"""Folder tests of the packaged ``ssv2`` benchmark.

Run only this folder: ``python -m pytest "$(yavih bench path ssv2)/tests" -q -p no:cacheprovider``.

- yaml / manifest / docs tests need nothing but pyyaml and the folder files;
- tests that go through ``Benchmark`` and the generic reader run on the core (always);
- the reader smoke writes tiny synthetic vp9 webm into ``tmp_path`` (PyAV with libvpx-vp9; skipped otherwise);
- the output-contract test runs ``bridge_template.py`` on synthetic raw files and machine-checks its
  ``(feats, pos, used)`` against ``output_contract`` and the declared rules (needs ``yavih.media.rules``,
  ``yavih.media.decode`` and ``grid_pos(region=...)``);
- ``make_manifest.build`` runs on a synthetic label fixture (needs torch for the draw);
- the real-data smoke and ``make_manifest --check`` run when the SSv2 root (with the official ``labels/``)
  resolves; ``--check`` also needs torch.
"""

from __future__ import annotations

import ast
import csv
import importlib
import importlib.util
import inspect
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_bytes, rows, bench = F.spec, F.manifest_bytes, F.rows, F.bench

RETIRED_KEYS = {
    "windows", "window_shift", "frame_rule", "frame_rule_params", "frame_grid", "frame_select", "pad_policy",
    "short_clip_policy", "crop", "audio_norm_context", "bridge_decode_allowed",
}
VIDEO_INFO_VOCAB = {"duration_s", "fps", "n_frames", "width", "height", "container", "bytes", "rotation"}
FPS = 12


def _assert_template_deps() -> None:
    """The media helpers the template uses exist (a regression fails, never skips)."""
    importlib.import_module("yavih.media.rules")
    importlib.import_module("yavih.media.decode")
    from yavih.bridge import grid_pos

    assert "region" in inspect.signature(grid_pos).parameters


# ----------------------------------------------------------------------------- files, yaml, hashes


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_folder_specific_files():
    """The folder layout is ``yavih bench lint``'s; these are this folder's own choices."""
    assert (HERE / "curation.json").is_file()
    assert not (HERE / "probe.py").exists()  # external: [] and no folder metric → no probe.py
    scripts = sorted(p.name for p in (HERE / "prepare").glob("*.sh"))
    assert scripts and all(re.match(r"^\d{2}_", s) for s in scripts), scripts


def test_yaml_is_schema_2_without_retired_keys(spec):
    assert spec["schema_version"] == 2 and spec["name"] == HERE.name == "ssv2"
    assert spec["readout"] == "clf" and spec["manifest"]["label"]["kind"] == "single"
    assert "requires" not in spec  # moved into output_contract
    stale = F.walk_keys({k: v for k, v in spec.items() if k != "changelog"}) & RETIRED_KEYS
    assert not stale, stale
    assert list(spec["streams"]) == ["video"]
    assert spec["streams"]["video"] == {"kind": "video", "media_column": "rel_media", "root": "media_root", "layout": "dir", "layout_params": {}, "required": True}
    assert set(spec["protocol"]) == {"clip_seconds", "train_draw", "views"}  # label facts only (none constrain the bridge)
    assert spec["protocol"]["clip_seconds"] is None and spec["protocol"]["views"] == "none"
    assert spec["source"]["kind"] == "manual_portal"
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in spec["source"]["sha256"].values()), spec["source"]["sha256"]
    assert spec["targets"] is None and spec["data"]["derived"] == {} and spec["readout_params"] == {}


def test_changelog(spec):
    log = spec["changelog"]
    assert [e["version"] for e in log] == [1] and spec["protocol_version"] == 1
    assert log[0]["invalidates"] is False and log[0]["supersedes"] == []


def test_train_draw_names_the_public_source(spec):
    """The draw is re-derived from the official train.json; drawn_from, curation.json and make_manifest agree."""
    draw = spec["protocol"]["train_draw"]
    assert (draw["cap"], draw["seed"], draw["balance"]) == (20000, 0, "class")
    sha = spec["source"]["sha256"]["labels/train.json"]
    assert sha in draw["drawn_from"] and "class_balanced_subsample" in draw["drawn_from"]
    cur = json.loads((HERE / "curation.json").read_text(encoding="utf-8"))
    assert cur["train_draw"] == draw
    assert cur["train_per_class"] == {"min": 91, "max": 116}
    mod = _load_make_manifest()
    assert mod.DRAWN_FROM == draw["drawn_from"]
    labels = {k: v for k, v in spec["source"]["sha256"].items() if k in ("labels/labels.json", "labels/train.json", "labels/validation.json")}
    assert mod.LABEL_FILES_SHA256 == labels == spec["manifest"]["upstream"]["upstream_files"]
    assert spec["manifest"]["upstream"]["sha256"] == sha


def test_input_contract(spec):
    ic = spec["input_contract"]
    assert set(ic) == {"handed_over", "info", "facts", "materialize"}
    assert set(ic["handed_over"]) == set(ic["info"]) == set(spec["streams"])
    assert "exactly as stored" in ic["handed_over"]["video"] and "nothing" in ic["handed_over"]["video"]
    assert ic["info"]["video"] == ["duration_s", "fps", "n_frames", "width", "height", "container", "bytes"]
    assert set(ic["info"]["video"]) <= VIDEO_INFO_VOCAB
    assert ic["facts"] == [] and ic["materialize"] is None


def test_output_contract_and_external(spec):
    oc = spec["output_contract"]
    assert oc == {"shape": "fixed", "coordinates": "media", "target_frame": None, "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None}}
    assert spec["external"] == []


def test_sha256_match(spec):
    F.assert_manifest_hashes(HERE, spec)
    cur = json.loads((HERE / "curation.json").read_text(encoding="utf-8"))
    assert cur["manifest_sha256"] == spec["manifest"]["sha256"] and cur["classes_sha256"] == spec["manifest"]["classes_sha256"]
    assert cur["curation"] == spec["manifest"]["upstream"]["curation"]
    assert cur["upstream_files"] == spec["manifest"]["upstream"]["upstream_files"]


def test_splits_assigned_once(spec, rows):
    assert F.assert_splits_assigned_once(spec, rows) == {"train": 20000, "val": 24777} == spec["budget"]["clips"]


def test_manifest_schema_and_order(rows, manifest_bytes):
    assert manifest_bytes.startswith(b"clip_id,rel_media,label,split\n") and b"\r\n" not in manifest_bytes
    assert [r["split"] for r in rows] == ["train"] * 20000 + ["val"] * 24777
    for r in rows:
        assert r["clip_id"].isdigit() and r["rel_media"] == f"20bn-something-something-v2/{r['clip_id']}.webm"
    # first / last rows of the frozen draw and of val (validation.json order, not sorted)
    assert rows[0]["clip_id"] == "34899" and rows[19999]["clip_id"] == "135421"
    assert rows[20000]["clip_id"] == "74225"


def test_classes_contiguous_labels_json_order(spec):
    crow = list(csv.DictReader((HERE / "classes.csv").open(newline="", encoding="utf-8")))
    assert list(crow[0].keys()) == ["index", "name"]
    assert [int(r["index"]) for r in crow] == list(range(174))
    names = [r["name"] for r in crow]
    assert len(set(names)) == 174
    assert names[0] == "Approaching something with your camera" and names[173] == "Wiping something off of something"
    assert names == sorted(names)  # labels.json index order happens to be Python-sorted
    assert not any(spec["manifest"]["label"]["sep"] in n for n in names)
    assert sum("," in n for n in names) == 12


def test_train_draw_is_class_balanced(rows):
    names = [r["name"] for r in csv.DictReader((HERE / "classes.csv").open(newline="", encoding="utf-8"))]
    idx = {n: i for i, n in enumerate(names)}
    train = Counter(idx[r["label"]] for r in rows if r["split"] == "train")
    assert len(train) == 174
    assert sorted(Counter(train.values()).items()) == [(91, 1), (111, 1), (115, 154), (116, 18)]
    # the floor: train majority (ties → lowest index = class 0, 116 rows) on the full val split
    val = Counter(idx[r["label"]] for r in rows if r["split"] == "val")
    assert max(train, key=lambda c: (train[c], -c)) == 0 and val[0] == 116
    assert min(val.values()) == 15 and max(val.values()) == 530 and len(val) == 174


def test_curation_sums(spec):
    """listed + excluded == total_upstream and listed == rows are ``yavih bench lint``'s (budget)."""
    cur = spec["manifest"]["upstream"]["curation"]
    assert cur["total_upstream"] == spec["manifest"]["upstream"]["rows"] == 193690
    for v in cur["per_split"].values():
        assert v["staged"] + v["missing"] == v["listed"]
    assert cur["excluded"] == {"draw_remainder": 168913 - 20000}


def test_metric_names_resolve(spec):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["metrics"]["headline"] == "acc" and spec["metrics"]["floors"] == ["acc_floor"] and spec["metrics"]["controls"] == []
    assert abs(spec["metrics"]["chance"]["acc"] - 1 / 174) < 1e-6


def test_yaml_loads_through_core(spec, bench):
    assert bench.name == "ssv2" and bench.spec.readout == "clf"
    assert len(bench.protocol_hash) == 64 and len(bench.manifest_hash) == 64
    assert bench.manifest_hash == "9bd6359dca83a92dd0739a3f112bbf04b7dcfd29d4de4a0bc21f14b47b64d051"
    assert bench.n_rows("train") == 20000 and bench.n_rows("val") == 24777
    view = bench.view()
    assert view.streams == {"video": "video"} and view.facts == {} and view.output_contract == spec["output_contract"]


# ----------------------------------------------------------------------------- documentation


def test_readme_facts():
    """Folder facts in README.md (the README structure is ``yavih bench lint``'s)."""
    text = (HERE / "README.md").read_text(encoding="utf-8")
    assert "@inproceedings{goyal2017something" in F.section(text, 2, "Citation")
    proto = F.section(text, 2, "In yavih")
    assert "class_balanced_subsample" in proto and "labels/train.json" in proto


def test_setup_facts(spec):
    """Folder facts in setup.md (the setup.md structure and contract copies are ``yavih bench lint``'s)."""
    text = (HERE / "setup.md").read_text(encoding="utf-8")
    assert spec["manifest"]["sha256"] in F.section(text, 2, "Prepare & verify")
    # every upstream sha256 the yaml pins is documented where the agent obtains the data
    obtain = F.section(text, 2, "Obtain the data")
    assert all(v in obtain for v in spec["source"]["sha256"].values())


# ----------------------------------------------------------------------------- no reader.py / bridge_template.py (static)


def test_folder_ships_no_reader_py():
    """The folder ships no reader.py: ``Benchmark.reader()`` falls back to the generic
    ``yavih.sample.iter_samples``, which hands over what benchmark.yaml declares; the reader smokes in this
    file go through ``bench.reader``."""
    assert not (HERE / "reader.py").exists()


def test_bridge_template_static():
    src = (HERE / "bridge_template.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert '"decode":' not in src and '"views": 1' in src and "decode_plan" not in src
    assert "Preprocessing source" in src and "crop_region" in src and "grid_pos(" in src
    enc = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "encode")
    rets = [n for n in ast.walk(enc) if isinstance(n, ast.Return)]
    assert rets and all(isinstance(r.value, ast.Tuple) and len(r.value.elts) == 3 for r in rets)  # (feats, pos, used)


# ----------------------------------------------------------------------------- synthetic raw files


def _write_webm(path: Path, n_frames: int, w: int = 64, h: int = 48, first_pts: int = 0) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w", format="webm") as c:
        vs = c.add_stream("libvpx-vp9", rate=FPS)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        for i in range(n_frames):
            img = np.full((h, w, 3), min(255, 8 + 8 * i), dtype=np.uint8)  # frame index encoded in brightness
            img[:, : w // 2, 0] = 255 - img[:, : w // 2, 0]  # a left/right difference so x positions matter
            fr = av.VideoFrame.from_ndarray(img, format="rgb24")
            fr.pts = first_pts + i
            for pkt in vs.encode(fr):
                c.mux(pkt)
        for pkt in vs.encode():
            c.mux(pkt)


def _stored_pts(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        return np.array(sorted(float(p.pts * vs.time_base) for p in c.demux(vs) if p.size > 0 and p.pts is not None))


def _need_vp9():
    av = pytest.importorskip("av")
    try:
        av.codec.Codec("libvpx-vp9", "w")
    except Exception:  # noqa: BLE001
        pytest.skip("PyAV build without a vp9 encoder")


def test_reader_smoke_on_synthetic_raw_files(tmp_path, spec):
    """Reader smoke: the stored bytes come back untouched with header facts; meta holds facts only."""
    _need_vp9()
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    for split, n_frames, w in (("train", 30, 64), ("val", 18, 80)):
        rel = b.manifest(split).media[0]["video"]
        f = tmp_path / rel
        _write_webm(f, n_frames, w=w)
        samples = list(b.reader(split, rows=range(0, 1), batch_size=1, canaries=0))
        assert len(samples) == 1 and samples[0].row == 0 and samples[0].split == split
        s = samples[0]
        assert set(s.streams) == {"video"} and s.meta == {"facts": {}}
        st = s.streams["video"]
        assert st.kind == "video" and st.name == "video"
        assert Path(st._real_path) == f and Path(st.path).resolve() == f.resolve() and st.bytes() == f.read_bytes()  # raw means stored
        with st.open() as fh:
            assert fh.read() == f.read_bytes()
        info = st.info
        assert set(spec["input_contract"]["info"]["video"]) <= set(info)
        assert info["width"] == w and info["height"] == 48 and info["container"] in ("webm", "matroska", "matroska,webm")
        assert info["bytes"] == f.stat().st_size and abs(info["fps"] - FPS) < 1e-6
        assert abs(info["duration_s"] - n_frames / FPS) < 0.05
        # n_frames is guaranteed through the site census (webm headers hold no count, setup.md "Input contract")
        assert info["n_frames"] in (None, n_frames)
    # canaries are exact copies (private _canary_of, never in meta)
    rel = b.manifest("val").media
    for i in range(1, 8):
        _write_webm(tmp_path / rel[i]["video"], 6 + i)
    out = list(b.reader("val", rows=range(0, 8), batch_size=1, canaries=1))
    assert len(out) == 9 and out[8]._canary_of in range(0, 8) and out[8].meta == {"facts": {}}
    orig = out[out[8]._canary_of]
    assert out[8].streams["video"].bytes() == orig.streams["video"].bytes() and out[8].row == orig.row


def test_reader_missing_media_is_a_missing_row(tmp_path):
    from yavih import Benchmark
    from yavih.media import MediaMissing

    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    with pytest.raises(MediaMissing):
        for s in b.reader("val", rows=range(0, 1)):
            s.streams["video"].bytes()


class _FakeRawStream:
    """Duck type of RawStream (layout dir) so the template can be checked without the core reader."""

    def __init__(self, path: Path, info: dict) -> None:
        self.name, self.kind, self.path, self.member, self.info = "video", "video", str(path), None, info

    def bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def open(self):
        return open(self.path, "rb")


class _FakeSample:
    def __init__(self, row: int, stream: _FakeRawStream) -> None:
        self.row, self.clip_id, self.split, self.streams, self.meta = row, f"c{row}", "val", {"video": stream}, {"facts": {}}


def _template():
    return F.load_module(HERE / "bridge_template.py", "ssv2_bridge_template")


class _View:
    name, aliases, readout, protocol_version, protocol_hash = "ssv2", [], "clf", 1, "0" * 64
    streams = {"video": "video"}
    facts: dict = {}

    def __init__(self, spec: dict) -> None:
        self.input_contract, self.output_contract = spec["input_contract"], spec["output_contract"]


def test_bridge_template_output_contract(tmp_path, spec):
    """Run the template on synthetic raw webm of different widths/lengths; machine-check (feats, pos, used)."""
    _assert_template_deps()
    _need_vp9()
    from yavih.media.rules import crop_region, rule_times, select_nearest

    mod = _template()
    clips = [(30, 64), (18, 80), (7, 48), (50, 96)]  # (frames, width): 7 frames < N_FRAMES → repeated frames
    samples = []
    for k, (n, w) in enumerate(clips):
        f = tmp_path / f"{k}.webm"
        _write_webm(f, n, w=w)
        info = {"duration_s": n / FPS, "fps": float(FPS), "n_frames": n, "width": w, "height": 48, "container": "webm", "bytes": f.stat().st_size}
        samples.append(_FakeSample(k, _FakeRawStream(f, info)))
    br = mod.Ssv2Bridge(_View(spec), {})
    d = br.describe()
    assert d["views"] == 1 and "decode" not in d and set(d["input"]) == {"video"}
    assert d["input"]["video"]["temporal"] == {"rule": "uniform", "n": mod.N_FRAMES, "span": "full", "edge": "error"}
    feats, pos, used = br.encode(samples)
    n_tok = (mod.N_FRAMES // mod.TUBELET) * mod.GRID * mod.GRID
    # shape: fixed [B, N, D]
    feats = np.asarray(feats)
    assert feats.shape[:2] == (len(samples), n_tok) and feats.dtype == np.float16 and np.isfinite(feats).all()
    pos = np.asarray(pos)
    assert pos.shape == (len(samples), n_tok, 4) and pos.dtype == np.float32
    assert len(used) == len(samples) and all(set(u) == {"video"} for u in used)
    for s, p, u in zip(samples, pos, used, strict=True):
        info = s.streams["video"].info
        times = np.asarray(u["video"]["times"])
        # E_TIMES_OUT_OF_MEDIA: every fed time inside [0, duration_s] (1 ms slack)
        assert len(times) == mod.N_FRAMES and times.min() >= -1e-3 and times.max() <= info["duration_s"] + 1e-3
        # E_RULE_MISMATCH (temporal): the declared rule re-evaluated on the facts; the fed frames are the stored
        # frames nearest (ties earlier) to those times, and their pts are what is reported
        want = rule_times(d["input"]["video"]["temporal"], info)
        stored = _stored_pts(Path(s.streams["video"].path))
        assert np.allclose(times, stored[select_nearest(stored, want)], atol=1e-6)
        if info["n_frames"] >= mod.N_FRAMES:  # validate's 0.5/fps + 1 ms holds unless frames are oversampled (end of
            assert np.abs(times - want).max() <= 0.5 / info["fps"] + 1e-3  # clip: up to 1/fps)
        # media coordinates: pos in [0, 1] ∪ {-1}; stream tag 0; t = time / duration_s; (y, x) inside the crop region
        assert set(np.unique(p[:, 0])) == {0.0}
        assert ((p[:, 1:] >= 0) & (p[:, 1:] <= 1)).all()
        y0, x0, y1, x1 = crop_region(d["input"]["video"]["spatial"], info["width"], info["height"])
        assert (p[:, 2] >= y0 - 1e-6).all() and (p[:, 2] <= y1 + 1e-6).all() and (p[:, 3] >= x0 - 1e-6).all() and (p[:, 3] <= x1 + 1e-6).all()
        t_tok = np.unique(p[:, 1])
        assert np.allclose(t_tok, times.reshape(-1, mod.TUBELET).mean(axis=1) / info["duration_s"], atol=1e-6)
    # the widths differ, so the regions (and pos) differ per clip → per_sample is required
    assert not np.allclose(pos[0], pos[3])
    # bridge check's hard checks in miniature: rerun bit-identical, alone and shuffled batches equal (instance-wise)
    f2, p2, u2 = br.encode(samples)
    assert np.array_equal(np.asarray(f2), feats) and np.array_equal(np.asarray(p2), pos) and u2 == used
    for k, s in enumerate(samples):
        fa, pa, ua = br.encode([s])
        assert np.array_equal(np.asarray(fa)[0], feats[k]) and np.array_equal(np.asarray(pa)[0], pos[k]) and ua[0] == used[k]
    order = [2, 0, 3, 1]
    fs, ps, us = br.encode([samples[i] for i in order])
    assert np.array_equal(np.asarray(fs), feats[order]) and [us[j] for j in np.argsort(order)] == used


def test_bridge_template_validates_against_core(bench, spec):
    _assert_template_deps()
    from yavih.bridge import validate_describe

    br = _template().Ssv2Bridge(bench.view(), {})
    d = br.describe()
    d["weights_sha256"] = "0" * 64  # the template's TODO is None; validate the rest of the schema
    validate_describe(d, br, bench.view())


# ----------------------------------------------------------------------------- make_manifest


def _load_make_manifest():
    return F.load_module(HERE / "make_manifest.py", "ssv2_make_manifest")


def test_draw_port_semantics():
    pytest.importorskip("torch")
    mod = _load_make_manifest()
    labels = ["b", "a", "b", "a", "c", "c", "b"]
    keep = mod.class_balanced_subsample(labels, 4, 0)
    assert keep == sorted(keep) and len(keep) == 4
    assert mod.class_balanced_subsample(labels, 0, 0) == list(range(7))
    assert mod.class_balanced_subsample(labels, 99, 0) == list(range(7))
    assert max(Counter(labels[i] for i in keep).values()) <= 2
    assert mod.template_to_label("Holding [something] next to [something]") == "Holding something next to something"


def test_make_manifest_build_on_synthetic_labels():
    """``build`` on a tiny label fixture in the release's JSON shape: rows, classes and the draw."""
    pytest.importorskip("torch")
    mod = _load_make_manifest()
    label_map = {"Pushing something": "1", "Holding something": "0", "Dropping something": "2"}
    tpl = {"Pushing something": "Pushing [something]", "Holding something": "Holding [something]", "Dropping something": "Dropping [something]"}
    order = ["Pushing something", "Holding something", "Pushing something", "Dropping something", "Holding something",
             "Pushing something", "Dropping something", "Holding something"]
    train = [{"id": str(100 + i), "label": n.lower(), "template": tpl[n], "placeholders": ["x"]} for i, n in enumerate(order)]
    val = [{"id": "7", "template": tpl["Dropping something"]}, {"id": "3", "template": tpl["Holding something"]}]
    d = mod.build(label_map, train, val, cap=6, seed=0)
    assert d["classes_csv"] == b"index,name\n0,Holding something\n1,Pushing something\n2,Dropping something\n"
    out = list(csv.DictReader(d["manifest_csv"].decode().splitlines()))
    assert [r["split"] for r in out] == ["train"] * 6 + ["val"] * 2
    assert [r["clip_id"] for r in out[6:]] == ["7", "3"]  # val keeps validation.json order
    ids = [int(r["clip_id"]) for r in out[:6]]
    assert ids == sorted(ids)  # the draw keeps ascending train.json order
    assert d["per_class"] == {"Holding something": 2, "Pushing something": 2, "Dropping something": 2}
    assert all(r["rel_media"] == f"20bn-something-something-v2/{r['clip_id']}.webm" for r in out)
    assert mod.build(label_map, train, val, cap=6, seed=0)["manifest_csv"] == d["manifest_csv"]
    with pytest.raises(SystemExit):
        mod.build(label_map, train + [{"id": "999", "template": "Unknown [something]"}], val)
    with pytest.raises(SystemExit):
        mod.build(label_map, train, val + [{"id": "100", "template": tpl["Holding something"]}])


def test_make_manifest_check_mode_matches_committed_files():
    """Re-derive from the official label files and compare; skipped where they or torch are absent."""
    import subprocess
    import sys

    root = F.site_root(HERE, "media_root", needs=("labels/train.json", "labels/validation.json", "labels/labels.json"))
    if root is None:
        pytest.skip("official SSv2 labels/ not staged here")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch needed for the class-balanced draw")
    r = subprocess.run([sys.executable, str(HERE / "make_manifest.py"), "--check", "--labels", str(root / "labels")],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert '"check": "identical"' in r.stdout


# ----------------------------------------------------------------------------- real data (optional)


def _real_root() -> str | None:
    root = F.site_root(HERE, "media_root", needs=("20bn-something-something-v2/34899.webm",))
    return str(root) if root else None


def test_reader_smoke_on_real_data():
    """Optional extra: the first rows of each split through the folder reader on the real release."""
    root = _real_root()
    if root is None:
        pytest.skip("SSv2 media not present on this machine")
    from yavih import Benchmark

    b = Benchmark.from_dir(HERE, data={"media_root": root})
    for split in ("train", "val"):
        samples = list(b.reader(split, rows=range(0, 8), batch_size=2, canaries=1))
        assert [s.row for s in samples[:8]] == list(range(8)) and len(samples) == 9 and samples[8]._canary_of in range(0, 4)
        for s in samples[:2]:
            st = s.streams["video"]
            assert set(s.streams) == {"video"} and s.meta == {"facts": {}}
            assert st.bytes() == Path(st.path).read_bytes()
            assert st.info["height"] == 240 and 240 <= st.info["width"] <= 483 and abs(st.info["fps"] - 12.0) < 1e-6
            assert 0.8 < st.info["duration_s"] < 14.2
    # val row 0 = clip 74225: 5.25 s, 63 frames at pts k/12
    st = next(iter(b.reader("val", rows=range(0, 1)))).streams["video"]
    assert abs(st.info["duration_s"] - 5.25) < 1e-6 and st.info["n_frames"] in (None, 63)


def test_doc_snippets_are_not_stale():
    """No retired reader-side vocabulary in the agent-facing files."""
    for name in ("setup.md", "README.md", "bridge_template.py"):
        text = (HERE / name).read_text(encoding="utf-8")
        for stale in ("frame_times(", ".frames(N", "decode_plan", "bridge_decode_allowed", "W_PREFETCH_MISS", "Stream.window"):
            assert stale not in text, stale
