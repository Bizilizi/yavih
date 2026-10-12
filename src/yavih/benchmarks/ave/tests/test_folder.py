"""Folder tests for ave (README.md, setup.md).

Run only this file: ``python -m pytest "$(yavih bench path ave)/tests" -q -p no:cacheprovider``.

Three tiers:

* pure (yaml, manifest, classes, README.md / setup.md, the official-accuracy transcription, the bridge
  template's describe() and its output on stub-decoded media) — always run;
* core (``Benchmark.from_dir``, metric resolution, the reader smoke on synthetic raw files) — always run; the
  synthetic smoke needs PyAV with an mp4 video encoder and aac;
* make_manifest.py on a synthetic upstream fixture — always runs (needs h5py);
* site (the real AVE tar + index.json and the public upstream files under the resolved ``media_root``, VGGSound's
  ``metadata/vggsound.csv``) — skipped elsewhere.
"""

from __future__ import annotations

import re
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from yavih.testing import folder as F
from yavih.versions import BENCHMARK_SCHEMA_VERSION

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_rows, classes, bench, template = F.spec, F.rows, F.classes, F.bench, F.template

ROWS = {"train": 2624, "val": 308, "test": 316}
FACTS = {"num_segments": 10, "segment_seconds": 1.0}

assert BENCHMARK_SCHEMA_VERSION >= 2  # the folder is schema 2; an older core is not supported


def _load(name: str, rel: str):
    return F.load_module(HERE / rel, name)  # registered in sys.modules: preprocessing_source() looks it up by name


@pytest.fixture(scope="module")
def view(spec):
    """A BenchmarkView built from the yaml alone: what a bridge receives."""
    from yavih.bridge import BenchmarkView

    return BenchmarkView(
        name="ave", aliases=[], readout="seq", protocol_version=spec["protocol_version"], protocol_hash="0" * 64,
        streams={k: v["kind"] for k, v in spec["streams"].items()}, input_contract=spec["input_contract"],
        output_contract=spec["output_contract"], facts={k: spec["protocol"][k] for k in spec["input_contract"]["facts"]},
    )


# ----------------------------------------------------------------------------- benchmark.yaml (schema 2)


def test_yaml_schema2_and_changelog(spec):
    assert spec["schema_version"] == 2 and spec["name"] == HERE.name == "ave" and spec["readout"] == "seq"
    assert spec["aliases"] == []
    assert spec["protocol_version"] == 1 == spec["changelog"][-1]["version"]
    assert [c["version"] for c in spec["changelog"]] == [1]
    v1 = spec["changelog"][0]
    assert v1["invalidates"] is False and v1["supersedes"] == []
    assert spec["header"]["name"] == "AVE" and spec["header"]["title"] == "Audio-Visual Event Localization in Unconstrained Videos"


def test_streams_and_input_contract(spec):
    st = spec["streams"]
    assert set(st) == {"video", "audio"}
    for name, kind in (("video", "video"), ("audio", "audio")):
        assert st[name] == {"kind": kind, "media_column": "rel_media", "root": "media_root", "layout": "tar_index",
                            "layout_params": {}, "required": True}
    assert spec["manifest"]["columns"]["rel_media"] == {"type": "str", "role": "media", "stream": ["video", "audio"]}
    assert spec["data"]["tar_index"] == "index.json" and spec["data"]["derived"] == {}
    ic = spec["input_contract"]
    assert set(ic) == {"handed_over", "info", "facts", "materialize"}
    assert set(ic["handed_over"]) == set(ic["info"]) == set(st)
    assert all(isinstance(v, str) and v for v in ic["handed_over"].values())
    assert ic["info"] == {"video": ["duration_s", "fps", "width", "height", "container", "bytes"],
                          "audio": ["duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"]}
    assert ic["facts"] == ["num_segments", "segment_seconds"] and all(f in spec["protocol"] for f in ic["facts"])
    assert ic["materialize"] is None


def test_output_contract_and_external(spec):
    assert spec["output_contract"] == {
        "shape": "fixed", "coordinates": "media", "target_frame": None,
        "requires": {"pos": ["t"], "streams": ["video"], "grid": None, "t_bins": 8},
    }
    assert spec["external"] == []
    assert not (HERE / "probe.py").exists(), "external is empty: no probe.py"
    assert spec["targets"] is None


def test_protocol_constants(spec):
    p = spec["protocol"]
    assert p == {"clip_seconds": 10.0, "train_draw": None, "views": "none", "num_segments": 10, "segment_seconds": 1.0,
                 "segments_scored": 10, "segment_offset": 0, "background_index": 28}
    assert spec["readout_params"] == {"T": 10} and spec["readout_params"]["T"] == p["segments_scored"]
    assert spec["output_contract"]["requires"]["t_bins"] <= p["segments_scored"]
    m = spec["metrics"]
    assert m["headline"] == "seg_acc"
    assert m["list"] == ["seg_acc", "clip_acc", "seg_acc_boundary", "n_boundary", "n_eval"]
    assert m["floors"] == ["seg_acc_floor", "seg_acc_boundary_floor"] and m["controls"] == []
    assert m["chance"] == {"seg_acc": 0.034483}
    assert m["ci"] == {"seg_acc": "bootstrap_clip", "seg_acc_boundary": "bootstrap_clip"}
    assert m["bootstrap"] == {"samples": 2000, "seed": 0} and m["aggregation"] is None


# ----------------------------------------------------------------------------- manifest and classes


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_sha256_match(spec, manifest_rows):
    F.assert_manifest_hashes(HERE, spec)  # also: gzip mtime 0, no file name (byte-identical re-freeze)
    assert len(manifest_rows) == sum(ROWS.values()) == spec["manifest"]["upstream"]["rows"]
    assert list(manifest_rows[0].keys()) == ["clip_id", "split", "label", "rel_media", "event_start", "event_end"]


def test_splits_assigned_once(spec, manifest_rows):
    roles = spec["splits"]
    assert dict(F.assert_splits_assigned_once(spec, manifest_rows)) == ROWS
    assert roles["train"] == ["train"] and roles["eval"] == ["test"] and roles["unused"] == ["val"]
    assert spec["budget"]["clips"] == {"train": 2624, "test": 316}


def test_classes_contiguous_background_pinned(spec, classes):
    assert list(classes[0].keys()) == ["index", "name"]
    assert [int(c["index"]) for c in classes] == list(range(29))
    names = [c["name"] for c in classes]
    assert len(set(names)) == 29
    assert names[28] == "background" == names[spec["protocol"]["background_index"]]
    assert names[:28] == sorted(names[:28])
    assert names[0] == "Accordion" and names[27] == "Violin, fiddle"


def test_rows_and_events(manifest_rows, classes):
    names = {c["name"] for c in classes} - {"background"}
    for r in manifest_rows:
        ytid, _, up_row = r["clip_id"].rpartition("#")
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,32}", ytid) and up_row.isdigit() and 0 <= int(up_row) < 4143
        assert r["rel_media"] == f"video/{ytid}.mp4"
        assert r["label"] in names
        a, b = int(r["event_start"]), int(r["event_end"])
        assert 0 <= a < b <= 10, r["clip_id"]
    ups = [int(r["clip_id"].rpartition("#")[2]) for r in manifest_rows]
    assert ups == sorted(ups)
    short = Counter(r["split"] for r in manifest_rows if int(r["event_end"]) - int(r["event_start"]) < 10)
    assert dict(short) == {"train": 838, "val": 112, "test": 104}
    assert sum(1 for r in manifest_rows if int(r["event_end"]) - int(r["event_start"]) == 10) == 2194
    by_split = {s: {r["rel_media"] for r in manifest_rows if r["split"] == s} for s in ROWS}
    assert not (by_split["train"] & by_split["test"]) and not (by_split["train"] & by_split["val"])
    shared = Counter(r["rel_media"] for r in manifest_rows)
    assert sum(1 for r in manifest_rows if shared[r["rel_media"]] > 1) == 32
    assert all(r["split"] == "train" for r in manifest_rows if shared[r["rel_media"]] > 1)


def test_curation_sums(spec, manifest_rows):
    cur = spec["manifest"]["upstream"]["curation"]
    listed = sum(v["listed"] for v in cur["per_split"].values())
    assert listed + sum(cur["excluded"].values()) == cur["total_upstream"] == 4143
    for split, v in cur["per_split"].items():
        assert v["listed"] == sum(r["split"] == split for r in manifest_rows)
        assert v["staged"] == v["listed"] and v["missing"] == 0
    assert cur["excluded"] == {"leakage_vggsound_train": 886, "leakage_ave_cross_split": 9}
    assert cur["leakage_audit"]["excluded"] == 895
    assert cur["short_clips"] == 74


# ----------------------------------------------------------------------------- targets, official accuracy, floors


def _targets(manifest_rows, classes, protocol, split):
    """Seq targets from the manifest (independent of core): second s → class iff start ≤ s < end."""
    idx = {c["name"]: int(c["index"]) for c in classes}
    off, T, bg = protocol["segment_offset"], protocol["segments_scored"], protocol["background_index"]
    rows = [r for r in manifest_rows if r["split"] == split]
    t = np.array([[idx[r["label"]] if int(r["event_start"]) <= s < int(r["event_end"]) else bg for s in range(off, off + T)]
                  for r in rows], dtype=np.int64)
    boundary = np.array([int(r["event_end"]) - int(r["event_start"]) < protocol["num_segments"] for r in rows])
    return t, boundary, rows


def _official_compute_acc(labels: np.ndarray, x_labels: np.ndarray, nb_batch: int) -> float:
    """YapengTian/AVE-ECCV18 ``supervised_main.py::compute_acc``, transcribed: argmax per segment of the
    one-hot labels and of the predictions over all 10 segments of every test video, then accuracy over the
    ``nb_batch * 10`` pooled segments (background included)."""
    N = int(nb_batch * 10)
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for i in range(nb_batch):
        for j in range(x_labels.shape[1]):
            pre_labels[c] = np.argmax(x_labels[i, j, :])
            real_labels[c] = np.argmax(labels[i, j, :])
            c += 1
    return float(np.mean(real_labels == pre_labels))


def test_seg_acc_is_the_official_ave_accuracy(spec):
    """README.md / setup.md: yavih's ``seg_acc`` with segments_scored 10 / offset 0 equals the authors' compute_acc."""
    from yavih.metrics.segment import seg_acc

    rng = np.random.default_rng(0)
    for n in (1, 3, 17):
        targets = rng.integers(0, 29, size=(n, 10))
        logits = rng.standard_normal((n, 10, 29)).astype(np.float32)
        logits[rng.random((n, 10)) < 0.5] *= 0.0  # ties at 0 → argmax picks index 0 in both implementations
        onehot = np.eye(29)[targets]
        assert seg_acc(logits, targets, spec["protocol"], {}) == pytest.approx(_official_compute_acc(onehot, logits, n))
    # hand-made: 2 videos, 10 segments; 17 of 20 segments right
    t = np.full((2, 10), 28)
    t[0, 2:6] = 4
    t[1, :] = 7
    pred = t.copy()
    pred[0, 2] = 28
    pred[1, 8:] = 28
    logits = np.eye(29)[pred].astype(np.float32)
    assert _official_compute_acc(np.eye(29)[t], logits, 2) == pytest.approx(0.85)
    assert seg_acc(logits, t, spec["protocol"], {}) == pytest.approx(0.85)


def test_segment_targets_and_floors(spec, manifest_rows, classes):
    """Seq targets over all ten seconds; floors 0.172 / 0.521 on test."""
    from yavih.metrics.segment import constant_segment_class, segment_targets

    p = spec["protocol"]
    ttr, _, _ = _targets(manifest_rows, classes, p, "train")
    tte, bnd, rows = _targets(manifest_rows, classes, p, "test")
    assert ttr.shape == (2624, 10) and tte.shape == (316, 10) and int(bnd.sum()) == 104
    idx = {c["name"]: int(c["index"]) for c in classes}
    assert all(segment_targets(r["event_start"], r["event_end"], idx[r["label"]], p) == tte[k].tolist() for k, r in enumerate(rows))
    k = constant_segment_class(ttr)
    assert k == 28 == Counter(ttr.ravel().tolist()).most_common(1)[0][0]
    assert round(float((tte == k).mean()), 3) == 0.172
    assert round(float((tte[bnd] == k).mean()), 3) == 0.521


# ----------------------------------------------------------------------------- README.md / setup.md


def test_readme_facts():
    """Folder facts in README.md (the README structure is ``yavih bench lint``'s)."""
    text = (HERE / "README.md").read_text(encoding="utf-8")
    cite, proto = F.section(text, 2, "Citation"), F.section(text, 2, "In yavih")
    assert "```bibtex" in cite and "tian2018ave" in cite
    assert "https://arxiv.org/abs/1803.08842" in text.split("\n\n")[1]
    for fact in ("0.172", "0.521", "2 624", "316", "make_manifest.py"):
        assert fact in proto, fact


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe(template, view):
    from yavih.bridge import preprocessing_source

    cls = template.AVEBridge
    assert cls.streams == ("video", "audio") and cls.pos_mode == "per_sample"
    assert not hasattr(cls, "decode_plan"), "decode_plan is not part of the Bridge API"
    assert preprocessing_source(cls) is None, "the template's slot must stay TODO (W_NO_PREPROCESSING_SOURCE until filled)"
    for reading, streams in (("av", ["video", "audio"]), ("v", ["video"]), ("a", ["audio"])):
        b = cls(view, {"reading": reading})
        d = b.describe()
        for key in ("weights_sha256", "weights_ref", "dtype", "batch_size", "streams", "pos_mode", "reading", "model",
                    "layer", "pooling", "views", "input", "precision", "args"):
            assert key in d, key
        assert d["reading"] == reading
        assert "decode" not in d and d["views"] == 1 and d["streams"] == streams == sorted(d["input"], key=streams.index)
        assert d["input"].get("video", {}).get("temporal", {"rule": "uniform"})["rule"] == "uniform"
        from yavih.bridge import validate_input

        validate_input(d["input"], view.streams, d["streams"])


class _StubStream:
    def __init__(self, name: str, kind: str, info: dict) -> None:
        self.name, self.kind, self.info = name, kind, info


class _StubSample:
    def __init__(self, clip_id: str, vinfo: dict, ainfo: dict) -> None:
        self.row, self.clip_id, self.split = 0, clip_id, "test"
        self.streams = {"video": _StubStream("video", "video", vinfo), "audio": _StubStream("audio", "audio", ainfo)}
        self.meta = {"facts": dict(FACTS)}


def _stub_decode() -> types.ModuleType:
    """A deterministic stand-in for ``yavih.media.decode`` keyed on the stream's facts only: nearest frame
    by pts at the declared fps, constant-gain audio. Lets the template's pos/used logic run anywhere."""
    mod = types.ModuleType("yavih.media.decode")

    def frames_at(stream, times):
        fps, h, w = stream.info["fps"], stream.info["height"], stream.info["width"]
        idx = np.clip(np.round(np.asarray(times) * fps - 0.5), 0, None).astype(np.int64)
        frames = (np.arange(len(idx))[:, None, None, None] * 7 + np.zeros((len(idx), h, w, 3))).astype(np.uint8)
        return frames, idx / fps

    def audio(stream, ranges):
        n = int(sum(int(hi) - int(lo) for lo, hi in ranges))
        return np.full((n, int(stream.info["channels"])), 0.1, np.float32), int(stream.info["sample_rate"])

    mod.frames_at, mod.audio = frames_at, audio
    mod.decoder_info = lambda: {"lib": "stub", "version": "0"}
    return mod


def _check_output_contract(spec, describe, feats, pos, used, infos) -> None:
    """The ave output contract on one batch: fixed [N, D]; pos in media coordinates;
    ``used`` inside the media; ≥ t_bins scored seconds with a timed video token per clip."""
    oc = spec["output_contract"]["requires"]
    p = spec["protocol"]
    feats, pos = np.asarray(feats), np.asarray(pos)
    B, N, D = feats.shape
    assert N >= 1 and D >= 1 and pos.shape == (B, N, 4) and pos.dtype == np.float32
    assert np.all(np.isfinite(feats.astype(np.float32)))
    ok = ((pos[..., 1:] >= 0) & (pos[..., 1:] <= 1)) | (pos[..., 1:] == -1)
    assert ok.all(), "pos outside [0, 1] ∪ {-1}"
    streams = describe["streams"]
    assert set(oc["streams"]) <= set(streams)
    vid = streams.index("video")
    for i in range(B):
        vi, ai = infos[i]
        u = used[i]
        assert set(u) == set(describe["input"])
        times = np.asarray(u["video"]["times"])
        assert ((times >= -1e-3) & (times <= vi["duration_s"] + 1e-3)).all()
        if "audio" in u:
            r = np.asarray(u["audio"]["samples"])
            assert (r[:, 0] >= 0).all() and (r[:, 1] <= ai["n_samples"]).all() and (r[:, 0] <= r[:, 1]).all()
        v = pos[i][(pos[i][:, 0] == vid) & (pos[i][:, 1] >= 0)]
        secs = np.floor(v[:, 1] * vi["duration_s"] / p["segment_seconds"]).astype(np.int64) - p["segment_offset"]
        hit = {int(s) for s in secs if 0 <= s < p["segments_scored"]}
        assert len(hit) >= oc["t_bins"], (i, sorted(hit))


def test_bridge_template_output_contract_on_stub_media(spec, template, view, monkeypatch):
    """The template's encode → (feats, pos, used) satisfies the output contract on two clips that differ in
    duration (a 9.73 s audio track), fps and aspect ratio, and is instance-wise (alone == batched)."""
    pytest.importorskip("yavih.media.rules")
    import yavih.media as ym

    stub = _stub_decode()
    monkeypatch.setitem(sys.modules, "yavih.media.decode", stub)
    monkeypatch.setattr(ym, "decode", stub, raising=False)
    from yavih.media.rules import crop_region

    infos = [
        (dict(duration_s=9.96, fps=25.0, width=320, height=240, container="mp4", bytes=1),
         dict(duration_s=9.73, sample_rate=44100, n_samples=429093, channels=2, container="mp4", bytes=1)),
        (dict(duration_s=10.01, fps=29.97, width=1280, height=720, container="mp4", bytes=1),
         dict(duration_s=10.0, sample_rate=44100, n_samples=441000, channels=1, container="mp4", bytes=1)),
    ]
    samples = [_StubSample(f"clip{i}#{i}", vi, ai) for i, (vi, ai) in enumerate(infos)]
    b = template.AVEBridge(view, {"reading": "av"})
    d = b.describe()
    feats, pos, used = b.encode(samples)
    _check_output_contract(spec, d, feats, pos, used, infos)
    # spatial positions lie in the declared crop region of the ORIGINAL frame (E_RULE_MISMATCH spatial)
    for i, (vi, _) in enumerate(infos):
        y0, x0, y1, x1 = crop_region(d["input"]["video"]["spatial"], vi["width"], vi["height"])
        v = pos[i][pos[i][:, 0] == 0]
        assert (v[:, 2] >= y0 - 1e-6).all() and (v[:, 2] <= y1 + 1e-6).all()
        assert (v[:, 3] >= x0 - 1e-6).all() and (v[:, 3] <= x1 + 1e-6).all()
    assert pos[1][pos[1][:, 0] == 0][:, 3].min() > pos[0][pos[0][:, 0] == 0][:, 3].min()  # 16:9 loses more width than 4:3
    # the short audio track: the template pads to 10 s and reports it
    assert used[0]["audio"]["pad"] == [0, 441000 - 429093] and used[0]["audio"]["samples"] == [[0, 429093]]
    # instance-wise: each clip alone gives the same features, pos and used
    for i, s in enumerate(samples):
        f1, p1, u1 = b.encode([s])
        np.testing.assert_array_equal(f1[0], feats[i])
        np.testing.assert_array_equal(p1[0], pos[i])
        assert u1[0] == used[i]


# ----------------------------------------------------------------------------- core: load, metrics, reader


def test_yaml_loads_in_core(bench):
    assert bench.name == "ave" and bench.spec.readout == "seq"
    assert len(bench.protocol_hash) == 64 and len(bench.manifest_hash) == 64
    assert bench.n_rows("train") == 2624 and bench.n_rows("test") == 316
    v = bench.view()
    assert v.facts == FACTS and v.streams == {"video": "video", "audio": "audio"}


def test_metric_names_resolve(spec, bench):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    assert set(m["list"]) | set(m["floors"]) | set(m["controls"]) == set(bench.metric_specs)


def test_folder_ships_no_reader_py():
    """The folder ships no reader.py: ``Benchmark.reader()`` falls back to the generic
    ``yavih.sample.iter_samples``, which hands over what benchmark.yaml declares; the reader smokes in this
    file go through ``bench.reader``."""
    assert not (HERE / "reader.py").exists()


def _write_upstream(root: Path, rows: list[tuple[str, str, int, int]], splits: dict[str, list[int]]) -> None:
    """A synthetic AVE root: upstream/Annotations.txt, upstream/{train,val,test}_order.h5 and index.json."""
    import json

    h5py = pytest.importorskip("h5py")
    (root / "upstream").mkdir(parents=True)
    lines = ["Category&VideoID&Quality&StartTime&EndTime"] + [f"{c}&{y}&Good&{a}&{b}" for c, y, a, b in rows]
    (root / "upstream" / "Annotations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for split, order in splits.items():
        with h5py.File(root / "upstream" / f"{split}_order.h5", "w") as fh:
            fh["order"] = np.asarray(order, dtype=np.int64)
    members = {f"video/{y}.mp4": [0, 512 * k, 100] for k, y in enumerate(sorted({r[1] for r in rows}))}
    (root / "index.json").write_text(json.dumps({"version": 1, "shards": ["AVE_Datset.tar"], "members": members}), encoding="utf-8")


def test_make_manifest_on_synthetic_upstream(tmp_path, monkeypatch):
    """The freeze rules on a tiny fixture: Annotations.txt order, both leakage filters, sorted classes + background."""
    import csv
    import gzip

    mod = _load("ave_make_manifest", "make_manifest.py")
    rows = [("Dog", "aaa", 0, 10), ("Cat", "bbb", 2, 5), ("Dog", "ccc", 0, 10), ("Bell", "aaa", 1, 4),
            ("Cat", "ddd", 0, 3), ("Bell", "eee", 4, 10), ("Dog", "fff", 0, 10)]
    _write_upstream(tmp_path / "ave", rows, {"train": [0, 1, 6], "val": [3], "test": [2, 4, 5]})
    vgg = tmp_path / "vggsound.csv"
    vgg.write_text("ccc,0,dog barking,train\nddd,0,cat,test\nfff,0,dog,train\n", encoding="utf-8")
    monkeypatch.setattr(mod, "TOTAL_UPSTREAM", 7)
    monkeypatch.setattr(mod, "EXPECTED_SPLIT_ROWS", {"train": 3, "val": 1, "test": 3})
    monkeypatch.setattr(mod, "EXPECTED_CATEGORIES", 3)
    monkeypatch.setattr(mod, "EXPECTED_YTIDS", 6)
    monkeypatch.setattr(mod, "BACKGROUND_INDEX", 3)
    out = tmp_path / "out"
    args = ["--media-root", str(tmp_path / "ave"), "--vggsound-csv", str(vgg), "--no-pins", "--out", str(out)]
    assert mod.main(args) == 0
    with gzip.open(out / "manifest.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    # fff (train) and ccc (test) are in VGGSound train; aaa#3 (val) repeats a train video
    assert [(r["clip_id"], r["split"], r["label"]) for r in got] == [
        ("aaa#0", "train", "Dog"), ("bbb#1", "train", "Cat"), ("ddd#4", "test", "Cat"), ("eee#5", "test", "Bell")]
    assert got[0]["rel_media"] == "video/aaa.mp4" and (got[1]["event_start"], got[1]["event_end"]) == ("2", "5")
    assert (out / "classes.csv").read_text(encoding="utf-8").splitlines() == ["index,name", "0,Bell", "1,Cat", "2,Dog", "3,background"]
    assert mod.main([*args, "--check"]) == 0
    with gzip.open(out / "manifest.csv.gz", "wb") as fh:
        fh.write(b"clip_id\n")
    assert mod.main([*args, "--check"]) == 1


def test_member_set_pin_ignores_the_shard_path():
    mod = _load("ave_make_manifest", "make_manifest.py")
    members = {"video/b.mp4": [0, 1024, 7], "video/a.mp4": [0, 512, 3]}
    assert mod.member_set_sha256(members) == mod.member_set_sha256({k: [3, *v[1:]] for k, v in reversed(members.items())})
    assert mod.member_set_sha256(members) != mod.member_set_sha256({**members, "video/a.mp4": [0, 512, 4]})


def test_make_manifest_check_mode():
    """``make_manifest.py --check`` re-derives the committed files byte for byte from the public upstream files."""
    mod = _load("ave_make_manifest", "make_manifest.py")
    root = F.site_root(HERE, "media_root", needs=("upstream/Annotations.txt", "index.json"))
    vgg = mod.default_vggsound_csv()
    if root is None or vgg is None or not vgg.is_file():
        pytest.skip("staged AVE upstream/ + index.json or VGGSound metadata/vggsound.csv not on this machine")
    pytest.importorskip("h5py")
    assert mod.main(["--check", "--media-root", str(root), "--vggsound-csv", str(vgg)]) == 0


def _has_encoder(av, name: str) -> bool:
    try:
        av.codec.Codec(name, "w")
        return True
    except Exception:  # noqa: BLE001
        return False


def _write_ave_mp4(path: Path, seconds: float = 10.0, audio_seconds: float | None = None, fps: int = 25,
                   rate: int = 44100, size: tuple[int, int] = (32, 48)) -> None:
    """Tiny AVE-shaped mp4: ``fps`` video (red channel = frame index) + mono AAC at ``rate`` Hz."""
    import av

    vcodec = next((c for c in ("libx264", "mpeg4") if _has_encoder(av, c)), None)
    if vcodec is None or not _has_encoder(av, "aac"):
        pytest.skip("PyAV build without an mp4 video encoder or aac")
    h, w = size
    with av.open(str(path), "w", format="mp4") as out:
        vs = out.add_stream(vcodec, rate=fps)
        vs.width, vs.height = w, h
        vs.pix_fmt = "yuv420p"
        aus = out.add_stream("aac", rate=rate)
        aus.layout = "mono"
        for i in range(int(round(seconds * fps))):
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            arr[..., 0] = i % 256
            for pkt in vs.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                out.mux(pkt)
        for pkt in vs.encode(None):
            out.mux(pkt)
        n = int(round((audio_seconds or seconds) * rate))
        wave = (0.3 * np.sin(2 * np.pi * 440.0 * np.arange(n) / rate)).astype(np.float32)
        for k in range(0, n, 1024):
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(wave[k : k + 1024][None, :]), format="fltp", layout="mono")
            fr.sample_rate = rate
            fr.pts = k
            for pkt in aus.encode(fr):
                out.mux(pkt)
        for pkt in aus.encode(None):
            out.mux(pkt)


def test_reader_smoke_on_synthetic_raw_files(tmp_path):
    """Reader smoke: two synthetic AVE mp4s packed as ``AVE_Dataset/AVE/<ytid>.mp4`` in a tar,
    indexed with ``prepare/03_index_tar.index_tar``; the reader hands over the stored member bytes as both
    streams, the guaranteed facts, and ``meta == {"facts": …}`` — nothing decoded, no window."""
    import json
    import tarfile

    pytest.importorskip("av")
    from yavih import Benchmark

    idx = _load("ave_index_tar", "prepare/03_index_tar.py")
    b = Benchmark.from_dir(HERE, data={"media_root": str(tmp_path)})
    man = b.manifest("train")
    ytids = [man.media[k]["video"].removeprefix("video/").removesuffix(".mp4") for k in (0, 1)]
    staging = tmp_path / "staging"
    staging.mkdir()
    tar_path = tmp_path / idx.TAR_FILENAME
    stored = {}
    with tarfile.open(tar_path, "w") as tar:
        tar.add(staging, arcname="AVE_Dataset/AVE")  # directory member: counted and skipped
        for k, y in enumerate(ytids):
            mp4 = staging / f"{y}.mp4"
            _write_ave_mp4(mp4, audio_seconds=9.73 if k == 0 else None, size=(32, 48) if k == 0 else (36, 48))
            stored[y] = mp4.read_bytes()
            tar.add(mp4, arcname=f"AVE_Dataset/AVE/{y}.mp4")
    members, stats = idx.index_tar(tar_path)
    assert sorted(members) == sorted(f"video/{y}.mp4" for y in ytids) and stats["directories"] == 1
    (tmp_path / "index.json").write_text(
        json.dumps({"version": 1, "shards": [idx.TAR_FILENAME], "members": members}, separators=(",", ":")), encoding="utf-8"
    )
    spec = yaml.safe_load((HERE / "benchmark.yaml").read_text(encoding="utf-8"))
    samples = list(b.reader("train", rows=range(0, 2)))
    assert [s.row for s in samples] == [0, 1]
    for s, y in zip(samples, ytids, strict=True):
        assert not hasattr(s, "media_key"), "Sample has no media_key"
        assert s.meta == {"facts": FACTS}
        assert set(s.streams) == {"video", "audio"}
        v, a = s.streams["video"], s.streams["audio"]
        assert (v.kind, a.kind) == ("video", "audio")
        assert v.bytes() == a.bytes() == stored[y], "both streams hand over the stored member byte for byte"
        assert v.open().read() == stored[y]
        for st, keys in ((v, spec["input_contract"]["info"]["video"]), (a, spec["input_contract"]["info"]["audio"])):
            missing = [k for k in keys if st.info.get(k) is None]
            assert not missing, (st.name, missing)
        assert v.info["width"] == 48 and v.info["fps"] == pytest.approx(25.0) and a.info["sample_rate"] == 44100
        assert v.info["bytes"] == a.info["bytes"] == len(stored[y])
    assert samples[0].streams["audio"].info["duration_s"] == pytest.approx(9.73, abs=0.05)
    assert samples[1].streams["audio"].info["duration_s"] == pytest.approx(10.0, abs=0.05)


def test_reader_smoke_if_media_present(bench, spec):
    """First rows of each read split through the folder reader against the real tar (skips elsewhere)."""
    root = F.site_root(HERE, "media_root")
    if root is None:
        pytest.skip("ave media_root not configured or not present on this host")
    if not (root / "index.json").is_file() or not (root / "AVE_Datset.tar").is_file():
        pytest.skip(f"AVE tar/index not present under {root}")
    for split in ("train", "test"):
        samples = list(bench.reader(split, rows=range(0, 2)))
        assert [s.row for s in samples] == [0, 1]
        s = samples[0]
        assert set(s.streams) == {"video", "audio"} and s.meta == {"facts": FACTS}
        data = s.streams["video"].bytes()
        assert data == s.streams["audio"].bytes() and data[4:8] == b"ftyp"
        for name in ("video", "audio"):
            info = s.streams[name].info
            assert all(info.get(k) is not None for k in spec["input_contract"]["info"][name]), (name, info)
        assert 9.7 <= s.streams["audio"].info["duration_s"] <= 10.1 and s.streams["audio"].info["sample_rate"] == 44100
