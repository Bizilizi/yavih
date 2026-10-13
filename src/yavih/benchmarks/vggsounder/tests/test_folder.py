"""Folder tests for vggsounder.

Run only this file: ``python -m pytest "$(yavih bench path vggsounder)/tests" -q -p no:cacheprovider``.

Pure checks (yaml, manifest, classes, docs, probe.py contract) run anywhere. Tests that load the folder
through core (``Benchmark``, ``RawStream``, ``yavih.media.rules``) always run; the reader and bridge smokes
run on synthetic raw files (tiny mp4 + wav), never on site media.
Tests that call the official scorer skip unless ``vggsounder==1.0`` (the PyPI release) is installed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_rows, classes = F.spec, F.rows, F.classes

SETS = {
    "audible": ("labels_a", "labels_av"),
    "visible": ("labels_v", "labels_av"),
    "all": ("labels_a", "labels_v", "labels_av"),
}
VGGSOUNDER_CSV_SHA = "400979ea3dee34bed145d3fd96966e16689352df63b645c4730caf8eb2cacb06"


def _load(name: str, path: Path):
    return F.load_module(path, name)


def _cells(cell: str) -> list[str]:
    return cell.split(";") if cell else []


@pytest.fixture(scope="module")
def probe_mod():
    return _load("vggsounder_folder_probe", HERE / "probe.py")


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


# ----------------------------------------------------------------------------- yaml


def test_yaml_identity(spec):
    assert spec["schema_version"] == 2 and spec["name"] == HERE.name == "vggsounder"
    assert spec["readout"] == "clf" and spec["protocol_version"] == 1
    assert spec["manifest"]["label"]["kind"] == "multi_sets"
    assert spec["manifest"]["label"]["sets"] == {k: list(v) for k, v in SETS.items()}
    (entry,) = spec["changelog"]
    assert entry["version"] == spec["protocol_version"] and entry["invalidates"] is False
    assert entry["supersedes"] == []


def test_protocol_keys(spec):
    # protocol holds label facts only, each consumed: generic keys, nothing that constrains the bridge
    assert set(spec["protocol"]) == {"clip_seconds", "train_draw", "views"}
    assert spec["protocol"]["train_draw"] == {
        **spec["protocol"]["train_draw"],
        "cap": 20000,
        "seed": 0,
        "balance": "class",
    }


def test_input_contract(spec):
    ic = spec["input_contract"]
    assert set(ic) == {"handed_over", "info", "facts", "materialize"}
    assert set(ic["handed_over"]) == set(ic["info"]) == set(spec["streams"]) == {"video", "audio"}
    assert ic["info"]["video"] == ["duration_s", "fps", "width", "height", "container", "bytes"]
    assert ic["info"]["audio"] == [
        "duration_s",
        "sample_rate",
        "n_samples",
        "channels",
        "container",
        "bytes",
    ]
    assert ic["facts"] == [] and ic["materialize"] is None
    assert {k: v["kind"] for k, v in spec["streams"].items()} == {
        "video": "video",
        "audio": "audio",
    }
    for s in spec["streams"].values():
        assert s["layout"] == "dir" and s["root"] == "media_root" and s["required"] is True


def test_output_contract(spec):
    oc = spec["output_contract"]
    assert oc == {
        "shape": "fixed",
        "coordinates": "media",
        "target_frame": None,
        "requires": {"pos": [], "streams": [], "grid": None, "t_bins": None},
    }
    assert spec["targets"] is None and spec["readout_params"] == {}


def test_external_entry(spec, probe_mod):
    (e,) = spec["external"]
    assert set(e) == {"name", "pip", "version_pin", "used_for"}
    assert e["name"] == "vggsounder" and e["version_pin"] == "1.0"
    assert e["pip"] == "pip install --no-deps vggsounder==1.0"
    assert probe_mod.external_entry() == {k: str(v) for k, v in e.items()}
    assert (HERE / "probe.py").is_file()  # non-empty external requires probe.py


# ----------------------------------------------------------------------------- manifest (setup.md)


def test_sha256_and_counts(spec, manifest_rows):
    F.assert_manifest_hashes(HERE, spec)
    assert len(manifest_rows) == 32008


def test_splits_assigned_once(spec, manifest_rows):
    assert F.assert_splits_assigned_once(spec, manifest_rows) == {"train": 20000, "test": 12008}


def test_classes_contiguous_vendored_order(classes):
    assert list(classes[0].keys()) == ["index", "name", "mid"]
    assert [int(r["index"]) for r in classes] == list(range(309))
    assert len({r["name"] for r in classes}) == 309 and len({r["mid"] for r in classes}) == 309
    assert classes[0]["name"] == "ambulance siren" and classes[308]["name"] == "air horn"
    assert [r["name"] for r in classes] != sorted(r["name"] for r in classes)
    assert classes[7]["mid"] == "vgg_7"


def test_label_cells(manifest_rows, classes):
    names = {r["name"] for r in classes}
    for r in manifest_rows:
        if r["split"] == "train":
            assert r["label"] in names and not (r["labels_a"] or r["labels_v"] or r["labels_av"])
        else:
            assert not r["label"] and (r["labels_a"] or r["labels_v"] or r["labels_av"])
            toks = [t for c in ("labels_a", "labels_v", "labels_av") for t in _cells(r[c])]
            assert len(set(toks)) == len(toks), r["clip_id"]  # disjoint A / V / AV partition
            assert set(toks) <= names
        assert (
            r["rel_video"] == f"video/{r['clip_id']}.mp4"
            and r["rel_audio"] == f"audio/{r['clip_id']}.wav"
        )
    train_y = {r["clip_id"].rsplit("_", 1)[0] for r in manifest_rows if r["split"] == "train"}
    eval_y = {r["clip_id"].rsplit("_", 1)[0] for r in manifest_rows if r["split"] == "test"}
    assert not (train_y & eval_y)  # leakage is a freeze-time fact


def test_train_draw_is_class_balanced(manifest_rows):
    counts = Counter(r["label"] for r in manifest_rows if r["split"] == "train")
    assert len(counts) == 309 and min(counts.values()) == 64 and max(counts.values()) == 65


def test_eval_set_populations_and_chance(spec, manifest_rows, classes):
    """The official scorer scores each set over the clips holding a label of it; chance = mean class
    prevalence over that population (= AP of a constant score), as benchmark.yaml records."""
    idx = {r["name"]: i for i, r in enumerate(classes)}
    ev = [r for r in manifest_rows if r["split"] == "test"]
    pops = {}
    for name, cols in SETS.items():
        y = np.zeros((len(ev), 309), bool)
        for i, r in enumerate(ev):
            for c in cols:
                for t in _cells(r[c]):
                    y[i, idx[t]] = True
        pop = y[y.any(axis=1)]
        pops[name] = len(pop)
        assert (pop.sum(0) > 0).all()  # every class has a positive in every set
        assert round(float(pop.mean(0).mean()), 5) == spec["metrics"]["chance"][f"auc_pr_{name}"]
        assert spec["metrics"]["chance"][f"auc_roc_{name}"] == 0.5
    assert pops == {"audible": 11911, "visible": 10182, "all": 12008}


def test_curation_sums(spec, manifest_rows):
    cur = spec["manifest"]["upstream"]["curation"]
    assert cur["leakage_audit"]["excluded"] == 6 == cur["excluded"]["leakage"]
    assert cur["total_upstream"] == 12014 + 183730
    mm = _load("vggsounder_make_manifest_consts", HERE / "make_manifest.py")
    assert cur["short_clips"] == mm.SHORT_CLIPS_MEASURED == 2350
    assert cur["excluded"]["leakage"] == mm.LEAKS and cur["total_upstream"] == mm.PACKAGE_VIDEOS + 183730
    up = spec["manifest"]["upstream"]
    assert up["sha256"] == mm.VGGSOUNDER_CSV_SHA and up["source"] == mm.UPSTREAM_SOURCE
    assert up["upstream_files"] == {
        "vggsounder/data/vggsounder.csv": mm.VGGSOUNDER_CSV_SHA,
        "vggsounder/data/classes.csv": mm.CLASSES_CSV_SHA,
        "vggsound/metadata/vggsound.csv": mm.VGGSOUND_CSV_SHA,
    }
    assert spec["source"]["sha256"]["vggsound/metadata/vggsound.csv"] == mm.VGGSOUND_CSV_SHA
    assert mm.VGGSOUND_CSV_SHA[:8] in spec["protocol"]["train_draw"]["drawn_from"]


def test_metric_names_resolve(spec, probe_mod):
    F.assert_metric_names_resolve(HERE, spec)
    names = [*spec["metrics"]["list"], *spec["metrics"]["floors"], *spec["metrics"]["controls"]]
    assert set(probe_mod.METRICS) == {n for n in names if n.startswith("auc_")}
    assert spec["metrics"]["headline"] == "auc_pr_all"
    assert spec["metrics"]["ci"] == {} and spec["metrics"]["controls"] == []


# ----------------------------------------------------------------------------- docs


def test_readme_facts():
    """Folder facts in README.md (the README structure is ``yavih bench lint``'s)."""
    text = (HERE / "README.md").read_text(encoding="utf-8")
    cite = F.section(text, 2, "Citation")
    assert re.search(r"@inproceedings\{\w+2025vggsounder,", cite) and "@inproceedings{chen2020vggsound," in cite


def test_bridge_template_describe(spec):
    pytest.importorskip("av")
    pytest.importorskip("soundfile")
    mod = _load("vggsounder_bridge_template", HERE / "bridge_template.py")
    from yavih.bridge import BenchmarkView

    view = BenchmarkView(
        name="vggsounder",
        aliases=[],
        readout="clf",
        protocol_version=1,
        protocol_hash="0" * 64,
        streams={"video": "video", "audio": "audio"},
        input_contract=spec["input_contract"],
        output_contract=spec["output_contract"],
        facts={},
    )
    for reading, streams in (("a", ["audio"]), ("v", ["video"]), ("av", ["video", "audio"])):
        b = mod.VGGSounderBridge(view, {"reading": reading})
        d = b.describe()
        assert d["streams"] == streams and sorted(d["input"]) == sorted(streams)
        assert "decode" not in d and d["views"] == 1 and d["reading"] == reading
        assert d["args"] == {"reading": reading, "layer": -1}
        assert all(v is not None for k, v in d["precision"].items() if k != "autocast")
        json.dumps(d)  # JSON primitives only
        for s in streams:
            assert {"temporal", "norm", "decoder"} <= set(d["input"][s])
            assert "source" in d["input"][s]["norm"]
    assert mod.VGGSounderBridge.pos_mode == "per_sample"
    assert not hasattr(mod.VGGSounderBridge, "decode_plan")


def test_make_manifest_draw():
    mod = _load("vggsounder_make_manifest", HERE / "make_manifest.py")
    pytest.importorskip("torch")
    labels = ["b", "a", "b", "a", "c", "c", "b"]
    keep = mod.class_balanced_subsample(labels, 4, 0)
    assert keep == sorted(keep) and len(keep) == 4
    assert mod.class_balanced_subsample(labels, 0, 0) == list(range(7))
    assert mod.class_balanced_subsample(labels, 99, 0) == list(range(7))
    assert max(Counter(labels[i] for i in keep).values()) <= 2


def _synthetic_public_inputs(tmp: Path) -> tuple[Path, Path, dict[str, str]]:
    """A tiny package ``data/`` (3 classes, 4 videos, one a VGGSound-train leak) and a ``vggsound.csv``
    (6 train rows, 2 test rows); returns the paths and their sha256s."""
    data = tmp / "pkg_data"
    data.mkdir()
    (data / "classes.csv").write_text('index,mid,display_name\n0,vgg_0,"dog"\n1,vgg_1,"cat, meowing"\n2,vgg_2,"rain"\n')
    rows = [
        ("aaaaaaaaaa1_000010", "dog", "A"), ("aaaaaaaaaa1_000010", "cat, meowing", "AV"),
        ("bbbbbbbbbb2_000000", "rain", "V"),
        ("leakleakle3_000030", "dog", "AV"),
        ("cccccccccc4_000005", "cat, meowing", "A"), ("cccccccccc4_000005", "rain", "A"),
    ]
    buf = "video_id,label,modality,background_music,static_image,voice_over\n" + "".join(
        f'{v},"{lab}",{m},False,False,False\n' for v, lab, m in rows)
    (data / "vggsounder.csv").write_text(buf)
    listing = tmp / "vggsound.csv"
    train = [("t1", 1, "dog"), ("t2", 2, "rain"), ("t3", 3, "dog"), ("t4", 4, "cat, meowing"), ("leakleakle3", 7, "rain"), ("t6", 6, "rain")]
    lines = [f'{y},{s},"{lab}",train' for y, s, lab in train] + ['aaaaaaaaaa1,10,dog,test', 'bbbbbbbbbb2,0,rain,test']
    listing.write_text("\n".join(lines) + "\n")
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (data / "classes.csv", data / "vggsounder.csv", listing)}
    return data, listing, sha


def test_make_manifest_public_route_on_synthetic_inputs(tmp_path, monkeypatch):
    """The freeze reads only the package CSVs and vggsound.csv: eval = package videos in file order minus
    VGGSound-train ytids, labels split by modality; train = the seeded draw over split == train rows."""
    pytest.importorskip("torch")
    mm = _load("vggsounder_make_manifest_synth", HERE / "make_manifest.py")
    data, listing, sha = _synthetic_public_inputs(tmp_path)
    monkeypatch.setattr(mm, "CLASSES_CSV_SHA", sha["classes.csv"])
    monkeypatch.setattr(mm, "VGGSOUNDER_CSV_SHA", sha["vggsounder.csv"])
    monkeypatch.setattr(mm, "VGGSOUND_CSV_SHA", sha["vggsound.csv"])
    monkeypatch.setattr(mm, "NUM_CLASSES", 3)
    monkeypatch.setattr(mm, "PACKAGE_VIDEOS", 4)
    monkeypatch.setattr(mm, "LEAKS", 1)
    out = mm.freeze(data, listing, None, cap=3, seed=0)
    text = out["manifest_bytes"].decode()
    lines = text.splitlines()
    assert lines[0] == ",".join(mm.MANIFEST_COLUMNS)
    train = [ln.split(",")[0] for ln in lines[1:] if ln.split(",")[1] == "train"]
    all_train = {"t1_000001", "t2_000002", "t3_000003", "t4_000004", "leakleakle3_000007", "t6_000006"}
    assert len(train) == 3 and set(train) <= all_train
    assert lines[-3:] == [
        'aaaaaaaaaa1_000010,test,,dog,,"cat, meowing",video/aaaaaaaaaa1_000010.mp4,audio/aaaaaaaaaa1_000010.wav',
        "bbbbbbbbbb2_000000,test,,,rain,,video/bbbbbbbbbb2_000000.mp4,audio/bbbbbbbbbb2_000000.wav",
        'cccccccccc4_000005,test,,"cat, meowing;rain",,,video/cccccccccc4_000005.mp4,audio/cccccccccc4_000005.wav',
    ]
    assert out["classes_bytes"] == b'index,name,mid\n0,dog,vgg_0\n1,"cat, meowing",vgg_1\n2,rain,vgg_2\n'
    cur = out["report"]["upstream"]["curation"]
    assert cur["total_upstream"] == 4 + 6 and cur["excluded"] == {"leakage": 1, "draw_remainder": 3, "train_unknown_label": 0}
    assert cur["per_split"]["train"]["staged"] is None  # no media root: presence not checked
    again = mm.freeze(data, listing, None, cap=3, seed=0)
    assert again["manifest_bytes"] == out["manifest_bytes"]  # deterministic
    monkeypatch.setattr(mm, "VGGSOUND_CSV_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="sha256"):
        mm.freeze(data, listing, None, cap=3, seed=0)


def test_make_manifest_check_against_public_files(monkeypatch):
    """``make_manifest.py --check`` reproduces the committed manifest and classes byte for byte (needs the
    vggsounder package, the configured VGGSound root's vggsound.csv and torch; skips otherwise)."""
    pytest.importorskip("torch")
    pytest.importorskip("vggsounder")
    mm = _load("vggsounder_make_manifest_check", HERE / "make_manifest.py")
    root = mm.default_root()
    if root is None or not (root / "metadata" / "vggsound.csv").is_file():
        pytest.skip("VGGSound root with metadata/vggsound.csv not configured")
    out = mm.freeze(mm.package_data_dir(), root / "metadata" / "vggsound.csv", None, mm.CAP, mm.SEED)
    from yavih.freeze import gzip_bytes

    assert gzip_bytes(out["manifest_bytes"]) == (HERE / "manifest.csv.gz").read_bytes()
    assert out["classes_bytes"] == (HERE / "classes.csv").read_bytes()


# ----------------------------------------------------------------------------- probe.py


def test_probe_imports_without_torch_and_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "vggsounder", None)
    mod = _load("vggsounder_probe_isolated", HERE / "probe.py")
    assert mod.SET_TO_PACKAGE == {"audible": "a", "visible": "v", "all": "all"}
    assert mod.PROTOCOL_KEYS == frozenset()
    for name, m in mod.METRICS.items():
        assert m.name == name and m.readouts == frozenset({"clf"}) and m.shape == "scalar"
        assert m.kind == ("floor" if name.endswith("_floor") else "metric")
        if hasattr(m, "external"):
            assert m.external == "vggsounder"
        if hasattr(m, "floor_of") and name.endswith("_floor"):
            assert m.floor_of == (name.removesuffix("_floor"),)
        with pytest.raises(RuntimeError, match="official vggsounder package"):
            m.fn(None, None, None, None)
    # the assert-import: absent package → not ready, message carries the pip line
    with pytest.raises(Exception, match=re.escape("pip install --no-deps vggsounder==1.0")) as exc:
        mod.require_vggsounder()
    assert getattr(exc.value, "exit_code", 3) == 3


def test_require_is_the_core_assert_import_with_the_version_pin(monkeypatch, probe_mod):
    """The authors' implementation through an assert-import: the one assert-import is yavih.external.require (no folder-local fallback)."""
    import importlib.metadata

    from yavih import external

    assert not hasattr(probe_mod, "_fallback_require")
    external.reset()
    monkeypatch.setitem(sys.modules, "vggsounder", types.ModuleType("vggsounder"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0.0")
    with pytest.raises(external.ExternalVersion, match="E_EXTERNAL_VERSION"):
        probe_mod.require_vggsounder()
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0")
    assert probe_mod.require_vggsounder() == "1.0" and external.versions_used() == {"vggsounder": "1.0"}
    external.reset()


_READING_INPUTS = {"a": ["audio"], "v": ["video"], "av": ["audio", "video"]}


def _fake_result(tmp: Path, name: str, ph: str, ids_sha: str, n: int, rng, *, reads=None, model: str = "m",
                 weights: str | None = "w" * 64, probe_hash: str = "q" * 64, standard: bool = True,
                 eval_hash: str | None = None) -> Path:
    d = tmp / name / "seed0"
    d.mkdir(parents=True)
    routing = name.split("_")[0].rstrip("0123456789")
    inputs = reads if reads is not None else _READING_INPUTS.get(routing, ["audio", "video"])
    rec = {
        "status": "ok",
        "standard": standard,
        "benchmark": {"name": "vggsounder", "protocol_hash": ph},
        "coverage": {"eval": {"ids_sha256": ids_sha}},
        "eval_hash": eval_hash or hashlib.sha256(f"{ph}:{probe_hash}".encode()).hexdigest(),
        "model": model,
        "reading": routing,
        "bridge": {"describe": {"model": model, "reading": routing, "weights_sha256": weights,
                                "input": {s: {} for s in inputs}},
                   "describe_sha256": hashlib.sha256(name.encode()).hexdigest()},
        "probe": {"hash": probe_hash},
    }
    (d / "result.json").write_text(json.dumps(rec))
    (d / "DONE").write_text("")
    np.save(d / "eval_logits.npy", rng.standard_normal((n, 309)).astype(np.float32))
    np.save(d / "eval_rows.npy", np.arange(n, dtype=np.int64))
    return d


class _StubBench:
    protocol_hash = "p" * 64

    def __init__(self, ids):
        self._ids = ids

    def roles(self):
        return {"eval": ["test"]}

    def manifest(self, split):
        return types.SimpleNamespace(clip_id=self._ids)


def test_join_refusals(tmp_path, probe_mod):
    rng = np.random.default_rng(0)
    bench = _StubBench([f"c{i}" for i in range(4)])
    ph = bench.protocol_hash
    a = _fake_result(tmp_path, "a", ph, "s", 4, rng)
    v = _fake_result(tmp_path, "v", ph, "s", 4, rng)
    av_other = _fake_result(tmp_path, "av", "q" * 64, "s", 4, rng)
    with pytest.raises(probe_mod.JoinRefused, match="E_JOIN_MISSING"):
        probe_mod.join(bench, {"a": a, "v": v})
    with pytest.raises(probe_mod.JoinRefused, match="E_JOIN_MIXED"):
        probe_mod.join(bench, {"a": a, "v": v, "av": av_other})
    av_pop = _fake_result(tmp_path, "av2", ph, "t", 4, rng)
    with pytest.raises(probe_mod.JoinRefused, match="E_JOIN_POPULATION"):
        probe_mod.join(bench, {"a": a, "v": v, "av": av_pop})
    av = _fake_result(tmp_path, "av_ok", ph, "s", 4, rng)
    # swapped routing: the video-only reading passed as a=, the audio-only one as v=
    with pytest.raises(probe_mod.JoinRefused, match="E_JOIN_ROUTING"):
        probe_mod.join(bench, {"a": v, "v": a, "av": av})
    # an "a" reading whose bridge also read the video
    a_av = _fake_result(tmp_path, "a_impure", ph, "s", 4, rng, reads=["audio", "video"])
    with pytest.raises(probe_mod.JoinRefused, match="E_JOIN_ROUTING"):
        probe_mod.join(bench, {"a": a_av, "v": v, "av": av})
    # two model names (E_JOIN_MODEL)
    v_other = _fake_result(tmp_path, "v_model", ph, "s", 4, rng, model="other")
    with pytest.raises(probe_mod.JoinRefused, match="E_JOIN_MODEL"):
        probe_mod.join(bench, {"a": a, "v": v_other, "av": av})
    # one protocol and probe but another eval_hash (eval_hash is the comparability key)
    v_eval = _fake_result(tmp_path, "v_eval", ph, "s", 4, rng, eval_hash="e" * 64)
    with pytest.raises(probe_mod.JoinRefused, match="E_JOIN_MIXED"):
        probe_mod.join(bench, {"a": a, "v": v_eval, "av": av})
    # one reading probed with another / a non-standard ProbeConfig
    v_probe = _fake_result(tmp_path, "v_probe", ph, "s", 4, rng, probe_hash="z" * 64)
    with pytest.raises(probe_mod.JoinRefused, match="E_JOIN_PROBE"):
        probe_mod.join(bench, {"a": a, "v": v_probe, "av": av})
    v_fast = _fake_result(tmp_path, "v_fast", ph, "s", 4, rng, standard=False)
    with pytest.raises(probe_mod.JoinRefused, match="E_JOIN_PROBE"):
        probe_mod.join(bench, {"a": a, "v": v_fast, "av": av})
    (a / "DONE").unlink()
    with pytest.raises(probe_mod.JoinRefused, match="DONE"):
        probe_mod.join(bench, {"a": a, "v": v, "av": av_pop})


def test_join_weights_differ_is_a_warning_not_a_refusal(tmp_path, probe_mod):
    """Different non-null ``weights_sha256`` among the three readings are ``W_JOIN_WEIGHTS``
    (recorded in the join output), never a refusal; a null fingerprint is not a difference."""
    rng = np.random.default_rng(1)
    ph = _StubBench.protocol_hash
    recs = {k: json.loads((_fake_result(tmp_path, k, ph, "s", 4, rng) / "result.json").read_text()) for k in ("a", "v", "av")}
    assert probe_mod._check_readings(recs) == []
    recs["a"]["bridge"]["describe"]["weights_sha256"] = None  # an unfingerprinted reading: no difference to report
    assert probe_mod._check_readings(recs) == []
    recs["v"]["bridge"]["describe"]["weights_sha256"] = "x" * 64
    warnings = probe_mod._check_readings(recs)
    assert [w["code"] for w in warnings] == ["W_JOIN_WEIGHTS"] and "m" in warnings[0]["message"]
    assert set(warnings[0]) == {"code", "message", "where"}


def test_join_reads_the_handed_over_record(tmp_path, probe_mod):
    """``report --join`` hands ``ResultRef``s whose ``.result`` is upgraded in memory; the hook reads
    that record, not the file; a record without a top-level ``model`` names it in ``describe()``."""
    rng = np.random.default_rng(2)
    ph = _StubBench.protocol_hash
    d = _fake_result(tmp_path, "a", ph, "s", 4, rng)
    upgraded = {**json.loads((d / "result.json").read_text()), "model": "from-ref"}
    ref = types.SimpleNamespace(seed_dir=d, result=upgraded)
    assert probe_mod._record(ref, d)["model"] == "from-ref"
    assert probe_mod._record(d, d)["model"] == "m"
    assert probe_mod._model({"bridge": {"describe": {"model": "x", "reading": "a"}}}) == "x"


def test_package_ground_truth_equals_label_sets(manifest_rows):
    """setup.md: the package's a / v / all ground truth equals the manifest's audible / visible / all sets
    on the eval population (and its AV-only `av` set is labels_av alone, which is why `all` maps to `all`)."""
    labels_mod = pytest.importorskip("vggsounder.labels")
    import vggsounder

    data = Path(vggsounder.__file__).parent / "data" / "vggsounder.csv"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == VGGSOUNDER_CSV_SHA
    ev = [r for r in manifest_rows if r["split"] == "test"]
    ids = {r["clip_id"] for r in ev}
    vs = labels_mod.VGGSounder(background_music=False, download_samples=False)
    assert len(vs) == 12014 and ids <= set(vs.keys())
    mine = {
        name: {r["clip_id"]: {t for c in cols for t in _cells(r[c])} for r in ev}
        for name, cols in SETS.items()
    }
    mine["av_only"] = {r["clip_id"]: set(_cells(r["labels_av"])) for r in ev}
    for name, key in (("audible", "a"), ("visible", "v"), ("all", "all"), ("av_only", "av")):
        vs.set_modality(key)
        pkg = {v.video_id: set(v.labels) for v in vs if v.video_id in ids}
        assert pkg == {k: s for k, s in mine[name].items() if s}, name


def _pinned_package_or_skip(probe_mod):
    try:
        return probe_mod.require_vggsounder()
    except Exception as exc:  # absent or not the PyPI 1.0 release
        pytest.skip(f"official scorer not available at its pin: {exc}")


def test_score_with_official_package(manifest_rows, spec, probe_mod):
    _pinned_package_or_skip(probe_mod)
    ev_ids = [r["clip_id"] for r in manifest_rows if r["split"] == "test"]
    bench = _StubBench(ev_ids)
    rows = np.arange(len(ev_ids))
    out = probe_mod.score(
        bench, "test", rows, np.random.default_rng(0).standard_normal((len(ev_ids), 309))
    )
    assert set(out) == {n for n in spec["metrics"]["list"] if n.startswith("auc_")}
    for k, v in out.items():
        assert (0.4 < v < 0.6) if "roc" in k else (0.0 < v < 0.02), (k, v)
    floors = probe_mod.score_floors(bench, "test", rows, np.full(309, 1 / 309))
    for s in SETS:
        assert round(floors[f"auc_pr_{s}_floor"], 5) == spec["metrics"]["chance"][f"auc_pr_{s}"]


# ----------------------------------------------------------------------------- core-backed


def _write_wav(path: Path, seconds: float, sr: int = 16000) -> None:
    import wave

    n = int(round(seconds * sr))
    t = np.arange(n) / sr
    x = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())


def _write_mp4(path: Path, n_frames: int, fps: int, h: int, w: int) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
        s = c.add_stream(codec, rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i in range(n_frames):
            img = np.full((h, w, 3), (i * 7) % 255, np.uint8)
            img[: h // 2, : w // 2] = 255 - img[: h // 2, : w // 2]
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


@pytest.fixture()
def synthetic_bench(tmp_path, monkeypatch, manifest_rows):
    """The packaged folder over a synthetic media root: rows 0, 1 of each split get a tiny mp4 + wav
    (one landscape 10 s clip, one short portrait clip with a 1.5 s wav)."""
    pytest.importorskip("av")
    pytest.importorskip("soundfile")
    root = tmp_path / "vggsound"
    for split in ("train", "test"):
        rows = [r for r in manifest_rows if r["split"] == split][:2]
        _write_mp4(root / rows[0]["rel_video"], 40, 4, 48, 64)  # 10 s landscape
        _write_wav(root / rows[0]["rel_audio"], 10.0)
        _write_mp4(root / rows[1]["rel_video"], 18, 12, 64, 36)  # 1.5 s portrait
        _write_wav(root / rows[1]["rel_audio"], 1.5)
    monkeypatch.setenv("YAVIH_DATA__VGGSOUNDER__MEDIA_ROOT", str(root))
    monkeypatch.setenv(
        "YAVIH_HOME", str(tmp_path / "home")
    )  # no site census: facts come from a live probe
    from yavih import Benchmark

    return Benchmark.from_dir(HERE), root


def test_bench_loads(synthetic_bench, spec):
    bench, _ = synthetic_bench
    assert bench.name == "vggsounder" and bench.spec.readout == "clf"
    assert len(bench.protocol_hash) == 64 and len(bench.manifest_hash) == 64
    assert bench.manifest_hash.startswith("da53029e")
    view = bench.view()
    assert view.streams == {"video": "video", "audio": "audio"} and view.facts == {}
    assert not hasattr(view, "manifest")


def test_reader_raw_handoff(synthetic_bench, spec):
    bench, root = synthetic_bench
    for split in ("train", "test"):
        samples = list(bench.reader(split, rows=range(0, 2)))
        assert [s.row for s in samples] == [0, 1]
        for s in samples:
            assert set(s.streams) == {"video", "audio"} and s.meta == {"facts": {}}
            assert not hasattr(s, "media_key")
            for name, rs in s.streams.items():
                assert rs.kind == name and rs.member is None
                ext = ".mp4" if name == "video" else ".wav"
                assert Path(rs._real_path) == root / name / f"{s.clip_id}{ext}"
                stored = Path(rs.path).read_bytes()
                with rs.open() as fh:
                    assert rs.bytes() == stored and fh.read() == stored, name
                for key in spec["input_contract"]["info"][name]:
                    assert rs.info[key] is not None, (name, key)
        v, a = samples[1].streams["video"].info, samples[1].streams["audio"].info
        assert (v["width"], v["height"]) == (36, 64) and abs(v["duration_s"] - 1.5) < 0.3
        assert a["sample_rate"] == 16000 and a["n_samples"] == 24000 and a["channels"] == 1


class _FakeRaw:
    """Duck-typed RawStream over a stored file (so the template runs without the core loader)."""

    def __init__(self, name: str, path: Path, info: dict) -> None:
        self.name = self.kind = name
        self.path, self.member, self.info = str(path), None, info

    def bytes(self) -> bytes:
        return Path(self.path).read_bytes()

    def open(self):
        return open(self.path, "rb")


def _facts(name: str, path: Path) -> dict:
    """Header facts the way prepare/04_media_facts.py reads them."""
    mf = _load("vggsounder_media_facts", HERE / "prepare" / "04_media_facts.py")
    return mf.video_facts(str(path)) if name == "video" else mf.audio_facts(str(path))


def test_bridge_template_output_contract(tmp_path, spec):
    """Output contract on synthetic raw files: fixed [N, D], pos in media coordinates inside the declared
    crop, `used` equal to what validate recomputes from the declared rules (E_RULE_MISMATCH tolerances),
    instance-wise features (bridge check's alone pass)."""
    pytest.importorskip("av")
    pytest.importorskip("soundfile")
    pytest.importorskip("PIL")
    rules = pytest.importorskip("yavih.media.rules")
    from yavih.bridge import BenchmarkView

    clips = {"c-land": ((40, 4, 48, 64), 10.0), "c-short-portrait": ((18, 12, 64, 36), 1.5)}
    samples = []
    for k, (cid, ((n, fps, h, w), secs)) in enumerate(clips.items()):
        vp, ap = tmp_path / "video" / f"{cid}.mp4", tmp_path / "audio" / f"{cid}.wav"
        _write_mp4(vp, n, fps, h, w)
        _write_wav(ap, secs)
        streams = {
            "video": _FakeRaw("video", vp, _facts("video", vp)),
            "audio": _FakeRaw("audio", ap, _facts("audio", ap)),
        }
        for name, rs in streams.items():
            for key in spec["input_contract"]["info"][name]:
                assert rs.info[key] is not None, (name, key)
        samples.append(
            types.SimpleNamespace(
                row=k, clip_id=cid, split="test", streams=streams, meta={"facts": {}}
            )
        )
    view = BenchmarkView(
        name="vggsounder",
        aliases=[],
        readout="clf",
        protocol_version=1,
        protocol_hash="0" * 64,
        streams={"video": "video", "audio": "audio"},
        input_contract=spec["input_contract"],
        output_contract=spec["output_contract"],
        facts={},
    )
    mod = _load("vggsounder_bridge_template_run", HERE / "bridge_template.py")
    b = mod.VGGSounderBridge(view, {"reading": "av"})
    feats, pos, used = F.encode_prepared(b, samples)
    feats = np.asarray(feats)
    assert (
        feats.shape == (2, 8 * 14 * 14 + mod.AUDIO_T * mod.AUDIO_F, mod.D)
        and feats.dtype == np.float16
    )
    assert np.isfinite(feats.astype(np.float32)).all()
    assert pos.shape == (2, feats.shape[1], 4) and pos.dtype == np.float32
    assert (((pos >= 0) & (pos <= 1)) | (pos == -1)).all()
    for i, s in enumerate(samples):
        vi, ai = s.streams["video"].info, s.streams["audio"].info
        times = np.asarray(used[i]["video"]["times"])
        assert (
            len(times) == mod.N_FRAMES
            and (times >= 0).all()
            and (times <= vi["duration_s"] + 1e-3).all()
        )
        want = rules.rule_times(mod.VIDEO_RULES["temporal"], vi)
        assert np.all(np.abs(times - want) <= 0.5 / vi["fps"] + 1e-3)
        ranges, pad = rules.audio_ranges(mod.AUDIO_RULES["temporal"], ai)
        assert used[i]["audio"] == {
            "samples": [[int(x) for x in r] for r in ranges],
            "pad": [int(x) for x in pad],
        }
        vid = pos[i][pos[i][:, 0] == 0]
        y0, x0, y1, x1 = rules.crop_region(mod.VIDEO_RULES["spatial"], vi["width"], vi["height"])
        assert (vid[:, 2] >= y0 - 1e-6).all() and (vid[:, 2] <= y1 + 1e-6).all()
        assert (vid[:, 3] >= x0 - 1e-6).all() and (vid[:, 3] <= x1 + 1e-6).all()
        assert ((vid[:, 1] >= 0) & (vid[:, 1] <= 1)).all()
    assert used[1]["audio"]["pad"] == [
        0,
        160000 - 24000,
    ]  # the short wav is zero-padded by the bridge, declared
    aud_short = pos[1][pos[1][:, 0] == 1]
    assert (aud_short[:, 1] == -1).any() and (aud_short[:, 2:] == -1).all()
    portrait = pos[1][pos[1][:, 0] == 0]
    # a centre crop of a tall frame keeps less of its height than of its width (positions stay full-frame)
    assert np.ptp(portrait[:, 2]) < np.ptp(portrait[:, 3]) and portrait[:, 2].min() > 0.25
    for i, s in enumerate(samples):  # instance-wise
        f1, p1, u1 = F.encode_prepared(b, [s])
        assert (
            np.array_equal(np.asarray(f1)[0], feats[i])
            and np.array_equal(p1[0], pos[i])
            and u1[0] == used[i]
        )
    f2, p2, u2 = F.encode_prepared(b, samples[::-1])  # order-invariant
    assert np.array_equal(np.asarray(f2)[::-1], feats) and u2[::-1] == used
