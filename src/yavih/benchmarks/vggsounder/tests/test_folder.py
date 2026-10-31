"""Folder tests for vggsounder.

Run only this file: ``python -m pytest "$(yavih bench path vggsounder)/tests" -q -p no:cacheprovider``.

Pure checks (yaml, manifest, classes, docs, eval.py contract) run anywhere. Tests that load the folder
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
def eval_mod():
    return _load("vggsounder_folder_eval", HERE / "eval.py")


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


# ----------------------------------------------------------------------------- yaml


def test_yaml_identity(spec):
    assert spec["schema_version"] == 3 and spec["name"] == HERE.name == "vggsounder"
    assert spec["scorer"] == "clf" and spec["protocol_version"] == 1
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
    assert set(ic["handed_over"]) == set(ic["info"]) == set(F.file_fields(spec)) == {"video", "audio"}
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
    assert {k: v["kind"] for k, v in F.file_fields(spec).items()} == {
        "video": "video",
        "audio": "audio",
    }
    for s in F.file_fields(spec).values():
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


def test_external_entry(spec, eval_mod):
    (e,) = spec["external"]
    assert set(e) == {"name", "pip", "version_pin", "used_for"}
    assert e["name"] == "vggsounder" and e["version_pin"] == "1.0"
    assert e["pip"] == "pip install --no-deps vggsounder==1.0"
    assert eval_mod.external_entry() == {k: str(v) for k, v in e.items()}
    assert (HERE / "eval.py").is_file()  # non-empty external requires eval.py


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


#: the package's ground-truth sets as manifest cells (``vggsounder.labels.VGGSounder.set_modality``).
PACKAGE_SETS = {
    "a": ("labels_a", "labels_av"),
    "v": ("labels_v", "labels_av"),
    "av": ("labels_av",),
    "a only": ("labels_a",),
    "v only": ("labels_v",),
    "all": ("labels_a", "labels_v", "labels_av"),
}


def test_eval_set_populations(manifest_rows):
    """The official scorer scores each set over the clips holding a label of it (setup.md gotcha 6)."""
    ev = [r for r in manifest_rows if r["split"] == "test"]
    pops = {name: sum(1 for r in ev if any(r[c] for c in cols)) for name, cols in PACKAGE_SETS.items()}
    assert pops == {"a": 11911, "v": 10182, "av": 9883, "a only": 6392, "v only": 850, "all": 12008}


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
    assert mm.VGGSOUND_CSV_SHA[:8] in spec["manifest"]["upstream"]["draw"]


def test_metric_names_resolve(spec, eval_mod):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    # only the package's own values: its 30 names on the reading's set and on its exclusive set, nothing else
    assert list(eval_mod.METRICS) == [*eval_mod.PACKAGE_METRICS, *(n + "_only" for n in eval_mod.PACKAGE_METRICS)]
    assert sorted(m["list"]) == sorted(eval_mod.METRICS) and len(m["list"]) == 60
    assert m["floors"] == [] and m["controls"] == [] and m["chance"] == {} and m["ci"] == {}
    assert m["headline"] == "auc_pr" and m["table"] == ["auc_pr", "hit_rate@1", "f1@1"]
    assert len(eval_mod.PACKAGE_METRICS) == 30
    assert {s.external for s in eval_mod.METRICS.values()} == {"vggsounder"}
    for name, s in eval_mod.METRICS.items():
        assert repr(name.removesuffix("_only")) in s.description and "vggsounder 1.0" in s.description


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


# ----------------------------------------------------------------------------- eval.py


def test_probe_imports_without_torch_and_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "vggsounder", None)
    mod = _load("vggsounder_probe_isolated", HERE / "eval.py")
    assert mod.READING_SETS == {"a": ("a", "a only"), "v": ("v", "v only"), "av": ("av", None)}
    assert mod.PROTOCOL_KEYS == frozenset()
    for name, m in {**mod.METRICS, **mod.JOIN_METRICS}.items():
        assert m.name == name and m.readouts == frozenset({"clf"}) and m.shape == "scalar"
        assert m.kind == "metric" and m.external == "vggsounder" and m.floor_of == ()
        with pytest.raises(RuntimeError, match="official vggsounder package"):
            m.fn(None, None, None, None)
    # mu: the package's definitions verbatim; display labels A / V / A∩V
    assert "both A and V top-1 hit when AV top-1 fails" in mod.JOIN_METRICS["mu_av"].description
    assert mod.MU_LABELS == {"mu_a": "A", "mu_v": "V", "mu_av": "A∩V"}
    # the assert-import: absent package → not ready, message carries the pip line
    with pytest.raises(Exception, match=re.escape("pip install --no-deps vggsounder==1.0")) as exc:
        mod.require_vggsounder()
    assert getattr(exc.value, "exit_code", 3) == 3


def test_require_is_the_core_assert_import_with_the_version_pin(monkeypatch, eval_mod):
    """The authors' implementation through an assert-import: the one assert-import is yavih.external.require (no folder-local fallback)."""
    import importlib.metadata

    from yavih import external

    assert not hasattr(eval_mod, "_fallback_require")
    external.reset()
    monkeypatch.setitem(sys.modules, "vggsounder", types.ModuleType("vggsounder"))
    for installed in ("1.1", "1.0.1", "0.9"):  # another version than the pin 1.0: refused, nothing recorded
        monkeypatch.setattr(importlib.metadata, "version", lambda name, v=installed: v)
        with pytest.raises(external.ExternalVersion, match="E_EXTERNAL_VERSION") as exc:
            eval_mod.require_vggsounder()
        assert exc.value.exit_code == 3 and external.versions_used() == {}
    # the pin compares as a PEP 440 version: 1.0.0 is 1.0; what is recorded is the installed string
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0.0")
    assert eval_mod.require_vggsounder() == "1.0.0" and external.versions_used() == {"vggsounder": "1.0.0"}
    external.reset()
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0")
    assert eval_mod.require_vggsounder() == "1.0" and external.versions_used() == {"vggsounder": "1.0"}
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


def test_join_refusals(tmp_path, eval_mod):
    rng = np.random.default_rng(0)
    bench = _StubBench([f"c{i}" for i in range(4)])
    ph = bench.protocol_hash
    a = _fake_result(tmp_path, "a", ph, "s", 4, rng)
    v = _fake_result(tmp_path, "v", ph, "s", 4, rng)
    av_other = _fake_result(tmp_path, "av", "q" * 64, "s", 4, rng)
    with pytest.raises(eval_mod.JoinRefused, match="E_JOIN_MISSING"):
        eval_mod.join(bench, {"a": a, "v": v})
    with pytest.raises(eval_mod.JoinRefused, match="E_JOIN_MIXED"):
        eval_mod.join(bench, {"a": a, "v": v, "av": av_other})
    av_pop = _fake_result(tmp_path, "av2", ph, "t", 4, rng)
    with pytest.raises(eval_mod.JoinRefused, match="E_JOIN_POPULATION"):
        eval_mod.join(bench, {"a": a, "v": v, "av": av_pop})
    av = _fake_result(tmp_path, "av_ok", ph, "s", 4, rng)
    # swapped routing: the video-only reading passed as a=, the audio-only one as v=
    with pytest.raises(eval_mod.JoinRefused, match="E_JOIN_ROUTING"):
        eval_mod.join(bench, {"a": v, "v": a, "av": av})
    # an "a" reading whose bridge also read the video
    a_av = _fake_result(tmp_path, "a_impure", ph, "s", 4, rng, reads=["audio", "video"])
    with pytest.raises(eval_mod.JoinRefused, match="E_JOIN_ROUTING"):
        eval_mod.join(bench, {"a": a_av, "v": v, "av": av})
    # two model names (E_JOIN_MODEL)
    v_other = _fake_result(tmp_path, "v_model", ph, "s", 4, rng, model="other")
    with pytest.raises(eval_mod.JoinRefused, match="E_JOIN_MODEL"):
        eval_mod.join(bench, {"a": a, "v": v_other, "av": av})
    # one protocol and probe but another eval_hash (eval_hash is the comparability key)
    v_eval = _fake_result(tmp_path, "v_eval", ph, "s", 4, rng, eval_hash="e" * 64)
    with pytest.raises(eval_mod.JoinRefused, match="E_JOIN_MIXED"):
        eval_mod.join(bench, {"a": a, "v": v_eval, "av": av})
    # one reading probed with another / a non-standard ProbeConfig
    v_probe = _fake_result(tmp_path, "v_probe", ph, "s", 4, rng, probe_hash="z" * 64)
    with pytest.raises(eval_mod.JoinRefused, match="E_JOIN_PROBE"):
        eval_mod.join(bench, {"a": a, "v": v_probe, "av": av})
    v_fast = _fake_result(tmp_path, "v_fast", ph, "s", 4, rng, standard=False)
    with pytest.raises(eval_mod.JoinRefused, match="E_JOIN_PROBE"):
        eval_mod.join(bench, {"a": a, "v": v_fast, "av": av})
    (a / "DONE").unlink()
    with pytest.raises(eval_mod.JoinRefused, match="DONE"):
        eval_mod.join(bench, {"a": a, "v": v, "av": av_pop})


def test_join_weights_differ_is_a_warning_not_a_refusal(tmp_path, eval_mod):
    """Different non-null ``weights_sha256`` among the three readings are ``W_JOIN_WEIGHTS``
    (recorded in the join output), never a refusal; a null fingerprint is not a difference."""
    rng = np.random.default_rng(1)
    ph = _StubBench.protocol_hash
    recs = {k: json.loads((_fake_result(tmp_path, k, ph, "s", 4, rng) / "result.json").read_text()) for k in ("a", "v", "av")}
    assert eval_mod._check_readings(recs) == []
    recs["a"]["bridge"]["describe"]["weights_sha256"] = None  # an unfingerprinted reading: no difference to report
    assert eval_mod._check_readings(recs) == []
    recs["v"]["bridge"]["describe"]["weights_sha256"] = "x" * 64
    warnings = eval_mod._check_readings(recs)
    assert [w["code"] for w in warnings] == ["W_JOIN_WEIGHTS"] and "m" in warnings[0]["message"]
    assert set(warnings[0]) == {"code", "message", "where"}


def test_join_reads_the_handed_over_record(tmp_path, eval_mod):
    """``report --join`` hands ``ResultRef``s whose ``.result`` is upgraded in memory; the hook reads
    that record, not the file; a record without a top-level ``model`` names it in ``describe()``."""
    rng = np.random.default_rng(2)
    ph = _StubBench.protocol_hash
    d = _fake_result(tmp_path, "a", ph, "s", 4, rng)
    upgraded = {**json.loads((d / "result.json").read_text()), "model": "from-ref"}
    ref = types.SimpleNamespace(seed_dir=d, result=upgraded)
    assert eval_mod._record(ref, d)["model"] == "from-ref"
    assert eval_mod._record(d, d)["model"] == "m"
    assert eval_mod._model({"bridge": {"describe": {"model": "x", "reading": "a"}}}) == "x"


def test_package_ground_truth_equals_label_sets(manifest_rows):
    """setup.md: the package's a / v / all ground truth equals the manifest's audible / visible / all sets
    on the eval population, and its `av` / `a only` / `v only` sets the labels_av / labels_a / labels_v cells."""
    labels_mod = pytest.importorskip("vggsounder.labels")
    import vggsounder

    data = Path(vggsounder.__file__).parent / "data" / "vggsounder.csv"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == VGGSOUNDER_CSV_SHA
    ev = [r for r in manifest_rows if r["split"] == "test"]
    ids = {r["clip_id"] for r in ev}
    vs = labels_mod.VGGSounder(background_music=False, download_samples=False)
    assert len(vs) == 12014 and ids <= set(vs.keys())
    for key, cols in PACKAGE_SETS.items():
        mine = {r["clip_id"]: {t for c in cols for t in _cells(r[c])} for r in ev}
        vs.set_modality(key)
        pkg = {v.video_id: set(v.labels) for v in vs if v.video_id in ids}
        assert pkg == {k: s for k, s in mine.items() if s}, key


def _pinned_package_or_skip(eval_mod):
    try:
        return eval_mod.require_vggsounder()
    except Exception as exc:  # absent or not the PyPI 1.0 release
        pytest.skip(f"official scorer not available at its pin: {exc}")


def test_score_with_official_package(manifest_rows, spec, eval_mod):
    _pinned_package_or_skip(eval_mod)
    ev_ids = [r["clip_id"] for r in manifest_rows if r["split"] == "test"]
    bench = _StubBench(ev_ids)
    rows = np.arange(len(ev_ids))
    rng = np.random.default_rng(0)
    logits = {k: rng.standard_normal((len(ev_ids), 309)) for k in ("a", "v", "av")}
    per_reading = {k: eval_mod.score(bench, "test", rows, logits[k], reading=k) for k in logits}
    for k, out in per_reading.items():
        assert set(out) == set(spec["metrics"]["list"])
        assert all(v is None for n, v in out.items() if n.endswith("_only")) == (k == "av")
        for n in ("auc_roc", "auc_pr"):
            assert (0.4 < out[n] < 0.6) if n == "auc_roc" else (0.0 < out[n] < 0.03), (k, n, out[n])
    # the per-reading scores are the package's compute_all_results numbers for the same logits
    payload = {c.removesuffix(".mp4"): {"logits": {k: logits[k][i].tolist() for k in logits}} for i, c in enumerate(ev_ids)}
    full = eval_mod.official_scores(payload, ev_ids, eval_mod.PACKAGE_SETS)
    for k, (own, only) in eval_mod.READING_SETS.items():
        for m in eval_mod.PACKAGE_METRICS:
            assert per_reading[k][m] == pytest.approx(full[own][m])
            if only is not None:
                assert per_reading[k][m + "_only"] == pytest.approx(full[only][m])
    assert {"a", "v", "av"} <= {s for s, d in full.items() if "mu" in d}


# ----------------------------------------------------------------------------- eval.py with a fake package


_SET_CODE = {"a": 1.0, "v": 2.0, "av": 3.0, "a only": 4.0, "v only": 5.0, "all": 6.0}


class _Frame:
    """The two DataFrame operations the package's per-set step uses on ``load_test_predictions`` output."""

    def __init__(self, **cols):
        self.cols = dict(cols)

    def drop(self, columns, inplace=False):
        assert inplace
        for c in columns:
            del self.cols[c]


@pytest.fixture
def fake_package(monkeypatch):
    """A fake ``vggsounder`` 1.0 with the package's routing semantics: ``load_test_predictions`` reads
    ``logits["a"]`` for ``a only`` and ``logits["v"]`` for ``v only`` (KeyError for an absent key);
    ``compare_labels_logits`` records (ground-truth set, logits key) and returns every package metric as
    ``set code + metric index / 1000``; ``compute_all_results`` adds ``mu`` for a / v / av."""
    from yavih import external

    calls: list[tuple[str, str]] = []

    class VGGSounder:
        def __init__(self, **kw):
            self.kw, self.modality = kw, None

        def get_all_labels(self):
            return [f"c{i}" for i in range(309)]

        def set_modality(self, modality):
            self.modality = modality

    def get_gt_for_modality(vs, modality, subset_ids=None):
        vs.set_modality(modality)
        return _Frame(set=modality, ids=subset_ids)

    def load_test_predictions(model, modality):
        key = {"a only": "a", "v only": "v"}.get(modality, modality)
        with open(model, "rb") as fh:
            payload = __import__("pickle").load(fh)
        return _Frame(labels=None, logits={i: d["logits"][key] for i, d in payload.items()}, modality=key)

    def compare_labels_logits(gt, logits_df, classes, ks=(1, 3, 5, 10)):
        assert "labels" not in logits_df.cols and len(classes) == 309
        calls.append((gt.cols["set"], logits_df.cols["modality"]))
        names = [f"{m}@{k}" for k in ks for m in ("f1", "f1_macro", "accuracy", "jaccard", "precision", "recall", "hit_rate")]
        return {n: _SET_CODE[gt.cols["set"]] + i / 1000 for i, n in enumerate([*names, "auc_roc", "auc_pr"])}

    def compute_all_results(models_path, subset_ids=None, modalities=(), vggsounder=None, verbose=True):
        (pkl,) = Path(models_path).glob("*.pkl")
        res = {}
        for m in modalities:
            res[m] = compare_labels_logits(get_gt_for_modality(vggsounder, m, subset_ids), _strip(load_test_predictions(str(pkl), m)),
                                           vggsounder.get_all_labels())
        for m in ("a", "v", "av"):
            res[m]["mu"] = {"a": 0.11, "v": 0.22, "av": 0.33}[m]
        return {pkl.stem: res}

    def _strip(frame):
        frame.drop(columns=["labels"], inplace=True)
        return frame

    pkg = types.ModuleType("vggsounder")
    bench_mod = types.ModuleType("vggsounder.benchmark")
    labels_mod = types.ModuleType("vggsounder.labels")
    for f in (get_gt_for_modality, load_test_predictions, compare_labels_logits, compute_all_results):
        setattr(bench_mod, f.__name__, f)
    labels_mod.VGGSounder = VGGSounder
    pkg.benchmark, pkg.labels = bench_mod, labels_mod
    for name, mod in (("vggsounder", pkg), ("vggsounder.benchmark", bench_mod), ("vggsounder.labels", labels_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    import importlib.metadata as md

    real_version = md.version
    monkeypatch.setattr(md, "version", lambda name: "1.0" if name == "vggsounder" else real_version(name))
    external.reset()
    yield calls
    external.reset()


def _value(set_name: str, metric: str, eval_mod) -> float:
    return _SET_CODE[set_name] + eval_mod.PACKAGE_METRICS.index(metric) / 1000


def test_score_ids_per_reading(spec, eval_mod, fake_package):
    """Reading a / v: exactly the 30 + 30 ids, all values; reading av: the 30 ids with values and the 30
    ``_only`` ids None (not applicable). The id set never depends on the reading."""
    ids = [f"c{i}_000000" for i in range(5)]
    bench, rows, out = _StubBench(ids), np.arange(5), np.zeros((5, 309))
    listed = set(spec["metrics"]["list"])
    for reading in ("a", "v"):
        got = eval_mod.score(bench, "test", rows, out, reading=reading)
        assert set(got) == listed and len(got) == 60 and all(isinstance(v, float) for v in got.values())
    got = eval_mod.score(bench, "test", rows, out, reading="av")
    assert set(got) == listed
    assert {n for n, v in got.items() if v is not None} == set(eval_mod.PACKAGE_METRICS)
    with pytest.raises(ValueError, match="reading"):
        eval_mod.score(bench, "test", rows, out, reading="rgb")
    with pytest.raises(ValueError, match="reading"):
        eval_mod.score(bench, "test", rows, out)


def test_score_routes_each_reading_to_its_own_sets_only(eval_mod, fake_package):
    """The logits of a reading go under its own pkl key and are scored only on its sets: a → a, a only;
    v → v, v only; av → av. An audio reading is never scored on a visible set."""
    ids = [f"c{i}_000000" for i in range(3)]
    bench, rows = _StubBench(ids), np.arange(3)
    expect = {"a": [("a", "a"), ("a only", "a")], "v": [("v", "v"), ("v only", "v")], "av": [("av", "av")]}
    for reading, calls in expect.items():
        fake_package.clear()
        got = eval_mod.score(bench, "test", rows, np.zeros((3, 309)), reading=reading)
        assert fake_package == calls
        own, only = eval_mod.READING_SETS[reading]
        assert got["hit_rate@1"] == _value(own, "hit_rate@1", eval_mod) and got["auc_pr"] == _value(own, "auc_pr", eval_mod)
        if only is not None:
            assert got["f1@10_only"] == _value(only, "f1@10", eval_mod)


def test_score_through_the_driver_hands_over_the_reading(spec, eval_mod, fake_package):
    """``evaluate_metrics`` with the folder's specs and ``ReadoutContext.scorer``: the hook receives the run's
    reading; the av reading's ``_only`` metrics are recorded as null."""
    from yavih.readouts.base import ReadoutContext, evaluate_metrics

    ids = [f"c{i}_000000" for i in range(4)]
    specs = {"metrics": {n: eval_mod.METRICS[n] for n in spec["metrics"]["list"]}, "floors": {}, "controls": {}}
    for reading in ("a", "av"):
        ctx = types.SimpleNamespace(eval_mod=eval_mod, bench=_StubBench(ids), reading=reading)
        scorer = ReadoutContext.scorer(ctx, ["test"], np.arange(4), None)
        blocks = evaluate_metrics(specs, outputs=np.zeros((4, 309)), targets=None, protocol={}, split_meta={},
                                  floor_outputs=None, scorer=scorer)
        assert list(blocks["metrics"]) == spec["metrics"]["list"] and blocks["floors"] == {}
        assert blocks["metrics"]["auc_pr"] == _value(reading, "auc_pr", eval_mod)
        assert (blocks["metrics"]["auc_pr_only"] is None) == (reading == "av")


def test_join_returns_the_full_package_output(tmp_path, eval_mod, fake_package):
    """``join``: compute_all_results on the five sets — 5 × 30 metrics + mu for a / v / av — and the compact
    table rows A / V / AV with mAP, HIT, F1 and mu (A / V / A∩V)."""
    rng = np.random.default_rng(3)
    bench = _StubBench([f"c{i}" for i in range(4)])
    runs = {k: _fake_result(tmp_path, k, bench.protocol_hash, "s", 4, rng) for k in ("a", "v", "av")}
    out = eval_mod.join(bench, runs)
    scores = out["scores"]
    assert list(scores) == list(eval_mod.PACKAGE_SETS)
    n_values = sum(len(d) for d in scores.values())
    assert n_values == 5 * 30 + 3 and {s for s, d in scores.items() if "mu" in d} == {"a", "v", "av"}
    for s in eval_mod.PACKAGE_SETS:
        assert set(scores[s]) - {"mu"} == set(eval_mod.PACKAGE_METRICS)
    # each set scored from the logits key the package reads it from
    assert sorted(fake_package) == sorted([("a", "a"), ("v", "v"), ("av", "av"), ("a only", "a"), ("v only", "v")])
    table = out["table"]
    assert table["header"] == ["", "mAP", "HIT", "F1", "mu (A, V, A∩V)"]
    assert [r[0] for r in table["rows"]] == ["A", "V", "AV"]
    assert table["rows"][2][1:] == [_value("av", "auc_pr", eval_mod), _value("av", "hit_rate@1", eval_mod),
                                    _value("av", "f1@1", eval_mod), 0.33]
    assert any("A∩V (mu_av)" in n and "both A and V top-1 hit when AV top-1 fails" in n for n in table["notes"])
    assert out["external"] == {"vggsounder": "1.0"} and out["n_eval"] == 4


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
