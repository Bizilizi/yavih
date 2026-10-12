"""Readout integration ("oracle_bridge clears every readout's floor … random_bridge does not"):
the REAL modules end to end — ``yavih.testing.make_synthetic_benchmark`` → ``Benchmark.from_dir``
→ ``oracle_bridge``/``random_bridge`` + ``run_extract`` → ``validate_features`` →
``probe`` (over the probe heads/trainer and the metrics). Marked ``integration`` + ``slow``; a sibling
module that is still a stub skips the test instead of failing it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from yavih import hashing
from yavih.readouts import ProbeRefused, check_result, probe, result_dir

pytestmark = [pytest.mark.integration, pytest.mark.slow]

HEADLINE = {"clf": ("acc", "acc_floor"), "seq": ("seg_acc", "seg_acc_floor"), "grid": ("d1", "d1_floor"), "box": ("iou_mean", "iou_floor")}


def _pipeline(tmp_path: Path, readout: str, *, bridge_kind: str, n_train: int, n_eval: int, seeds=(0,), **probe_kw):
    """The REAL chain; every sibling is implemented, so a NotImplementedError anywhere in it FAILS the
    contract test instead of skipping it."""
    from yavih.benchmark import Benchmark
    from yavih.extract import run_extract
    from yavih.io.validate import validate_features
    from yavih.probes.config import ProbeConfig
    from yavih.testing import make_synthetic_benchmark, oracle_bridge, random_bridge

    folder = make_synthetic_benchmark(tmp_path, readout, n_train=n_train, n_eval=n_eval, media="tiny")
    bench = Benchmark.from_dir(folder)
    make = oracle_bridge if bridge_kind == "oracle" else random_bridge
    bridge = make(bench, N=4, D=16)
    run = tmp_path / "runs" / bench.name / bridge.model_name / bridge.reading  # runs/<bench>/<model>/<reading>
    splits = list(bench.spec.splits.train) + list(bench.spec.splits.eval) + list(bench.spec.splits.floor) + list(bench.spec.splits.folds or ())
    for split in splits:
        out = run_extract(bench=bench, bridge=bridge, split=split, shard=(0, 1), out=run, canaries=2)
        assert out["status"] == "ok", out
    for split in splits:
        rep = validate_features(run / "features" / split, bench=bench)
        assert rep.ok, [(e.code, e.message) for e in rep.errors]
    cfg = ProbeConfig.pinned(readout, bench).fast()
    res = probe(run, bench=bench, cfg=cfg, seeds=seeds, device="cpu", log_samples=True, **probe_kw)
    return bench, run, cfg, res


def _check_store(res, bench) -> None:
    """docs/results.md: one results file for the invocation next to the runs root, one samples line per eval row."""
    import json as _json

    rf = res[0].results_file
    assert rf is not None and rf.parent.name == res[0].result["model"] and rf.parent.parent.name == "results"
    assert rf.parent.parent.parent == res[0].path.parents[7]  # `results` next to the runs root
    doc = _json.loads(rf.read_text())
    key = f"{bench.name}:{res[0].result['reading']}"  # <benchmark>:<reading>
    entry = doc["results"][key]
    assert entry["status"] == "ok" and entry["headline"] == bench.spec.metrics.headline
    assert doc["configs"][key]["protocol_hash"] == bench.protocol_hash and doc["versions"][key] == bench.spec.protocol_version
    assert doc["configs"][key]["eval_hash"] == res[0].result["eval_hash"]
    lines = res[0].samples_file.read_text().splitlines()
    assert len(lines) == res[0].result["metrics"]["n_eval"]


def test_clf_oracle_clears_floor_and_random_does_not(tmp_path):
    bench, run, cfg, res = _pipeline(tmp_path / "oracle", "clf", bridge_kind="oracle", n_train=24, n_eval=12)
    assert [r.status for r in res] == ["ok"]
    rec = res[0].result
    check_result(rec, expect={"metrics": list(bench.spec.metrics.list), "floors": list(bench.spec.metrics.floors),
                              "controls": list(bench.spec.metrics.controls), "ci": dict(bench.spec.metrics.ci)})
    metric, floor = HEADLINE["clf"]
    assert rec["metrics"][metric] > rec["floors"][floor], rec
    assert rec["standard"] is False and rec["probe"]["hash"] == hashing.probe_hash(cfg)
    assert rec["benchmark"]["protocol_hash"] == bench.protocol_hash and rec["benchmark"]["manifest_hash"] == bench.manifest_hash
    assert rec["features"]["N"] == 4 and rec["features"]["D"] == 16 and rec["features"]["backend"] in {"ram", "memmap"}
    assert rec["coverage"]["eval"]["present"] == 12 and rec["metrics"]["n_eval"] == 12
    out = res[0].path.parent
    assert out == result_dir(run, bench.protocol_hash, hashing.probe_hash(cfg), 0)
    assert np.load(out / "eval_logits.npy").shape == (12, len(bench.classes()))
    assert np.load(out / "eval_rows.npy").tolist() == list(range(12))
    done = json.loads((out / "DONE").read_text())
    assert done["protocol_hash"] == bench.protocol_hash and done["probe_hash"] == hashing.probe_hash(cfg)
    assert done["features_fingerprint"] == rec["features"]["fingerprint"]
    fit = json.loads((out / "probe_fit.json").read_text())
    assert fit["epochs_run"] >= 1 and fit["s_per_epoch"] > 0
    _check_store(res, bench)
    # the same call again is a skip (matching DONE), not a refit
    assert probe(run, bench=bench, cfg=cfg, device="cpu")[0].skipped is True

    _, _, _, rnd = _pipeline(tmp_path / "random", "clf", bridge_kind="random", n_train=24, n_eval=12)
    r = rnd[0].result
    assert r["status"] == "ok"
    # random features: within the Wilson interval of the floor (never confidently above it)
    assert r["ci95"][metric][0] <= max(r["floors"][floor], 1 / len(bench.classes())) + 1e-9


@pytest.mark.parametrize("readout", ["seq", "grid", "box"])
def test_dense_readouts_run_on_oracle_features(tmp_path, readout):
    bench, run, cfg, res = _pipeline(tmp_path, readout, bridge_kind="oracle", n_train=16, n_eval=8)
    assert res[0].status == "ok"
    rec = res[0].result
    check_result(rec, expect={"metrics": list(bench.spec.metrics.list), "floors": list(bench.spec.metrics.floors),
                              "controls": list(bench.spec.metrics.controls), "ci": dict(bench.spec.metrics.ci)})
    metric, floor = HEADLINE[readout]
    assert np.isfinite(rec["metrics"][metric]) and np.isfinite(rec["floors"][floor])
    assert rec["metrics"]["n_eval"] == rec["coverage"]["eval"]["present"]
    assert (res[0].path.parent / "DONE").exists() and (res[0].path.parent / "eval_logits.npy").exists()
    _check_store(res, bench)


def test_probe_refuses_stale_input_hash_against_real_meta(tmp_path):
    bench, run, cfg, _ = _pipeline(tmp_path, "clf", bridge_kind="oracle", n_train=12, n_eval=6)
    meta_path = run / "features" / "val" / "_meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["benchmark"]["input_hash"] == bench.input_hash  # the writer records the input side
    meta["benchmark"]["protocol_hash"] = "0" * 64  # a scoring-side difference alone is no refusal ...
    meta_path.write_text(json.dumps(meta))
    probe(run, bench=bench, cfg=cfg, seeds=(1,), device="cpu", write_results=False)
    meta["benchmark"]["input_hash"] = "0" * 64  # ... an input-side one is
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=cfg, seeds=(1,), device="cpu")
    assert e.value.code == "E_HASH_MISMATCH" and e.value.exit_code == 2


def _na_variant(base):
    """``base`` (a testing bridge class) whose ``encode`` raises ``NotApplicable``: another configuration of the same
    model name and reading that is not applicable at run time."""
    from yavih.bridge import NotApplicable

    class NotApplicableVariant(base):
        def encode(self, samples):
            raise NotApplicable("config B cannot serve this media")

    return NotApplicableVariant


def test_a_reconfigured_not_applicable_bridge_never_reuses_the_old_ok_result(tmp_path):
    """Config A (oracle, D 16, model m1, reading a) is extracted, validated
    and probed (ok). Config B (same m1/a, D 32) raises NotApplicable in encode and is extracted into the SAME run
    directory: its NOT_APPLICABLE.json must win. Extract moves A's shards AND A's _meta.json to _stale/; the probe never
    reads a _meta.json beside NOT_APPLICABLE.json, so it records not_applicable with
    B's describe_sha256 and moves A's DONE to probe/_stale/ instead of returning A's 'ok' as skipped."""
    from yavih.benchmark import Benchmark
    from yavih.extract import run_extract
    from yavih.io.format import META_FILE, NOT_APPLICABLE_FILE, STALE_DIR
    from yavih.io.validate import validate_features
    from yavih.probes.config import ProbeConfig
    from yavih.testing import make_synthetic_benchmark, oracle_bridge

    folder = make_synthetic_benchmark(tmp_path, "clf", n_train=12, n_eval=6, media="tiny")
    bench = Benchmark.from_dir(folder)
    run = tmp_path / "runs" / bench.name / "m1" / "a"
    splits = list(bench.spec.splits.train) + list(bench.spec.splits.eval)
    a = oracle_bridge(bench, N=4, D=16, model="m1", reading="a")
    for split in splits:
        assert run_extract(bench=bench, bridge=a, split=split, shard=(0, 1), out=run, canaries=0)["status"] == "ok"
        assert validate_features(run / "features" / split, bench=bench).ok
    cfg = ProbeConfig.pinned("clf", bench).fast()
    (ok,) = probe(run, bench=bench, cfg=cfg, seeds=(0,), device="cpu")
    assert ok.status == "ok"
    dsha_a = ok.result["bridge"]["describe_sha256"]
    seed_dir = ok.path.parent

    b = _na_variant(oracle_bridge(bench, N=4, D=32, model="m1", reading="a"))
    for split in splits:
        out = run_extract(bench=bench, bridge=b, split=split, shard=(0, 1), out=run, canaries=0)
        assert out["status"] == "not_applicable", out
        d = run / "features" / split
        assert sorted(p.name for p in d.iterdir() if p.is_file()) == [NOT_APPLICABLE_FILE], list(d.iterdir())
        moved = sorted(p.name for p in (d / STALE_DIR).glob("*/*"))
        assert moved == ["_layout.json", META_FILE, "shard-000000.json", "shard-000000.tar"], moved  # A's frozen layout too
    dsha_b = json.loads((run / "features" / splits[0] / NOT_APPLICABLE_FILE).read_text())["describe_sha256"]
    assert dsha_b and dsha_b != dsha_a

    (na,) = probe(run, bench=bench, cfg=cfg, seeds=(0,), device="cpu")
    assert na.status == "not_applicable" and na.skipped is False, (na.status, na.skipped)
    assert na.result["bridge"]["describe_sha256"] == dsha_b and na.result["features"]["D"] is None
    assert na.path.parent == seed_dir and json.loads((seed_dir / "DONE").read_text())["describe_sha256"] == dsha_b
    old = list((run / "probe" / "_stale").glob(f"*/{seed_dir.parent.name}/{seed_dir.name}/result.json"))
    assert len(old) == 1 and json.loads(old[0].read_text())["bridge"]["describe_sha256"] == dsha_a
    assert json.loads(old[0].read_text())["status"] == "ok"
    # the same call again: B's own verdict is the skip now
    (again,) = probe(run, bench=bench, cfg=cfg, seeds=(0,), device="cpu")
    assert again.skipped is True and again.status == "not_applicable" and again.result["bridge"]["describe_sha256"] == dsha_b

    # a stale _meta.json beside NOT_APPLICABLE.json is never read
    for split in splits:
        d = run / "features" / split
        (stale_meta,) = list((d / STALE_DIR).glob(f"*/{META_FILE}"))
        (d / META_FILE).write_bytes(stale_meta.read_bytes())
    (still,) = probe(run, bench=bench, cfg=cfg, seeds=(0,), device="cpu")
    assert still.status == "not_applicable" and still.result["bridge"]["describe_sha256"] == dsha_b
    assert still.result["features"]["D"] is None and still.result["features"]["fingerprint"] is None


def test_probe_refuses_a_not_applicable_split_beside_another_configurations_features(tmp_path):
    """``_agree`` covers describe_sha256 across NOT_APPLICABLE.json records and the other splits' metas. A
    train split re-extracted by config B (NotApplicable) next to an eval split still holding config A's validated
    features is two configurations: E_DESCRIBE_MISMATCH, never a verdict borrowed from either."""
    from yavih.benchmark import Benchmark
    from yavih.extract import run_extract
    from yavih.io.validate import validate_features
    from yavih.probes.config import ProbeConfig
    from yavih.testing import make_synthetic_benchmark, oracle_bridge

    folder = make_synthetic_benchmark(tmp_path, "clf", n_train=12, n_eval=6, media="tiny")
    bench = Benchmark.from_dir(folder)
    run = tmp_path / "runs" / bench.name / "m1" / "a"
    a = oracle_bridge(bench, N=4, D=16, model="m1", reading="a")
    (train,) = bench.spec.splits.train
    (val,) = bench.spec.splits.eval
    for split in (train, val):
        run_extract(bench=bench, bridge=a, split=split, shard=(0, 1), out=run, canaries=0)
        assert validate_features(run / "features" / split, bench=bench).ok
    b = _na_variant(oracle_bridge(bench, N=4, D=32, model="m1", reading="a"))
    assert run_extract(bench=bench, bridge=b, split=train, shard=(0, 1), out=run, canaries=0)["status"] == "not_applicable"
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=ProbeConfig.pinned("clf", bench).fast(), seeds=(0,), device="cpu")
    assert e.value.code == "E_DESCRIBE_MISMATCH" and "re-extract every split" in str(e.value)
