"""Tests for ``yavih.report``: rows, comparability, seeds, floors, ``DONE`` markers and result migration.

Synthetic ``runs/`` trees with ``result.json`` + ``DONE`` are written directly (no probe involved). Rows are keyed
``(benchmark, eval_hash, model, reading)``; bridge decisions are recorded, never keys.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from yavih import hashing
from yavih.report import (
    Report,
    ReportError,
    build_index,
    build_report,
    column_key,
    comparable,
    iter_moved_aside,
    iter_results,
    report,
    row_key,
    tables,
)
from yavih.report.index import INDEX_FILE, runs_root_of
from yavih.report.migrate import PROTOCOL_HASH_UPGRADES, MigrationError, migrate
from yavih.versions import RESULT_SCHEMA_VERSION

PH = "a" * 64
PH2 = "b" * 64
OLD_PH = "0123" * 16  # a protocol hash a synthetic changelog supersedes
QH = "c" * 64
QH2 = "d" * 64
IDS = "e" * 64
IDS2 = "f" * 64
FP = "5" * 64
FP2 = "6" * 64
EH = hashing.eval_hash(PH, QH)


def make_result(
    *,
    bench: str = "k400",
    model: str = "vjepa",
    reading: str = "v",
    bridge_name: str = "vjepa-video",
    seed: int = 0,
    acc: float = 0.5,
    acc_floor: float = 0.1,
    ci: list[float] | None = None,
    protocol_hash: str = PH,
    probe_hash: str = QH,
    protocol_version: int = 2,
    standard: bool = True,
    status: str = "ok",
    ids_sha256: str = IDS,
    eval_present: int = 100,
    eval_nominal: int = 100,
    controls: dict[str, Any] | None = None,
    describe_extra: dict[str, Any] | None = None,
    weights_sha256: str | None = "0" * 64,
    readout: str = "clf",
    metrics: dict[str, Any] | None = None,
    floors: dict[str, Any] | None = None,
    fingerprint: str = FP,
    created: str = "2026-10-07T14:00:00Z",
    rules_checked: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A schema-3 ``result.json``: ``eval_hash``, ``model``, ``reading``; the bridge block is recorded only."""
    describe = {
        "model": model, "reading": reading,
        "weights_sha256": weights_sha256, "weights_ref": "/w/x.pt", "dtype": "float16", "batch_size": 32,
        "streams": ["video"], "pos_mode": "constant",
        "layer": 12, "pooling": "none", "views": 1,
        "input": {"video": {"temporal": {"rule": "uniform", "n": 16, "span": "full"},
                            "spatial": {"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True},
                            "norm": {"mean": [0.5], "std": [0.5], "source": "vjepa@0 transforms.py"}, "decoder": {"lib": "pyav", "version": "14.3.0"}}},
        "precision": {"autocast": "bf16", "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
        "args": {"layer": 12},
    }
    describe.update(describe_extra or {})
    ok = status == "ok"
    ci = [acc - 0.05, acc + 0.05] if ci is None else ci
    return {
        "schema_version": 4,
        "status": status,
        "reason": None if ok else "eligibility: missing pos",
        "standard": standard,
        "eval_hash": hashing.eval_hash(protocol_hash, probe_hash),
        "benchmark": {"name": bench, "protocol_version": protocol_version, "readout": readout, "readout_version": 1 if readout == "clf" else 2,
                      "manifest_hash": "9" * 64, "protocol_hash": protocol_hash, "folder_source": {"probe": None, "reader": None}},
        "model": model,
        "reading": reading,
        "bridge": {"name": bridge_name, "describe": describe, "describe_sha256": hashing.describe_sha256(describe), "source_sha256": "7" * 64},
        "features": {"fingerprint": fingerprint, "dtype": "float16", "N": 1568, "D": 1024, "pos_mode": "constant", "bytes": 1e9,
                     "backend": "ram", "canary_max_rel_l2": 1e-6, "batch_tolerance": 1e-5,
                     "rules_checked": rules_checked if rules_checked is not None else {"video": {"temporal": "uniform", "spatial": True}}},
        "probe": {"hash": probe_hash, "config": {"head": {"family": "attentive_probe_v1"}}, "seed": seed, "selected_lr": 3e-4,
                  "epochs_run": 10, "best_epoch": 5, "stop": "patience", "n_train": 90, "n_val": 10},
        "metrics": (metrics if metrics is not None else {"acc": acc, "n_eval": eval_present}) if ok else {},
        "ci95": {"acc": ci} if ok and "acc" in (metrics or {"acc": 1}) else {},
        "floors": (floors if floors is not None else {"acc_floor": acc_floor}) if ok else {},
        "controls": (controls or {}) if ok else {},
        "chance": {"acc": 0.0025},
        "external": {},
        "coverage": {"train": {"present": 1000, "nominal": 1000, "ids_sha256": "1" * 64},
                     "eval": {"present": eval_present, "nominal": eval_nominal, "ids_sha256": ids_sha256}, "floor": None},
        "not_applicable": None if ok else {"missing": ["pos.y"], "reason": "no spatial tokens", "decided_by": "validate"},
        "timing": {"validate_s": 1.0, "materialize_s": 1.0, "fit_s": 1.0, "score_s": 1.0, "wall_s": 4.0, "s_per_epoch": 0.1},
        "env": {"host": "h", "gpu": "none", "torch": "2.3.1", "python": "3.12", "yavih": "0.1.0", "slurm_job_id": None},
        "warnings": [],
        "created": created,
    }


def seed_dir_of(runs: Path, result: dict[str, Any]) -> Path:
    """``runs/<bench>/<model>/<reading>/probe/<eval_hash12>/seed<k>/``."""
    return (runs / result["benchmark"]["name"] / result["model"] / result["reading"] / "probe" / result["eval_hash"][:12]
            / f"seed{result['probe']['seed']}")


def write_result(runs: Path, result: dict[str, Any], *, done: bool | str = True, with_npy: bool = False, seed_dir: Path | None = None) -> Path:
    """``result.json`` (+ the ``DONE`` marker) in the run layout, or under ``seed_dir``."""
    seed_dir = seed_dir if seed_dir is not None else seed_dir_of(runs, result)
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / "result.json"
    data = json.dumps(result, sort_keys=True, indent=1).encode()
    path.write_bytes(data)
    if done is True:
        marker = {"result_sha256": hashlib.sha256(data).hexdigest(), "protocol_hash": result["benchmark"]["protocol_hash"],
                  "probe_hash": result["probe"]["hash"], "features_fingerprint": (result.get("features") or {}).get("fingerprint"),
                  "yavih_version": "0.1.0", "created": result["created"]}
        marker.update(eval_hash=result["eval_hash"], model=result["model"], reading=result["reading"],
                      describe_sha256=result["bridge"]["describe_sha256"])
        (seed_dir / "DONE").write_text(json.dumps(marker))
    elif done == "empty":
        (seed_dir / "DONE").write_text("")
    elif done == "stale":
        (seed_dir / "DONE").write_text(json.dumps({"result_sha256": "0" * 64}))
    if with_npy:
        import numpy as np

        n = result["coverage"]["eval"]["present"]
        np.save(seed_dir / "eval_logits.npy", np.zeros((n, 4), dtype=np.float32))
        np.save(seed_dir / "eval_rows.npy", np.arange(n, dtype=np.int64))
    return path


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    r = tmp_path / "runs"
    r.mkdir()
    return r


# --------------------------------------------------------------------------- migrate


def test_migrate_identity_for_current_schema():
    r = make_result()
    out = migrate(r)
    assert out == r and out is not r
    assert out["schema_version"] == RESULT_SCHEMA_VERSION == 4


def test_migrate_refuses_newer_schema_and_bad_types():
    with pytest.raises(MigrationError):
        migrate({"schema_version": RESULT_SCHEMA_VERSION + 1})
    assert isinstance(MigrationError("x"), ValueError)
    with pytest.raises(MigrationError):
        migrate({"schema_version": "1"})
    with pytest.raises(MigrationError):
        migrate({"schema_version": 0})
    with pytest.raises(TypeError):
        migrate([])  # type: ignore[arg-type]


def test_migrate_refuses_a_missing_or_older_schema():
    r = make_result()
    del r["schema_version"]
    with pytest.raises(MigrationError):
        migrate(r)
    with pytest.raises(MigrationError, match="older"):
        migrate({**make_result(), "schema_version": 2})  # 3 has an upgrade step; 2 and earlier have none


@pytest.fixture
def superseded():
    """A synthetic folder changelog whose current protocol hash ``PH`` supersedes ``OLD_PH``."""
    from yavih import hash_history

    hash_history.register("synth_report", (), [{"version": 2, "protocol_hash": PH, "input_hash": "6" * 64, "supersedes": [OLD_PH]}])
    yield
    hash_history.reset()


def test_migrate_maps_a_superseded_protocol_hash(superseded):
    """A record of a superseded protocol hash is read under the newest one (``eval_hash`` recomputed); any other hash
    stays its own column."""
    assert PROTOCOL_HASH_UPGRADES[OLD_PH] == PH
    out = migrate(make_result(protocol_hash=OLD_PH))
    assert out["benchmark"]["protocol_hash"] == PH and out["eval_hash"] == hashing.eval_hash(PH, QH) == EH
    out = migrate(make_result(protocol_hash=PH2))
    assert out["benchmark"]["protocol_hash"] == PH2 and out["eval_hash"] == hashing.eval_hash(PH2, QH)


# --------------------------------------------------------------------------- index


def test_iter_results_ignores_missing_or_stale_done(runs: Path, capsys):
    a = write_result(runs, make_result(model="m1"))
    write_result(runs, make_result(model="m2"), done=False)  # no DONE → ignored
    write_result(runs, make_result(model="m3"), done="stale")  # sha mismatch → treated as absent
    write_result(runs, make_result(model="m4"), done="empty")  # a bare marker is not one the harness writes → absent
    # a marker whose sha matches but whose recorded names or hashes disagree with the result is also absent
    e = write_result(runs, make_result(model="m5"))
    marker = json.loads((e.parent / "DONE").read_text())
    marker["model"] = "other"
    (e.parent / "DONE").write_text(json.dumps(marker))
    f = write_result(runs, make_result(model="m6"))
    marker = json.loads((f.parent / "DONE").read_text())
    marker["eval_hash"] = "0" * 64
    (f.parent / "DONE").write_text(json.dumps(marker))
    found = iter_results([str(runs / "*")])
    assert found == [a.resolve()]
    err = capsys.readouterr().err
    assert "m4" in err and "result_sha256" in err and "m5" in err and "DONE.model" in err and "DONE.eval_hash" in err
    # file globs, seed dirs and the runs root itself all work and de-duplicate
    assert iter_results([str(runs), str(a), str(a.parent), str(runs / "**" / "result.json")]) == found
    assert iter_results([str(tmp := runs / "nope")]) == [] and not tmp.exists()


def test_iter_results_skips_moved_aside_stale_outputs(runs: Path):
    """A probe output moved to ``probe/_stale/<utc>/<eval_hash12>/seed<k>/`` keeps a
    valid DONE but is never found by ``iter_results`` (never a row); ``iter_moved_aside`` lists it; naming the stale
    file explicitly still reports it as live. A moved-aside output of the SAME configuration (a re-extraction: other
    features, one describe_sha256) is no older configuration and no older feature set: no W_BRIDGE_CHANGED, no
    W_FEATURES_DIFFER."""
    live = write_result(runs, make_result(model="m1", acc=0.6))
    old = make_result(model="m1", acc=0.4, fingerprint=FP2, created="2026-10-07T13:00:00Z")
    assert old["bridge"]["describe_sha256"] == make_result(model="m1")["bridge"]["describe_sha256"]
    stale_dir = live.parents[2] / "_stale" / "20261009T000000Z" / live.parents[1].name / live.parent.name
    stale = write_result(runs, old, seed_dir=stale_dir)
    assert (stale.parent / "DONE").is_file()
    assert iter_results([str(runs / "*")]) == [live.resolve()]
    assert iter_results([str(runs)]) == [live.resolve()]
    assert iter_moved_aside([str(runs / "*")]) == [stale.resolve()] and iter_moved_aside([str(runs)]) == [stale.resolve()]
    assert iter_results([str(stale)]) == [stale.resolve()] and iter_moved_aside([str(stale)]) == []
    rep = report([str(runs / "*")], fmt="json", detail=True)
    assert not [w for w in rep.warnings if w["code"] in ("W_BRIDGE_CHANGED", "W_FEATURES_DIFFER")], rep.warnings
    (row,) = rep.rows
    assert rep.n_results == 1 and row["metrics"]["acc"]["value"] == pytest.approx(0.6) and row["n_configs"] == 1
    d = rep.detail[rep.columns[0]["id"]]["m1:v"]
    assert d["older_configs"] == [] and d["older_feature_sets"] == []
    # the index file lists the moved-aside record, flagged; the runs root is still inferred from the live layout
    lines = [json.loads(line) for line in (runs / INDEX_FILE).read_text().splitlines()]
    assert {(ln["path"], bool(ln.get("moved_aside"))) for ln in lines} == {(str(live.resolve()), False), (str(stale.resolve()), True)}
    assert (runs / "_report" / "report.json").is_file()


def test_in_place_reconfiguration_in_one_runs_root_is_w_bridge_changed(runs: Path):
    """A reused model name re-run under another configuration in the SAME
    runs root moves the old probe output to ``probe/_stale/``. Its numbers must not vanish silently: the moved-
    aside configuration is an older configuration of the live row (``W_BRIDGE_CHANGED``, ``--detail`` with
    ``moved_aside: true``), never the shown row, never a row of its own — even when it is newer than the live one."""
    old = make_result(model="m1", acc=0.40, describe_extra={"layer": 6}, fingerprint=FP2, created="2026-10-07T13:00:00Z")
    live = make_result(model="m1", acc=0.60, describe_extra={"layer": 12}, created="2026-10-07T14:00:00Z")
    assert old["bridge"]["describe_sha256"] != live["bridge"]["describe_sha256"]
    live_path = write_result(runs, live)
    stale_root = live_path.parents[2] / "_stale" / "20261007T135959Z"
    write_result(runs, old, seed_dir=stale_root / live_path.parents[1].name / live_path.parent.name)
    rep = report([str(runs / "*")], fmt="json", detail=True)
    (w,) = [w for w in rep.warnings if w["code"] == "W_BRIDGE_CHANGED"]
    assert "m1 v on k400" in w["message"] and "2 bridge configurations" in w["message"] and "layer" in w["message"]
    assert "probe/_stale/" in w["message"] and old["bridge"]["describe_sha256"][:8] in w["message"]
    (row,) = rep.rows
    assert rep.n_results == 1 and row["describe_sha256"] == live["bridge"]["describe_sha256"] and row["n_configs"] == 2
    assert row["metrics"]["acc"]["value"] == pytest.approx(0.60) and not any("bridge" in f or "config" in f or "stale" in f for f in row["flags"])
    d = rep.detail[rep.columns[0]["id"]]["m1:v"]
    (older,) = d["older_configs"]
    assert older["describe_sha256"] == old["bridge"]["describe_sha256"] and older["moved_aside"] is True
    assert older["diff"]["layer"] == {"shown": 12, "other": 6} and older["headline"] == pytest.approx(0.40)
    assert d["older_feature_sets"] == []  # the old configuration's features are not a feature set of the live one
    assert [r["value"] for r in rep.lm_eval["m1"] if r["metric"] == "acc"] == [pytest.approx(0.60)]
    md = report([str(runs / "*")], detail=True)
    assert "W_BRIDGE_CHANGED" in md.text and "moved aside to probe/_stale/" in md.text

    # a moved-aside configuration NEWER than the live one is still never shown
    newer = make_result(model="m1", acc=0.90, describe_extra={"layer": 18}, created="2026-10-08T09:00:00Z")
    write_result(runs, newer, seed_dir=live_path.parents[2] / "_stale" / "20261008T090001Z" / live_path.parents[1].name / live_path.parent.name)
    rep = report([str(runs / "*")], fmt="json", detail=True)
    (row,) = rep.rows
    assert row["metrics"]["acc"]["value"] == pytest.approx(0.60) and row["n_configs"] == 3
    assert [oc["moved_aside"] for oc in rep.detail[rep.columns[0]["id"]]["m1:v"]["older_configs"]] == [True, True]

    # a row key known ONLY from moved-aside outputs gets no row
    other = make_result(model="m2", acc=0.5)
    m2_live = seed_dir_of(runs, other)
    write_result(runs, other, seed_dir=m2_live.parents[1] / "_stale" / "20261008T100000Z" / m2_live.parent.name / m2_live.name)
    rep = report([str(runs / "*")], fmt="json")
    assert {r["model"] for r in rep.rows} == {"m1"} and "m2" not in rep.lm_eval


def test_build_index_writes_and_reuses_cache(runs: Path, monkeypatch: pytest.MonkeyPatch):
    p = write_result(runs, make_result())
    recs = build_index([str(runs / "*")])
    assert len(recs) == 1 and recs[0]["path"] == str(p.resolve()) and recs[0]["result"]["model"] == "vjepa"
    index = runs / INDEX_FILE
    assert index.is_file()
    lines = [json.loads(line) for line in index.read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["mtime_ns"] == p.stat().st_mtime_ns

    # cache hit: a tampered-but-same-mtime index line is reused verbatim (proves no re-read)
    lines[0]["result"]["model"] = "cached"
    index.write_text(json.dumps(lines[0]) + "\n")
    recs = build_index([str(runs / "*")])
    assert recs[0]["result"]["model"] == "cached"

    # mtime change → re-read
    new = make_result(acc=0.9)
    time.sleep(0.01)
    write_result(runs, new)
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 10_000_000))
    recs = build_index([str(runs / "*")])
    assert recs[0]["result"]["metrics"]["acc"] == 0.9
    assert json.loads(index.read_text().splitlines()[0])["result"]["metrics"]["acc"] == 0.9


def test_build_index_skips_unreadable_and_newer_schema(runs: Path, capsys):
    good = write_result(runs, make_result(model="m1"))
    bad = make_result(model="m2")
    bad["schema_version"] = RESULT_SCHEMA_VERSION + 5
    write_result(runs, bad)
    broken = write_result(runs, make_result(model="m3"), done="empty")
    broken.write_text("{not json")
    recs = build_index([str(runs / "*")])
    assert [r["path"] for r in recs] == [str(good.resolve())]
    err = capsys.readouterr().err
    assert "newer" in err and "skipping" in err


def test_a_schema_3_result_loads_and_reports(runs: Path):
    """Schema 4 added ``benchmark.folder_source``; a schema-3 ``result.json`` is upgraded in memory (``null``: unknown)
    and reported next to a current one, with no W_FOLDER_SOURCE_DIFFERS (an unknown source is not a different one)."""
    old = make_result(model="m1", acc=0.4)
    old["schema_version"] = 3
    del old["benchmark"]["folder_source"]
    up = migrate(old)
    assert up["schema_version"] == RESULT_SCHEMA_VERSION == 4 and up["benchmark"]["folder_source"] is None
    assert "folder_source" not in old["benchmark"]  # in memory only
    cur = make_result(model="m2", acc=0.6)
    cur["benchmark"]["folder_source"] = {"probe": "1" * 64, "reader": None}
    write_result(runs, old)
    write_result(runs, cur)
    rep = report([str(runs / "*")], write=False)
    assert rep.n_results == 2 and len(rep.columns) == 1
    assert {r["model"]: r["metrics"]["acc"]["value"] for r in rep.rows} == {"m1": pytest.approx(0.4), "m2": pytest.approx(0.6)}
    assert "W_FOLDER_SOURCE_DIFFERS" not in {w["code"] for w in rep.warnings}
    with pytest.raises(MigrationError, match="older than this yavih"):
        migrate({**old, "schema_version": 2})


def test_folder_source_differs_compares_known_sources_only():
    """Two known, different sources in one column warn; a missing one next to known ones that agree does not."""
    a, b, c = (make_result(model=m) for m in ("m1", "m2", "m3"))
    a["benchmark"]["folder_source"] = {"probe": "1" * 64, "reader": None}
    b["benchmark"]["folder_source"] = {"probe": "1" * 64, "reader": None}
    c["benchmark"]["folder_source"] = None
    rep = build_report(_records(a, b, c))
    assert "W_FOLDER_SOURCE_DIFFERS" not in {w["code"] for w in rep.warnings}
    assert "folder_source_differs" not in rep.columns[0]["flags"]
    b["benchmark"]["folder_source"] = {"probe": "2" * 64, "reader": None}
    rep = build_report(_records(a, b, c))
    assert [w["code"] for w in rep.warnings].count("W_FOLDER_SOURCE_DIFFERS") == 1
    assert "folder_source_differs" in rep.columns[0]["flags"]


def test_runs_root_inference(runs: Path):
    p = write_result(runs, make_result())
    assert p.parents[6] == runs and runs_root_of([p]) == runs.resolve()
    assert runs_root_of([]) is None
    assert runs_root_of([Path("/a/result.json")]) is None
    explicit = runs.parent / "elsewhere"
    build_index([str(runs / "*")], runs_root=explicit)
    assert (explicit / INDEX_FILE).is_file()


def test_a_superseded_protocol_hash_lands_in_the_current_column(runs: Path, superseded):
    """Result files are found at any depth; a result of a superseded protocol hash lands in the same row as one of the
    current hash for the same model and reading."""
    old = write_result(runs, make_result(model="vjepa", reading="v", protocol_hash=OLD_PH, seed=0, acc=0.4, created="2026-10-07T10:00:00Z"))
    cur = write_result(runs, make_result(model="vjepa", reading="v", protocol_hash=PH, seed=1, acc=0.6))
    assert runs_root_of([old]) == runs_root_of([cur]) == runs.resolve() == runs_root_of([old, cur])
    rep = report([str(runs)], write=False)
    assert rep.n_results == 2 and len(rep.columns) == 1 and rep.columns[0]["eval_hash"] == EH
    (row,) = rep.rows
    assert (row["model"], row["reading"], row["n_seeds"]) == ("vjepa", "v", 2)


# --------------------------------------------------------------------------- comparability


def test_column_key_and_comparable():
    a, b = make_result(), make_result(model="other")
    assert column_key(a) == ("k400", EH) and row_key(a) == ("k400", EH, "vjepa", "v")
    assert comparable(a, b, aliases={})
    assert not comparable(a, make_result(probe_hash=QH2), aliases={})
    assert not comparable(a, make_result(protocol_hash=PH2), aliases={})
    assert not comparable(a, make_result(bench="ave"), aliases={})
    assert not comparable(a, make_result(standard=False), aliases={})
    # alias collapses onto the canonical name
    assert column_key(make_result(bench="audioset20k_mirror"), aliases={"audioset20k_mirror": "audioset20k"})[0] == "audioset20k"


@pytest.mark.parametrize("extra", [
    {"layer": 18}, {"dtype": "bfloat16"}, {"batch_size": 8}, {"weights_sha256": "f" * 64}, {"weights_sha256": None},
    {"input": {"video": {"temporal": {"rule": "custom", "description": "the model's own clip sampler"}}}},
    {"precision": {"autocast": "fp32"}}, {"args": {"layer": 18}},
])
def test_bridge_decisions_never_enter_a_key(extra):
    """A changed ``describe()`` key changes ``describe_sha256`` and nothing that keys
    or compares results: same column, same row key, comparable."""
    a, b = make_result(), make_result(describe_extra=extra)
    # yavih-describe-2: the batch size is provenance and leaves even describe_sha256 alone
    assert (a["bridge"]["describe_sha256"] != b["bridge"]["describe_sha256"]) == ("batch_size" not in extra)
    assert column_key(a, {}) == column_key(b, {}) and row_key(a, {}) == row_key(b, {}) and comparable(a, b, aliases={})
    assert a["eval_hash"] == b["eval_hash"]


def _records(*results: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"path": f"/r/{i}/result.json", "mtime_ns": 0, "size": 0, "result": r} for i, r in enumerate(results)]


def test_report_refuses_to_merge_columns_with_different_eval_hashes():
    rep = build_report(_records(
        make_result(model="m1"),
        make_result(model="m2", probe_hash=QH2),
        make_result(model="m3", protocol_hash=PH2),
        make_result(model="m4", bench="ave", readout="seq"),
    ))
    assert len(rep.columns) == 4
    assert all(c["n_rows"] == 1 for c in rep.columns)
    ids = {f"k400:{EH[:12]}", f"k400:{hashing.eval_hash(PH, QH2)[:12]}", f"k400:{hashing.eval_hash(PH2, QH)[:12]}", f"ave:{EH[:12]}"}
    assert {c["id"] for c in rep.columns} == ids
    assert {c["eval_hash"] for c in rep.columns if c["benchmark"] == "k400"} == {EH, hashing.eval_hash(PH, QH2), hashing.eval_hash(PH2, QH)}


def test_two_models_and_two_configurations_share_one_column():
    rep = build_report(_records(make_result(model="m1"), make_result(model="m2", describe_extra={"layer": 3, "dtype": "float32"})))
    assert len(rep.columns) == 1 and {(r["model"], r["reading"]) for r in rep.rows} == {("m1", "v"), ("m2", "v")}


def test_nonstandard_results_never_share_a_comparable_column():
    rep = build_report(_records(make_result(model="m1"), make_result(model="m2", standard=False)))
    assert len(rep.columns) == 2
    ns = [c for c in rep.columns if not c["standard"]]
    assert len(ns) == 1 and ns[0]["comparable"] is False and ns[0]["id"].endswith(":nonstandard")
    assert any(w["code"] == "W_NONSTANDARD" for w in rep.warnings)


# --------------------------------------------------------------------------- seeds


def test_seed_aggregation_mean_std_n_seeds():
    rep = build_report(_records(
        make_result(seed=0, acc=0.4, ci=[0.3, 0.5]),
        make_result(seed=1, acc=0.5, ci=[0.4, 0.6]),
        make_result(seed=2, acc=0.6, ci=[0.5, 0.7]),
    ))
    assert len(rep.rows) == 1
    row = rep.rows[0]
    assert row["n_seeds"] == 3 and row["seeds"] == [0, 1, 2] and row["seed"] is None
    cell = row["metrics"]["acc"]
    assert cell["value"] == pytest.approx(0.5) and cell["std"] == pytest.approx(0.1) and cell["floor"] == pytest.approx(0.1)
    assert cell["ci"] == pytest.approx([0.4, 0.6])
    assert "single_seed" not in row["flags"] and "below_floor" not in row["flags"]
    assert "0.5 ± 0.1 (0.1) [0.4, 0.6]" in rep.text


def test_seeds_list_mode_one_row_per_seed():
    rep = build_report(_records(make_result(seed=0, acc=0.4), make_result(seed=1, acc=0.5)), seeds="list")
    assert [r["seed"] for r in rep.rows] == [1, 0]  # sorted by headline value desc
    assert all(r["n_seeds"] == 1 and r["metrics"]["acc"]["std"] is None and "single_seed" not in r["flags"] for r in rep.rows)
    with pytest.raises(ValueError):
        build_report([], seeds="nope")


def test_single_seed_row_is_marked():
    rep = build_report(_records(make_result()))
    assert "single_seed" in rep.rows[0]["flags"] and rep.rows[0]["metrics"]["acc"]["std"] is None


# --------------------------------------------------------------------------- one row per key: configurations and feature sets


def test_newest_configuration_is_shown_with_w_bridge_changed():
    """One row key, two ``describe_sha256`` → one row showing the newest configuration
    (its seeds only), ``W_BRIDGE_CHANGED`` naming the differing keys, the older one under ``--detail``."""
    old0 = make_result(seed=0, acc=0.30, describe_extra={"layer": 12}, created="2026-10-07T10:00:00Z")
    old1 = make_result(seed=1, acc=0.32, describe_extra={"layer": 12}, created="2026-10-07T11:00:00Z")
    new0 = make_result(seed=0, acc=0.70, describe_extra={"layer": 18}, created="2026-10-08T10:00:00Z")
    rep = build_report(_records(old0, old1, new0), detail=True)
    (row,) = rep.rows
    assert row["describe_sha256"] == new0["bridge"]["describe_sha256"] and row["n_seeds"] == 1
    assert row["metrics"]["acc"]["value"] == pytest.approx(0.70) and row["n_configs"] == 2
    assert not any("bridge" in f or "config" in f for f in row["flags"])  # recorded, never a flag
    (w,) = [w for w in rep.warnings if w["code"] == "W_BRIDGE_CHANGED"]
    assert "vjepa v on k400" in w["message"] and "layer" in w["message"] and "2 bridge configurations" in w["message"]
    d = rep.detail[rep.columns[0]["id"]]["vjepa:v"]
    assert d["describe_sha256"] == new0["bridge"]["describe_sha256"] and d["describe"]["layer"] == 18
    (older,) = d["older_configs"]
    assert older["describe_sha256"] == old0["bridge"]["describe_sha256"] and older["seeds"] == [0, 1]
    assert older["diff"]["layer"] == {"shown": 18, "other": 12} and older["headline"] == pytest.approx(0.31)
    assert "older configuration" in rep.text and "`layer`" in rep.text
    lm = rep.lm_eval["vjepa"]
    assert [r["value"] for r in lm if r["metric"] == "acc"] == [pytest.approx(0.70)]


def test_newest_feature_set_is_shown_with_w_features_differ_and_distinct_seeds():
    a0 = make_result(seed=0, acc=0.40)
    a1 = make_result(seed=1, acc=0.50)
    b0 = make_result(seed=0, acc=0.90, fingerprint=FP2, created="2026-10-08T14:00:00Z")  # re-extraction, same configuration
    b1 = make_result(seed=1, acc=0.80, fingerprint=FP2, created="2026-10-08T14:30:00Z")
    dup = make_result(seed=1, acc=0.85, fingerprint=FP2, created="2026-10-08T15:00:00Z")  # the same seed probed again: newest wins
    rep = build_report(_records(a0, a1, b0, b1, dup), detail=True)
    (row,) = rep.rows
    assert row["features_fingerprint"] == FP2 and row["n_seeds"] == 2 and row["n_feature_sets"] == 2
    assert row["metrics"]["acc"]["value"] == pytest.approx(0.875)  # never averaged with the older feature set
    assert any(w["code"] == "W_FEATURES_DIFFER" for w in rep.warnings) and not any(w["code"] == "W_BRIDGE_CHANGED" for w in rep.warnings)
    assert not any(f.startswith("features ") for f in row["flags"])
    (older,) = rep.detail[rep.columns[0]["id"]]["vjepa:v"]["older_feature_sets"]
    assert older["fingerprint"] == FP and older["seeds"] == [0, 1] and older["headline"] == pytest.approx(0.45)
    lm = rep.lm_eval["vjepa"]
    assert {r["benchmark"] for r in lm} == {"k400"} and [r["n_seeds"] for r in lm if r["metric"] == "acc"] == [2]


def test_a_capped_run_is_its_own_partial_row_in_every_format():
    """A result with ``sample`` (``plan build --sample N``) never displaces, or averages with, the full run of its row
    key, however new: it is a second row marked ``partial`` (json flag and key, md suffix, csv column), its subset
    population never makes ``population_differs``, and suite completeness never counts it."""
    from yavih.suites import completeness

    full = make_result(acc=0.5)
    part = make_result(acc=0.9, fingerprint=FP2, ids_sha256=IDS2, eval_present=4, eval_nominal=4, created="2026-10-09T14:00:00Z")
    part["sample"] = {"train": [0, 4], "val": [0, 4]}
    rep = build_report(_records(full, part), detail=True)
    full_row, part_row = rep.rows
    assert (full_row["partial"], part_row["partial"]) == (False, True) and part_row["sample"] == part["sample"]
    assert full_row["metrics"]["acc"]["value"] == 0.5 and part_row["metrics"]["acc"]["value"] == 0.9
    assert "partial" in part_row["flags"] and "partial" not in full_row["flags"] and not part_row["greyed"]
    assert full_row["n_feature_sets"] == 1 and not any(w["code"] in ("W_FEATURES_DIFFER", "W_POPULATION_DIFFERS") for w in rep.warnings)
    assert not rep.columns[0]["population_differs"] and set(rep.detail[rep.columns[0]["id"]]) == {"vjepa:v", "vjepa:v:partial"}
    lm = [r for r in rep.lm_eval["vjepa"] if r["metric"] == "acc"]
    assert [r.get("partial", False) for r in lm] == [False, True]
    md = tables.render(rep, "md")
    assert sum("(partial)" in ln for ln in md.splitlines() if ln.startswith("| k400")) == 2  # view 1: acc and n_eval
    assert any(ln.startswith("| vjepa") and "0.9 (0.1) [0.85, 0.95] (partial)" in ln and "partial" in ln.split("|")[-2] for ln in md.splitlines())
    csv_lines = tables.render(rep, "csv").splitlines()
    col = csv_lines[0].split(",").index("partial")
    assert {ln.split(",")[col] for ln in csv_lines[1:]} == {"true", "false"}
    rows = completeness("default", [{"result": part}], models=["vjepa"])
    k400 = next(r for r in rows if r["benchmark"] == "k400")
    assert k400["status"] == "missing" and "partial only" in k400["note"]
    assert next(r for r in completeness("default", [{"result": full}], models=["vjepa"]) if r["benchmark"] == "k400")["status"] == "done"


# --------------------------------------------------------------------------- floors / greying


def test_greys_out_rows_whose_headline_ci_overlaps_floor():
    rep = build_report(_records(
        make_result(model="good", acc=0.5, acc_floor=0.1, ci=[0.45, 0.55]),
        make_result(model="weak", acc=0.12, acc_floor=0.1, ci=[0.08, 0.16]),  # CI contains the floor
        make_result(model="under", acc=0.05, acc_floor=0.1, ci=[0.02, 0.08]),  # entirely under
    ))
    by = {r["model"]: r for r in rep.rows}
    assert not by["good"]["greyed"]
    assert by["weak"]["greyed"] and "below_floor" in by["weak"]["flags"]
    assert by["under"]["greyed"] and "below_floor" in by["under"]["flags"]
    md = rep.text
    assert "~~weak~~" in md and "~~under~~" in md and "~~good~~" not in md
    assert "| **acc** |" in md  # headline marked
    csv_text = tables.render_csv(rep)
    lines = [line for line in csv_text.splitlines() if "weak" in line and ",acc," in line]
    assert lines and "below_floor" in lines[0]
    good_lines = [line for line in csv_text.splitlines() if ",good," in line]
    assert good_lines and all("below_floor" not in line for line in good_lines)


def test_point_value_decides_without_ci():
    r = make_result(acc=0.09, acc_floor=0.1)
    r["ci95"] = {}
    rep = build_report(_records(r))
    assert rep.rows[0]["greyed"]
    r2 = make_result(acc=0.3, acc_floor=0.1)
    r2["ci95"] = {}
    assert not build_report(_records(r2)).rows[0]["greyed"]


def test_controls_grey_rows_floor_violated_and_audio_blind():
    rep = build_report(_records(
        make_result(model="v1", controls={"acc_floor_arm": 0.3, "floor_violated": True}),
        make_result(model="b2", controls={"iou_shuffled": 0.2, "audio_blind": True}),
        make_result(model="c3", controls={"acc_floor_arm": 0.1, "floor_violated": False}),
    ))
    by = {r["model"]: r for r in rep.rows}
    assert by["v1"]["greyed"] and "floor_violated" in by["v1"]["flags"]
    assert by["b2"]["greyed"] and "audio_blind" in by["b2"]["flags"]
    assert not by["c3"]["greyed"]
    assert by["v1"]["controls"]["floor_violated"] is True and by["v1"]["controls"]["acc_floor_arm"] == pytest.approx(0.3)
    col = rep.columns[0]
    assert set(col["controls"]) == {"acc_floor_arm", "floor_violated", "iou_shuffled", "audio_blind"}


def test_custom_rule_and_unfingerprinted_rows_are_never_greyed_or_flagged():
    """Rule opt-out: a ``custom`` temporal/spatial rule is a legitimate opt-out — the row is
    neither greyed nor flagged (``rules_checked`` stays in the record and ``--detail``); a null weights fingerprint is
    one report-level ``W_WEIGHTS_UNFINGERPRINTED`` warning, never a row flag."""
    custom = {"input": {"video": {"temporal": {"rule": "custom", "description": "the model's own clip sampler"},
                                  "spatial": {"rule": "custom", "description": "random-resized-crop eval transform"}}}}
    rc = {"video": {"temporal": "custom", "spatial": False}}
    rep = build_report(_records(
        make_result(model="optout", describe_extra=custom, rules_checked=rc, weights_sha256=None, seed=0),
        make_result(model="optout", describe_extra=custom, rules_checked=rc, weights_sha256=None, seed=1, acc=0.52),
        make_result(model="plain", weights_sha256=None),
    ), detail=True)
    by = {r["model"]: r for r in rep.rows}
    assert by["optout"]["flags"] == [] and not by["optout"]["greyed"] and by["optout"]["rules_checked"] == rc
    assert by["plain"]["flags"] == ["single_seed"] and not by["plain"]["greyed"]
    ws = [w for w in rep.warnings if w["code"] == "W_WEIGHTS_UNFINGERPRINTED"]
    assert len(ws) == 1 and "optout v on k400" in ws[0]["message"] and "plain v on k400" in ws[0]["message"]
    assert "~~optout~~" not in rep.text and "custom" not in ",".join(by["optout"]["flags"])
    d = rep.detail[rep.columns[0]["id"]]["optout:v"]
    assert d["rules_checked"] == rc and d["describe"]["input"]["video"]["temporal"]["rule"] == "custom"
    assert "the model's own clip sampler" in rep.text  # --detail prints the custom-rule description


# --------------------------------------------------------------------------- population differs


def test_population_differs_marks_column_and_rows():
    rep = build_report(_records(make_result(model="m1"), make_result(model="m2", ids_sha256=IDS2)))
    assert rep.columns[0]["population_differs"] is True
    assert all("population_differs" in r["flags"] for r in rep.rows)
    assert any(w["code"] == "W_POPULATION_DIFFERS" for w in rep.warnings)
    assert "population differs" in rep.text


def test_population_differs_below_min_coverage_refused_unless_allowed():
    recs = _records(make_result(model="m1"), make_result(model="m2", ids_sha256=IDS2, eval_present=80, eval_nominal=100))
    with pytest.raises(ReportError) as ei:
        build_report(recs)
    assert ei.value.code == "E_POPULATION_DIFFERS" and ei.value.exit_code == 2
    rep = build_report(recs, allow_coverage_diff=True)
    assert rep.columns[0]["population_differs"]
    assert {r["model"]: r["coverage"]["eval"] for r in rep.rows} == {"m1": 1.0, "m2": 0.8}


def test_same_population_low_coverage_is_not_refused():
    rep = build_report(_records(make_result(model="m1", eval_present=50, eval_nominal=100), make_result(model="m2", eval_present=50, eval_nominal=100)))
    assert not rep.columns[0]["population_differs"]


# --------------------------------------------------------------------------- superseded protocol versions


def test_superseded_protocol_versions_listed():
    rep = build_report(_records(
        make_result(model="m1", protocol_version=1, protocol_hash=PH),
        make_result(model="m2", protocol_version=2, protocol_hash=PH2),
    ))
    assert rep.superseded == [{"benchmark": "k400", "protocol_version": 1, "latest": 2, "eval_hashes": [EH], "n_results": 1}]
    cols = {c["protocol_hash"]: c for c in rep.columns}
    assert cols[PH]["superseded"] is True and cols[PH2]["superseded"] is False
    assert "Superseded protocol versions" in rep.text and "superseded" in cols[PH]["flags"]
    assert build_report(_records(make_result())).superseded == []


# --------------------------------------------------------------------------- model / reading grouping with MISSING


def test_missing_readings_become_missing_cells():
    rep = build_report(_records(
        make_result(bench="ave", readout="seq", model="avm", reading="a"),
        make_result(bench="ave", readout="seq", model="avm", reading="v"),
        make_result(bench="ave", readout="seq", model="avm", reading="av"),
        make_result(bench="k400", model="avm", reading="v"),
        make_result(bench="k400", model="solo", reading="v"),
    ))
    k400 = [r for r in rep.rows if r["benchmark"] == "k400"]
    missing = [r for r in k400 if r["missing"]]
    assert {(r["model"], r["reading"]) for r in missing} == {("avm", "a"), ("avm", "av")}
    assert all(r["status"] == "missing" and r["flags"] == ["missing"] and not r["greyed"] for r in missing)
    ave = [r for r in rep.rows if r["benchmark"] == "ave"]
    assert not any(r["missing"] for r in ave)
    col = next(c for c in rep.columns if c["benchmark"] == "k400")
    assert col["models"] == {"avm": ["a", "av", "v"], "solo": ["v"]}
    assert "MISSING" in rep.text
    assert sum(w["code"] == "W_MISSING_READING" for w in rep.warnings) == 2
    # rows are grouped by model then reading
    assert [(r["model"], r["reading"]) for r in k400] == sorted((r["model"], r["reading"]) for r in k400)
    csv_text = tables.render_csv(rep)
    assert sum(",MISSING," in line for line in csv_text.splitlines()) == 2


def test_missing_readings_limited_to_the_benchmark_modalities():
    """An audio reading is never MISSING on video-only k400, a visual
    one never on audio-only esc50; an AV benchmark still expects every reading of the model."""
    a, v, av = {"streams": ["audio"]}, {"streams": ["video"]}, {"streams": ["video", "audio"]}
    rep = build_report(_records(
        make_result(bench="esc50", model="avmodel", reading="a", describe_extra=a),
        make_result(bench="k400", model="avmodel", reading="v", describe_extra=v),
        make_result(bench="in1k", model="avmodel", reading="v", describe_extra={"streams": ["image"]}),
        make_result(bench="ave", readout="seq", model="avmodel", reading="av", describe_extra=av),
        make_result(bench="vggsounder", model="avmodel", reading="av", describe_extra=av),
    ))
    missing = {(r["benchmark"], r["reading"]) for r in rep.rows if r["missing"]}
    assert missing == {("ave", "a"), ("ave", "v"), ("vggsounder", "a"), ("vggsounder", "v")}
    models = {c["benchmark"]: c["models"]["avmodel"] for c in rep.columns}
    assert models["esc50"] == ["a"] and models["k400"] == ["v"] and models["in1k"] == ["v"] and models["ave"] == ["a", "av", "v"]
    assert sum(w["code"] == "W_MISSING_READING" for w in rep.warnings) == 4


def test_missing_readings_unknown_modality_or_benchmark_keeps_v2_behaviour():
    # a stream the benchmark does not list and that is not a kind name: the reading stays expected
    rep = build_report(_records(
        make_result(bench="esc50", model="avmodel", reading="a", describe_extra={"streams": ["audio"]}),
        make_result(bench="k400", model="avmodel", reading="x", describe_extra={"streams": ["spectro_patches"]}),
    ))
    assert {(r["benchmark"], r["reading"]) for r in rep.rows if r["missing"]} == {("esc50", "x")}
    # an unpackaged benchmark (no stream kinds): every reading of the model is expected there
    rep = build_report(_records(
        make_result(bench="my_private_bench", model="avmodel", reading="a", describe_extra={"streams": ["audio"]}),
        make_result(bench="k400", model="avmodel", reading="v", describe_extra={"streams": ["video"]}),
    ))
    assert {(r["benchmark"], r["reading"]) for r in rep.rows if r["missing"]} == {("my_private_bench", "v")}


# --------------------------------------------------------------------------- not_applicable


def test_not_applicable_rows_have_no_metrics():
    rep = build_report(_records(make_result(model="na", status="not_applicable"), make_result(model="ok")))
    by = {r["model"]: r for r in rep.rows}
    assert by["na"]["status"] == "not_applicable" and by["na"]["metrics"] == {} and "not_applicable" in by["na"]["flags"]
    assert by["na"]["not_applicable"]["missing"] == ["pos.y"]
    assert "_not_applicable_" in rep.text and "pos.y" in rep.text
    assert rep.columns[0]["metrics"] == ["acc", "n_eval"]  # from the ok row only


# --------------------------------------------------------------------------- detail


def test_detail_prints_every_rows_configuration():
    rep = build_report(_records(make_result(model="m1"), make_result(model="m2", describe_extra={"layer": 3})), detail=True)
    d = rep.detail[rep.columns[0]["id"]]
    assert set(d) == {"m1:v", "m2:v"} and d["m2:v"]["describe"]["layer"] == 3 and d["m2:v"]["older_configs"] == []
    assert d["m1:v"]["describe_sha256"] == make_result(model="m1")["bridge"]["describe_sha256"]
    assert "Detail" in rep.text
    assert build_report(_records(make_result()), detail=False).detail == {}


# --------------------------------------------------------------------------- renderers


def test_format_value():
    assert tables.format_value(0.4123, 0.164, [0.36, 0.47]) == "0.4123 (0.164) [0.36, 0.47]"
    assert tables.format_value(0.5, None, None, std=0.01, n_seeds=3) == "0.5 ± 0.01 ×3"
    assert tables.format_value(None) == ""
    assert tables.format_number(1568) == "1568" and tables.format_number(True) == "true" and tables.format_number([1, 2.5]) == "[1, 2.5]"


def test_render_formats_agree_and_json_round_trips():
    recs = _records(make_result(model="m1"), make_result(model="m2", acc=0.05, ci=[0.0, 0.1]))
    md = build_report(recs, fmt="md")
    csv_rep = build_report(recs, fmt="csv")
    js = build_report(recs, fmt="json")
    assert md.text.startswith("# yavih report") and "| model | reading |" in md.text and f"eval_hash `{EH[:12]}`" in md.text
    header = csv_rep.text.splitlines()[0].split(",")
    assert header[:8] == ["benchmark", "eval_hash", "protocol_hash", "probe_hash", "model", "reading", "describe_sha256", "features_fingerprint"]
    assert "flag" in header
    data = json.loads(js.text)
    assert data["columns"] == md.columns and data["rows"] == md.rows and data["fmt"] == "json"
    assert all({"eval_hash", "model", "reading", "describe_sha256", "features_fingerprint"} <= set(r) for r in data["rows"])
    assert Report().to_json()["rows"] == []
    with pytest.raises(ValueError):
        tables.render(md, "xml")
    with pytest.raises(ValueError):
        build_report(recs, fmt="xml")


def test_empty_report():
    rep = build_report([])
    assert rep.columns == [] and rep.rows == [] and "no result.json" in rep.text


# --------------------------------------------------------------------------- report() end to end on disk


def test_report_builds_index_tables_and_done(runs: Path):
    write_result(runs, make_result(model="m1", seed=0))
    write_result(runs, make_result(model="m1", seed=1))
    write_result(runs, make_result(model="m2", acc=0.05, ci=[0.0, 0.1]))
    write_result(runs, make_result(model="m3"), done=False)
    rep = report([str(runs / "*")], fmt="md")
    assert rep.n_results == 3 and len(rep.rows) == 2
    by = {r["model"]: r for r in rep.rows}
    assert by["m1"]["n_seeds"] == 2 and by["m2"]["greyed"]
    assert (runs / INDEX_FILE).is_file() and len((runs / INDEX_FILE).read_text().splitlines()) == 3
    done = json.loads((runs / "_report" / "DONE").read_text())
    assert done["n_results"] == 3 and done["fmt"] == "md" and (runs / "_report" / "report.md").read_text() == rep.text
    assert rep.paths["done"].endswith("DONE")
    # csv to an explicit runs root, without side effects when write=False
    rep2 = report([str(runs / "k400")], fmt="csv", runs_root=runs, write=False)
    assert rep2.text.splitlines()[0].startswith("benchmark,") and not (runs / "_report" / "report.csv").exists()
    # a single string glob is accepted
    assert report(str(runs / "*"), write=False).n_results == 3


def test_report_unknown_join_refused(runs: Path):
    write_result(runs, make_result())
    with pytest.raises(ReportError) as ei:
        report([str(runs / "*")], join={"nope": {}}, write=False)
    assert ei.value.exit_code == 1


def test_cli_report_build_json_envelope(runs: Path, capsys):
    from yavih.cli import main

    write_result(runs, make_result(model="m1"))
    code = main(["--json", "report", "build", str(runs / "*"), "--fmt", "md", "--out", str(runs / "out.md")])
    out = capsys.readouterr().out
    env = json.loads(out)
    assert code == 0 and env["status"] == "ok" and len(env["data"]["rows"]) == 1
    assert env["data"]["rows"][0]["eval_hash"] == EH and env["data"]["rows"][0]["model"] == "m1"
    assert (runs / "out.md").read_text() == (runs / "_report" / "report.md").read_text()


def test_cli_report_population_refusal_exit_2(runs: Path, capsys):
    from yavih.cli import main

    write_result(runs, make_result(model="m1"))
    write_result(runs, make_result(model="m2", ids_sha256=IDS2, eval_present=50))
    code = main(["--json", "report", "build", str(runs / "*")])
    env = json.loads(capsys.readouterr().out)
    assert code == 2 and env["status"] == "error" and env["errors"][0]["code"] == "E_POPULATION_DIFFERS"
    assert main(["--json", "report", "build", str(runs / "*"), "--allow-coverage-diff"]) == 0


# ----------------------------------------------------------------------------- floors and coverage flags


def test_floor_resolution_covers_folds_and_box_headlines():
    """Report greying needs each headline's registered floor (esc50 ``acc_mean``/``acc_floor``,
    vggss_box ``iou_mean``/``iou_floor``); the literal ``<name>_floor`` lookup missed both."""
    from yavih.report.compare import build_report, floor_name_for

    assert floor_name_for("acc", ["acc", "n_eval"], ["acc_floor"]) == "acc_floor"
    assert floor_name_for("acc_mean", ["acc_mean", "acc_pstd", "acc_per_fold", "n_folds", "n_eval"], ["acc_floor"]) == "acc_floor"
    assert floor_name_for("acc_pstd", ["acc_mean", "acc_pstd"], ["acc_floor"]) == "acc_floor"  # same stem: harmless, never the headline
    assert floor_name_for("acc_mean", ["acc", "acc_mean"], ["acc_floor"]) is None  # the floor belongs to ``acc`` when it exists
    assert floor_name_for("iou_mean", ["iou_mean", "iou_shuffled", "n_eval"], ["iou_floor"]) == "iou_floor"
    assert floor_name_for("seg_acc_boundary", ["seg_acc", "seg_acc_boundary"], ["seg_acc_floor", "seg_acc_boundary_floor"]) == "seg_acc_boundary_floor"
    assert floor_name_for("n_eval", ["acc", "n_eval"], ["acc_floor"]) is None

    esc_low = make_result(bench="esc50_like", model="low", metrics={"acc_mean": 0.01, "acc_pstd": 0.0, "n_eval": 400}, floors={"acc_floor": 0.02})
    esc_ok = make_result(bench="esc50_like", model="high", metrics={"acc_mean": 0.90, "acc_pstd": 0.01, "n_eval": 400}, floors={"acc_floor": 0.02})
    box_low = make_result(bench="vggss_box_like", model="box", readout="box", metrics={"iou_mean": 0.30, "n_eval": 400}, floors={"iou_floor": 0.44})
    rep = build_report([{"path": f"/r/{i}", "result": r} for i, r in enumerate([esc_low, esc_ok, box_low])])
    rows = {row["model"]: row for row in rep.rows}
    assert rows["low"]["headline"] == "acc_mean" and rows["low"]["metrics"]["acc_mean"]["floor"] == 0.02
    assert "below_floor" in rows["low"]["flags"] and rows["low"]["greyed"]
    assert "below_floor" not in rows["high"]["flags"] and not rows["high"]["greyed"]
    assert rows["box"]["headline"] == "iou_mean" and rows["box"]["metrics"]["iou_mean"]["floor"] == 0.44
    assert "below_floor" in rows["box"]["flags"] and rows["box"]["greyed"]
    assert "~~low~~" in rep.text and "~~box~~" in rep.text and "~~high~~" not in rep.text


def test_lone_low_coverage_result_is_flagged():
    """A result scored on a subset of the eval population (``--allow-low-coverage``, W_COVERAGE) is annotated
    even as the only row of its column, where ``population_differs`` cannot fire."""
    from yavih.report.compare import build_report

    low = make_result(model="low", eval_present=80, eval_nominal=100)
    rep = build_report([{"path": "/r/low", "result": low}])
    assert rep.rows[0]["flags"] == ["single_seed", "low_coverage"] and rep.rows[0]["coverage"]["eval"] == 0.8
    warned = make_result(model="warn")
    warned["warnings"] = [{"code": "W_COVERAGE", "message": "eval 0.9 < 0.95", "where": "val"}]
    rep = build_report([{"path": "/r/w", "result": warned}])
    assert "low_coverage" in rep.rows[0]["flags"]
    full = make_result(model="full")
    assert "low_coverage" not in build_report([{"path": "/r/f", "result": full}]).rows[0]["flags"]
