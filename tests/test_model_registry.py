"""The model registry (``yavih.registry``): export from a results store, the entry contract, ``check``, and
``report build --registry`` comparing local results with registry rows under the one comparability rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_report import make_result, write_result
from yavih import conformance as C
from yavih import hashing
from yavih import registry as R
from yavih import testing as T
from yavih.benchmark import Benchmark
from yavih.report import store


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A synthetic clf folder resolvable by name, its fixture recorded, two seeds of a local run and a results file."""
    from yavih.probes.config import ProbeConfig

    folder = T.make_synthetic_benchmark(tmp_path / "folders", "clf", n_train=12, n_eval=8, media="none")
    monkeypatch.setenv("YAVIH_BENCH_PATH", str(tmp_path / "folders"))
    bench = Benchmark.load(folder.name)
    C.record(bench)
    qh = hashing.probe_hash(ProbeConfig.pinned("clf", bench))
    runs = tmp_path / "runs"
    results = [make_result(bench=bench.name, model="toy", reading="v", seed=s, acc=0.60 + 0.02 * s, protocol_hash=bench.protocol_hash,
                           probe_hash=qh, protocol_version=bench.spec.protocol_version, created=f"2026-10-0{s + 1}T00:00:00Z")
               for s in (0, 1)]
    for r in results:
        r["benchmark"]["folder_source"] = dict(bench.folder_source)
        r["benchmark"]["readout_version"] = 1
    paths = [write_result(runs, r) for r in results]
    results_root = tmp_path / "results"
    file = store.write_invocation(results, results_root=results_root, bench=bench, result_paths=[str(p) for p in paths])["results_file"]
    return {"bench": bench, "qh": qh, "runs": runs, "results_root": results_root, "file": Path(file), "results": results}


def test_export_writes_entries_and_protocol_payloads(world, tmp_path):
    reg = tmp_path / "registry"
    out = R.export(world["results_root"], org="lab", out=reg)
    assert len(out["entries"]) == 1 and len(out["protocols"]) == 1 and out["skipped"] == []
    bench = world["bench"]
    entry = json.loads(Path(out["entries"][0]).read_text())
    assert Path(out["entries"][0]) == reg / "entries" / bench.name / hashing.hash12(hashing.eval_hash(bench.protocol_hash, world["qh"])) / "lab" / "toy" / "v.json"
    assert entry["registry_version"] == R.REGISTRY_ENTRY_VERSION
    assert entry["id"] == f"{bench.name}/{hashing.hash12(hashing.eval_hash(bench.protocol_hash, world['qh']))}/lab/toy/v"
    i = entry["identity"]
    assert (i["protocol_hash"], i["probe_hash"], i["input_hash"], i["scoring_hash"]) == (bench.protocol_hash, world["qh"], bench.input_hash, bench.scoring_hash)
    assert i["standard"] is True and i["protocol_version"] == bench.spec.protocol_version
    assert entry["model"] == {**entry["model"], "name": "lab/toy", "org": "lab", "model": "toy"}
    s = entry["score"]
    assert s["metrics"]["acc"] == pytest.approx(0.61) and s["n_seeds"] == 2 and [sd["seed"] for sd in s["seeds"]] == [0, 1]
    assert s["seeds"][0]["metrics"]["acc"] == 0.60 and s["seeds"][1]["metrics"]["acc"] == 0.62  # per-seed numbers from the unit results
    assert s["floors"]["acc"] == pytest.approx(0.1) and "acc_floor" in s["floor_names"]
    assert entry["conformance"]["passed"] is True and entry["conformance"]["tolerance"]["acc"] == C.DEFAULT_TOLERANCE
    assert entry["provenance"]["folder_source"] == bench.folder_source and entry["provenance"]["results_file"].endswith(".json")
    assert entry["population"]["coverage"]["eval"]["ids_sha256"] == "e" * 64
    payload = json.loads(Path(out["protocols"][0]).read_text())
    assert payload["protocol_hash"] == bench.protocol_hash and payload["expected"]["protocol_version"] == bench.spec.protocol_version
    assert R._recompute_protocol_hash(payload["explain"]) == bench.protocol_hash
    # a resubmission replaces the file and records the previous one
    out2 = R.export(world["results_root"], org="lab", out=reg)
    assert len(out2["protocols"]) == 0 and json.loads(Path(out2["entries"][0]).read_text())["provenance"]["replaces"]


def test_check_accepts_a_clean_registry_and_names_every_defect(world, tmp_path):
    reg = tmp_path / "registry"
    R.export(world["results_root"], org="lab", out=reg)
    rep = R.check(reg)
    assert rep.ok and rep.n_entries == 1 and rep.n_protocols == 1 and rep.findings == []
    path = next((reg / "entries").rglob("*.json"))
    entry = json.loads(path.read_text())

    def with_entry(mutate):
        e = json.loads(json.dumps(entry))
        mutate(e)
        path.write_text(json.dumps(e))
        return R.check(reg)

    assert [f.code for f in with_entry(lambda e: e["identity"].__setitem__("probe_hash", "0" * 64)).errors] == [R.E_HASH]
    assert R.E_CONFORMANCE in [f.code for f in with_entry(lambda e: e["conformance"].__setitem__("passed", False)).errors]
    assert R.E_ID in [f.code for f in with_entry(lambda e: e.__setitem__("id", "x")).errors]
    assert R.W_NONSTANDARD in [f.code for f in with_entry(lambda e: e["identity"].__setitem__("standard", False)).warnings]
    assert R.W_COVERAGE in [f.code for f in with_entry(lambda e: e["population"]["coverage"]["eval"].update(present=50, nominal=100)).warnings]
    assert R.E_SCHEMA in [f.code for f in with_entry(lambda e: e.pop("score")).errors]
    path.write_text(json.dumps(entry))
    for p in (reg / "protocols").rglob("*.json"):
        p.unlink()
    assert [f.code for f in R.check(reg).errors] == [R.E_PROTOCOL]


def test_export_skips_what_is_not_a_score(world, tmp_path):
    from test_report import make_result as mk

    bench = world["bench"]
    bad = [mk(bench=bench.name, model="toy2", reading="v", seed=0, protocol_hash=bench.protocol_hash, probe_hash=world["qh"], standard=False)]
    file = store.write_invocation(bad, results_root=world["results_root"], bench=bench, result_paths=[])["results_file"]
    out = R.export(file, org="lab", out=tmp_path / "r2")
    assert out["entries"] == [] and out["skipped"] and "non-standard" in out["skipped"][0]["reason"]
    out = R.export(file, org="lab", out=tmp_path / "r2", include_nonstandard=True)
    assert len(out["entries"]) == 1
    with pytest.raises(R.RegistryError):
        R.export(world["results_root"], org="not valid!", out=tmp_path / "r3")


def test_report_build_compares_local_rows_with_the_registry(world, tmp_path):
    from yavih.report import report

    reg = tmp_path / "registry"
    R.export(world["results_root"], org="lab", out=reg)
    bench = world["bench"]
    other = [make_result(bench=bench.name, model="mine", reading="v", seed=0, acc=0.7, protocol_hash=bench.protocol_hash, probe_hash=world["qh"],
                         protocol_version=bench.spec.protocol_version)]
    other[0]["benchmark"]["folder_source"] = dict(bench.folder_source)
    runs2 = tmp_path / "runs2"
    write_result(runs2, other[0])
    rep = report([str(runs2 / "*")], registry=reg, write=False)
    rows = {(r["model"], r["reading"]): r for r in rep.rows}
    assert ("mine", "v") in rows and ("lab/toy", "v") in rows
    mine, theirs = rows[("mine", "v")], rows[("lab/toy", "v")]
    assert mine["column"] == theirs["column"] and "registry" in theirs["flags"] and "registry" not in mine["flags"]
    assert theirs["n_seeds"] == 2 and theirs["metrics"]["acc"]["value"] == pytest.approx(0.61) and theirs["registry"].endswith("/lab/toy/v")
    assert mine["metrics"]["acc"]["value"] == 0.7
    assert len(rep.columns) == 1 and rep.columns[0]["n_rows"] == 2
    # an entry whose conformance did not pass is not a row
    path = next((reg / "entries").rglob("*.json"))
    e = json.loads(path.read_text())
    e["conformance"]["passed"] = False
    path.write_text(json.dumps(e))
    assert ("lab/toy", "v") not in {(r["model"], r["reading"]) for r in report([str(runs2 / "*")], registry=reg, write=False).rows}


def test_cli_registry_export_and_check(world, tmp_path, capsys):
    from yavih import cli

    reg = tmp_path / "registry"

    def run(argv):
        rc = cli.main(["--json", *argv])
        return rc, json.loads(capsys.readouterr().out)

    rc, env = run(["registry", "export", str(world["results_root"]), "--org", "lab", "--out", str(reg)])
    assert rc == 0 and len(env["data"]["entries"]) == 1 and env["paths"]["out"] == str(reg)
    rc, env = run(["registry", "check", str(reg)])
    assert rc == 0 and env["data"]["ok"] is True and env["data"]["n_entries"] == 1
    rc, env = run(["report", "build", str(world["runs"] / "*"), "--registry", str(reg), "--fmt", "json"])
    assert rc == 0 and any(r.get("registry") for r in env["data"]["rows"])
