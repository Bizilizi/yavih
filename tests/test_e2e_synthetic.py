"""End-to-end synthetic run: for every readout, ``make_synthetic_benchmark`` (raw tiny media) →
``yavih bench verify --census`` → ``yavih bridge check`` (hard invariance checks) → ``python -m yavih.extract``
with ``random_bridge`` and ``oracle_bridge`` → ``yavih features validate`` (rules checked, eligibility in media
coordinates) → ``probe`` under ``ProbeConfig.pinned(r).fast()`` (results store written) → ``yavih report build``;
the oracle clears the floor, the random bridge does not, and the report greys the random row out. Run
directories are ``runs/<bench>/<model>/<reading>/`` from the profile, both bridges share one ``eval_hash`` (what
yavih controls) whatever they declare, the random bridge's ``custom`` rules are the note ``N_RULE_CUSTOM`` (never a
warning), and the results store is ``<results_root>/<model>/`` with entries keyed ``<benchmark>:<reading>``.

Marked ``integration`` (and ``slow``): it depends on every module (M1–M10) and passes only once they are
integrated. Deselect with ``-m "not integration"``. The synthetic benchmark lives under
``$YAVIH_HOME/benchmarks/<name>`` so every stage can resolve it by name (benchmark resolution order).
GPU is used when available; the CPU path is the same code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from yavih import testing as T

pytestmark = [pytest.mark.integration, pytest.mark.slow]

N_TRAIN, N_EVAL, N_SHARDS, D = 64, 32, 2, 32
HEADLINE = {"clf": ("acc", "acc_floor"), "seq": ("seg_acc", "seg_acc_floor"), "grid": ("d1", "d1_floor"), "box": ("iou_mean", "iou_floor")}


@pytest.fixture(scope="module")
def integration_env():
    """Kept as a no-op hook: the core is complete, so a missing entry point is a FAILURE, never a skip."""
    return None


def _run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic CuBLAS under use_deterministic_algorithms
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, env=env, timeout=kw.pop("timeout", 900), **kw)


def _ok(proc: subprocess.CompletedProcess, what: str) -> subprocess.CompletedProcess:
    assert proc.returncode == 0, f"{what} exit {proc.returncode}\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
    return proc


def _codes(findings) -> set[str]:
    """Codes of a findings list (dicts with ``code``, or ``"CODE: message"`` strings)."""
    return {f["code"] if isinstance(f, dict) else str(f).split(":", 1)[0] for f in (findings or [])}


def _json(proc: subprocess.CompletedProcess) -> dict:
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 1, f"--json must print exactly one object, got:\n{proc.stdout}"
    env = json.loads(lines[0])
    assert set(env) == {"status", "data", "errors", "warnings", "paths"}
    return env


def _extract_all(bench_name: str, bridge_file: Path, run_dir: Path, splits: list[str]) -> None:
    for split in splits:
        for i in range(N_SHARDS):
            _ok(
                _run(["-m", "yavih.extract", "--bench", bench_name, "--bridge", str(bridge_file), "--split", split, "--shard", f"{i}/{N_SHARDS}", "--out", str(run_dir), "--canaries", "2"]),
                f"extract {bench_name} {bridge_file.name} {split} {i}/{N_SHARDS}",
            )


@pytest.mark.parametrize("readout", T.READOUTS)
def test_end_to_end_synthetic(readout, integration_env, yavih_home, runs_dir, tmp_path, monkeypatch):
    from yavih.benchmark import Benchmark
    from yavih.probes.config import ProbeConfig
    from yavih.readouts import probe

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # 1. benchmark folder under $YAVIH_HOME/benchmarks so `--bench <name>` resolves everywhere
    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", readout, n_train=N_TRAIN, n_eval=N_EVAL, media="tiny")
    name = folder.name
    bench = Benchmark.load(name)
    assert bench.folder.resolve() == folder.resolve() and bench.spec.readout == readout
    roles = bench.roles()
    splits = list(roles["train"]) + list(roles["eval"])
    headline, floor_name = HEADLINE[readout]
    metrics_spec = bench.spec.metrics

    # bench verify --census writes the media-info table; doctor then sees it
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "bench", "verify", name, "--census", "--decode-sample", "4"]), "bench verify"))
    d = env["data"]
    assert d["manifest_sha_ok"] and d["labels_ok"] and d["curation_invariants_ok"]
    assert d["read"]["failed"] == 0 and d["read"]["ok"] > 0 and d["decode_sample"]["failed"] == 0 and d["decode_sample"]["ok"] > 0
    assert all(v["present"] == v["nominal"] for v in d["per_split"].values()), d["per_split"]
    assert d["media_info"]["present"] and d["media_info"]["rows"] > 0 and len(d["media_info"]["sha256"]) == 64
    assert d["external"] == {} and "materialized_sha_ok" not in d
    assert bench.media_info() is not None  # the census the reader reads facts from (no decoder needed)
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "doctor", "--bench", name]), "doctor"))
    assert env["status"] == "ok" and env["data"]["media_read"]["ok"] is True and env["data"]["media_info"]["present"] is True
    assert all(r["exists"] for r in env["data"]["data_roots"].values())
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "bench", "info", name, "--setup"]), "bench info --setup"))
    assert env["data"]["setup"].startswith(f"# {name} — setup")
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "bench", "list"]), "bench list"))
    row = next(r for r in env["data"] if r["name"] == name)
    assert row["readout"] == readout and row["clips"] == {"train": N_TRAIN, "val": N_EVAL}

    # 2. bridges as files (what --bridge loads)
    bridges = {kind: T.write_bridge_file(tmp_path / f"{kind}_bridge.py", kind, D=D) for kind in ("random", "oracle")}

    # bridge check: rerun bit-identical, alone (scrambled identity) and shuffled passes invariant;
    # profile written with `invariance`. The oracle declares checkable rules; the random bridge `custom`.
    profiles = {}
    for kind in ("oracle", "random"):
        profile = tmp_path / f"{kind}.profile.json"
        env = _json(_ok(_run(["-m", "yavih.cli", "--json", "bridge", "check", str(bridges[kind]), "--bench", name, "--n", "8", "--out", str(profile)]), f"bridge check {kind}"))
        p = env["data"]
        profiles[kind] = p
        inv = p["invariance"]
        assert inv["rerun_bit_identical"] is True and inv["batch_rel_l2"] == 0.0 and inv["order_rel_l2"] == 0.0 and inv["tolerance"] == 1e-5, inv
        on_disk = json.loads(profile.read_text())
        # the profile names the model and reading (describe())
        assert p["D"] == D and profile.is_file() and (on_disk["model"], on_disk["reading"]) == (p["model"], p["reading"]) == (f"yavih-{kind}", "synthetic")
        assert on_disk["describe_sha256"] == p["describe_sha256"]
        # the random bridge's custom rules are an opt-out: a note, never W_RULE_UNCHECKED
        assert ("N_RULE_CUSTOM" in _codes(p.get("notes"))) == (kind == "random"), (kind, p.get("notes"))
        assert "W_RULE_UNCHECKED" not in _codes(env["warnings"]), env["warnings"]
        assert set(p["used"]) == set(bench.spec.streams) and set(p["media"]) == set(bench.spec.streams)
        expect_temporal = "custom" if kind == "random" else None
        for sname, rc in p["rules_checked"].items():
            if bench.spec.streams[sname].kind != "image":
                assert (rc["temporal"] == "custom") == (expect_temporal == "custom"), (kind, sname, rc)
        assert not [w for w in env["warnings"] if w["code"].startswith("E_")]

    # 3. extract every split in N_SHARDS shards per bridge (python -m yavih.extract), one run dir per bridge:
    # runs/<benchmark>/<model>/<reading>/
    run_dirs = {kind: runs_dir / name / profiles[kind]["model"] / profiles[kind]["reading"] for kind in bridges}
    t0 = time.time()
    for kind, bfile in bridges.items():
        _extract_all(name, bfile, run_dirs[kind], splits)
    extract_s = time.time() - t0
    for kind in bridges:
        for split in splits:
            sidecars = sorted((run_dirs[kind] / "features" / split).glob("shard-*.json"))
            assert len(sidecars) == N_SHARDS, f"{kind}/{split}: {sidecars}"
            sc = json.loads(sidecars[0].read_text())
            assert sc["rows"]["n_missing"] == 0 and sc["features"]["D"] == D and sc["shard"]["n"] == N_SHARDS
            assert (sc["model"], sc["reading"]) == (profiles[kind]["model"], profiles[kind]["reading"]), sc.get("model")
    # resumption: a matching sidecar → exit 0, nothing rewritten
    sidecar = run_dirs["oracle"] / "features" / splits[0] / "shard-000000.json"
    before = sidecar.stat().st_mtime_ns
    _ok(_run(["-m", "yavih.extract", "--bench", name, "--bridge", str(bridges["oracle"]), "--split", splits[0], "--shard", f"0/{N_SHARDS}", "--out", str(run_dirs["oracle"]), "--canaries", "2"]), "extract rerun")
    assert sidecar.stat().st_mtime_ns == before

    # 4. validate (CLI, both forms): ok, 100 % coverage, eligible, _meta.json written
    for kind in bridges:
        env = _json(_ok(_run(["-m", "yavih.cli", "--json", "features", "validate", str(run_dirs[kind] / "features")]), f"validate {kind}"))
        assert env["status"] == "ok" and env["errors"] == [], env["errors"][:5]
        reports = env["data"] if isinstance(env["data"], list) else [env["data"]]
        assert {r["split"] for r in reports} == set(splits)
        for r in reports:
            assert r["ok"] and r["coverage"]["present"] == r["coverage"]["nominal"] and r["eligibility"]["eligible"] is True, r
            assert (run_dirs[kind] / "features" / r["split"] / "_meta.json").is_file()
            # a custom rule is the note N_RULE_CUSTOM and no warning; the oracle's rules are checked
            assert ("N_RULE_CUSTOM" in _codes(r["notes"])) == (kind == "random"), (kind, r["notes"])
            assert "W_RULE_UNCHECKED" not in _codes(r["warnings"]), (kind, r["warnings"])
    text = _ok(_run(["-m", "yavih.cli", "features", "validate", str(run_dirs["oracle"] / "features")]), "validate human").stdout
    assert f"eligible for {readout}: yes" in text and "missing shards: 0" in text and "rules checked:" in text
    assert "opted out" not in text and "N_RULE_CUSTOM" not in text
    # the random bridge's opt-out prints its description on the rules line and as a note, never a warning
    text = _ok(_run(["-m", "yavih.cli", "features", "validate", str(run_dirs["random"] / "features")]), "validate human random")
    rules = next(ln for ln in text.stdout.splitlines() if "rules checked:" in ln)
    # "audio custom (opted out: <d>)"; an image stream has no temporal rule: "image (opted out: spatial: <d>)"
    assert "(opted out: " in rules and "no decoding: the stored bytes are hashed" in rules, rules
    assert "note N_RULE_CUSTOM:" in text.stdout and "W_RULE_UNCHECKED" not in text.stdout + text.stderr

    # 5. probe under ProbeConfig.pinned(r).fast(): oracle clears the floor, random does not
    from yavih.hashing import eval_hash

    cfg = T.fast(ProbeConfig.pinned(readout, bench))
    results = {}
    for kind in bridges:
        res = probe(run_dirs[kind], seeds=(0,), cfg=cfg)
        assert len(res) == 1 and res[0].status == "ok", (kind, res[0].status, res[0].reason)
        results[kind] = res[0]
        rj = res[0].result
        assert rj["standard"] is False and rj["schema_version"] == 4 and rj["benchmark"]["readout"] == readout and rj["external"] == {}
        assert rj["features"]["rules_checked"] and rj["features"]["batch_tolerance"] == 1e-5
        # model and reading recorded; eval_hash = sha256(canon({protocol_hash, probe_hash}))
        assert (rj["model"], rj["reading"]) == (profiles[kind]["model"], profiles[kind]["reading"])
        assert rj["eval_hash"] == eval_hash(rj["benchmark"]["protocol_hash"], rj["probe"]["hash"])
        assert res[0].path.parent.name == "seed0" and res[0].path.parent.parent.name == rj["eval_hash"][:12]
        assert res[0].path.parent.parent.parent == run_dirs[kind] / "probe"
        assert rj["bridge"]["describe_sha256"] == profiles[kind]["describe_sha256"]
        temporal = {s: rc.get("temporal") for s, rc in rj["features"]["rules_checked"].items() if bench.spec.streams[s].kind != "image"}
        assert all((v == "custom") == (kind == "random") for v in temporal.values()), (kind, temporal)
        assert set(rj["metrics"]) == set(metrics_spec.list), (set(rj["metrics"]), metrics_spec.list)
        assert set(rj["floors"]) == set(metrics_spec.floors) and set(rj["controls"]) == set(metrics_spec.controls)
        assert set(rj["ci95"]) == {k for k, v in metrics_spec.ci.items() if v}
        assert res[0].path.is_file() and (res[0].path.parent / "DONE").is_file() and (res[0].path.parent / "eval_logits.npy").is_file()
        assert rj["coverage"]["eval"]["present"] == N_EVAL
        # the lm-eval style results store: <results_root>/<model>/results_<timestamp>.json, key <benchmark>:<reading>
        assert res[0].results_file is not None and res[0].results_file.is_file(), res[0].results_file
        doc = json.loads(res[0].results_file.read_text())
        key = f"{name}:{rj['reading']}"
        assert res[0].results_file.parent.name == rj["model"] and res[0].results_file.parent.parent == runs_dir.parent / "results"
        assert doc["results_version"] == 2 and doc["versions"] == {key: 1}
        assert doc["results"][key][headline] == rj["metrics"][headline] and doc["config"]["model"] == rj["model"]
        c = doc["configs"][key]
        assert (c["benchmark"], c["reading"], c["eval_hash"]) == (name, rj["reading"], rj["eval_hash"])
        assert c["bridge"]["describe_sha256"] == rj["bridge"]["describe_sha256"]
    floor = results["oracle"].result["floors"][floor_name]
    assert results["random"].result["floors"][floor_name] == floor
    oracle_lo, _ = results["oracle"].result["ci95"][headline]
    random_lo, _ = results["random"].result["ci95"][headline]
    assert oracle_lo > floor, f"oracle {headline} CI lower {oracle_lo} <= floor {floor}: {results['oracle'].result['metrics']}"
    assert random_lo <= floor, f"random {headline} CI lower {random_lo} > floor {floor}: {results['random'].result['metrics']}"
    assert results["oracle"].result["metrics"][headline] > results["random"].result["metrics"][headline]
    # the readout's conformance band (readouts/<name>.py CONFORMANCE): a trainer or readout change that leaves it bumps VERSION
    import importlib

    band = importlib.import_module(f"yavih.readouts.{readout}").CONFORMANCE
    assert results["oracle"].result["metrics"][headline] >= band["oracle_min"], band
    assert results["random"].result["metrics"][headline] <= band["random_max"], band
    # comparability is what yavih controls: two bridges declaring different rules, weights and args share one eval_hash
    assert results["oracle"].result["eval_hash"] == results["random"].result["eval_hash"]
    assert results["oracle"].result["bridge"]["describe_sha256"] != results["random"].result["bridge"]["describe_sha256"]

    # the CLI probe path with the PINNED config (standard result) on the oracle, clf only (dense lr grids are slow)
    if readout == "clf":
        env = _json(_ok(_run(["-m", "yavih.cli", "--json", "probe", "run", str(run_dirs["oracle"]), "--seeds", "1"]), "probe run"))
        assert env["status"] == "ok" and env["data"][0]["status"] == "ok" and env["data"][0]["seed"] == 1
        assert env["data"][0]["metrics"][headline] > env["data"][0]["floors"][floor_name]

    # 6. report: both rows in one column (one eval_hash), the random row greyed out for its floor, never for its
    # custom rules
    proc = _ok(_run(["-m", "yavih.cli", "report", "build", str(runs_dir / name / "*")]), "report build")
    oracle_m, random_m = results["oracle"].result["model"], results["random"].result["model"]
    assert oracle_m in proc.stdout and random_m in proc.stdout and results["oracle"].result["eval_hash"][:12] in proc.stdout
    lines = proc.stdout.splitlines()
    assert any("~~" in ln for ln in lines if random_m in ln), [ln for ln in lines if random_m in ln]
    assert not any("~~" in ln for ln in lines if oracle_m in ln), [ln for ln in lines if oracle_m in ln]
    assert "W_BRIDGE_CHANGED" not in proc.stdout + proc.stderr  # one configuration per (model, reading)
    out = tmp_path / "report.json"
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "report", "build", str(runs_dir / name / "*"), "--fmt", "json", "--out", str(out)]), "report json"))
    assert env["status"] == "ok" and env["paths"]["out"] == str(out) and out.is_file()
    sys.stderr.write(f"\n[e2e {readout}] extract {extract_s:.1f}s; oracle {headline}={results['oracle'].result['metrics'][headline]:.3f} random={results['random'].result['metrics'][headline]:.3f} floor={floor:.3f}\n")


def test_features_stage_roundtrip(integration_env, yavih_home, runs_dir, tmp_path):
    """``features stage`` copies a finished feature set; the staged copy validates like the original."""
    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=8, n_eval=4, media="tiny")
    bfile = T.write_bridge_file(tmp_path / "rb.py", "random", D=16)
    run_dir = runs_dir / folder.name / "yavih-random" / "synthetic"
    _extract_all(folder.name, bfile, run_dir, ["train", "val"])
    _ok(_run(["-m", "yavih.cli", "features", "validate", str(run_dir / "features")]), "validate")
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "features", "stage", str(run_dir / "features"), str(tmp_path / "scratch")]), "stage"))
    staged = Path(env["data"]["staged"])
    assert staged.parent == tmp_path / "scratch" and set(env["data"]["splits"]) == {"train", "val"}
    # <scratch>/<unit|run>.<host>.<pid>/features/<split>: the layout `probe run --scratch <staged>` consumes
    assert (staged / "features" / "val" / "_meta.json").is_file() and len(list((staged / "features" / "val").glob("shard-*.tar"))) == N_SHARDS
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "features", "validate", str(staged / "features"), "--no-write-meta"]), "validate staged"))
    assert env["status"] == "ok" and all(r["ok"] for r in (env["data"] if isinstance(env["data"], list) else [env["data"]]))


def _unit_path(unit: dict, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(unit["cwd"]) / p


def test_plan_build_exec_report_chain(integration_env, yavih_home, runs_dir, tmp_path, monkeypatch):
    """The REAL chain: ``bridge check`` → ``plan build`` → ``exec`` every unit (the
    extract child is ``python -m yavih.extract``, probe and report run in-process) → ``plan status`` all done →
    the report names the model; a second ``exec`` of a finished unit is a skip."""
    import yaml

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=32, n_eval=16, media="tiny")
    name = folder.name
    work = tmp_path / "work"
    (work / "bridges").mkdir(parents=True)
    bfile = T.write_bridge_file(work / "bridges" / "oracle.py", "oracle", D=16)
    profile = work / "bridges" / "oracle.profile.json"
    _ok(_run(["-m", "yavih.cli", "bridge", "check", str(bfile), "--bench", name, "--n", "8", "--out", str(profile)], cwd=work), "bridge check")
    prof = json.loads(profile.read_text())
    model, reading = prof["model"], prof["reading"]
    assert (model, reading) == ("yavih-oracle", "synthetic")
    matrix = {
        "schema_version": 1, "out": "runs", "seeds": [0], "canaries": 2, "shard_minutes": 30,
        "defaults": {"launcher": sys.executable, "cpus": 2},
        "bridges": {"ob": {"script": "bridges/oracle.py", "benchmark": name, "args": {}, "profile": "bridges/oracle.profile.json"}},
    }
    (work / "matrix.yaml").write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    plan = work / "plan.jsonl"
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "plan", "build", str(work / "matrix.yaml"), "--out", str(plan)], cwd=work), "plan build"))
    units = env["data"]
    kinds = [u["kind"] for u in units]
    assert kinds.count("probe") == 1 and kinds.count("report") == 1 and kinds.count("extract") >= 2 and kinds == sorted(kinds, key=["extract", "probe", "report"].index)
    for u in units:
        if u["kind"] == "extract":
            cmd = u["cmd"]
            # `--out` is the RUN dir runs/<benchmark>/<model>/<reading>: the child's
            # shard lands exactly at `done`; drift is checked against the profile alone (no --expect-model-id)
            assert cmd[cmd.index("--out") + 1] == os.path.join("runs", name, model, reading), cmd
            assert "--expect-profile" in cmd, cmd
            assert u["done"] == os.path.join(cmd[cmd.index("--out") + 1], "features", u["split"], f"shard-{u['shard'][0]:06d}.json")
    for i, u in enumerate(units):
        env = _json(_ok(_run(["-m", "yavih.cli", "--json", "exec", str(plan), "--index", str(i)], cwd=work, timeout=1200), f"exec {i} {u['kind']}"))
        assert env["status"] == "ok" and env["data"]["status"] == "done", (u["kind"], env["data"])
        assert _unit_path(u, u["done"]).is_file(), (u["kind"], u["done"])
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "plan", "status", str(plan)], cwd=work), "plan status"))
    st = env["data"]
    assert st["done"] == st["total"] == len(units) and st["failed"] == 0 and st["not_ready"] == 0 and st["pending"] == 0, st
    report_md = (work / "runs" / "_report" / "report.md").read_text()
    assert model in report_md and "synth_clf" in report_md
    result = json.loads((_unit_path(units[kinds.index("probe")], units[kinds.index("probe")]["done"]).parent / "result.json").read_text())
    # the planned probe runs the PINNED schedule (standard result); on 32 synthetic rows that schedule
    # underfits, so only the chain is asserted here — "oracle clears the floor" is the fast() contract above
    assert result["status"] == "ok" and result["standard"] is True and (result["model"], result["reading"]) == (model, reading)
    assert "acc" in result["metrics"] and "acc_floor" in result["floors"]
    done = json.loads(_unit_path(units[kinds.index("probe")], units[kinds.index("probe")]["done"]).read_text())
    assert (done["model"], done["reading"]) == (model, reading) and done["features_fingerprint"] == result["features"]["fingerprint"]
    # the probe DONE carries eval_hash and the configuration's describe_sha256 (staleness, not identity)
    assert done["eval_hash"] == result["eval_hash"] and done["describe_sha256"] == prof["describe_sha256"]
    # a finished unit is skipped, nothing re-run
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "exec", str(plan), "--index", "0"], cwd=work), "exec again"))
    assert env["status"] == "skipped" and env["data"]["status"] == "skipped"
    # --pending emits nothing once everything is done
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "plan", "build", str(work / "matrix.yaml"), "--pending", "--format", "count"], cwd=work), "plan pending"))
    assert env["data"]["count"] == 0


def test_plan_build_sample_chain(integration_env, yavih_home, runs_dir, tmp_path, monkeypatch):
    """``plan build --sample 4``: the exec chain of a capped run. One ``--rows 0:4`` extract unit per split (the first
    rows in manifest order), under ``<out>-sample4`` (its own runs root), with unit ids distinct from the full plan's;
    the layouts and ``_meta.json`` record ``sample``; the probe writes no results store; the report row is ``partial``
    in md, csv and json; and a full extract into the sampled run directory is refused (``E_LAYOUT``)."""
    import yaml

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=32, n_eval=16, media="tiny")
    name = folder.name
    work = tmp_path / "work"
    (work / "bridges").mkdir(parents=True)
    bfile = T.write_bridge_file(work / "bridges" / "oracle.py", "oracle", D=16)
    profile = work / "bridges" / "oracle.profile.json"
    _ok(_run(["-m", "yavih.cli", "bridge", "check", str(bfile), "--bench", name, "--n", "8", "--out", str(profile)], cwd=work), "bridge check")
    prof = json.loads(profile.read_text())
    model, reading = prof["model"], prof["reading"]
    matrix = {
        "schema_version": 1, "out": "runs", "seeds": [0], "canaries": 2, "shard_minutes": 30,
        "defaults": {"launcher": sys.executable, "cpus": 2},
        "bridges": {"ob": {"script": "bridges/oracle.py", "benchmark": name, "args": {}, "profile": "bridges/oracle.profile.json"}},
    }
    (work / "matrix.yaml").write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    plan = work / "plan-sample.jsonl"
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "plan", "build", str(work / "matrix.yaml"), "--sample", "4", "--out", str(plan)], cwd=work),
                    "plan build --sample"))
    units = env["data"]
    full = _json(_ok(_run(["-m", "yavih.cli", "--json", "plan", "build", str(work / "matrix.yaml")], cwd=work), "plan build"))["data"]
    again = _json(_ok(_run(["-m", "yavih.cli", "--json", "plan", "build", str(work / "matrix.yaml"), "--sample", "4"], cwd=work), "again"))["data"]
    assert [u["id"] for u in again] == [u["id"] for u in units] and not {u["id"] for u in units} & {u["id"] for u in full}
    run_rel = os.path.join("runs-sample4", name, model, reading)
    extracts = [u for u in units if u["kind"] == "extract"]
    assert sorted(u["split"] for u in extracts) == ["train", "val"] and all(u["shard"] == [0, 1] for u in extracts)
    for u in extracts:
        cmd = u["cmd"]
        assert cmd[cmd.index("--rows") + 1] == "0:4" and "--shard" not in cmd and cmd[cmd.index("--out") + 1] == run_rel, cmd
    (probe_u,) = [u for u in units if u["kind"] == "probe"]
    assert "--no-results" in probe_u["cmd"] and "--results" not in probe_u["cmd"] and probe_u["out"] == run_rel
    (report_u,) = [u for u in units if u["kind"] == "report"]
    assert report_u["out"] == os.path.join("runs-sample4", "_report")
    for i, u in enumerate(units):
        env = _json(_ok(_run(["-m", "yavih.cli", "--json", "exec", str(plan), "--index", str(i)], cwd=work, timeout=1200), f"exec {i} {u['kind']}"))
        assert env["status"] == "ok" and env["data"]["status"] == "done", (u["kind"], env["data"])
    st = _json(_ok(_run(["-m", "yavih.cli", "--json", "plan", "status", str(plan)], cwd=work), "plan status"))["data"]
    assert st["done"] == st["total"] == len(units) and st["failed"] == 0, st
    run_dir = work / run_rel
    for split in ("train", "val"):
        sd = run_dir / "features" / split
        assert json.loads((sd / "_layout.json").read_text())["sample"] == [0, 4]
        meta = json.loads((sd / "_meta.json").read_text())
        assert meta["sample"] == [0, 4] and meta["coverage"]["nominal"] == meta["coverage"]["present"] == 4
    result = json.loads((_unit_path(probe_u, probe_u["done"]).parent / "result.json").read_text())
    assert result["status"] == "ok" and result["sample"] == {"train": [0, 4], "val": [0, 4]}
    assert not list(tmp_path.rglob("results_*.json")), "a sampled result must never reach the results store"
    assert not (work / "runs").exists()  # nothing under the full run's root
    md = (work / "runs-sample4" / "_report" / "report.md").read_text()
    assert "(partial)" in md and model in md
    rep = _json(_ok(_run(["-m", "yavih.cli", "--json", "report", "build", str(work / "runs-sample4" / "*"), "--fmt", "json"]), "report json"))["data"]
    assert rep["rows"] and all(r["partial"] is True and "partial" in r["flags"] for r in rep["rows"])
    assert all(r.get("partial") for rows in rep["lm_eval"].values() for r in rows)
    csv_text = _ok(_run(["-m", "yavih.cli", "report", "build", str(work / "runs-sample4" / "*"), "--fmt", "csv"]), "report csv").stdout
    header, *lines = csv_text.splitlines()
    col = header.split(",").index("partial")
    assert lines and all(ln.split(",")[col] == "true" for ln in lines)
    # the sampled run directory never takes a full-split shard
    proc = _run(["-m", "yavih.extract", "--bench", name, "--bridge", str(bfile), "--split", "train", "--shard", "0/1", "--out", str(run_dir),
                 "--canaries", "0"], cwd=work)
    assert proc.returncode == 1 and "E_LAYOUT" in proc.stderr and "sample" in proc.stderr, proc.stderr[-2000:]


def test_features_validate_ineligible_is_ok_and_probe_records_not_applicable(integration_env, yavih_home, runs_dir, tmp_path):
    """A pooled bridge (``pos_mode: none``) on a dense readout validates ``ok`` with
    ``eligibility.eligible == false`` (exit 0, real benchmark through the CLI), and ``yavih probe run``
    then records ``not_applicable`` decided by validate, with a DONE marker."""
    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "seq", n_train=8, n_eval=4, media="tiny")
    bfile = T.write_bridge_file(tmp_path / "pooled_bridge.py", "random", N=1, D=16, pos_mode="none", model="pooled")
    run_dir = runs_dir / folder.name / "pooled" / "synthetic"
    _extract_all(folder.name, bfile, run_dir, ["train", "val"])
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "features", "validate", str(run_dir / "features")]), "validate ineligible"))
    assert env["status"] == "ok" and env["errors"] == []
    reports = env["data"] if isinstance(env["data"], list) else [env["data"]]
    assert {r["split"] for r in reports} == {"train", "val"}
    for r in reports:
        assert r["ok"] is True and r["eligibility"]["eligible"] is False, r["eligibility"]
        assert any(m.startswith("pos.") for m in r["eligibility"]["missing"]), r["eligibility"]
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "probe", "run", str(run_dir)]), "probe run ineligible"))
    assert env["status"] == "not_applicable" and env["data"][0]["status"] == "not_applicable"
    result_path = Path(env["data"][0]["path"])
    rj = json.loads(result_path.read_text())
    assert rj["status"] == "not_applicable" and rj["not_applicable"]["decided_by"] == "validate" and "metrics" not in rj
    assert (result_path.parent / "DONE").is_file()


def test_one_model_two_readings_two_configurations(integration_env, yavih_home, runs_dir, tmp_path, monkeypatch):
    """Integration on the REAL chain: one model name ``m1``
    read two ways (``a``, ``v``) and, for ``a``, under two bridge configurations (``D`` 16 vs 32: two
    ``describe_sha256``). Bridge decisions never enter paths, keys or comparability: every result shares one
    ``eval_hash`` (= sha256(canon({protocol_hash, probe_hash}))), run and probe paths are
    ``runs/<b>/m1/<reading>/probe/<eval_hash12>/seed0``, the results store is ``<results_root>/m1/`` keyed
    ``<b>:a`` / ``<b>:v``, and the report shows one column and one row per reading with ``W_BRIDGE_CHANGED`` for
    ``a`` (newest configuration shown, the older under ``--detail``). Re-extracting into the SAME run directory
    under the other configuration moves the old shards to ``_stale/`` (keyed on ``describe_sha256``) and the probe
    re-runs, moving its old output to ``probe/_stale/``; the report never shows that output as a row but lists its
    configuration as an older one (``W_BRIDGE_CHANGED``, ``moved_aside``), on one runs root as on two."""
    from yavih.benchmark import Benchmark
    from yavih.hashing import eval_hash
    from yavih.probes.config import ProbeConfig
    from yavih.readouts import probe

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "clf", n_train=32, n_eval=16, media="tiny")
    name = folder.name
    bench = Benchmark.load(name)
    splits = list(bench.roles()["train"]) + list(bench.roles()["eval"])
    bridges = {  # bridge FILE labels differ on purpose: `name` is recorded only, never a key or a path
        "a1": T.write_bridge_file(tmp_path / "m1_audio.py", "oracle", D=16, model="m1", reading="a", name="m1-audio-narrow"),
        "v": T.write_bridge_file(tmp_path / "m1_visual.py", "oracle", D=16, model="m1", reading="v", name="m1-visual"),
        "a2": T.write_bridge_file(tmp_path / "m1_audio_wide.py", "oracle", D=32, model="m1", reading="a", name="m1-audio-wide"),
    }
    profiles = {}
    for k, bfile in bridges.items():
        out = tmp_path / f"{k}.profile.json"
        proc = _ok(_run(["-m", "yavih.cli", "bridge", "check", str(bfile), "--bench", name, "--n", "4", "--out", str(out)]), f"bridge check {k}")
        profiles[k] = json.loads(out.read_text())
        assert (profiles[k]["model"], profiles[k]["reading"]) == ("m1", "v" if k == "v" else "a")
        assert f"run directory: runs/{name}/m1/{profiles[k]['reading']}/" in proc.stdout, proc.stdout
    assert len({p["describe_sha256"] for p in profiles.values()}) == 3  # three configurations

    # a second runs root ("another site") holds the other configuration of m1/a; both share one results root
    runs_b = tmp_path / "runs_site_b"
    run_dirs = {"a1": runs_dir / name / "m1" / "a", "v": runs_dir / name / "m1" / "v", "a2": runs_b / name / "m1" / "a"}
    results_root = runs_dir.parent / "results"
    assert runs_b.parent / "results" == results_root
    cfg = T.fast(ProbeConfig.pinned("clf", bench))
    res = {}
    for k in ("a1", "v", "a2"):  # a2 is probed last: the newest configuration of m1/a
        _extract_all(name, bridges[k], run_dirs[k], splits)
        _ok(_run(["-m", "yavih.cli", "features", "validate", str(run_dirs[k] / "features")]), f"validate {k}")
        out = probe(run_dirs[k], seeds=(0,), cfg=cfg)
        assert len(out) == 1 and out[0].status == "ok", (k, out[0].status, out[0].reason)
        res[k] = out[0]
        time.sleep(1.1)  # distinct `created` seconds
    rj = {k: r.result for k, r in res.items()}
    eh = rj["a1"]["eval_hash"]
    assert eh == eval_hash(rj["a1"]["benchmark"]["protocol_hash"], rj["a1"]["probe"]["hash"])
    assert {r["eval_hash"] for r in rj.values()} == {eh}, "bridge decisions (D, reading, file label) must not enter eval_hash"
    assert {r["benchmark"]["protocol_hash"] for r in rj.values()} == {bench.protocol_hash}
    assert len({r["probe"]["hash"] for r in rj.values()}) == 1
    for k, r in res.items():
        assert r.path == run_dirs[k] / "probe" / eh[:12] / "seed0" / "result.json", (k, r.path)
        assert (rj[k]["model"], rj[k]["reading"]) == ("m1", profiles[k]["reading"])
        assert rj[k]["bridge"]["describe_sha256"] == profiles[k]["describe_sha256"]
        # nothing bridge-side names a directory: neither the file label nor the describe digest
        rel = r.path.relative_to(run_dirs[k].parents[2]).parts
        assert not any(rj[k]["bridge"]["name"] in p or rj[k]["bridge"]["describe_sha256"][:12] in p for p in rel), rel

    # results store: <results_root>/m1/results_<ts>.json, one per probe invocation, keyed <b>:<reading>
    assert sorted(p.name for p in results_root.iterdir()) == ["m1"]
    files = sorted((results_root / "m1").glob("results_*.json"))
    assert len(files) == 3 and {res[k].results_file for k in res} == set(files)
    docs = {k: json.loads(res[k].results_file.read_text()) for k in res}
    for k, doc in docs.items():
        key = f"{name}:{profiles[k]['reading']}"
        assert doc["results_version"] == 2 and set(doc["results"]) == {key} and doc["config"]["model"] == "m1"
        c = doc["configs"][key]
        assert (c["benchmark"], c["reading"], c["eval_hash"]) == (name, profiles[k]["reading"], eh)
        assert c["bridge"]["describe_sha256"] == profiles[k]["describe_sha256"]
    assert docs["a1"]["configs"][f"{name}:a"]["bridge"]["describe_sha256"] != docs["a2"]["configs"][f"{name}:a"]["bridge"]["describe_sha256"]

    # report over both runs roots: ONE column (one eval_hash), two rows (m1 a, m1 v), W_BRIDGE_CHANGED for m1 a only
    globs = [str(runs_dir / name / "*"), str(runs_b / name / "*")]
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "report", "build", *globs, "--fmt", "json", "--detail"]), "report json"))
    rep = env["data"]
    cols = rep["columns"]
    cols = list(cols.values()) if isinstance(cols, dict) else cols
    assert len(cols) == 1 and cols[0]["eval_hash"] == eh and cols[0]["benchmark"] == name, cols
    col_id = cols[0]["id"]
    assert eh[:12] in col_id
    changed = [w for w in rep["warnings"] if w["code"] == "W_BRIDGE_CHANGED"]
    assert len(changed) == 1 and "m1 a on" in changed[0]["message"] and changed[0]["column"] == col_id, rep["warnings"]
    assert "D" in changed[0]["message"]  # the differing describe() key is named
    # the envelope carries the report's warnings as Findings too, where = the column id
    assert [(w["code"], w["where"]) for w in env["warnings"] if w["code"] == "W_BRIDGE_CHANGED"] == [("W_BRIDGE_CHANGED", col_id)]
    det = rep["detail"][col_id]
    assert set(det) == {"m1:a", "m1:v"}, det.keys()
    assert det["m1:a"]["describe_sha256"] == profiles["a2"]["describe_sha256"]  # newest shown
    older = det["m1:a"]["older_configs"]
    assert len(older) == 1 and older[0]["describe_sha256"] == profiles["a1"]["describe_sha256"], older
    assert det["m1:v"]["older_configs"] == []
    md = _ok(_run(["-m", "yavih.cli", "report", "build", *globs]), "report md").stdout
    # view 1 (per model): one block "## m1", the eval_hash column on every line of both readings
    view1 = [ln for ln in md.splitlines() if ln.startswith(f"| {name}")]
    assert {ln.split("|")[2].strip() for ln in view1} == {"a", "v"} and all(f"`{eh[:12]}`" in ln for ln in view1), view1
    assert md.count("\n## m1\n") == 1
    # view 2 (comparability columns): one column headed by the eval_hash, one row per (model, reading)
    view2 = [ln for ln in md.splitlines() if ln.startswith("| m1 |")]
    assert [ln.split("|")[2].strip() for ln in view2] == ["a", "v"], view2
    assert md.count(f"eval_hash `{eh[:12]}`") == 1
    assert "W_BRIDGE_CHANGED" in md and not any("~~" in ln for ln in view1 + view2)  # a reconfiguration never greys a row

    # --merge-results: one merged file in m1/, newest entry per (benchmark, reading, eval_hash)
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "report", "build", *globs, "--merge-results", "--results", str(results_root)]), "merge"))
    merged = [Path(p) for p in env["paths"]["merged"]]
    assert len(merged) == 1 and merged[0].parent == results_root / "m1"
    mdoc = json.loads(merged[0].read_text())
    assert set(mdoc["results"]) == {f"{name}:a", f"{name}:v"}, mdoc["results"].keys()
    assert mdoc["configs"][f"{name}:a"]["bridge"]["describe_sha256"] == profiles["a2"]["describe_sha256"]

    # in-place reconfiguration: a2's bridge into a1's run directory → old shards to _stale/, probe re-runs
    old_sidecar = json.loads((run_dirs["a1"] / "features" / splits[0] / "shard-000000.json").read_text())
    assert old_sidecar["bridge"]["describe_sha256"] == profiles["a1"]["describe_sha256"]
    _extract_all(name, bridges["a2"], run_dirs["a1"], splits)
    for split in splits:
        stale = list((run_dirs["a1"] / "features" / split / "_stale").glob("*/shard-*.json"))
        assert len(stale) == N_SHARDS, (split, stale)
        assert {json.loads(p.read_text())["bridge"]["describe_sha256"] for p in stale} == {profiles["a1"]["describe_sha256"]}
        new = json.loads((run_dirs["a1"] / "features" / split / "shard-000000.json").read_text())
        assert new["bridge"]["describe_sha256"] == profiles["a2"]["describe_sha256"] and (new["model"], new["reading"]) == ("m1", "a")
    _ok(_run(["-m", "yavih.cli", "features", "validate", str(run_dirs["a1"] / "features")]), "validate after reconfiguration")
    again = probe(run_dirs["a1"], seeds=(0,), cfg=cfg)
    assert again[0].status == "ok" and again[0].path == res["a1"].path
    assert again[0].result["bridge"]["describe_sha256"] == profiles["a2"]["describe_sha256"]
    moved = list((run_dirs["a1"] / "probe" / "_stale").glob(f"*/{eh[:12]}/seed0/result.json"))
    assert len(moved) == 1 and json.loads(moved[0].read_text())["bridge"]["describe_sha256"] == profiles["a1"]["describe_sha256"]
    # the moved-aside result is never a row, but its configuration does not vanish silently:
    # on ONE runs root, m1/a now shows the a2 configuration and lists a1's moved-aside output as an older configuration
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "report", "build", str(runs_dir / name / "*"), "--fmt", "json", "--detail"]),
                    "report after reconfiguration, one runs root"))
    rep = env["data"]
    changed = [w for w in rep["warnings"] if w["code"] == "W_BRIDGE_CHANGED"]
    assert len(changed) == 1 and "m1 a on" in changed[0]["message"] and "probe/_stale/" in changed[0]["message"], rep["warnings"]
    assert rep["n_results"] == 2 and sorted((r["model"], r["reading"]) for r in rep["rows"]) == [("m1", "a"), ("m1", "v")]
    det = rep["detail"][col_id]["m1:a"]
    assert det["describe_sha256"] == profiles["a2"]["describe_sha256"]
    (older,) = det["older_configs"]
    assert older["describe_sha256"] == profiles["a1"]["describe_sha256"] and older["moved_aside"] is True, older
    assert det["older_feature_sets"] == []
    # over both roots: the a2 configuration on both sites is ONE configuration (two feature sets at most), a1 the older
    env = _json(_ok(_run(["-m", "yavih.cli", "--json", "report", "build", *globs, "--fmt", "json", "--detail"]), "report after reconfiguration"))
    changed = [w for w in env["data"]["warnings"] if w["code"] == "W_BRIDGE_CHANGED"]
    assert len(changed) == 1 and "2 bridge configurations" in changed[0]["message"], env["data"]["warnings"]
    older = env["data"]["detail"][col_id]["m1:a"]["older_configs"]
    assert [(o["describe_sha256"], o["moved_aside"]) for o in older] == [(profiles["a1"]["describe_sha256"], True)], older
