"""``plan exec``: dependencies (exit 3), DONE skip, deadline refusal (exit 4), launcher handling, env,
tee logs, signal-killed child → exit 4 preempted, last.json, --slots, in-process probe/report, stdlib-only
controller for extract units; exit codes through the CLI envelope."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from test_plan_helpers import (
    fake_unit,
    python_launcher,
    write_done_probe,
    write_fake_script,
)
from yavih import cli
from yavih.deadline import DeadlineError
from yavih.plan import exec as X
from yavih.plan import units as U
from yavih.plan.units import Unit, write_plan


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root, write_fake_script(root)


def _plan(root, *units: Unit) -> str:
    p = root / "plan.jsonl"
    write_plan(list(units), str(p))
    return str(p)


def _last(unit: Unit) -> dict:
    return json.loads(open(os.path.join(U.resolve_path(unit, unit.out), "logs", f"last.{unit.id}.json")).read())


# ----------------------------------------------------------------------------- pure helpers


def test_build_command_launcher_rules(workspace):
    root, script = workspace
    u = fake_unit(root, script)
    assert X.build_command(u) == u.cmd
    u.launcher = "/envs/vjepa/bin/python"
    assert X.build_command(u) == ["/envs/vjepa/bin/python", *u.cmd[1:]]
    u.launcher = "conda run -n vjepa python3.11"
    assert X.build_command(u) == ["conda", "run", "-n", "vjepa", "python3.11", *u.cmd[1:]]
    u.launcher = "apptainer exec --nv --bind /e img.sif"
    assert X.build_command(u) == ["apptainer", "exec", "--nv", "--bind", "/e", "img.sif", *u.cmd]
    u.launcher = "   "
    assert X.build_command(u) == u.cmd


def test_resolve_deadline_sources():
    now = time.time()
    assert X.resolve_deadline(None) is None
    assert X.resolve_deadline("") is None
    assert X.resolve_deadline(1234.5) == 1234.5  # a computed epoch is taken as is
    assert X.resolve_deadline("1700000000") == 1700000000.0
    assert abs(X.resolve_deadline("+3600") - (now + 3600)) < 5  # relative, not an epoch
    assert abs(X.resolve_deadline("2h") - (now + 7200)) < 5
    assert X.resolve_deadline("none") is None
    assert X.resolve_deadline("slurm", margin_s=300, env={"SLURM_JOB_END_TIME": "100000", "SLURM_JOB_ID": "1"}) == 100000 - 300
    for env in ({"SLURM_JOB_ID": "1"}, {"SLURM_JOB_ID": "1", "SLURM_JOB_END_TIME": "N/A"}):
        with pytest.raises(DeadlineError) as ei:  # inside a job without a usable end time: exit 1, never "now"
            X.resolve_deadline("slurm", env=env)
        assert ei.value.exit_code == 1 and ei.value.code == "E_DEADLINE_SPEC"
    assert X.resolve_deadline("slurm", env={}) is None  # outside SLURM → no deadline
    for bad in ("tomorrow", "3600", "+soon"):  # "3600" is an epoch in 1970: a duration typed without "+"
        with pytest.raises(DeadlineError):
            X.resolve_deadline(bad)


# ----------------------------------------------------------------------------- extract paths


def test_extract_unit_done_then_skipped(workspace, capsys):
    root, script = workspace
    u = fake_unit(root, script, mode="ok", launcher=python_launcher())
    plan = _plan(root, u)
    last = X.exec_unit(plan, index=0)
    assert last["status"] == "done" and last["exit_code"] == 0 and last["id"] == u.id
    assert os.path.exists(U.resolve_path(u, u.done))
    assert set(last) >= {"id", "status", "exit_code", "wall_s", "host", "gpu", "slurm_job_id", "inputs_hash", "started", "ended"}
    assert last["inputs_hash"] == u.inputs_hash and last["needs"] == u.needs
    assert _last(u) == last and json.load(open(os.path.join(root, u.out, "logs", "last.json")))["id"] == u.id
    assert last["log"] and os.path.exists(last["log"]) and "child start ok" in open(last["log"]).read()
    err = capsys.readouterr().err
    assert f"[yavih] unit {u.id} extract start" in err and f"[yavih] unit {u.id} extract done wall_s=" in err
    # second run: DONE check → skipped, exit 0, child not spawned
    again = X.exec_unit(plan, unit_id=u.id)
    assert again["status"] == "skipped" and again["exit_code"] == 0
    assert "already done" in again["message"]


def test_env_and_launcher_substitution_reach_the_child(workspace):
    root, script = workspace
    u = fake_unit(root, script, mode="env", launcher=python_launcher(), env={"HF_HUB_OFFLINE": "1"})
    plan = _plan(root, u)
    dl = time.time() + 3600
    last = X.exec_unit(plan, index=0, deadline=dl)
    assert last["status"] == "done"
    log = open(last["log"]).read()
    assert f"UNIT={u.id}" in log and f"DL={float(dl)!r}" in log and "HF=1" in log


def test_dependency_missing_exits_3_nothing_run(workspace):
    root, script = workspace
    dep = fake_unit(root, script, mode="ok", shard=(0, 2), launcher=python_launcher())
    u = fake_unit(root, script, mode="ok", shard=(1, 2), depends_on=[dep.id], launcher=python_launcher())
    plan = _plan(root, dep, u)
    last = X.exec_unit(plan, unit_id=u.id)
    assert last["status"] == "not_ready" and last["exit_code"] == 3
    assert not os.path.exists(U.resolve_path(u, u.done)) and last["log"] is None
    assert dep.id in last["message"]
    # run the dependency, then the dependent succeeds
    assert X.exec_unit(plan, unit_id=dep.id)["exit_code"] == 0
    assert X.exec_unit(plan, unit_id=u.id)["status"] == "done"
    # a dependency id that is not in the plan is also not ready
    ghost = fake_unit(root, script, mode="ok", shard=(0, 3), depends_on=["000000000000"], launcher=python_launcher())
    assert X.exec_unit(_plan(root, ghost), index=0)["exit_code"] == 3


def test_deadline_headroom_refusal_exit_4(workspace):
    root, script = workspace
    u = fake_unit(root, script, mode="ok", launcher=python_launcher(), minutes=100)
    plan = _plan(root, u)
    last = X.exec_unit(plan, index=0, deadline=time.time() + 60)  # need max(600, 0.25*100*60=1500) s
    assert last["status"] == "deadline" and last["exit_code"] == 4 and last["log"] is None
    assert not os.path.exists(U.resolve_path(u, u.done))
    ok = X.exec_unit(plan, index=0, deadline=time.time() + 2000)
    assert ok["status"] == "done"


def test_child_exit_codes_map_through(workspace):
    root, script = workspace
    # exit 1 (usage / configuration) is `misconfigured`: no resubmission heals it
    cases = {"exit1": ("misconfigured", 1), "exit2": ("failed", 2), "exit3": ("not_ready", 3), "exit4": ("deadline", 4), "exit5": ("failed", 5)}
    for i, (mode, (status, code)) in enumerate(cases.items()):
        u = fake_unit(root, script, mode=mode, shard=(i, 10), launcher=python_launcher())
        last = X.exec_unit(_plan(root, u), index=0)
        assert (last["status"], last["exit_code"]) == (status, code), mode
        assert not os.path.exists(U.resolve_path(u, u.done))
    # exit 0 without a DONE marker is a failure, never silently "done"
    u = fake_unit(root, script, mode="noop", shard=(7, 10), launcher=python_launcher())
    last = X.exec_unit(_plan(root, u), index=0)
    assert last["status"] == "failed" and last["exit_code"] == 5 and "missing" in last["message"]


def test_signal_killed_child_is_preempted_exit_4(workspace):
    root, script = workspace
    u = fake_unit(root, script, mode="sigterm", launcher=python_launcher())
    last = X.exec_unit(_plan(root, u), index=0)
    assert last["status"] == "preempted" and last["exit_code"] == 4
    assert "signal" in last["message"]


def test_bad_launcher_binary_is_a_failure_exit_1(workspace):
    root, script = workspace
    u = fake_unit(root, script, mode="ok", launcher="/no/such/python")
    last = X.exec_unit(_plan(root, u), index=0)
    assert last["status"] == "misconfigured" and last["exit_code"] == 1


def test_exec_selection_errors(workspace):
    root, script = workspace
    u = fake_unit(root, script, launcher=python_launcher())
    plan = _plan(root, u)
    with pytest.raises(X.ExecError):
        X.exec_unit(plan)
    with pytest.raises(X.ExecError):
        X.exec_unit(plan, index=0, unit_id=u.id)
    with pytest.raises(X.ExecError):
        X.exec_unit(plan, index=5)
    with pytest.raises(X.ExecError):
        X.exec_unit(plan, unit_id="nope")
    with pytest.raises(X.ExecError):
        X.exec_unit(plan, unit_id=u.id, slots=2)
    with pytest.raises(U.PlanError):
        X.exec_unit(str(root / "none.jsonl"), index=0)


def test_slots_run_concurrently_with_gpu_and_cpu_packing(workspace):
    root, script = workspace
    us = [fake_unit(root, script, mode="env", shard=(i, 4), launcher=python_launcher(), extra_argv=["--decode-workers", "3"]) for i in range(4)]
    plan = _plan(root, *us)
    res = X.exec_unit(plan, index=1, slots=2)  # units 2 and 3
    assert res["status"] == "done" and res["exit_code"] == 0 and len(res["slots"]) == 2
    assert [r["id"] for r in res["slots"]] == [us[2].id, us[3].id]
    for g, r in enumerate(res["slots"]):
        log = open(r["log"]).read()
        assert f"CVD={g}" in log and r["gpu"] == str(g)
        assert r["needs"]["cpus"] == us[0].needs["cpus"] // 2  # 4 // 2
        assert "ARGV=--decode-workers 1" in log  # cpus // K − 1 decode workers
    with pytest.raises(X.ExecError):
        X.exec_unit(plan, index=2, slots=2)
    # units 0/1 untouched
    assert U.check_done(us[0]) is None and U.check_done(us[2]) == "done"


def test_slots_exit_code_is_the_worst(workspace):
    root, script = workspace
    a = fake_unit(root, script, mode="ok", shard=(0, 2), launcher=python_launcher())
    b = fake_unit(root, script, mode="exit3", shard=(1, 2), launcher=python_launcher())
    res = X.exec_unit(_plan(root, a, b), index=0, slots=2)
    assert res["exit_code"] == 3 and res["status"] == "not_ready"


# ----------------------------------------------------------------------------- stdlib-only controller + signal forwarding (subprocess)


def test_extract_exec_loads_no_torch_or_probe_modules(workspace):
    root, script = workspace
    u = fake_unit(root, script, mode="ok", launcher=python_launcher())
    plan = _plan(root, u)
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "from yavih.plan.exec import exec_unit\n"
        f"last = exec_unit({plan!r}, index=0)\n"
        "assert last['status'] == 'done', last\n"
        "bad = [m for m in sys.modules if m.startswith(('yavih.readouts', 'yavih.probes', 'yavih.report', 'sklearn'))]\n"
        "assert not bad, bad\n"
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(root))
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


def test_sigterm_to_exec_is_forwarded_and_reported_preempted(workspace):
    root, script = workspace
    sentinel = root / "child.pid"
    u = fake_unit(root, script, mode="sleep", launcher=python_launcher(), sentinel=sentinel)
    plan = _plan(root, u)
    proc = subprocess.Popen(
        [sys.executable, "-m", "yavih.cli", "--json", "exec", plan, "--index", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(root),
    )
    deadline = time.time() + 30
    while not sentinel.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert sentinel.exists(), proc.stderr.read() if proc.poll() is not None else "child never started"
    time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=30)
    env = json.loads(out.strip())
    assert proc.returncode == 4, (out, err)
    assert env["status"] == "error" and env["data"]["status"] == "preempted" and env["data"]["exit_code"] == 4
    assert "child got signal" in err
    assert _last(u)["status"] == "preempted"
    child_pid = int(sentinel.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


# ----------------------------------------------------------------------------- in-process probe / report


def test_probe_unit_runs_in_process_via_readouts_probe(workspace, monkeypatch):
    import yavih.readouts as R

    root, script = workspace
    u = fake_unit(root, script, kind="probe", seed=0)
    calls = []

    def fake_probe(run_dir, *, seeds=(0,), **kw):
        import inspect

        from yavih.readouts import base

        inspect.signature(base.probe).bind(run_dir, seeds=seeds, **kw)  # TypeError on a keyword the real probe lacks
        calls.append((run_dir, seeds, kw))
        write_done_probe(u)
        return []

    monkeypatch.setattr(R, "probe", fake_probe)
    dl = time.time() + 7200
    last = X.exec_unit(_plan(root, u), index=0, deadline=dl)
    assert last["status"] == "done" and last["exit_code"] == 0
    assert calls and calls[0][0] == U.resolve_path(u, u.out) and calls[0][1] == (0,)
    # the probe prologue is pinned: validate every split first; deadline passed through; no scratch
    assert calls[0][2] == {"deadline": pytest.approx(dl, abs=5), "scratch": None, "validate_first": True}


def test_probe_unit_failure_and_refusal_codes(workspace, monkeypatch):
    import yavih.readouts as R

    root, script = workspace
    u = fake_unit(root, script, kind="probe", seed=1)
    plan = _plan(root, u)

    def refuse(run_dir, **kw):
        raise R.ProbeRefused("E_COVERAGE", "eval coverage 0.5 < 0.95")

    monkeypatch.setattr(R, "probe", refuse)
    last = X.exec_unit(plan, index=0)
    assert last["status"] == "failed" and last["exit_code"] == 2 and "E_COVERAGE" in last["message"]

    def deadline(run_dir, **kw):
        raise TimeoutError("deadline between epochs")

    monkeypatch.setattr(R, "probe", deadline)
    last = X.exec_unit(plan, index=0)
    assert last["status"] == "deadline" and last["exit_code"] == 4

    def boom(run_dir, **kw):
        raise RuntimeError("cuda went away")

    monkeypatch.setattr(R, "probe", boom)
    last = X.exec_unit(plan, index=0)
    assert last["status"] == "failed" and last["exit_code"] == 5

    monkeypatch.setattr(R, "probe", lambda run_dir, **kw: [])  # returns without DONE
    last = X.exec_unit(plan, index=0)
    assert last["status"] == "failed" and last["exit_code"] == 5 and "without writing" in last["message"]


def test_probe_unit_scratch_preflight_exit_3(workspace, monkeypatch):
    import yavih.readouts as R

    root, script = workspace
    u = fake_unit(root, script, kind="probe", seed=0)
    u.needs["scratch_gb"] = 10**6  # a petabyte: no filesystem has it
    monkeypatch.delenv("YAVIH_SCRATCH", raising=False)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    # no scratch configured: like `probe run`, the probe reads the features in place (memmap, W_SIZE) instead of an
    # exit-3 loop
    seen: list = []

    def fake_probe(run_dir, **kw):
        seen.append(kw["scratch"])
        write_done_probe(u)
        return []

    monkeypatch.setattr(R, "probe", fake_probe)
    last = X.exec_unit(_plan(root, u), index=0)
    assert last["status"] == "done" and seen == [None]
    assert "read in place" in open(last["log"]).read()
    os.remove(U.resolve_path(u, u.done))
    monkeypatch.setattr(R, "probe", lambda run_dir, **kw: pytest.fail("probe must not start"))
    scratch = root / "scratch"
    scratch.mkdir()
    for how in ("env", "config"):  # YAVIH_SCRATCH, or the config's [scratch] root
        if how == "env":
            monkeypatch.setenv("YAVIH_SCRATCH", str(scratch))
        else:
            monkeypatch.delenv("YAVIH_SCRATCH", raising=False)
            cfg = root / "config.toml"
            cfg.write_text(f'[scratch]\nroot = "{scratch}"\n', encoding="utf-8")
            monkeypatch.setenv("YAVIH_CONFIG", str(cfg))
        last = X.exec_unit(_plan(root, u), index=0)
        assert last["status"] == "not_ready" and last["exit_code"] == 3 and "free" in last["message"] and str(scratch) in last["message"], how


def test_scratch_sweep_removes_dead_and_old(tmp_path):
    import socket

    host = socket.gethostname()
    root = tmp_path / "scratch"
    root.mkdir()
    dead = root / f"{'1' * 12}.{host}.999999"
    alive = root / f"{'2' * 12}.{host}.{os.getpid()}"
    foreign_new = root / f"{'3' * 12}.otherhost.1"
    foreign_old = root / f"{'4' * 12}.otherhost.2"
    unrelated = root / "keep_me"
    for d in (dead, alive, foreign_new, foreign_old, unrelated):
        d.mkdir()
    old = time.time() - 40 * 3600
    os.utime(foreign_old, (old, old))
    removed = X._sweep_scratch(str(root))
    assert set(removed) == {str(dead), str(foreign_old)}
    assert alive.exists() and foreign_new.exists() and unrelated.exists()


def test_plan_time_not_applicable_probe_uses_readouts_hook(workspace, monkeypatch):
    """A planned ``not_applicable`` probe unit never runs the ordinary probe; it goes
    through ``yavih.readouts.probe_not_applicable(run_dir, *, unit, profile, seed)``."""
    import yavih.readouts as R

    root, script = workspace
    u = fake_unit(root, script, kind="probe", seed=0, expected_status="not_applicable")
    u.not_applicable = {"missing": ["pos.t"], "reason": "pooled", "decided_by": "plan"}
    plan = _plan(root, u)
    monkeypatch.setattr(R, "probe", lambda *a, **k: pytest.fail("ordinary probe must not run for a planned not_applicable"))
    assert callable(R.probe_not_applicable)

    seen = {}

    def fake_na(run_dir, *, unit, profile, seed, **kw):
        import inspect

        from yavih.readouts import base

        inspect.signature(base.probe_not_applicable).bind(run_dir, unit=unit, profile=profile, seed=seed, **kw)
        seen.update(unit=unit, seed=seed)
        write_done_probe(u, status="not_applicable")

    monkeypatch.setattr(R, "probe_not_applicable", fake_na, raising=False)
    last = X.exec_unit(plan, index=0)
    assert last["status"] == "not_applicable" and last["exit_code"] == 0
    assert seen["unit"]["id"] == u.id and seen["seed"] == 0
    # the record the hook builds result.json/DONE from carries the names and the keys
    assert {k: seen["unit"][k] for k in ("model", "reading", "eval_hash", "describe_sha256")} == {
        "model": u.model, "reading": u.reading, "eval_hash": u.eval_hash, "describe_sha256": u.describe_sha256}


def test_probe_unit_reruns_over_a_done_of_another_configuration(workspace, monkeypatch):
    """Via exec step 3: a probe DONE written under another ``describe_sha256`` (the bridge was
    reconfigured under the same model name) is not ``skipped``; the probe runs (and moves the old seed dir to
    ``_stale/``, readouts' job)."""
    import yavih.readouts as R

    root, script = workspace
    u = fake_unit(root, script, kind="probe", seed=0)
    write_done_probe(u, describe_sha256="7" * 64)
    calls = []

    def fake_probe(run_dir, **kw):
        calls.append(run_dir)
        write_done_probe(u)
        return []

    monkeypatch.setattr(R, "probe", fake_probe)
    last = X.exec_unit(_plan(root, u), index=0)
    assert last["status"] == "done" and len(calls) == 1
    assert X.exec_unit(_plan(root, u), index=0)["status"] == "skipped" and len(calls) == 1


def test_report_unit_runs_in_process(workspace, monkeypatch):
    import yavih.report as REP

    root, script = workspace
    u = fake_unit(root, script, kind="report")
    seen = {}

    def fake_report(globs, *, fmt="md", **kw):
        seen["globs"] = list(globs)
        return REP.Report(columns=[{"c": 1}], rows=[{"r": 1}], text="| a |\n", fmt=fmt)

    monkeypatch.setattr(REP, "report", fake_report)
    last = X.exec_unit(_plan(root, u), index=0)
    assert last["status"] == "done" and last["exit_code"] == 0
    out = U.resolve_path(u, u.out)
    assert seen["globs"] == [os.path.join(os.path.dirname(out), "*")]
    assert open(os.path.join(out, "report.md")).read() == "| a |\n"
    done = json.load(open(U.resolve_path(u, u.done)))
    assert done["n_rows"] == 1 and "created" in done
    assert X.exec_unit(_plan(root, u), index=0)["status"] == "skipped"


# ----------------------------------------------------------------------------- CLI envelope


def test_cli_exec_envelope_and_exit_codes(workspace, capsys):
    root, script = workspace
    ok = fake_unit(root, script, mode="ok", shard=(0, 3), launcher=python_launcher())
    dep = fake_unit(root, script, mode="ok", shard=(1, 3), depends_on=[ok.id], launcher=python_launcher())
    bad = fake_unit(root, script, mode="exit2", shard=(2, 3), launcher=python_launcher())
    plan = _plan(root, ok, dep, bad)
    rc = cli.main(["--json", "exec", plan, "--id", dep.id])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 3 and env["status"] == "error" and env["data"]["status"] == "not_ready"
    rc = cli.main(["--json", "exec", plan, "--index", "0"])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 0 and env["status"] == "ok" and env["data"]["status"] == "done"
    rc = cli.main(["--json", "exec", plan, "--index", "0"])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 0 and env["status"] == "skipped"
    rc = cli.main(["--json", "exec", plan, "--index", "2"])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 2 and env["data"]["exit_code"] == 2
    rc = cli.main(["--json", "exec", plan, "--index", "9"])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 1 and env["errors"][0]["code"] == "E_EXEC"
    rc = cli.main(["--json", "exec", plan, "--index", "1", "--deadline", "slurm"])  # outside SLURM: no deadline
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 0 and env["data"]["status"] == "done"
