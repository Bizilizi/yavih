"""The documented plan/exec/SLURM driver loop end to end:

- the documented driver loop (pending extract → pending probe → pending report) runs end to end with
  ``exec --deps plan.jsonl``; without it a probe from a pending file is ``not_ready`` with a hint;
- the SLURM templates replace (or forward TERM from) the batch shell, so ``--signal=B:TERM`` reaches
  ``yavih exec`` — checked on the text of slurm.md and by signalling a real bash running the documented lines;
- ``--deadline-margin`` defaults above the templates' signal lead;
- a ``running`` record while a unit runs, ``plan status`` running / stale running, text summary;
- failed in-process units leave a log with the traceback;
- probe minutes count the shuffled control's second fit and the re-derived dense ``probe_minutes_ref``.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from test_plan_helpers import (
    fake_unit,
    make_info,
    python_launcher,
    standard_matrix,
    write_done_extract,
    write_done_probe,
    write_fake_script,
    write_last,
)
from yavih import cli
from yavih.plan import estimate as E
from yavih.plan import exec as X
from yavih.plan import units as U
from yavih.plan.status import RUNNING_GRACE_S, format_status, plan_status, running_alive
from yavih.plan.units import plan, read_plan, write_plan

SLURM_MD = Path(cli.__file__).parent / "skill" / "yavih" / "references" / "slurm.md"


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root, write_fake_script(root)


def _plan(root: Path, *units, name: str = "plan.jsonl") -> str:
    p = root / name
    write_plan(list(units), str(p))
    return str(p)


def _last(unit) -> dict:
    return json.loads(open(os.path.join(U.resolve_path(unit, unit.out), "logs", f"last.{unit.id}.json")).read())


# ============================================================================= driver loop / --deps


def test_documented_driver_loop_runs_probe_and_report_units_with_deps(tmp_path, monkeypatch, capsys):
    """slurm.md's loop: full plan once, then per kind a ``--pending --kind K`` file executed by index. The probe
    file holds no extract units; ``--deps plan.jsonl`` resolves them."""
    import yavih.readouts as R
    import yavih.report as REP

    mp = standard_matrix(tmp_path, monkeypatch, seeds=[0, 1], shard_minutes=1, clips_per_s=0.1)
    full = tmp_path / "plan.jsonl"
    write_plan(plan(str(mp)), str(full))
    by_seed: dict[int, U.Unit] = {}

    def fake_probe(run_dir, *, seeds=(0,), **kw):
        write_done_probe(by_seed[seeds[0]])
        return []

    monkeypatch.setattr(R, "probe", fake_probe)
    monkeypatch.setattr(REP, "report", lambda globs, *, fmt="md", **kw: REP.Report(columns=[], rows=[], text="| t |\n", fmt=fmt))

    rounds: dict[str, int] = {}
    for kind in ("extract", "probe", "report"):
        while True:
            pending = plan(str(mp), pending=True, kind=kind)
            if not pending:
                break
            rounds[kind] = rounds.get(kind, 0) + 1
            assert rounds[kind] <= 2, f"{kind} pending never drains"
            pf = str(tmp_path / f"pending.{kind}.jsonl")
            write_plan(pending, pf)
            if kind == "extract":  # the array would run `python -m yavih.extract`; its DONE marker is what matters
                for u in pending:
                    write_done_extract(u)
                continue
            if kind == "probe":
                by_seed.update({u.seed: u for u in pending})
                # without --deps a probe unit is not_ready, with a hint
                miss = X.exec_unit(pf, index=0)
                assert miss["status"] == "not_ready" and miss["exit_code"] == 3
                assert "--deps plan.jsonl" in miss["message"]
            for i in range(len(pending)):
                rc = cli.main(["--json", "exec", pf, "--deps", str(full), "--index", str(i)])
                env = json.loads(capsys.readouterr().out.strip())
                assert rc == 0 and env["data"]["status"] == "done", env
    assert rounds == {"extract": 1, "probe": 1, "report": 1}
    st = plan_status(str(full))
    assert st["done"] == st["total"] and st["pending"] == 0 and st["failed"] == 0


def test_deps_is_lookup_only_selection_stays_in_the_plan_file(workspace):
    root, script = workspace
    dep = fake_unit(root, script, mode="ok", shard=(0, 2), launcher=python_launcher())
    u = fake_unit(root, script, mode="ok", shard=(1, 2), depends_on=[dep.id], launcher=python_launcher())
    full = _plan(root, dep, u, name="plan.jsonl")
    pend = _plan(root, u, name="pending.jsonl")
    with pytest.raises(X.ExecError):  # --id selects from the pending file only
        X.exec_unit(pend, unit_id=dep.id, deps=full)
    assert X.exec_unit(pend, index=0, deps=full)["status"] == "not_ready"  # dep found, not done
    assert X.exec_unit(full, unit_id=dep.id)["status"] == "done"
    assert X.exec_unit(pend, index=0, deps=full)["status"] == "done"
    with pytest.raises(U.PlanError):  # a missing --deps file is a usage error, not a silent fallback
        X.exec_unit(pend, index=0, deps=str(root / "nope.jsonl"))


def test_deps_also_resolves_under_slots(workspace):
    root, script = workspace
    dep = fake_unit(root, script, mode="ok", shard=(0, 3), launcher=python_launcher())
    a = fake_unit(root, script, mode="ok", shard=(1, 3), depends_on=[dep.id], launcher=python_launcher())
    b = fake_unit(root, script, mode="ok", shard=(2, 3), depends_on=[dep.id], launcher=python_launcher())
    full = _plan(root, dep, a, b)
    pend = _plan(root, a, b, name="pending.jsonl")
    X.exec_unit(full, unit_id=dep.id)
    res = X.exec_unit(pend, index=0, slots=2, deps=full)
    assert res["exit_code"] == 0 and [r["status"] for r in res["slots"]] == ["done", "done"]


# ============================================================================= SIGTERM forwarding


def test_default_deadline_margin_exceeds_the_documented_signal_lead():
    leads = [int(m) for m in re.findall(r"--signal=B:TERM@(\d+)", SLURM_MD.read_text(encoding="utf-8"))]
    assert leads, "slurm.md documents no --signal lead?"
    assert X.DEFAULT_DEADLINE_MARGIN_S > max(leads)
    env = {"SLURM_JOB_END_TIME": "100000", "SLURM_JOB_ID": "1"}
    assert X.resolve_deadline("slurm", env=env) == 100000 - X.DEFAULT_DEADLINE_MARGIN_S
    ns = cli.build_parser().parse_args(["exec", "p.jsonl", "--index", "0"])
    assert ns.deadline_margin == X.DEFAULT_DEADLINE_MARGIN_S and ns.deps is None
    import inspect

    assert inspect.signature(X.exec_unit).parameters["deadline_margin"].default == X.DEFAULT_DEADLINE_MARGIN_S


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", text, flags=re.S)


def test_every_slurm_template_replaces_or_forwards_the_batch_shell():
    """Each documented ``yavih exec`` invocation inside a batch script is either ``exec …`` (the shell is replaced)
    or backgrounded with a TERM trap that forwards to it; each passes ``--deps``."""
    text = SLURM_MD.read_text(encoding="utf-8")
    calls = 0
    for block in _bash_blocks(text):
        lines = block.splitlines()
        for i, line in enumerate(lines):
            code = line.split("#", 1)[0].strip() if not line.lstrip().startswith("#") else ""
            if not re.search(r"\byavih exec\b", code):
                continue
            calls += 1
            assert "--deps" in code, f"slurm.md: `{code}` lacks --deps"
            if code.startswith("exec "):
                continue
            assert code.endswith("&"), f"slurm.md: `{code}` neither replaces the shell (exec …) nor runs in the background"
            rest = "\n".join(lines[i + 1:])
            assert re.search(r"trap '.*kill -TERM \"\$pid\".*' TERM", rest), "backgrounded yavih exec without a TERM forward"
            assert re.search(r"wait \"\$pid\"", rest) and 'exit "$rc"' in rest
    assert calls >= 3, "templates A, B and C all start yavih exec"


def test_slurm_templates_are_generic():
    """Three templates (A, B, C) with site values left as placeholders."""
    text = SLURM_MD.read_text(encoding="utf-8")
    assert re.findall(r"^## Template ([A-Z])\b", text, flags=re.M) == ["A", "B", "C"]
    for placeholder in ("<partition>", "<account>"):
        assert placeholder in text, placeholder


def _template_c_forwarding() -> str:
    """The lines of Template C from the backgrounded ``yavih exec … &`` through ``exit "$rc"``, with the command
    replaced by ``"$@" &`` (the test passes the real ``yavih exec`` argv)."""
    text = SLURM_MD.read_text(encoding="utf-8")
    sec = text.split("## Template C", 1)[1]
    block = _bash_blocks(sec)[0].splitlines()
    start = next(i for i, ln in enumerate(block) if "yavih exec" in ln and ln.rstrip().endswith("&"))
    end = next(i for i, ln in enumerate(block) if ln.strip() == 'exit "$rc"')
    return "\n".join(['"$@" &', *block[start + 1:end + 1]]) + "\n"


@pytest.mark.parametrize("style", ["exec", "forward"])
def test_sigterm_to_the_batch_shell_reaches_exec_and_is_recorded(workspace, style):
    """``scancel --signal=TERM --batch`` / ``--signal=B:TERM``: TERM goes to the batch shell only. With the documented
    lines the extract child is told, exec records ``preempted`` and the shell exits 4."""
    root, script = workspace
    sentinel = root / "child.pid"
    u = fake_unit(root, script, mode="sleep", launcher=python_launcher(), sentinel=sentinel)
    planf = _plan(root, u)
    argv = [sys.executable, "-m", "yavih.cli", "exec", planf, "--deps", planf, "--index", "0"]
    if style == "exec":
        body = 'set -euo pipefail\nexec "$@"\n'
    else:
        body = "set -euo pipefail\n" + _template_c_forwarding()
    sh = root / "job.sh"
    sh.write_text(body, encoding="utf-8")
    proc = subprocess.Popen(["bash", str(sh), *argv], cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 30
    while not sentinel.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert sentinel.exists(), proc.communicate(timeout=30)
    time.sleep(0.3)
    os.kill(proc.pid, signal.SIGTERM)  # the batch shell's pid (with `exec` it is yavih itself)
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 4, (style, out, err)
    assert "child got signal" in err
    assert _last(u)["status"] == "preempted"


# ============================================================================= running records and status


def test_failed_in_process_unit_leaves_its_traceback(workspace, monkeypatch):
    import yavih.readouts as R

    root, script = workspace
    u = fake_unit(root, script, kind="probe", seed=0)
    planf = _plan(root, u)

    def deep_failure():
        raise RuntimeError("cuda went away")

    def boom(run_dir, **kw):
        deep_failure()

    monkeypatch.setattr(R, "probe", boom)
    last = X.exec_unit(planf, index=0)
    assert last["status"] == "failed" and last["exit_code"] == 5
    assert last["log"] and os.path.exists(last["log"])
    log = open(last["log"], encoding="utf-8").read()
    assert "Traceback (most recent call last)" in log and "deep_failure" in log and "RuntimeError: cuda went away" in log
    assert last["message"].startswith("RuntimeError: cuda went away") and last["log"] in last["message"]
    # a successful in-process unit still names its log (header only)
    monkeypatch.setattr(R, "probe", lambda run_dir, **kw: write_done_probe(u))
    ok = X.exec_unit(planf, index=0)
    assert ok["status"] == "done" and ok["log"] and "in-process" in open(ok["log"], encoding="utf-8").read()


def test_running_record_while_a_unit_runs_then_final(workspace):
    root, script = workspace
    sentinel = root / "child.pid"
    u = fake_unit(root, script, mode="sleep", launcher=python_launcher(), sentinel=sentinel)
    planf = _plan(root, u)
    proc = subprocess.Popen([sys.executable, "-m", "yavih.cli", "exec", planf, "--index", "0"], cwd=str(root),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 30
        while not sentinel.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert sentinel.exists()
        rec = _last(u)
        assert rec["status"] == "running" and rec["pid"] == proc.pid and rec["host"] == socket.gethostname()
        assert rec["exit_code"] is None and rec["log"] and isinstance(rec["started_epoch"], float)
        # the shared last.json is never a provisional record
        assert not os.path.exists(os.path.join(U.resolve_path(u, u.out), "logs", "last.json"))
        st = plan_status(planf)
        assert st["running"] == 1 and st["pending"] == 0 and st["units"][0]["host"] == socket.gethostname()
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=60)
    assert _last(u)["status"] == "preempted" and plan_status(planf)["preempted"] == 1


def test_stale_running_records_count_as_pending(workspace):
    root, script = workspace
    u = fake_unit(root, script, mode="ok", launcher=python_launcher(), minutes=10)
    planf = _plan(root, u)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    now = time.time()
    write_last(u, status="running", exit_code=None, pid=dead.pid, host=socket.gethostname(), started_epoch=now)
    st = plan_status(planf)
    assert (st["running"], st["pending"]) == (0, 1) and st["stale_running"] == [u.id]
    assert "stale running record" in st["units"][0]["message"]
    # another host: believed within 2 × minutes + grace, and not past its deadline + grace
    rec = {"status": "running", "host": "elsewhere", "pid": 1, "started_epoch": now - 60, "needs": {"minutes": 10}, "deadline": None}
    assert running_alive(rec, now=now)
    assert not running_alive(dict(rec, started_epoch=now - 2 * 600 - RUNNING_GRACE_S - 1), now=now)
    assert not running_alive(dict(rec, deadline=now - RUNNING_GRACE_S - 1), now=now)
    assert not running_alive(dict(rec, started_epoch=None), now=now)


def test_plan_status_text_is_a_summary_json_keeps_every_unit(tmp_path, monkeypatch, capsys):
    mp = standard_matrix(tmp_path, monkeypatch, seeds=[0, 1], shard_minutes=1, clips_per_s=0.1)
    units = plan(str(mp))
    p = tmp_path / "plan.jsonl"
    write_plan(units, str(p))
    ex = [u for u in units if u.kind == "extract"]
    pr = [u for u in units if u.kind == "probe"]
    write_done_extract(ex[0])
    write_last(ex[1], status="preempted", exit_code=4, message="child exit 4")
    write_last(pr[0], status="failed", exit_code=5, log="/x/logs/u.log",
               message="RuntimeError: first line\nsecond line: the one that matters")
    rc = cli.main(["plan", "status", str(p)])
    out = capsys.readouterr().out
    assert rc == 0 and not out.lstrip().startswith("{") and '"units"' not in out
    lines = out.splitlines()
    assert lines[0].startswith(f"{p}: {len(units)} units") and "failed 1" in lines[0] and "preempted 1" in lines[0]
    assert any(ln.strip().startswith("extract ") for ln in lines) and any(ln.strip().startswith("probe ") for ln in lines)
    i = lines.index("failed:")
    assert pr[0].id in lines[i + 1] and lines[i + 1].endswith("second line: the one that matters") and "exit=5" in lines[i + 1]
    assert lines[i + 2].strip() == "log: /x/logs/u.log"
    assert "preempted:" in lines and any(ex[1].id in ln for ln in lines)
    rc = cli.main(["--json", "plan", "status", str(p)])
    data = json.loads(capsys.readouterr().out)["data"]
    assert len(data["units"]) == len(units) and data["failed"] == 1 and "running" in data and data["stale_running"] == []
    # the formatter caps long lists and points to --json
    many = {"total": 30, "failed": 30, "by_kind": {}, "units": [{"id": f"{i:012d}", "kind": "probe", "status": "failed",
                                                                  "exit_code": 5, "message": "x"} for i in range(30)]}
    assert "… 10 more (plan status --json lists every unit)" in format_status(many, max_listed=20)


# ============================================================================= dense probe estimates


def test_probe_minutes_count_the_shuffled_control_fit():
    ram = 16 * 2**30
    one = E.probe_needs(bytes_train=0, N=2080, D=768, probe_minutes_ref=6, seeds=1, ram_budget_bytes=ram)
    two = E.probe_needs(bytes_train=0, N=2080, D=768, probe_minutes_ref=6, seeds=1, ram_budget_bytes=ram, fits=2)
    assert one["minutes"] == 6 * 24 and two["minutes"] == 2 * 6 * 24
    with pytest.raises(ValueError):
        E.probe_needs(bytes_train=0, N=1, D=1, probe_minutes_ref=1, seeds=1, ram_budget_bytes=ram, fits=0)
    assert U.probe_fits_for({"shuffled_control": True}) == 2
    assert U.probe_fits_for({"shuffled_control": False}) == 1 and U.probe_fits_for({}) == 1 and U.probe_fits_for(None) == 1


def test_plan_doubles_probe_minutes_for_a_shuffled_control_benchmark(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    base = {u.seed: u.needs["minutes"] for u in plan(str(standard_matrix(tmp_path / "a", monkeypatch))) if u.kind == "probe"}
    info = make_info(probe_minutes_ref=40.0)
    info.probe_fits = 2
    (tmp_path / "b").mkdir()
    mp = standard_matrix(tmp_path / "b", monkeypatch, infos={"synth_clf": info})
    doubled = {u.seed: u.needs["minutes"] for u in plan(str(mp)) if u.kind == "probe"}
    assert base == {0: 30} and doubled == {0: 2 * 40}  # floor 30 at ref 1; ref 40 × token scale 1 × 2 fits


@pytest.mark.parametrize(("name", "N", "D", "fits", "minutes"), [
    ("ave", 2080, 768, 1, 120),          # measured: 32–102 min per seed
    ("vggss_box", 2080, 768, 2, 288),    # measured: main + shuffled ≈ 133–157 GPU-min, ≈ 260 min across a kill
    ("nyuv2_depth", 196, 768, 1, 123),   # measured: 31–105 min per seed
])
def test_dense_probe_budgets_match_measured_runs(name, N, D, fits, minutes):
    from yavih.benchmark import Benchmark

    info = U.benchmark_info(Benchmark.load(name))
    assert info.probe_fits == fits
    got = E.probe_needs(bytes_train=0, N=N, D=D, probe_minutes_ref=info.probe_minutes_ref, seeds=1,
                        ram_budget_bytes=16 * 2**30, fits=info.probe_fits)["minutes"]
    assert got == minutes


def test_read_plan_of_a_pending_file_has_no_other_kinds(tmp_path, monkeypatch):
    """The premise of ``--deps``: a ``--pending --kind probe`` file carries probe units only."""
    mp = standard_matrix(tmp_path, monkeypatch)
    pf = tmp_path / "pending.probe.jsonl"
    write_plan(plan(str(mp), pending=True, kind="probe"), str(pf))
    units = read_plan(str(pf))
    assert units and {u.kind for u in units} == {"probe"} and all(u.depends_on for u in units)


# ============================================================================= matrix results key


def test_matrix_results_key_is_accepted_and_reaches_the_probe(tmp_path, monkeypatch):
    """The matrix ``results`` key is accepted and reaches the probe."""
    import yaml

    import yavih.readouts as R
    from yavih.plan.matrix import MatrixError, load_matrix

    mp = standard_matrix(tmp_path, monkeypatch)
    raw = yaml.safe_load(mp.read_text())
    raw["results"] = None
    mp.write_text(yaml.safe_dump(raw))
    assert load_matrix(str(mp)).results is None
    plain = plan(str(mp))
    assert all("--results" not in u.cmd for u in plain)
    raw["results"] = "res/"
    mp.write_text(yaml.safe_dump(raw))
    units = plan(str(mp))
    pr = [u for u in units if u.kind == "probe"]
    assert pr and all(u.cmd[-2:] == ["--results", "res/"] for u in pr)
    assert {u.id for u in units} == {u.id for u in plain}  # unit ids never depend on where results go
    assert all("--results" not in u.cmd for u in units if u.kind != "probe")
    seen = {}

    def fake_probe(run_dir, **kw):
        seen.update(kw)
        write_done_probe(pr[0])
        return []

    monkeypatch.setattr(R, "probe", fake_probe)
    pf = tmp_path / "plan.jsonl"
    write_plan(units, str(pf))
    for u in units:
        if u.kind == "extract":
            write_done_extract(u)
    assert X.exec_unit(str(pf), unit_id=pr[0].id)["status"] == "done"
    assert seen["results_root"] == os.path.normpath(os.path.join(pr[0].cwd, "res/"))
    for bad in (5, "", "  "):
        raw["results"] = bad
        mp.write_text(yaml.safe_dump(raw))
        with pytest.raises(MatrixError, match="matrix.results"):
            load_matrix(str(mp))
