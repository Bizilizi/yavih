"""``plan status`` counts and ``plan calibrate``: exit-4 bump, idempotence, multipliers."""

from __future__ import annotations

import json
import os

import pytest

from test_plan_helpers import (
    fake_unit,
    python_launcher,
    standard_matrix,
    write_done_extract,
    write_done_probe,
    write_fake_script,
    write_last,
)
from yavih import cli
from yavih.plan import calibrate as C
from yavih.plan import exec as X
from yavih.plan import units as U
from yavih.plan.status import format_status, plan_status, unit_status
from yavih.plan.units import plan, write_plan


def test_status_counts_from_markers_and_last_json(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch, seeds=[0, 1], shard_minutes=1, clips_per_s=0.1)  # 6 rows/shard: 2 + 1 shards
    units = plan(str(mp))
    p = tmp_path / "plan.jsonl"
    write_plan(units, str(p))
    ex = [u for u in units if u.kind == "extract"]
    pr = [u for u in units if u.kind == "probe"]
    assert len(ex) == 3
    st = plan_status(str(p))
    assert st["pending"] == len(units) and st["done"] == 0 and st["total"] == len(units)
    write_done_extract(ex[0])
    write_last(ex[1], status="preempted", exit_code=4)
    write_last(ex[2], status="failed", exit_code=2)
    write_done_probe(pr[0], status="not_applicable")
    write_last(pr[1], status="not_ready", exit_code=3)
    st = plan_status(str(p))
    assert (st["done"], st["preempted"], st["failed"], st["not_applicable"], st["not_ready"], st["pending"]) == (1, 1, 1, 1, 1, 1)
    assert st["by_kind"]["extract"] == {"done": 1, "pending": 0, "running": 0, "not_ready": 0, "preempted": 1, "misconfigured": 0,
                                        "failed": 1, "not_applicable": 0}
    assert st["by_kind"]["probe"]["not_applicable"] == 1 and st["by_kind"]["report"]["pending"] == 1
    rows = {r["id"]: r for r in st["units"]}
    assert rows[ex[1].id]["status"] == "preempted" and rows[ex[1].id]["exit_code"] == 4
    # a deadline refusal is exit 4 too → preempted (resubmittable), never failed
    write_last(ex[2], status="deadline", exit_code=4)
    assert plan_status(str(p))["preempted"] == 2 and plan_status(str(p))["failed"] == 0
    # exec said done but the marker vanished (re-plan moved it to _stale/) → pending again
    write_last(ex[2], status="done", exit_code=0)
    assert unit_status(ex[2]) == ("pending", 0)
    # exit 1 is its own status: a configuration error a driver loop must stop on
    write_last(ex[2], status="misconfigured", exit_code=1, message="E_LAYOUT: re-run plan build")
    st = plan_status(str(p))
    assert st["misconfigured"] == 1 and st["failed"] == 0 and "misconfigured" in format_status(st)
    # the same exit 3 three times in a row is flagged: resubmitting will not help
    write_last(pr[1], status="not_ready", exit_code=3, history={"attempts": 3, "status": "not_ready", "streak": 3})
    st = plan_status(str(p))
    assert st["repeated"] == [pr[1].id] and "repeated" in format_status(st)
    write_last(pr[1], status="not_ready", exit_code=3, history={"attempts": 3, "status": "not_ready", "streak": 2})
    assert plan_status(str(p))["repeated"] == []


def test_status_after_real_exec(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    script = write_fake_script(root)
    a = fake_unit(root, script, mode="ok", shard=(0, 3), launcher=python_launcher())
    b = fake_unit(root, script, mode="sigterm", shard=(1, 3), launcher=python_launcher())
    c = fake_unit(root, script, mode="exit3", shard=(2, 3), launcher=python_launcher())
    p = root / "plan.jsonl"
    write_plan([a, b, c], str(p))
    for i in range(3):
        X.exec_unit(str(p), index=i)
    st = plan_status(str(p))
    assert (st["done"], st["preempted"], st["not_ready"], st["failed"], st["pending"]) == (1, 1, 1, 0, 0)


def test_cli_plan_status_envelope(tmp_path, monkeypatch, capsys):
    mp = standard_matrix(tmp_path, monkeypatch)
    p = tmp_path / "plan.jsonl"
    write_plan(plan(str(mp)), str(p))
    rc = cli.main(["--json", "plan", "status", str(p)])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 0 and env["status"] == "ok"
    assert set(env["data"]) >= {"done", "pending", "not_ready", "preempted", "failed", "not_applicable", "by_kind"}
    rc = cli.main(["--json", "plan", "status", str(tmp_path / "missing.jsonl")])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 1 and env["errors"][0]["code"] == "E_PLAN"


# ----------------------------------------------------------------------------- calibrate


def test_calibrate_bumps_exit_4_units_once_per_attempt(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch)
    units = plan(str(mp))
    ex = [u for u in units if u.kind == "extract"]
    runs = os.path.join(str(tmp_path / "m"), "runs")
    os.makedirs(runs, exist_ok=True)
    write_last(ex[0], status="deadline", exit_code=4, ended="2026-10-07T01:00:00Z")
    rec = C.calibrate(runs)
    assert rec["units"][ex[0].id]["minutes"] == pytest.approx(1.5)
    assert rec["observations"]["bumped"] == [ex[0].id]
    assert os.path.exists(os.path.join(runs, C.CALIBRATION_FILE))
    # idempotent: no new attempt → no compounding
    rec = C.calibrate(runs)
    assert rec["units"][ex[0].id]["minutes"] == pytest.approx(1.5) and rec["observations"]["bumped"] == []
    # a preemption (SIGTERM) or a refusal at start says nothing about the unit's length: no bump
    write_last(ex[0], status="preempted", exit_code=4, ended="2026-10-07T01:30:00Z")
    assert C.calibrate(runs)["units"][ex[0].id]["minutes"] == pytest.approx(1.5)
    write_last(ex[0], status="deadline", exit_code=4, ended="2026-10-07T01:40:00Z", refused_at_start=True,
               message="300s left before the deadline, need 600s")
    assert C.calibrate(runs)["units"][ex[0].id]["minutes"] == pytest.approx(1.5)
    # a new real deadline stop compounds
    write_last(ex[0], status="deadline", exit_code=4, ended="2026-10-07T02:00:00Z")
    rec = C.calibrate(runs)
    assert rec["units"][ex[0].id]["minutes"] == pytest.approx(2.25)
    # the plan picks it up: that unit's minutes are 2.25× the uncalibrated estimate (ceil)
    before = ex[0].needs["minutes"]
    after = next(u for u in plan(str(mp)) if u.id == ex[0].id).needs["minutes"]
    assert after == -(-before * 2.25 // 1)
    # once the unit finishes, its bump is dropped
    write_last(ex[0], status="done", exit_code=0, ended="2026-10-07T03:00:00Z")
    rec = C.calibrate(runs)
    assert ex[0].id not in rec["units"]


def test_calibrate_multipliers_from_observed_walls(tmp_path, monkeypatch):
    mp = standard_matrix(tmp_path, monkeypatch, seeds=[0, 1])
    units = plan(str(mp))
    runs = os.path.join(str(tmp_path / "m"), "runs")
    os.makedirs(runs, exist_ok=True)
    ex = [u for u in units if u.kind == "extract"]
    pr = [u for u in units if u.kind == "probe"]
    # extract units took twice their plan; probes took a tenth
    for u in ex:
        write_done_extract(u)
        write_last(u, status="done", exit_code=0, wall_s=u.needs["minutes"] * 60 * 2.0)
    for u in pr:
        d = write_done_probe(u)
        (d.parent / "probe_fit.json").write_text(json.dumps({"epochs_run": 5, "s_per_epoch": 2.5, "readout": "clf"}))
        write_last(u, status="done", exit_code=0, wall_s=u.needs["minutes"] * 60 * 0.1)
    rec = C.calibrate(runs)
    assert rec["multipliers"]["extract_minutes"] == pytest.approx(2.0)
    assert rec["multipliers"]["probe_minutes"] == pytest.approx(C.MULT_FLOOR)  # never below the floor
    assert rec["observations"]["probe_s_per_epoch_median"] == {"clf": 2.5}
    assert rec["observations"]["probe_wall_s_median"] != {}
    # the plan applies the multipliers
    after = {u.id: u.needs["minutes"] for u in plan(str(mp))}
    for u in ex:
        assert after[u.id] >= u.needs["minutes"] * 2 - 1
    # a second calibrate compounds on the already-applied plan only through fresh observations; with the
    # same records the extract multiplier doubles again but is clamped at the ceiling eventually
    rec2 = C.calibrate(runs)
    assert rec2["multipliers"]["extract_minutes"] == pytest.approx(min(C.MULT_CEIL, 4.0))


def test_calibrate_missing_root_and_cli(tmp_path, capsys):
    with pytest.raises(FileNotFoundError):
        C.calibrate(str(tmp_path / "nope"))
    runs = tmp_path / "runs"
    runs.mkdir()
    rc = cli.main(["--json", "plan", "calibrate", str(runs)])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 0 and env["data"]["multipliers"] == {"extract_minutes": 1.0, "probe_minutes": 1.0} and env["data"]["units"] == {}
    assert json.load(open(runs / C.CALIBRATION_FILE))["schema"] == C.CALIBRATION_SCHEMA


def test_calibration_ignores_stale_dirs(tmp_path):
    runs = tmp_path / "runs"
    (runs / "_stale" / "x" / "logs").mkdir(parents=True)
    (runs / "_stale" / "x" / "logs" / "last.abc.json").write_text(json.dumps({"id": "abc", "exit_code": 4, "ended": "t"}))
    rec = C.calibrate(str(runs))
    assert rec["units"] == {} and rec["observations"]["n_last"] == 0


def test_resolve_path_relative_to_cwd(tmp_path):
    u = U.Unit(id="x" * 12, kind="report", benchmark="_report", bridge="_report", model="_report", reading="_report", split=None,
               shard=None, seed=None, protocol_hash="", probe_hash=None, eval_hash=None, depends_on=[], needs={}, cmd=[], launcher=None, env={},
               cwd=str(tmp_path), out="runs/_report", done="runs/_report/DONE", inputs_hash="")
    assert U.resolve_path(u, u.done) == str(tmp_path / "runs" / "_report" / "DONE")
    assert U.resolve_path(u, "/abs/x") == "/abs/x"
