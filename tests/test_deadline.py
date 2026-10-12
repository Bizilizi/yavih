"""``yavih.deadline``: the one ``--deadline`` parser exec, extract and probe run share."""

from __future__ import annotations

import time

import pytest

from yavih import extract
from yavih.deadline import DeadlineError, parse_deadline, parse_duration
from yavih.plan import exec as X
from yavih.readouts import base

NOW = 1_800_000_000.0


@pytest.mark.parametrize("text,secs", [("90", 90), ("90s", 90), ("15m", 900), ("2h", 7200), ("1h30m", 5400), ("1d", 86400),
                                       ("1.5h", 5400), ("1m30", 90)])
def test_parse_duration(text, secs):
    assert parse_duration(text) == secs


@pytest.mark.parametrize("text", ["", "h", "soon", "1x", "-5m", "1h-30m"])
def test_parse_duration_rejects(text):
    assert parse_duration(text) is None


@pytest.mark.parametrize("spec,want", [
    ("+3600", NOW + 3600), ("+90m", NOW + 5400), ("2h", NOW + 7200), ("45s", NOW + 45), ("1700000000", 1700000000.0),
    ("2027-01-15T12:00:00Z", 1800014400.0), ("2027-01-15T12:00:00", 1800014400.0), ("2027-01-15T13:00:00+01:00", 1800014400.0),
    ("none", None), ("OFF", None), (None, None), ("", None), (1234.5, 1234.5),
])
def test_forms(spec, want):
    got = parse_deadline(spec, env={}, now=NOW)
    assert got == (pytest.approx(want) if want is not None else None)


def test_slurm_source():
    env = {"SLURM_JOB_ID": "7", "SLURM_JOB_END_TIME": str(int(NOW))}
    assert parse_deadline("slurm", env=env) == NOW - 900.0
    assert parse_deadline("SLURM", env=env, margin_s=60) == NOW - 60.0
    assert parse_deadline("slurm", env={}) is None  # outside SLURM: no deadline
    for bad in ({"SLURM_JOB_ID": "7"}, {"SLURM_JOB_ID": "7", "SLURM_JOB_END_TIME": "Unknown"}):
        with pytest.raises(DeadlineError) as ei:  # never "now": that refused every unit and looped forever
            parse_deadline("slurm", env=bad)
        assert ei.value.exit_code == 1 and ei.value.code == "E_DEADLINE_SPEC" and "+<seconds>" in str(ei.value)


@pytest.mark.parametrize("spec", ["3600", "0", "tomorrow", "+", "+soon", "nan", "inf", "12:00"])
def test_misconfigured_specs_are_exit_1(spec):
    with pytest.raises(DeadlineError) as ei:
        parse_deadline(spec, env={})
    assert ei.value.exit_code == 1


def test_inherit_yavih_deadline():
    assert parse_deadline(None, env={"YAVIH_DEADLINE": "1700000000.5"}) == 1700000000.5
    assert parse_deadline(None, env={"YAVIH_DEADLINE": "1700000000.5"}, inherit=False) is None


def test_every_entry_point_uses_the_shared_parser():
    """exec (no inherit), extract and probe run agree on every form, including '+3600' (relative, never an epoch)."""
    now = time.time()
    env: dict[str, str] = {}
    for f in (lambda s: X.resolve_deadline(s, env=env), lambda s: extract.parse_deadline(s, env),
              lambda s: base.resolve_deadline(s, env=env)):
        assert abs(f("+3600") - (now + 3600)) < 5
        assert f("none") is None
        with pytest.raises(DeadlineError):
            f("3600")
    with pytest.raises(DeadlineError):
        X.resolve_deadline("slurm", env={"SLURM_JOB_ID": "1"})


def test_cli_exec_bad_deadline_is_one_usage_envelope(tmp_path, capsys):
    import json

    from yavih import cli

    plan = tmp_path / "plan.jsonl"
    plan.write_text("", encoding="utf-8")
    rc = cli.main(["--json", "exec", str(plan), "--index", "0", "--deadline", "3600"])
    env = json.loads(capsys.readouterr().out.strip())
    assert rc == 1 and env["status"] == "error" and env["errors"][0]["code"] == "E_DEADLINE_SPEC"
