"""The feature cleanup policy (``yavih.plan.clean``): ``[features] keep``, the ``cleaned`` state of an extract unit,
``exec`` dropping the tars after the last probe of a run directory, ``plan clean``, ``features keep`` writing the
project file, and the readers that meet a cleaned split (validate, the probe)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from test_plan_helpers import (
    fake_unit,
    write_done_extract,
    write_done_probe,
    write_fake_script,
    write_last,
)
from yavih import cli, config
from yavih.io.format import CLEANED_FILE
from yavih.plan import clean as C
from yavih.plan import exec as X
from yavih.plan import units as U
from yavih.plan.status import plan_status
from yavih.plan.units import Unit, write_plan


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root, write_fake_script(root)


def _config(tmp_path: Path, monkeypatch, body: str) -> Path:
    p = tmp_path / "yavih.toml"
    p.write_text(body, encoding="utf-8")
    monkeypatch.setenv("YAVIH_CONFIG", str(p))
    return p


def _run(root: Path, script: Path, *, seeds=(0, 1), shards=2, model="fake") -> tuple[list[Unit], list[Unit], str]:
    """Extract units (sidecar + a 1 MB tar each) and probe units over them; returns (extracts, probes, plan path)."""
    ex = [fake_unit(root, script, shard=(i, shards), model=model) for i in range(shards)]
    for u in ex:
        side = write_done_extract(u)
        side.with_suffix(".tar").write_bytes(b"\0" * 1_000_000)
    probes = [fake_unit(root, script, kind="probe", seed=s, model=model) for s in seeds]
    for p in probes:
        p.depends_on = [u.id for u in ex]
    plan = root / "plan.jsonl"
    write_plan([*ex, *probes], str(plan))
    return ex, probes, str(plan)


def _tars(u: Unit) -> Path:
    return Path(U.resolve_path(u, u.done)).with_suffix(".tar")


def _finish(probe: Unit, monkeypatch) -> dict:
    """Run one probe unit through exec with a fake in-process probe that writes its DONE."""
    import yavih.readouts as R

    monkeypatch.setattr(R, "probe", lambda run_dir, **kw: write_done_probe(probe) and [])
    return X.exec_unit(str(Path(probe.cwd) / "plan.jsonl"), unit_id=probe.id)


# ----------------------------------------------------------------------------- the cleaned state


def test_check_done_is_cleaned_only_with_the_record(workspace):
    root, script = workspace
    (u,), _, _ = _run(root, script, shards=1, seeds=(0,))
    assert U.check_done(u) == "done"
    _tars(u).unlink()
    assert U.check_done(u) == "done"  # a lost tar without the cleanup record: validation's E_SHARD_SET, as before
    (_tars(u).parent / CLEANED_FILE).write_text("{}", encoding="utf-8")
    assert U.check_done(u) == U.CLEANED


def test_resolve_done_brings_a_cleaned_extraction_back_for_a_pending_probe(workspace):
    root, script = workspace
    ex, probes, _ = _run(root, script)
    for p in probes:
        write_done_probe(p)
    C.drop_features(U.resolve_path(probes[0], probes[0].out))
    states = U.resolve_done([*ex, *probes])
    assert [states[u.id] for u in ex] == [U.CLEANED, U.CLEANED]
    os.remove(U.resolve_path(probes[1], probes[1].done))  # a probe must run again (a new seed, a bumped protocol)
    states = U.resolve_done([*ex, *probes])
    assert [states[u.id] for u in ex] == [None, None] and states[probes[0].id] == "done"
    # no probe in sight (a pending extract file without its probes): planned because needed
    assert U.resolve_done(ex) == {u.id: None for u in ex}


# ----------------------------------------------------------------------------- exec


def test_exec_drops_the_tars_after_the_last_probe_and_keeps_the_records(workspace, monkeypatch):
    root, script = workspace
    ex, probes, plan = _run(root, script)
    first = _finish(probes[0], monkeypatch)
    assert first["status"] == "done" and "cleanup" not in first  # seed 1 still to run: nothing dropped
    assert all(_tars(u).exists() for u in ex)
    last = _finish(probes[1], monkeypatch)
    assert last["status"] == "done" and last["cleanup"]["status"] == "dropped"
    assert last["cleanup"]["bytes"] == 2_000_000 and last["cleanup"]["policy"] == {"keep": False, "source": "default"}
    assert "To keep features: yavih features keep" in last["cleanup"]["message"]
    split = _tars(ex[0]).parent
    assert not any(split.glob("*.tar")) and len(list(split.glob("shard-*.json"))) == 2
    rec = json.loads((split / CLEANED_FILE).read_text())
    assert rec["cleanups"][-1]["tars"] == 2 and rec["cleanups"][-1]["by"] == f"exec {probes[1].id}"
    st = plan_status(plan)
    assert st["cleaned"] == 2 and st["done"] == 2 and st["pending"] == 0
    units = U.read_plan(plan)
    assert all(v is not None for v in U.resolve_done(units).values())  # plan build --pending: nothing to run


def test_a_kept_run_directory_is_left_alone(workspace, monkeypatch, tmp_path):
    root, script = workspace
    _config(tmp_path, monkeypatch, '[features]\nkeep = ["synth_clf/fake"]\n')
    ex, probes, _ = _run(root, script, seeds=(0,))
    last = _finish(probes[0], monkeypatch)
    assert last["cleanup"]["status"] == "kept" and last["cleanup"]["policy"]["keep"] == ["synth_clf/fake/*"]
    assert all(_tars(u).exists() for u in ex)


def test_a_running_extract_of_the_run_directory_blocks_the_cleanup(workspace, monkeypatch):
    root, script = workspace
    ex, probes, _ = _run(root, script, seeds=(0,))
    write_last(ex[0], status="running", host=__import__("socket").gethostname(), pid=os.getpid())
    last = _finish(probes[0], monkeypatch)
    assert "cleanup" not in last and all(_tars(u).exists() for u in ex)


def test_exec_refuses_a_pending_probe_over_cleaned_shards_and_reextracts_them(workspace, monkeypatch):
    root, script = workspace
    ex, probes, plan = _run(root, script)
    for p in probes:
        write_done_probe(p)
    C.drop_features(U.resolve_path(probes[0], probes[0].out))
    # every probe done: the cleaned shard is skipped, the probes too
    assert X.exec_unit(plan, unit_id=ex[0].id)["message"].startswith("already cleaned")
    os.remove(U.resolve_path(probes[1], probes[1].done))
    rec = X.exec_unit(plan, unit_id=probes[1].id)
    assert rec["status"] == "not_ready" and rec["exit_code"] == 3 and "plan build --pending" in rec["message"]
    # the shard is needed again: exec runs the extraction instead of skipping it
    ran = []

    def fake_extract(unit, **kw):
        ran.append(unit.id)
        _tars(unit).write_bytes(b"\0" * 10)
        return 0, False

    monkeypatch.setattr(X, "_run_extract", fake_extract)
    again = X.exec_unit(plan, unit_id=ex[0].id)
    assert ran == [ex[0].id] and again["status"] == "done" and U.check_done(ex[0]) == "done"


# ----------------------------------------------------------------------------- plan clean


def test_plan_clean_dry_run_then_drop(workspace, monkeypatch, capsys):
    root, script = workspace
    ex, probes, plan = _run(root, script)
    write_done_probe(probes[0])
    data = C.clean_plan(plan)
    assert data["not_finished"] == 1 and data["bytes"] == 0  # seed 1 not done
    write_done_probe(probes[1])
    (_tars(ex[0]).parent / "_stale" / "x").mkdir(parents=True)
    (_tars(ex[0]).parent / "_stale" / "x" / "shard-000000.tar").write_bytes(b"\0" * 500)
    dry = C.clean_plan(plan, dry_run=True)
    assert dry["would_drop"] == 1 and dry["bytes"] == 2_000_500 and all(_tars(u).exists() for u in ex)
    assert cli.main(["--json", "plan", "clean", plan]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["data"]["dropped"] == 1 and env["data"]["files"] == 3
    assert not any(_tars(u).exists() for u in ex) and not (_tars(ex[0]).parent / "_stale").exists()
    assert C.clean_plan(plan)["nothing"] == 1  # idempotent


# ----------------------------------------------------------------------------- readers of a cleaned split


def test_validate_and_the_probe_name_a_cleaned_split(workspace):
    from yavih.io.validate import CODES, validate_features
    from yavih.readouts.base import _cleaned_shards

    root, script = workspace
    ex, probes, _ = _run(root, script)
    split = _tars(ex[0]).parent
    assert _cleaned_shards(split) == 0
    C.drop_features(U.resolve_path(probes[0], probes[0].out))
    rep = validate_features(split, write_meta=False)
    assert [f.code for f in rep.errors] == ["E_FEATURES_CLEANED"] and CODES["E_FEATURES_CLEANED"] == "error"
    assert _cleaned_shards(split) == 2


# ----------------------------------------------------------------------------- configuration and features keep


def test_the_keep_schema_is_strict(tmp_path, monkeypatch):
    for body, keep in (("[features]\nkeep = true\n", True), ('[features]\nkeep = ["k400", "*/m", "a/b/c"]\n', ("k400/*/*", "*/m/*", "a/b/c"))):
        _config(tmp_path, monkeypatch, body)
        assert config.load(refresh=True).features_keep() == keep
    s = config.load(refresh=True)
    assert s.keeps_features("a", "b", "c") and not s.keeps_features("a", "b", "d") and s.keeps_features("x", "m", "v")
    for bad in ('[features]\nkeep = "yes"\n', '[features]\nkeep = ["a/b/c/d"]\n', "[features]\nkeep_all = true\n"):
        _config(tmp_path, monkeypatch, bad)
        with pytest.raises(config.ConfigError):
            config.load(refresh=True)


def test_set_features_keep_edits_one_line_and_keeps_the_rest(tmp_path):
    p = tmp_path / "yavih.toml"
    p.write_text('# head\n[project]\nyavih = ">=1"\n\n[features]\n# why\nkeep = false   # inline\n\n[data]\nroot = "/d"\n')
    C.set_features_keep(p, ["k400/*/*"], comment="2026-01-01 keep: a diff")
    text = p.read_text()
    assert '[features]\n# why\n# 2026-01-01 keep: a diff\nkeep = ["k400/*/*"]\n\n[data]\nroot = "/d"\n' in text and text.startswith("# head\n")
    q = tmp_path / "bare.toml"
    q.write_text('[project]\nyavih = ">=1"\n')
    C.set_features_keep(q, True)
    assert q.read_text() == '[project]\nyavih = ">=1"\n\n[features]\nkeep = true\n'
    p.write_text('[features]\nkeep = [\n  "a/b/c",\n]\n')
    with pytest.raises(C.KeepEditError):
        C.set_features_keep(p, False)


def test_features_keep_cli_records_the_request_in_the_project_file(tmp_path, monkeypatch, capsys):
    proj = tmp_path / "repo" / "yavih-bench"
    proj.mkdir(parents=True)
    (tmp_path / "repo" / ".git").mkdir()
    toml = proj / "yavih.toml"
    toml.write_text('[project]\nyavih = ">=0"\n', encoding="utf-8")
    monkeypatch.delenv("YAVIH_CONFIG")
    monkeypatch.chdir(tmp_path / "repo")
    assert cli.main(["--json", "features", "keep"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["keep"] is False
    assert cli.main(["features", "keep", "k400"]) == 1  # --reason is required
    capsys.readouterr()
    assert cli.main(["--json", "features", "keep", "k400", "*/mymodel/v", "--reason", "diff against v2"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["data"]["keep"] == ["k400/*/*", "*/mymodel/v"] and env["data"]["changed"] is True
    assert "yavih features keep k400/*/*, */mymodel/v: diff against v2" in toml.read_text()
    assert cli.main(["--json", "features", "keep", "k400", "--reason", "again"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["changed"] is False
    assert cli.main(["--json", "config", "get", "features.keep"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["value"] == ["k400/*/*", "*/mymodel/v"]
    assert cli.main(["--json", "features", "keep", "--none"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["keep"] is False
