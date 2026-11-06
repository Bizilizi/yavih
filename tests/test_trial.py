"""``yavih bench trial <name|folder> --synthetic`` (``yavih.trial.run_trial``): every packaged folder's whole pipeline
on its synthetic stand-in, per-stage status and exit codes, and the isolation contract (no user root, store, home or
configuration is read or written; the environment and the CLI configuration layer are restored)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from yavih import cli, config
from yavih import testing as T
from yavih.benchmarks import packaged_folder, packaged_names
from yavih.scaffold import DEFAULT_LIKE
from yavih.trial import STAGES, TRIAL_DOC, TrialError, run_trial

pytest.importorskip("torch")
pytest.importorskip("av")

PACKAGED = packaged_names()


def _json(capsys, argv: list[str]) -> tuple[int, dict]:
    rc = cli.main(["--json", *argv])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1, out
    return rc, json.loads(out[0])


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


def _yavih_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k.startswith("YAVIH_") or k in ("XDG_CONFIG_HOME", "CUBLAS_WORKSPACE_CONFIG")}


def test_every_default_like_readout_is_covered():
    """The packaged folders cover every readout of the scaffold's ``--like`` defaults (clf, seq, grid, box)."""
    assert set(DEFAULT_LIKE.values()) <= set(PACKAGED) and set(DEFAULT_LIKE) == set(T.READOUTS)


def _yaml_scorer(name: str) -> str:
    import yaml

    from yavih.benchmarks import packaged_folder

    return str(yaml.safe_load((packaged_folder(name) / "benchmark.yaml").read_text(encoding="utf-8"))["scorer"])


def _trial_kind(name: str) -> tuple:
    """What selects the synthetic stand-in's pipeline: the scorer and the label kind (the stand-in writes each stream
    kind with the same writers whatever the folder)."""
    import yaml

    from yavih.benchmarks import packaged_folder

    y = yaml.safe_load((packaged_folder(name) / "benchmark.yaml").read_text(encoding="utf-8"))
    return (y["scorer"], (y["manifest"].get("label") or {}).get("kind"))


#: One folder per scorer and label kind: the trial runs the same pipeline for every folder of a kind, so a new folder
#: of a known kind adds nothing and a new kind is covered on its own (B8.3: the trial per scorer kind, not per folder).
TRIAL_REPRESENTATIVES = sorted({_trial_kind(n): n for n in reversed(PACKAGED)}.values())


@pytest.mark.parametrize("name", TRIAL_REPRESENTATIVES)
def test_bench_trial_synthetic(name, capsys, yavih_home, runs_dir, monkeypatch):
    """Every stage ``ok`` and the oracle clears the headline floor under ``--readout-fast``; the user's home and runs
    root stay untouched and the environment comes back exactly as it was."""
    monkeypatch.setenv("YAVIH_RESULTS", str(runs_dir.parent / "results"))
    home_before, env_before, cli_before = _tree(yavih_home), _yavih_env(), config.cli_overrides()
    no_stand_in = _yaml_scorer(name) not in T.READOUTS and not (packaged_folder(name) / "testing.py").is_file()
    if no_stand_in:  # the conformance stage is the folder's own check (tests/test_conformance.py); here: what the trial does without a stand-in
        import yavih.conformance as C

        monkeypatch.setattr(C, "check", lambda bench: C.ConformanceReport(name=bench.name, folder=str(bench.folder), protocol_version=1, expected_version=1, ok=True, skipped=False))
    rc, env = _json(capsys, ["bench", "trial", name, "--synthetic", "--readout-fast"])
    if no_stand_in:  # a scorer no stand-in covers (stream, the generation beacons' own): conformance, then not_applicable
        rec = env["data"]
        assert rc == 0 and env["status"] == "ok" and rec["stand_in"] == "none", env["errors"]
        assert [s["status"] for s in rec["stages"]] == ["ok"] + ["not_applicable"] * (len(STAGES) - 1)
        assert any("no synthetic stand-in" in lim for lim in rec["limits"])
        assert _tree(yavih_home) == home_before and _tree(runs_dir) == []
        return
    rec = env["data"]
    assert rc == 0 and env["status"] == "ok" and env["errors"] == [], (rec.get("failed"), rec.get("stages"))
    assert [s["stage"] for s in rec["stages"]] == list(STAGES) and all(s["status"] == "ok" for s in rec["stages"])
    assert rec["benchmark"] == name and rec["synthetic"] is True and rec["exit_code"] == 0 and rec["paths"] == {}
    h = rec["headline"]
    assert h["clears_floor"] is True and h["standard"] is False and h["value"] > h["floor"], h
    assert any("synthetic stand-in" in lim for lim in rec["limits"])
    if name in ("fsd50k", "audioset20k", "vggsounder"):  # multi / multi_sets: trialled single-label
        assert any("single-label" in lim for lim in rec["limits"]), rec["limits"]
    assert _tree(yavih_home) == home_before and _tree(runs_dir) == [] and not (runs_dir.parent / "results").exists()
    assert _yavih_env() == env_before and config.cli_overrides() == cli_before


def test_bench_trial_refuses_without_synthetic(capsys):
    rc, env = _json(capsys, ["bench", "trial", "esc50"])
    assert rc == 1 and env["errors"][0]["code"] == "E_USAGE" and TRIAL_DOC in env["errors"][0]["message"]
    assert TRIAL_DOC == "https://github.com/Bizilizi/yavih/blob/main/docs/trial-run.md"


def test_bench_trial_keep_returns_paths_and_refuses_a_used_directory(tmp_path, capsys):
    keep = tmp_path / "trial"
    rc, env = _json(capsys, ["bench", "trial", "esc50", "--synthetic", "--readout-fast", "--keep", str(keep)])
    assert rc == 0, env["errors"]
    paths = env["paths"]
    assert set(paths) == {"work", "folder", "bridge", "profile", "run_dir", "report"} and paths["work"] == str(keep)
    assert all(Path(p).exists() for p in paths.values()) and "esc50" in Path(paths["report"]).read_text()
    assert Path(paths["run_dir"]).relative_to(keep).parts[:2] == ("runs", "esc50")
    assert not list(keep.rglob("results_*.json"))  # the probe never writes the results store
    rc, env = _json(capsys, ["bench", "trial", "esc50", "--synthetic", "--keep", str(keep)])
    assert rc == 1 and env["errors"][0]["code"] == "E_EXISTS"


def test_bench_trial_not_ready_when_no_stand_in_exists(capsys, monkeypatch):
    """A readout or stream kind the synthetic fixtures do not cover: the conformance stage, then ``not_applicable``
    (exit 0); no media writer: exit 3, a message, no traceback."""
    monkeypatch.setattr(T, "READOUTS", ("seq", "grid", "box"))
    rc, env = _json(capsys, ["bench", "trial", "esc50", "--synthetic"])
    assert rc == 0 and env["data"]["stand_in"] == "none" and env["data"]["stages"][1]["status"] == "not_applicable"
    assert "no synthetic stand-in" in env["data"]["stages"][1]["detail"]["reason"]
    monkeypatch.undo()

    def no_writer(*a, **k):
        raise ImportError("No module named 'soundfile'")

    monkeypatch.setattr(T, "make_synthetic_like", no_writer)
    env_before = _yavih_env()
    rc, env = _json(capsys, ["bench", "trial", "esc50", "--synthetic"])
    assert rc == 3 and env["errors"][0]["code"] == "E_TRIAL_NOT_READY" and "soundfile" in env["errors"][0]["message"]
    assert env["data"]["failed"]["stage"] == "stand_in" and [s["status"] for s in env["data"]["stages"]][2:] == ["skipped"] * 7
    assert _yavih_env() == env_before
    with pytest.raises(TrialError) as ei:
        run_trial("esc50", bridge="nope")
    assert ei.value.exit_code == 1


def test_bench_trial_failing_stage_is_exit_2(tmp_path):
    """The random bridge cannot clear the floor: under ``readout_fast`` the headline stage fails a contract (exit 2),
    every earlier stage is ``ok`` and the environment is restored."""
    env_before = _yavih_env()
    rec = run_trial("esc50", readout_fast=True, bridge="random")
    assert rec["ok"] is False and rec["exit_code"] == 2 and rec["failed"]["stage"] == "headline"
    assert [s["status"] for s in rec["stages"]] == ["ok"] * 8 + ["failed"] and rec["headline"]["clears_floor"] is False
    assert _yavih_env() == env_before
