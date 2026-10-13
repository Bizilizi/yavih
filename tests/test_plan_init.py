"""``yavih plan init`` (``yavih.plan.skeleton``): the matrix skeleton of one model on a suite and its bridge stubs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from test_plan_helpers import make_profile, write_profile
from yavih import cli
from yavih.plan.matrix import check_profile, load_matrix, load_profile
from yavih.plan.skeleton import plan_init, write_matrix_skeleton
from yavih.suites import SuiteError


def _json(capsys, argv: list[str]) -> tuple[int, dict]:
    rc = cli.main(["--json", *argv])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1, out
    return rc, json.loads(out[0])


def test_skeleton_written_and_bridges_created(tmp_path, runs_dir):
    m = tmp_path / "proj" / "matrix.yaml"
    rec = plan_init("minimal", model="toy", launcher="/envs/toy/bin/python", out=m)
    assert [e["key"] for e in rec["entries"]] == ["toy-esc50-a", "toy-ssv2-v", "toy-vggsounder-a", "toy-vggsounder-v",
                                                  "toy-vggsounder-av"]
    # ONE bridge file per benchmark, even for the three vggsounder readings
    assert [(b["benchmark"], b["action"]) for b in rec["bridges"]] == [("esc50", "created"), ("ssv2", "created"), ("vggsounder", "created")]
    for b in ("esc50", "ssv2", "vggsounder"):
        text = (m.parent / "bridges" / f"toy_{b}.py").read_text(encoding="utf-8")
        assert '"model": "toy"' in text and "TODO" in text
    raw = yaml.safe_load(m.read_text(encoding="utf-8"))
    assert raw["defaults"]["launcher"] == "/envs/toy/bin/python" and raw["out"] is None
    ent = raw["bridges"]
    assert ent["toy-esc50-a"] == {"script": "bridges/toy_esc50.py", "benchmark": "esc50", "profile": "bridges/toy-esc50-a.profile.json"}
    assert ent["toy-vggsounder-av"] == {"script": "bridges/toy_vggsounder.py", "benchmark": "vggsounder", "args": {"reading": "av"},
                                        "profile": "bridges/toy-vggsounder-av.profile.json"}
    text = m.read_text(encoding="utf-8")
    assert "# waive:" in text and "E_PROFILE" in text and "--join vggsounder" in text
    # next steps: one bridge check per entry (--arg only for the multi-reading template), then plan build
    nxt = rec["next"]
    assert len(nxt) == 6 and nxt[-1].startswith("yavih plan build ") and nxt[-1].endswith("plan.jsonl")
    assert nxt[0].startswith("yavih bridge check ") and "--bench esc50 --out " in nxt[0] and "--arg" not in nxt[0]
    assert "--bench vggsounder --arg reading=v --out " in nxt[3] and nxt[3].endswith("toy-vggsounder-v.profile.json")


def test_skeleton_loads_through_load_matrix_once_profiles_exist(tmp_path, runs_dir):
    m = tmp_path / "matrix.yaml"
    rec = plan_init("default", model="toy", launcher="python", out=m)
    matrix = load_matrix(str(m))
    assert len(matrix.entries) == len(rec["entries"]) == 13 and matrix.out == str(runs_dir)
    assert len({e.script for e in matrix.entries}) == 9  # one bridge file per benchmark
    for ent in matrix.entries:
        bench, reading = ent.benchmark, ent.key.rsplit("-", 1)[1]
        prof = make_profile(bench, model="toy", reading=reading, args=dict(ent.args))
        write_profile(tmp_path, ent.profile, prof)
        check_profile(load_profile(str(tmp_path / ent.profile)), ent)
        assert ent.launcher == "python" and ent.cpus == 8
        multi = bench in ("vggsounder", "audioset20k", "avsync")
        assert ent.args == ({"reading": reading} if multi else {})


def test_existing_bridges_kept_and_matrix_refused(tmp_path, runs_dir, capsys):
    m = tmp_path / "matrix.yaml"
    (tmp_path / "bridges").mkdir()
    (tmp_path / "bridges" / "toy_esc50.py").write_text("# filled by hand\n", encoding="utf-8")
    rec = plan_init("minimal", model="toy", launcher="python", out=m)
    assert {b["benchmark"]: b["action"] for b in rec["bridges"]} == {"esc50": "kept", "ssv2": "created", "vggsounder": "created"}
    assert (tmp_path / "bridges" / "toy_esc50.py").read_text(encoding="utf-8") == "# filled by hand\n"
    with pytest.raises(SuiteError) as ei:
        plan_init("minimal", model="toy", launcher="python", out=m)
    assert ei.value.code == "E_EXISTS" and ei.value.exit_code == 1
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy", "--launcher", "python", "--out", str(m)])
    assert rc == 1 and env["errors"][0]["code"] == "E_EXISTS"
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy", "--launcher", "py3", "--out", str(m), "--force"])
    assert rc == 0 and {b["action"] for b in env["data"]["bridges"]} == {"kept"}
    assert yaml.safe_load(m.read_text(encoding="utf-8"))["defaults"]["launcher"] == "py3"
    assert env["paths"]["matrix"] == str(m.resolve())
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "_bad", "--launcher", "py3", "--out", str(tmp_path / "x.yaml")])
    assert rc == 1 and env["errors"][0]["code"] == "E_ARG"


def test_pending_entries_become_comments(tmp_path, runs_dir):
    suite = tmp_path / "mine.yaml"
    suite.write_text(yaml.safe_dump({
        "schema_version": 1, "name": "mine", "description": "one packaged, one pending",
        "groups": [{"capability": "audio", "entries": [{"benchmark": "esc50", "readings": ["a"]}]},
                   {"capability": "temporal", "entries": [{"benchmark": "future_sync", "readings": ["av"], "status": "pending"}]}],
    }), encoding="utf-8")
    m = tmp_path / "matrix.yaml"
    rec = write_matrix_skeleton(str(suite), model="toy", launcher="python", out=m)
    assert rec["pending"] == [{"capability": "temporal", "benchmark": "future_sync", "readings": ["av"]}]
    text = m.read_text(encoding="utf-8")
    assert "#   temporal: future_sync [av]" in text and "future_sync" not in "".join(yaml.safe_load(text)["bridges"])
    assert [e.key for e in load_matrix(str(m)).entries] == ["toy-esc50-a"]


def test_cli_plan_init_text_lists_next_steps(tmp_path, runs_dir, capsys):
    m = tmp_path / "matrix.yaml"
    assert cli.main(["plan", "init", "--suite", "minimal", "--model", "toy", "--launcher", "python", "--out", str(m)]) == 0
    out = capsys.readouterr().out
    checks = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("yavih bridge check ")]
    assert len(checks) == 5 and sum("--arg reading=" in c for c in checks) == 3
    assert f"yavih plan build {m} --out {tmp_path / 'plan.jsonl'}" in out
    assert "yavih suite missing minimal --model toy" in out
    assert Path(m).is_file()
