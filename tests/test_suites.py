"""Suites (``yavih.suites``): the shipped files, validation, project shadowing, the ``yavih suite`` commands,
``bench list --suite`` and the ``report build --suite`` completeness appendix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from test_report import make_result, write_result
from yavih import cli, suites
from yavih.benchmarks import packaged_folder
from yavih.lint import template_readings
from yavih.report import report
from yavih.suites import SuiteError, load_suite, validate_suite

MINIMAL = {"audio": {"esc50": ["a"]}, "visual": {"ssv2": ["v"]}, "audio-visual": {"vggsounder": ["a", "v", "av"]}}
DEFAULT = {
    "audio": {"esc50": ["a"], "fsd50k": ["a"]},
    "visual": {"ssv2": ["v"], "k400": ["v"], "in1k": ["v"]},
    "audio-visual": {"vggsounder": ["a", "v", "av"], "audioset20k": ["a", "v", "av"]},
    "temporal": {"avsync": ["av"]},
    "spatial": {"vggss_box": ["av"]},
}


def _json(capsys, argv: list[str]) -> tuple[int, dict]:
    rc = cli.main(["--json", *argv])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1, out
    return rc, json.loads(out[0])


def _shape(suite) -> dict[str, dict[str, list[str]]]:
    return {g.capability: {e.benchmark: list(e.readings) for e in g.entries} for g in suite.groups}


def _raw(**over) -> dict:
    raw = {"schema_version": 1, "name": "mine", "description": "test suite",
           "groups": [{"capability": "audio", "entries": [{"benchmark": "esc50", "readings": ["a"]}]}]}
    raw.update(over)
    return raw


# ----------------------------------------------------------------------------- shipped files


def test_shipped_suites_validate():
    recs = {r["name"]: r for r in suites.list_suites()}
    assert {"minimal", "default"} <= set(recs)
    for name in ("minimal", "default"):
        assert recs[name]["origin"] == "shipped" and recs[name]["error"] is None and not recs[name]["shadows"]
        s = load_suite(name)
        assert s.origin == "shipped" and s.path == suites.shipped_dir() / f"{name}.yaml" and s.description
        assert "\n" not in s.description
    assert _shape(load_suite("minimal")) == MINIMAL
    assert _shape(load_suite("default")) == DEFAULT
    joins = {e.benchmark: e.join_reading for _c, e in load_suite("default").entries() if e.join}
    assert joins == {"vggsounder": "a+v+av"}
    assert {e.benchmark: e.join for _c, e in load_suite("minimal").entries()}["vggsounder"] is True


def test_default_readings_subset_of_templates():
    """Every reading of a shipped suite is one its folder's bridge template declares, and every join names a folder
    whose probe.py defines join() (what `report build --join` calls)."""
    for name in ("minimal", "default"):
        for _cap, e in load_suite(name).entries():
            folder = packaged_folder(e.benchmark)
            assert folder is not None and e.status == "packaged" and e.required
            assert set(e.readings) <= template_readings((folder / "bridge_template.py").read_text(encoding="utf-8")), e
            if e.join:
                assert "\ndef join(" in (folder / "probe.py").read_text(encoding="utf-8")
    assert suites.expected_store_keys("minimal") == ["esc50:a", "ssv2:v", "vggsounder:a", "vggsounder:v", "vggsounder:av",
                                                     "vggsounder:a+v+av"]
    assert [p["benchmark"] for p in suites.pairs("default")].count("audioset20k") == 3


# ----------------------------------------------------------------------------- validation


def test_unknown_key_is_E_SUITE(tmp_path):
    with pytest.raises(SuiteError) as ei:
        validate_suite(_raw(owner="me"))
    assert ei.value.code == "E_SUITE" and ei.value.exit_code == 1 and "owner" in str(ei.value)
    bad_entry = _raw(groups=[{"capability": "audio", "entries": [{"benchmark": "esc50", "readings": ["a"], "weight": 2}]}])
    with pytest.raises(SuiteError, match="weight"):
        validate_suite(bad_entry)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(_raw(owner="me")), encoding="utf-8")
    assert cli.main(["suite", "show", str(p)]) == 1
    assert cli.exit_code_for(SuiteError("x")) == (1, "E_SUITE")
    assert cli.exit_code_for(SuiteError("x", "E_EXISTS")) == (1, "E_EXISTS")


@pytest.mark.parametrize("over, needle", [
    ({"schema_version": 2}, "schema_version"),
    ({"name": "Bad Name"}, "name"),
    ({"description": ""}, "description"),
    ({"groups": []}, "groups"),
    ({"groups": [{"capability": "smell", "entries": [{"benchmark": "esc50", "readings": ["a"]}]}]}, "capability"),
    ({"groups": [{"capability": "audio", "entries": [{"benchmark": "esc50", "readings": ["v"]}]}]}, "not readings"),
    ({"groups": [{"capability": "audio", "entries": [{"benchmark": "esc50", "readings": ["a"], "join": True}]}]}, "join"),
    ({"groups": [{"capability": "audio", "entries": [{"benchmark": "nope_bench", "readings": ["a"]}]}]}, "does not resolve"),
    ({"groups": [{"capability": "audio", "entries": [{"benchmark": "esc50", "readings": ["a"]},
                                                     {"benchmark": "esc50", "readings": ["a"]}]}]}, "twice"),
    ({"groups": [{"capability": "audio", "entries": [{"benchmark": "esc50", "readings": ["a"], "status": "maybe"}]}]}, "status"),
])
def test_validation_refusals(over, needle):
    with pytest.raises(SuiteError, match=needle):
        validate_suite(_raw(**over))


def test_pending_entry_allowed(tmp_path):
    raw = _raw(groups=[{"capability": "audio", "entries": [{"benchmark": "esc50", "readings": ["a"]}]},
                       {"capability": "retrieval", "entries": [{"benchmark": "vggsound_retrieval", "readings": ["av"],
                                                                "status": "pending", "required": False}]}])
    s = validate_suite(raw)
    e = s.groups[1].entries[0]
    assert e.status == "pending" and not e.required and e.benchmark == "vggsound_retrieval"
    assert suites.expected_store_keys(s) == ["esc50:a"]
    assert [p["status"] for p in suites.pairs(s)] == ["packaged", "pending"]
    rows = suites.completeness(s, [], models=["m"])
    assert [r["status"] for r in rows] == ["missing", "pending"]


def test_alias_is_canonicalized():
    s = validate_suite(_raw(groups=[{"capability": "visual", "entries": [{"benchmark": "imagenet1k", "readings": ["v"]}]}]))
    assert s.groups[0].entries[0].benchmark == "in1k"


# ----------------------------------------------------------------------------- locations


def _project(tmp_path: Path, monkeypatch) -> Path:
    proj = tmp_path / "proj"
    (proj / "suites").mkdir(parents=True)
    (proj / "yavih.toml").write_text("", encoding="utf-8")
    monkeypatch.setenv("YAVIH_CONFIG", str(proj / "yavih.toml"))  # conftest pins it to a missing file
    return proj


def test_project_suite_shadows_shipped(tmp_path, monkeypatch):
    proj = _project(tmp_path, monkeypatch)
    assert suites.project_suites_dir() == (proj / "suites").resolve()
    text = (suites.shipped_dir() / "minimal.yaml").read_text(encoding="utf-8").replace("name: minimal", "name: default")
    (proj / "suites" / "default.yaml").write_text(text, encoding="utf-8")
    (proj / "suites" / "mine.yaml").write_text(yaml.safe_dump(_raw()), encoding="utf-8")
    recs = {r["name"]: r for r in suites.list_suites()}
    assert recs["default"]["origin"] == "project" and recs["default"]["shadows"]
    assert recs["minimal"]["origin"] == "shipped" and recs["mine"]["origin"] == "project" and not recs["mine"]["shadows"]
    s = load_suite("default")
    assert s.origin == "project" and _shape(s) == MINIMAL
    # YAVIH_SUITES beats the directory next to yavih.toml
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("YAVIH_SUITES", str(other))
    assert suites.project_suites_dir() == other and load_suite("default").origin == "shipped"


def test_project_suite_name_must_match_the_file(tmp_path, monkeypatch):
    proj = _project(tmp_path, monkeypatch)
    (proj / "suites" / "x.yaml").write_text(yaml.safe_dump(_raw()), encoding="utf-8")
    with pytest.raises(SuiteError, match="differs from the file name"):
        load_suite("x")
    assert {r["name"]: r for r in suites.list_suites()}["x"]["error"]
    with pytest.raises(SuiteError, match="unknown suite"):
        load_suite("nope")


# ----------------------------------------------------------------------------- yavih suite


def test_suite_list_json_marks_origin(tmp_path, monkeypatch, capsys):
    proj = _project(tmp_path, monkeypatch)
    (proj / "suites" / "mine.yaml").write_text(yaml.safe_dump(_raw()), encoding="utf-8")
    rc, env = _json(capsys, ["suite", "list"])
    assert rc == 0 and env["status"] == "ok"
    origins = {r["name"]: r["origin"] for r in env["data"]}
    assert origins == {"default": "shipped", "minimal": "shipped", "mine": "project"}
    rc, env = _json(capsys, ["suite", "show", "minimal"])
    assert rc == 0 and env["data"]["name"] == "minimal" and env["data"]["groups"][2]["entries"][0]["join_reading"] == "a+v+av"
    rc, env = _json(capsys, ["suite", "show", "nope"])
    assert rc == 1 and env["errors"][0]["code"] == "E_SUITE"


def test_suite_new_copies_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    proj = _project(tmp_path, monkeypatch)
    rc, env = _json(capsys, ["suite", "new", "mine", "--from", "minimal"])
    assert rc == 0, env
    path = Path(env["data"]["path"])
    assert path == (proj / "suites" / "mine.yaml").resolve()
    text = path.read_text(encoding="utf-8")
    assert "name: mine" in text and "# audio:" in text  # comments survive the copy
    s = load_suite("mine")
    assert s.origin == "project" and _shape(s) == MINIMAL
    rc, env = _json(capsys, ["suite", "new", "mine"])
    assert rc == 1 and env["errors"][0]["code"] == "E_EXISTS"
    rc, env = _json(capsys, ["suite", "new", "mine", "--force", "--out", str(tmp_path / "elsewhere") + "/"])
    assert rc == 0 and Path(env["data"]["path"]) == (tmp_path / "elsewhere" / "mine.yaml").resolve()
    assert _shape(load_suite(env["data"]["path"])) == DEFAULT


def test_suite_missing_with_synthetic_runs(runs_dir, capsys):
    write_result(runs_dir, make_result(bench="esc50", model="m1", reading="a"))
    write_result(runs_dir, make_result(bench="vggsounder", model="m1", reading="v"))
    rc, env = _json(capsys, ["suite", "missing", "minimal", "--model", "m1"])
    assert rc == 0 and env["status"] == "ok"
    status = {(r["benchmark"], r["reading"]): r["status"] for r in env["data"]["rows"]}
    assert status == {("esc50", "a"): "done", ("ssv2", "v"): "missing", ("vggsounder", "a"): "missing",
                      ("vggsounder", "v"): "done", ("vggsounder", "av"): "missing", ("vggsounder", "a+v+av"): "join-missing"}
    hint = next(r["hint"] for r in env["data"]["rows"] if r["reading"] == "a+v+av")
    assert hint == (f"yavih report build {runs_dir}/vggsounder/m1 --join vggsounder a={runs_dir}/vggsounder/m1/a "
                    f"v={runs_dir}/vggsounder/m1/v av={runs_dir}/vggsounder/m1/av")
    assert env["data"]["counts"] == {"done": 2, "join-missing": 1, "missing": 3}
    # a stored join entry of the model makes the join row done
    from yavih.report import store

    results = Path(env["data"]["results_root"])
    store.write_join(results, "m1", benchmark="vggsounder", joined={"acc": 0.5},
                     refs={r: make_result(bench="vggsounder", model="m1", reading=r) for r in ("a", "v", "av")})
    rc, env = _json(capsys, ["suite", "missing", "minimal", str(runs_dir), "--model", "m1"])
    assert {(r["benchmark"], r["reading"]): r["status"] for r in env["data"]["rows"]}[("vggsounder", "a+v+av")] == "done"
    # text mode prints the command next to a missing join
    assert cli.main(["suite", "missing", "--model", "m2", "minimal", str(runs_dir)]) == 0
    out = capsys.readouterr().out
    assert "esc50:a" in out and "missing" in out and "--join vggsounder" in out


# ----------------------------------------------------------------------------- bench list / report build


def test_bench_list_suite_filters(capsys):
    rc, env = _json(capsys, ["bench", "list", "--suite", "minimal"])
    assert rc == 0
    rows = env["data"]
    assert [r["name"] for r in rows] == ["esc50", "ssv2", "vggsounder"]
    assert [(r["capability"], r["readings"]) for r in rows] == [("audio", ["a"]), ("visual", ["v"]),
                                                               ("audio-visual", ["a", "v", "av", "a+v+av"])]
    assert cli.main(["bench", "list", "--suite", "default"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("suite default (shipped)") and "capability" in out and "vggss_box" in out and "nyuv2_depth" not in out
    rc, env = _json(capsys, ["bench", "list", "--suite", "nope"])
    assert rc == 1 and env["errors"][0]["code"] == "E_SUITE"


def test_report_suite_appendix_lists_missing_pairs(runs_dir, capsys):
    write_result(runs_dir, make_result(bench="esc50", model="m1", reading="a"))
    write_result(runs_dir, make_result(bench="ssv2", model="m2", reading="v"))
    rep = report([str(runs_dir)], fmt="md", suite="minimal", write=False)
    assert rep.suite["name"] == "minimal"
    status = {(r["model"], r["benchmark"], r["reading"]): r["status"] for r in rep.suite["rows"]}
    assert status[("m1", "esc50", "a")] == "done" and status[("m1", "ssv2", "v")] == "missing"
    assert status[("m2", "ssv2", "v")] == "done" and status[("m2", "esc50", "a")] == "missing"
    assert status[("m1", "vggsounder", "a+v+av")] == "join-missing"
    assert len(rep.suite["rows"]) == 2 * 6
    md = rep.text
    assert "## Suite completeness: minimal" in md and "| capability | benchmark | reading | m1 | m2 |" in md
    assert "| audio | esc50 | a | done | missing |" in md and "a+v+av (join)" in md and "--join vggsounder" in md
    js = json.loads(report([str(runs_dir)], fmt="json", suite="minimal", write=False).text)
    assert js["suite"]["rows"] == rep.suite["rows"]
    csv_text = report([str(runs_dir)], fmt="csv", suite="minimal", write=False).text
    assert csv_text.splitlines()[0].startswith("benchmark,") and "Suite completeness" not in csv_text
    assert report([str(runs_dir)], write=False).suite is None
    # CLI: --suite on report build; an unknown suite is exit 1 before anything is indexed
    rc, env = _json(capsys, ["report", "build", str(runs_dir), "--suite", "minimal", "--fmt", "json"])
    assert rc == 0 and env["data"]["suite"]["counts"]["done"] == 2
    rc, env = _json(capsys, ["report", "build", str(runs_dir), "--suite", "nope"])
    assert rc == 1 and env["errors"][0]["code"] == "E_SUITE"
