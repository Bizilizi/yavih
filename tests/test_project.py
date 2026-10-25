"""The project folder ``yavih-bench/`` (``yavih.project``, ``yavih.config``): ``yavih init``, discovery up to the git
root, the ``[project]`` table, launchers by name per site, ``plan init`` / ``bridge new`` / suites / ``bench new``
defaults inside a project, and the layout and version findings of ``yavih doctor``."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from yavih import cli, config, suites
from yavih.plan.exec import ExecError, resolve_launcher
from yavih.plan.matrix import load_matrix
from yavih.project import ProjectError, check_requirement, doctor_block, init_project, layout_report


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A git repository (a ``.git`` directory is enough for discovery) with an isolated user config, cwd at its root."""
    xdg = tmp_path / "xdg"
    (xdg / "yavih").mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.delenv("YAVIH_CONFIG", raising=False)
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    monkeypatch.chdir(root)
    yield root
    config.set_cli()


def _json(capsys, argv) -> tuple[int, dict]:
    rc = cli.main(["--json", *argv])
    out = capsys.readouterr().out.strip().splitlines()
    return rc, json.loads(out[-1])


# ----------------------------------------------------------------------------- init and discovery


def test_init_writes_the_folder_and_never_the_user_config(repo, capsys):
    rc, env = _json(capsys, ["init", "--site", "c1", "--hosts", "*", "--launcher", "/envs/m/bin/python"])
    assert rc == 0, env["errors"]
    folder = repo / "yavih-bench"
    for name in ("yavih.toml", "README.md", ".gitignore", "bridges/.gitkeep", "matrices/.gitkeep", "suites/.gitkeep",
                 "benchmarks/.gitkeep", "tools/.gitkeep"):
        assert (folder / name).is_file(), name
    assert env["paths"]["dir"] == str(folder)
    # only what yavih itself creates inside the folder; results/ is committed, so it is not listed
    assert (folder / ".gitignore").read_text(encoding="utf-8") == "runs/\nlogs/\n__pycache__/\n"
    user = Path(config.user_config_path()[0])
    assert not user.exists() and env["data"]["user_config"] == str(user)  # the user file holds personal overrides only
    s = config.load(refresh=True)
    assert s.project_dir() == folder and s.project_file().how == "discovered"
    assert s.site == "c1" and s.launcher() == "/envs/m/bin/python" and s.launchers() == {}
    assert s.project_requirement() and s.project_requirement().startswith(">=")
    assert s.project_path("suites") == folder / "suites" and s.project_path("benchmarks") == folder / "benchmarks"
    # runs and results default to the project folder (absolute), and init reports them
    assert s.get("runs", "root") is None and s.runs_root() == folder / "runs" and s.results_root() == folder / "results"
    assert env["data"]["runs_root"] == str(folder / "runs") and env["data"]["results_root"] == str(folder / "results")
    shown = s.show()
    assert shown["project"]["dir"] == str(folder) and shown["values"]["launcher"]["value"] == "/envs/m/bin/python"
    assert shown["values"]["runs.root"] == {"value": str(folder / "runs"), "source": "default (project)"}
    assert shown["values"]["results.root"] == {"value": str(folder / "results"), "source": "default (project)"}
    assert cli.main(["init"]) == 1  # E_EXISTS without --force
    capsys.readouterr()
    rc, env = _json(capsys, ["config", "get", "runs.root"])
    assert rc == 0 and env["data"] == {"key": "runs.root", "value": str(folder / "runs"), "source": "default (project)"}


def test_default_roots_outside_a_project_are_absolute(repo, monkeypatch, tmp_path):
    """Without a project the fallback is ``runs`` under the current directory, absolute, and ``results`` next to it;
    ``config show`` says so."""
    s = config.load(refresh=True)
    assert s.project_dir() is None
    assert s.runs_root() == repo / "runs" and s.runs_root().is_absolute() and s.results_root() == repo / "results"
    assert s.show()["values"]["runs.root"]["source"] == "default (./runs)"
    assert s.show()["values"]["results.root"]["source"] == "default (<runs root>/../results)"
    # a configured runs root without a results root: results next to it (no project to anchor them)
    monkeypatch.setenv("YAVIH_RUNS", str(tmp_path / "scratch" / "runs"))
    s = config.load(refresh=True)
    assert s.runs_root() == tmp_path / "scratch" / "runs" and s.results_root() == tmp_path / "scratch" / "results"
    # inside a project a configured runs root leaves results with the project
    monkeypatch.delenv("YAVIH_RUNS")
    init_project()
    (repo / "yavih-bench" / "yavih.toml").write_text('[runs]\nroot = "/scratch/me/runs"\n', encoding="utf-8")
    s = config.load(refresh=True)
    assert s.runs_root() == Path("/scratch/me/runs") and s.results_root() == repo / "yavih-bench" / "results"
    assert s.runs_root_source().startswith("file:") and s.results_root_source() == "default (project)"
    assert s.results_root("/elsewhere/runs") == repo / "yavih-bench" / "results"  # a probe's run root changes nothing in a project


def test_matrix_out_null_is_the_config_runs_root(repo, capsys):
    """A matrix without ``out`` lands where ``config get runs.root`` points: the project default, absolute."""
    init_project()
    m = repo / "yavih-bench" / "matrices" / "m.yaml"
    m.write_text("schema_version: 1\nbridges:\n  a: {script: s.py, benchmark: k400, profile: p.json}\n", encoding="utf-8")
    config.load(refresh=True)
    rc, env = _json(capsys, ["config", "get", "runs.root"])
    assert rc == 0 and load_matrix(str(m)).out == env["data"]["value"] == str(repo / "yavih-bench" / "runs")


def test_discovery_prefers_the_folder_and_stops_at_the_git_root(repo, tmp_path):
    (tmp_path / "yavih.toml").write_text('[runs]\nroot = "/above-the-repo"\n', encoding="utf-8")  # above .git: never read
    sub = repo / "experiments" / "x"
    sub.mkdir(parents=True)
    assert config._discover_project(sub) is None
    (repo / "yavih.toml").write_text('[runs]\nroot = "/root-file"\n', encoding="utf-8")
    assert config._discover_project(sub) == repo / "yavih.toml"
    (repo / "yavih-bench").mkdir()
    (repo / "yavih-bench" / "yavih.toml").write_text('[runs]\nroot = "/folder-file"\n', encoding="utf-8")
    assert config._discover_project(sub) == repo / "yavih-bench" / "yavih.toml"
    assert config.load(refresh=True, cwd=sub).runs_root() == Path("/folder-file")
    assert config.git_root(sub) == repo and config.git_root(tmp_path) is None


def test_second_init_is_refused(repo, capsys, monkeypatch):
    init_project()
    with pytest.raises(ProjectError) as ei:
        init_project()
    assert ei.value.code == "E_EXISTS"
    (repo / "yavih-bench" / "README.md").write_text("edited\n", encoding="utf-8")
    toml_before = (repo / "yavih-bench" / "yavih.toml").read_text(encoding="utf-8")
    rec = init_project(force=True)
    assert (repo / "yavih-bench" / "yavih.toml").read_text(encoding="utf-8") == toml_before
    assert str(repo / "yavih-bench" / "yavih.toml") in rec["kept"] and (repo / "yavih-bench" / "README.md").read_text(encoding="utf-8") != "edited\n"
    sub = repo / "experiments" / "wave2"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    rc, env = _json(capsys, ["init"])
    assert rc == 1 and env["errors"][0]["code"] == "E_PROJECT_EXISTS" and "subprojects" in env["errors"][0]["message"]
    assert not (sub / "yavih-bench").exists()
    rc, env = _json(capsys, ["init", "--site", "bad id"])
    assert rc == 1 and env["errors"][0]["code"] == "E_ARG"


# ----------------------------------------------------------------------------- [project] and launchers


def _project(repo: Path, text: str) -> Path:
    folder = repo / "yavih-bench"
    folder.mkdir(exist_ok=True)
    p = folder / "yavih.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_project_table_relocations_requirement_and_subprojects(repo):
    _project(repo, '[project]\nyavih = ">=1,<2"\nbridges = "../src/mymodel/yavih"\nsubprojects = ["../experiments/old"]\n')
    s = config.load(refresh=True)
    assert s.project_requirement() == ">=1,<2"
    assert s.project_path("bridges") == (repo / "src" / "mymodel" / "yavih").resolve()
    assert s.project_path("matrices") == repo / "yavih-bench" / "matrices"
    assert s.subprojects() == [(repo / "experiments" / "old").resolve()]
    assert s.show()["project"]["bridges"] == str((repo / "src" / "mymodel" / "yavih").resolve())
    with pytest.raises(ValueError):
        s.project_path("tools")


@pytest.mark.parametrize("text, match", [
    ('[project]\nbridgez = "x"\n', "unknown key \\[project\\] bridgez, did you mean 'bridges'"),
    ('[project]\nyavih = 3\n', "version requirement string"),
    ('[project]\nsubprojects = "a"\n', "list of directory paths"),
    ('[site.c]\nhosts = ["*"]\n[site.c.project]\nyavih = ">=1"\n', "not a site table"),
    ('launcher = 3\n', "launcher must be a non-empty string"),
    ('[launchers]\n"bad name" = "python"\n', "launcher name matches"),
    ('[launchers]\nvjepa = ""\n', "must be a non-empty string"),
])
def test_project_and_launcher_schema_errors(repo, text, match):
    _project(repo, text)
    with pytest.raises(config.ConfigError, match=match):
        config.load(refresh=True)


def test_project_table_belongs_to_the_project_file(repo, tmp_path):
    user = Path(config.user_config_path()[0])
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text('[project]\nyavih = ">=1"\n', encoding="utf-8")
    with pytest.raises(config.ConfigError, match="belongs to the project file"):
        config.load(refresh=True)
    assert config.load(refresh=True, lenient=True).project_requirement() is None


def test_launchers_by_name_per_site(repo, monkeypatch):
    _project(repo, 'launcher = "python"\n[launchers]\nvjepa = "~/envs/vjepa/bin/python"\n'
                   '[site.a]\nhosts = ["nomatch-*"]\nlauncher = "/a/bin/python"\n[site.a.launchers]\nvjepa = "/a/envs/vjepa/bin/python"\n'
                   '[site.b]\nhosts = ["nomatch-*"]\n[site.b.launchers]\nhf = "apptainer exec --nv img.sif python"\n')
    s = config.load(refresh=True)
    assert s.launcher() == "python" and s.launcher("vjepa") == str(Path("~/envs/vjepa/bin/python").expanduser())
    monkeypatch.setenv("YAVIH_SITE", "a")
    s = config.load(refresh=True)
    assert s.launcher() == "/a/bin/python" and s.launcher("vjepa") == "/a/envs/vjepa/bin/python" and s.launcher("hf") is None
    assert resolve_launcher(None) == "/a/bin/python" and resolve_launcher("vjepa") == "/a/envs/vjepa/bin/python"
    assert resolve_launcher("/x/bin/python") == "/x/bin/python" and resolve_launcher("conda run -n m python") == "conda run -n m python"
    assert resolve_launcher(sys.executable) == sys.executable
    with pytest.raises(ExecError, match="neither a \\[launchers\\] name .* nor a command on PATH"):
        resolve_launcher("no-such-launcher-xyz")
    monkeypatch.setenv("YAVIH_SITE", "b")
    s = config.load(refresh=True)
    assert s.launcher() == "python" and s.launchers() == {"hf": "apptainer exec --nv img.sif python", "vjepa": str(Path("~/envs/vjepa/bin/python").expanduser())}
    monkeypatch.setenv("YAVIH_LAUNCHER", "/env/bin/python")
    assert config.load(refresh=True).launcher() == "/env/bin/python"


# ----------------------------------------------------------------------------- defaults inside a project


def test_plan_init_bridge_new_suites_and_bench_new_default_into_the_project(repo, runs_dir, capsys, monkeypatch):
    init_project()
    folder = repo / "yavih-bench"
    work = repo / "experiments" / "wave1"
    work.mkdir(parents=True)
    monkeypatch.chdir(work)
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy"])
    assert rc == 0, env["errors"]
    m = folder / "matrices" / "toy.yaml"
    assert env["paths"]["matrix"] == str(m) and env["data"]["project"] == str(folder)
    assert [w["code"] for w in env["warnings"]] == ["W_LAUNCHER_UNSET"]
    assert sorted(p.name for p in (folder / "bridges").glob("*.py")) == ["toy_esc50.py", "toy_ssv2.py", "toy_vggsounder.py"]
    raw = yaml.safe_load(m.read_text(encoding="utf-8"))
    assert raw["defaults"]["launcher"] is None
    assert raw["bridges"]["toy-esc50-a"] == {"script": "../bridges/toy_esc50.py", "benchmark": "esc50",
                                            "profile": "toy.profiles/toy-esc50-a.profile.json"}
    matrix = load_matrix(str(m))
    assert matrix.root == str(folder / "matrices") and all(e.launcher is None for e in matrix.entries)
    # a configured launcher silences the warning
    (folder / "yavih.toml").write_text('[site.here]\nhosts = ["*"]\nlauncher = "/envs/toy/bin/python"\n', encoding="utf-8")
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy", "--force"])
    assert rc == 0 and env["warnings"] == [] and {b["action"] for b in env["data"]["bridges"]} == {"kept"}
    rc, env = _json(capsys, ["bridge", "new", "esc50", "--model", "other"])
    assert rc == 0 and env["data"]["path"] == str(folder / "bridges" / "other_esc50.py")
    assert suites.project_suites_dir() == folder / "suites"
    rc, env = _json(capsys, ["suite", "new", "mine"])
    assert rc == 0 and Path(env["data"]["path"]).parent == folder / "suites"
    from yavih.scaffold import _dest

    assert _dest(None) == folder / "benchmarks" and _dest("/elsewhere") == Path("/elsewhere")
    assert config.project_benchmarks_dir() == folder / "benchmarks"


# ----------------------------------------------------------------------------- doctor: layout and version


def test_init_of_a_listed_subproject(repo, capsys, monkeypatch, runs_dir):
    """A directory the governing project lists in ``[project] subprojects`` (relative to that project file) may get its
    own project folder; discovery inside it then finds the subproject's file, and doctor warns from neither side."""
    init_project()
    root_file = repo / "yavih-bench" / "yavih.toml"
    sub = repo / "services" / "asr"
    sibling = repo / "services" / "tts"
    for d in (sub, sibling):
        d.mkdir(parents=True)
    (repo / "yavih-bench" / "matrices" / "matrix.old.yaml").write_text("schema_version: 1\n", encoding="utf-8")  # the root's: no stray from the subproject
    root_file.write_text('[project]\nsubprojects = ["../services/asr", "../services/tts"]\n', encoding="utf-8")
    monkeypatch.chdir(sub)
    rec = init_project()
    assert rec["file"] == str(sub / "yavih-bench" / "yavih.toml") and rec["existing"] == str(root_file)
    assert config._discover_project(sub) == sub / "yavih-bench" / "yavih.toml"
    assert config._discover_project(sub / "deeper") == sub / "yavih-bench" / "yavih.toml"
    (sibling / "yavih.toml").write_text("[runs]\nroot = \"r\"\n", encoding="utf-8")
    s = config.load(refresh=True, cwd=sub)
    assert s.project_dir() == sub / "yavih-bench"
    data, warnings, _n = doctor_block(s, cwd=sub)
    assert data["nested"] == [] and data["stray"] == [] and warnings == []
    monkeypatch.chdir(repo)
    data, warnings, _n = doctor_block(config.load(refresh=True))
    assert data["nested"] == [] and warnings == []
    # an unlisted directory is still refused, with the same message
    other = repo / "services" / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    with pytest.raises(ProjectError) as ei:
        init_project()
    assert ei.value.code == "E_PROJECT_EXISTS" and "subprojects" in str(ei.value)
    # from inside the subproject, an unlisted project file elsewhere is still nested
    (other / "yavih.toml").write_text("[runs]\nroot = \"r\"\n", encoding="utf-8")
    data, warnings, _n = doctor_block(config.load(refresh=True, cwd=sub), cwd=sub)
    assert data["nested"] == [str(other / "yavih.toml")] and [c for c, _m, _w in warnings] == ["W_PROJECT_NESTED"]


def test_layout_report_and_doctor_findings(repo, runs_dir):
    init_project()
    s = config.load(refresh=True)
    rep = layout_report(s)
    assert rep["dir"] == str(repo / "yavih-bench") and rep["root"] == str(repo) and rep["nested"] == [] and rep["stray"] == []
    (repo / "experiments" / "old").mkdir(parents=True)
    (repo / "experiments" / "old" / "yavih.toml").write_text("[runs]\nroot = \"r\"\n", encoding="utf-8")
    (repo / "experiments" / "old" / "matrix.cluster_b.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (repo / "yavih-bench" / "matrices").mkdir(exist_ok=True)
    (repo / "yavih-bench" / "matrices" / "matrix.yaml").write_text("schema_version: 1\n", encoding="utf-8")  # inside: fine
    venv = repo / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "matrix.yaml").write_text("x\n", encoding="utf-8")  # a dot-directory: never scanned
    data, warnings, notes = doctor_block(config.load(refresh=True))
    assert data["nested"] == [str(repo / "experiments" / "old" / "yavih.toml")]
    assert data["stray"] == [str(repo / "experiments" / "old" / "matrix.cluster_b.yaml")]
    assert [c for c, _m, _w in warnings] == ["W_PROJECT_NESTED", "W_PROJECT_STRAY"]
    assert data["requirement"]["spec"] and data["requirement"]["ok"] is True
    # listed as a subproject, the nested folder and its matrix are fine
    (repo / "yavih-bench" / "yavih.toml").write_text('[project]\nsubprojects = ["../experiments/old"]\n', encoding="utf-8")
    data, warnings, _n = doctor_block(config.load(refresh=True))
    assert data["nested"] == [] and data["stray"] == [] and warnings == []
    # a requirement the installed version does not satisfy
    (repo / "yavih-bench" / "yavih.toml").write_text('[project]\nyavih = ">=99"\nsubprojects = ["../experiments/old"]\n', encoding="utf-8")
    data, warnings, _n = doctor_block(config.load(refresh=True))
    assert data["requirement"]["ok"] is False and [c for c, _m, _w in warnings] == ["W_PROJECT_VERSION"]


def test_doctor_skips_the_default_roots_and_warns_on_a_large_runs_root(repo, monkeypatch):
    """The layout scan never enters the runs and results roots in effect, configured or the project defaults (a
    results file or a plan's matrix copy there is not a stray); ``W_RUNS_SIZE`` when the project-default runs root
    exceeds ``RUNS_SIZE_WARN_GB``, never for a runs root configured elsewhere."""
    from yavih import project as P

    init_project()
    s = config.load(refresh=True)
    runs, results = s.runs_root(), s.results_root()
    assert runs == repo / "yavih-bench" / "runs" and results == repo / "yavih-bench" / "results"
    for d in (runs / "_plans", results / "m"):
        d.mkdir(parents=True)
        (d / "matrix.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (results / "m" / "results_1.json").write_text("{}", encoding="utf-8")
    data, warnings, _n = doctor_block(s)
    assert data["stray"] == [] and warnings == []
    # the same roots relocated outside the project folder, by configuration: still skipped
    (repo / "yavih-bench" / "yavih.toml").write_text('[runs]\nroot = "../scratch/runs"\n[results]\nroot = "../scratch/results"\n', encoding="utf-8")
    s = config.load(refresh=True)
    for d in (s.runs_root() / "_plans", s.results_root()):
        d.mkdir(parents=True)
        (d / "matrix.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (repo / "scratch" / "matrix.other.yaml").write_text("schema_version: 1\n", encoding="utf-8")  # outside both: a stray
    data, warnings, _n = doctor_block(s)
    assert data["stray"] == [str(repo / "scratch" / "matrix.other.yaml")] and [c for c, _m, _w in warnings] == ["W_PROJECT_STRAY"]
    # size: the project default over the threshold warns and says how to move it; a configured root elsewhere never does
    monkeypatch.setattr(P, "runs_size", lambda root, stop_at=None: int(P.RUNS_SIZE_WARN_GB * 1e9) + 1)
    data, warnings, _n = doctor_block(s)
    assert [c for c, _m, _w in warnings] == ["W_PROJECT_STRAY"]
    import shutil

    shutil.rmtree(repo / "scratch")
    (repo / "yavih-bench" / "yavih.toml").write_text("", encoding="utf-8")
    s = config.load(refresh=True)
    data, warnings, _n = doctor_block(s)
    assert [c for c, _m, _w in warnings] == ["W_RUNS_SIZE"]
    code, msg, where = warnings[0]
    assert where == str(runs) and "[runs] root" in msg and "mv " in msg and str(results) in msg
    monkeypatch.setattr(P, "runs_size", lambda root, stop_at=None: int(P.RUNS_SIZE_WARN_GB * 1e9))
    assert doctor_block(config.load(refresh=True))[1] == []


def test_runs_size_walks_files_and_stops_early(tmp_path):
    from yavih.project import runs_size

    root = tmp_path / "runs"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "f1").write_bytes(b"x" * 1000)
    (root / "a" / "f2").write_bytes(b"y" * 500)
    (root / "link").symlink_to(root / "a")  # symlinks are not followed, so nothing is counted twice
    assert runs_size(root) == 1500
    assert runs_size(root, stop_at=100) >= 500  # stops once over the threshold, never below it
    assert runs_size(tmp_path / "absent") == 0


def test_sibling_bare_projects_outside_a_repository_are_not_nested(repo, runs_dir, monkeypatch, tmp_path):
    """Outside any repository the scan covers a bare ``yavih.toml`` project's own directory, so a sibling directory
    with its own project file and matrices is another project, never ``W_PROJECT_NESTED`` / ``W_PROJECT_STRAY``; a
    ``yavih-bench/`` folder's scan covers the directory it sits in (its would-be repository root)."""
    work = tmp_path / "work"
    for name in ("a", "b"):
        (work / name).mkdir(parents=True)
        (work / name / "yavih.toml").write_text("", encoding="utf-8")
        (work / name / "matrix.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    assert config.git_root(work / "a") is None
    monkeypatch.chdir(work / "a")
    s = config.load(refresh=True)
    rep = layout_report(s)
    assert rep["dir"] == str(work / "a") and rep["root"] == str(work / "a") and rep["nested"] == [] and rep["stray"] == []
    _d, warnings, _n = doctor_block(s)
    assert [c for c, _m, _w in warnings if c.startswith("W_PROJECT")] == []
    # a yavih-bench/ folder: the directory above it is the root, a sibling project inside it is nested
    (work / "c" / "yavih-bench").mkdir(parents=True)
    (work / "c" / "yavih-bench" / "yavih.toml").write_text("", encoding="utf-8")
    (work / "c" / "other").mkdir()
    (work / "c" / "other" / "yavih.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(work / "c")
    rep = layout_report(config.load(refresh=True))
    assert rep["root"] == str(work / "c") and rep["nested"] == [str(work / "c" / "other" / "yavih.toml")]


def test_doctor_without_a_project_notes_init(repo, runs_dir):
    data, warnings, notes = doctor_block(config.load(refresh=True))
    assert data["dir"] is None and warnings == [] and any("yavih init" in n for n in notes)


def test_doctor_cli_carries_the_project_block(repo, runs_dir, capsys, monkeypatch):
    init_project()
    (repo / "stray").mkdir()
    (repo / "stray" / "matrix.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    rc, env = _json(capsys, ["doctor"])
    assert rc == 0 and env["data"]["project"]["dir"] == str(repo / "yavih-bench")
    assert [w["code"] for w in env["warnings"] if w["code"].startswith("W_PROJECT")] == ["W_PROJECT_STRAY"]
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "  project     " in out and "STRAY matrix" in out


@pytest.mark.parametrize("version, spec, ok, dev", [
    ("1.2.0", ">=1,<2", True, False),
    ("2.0.0", ">=1,<2", False, False),
    ("1.0.0", "==1.0.0", True, False),
    ("0.0.1.dev62+g3e66da1.d20261012", ">=0,<1", True, True),
    ("1.0.1.dev3+gabc", ">=1,<2", True, True),
    ("1.5.0", "~=1.4", True, False),
    ("2.1.0", "~=1.4", False, False),
])
def test_check_requirement(version, spec, ok, dev, monkeypatch):
    rec = check_requirement(spec, version)
    assert rec["ok"] is ok and rec["dev"] is dev, rec
    # the release-only reading (no `packaging`) agrees on these
    monkeypatch.setitem(sys.modules, "packaging.specifiers", None)
    monkeypatch.setitem(sys.modules, "packaging.version", None)
    rec = check_requirement(spec, version)
    assert rec["ok"] is ok and "release-only" in rec["reason"], rec


def test_check_requirement_edge_cases():
    assert check_requirement(None, "1.0.0") == {"spec": None, "version": "1.0.0", "ok": None, "dev": False, "reason": None}
    rec = check_requirement("not a spec", "1.0.0")
    assert rec["ok"] is None and rec["reason"]


# ----------------------------------------------------------------------------- review fixes


def test_exec_resolves_the_launcher_from_the_unit_cwd_before_running(repo, tmp_path, monkeypatch):
    """A ``[launchers]`` name resolves through the project discovered from the UNIT's cwd (the matrix directory), wherever
    ``exec`` runs; a name the site does not define is ``misconfigured`` before any running record exists."""
    from test_plan_helpers import fake_unit, python_launcher, write_fake_script
    from yavih.plan import exec as X
    from yavih.plan import units as U
    from yavih.plan.units import write_plan

    init_project()
    (repo / "yavih-bench" / "yavih.toml").write_text(f'[launchers]\nvjepa = {json.dumps(python_launcher())}\n', encoding="utf-8")
    ws = repo / "ws"
    ws.mkdir()
    script = write_fake_script(ws)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # a job started from $HOME: no project below the cwd
    assert config.load(refresh=True).project_dir() is None
    u = fake_unit(ws, script, launcher="vjepa")
    plan = str(ws / "plan.jsonl")
    write_plan([u], plan)
    last = X.exec_unit(plan, index=0)
    assert last["status"] == "done", last
    bad = fake_unit(ws, script, launcher="no-such-launcher-xyz", shard=(1, 2))
    write_plan([bad], plan)
    last = X.exec_unit(plan, index=0)
    assert last["status"] == "misconfigured" and last["exit_code"] == 1 and "launcher" in last["message"]
    assert not os.path.exists(U.resolve_path(bad, bad.done))
    assert not os.path.exists(os.path.join(U.resolve_path(bad, bad.out), "logs", f"running.{bad.id}.json"))


def test_plan_init_out_elsewhere_keeps_the_project_bridges(repo, runs_dir, capsys, tmp_path):
    init_project()
    out = tmp_path / "elsewhere" / "toy.yaml"
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy", "--out", str(out)])
    assert rc == 0, env["errors"]
    assert (repo / "yavih-bench" / "bridges" / "toy_esc50.py").is_file() and not (tmp_path / "bridges").exists()
    raw = yaml.safe_load(out.read_text(encoding="utf-8"))
    script = raw["bridges"]["toy-esc50-a"]["script"]
    assert (out.parent / script).resolve() == (repo / "yavih-bench" / "bridges" / "toy_esc50.py").resolve()
    assert raw["bridges"]["toy-esc50-a"]["profile"] == "toy.profiles/toy-esc50-a.profile.json"


def test_launcher_anywhere_in_the_configuration_silences_the_unset_warning(repo, runs_dir, capsys, monkeypatch):
    """``plan init`` warns ``W_LAUNCHER_UNSET`` only when no DEFAULT ``launcher`` is defined anywhere: one under another
    site's table (a login node that is active on no site) counts as configured. A ``[launchers]`` name does not: the
    skeleton's ``launcher: null`` resolves to the default ``launcher`` only, never to a named entry."""
    init_project()
    toml = repo / "yavih-bench" / "yavih.toml"
    toml.write_text("", encoding="utf-8")
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy"])
    assert rc == 0 and [w["code"] for w in env["warnings"]] == ["W_LAUNCHER_UNSET"] and not env["data"]["launcher_configured"]
    assert "anywhere in the configuration" in env["warnings"][0]["message"]
    # two sites, this host active on neither: site a's default launcher is that cluster's, not missing
    toml.write_text('[site.a]\nhosts = ["no-such-host-a*"]\nlauncher = "/a/bin/python"\n'
                    '[site.b]\nhosts = ["no-such-host-b*"]\n[site.b.launchers]\nvjepa = "/b/vjepa/bin/python"\n', encoding="utf-8")
    s = config.load(refresh=True)
    assert s.site is None and s.launcher() is None and s.launchers() == {} and s.any_site_launcher()
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy", "--force"])
    assert rc == 0 and env["warnings"] == [] and env["data"]["launcher_configured"] is True
    # named entries alone, in an inactive site table or at the top level: `null` resolves to none of them, so it warns
    toml.write_text('[site.b]\nhosts = ["no-such-host-b*"]\n[site.b.launchers]\nvjepa = "/b/vjepa/bin/python"\n', encoding="utf-8")
    assert not config.load(refresh=True).any_site_launcher()
    toml.write_text('[launchers]\nvjepa = "/envs/vjepa/bin/python"\n', encoding="utf-8")
    s = config.load(refresh=True)
    assert s.launchers() == {"vjepa": "/envs/vjepa/bin/python"} and resolve_launcher(None) is None and not s.any_site_launcher()
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy", "--force"])
    assert rc == 0 and [w["code"] for w in env["warnings"]] == ["W_LAUNCHER_UNSET"] and not env["data"]["launcher_configured"]
    # naming the entry on the command line is a launcher: no warning
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy", "--force", "--launcher", "vjepa"])
    assert rc == 0 and env["warnings"] == [] and env["data"]["launcher"] == "vjepa"
    toml.write_text('[site.b]\nhosts = ["no-such-host-b*"]\n[site.b.data]\nroot = "/d"\n', encoding="utf-8")
    assert not config.load(refresh=True).any_site_launcher()  # a site table without a launcher defines none


def test_init_launcher_without_site_and_hosts_without_site(repo, capsys, tmp_path):
    rc, env = _json(capsys, ["init", "--hosts", "*"])
    assert rc == 1 and env["errors"][0]["code"] == "E_ARG" and "--site" in env["errors"][0]["message"]
    rc, env = _json(capsys, ["init", "--launcher", "/envs/m/bin/python"])
    assert rc == 0 and config.load(refresh=True).launcher() == "/envs/m/bin/python"
    # --dir: the new project is validated from its own directory, not from the cwd
    other = tmp_path / "other"
    (other / ".git").mkdir(parents=True)
    rc, env = _json(capsys, ["init", "--dir", str(other), "--site", "c", "--launcher", "/o/bin/python"])
    assert rc == 0 and env["paths"]["dir"] == str(other / "yavih-bench")
    assert config.load(refresh=True, cwd=other).launcher() == "/o/bin/python"


def test_relocated_directories_are_not_strays_and_show_never_raises(repo, runs_dir):
    init_project()
    (repo / "experiments" / "matrices").mkdir(parents=True)
    (repo / "experiments" / "matrices" / "matrix.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (repo / "yavih-bench" / "yavih.toml").write_text('[project]\nmatrices = "../experiments/matrices"\n', encoding="utf-8")
    data, warnings, _n = doctor_block(config.load(refresh=True))
    assert data["stray"] == [] and warnings == []
    (repo / "yavih-bench" / "yavih.toml").write_text('[project]\nbridges = "${data.root}/bridges"\n', encoding="utf-8")
    s = config.load(refresh=True)
    shown = s.show()["project"]
    assert shown["bridges"].startswith("<") and "data.root" in shown["bridges"] and shown["matrices"].endswith("matrices")
    with pytest.raises(config.ConfigError):
        s.project_path("bridges")


def test_discovery_walks_through_a_submodule_but_stops_at_a_worktree(repo, tmp_path):
    (repo / "yavih-bench").mkdir()
    (repo / "yavih-bench" / "yavih.toml").write_text("[runs]\nroot = \"/r\"\n", encoding="utf-8")
    sub = repo / "models" / "vjepa"
    sub.mkdir(parents=True)
    (sub / ".git").write_text("gitdir: ../../.git/modules/models/vjepa\n", encoding="utf-8")
    assert config._discover_project(sub / "src") == repo / "yavih-bench" / "yavih.toml"
    assert config.git_root(sub) == repo
    wt = repo / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {tmp_path}/elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    assert config._discover_project(wt) is None and config.git_root(wt) == wt


def test_project_file_is_parsed_once_per_version(repo, monkeypatch):
    init_project()
    calls = []
    real = config._read_file

    def counting(path, layer, how):
        calls.append(str(path))
        return real(path, layer, how)

    monkeypatch.setattr(config, "_read_file", counting)
    config._PROJECT_FILE_CACHE.clear()
    for _ in range(5):
        config.project_benchmarks_dir()
    assert calls.count(str(repo / "yavih-bench" / "yavih.toml")) == 1
    (repo / "yavih-bench" / "yavih.toml").write_text('[project]\nbenchmarks = "../benches"\n', encoding="utf-8")
    assert config.project_benchmarks_dir() == (repo / "benches").resolve()
    assert calls.count(str(repo / "yavih-bench" / "yavih.toml")) == 2
