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


def test_init_writes_the_folder_and_the_user_config(repo, capsys):
    rc, env = _json(capsys, ["init", "--site", "c1", "--hosts", "*", "--launcher", "/envs/m/bin/python"])
    assert rc == 0, env["errors"]
    folder = repo / "yavih-bench"
    for name in ("yavih.toml", "README.md", ".gitignore", "bridges/.gitkeep", "matrices/.gitkeep", "suites/.gitkeep",
                 "benchmarks/.gitkeep", "tools/.gitkeep"):
        assert (folder / name).is_file(), name
    assert env["data"]["user_config_written"] is True and env["paths"]["dir"] == str(folder)
    assert "*.jsonl" in (folder / ".gitignore").read_text(encoding="utf-8").splitlines()  # plan files are run state
    user = Path(env["data"]["user_config"])
    assert user.is_file() and "[runs]" in user.read_text(encoding="utf-8")
    s = config.load(refresh=True)
    assert s.project_dir() == folder and s.project_file().how == "discovered"
    assert s.site == "c1" and s.launcher() == "/envs/m/bin/python" and s.launchers() == {}
    assert s.project_requirement() and s.project_requirement().startswith(">=")
    assert s.project_path("suites") == folder / "suites" and s.project_path("benchmarks") == folder / "benchmarks"
    # the committed project file carries no personal root: runs come from the user file
    assert s.get("runs", "root").source.startswith(f"file:{user}")
    assert s.runs_root() == Path("~/yavih/runs").expanduser()
    shown = s.show()
    assert shown["project"]["dir"] == str(folder) and shown["values"]["launcher"]["value"] == "/envs/m/bin/python"


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
    init_project(user_file=False)
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
    init_project(user_file=False)
    root_file = repo / "yavih-bench" / "yavih.toml"
    sub = repo / "services" / "asr"
    sibling = repo / "services" / "tts"
    for d in (sub, sibling):
        d.mkdir(parents=True)
    (repo / "yavih-bench" / "matrices" / "matrix.old.yaml").write_text("schema_version: 1\n", encoding="utf-8")  # the root's: no stray from the subproject
    root_file.write_text('[project]\nsubprojects = ["../services/asr", "../services/tts"]\n', encoding="utf-8")
    monkeypatch.chdir(sub)
    rec = init_project(user_file=False)
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
        init_project(user_file=False)
    assert ei.value.code == "E_PROJECT_EXISTS" and "subprojects" in str(ei.value)
    # from inside the subproject, an unlisted project file elsewhere is still nested
    (other / "yavih.toml").write_text("[runs]\nroot = \"r\"\n", encoding="utf-8")
    data, warnings, _n = doctor_block(config.load(refresh=True, cwd=sub), cwd=sub)
    assert data["nested"] == [str(other / "yavih.toml")] and [c for c, _m, _w in warnings] == ["W_PROJECT_NESTED"]


def test_layout_report_and_doctor_findings(repo, runs_dir):
    init_project(user_file=False)
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


def test_doctor_without_a_project_notes_init(repo, runs_dir):
    data, warnings, notes = doctor_block(config.load(refresh=True))
    assert data["dir"] is None and warnings == [] and any("yavih init" in n for n in notes)


def test_doctor_cli_carries_the_project_block(repo, runs_dir, capsys, monkeypatch):
    init_project(user_file=False)
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

    init_project(user_file=False)
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
    init_project(user_file=False)
    out = tmp_path / "elsewhere" / "toy.yaml"
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy", "--out", str(out)])
    assert rc == 0, env["errors"]
    assert (repo / "yavih-bench" / "bridges" / "toy_esc50.py").is_file() and not (tmp_path / "bridges").exists()
    raw = yaml.safe_load(out.read_text(encoding="utf-8"))
    script = raw["bridges"]["toy-esc50-a"]["script"]
    assert (out.parent / script).resolve() == (repo / "yavih-bench" / "bridges" / "toy_esc50.py").resolve()
    assert raw["bridges"]["toy-esc50-a"]["profile"] == "toy.profiles/toy-esc50-a.profile.json"


def test_named_launchers_alone_do_not_silence_the_unset_warning(repo, runs_dir, capsys):
    init_project(user_file=False)
    (repo / "yavih-bench" / "yavih.toml").write_text('[launchers]\nvjepa = "/envs/vjepa/bin/python"\n', encoding="utf-8")
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy"])
    assert rc == 0 and [w["code"] for w in env["warnings"]] == ["W_LAUNCHER_UNSET"]
    rc, env = _json(capsys, ["plan", "init", "--suite", "minimal", "--model", "toy", "--force", "--launcher", "vjepa"])
    assert rc == 0 and env["warnings"] == [] and env["data"]["launcher"] == "vjepa"


def test_init_launcher_without_site_and_hosts_without_site(repo, capsys, tmp_path):
    rc, env = _json(capsys, ["init", "--hosts", "*"])
    assert rc == 1 and env["errors"][0]["code"] == "E_ARG" and "--site" in env["errors"][0]["message"]
    rc, env = _json(capsys, ["init", "--launcher", "/envs/m/bin/python", "--no-user-config"])
    assert rc == 0 and config.load(refresh=True).launcher() == "/envs/m/bin/python"
    # --dir: the new project is validated from its own directory, not from the cwd
    other = tmp_path / "other"
    (other / ".git").mkdir(parents=True)
    rc, env = _json(capsys, ["init", "--dir", str(other), "--site", "c", "--launcher", "/o/bin/python", "--no-user-config"])
    assert rc == 0 and env["paths"]["dir"] == str(other / "yavih-bench")
    assert config.load(refresh=True, cwd=other).launcher() == "/o/bin/python"


def test_relocated_directories_are_not_strays_and_show_never_raises(repo, runs_dir):
    init_project(user_file=False)
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
    init_project(user_file=False)
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
