"""One command surface for shipped files (``yavih.shipped``): ``bench files|show|path|prepare``, ``bridge new``,
``skill show --ref``, the skill install stamp and ``doctor``'s ``installed_from`` / ``W_SKILL_STALE``. An agent reaches
every packaged document, template and prepare script through these commands, from a wheel as from a checkout."""

from __future__ import annotations

import json
import shutil
import stat
import sys
from pathlib import Path

import pytest

import yavih
from yavih import cli, shipped
from yavih.benchmarks import packaged_folder

ESC50 = packaged_folder("esc50")
PACKAGED = sorted(p.name for p in Path(yavih.__file__).resolve().parent.joinpath("benchmarks").iterdir()
                  if (p / "benchmark.yaml").is_file())


def _json(capsys, argv: list[str]) -> tuple[int, dict]:
    rc = cli.main(["--json", *argv])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1, out
    return rc, json.loads(out[0])


# ----------------------------------------------------------------------------- bench files / show / path


def test_bench_files_lists_every_shipped_file_with_kind_and_absolute_path(capsys):
    rc, env = _json(capsys, ["bench", "files", "esc50"])
    assert rc == 0 and env["status"] == "ok"
    files = env["data"]["files"]
    kinds = {f["kind"] for f in files}
    assert {"readme", "setup", "yaml", "template", "manifest", "classes", "prepare", "test"} <= kinds
    assert kinds <= set(shipped.FILE_KINDS)
    for f in files:
        p = Path(f["path"])
        assert p.is_absolute() and p.is_file() and p.stat().st_size == f["size"]
        assert "notes" not in Path(f["rel"]).parts and "__pycache__" not in f["rel"] and not f["rel"].endswith(".pyc")
    rels = {f["rel"] for f in files}
    assert {"setup.md", "bridge_template.py", "prepare/01_download.sh", "tests/test_folder.py"} <= rels
    assert env["data"]["folder"] == str(ESC50.resolve())


def test_python_api_mirrors_the_commands():
    from yavih.benchmark import bench_file_text, bench_files

    assert {f["rel"] for f in bench_files("esc50")} == {f["rel"] for f in shipped.bench_files("esc50")}
    assert bench_file_text("esc50", "setup") == (ESC50 / "setup.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("what, rel", [("readme", "README.md"), ("setup", "setup.md"), ("yaml", "benchmark.yaml"),
                                       ("template", "bridge_template.py"), ("classes", "classes.csv"),
                                       ("prepare/02_verify.sh", "prepare/02_verify.sh"), ("tests/test_folder.py", "tests/test_folder.py")])
def test_bench_show_prints_one_file(capsys, what, rel):
    rc, env = _json(capsys, ["bench", "show", "esc50", what])
    assert rc == 0 and env["data"]["path"] == str((ESC50 / rel).resolve())
    assert env["data"]["text"] == (ESC50 / rel).read_text(encoding="utf-8")
    assert cli.main(["bench", "show", "esc50", what]) == 0
    assert capsys.readouterr().out.rstrip("\n") == (ESC50 / rel).read_text(encoding="utf-8").rstrip("\n")


def test_bench_show_probe_of_a_folder_with_probe_py(capsys):
    rc, env = _json(capsys, ["bench", "show", "avsync", "probe"])
    assert rc == 0 and env["data"]["path"].endswith("avsync/probe.py")


@pytest.mark.parametrize("what, code", [("../../cli.py", "E_PATH"), ("/etc/passwd", "E_PATH"), ("prepare/../../shipped.py", "E_PATH"),
                                        ("notes/anything.md", "E_PATH"), ("nope.md", "E_NOT_FOUND"), ("manifest", "E_BINARY"),
                                        ("probe", "E_NOT_FOUND")])
def test_bench_show_refuses_paths_outside_and_binaries(capsys, what, code):
    rc, env = _json(capsys, ["bench", "show", "esc50", what])
    assert rc == 1 and env["status"] == "error" and env["errors"][0]["code"] == code, env


def test_bench_info_setup_is_bench_show_setup(capsys):
    _, show = _json(capsys, ["bench", "show", "esc50", "setup"])
    rc, info = _json(capsys, ["bench", "info", "esc50", "--setup"])
    assert rc == 0 and info["data"]["setup"] == show["data"]["text"] and info["paths"]["setup"] == show["data"]["path"]
    _, readme = _json(capsys, ["bench", "info", "esc50", "--readme-only"])
    assert readme["data"]["readme"] == (ESC50 / "README.md").read_text(encoding="utf-8")


def test_bench_path_prints_the_folder_or_one_file(capsys):
    assert cli.main(["bench", "path", "esc50"]) == 0
    assert capsys.readouterr().out.strip() == str(ESC50.resolve())
    assert cli.main(["bench", "path", "k400", "prepare/01_download_torrent.sh"]) == 0
    assert capsys.readouterr().out.strip() == str((packaged_folder("k400") / "prepare" / "01_download_torrent.sh").resolve())


def test_unknown_benchmark_is_a_usage_error(capsys):
    rc, env = _json(capsys, ["bench", "files", "nope"])
    assert rc == 1 and env["status"] == "error"


# ----------------------------------------------------------------------------- bench prepare


@pytest.fixture
def folder_copy(tmp_path):
    dst = tmp_path / "esc50"
    shutil.copytree(ESC50, dst, ignore=shutil.ignore_patterns("__pycache__", "notes"))
    (dst / "prepare" / "09_echo.sh").write_text('#!/usr/bin/env bash\n# print what the step sees\n'
                                                'echo "FOLDER=$FOLDER BENCH=$YAVIH_BENCH ARGS=$* PWD=$PWD"\n', encoding="utf-8")
    (dst / "prepare" / "10_py.py").write_text('"""print the interpreter"""\nimport os, sys\n'
                                              'print("PY", sys.executable, os.environ["YAVIH_BENCH"], sys.argv[1:])\n', encoding="utf-8")
    (dst / "prepare" / "11_fail.sh").write_text("#!/usr/bin/env bash\nexit 3\n", encoding="utf-8")
    for p in (dst / "prepare").iterdir():
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return dst


def test_bench_prepare_lists_numbered_steps(capsys):
    rc, env = _json(capsys, ["bench", "prepare", "esc50"])
    assert rc == 0
    steps = env["data"]["steps"]
    assert [s["file"] for s in steps] == ["01_download.sh", "02_verify.sh", "03_make_manifest.sh"]
    assert all(s["interpreter"] == "bash" and Path(s["path"]).is_file() for s in steps)
    rc, env = _json(capsys, ["bench", "prepare", "avsync"])
    assert {s["file"]: s["interpreter"] for s in env["data"]["steps"]}["03_materialize.py"] == "python"


def test_bench_prepare_runs_a_step_with_folder_and_bench_set(capfd, folder_copy, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["bench", "prepare", str(folder_copy), "09", "--", "a", "--b"]) == 0
    out = capfd.readouterr().out
    assert f"FOLDER={folder_copy.resolve()} BENCH=esc50 ARGS=a --b PWD={tmp_path}" in out
    assert cli.main(["bench", "prepare", str(folder_copy), "py", "--", "x"]) == 0  # by name, python = this interpreter
    assert f"PY {sys.executable} esc50 ['x']" in capfd.readouterr().out
    assert cli.main(["bench", "prepare", str(folder_copy), "11_fail.sh"]) == 3  # the step's own exit code
    assert "E_PREPARE_FAILED" in capfd.readouterr().err
    assert cli.main(["bench", "prepare", str(folder_copy), "42"]) == 1
    assert "E_PREPARE_STEP" in capfd.readouterr().err


def test_yavih_bench_is_a_known_env_var():
    from yavih.config import ENV_VARS

    assert "YAVIH_BENCH" in ENV_VARS  # strict config mode rejects unknown YAVIH_* names in the step's children


# ----------------------------------------------------------------------------- bridge new


def test_bridge_new_fills_the_model_and_keeps_the_template_guard(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc, env = _json(capsys, ["bridge", "new", "esc50", "--model", "demo"])
    assert rc == 0, env
    path = tmp_path / "bridges" / "demo_esc50.py"
    assert env["data"]["path"] == str(path.resolve())
    text = path.read_text(encoding="utf-8")
    assert '"model": "demo"' in text and '"TODO-model"' not in text and 'name = "demo-esc50"' in text
    assert "yavih bench show esc50 setup" in text.split('"""', 1)[0]  # the header names the contract command
    assert '"weights_ref": "TODO' in text  # still a template: bridge check refuses it
    compile(text, str(path), "exec")
    rc, env = _json(capsys, ["bridge", "new", "esc50", "--model", "demo"])
    assert rc == 1 and env["errors"][0]["code"] == "E_EXISTS"
    rc, env = _json(capsys, ["bridge", "new", "esc50", "--model", "demo", "--reading", "v", "--force"])
    assert rc == 0 and '"reading": "v"' in path.read_text(encoding="utf-8")
    rc, env = _json(capsys, ["bridge", "check", str(path), "--bench", "esc50", "--synthetic"])
    assert rc == 2 and env["errors"][0]["code"] == "E_TEMPLATE_UNFILLED", env


def test_bridge_new_reading_arg_and_out(capsys, tmp_path):
    rc, env = _json(capsys, ["bridge", "new", "vggsounder", "--model", "m1", "--reading", "v", "--out", str(tmp_path / "x") + "/"])
    assert rc == 0
    text = (tmp_path / "x" / "m1_vggsounder.py").read_text(encoding="utf-8")
    assert 'default="v", choices=("a", "v", "av")' in text
    rc, env = _json(capsys, ["bridge", "new", "esc50", "--model", "_hidden"])
    assert rc == 1 and env["errors"][0]["code"] == "E_ARG"


@pytest.mark.parametrize("name", PACKAGED)
def test_bridge_new_works_for_every_packaged_template(tmp_path, name):
    rec = shipped.bridge_new(name, model="m", out=tmp_path / f"m_{name}.py")
    text = Path(rec["path"]).read_text(encoding="utf-8")
    assert "describe()['model']" in rec["filled"] and '"model": "m"' in text
    assert '"weights_ref": "TODO' in text  # E_TEMPLATE_UNFILLED stays meaningful
    compile(text, rec["path"], "exec")


# ----------------------------------------------------------------------------- skill


def test_skill_show_ref(capsys):
    rc, env = _json(capsys, ["skill", "show", "--ref", "bridge"])
    assert rc == 0 and env["data"]["text"] == (shipped.skill_root() / "references" / "bridge.md").read_text(encoding="utf-8")
    rc, env = _json(capsys, ["skill", "show", "--ref", "nope"])
    assert rc == 1 and env["status"] == "error"
    rc, env = _json(capsys, ["skill", "show", "--all"])
    assert rc == 0 and len(env["data"]["references"]) == 5


def test_skill_install_stamp_and_doctor_staleness(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc, env = _json(capsys, ["skill", "install"])
    assert rc == 0 and shipped.SKILL_STAMP_FILE in env["data"]["files"]
    stamp_path = tmp_path / ".claude" / "skills" / "yavih" / shipped.SKILL_STAMP_FILE
    assert json.loads(stamp_path.read_text()) == shipped.skill_stamp()
    rc, env = _json(capsys, ["skill", "install", "--target", "cursor"])
    assert rc == 0
    rc, doc = _json(capsys, ["doctor"])
    assert doc["data"]["installed_from"] in ("wheel", "editable", "checkout")
    assert Path(doc["data"]["package_dir"]) == Path(yavih.__file__).resolve().parent
    installed = {i["target"]: i for i in doc["data"]["skill"]["installed"]}
    assert set(installed) == {"claude", "cursor"} and not any(i["stale"] for i in installed.values())
    assert "W_SKILL_STALE" not in {w["code"] for w in doc["warnings"]}
    stamp_path.write_text(json.dumps({"yavih_version": "0.0.1", "sha256": "0" * 64}))
    cursor = tmp_path / ".cursor" / "rules" / "yavih.mdc"
    cursor.write_text(cursor.read_text().split("<!-- yavih skill stamp")[0])  # an unstamped (older) install
    rc, doc = _json(capsys, ["doctor"])
    stale = [w for w in doc["warnings"] if w["code"] == "W_SKILL_STALE"]
    assert len(stale) == 2 and any("0.0.1" in w["message"] for w in stale) and all("yavih skill install" in w["message"] for w in stale)


def test_installed_from_editable_checkout():
    info = shipped.installed_from()
    assert info["installed_from"] in ("editable", "checkout")  # the test suite runs from the source tree
