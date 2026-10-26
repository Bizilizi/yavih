"""``yavih.external``: the assert-import of original-author packages, including the git-checkout
route for a package importable without distribution metadata (a git checkout on ``PYTHONPATH``)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from yavih import external

SHA = "a1f14640076e38b8bc07b66d0fe2d01d15691e9d"
OTHER = "0123456789abcdef0123456789abcdef01234567"
PIP = f"pip install loguru -e git+https://github.com/example/authors.git@{SHA}#egg=authpkg_ext"


@pytest.fixture(autouse=True)
def _clean():
    external.reset()
    yield
    external.reset()
    for m in [m for m in sys.modules if m == "authpkg_ext" or m.startswith("authpkg_ext.")]:
        del sys.modules[m]


def _checkout(tmp_path, *, head: str | None = SHA, version: str | None = "1.0", ref: bool = False, packed: bool = False,
              init_py: bool = False):
    """A Probe3D-shaped checkout: ``setup.py`` + a namespace package ``authpkg_ext/utils/metrics.py`` (no
    ``__init__.py`` at the top, like Probe3D's ``evals``), no dist-info anywhere; ``.git`` written by hand."""
    root = tmp_path / "authors"
    (root / "authpkg_ext" / "utils").mkdir(parents=True)
    (root / "authpkg_ext" / "utils" / "__init__.py").write_text("")
    (root / "authpkg_ext" / "utils" / "metrics.py").write_text("def evaluate():\n    return 1\n")
    if init_py:
        (root / "authpkg_ext" / "__init__.py").write_text("")
    if version is not None:
        (root / "setup.py").write_text(f'from setuptools import setup\nsetup(name="authpkg_ext", version="{version}")\n')
    if head is not None:
        g = root / ".git"
        g.mkdir()
        if ref or packed:
            (g / "HEAD").write_text("ref: refs/heads/main\n")
            if packed:
                (g / "packed-refs").write_text(f"# pack-refs with: peeled\n{head} refs/heads/main\n")
            else:
                (g / "refs" / "heads").mkdir(parents=True)
                (g / "refs" / "heads" / "main").write_text(head + "\n")
        else:
            (g / "HEAD").write_text(head + "\n")
    return root


def test_same_version_is_pep440_equality():
    """``require`` / ``status`` compare the installed version with ``version_pin`` as PEP 440 versions."""
    assert external.same_version("1.0.0", "1.0") and external.same_version("1.0", "1.0.0") and external.same_version("1.0", "1.0")
    assert not external.same_version("1.0.1", "1.0") and not external.same_version("1.1", "1.0") and not external.same_version("0.9", "1.0")
    assert not external.same_version(None, "1.0")
    assert external.same_version("1.0+a1f14640 (git checkout)", "1.0+a1f14640 (git checkout)")  # not a version: text equality
    assert not external.same_version("1.0+a1f14640 (git checkout)", "1.0")


def test_pinned_commit():
    assert external.pinned_commit(PIP) == SHA
    assert external.pinned_commit("pip install git+ssh://git@github.com/x/y.git@ABCDEF1#egg=y") == "abcdef1"
    assert external.pinned_commit("pip install --no-deps vggsounder==1.0") is None
    assert external.pinned_commit("pip install git+https://x/y.git@main#egg=y") is None  # a branch is no pin


@pytest.mark.parametrize("kw", [{}, {"ref": True}, {"packed": True}, {"init_py": True}], ids=["detached", "ref", "packed-ref", "regular-pkg"])
def test_checkout_at_pinned_commit_is_accepted(tmp_path, monkeypatch, kw):
    root = _checkout(tmp_path, **kw)
    monkeypatch.syspath_prepend(str(root))
    st = external.status("authpkg_ext", PIP, "1.0")  # without importing (doctor, bench verify)
    assert st["ok"] is True and st["version"] == f"1.0+{SHA[:8]} (git checkout)"
    assert st["checkout"]["root"] == str(root.resolve()) and st["checkout"]["head"] == SHA
    assert "authpkg_ext" not in sys.modules
    mod = external.require("authpkg_ext", PIP, "1.0")
    assert mod.__name__ == "authpkg_ext"
    assert external.versions_used() == {"authpkg_ext": f"1.0+{SHA[:8]} (git checkout)"}
    external.record_version("authpkg_ext", "1.0+a1f14640 (metrics.py sha256 0123)")  # a folder's finer pin still works


def test_checkout_at_other_commit_is_refused(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_checkout(tmp_path, head=OTHER)))
    st = external.status("authpkg_ext", PIP, "1.0")
    assert st["ok"] is False and st["version"] is None and OTHER in st["checkout"]["reason"]
    with pytest.raises(external.ExternalVersion, match=r"E_EXTERNAL_VERSION.*HEAD is 0123456789ab.*the pin is a1f14640"):
        external.require("authpkg_ext", PIP, "1.0")
    assert external.versions_used() == {}


def test_checkout_without_git_is_refused(tmp_path, monkeypatch):
    """A plain copy (tarball, no .git) cannot prove its commit; the pip line's checkout hint is in the message."""
    monkeypatch.syspath_prepend(str(_checkout(tmp_path, head=None)))
    if external._git_head(tmp_path) is not None:  # pragma: no cover - tmp under some git work tree
        pytest.skip("tmp_path lies inside a git work tree")
    with pytest.raises(external.ExternalVersion, match="not inside a git checkout") as ei:
        external.require("authpkg_ext", PIP, "1.0")
    assert f"git -C <dir> checkout {SHA}" in str(ei.value) and "PYTHONPATH" in str(ei.value)
    assert external.status("authpkg_ext", PIP, "1.0")["ok"] is False


def test_checkout_declaring_another_version_is_refused(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_checkout(tmp_path, version="2.0")))
    with pytest.raises(external.ExternalVersion, match="declares version 2.0, the pin is 1.0"):
        external.require("authpkg_ext", PIP, "1.0")


def test_no_commit_pin_keeps_the_metadata_rule(tmp_path, monkeypatch):
    """vggsounder-style pins (``pkg==1.0``) never take the checkout route: no metadata = unknown version."""
    monkeypatch.syspath_prepend(str(_checkout(tmp_path)))
    pip = "pip install --no-deps authpkg_ext==1.0"
    assert external.status("authpkg_ext", pip, "1.0") == {"ok": False, "version": None, "version_pin": "1.0", "pip": pip}
    with pytest.raises(external.ExternalVersion, match=r"\(unknown version\)") as ei:
        external.require("authpkg_ext", pip, "1.0")
    assert "git clone" not in str(ei.value)


def test_dist_metadata_still_decides_when_present(tmp_path, monkeypatch):
    """With distribution metadata the version check is unchanged, even inside a checkout at the pinned commit."""
    from importlib import metadata

    monkeypatch.syspath_prepend(str(_checkout(tmp_path)))
    monkeypatch.setattr(metadata, "version", lambda name: "0.9")
    assert external.status("authpkg_ext", PIP, "1.0")["ok"] is False
    with pytest.raises(external.ExternalVersion, match="authpkg_ext 0.9 is installed"):
        external.require("authpkg_ext", PIP, "1.0")
    monkeypatch.setattr(metadata, "version", lambda name: "1.0")
    external.require("authpkg_ext", PIP, "1.0")
    assert external.versions_used() == {"authpkg_ext": "1.0"}


def test_missing_package_message_names_both_routes():
    with pytest.raises(external.ExternalMissing) as ei:
        external.require("authpkg_ext_absent", PIP.replace("authpkg_ext", "authpkg_ext_absent"), "1.0")
    msg = str(ei.value)
    assert "pip install loguru -e git+https://github.com/example/authors.git" in msg
    assert f"git clone https://github.com/example/authors.git <dir> && git -C <dir> checkout {SHA}" in msg
    assert external.status("authpkg_ext_absent", PIP, "1.0")["ok"] is False


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_real_git_checkout_and_local_edits(tmp_path, monkeypatch):
    """A real ``git clone``-like checkout: accepted at the pinned commit, refused once a tracked file in the
    package is edited (the dirty check needs git on PATH)."""
    root = _checkout(tmp_path, head=None)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True, text=True, env=env)  # noqa: E731
    run("init", "-q")
    run("add", "-A")
    run("commit", "-q", "-m", "x")
    head = run("rev-parse", "HEAD").stdout.strip()
    pip = PIP.replace(SHA, head)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    st = external.status("authpkg_ext", pip, "1.0")
    if st["checkout"]["modified"] is None:  # pragma: no cover - git refuses the tree (ownership policy)
        pytest.skip("git status unavailable for the temporary checkout")
    assert st["ok"] is True and st["checkout"]["modified"] == []
    (root / "authpkg_ext" / "utils" / "metrics.py").write_text("def evaluate():\n    return 2\n")
    st = external.status("authpkg_ext", pip, "1.0")
    assert st["ok"] is False and "metrics.py" in st["checkout"]["reason"]
    with pytest.raises(external.ExternalVersion, match="modified tracked files"):
        external.require("authpkg_ext", pip, "1.0")


def test_nyuv2_setup_documents_the_checkout_route():
    """The documented clone + checkout is exactly the commit :func:`pinned_commit` reads from the yaml pip line."""
    import yaml

    from yavih.benchmarks import packaged_folder

    folder = packaged_folder("nyuv2_depth")
    pip = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))["external"][0]["pip"]
    commit = external.pinned_commit(pip)
    text = (folder / "setup.md").read_text(encoding="utf-8")
    deps = text.split("## Dependencies", 1)[1].split("\n## ", 1)[0]
    assert pip in deps  # the verbatim pip line stays
    assert f"checkout {commit}" in deps and "export PYTHONPATH=" in deps and "uv pip install" in deps
    assert "(git checkout)" in deps


def test_check_externals_records_the_checkout_identity(tmp_path, monkeypatch):
    """Integration: ``readouts.base.check_externals`` (probe prologue AND ``report`` joins) records
    the identity ``require`` accepted, ``"<pin>+<sha8> (git checkout)"``, not ``"unknown"`` (a checkout has neither
    dist metadata nor ``__version__``)."""
    from types import SimpleNamespace

    from yavih.readouts.base import check_externals

    root = _checkout(tmp_path)
    monkeypatch.syspath_prepend(str(root))
    spec = SimpleNamespace(external=[{"name": "authpkg_ext", "pip": PIP, "version_pin": "1.0"}])
    assert check_externals(spec) == {"authpkg_ext": f"1.0+{SHA[:8]} (git checkout)"}
