"""``yavih.buildinfo``: which source commit is running.

Resolution git checkout (source layout) > setuptools-scm version file (installed wheel) > None, caching, the
``+dirty`` rule, and that nothing ever raises. The version file itself is exercised by a wheel build
(tests/test_import_boundary.py), not here."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from yavih import buildinfo

HEAD = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    import yavih

    monkeypatch.delitem(sys.modules, "yavih._version", raising=False)
    monkeypatch.delattr(yavih, "_version", raising=False)
    buildinfo._reset_cache()
    yield
    buildinfo._reset_cache()


def _stamp(monkeypatch, **attrs) -> None:
    """A setuptools-scm version file (``__version__``, ``__commit_id__`` = "g" + commit, ``__dirty__``)."""
    mod = types.ModuleType("yavih._version")
    mod.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, "yavih._version", mod)


def _installed(monkeypatch, tmp_path) -> None:
    """Run as an installed wheel: the package directory is ``site-packages/yavih``, not ``<top>/src/yavih``."""
    monkeypatch.setattr(buildinfo, "_PKG_DIR", tmp_path / "site-packages" / "yavih")


def _no_git(monkeypatch) -> list:
    calls: list = []

    def boom(*a, **k):
        calls.append(a)
        raise FileNotFoundError("git")

    monkeypatch.setattr(buildinfo.subprocess, "run", boom)
    return calls


def test_installed_wheel_uses_the_version_file_and_git_is_not_called(monkeypatch, tmp_path):
    _installed(monkeypatch, tmp_path)
    _stamp(monkeypatch, __version__="9.9.9", __commit_id__="g" + HEAD, __dirty__=False)
    calls = _no_git(monkeypatch)
    assert buildinfo.commit() == HEAD
    assert buildinfo.describe() == {"version": "9.9.9", "commit": HEAD, "source": "build"}
    assert calls == []


def test_version_file_dirty_and_without_version(monkeypatch, tmp_path):
    from yavih.versions import __version__

    _installed(monkeypatch, tmp_path)
    _stamp(monkeypatch, __commit_id__="g" + HEAD, __dirty__=True)
    _no_git(monkeypatch)
    assert buildinfo.describe() == {"version": __version__, "commit": HEAD + "+dirty", "source": "build"}


@pytest.mark.parametrize("bad", [None, "", "deadbeef", "g0123456789", "g" + HEAD.upper(), HEAD + "+wip", 42])
def test_unusable_version_file_is_unknown(monkeypatch, tmp_path, bad):
    """A wheel built from an sdist or without git metadata records no full commit: unknown, never a guess."""
    _installed(monkeypatch, tmp_path)
    _stamp(monkeypatch, __version__="1.0", __commit_id__=bad, __dirty__=False)
    calls = _no_git(monkeypatch)
    assert buildinfo.describe() == {"version": buildinfo._package_version(), "commit": None, "source": None}
    assert calls == []


def test_stamped_commit():
    assert buildinfo.stamped_commit("g" + HEAD) == HEAD == buildinfo.stamped_commit(HEAD)
    assert buildinfo.stamped_commit("g" + HEAD, True) == HEAD + "+dirty"
    assert buildinfo.stamped_commit("g" + HEAD, "yes") == HEAD  # only a real True marks it dirty
    assert buildinfo.stamped_commit("g" + HEAD + "+dirty") is None and buildinfo.stamped_commit(None) is None


def test_checkout_asks_git_and_ignores_the_install_time_version_file(monkeypatch):
    """In ``<top>/src/yavih`` the version file is from the last ``pip install -e``; git names the current commit."""
    monkeypatch.setattr(buildinfo, "_PKG_DIR", Path("/x/src/yavih"))
    _stamp(monkeypatch, __version__="1.0", __commit_id__="g" + "f" * 40, __dirty__=False)
    monkeypatch.setattr(buildinfo, "_from_checkout", lambda: HEAD + "+dirty")
    d = buildinfo.describe()
    assert d["commit"] == HEAD + "+dirty" and d["source"] == "git" and isinstance(d["version"], str)


def test_checkout_git_failure_gives_none_and_never_raises(monkeypatch):
    monkeypatch.setattr(buildinfo, "_PKG_DIR", Path("/x/src/yavih"))
    _stamp(monkeypatch, __version__="1.0", __commit_id__="g" + HEAD, __dirty__=False)
    _no_git(monkeypatch)
    assert buildinfo.commit() is None
    assert buildinfo.describe()["commit"] is None and buildinfo.describe()["source"] is None


@pytest.mark.parametrize("exc", [subprocess.TimeoutExpired("git", 5), PermissionError("x"), ValueError("x")])
def test_git_errors_are_swallowed(monkeypatch, exc):
    def raise_(*a, **k):
        raise exc

    monkeypatch.setattr(buildinfo.subprocess, "run", raise_)
    assert buildinfo.git_commit(Path(buildinfo.__file__).parents[2]) is None


def test_broken_version_file_never_raises(monkeypatch, tmp_path):
    """A version file that fails on attribute access (corrupt file) is 'no stamp', not a crash."""
    class Exploding(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError("corrupt stamp")

    _installed(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "yavih._version", Exploding("yavih._version"))
    assert buildinfo.commit() is None and buildinfo.describe()["source"] is None


def test_result_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(buildinfo, "_PKG_DIR", Path("/x/src/yavih"))
    monkeypatch.setattr(buildinfo, "_from_checkout", lambda: calls.append(1) or HEAD)
    assert buildinfo.commit() == HEAD and buildinfo.commit() == HEAD and buildinfo.describe()["commit"] == HEAD
    assert calls == [1]
    d = buildinfo.describe()
    d["commit"] = "mutated"
    assert buildinfo.commit() == HEAD  # describe() hands out a copy


def test_import_is_lazy():
    """Importing the module runs no git (a fresh interpreter, subprocess.run patched to fail before import)."""
    code = ("import subprocess\n"
            "def boom(*a, **k): raise SystemExit('git called at import')\n"
            "subprocess.run = boom\n"
            "import yavih.buildinfo as b\n"
            "assert b._cache is None\n"
            "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def _git_repo(root: Path) -> None:
    def g(*a):
        subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True,
                       env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                            "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(root), "PATH": "/usr/bin:/bin"})

    (root / "src" / "yavih").mkdir(parents=True)
    (root / "src" / "yavih" / "x.py").write_text("x = 1\n", encoding="utf-8")
    g("init", "-q")
    g("add", "-A")
    g("commit", "-q", "-m", "init")


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True).returncode != 0, reason="no git")
def test_git_commit_on_a_real_checkout(tmp_path):
    root = tmp_path / "yavih"
    _git_repo(root)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert buildinfo.git_commit(root) == head
    (root / "untracked.txt").write_text("u", encoding="utf-8")
    assert buildinfo.git_commit(root) == head  # untracked files do not make a tree dirty
    (root / "src" / "yavih" / "x.py").write_text("x = 2\n", encoding="utf-8")
    assert buildinfo.git_commit(root) == head + "+dirty"
    assert buildinfo.git_commit(root / "src") is None  # not the work-tree top level
    other = tmp_path / "notyavih"
    other.mkdir()
    assert buildinfo.git_commit(other) is None  # not a git tree at all


def test_installed_copy_inside_a_checkout_is_not_a_checkout(monkeypatch, tmp_path):
    """``.venv/lib/.../site-packages/yavih`` under a work tree must not report the work tree's commit."""
    pkg = tmp_path / "repo" / ".venv" / "lib" / "site-packages" / "yavih"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(buildinfo, "_PKG_DIR", pkg)
    monkeypatch.setattr(buildinfo, "git_commit", lambda root: pytest.fail("git consulted for a non-src layout"))
    assert buildinfo._from_checkout() is None
