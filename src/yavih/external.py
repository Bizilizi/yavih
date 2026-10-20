"""Original-author packages: the ``external`` block of ``benchmark.yaml``.

A benchmark folder's ``probe.py`` reaches the benchmark authors' own scorer / loader (VGGSounder's
``vggsounder`` package, Probe3D's depth metrics, ...) ONLY through :func:`require` — the assert-import:

- the package absent → :class:`ExternalMissing` (``E_EXTERNAL_MISSING``, exit 3 "not ready"); the message is the
  ``pip`` line of ``benchmark.yaml[external]``, so the agent installs it independently;
- installed at another version than ``version_pin`` → :class:`ExternalVersion` (``E_EXTERNAL_VERSION``, exit 3);
- importable WITHOUT distribution metadata (a git checkout on ``PYTHONPATH``, no ``pip install``): accepted only
  when the ``pip`` line pins a git commit (``git+<url>@<sha>``) and the checkout the module is imported from is
  at that commit (``.git/HEAD``, read without git), its package directory has no modified tracked files (``git
  status``, when ``git status`` works) and its declared version (``setup.py`` / ``pyproject.toml``, when present)
  equals ``version_pin``; otherwise :class:`ExternalVersion`. A pip line without a commit (``vggsounder==1.0``)
  never takes this route. A folder with a finer pin (Probe3D's ``metrics.py`` sha256) still checks it after
  :func:`require` (a uv venv without pip cannot do the editable install).

Such packages are NEVER yavih dependencies: nothing here installs anything, and nothing in yavih
silently re-implements a missing package. :func:`versions_used` reports every package required in this
process (``result.json[external]``, the results file's ``config.external``).

Stdlib only.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "ExternalError",
    "ExternalMissing",
    "ExternalVersion",
    "require",
    "installed_version",
    "pinned_commit",
    "checkout_status",
    "status",
    "versions_used",
    "record_version",
    "reset",
]

_USED: dict[str, str] = {}


class ExternalError(RuntimeError):
    """Base: an original-author package is not usable as pinned (exit 3, "not ready")."""

    code = "E_EXTERNAL"
    exit_code = 3

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name


class ExternalMissing(ExternalError):
    """``import <name>`` failed (``E_EXTERNAL_MISSING``)."""

    code = "E_EXTERNAL_MISSING"


class ExternalVersion(ExternalError):
    """The installed version differs from ``version_pin`` (``E_EXTERNAL_VERSION``)."""

    code = "E_EXTERNAL_VERSION"


def same_version(installed: str | None, pin: str) -> bool:
    """``installed == pin`` as PEP 440 versions (``1.0`` is ``1.0.0``); a string that is not a version compares as text."""
    if installed is None:
        return False
    a, b = str(installed), str(pin)
    if a == b:
        return True
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:  # pragma: no cover - packaging is ubiquitous but not a dependency
        return _strip_zeros(a) == _strip_zeros(b)
    try:
        return Version(a) == Version(b)
    except InvalidVersion:
        return _strip_zeros(a) == _strip_zeros(b)


def _strip_zeros(v: str) -> str:
    parts = v.split(".")
    while len(parts) > 1 and parts[-1] == "0":
        parts.pop()
    return ".".join(parts)


def installed_version(name: str, module: Any = None) -> str | None:
    """``importlib.metadata.version(name)``, else the module's ``__version__``, else ``None``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
    if module is None:
        return None
    v = getattr(module, "__version__", None)
    return None if v is None else str(v)


def require(name: str, pip: str, version_pin: str) -> Any:
    """The assert-import: import ``name`` and check its version equals ``version_pin``; returns the
    module. Records ``{name: version}`` for :func:`versions_used`. A package without distribution metadata is
    accepted only as a git checkout at the commit the ``pip`` line pins (:func:`checkout_status`)."""
    try:
        module = importlib.import_module(name)
    except ImportError as e:
        raise ExternalMissing(
            name,
            f"E_EXTERNAL_MISSING: the benchmark authors' package {name!r} is not installed. It is never a yavih "
            f"dependency; install it independently in the probe env (setup.md '## Dependencies'): {_hint(pip)}"
            f"{_checkout_hint(pip)}",
        ) from e
    version = installed_version(name, module)
    if version is None:
        co = checkout_status(name, pip, version_pin, module=module)
        if co is not None:
            if not co["ok"]:
                raise ExternalVersion(
                    name,
                    f"E_EXTERNAL_VERSION: {name} is imported from {co['root'] or '(no git checkout)'} without "
                    f"distribution metadata and {co['reason']}; install exactly: {_hint(pip)}{_checkout_hint(pip)}",
                )
            _USED[name] = co["version"]
            return module
    if not same_version(version, str(version_pin)):
        raise ExternalVersion(
            name,
            f"E_EXTERNAL_VERSION: {name} {version or '(unknown version)'} is installed but this protocol pins "
            f"{version_pin}; install exactly: {_hint(pip)}{_checkout_hint(pip) if version is None else ''}",
        )
    _USED[name] = version
    return module


def status(name: str, pip: str, version_pin: str) -> dict[str, Any]:
    """``{ok, version, version_pin, pip}`` without importing the package (``yavih doctor``, ``bench verify``); a
    checkout accepted by :func:`checkout_status` reports its identity as ``version`` and adds ``checkout``."""
    try:
        v: str | None = metadata.version(name)
    except metadata.PackageNotFoundError:
        v = None
    out: dict[str, Any] = {"ok": same_version(v, str(version_pin)), "version": v, "version_pin": str(version_pin), "pip": _hint(pip)}
    if v is None:
        co = checkout_status(name, pip, version_pin)
        if co is not None:
            out.update(ok=co["ok"], version=co["version"] if co["ok"] else None, checkout=co)
    return out


# ----------------------------------------------------------------------------- git checkout without dist metadata

_GIT_PIN = re.compile(r"git\+(\S+?)@([0-9a-fA-F]{7,40})(?=[#\s]|$)")
_SETUP_VERSION = re.compile(r"""\bversion\s*=\s*["']([^"']+)["']""")


def pinned_commit(pip: str) -> str | None:
    """The commit a ``git+<url>@<sha>`` pip line pins (lower-case hex, 7-40 chars), else ``None``."""
    m = _GIT_PIN.search(str(pip))
    return m.group(2).lower() if m else None


def _checkout_hint(pip: str) -> str:
    m = _GIT_PIN.search(str(pip))
    if not m:
        return ""
    url, sha = m.group(1), m.group(2)
    return (f" -- or, where pip cannot install (a uv venv, a read-only env): git clone {url} <dir> && "
            f"git -C <dir> checkout {sha} && export PYTHONPATH=<dir>:$PYTHONPATH (plus the line's other packages)")


def _module_dirs(name: str, module: Any = None) -> list[Path]:
    """Directories the package is (or would be) imported from; ``find_spec`` does not execute it."""
    if module is None:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            return []
        if spec is None:
            return []
        locs = list(spec.submodule_search_locations or [])
        origin = spec.origin if spec.has_location else None
    else:
        locs = list(getattr(module, "__path__", None) or [])
        origin = getattr(module, "__file__", None)
    dirs = [Path(p) for p in locs] or ([Path(origin).parent] if origin else [])
    return [d.resolve() for d in dirs if d.is_dir()]


def _git_dir(root: Path) -> Path | None:
    g = root / ".git"
    if g.is_dir():
        return g
    if g.is_file():  # worktree / submodule: "gitdir: <path>"
        text = g.read_text(encoding="utf-8", errors="replace").strip()
        if text.startswith("gitdir:"):
            p = Path(text[len("gitdir:"):].strip())
            p = p if p.is_absolute() else (root / p)
            return p.resolve() if p.is_dir() else None
    return None


def _resolve_ref(git_dir: Path, ref: str) -> str | None:
    common = git_dir
    cd = git_dir / "commondir"
    if cd.is_file():
        c = Path(cd.read_text(encoding="utf-8").strip())
        common = (c if c.is_absolute() else git_dir / c).resolve()
    for base in dict.fromkeys((git_dir, common)):
        f = base / ref
        if f.is_file():
            return f.read_text(encoding="utf-8").strip().lower()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(" ", 1)
            if len(parts) == 2 and parts[1] == ref and not line.startswith(("#", "^")):
                return parts[0].lower()
    return None


def _git_head(start: Path) -> tuple[Path, str | None] | None:
    """``(checkout root, HEAD commit)`` of the git work tree containing ``start``, read from ``.git`` alone."""
    for d in (start, *start.parents):
        gd = _git_dir(d)
        if gd is None:
            continue
        head_file = gd / "HEAD"
        if not head_file.is_file():
            return d, None
        head = head_file.read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            return d, _resolve_ref(gd, head[len("ref:"):].strip())
        return d, head.lower()
    return None


def _declared_version(root: Path) -> str | None:
    """``version="…"`` of the checkout's ``setup.py``, else ``pyproject.toml``; ``None`` if neither states one."""
    for fname in ("setup.py", "pyproject.toml"):
        f = root / fname
        if f.is_file():
            m = _SETUP_VERSION.search(f.read_text(encoding="utf-8", errors="replace"))
            if m:
                return m.group(1)
    return None


def _modified(root: Path, dirs: list[Path]) -> list[str] | None:
    """Tracked files under ``dirs`` with local modifications (``git status``); ``None`` when git is unavailable."""
    git = shutil.which("git")
    if git is None:
        return None
    rel = [str(d.relative_to(root)) if d != root else "." for d in dirs if d == root or root in d.parents]
    try:
        r = subprocess.run([git, "-C", str(root), "status", "--porcelain", "--untracked-files=no", "--", *rel],
                           capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return [ln[3:] for ln in r.stdout.splitlines() if ln.strip()]


def checkout_status(name: str, pip: str, version_pin: str, module: Any = None) -> dict[str, Any] | None:
    """For a package importable WITHOUT distribution metadata: ``{ok, version, root, head, commit_pin,
    declared_version, modified, reason}``; ``None`` when the ``pip`` line pins no git commit or nothing named
    ``name`` is importable (the metadata route alone decides then). ``ok`` needs the checkout HEAD to be the pinned
    commit, no modified tracked file in the package directory (when git can tell) and a declared version, if any,
    equal to ``version_pin``. ``version`` is ``"<pin>+<commit[:8]> (git checkout)"``."""
    pin = pinned_commit(pip)
    if pin is None:
        return None
    dirs = _module_dirs(name, module)
    if not dirs:
        return None
    out: dict[str, Any] = {"ok": False, "version": None, "root": None, "head": None, "commit_pin": pin,
                           "declared_version": None, "modified": None, "reason": ""}
    found = _git_head(dirs[0])
    if found is None:
        out["reason"] = f"{dirs[0]} is not inside a git checkout, so its commit cannot be checked against {pin}"
        return out
    root, head = found
    out.update(root=str(root), head=head)
    if not head or not head.startswith(pin):
        out["reason"] = f"the checkout's HEAD is {head or '(unreadable)'}, the pin is {pin}"
        return out
    declared = _declared_version(root)
    out["declared_version"] = declared
    if declared is not None and declared != str(version_pin):
        out["reason"] = f"the checkout declares version {declared}, the pin is {version_pin}"
        return out
    modified = _modified(root, [d for d in dirs if d == root or root in d.parents])
    out["modified"] = modified
    if modified:
        out["reason"] = f"the checkout has modified tracked files {modified[:5]}"
        return out
    out.update(ok=True, version=f"{version_pin}+{head[:8]} (git checkout)",
               reason="git checkout at the pinned commit" + ("" if modified is not None else " (git status unavailable: local edits unchecked)"))
    return out


def _hint(pip: str) -> str:
    """The runnable install command: ``external[].pip`` is the full command (``bench lint`` checks it)."""
    return str(pip).strip()


def record_version(name: str, version: str) -> None:
    """Record a more precise identity of a package :func:`require` returned: a folder whose pin is
    finer than the package version (Probe3D reports ``1.0`` at every commit; the folder checks the commit's
    file digest) records e.g. ``"1.0+a1f14640 (metrics.py sha256 39c0c919f047)"`` after its own check, and
    that string is what ``result.json[external]`` and the results file's ``config.external`` carry."""
    if name not in _USED:
        raise KeyError(f"record_version({name!r}): require() it first")
    _USED[name] = str(version)


def versions_used() -> dict[str, str]:
    """Every package :func:`require` returned in this process, ``{name: version}``."""
    return dict(_USED)


def reset() -> None:
    """Forget :func:`versions_used` (tests)."""
    _USED.clear()
