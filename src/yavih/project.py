"""The project folder ``yavih-bench/``: ``yavih init`` writes it, ``yavih doctor`` checks it.

One folder per repository holds everything a project keeps for yavih, text only::

    yavih-bench/
      yavih.toml      [project] (the yavih requirement, relocations, subprojects), data roots, one [site.<id>] per
                      cluster with its launcher(s)
      README.md       what lives here and where runs go
      .gitignore      runs/, logs/, __pycache__/ (results/ is committed)
      bridges/        one file per (model, benchmark) plus the shared <model>_common.py
      matrices/       <model>.yaml and <model>.profiles/ next to it
      suites/         project suites
      benchmarks/     project-local benchmark folders
      tools/          the project's own glue (drivers, generators, sync scripts); yavih never reads it
      runs/           the default runs root (ignored by git; plan files under runs/_plans/)
      results/        the default results root: results/<model>/results_<timestamp>.json, small text, committed

Runs and results default to ``yavih-bench/runs`` and ``yavih-bench/results`` (:meth:`yavih.config.Settings.runs_root`);
a ``[runs] root`` in the user file (``~/.config/yavih/config.toml``) or in a ``[site.<id>]`` table puts the runs of a
large suite on scratch. ``init`` never writes the user file: it holds personal overrides only. ``doctor`` warns
(``W_RUNS_SIZE``) when the project-default runs root has grown past :data:`RUNS_SIZE_WARN_GB`.

Discovery (``yavih.config``) takes the nearest ``yavih-bench/yavih.toml`` (else ``yavih.toml``) walking up from the
current directory and stops at the git root: a repository has one project. ``doctor`` warns about a second project
file inside the repository (``W_PROJECT_NESTED``, unless listed in ``[project] subprojects``), a matrix file outside
the project folder (``W_PROJECT_STRAY``) and an installed yavih outside ``[project] yavih`` (``W_PROJECT_VERSION``).
Bridges shipped inside a model package (``module:Class`` in the matrix), a benchmark folder kept at the top level
of a repository (``[project] benchmarks`` or ``$YAVIH_BENCH_PATH``) and job scripts kept elsewhere are not strays.

Documentation: https://github.com/Bizilizi/yavih/blob/main/docs/project-layout.md
"""

from __future__ import annotations

import os
import re
import socket
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from yavih.config import (
    PROJECT_DIR,
    PROJECT_FILE,
    PROJECT_KINDS,
    ConfigError,
    git_root,
    user_config_path,
)

__all__ = ["ProjectError", "init_project", "layout_report", "check_requirement", "doctor_block", "runs_size", "DOCS",
           "RUNS_SIZE_WARN_GB"]

DOCS = "https://github.com/Bizilizi/yavih/blob/main/docs/project-layout.md"

#: Directories ``init`` creates under the project folder (each with a ``.gitkeep`` so git keeps it).
PROJECT_SUBDIRS = (*PROJECT_KINDS, "tools")
#: Directory names the layout scan never enters.
_SKIP_DIRS = {"__pycache__", "node_modules", "site-packages", "dist", "build", "_stale", "_report"}
_MATRIX_FILE = re.compile(r"^matrix[A-Za-z0-9_.-]*\.ya?ml$")
_SCAN_MAX_DEPTH = 8
_SCAN_MAX_DIRS = 50_000
#: ``doctor`` warns ``W_RUNS_SIZE`` when the project-default runs root (``yavih-bench/runs``) holds more than this.
RUNS_SIZE_WARN_GB = 50.0


class ProjectError(ValueError):
    """``yavih init`` refused (exit 1)."""

    def __init__(self, message: str, code: str = "E_USAGE") -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = 1


# ----------------------------------------------------------------------------- init


def _listed_dirs(file: Path) -> tuple[list[Path], list[Path]]:
    """``([project] subprojects, relocated [project] directories)`` of project file ``file``, resolved against the
    file's directory as those keys are documented (``~`` expanded); ``([], [])`` when the file does not parse."""
    from yavih.config import tomllib  # stdlib tomllib, or the 3.10 fallback

    try:
        table = tomllib.loads(file.read_text(encoding="utf-8")).get("project") or {}
    except (OSError, ValueError):
        return [], []

    def anchor(v: Any) -> Path | None:
        if not isinstance(v, str) or not v:
            return None
        q = Path(os.path.expanduser(v))
        return (q if q.is_absolute() else file.parent / q).resolve()

    subs = table.get("subprojects") if isinstance(table, dict) else None
    listed = [a for a in (anchor(v) for v in (subs if isinstance(subs, list) else [])) if a is not None]
    relocated = [a for a in (anchor(table.get(k)) for k in PROJECT_KINDS) if a is not None] if isinstance(table, dict) else []
    return listed, relocated


def _covered(path: Path, dirs: Iterable[Path]) -> bool:
    """``path`` is one of ``dirs`` or inside one."""
    r = path.resolve()
    return any(r == d or d in r.parents for d in dirs)


def _requirement_for(version: str) -> str:
    """``">=M,<M+1"`` for the installed major version ``M`` (``0`` for a pre-release checkout)."""
    m = re.match(r"(\d+)", version or "")
    major = int(m.group(1)) if m else 0
    return f">={major},<{major + 1}"


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def project_toml(*, version: str, site: str | None = None, hosts: Iterable[str] = (), launcher: str | None = None) -> str:
    """The ``yavih.toml`` ``init`` writes: the requirement, commented relocations, a commented data root, the launcher
    (inside the site table with ``site``, else at the top level, else a commented example) and one ``[site.<id>]``
    table (a real one with ``site``, else a commented example)."""
    hosts = list(hosts)
    lines = [
        "# yavih project configuration (`yavih init`). Layers, lowest first: the user file (~/.config/yavih/config.toml) →",
        "# this file → its active [site.<id>] table → YAVIH_* variables → command-line flags.",
        "# Project facts only: the yavih version, data roots, the model environments. Runs and results default to",
        "# yavih-bench/runs and yavih-bench/results; a personal [runs] root belongs in the user file. Schema: `yavih config show`,",
        "# https://github.com/Bizilizi/yavih/blob/main/docs/configuration.md.",
        "",
        "# The launcher is how a model environment's python is started (a path, or a prefix such as `conda run -n m python`):",
        "# the top-level default (a key, so it comes before any table), per site under [site.<id>], or by name under",
        "# [launchers], which a matrix entry refers to as `launcher: <name>` and gets on whichever site runs it.",
        (f"launcher = {_toml_str(launcher)}" if launcher and not site else "# launcher = \"/envs/mymodel/bin/python\""),
        "",
        "[project]",
        f"yavih = {_toml_str(_requirement_for(version))}            # the yavih this project expects; `yavih doctor` warns on a mismatch",
        "# bridges = \"bridges\"          # relocate a directory (relative to this file); also matrices, suites, benchmarks",
        "# subprojects = []             # nested project folders inside this repository that are not strays",
        "",
        "[data]",
        "# root = \"/data/datasets\"      # ${data.root}; one benchmark's root: [data.<bench>] <root> = \"...\"",
        "",
        "# [launchers]",
        "# mymodel = \"/envs/mymodel/bin/python\"",
        "",
    ]
    if site:
        lines += [f"[site.{site}]                   # active when YAVIH_SITE={site} or a hosts glob matches the hostname",
                  f"hosts = [{', '.join(_toml_str(h) for h in hosts) or _toml_str(socket.gethostname())}]"]
        if launcher:
            lines.append(f"launcher = {_toml_str(launcher)}")
        lines += [f"# [site.{site}.launchers]", "# mymodel = \"/envs/mymodel/bin/python\"",
                  f"# [site.{site}.data]", "# root = \"/cluster/datasets\""]
    else:
        lines += ["# One table per cluster; the first whose hosts glob matches the hostname is active (or YAVIH_SITE=<id>):",
                  "# [site.cluster_a]", "# hosts = [\"gpu*\", \"*.cluster-a.example\"]",
                  "# launcher = \"/envs/mymodel/bin/python\"", "# [site.cluster_a.launchers]",
                  "# mymodel = \"/envs/mymodel/bin/python\"", "# [site.cluster_a.data]", "# root = \"/cluster/datasets\""]
    return "\n".join(lines) + "\n"


PROJECT_README = """# yavih-bench

Everything this repository keeps for [yavih](https://github.com/Bizilizi/yavih), text only. One such folder per
repository; `yavih` finds it from any directory below the repository root.

| path | holds |
|---|---|
| `yavih.toml` | the project configuration: the yavih version it expects, data roots, one `[site.<id>]` per cluster with its launcher(s) |
| `bridges/` | one bridge file per model and benchmark (`yavih bridge new`, `yavih plan init`) and the shared `<model>_common.py` |
| `matrices/` | one matrix per model (`matrices/<model>.yaml`) with its profiles next to it (`<model>.profiles/`) |
| `suites/` | project benchmark suites (`yavih suite new`) |
| `benchmarks/` | project-local benchmark folders (`yavih bench new`), resolved by name |
| `tools/` | the project's own glue: drivers, matrix generators, sync scripts; yavih never reads it |
| `runs/` | the default runs root: run directories, features, plan files (`runs/_plans/`); ignored by git |
| `results/` | the default results root: `results/<model>/results_<timestamp>.json`, small text files, committed |

`yavih config show` prints the roots in effect. For a large suite set `[runs] root` in the user file
(`~/.config/yavih/config.toml`) or in a `[site.<id>]` table to put the runs on scratch; results stay here.

Bridges shipped inside a model package (`script: mymodel.yavih:Bridge`), a benchmark folder kept at the top level
of the repository (`[project] benchmarks`) and job scripts kept with the rest of the project's infrastructure are
fine outside this folder; a second `yavih.toml` or a stray `matrix.yaml` elsewhere is what `yavih doctor` warns about.

Commands: `yavih doctor` (checks this folder), `yavih plan init --suite default --model <name>` (it prints the
`yavih bridge check` and `yavih plan build matrices/<name>.yaml --out <runs root>/_plans/<name>.jsonl` commands).
Layout reference: {docs}
"""

#: Only what yavih itself creates inside the folder: the default runs root (features, plan files), the sbatch logs
#: and bytecode. ``results/`` is not listed: results files are small text, committed with the project.
GITIGNORE = "runs/\nlogs/\n__pycache__/\n"


def init_project(dir: str | os.PathLike | None = None, *, site: str | None = None, hosts: Iterable[str] = (),
                 launcher: str | None = None, force: bool = False) -> dict[str, Any]:
    """Write ``<dir>/yavih-bench/`` (``dir`` default: the current directory): ``yavih.toml`` (never overwritten),
    ``README.md`` and ``.gitignore`` (rewritten with ``force``) and the subdirectories with a ``.gitkeep`` each. The
    user file (``~/.config/yavih/config.toml``) is never written: runs and results default to the folder's ``runs/``
    and ``results/``. Refuses (``E_PROJECT_EXISTS``) when another project file already governs ``dir`` and does not
    list it (or a folder containing it) in ``[project] subprojects`` (one project per repository; a listed subproject
    gets its own folder, which discovery from inside it then finds first), and (``E_EXISTS``) when the folder's
    ``yavih.toml`` exists, unless ``force`` refreshes the other files. Returns ``{dir, file, written, kept, existing,
    runs_root, results_root, next}``."""
    from yavih import config
    from yavih.versions import __version__

    base = Path(dir).expanduser().resolve() if dir is not None else config._cwd().resolve()
    folder = base / PROJECT_DIR
    file = folder / PROJECT_FILE
    if site is not None and not re.match(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$", site):
        raise ProjectError(f"--site {site!r}: a site id matches [A-Za-z][A-Za-z0-9_-]*", "E_ARG")
    hosts = list(hosts)
    if hosts and site is None:
        raise ProjectError("--hosts names the hosts of a site table: pass --site <id> as well", "E_ARG")
    existing = config._discover_project(base)
    if existing is not None and existing.resolve() != file.resolve() and not _covered(base, _listed_dirs(existing)[0]):
        raise ProjectError(f"{existing} already governs {base} (one project per repository; discovery stops at the git root). "
                           f"Work in that project, or list this folder in its [project] subprojects", "E_PROJECT_EXISTS")
    if file.exists() and not force:
        raise ProjectError(f"{file} exists; pass --force to refresh README.md, .gitignore and the directories (yavih.toml is never overwritten)",
                           "E_EXISTS")
    written: list[str] = []
    kept: list[str] = []
    folder.mkdir(parents=True, exist_ok=True)
    if file.exists():
        kept.append(str(file))
    else:
        file.write_text(project_toml(version=__version__, site=site, hosts=hosts, launcher=launcher), encoding="utf-8")
        written.append(str(file))
    for name, text in (("README.md", PROJECT_README.replace("{docs}", DOCS)), (".gitignore", GITIGNORE)):
        p = folder / name
        if p.exists() and not force:
            kept.append(str(p))
        else:
            p.write_text(text, encoding="utf-8")
            written.append(str(p))
    for sub in PROJECT_SUBDIRS:
        d = folder / sub
        d.mkdir(exist_ok=True)
        keep = d / ".gitkeep"
        if not keep.exists() and not any(d.iterdir()):
            keep.write_text("", encoding="utf-8")
            written.append(str(keep))
    try:
        settings = config.load(refresh=True, cwd=base)  # the new file (discovered from its own repository) must parse under the strict schema
    except ConfigError as e:  # pragma: no cover - the template is valid; a pre-existing user file may not be
        raise ProjectError(f"{file} written, but the configuration does not load: {e}") from e
    return {"dir": str(folder), "file": str(file), "written": written, "kept": kept,
            "existing": str(existing) if existing is not None else None,
            "runs_root": str(settings.runs_root()), "results_root": str(settings.results_root()),
            "user_config": str(user_config_path()[0]),
            "next": [f"edit {file}: data roots and one [site.<id>] per cluster with its launcher",
                     "yavih doctor", "yavih plan init --suite default --model <name>"]}


# ----------------------------------------------------------------------------- version requirement


def _release(version: str) -> tuple[int, ...]:
    m = re.match(r"v?(\d+(?:\.\d+)*)", version.strip())
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def _cmp(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    n = max(len(a), len(b))
    a2, b2 = a + (0,) * (n - len(a)), b + (0,) * (n - len(b))
    return (a2 > b2) - (a2 < b2)


def _minimal_check(spec: str, version: str) -> bool:
    """A release-only reading of a PEP 440 specifier set (``>=``, ``<=``, ``==``, ``!=``, ``>``, ``<``, ``~=``), used
    when ``packaging`` is not installed; pre-release and local segments are ignored."""
    v = _release(version)
    for clause in (c.strip() for c in spec.split(",") if c.strip()):
        m = re.match(r"^(~=|==|!=|<=|>=|<|>)\s*v?(\d+(?:\.\d+)*)(?:\.\*)?$", clause)
        if not m:
            raise ValueError(f"unsupported requirement clause {clause!r}")
        op, want = m.group(1), _release(m.group(2))
        c = _cmp(v, want)
        if op == "~=":  # ~=X.Y → >=X.Y,<X+1; ~=X.Y.Z → >=X.Y.Z,<X.Y+1
            upper = (*want[:-2], want[-2] + 1) if len(want) > 1 else (want[0] + 1,)
            ok = c >= 0 and _cmp(v, upper) < 0
        else:
            ok = {"==": c == 0, "!=": c != 0, "<=": c <= 0, ">=": c >= 0, "<": c < 0, ">": c > 0}[op]
        if not ok:
            return False
    return True


def check_requirement(spec: str | None, version: str) -> dict[str, Any]:
    """``{spec, version, ok, dev, reason}``: whether the installed ``version`` satisfies ``[project] yavih``. ``ok`` is
    ``None`` without a requirement or when the clause cannot be read; a development build (``.devN``, a dirty tree, a
    checkout without tags) sets ``dev`` and is reported, never refused (``ok`` stays ``True`` when the release part
    fits). Uses ``packaging`` when installed, else a release-only reading of the specifier."""
    dev = bool(re.search(r"\.dev\d*|\+|^0\.0\.0", version or ""))
    out: dict[str, Any] = {"spec": spec, "version": version, "ok": None, "dev": dev, "reason": None}
    if not spec:
        return out
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        ok = Version(version) in SpecifierSet(spec, prereleases=True)
        how = "packaging"
    except ImportError:
        try:
            ok = _minimal_check(spec, version)
            how = "release-only"
        except ValueError as e:
            out["reason"] = str(e)
            return out
    except Exception as e:  # noqa: BLE001 - an unreadable specifier or version is a finding, not a crash
        out["reason"] = f"{e.__class__.__name__}: {e}"
        return out
    out["ok"] = bool(ok)
    out["reason"] = f"{version} {'satisfies' if ok else 'does not satisfy'} {spec} ({how})"
    return out


# ----------------------------------------------------------------------------- layout


def _scan_root(settings: Any, cwd: Path) -> Path | None:
    """The directory the layout scan covers: the repository root, else (outside any repository) the directory a
    ``yavih-bench/`` folder sits in, else the bare project's own directory (a sibling directory with its own
    ``yavih.toml`` is another project, not a nested one), else ``None``."""
    project = settings.project_dir()
    start = project if project is not None else cwd
    root = git_root(start)
    if root is not None:
        return root
    if project is None:
        return None
    return project.parent if project.name == PROJECT_DIR else project


def layout_report(settings: Any, cwd: str | os.PathLike | None = None) -> dict[str, Any]:
    """``{dir, file, root, nested, stray, skipped}``: the project folder, the repository root the scan covers, other
    project files inside it (``nested``: not listed in ``[project] subprojects``) and matrix files outside the project
    folder and its relocated directories (``stray``). A project file that lists this project in its own ``[project]
    subprojects`` is its parent, not nested: its folder, relocated directories and other subprojects are not reported.
    The scan never enters dot-directories, virtual environments, ``__pycache__``, build and ``node_modules`` trees, the
    runs and results roots in effect (configured or the project defaults), or deeper than 8 levels; ``skipped`` says
    when the directory cap stopped it."""
    from yavih import config

    cwd_path = Path(cwd) if cwd is not None else config._cwd()
    project = settings.project_dir()
    pfile = settings.project_file()
    root = _scan_root(settings, cwd_path)
    out: dict[str, Any] = {"dir": str(project) if project else None, "file": str(pfile.path) if pfile else None,
                           "root": str(root) if root else None, "nested": [], "stray": [], "skipped": False}
    if root is None:
        return out
    allowed = {p.resolve() for p in settings.subprojects()}
    if project is not None:
        allowed.add(project.resolve())
        for kind in PROJECT_KINDS:  # a relocated project directory is where its files belong, never a stray
            try:
                allowed.add(settings.project_path(kind).resolve())
            except ConfigError:
                continue
    avoid = set()
    for getter in (settings.runs_root, settings.results_root):
        try:
            avoid.add(Path(getter()).resolve())
        except (ConfigError, OSError):
            pass
    found: list[Path] = []
    matrices: list[Path] = []
    n_dirs = 0
    root_depth = len(root.resolve().parts)
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        here = Path(dirpath)
        n_dirs += 1
        if n_dirs > _SCAN_MAX_DIRS:
            out["skipped"] = True
            break
        keep = []
        for d in dirnames:
            sub = here / d
            if d.startswith(".") or d in _SKIP_DIRS or (sub / "pyvenv.cfg").exists():
                continue
            try:
                r = sub.resolve()
            except OSError:
                continue
            if r in avoid or len(r.parts) - root_depth >= _SCAN_MAX_DEPTH:
                continue
            if project is not None and r == project.resolve():
                continue  # the project folder itself holds the matrices and config by design
            keep.append(d)
        dirnames[:] = keep
        for name in filenames:
            p = here / name
            if name == PROJECT_FILE and (pfile is None or p.resolve() != Path(pfile.path).resolve()):
                found.append(p)
            elif _MATRIX_FILE.match(name) and project is not None:
                matrices.append(p)
    if project is not None:  # a project that lists this one as a subproject governs its siblings, not a stray
        for p in found:
            listed, relocated = _listed_dirs(p)
            if _covered(project.parent, listed):
                allowed.update({p.parent.resolve(), *listed, *relocated})
    out["nested"] = sorted(str(p) for p in found if not _covered(p.parent, allowed))
    out["stray"] = sorted(str(p) for p in matrices if not _covered(p.parent, allowed))
    return out


def runs_size(root: str | os.PathLike, *, stop_at: float | None = None) -> int:
    """Bytes under ``root`` (``du``-style: regular files, symlinks not followed, a missing root is 0). With ``stop_at``
    the walk stops as soon as the total exceeds it, so a root far over the threshold costs little to check."""
    total = 0
    stack = [str(root)]
    while stack:
        here = stack.pop()
        try:
            with os.scandir(here) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
        if stop_at is not None and total > stop_at:
            break
    return total


def _runs_size_finding(settings: Any, project: Path) -> tuple[str, str, str] | None:
    """``W_RUNS_SIZE`` when the runs root is the project default (under the project folder) and holds more than
    :data:`RUNS_SIZE_WARN_GB`; ``None`` for a configured root elsewhere or a small one."""
    try:
        root = Path(settings.runs_root()).resolve()
    except (ConfigError, OSError):
        return None
    if not (project.resolve() == root or project.resolve() in root.parents) or not root.is_dir():
        return None
    limit = RUNS_SIZE_WARN_GB * 1e9
    size = runs_size(root, stop_at=limit)
    if size <= limit:
        return None
    return ("W_RUNS_SIZE", f"the runs root {root} holds more than {RUNS_SIZE_WARN_GB:g} GB inside the project folder. Move it "
            f"to scratch: set `[runs] root = \"/scratch/<you>/yavih-runs\"` in {user_config_path(settings.env)[0]} (or under a "
            f"[site.<id>] table), `mv {root} <that path>`, then rebuild the plans (`yavih plan build … --out "
            f"<new root>/_plans/<model>.jsonl`); results stay in {Path(settings.results_root())}", str(root))


def doctor_block(settings: Any, cwd: str | os.PathLike | None = None) -> tuple[dict[str, Any], list[tuple[str, str, str | None]], list[str]]:
    """The ``project`` block of ``yavih doctor``: the layout report plus the version check and the runs-root size
    check (``W_RUNS_SIZE``), with the warnings (``(code, message, where)``) and notes to attach."""
    from yavih.versions import __version__

    warnings: list[tuple[str, str, str | None]] = []
    notes: list[str] = []
    data = layout_report(settings, cwd)
    data["requirement"] = check_requirement(settings.project_requirement(), __version__)
    if data["dir"] is None:
        notes.append(f"no project folder: `yavih init` writes {PROJECT_DIR}/ (yavih.toml, bridges/, matrices/, suites/, benchmarks/, tools/); see {DOCS}")
        return data, warnings, notes
    req = data["requirement"]
    if req["ok"] is False:
        warnings.append(("W_PROJECT_VERSION", f"installed yavih {req['version']} is outside the project's requirement {req['spec']!r} "
                         f"([project] yavih in {data['file']}): install the version the project expects, or update the requirement", data["file"]))
    elif req["spec"] and req["ok"] is None:
        warnings.append(("W_PROJECT_VERSION", f"[project] yavih = {req['spec']!r} cannot be checked: {req['reason']}", data["file"]))
    if req["dev"]:
        notes.append(f"yavih {req['version']} is a development build (not a release tag); the requirement {req['spec'] or 'none'} is advisory for it")
    for p in data["nested"]:
        warnings.append(("W_PROJECT_NESTED", f"a second project file inside the repository: {p} (one project per repository; "
                         f"discovery stops at the git root, so commands run below it use that file). Merge it into {data['dir']}, "
                         f"or list its folder in [project] subprojects of {data['file']}", p))
    for p in data["stray"]:
        warnings.append(("W_PROJECT_STRAY", f"a matrix file outside the project folder: {p}; matrices live in "
                         f"{settings.project_path('matrices')} (`yavih plan init` writes them there)", p))
    if data["skipped"]:
        notes.append(f"layout scan stopped after {_SCAN_MAX_DIRS} directories under {data['root']}")
    size = _runs_size_finding(settings, Path(data["dir"]))
    if size is not None:
        warnings.append(size)
    return data, warnings, notes
