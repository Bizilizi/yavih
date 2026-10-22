"""The project folder ``yavih-bench/``: ``yavih init`` writes it, ``yavih doctor`` checks it.

One folder per repository holds everything a project keeps for yavih, text only::

    yavih-bench/
      yavih.toml      [project] (the yavih requirement, relocations, subprojects), data roots, one [site.<id>] per
                      cluster with its launcher(s)
      README.md       what lives here and where runs go
      .gitignore      __pycache__/, logs/, _stale/
      bridges/        one file per (model, benchmark) plus the shared <model>_common.py
      matrices/       <model>.yaml and <model>.profiles/ next to it
      suites/         project suites
      benchmarks/     project-local benchmark folders
      tools/          the project's own glue (drivers, generators, sync scripts); yavih never reads it

Runs, results, scratch and plan state live at the configured roots, never here. Personal roots belong in the
user file (``~/.config/yavih/config.toml``), which ``init`` writes when absent, so the project file can be committed.

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

__all__ = ["ProjectError", "init_project", "layout_report", "check_requirement", "doctor_block", "DOCS"]

DOCS = "https://github.com/Bizilizi/yavih/blob/main/docs/project-layout.md"

#: Directories ``init`` creates under the project folder (each with a ``.gitkeep`` so git keeps it).
PROJECT_SUBDIRS = (*PROJECT_KINDS, "tools")
#: Directory names the layout scan never enters.
_SKIP_DIRS = {"__pycache__", "node_modules", "site-packages", "dist", "build", "_stale", "_report"}
_MATRIX_FILE = re.compile(r"^matrix[A-Za-z0-9_.-]*\.ya?ml$")
_SCAN_MAX_DEPTH = 8
_SCAN_MAX_DIRS = 50_000


class ProjectError(ValueError):
    """``yavih init`` refused (exit 1)."""

    def __init__(self, message: str, code: str = "E_USAGE") -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = 1


# ----------------------------------------------------------------------------- init


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
        "# Project facts only: the yavih version, data roots, the model environments. Personal roots (runs, results,",
        "# scratch) belong in the user file, so this file can be committed. Schema: `yavih config show`,",
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

Runs, features, results, scratch and plan state are written at the configured roots, never here
(`yavih config show` prints them; personal roots belong in `~/.config/yavih/config.toml`).

Bridges shipped inside a model package (`script: mymodel.yavih:Bridge`), a benchmark folder kept at the top level
of the repository (`[project] benchmarks`) and job scripts kept with the rest of the project's infrastructure are
fine outside this folder; a second `yavih.toml` or a stray `matrix.yaml` elsewhere is what `yavih doctor` warns about.

Commands: `yavih doctor` (checks this folder), `yavih plan init --suite default --model <name>` (it prints the
`yavih bridge check` and `yavih plan build matrices/<name>.yaml --out <runs root>/_plans/<name>.jsonl` commands).
Layout reference: {docs}
"""

#: ``*.jsonl``: plan files belong under the runs root (``plan init`` prints ``--out <runs root>/_plans/<model>.jsonl``);
#: a plan built here anyway is run state, not a project fact.
GITIGNORE = "__pycache__/\n*.pyc\nlogs/\n_stale/\n# plan files (yavih plan build --out) are run state: keep them under the runs root\n*.jsonl\n"

USER_CONFIG = """# yavih user configuration (`yavih init`): personal roots, kept out of the committed project file.
# Site tables work here too ([site.<id>] with hosts = [...]). Schema: `yavih config show`.
[runs]
root = "~/yavih/runs"                   # run directories <runs root>/<benchmark>/<model>/<reading>/
[results]
root = "~/yavih/results"                # the results store, results/<model>/results_<timestamp>.json
# [scratch]
# root = "/scratch/$USER/yavih"         # staged probe shards; default $SLURM_TMPDIR
"""


def init_project(dir: str | os.PathLike | None = None, *, site: str | None = None, hosts: Iterable[str] = (),
                 launcher: str | None = None, force: bool = False, user_file: bool = True) -> dict[str, Any]:
    """Write ``<dir>/yavih-bench/`` (``dir`` default: the current directory): ``yavih.toml`` (never overwritten),
    ``README.md`` and ``.gitignore`` (rewritten with ``force``), the subdirectories with a ``.gitkeep`` each, and the
    user config file with personal roots when it does not exist (``user_file``). Refuses (``E_PROJECT_EXISTS``) when
    another project file already governs ``dir`` (one project per repository; nested ones go in ``[project]
    subprojects``), and (``E_EXISTS``) when the folder's ``yavih.toml`` exists, unless ``force`` refreshes the other
    files. Returns ``{dir, file, written, kept, user_config, user_config_written, existing}``."""
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
    if existing is not None and existing.resolve() != file.resolve():
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
    upath, _how = user_config_path()
    user_written = False
    if user_file and not upath.exists():
        upath.parent.mkdir(parents=True, exist_ok=True)
        upath.write_text(USER_CONFIG, encoding="utf-8")
        user_written = True
    try:
        config.load(refresh=True, cwd=base)  # the new file (discovered from its own repository) must parse under the strict schema
    except ConfigError as e:  # pragma: no cover - the template is valid; a pre-existing user file may not be
        raise ProjectError(f"{file} written, but the configuration does not load: {e}") from e
    return {"dir": str(folder), "file": str(file), "written": written, "kept": kept, "user_config": str(upath),
            "user_config_written": user_written, "existing": str(existing) if existing is not None else None,
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
    project = settings.project_dir()
    start = project if project is not None else cwd
    root = git_root(start)
    return root if root is not None else (project.parent if project is not None else None)


def layout_report(settings: Any, cwd: str | os.PathLike | None = None) -> dict[str, Any]:
    """``{dir, file, root, nested, stray, skipped}``: the project folder, the repository root the scan covers, other
    project files inside it (``nested``: not listed in ``[project] subprojects``) and matrix files outside the project
    folder and its relocated directories (``stray``). The scan never enters dot-directories, virtual environments, ``__pycache__``, build and
    ``node_modules`` trees, the configured runs / results roots, or deeper than 8 levels; ``skipped`` says when the
    directory cap stopped it."""
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
        in_allowed = any(here.resolve() == a or a in here.resolve().parents for a in allowed)
        for name in filenames:
            p = here / name
            if name == PROJECT_FILE and (pfile is None or p.resolve() != Path(pfile.path).resolve()) and not in_allowed:
                out["nested"].append(str(p))
            elif _MATRIX_FILE.match(name) and project is not None and not in_allowed:
                out["stray"].append(str(p))
    out["nested"].sort()
    out["stray"].sort()
    return out


def doctor_block(settings: Any, cwd: str | os.PathLike | None = None) -> tuple[dict[str, Any], list[tuple[str, str, str | None]], list[str]]:
    """The ``project`` block of ``yavih doctor``: the layout report plus the version check, with the warnings
    (``(code, message, where)``) and notes to attach."""
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
    return data, warnings, notes
