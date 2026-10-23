"""Site configuration: one typed :class:`Settings` per process.

Layers, lowest to highest precedence (every later layer overrides an earlier one, key by key):

1. built-in defaults and ``benchmark.yaml`` root defaults (``${data.root}``, ``${datasets.<name>}``);
2. the USER file ``$XDG_CONFIG_HOME/yavih/config.toml`` (else ``~/.config/yavih/config.toml``);
3. the active ``[site.<id>]`` table of the user file;
4. the PROJECT file: ``--config FILE``, else ``$YAVIH_CONFIG``, else the nearest ``yavih-bench/yavih.toml``
   (else ``yavih.toml``) found walking up from the current directory, stopping at the git root;
5. the active ``[site.<id>]`` table of the project file;
6. environment variables ``YAVIH_*`` (:data:`ENV_VARS`, ``YAVIH_DATA__<BENCH>__<ROOT>``,
   ``YAVIH_DATASETS__<NAME>``);
7. CLI flags (``yavih --config F --runs-root D --results D --data <bench>.<root>=PATH``), installed
   by :func:`set_cli`; programmatic ``Benchmark.load(data=...)`` overrides sit above everything.

The active site is ``$YAVIH_SITE`` or the first ``[site.<id>]`` whose ``hosts`` globs match
``socket.gethostname()``. A site table also carries the model environments of that cluster:
``launcher`` (the default) and ``[launchers] name = command`` (named ones a matrix entry refers to).
The project file's ``[project]`` table pins the yavih version it expects (``yavih = ">=1,<2"``,
checked by ``doctor``), relocates the project directories (``bridges``, ``matrices``, ``suites``,
``benchmarks``) and lists ``subprojects`` (nested project folders that are not strays). Values get ``~`` and ``$VAR`` expansion; a relative path in a file is
anchored to that file's directory; ``${data.root}`` and ``${datasets.<name>}`` are substituted in
yaml defaults and config values alike (``${datasets.X}`` = ``[datasets] X``, else ``${data.root}/X``).

The schema is strict: an unknown table, key, benchmark (a near miss of a known name such as
``[data.ESC50]``), root of a known benchmark, or ``YAVIH_*`` variable is a :class:`ConfigError`
(exit 1). ``--lenient`` / ``YAVIH_LENIENT=1`` turns those errors into warnings: the escape for a
HOME shared by installs of different yavih versions, where a newer key must not brick an older
install. A ``[data.<name>]`` table naming no known benchmark at all is only a warning, because a
folder loaded by path may carry any name; its keys are checked when that folder is loaded.

Every value keeps its source (``cli``, ``env:NAME``, ``file:PATH:LINE``, ``yaml:PATH``,
``default``); ``yavih config show`` prints them. Comparability knobs (``ProbeConfig``, readout
parameters) are pinned in code and yaml and never configurable here: only site knobs are.

stdlib only: ``tomllib`` on Python >= 3.11; on 3.10 an installed ``tomli``, else yavih's small vendored reader
(``yavih._toml``), so ``pip install --no-deps`` into a 3.10 model env still reads the config.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import re
import socket
import sys
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised by the Python 3.10 install check
    try:
        import tomli as tomllib
    except ImportError:
        from yavih import _toml as tomllib

__all__ = [
    "ConfigError",
    "Value",
    "ConfigFile",
    "Settings",
    "load",
    "set_cli",
    "cli_overrides",
    "register_benchmark",
    "warn_once",
    "drain_warnings",
    "yavih_home",
    "runs_root",
    "results_root",
    "configured_results_root",
    "scratch_root",
    "configured_scratch_root",
    "config_path",
    "user_config_path",
    "load_config",
    "data_root",
    "dataset",
    "resolve_data",
    "resolve_data_sources",
    "folder_roots",
    "folder_root",
    "roots_digest",
    "data_env_var",
    "DATA_ENV_PREFIX",
    "DATASETS_ENV_PREFIX",
    "ENV_VARS",
    "PROJECT_FILE",
    "PROJECT_DIR",
    "PROJECT_KINDS",
    "git_root",
    "project_benchmarks_dir",
]

DATA_ENV_PREFIX = "YAVIH_DATA__"
DATASETS_ENV_PREFIX = "YAVIH_DATASETS__"
PROJECT_FILE = "yavih.toml"
#: The project folder ``yavih init`` writes and discovery looks for first: ``<repo>/yavih-bench/yavih.toml``.
PROJECT_DIR = "yavih-bench"
#: The directories a project keeps under :data:`PROJECT_DIR` (each relocatable by ``[project] <kind> = path``).
PROJECT_KINDS = ("bridges", "matrices", "suites", "benchmarks")
#: ``[project]`` keys.
_PROJECT_KEYS = ("yavih", *PROJECT_KINDS, "subprojects")
_LAUNCHER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")

#: Every ``YAVIH_*`` variable yavih reads or sets, with what it means. Anything else named ``YAVIH_*``
#: is an error (a warning with ``--lenient``), so a typo or an unknown variable never goes unnoticed.
ENV_VARS: dict[str, str] = {
    "YAVIH_HOME": "yavih home (fetched benchmarks, data_status/, media_info/); config [home] root; default ~/.yavih",
    "YAVIH_RUNS": "runs root; config [runs] root; default <project dir>/runs (yavih-bench/runs), else ./runs",
    "YAVIH_RESULTS": "results-store root; config [results] root; default <project dir>/results (yavih-bench/results), else <runs root>/../results",
    "YAVIH_SCRATCH": "scratch for staged shards; config [scratch] root; else SLURM_TMPDIR; never TMPDIR",
    "YAVIH_CONFIG": "the project config file (beats the yavih-bench/yavih.toml discovery); same as --config",
    "YAVIH_SITE": "the active [site.<id>] table (else the first whose hosts glob matches the hostname)",
    "YAVIH_LENIENT": "1 = unknown config keys / YAVIH_* variables / stale benchmark copies warn instead of failing",
    "YAVIH_BENCH_PATH": "colon-separated directories of out-of-tree benchmark folders (after $YAVIH_HOME/benchmarks, the packaged ones and the project's benchmarks/)",
    "YAVIH_SUITES": "directory of project suites (<name>.yaml; beats the project's suites/, else ./suites)",
    "YAVIH_LAUNCHER": "the default launcher (the model environment's python, or a prefix command); config `launcher` (top level or [site.<id>])",
    "YAVIH_DATA__ROOT": "the site data root (${data.root}); config [data] root",
    "YAVIH_RAM_BUDGET_GB": "RAM budget of a probe / validate in GiB; config [resources] ram_budget_gb",
    "YAVIH_CPUS": "CPUs this process may use (set per slot by `yavih exec --slots`); config [resources] cpus",
    "YAVIH_DECODE_WORKERS": "default decode workers; config [resources] decode_workers",
    "YAVIH_VALIDATE_WORKERS": "threads scanning shards in `features validate` (1 = one shard at a time); default min(shards, CPUs, RAM budget / largest shard tar); the report never depends on it",
    "YAVIH_JSON": "1 = every command emits one JSON envelope",
    "YAVIH_TESTING": "1 = non-pinned ProbeConfig allowed (results get standard: false)",
    "YAVIH_UNIT_ID": "set by `yavih exec` for the child: the plan unit id",
    "YAVIH_DEADLINE": "the deadline as an epoch (set by `yavih exec` for the child; inherited by extract / probe run)",
    "YAVIH_PROBE_CKPT_EVERY_S": "seconds between probe checkpoints (default 600)",
    "YAVIH_SLOT_GPU": "set by `yavih exec --slots` for each slot: its device mask, applied after the unit's env",
    "YAVIH_CLAIM_LEASE_S": "shard-claim heartbeat lease in seconds (default 900); a staler claim is taken over",
    "YAVIH_BENCH": "set by `yavih bench prepare <b> <step>` for the step it runs: the benchmark name (FOLDER = its folder)",
}

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ENV_REF = re.compile(r"\$(\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_PLACEHOLDER = re.compile(r"\$\{(data\.root|datasets\.([a-z][a-z0-9_]*))\}")

#: Allowed scalar keys per table (``[data]`` and ``[datasets]`` are free-form, checked separately).
_ROOT_TABLES = ("home", "runs", "results", "scratch")
_RESOURCE_KEYS: dict[str, type] = {"ram_budget_gb": float, "cpus": int, "decode_workers": int}
_TOP_TABLES = (*_ROOT_TABLES, "data", "datasets", "resources", "launchers", "project", "site")
_ENV_SCALARS: dict[str, tuple[str, ...]] = {
    "YAVIH_HOME": ("home", "root"),
    "YAVIH_RUNS": ("runs", "root"),
    "YAVIH_RESULTS": ("results", "root"),
    "YAVIH_SCRATCH": ("scratch", "root"),
    "YAVIH_DATA__ROOT": ("data", "root"),
    "YAVIH_RAM_BUDGET_GB": ("resources", "ram_budget_gb"),
    "YAVIH_CPUS": ("resources", "cpus"),
    "YAVIH_DECODE_WORKERS": ("resources", "decode_workers"),
    "YAVIH_LAUNCHER": ("launcher", "default"),
}


class ConfigError(ValueError):
    """A config file or data root could not be resolved (exit 1 territory)."""


# ----------------------------------------------------------------------------- values and files


@dataclass(frozen=True)
class Value:
    """One configured value and where it came from. ``base`` anchors a relative path (the directory of
    the file that set it); ``None`` leaves a relative path relative to the current directory."""

    value: Any
    source: str
    base: str | None = None
    order: int = 0  # position in layer order (later = higher precedence); set by load()

    def to_json(self) -> dict[str, Any]:
        return {"value": self.value, "source": self.source}


@dataclass
class ConfigFile:
    """A config file of the user or project layer, present or not."""

    path: Path
    layer: str  # "user" | "project"
    how: str  # "XDG_CONFIG_HOME" | "~/.config" | "--config" | "YAVIH_CONFIG" | "discovered" | "config="
    exists: bool
    data: dict = field(default_factory=dict)
    lines: dict[tuple[str, ...], int] = field(default_factory=dict)

    def where(self, *keys: str) -> str:
        line = self.lines.get(tuple(keys))
        return f"file:{self.path}:{line}" if line else f"file:{self.path}"

    def to_json(self) -> dict[str, Any]:
        return {"path": str(self.path), "layer": self.layer, "how": self.how, "exists": self.exists}


def _split_dotted(text: str) -> list[str]:
    parts, cur, quote = [], "", None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            else:
                cur += ch
        elif ch in "\"'":
            quote = ch
        elif ch == ".":
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts]


_HEADER_RE = re.compile(r"^\s*\[\s*([^\[\]]+?)\s*\]\s*(#.*)?$")
_KEY_RE = re.compile(r"^\s*((?:[A-Za-z0-9_\-]+|\"[^\"]*\"|'[^']*')(?:\s*\.\s*(?:[A-Za-z0-9_\-]+|\"[^\"]*\"|'[^']*'))*)\s*=")


def _key_lines(text: str) -> dict[tuple[str, ...], int]:
    """``(table..., key...) -> line`` for the plain TOML the config uses (headers, ``key = value``, dotted
    keys). Only for provenance (``file:PATH:LINE``); ``tomllib`` is what parses the values."""
    out: dict[tuple[str, ...], int] = {}
    table: tuple[str, ...] = ()
    for i, line in enumerate(text.splitlines(), 1):
        m = _HEADER_RE.match(line)
        if m and not line.lstrip().startswith("[["):
            table = tuple(_split_dotted(m.group(1)))
            out.setdefault(table, i)
            continue
        m = _KEY_RE.match(line)
        if m:
            out.setdefault(table + tuple(_split_dotted(m.group(1))), i)
    return out


def _read_file(path: Path, layer: str, how: str) -> ConfigFile:
    if not path.is_file():
        return ConfigFile(path=path, layer=layer, how=how, exists=False)
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"malformed config {path}: {e}") from e
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    return ConfigFile(path=path, layer=layer, how=how, exists=True, data=data, lines=_key_lines(raw.decode("utf-8")))


# ----------------------------------------------------------------------------- process state

_LOCK = threading.RLock()
_CLI: dict[str, Any] = {}
_CACHE: dict[str, Any] = {"key": None, "settings": None, "stamps": None}
_REGISTERED: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {}
_REGISTRY_CACHE: dict[str, Any] = {"key": None, "value": None}
_PROJECT_FILE_CACHE: dict[str, tuple[tuple, ConfigFile]] = {}
_WARNED: set[str] = set()
_PENDING_WARNINGS: list[dict[str, Any]] = []


def _cwd() -> Path:
    try:
        return Path.cwd()
    except OSError:  # the current directory was removed: no project discovery from it
        return Path("/")


def warn_once(code: str, message: str, where: str | None = None) -> None:
    """Print ``[yavih] warning CODE: message`` to stderr once per process and keep it for the CLI
    envelope (:func:`drain_warnings`)."""
    key = f"{code}\0{message}"
    with _LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
        _PENDING_WARNINGS.append({"code": code, "message": message, "where": where})
    try:
        sys.stderr.write(f"[yavih] warning {code}: {message}" + (f" ({where})" if where else "") + "\n")
    except (OSError, ValueError):  # pragma: no cover - closed stderr
        pass


def drain_warnings() -> list[dict[str, Any]]:
    """The warnings :func:`warn_once` emitted since the last call (the CLI adds them to its envelope)."""
    with _LOCK:
        out = list(_PENDING_WARNINGS)
        _PENDING_WARNINGS.clear()
    return out


def set_cli(
    *,
    config_file: str | os.PathLike | None = None,
    runs_root: str | os.PathLike | None = None,
    results: str | os.PathLike | None = None,
    data: Mapping[str, str] | Iterable[str] | None = None,
    lenient: bool | None = None,
) -> None:
    """Install the CLI layer (``yavih --config/--runs-root/--results/--data/--lenient``). ``data`` is
    ``{"<bench>.<root>": path}`` or ``["<bench>.<root>=PATH", ...]``; ``root=PATH`` sets ``[data] root``.
    Relative paths are made absolute against the current directory. Calling with no argument clears it."""
    cwd = _cwd()

    def absolute(p: Any) -> str:
        q = Path(os.path.expanduser(os.path.expandvars(str(p))))
        return str(q if q.is_absolute() else cwd / q)

    pairs: dict[str, str] = {}
    if isinstance(data, Mapping):
        pairs = {str(k): str(v) for k, v in data.items()}
    elif data is not None:
        for item in data:
            if "=" not in str(item):
                raise ConfigError(f"--data expects <bench>.<root>=PATH (or root=PATH), got {item!r}")
            k, v = str(item).split("=", 1)
            pairs[k.strip()] = v
    entries: dict[tuple[str, ...], str] = {}
    for k, v in pairs.items():
        parts = k.split(".")
        if parts == ["root"]:
            entries[("data", "root")] = absolute(v)
        elif len(parts) == 2 and parts[0] == "datasets":
            entries[("datasets", parts[1])] = absolute(v)
        elif len(parts) == 2 and all(parts):
            entries[("data", parts[0], parts[1])] = absolute(v)
        else:
            raise ConfigError(f"--data expects <bench>.<root>=PATH, root=PATH or datasets.<name>=PATH, got {k!r}")
    with _LOCK:
        _CLI.clear()
        if config_file is not None:
            _CLI["config_file"] = absolute(config_file)
        if runs_root is not None:
            entries[("runs", "root")] = absolute(runs_root)
        if results is not None:
            entries[("results", "root")] = absolute(results)
        if entries:
            _CLI["entries"] = entries
        if lenient:
            _CLI["lenient"] = True
        _CACHE["key"] = None


def cli_overrides() -> dict[str, Any]:
    """A copy of the installed CLI layer (``{config_file?, entries?, lenient?}``)."""
    with _LOCK:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in _CLI.items()}


def register_benchmark(name: str, aliases: Iterable[str] = (), roots: Mapping[str, Any] | None = None) -> None:
    """Make a benchmark known to the schema check (``Benchmark.from_dir`` calls this, so a folder loaded
    by path validates its ``[data.<name>]`` table and ``YAVIH_DATA__<NAME>__*`` variables)."""
    entry = (tuple(str(a) for a in aliases), dict(roots or {}))
    with _LOCK:
        if _REGISTERED.get(name) != entry:
            _REGISTERED[name] = entry
            _CACHE["key"] = None


# ----------------------------------------------------------------------------- benchmark registry


def _registry() -> dict[str, tuple[tuple[str, ...], dict[str, Any]]]:
    """``name -> (aliases, roots)`` over packaged folders, ``<home>/benchmarks`` and registered folders.
    Read from the yamls once per process (and again when the home directory changes)."""
    home = _home_from_env(os.environ)
    key = f"{home}|{os.environ.get('YAVIH_BENCH_PATH', '')}|{project_benchmarks_dir() or ''}"
    with _LOCK:
        if _REGISTRY_CACHE["key"] != key:
            _REGISTRY_CACHE["value"] = _scan_registry(home)
            _REGISTRY_CACHE["key"] = key
        out = dict(_REGISTRY_CACHE["value"])
        out.update(_REGISTERED)
    return out


def _scan_registry(home: Path) -> dict[str, tuple[tuple[str, ...], dict[str, Any]]]:
    import yaml  # core dependency; imported lazily so `import yavih.config` stays stdlib

    from yavih.benchmarks import BENCHMARK_YAML, external_folders, packaged_folder, packaged_names

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    folders: dict[str, Path] = {}
    for name, p in external_folders().items():  # $YAVIH_BENCH_PATH and entry-point packs: lowest precedence
        folders[name] = p
    for name in packaged_names():
        p = packaged_folder(name)
        if p is not None:
            folders[name] = p
    hb = home / "benchmarks"
    if hb.is_dir():
        for p in sorted(hb.iterdir()):
            if (p / BENCHMARK_YAML).is_file():
                folders[p.name] = p
    out: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {}
    for name, folder in folders.items():
        try:
            with open(folder / BENCHMARK_YAML, encoding="utf-8") as fh:
                d = yaml.load(fh, Loader=loader)
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(d, Mapping):
            continue
        aliases = d.get("aliases") if isinstance(d.get("aliases"), list) else []
        roots = ((d.get("data") or {}).get("roots") or {}) if isinstance(d.get("data"), Mapping) else {}
        out[str(d.get("name", name))] = (tuple(str(a) for a in aliases), dict(roots) if isinstance(roots, Mapping) else {})
    return out


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _canonical(name: str, registry: Mapping[str, tuple[tuple[str, ...], dict]]) -> str | None:
    if name in registry:
        return name
    for canon, (aliases, _roots) in registry.items():
        if name in aliases:
            return canon
    return None


def _near_miss(name: str, registry: Mapping[str, tuple[tuple[str, ...], dict]]) -> str | None:
    """A known name ``name`` is probably a typo of (case, separators, one or two letters)."""
    known = {n: n for n in registry}
    for canon, (aliases, _r) in registry.items():
        for a in aliases:
            known[a] = canon
    for k in known:
        if _norm(k) == _norm(name):
            return k
    close = difflib.get_close_matches(name.lower(), list(known), n=1, cutoff=0.8)
    return close[0] if close else None


# ----------------------------------------------------------------------------- discovery


def _home_from_env(env: Mapping[str, str]) -> Path:
    raw = env.get("YAVIH_HOME")
    return Path(os.path.expanduser(raw)) if raw else Path.home() / ".yavih"


def user_config_path(env: Mapping[str, str] | None = None) -> tuple[Path, str]:
    """``($XDG_CONFIG_HOME/yavih/config.toml | ~/.config/yavih/config.toml, how)``."""
    env = os.environ if env is None else env
    xdg = env.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(os.path.expanduser(xdg)) / "yavih" / "config.toml", "XDG_CONFIG_HOME"
    return Path.home() / ".config" / "yavih" / "config.toml", "~/.config"


def _is_repo_root(d: Path) -> bool:
    """``d`` holds a ``.git`` directory, or a ``.git`` file of a linked worktree. A submodule's ``.git`` file (its
    ``gitdir:`` points into the superproject's ``.git/modules/``) does not count: the superproject's project
    governs it."""
    g = d / ".git"
    if g.is_dir():
        return True
    if not g.is_file():
        return False
    try:
        head = g.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return True
    target = head[len("gitdir:"):].strip() if head.startswith("gitdir:") else ""
    return "/modules/" not in target.replace("\\", "/")


def git_root(start: Path) -> Path | None:
    """The nearest repository root at or above ``start`` (:func:`_is_repo_root`), else ``None``."""
    for d in (start, *start.parents):
        if _is_repo_root(d):
            return d
    return None


def _discover_project(cwd: Path) -> Path | None:
    """The nearest ``yavih-bench/yavih.toml``, else ``yavih.toml``, walking up from ``cwd``; the walk stops at the
    repository root (a repository has one project; a submodule checkout is walked through to its superproject), and
    runs to the filesystem root outside any repository."""
    for d in (cwd, *cwd.parents):
        for p in (d / PROJECT_DIR / PROJECT_FILE, d / PROJECT_FILE):
            if p.is_file():
                return p
        if _is_repo_root(d):
            return None
    return None


def project_benchmarks_dir(env: Mapping[str, str] | None = None, cwd: str | os.PathLike | None = None) -> Path | None:
    """The project's benchmark folders directory (``[project] benchmarks``, else ``benchmarks/`` next to the project
    file), without building the settings: read by the benchmark registry, which the settings build itself needs.
    ``None`` without a project file."""
    env = os.environ if env is None else env
    cwd_path = Path(cwd) if cwd is not None else _cwd()
    p, how = _project_path(env, cwd_path, cli_overrides() if env is os.environ else {})
    if p is None or not p.is_file():
        return None
    try:
        st = p.stat()
        stamp = (str(p), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    with _LOCK:  # parsed once per file version: this sits on the registry's cache key, i.e. on every settings build
        cached = _PROJECT_FILE_CACHE.get(str(p))
        if cached is not None and cached[0] == stamp:
            f = cached[1]
        else:
            try:
                f = _read_file(p, "project", how)
            except ConfigError:
                return None
            _PROJECT_FILE_CACHE[str(p)] = (stamp, f)
    proj = f.data.get("project")
    raw = proj.get("benchmarks") if isinstance(proj, Mapping) else None
    if isinstance(raw, str) and raw.strip():
        try:
            s = os.path.expanduser(_expand_env(raw, env, "[project] benchmarks", None))
        except ConfigError:
            return None
        return Path(s) if os.path.isabs(s) else Path(os.path.normpath(p.parent / s))
    return p.parent / "benchmarks"


def _project_path(env: Mapping[str, str], cwd: Path, cli: Mapping[str, Any]) -> tuple[Path | None, str]:
    if cli.get("config_file"):
        return Path(cli["config_file"]), "--config"
    raw = env.get("YAVIH_CONFIG")
    if raw:
        p = Path(os.path.expanduser(raw))
        return (p if p.is_absolute() else cwd / p), "YAVIH_CONFIG"
    found = _discover_project(cwd)
    return (found, "discovered") if found is not None else (None, "")


def config_path(env: Mapping[str, str] | None = None) -> Path:
    """The highest-precedence config file location: ``--config`` / ``$YAVIH_CONFIG`` / the discovered
    ``yavih.toml``, else the user file (where a missing key is best added)."""
    env = os.environ if env is None else env
    p, _how = _project_path(env, _cwd(), cli_overrides())
    return p if p is not None else user_config_path(env)[0]


def load_config(path: str | os.PathLike | None = None, env: Mapping[str, str] | None = None) -> dict:
    """Parse ONE TOML file (``path``, default :func:`config_path`): a missing file is an empty config, a
    malformed one is an error. The layered view is :func:`load`."""
    p = Path(path) if path is not None else config_path(env)
    return _read_file(p, "project", "config=").data


# ----------------------------------------------------------------------------- expansion


class _Collector:
    """Strict-schema findings: errors (or warnings when lenient) and warnings."""

    def __init__(self, lenient: bool) -> None:
        self.lenient = lenient
        self.errors: list[str] = []
        self.warnings: list[tuple[str, str]] = []

    def strict(self, code: str, message: str) -> None:
        if self.lenient:
            self.warnings.append((code, message + " (ignored: lenient)"))
        else:
            self.errors.append(message)

    def warn(self, code: str, message: str) -> None:
        self.warnings.append((code, message))


def _expand_env(raw: str, env: Mapping[str, str], where: str, col: _Collector | None) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(2) or m.group(3)
        if name in env:
            return env[name]
        if col is not None:
            col.strict("W_CONFIG_UNSET_VAR", f"{where}: ${name} is not set")
        return m.group(0)

    return os.path.expanduser(_ENV_REF.sub(sub, raw))


# ----------------------------------------------------------------------------- Settings


@dataclass
class Settings:
    """The resolved configuration of this process. Values are :class:`Value` (value + source); derived
    paths come from the methods. Built by :func:`load`; never mutate one."""

    files: list[ConfigFile]
    site: str | None
    site_source: str | None
    lenient: bool
    values: dict[tuple[str, ...], Value]
    warnings: list[tuple[str, str]]
    env: Mapping[str, str]
    cwd: Path

    # -- scalar roots ---------------------------------------------------------------------
    def get(self, *key: str) -> Value | None:
        return self.values.get(tuple(key))

    def path_of(self, v: Value) -> str:
        """``v`` with placeholders substituted, ``~`` expanded and a relative path anchored to its file."""
        s = self._subst(str(v.value), v.source)
        s = os.path.expanduser(s)
        if v.base is not None and not os.path.isabs(s):
            s = os.path.normpath(os.path.join(v.base, s))
        return s

    def yavih_home(self) -> Path:
        v = self.get("home", "root")
        return Path(self.path_of(v)) if v else Path.home() / ".yavih"

    def runs_root(self) -> Path:
        """``--runs-root`` → ``$YAVIH_RUNS`` → ``[runs] root`` → ``<project dir>/runs`` (a project is discovered) →
        ``<current directory>/runs``. Always absolute, so a matrix ``out: null`` and ``config get runs.root`` name the
        same directory."""
        v = self.get("runs", "root")
        if v:
            return Path(self.path_of(v))
        project = self.project_dir()
        if project is not None:
            return project / "runs"
        return Path(os.path.abspath(Path(self.cwd) / "runs"))

    def runs_root_source(self) -> str:
        """The ``source`` of :meth:`runs_root`: the configured value's, else ``default (project)`` / ``default (./runs)``."""
        v = self.get("runs", "root")
        if v:
            return v.source
        return "default (project)" if self.project_dir() is not None else "default (./runs)"

    def configured_results_root(self) -> Path | None:
        """``--results`` / ``$YAVIH_RESULTS`` / ``[results] root``, else ``None``."""
        v = self.get("results", "root")
        return Path(self.path_of(v)) if v else None

    def results_root(self, runs_root_path: str | os.PathLike | None = None) -> Path:
        """``--results`` → ``$YAVIH_RESULTS`` → ``[results] root`` → ``<project dir>/results`` (a project is discovered;
        results files are small text, kept with the project) → ``results/`` next to ``runs_root_path`` (the runs root a
        probe's run directory lives under), else next to :meth:`runs_root`."""
        configured = self.configured_results_root()
        if configured is not None:
            return configured
        project = self.project_dir()
        if project is not None:
            return project / "results"
        base = Path(runs_root_path).expanduser() if runs_root_path is not None else self.runs_root()
        return base.absolute().parent / "results"

    def results_root_source(self) -> str:
        """The ``source`` of :meth:`results_root`: the configured value's, else ``default (project)`` /
        ``default (<runs root>/../results)``."""
        v = self.get("results", "root")
        if v:
            return v.source
        return "default (project)" if self.project_dir() is not None else "default (<runs root>/../results)"

    def configured_scratch_root(self) -> Path | None:
        """``YAVIH_SCRATCH`` / ``[scratch] root`` (env beats files: explicit configuration beats the site's
        scheduler variable) → ``$SLURM_TMPDIR`` → ``None``."""
        v = self.get("scratch", "root")
        if v:
            return Path(self.path_of(v))
        raw = self.env.get("SLURM_TMPDIR")
        return Path(raw).expanduser() if raw else None

    def scratch_root(self, run_dir: str | os.PathLike | None = None) -> Path:
        """:meth:`configured_scratch_root` → ``run_dir`` (never ``TMPDIR``)."""
        root = self.configured_scratch_root()
        if root is not None:
            return root
        if run_dir is None:
            raise ConfigError(
                "no scratch: set YAVIH_SCRATCH or [scratch] root (or run inside SLURM with SLURM_TMPDIR) or pass "
                "run_dir; TMPDIR is deliberately not used"
            )
        return Path(run_dir)

    def data_root(self) -> str | None:
        v = self.get("data", "root")
        if v is None:
            return None
        if _PLACEHOLDER.search(str(v.value)):
            raise ConfigError(f"[data] root = {v.value!r} ({v.source}) cannot use ${{...}} placeholders")
        return self.path_of(v)

    def dataset(self, name: str, *, fallback: bool = True) -> str | None:
        """``[datasets] <name>``, else (``fallback``) ``${data.root}/<name>``, else ``None``."""
        v = self.get("datasets", name)
        if v is not None:
            return self.path_of(v)
        if not fallback:
            return None
        root = self.data_root()
        return os.path.join(root, name) if root is not None else None

    def resource(self, key: str) -> Value | None:
        return self.get("resources", key)

    # -- placeholders --------------------------------------------------------------------
    def _subst(self, value: str, where: str) -> str:
        def sub(m: re.Match) -> str:
            if m.group(1) == "data.root":
                root = self.data_root()
                if root is None:
                    raise ConfigError(f"{where}: {value!r} uses ${{data.root}} but no data root is configured; set [data] root "
                                      f"in {self.edit_hint()} or YAVIH_DATA__ROOT")
                return root
            name = m.group(2)
            v = self.get("datasets", name)
            if v is not None:
                inner = str(v.value)
                if _PLACEHOLDER.search(inner) and "datasets." in inner:
                    raise ConfigError(f"[datasets] {name} = {inner!r}: a dataset may use ${{data.root}} but not another ${{datasets.*}}")
                return self.path_of(v)
            root = self.data_root()
            if root is None:
                raise ConfigError(f"{where}: {value!r} uses ${{datasets.{name}}} but neither [datasets] {name} nor [data] root "
                                  f"is configured; set one in {self.edit_hint()} (or YAVIH_DATASETS__{name.upper()} / YAVIH_DATA__ROOT)")
            return os.path.join(root, name)

        return _PLACEHOLDER.sub(sub, value)

    # -- the project (yavih-bench/) ---------------------------------------------------------
    def project_file(self) -> ConfigFile | None:
        """The project config file that exists (``--config``, ``$YAVIH_CONFIG`` or discovered), else ``None``."""
        return next((f for f in self.files if f.layer == "project" and f.exists and f.how != "config="), None)

    def project_dir(self) -> Path | None:
        """The project directory: the directory holding the project file (``yavih-bench/``), else ``None``."""
        f = self.project_file()
        return Path(f.path).resolve().parent if f is not None else None

    def project_path(self, kind: str) -> Path | None:
        """Where the project keeps its ``kind`` (``bridges``, ``matrices``, ``suites``, ``benchmarks``): ``[project] <kind>``,
        else ``<project dir>/<kind>``; ``None`` without a project."""
        if kind not in PROJECT_KINDS:
            raise ValueError(f"project_path: unknown kind {kind!r}; one of {PROJECT_KINDS}")
        root = self.project_dir()
        if root is None:
            return None
        v = self.get("project", kind)
        return Path(self.path_of(v)) if v else root / kind

    def project_requirement(self) -> str | None:
        """``[project] yavih``: the version requirement the project expects (``doctor`` checks it), else ``None``."""
        v = self.get("project", "yavih")
        return str(v.value) if v else None

    def subprojects(self) -> list[Path]:
        """``[project] subprojects``: nested project directories that are not strays, anchored to the project file."""
        v = self.get("project", "subprojects")
        if not v:
            return []
        return [Path(self.path_of(Value(s, v.source, v.base))) for s in v.value]

    def launcher(self, name: str | None = None) -> str | None:
        """The configured launcher: with ``name`` the ``[launchers] name`` entry of the active layers, else the default
        ``launcher``; ``None`` when not configured. A launcher is a command (``~`` expanded), never anchored to a file."""
        v = self.get("launchers", name) if name is not None else self.get("launcher", "default")
        return os.path.expanduser(str(v.value)) if v else None

    def launchers(self) -> dict[str, str]:
        """Every named launcher of the active layers, ``name -> command``."""
        return {k[1]: os.path.expanduser(str(v.value)) for k, v in sorted(self.values.items()) if k[0] == "launchers"}

    def any_site_launcher(self) -> bool:
        """Whether a launcher is defined anywhere in the configuration: the active layers (``launcher``, ``[launchers]``,
        ``YAVIH_LAUNCHER``), or the top level or ANY ``[site.<id>]`` table of a config file, active here or not. A
        launcher written for another cluster's site table is that cluster's, not missing: ``plan init`` warns
        ``W_LAUNCHER_UNSET`` only when this is false."""
        if self.launcher() is not None or self.launchers():
            return True

        def defines(table: Any) -> bool:
            if not isinstance(table, Mapping):
                return False
            lv, named = table.get("launcher"), table.get("launchers")
            return (isinstance(lv, str) and bool(lv.strip())) or (isinstance(named, Mapping) and bool(named))

        for f in self.files:
            data = f.data if isinstance(f.data, Mapping) else {}
            sites = data.get("site")
            if defines(data) or (isinstance(sites, Mapping) and any(defines(b) for b in sites.values())):
                return True
        return False

    def edit_hint(self) -> str:
        project = next((f for f in self.files if f.layer == "project" and f.exists), None)
        user = next((f for f in self.files if f.layer == "user"), None)
        return str((project or user).path) if (project or user) else str(user_config_path(self.env)[0])

    # -- data roots ----------------------------------------------------------------------
    def _bench_values(self, bench: str, aliases: Iterable[str]) -> dict[str, Value]:
        names = (bench, *aliases)
        out: dict[str, Value] = {}
        # values are stored in layer order per (data, name, key); the later layer has already won per key,
        # so among a benchmark and its aliases the higher-precedence source wins
        for (t, *rest), v in self.values.items():
            if t != "data" or len(rest) != 2 or rest[0] not in names:
                continue
            key = rest[1]
            prev = out.get(key)
            if prev is None or v.order >= prev.order:
                out[key] = v
        return out

    def resolve_data_sources(
        self,
        bench: str,
        roots: Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
        *,
        aliases: Iterable[str] = (),
        strict_missing: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Per root ``{value, source, env_var, config_key, error}``; with ``strict_missing`` an unresolvable
        root raises, else it is reported in ``error`` with ``value: None`` (``doctor``)."""
        overrides = {k: v for k, v in dict(overrides or {}).items()}
        unknown = sorted(set(overrides) - set(roots))
        if unknown:
            raise ConfigError(f"unknown data root override(s) for '{bench}': {unknown}; declared: {sorted(roots)}")
        aliases = tuple(aliases) or _aliases_for(bench)
        configured = self._bench_values(bench, aliases)
        bad = sorted(set(configured) - set(roots))
        if bad:
            msgs = []
            for k in bad:
                close = difflib.get_close_matches(k, list(roots), n=1, cutoff=0.6)
                msgs.append(f"{k!r} ({configured[k].source})" + (f", did you mean {close[0]!r}?" if close else ""))
            message = f"unknown data root(s) of benchmark '{bench}': {'; '.join(msgs)}; declared roots: {sorted(roots)}"
            if self.lenient:
                warn_once("W_CONFIG_UNKNOWN_ROOT", message + " (ignored: lenient)")
            else:
                raise ConfigError(message)
        out: dict[str, dict[str, Any]] = {}
        for key, spec in roots.items():
            default = spec.get("default") if isinstance(spec, Mapping) else spec
            rec: dict[str, Any] = {"value": None, "source": None, "env_var": data_env_var(bench, key), "config_key": f"[data.{bench}] {key}", "error": None}
            try:
                if key in overrides and overrides[key] is not None:
                    rec["source"] = "argument"
                    path = os.path.expanduser(str(overrides[key]))
                elif key in configured:
                    v = configured[key]
                    rec["source"] = v.source
                    path = self.path_of(v)
                elif default is not None:
                    rec["source"] = f"yaml default {default}"
                    path = os.path.expanduser(self._subst(str(default), f"data root '{key}' of benchmark '{bench}'"))
                else:
                    raise ConfigError(f"data root '{key}' of benchmark '{bench}' has no default and is not configured")
                rec["value"] = path
            except ConfigError as e:
                msg = str(e)
                if key not in configured and key not in overrides:
                    msg += f"; or set this root alone: {rec['env_var']}, --data {bench}.{key}=PATH, or [data.{bench}] {key}"
                if strict_missing:
                    raise ConfigError(msg) from None
                rec["error"] = msg
            out[key] = rec
        return out

    def resolve_data(self, bench: str, roots: Mapping[str, Any], overrides: Mapping[str, Any] | None = None, *, aliases: Iterable[str] = ()) -> dict[str, str]:
        """``{root: path}``. A root that does not resolve raises, except a ``prepare_only`` root (read only by
        ``prepare/``), which is left out so it never blocks reading the benchmark."""
        recs = self.resolve_data_sources(bench, roots, overrides, aliases=aliases, strict_missing=False)
        out: dict[str, str] = {}
        for k, r in recs.items():
            if r["error"] is None:
                out[k] = r["value"]
            elif not (isinstance(roots.get(k), Mapping) and roots[k].get("prepare_only")):
                raise ConfigError(r["error"])
        return out

    # -- display -------------------------------------------------------------------------
    def show(self, bench: Any | None = None) -> dict[str, Any]:
        """``yavih config show`` data: files, site, every value with its source, and (``bench``: a
        ``Benchmark``) its roots with existence; derived resource defaults included."""
        from yavih import resources

        vals: dict[str, Any] = {}
        vals["home.root"] = {"value": str(self.yavih_home()), "source": (self.get("home", "root") or Value(None, "default")).source}
        vals["runs.root"] = {"value": str(self.runs_root()), "source": self.runs_root_source()}
        vals["results.root"] = {"value": str(self.results_root()), "source": self.results_root_source()}
        try:
            vals["scratch.root"] = {"value": str(self.scratch_root()), "source": (self.get("scratch", "root") or Value(None, "env:SLURM_TMPDIR")).source}
        except ConfigError:
            vals["scratch.root"] = {"value": None, "source": "unset (YAVIH_SCRATCH / [scratch] root / SLURM_TMPDIR)"}
        dv = self.get("data", "root")
        vals["data.root"] = {"value": self.data_root() if dv else None, "source": dv.source if dv else "unset"}
        for (t, *rest), v in sorted(self.values.items()):
            if t == "datasets":
                vals[f"datasets.{rest[0]}"] = {"value": self._safe_path(v), "source": v.source}
            elif t == "data" and len(rest) == 2:
                vals[f"data.{rest[0]}.{rest[1]}"] = {"value": self._safe_path(v), "source": v.source}
        for key, rec in resources.describe(self).items():
            vals[f"resources.{key}"] = rec
        lv = self.get("launcher", "default")
        vals["launcher"] = {"value": self.launcher(), "source": lv.source if lv else "unset (the unit's own launcher, else the current python)"}
        for name, cmd in self.launchers().items():
            vals[f"launchers.{name}"] = {"value": cmd, "source": self.get("launchers", name).source}
        project: dict[str, Any] | None = None
        pf = self.project_file()
        if pf is not None:

            def safe(get: Any) -> Any:  # a [project] path that does not resolve is shown, never raised (as every value)
                try:
                    v = get()
                except ConfigError as e:
                    return f"<{e}>"
                return [str(x) for x in v] if isinstance(v, list) else str(v)

            project = {"file": str(pf.path), "dir": str(self.project_dir()), "yavih": self.project_requirement(),
                       **{k: safe(lambda k=k: self.project_path(k)) for k in PROJECT_KINDS}, "subprojects": safe(self.subprojects)}
        data: dict[str, Any] = {
            "project": project,
            "files": [f.to_json() for f in self.files],
            "site": self.site,
            "site_source": self.site_source,
            "lenient": self.lenient,
            "values": vals,
            "warnings": [{"code": c, "message": m} for c, m in self.warnings],
        }
        if bench is not None:
            spec = getattr(bench, "spec", None)
            raw = spec if isinstance(spec, Mapping) else (getattr(spec, "raw", None) or {})
            raw_roots = (raw.get("data") or {}).get("roots") or {}
            aliases = tuple(getattr(spec, "aliases", None) or raw.get("aliases") or ())
            recs = self.resolve_data_sources(bench.name, raw_roots, getattr(bench, "data_overrides", None),
                                             aliases=aliases, strict_missing=False)
            for key, rec in recs.items():
                spec = raw_roots.get(key) or {}
                rec["exists"] = bool(rec["value"]) and os.path.exists(rec["value"])
                rec["prepare_only"] = bool(spec.get("prepare_only")) if isinstance(spec, Mapping) else False
                rec["default"] = spec.get("default") if isinstance(spec, Mapping) else spec
            data["bench"] = {"name": bench.name, "folder": str(bench.folder), "roots": recs}
        return data

    def _safe_path(self, v: Value) -> str:
        try:
            return self.path_of(v)
        except ConfigError as e:
            return f"<{e}>"


def _aliases_for(bench: str) -> tuple[str, ...]:
    entry = _REGISTERED.get(bench) or (_REGISTRY_CACHE["value"] or {}).get(bench)
    return tuple(entry[0]) if entry else ()


# ----------------------------------------------------------------------------- building


def _schema_entries(
    f: ConfigFile, data: Mapping[str, Any], prefix: tuple[str, ...], env: Mapping[str, str], col: _Collector, *, in_site: bool
) -> list[tuple[tuple[str, ...], Value]]:
    """Validate one file (or one ``[site.<id>]`` table of it) and flatten it to ``(key, Value)``."""
    out: list[tuple[tuple[str, ...], Value]] = []
    base = str(f.path.parent)
    label = f"{f.path}" + (f" [{'.'.join(prefix)}]" if prefix else "")

    def src(*keys: str) -> str:
        return f.where(*prefix, *keys) + (f" [{'.'.join(prefix)}]" if prefix else "")

    def as_path(value: Any, *keys: str) -> Value | None:
        where = f"{label}: {'.'.join(keys)}"
        if not isinstance(value, str):
            col.strict("W_CONFIG_TYPE", f"{where} must be a string path, got {type(value).__name__}")
            return None
        return Value(_expand_env(value, env, where, col), src(*keys), base)

    for table, body in data.items():
        if table == "hosts" and in_site:
            continue
        if table == "launcher":
            if not isinstance(body, str) or not body.strip():
                col.strict("W_CONFIG_TYPE", f"{label}: launcher must be a non-empty string (the model environment's python, or a prefix command)")
                continue
            out.append((("launcher", "default"), Value(_expand_env(body, env, f"{label}: launcher", col), src("launcher"))))
            continue
        if table == "launchers":
            if not isinstance(body, Mapping):
                col.strict("W_CONFIG_TYPE", f"{label}: [launchers] must be a table of name = command")
                continue
            for k, v in body.items():
                if not _LAUNCHER_NAME_RE.match(str(k)):
                    col.strict("W_CONFIG_UNKNOWN", f"{label}: [launchers] {k}: a launcher name matches [A-Za-z][A-Za-z0-9_.-]*")
                    continue
                if not isinstance(v, str) or not v.strip():
                    col.strict("W_CONFIG_TYPE", f"{label}: [launchers] {k} must be a non-empty string")
                    continue
                out.append((("launchers", str(k)), Value(_expand_env(v, env, f"{label}: launchers.{k}", col), src("launchers", str(k)))))
            continue
        if table == "project":
            if in_site:
                col.strict("W_CONFIG_UNKNOWN", f"{label}: [project] is not a site table")
                continue
            if f.layer != "project":
                col.strict("W_CONFIG_UNKNOWN", f"{label}: [project] belongs to the project file ({PROJECT_DIR}/{PROJECT_FILE}), not the user file")
                continue
            if not isinstance(body, Mapping):
                col.strict("W_CONFIG_TYPE", f"{label}: [project] must be a table")
                continue
            for k, v in body.items():
                if k == "yavih":
                    if not isinstance(v, str) or not v.strip():
                        col.strict("W_CONFIG_TYPE", f"{label}: [project] yavih must be a version requirement string such as \">=1,<2\"")
                        continue
                    out.append((("project", "yavih"), Value(v.strip(), src("project", "yavih"))))
                elif k in PROJECT_KINDS:
                    val = as_path(v, "project", k)
                    if val is not None:
                        out.append((("project", k), val))
                elif k == "subprojects":
                    if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
                        col.strict("W_CONFIG_TYPE", f"{label}: [project] subprojects must be a list of directory paths")
                        continue
                    paths = tuple(_expand_env(x, env, f"{label}: project.subprojects", col) for x in v)
                    out.append((("project", "subprojects"), Value(paths, src("project", "subprojects"), base)))
                else:
                    close = difflib.get_close_matches(str(k), _PROJECT_KEYS, n=1, cutoff=0.6)
                    col.strict("W_CONFIG_UNKNOWN", f"{label}: unknown key [project] {k}" + (f", did you mean {close[0]!r}?" if close else "")
                               + f"; known: {', '.join(_PROJECT_KEYS)}")
            continue
        if table not in _TOP_TABLES or (table == "site" and in_site):
            close = difflib.get_close_matches(table, _TOP_TABLES, n=1, cutoff=0.6)
            col.strict("W_CONFIG_UNKNOWN", f"{label}: unknown table [{table}]" + (f", did you mean [{close[0]}]?" if close else "")
                       + f"; known: {', '.join(t for t in _TOP_TABLES if not (in_site and t in ('site', 'project')))}, and the key launcher")
            continue
        if table == "site":
            continue  # handled by the caller
        if not isinstance(body, Mapping):
            col.strict("W_CONFIG_TYPE", f"{label}: [{table}] must be a table")
            continue
        if table in _ROOT_TABLES:
            for k, v in body.items():
                if k != "root":
                    close = difflib.get_close_matches(k, ["root"], n=1, cutoff=0.5)
                    col.strict("W_CONFIG_UNKNOWN", f"{label}: unknown key [{table}] {k}" + (f", did you mean {close[0]!r}?" if close else "") + "; known: root")
                    continue
                val = as_path(v, table, k)
                if val is not None:
                    out.append(((table, "root"), val))
        elif table == "resources":
            for k, v in body.items():
                if k not in _RESOURCE_KEYS:
                    close = difflib.get_close_matches(k, list(_RESOURCE_KEYS), n=1, cutoff=0.6)
                    col.strict("W_CONFIG_UNKNOWN", f"{label}: unknown key [resources] {k}" + (f", did you mean {close[0]!r}?" if close else "")
                               + f"; known: {', '.join(_RESOURCE_KEYS)}")
                    continue
                try:
                    out.append((("resources", k), Value(_coerce_resource(k, v), src("resources", k))))
                except ConfigError as e:
                    col.errors.append(f"{label}: {e}")
        elif table == "datasets":
            for k, v in body.items():
                if not _NAME_RE.match(str(k)):
                    col.strict("W_CONFIG_UNKNOWN", f"{label}: [datasets] {k}: dataset names must match [a-z][a-z0-9_]*")
                    continue
                val = as_path(v, "datasets", k)
                if val is not None:
                    out.append((("datasets", k), val))
        elif table == "data":
            for k, v in body.items():
                if isinstance(v, Mapping):
                    for rk, rv in v.items():
                        if isinstance(rv, Mapping):
                            col.strict("W_CONFIG_TYPE", f"{label}: [data.{k}.{rk}] is a table; roots are 'key = \"path\"'")
                            continue
                        val = as_path(rv, "data", k, rk)
                        if val is not None:
                            out.append((("data", str(k), str(rk)), val))
                elif k == "root":
                    val = as_path(v, "data", "root")
                    if val is not None:
                        out.append((("data", "root"), val))
                else:
                    col.strict("W_CONFIG_UNKNOWN", f"{label}: unknown key [data] {k}; [data] takes 'root' and [data.<benchmark>] tables")
    return out


def _coerce_resource(key: str, v: Any) -> int | float:
    kind = _RESOURCE_KEYS[key]
    try:
        if isinstance(v, bool):
            raise ValueError
        x = kind(float(v)) if kind is int and float(v) == int(float(v)) else kind(v)
    except (TypeError, ValueError):
        raise ConfigError(f"[resources] {key} = {v!r} is not a{'n integer' if kind is int else ' number'}") from None
    if x < 0 or (x == 0 and key != "ram_budget_gb"):
        raise ConfigError(f"[resources] {key} = {v!r} must be {'>= 0' if key == 'ram_budget_gb' else '> 0'}")
    return x


def _active_site(files: list[ConfigFile], env: Mapping[str, str], col: _Collector) -> tuple[str | None, str | None]:
    sites: list[tuple[str, Any, ConfigFile]] = []
    for f in files:
        for sid, body in (f.data.get("site") or {}).items() if isinstance(f.data.get("site"), Mapping) else ():
            sites.append((str(sid), body, f))
    chosen = env.get("YAVIH_SITE")
    if chosen:
        if not any(s == chosen for s, _b, _f in sites):
            col.strict("W_CONFIG_SITE", f"YAVIH_SITE={chosen!r} but no config file has a [site.{chosen}] table; defined: {sorted({s for s, _b, _f in sites})}")
        return chosen, "env:YAVIH_SITE"
    host = socket.gethostname()
    matches: list[tuple[str, str]] = []
    for sid, body, f in sites:
        hosts = body.get("hosts") if isinstance(body, Mapping) else None
        if hosts is None:
            continue
        if isinstance(hosts, str):
            hosts = [hosts]
        if not isinstance(hosts, list) or not all(isinstance(h, str) for h in hosts):
            col.strict("W_CONFIG_TYPE", f"{f.path}: [site.{sid}] hosts must be a list of hostname globs")
            continue
        for h in hosts:
            if fnmatch.fnmatch(host, h):
                matches.append((sid, f"host {host} matches {h!r} ({f.where('site', sid, 'hosts')})"))
                break
    ids = list(dict.fromkeys(s for s, _ in matches))
    if len(ids) > 1:
        col.warn("W_CONFIG_SITE", f"host {host} matches several sites {ids}; using {ids[0]!r} (set YAVIH_SITE to choose)")
    return (ids[0], matches[0][1]) if ids else (None, None)


def _env_entries(env: Mapping[str, str], col: _Collector) -> list[tuple[tuple[str, ...], Value]]:
    out: list[tuple[tuple[str, ...], Value]] = []
    for name in sorted(env):
        if not name.startswith("YAVIH_"):
            continue
        raw = env[name]
        if name in _ENV_SCALARS:
            key = _ENV_SCALARS[name]
            if raw == "":
                continue
            if key[0] == "resources":
                try:
                    out.append((key, Value(_coerce_resource(key[1], raw), f"env:{name}")))
                except ConfigError as e:
                    col.errors.append(f"{name}: {e}")
            else:
                out.append((key, Value(_expand_env(raw, env, name, col), f"env:{name}")))
            continue
        if name in ENV_VARS:
            continue
        if name.startswith(DATA_ENV_PREFIX):
            parts = name[len(DATA_ENV_PREFIX):].split("__")
            if len(parts) == 2 and all(parts):
                if raw != "":
                    out.append((("data", parts[0].lower(), parts[1].lower()), Value(_expand_env(raw, env, name, col), f"env:{name}")))
                continue
            col.strict("W_CONFIG_ENV", f"{name}: expected YAVIH_DATA__<BENCH>__<ROOT> or YAVIH_DATA__ROOT")
            continue
        if name.startswith(DATASETS_ENV_PREFIX):
            ds = name[len(DATASETS_ENV_PREFIX):].lower()
            if _NAME_RE.match(ds):
                if raw != "":
                    out.append((("datasets", ds), Value(_expand_env(raw, env, name, col), f"env:{name}")))
                continue
        close = difflib.get_close_matches(name, list(ENV_VARS), n=1, cutoff=0.75)
        col.strict("W_CONFIG_ENV", f"unknown environment variable {name}" + (f", did you mean {close[0]}?" if close else "")
                   + " (unset it; the known ones are listed by `yavih config show` and references/cli.md)")
    return out


def _check_benchmarks(values: Mapping[tuple[str, ...], Value], col: _Collector) -> dict[tuple[str, ...], Value]:
    """Map ``[data.<alias>]`` to canonical names, refuse near misses and unknown roots of known benchmarks."""
    names = {k[1] for k in values if k[0] == "data" and len(k) == 3}
    if not names:
        return dict(values)
    reg = _registry()
    out: dict[tuple[str, ...], Value] = {}
    for key, v in values.items():
        if not (key[0] == "data" and len(key) == 3):
            out[key] = v
            continue
        _, name, root = key
        canon = _canonical(name, reg)
        if canon is None:
            near = _near_miss(name, reg)
            if near is not None:
                col.strict("W_CONFIG_UNKNOWN_BENCH", f"{v.source}: unknown benchmark {name!r}, did you mean {near!r}? (names are lower case: [data.{near}], YAVIH_DATA__{near.upper()}__<ROOT>)")
                continue
            col.warn("W_CONFIG_UNKNOWN_BENCH", f"{v.source}: [data.{name}] names no packaged or $YAVIH_HOME/benchmarks benchmark "
                     f"(known: {', '.join(sorted(reg))}); kept for a folder of that name loaded by path")
            out[key] = v
            continue
        roots = reg[canon][1]
        if roots and root not in roots:
            close = difflib.get_close_matches(root, list(roots), n=1, cutoff=0.6)
            col.strict("W_CONFIG_UNKNOWN_ROOT", f"{v.source}: benchmark {canon!r} has no data root {root!r}" + (f", did you mean {close[0]!r}?" if close else "")
                       + f"; declared: {sorted(roots)}")
            continue
        ckey = ("data", canon, root)
        prev = out.get(ckey)
        if prev is None or v.order >= prev.order:
            out[ckey] = v
    return out


def _build(env: Mapping[str, str], cwd: Path, cli: Mapping[str, Any], explicit: Mapping[str, Any] | None, lenient: bool | None) -> Settings:
    lenient_flag = bool(lenient) if lenient is not None else (bool(cli.get("lenient")) or env.get("YAVIH_LENIENT", "") not in ("", "0"))
    col = _Collector(lenient_flag)
    files: list[ConfigFile] = []
    if explicit is not None:
        files.append(ConfigFile(path=cwd / "<config=>", layer="project", how="config=", exists=True, data=dict(explicit)))
    else:
        upath, uhow = user_config_path(env)
        files.append(_read_file(upath, "user", uhow))
        ppath, phow = _project_path(env, cwd, cli)
        if ppath is not None:
            pf = _read_file(ppath, "project", phow)
            if not pf.exists and phow == "--config":
                raise ConfigError(f"--config {ppath}: no such file")
            if not (pf.exists and files[0].exists and pf.path.resolve() == files[0].path.resolve()):
                files.append(pf)  # the user file named again as the project file is read once
    site, site_source = _active_site([f for f in files if f.exists], env, col)
    entries: list[tuple[tuple[str, ...], Value]] = []
    for f in files:
        if not f.exists:
            continue
        entries += _schema_entries(f, f.data, (), env, col, in_site=False)
        sites = f.data.get("site")
        if sites is not None and not isinstance(sites, Mapping):
            col.strict("W_CONFIG_TYPE", f"{f.path}: [site] must hold [site.<id>] tables")
            sites = None
        for sid, body in (sites or {}).items():
            if not isinstance(body, Mapping):
                col.strict("W_CONFIG_TYPE", f"{f.path}: [site.{sid}] must be a table")
                continue
            sub = _schema_entries(f, body, ("site", str(sid)), env, col, in_site=True)
            if str(sid) == site:
                entries += sub
    entries += _env_entries(env, col)
    for key, raw in (cli.get("entries") or {}).items():
        entries.append((tuple(key), Value(raw, "cli")))
    values: dict[tuple[str, ...], Value] = {}
    for i, (key, v) in enumerate(entries, 1):
        values[key] = Value(v.value, v.source, v.base, i)
    values = _check_benchmarks(values, col)
    if col.errors:
        hint = "fix the config, or pass --lenient / set YAVIH_LENIENT=1 to downgrade schema errors to warnings"
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(col.errors) + f"\n({hint}; `yavih config show` lists the sources)")
    return Settings(files=files, site=site, site_source=site_source, lenient=lenient_flag, values=values, warnings=col.warnings, env=dict(env), cwd=cwd)


def _file_stamps(settings: Settings) -> tuple:
    out = []
    for f in settings.files:
        try:
            st = f.path.stat()
            out.append((str(f.path), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(f.path), None, None))
    return tuple(out)


def load(
    *,
    env: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    cwd: str | os.PathLike | None = None,
    lenient: bool | None = None,
    refresh: bool = False,
) -> Settings:
    """The process :class:`Settings` (cached; rebuilt when the environment, the current directory, the CLI
    layer, the registered benchmarks or a config file changes). ``config`` replaces both files by one
    in-memory table (tests); ``env`` replaces ``os.environ``."""
    real_env = env is None
    env_map: Mapping[str, str] = os.environ if env is None else env
    cwd_path = Path(cwd) if cwd is not None else _cwd()
    cli = cli_overrides() if real_env else {}
    if config is not None or not real_env:
        return _build(env_map, cwd_path, cli, config, lenient)
    key = json.dumps(
        [sorted((k, v) for k, v in env_map.items() if k.startswith("YAVIH_") or k in ("XDG_CONFIG_HOME", "HOME", "SLURM_TMPDIR")),
         str(cwd_path), str(_discover_project(cwd_path)), repr(sorted((k, repr(v)) for k, v in cli.items())), lenient, sorted(_REGISTERED)],
    )
    with _LOCK:
        cached: Settings | None = _CACHE["settings"]
        if not refresh and cached is not None and _CACHE["key"] == key and _CACHE["stamps"] == _file_stamps(cached):
            return cached
    settings = _build(env_map, cwd_path, cli, None, lenient)
    with _LOCK:
        _CACHE.update(key=key, settings=settings, stamps=_file_stamps(settings))
    for code, msg in settings.warnings:
        warn_once(code, msg)
    return settings


# ----------------------------------------------------------------------------- the function API


def _settings(env: Mapping[str, str] | None, config: Mapping[str, Any] | None) -> Settings:
    return load(env=env, config=config)


def yavih_home(env: Mapping[str, str] | None = None) -> Path:
    """``$YAVIH_HOME`` → ``[home] root`` → ``~/.yavih``. Never raises on a bad config file (the home
    directory is needed to report it): an invalid configuration falls back to ``$YAVIH_HOME``."""
    try:
        return _settings(env, None).yavih_home()
    except ConfigError:
        return _home_from_env(os.environ if env is None else env)


def runs_root(env: Mapping[str, str] | None = None, config: dict | None = None) -> Path:
    """``--runs-root`` → ``$YAVIH_RUNS`` → ``[runs] root`` (project, then user file) → ``<project dir>/runs`` →
    ``<current directory>/runs``; always absolute (:meth:`Settings.runs_root`)."""
    return _settings(env, config).runs_root()


def configured_results_root(env: Mapping[str, str] | None = None, config: dict | None = None) -> Path | None:
    """The results-store root the site configured: ``--results`` → ``$YAVIH_RESULTS`` → config ``[results] root``
    → ``None`` (``None`` = derive it from a standard run layout, or refuse)."""
    return _settings(env, config).configured_results_root()


def results_root(
    runs_root_path: str | os.PathLike | None = None,
    env: Mapping[str, str] | None = None,
    config: dict | None = None,
) -> Path:
    """The lm-evaluation-harness style results store: ``--results`` → ``$YAVIH_RESULTS`` →
    ``[results] root`` → ``<project dir>/results`` → ``Path(runs_root).parent / "results"``. ``runs_root_path`` is the
    runs root of the invocation (a probe passes the root its run directory lives under); ``None`` = :func:`runs_root`."""
    return _settings(env, config).results_root(runs_root_path)


def configured_scratch_root(env: Mapping[str, str] | None = None, config: dict | None = None) -> Path | None:
    """The configured scratch root: ``$YAVIH_SCRATCH`` / ``[scratch] root`` (env beats files) → ``$SLURM_TMPDIR``
    → ``None`` (never ``TMPDIR``). ``yavih exec`` and ``probe run`` share it: without one a probe reads the
    features in place (memmap backend, ``W_SIZE``) instead of staging them."""
    return _settings(env, config).configured_scratch_root()


def scratch_root(run_dir: str | os.PathLike | None = None, env: Mapping[str, str] | None = None,
                 config: dict | None = None) -> Path:
    """:func:`configured_scratch_root` → the run directory (never ``TMPDIR``)."""
    return _settings(env, config).scratch_root(run_dir)


def data_root(config: dict | None = None, env: Mapping[str, str] | None = None) -> str | None:
    """The site-wide ``[data] root`` (``${data.root}`` in benchmark.yaml defaults); env
    ``YAVIH_DATA__ROOT`` overrides it."""
    return _settings(env, config).data_root()


def dataset(name: str, *, fallback: bool = True, env: Mapping[str, str] | None = None, config: dict | None = None) -> str | None:
    """``[datasets] <name>`` (``YAVIH_DATASETS__<NAME>``), else ``${data.root}/<name>`` when ``fallback``."""
    return _settings(env, config).dataset(name, fallback=fallback)


def data_env_var(bench: str, key: str) -> str:
    """Name of the env var overriding root ``key`` of benchmark ``bench``:
    ``YAVIH_DATA__<NAME>__<KEY>`` with both parts upper-cased."""
    return f"{DATA_ENV_PREFIX}{bench.upper()}__{key.upper()}"


def resolve_data(
    bench: str,
    roots: Mapping[str, Any],
    overrides: Mapping[str, str] | None = None,
    *,
    config: dict | None = None,
    env: Mapping[str, str] | None = None,
    aliases: Iterable[str] = (),
) -> dict[str, str]:
    """Resolve every root of ``benchmark.yaml[data.roots]`` to a path string.

    ``roots`` maps root name → RootSpec dict (``{default, description}``) or a bare default string.
    Precedence per root: ``overrides`` → ``--data`` → env ``YAVIH_DATA__<NAME>__<KEY>`` → project file
    (its site table first) → user file (likewise) ``[data.<name>] <key>`` → the yaml default with
    ``${data.root}`` / ``${datasets.X}`` substituted. Paths are ``~``/``$VAR``-expanded and NOT checked
    for existence (``bench verify`` does). Unknown override keys and configured keys the benchmark does
    not declare are errors (nothing is silently dropped)."""
    return _settings(env, config).resolve_data(bench, roots, overrides, aliases=aliases)


def resolve_data_sources(bench: str, roots: Mapping[str, Any], overrides: Mapping[str, str] | None = None, *,
                         aliases: Iterable[str] = (), strict_missing: bool = True) -> dict[str, dict[str, Any]]:
    """:func:`resolve_data` with each root's ``{value, source, env_var, config_key, error}``."""
    return load().resolve_data_sources(bench, roots, overrides, aliases=aliases, strict_missing=strict_missing)


def _folder_yaml(folder: str | os.PathLike) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    import yaml

    p = Path(folder)
    try:
        with open(p / "benchmark.yaml", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"{p}: cannot read benchmark.yaml: {e}") from e
    roots = ((d or {}).get("data") or {}).get("roots") or {}
    return str(d.get("name", p.name)), tuple(str(a) for a in (d.get("aliases") or [])), dict(roots)


def folder_roots(folder: str | os.PathLike, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """The roots of the benchmark folder ``folder`` (its yaml's ``data.roots``) resolved through the
    layers, WITHOUT loading the manifest: what ``make_manifest.py`` and ``prepare/*.py`` use (a re-freeze
    runs while the manifest is being rewritten)."""
    name, aliases, roots = _folder_yaml(folder)
    register_benchmark(name, aliases, roots)
    return load().resolve_data(name, roots, overrides, aliases=aliases)


def folder_root(folder: str | os.PathLike, key: str, override: str | os.PathLike | None = None) -> Path:
    """One root of :func:`folder_roots` as a ``Path``; ``override`` (a script's ``--<root>`` flag) wins.
    Raises :class:`ConfigError` naming the config key and env var when the root cannot be resolved."""
    if override is not None:
        return Path(os.path.expanduser(str(override)))
    name, aliases, roots = _folder_yaml(folder)
    if key not in roots:
        raise ConfigError(f"benchmark {name!r} declares no data root {key!r}; declared: {sorted(roots)}")
    register_benchmark(name, aliases, roots)
    rec = load().resolve_data_sources(name, roots, aliases=aliases, strict_missing=False)[key]
    if rec["error"]:
        raise ConfigError(rec["error"])
    return Path(rec["value"])


def roots_digest(roots: Mapping[str, str]) -> str:
    """12 hex chars of sha256 over the resolved roots: the key of the per-site caches
    (``data_status/<name>-<digest>.json``), so two sites sharing one HOME never void each other."""
    payload = json.dumps({str(k): str(v) for k, v in sorted(roots.items())}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]
