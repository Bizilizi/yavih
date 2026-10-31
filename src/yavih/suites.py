"""Suites: named benchmark sets ("benchmark my model" = run a suite).

A suite is a yaml file naming, per capability, the benchmarks and readings a model is run on::

    schema_version: 1
    name: default
    description: one line
    groups:
      - capability: audio-visual          # one of CAPABILITIES
        entries:
          - {benchmark: vggsounder, readings: [a, v, av], join: true}
          - {benchmark: audioset20k, readings: [a, v, av], required: true, status: packaged}

``status: pending`` marks a benchmark without a packaged folder yet (listed, never planned or checked).
``join: true`` names the folder's ``eval.py`` ``join()`` hook (``yavih report build --join <benchmark>
<reading>=RUN ...``), whose result is stored under ``<benchmark>:<readings joined by +>``.

Resolution (``load_suite``): an existing ``.yaml`` path; else ``<project suites dir>/<name>.yaml`` (the
directory ``$YAVIH_SUITES``, else the project's ``suites/`` next to ``yavih-bench/yavih.toml``, else ``./suites``); else the
shipped ``yavih/suites/<name>.yaml``. A project suite shadows a shipped one of the same name. Suites are files
only: there is no configuration table for them.

"suite" is never "profile": a profile is the ``profile.json`` that ``yavih bridge check`` writes for one
bridge. Every refusal here is :class:`SuiteError` (``E_SUITE``, exit 1). stdlib at module scope; ``yaml``
(a core dependency) and the benchmark resolution are imported lazily.
Documentation: https://github.com/Bizilizi/yavih/blob/main/docs/suites.md
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CAPABILITIES",
    "ENTRY_STATUSES",
    "COMPLETENESS",
    "SUITE_SCHEMA_VERSION",
    "SUITES_ENV",
    "Suite",
    "SuiteGroup",
    "SuiteEntry",
    "SuiteError",
    "shipped_dir",
    "project_suites_dir",
    "list_suites",
    "load_suite",
    "validate_suite",
    "pairs",
    "expected_store_keys",
    "new_suite",
    "completeness",
    "completeness_json",
    "join_command",
]

#: The capability groups a suite may use, in display order.
CAPABILITIES: tuple[str, ...] = ("audio", "visual", "audio-visual", "temporal", "spatial", "retrieval", "generation")
#: ``status`` of a suite entry: ``packaged`` (a resolvable benchmark folder) or ``pending`` (none yet).
ENTRY_STATUSES: tuple[str, ...] = ("packaged", "pending")
#: Completeness of one ``(benchmark, reading, model)``: a DONE-marked result exists (``done``), it does not
#: (``missing``), the benchmark is not packaged (``pending``); a join entry is ``done``, ``join-missing`` (not in the
#: results store) or ``join-unknown`` (no results store to look in).
COMPLETENESS: tuple[str, ...] = ("done", "missing", "pending", "join-missing", "join-unknown")
SUITE_SCHEMA_VERSION = 1
#: The directory of project suites; beats the project's ``suites/``.
SUITES_ENV = "YAVIH_SUITES"
_SUITES_DIR = "suites"
_TOP_KEYS = ("schema_version", "name", "description", "groups")
_GROUP_KEYS = ("capability", "entries")
_ENTRY_KEYS = ("benchmark", "readings", "join", "required", "status")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_BENCH_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
#: ``yavih.plan.matrix.READING_RE`` (kept here so the module stays import-light).
_READING_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_JOIN_HOOK_RE = re.compile(r"^def join\(", flags=re.M)
_DOCS = "https://github.com/Bizilizi/yavih/blob/main/docs/suites.md"


class SuiteError(ValueError):
    """A suite that cannot be found, parsed or validated, or a ``suite new`` / ``plan init`` destination that
    exists (``E_EXISTS``): exit 1."""

    exit_code = 1

    def __init__(self, message: str, code: str = "E_SUITE") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SuiteEntry:
    """One benchmark of a suite: the canonical ``benchmark`` name (as given for a pending entry), the ``readings``
    to run, whether the folder's ``join()`` combines them, whether the entry is ``required`` for completeness, and
    its ``status`` (:data:`ENTRY_STATUSES`). ``multi_reading``: the bridge template takes the reading as an
    argument (one bridge file serves every reading; matrix entries pass ``args: {reading: ...}``)."""

    benchmark: str
    readings: tuple[str, ...]
    join: bool = False
    required: bool = True
    status: str = "packaged"
    multi_reading: bool = False

    @property
    def join_reading(self) -> str | None:
        """The reading tag of the joined entry (``a+v+av``), or ``None`` without ``join``."""
        if not self.join:
            return None
        from yavih.report.store import join_reading

        return join_reading(self.readings)

    def to_json(self) -> dict[str, Any]:
        return {"benchmark": self.benchmark, "readings": list(self.readings), "join": self.join, "required": self.required,
                "status": self.status, "multi_reading": self.multi_reading, "join_reading": self.join_reading}


@dataclass(frozen=True)
class SuiteGroup:
    capability: str
    entries: tuple[SuiteEntry, ...]

    def to_json(self) -> dict[str, Any]:
        return {"capability": self.capability, "entries": [e.to_json() for e in self.entries]}


@dataclass(frozen=True)
class Suite:
    """A validated suite. ``origin``: ``shipped``, ``project`` (the project suites directory) or ``path`` (loaded
    from an explicit file)."""

    name: str
    description: str
    groups: tuple[SuiteGroup, ...]
    path: Path | None = None
    origin: str = "path"
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def entries(self) -> list[tuple[str, SuiteEntry]]:
        """``(capability, entry)`` in file order."""
        return [(g.capability, e) for g in self.groups for e in g.entries]

    def benchmarks(self) -> list[str]:
        """The distinct benchmark names, in file order."""
        return list(dict.fromkeys(e.benchmark for _c, e in self.entries()))

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "origin": self.origin,
                "path": str(self.path) if self.path is not None else None, "groups": [g.to_json() for g in self.groups]}


# ----------------------------------------------------------------------------- locations


def shipped_dir() -> Path:
    """The packaged ``yavih/suites/`` directory (``minimal.yaml``, ``default.yaml``)."""
    return Path(__file__).resolve().parent / _SUITES_DIR


def project_suites_dir() -> Path:
    """Where project suites live and ``suite new`` writes: ``$YAVIH_SUITES``; else the project's suites directory
    (``[project] suites``, else ``suites/`` next to the project config file: ``--config``, ``$YAVIH_CONFIG`` or the
    discovered ``yavih-bench/yavih.toml``); else ``./suites``."""
    env = os.environ.get(SUITES_ENV)
    if env:
        return Path(os.path.expanduser(env))
    from yavih import config

    try:
        settings = config.load()
    except config.ConfigError:
        settings = None
    if settings is not None:
        p = settings.project_path("suites")
        if p is not None:
            return p
    return config._cwd() / _SUITES_DIR


def _yaml_files(d: Path) -> dict[str, Path]:
    if not d.is_dir():
        return {}
    return {p.stem: p for p in sorted(d.glob("*.yaml")) if p.is_file()}


def list_suites() -> list[dict[str, Any]]:
    """Every suite by name, shipped first then project ones, each ``{name, origin, path, description, shadows,
    error}``: a project suite of a shipped name replaces it (``shadows: true``); a file that does not validate is
    listed with its ``error`` instead of failing the listing."""
    shipped = _yaml_files(shipped_dir())
    project = _yaml_files(project_suites_dir())
    out: list[dict[str, Any]] = []
    for name in [*shipped, *[n for n in project if n not in shipped]]:
        origin = "project" if name in project else "shipped"
        path = project.get(name) or shipped[name]
        rec: dict[str, Any] = {"name": name, "origin": origin, "path": str(path), "description": None,
                               "shadows": origin == "project" and name in shipped, "error": None}
        try:
            rec["description"] = _load_file(path, origin=origin, expect_name=name).description
        except SuiteError as e:
            rec["error"] = str(e)
        out.append(rec)
    return out


def _read_yaml(path: Path) -> Any:
    import yaml  # core dependency

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SuiteError(f"suite {path}: cannot read: {e}") from e
    except yaml.YAMLError as e:
        raise SuiteError(f"suite {path} does not parse: {e}") from e


def _load_file(path: Path, *, origin: str, expect_name: str | None = None) -> Suite:
    suite = validate_suite(_read_yaml(path), where=str(path))
    if expect_name is not None and suite.name != expect_name:
        raise SuiteError(f"suite {path}: name {suite.name!r} differs from the file name {expect_name!r}")
    return Suite(name=suite.name, description=suite.description, groups=suite.groups, path=path, origin=origin, raw=suite.raw)


def load_suite(name_or_path: str | os.PathLike | Suite) -> Suite:
    """A :class:`Suite` by name (the project suites directory first, then the shipped ones) or by an existing
    ``.yaml`` / ``.yml`` path. Unknown name or invalid file: :class:`SuiteError` (exit 1)."""
    if isinstance(name_or_path, Suite):
        return name_or_path
    ref = str(name_or_path)
    p = Path(ref).expanduser()
    if p.suffix in (".yaml", ".yml") or "/" in ref or os.sep in ref:
        if not p.is_file():
            raise SuiteError(f"suite file {ref} does not exist")
        return _load_file(p, origin="path")
    if not _NAME_RE.match(ref):
        raise SuiteError(f"suite name {ref!r} must match [a-z0-9][a-z0-9_-]* (or name a .yaml file)")
    project = project_suites_dir() / f"{ref}.yaml"
    if project.is_file():
        return _load_file(project, origin="project", expect_name=ref)
    shipped = shipped_dir() / f"{ref}.yaml"
    if shipped.is_file():
        return _load_file(shipped, origin="shipped", expect_name=ref)
    known = sorted({r["name"] for r in list_suites()})
    raise SuiteError(f"unknown suite {ref!r}; known: {', '.join(known) or 'none'} (`yavih suite list`; a custom one: "
                     f"`yavih suite new {ref}`)")


# ----------------------------------------------------------------------------- validation


def _check_keys(obj: Any, allowed: Sequence[str], where: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise SuiteError(f"{where} must be a mapping")
    unknown = sorted(set(map(str, obj)) - set(allowed))
    if unknown:
        raise SuiteError(f"{where}: unknown keys {unknown} (allowed: {', '.join(allowed)}; {_DOCS})")
    return obj


def _bool(raw: Any, default: bool, where: str) -> bool:
    if raw is None:
        return default
    if not isinstance(raw, bool):
        raise SuiteError(f"{where} must be true or false")
    return raw


def _packaged_facts(bench: str, where: str) -> tuple[str, set[str], bool, bool]:
    """``(canonical name, template readings, multi-reading template, eval.py defines join)`` of a resolvable folder."""
    from yavih.benchmark import resolve_folder
    from yavih.lint import template_readings
    from yavih.shipped import _bench_name

    folder = resolve_folder(bench)
    if folder is None:
        raise SuiteError(f"{where}: benchmark {bench!r} does not resolve to a folder (`yavih bench list`); "
                         "mark it `status: pending` while it has none")
    template = Path(folder) / "bridge_template.py"
    if not template.is_file():
        raise SuiteError(f"{where}: benchmark {bench!r} ships no bridge_template.py ({folder})")
    src = template.read_text(encoding="utf-8")
    probe = Path(folder) / "eval.py"
    has_join = probe.is_file() and bool(_JOIN_HOOK_RE.search(probe.read_text(encoding="utf-8")))
    multi = bool(re.search(r'"reading":\s*ArgSpec\(', src))
    return _bench_name(Path(folder)), template_readings(src), multi, has_join


def validate_suite(raw: Any, *, where: str = "suite") -> Suite:
    """Check a parsed suite file and return the :class:`Suite` (``origin="path"``, no ``path``). Refused
    (:class:`SuiteError`): unknown keys at any level, ``schema_version`` other than 1, a capability outside
    :data:`CAPABILITIES`, a packaged benchmark that does not resolve, readings outside its
    ``bridge_template.py``'s (a literal ``"reading"`` or the ``choices`` of its ``reading`` argument), ``join`` on
    a folder whose ``eval.py`` defines no ``join()``, and a ``(benchmark, reading)`` listed twice. Packaged
    benchmark names are canonicalized (an alias becomes the folder's name)."""
    top = _check_keys(raw, _TOP_KEYS, where)
    if isinstance(top.get("schema_version"), bool) or top.get("schema_version") != SUITE_SCHEMA_VERSION:
        raise SuiteError(f"{where}: schema_version must be {SUITE_SCHEMA_VERSION}, got {top.get('schema_version')!r}")
    name = top.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise SuiteError(f"{where}: name {name!r} must match [a-z0-9][a-z0-9_-]{{0,63}}")
    desc = top.get("description")
    if not isinstance(desc, str) or not desc.strip() or "\n" in desc.strip():
        raise SuiteError(f"{where}: description must be one non-empty line")
    groups_raw = top.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise SuiteError(f"{where}: groups must be a non-empty list of {{capability, entries}}")
    seen: set[tuple[str, str]] = set()
    seen_caps: set[str] = set()
    groups: list[SuiteGroup] = []
    for gi, graw in enumerate(groups_raw):
        gwhere = f"{where}: groups[{gi}]"
        g = _check_keys(graw, _GROUP_KEYS, gwhere)
        cap = g.get("capability")
        if cap not in CAPABILITIES:
            raise SuiteError(f"{gwhere}.capability {cap!r} must be one of {', '.join(CAPABILITIES)}")
        if cap in seen_caps:
            raise SuiteError(f"{gwhere}: capability {cap!r} appears in two groups")
        seen_caps.add(cap)
        ents = g.get("entries")
        if not isinstance(ents, list) or not ents:
            raise SuiteError(f"{gwhere}.entries must be a non-empty list")
        entries: list[SuiteEntry] = []
        for ei, eraw in enumerate(ents):
            ewhere = f"{gwhere}.entries[{ei}]"
            e = _check_keys(eraw, _ENTRY_KEYS, ewhere)
            bench = e.get("benchmark")
            if not isinstance(bench, str) or not bench:
                raise SuiteError(f"{ewhere}.benchmark is required")
            readings = e.get("readings")
            if (not isinstance(readings, list) or not readings
                    or not all(isinstance(r, str) and _READING_RE.match(r) for r in readings)):
                raise SuiteError(f"{ewhere}.readings must be a non-empty list of reading tags (a, v, av)")
            if len(set(readings)) != len(readings):
                raise SuiteError(f"{ewhere}.readings lists a reading twice: {readings}")
            status = e.get("status", "packaged")
            if status not in ENTRY_STATUSES:
                raise SuiteError(f"{ewhere}.status {status!r} must be one of {', '.join(ENTRY_STATUSES)}")
            join = _bool(e.get("join"), False, f"{ewhere}.join")
            required = _bool(e.get("required"), True, f"{ewhere}.required")
            multi = len(readings) > 1
            if status == "pending":
                if not _BENCH_RE.match(bench):
                    raise SuiteError(f"{ewhere}.benchmark {bench!r} must match [a-z0-9_]+")
            else:
                bench, have, multi, has_join = _packaged_facts(bench, ewhere)
                if not have:
                    raise SuiteError(f"{ewhere}: {bench}'s bridge_template.py declares no reading")
                extra = [r for r in readings if r not in have]
                if extra:
                    raise SuiteError(f"{ewhere}: readings {extra} are not readings of {bench}'s bridge_template.py "
                                     f"(it declares {sorted(have)})")
                if join and not has_join:
                    raise SuiteError(f"{ewhere}: join: true, but {bench}'s eval.py defines no join() "
                                     "(`yavih report build --join` needs that hook)")
            if join and len(readings) < 2:
                raise SuiteError(f"{ewhere}: join needs at least two readings")
            for r in readings:
                if (bench, r) in seen:
                    raise SuiteError(f"{ewhere}: ({bench}, {r}) is listed twice")
                seen.add((bench, r))
            entries.append(SuiteEntry(benchmark=bench, readings=tuple(readings), join=join, required=required, status=status,
                                      multi_reading=multi))
        groups.append(SuiteGroup(capability=str(cap), entries=tuple(entries)))
    return Suite(name=name, description=desc.strip(), groups=tuple(groups), raw=dict(top))


# ----------------------------------------------------------------------------- views


def pairs(suite: Suite | str) -> list[dict[str, Any]]:
    """One record per ``(benchmark, reading)`` in file order: ``{capability, benchmark, reading, required,
    status, multi_reading}`` (pending entries included, with ``status: pending``)."""
    suite = load_suite(suite)
    return [{"capability": cap, "benchmark": e.benchmark, "reading": r, "required": e.required, "status": e.status,
             "multi_reading": e.multi_reading}
            for cap, e in suite.entries() for r in e.readings]


def expected_store_keys(suite: Suite | str) -> list[str]:
    """The results-store entry keys a complete packaged suite produces per model: ``<benchmark>:<reading>`` per
    pair and ``<benchmark>:<a+v+av>`` per join (``yavih.report.store.entry_key``)."""
    from yavih.report.store import entry_key

    suite = load_suite(suite)
    keys: list[str] = []
    for _cap, e in suite.entries():
        if e.status != "packaged":
            continue
        keys += [entry_key(e.benchmark, r) for r in e.readings]
        if e.join:
            keys.append(entry_key(e.benchmark, str(e.join_reading)))
    return keys


def new_suite(name: str, *, from_: str = "default", out: str | os.PathLike | None = None, force: bool = False) -> dict[str, Any]:
    """Copy suite ``from_`` (a name or a path) to ``<project suites dir>/<name>.yaml`` (``out``: a directory or a
    file) with its ``name`` set to ``name``; comments and order are kept. Refuses an existing file unless ``force``
    (``E_EXISTS``). Returns ``{name, from, path}``."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise SuiteError(f"suite name {name!r} must match [a-z0-9][a-z0-9_-]{{0,63}}")
    src = load_suite(from_)
    if src.path is None:  # pragma: no cover - load_suite always sets it
        raise SuiteError(f"suite {from_!r} has no file to copy")
    dest = Path(out) if out is not None else project_suites_dir() / f"{name}.yaml"
    if out is not None and (dest.is_dir() or str(out).endswith(("/", os.sep))):
        dest = dest / f"{name}.yaml"
    if dest.exists() and not force:
        raise SuiteError(f"{dest} exists; pass --force to overwrite it (or --out another path)", "E_EXISTS")
    text = src.path.read_text(encoding="utf-8")
    text, n = re.subn(r"^name:.*$", f"name: {name}", text, count=1, flags=re.M)
    if not n:  # pragma: no cover - a validated suite has a top-level name line
        raise SuiteError(f"suite {src.path} has no top-level `name:` line to replace")
    validate_suite(_parse_text(text), where=str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return {"name": name, "from": src.name, "from_path": str(src.path), "path": str(dest.resolve())}


def _parse_text(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


# ----------------------------------------------------------------------------- completeness


def join_command(entry: SuiteEntry, model: str, runs_root: str | os.PathLike | None) -> str:
    """The ``yavih report build`` line that computes ``entry``'s join for ``model`` from the standard run layout
    ``<runs root>/<benchmark>/<model>/<reading>``."""
    root = str(runs_root) if runs_root is not None else "<runs root>"
    routes = " ".join(f"{r}={root}/{entry.benchmark}/{model}/{r}" for r in entry.readings)
    return f"yavih report build {root}/{entry.benchmark}/{model} --join {entry.benchmark} {routes}"


def _stored_joins(results_root: Path, model: str) -> set[tuple[str, str]]:
    from yavih.report.store import read_results_files

    out: set[tuple[str, str]] = set()
    model_dir = results_root / model
    if not model_dir.is_dir():
        return out
    for _path, doc in read_results_files(model_dir):
        for cfg in (doc.get("configs") or {}).values():
            if isinstance(cfg, Mapping) and cfg.get("joined") and cfg.get("benchmark") and cfg.get("reading"):
                out.add((str(cfg["benchmark"]), str(cfg["reading"])))
    return out


def completeness(suite: Suite | str, records: Iterable[Mapping[str, Any]], *, models: Sequence[str] | None = None,
                 results_root: str | os.PathLike | None = None, runs_root: str | os.PathLike | None = None,
                 joined: Iterable[tuple[str, str, str]] = ()) -> list[dict[str, Any]]:
    """Per model and suite ``(benchmark, reading)`` — plus one ``<benchmark>:<a+v+av>`` row per join entry — the
    status (:data:`COMPLETENESS`): ``{capability, benchmark, reading, model, status, required, note, hint}``.

    ``records`` are report index records (``{"path", "result"}``; moved-aside ones are ignored): a pair is ``done``
    when a DONE-marked result of that ``(benchmark after alias, model, reading)`` exists (``note`` says when none is
    ``ok`` or none is standard; a capped-run result, ``sample``, never counts: ``missing`` with the note ``partial only``). A join row is ``done`` when ``joined`` holds ``(model, benchmark, reading)`` (a
    join of this invocation) or the model's results directory under ``results_root`` holds that entry; with no
    ``results_root`` it is ``join-unknown``. ``hint`` names the command that fills a missing join.
    ``models=None``: every model of the records."""
    from yavih.report.compare import _alias_map, row_key
    from yavih.report.index import MOVED_ASIDE

    suite = load_suite(suite)
    aliases = _alias_map()
    seen: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    partial: set[tuple[str, str, str]] = set()  # pairs with a capped-run result only (`plan build --sample N`)
    for rec in records:
        if rec.get(MOVED_ASIDE):
            continue
        res = rec.get("result")
        if not isinstance(res, Mapping):
            continue
        bench, _eh, mdl, rd = row_key(dict(res), aliases)
        if res.get("sample"):  # a capped run never completes a pair
            partial.add((bench, mdl, rd))
            continue
        seen.setdefault((bench, mdl, rd), []).append(res)
    if models is None:
        models = sorted({m for (_b, m, _r) in (*seen, *partial) if m})
    joined_set = {tuple(map(str, j)) for j in joined}
    store_root = Path(results_root) if results_root is not None else None
    rows: list[dict[str, Any]] = []
    for model in models:
        stored = _stored_joins(store_root, model) if store_root is not None else set()
        for cap, e in suite.entries():
            for r in e.readings:
                row = {"capability": cap, "benchmark": e.benchmark, "reading": r, "model": model, "required": e.required,
                       "status": "pending", "note": None, "hint": None}
                if e.status == "packaged":
                    found = seen.get((e.benchmark, model, r), [])
                    row["status"] = "done" if found else "missing"
                    if not found and (e.benchmark, model, r) in partial:
                        row["note"] = "partial only (plan build --sample)"
                    elif found and not any(x.get("status") == "ok" for x in found):
                        row["note"] = ", ".join(sorted({str(x.get("status")) for x in found}))
                    elif found and not any(x.get("standard") for x in found if x.get("status") == "ok"):
                        row["note"] = "non-standard only"
                rows.append(row)
            if e.join:
                jr = str(e.join_reading)
                row = {"capability": cap, "benchmark": e.benchmark, "reading": jr, "model": model, "required": e.required,
                       "status": "pending", "note": "join", "hint": None}
                if e.status == "packaged":
                    if (model, e.benchmark, jr) in joined_set or (e.benchmark, jr) in stored:
                        row["status"] = "done"
                    else:
                        row["status"] = "join-missing" if store_root is not None else "join-unknown"
                        row["hint"] = join_command(e, model, runs_root)
                rows.append(row)
    return rows


def completeness_json(suite: Suite, rows: list[dict[str, Any]], *, results_root: str | os.PathLike | None = None) -> dict[str, Any]:
    """The ``{name, description, origin, path, results_root, rows, counts}`` block of ``report build --suite`` and
    ``suite missing``."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {"name": suite.name, "description": suite.description, "origin": suite.origin,
            "path": str(suite.path) if suite.path is not None else None,
            "results_root": str(results_root) if results_root is not None else None,
            "rows": rows, "counts": dict(sorted(counts.items()))}
