"""The benchmark-folder contract as a tool: ``yavih bench lint <name|path> [--json]``.

A benchmark folder is the single source of truth for one protocol. This module checks that a folder, packaged or
not, has everything the harness, an agent and a reader of the docs rely on, so a new or edited folder cannot drift
from the rest and an out-of-tree folder (``$YAVIH_BENCH_PATH``, ``$YAVIH_HOME/benchmarks``, an entry-point pack, a
path) gets the same checks as the shipped ones. The repo test-suite runs it over every packaged folder.

What is checked (one ``CHECKS`` entry each, so a test or a caller can run a subset):

- ``yaml``: ``benchmark.yaml`` parses with pyyaml and loads strictly (``Benchmark.load``); ``name`` = folder name.
- ``hashes``: the newest ``changelog`` entry records the ``protocol_hash`` and ``input_hash`` the folder computes
  (``E_CHANGELOG_HASH``, with the exact lines to write).
- ``layout``: the required files, a non-empty ``prepare/``, ``classes.csv`` iff ``manifest.classes``, no
  ``__init__.py`` in the folder or its ``tests/`` (a folder is data + scripts, never a package).
- ``reader``: an optional folder ``reader.py`` accepts ``shard=`` (per-shard canaries); without one the generic
  ``yavih.sample.iter_samples`` reads the folder.
- ``budget``: ``budget.clips`` and ``manifest.upstream.curation`` agree with the frozen rows.
- ``header``: the ``header: {name, title, venue, abstract}`` block the README header image is rendered from
  (``python -m yavih.headers``; the image itself is not shipped and not required). When ``assets/header.png`` exists
  and the header template is at hand (a source checkout), ``assets/header.src.sha256`` must equal the sha256 of the
  current header HTML (``W_HEADER_STALE``, a warning: the image predates its yaml, chips or template).
- ``readme``: the first line is the header image ``![<header.name>](assets/header.png)``, the second paragraph the
  links line (naming ``setup.md``); the ``##`` sections Description, Citation (with a fenced ``bibtex`` block) and
  In yavih in that order; "## In yavih" names ``protocol_version``, the headline metric and the re-freeze
  (``make_manifest.py``); a printed ``protocol_hash`` is the current one.
- ``setup``: H1 ``# <name> — setup``; the six ``##`` sections in order; dependencies carry every ``external[].pip``
  (or say "None"); the contract yaml copies equal ``benchmark.yaml``; ``yavih bench verify <name>``;
  the prepare steps run as ``yavih bench prepare <name> <step>``; ``source.revision`` in "## Obtain the data".
- ``external``: every ``external[].pip`` is a runnable ``pip install …`` command.
- ``obtain``: the ``obtain: {tier, how}`` block.
- ``capability``: a non-empty ``capability:`` list of ``yavih.suites.CAPABILITIES`` ids (``E_CAPABILITY``; optional for
  the loader); a shipped suite that lists the benchmark under a capability it does not carry is a warning
  (``W_SUITE_CAPABILITY``).
- ``template``: ``bridge_template.py`` declares a literal ``describe()["model"]`` and a ``reading`` in a/v/av.
- ``citations``: the shipped docs cite nothing a pip-installed user cannot open (the repository's docs/ pages by
  relative path, the retired SPEC / DESIGN documents, repo notes,
  the source tree); a ``https://github.com/Bizilizi/yavih`` URL is the exception. No shipped text file of the folder
  (``*.md``, ``*.yaml``, ``*.py``, ``*.sh`` in the folder, ``prepare/``, ``tests/``) names a ``notes/`` path
  (``E_NOTES_LINK``): a folder may keep maintainer ``notes/``, but they are never packaged.
- ``imports``: ``reader.py`` / ``probe.py`` import with torch and the folder's external packages blocked.
- ``todo``: unfilled ``TODO`` markers (a ``yavih bench new`` scaffold) are reported as warnings: ``TODO`` in the
  folder's yaml, docs and scripts, ``TODO:`` in ``tests/*.py`` and ``prepare/*`` (where a test may name the bridge
  template's own ``TODO`` slots).

Errors fail the lint (exit 2, a contract violation); warnings do not. Stdlib + pyyaml at import time.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "CHECKS",
    "Finding",
    "HEADER_IMAGE",
    "LintReport",
    "OBTAIN_TIERS",
    "README_SECTIONS",
    "REQUIRED_FILES",
    "RETIRED_HEADINGS",
    "SETUP_SECTIONS",
    "fenced_blocks",
    "headings",
    "items",
    "lines_outside_fences",
    "lint",
    "lint_folder",
    "section",
    "template_readings",
]

#: Files every folder holds. ``classes.csv`` is required iff ``manifest.classes`` is set.
REQUIRED_FILES: tuple[str, ...] = ("README.md", "setup.md", "benchmark.yaml", "manifest.csv.gz",
                                   "bridge_template.py", "make_manifest.py", "tests/test_folder.py")
README_SECTIONS: tuple[str, ...] = ("Description", "Citation", "In yavih")
SETUP_SECTIONS: tuple[str, ...] = ("Obtain the data", "Dependencies", "Input contract", "Output contract",
                                   "Prepare & verify", "Gotchas")
#: The README's first line points at the header image (rendered by ``python -m yavih.headers``; GitHub only).
HEADER_IMAGE = "assets/header.png"
OBTAIN_TIERS: frozenset[str] = frozenset({"auto", "manual-download", "needs-licence", "needs-data-owner", "site-only"})
#: Headings neither README.md nor setup.md carries: the yaml ``changelog`` is the version record, and the
#: re-freeze notes live in README "## In yavih".
RETIRED_HEADINGS: frozenset[str] = frozenset({"Changelog", "History", "Checklist", "Maintainers"})
READINGS: frozenset[str] = frozenset({"a", "v", "av"})
#: A numbered prepare step (``prepare/01_download.sh``): what ``yavih bench prepare <name> <step>`` runs.
_STEP_FILE_RE = re.compile(r"^\d+_[A-Za-z0-9_.-]+\.(?:sh|py)$")
#: What a pip-installed user cannot open: the repository's docs/ pages by relative path (the wheel does not ship them;
#: link https://github.com/Bizilizi/yavih/blob/main/docs/<page>.md), the retired SPEC / DESIGN documents, repo notes/
#: paths, the source tree, a repo venv.
UNOPENABLE = re.compile(r"(^|[^/\w.-])docs/[\w-]+\.md|\bSPEC\b|DESIGN(\.md| \")|Philosophy v3|IMPLEMENTATION|(^|[^/A-Za-z_$])notes/|src/yavih/|"
                        r"\.venv/|repo root")
REPO_URL = "github.com/Bizilizi/yavih"
_REPO_URL_TOKEN = re.compile(r"\S*" + re.escape(REPO_URL) + r"\S*")
CITATION_DOCS: tuple[str, ...] = ("setup.md", "README.md", "bridge_template.py")
#: A ``notes/`` path (``notes/decisions.md``, ``$FOLDER/notes/…``, ``<name>/notes/…``): never shipped in the wheel.
NOTES_LINK = re.compile(r"\bnotes/")
#: The folder files the wheel ships and the notes-link rule reads (pyproject package-data: the folder, prepare/, tests/).
SHIPPED_DIRS: tuple[str, ...] = (".", "prepare", "tests")
SHIPPED_TEXT_SUFFIXES: frozenset[str] = frozenset({".md", ".yaml", ".yml", ".py", ".sh", ".txt"})
_TODO = re.compile(r"\bTODO\b")
#: The marker in ``tests/*.py`` and ``prepare/*``: ``TODO:`` (a folder test may assert on the template's ``TODO`` slots).
_TODO_MARK = re.compile(r"\bTODO:")
#: The folder files ``todo`` reads with ``_TODO`` (the bridge template's TODOs are for the bridge author).
_TODO_FILES: tuple[str, ...] = ("benchmark.yaml", "README.md", "setup.md", "reader.py", "probe.py", "make_manifest.py",
                                "classes.csv")


# ----------------------------------------------------------------------------- findings


@dataclass(frozen=True)
class Finding:
    """One contract finding. ``severity`` is ``"error"`` (fails the lint) or ``"warning"``."""

    code: str
    message: str
    check: str
    file: str | None = None
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "check": self.check, "file": self.file, "severity": self.severity}


@dataclass
class LintReport:
    folder: Path
    name: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity != "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "folder": str(self.folder), "ok": self.ok,
                "errors": [f.to_dict() for f in self.errors], "warnings": [f.to_dict() for f in self.warnings]}


# ----------------------------------------------------------------------------- markdown helpers

# a number as a whole token: not inside a longer number or a decimal ("2." at a sentence end is fine)
_NUM_L, _NUM_R = r"(?<!\d)(?<!\d\.)", r"(?!\d|\.\d)"
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")


def lines_outside_fences(text: str) -> list[tuple[int, str]]:
    """``(index, line)`` of every line not inside a fenced code block (``` or ~~~, CommonMark closing rule)."""
    out, fence = [], None
    for i, line in enumerate(text.splitlines()):
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence = m.group(1)
                continue
            out.append((i, line))
        elif m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not m.group(2).strip():
            fence = None
    return out


def headings(text: str, level: int) -> list[tuple[int, str]]:
    pat = re.compile(rf"^#{{{level}}}(?!#)\s+(.*?)\s*#*\s*$")
    return [(i, m.group(1)) for i, line in lines_outside_fences(text) if (m := pat.match(line))]


def section(text: str, level: int, heading: str) -> str:
    """Body of the heading at ``level`` named ``heading``, up to the next heading of the same or a higher level.
    Raises ``KeyError`` when the heading is absent."""
    lines = text.splitlines()
    starts = sorted((i, lvl, h) for lvl in range(1, level + 1) for i, h in headings(text, lvl))
    for k, (i, lvl, h) in enumerate(starts):
        if lvl == level and h == heading:
            end = next((j for j, _, _ in starts[k + 1:]), len(lines))
            return "\n".join(lines[i + 1:end])
    raise KeyError(f"heading {'#' * level} {heading} not found")


def fenced_blocks(body: str, info: str) -> list[str]:
    """The contents of the fenced blocks of ``body`` whose info string starts with ``info``."""
    blocks, fence, buf = [], None, []
    for line in body.splitlines():
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence, buf = (m.group(1), m.group(2).strip().split(" ")[0].lower()), []
        elif m and m.group(1)[0] == fence[0][0] and len(m.group(1)) >= len(fence[0]) and not m.group(2).strip():
            if fence[1] == info:
                blocks.append("\n".join(buf))
            fence = None
        else:
            buf.append(line)
    return blocks


def items(body: str) -> list[str]:
    """Bullets, numbered items and table rows, each joined with its continuation lines."""
    out: list[str] = []
    for line in body.splitlines():
        if re.match(r"^\s*([-*+]|\d+\.)\s+|^\s*\|", line):
            out.append(line)
        elif out and line.strip():
            out[-1] += " " + line.strip()
    return out


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str)


# ----------------------------------------------------------------------------- context


class _Ctx:
    """One folder under lint: lazily read texts, the raw yaml (pyyaml) and the loaded ``Benchmark``."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._texts: dict[str, str | None] = {}
        self.raw: dict[str, Any] | None = None
        self.raw_error: str | None = None
        self.bench: Any = None
        self.bench_error: str | None = None
        try:
            d = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
            if isinstance(d, dict):
                self.raw = d
            else:
                self.raw_error = f"benchmark.yaml is not a mapping ({type(d).__name__})"
        except (OSError, yaml.YAMLError) as e:
            self.raw_error = f"{e.__class__.__name__}: {e}"
        if self.raw is not None:
            from yavih.benchmark import Benchmark

            try:
                self.bench = Benchmark.load(str(folder))
            except Exception as e:  # noqa: BLE001 - every load failure is a finding
                self.bench_error = f"{e.__class__.__name__}: {e}"
        self.name = str((self.raw or {}).get("name") or folder.name)

    def text(self, rel: str) -> str | None:
        if rel not in self._texts:
            try:
                self._texts[rel] = (self.folder / rel).read_text(encoding="utf-8")
            except OSError:
                self._texts[rel] = None
        return self._texts[rel]


Check = Callable[[_Ctx], Iterable[Finding]]
#: ``(id, function)`` in run order. ``lint_folder(..., checks=[...])`` selects by id.
CHECKS: list[tuple[str, Check]] = []


def _check(cid: str) -> Callable[[Check], Check]:
    def deco(fn: Check) -> Check:
        CHECKS.append((cid, fn))
        return fn
    return deco


def _err(check: str, code: str, msg: str, file: str | None = None) -> Finding:
    return Finding(code=code, message=msg, check=check, file=file)


def _warn(check: str, code: str, msg: str, file: str | None = None) -> Finding:
    return Finding(code=code, message=msg, check=check, file=file, severity="warning")


# ----------------------------------------------------------------------------- checks


@_check("yaml")
def _c_yaml(c: _Ctx) -> Iterator[Finding]:
    if c.raw is None:
        yield _err("yaml", "E_YAML", c.raw_error or "benchmark.yaml unreadable", "benchmark.yaml")
        return
    if c.bench_error is not None:
        yield _err("yaml", "E_SCHEMA", c.bench_error, "benchmark.yaml")
    if c.raw.get("name") != c.folder.name:
        yield _err("yaml", "E_NAME", f"benchmark.yaml name {c.raw.get('name')!r} != folder name {c.folder.name!r}", "benchmark.yaml")


@_check("hashes")
def _c_hashes(c: _Ctx) -> Iterator[Finding]:
    if c.bench is None:
        return
    from yavih.hash_history import check_recorded

    for msg in check_recorded(c.bench):
        yield _err("hashes", "E_CHANGELOG_HASH", msg, "benchmark.yaml")


@_check("layout")
def _c_layout(c: _Ctx) -> Iterator[Finding]:
    for rel in REQUIRED_FILES:
        if not (c.folder / rel).is_file():
            yield _err("layout", "E_LAYOUT", f"required file {rel} missing", rel)
    prep = c.folder / "prepare"
    if not prep.is_dir() or not any(p.name != "__pycache__" for p in prep.iterdir()):
        yield _err("layout", "E_LAYOUT", "prepare/ is missing or empty", "prepare/")
    if c.raw is not None:
        wants = bool((c.raw.get("manifest") or {}).get("classes"))
        if (c.folder / "classes.csv").is_file() != wants:
            yield _err("layout", "E_LAYOUT", "classes.csv must exist iff manifest.classes is set", "classes.csv")
    for rel in ("__init__.py", "tests/__init__.py"):
        if (c.folder / rel).exists():
            yield _err("layout", "E_LAYOUT", f"{rel}: a folder is data + scripts, never a Python package", rel)


@_check("reader")
def _c_reader(c: _Ctx) -> Iterator[Finding]:
    if c.bench is None or not (c.folder / "reader.py").is_file():
        return
    try:
        params = inspect.signature(c.bench._folder_reader().iter_samples).parameters
    except Exception as e:  # noqa: BLE001
        yield _err("reader", "E_READER", f"reader.py does not load: {e.__class__.__name__}: {e}", "reader.py")
        return
    if "shard" not in params and not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        yield _err("reader", "E_READER", "iter_samples must accept shard=(i, n) (per-shard canaries)", "reader.py")


@_check("budget")
def _c_budget(c: _Ctx) -> Iterator[Finding]:
    if c.bench is None:
        return
    b = c.bench
    try:
        counts = {s: b.n_rows(s) for s in b.spec.splits.read_values()}
    except Exception as e:  # noqa: BLE001
        yield _err("budget", "E_MANIFEST", f"manifest unreadable: {e.__class__.__name__}: {e}", "manifest.csv.gz")
        return
    clips = dict((c.raw.get("budget") or {}).get("clips") or {})
    if clips != counts:
        yield _err("budget", "E_BUDGET", f"budget.clips {clips} != frozen rows {counts}", "benchmark.yaml")
    cur = ((c.raw.get("manifest") or {}).get("upstream") or {}).get("curation")
    if not isinstance(cur, dict):
        yield _err("budget", "E_CURATION", "manifest.upstream.curation is missing", "benchmark.yaml")
        return
    listed = {s: (v or {}).get("listed") for s, v in (cur.get("per_split") or {}).items()}
    if {s: listed.get(s) for s in counts} != counts:
        yield _err("budget", "E_CURATION", f"curation.per_split listed {listed} != frozen rows {counts}", "benchmark.yaml")
    try:
        total = sum(listed.values()) + sum((cur.get("excluded") or {}).values())
    except TypeError:
        total = None
    if total != cur.get("total_upstream"):
        yield _err("budget", "E_CURATION", f"listed + excluded = {total} != total_upstream {cur.get('total_upstream')}", "benchmark.yaml")


@_check("header")
def _c_header(c: _Ctx) -> Iterator[Finding]:
    if c.raw is None:
        return
    if not isinstance(c.raw.get("header"), dict):
        yield _err("header", "E_HEADER", "benchmark.yaml needs `header: {name, title, venue, abstract}` (the README "
                   "header image, `python -m yavih.headers`)", "benchmark.yaml")
        return
    if c.bench is None or not (c.folder / HEADER_IMAGE).is_file():
        return
    from yavih import headers

    if not (headers.TEMPLATE_DIR / "template.html").is_file():
        return  # an installed wheel: no template to compare with (and no assets/ shipped)
    try:
        want = headers.header_source_sha256(c.bench)
    except headers.HeaderError:
        return
    rel = f"assets/{headers.SOURCE_SHA_FILE}"
    got = (c.text(rel) or "").strip()
    if got != want:
        yield _warn("header", "W_HEADER_STALE", f"{HEADER_IMAGE} was not rendered from the current header ({rel} "
                    f"{'is missing' if not got else 'records ' + got[:12] + '…'}, the header HTML is {want[:12]}…): "
                    f"re-render and commit both with `python -m yavih.headers {c.name}` (without Chrome: the PNG from "
                    "the `headers` CI artifact, the sha file from `--html-only`)", HEADER_IMAGE)


def _retired_headings(text: str, f: str, check: str) -> Iterator[Finding]:
    for level in range(1, 7):
        for _, h in headings(text, level):
            if h in RETIRED_HEADINGS:
                yield _err(check, "E_README" if check == "readme" else "E_SETUP",
                           f"heading {'#' * level} {h}: not part of the folder docs (the benchmark.yaml changelog is the "
                           "version record; re-freeze notes go in README.md \"## In yavih\")", f)


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


@_check("readme")
def _c_readme(c: _Ctx) -> Iterator[Finding]:
    text, f = c.text("README.md"), "README.md"
    if text is None or c.raw is None:
        return
    y = c.raw
    paras = _paragraphs(text)
    name = (y.get("header") or {}).get("name") if isinstance(y.get("header"), dict) else None
    image = f"![{name}]({HEADER_IMAGE})" if name else None
    first = paras[0] if paras else ""
    if image is not None and first != image:
        yield _err("readme", "E_README", f"the first line must be the header image {image!r}, got {first[:80]!r}", f)
    if len(paras) < 2 or "(setup.md)" not in paras[1]:
        yield _err("readme", "E_README", "the second paragraph must be the links line (Paper · Homepage · Dataset · "
                   "[setup.md](setup.md), whichever exist)", f)
    yield from _retired_headings(text, f, "readme")
    h1 = headings(text, 1)
    if h1:
        yield _err("readme", "E_README", f"no H1 (the header image carries the name), got {[h for _, h in h1]}", f)
    got = [h for _, h in headings(text, 2)]
    if got != list(README_SECTIONS):
        yield _err("readme", "E_README", f"## headings {got} != {list(README_SECTIONS)}", f)
        return
    if not fenced_blocks(section(text, 2, "Citation"), "bibtex"):
        yield _err("readme", "E_README", "## Citation must hold a fenced bibtex block", f)
    proto = section(text, 2, "In yavih")
    pv = y.get("protocol_version")
    if "protocol_version" not in proto or not re.search(rf"{_NUM_L}{pv}{_NUM_R}", proto):
        yield _err("readme", "E_README", f"## In yavih must name protocol_version {pv}", f)
    headline = (y.get("metrics") or {}).get("headline")
    if not headline or headline not in proto:
        yield _err("readme", "E_README", f"## In yavih must name the headline metric {headline!r}", f)
    if "make_manifest" not in proto:
        yield _err("readme", "E_README", "## In yavih must say how the manifest is re-frozen (make_manifest.py)", f)
    if c.bench is not None:
        for h in re.findall(r"protocol_hash`?\s*`?\s*([0-9a-f]{12,64})", text):
            if not c.bench.protocol_hash.startswith(h):
                yield _err("readme", "E_README_HASH", f"quotes protocol_hash {h[:12]}…, current {c.bench.protocol_hash[:12]}…", f)


@_check("setup")
def _c_setup(c: _Ctx) -> Iterator[Finding]:
    text, f = c.text("setup.md"), "setup.md"
    if text is None or c.raw is None:
        return
    y, name = c.raw, c.name
    yield from _retired_headings(text, f, "setup")
    h1 = [h for _, h in headings(text, 1)]
    if h1 != [f"{name} — setup"]:
        yield _err("setup", "E_SETUP", f"H1 {h1} != ['{name} — setup']", f)
    got = [h for _, h in headings(text, 2)]
    if got != list(SETUP_SECTIONS):
        yield _err("setup", "E_SETUP", f"## headings {got} != {list(SETUP_SECTIONS)}", f)
        return
    deps = section(text, 2, "Dependencies")
    external = y.get("external") or []
    for ext in external:
        if str(ext.get("pip")) not in deps:
            yield _err("setup", "E_SETUP", f"## Dependencies must carry the pip line {ext.get('pip')!r} verbatim", f)
    if not external and not re.search(r"\bNone\b", deps):
        yield _err("setup", "E_SETUP", "## Dependencies must say 'None' when external is empty", f)
    for heading, key in (("Input contract", "input_contract"), ("Output contract", "output_contract")):
        blocks = fenced_blocks(section(text, 2, heading), "yaml")
        if not blocks:
            yield _err("setup", "E_SETUP", f"## {heading} must end with a fenced yaml copy of benchmark.yaml `{key}`", f)
            continue
        try:
            parsed = yaml.safe_load(blocks[-1])
        except yaml.YAMLError as e:
            yield _err("setup", "E_SETUP", f"## {heading}: the yaml block does not parse: {e}", f)
            continue
        if isinstance(parsed, dict) and set(parsed) == {key}:  # a copy that keeps the top-level key
            parsed = parsed[key]
        if key not in y:
            yield _err("setup", "E_SETUP", f"benchmark.yaml has no `{key}`", "benchmark.yaml")
        elif _canon(parsed) != _canon(y[key]):
            yield _err("setup", "E_SETUP", f"the last yaml block of ## {heading} differs from benchmark.yaml `{key}`", f)
    if f"yavih bench verify {name}" not in section(text, 2, "Prepare & verify"):
        yield _err("setup", "E_SETUP", f"## Prepare & verify must contain `yavih bench verify {name}`", f)
    obtain = section(text, 2, "Obtain the data")
    has_steps = any(_STEP_FILE_RE.match(q.name) for q in (c.folder / "prepare").glob("*")) if (c.folder / "prepare").is_dir() else False
    if has_steps and f"yavih bench prepare {name}" not in obtain:
        # one command surface for shipped files (a wheel install has no checkout to cd into): the steps run through
        # `yavih bench prepare <name> <step>`, which sets FOLDER and YAVIH_BENCH for them
        yield _err("setup", "E_SETUP", f"## Obtain the data must run its prepare steps through `yavih bench prepare {name} <step>`", f)
    rev = (y.get("source") or {}).get("revision")
    if rev and str(rev) not in obtain:
        yield _err("setup", "E_SETUP", f"## Obtain the data must contain source.revision {rev!r} (the pinned download)", f)


@_check("external")
def _c_external(c: _Ctx) -> Iterator[Finding]:
    for ext in (c.raw or {}).get("external") or []:
        if not str(ext.get("pip", "")).startswith("pip install "):
            yield _err("external", "E_EXTERNAL", f"external {ext.get('name')!r}: pip {ext.get('pip')!r} is not a full `pip install …` command", "benchmark.yaml")


@_check("obtain")
def _c_obtain(c: _Ctx) -> Iterator[Finding]:
    if c.raw is None:
        return
    ob = c.raw.get("obtain")
    if not isinstance(ob, dict) or set(ob) != {"tier", "how"}:
        yield _err("obtain", "E_OBTAIN", "benchmark.yaml needs `obtain: {tier, how}` (who must act before the data can be staged)", "benchmark.yaml")
        return
    if ob["tier"] not in OBTAIN_TIERS:
        yield _err("obtain", "E_OBTAIN", f"obtain.tier {ob['tier']!r} not in {sorted(OBTAIN_TIERS)}", "benchmark.yaml")
    if not str(ob.get("how") or "").strip():
        yield _err("obtain", "E_OBTAIN", "obtain.how is empty", "benchmark.yaml")


def _shipped_suite_groups() -> list[tuple[str, str, list[str]]]:
    """``(suite file name, group capability, benchmark names)`` of every shipped suite, read raw (pyyaml; a group
    that does not parse is skipped: ``yavih suite show`` reports it)."""
    from yavih.suites import shipped_dir

    out: list[tuple[str, str, list[str]]] = []
    for p in sorted(shipped_dir().glob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        groups = d.get("groups") if isinstance(d, dict) else None
        for g in groups if isinstance(groups, list) else []:
            if isinstance(g, dict):
                out.append((p.name, str(g.get("capability")),
                            [str(e.get("benchmark")) for e in g.get("entries") or [] if isinstance(e, dict)]))
    return out


@_check("capability")
def _c_capability(c: _Ctx) -> Iterator[Finding]:
    if c.raw is None:
        return
    from yavih.suites import CAPABILITIES

    caps = c.raw.get("capability")
    if not isinstance(caps, list) or not caps:
        yield _err("capability", "E_CAPABILITY", "benchmark.yaml needs `capability: [...]`, one or more of "
                   f"{', '.join(CAPABILITIES)} (the README and the benchmarks index group benchmarks by it)", "benchmark.yaml")
        return
    bad = [x for x in caps if x not in CAPABILITIES]
    if bad:
        yield _err("capability", "E_CAPABILITY", f"capability {bad} not in the vocabulary {', '.join(CAPABILITIES)}", "benchmark.yaml")
        return
    names = {c.name, *map(str, c.raw.get("aliases") or [])}
    for suite, cap, benches in _shipped_suite_groups():
        if names & set(benches) and cap not in caps:
            yield _warn("capability", "W_SUITE_CAPABILITY", f"the shipped suite {suite} lists {c.name} under capability "
                        f"{cap!r}, which is not among its capability tags {caps}", "benchmark.yaml")


def template_readings(src: str) -> set[str]:
    """The ``describe()["reading"]`` values a bridge template declares: a literal, or the ``choices`` of its
    ``ArgSpec``."""
    readings = set(re.findall(r'"reading":\s*"([^"]+)"', src))
    for block in re.findall(r'"reading":\s*ArgSpec\((.*?)\)\s*,\s*\n', src, flags=re.S):
        m = re.search(r"choices=\(([^)]*)\)", block)
        if m:
            readings |= set(re.findall(r'"([^"]+)"', m.group(1)))
    return readings


@_check("template")
def _c_template(c: _Ctx) -> Iterator[Finding]:
    src, f = c.text("bridge_template.py"), "bridge_template.py"
    if src is None:
        return
    if not re.search(r'^\s+"model": "[A-Za-z0-9_.-]{1,64}",', src, flags=re.M):
        yield _err("template", "E_TEMPLATE", 'describe() lacks a literal "model" (the name results are filed under)', f)
    readings = template_readings(src)
    if not readings or not readings <= READINGS:
        yield _err("template", "E_TEMPLATE", f"describe()['reading'] {sorted(readings)} must be in {sorted(READINGS)}", f)


def _shipped_texts(folder: Path) -> list[str]:
    """The folder-relative paths of the shipped text files (``SHIPPED_DIRS`` x ``SHIPPED_TEXT_SUFFIXES``), sorted."""
    out = []
    for d in SHIPPED_DIRS:
        base = folder / d
        if base.is_dir():
            out += [p.relative_to(folder).as_posix() for p in base.iterdir()
                    if p.is_file() and p.suffix in SHIPPED_TEXT_SUFFIXES]
    return sorted(out)


@_check("citations")
def _c_citations(c: _Ctx) -> Iterator[Finding]:
    for doc in _shipped_texts(c.folder):
        text = c.text(doc)
        if text is None:
            continue
        for k, line in enumerate(text.splitlines(), start=1):
            if NOTES_LINK.search(_REPO_URL_TOKEN.sub(" ", line)):  # a repo URL is openable; a path beside it is not
                yield _err("citations", "E_NOTES_LINK", f"line {k} names a notes/ path, which the wheel does not ship "
                           f"(maintainer notes stay in the source repository; use a {REPO_URL} URL or drop the "
                           f"pointer): {line.strip()[:140]!r}", doc)
            if doc in CITATION_DOCS and REPO_URL not in line and UNOPENABLE.search(NOTES_LINK.sub("", line)):
                yield _err("citations", "E_CITATION", f"line {k} cites what a pip-installed user cannot open "
                           f"(a relative docs/ page, notes/, the source tree; use a {REPO_URL} URL): {line.strip()[:140]!r}", doc)


_BLOCKED_IMPORT = r'''
import importlib.util, os, sys
sys.path.insert(0, sys.argv[2])  # the yavih the linter runs from, not whichever one -I would find
for m in sys.argv[3:]:
    sys.modules[m] = None   # `import m` raises ImportError, as in an env without the package
folder = sys.argv[1]
for f in ("reader.py", "probe.py"):
    p = os.path.join(folder, f)
    if not os.path.isfile(p):
        continue
    spec = importlib.util.spec_from_file_location("blocked_" + f[:-3], p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
print("ok")
'''


@_check("imports")
def _c_imports(c: _Ctx) -> Iterator[Finding]:
    externals = [str(e.get("name")) for e in ((c.raw or {}).get("external") or [])]
    yavih_root = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run([sys.executable, "-I", "-c", _BLOCKED_IMPORT, str(c.folder), yavih_root, "torch", *externals],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout.strip().endswith("ok"):
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["?"]
        yield _err("imports", "E_IMPORT", "reader.py / probe.py must import without torch and without the external "
                   f"packages (assert-import at call time): {tail[0][:300]}", "reader.py")


def _todo_scripts(folder: Path) -> list[str]:
    """``tests/*.py`` and the files of ``prepare/``, folder-relative and sorted."""
    out = [p for p in (folder / "tests").glob("*.py") if p.is_file()]
    if (folder / "prepare").is_dir():
        out += [p for p in (folder / "prepare").iterdir() if p.is_file()]
    return sorted(p.relative_to(folder).as_posix() for p in out)


@_check("todo")
def _c_todo(c: _Ctx) -> Iterator[Finding]:
    for rel, pattern in [*((r, _TODO) for r in _TODO_FILES), *((r, _TODO_MARK) for r in _todo_scripts(c.folder))]:
        try:
            text = c.text(rel)
        except UnicodeDecodeError:
            continue  # a binary prepare/ asset holds no marker
        if text is None:
            continue
        n = len(pattern.findall(text))
        if n:
            yield _warn("todo", "W_TODO", f"{n} TODO marker(s) still to fill", rel)


# ----------------------------------------------------------------------------- entry points


def lint_folder(folder: str | Path, *, checks: Iterable[str] | None = None, skip: Iterable[str] = ()) -> LintReport:
    """Run the folder contract on ``folder`` (a directory holding ``benchmark.yaml``)."""
    folder = Path(folder)
    if not (folder / "benchmark.yaml").is_file():
        return LintReport(folder=folder, name=folder.name, findings=[
            _err("yaml", "E_LAYOUT", f"{folder} holds no benchmark.yaml", "benchmark.yaml")])
    wanted = set(checks) if checks is not None else None
    unknown = (wanted or set()) | set(skip)
    unknown -= {cid for cid, _ in CHECKS}
    if unknown:
        raise ValueError(f"unknown lint checks {sorted(unknown)}; known: {[cid for cid, _ in CHECKS]}")
    ctx = _Ctx(folder)
    report = LintReport(folder=folder, name=ctx.name)
    for cid, fn in CHECKS:
        if (wanted is not None and cid not in wanted) or cid in skip:
            continue
        try:
            report.findings.extend(fn(ctx))
        except Exception as e:  # noqa: BLE001 - a crashing check is itself a finding
            report.findings.append(_err(cid, "E_LINT_CRASH", f"check {cid} crashed: {e.__class__.__name__}: {e}"))
    return report


def lint(ref: str, **kw: Any) -> LintReport:
    """``lint_folder`` on a benchmark name, alias or path (the ``Benchmark.load`` resolution order)."""
    from yavih.benchmark import UnknownBenchmark, resolve_folder

    p = Path(ref).expanduser()
    if p.is_dir():
        return lint_folder(p, **kw)
    folder = resolve_folder(ref)
    if folder is None:
        raise UnknownBenchmark(f"no benchmark folder named {ref!r} (a name, an alias or a directory holding benchmark.yaml)")
    return lint_folder(folder, **kw)
