"""Benchmark tables generated from the folders: the folder is the single source of truth.

Every list of the packaged benchmarks is generated: in the repo ``README.md`` (the table, the controls bullet), the
"Packaged benchmarks" table of https://github.com/Bizilizi/yavih/blob/main/docs/benchmark-folder.md, the index
``yavih/benchmarks/README.md`` (the tables per capability, the ``setup.md`` links) and the skill's
``references/data.md`` (the obtainability table, the ``prepare_only`` roots) and ``references/readouts.md`` (the
benchmarks per readout). The ``capabilities`` kind (one row per capability with a few linked examples and a link to
the index) is written for the repo README and links by GitHub URL. Each holds its block between two marker
comments::

    <!-- yavih:bench-table index begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
    ...
    <!-- yavih:bench-table index end -->

and this module renders the block from each folder's ``benchmark.yaml`` (read with pyyaml; nothing is loaded or
scanned; ``probe.py`` is only tested for existence), so adding a folder changes no hand-written row. ``python -m yavih.doctables --write`` rewrites the blocks
in place; ``--check`` exits 1 when a committed block differs (the repo test does the same). ``yavih bench list
--fmt md`` prints the ``readme`` table for any set of discoverable folders.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

__all__ = ["KINDS", "SOURCE_WORDS", "UNTAGGED", "render", "replace_blocks", "repo_documents", "main"]

SOURCE_WORDS: dict[str, str] = {"public_url": "public", "torrent": "torrent", "manual_portal": "manual portal",
                                "private_mirror": "private mirror", "derived": "derived"}
_BEGIN = re.compile(r"^<!-- yavih:bench-table (?P<kind>[a-z_]+) begin\b.*-->\s*$")
_END = "<!-- yavih:bench-table {kind} end -->"
#: The group of a folder whose ``benchmark.yaml`` declares no ``capability`` (``bench lint`` requires one).
UNTAGGED = "untagged"
#: Where the ``capabilities`` kind links: the packaged folders on GitHub (it is written for the repo README, which
#: PyPI renders too, so relative links would not resolve).
_BENCH_URL = "https://github.com/Bizilizi/yavih/blob/main/src/yavih/benchmarks"
#: How many example benchmarks a ``capabilities`` row names.
_EXAMPLES = 4


def _num(n: Any) -> str:
    return f"{int(n):,}".replace(",", " ") if isinstance(n, int) else str(n)


def _yaml(folder: Path) -> dict[str, Any]:
    d = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    return d if isinstance(d, dict) else {}


def _aliases(y: Mapping[str, Any]) -> str:
    al = [f"`{a}`" for a in (y.get("aliases") or [])]
    return f"; alias{'es' if len(al) > 1 else ''} {', '.join(al)}" if al else ""


def _rows(y: Mapping[str, Any]) -> str:
    """Frozen rows per split (``budget.clips``, pinned to the manifest by ``bench lint``)."""
    clips = dict((y.get("budget") or {}).get("clips") or {})
    folds = (y.get("splits") or {}).get("folds")
    if folds:
        counts = [clips.get(str(f)) for f in folds]
        if len(set(counts)) == 1 and counts[0] is not None:
            return f"{_num(sum(counts))} in folds {folds[0]}–{folds[-1]} ({_num(counts[0])} each)"
        return " / ".join(f"{_num(clips.get(str(f)))} fold {f}" for f in folds)
    return " / ".join(f"{_num(n)} {s}" for s, n in clips.items())


def _media(y: Mapping[str, Any]) -> str:
    kinds = [str((s or {}).get("kind")) for s in (y.get("streams") or {}).values()]
    seen: list[str] = []
    for k in kinds:
        if k not in seen:
            seen.append(k)
    out = " + ".join(seen)
    if ((y.get("input_contract") or {}).get("materialize")) is not None:
        out += " (materialized at prepare time)"
    return out


def _readout(y: Mapping[str, Any]) -> str:
    label = (((y.get("manifest") or {}).get("label")) or {}).get("kind")
    extra = [str(label)] if label and label not in ("single", "none") else []
    if (y.get("splits") or {}).get("folds"):
        extra.append("folds")
    if (y.get("splits") or {}).get("floor"):
        extra.append("floor role")
    return f"{y.get('readout')}" + (f" ({', '.join(extra)})" if extra else "")


def _source(y: Mapping[str, Any]) -> str:
    return SOURCE_WORDS.get(str((y.get("source") or {}).get("kind")), str((y.get("source") or {}).get("kind")))


def _tier(y: Mapping[str, Any]) -> str:
    return str(((y.get("obtain") or {}).get("tier")) or "-")


def _cells(cells: Iterable[str]) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in cells) + " |"


def _table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join([_cells(header), "|" + "---|" * len(header), *(_cells(r) for r in rows)])


def _capability_groups(folders: Mapping[str, Path]) -> dict[str, list[str]]:
    """``{capability: names sorted}`` in ``yavih.suites.CAPABILITIES`` order, only non-empty groups; a benchmark
    with several capabilities is in each of their groups, one without any in :data:`UNTAGGED` (last)."""
    from yavih.suites import CAPABILITIES

    groups: dict[str, list[str]] = {c: [] for c in CAPABILITIES}
    for name, folder in folders.items():
        caps = [str(c) for c in _yaml(folder).get("capability") or []] or [UNTAGGED]
        for c in dict.fromkeys(caps):
            groups.setdefault(c, []).append(name)
    if UNTAGGED in groups:
        groups[UNTAGGED] = groups.pop(UNTAGGED)
    return {c: sorted(names) for c, names in groups.items() if names}


def _index(folders: Mapping[str, Path]) -> str:
    """The ``benchmarks/README.md`` index, grouped by capability: per group a ``### <capability> (<n>)`` heading and
    the table ``| Benchmark | Description | Readout | Media | Source |`` sorted by name. A benchmark with several
    capabilities is listed under each."""
    blocks = []
    for cap, names in _capability_groups(folders).items():
        rows = []
        for name in names:
            y = _yaml(folders[name])
            rows.append([f"[{name}]({name}/README.md)", f"{y.get('title')}{_aliases(y)}", str(y.get("readout")),
                         _media(y), f"{_source(y)} (obtain: {_tier(y)})"])
        blocks.append(f"### {cap} ({len(names)})\n\n" + _table(["Benchmark", "Description", "Readout", "Media", "Source"], rows))
    return "\n\n".join(blocks)


def _capabilities(folders: Mapping[str, Path]) -> str:
    """The repo README's capability overview: ``| capability | benchmarks | examples |``, one row per capability that
    has a benchmark (the first four names by name, each linked to its folder README, ``+ N more`` after them), then
    a link to the full index. Links are GitHub URLs."""
    rows = []
    for cap, names in _capability_groups(folders).items():
        shown = ", ".join(f"[{n}]({_BENCH_URL}/{n}/README.md)" for n in names[:_EXAMPLES])
        more = f" + {len(names) - _EXAMPLES} more" if len(names) > _EXAMPLES else ""
        rows.append([f"`{cap}`", str(len(names)), shown + more])
    n = len(folders)
    return (_table(["capability", "benchmarks", "examples"], rows)
            + f"\n\n[All {n} benchmark{'s' if n != 1 else ''} → the index]({_BENCH_URL}/README.md)")


def _readme(folders: Mapping[str, Path]) -> str:
    """Repo ``README.md`` / ``yavih bench list --fmt md``: name, task, readout, headline, source, frozen rows."""
    rows = []
    for name, folder in folders.items():
        y = _yaml(folder)
        rows.append([f"`{name}`", f"{y.get('title')}{_aliases(y)}", f"`{y.get('readout')}`",
                     f"`{(y.get('metrics') or {}).get('headline')}`", _source(y), _rows(y)])
    return _table(["benchmark", "task", "readout", "headline", "source", "frozen rows"], rows)


def _data(folders: Mapping[str, Path]) -> str:
    """Skill ``references/data.md``: the obtainability and budget table from the frozen yaml."""
    rows = []
    for name, folder in folders.items():
        y = _yaml(folder)
        budget = y.get("budget") or {}
        gb = budget.get("media_gb")
        src = y.get("source") or {}
        sbytes = src.get("bytes")
        rows.append([f"{name}{' (' + _aliases(y)[2:] + ')' if _aliases(y) else ''}", _readout(y),
                     f"`{src.get('kind')}`" + (f"; {sbytes / 1e9:.1f} GB = `source.bytes`" if isinstance(sbytes, (int, float)) else ""),
                     f"{gb:g} GB = `budget.media_gb`" if isinstance(gb, (int, float)) else "TODO",
                     _rows(y), _media(y), _tier(y), f"`yavih bench info {name} --setup`"])
    return _table(["benchmark", "readout", "source", "media touched (frozen)", "frozen rows", "handed over", "obtain",
                   "how to stage it"], rows)


def _setup_list(folders: Mapping[str, Path]) -> str:
    """``benchmarks/README.md``: the setup.md links, one per folder."""
    return " ·\n".join(f"[{name}]({name}/setup.md)" for name in folders) + "."


def _names(folders: Mapping[str, Path]) -> str:
    return ", ".join(f"`{name}`" for name in folders) + "."


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _folder(folders: Mapping[str, Path]) -> str:
    """The benchmark-folder page's "Packaged benchmarks" table: readout, label kind, streams, headline, ``probe.py``."""
    rows = []
    for name, folder in folders.items():
        y = _yaml(folder)
        folds = (y.get("splits") or {}).get("folds")
        label = str((((y.get("manifest") or {}).get("label")) or {}).get("kind"))
        if label == "none" and y.get("targets"):
            label += " (targets)"
        streams = ", ".join(y.get("streams") or {})
        if ((y.get("input_contract") or {}).get("materialize")) is not None:
            streams += " (materialized)"
        probe = "yes" if (folder / "probe.py").is_file() else ""
        if probe and y.get("external"):
            probe += ", `external`"
        rows.append([f"{name}{' (' + _aliases(y)[2:] + ')' if _aliases(y) else ''}",
                     f"{y.get('readout')}" + (f" ({len(folds)} folds)" if folds else ""), label, streams,
                     f"`{(y.get('metrics') or {}).get('headline')}`", probe])
    return _table(["benchmark", "readout", "label kind", "streams", "headline", "probe.py"], rows)


def _readout_groups(folders: Mapping[str, Path]) -> str:
    """Skill ``references/readouts.md``: the packaged benchmarks per readout (registry order, then any other)."""
    from yavih.readouts import registry

    groups: dict[str, list[str]] = {r: [] for r in registry.names()}
    for name, folder in folders.items():
        groups.setdefault(str(_yaml(folder).get("readout")), []).append(name)
    return "Packaged benchmarks per readout: " + " · ".join(
        f"`{r}` {', '.join(names)}" for r, names in groups.items() if names) + "."


def _prepare_roots(folders: Mapping[str, Path]) -> str:
    """Skill ``references/data.md``: the ``data.roots`` marked ``prepare_only``, per folder."""
    out = []
    for name, folder in folders.items():
        roots = [f"`{r}`" for r, spec in (((_yaml(folder).get("data") or {}).get("roots")) or {}).items()
                 if isinstance(spec, dict) and spec.get("prepare_only")]
        if roots:
            out.append(f"{name} {_and(roots)}")
    return ("Roots marked `prepare_only`: " + "; ".join(out) + ".") if out else "No root is marked `prepare_only`."


def _controls(folders: Mapping[str, Path]) -> str:
    """Repo ``README.md`` "What yavih enforces": the folders whose ``metrics.controls`` void a report row."""
    from yavih.report.compare import _VOIDING_CONTROLS

    names, controls = [], []
    for name, folder in folders.items():
        mine = [c for c in ((_yaml(folder).get("metrics") or {}).get("controls") or []) if c in _VOIDING_CONTROLS]
        if mine:
            names.append(f"`{name}`")
            controls += [f"`{c}`" for c in mine if f"`{c}`" not in controls]
    if not names:
        return "- No benchmark emits a control that voids a row."
    return (f"- {_and(names)} emit{'s' if len(names) == 1 else ''} controls ({', '.join(controls)}); the report greys "
            "out the rows they void.")


KINDS: dict[str, Callable[[Mapping[str, Path]], str]] = {
    "names": _names,
    "index": _index,
    "capabilities": _capabilities,
    "readme": _readme,
    "data": _data,
    "setup_links": _setup_list,
    "folder": _folder,
    "readout_groups": _readout_groups,
    "prepare_roots": _prepare_roots,
    "controls": _controls,
}


def packaged() -> dict[str, Path]:
    from yavih.benchmarks import packaged_folder, packaged_names

    return {n: p for n in packaged_names() if (p := packaged_folder(n)) is not None}


def render(kind: str, folders: Mapping[str, Path] | None = None) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown table kind {kind!r}; known: {sorted(KINDS)}")
    return KINDS[kind](packaged() if folders is None else folders)


def replace_blocks(text: str, folders: Mapping[str, Path] | None = None) -> str:
    """``text`` with every marked block re-rendered (the marker lines themselves are kept)."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _BEGIN.match(lines[i].rstrip("\n"))
        out.append(lines[i])
        i += 1
        if not m:
            continue
        end = _END.format(kind=m.group("kind"))
        j = i
        while j < len(lines) and lines[j].strip() != end:
            j += 1
        if j == len(lines):
            raise ValueError(f"marker {lines[i - 1].strip()!r} has no {end!r}")
        out.append(render(m.group("kind"), folders) + "\n")
        out.append(lines[j])
        i = j + 1
    return "".join(out)


def repo_documents() -> list[Path]:
    """The documents with generated blocks: the packaged index and skill references, plus the repo README and
    the benchmark-folder page of the repository's docs when the package runs from a source checkout."""
    pkg = Path(__file__).resolve().parent
    refs = pkg / "skill" / "yavih" / "references"
    docs = [pkg / "benchmarks" / "README.md", refs / "data.md", refs / "readouts.md"]
    repo = pkg.parent.parent
    if (repo / "pyproject.toml").is_file():
        docs = [p for p in (repo / "README.md", repo / "docs" / "benchmark-folder.md") if p.is_file()] + docs
    return docs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m yavih.doctables", description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true", help="rewrite the generated blocks in place")
    g.add_argument("--check", action="store_true", help="exit 1 when a committed block is stale")
    ap.add_argument("--print", dest="kind", choices=sorted(KINDS), help="print one table")
    ns = ap.parse_args(argv)
    if ns.kind:
        sys.stdout.write(render(ns.kind) + "\n")
        return 0
    stale = []
    for doc in repo_documents():
        text = doc.read_text(encoding="utf-8")
        new = replace_blocks(text)
        if new != text:
            stale.append(doc)
            if ns.write:
                doc.write_text(new, encoding="utf-8")
    for doc in stale:
        print(f"{'rewrote' if ns.write else 'stale'}: {doc}")
    return 1 if stale and not ns.write else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
