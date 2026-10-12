"""Benchmark tables generated from the folders: the folder is the single source of truth.

Three documents list the packaged benchmarks: the repo ``README.md``, the index ``yavih/benchmarks/README.md`` and
the skill's ``references/data.md``. Each holds its table between two marker comments::

    <!-- yavih:bench-table index begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
    ...
    <!-- yavih:bench-table index end -->

and this module renders the block from each folder's ``benchmark.yaml`` (read with pyyaml; nothing is loaded or
scanned), so adding a folder changes no hand-written row. ``python -m yavih.doctables --write`` rewrites the blocks
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

__all__ = ["KINDS", "SOURCE_WORDS", "render", "replace_blocks", "repo_documents", "main"]

SOURCE_WORDS: dict[str, str] = {"public_url": "public", "torrent": "torrent", "manual_portal": "manual portal",
                                "private_mirror": "private mirror", "derived": "derived"}
_BEGIN = re.compile(r"^<!-- yavih:bench-table (?P<kind>[a-z_]+) begin\b.*-->\s*$")
_END = "<!-- yavih:bench-table {kind} end -->"


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


def _index(folders: Mapping[str, Path]) -> str:
    """The ``benchmarks/README.md`` table: ``| Benchmark | Description | Readout | Media | Source |``."""
    rows = []
    for name, folder in folders.items():
        y = _yaml(folder)
        rows.append([f"[{name}]({name}/README.md)", f"{y.get('title')}{_aliases(y)}", str(y.get("readout")), _media(y),
                     f"{_source(y)} (obtain: {_tier(y)})"])
    return _table(["Benchmark", "Description", "Readout", "Media", "Source"], rows)


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


KINDS: dict[str, Callable[[Mapping[str, Path]], str]] = {
    "names": _names,
    "index": _index,
    "readme": _readme,
    "data": _data,
    "setup_links": _setup_list,
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
    """The documents with generated blocks: the packaged index and skill reference, plus the repo README when the
    package runs from a source checkout."""
    pkg = Path(__file__).resolve().parent
    docs = [pkg / "benchmarks" / "README.md", pkg / "skill" / "yavih" / "references" / "data.md"]
    readme = pkg.parent.parent / "README.md"
    if readme.is_file() and (pkg.parent.parent / "pyproject.toml").is_file():
        docs.insert(0, readme)
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
