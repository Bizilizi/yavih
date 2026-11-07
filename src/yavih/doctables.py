"""Benchmark tables generated from the folders: the folder is the single source of truth.

Every list of the repository's benchmarks is generated: in the repo ``README.md`` (the capability table), the controls
bullet of https://github.com/Bizilizi/yavih/blob/main/docs/quickstart.md, the
"Packaged benchmarks" table of https://github.com/Bizilizi/yavih/blob/main/docs/benchmark-folder.md, the index
``benchmarks/README.md`` at the repository root (the tables per capability, the ``setup.md`` links) and the skill's
``references/data.md`` (the obtainability table, the ``prepare_only`` roots) and ``references/readouts.md`` (the
benchmarks per readout). The ``capabilities`` kind (one row per capability with a few linked examples and a link to
the index) is written for the repo README and links by GitHub URL. Each holds its block between two marker
comments::

    <!-- yavih:bench-table index begin (generated from each folder's benchmark.yaml: python -m yavih.doctables --write) -->
    ...
    <!-- yavih:bench-table index end -->

and this module renders the block from each folder's ``benchmark.yaml`` (read with pyyaml; nothing is loaded or
scanned; ``eval.py`` is only tested for existence), so adding a folder changes no hand-written row. ``python -m yavih.doctables --write`` rewrites the blocks
in place; ``--check`` exits 1 when a committed block differs (the repo test does the same). ``yavih bench list
--fmt md`` prints the ``readme`` table for any set of discoverable folders.

A folder's own ``README.md`` (the human page) carries one more generated block, the "at a glance" table, between its
own markers::

    <!-- yavih:glance begin (generated from benchmark.yaml: python -m yavih.doctables --glance <name> --write) -->
    ...
    <!-- yavih:glance end -->

rendered from that folder's ``benchmark.yaml`` (plus the row count of ``classes.csv`` and the readings of
``bridge_template.py``; nothing is loaded): task, modality, size, headline, readings, data, licence, protocol.
``--glance <name>`` prints it, ``--glance <name> --write`` rewrites one folder's README (inside a benchmark
contribution's footprint), plain ``--write`` also refreshes every packaged folder README that holds a block, and
``bench lint`` reports a missing or stale one (``E_README_GLANCE``).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from yavih.fields import cell_fields, file_fields

__all__ = ["GLANCE_BEGIN", "GLANCE_END", "KINDS", "SCORER_WORDS", "SOURCE_WORDS", "UNTAGGED", "glance_block", "glance_documents",
           "main", "render", "render_glance", "replace_blocks", "replace_glance", "repo_documents"]

SOURCE_WORDS: dict[str, str] = {"public_url": "public", "torrent": "torrent", "manual_portal": "manual portal",
                                "private_mirror": "private mirror", "derived": "derived"}
_BEGIN = re.compile(r"^<!-- yavih:bench-table (?P<kind>[a-z_]+) begin\b.*-->\s*$")
_END = "<!-- yavih:bench-table {kind} end -->"
#: The group of a folder whose ``benchmark.yaml`` declares no ``capability`` (``bench lint`` requires one).
UNTAGGED = "untagged"
#: Where the ``capabilities`` kind links: the packaged folders on GitHub (it is written for the repo README, which
#: PyPI renders too, so relative links would not resolve).
_BENCH_URL = "https://github.com/Bizilizi/yavih/blob/main/benchmarks"
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
    """The kinds handed over: file fields by media kind, cell fields as ``<kind> (cell)`` (a caption: ``text (cell)``)."""
    kinds = [str((s or {}).get("kind")) for s in file_fields(y.get("fields")).values()]
    kinds += [f"{(c or {}).get('kind')} (cell)" for c in cell_fields(y.get("fields")).values()]
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
    return f"{y.get('scorer')}" + (f" ({', '.join(extra)})" if extra else "")


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
            rows.append([f"[{name}]({name}/README.md)", f"{y.get('title')}{_aliases(y)}", str(y.get("scorer")),
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
        rows.append([f"`{name}`", f"{y.get('title')}{_aliases(y)}", f"`{y.get('scorer')}`",
                     f"`{(y.get('metrics') or {}).get('headline')}`", _source(y), _rows(y)])
    return _table(["benchmark", "task", "scorer", "headline", "source", "frozen rows"], rows)


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
    return _table(["benchmark", "scorer", "source", "media touched (frozen)", "frozen rows", "handed over", "obtain",
                   "how to stage it"], rows)


def _setup_list(folders: Mapping[str, Path]) -> str:
    """``benchmarks/README.md``: the setup.md links, one per folder."""
    return " ·\n".join(f"[{name}]({name}/setup.md)" for name in folders) + "."


def _names(folders: Mapping[str, Path]) -> str:
    return ", ".join(f"`{name}`" for name in folders) + "."


def _and(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _folder(folders: Mapping[str, Path]) -> str:
    """The benchmark-folder page's "Packaged benchmarks" table: readout, label kind, streams, headline, ``eval.py``."""
    rows = []
    for name, folder in folders.items():
        y = _yaml(folder)
        folds = (y.get("splits") or {}).get("folds")
        label = str((((y.get("manifest") or {}).get("label")) or {}).get("kind"))
        if label == "none" and y.get("targets"):
            label += " (targets)"
        streams = ", ".join([*file_fields(y.get("fields")), *(f"{n} (cell)" for n in cell_fields(y.get("fields")))])
        if ((y.get("input_contract") or {}).get("materialize")) is not None:
            streams += " (materialized)"
        probe = "yes" if (folder / "eval.py").is_file() else ""
        if probe and y.get("external"):
            probe += ", `external`"
        rows.append([f"{name}{' (' + _aliases(y)[2:] + ')' if _aliases(y) else ''}",
                     f"{y.get('scorer')}" + (f" ({len(folds)} folds)" if folds else ""), label, streams,
                     f"`{(y.get('metrics') or {}).get('headline')}`", probe])
    return _table(["benchmark", "scorer", "label kind", "streams", "headline", "eval.py"], rows)


def _readout_groups(folders: Mapping[str, Path]) -> str:
    """Skill ``references/readouts.md``: the packaged benchmarks per readout: the built-in scorers in their fixed order,
    then every other scorer (a folder's own, a pack's) by name. Never the registry's order, which depends on which
    folders a process happened to load first."""
    from yavih.readouts import registry

    builtin = [m.rsplit(".", 1)[-1] for m in registry.BUILTIN]
    groups: dict[str, list[str]] = {r: [] for r in builtin}
    for name, folder in folders.items():
        groups.setdefault(str(_yaml(folder).get("scorer")), []).append(name)
    order = builtin + sorted(r for r in groups if r not in builtin)
    return "Packaged benchmarks per readout: " + " · ".join(
        f"`{r}` {', '.join(groups[r])}" for r in order if groups[r]) + "."


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


# ----------------------------------------------------------------------------- the README "at a glance" block

#: The glance block's markers (one block per folder README, before its first ``##`` heading).
GLANCE_BEGIN = ("<!-- yavih:glance begin (generated from benchmark.yaml: python -m yavih.doctables --glance {name} --write)"
                " -->")
GLANCE_END = "<!-- yavih:glance end -->"
_GLANCE_BEGIN = re.compile(r"^<!-- yavih:glance begin\b.*-->\s*$")
#: The task a core scorer scores, in words (a folder's own scorer is named as such).
SCORER_WORDS: dict[str, str] = {
    "clf": "classification", "knn": "classification (k-NN, no training)", "seq": "per-segment classification over time",
    "grid": "dense prediction on a spatial grid", "box": "box localisation", "stream": "online prediction on a stream",
}


def _first_sentence(text: str) -> str:
    """The first sentence of ``text`` (up to ``. `` followed by an upper-case letter), `` …`` when more follows."""
    text = " ".join(str(text).split())
    m = re.search(r"\.\s+(?=[A-Z])", text)
    return text if m is None else text[:m.start() + 1] + " …"


def _class_count(folder: Path, y: Mapping[str, Any]) -> int | None:
    rel = (y.get("manifest") or {}).get("classes")
    path = folder / str(rel) if rel else None
    if path is None or not path.is_file():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return max(0, len(lines) - 1)


def _readings(folder: Path) -> list[str]:
    """The readings ``bridge_template.py`` declares: a / v / av first, then the folder's own, sorted."""
    from yavih.lint import template_readings

    tpl = folder / "bridge_template.py"
    got = template_readings(tpl.read_text(encoding="utf-8")) if tpl.is_file() else set()
    base = [r for r in ("a", "v", "av", "t") if r in got]
    return base + sorted(got - set(base))


def _number(x: Any) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def render_glance(folder: str | Path) -> str:
    """The "at a glance" block of ``folder/README.md``, both markers included (no trailing newline), from the folder's
    ``benchmark.yaml`` (read with pyyaml), ``classes.csv`` and ``bridge_template.py``."""
    from yavih.headers import metric_label
    from yavih.readouts import registry

    folder = Path(folder)
    y = _yaml(folder)
    name = str(y.get("name") or folder.name)
    scorer = str(y.get("scorer"))
    builtin = {m.rsplit(".", 1)[-1] for m in registry.BUILTIN}
    label = str((((y.get("manifest") or {}).get("label")) or {}).get("kind"))
    if scorer in builtin:
        task = SCORER_WORDS.get(scorer, scorer) + (", multi-label" if label in ("multi", "multi_sets") else "") + f" (scorer `{scorer}`)"
    else:
        task = f"the folder's own scorer (`{scorer}`)" + (", multi-label" if label in ("multi", "multi_sets") else "")
    caps = [f"`{c}`" for c in y.get("capability") or []]
    if caps:
        task += " · " + ", ".join(caps)
    size = _rows(y)
    n_classes = _class_count(folder, y)
    if n_classes:
        size += f" · {_num(n_classes)} classes"
    m = y.get("metrics") or {}
    headline = str(m.get("headline"))
    head = f"{metric_label(headline)}, `{headline}`" if metric_label(headline) != headline else f"`{headline}`"
    chance = (m.get("chance") or {}).get(headline)
    if isinstance(chance, (int, float)):
        head += f" · chance {_number(chance)}"
    modality = _media(y)
    if "generation" in (y.get("capability") or []):  # what the model is given -> what it generates (output codecs)
        from yavih.headers import _CODEC_KIND

        schema = ((y.get("output_contract") or {}).get("schema")) or {}
        made = sorted({_CODEC_KIND.get(str((o or {}).get("type")), str(k)) for k, o in schema.items()})
        if made:
            modality += f" → generated {' + '.join(made)}"
    rows: list[list[str]] = [["**Task**", task], ["**Modality**", modality], ["**Size**", size], ["**Headline**", head]]
    readings = _readings(folder)
    if len(readings) > 1:
        rows.append(["**Readings**", " / ".join(f"`{r}`" for r in readings)])
    gb = (y.get("budget") or {}).get("media_gb")
    data = f"{_source(y)} source · obtain tier `{_tier(y)}`" + (f" · {_number(gb)} GB of media" if isinstance(gb, (int, float)) else "")
    rows.append(["**Data**", data + " · [setup.md](setup.md)"])
    licence = (y.get("source") or {}).get("licence")
    if licence:
        rows.append(["**Licence**", _first_sentence(licence)])
    proto = f"`protocol_version` {y.get('protocol_version')}"
    if y.get("core"):
        proto += f" · yavih `{y.get('core')}`"
    proto += f" · `yavih bench fetch {name}`" + _aliases(y)
    rows.append(["**Protocol**", proto])
    return "\n".join([GLANCE_BEGIN.format(name=name), _table(["At a glance", ""], rows), GLANCE_END])


def _glance_spans(text: str) -> list[tuple[int, int]]:
    """``(begin, end)`` line indices of every glance block of ``text``; a begin without an end raises ``ValueError``."""
    lines = text.splitlines()
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if _GLANCE_BEGIN.match(lines[i]):
            j = next((k for k in range(i + 1, len(lines)) if lines[k].strip() == GLANCE_END), None)
            if j is None:
                raise ValueError(f"line {i + 1}: {lines[i].strip()!r} has no {GLANCE_END!r}")
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


def glance_block(text: str) -> str | None:
    """The first glance block of ``text`` (both marker lines), ``None`` when there is none; ``ValueError`` on a begin
    marker without its end."""
    spans = _glance_spans(text)
    if not spans:
        return None
    lines = text.splitlines()
    return "\n".join(lines[spans[0][0]:spans[0][1] + 1])


def replace_glance(text: str, folder: str | Path) -> str:
    """``text`` (a folder README) with its glance block(s) re-rendered from ``folder``; without one, the block is
    inserted before the first ``##`` heading (at the end when there is none). ``ValueError`` on a begin marker
    without its end."""
    from yavih.lint import headings

    block = render_glance(folder)
    lines = text.splitlines()
    spans = _glance_spans(text)
    if spans:
        out: list[str] = []
        prev = 0
        for b, e in spans:
            out += lines[prev:b] + [block]
            prev = e + 1
        out += lines[prev:]
    else:
        h2 = headings(text, 2)
        at = h2[0][0] if h2 else len(lines)
        head = lines[:at]
        while head and not head[-1].strip():
            head.pop()
        out = head + ([""] if head else []) + [block, ""] + lines[at:]
    return "\n".join(out) + ("\n" if text.endswith("\n") or not spans else "")


def glance_documents() -> list[Path]:
    """The packaged folder READMEs that hold a glance block (what plain ``--write`` / ``--check`` refresh)."""
    out = []
    for folder in packaged().values():
        readme = folder / "README.md"
        if readme.is_file() and any(_GLANCE_BEGIN.match(ln) for ln in readme.read_text(encoding="utf-8").splitlines()):
            out.append(readme)
    return out


def _glance_folder(ref: str) -> Path:
    """A folder by name (packaged, project, fetched, ``$YAVIH_BENCH_PATH``, alias) or by path."""
    p = Path(ref)
    if (p / "benchmark.yaml").is_file():
        return p
    from yavih.benchmark import resolve_folder

    folder = resolve_folder(ref)
    if folder is None:
        raise SystemExit(f"python -m yavih.doctables --glance: unknown benchmark {ref!r} (a name or a folder path)")
    return Path(folder)


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
    """The documents with generated blocks: the skill references, plus, when the package runs from a source checkout,
    the repo README, the quickstart, the benchmark-folder page and the benchmarks index (``benchmarks/README.md``)."""
    pkg = Path(__file__).resolve().parent
    refs = pkg / "skill" / "yavih" / "references"
    docs = [refs / "data.md", refs / "readouts.md"]
    repo = pkg.parent.parent
    if (repo / "pyproject.toml").is_file():
        docs = [p for p in (repo / "README.md", repo / "docs" / "quickstart.md", repo / "docs" / "benchmark-folder.md",
                            repo / "benchmarks" / "README.md") if p.is_file()] + docs
    return docs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m yavih.doctables", description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true", help="rewrite the generated blocks in place")
    g.add_argument("--check", action="store_true", help="exit 1 when a committed block is stale")
    ap.add_argument("--print", dest="kind", choices=sorted(KINDS), help="print one table")
    ap.add_argument("--glance", metavar="NAME", help="one folder's README \"at a glance\" block: print it; with --write "
                    "rewrite that README only; with --check exit 1 when it is stale")
    ns = ap.parse_args(argv)
    if ns.kind:
        sys.stdout.write(render(ns.kind) + "\n")
        return 0
    if ns.glance:
        folder = _glance_folder(ns.glance)
        readme = folder / "README.md"
        if not ns.write and not ns.check:
            sys.stdout.write(render_glance(folder) + "\n")
            return 0
        text = readme.read_text(encoding="utf-8")
        new = replace_glance(text, folder)
        if new == text:
            print(f"ok: {readme}")
            return 0
        if ns.write:
            readme.write_text(new, encoding="utf-8")
        print(f"{'rewrote' if ns.write else 'stale'}: {readme}")
        return 0 if ns.write else 1
    stale = []
    if ns.write:  # the integrator's refresh: every packaged folder README that already holds a glance block
        for readme in glance_documents():
            text = readme.read_text(encoding="utf-8")
            new = replace_glance(text, readme.parent)
            if new != text:
                readme.write_text(new, encoding="utf-8")
                print(f"rewrote: {readme}")
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
