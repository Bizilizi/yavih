"""Comparability and the ``report()`` entry point.

Comparability is decided only by what yavih controls: two results are comparable iff the benchmark name
(after alias), ``eval_hash = sha256(canon({protocol_hash, probe_hash}))`` and ``standard`` agree. Bridge
decisions (``describe()``, ``describe_sha256``, weights, dtype, batch size) are recorded and shown, never a key.

Selection (both views): results are keyed ``(benchmark, eval_hash, model, reading)``; within a key the
configurations (``bridge.describe_sha256``) are told apart — the newest is shown, the others go under
``--detail`` (``W_BRIDGE_CHANGED``) — and within the shown configuration the feature sets
(``features.fingerprint``) — the newest is shown (``W_FEATURES_DIFFER``). Seeds aggregate over ONE
configuration and ONE feature set, deduplicated on ``probe.seed``. Neither split ever makes two rows.
Results a re-probe moved to ``probe/_stale/`` (index records ``moved_aside``) are never shown: one whose
``describe_sha256`` differs from every live configuration of its row key is an older configuration of that row
(``W_BRIDGE_CHANGED``, ``--detail``; the in-place reconfiguration of one run directory); one of a live configuration
(a re-extraction or a re-probe of the same configuration) is ignored.

View 1: per model, the lm-evaluation-harness table ``benchmark | reading | version | eval_hash | readout |
metric | value | ci95 | floor`` (:func:`lm_eval_tables`; the floor is the one whose ``floor_of`` names the
metric). View 2: columns ``(benchmark, eval_hash)`` (standard and non-standard apart), rows ``(model,
reading)``; ``mean ± std`` with ``n_seeds``; a row is greyed when its headline CI overlaps its floor or it
carries ``floor_violated`` / ``audio_blind`` — never because of a bridge decision (a ``custom`` rule, an
unfingerprinted checkpoint); ``population differs`` marks a column whose eval ``ids_sha256`` disagree;
superseded ``protocol_version``\\s are listed; readings are grouped by ``model`` with ``MISSING`` cells (only
for readings whose stream modalities the column's benchmark has).

A result of a capped run (``plan build --sample N``: ``result["sample"]``) is never mixed with a full run's: within a
row key it forms its own ``partial`` row (view 1 and view 2), marked in every format (``partial: true`` and the flag
``partial`` in json, the ``(partial)`` suffix on its values in md, the ``partial`` column and flag in csv). Such a
result never reaches the results store, so ``--merge-results`` never sees it.

Ties: in a standard column, the values of one metric that lie within that metric's conformance tolerance of the best
value are a tie, not an order (:func:`_mark_ties`): each such cell carries ``tie: true`` (json), the ``tie`` column
(csv) or a leading ``≈`` (md), and tied rows are never ordered by value.
stdlib only (the folder ``join`` is a lazy import).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yavih import hashing
from yavih.report import tables
from yavih.report.index import MOVED_ASIDE, build_index, runs_root_of
from yavih.versions import __version__

__all__ = [
    "describe_diff",
    "by_fingerprint",
    "row_key",
    "select_newest",
    "add_moved_aside",
    "Selection",
    "lm_eval_tables",
    "Report",
    "ReportError",
    "report",
    "comparable",
    "column_key",
    "floor_name_for",
    "build_report",
    "is_partial",
    "DEFAULT_MIN_EVAL_COVERAGE",
    "REPORT_DIR",
]

#: ``splits.min_eval_coverage`` default used when the benchmark yaml is not loadable.
DEFAULT_MIN_EVAL_COVERAGE = 0.95
#: Where the rendered tables and the ``DONE`` marker go, under the runs root.
REPORT_DIR = "_report"
#: Controls whose truthiness voids the headline (avsync / vggss_box).
_VOIDING_CONTROLS = ("floor_violated", "audio_blind")
#: ``yavih.conformance.DEFAULT_TOLERANCE``, repeated here because this module is stdlib only (the conformance module
#: needs numpy); ``tests/test_report.py`` checks that the two agree.
DEFAULT_TOLERANCE = 0.001
#: Disclosure-like names never chosen as a fallback headline.
_COUNT_PREFIX = "n_"


class ReportError(Exception):
    """A refused comparison (exit 2): e.g. eval populations differ below ``min_eval_coverage`` and
    ``allow_coverage_diff`` is false."""

    def __init__(self, code: str, message: str, exit_code: int = 2) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.exit_code = exit_code


@dataclass
class Report:
    """Rendered tables plus the structured rows (``report build --fmt json``)."""

    columns: list[dict[str, Any]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    superseded: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    fmt: str = "md"
    detail: dict[str, Any] = field(default_factory=dict)
    joins: dict[str, Any] = field(default_factory=dict)
    n_results: int = 0
    paths: dict[str, str] = field(default_factory=dict)
    #: view 1: ``{model: [row]}`` with rows ``{benchmark, reading, version, eval_hash, readout, metric, value,
    #: std, n_seeds, ci95, floor, status, reason}``
    lm_eval: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    merged: list[str] = field(default_factory=list)
    #: ``report(suite=...)``: the suite completeness appendix ``{name, description, origin, path, results_root, rows,
    #: counts}`` with rows ``{capability, benchmark, reading, model, status, required, note, hint}`` (``yavih.suites``)
    suite: dict[str, Any] | None = None
    #: ``--detail``: the markdown also shows the metrics outside a column's ``metrics.table`` and a join's full output
    show_all: bool = False

    def to_json(self) -> dict:
        return {
            "lm_eval": self.lm_eval,
            "merged": self.merged,
            "suite": self.suite,
            "columns": self.columns,
            "rows": self.rows,
            "warnings": self.warnings,
            "superseded": self.superseded,
            "fmt": self.fmt,
            "detail": self.detail,
            "joins": self.joins,
            "n_results": self.n_results,
            "paths": self.paths,
        }


# --------------------------------------------------------------------------- comparability


def _alias_map() -> dict[str, str]:
    """``alias → canonical name`` over the packaged benchmarks (``benchmark.yaml`` ``aliases``), or ``{}`` when
    the benchmark loader is unavailable. ``result.json`` already records the canonical name
    (``Benchmark.load(alias)`` resolves before anything is written), so this is a safety net."""
    try:
        from yavih.benchmark import load_yaml, packaged_benchmarks  # lazy: numpy + pyyaml

        out: dict[str, str] = {}
        for name in packaged_benchmarks():
            spec = _packaged_spec(name, load_yaml)
            if spec is None:
                continue
            for alias in getattr(spec, "aliases", ()) or ():
                out[str(alias)] = name
        return out
    except Exception:  # noqa: BLE001 - any failure means "no alias information"
        return {}


def _packaged_spec(name: str, load_yaml) -> Any | None:
    try:
        from importlib import resources

        folder = resources.files("yavih.benchmarks") / name
        yaml_path = folder / "benchmark.yaml"
        if not yaml_path.is_file():
            return None
        return load_yaml(str(yaml_path))
    except Exception:  # noqa: BLE001
        return None


def _bench_meta(name: str) -> dict[str, Any]:
    """``{headline, table, tolerance, min_eval_coverage, protocol_version, stream_kinds}`` from the packaged yaml when
    loadable (``table`` / ``tolerance`` only when ``metrics.table`` / ``metrics.tolerance`` is declared); ``{}``
    otherwise (the report then falls back to defaults and heuristics)."""
    try:
        from yavih.benchmark import load_yaml

        spec = _packaged_spec(name, load_yaml)
        if spec is None:
            return {}
        out: dict[str, Any] = {}
        metrics = getattr(spec, "metrics", None)
        headline = getattr(metrics, "headline", None)
        if headline:
            out["headline"] = str(headline)
        table = getattr(metrics, "table", None)
        if table:
            out["table"] = [str(m) for m in table]
        tolerance = getattr(metrics, "tolerance", None)
        if tolerance:
            out["tolerance"] = {str(k): float(v) for k, v in tolerance.items()}
        splits = getattr(spec, "splits", None)
        mec = getattr(splits, "min_eval_coverage", None)
        if isinstance(mec, (int, float)):
            out["min_eval_coverage"] = float(mec)
        pv = getattr(spec, "protocol_version", None)
        if isinstance(pv, int):
            out["protocol_version"] = pv
        streams = getattr(spec, "streams", None)
        if isinstance(streams, Mapping) and streams:
            kinds = {str(k): str(getattr(v, "kind", "") or (v.get("kind", "") if isinstance(v, Mapping) else "")) for k, v in streams.items()}
            out["stream_kinds"] = {k: v for k, v in kinds.items() if v}
        return out
    except Exception:  # noqa: BLE001
        return {}


def canonical_benchmark(name: str, aliases: Mapping[str, str] | None = None) -> str:
    aliases = _alias_map() if aliases is None else aliases
    return aliases.get(name, name)


def _eval_hash_of(result: dict) -> str:
    """``result["eval_hash"]``, else computed from the two components, else ``""``."""
    eh = result.get("eval_hash")
    if isinstance(eh, str) and eh:
        return eh
    ph = (result.get("benchmark") or {}).get("protocol_hash")
    qh = (result.get("probe") or {}).get("hash")
    if isinstance(ph, str) and ph and isinstance(qh, str) and qh:
        return hashing.eval_hash(ph, qh)
    return ""


def column_key(result: dict, aliases: Mapping[str, str] | None = None) -> tuple[str, str]:
    """``(benchmark.name after alias, eval_hash)``: what yavih controls, nothing else."""
    bench = result.get("benchmark") or {}
    return (canonical_benchmark(str(bench.get("name", "")), aliases), _eval_hash_of(result))


def row_key(result: dict, aliases: Mapping[str, str] | None = None) -> tuple[str, str, str, str]:
    """``(benchmark after alias, eval_hash, model, reading)``: the report row key."""
    return (*column_key(result, aliases), _model(result), _reading(result))


def comparable(a: dict, b: dict, aliases: Mapping[str, str] | None = None) -> bool:
    """Same benchmark (after alias), same ``eval_hash`` and both ``standard``. Nothing the bridge
    decided enters: two configurations of one model are comparable, and so are two models; a registry entry
    (``yavih.registry``) is compared by the same rule, and differences within a metric's recorded conformance
    tolerance are ties (:func:`_mark_ties`)."""
    if aliases is None:
        aliases = _alias_map()
    return bool(a.get("standard")) and bool(b.get("standard")) and column_key(a, aliases) == column_key(b, aliases)


# --------------------------------------------------------------------------- helpers


def is_partial(result: dict) -> bool:
    """A result of a capped run (``plan build --sample N``): it carries ``sample`` (``{split: [lo, hi]}``)."""
    return bool(result.get("sample"))


def _describe(result: dict) -> dict[str, Any]:
    bridge = result.get("bridge") or {}
    d = bridge.get("describe")
    return d if isinstance(d, dict) else {}


def _model(result: dict) -> str:
    """``result["model"]`` (``describe()["model"]``, the user's name for the model)."""
    return str(result.get("model") or _describe(result).get("model") or "")


def _reading(result: dict) -> str:
    return str(result.get("reading") or _describe(result).get("reading") or "")


def _describe_sha(result: dict) -> str:
    return str((result.get("bridge") or {}).get("describe_sha256") or "")


def _fingerprint(r: dict) -> str:
    f = r.get("features")
    return str((f or {}).get("fingerprint") or "") if isinstance(f, dict) else ""


def _created(r: dict) -> str:
    return str(r.get("created") or "")


def _newest(results: Sequence[dict]) -> str:
    return max((_created(r) for r in results), default="")


def _dedup_seeds(results: Sequence[dict]) -> list[dict]:
    """One result per ``probe.seed`` (the newest ``created`` wins), sorted by seed: ``n_seeds`` counts distinct seeds."""
    by_seed: dict[int, dict] = {}
    for r in results:
        sd = int((r.get("probe") or {}).get("seed", 0) or 0)
        if sd not in by_seed or _created(r) > _created(by_seed[sd]):
            by_seed[sd] = r
    return [by_seed[k] for k in sorted(by_seed)]


def _group_newest_first(results: Sequence[dict], key) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)
    return sorted(groups.items(), key=lambda kv: _newest(kv[1]), reverse=True)


def by_fingerprint(results: Sequence[dict]) -> list[tuple[str, list[dict]]]:
    """Results split by ``features.fingerprint``, newest feature set first, each deduplicated on
    ``probe.seed``: seeds are aggregated only over ONE feature set."""
    return [(fp, _dedup_seeds(rs)) for fp, rs in _group_newest_first(results, _fingerprint)]


@dataclass
class Selection:
    """What the report shows for one row key ``(benchmark, eval_hash, model, reading)``.

    ``shown`` = the seeds (deduplicated) of the newest feature set of the newest configuration; ``configs`` =
    every configuration ``[(describe_sha256, results)]`` newest first (``configs[0]`` is the shown one);
    ``feature_sets`` = the shown configuration's feature sets ``[(fingerprint, seeds)]`` newest first."""

    shown: list[dict]
    configs: list[tuple[str, list[dict]]]
    feature_sets: list[tuple[str, list[dict]]]
    #: ``describe_sha256`` of the configurations known only from moved-aside outputs (``probe/_stale/``): listed in
    #: ``configs`` after every live one, never shown (:func:`add_moved_aside`)
    moved_aside: frozenset[str] = frozenset()

    @property
    def describe_sha256(self) -> str:
        return self.configs[0][0] if self.configs else ""

    @property
    def fingerprint(self) -> str:
        return self.feature_sets[0][0] if self.feature_sets else ""


def select_newest(results: Sequence[dict]) -> Selection:
    """The report's selection for the results of ONE row key: group by ``bridge.describe_sha256`` (the
    configuration whose results hold the newest ``created`` is shown), then by ``features.fingerprint`` within
    it (the newest feature set is shown), then deduplicate seeds. Never averages across either split."""
    configs = _group_newest_first(results, _describe_sha)
    feature_sets = by_fingerprint(configs[0][1]) if configs else []
    shown = feature_sets[0][1] if feature_sets else []
    return Selection(shown=shown, configs=configs, feature_sets=feature_sets)


def add_moved_aside(sel: Selection, moved: Sequence[dict]) -> Selection:
    """The moved-aside results (``probe/_stale/``) of ONE row key become older configurations of the live
    selection when their ``describe_sha256`` differs from every live configuration (an in-place reconfiguration under a
    reused model name: the old numbers must not vanish silently); those of a live configuration are dropped (a
    re-extraction or re-probe of the same configuration is no "older configuration"). Never changes what is
    shown; a row key without live results gets no row from them."""
    if not sel.configs or not moved:
        return sel
    live = {d for d, _ in sel.configs}
    extra = _group_newest_first([r for r in moved if _describe_sha(r) not in live], _describe_sha)
    if not extra:
        return sel
    return Selection(shown=sel.shown, configs=[*sel.configs, *extra], feature_sets=sel.feature_sets,
                     moved_aside=frozenset(d for d, _ in extra))


def _flatten(d: Mapping[str, Any]) -> dict[str, Any]:
    """``describe()`` one level down (``input.video``, ``precision.autocast``, ``args.layer``)."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                items[f"{k}.{sk}"] = sv
        else:
            items[k] = v
    return items


def describe_diff(shown: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{key: {"shown": value, "other": value}}`` for the ``describe()`` keys (one level down) whose values
    differ between two configurations of one row key (``--detail``, ``W_BRIDGE_CHANGED``)."""
    a, b = _flatten(shown), _flatten(other)
    out: dict[str, dict[str, Any]] = {}
    for k in _ordered_union([a, b]):
        va, vb = a.get(k, None), b.get(k, None)
        if (k in a) != (k in b) or json.dumps(va, sort_keys=True, default=str) != json.dumps(vb, sort_keys=True, default=str):
            out[k] = {"shown": va, "other": vb}
    return out


def _unfingerprinted(result: dict) -> bool:
    """``describe()["weights_sha256"]`` declared null, or the result recorded ``W_WEIGHTS_UNFINGERPRINTED``."""
    d = _describe(result)
    return ("weights_sha256" in d and d["weights_sha256"] is None) or _has_warning([result], "W_WEIGHTS_UNFINGERPRINTED")


#: stream kind → modality (``streams.*.kind``): what a reading can be expected to need.
_MODALITY = {"audio": "audio", "video": "visual", "image": "visual"}


def _stream_modalities(result: dict, stream_kinds: Mapping[str, str] | None) -> frozenset[str] | None:
    """Modalities of the streams the bridge read (``describe()["streams"]``, names or ``{"name": …}``),
    via the benchmark's ``streams.*.kind`` or, for a stream the spec does not list, a name that IS a
    kind; ``None`` when any stream cannot be placed (the caller then keeps the reading expected)."""
    streams = _describe(result).get("streams") or []
    out: set[str] = set()
    for s in streams:
        name = str(s.get("name") if isinstance(s, Mapping) else s)
        kind = (stream_kinds or {}).get(name) or name
        mod = _MODALITY.get(kind)
        if mod is None:
            return None
        out.add(mod)
    return frozenset(out) if out else None


def _coverage_frac(block: Any) -> float | None:
    if not isinstance(block, dict):
        return None
    present, nominal = block.get("present"), block.get("nominal")
    if not isinstance(present, (int, float)) or not isinstance(nominal, (int, float)) or nominal <= 0:
        return None
    return float(present) / float(nominal)


def _mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]
    if not vals:
        return None, None
    mean = statistics.fmean(vals)
    # Sample std: seeds are replicates of one estimator; a single seed has no spread (None, not 0),
    # which is how the table marks single-seed rows.
    std = statistics.stdev(vals) if len(vals) >= 2 else None
    return mean, std


def _ordered_union(lists: Iterable[Iterable[str]]) -> list[str]:
    seen: dict[str, None] = {}
    for names in lists:
        for n in names:
            seen.setdefault(n, None)
    return list(seen)


def floor_name_for(metric: str, metric_names: Sequence[str], floor_names: Sequence[str]) -> str | None:
    """The registered floor that belongs to ``metric`` (the floor that greys its cell).

    Exact ``<metric>_floor`` first (``acc`` → ``acc_floor``, ``d1`` → ``d1_floor``,
    ``seg_acc_boundary`` → ``seg_acc_boundary_floor``). Otherwise a floor whose stem is a prefix of the
    metric (``iou_floor`` → ``iou_mean``, ``acc_floor`` → ``acc_mean`` under folds) — but only while no
    metric carries that stem itself (``acc_floor`` belongs to ``acc`` when both ``acc`` and ``acc_mean``
    exist); the longest stem wins. ``None`` when the metric has no floor."""
    if f"{metric}_floor" in floor_names:
        return f"{metric}_floor"
    best: str | None = None
    for f in floor_names:
        if not f.endswith("_floor"):
            continue
        stem = f[: -len("_floor")]
        if stem in metric_names or not metric.startswith(stem + "_"):
            continue
        if best is None or len(stem) > len(best) - len("_floor"):
            best = f
    return best


def _pick_headline(metric_names: Sequence[str], floor_names: Sequence[str], meta: dict[str, Any]) -> str | None:
    """``benchmark.yaml[metrics.headline]`` when loadable; else the first metric that has a floor
    (:func:`floor_name_for`); else the first non-count metric."""
    h = meta.get("headline")
    if h and h in metric_names:
        return h
    for m in metric_names:
        if floor_name_for(m, metric_names, floor_names) is not None:
            return m
    for m in metric_names:
        if not m.startswith(_COUNT_PREFIX):
            return m
    return metric_names[0] if metric_names else None


def _aggregate_metric(name: str, results: Sequence[dict], floor_name: str | None = None) -> dict[str, Any]:
    """One metric cell over seeds: ``value`` (mean), ``std``, ``floor`` (mean of per-seed floors; they
    depend on train targets only and should agree), ``ci`` ([mean lo, mean hi]), ``chance``.
    ``floor_name`` is the registered floor of this metric (:func:`floor_name_for`); the default is the
    literal ``<name>_floor``."""
    values = [r.get("metrics", {}).get(name) for r in results]
    scalar = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    cell: dict[str, Any] = {"value": None, "std": None, "floor": None, "ci": None, "chance": None, "n": len(scalar)}
    if scalar and all(isinstance(v, int) for v in scalar) and len(set(scalar)) == 1:
        # count disclosures (n_eval, n_boundary, ...) agree across seeds: an int, no spread
        cell["value"], cell["std"] = scalar[0], None
    elif scalar:
        cell["value"], cell["std"] = _mean_std(scalar)
    elif values and all(isinstance(v, list) for v in values if v is not None):
        # shape: list metrics (acc_per_fold) are carried through as the per-seed lists.
        cell["value"] = values[0] if len(values) == 1 else values
    floor_name = floor_name or f"{name}_floor"
    floors = [(r.get("floors") or {}).get(floor_name) for r in results]
    floors = [f for f in floors if isinstance(f, (int, float)) and not isinstance(f, bool)]
    if floors:
        cell["floor"], _ = _mean_std(floors)
    cis = [r.get("ci95", {}).get(name) for r in results]
    cis = [c for c in cis if isinstance(c, list) and len(c) == 2 and all(isinstance(x, (int, float)) for x in c)]
    if cis:
        lo, _ = _mean_std([c[0] for c in cis])
        hi, _ = _mean_std([c[1] for c in cis])
        cell["ci"] = [lo, hi]
    chance = [r.get("chance", {}).get(name) for r in results]
    chance = [c for c in chance if isinstance(c, (int, float)) and not isinstance(c, bool)]
    if chance:
        cell["chance"] = chance[0]
    return cell


def _below_floor(cell: dict[str, Any]) -> bool:
    """Headline CI overlaps (or lies under) its floor; without a CI the point value decides.
    Floors are all higher-is-better, so "clears" means ``lo > floor``."""
    floor = cell.get("floor")
    value = cell.get("value")
    if floor is None or not isinstance(value, (int, float)):
        return False
    ci = cell.get("ci")
    if ci and isinstance(ci[0], (int, float)):
        return ci[0] <= floor
    return value <= floor


def _control_fired(results: Sequence[dict], name: str) -> bool:
    for r in results:
        v = (r.get("controls") or {}).get(name)
        if isinstance(v, bool):
            if v:
                return True
        elif isinstance(v, (int, float)) and v:
            return True
    return False


def _has_warning(results: Sequence[dict], code: str) -> bool:
    for r in results:
        for w in r.get("warnings") or []:
            if isinstance(w, dict) and w.get("code") == code:
                return True
    return False


def waived_of(results: Sequence[dict]) -> list[dict[str, Any]]:
    """The validation waivers recorded in ``results`` (``result.json`` ``waived``), one per ``(code, source)``:
    ``[{code, reason, source, count}]``. The report MARKS such a row (``†`` and a footnote); it never greys, excludes
    or flags it — a waiver with a reason is a legitimate, recorded decision, like a ``custom`` rule."""
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        for w in r.get("waived") or []:
            if isinstance(w, dict) and w.get("code"):
                key = (str(w["code"]), str(w.get("source") or "describe"))
                by.setdefault(key, {"code": key[0], "reason": str(w.get("reason") or ""), "source": key[1], "count": int(w.get("count") or 0)})
    return [by[k] for k in sorted(by)]


def _row_from(results: Sequence[dict], *, column_id: str, key: tuple[str, str], metric_names: Sequence[str],
              control_names: Sequence[str], headline: str | None, seed: int | None, paths: Sequence[str],
              floor_names: Sequence[str] = (), min_eval_coverage: float = DEFAULT_MIN_EVAL_COVERAGE) -> dict[str, Any]:
    """One view-2 row from the shown results of one row key. Flags come from yavih-side facts only (status,
    seeds, floors, controls, coverage): nothing the bridge declared — a ``custom`` rule, a null weights
    fingerprint, a dtype — greys or flags a row."""
    first = results[0]
    statuses = [r.get("status") for r in results]
    ok = [r for r in results if r.get("status") == "ok"]
    status = "ok" if ok else (statuses[0] if statuses else "failed")
    bridge = first.get("bridge") or {}
    cov = first.get("coverage") or {}
    metrics = {m: _aggregate_metric(m, ok, floor_name_for(m, metric_names, floor_names)) for m in metric_names} if ok else {}
    controls: dict[str, Any] = {}
    for c in control_names:
        vals = [(r.get("controls") or {}).get(c) for r in ok]
        vals = [v for v in vals if v is not None]
        if not vals:
            controls[c] = None
        elif all(isinstance(v, bool) for v in vals):
            controls[c] = any(vals)
        elif all(isinstance(v, (int, float)) for v in vals):
            controls[c] = _mean_std(vals)[0]
        else:
            controls[c] = vals[0]
    flags: list[str] = []
    partial = is_partial(first)
    if partial:
        flags.append("partial")
    if status != "ok":
        flags.append(status)
    n_seeds = sum(int((r.get("registry") or {}).get("n_seeds") or 1) for r in ok)
    if n_seeds == 1 and seed is None:
        flags.append("single_seed")
    if any(r.get("registry") for r in results):
        flags.append("registry")
    for c in _VOIDING_CONTROLS:
        if _control_fired(ok, c):
            flags.append(c)
    if headline and ok and _below_floor(metrics.get(headline, {})):
        flags.append("below_floor")
    eval_frac = _coverage_frac(cov.get("eval")) if isinstance(cov, dict) else None
    train_frac = _coverage_frac(cov.get("train")) if isinstance(cov, dict) else None
    # A result scored on a subset of the eval population (``--allow-low-coverage``, W_COVERAGE)
    # is annotated even when it is the only row of its column (population_differs needs two rows).
    if ok and ((eval_frac is not None and eval_frac < min_eval_coverage) or _has_warning(ok, "W_COVERAGE")):
        flags.append("low_coverage")
    ids = (cov.get("eval") or {}).get("ids_sha256") if isinstance(cov, dict) else None
    greyed = any(f in flags for f in ("below_floor", *_VOIDING_CONTROLS))
    na = first.get("not_applicable") if status == "not_applicable" else None
    feats = first.get("features") if isinstance(first.get("features"), dict) else {}
    return {
        "column": column_id,
        "benchmark": key[0],
        "eval_hash": key[1],
        "protocol_hash": (first.get("benchmark") or {}).get("protocol_hash"),
        "probe_hash": (first.get("probe") or {}).get("hash"),
        "model": _model(first),
        "reading": _reading(first),
        "bridge_name": str(bridge.get("name", "")),
        "describe_sha256": bridge.get("describe_sha256"),
        "features_fingerprint": feats.get("fingerprint"),
        "standard": bool(first.get("standard")),
        "status": status,
        "reason": first.get("reason") if status != "ok" else None,
        "not_applicable": na,
        "seed": seed,
        "seeds": sorted({int((r.get("probe") or {}).get("seed", 0)) for r in ok}),
        "n_seeds": n_seeds,
        "registry": next((r["registry"].get("id") for r in results if r.get("registry")), None),
        "headline": headline,
        "metrics": metrics,
        "controls": controls,
        "coverage": {"train": train_frac, "eval": eval_frac, "eval_ids_sha256": ids},
        "features_dtype": feats.get("dtype"),
        "rules_checked": feats.get("rules_checked"),
        "waived": waived_of(results),
        "n_configs": 1,
        "n_feature_sets": 1,
        "flags": flags,
        "greyed": greyed,
        "missing": False,
        "partial": partial,
        "sample": first.get("sample") if partial else None,
        "paths": list(paths),
    }


def _missing_row(column: dict[str, Any], model: str, reading: str) -> dict[str, Any]:
    return {
        "column": column["id"],
        "benchmark": column["benchmark"],
        "eval_hash": column["eval_hash"],
        "protocol_hash": column["protocol_hash"],
        "probe_hash": column["probe_hash"],
        "model": model,
        "reading": reading,
        "bridge_name": "",
        "describe_sha256": None,
        "features_fingerprint": None,
        "standard": column["standard"],
        "status": "missing",
        "reason": None,
        "not_applicable": None,
        "seed": None,
        "seeds": [],
        "n_seeds": 0,
        "headline": column.get("headline"),
        "metrics": {},
        "controls": {},
        "coverage": {"train": None, "eval": None, "eval_ids_sha256": None},
        "features_dtype": None,
        "rules_checked": None,
        "waived": [],
        "n_configs": 0,
        "n_feature_sets": 0,
        "flags": ["missing"],
        "greyed": False,
        "missing": True,
        "partial": False,
        "sample": None,
        "paths": [],
    }


def _rankable(row: dict[str, Any]) -> bool:
    """A row whose values take part in a tie: ok, the full population at coverage, not greyed."""
    flags = row.get("flags") or []
    return (row.get("status") == "ok" and not row.get("missing") and not row.get("partial") and not row.get("greyed")
            and "low_coverage" not in flags)


def _mark_ties(rows: Sequence[dict[str, Any]], results: Sequence[dict], metric_names: Sequence[str],
               meta: Mapping[str, Any]) -> None:
    """Ties within one standard column ("Ties" in https://github.com/Bizilizi/yavih/blob/main/docs/results.md). Per metric with a direction (``higher_is_better``
    from the metric registry, or recorded by a registry entry; a disclosure has none), over the rankable rows
    (:func:`_rankable`) with a scalar value: the cells within the metric's tolerance of the best value are a tie when
    there are at least two of them; each gets ``tie: True`` and every other rankable cell ``tie: False`` (no key on
    other cells). Tolerance: the packaged folder's ``metrics.tolerance``, else the largest one a
    registry entry of the column recorded (``conformance.tolerance``), else :data:`DEFAULT_TOLERANCE`."""
    from yavih.report.store import metric_info

    ranked = [row for row in rows if _rankable(row)]
    if len(ranked) < 2:
        return
    info = metric_info(metric_names)
    folder_tol = dict(meta.get("tolerance") or {})
    for m in metric_names:
        hib = info.get(m, {}).get("higher_is_better")
        recorded = [((r.get("registry") or {}).get("higher_is_better") or {}).get(m) for r in results]
        recorded = [v for v in recorded if isinstance(v, bool)]
        if recorded:
            hib = recorded[0]
        if hib is None:
            continue
        cells = [(row["metrics"].get(m) or {}) for row in ranked]
        cells = [c for c in cells if isinstance(c.get("value"), (int, float)) and not isinstance(c.get("value"), bool)
                 and math.isfinite(float(c["value"]))]
        if len(cells) < 2:
            continue
        if m in folder_tol:
            tol = folder_tol[m]
        else:
            entry_tols = [((r.get("registry") or {}).get("tolerance") or {}).get(m) for r in results]
            entry_tols = [float(v) for v in entry_tols if isinstance(v, (int, float)) and not isinstance(v, bool)]
            tol = max(entry_tols) if entry_tols else DEFAULT_TOLERANCE
        best = (max if hib else min)(float(c["value"]) for c in cells)
        tied = [abs(float(c["value"]) - best) <= tol + 1e-12 for c in cells]
        several = sum(tied) >= 2
        for c, is_tied in zip(cells, tied, strict=True):
            c["tie"] = bool(several and is_tied)


def _sort_key_rows(row: dict[str, Any]) -> tuple:
    h = row.get("headline")
    cell = row["metrics"].get(h) if h else None
    value = cell.get("value") if isinstance(cell, dict) else None
    v = -float(value) if isinstance(value, (int, float)) else math.inf
    if isinstance(cell, dict) and cell.get("tie"):
        v = -math.inf  # tied with the best: never ordered by value (the seed decides among them)
    return (row["model"], row["reading"], bool(row.get("partial")), row["missing"], v, row["seed"] if row["seed"] is not None else -1)


def _headline_value(results: Sequence[dict], headline: str | None) -> Any:
    ok = [r for r in results if r.get("status") == "ok"]
    if not headline or not ok:
        return None
    return _aggregate_metric(headline, ok).get("value")


def _row_detail(sel: Selection, headline: str | None) -> dict[str, Any]:
    """``--detail`` for one row: the shown configuration's ``describe()`` (input rules, custom-rule
    descriptions and batch tolerance included) and ``describe_sha256``, the older configurations with their
    ``describe()`` diff against the shown one, seeds and headline values, the older feature sets, and the
    recorded warnings of the shown results (``W_MEDIA_FP``, ``W_WEIGHTS_UNFINGERPRINTED``, …)."""
    shown_describe = _describe(sel.shown[0]) if sel.shown else {}
    first = sel.shown[0] if sel.shown else {}
    return {
        "describe_sha256": sel.describe_sha256 or None,
        "describe": shown_describe,
        "rules_checked": (first.get("features") or {}).get("rules_checked") if isinstance(first.get("features"), dict) else None,
        "features_fingerprint": sel.fingerprint or None,
        "older_configs": [
            {"describe_sha256": dsha or None, "diff": describe_diff(shown_describe, _describe(rs[0])),
             "seeds": sorted({int((r.get("probe") or {}).get("seed", 0) or 0) for r in rs}),
             "headline": _headline_value(_dedup_seeds(rs), headline), "created": _newest(rs),
             "moved_aside": dsha in sel.moved_aside}
            for dsha, rs in sel.configs[1:]
        ],
        "older_feature_sets": [
            {"fingerprint": fp or None, "seeds": [int((r.get("probe") or {}).get("seed", 0) or 0) for r in rs],
             "headline": _headline_value(rs, headline), "created": _newest(rs)}
            for fp, rs in sel.feature_sets[1:]
        ],
        "warnings": sorted({str(w.get("code")) for r in sel.shown for w in (r.get("warnings") or []) if isinstance(w, dict) and w.get("code")}),
    }


# --------------------------------------------------------------------------- view 1: lm-eval tables


def lm_eval_tables(records: Sequence[dict], aliases: Mapping[str, str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The per-model view: per ``model``, one row per ``(benchmark, reading)`` and ``metrics.list`` metric that
    has a value (headline first, disclosures such as ``n_eval`` last). ``eval_hash`` = the 12-hex comparability
    key (two numbers of one benchmark compare iff it is equal); ``value`` = mean over seeds (``std`` and
    ``n_seeds`` when several), ``ci95`` = the interval of the lowest seed (as in the results store) or
    ``None``, ``floor`` = the value of the floor whose ``floor_of`` names the metric, ``in_table`` = whether the
    metric is in the benchmark's ``metrics.table`` (all are when it declares none). Each row key shows
    its newest configuration and feature set (:func:`select_newest`). A ``not_applicable`` benchmark prints one
    row with ``metric = —`` and the reason; a non-standard probe is labelled ``(nonstandard)``."""
    from yavih.report.store import metric_info

    aliases = _alias_map() if aliases is None else aliases
    by_model: dict[str, dict[tuple[str, str, str, bool, bool], list[dict]]] = {}
    for rec in records:
        if rec.get(MOVED_ASIDE):
            continue  # never a row
        res = rec["result"]
        bench, eh, model, reading = row_key(res, aliases)
        by_model.setdefault(model, {}).setdefault((bench, reading, eh, bool(res.get("standard")), is_partial(res)), []).append(res)
    out: dict[str, list[dict[str, Any]]] = {}
    for model in sorted(by_model):
        groups = by_model[model]
        rows: list[dict[str, Any]] = []

        def order(kv):
            (bench, reading, eh, std, partial), rs = kv
            pv = (rs[0].get("benchmark") or {}).get("protocol_version")
            return (bench, reading, partial, not std, -(pv if isinstance(pv, int) else -1), eh)

        for (bench, reading, eh, std, partial), all_results in sorted(groups.items(), key=order):
            results = select_newest(all_results).shown
            first = results[0]
            bmeta = first.get("benchmark") or {}
            base = {"benchmark": bench + ("" if std else " (nonstandard)"), "reading": reading, "version": bmeta.get("protocol_version"),
                    "eval_hash": eh[:12], "readout": bmeta.get("readout")}
            waived = waived_of(results)
            if waived:  # marked († and a footnote), never greyed
                base["waived"] = waived
            if partial:  # a capped run: its own row, marked in every format
                base["partial"] = True
            ok = [r for r in results if r.get("status") == "ok"]
            if not ok:
                rows.append({**base, "metric": "—", "value": None, "std": None, "n_seeds": len(results), "ci95": None,
                             "floor": None, "status": str(first.get("status")), "reason": first.get("reason")})
                continue
            metric_names = _ordered_union([list((r.get("metrics") or {}).keys()) for r in ok])
            floor_names = _ordered_union([list((r.get("floors") or {}).keys()) for r in ok])
            info = metric_info(metric_names, floors=floor_names)
            meta = _bench_meta(bench)
            headline = _pick_headline(metric_names, floor_names, meta)
            table = (set(meta.get("table") or ()) & set(metric_names)) or set(metric_names)
            disclosures = [m for m in metric_names if info[m]["kind"] == "disclosure"]
            names = ([headline] if headline in metric_names else []) + [m for m in metric_names if m != headline and m not in disclosures] + [
                m for m in disclosures if m != headline]
            for m in names:
                values = [(r.get("metrics") or {}).get(m) for r in ok]
                if all(v is None for v in values):
                    continue
                cell = _aggregate_metric(m, ok, info[m].get("floor"))
                ci = (ok[0].get("ci95") or {}).get(m)
                rows.append({**base, "metric": m, "value": cell["value"], "std": cell["std"], "n_seeds": len(ok),
                             "ci95": list(ci) if isinstance(ci, list) else None, "floor": cell["floor"],
                             "status": "ok", "reason": None, "headline": m == headline, "in_table": m in table})
        out[model] = rows
    return out


# --------------------------------------------------------------------------- the build


def build_report(records: Sequence[dict], *, fmt: str = "md", seeds: str = "aggregate",
                 allow_coverage_diff: bool = False, detail: bool = False) -> Report:
    """Group index records (``{"path", "result"}``) into columns and rows without
    touching the filesystem; :func:`report` wraps it."""
    if seeds not in ("aggregate", "list"):
        raise ValueError(f"seeds must be 'aggregate' or 'list', got {seeds!r}")
    if fmt not in ("md", "csv", "json"):
        raise ValueError(f"fmt must be md, csv or json, got {fmt!r}")
    aliases = _alias_map()
    # moved-aside outputs (probe/_stale/) are never rows: only older configurations of a live row key
    moved: dict[tuple[tuple[str, str], bool, str, str], list[dict]] = {}
    for rec in records:
        if rec.get(MOVED_ASIDE):
            res = rec["result"]
            moved.setdefault((column_key(res, aliases), bool(res.get("standard")), _model(res), _reading(res)), []).append(res)
    records = [rec for rec in records if not rec.get(MOVED_ASIDE)]
    rep = Report(fmt=fmt, n_results=len(records), show_all=bool(detail))
    rep.lm_eval = lm_eval_tables(records, aliases)

    # --- columns: (benchmark, eval_hash), standard and non-standard apart
    groups: dict[tuple[tuple[str, str], bool], list[dict]] = {}
    paths_of: dict[int, str] = {}
    for rec in records:
        res = rec["result"]
        paths_of[id(res)] = str(rec.get("path", ""))
        groups.setdefault((column_key(res, aliases), bool(res.get("standard"))), []).append(res)

    # --- superseded protocol versions: per benchmark, every version below the newest seen
    # (or below the packaged yaml's current version when loadable) is superseded.
    versions: dict[str, dict[int, set[str]]] = {}
    for (key, _std), results in groups.items():
        for r in results:
            pv = (r.get("benchmark") or {}).get("protocol_version")
            if isinstance(pv, int):
                versions.setdefault(key[0], {}).setdefault(pv, set()).add(key[1])
    bench_meta = {name: _bench_meta(name) for name in versions}

    def meta_of(name: str) -> dict[str, Any]:
        if name not in bench_meta:
            bench_meta[name] = _bench_meta(name)
        return bench_meta[name]
    latest: dict[str, int] = {}
    for name, by_v in versions.items():
        latest[name] = max([*by_v, bench_meta[name].get("protocol_version", -1)])
    for name, by_v in sorted(versions.items()):
        for pv in sorted(by_v):
            if pv < latest[name]:
                rep.superseded.append({
                    "benchmark": name, "protocol_version": pv, "latest": latest[name],
                    "eval_hashes": sorted(by_v[pv]),
                    "n_results": sum(1 for (k, _s), rs in groups.items() if k[0] == name and k[1] in by_v[pv] for _ in rs),
                })

    # --- expected readings per model across the whole report (so an AV model that reported only two of three
    # readings in one column gets a MISSING cell there). A reading is expected only in columns whose benchmark
    # has every modality it read (an audio reading is never MISSING on k400, a visual one never on esc50). Its
    # modality set is the union over all its results; unknown (unplaceable stream, unpackaged benchmark) keeps
    # the reading expected everywhere.
    readings_by_model: dict[str, list[str]] = {}
    reading_mods: dict[tuple[str, str], frozenset[str] | None] = {}
    for (key, _std), results in groups.items():
        kinds = meta_of(key[0]).get("stream_kinds")
        for r in results:
            mdl, rd = _model(r), _reading(r)
            if mdl and rd and rd not in readings_by_model.setdefault(mdl, []):
                readings_by_model[mdl].append(rd)
            if mdl and rd:
                mods = _stream_modalities(r, kinds)
                prev = reading_mods.get((mdl, rd), frozenset())
                reading_mods[(mdl, rd)] = None if (mods is None or prev is None) else prev | mods

    def expected_readings(mdl: str, bench: str) -> list[str]:
        kinds = meta_of(bench).get("stream_kinds")
        if not kinds or any(k not in _MODALITY for k in kinds.values()):
            return list(readings_by_model.get(mdl, []))
        have = {_MODALITY[k] for k in kinds.values()}
        return [rd for rd in readings_by_model.get(mdl, []) if not reading_mods.get((mdl, rd)) or reading_mods[(mdl, rd)] <= have]

    def col_sort(item):
        (key, std), results = item
        pv = max(((r.get("benchmark") or {}).get("protocol_version") or -1 for r in results), default=-1)
        return (key[0], not std, -pv, key[1])

    unfingerprinted: list[str] = []
    for (key, std), results in sorted(groups.items(), key=col_sort):
        bench, eh = key
        column_id = f"{bench}:{eh[:12]}" + ("" if std else ":nonstandard")
        # rows by (model, reading): newest configuration, newest feature set
        # a capped run's results (`sample`) form their own row: never selected against, or averaged with, a full run's
        by_row: dict[tuple[str, str, bool], list[dict]] = {}
        for r in results:
            by_row.setdefault((_model(r), _reading(r), is_partial(r)), []).append(r)
        selections = {rk: add_moved_aside(select_newest(rs), [] if rk[2] else moved.get((key, std, *rk[:2]), []))
                      for rk, rs in by_row.items()}
        shown_all = [r for sel in selections.values() for r in sel.shown]
        ok_results = [r for r in shown_all if r.get("status") == "ok"]
        metric_names = _ordered_union([list((r.get("metrics") or {}).keys()) for r in ok_results])
        floor_names = _ordered_union([list((r.get("floors") or {}).keys()) for r in ok_results])
        control_names = _ordered_union([list((r.get("controls") or {}).keys()) for r in ok_results])
        meta = bench_meta.get(bench, {})
        headline = _pick_headline(metric_names, floor_names, meta)
        pvs = sorted({(r.get("benchmark") or {}).get("protocol_version") for r in results if isinstance((r.get("benchmark") or {}).get("protocol_version"), int)})
        readouts = sorted({str((r.get("benchmark") or {}).get("readout", "")) for r in results})
        first = results[0]
        column: dict[str, Any] = {
            "id": column_id,
            "benchmark": bench,
            "eval_hash": eh,
            "protocol_hash": (first.get("benchmark") or {}).get("protocol_hash"),
            "probe_hash": (first.get("probe") or {}).get("hash"),
            "protocol_version": pvs[0] if len(pvs) == 1 else (pvs or None),
            "readout": readouts[0] if len(readouts) == 1 else readouts,
            "standard": std,
            "comparable": std,
            "metrics": metric_names,
            "floors": floor_names,
            "controls": control_names,
            "headline": headline,
            # the default markdown columns: metrics.table (in metrics order), else (or when none is present) every metric
            "table": [m for m in metric_names if m in set(meta.get("table") or ())] or list(metric_names),
            "population_differs": False,
            "superseded": any(pv < latest.get(bench, pv) for pv in pvs),
            "n_results": len(results),
            "n_rows": 0,
            "models": {},
            "flags": [],
        }
        if not std:
            column["flags"].append("nonstandard")
            rep.warnings.append({
                "code": "W_NONSTANDARD", "column": column_id,
                "message": "non-pinned ProbeConfig (standard=false): rows in this column are not comparable with anything",
            })
        if column["superseded"]:
            column["flags"].append("superseded")
        # a record without folder_source (a schema-3 result, a registry entry exported without it) is unknown, not
        # different: only the known sources are compared
        sources = sorted({json.dumps(fs, sort_keys=True) for r in results
                          if isinstance(fs := (r.get("benchmark") or {}).get("folder_source"), dict) and fs})
        if len(sources) > 1:
            column["flags"].append("folder_source_differs")
            rep.warnings.append({
                "code": "W_FOLDER_SOURCE_DIFFERS", "column": column_id,
                "message": f"{bench}: results under one protocol hash were scored by {len(sources)} different versions of the "
                           "folder's probe.py / reader.py (benchmark.folder_source); comparable by declaration (the "
                           "conformance fixture holds them within tolerance), shown for diagnosis",
            })

        rows: list[dict[str, Any]] = []
        min_cov = float(meta.get("min_eval_coverage", DEFAULT_MIN_EVAL_COVERAGE))
        row_kw = dict(column_id=column_id, key=key, metric_names=metric_names, control_names=control_names, headline=headline,
                      floor_names=floor_names, min_eval_coverage=min_cov)
        for (mdl, rd, partial), sel in selections.items():
            if len(sel.configs) > 1:
                shown_d = _describe(sel.configs[0][1][0])
                keys = _ordered_union(describe_diff(shown_d, _describe(rs[0])).keys() for _d, rs in sel.configs[1:])
                aside_note = (f"; {', '.join(sorted(d[:8] or '?' for d in sel.moved_aside))} only in outputs moved to probe/_stale/ "
                              "by an in-place reconfiguration") if sel.moved_aside else ""
                rep.warnings.append({
                    "code": "W_BRIDGE_CHANGED", "column": column_id,
                    "message": f"{mdl} {rd} on {bench}: results from {len(sel.configs)} bridge configurations (describe "
                               f"{', '.join(d[:8] or '?' for d, _ in sel.configs)}; differing keys {keys}{aside_note}); the newest "
                               "is shown, the others under --detail",
                })
            if len(sel.feature_sets) > 1:
                rep.warnings.append({
                    "code": "W_FEATURES_DIFFER", "column": column_id,
                    "message": f"{mdl} {rd} on {bench}: {len(sel.feature_sets)} feature sets under one configuration (fingerprints "
                               f"{', '.join(fp[:8] or '?' for fp, _ in sel.feature_sets)}: a re-extraction, another site, differently "
                               "staged media); the newest is shown, never averaged, the others under --detail",
                })
            if any(_unfingerprinted(r) for r in sel.shown):
                unfingerprinted.append(f"{mdl} {rd} on {bench}")
            if seeds == "aggregate":
                part_rows = [_row_from(sel.shown, seed=None, paths=[paths_of.get(id(r), "") for r in sel.shown], **row_kw)]
            else:
                part_rows = [_row_from([r], seed=int((r.get("probe") or {}).get("seed", 0)), paths=[paths_of.get(id(r), "")], **row_kw)
                             for r in sel.shown]
            for row in part_rows:
                row["n_configs"], row["n_feature_sets"] = len(sel.configs), len(sel.feature_sets)
            rows.extend(part_rows)
            if detail:
                rep.detail.setdefault(column_id, {})[f"{mdl}:{rd}" + (":partial" if partial else "")] = _row_detail(sel, headline)

        # population differs: distinct eval ids_sha256 among ok rows (a partial row is a subset by construction: it is
        # marked partial, never compared here)
        full_ok = [row for row in rows if row["status"] == "ok" and not row.get("partial")]
        ids = {row["coverage"]["eval_ids_sha256"] for row in full_ok if row["coverage"]["eval_ids_sha256"]}
        if len(ids) > 1:
            column["population_differs"] = True
            column["flags"].append("population_differs")
            for row in full_ok:
                row["flags"].append("population_differs")
            low = [row for row in full_ok if row["coverage"]["eval"] is not None and row["coverage"]["eval"] < min_cov]
            if low and not allow_coverage_diff:
                names = ", ".join(sorted({f"{row['model']} {row['reading']}" for row in low}))
                raise ReportError(
                    "E_POPULATION_DIFFERS",
                    f"column {column_id}: eval populations differ and {names} fall below min_eval_coverage={min_cov}; "
                    f"pass --allow-coverage-diff to compare anyway",
                )
            rep.warnings.append({
                "code": "W_POPULATION_DIFFERS", "column": column_id,
                "message": f"eval ids_sha256 differ between rows ({len(ids)} populations); numbers are not on one population",
            })

        if std:
            _mark_ties(rows, [r for sel in selections.values() for r in sel.shown], metric_names, meta)
            if any(cell.get("tie") for row in rows for cell in (row.get("metrics") or {}).values() if isinstance(cell, dict)):
                column["flags"].append("ties")

        # model / reading grouping with MISSING cells
        present: dict[str, set[str]] = {}
        for row in rows:
            if row["model"]:
                present.setdefault(row["model"], set()).add(row["reading"])
        for mdl in readings_by_model:
            if mdl not in present:
                continue
            for rd in expected_readings(mdl, bench):
                if rd not in present[mdl]:
                    rows.append(_missing_row(column, mdl, rd))
                    rep.warnings.append({
                        "code": "W_MISSING_READING", "column": column_id,
                        "message": f"model {mdl!r} has no {rd!r} reading in this column",
                    })
        column["models"] = {mdl: sorted(present[mdl] | set(expected_readings(mdl, bench))) for mdl in sorted(present)}
        rows.sort(key=_sort_key_rows)
        column["n_rows"] = len(rows)
        rep.columns.append(column)
        rep.rows.extend(rows)

    if unfingerprinted:
        rep.warnings.append({
            "code": "W_WEIGHTS_UNFINGERPRINTED", "column": None,
            "message": f"describe()['weights_sha256'] is null for {', '.join(sorted(set(unfingerprinted)))}: recorded, nothing else "
                       "(the model name is the user's)",
        })
    rep.text = tables.render(rep, fmt)
    return rep


def _write_report_outputs(rep: Report, runs_root: Path, globs: Sequence[str]) -> None:
    """``runs/_report/report.<fmt>`` + ``DONE`` (written when the tables are written)."""
    out_dir = runs_root / REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"report.{rep.fmt}"
    tmp = table_path.with_name(table_path.name + f".tmp.{os.getpid()}")
    tmp.write_text(rep.text, encoding="utf-8")
    os.replace(tmp, table_path)
    marker = {
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "yavih_version": __version__,
        "fmt": rep.fmt,
        "n_results": rep.n_results,
        "n_columns": len(rep.columns),
        "n_rows": len(rep.rows),
        "globs": [str(g) for g in globs],
        "table": str(table_path),
    }
    done = out_dir / "DONE"
    tmp = done.with_name(f"DONE.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(marker, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(tmp, done)
    rep.paths["report"] = str(table_path)
    rep.paths["done"] = str(done)


def report(
    globs: Sequence[str],
    *,
    fmt: str = "md",
    seeds: str = "aggregate",
    allow_coverage_diff: bool = False,
    join: dict | None = None,
    detail: bool = False,
    results_root: str | Path | None = None,
    merge_results: bool = False,
    runs_root: str | Path | None = None,
    write: bool = True,
    suite: Any = None,
    registry: str | Path | None = None,
) -> Report:
    """Index → the per-model lm-eval tables + the comparability columns → rendered text;
    writes ``runs/_index.jsonl`` and, when the runs root is known, ``runs/_report/report.<fmt>`` +
    ``DONE``. ``join={"<benchmark>": {"<reading>": run, ...}}`` calls that benchmark folder's
    ``probe.py`` ``join()`` (:func:`yavih.report.join.run_join`; vggsounder: ``a``/``v``/``av``) and writes
    the joined numbers to the model's results directory under ``<benchmark>:a+v+av``. ``merge_results`` merges every model's
    ``results_*.json`` under ``results_root`` (default ``yavih.config.results_root`` of the runs root)
    into one ``results_<timestamp>.json`` per model. ``suite`` (a name, a path or a ``yavih.suites.Suite``) adds the
    completeness appendix ``rep.suite``: per model of the report and suite ``(benchmark, reading)`` ``done`` /
    ``missing`` / ``pending``, and per join entry ``done`` / ``join-missing`` (this invocation's joins and the results
    store under ``results_root``) (:func:`yavih.suites.completeness`)."""
    if isinstance(globs, (str, os.PathLike)):
        globs = [str(globs)]
    the_suite = None
    if suite is not None:
        from yavih.suites import load_suite

        the_suite = load_suite(suite)  # an unknown or invalid suite fails before anything is indexed
    records = build_index(globs, runs_root=runs_root)
    if registry is not None:
        # registry entries join the comparison as rows under the same rule (same benchmark, same eval_hash, both standard);
        # their seeds and provenance are theirs, their model name is namespaced <org>/<model>
        from yavih.registry import records as registry_records

        records = [*records, *registry_records(registry)]
    rep = build_report(records, fmt=fmt, seeds=seeds, allow_coverage_diff=allow_coverage_diff, detail=detail)
    records = [r for r in records if not r.get(MOVED_ASIDE)]
    root = Path(runs_root) if runs_root is not None else runs_root_of([Path(r["path"]) for r in records])
    if results_root is None and (join or merge_results or the_suite is not None):
        from yavih.config import results_root as _results_root

        results_root = _results_root(root)
    if join:
        from yavih.report.join import JoinError, run_join

        for kind, spec in join.items():
            try:
                rep.joins[kind] = run_join(str(kind), dict(spec), results_root=results_root, write=write)
            except JoinError as e:
                raise ReportError(e.code, str(e).split(": ", 1)[-1], exit_code=e.exit_code) from e
        rep.text = tables.render(rep, fmt)
    if merge_results:
        from yavih.report import store

        models = sorted({_model(r["result"]) for r in records} - {""}) or None
        rep.merged = [str(p) for p in store.merge_results(results_root, models)]
        if rep.merged:
            rep.paths["merged"] = ";".join(rep.merged)
    if the_suite is not None:
        from yavih.suites import completeness, completeness_json

        joined = [(j.get("model"), j.get("benchmark"), j.get("reading")) for j in rep.joins.values() if isinstance(j, dict)]
        rows = completeness(the_suite, records, results_root=results_root, runs_root=root, joined=joined)
        rep.suite = completeness_json(the_suite, rows, results_root=results_root)
        rep.text = tables.render(rep, fmt)
    if write:
        if root is not None:
            try:
                _write_report_outputs(rep, root, globs)
            except OSError as exc:
                print(f"[yavih.report] could not write {root / REPORT_DIR}: {exc}", file=sys.stderr)
    return rep
