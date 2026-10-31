"""Metric registry: ``MetricSpec`` and the complete ``REGISTRY`` table.

Importable without scikit-learn or torch: every ``fn`` is a ``LazyFn`` holding a
``"yavih.metrics.<module>:<function>"`` reference that is resolved on first call.
``Benchmark.load`` resolves ``metrics.list/floors/controls``
names here and only needs names, ``readouts`` and ``protocol_keys``.

Per-set metrics (``auc_pr_<set>``, ``auc_roc_<set>``, ``auc_pr_<set>_floor``, the
clf ``multi_sets`` metrics) are registered as ``PATTERNS`` and expanded at
``Benchmark.load`` with ``expand_sets(set_names)``: the expansion returns concrete
``MetricSpec`` objects whose ``fn`` is the template function with ``set_name=<set>``
bound as a keyword argument.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MetricSpec",
    "MetricPattern",
    "LazyFn",
    "REGISTRY",
    "PATTERNS",
    "CI_KINDS",
    "CI_ADMISSIBLE",
    "KINDS",
    "SHAPES",
    "LOWER_IS_BETTER",
    "FLOOR_OF",
    "UnknownMetric",
    "MetricNotAllowed",
    "register",
    "expand_pattern",
    "expand_sets",
    "match_pattern",
    "resolve",
    "resolve_names",
    "protocol_keys_for",
    "allows",
    "ANY_READOUT",
]

KINDS: frozenset[str] = frozenset({"metric", "floor", "control", "disclosure"})
SHAPES: frozenset[str] = frozenset({"scalar", "list"})
#: Allowed values of ``benchmark.yaml[metrics.ci][name]`` (intervals are not metrics).
CI_KINDS: frozenset[str] = frozenset({"binomial", "bootstrap_clip", "pooled_binomial"})
_CI_BINARY = frozenset({"acc", "acc@0.5", "acc@0.3", "center_in_gt"})
#: Which metric names each interval kind can be computed for: ``binomial`` needs a
#: per-row 0/1 quantity; ``bootstrap_clip`` a per-clip quantity or a statistic recomputed per resample
#: (``map``); ``pooled_binomial`` pools the fold accuracies. ``Benchmark.load`` refuses any other
#: ``metrics.ci`` pairing;
#: ``tests/test_metrics_intervals.py`` pins this table equal to ``yavih.metrics.intervals.admissible``. A folder
#: metric (``eval.py`` ``METRICS``) adds its own: ``MetricSpec.ci`` names the kinds its ``per_clip`` supports.
CI_ADMISSIBLE: dict[str, frozenset[str]] = {
    "binomial": _CI_BINARY,
    "bootstrap_clip": _CI_BINARY | frozenset({
        "seg_acc", "seg_acc_boundary", "clip_acc", "iou_mean", "d1", "d2", "d3", "absrel", "rmse", "map",
    }),
    "pooled_binomial": _CI_BINARY | frozenset({"acc_mean", "acc_pstd", "acc_per_fold"}),
}

#: ``MetricSpec.readouts`` wildcard: the metric serves every readout, including one added later (``n_eval``).
ANY_READOUT = "*"
_READOUT_NAME = re.compile(r"^(\*|[a-z][a-z0-9_]*)$")


class UnknownMetric(KeyError):
    """A ``metrics.*`` name is not in ``REGISTRY`` (and matches no pattern)."""


class MetricNotAllowed(ValueError):
    """A registered name is not allowed for this readout (``MetricSpec.readouts``)."""


class LazyFn:
    """A callable that imports ``"pkg.module:function"`` on first call (the
    registry is importable without sklearn). Optional ``bound`` keyword arguments are
    passed on every call (used by set-pattern expansions: ``set_name=…``)."""

    __slots__ = ("ref", "bound", "_fn")

    def __init__(self, ref: str, **bound: Any) -> None:
        if ":" not in ref:
            raise ValueError(f"LazyFn ref must be 'module:function', got {ref!r}")
        self.ref = ref
        self.bound = dict(bound)
        self._fn: Callable[..., Any] | None = None

    def resolve(self) -> Callable[..., Any]:
        if self._fn is None:
            module_name, func_name = self.ref.split(":", 1)
            module = importlib.import_module(module_name)
            fn = getattr(module, func_name, None)
            if fn is None or not callable(fn):
                raise AttributeError(f"metric function {self.ref!r} is not defined")
            self._fn = fn
        return self._fn

    def __call__(self, outputs: Any, targets: Any, protocol: Any, split_meta: Any) -> Any:
        return self.resolve()(outputs, targets, protocol, split_meta, **self.bound)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        b = f", {self.bound}" if self.bound else ""
        return f"LazyFn({self.ref!r}{b})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LazyFn) and (self.ref, self.bound) == (other.ref, other.bound)

    def __hash__(self) -> int:
        return hash((self.ref, tuple(sorted(self.bound.items()))))


@dataclass(frozen=True)
class MetricSpec:
    """One registered metric. ``fn(outputs, targets, protocol, split_meta) -> float | list[float]``."""

    name: str
    fn: Callable[..., Any]
    readouts: frozenset[str]
    protocol_keys: frozenset[str] = field(default_factory=frozenset)
    shape: str = "scalar"
    kind: str = "metric"
    #: The results store's ``higher_is_better``; ``None`` for disclosures and controls.
    higher_is_better: bool | None = True
    #: floors only: the metrics this floor is the floor of (``acc_floor`` → ``(acc, acc_mean)``).
    floor_of: tuple[str, ...] = ()
    #: the ``external[].name`` whose code computes it (folder metrics only); ``None`` = yavih's own.
    external: str | None = None
    #: folder metrics: ``per_clip(outputs, targets, protocol, split_meta) -> [n]`` values (NaN = unscored) on the
    #: population the metric scores, for intervals; registry metrics use ``yavih.metrics.intervals.PER_CLIP``.
    per_clip: Callable[..., Any] | None = field(default=None, compare=False)
    #: folder metrics: the ``metrics.ci`` kinds this metric admits (``binomial`` / ``pooled_binomial`` need a per-row
    #: 0/1 ``per_clip``, ``bootstrap_clip`` any ``per_clip``); registry metrics use ``CI_ADMISSIBLE``.
    ci: frozenset[str] = frozenset()
    #: folder metrics whose statistic is not a mean of per-clip values (a distribution distance such as KAD):
    #: ``interval(kind, outputs, targets, protocol, split_meta) -> (lo, hi)``, used for the kinds in ``ci`` in place of
    #: the per-clip bootstrap; it must resample clips with ``yavih.metrics.intervals.bootstrap_draws`` and the
    #: ``split_meta["bootstrap"]`` settings, so every interval of a result shares one resampling scheme.
    interval: Callable[..., Any] | None = field(default=None, compare=False)
    #: folder metrics: one line naming what computes it (an external package's own metric name and definition)
    description: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"MetricSpec {self.name}: shape must be one of {sorted(SHAPES)}")
        if self.kind not in KINDS:
            raise ValueError(f"MetricSpec {self.name}: kind must be one of {sorted(KINDS)}")
        object.__setattr__(self, "readouts", frozenset(self.readouts))
        bad = sorted(r for r in self.readouts if not isinstance(r, str) or not _READOUT_NAME.match(r))
        if bad or not self.readouts:
            raise ValueError(f"MetricSpec {self.name}: readouts must be readout names (or {ANY_READOUT!r}), got {sorted(self.readouts)}")
        if self.higher_is_better is not None and not isinstance(self.higher_is_better, bool):
            raise ValueError(f"MetricSpec {self.name}: higher_is_better must be a bool or None")
        object.__setattr__(self, "floor_of", tuple(self.floor_of or ()))
        if self.floor_of and self.kind != "floor":
            raise ValueError(f"MetricSpec {self.name}: floor_of is for floors only (kind {self.kind!r})")
        if self.external is not None and (not isinstance(self.external, str) or not self.external):
            raise ValueError(f"MetricSpec {self.name}: external must be a package name or None")
        object.__setattr__(self, "ci", frozenset(self.ci or ()))
        if self.ci - CI_KINDS:
            raise ValueError(f"MetricSpec {self.name}: ci kinds {sorted(self.ci - CI_KINDS)} not in {sorted(CI_KINDS)}")
        if self.ci and self.per_clip is None and self.interval is None:
            raise ValueError(f"MetricSpec {self.name}: declaring ci {sorted(self.ci)} needs a per_clip or an interval function")
        if self.interval is not None and not callable(self.interval):
            raise ValueError(f"MetricSpec {self.name}: interval must be callable")
        if self.per_clip is not None and not callable(self.per_clip):
            raise ValueError(f"MetricSpec {self.name}: per_clip must be callable")


@dataclass(frozen=True)
class MetricPattern:
    """A family of metrics parametrized by a ``label.sets`` name, e.g. ``auc_pr_<set>``.
    ``regex`` has one named group ``set``; ``fn_ref`` is the template function which
    receives ``set_name=<set>`` as a bound keyword."""

    template: str  # e.g. "auc_pr_<set>"
    fn_ref: str
    readouts: frozenset[str]
    kind: str = "metric"
    shape: str = "scalar"
    protocol_keys: frozenset[str] = field(default_factory=frozenset)
    floor_of: tuple[str, ...] = ()  # templates, e.g. ("auc_pr_<set>",)

    @property
    def regex(self) -> re.Pattern[str]:
        pat = re.escape(self.template).replace(re.escape("<set>"), r"(?P<set>[a-z0-9_]+?)")
        return re.compile(rf"^{pat}$")

    def name_for(self, set_name: str) -> str:
        return self.template.replace("<set>", set_name)

    def expand(self, set_name: str) -> MetricSpec:
        return MetricSpec(
            name=self.name_for(set_name),
            fn=LazyFn(self.fn_ref, set_name=set_name),
            readouts=self.readouts,
            protocol_keys=self.protocol_keys,
            shape=self.shape,
            kind=self.kind,
            higher_is_better=_higher_is_better(self.kind, self.name_for(set_name)),
            floor_of=tuple(t.replace("<set>", set_name) for t in self.floor_of),
        )


REGISTRY: dict[str, MetricSpec] = {}
PATTERNS: dict[str, MetricPattern] = {}

#: Registry metrics where lower is better (a folder metric sets ``higher_is_better`` itself).
LOWER_IS_BETTER: frozenset[str] = frozenset({"absrel", "rmse"})

#: ``floor_of``: which metrics each registered floor is the floor of (the report's floor column).
FLOOR_OF: dict[str, tuple[str, ...]] = {
    "acc_floor": ("acc", "acc_mean"),
    "map_floor": ("map",),
    "seg_acc_floor": ("seg_acc",),
    "seg_acc_boundary_floor": ("seg_acc_boundary",),
    "d1_floor": ("d1",),
    "iou_floor": ("iou_mean",),
}


def _higher_is_better(kind: str, name: str) -> bool | None:
    """``None`` for disclosures and controls (no direction), ``False`` for the lower-is-better metrics."""
    if kind in ("disclosure", "control"):
        return None
    return name not in LOWER_IS_BETTER


def register(spec: MetricSpec, *, replace: bool = False) -> MetricSpec:
    """Add a spec; re-registering a name is an error unless ``replace``."""
    if spec.name in REGISTRY and not replace:
        raise ValueError(f"metric {spec.name!r} already registered")
    REGISTRY[spec.name] = spec
    return spec


def _reg(
    names: Iterable[str],
    module: str,
    readouts: Iterable[str],
    *,
    keys: Iterable[str] = (),
    kind: str = "metric",
    shape: str = "scalar",
    fn_names: Mapping[str, str] | None = None,
) -> None:
    fn_names = dict(fn_names or {})
    for name in names:
        fn_name = fn_names.get(name, name)
        register(
            MetricSpec(
                name=name,
                fn=LazyFn(f"yavih.metrics.{module}:{fn_name}"),
                readouts=frozenset(readouts),
                protocol_keys=frozenset(keys),
                shape=shape,
                kind=kind,
                higher_is_better=_higher_is_better(kind, name),
                floor_of=FLOOR_OF.get(name, ()) if kind == "floor" else (),
            )
        )


def _pattern(
    template: str, module: str, fn_name: str, readouts: Iterable[str], *, kind: str = "metric", floor_of: tuple[str, ...] = ()
) -> None:
    PATTERNS[template] = MetricPattern(
        template=template, fn_ref=f"yavih.metrics.{module}:{fn_name}", readouts=frozenset(readouts), kind=kind, floor_of=floor_of
    )


# ----------------------------------------------------------------------------- the metric table
# clf: exact-argmax accuracy and disclosures
_reg(["acc"], "classification", ["clf"])
_reg(["n_eval"], "classification", [ANY_READOUT], kind="disclosure")  # listed by every readout
_reg(["n_train"], "classification", ["clf"], kind="disclosure")
_reg(["acc_floor"], "classification", ["clf"], kind="floor")
# clf multi-label
_reg(["map"], "classification", ["clf"])
_reg(["effective_pos_classes"], "classification", ["clf"], kind="disclosure")
_reg(["map_floor"], "classification", ["clf"], kind="floor")
# clf multi_sets: one per label.sets name, expanded at Benchmark.load (expand_sets)
_pattern("auc_pr_<set>", "classification", "auc_pr", ["clf"])
_pattern("auc_roc_<set>", "classification", "auc_roc", ["clf"])
_pattern("auc_pr_<set>_floor", "classification", "auc_pr_floor", ["clf"], kind="floor", floor_of=("auc_pr_<set>",))
# clf folds
_reg(["acc_mean", "acc_pstd"], "classification", ["clf"])
_reg(["acc_per_fold"], "classification", ["clf"], shape="list")
_reg(["n_folds"], "classification", ["clf"], kind="disclosure")
# (avsync's offset metrics and floor-arm controls live in that folder's eval.py METRICS)
# seq
_SEQ_KEYS = ["num_segments", "segments_scored", "segment_offset", "background_index"]
_reg(["seg_acc", "clip_acc", "seg_acc_boundary"], "segment", ["seq"], keys=_SEQ_KEYS)
_reg(["n_boundary"], "segment", ["seq"], keys=_SEQ_KEYS, kind="disclosure")
_reg(["seg_acc_floor", "seg_acc_boundary_floor"], "segment", ["seq"], keys=_SEQ_KEYS, kind="floor")
# grid
_GRID_KEYS = ["max_depth", "target_size"]
_reg(["d1", "d2", "d3", "absrel", "rmse"], "depth", ["grid"], keys=_GRID_KEYS)
_reg(["d1_floor"], "depth", ["grid"], keys=_GRID_KEYS, kind="floor")
# box
_reg(
    ["iou_mean", "acc@0.5", "acc@0.3", "center_in_gt"],
    "box",
    ["box"],
    fn_names={"acc@0.5": "acc_at_05", "acc@0.3": "acc_at_03"},
)
_reg(["n_excluded_degenerate"], "box", ["box"], kind="disclosure")
_reg(["iou_floor"], "box", ["box"], kind="floor")
_reg(["iou_shuffled", "audio_blind"], "box", ["box"], kind="control")


# ----------------------------------------------------------------------------- resolution helpers


def match_pattern(name: str) -> tuple[MetricPattern, str] | None:
    """``(pattern, set_name)`` when ``name`` is an instance of a registered pattern.
    Longer templates win so ``auc_pr_x_floor`` matches the floor pattern, not ``auc_pr_<set>``
    with ``set = "x_floor"``."""
    for template in sorted(PATTERNS, key=len, reverse=True):
        m = PATTERNS[template].regex.match(name)
        if m:
            return PATTERNS[template], m.group("set")
    return None


def expand_pattern(template: str, set_name: str) -> MetricSpec:
    """Concrete spec for one pattern and one set name (not added to ``REGISTRY``)."""
    return PATTERNS[template].expand(set_name)


def expand_sets(set_names: Iterable[str]) -> dict[str, MetricSpec]:
    """All pattern expansions for the given ``label.sets`` names (expanded at load)."""
    out: dict[str, MetricSpec] = {}
    for s in set_names:
        for pat in PATTERNS.values():
            spec = pat.expand(s)
            out[spec.name] = spec
    return out


def resolve(name: str, *, sets: Iterable[str] | None = None) -> MetricSpec:
    """Look ``name`` up in ``REGISTRY``, else expand it from a pattern when its set name is
    one of ``sets`` (the benchmark's ``label.sets``). Raises ``UnknownMetric``."""
    if name in REGISTRY:
        return REGISTRY[name]
    hit = match_pattern(name)
    if hit is not None:
        pat, set_name = hit
        if sets is not None and set_name in set(sets):
            return pat.expand(set_name)
        raise UnknownMetric(
            f"metric {name!r} matches pattern {pat.template!r} but {set_name!r} is not a declared label set "
            f"({sorted(sets) if sets is not None else 'none declared'})"
        )
    raise UnknownMetric(f"unknown metric {name!r}; registered: {sorted(REGISTRY)} + patterns {sorted(PATTERNS)}")


def resolve_names(
    names: Iterable[str],
    readout: str | Iterable[str],
    *,
    sets: Iterable[str] | None = None,
    kinds: Iterable[str] | None = None,
) -> dict[str, MetricSpec]:
    """Resolve every name for ``readout`` (load validation): unknown → ``UnknownMetric``;
    a spec whose ``readouts`` exclude the readout → ``MetricNotAllowed``; with ``kinds``, a
    spec whose kind is not listed → ``MetricNotAllowed`` (e.g. a floor named in ``metrics.list``)."""
    kinds_set = frozenset(kinds) if kinds is not None else None
    out: dict[str, MetricSpec] = {}
    for name in names:
        spec = resolve(name, sets=sets)
        if not allows(spec, readout):
            shown = readout if isinstance(readout, str) else "/".join(sorted(readout))
            raise MetricNotAllowed(f"metric {name!r} is not allowed for scorer {shown!r} (allowed: {sorted(spec.readouts)})")
        if kinds_set is not None and spec.kind not in kinds_set:
            raise MetricNotAllowed(f"metric {name!r} has kind {spec.kind!r}, expected one of {sorted(kinds_set)}")
        out[name] = spec
    return out


def allows(spec: Any, readout: str | Iterable[str]) -> bool:
    """Whether ``spec`` (a ``MetricSpec``) may be listed by a benchmark scored by ``readout``: one scorer name, or the
    names a scorer answers to (``Scorer.families()``: itself and its family)."""
    rs = frozenset(getattr(spec, "readouts", ()) or ())
    names = {readout} if isinstance(readout, str) else set(readout)
    return bool(names & rs) or ANY_READOUT in rs


def protocol_keys_for(specs: Iterable[MetricSpec]) -> frozenset[str]:
    """Union of ``protocol_keys`` — the per-benchmark protocol keys ``Benchmark.load`` requires
    to be present (and refuses undeclared ones)."""
    keys: set[str] = set()
    for s in specs:
        keys |= s.protocol_keys
    return frozenset(keys)
