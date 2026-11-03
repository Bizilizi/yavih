"""Intervals (intervals are not metrics): ``binomial`` = Wilson 95 %; ``bootstrap_clip`` =
seeded clip-level bootstrap (``metrics.bootstrap``), 2.5/97.5 percentiles; ``pooled_binomial`` = Wilson
on the pooled correct count over all held-out fold rows.

``interval(kind, name, …)`` is the driver's entry point: ``metrics.ci[name]`` → ``result.ci95[name]``.
The per-metric resampling unit is the CLIP (row): a clip's segments or pixels are anything but
independent, so ``PER_CLIP`` maps every metric that admits an interval to its per-clip quantity
(``float64 [n]`` or ``[n, k]`` with NaN for unscored cells) and the bootstrap pools the resampled
clips exactly as the point estimate does. Ranking metrics have no per-clip quantity; ``RESAMPLED`` maps
them to a statistic recomputed on every resample (``map``: macro AP over the fixed ``C``, evaluated with
the resample's integer clip weights, so a duplicated clip is one more tied row exactly as in
``classification.macro_average_precision`` on the materialised resample). ``registry.CI_ADMISSIBLE`` is
the declarative table ``Benchmark.load`` checks; ``tests/test_metrics_intervals.py`` pins it equal to
what this module computes. A folder metric (``eval.py`` ``METRICS``) brings its own per-clip function and the
interval kinds it admits (``MetricSpec.per_clip`` / ``MetricSpec.ci``); ``interval(..., spec=...)`` uses them."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from yavih.metrics import box, classification, depth, segment
from yavih.metrics._common import as_array, meta

__all__ = ["wilson", "bootstrap_clip", "pooled_binomial", "interval", "PER_CLIP", "per_clip_values",
           "RESAMPLED", "map_resample_stats", "bootstrap_draws", "admissible", "DEFAULT_BOOTSTRAP"]

Z95 = 1.959963984540054
DEFAULT_BOOTSTRAP: dict[str, int] = {"samples": 2000, "seed": 0}


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (``n ≥ 1``), clipped to ``[0, 1]``."""
    if n < 1:
        raise ValueError("wilson: n must be ≥ 1")
    if not 0 <= k <= n:
        raise ValueError(f"wilson: k must lie in [0, n], got k={k}, n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    # The bounds are exactly 0 / 1 at k == 0 / k == n; rounding must not report 0.9999999999999999.
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (lo, hi)


def pooled_binomial(correct_per_fold, n_per_fold) -> tuple[float, float]:
    """Wilson on the pooled correct count over all held-out rows (ESC-50: 2 000)."""
    k = int(np.sum(as_array(correct_per_fold)))
    n = int(np.sum(as_array(n_per_fold)))
    return wilson(k, n)


def bootstrap_clip(per_clip_values, *, samples: int, seed: int, statistic=None) -> tuple[float, float]:
    """Seeded clip-level percentile bootstrap (2.5 / 97.5), ``numpy.random.default_rng(seed)``.

    ``per_clip_values`` is ``[n]`` (one number per clip) or ``[n, k]`` (per-clip cells, NaN = unscored);
    the default statistic is the POOLED mean over the resampled clips' finite cells — for ``[n]`` the
    plain mean, for ``seg_acc``'s ``[n, T]`` the pooled segment accuracy. A custom ``statistic(values)``
    receives the resampled rows and returns a float."""
    v = as_array(per_clip_values, np.float64)
    if v.ndim == 1:
        v = v[:, None]
    if v.ndim != 2 or v.shape[0] == 0:
        raise ValueError("bootstrap_clip: per_clip_values must be a non-empty [n] or [n, k] array")
    if samples < 1:
        raise ValueError("bootstrap_clip: samples must be ≥ 1")
    n = v.shape[0]
    draws = bootstrap_draws(n, samples=samples, seed=seed)
    if statistic is None:
        clip_sum = np.nansum(v, axis=1)
        clip_cnt = np.isfinite(v).sum(axis=1).astype(np.float64)
        if clip_cnt.sum() == 0:
            raise ValueError("bootstrap_clip: no finite value")
        num = clip_sum[draws].sum(axis=1)
        den = clip_cnt[draws].sum(axis=1)
        stats = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
    else:
        stats = np.asarray([float(statistic(v[d])) for d in draws], dtype=np.float64)
    return _percentile_interval(stats)


def bootstrap_draws(n: int, *, samples: int, seed: int) -> np.ndarray:
    """``int64 [samples, n]`` clip indices drawn with replacement, ``numpy.random.default_rng(seed)``:
    the one resampling scheme every ``bootstrap_clip`` interval shares."""
    if n < 1:
        raise ValueError("bootstrap_clip: needs at least one clip")
    if samples < 1:
        raise ValueError("bootstrap_clip: samples must be ≥ 1")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, n, size=(int(samples), n))


def _percentile_interval(stats: np.ndarray) -> tuple[float, float]:
    stats = np.asarray(stats, dtype=np.float64)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        raise ValueError("bootstrap_clip: every resample was undefined")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (float(lo), float(hi))


def map_resample_stats(outputs, targets, draws: np.ndarray) -> np.ndarray:
    """Macro AP (``classification.macro_average_precision``: fixed ``C``, zero-positive classes count
    ``AP = 0``) of every resample in ``draws`` (``[samples, n]`` clip indices), ``float64 [samples]``.

    Vectorised over clips and classes: each class column is sorted once (descending, stable) and its
    tie groups found once; a resample is the integer weight vector ``w = bincount(draw)``, and
    ``AP_c = Σ_k w_k y_k · P(end of k's tie group) / Σ_k w_k y_k`` with ``P = cumΣ(w·y) / cumΣ(w)``.
    That equals AP on the materialised resample: copies of a clip share its score, hence its threshold.
    Cost ≈ six passes over ``[n, C]`` per resample (FSD50K eval, 10 231 × 200: ~20 ms)."""
    o = as_array(outputs, np.float64)
    t = as_array(targets)
    if o.ndim != 2 or t.ndim != 2 or o.shape != t.shape or o.shape[0] == 0:
        raise ValueError(f"map interval: outputs/targets must be the same non-empty [n, C], got {o.shape} / {t.shape}")
    y = (t > 0.5) if not np.issubdtype(t.dtype, np.bool_) else t
    n, _ = o.shape
    order = np.argsort(-o, axis=0, kind="stable")
    s_sorted = np.take_along_axis(o, order, axis=0)
    y_sorted = np.take_along_axis(y, order, axis=0).astype(np.float64)
    # index of the last row of each row's tie group (per column): the threshold the row is scored at
    is_end = np.ones_like(s_sorted, dtype=bool)
    is_end[:-1] = s_sorted[:-1] != s_sorted[1:]
    ends = np.where(is_end, np.arange(n)[:, None], n)
    group_end = np.minimum.accumulate(ends[::-1], axis=0)[::-1]
    draws = np.asarray(draws)
    out = np.empty(draws.shape[0], dtype=np.float64)
    for b, draw in enumerate(draws):
        w = np.bincount(draw, minlength=n).astype(np.float64)[order]
        wy = w * y_sorted
        cum_w = np.cumsum(w, axis=0)
        cum_tp = np.cumsum(wy, axis=0)
        prec = np.divide(cum_tp, cum_w, out=np.zeros_like(cum_tp), where=cum_w > 0)
        pos = cum_tp[-1]
        ap = np.divide((wy * np.take_along_axis(prec, group_end, axis=0)).sum(axis=0), pos,
                       out=np.zeros_like(pos), where=pos > 0)
        out[b] = float(ap.mean())
    return out


#: metric name → ``fn(outputs, targets, draws) -> float64 [samples]`` for metrics with no per-clip
#: quantity whose ``bootstrap_clip`` interval recomputes the statistic on every resample.
RESAMPLED: dict[str, Callable[..., np.ndarray]] = {
    "map": map_resample_stats,
}


# ----------------------------------------------------------------------------- per-clip quantities


def _as_float(mask: np.ndarray) -> np.ndarray:
    return as_array(mask).astype(np.float64)


def _seg_boundary(outputs, targets, protocol, split_meta) -> np.ndarray:
    m = segment.boundary_mask(targets, protocol, split_meta)
    return segment.segment_correct(outputs, targets)[m]


def _clip_acc(outputs, targets, protocol, split_meta) -> np.ndarray:
    c = segment.segment_correct(outputs, targets)
    cnt = np.isfinite(c).sum(axis=1)
    return np.where(cnt > 0, np.nansum(c, axis=1) / np.maximum(cnt, 1), np.nan)


def _depth(key: str) -> Callable[..., np.ndarray]:
    def fn(outputs, targets, protocol, split_meta) -> np.ndarray:
        return depth.depth_per_image(outputs, targets, protocol)[key]

    return fn


#: metric name → per-clip values ``[n]`` or ``[n, k]`` (NaN = unscored), on the same population the
#: scalar metric scores. ``binomial`` requires the ``[n]`` 0/1 kind.
PER_CLIP: dict[str, Callable[..., np.ndarray]] = {
    "acc": lambda o, t, p, m: _as_float(classification.correct(o, t)),
    "seg_acc": lambda o, t, p, m: segment.segment_correct(o, t),
    "seg_acc_boundary": _seg_boundary,
    "clip_acc": _clip_acc,
    "iou_mean": lambda o, t, p, m: box.ious(o, t),
    "acc@0.5": lambda o, t, p, m: _as_float(box.ious(o, t) >= 0.5),
    "acc@0.3": lambda o, t, p, m: _as_float(box.ious(o, t) >= 0.3),
    "center_in_gt": lambda o, t, p, m: _as_float(box.center_inside(o, t)),
    "d1": _depth("d1"),
    "d2": _depth("d2"),
    "d3": _depth("d3"),
    "absrel": _depth("absrel"),
    "rmse": _depth("rmse"),
}

_BINARY = frozenset({"acc", "acc@0.5", "acc@0.3", "center_in_gt"})
_FOLD_ACC = frozenset({"acc_mean", "acc_pstd", "acc_per_fold"})


def per_clip_values(name: str, outputs, targets, protocol, split_meta) -> np.ndarray:
    if name not in PER_CLIP:
        raise ValueError(f"no per-clip quantity is defined for metric {name!r} (supported: {sorted(PER_CLIP)})")
    return PER_CLIP[name](outputs, targets, protocol, split_meta)


def admissible(kind: str) -> frozenset[str]:
    """The metric names this module can compute a ``kind`` interval for (pinned equal to
    ``registry.CI_ADMISSIBLE`` by the tests)."""
    if kind == "binomial":
        return frozenset(_BINARY)
    if kind == "bootstrap_clip":
        return frozenset(PER_CLIP) | frozenset(RESAMPLED)
    if kind == "pooled_binomial":
        return frozenset(_BINARY) | _FOLD_ACC
    raise ValueError(f"unknown interval kind {kind!r} (CI_KINDS)")


def _fold_successes(name: str, split_meta: Any) -> tuple[int, int]:
    folds = meta(split_meta, "folds")
    if not isinstance(folds, Mapping) or not folds:
        raise ValueError("pooled_binomial needs split_meta['folds'] = {fold: {outputs, targets}}")
    base = "acc" if name in _FOLD_ACC | {"acc"} else name
    if base not in _BINARY:
        raise ValueError(f"pooled_binomial is defined for per-row 0/1 metrics, not {name!r}")
    k = n = 0
    for f in folds.values():
        v = per_clip_values(base, f["outputs"], f["targets"], None, f)
        k += int(np.nansum(v))
        n += int(np.isfinite(v).sum())
    return k, n


def _folder_interval(kind: str, name: str, per_clip: Callable[..., Any], outputs, targets, protocol, split_meta) -> list[float]:
    """An interval for a folder metric from its own ``MetricSpec.per_clip`` (``Benchmark.load`` already checked that
    ``kind`` is in the spec's ``ci``): the same Wilson / clip bootstrap / pooled Wilson as the registry metrics."""
    if kind == "binomial":
        v = as_array(per_clip(outputs, targets, protocol, split_meta)).astype(np.float64)
        return list(wilson(int(np.nansum(v)), int(np.isfinite(v).sum())))
    if kind == "bootstrap_clip":
        cfg = dict(DEFAULT_BOOTSTRAP)
        cfg.update(meta(split_meta, "bootstrap") or {})
        v = as_array(per_clip(outputs, targets, protocol, split_meta)).astype(np.float64)
        return list(bootstrap_clip(v, samples=int(cfg["samples"]), seed=int(cfg["seed"])))
    if kind == "pooled_binomial":
        folds = meta(split_meta, "folds")
        if not isinstance(folds, Mapping) or not folds:
            raise ValueError("pooled_binomial needs split_meta['folds'] = {fold: {outputs, targets}}")
        k = n = 0
        for f in folds.values():
            v = as_array(per_clip(f["outputs"], f["targets"], protocol, f)).astype(np.float64)
            k += int(np.nansum(v))
            n += int(np.isfinite(v).sum())
        return list(wilson(k, n))
    raise ValueError(f"unknown interval kind {kind!r} (CI_KINDS)")


def interval(kind: str, name: str, outputs, targets, protocol, split_meta, *, spec: Any = None) -> list[float]:
    """Dispatch on ``metrics.ci[name]`` → ``[lo, hi]`` for ``result.ci95[name]``.
    ``bootstrap_clip`` reads ``samples``/``seed`` from ``split_meta["bootstrap"]`` (``metrics.bootstrap``;
    default 2 000 / 0). ``spec`` is the metric's ``MetricSpec``: a folder metric that declares ``kind`` in its
    ``ci`` is computed by its own ``interval`` when it has one, else from its own ``per_clip``; every other name
    uses this module's tables."""
    own = getattr(spec, "interval", None)
    if own is not None and kind in (getattr(spec, "ci", None) or ()):
        lo, hi = own(kind, outputs, targets, protocol, split_meta)
        return [float(lo), float(hi)]
    per_clip = getattr(spec, "per_clip", None)
    if per_clip is not None and kind in (getattr(spec, "ci", None) or ()):
        return _folder_interval(kind, name, per_clip, outputs, targets, protocol, split_meta)
    if kind == "binomial":
        if name not in _BINARY:
            raise ValueError(f"binomial interval is defined for per-row 0/1 metrics, not {name!r}")
        v = per_clip_values(name, outputs, targets, protocol, split_meta)
        return list(wilson(int(np.nansum(v)), int(np.isfinite(v).sum())))
    if kind == "bootstrap_clip":
        cfg = dict(DEFAULT_BOOTSTRAP)
        cfg.update(meta(split_meta, "bootstrap") or {})
        if name in RESAMPLED:
            n = int(as_array(outputs).shape[0])
            draws = bootstrap_draws(n, samples=int(cfg["samples"]), seed=int(cfg["seed"]))
            return list(_percentile_interval(RESAMPLED[name](outputs, targets, draws)))
        v = per_clip_values(name, outputs, targets, protocol, split_meta)
        return list(bootstrap_clip(v, samples=int(cfg["samples"]), seed=int(cfg["seed"])))
    if kind == "pooled_binomial":
        k, n = _fold_successes(name, split_meta)
        return list(wilson(k, n))
    raise ValueError(f"unknown interval kind {kind!r} (CI_KINDS)")
