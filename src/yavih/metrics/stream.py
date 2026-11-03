"""Metrics of streaming benchmarks, with intervals resampled by recording.

The quantities follow the reference evaluators of the online tasks:

* :func:`perframe_map` — online action detection's frame-level mean AP (LSTR / TeSTra
  ``perframe_average_precision``): AP per class over every scored frame of every recording, the background class
  and the ignored class skipped, a class without a positive frame left out of the mean; THUMOS'14 first removes the
  frames whose ground truth is ``Ambiguous`` (``valid``).
* :func:`calibrated_ap` — TVSeries' calibrated AP (De Geest et al. 2016), with the reference's exact formula (the
  ratio of negatives to positives and the ``eps`` terms of ``calibrated_average_precision_score``).
* :func:`perstage_map` — the action-progress breakdown (``perstage_average_precision``): AP over the background
  frames plus the frames of each action instance's ``[s, e)`` decile of progress.
* :func:`bin_macro_ap` — macro AP over a frozen class set on dense bins (a multilabel causal recognition task).
* :func:`class_mean_topk_recall` — EPIC-KITCHENS anticipation's class-mean top-k recall (RU-LSTM ``topk_recall``):
  per class, the fraction of its instances whose top-k scores contain it, averaged over the classes present.

Every metric is a statistic of rows (frames, bins, anchors) grouped into recordings; :func:`recording_interval`
resamples recordings (``yavih.metrics.intervals.bootstrap_draws``) and evaluates the statistic with integer row
weights, so a resample equals the materialised one. numpy only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from yavih.metrics.intervals import bootstrap_draws

__all__ = [
    "weighted_ap",
    "weighted_calibrated_ap",
    "perframe_map",
    "calibrated_ap",
    "perstage_map",
    "bin_macro_ap",
    "class_mean_topk_recall",
    "segment_accuracy",
    "recording_interval",
]

_EPS = float(np.finfo(float).eps)


def weighted_ap(scores: Any, y: Any, w: Any | None = None) -> float:
    """Average precision of one binary column (== ``sklearn.metrics.average_precision_score`` on the rows repeated
    ``w`` times): tied scores form one threshold. ``nan`` without a positive."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    t = np.asarray(y).ravel().astype(bool)
    w = np.ones(s.size) if w is None else np.asarray(w, dtype=np.float64).ravel()
    keep = w > 0
    s, t, w = s[keep], t[keep], w[keep]
    pos = float(w[t].sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    s, t, w = s[order], t[order], w[order]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tp = np.cumsum(w * t)[last]
    tot = np.cumsum(w)[last]
    precision = tp / tot
    recall = tp / pos
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def weighted_calibrated_ap(scores: Any, y: Any, w: Any | None = None) -> float:
    """Calibrated AP (``calibrated_average_precision_score``): rows sorted by decreasing score, ``cprec = TP / (TP +
    FP / ratio)`` with ``ratio = #negatives / #positives``, averaged over the positives. Integer weights ``w`` stand
    for repeated rows (consecutive copies). ``nan`` without a positive."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    t = np.asarray(y).ravel().astype(bool)
    w = np.ones(s.size, dtype=np.int64) if w is None else np.asarray(w).ravel().astype(np.int64)
    keep = w > 0
    s, t, w = s[keep], t[keep], w[keep]
    pos = int(w[t].sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    t, w = t[order], w[order]
    ratio = float(w[~t].sum()) / pos
    tp_before = np.cumsum(w * t) - w * t
    fp_cum = np.cumsum(w * ~t)
    total = 0.0
    wp, tb, fp = w[t], tp_before[t], fp_cum[t]
    for i in range(1, int(wp.max()) + 1):
        m = wp >= i
        tps = tb[m] + i
        total += float(np.sum(tps / (tps + fp[m] / (ratio + _EPS) + _EPS)))
    return total / pos


def _columns(n_classes: int, ignore: Sequence[int]) -> list[int]:
    skip = {int(i) for i in ignore}
    return [c for c in range(n_classes) if c not in skip]


def perframe_map(scores: Any, labels: Any, *, ignore: Sequence[int] = (0,), valid: Any | None = None, w: Any | None = None,
                 calibrated: bool = False, per_class: bool = False) -> Any:
    """Frame-level mean (calibrated) AP: ``scores`` / ``labels`` ``[M, C]`` (labels multi-hot, background column
    included), columns in ``ignore`` skipped, rows with ``valid`` false removed, classes without a positive frame left
    out of the mean. ``per_class`` returns ``(mean, {class: ap})``."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if valid is not None:
        keep = np.asarray(valid, dtype=bool)
        s, y = s[keep], y[keep]
        w = None if w is None else np.asarray(w)[keep]
    fn = weighted_calibrated_ap if calibrated else weighted_ap
    aps = {}
    for c in _columns(s.shape[1], ignore):
        if np.any(y[:, c] if w is None else (y[:, c].astype(bool) & (np.asarray(w) > 0))):
            aps[c] = fn(s[:, c], y[:, c], w)
    mean = float(np.mean(list(aps.values()))) if aps else float("nan")
    return (mean, aps) if per_class else mean


def calibrated_ap(scores: Any, labels: Any, **kw: Any) -> Any:
    """:func:`perframe_map` with calibrated AP (TVSeries)."""
    return perframe_map(scores, labels, calibrated=True, **kw)


def perstage_map(scores: Any, labels: Any, recording: Any, *, ignore: Sequence[int] = (0,), calibrated: bool = True) -> dict[str, float]:
    """Mean (calibrated) AP per action-progress decile (``"0-10"`` … ``"90-100"``), per the reference: for each class,
    the background frames of that class column plus, of every action instance (a maximal run of positive frames within
    one recording), the frames from ``start + int(len * s)`` to ``max(that + 1, start + int(len * e))``."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    rec = np.asarray(recording)
    fn = weighted_calibrated_ap if calibrated else weighted_ap
    out = {}
    for k in range(10):
        ps, pe = k / 10, (k + 1) / 10
        aps = []
        for c in _columns(s.shape[1], ignore):
            col = y[:, c]
            idx = list(np.flatnonzero(~col))
            starts = np.flatnonzero(col & ~np.r_[False, col[:-1] & (rec[1:] == rec[:-1])])
            ends = np.flatnonzero(col & ~np.r_[col[1:] & (rec[1:] == rec[:-1]), False])
            for a, b in zip(starts, ends, strict=True):
                n = int(b - a)
                lo = int(a) + int(n * ps)
                hi = max(lo + 1, int(a) + int(n * pe))
                idx.extend(range(lo, hi))
            idx_a = np.asarray(sorted(idx), dtype=np.int64)
            if col[idx_a].any():
                aps.append(fn(s[idx_a, c], col[idx_a]))
        out[f"{k * 10}-{(k + 1) * 10}"] = float(np.mean(aps)) if aps else float("nan")
    return out


def bin_macro_ap(scores: Any, labels: Any, *, classes: Sequence[int], w: Any | None = None) -> float:
    """Macro AP over the frozen class set ``classes`` on multilabel bins (``labels`` ``[M, C]`` in ``{0, 1}``); every
    listed class must have a positive (the folder freezes the set on the evaluation split)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    aps = [weighted_ap(s[:, c], y[:, c], w) for c in classes]
    if any(np.isnan(a) for a in aps):
        raise ValueError("bin_macro_ap: a class of the frozen set has no positive in these rows")
    return float(np.mean(aps))


def class_mean_topk_recall(scores: Any, labels: Any, *, k: int = 5, classes: Sequence[int] | None = None, w: Any | None = None) -> float:
    """Class-mean top-k recall (RU-LSTM ``topk_recall``): over the classes present in ``labels`` (restricted to
    ``classes`` when given, e.g. the many-shot set), the fraction of each class's instances whose top-``k`` scores
    contain it, averaged over those classes. ``labels`` ``[M]`` int (``-1`` = not scored)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(np.int64)
    w = np.ones(y.size) if w is None else np.asarray(w, dtype=np.float64)
    keep = (y >= 0) & (w > 0)
    s, y, w = s[keep], y[keep], w[keep]
    top = np.argpartition(-s, kth=min(k, s.shape[1]) - 1, axis=1)[:, :k]
    hit = (top == y[:, None]).any(axis=1)
    present = np.unique(y)
    if classes is not None:
        present = np.intersect1d(present, np.asarray(list(classes), dtype=np.int64))
    if present.size == 0:
        return float("nan")
    rec = [float(w[(y == c) & hit].sum() / w[y == c].sum()) for c in present]
    return float(np.mean(rec))


def segment_accuracy(scores: Any, labels: Any, *, w: Any | None = None) -> float:
    """Accuracy of the arg-max over single-label bins (``labels`` ``[M]`` int, ``-1`` = not scored)."""
    s = np.asarray(scores)
    y = np.asarray(labels).astype(np.int64)
    w = np.ones(y.size) if w is None else np.asarray(w, dtype=np.float64)
    keep = y >= 0
    return float(np.sum(w[keep] * (s[keep].argmax(axis=1) == y[keep])) / np.sum(w[keep]))


def recording_interval(stat: Callable[[np.ndarray], float], recording: Any, *, samples: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap (2.5 / 97.5) resampling RECORDINGS: ``stat(w)`` evaluates the metric with integer row
    weights ``w`` (a row's weight = how often its recording was drawn). Resamples where the statistic is undefined
    (``nan``) are dropped."""
    rec = np.asarray(recording)
    uniq, inv = np.unique(rec, return_inverse=True)
    draws = bootstrap_draws(len(uniq), samples=samples, seed=seed)
    vals = []
    for d in draws:
        counts = np.bincount(d, minlength=len(uniq))
        v = stat(counts[inv])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        raise ValueError("recording_interval: the statistic is undefined on every resample")
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return (float(lo), float(hi))
