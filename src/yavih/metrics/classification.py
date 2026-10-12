"""clf metrics: accuracy, mAP, per-set AUCs, folds aggregates, disclosures.

Every function has the registry signature ``fn(outputs, targets, protocol, split_meta)``;
per-set functions additionally take ``set_name`` (bound by ``registry.expand_sets``).
``outputs`` = the head's raw eval outputs (``float32 [n, C]`` logits), ``targets`` = ``int64 [n]``
(single) or ``float32 [n, C]`` (multi / per set), ``split_meta`` = ``{n_train, n_eval, train_targets,
folds: {fold: {outputs, targets}}, ...}`` (see ``yavih.metrics._common``). Pure numpy: ``average_precision``
and ``roc_auc`` reproduce scikit-learn's ``average_precision_score`` / ``roc_auc_score`` exactly
(tie groups collapsed to one threshold; pinned against sklearn in ``tests/test_metrics_classification.py``),
so the probe env needs no sklearn for scoring.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from yavih.metrics._common import (
    as_array,
    count_rows,
    float_or_raise,
    meta,
    predictions,
    require_meta,
    set_targets,
)

__all__ = [
    "acc", "n_eval", "n_train", "acc_floor",
    "map", "effective_pos_classes", "map_floor",
    "auc_pr", "auc_roc", "auc_pr_floor",
    "acc_mean", "acc_pstd", "acc_per_fold", "n_folds",
    "average_precision", "roc_auc", "macro_average_precision", "macro_roc_auc",
    "majority_class", "train_prevalence", "correct",
]


# ----------------------------------------------------------------------------- single-label


def _single_targets(targets: Any) -> np.ndarray:
    t = as_array(targets)
    if t.ndim == 2:  # a one-hot matrix is accepted for a single-label population
        t = np.argmax(t, axis=-1)
    if t.ndim != 1:
        raise ValueError(f"single-label targets must be [n], got shape {t.shape}")
    return t.astype(np.int64)


def correct(outputs: Any, targets: Any) -> np.ndarray:
    """Per-row exact-argmax success ``bool [n]`` (shared with the binomial intervals)."""
    preds = predictions(outputs, ndim_scores=2)
    t = _single_targets(targets)
    if preds.shape != t.shape:
        raise ValueError(f"outputs rows {preds.shape} do not match targets {t.shape}")
    return preds == t


def acc(outputs, targets, protocol, split_meta) -> float:
    """Exact-argmax accuracy."""
    c = correct(outputs, targets)
    if c.size == 0:
        raise ValueError("acc: empty eval population")
    return float(c.mean())


def n_eval(outputs, targets, protocol, split_meta) -> float:
    """Disclosure: number of scored eval rows (registered for every readout)."""
    return float(count_rows(targets, split_meta, "n_eval"))


def n_train(outputs, targets, protocol, split_meta) -> float:
    """Disclosure: number of train rows (``split_meta["n_train"]`` else ``len(train_targets)``)."""
    n = meta(split_meta, "n_train")
    if n is None:
        n = as_array(require_meta(split_meta, "train_targets", "n_train")).shape[0]
    return float(n)


def majority_class(train_targets: Any, num_classes: int | None = None) -> int:
    """Train majority class; ties go to the lowest index (``bincount().argmax()``)."""
    t = _single_targets(train_targets)
    if t.size == 0:
        raise ValueError("majority_class: no train targets")
    return int(np.argmax(np.bincount(t, minlength=num_classes or 0)))


def acc_floor(outputs, targets, protocol, split_meta) -> float:
    """Accuracy of the train majority class on eval. ``outputs`` is the floor's constant
    prediction when the driver provides it, else derived from ``split_meta["train_targets"]``."""
    t = _single_targets(targets)
    if outputs is None:
        cls = majority_class(require_meta(split_meta, "train_targets", "floors"))
        outputs = np.full(t.shape[0], cls, dtype=np.int64)
    return acc(outputs, t, protocol, split_meta)


# ----------------------------------------------------------------------------- ranking metrics


def _binary(y_true: np.ndarray) -> np.ndarray:
    y = as_array(y_true)
    return (y > 0.5) if not np.issubdtype(y.dtype, np.bool_) else y


def average_precision(scores: Any, y_true: Any) -> float:
    """AP of one binary column = ``sum_k (R_k − R_{k−1}) P_k`` over the distinct score thresholds in
    decreasing order (== ``sklearn.metrics.average_precision_score``). A column with no positive
    has no recall curve; the caller decides its value (``AP = 0``)."""
    s = as_array(scores, np.float64).ravel()
    y = _binary(y_true).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("average_precision: no positive in this column")
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    # Tied scores form one threshold: evaluate (tp, fp) at the last index of every tie group.
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tp = np.cumsum(y)[last].astype(np.float64)
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision))


def roc_auc(scores: Any, y_true: Any) -> float:
    """ROC-AUC of one binary column via the Mann–Whitney U with average ranks for ties (== trapezoidal
    area under the ROC curve, ``sklearn.metrics.roc_auc_score``). Undefined without both a positive
    and a negative (``ValueError``)."""
    s = as_array(scores, np.float64).ravel()
    y = _binary(y_true).ravel()
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc: needs at least one positive and one negative")
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    # Average ranks (1-based) over tie groups.
    bounds = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1, s.size]
    ranks_sorted = np.empty(s.size, dtype=np.float64)
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        ranks_sorted[lo:hi] = (lo + 1 + hi) / 2.0
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _multi_pair(outputs: Any, targets: Any) -> tuple[np.ndarray, np.ndarray]:
    o = as_array(outputs, np.float64)
    t = as_array(targets)
    if o.ndim != 2 or t.ndim != 2:
        raise ValueError(f"multi-label outputs/targets must be [n, C], got {o.shape} / {t.shape}")
    if o.shape != t.shape:
        raise ValueError(f"outputs {o.shape} and targets {t.shape} differ (fixed C)")
    if o.shape[0] == 0:
        raise ValueError("multi-label metric: empty eval population")
    return o, _binary(t)


def macro_average_precision(outputs: Any, targets: Any) -> float:
    """Macro AP over the fixed ``C`` columns; zero-positive classes count ``AP = 0``.
    Ranking-based, so logits and sigmoid agree."""
    o, t = _multi_pair(outputs, targets)
    aps = [average_precision(o[:, c], t[:, c]) if t[:, c].any() else 0.0 for c in range(o.shape[1])]
    return float(np.mean(aps))


def macro_roc_auc(outputs: Any, targets: Any) -> float:
    """Macro ROC-AUC over the classes for which it is defined (both a positive and a negative in
    eval); the undefined classes are excluded from the mean (the VGGSounder scorer's policy — ROC
    has no natural zero). Raises when no class is defined."""
    o, t = _multi_pair(outputs, targets)
    pos = t.sum(axis=0)
    defined = (pos > 0) & (pos < t.shape[0])
    if not defined.any():
        raise ValueError("macro_roc_auc: no class has both a positive and a negative")
    return float(np.mean([roc_auc(o[:, c], t[:, c]) for c in np.flatnonzero(defined)]))


def map(outputs, targets, protocol, split_meta) -> float:  # noqa: A001 - the registry name is `map`
    """Macro AP over the fixed C; zero-positive classes count AP = 0."""
    return float_or_raise(macro_average_precision(outputs, targets), "map")


def effective_pos_classes(outputs, targets, protocol, split_meta) -> float:
    """Disclosure: how many of the fixed ``C`` classes have ≥ 1 positive in eval."""
    t = _binary(as_array(targets))
    if t.ndim != 2:
        raise ValueError(f"effective_pos_classes: targets must be [n, C], got {t.shape}")
    return float(int((t.sum(axis=0) > 0).sum()))


def train_prevalence(train_targets: Any, num_classes: int) -> np.ndarray:
    """Per-class positive rate over the train rows ``float64 [C]``; train targets may be a multi-hot
    ``[n_train, C]`` matrix or single-label indices ``[n_train]`` (VGGSounder trains one-hot)."""
    t = as_array(train_targets)
    if t.ndim == 1:
        idx = t.astype(np.int64)
        if idx.size == 0:
            raise ValueError("train_prevalence: no train targets")
        return np.bincount(idx, minlength=num_classes).astype(np.float64) / idx.size
    if t.ndim != 2 or t.shape[1] != num_classes:
        raise ValueError(f"train targets must be [n_train, {num_classes}] or [n_train], got {t.shape}")
    if t.shape[0] == 0:
        raise ValueError("train_prevalence: no train targets")
    return _binary(t).mean(axis=0)


def _prevalence_outputs(outputs: Any, targets: np.ndarray, split_meta: Any) -> np.ndarray:
    """The constant train-prevalence score ``[n_eval, C]`` unless the driver supplied ``outputs``."""
    if outputs is not None:
        return as_array(outputs)
    prev = train_prevalence(require_meta(split_meta, "train_targets", "floors"), targets.shape[1])
    return np.broadcast_to(prev, targets.shape)


def map_floor(outputs, targets, protocol, split_meta) -> float:
    """Macro AP of the constant train-prevalence score."""
    t = as_array(targets)
    return float_or_raise(macro_average_precision(_prevalence_outputs(outputs, t, split_meta), t), "map_floor")


def auc_pr(outputs, targets, protocol, split_meta, *, set_name: str) -> float:
    """``auc_pr_<set>``: macro AP against the set's target matrix (same policy as ``map``)."""
    t = set_targets(targets, split_meta, set_name)
    return float_or_raise(macro_average_precision(outputs, t), f"auc_pr_{set_name}")


def auc_roc(outputs, targets, protocol, split_meta, *, set_name: str) -> float:
    """``auc_roc_<set>``: macro ROC-AUC over the defined classes of the set."""
    t = set_targets(targets, split_meta, set_name)
    return float_or_raise(macro_roc_auc(outputs, t), f"auc_roc_{set_name}")


def auc_pr_floor(outputs, targets, protocol, split_meta, *, set_name: str) -> float:
    """``auc_pr_<set>_floor``: macro AP of the constant train-prevalence score on the set."""
    t = set_targets(targets, split_meta, set_name)
    return float_or_raise(
        macro_average_precision(_prevalence_outputs(outputs, t, split_meta), t), f"auc_pr_{set_name}_floor"
    )


# ----------------------------------------------------------------------------- folds


def _folds(split_meta: Any) -> Mapping[str, Mapping[str, Any]]:
    folds = require_meta(split_meta, "folds", "folds")
    if not isinstance(folds, Mapping) or not folds:
        raise ValueError("split_meta['folds'] must be a non-empty {fold: {outputs, targets}} mapping")
    return folds


def acc_per_fold(outputs, targets, protocol, split_meta) -> list[float]:
    """Held-out accuracy per fold, in ``splits.folds`` order (``shape: list``)."""
    return [acc(f["outputs"], f["targets"], protocol, split_meta) for f in _folds(split_meta).values()]


def acc_mean(outputs, targets, protocol, split_meta) -> float:
    return float(np.mean(acc_per_fold(outputs, targets, protocol, split_meta)))


def acc_pstd(outputs, targets, protocol, split_meta) -> float:
    """Population standard deviation (``ddof = 0``) over the folds."""
    return float(np.std(acc_per_fold(outputs, targets, protocol, split_meta), ddof=0))


def n_folds(outputs, targets, protocol, split_meta) -> float:
    return float(len(_folds(split_meta)))
