"""Label-only floors (computed from TRAIN targets only): the constant
predictions each floor metric scores — train majority class (``acc_floor``), train-prevalence
constant score (``map_floor``, ``auc_pr_<set>_floor``), most frequent train segment label
(``seg_acc_floor``), per-pixel mean train log-depth map (``d1_floor``, one streaming pass over the
train target memmap), mean train box (``iou_floor``).

numpy only: floors are scored "through the same metric function on the eval rows", so the
outputs are the head's raw-output shape repeated ``n_eval`` times and the metric never knows whether
a head or a floor produced them."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["floor_outputs", "mean_log_depth_map", "majority_class", "class_prevalence", "mean_box"]

_CHUNK_ROWS = 256


def majority_class(labels: np.ndarray, num_classes: int | None = None) -> int:
    """Most frequent label (ties → smallest index, so the floor is a deterministic function of the counts)."""
    lab = np.asarray(labels).reshape(-1)
    lab = lab[np.isfinite(lab.astype(np.float64))] if lab.dtype.kind == "f" else lab
    lab = lab[lab >= 0].astype(np.int64)
    if lab.size == 0:
        raise ValueError("floor: no valid train labels")
    c = int(num_classes) if num_classes is not None else int(lab.max()) + 1
    return int(np.argmax(np.bincount(lab, minlength=c)))


def class_prevalence(multi_hot: np.ndarray) -> np.ndarray:
    """Per-class positive rate over the train rows (NaN cells = "does not apply" are skipped) → ``[C]``."""
    m = np.asarray(multi_hot, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"floor: multi-label train targets must be [n, C], got {m.shape}")
    with np.errstate(invalid="ignore"):
        prev = np.nanmean(m, axis=0)
    return np.nan_to_num(prev, nan=0.0).astype(np.float32)


def mean_box(boxes: np.ndarray) -> np.ndarray:
    """Mean of the finite ``(cx, cy, w, h)`` train boxes → ``[4]``."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    ok = np.isfinite(b).all(axis=1)
    if not ok.any():
        raise ValueError("floor: no finite train boxes")
    return b[ok].mean(axis=0).astype(np.float32)


def _iter_chunks(targets: Any) -> Any:
    """Yield ``[k, ...]`` chunks of a train target source: ``TargetReader``-like (``.rows(idx)`` + ``.shape``),
    memmap/ndarray (sliced), never the whole array at once."""
    n = int(targets.shape[0])
    rows = getattr(targets, "rows", None)
    for lo in range(0, n, _CHUNK_ROWS):
        hi = min(n, lo + _CHUNK_ROWS)
        yield np.asarray(rows(np.arange(lo, hi)) if callable(rows) else targets[lo:hi])


def mean_log_depth_map(train_targets: Any, *, max_depth: float | None = None) -> np.ndarray:
    """Per-pixel mean train log-depth over valid pixels (``0 < d <= max_depth``), one streaming pass;
    pixels valid in no train image fall back to the global mean log-depth → ``[S, S]`` float32."""
    acc = None
    cnt = None
    for chunk in _iter_chunks(train_targets):
        d = chunk.astype(np.float64)
        valid = np.isfinite(d) & (d > 0)
        if max_depth is not None:
            valid &= d <= float(max_depth)
        logd = np.where(valid, np.log(np.where(valid, d, 1.0)), 0.0)
        if acc is None:
            acc = np.zeros(d.shape[1:], dtype=np.float64)
            cnt = np.zeros(d.shape[1:], dtype=np.float64)
        acc += logd.sum(axis=0)
        cnt += valid.sum(axis=0)
    if acc is None or cnt.sum() == 0:
        raise ValueError("floor: no valid train depth pixels")
    global_mean = acc.sum() / cnt.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), global_mean)
    return mean.astype(np.float32)


def floor_outputs(readout: str, train_targets: Any, n_eval: int, *, protocol: dict, label_kind: str,
                  num_classes: int | None = None) -> np.ndarray:
    """Constant eval outputs in the head's raw-output shape, fed to the ``kind == "floor"`` metric functions.
    Each readout defines its own (``Readout.floor_outputs`` in ``yavih/readouts/<name>.py``):

    - ``clf`` single: one-hot logits of the train majority class → ``[n_eval, C]``.
    - ``clf`` multi / multi_sets: train prevalence per class as the constant score → ``[n_eval, C]``.
    - ``seq``: one-hot logits of the most frequent train segment label (pooled over all (clip, segment)
      cells, NaN/negative = unscored) → ``[n_eval, T, C]``.
    - ``grid``: ``exp`` of the per-pixel mean train log-depth map → ``[n_eval, S, S]`` (a depth, like the head).
    - ``box``: the mean train box → ``[n_eval, 4]``.
    """
    from yavih.readouts import registry

    try:
        r = registry.get(readout)
    except registry.UnknownScorer as e:
        raise ValueError(f"floor_outputs: unknown readout {readout!r}") from e
    if r.floor_outputs is None:
        raise ValueError(f"floor_outputs: readout {readout!r} declares no floor_outputs")
    return r.floor_outputs(train_targets, int(n_eval), protocol=protocol, label_kind=label_kind, num_classes=num_classes)
