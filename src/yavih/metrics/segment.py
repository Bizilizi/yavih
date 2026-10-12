"""seq metrics: ``seg_acc`` (pooled over valid (clip, segment)), ``clip_acc``,
``seg_acc_boundary`` on rows with ``event_end − event_start < num_segments``, ``n_boundary``, floors.
Targets ``int64 [n, T]`` with ``T = segments_scored``; segment ``k`` is media second
``segment_offset + k``; a negative target marks an unscored segment (masked from numerator and
denominator). Outputs ``[n, T, C]`` scores or ``[n, T]`` predictions (floors)."""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.metrics._common import as_array, meta, predictions, require_meta

__all__ = ["seg_acc", "clip_acc", "seg_acc_boundary", "n_boundary", "seg_acc_floor", "seg_acc_boundary_floor",
           "segment_targets", "segment_correct", "boundary_mask", "constant_segment_class"]


def segment_targets(event_start, event_end, label, protocol) -> list[int]:
    """Segment ``s`` takes the row's class iff ``event_start ≤ s < event_end`` else ``background_index``,
    for ``s in [segment_offset, segment_offset + segments_scored)``."""
    off = int(protocol["segment_offset"])
    t = int(protocol["segments_scored"])
    bg = int(protocol["background_index"])
    s0, s1, lab = int(event_start), int(event_end), int(label)
    return [lab if s0 <= s < s1 else bg for s in range(off, off + t)]


def _pair(outputs: Any, targets: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = as_array(targets)
    if t.ndim != 2:
        raise ValueError(f"seq targets must be [n, T], got shape {t.shape}")
    t = t.astype(np.int64)
    preds = predictions(outputs, ndim_scores=3)
    if preds.shape != t.shape:
        raise ValueError(f"outputs {preds.shape} do not match targets {t.shape}")
    return preds, t, t >= 0


def segment_correct(outputs: Any, targets: Any) -> np.ndarray:
    """``float64 [n, T]``: 1/0 per valid segment, NaN where the target is unscored (``< 0``) — the
    quantity ``seg_acc`` pools and ``bootstrap_clip`` resamples by clip."""
    preds, t, valid = _pair(outputs, targets)
    out = (preds == t).astype(np.float64)
    out[~valid] = np.nan
    return out


def _pooled(c: np.ndarray, what: str) -> float:
    n = np.isfinite(c).sum()
    if n == 0:
        raise ValueError(f"{what}: no valid segment to score")
    return float(np.nansum(c) / n)


def seg_acc(outputs, targets, protocol, split_meta) -> float:
    """Pooled accuracy over all valid (clip, segment) pairs (the AVE convention)."""
    return _pooled(segment_correct(outputs, targets), "seg_acc")


def clip_acc(outputs, targets, protocol, split_meta) -> float:
    """Mean of per-clip accuracies over clips with ≥ 1 valid segment (diverges from ``seg_acc`` only
    when clips carry different numbers of scored segments)."""
    c = segment_correct(outputs, targets)
    counts = np.isfinite(c).sum(axis=1)
    scored = counts > 0
    if not scored.any():
        raise ValueError("clip_acc: no clip has a valid segment")
    per_clip = np.nansum(c[scored], axis=1) / counts[scored]
    return float(per_clip.mean())


def boundary_mask(targets: Any, protocol: Any, split_meta: Any) -> np.ndarray:
    """``bool [n]``: rows whose event is shorter than the clip, ``event_end − event_start <
    num_segments``, from ``split_meta["event_start"/"event_end"]`` (manifest seconds) or
    a precomputed ``split_meta["boundary"]``. The scored window ``[segment_offset, +T)`` is narrower
    than the clip, so the mask cannot be read off the targets alone."""
    n = as_array(targets).shape[0]
    pre = meta(split_meta, "boundary")
    if pre is not None:
        m = as_array(pre).astype(bool)
    else:
        start = meta(split_meta, "event_start")
        end = meta(split_meta, "event_end")
        if start is None or end is None:
            raise ValueError(
                "seg_acc_boundary needs split_meta['event_start'] and ['event_end'] (or ['boundary']) per eval row"
            )
        m = (as_array(end, np.float64) - as_array(start, np.float64)) < float(protocol["num_segments"])
    if m.shape != (n,):
        raise ValueError(f"boundary mask has shape {m.shape}, expected ({n},)")
    return m


def seg_acc_boundary(outputs, targets, protocol, split_meta) -> float:
    """``seg_acc`` restricted to the boundary rows (AVE test: 104 of 316)."""
    m = boundary_mask(targets, protocol, split_meta)
    if not m.any():
        raise ValueError("seg_acc_boundary: no boundary row in the eval population")
    return _pooled(segment_correct(outputs, targets)[m], "seg_acc_boundary")


def n_boundary(outputs, targets, protocol, split_meta) -> float:
    """Disclosure: number of boundary rows."""
    return float(int(boundary_mask(targets, protocol, split_meta).sum()))


def constant_segment_class(train_targets: Any) -> int:
    """Most frequent TRAIN segment label over the valid segments; ties to the lowest index."""
    t = as_array(train_targets).astype(np.int64).ravel()
    t = t[t >= 0]
    if t.size == 0:
        raise ValueError("constant_segment_class: no valid train segment")
    return int(np.argmax(np.bincount(t)))


def _floor_outputs(outputs: Any, targets: Any, split_meta: Any) -> Any:
    if outputs is not None:
        return outputs
    cls = constant_segment_class(require_meta(split_meta, "train_targets", "floors"))
    return np.full(as_array(targets).shape[:2], cls, dtype=np.int64)


def seg_acc_floor(outputs, targets, protocol, split_meta) -> float:
    """Most frequent train segment label scored through ``seg_acc`` (AVE ≈ 0.164)."""
    return seg_acc(_floor_outputs(outputs, targets, split_meta), targets, protocol, split_meta)


def seg_acc_boundary_floor(outputs, targets, protocol, split_meta) -> float:
    """The same constant predictor read on the boundary rows (AVE ≈ 0.499)."""
    return seg_acc_boundary(_floor_outputs(outputs, targets, split_meta), targets, protocol, split_meta)
