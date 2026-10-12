"""box metrics: ``iou_mean``, ``acc@0.5`` (``acc_at_05``), ``acc@0.3`` (``acc_at_03``),
``center_in_gt``, ``n_excluded_degenerate``, ``iou_floor`` (mean train box), controls ``iou_shuffled`` and
``audio_blind = not (McNemar z over paired acc@0.5 successes > 3)``. Boxes ``(cx, cy, w, h)`` from
``x1|y1|x2|y2`` sanitized ``max(x, 0)``; degenerate boxes excluded and counted. A target row with a
non-finite coordinate (``parse_boxes`` writes NaN for a degenerate cell) is dropped from every
denominator."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from yavih.metrics._common import as_array, float_or_raise, meta, require_meta

__all__ = ["iou_mean", "acc_at_05", "acc_at_03", "center_in_gt", "n_excluded_degenerate", "iou_floor",
           "iou_shuffled", "audio_blind", "box_iou", "parse_box", "parse_boxes", "valid_rows", "ious",
           "center_inside", "mean_train_box", "mcnemar_z", "AUDIO_BLIND_SIGMAS"]

#: The supervised arm must beat the shuffled arm by more than this many McNemar σ.
AUDIO_BLIND_SIGMAS = 3.0
_EPS = 1e-12


def parse_box(cell: str) -> tuple[float, float, float, float] | None:
    """``x1|y1|x2|y2`` → ``(cx, cy, w, h)``; ``None`` when degenerate (``x2 ≤ x1 or y2 ≤ y1``) after the
    ``max(x, 0)`` sanitization."""
    parts = str(cell).strip().split("|")
    if len(parts) != 4:
        raise ValueError(f"box cell must be 'x1|y1|x2|y2', got {cell!r}")
    x1, y1, x2, y2 = (max(float(p), 0.0) for p in parts)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def parse_boxes(cells: Iterable[str]) -> tuple[np.ndarray, int]:
    """All rows → ``float32 [n, 4]`` targets with NaN rows for degenerate boxes, plus their count."""
    rows: list[tuple[float, float, float, float]] = []
    excluded = 0
    nan_row = (math.nan,) * 4
    for cell in cells:
        box = parse_box(cell)
        if box is None:
            excluded += 1
            rows.append(nan_row)
        else:
            rows.append(box)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4), excluded


def _xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _boxes(x: Any, what: str) -> np.ndarray:
    b = as_array(x, np.float64)
    if b.ndim == 1 and b.shape[0] == 4:
        b = b[None, :]
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"{what} must be [n, 4] (cx, cy, w, h), got {b.shape}")
    return b


def box_iou(pred, gt):
    """Row-aligned IoU of ``(cx, cy, w, h)`` boxes ``[n, 4]`` → ``float64 [n]`` (zero for an empty union)."""
    p = _xyxy(_boxes(pred, "pred"))
    g = _xyxy(_boxes(gt, "gt"))
    if p.shape != g.shape:
        raise ValueError(f"pred {p.shape} and gt {g.shape} differ")
    area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    area_g = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
    iw = np.clip(np.minimum(p[:, 2], g[:, 2]) - np.maximum(p[:, 0], g[:, 0]), 0, None)
    ih = np.clip(np.minimum(p[:, 3], g[:, 3]) - np.maximum(p[:, 1], g[:, 1]), 0, None)
    inter = iw * ih
    union = area_p + area_g - inter
    return np.where(union > 0, inter / np.maximum(union, _EPS), 0.0)


def valid_rows(targets: Any) -> np.ndarray:
    """``bool [n]``: target rows with four finite coordinates (degenerate rows carry NaN)."""
    return np.isfinite(_boxes(targets, "targets")).all(axis=1)


def _scored(outputs: Any, targets: Any) -> tuple[np.ndarray, np.ndarray]:
    t = _boxes(targets, "targets")
    p = _boxes(outputs, "outputs")
    if p.shape[0] == 1 and t.shape[0] > 1:
        p = np.broadcast_to(p, t.shape)
    if p.shape != t.shape:
        raise ValueError(f"outputs {p.shape} do not match targets {t.shape}")
    v = valid_rows(t)
    if not v.any():
        raise ValueError("box metric: no valid (non-degenerate) target box to score")
    return p[v], t[v]


def ious(outputs: Any, targets: Any) -> np.ndarray:
    """Per scored row IoU ``float64 [n_valid]`` (shared with the intervals)."""
    p, t = _scored(outputs, targets)
    return box_iou(p, t)


def center_inside(outputs: Any, targets: Any) -> np.ndarray:
    """``bool [n_valid]``: predicted centre inside the GT box (inclusive edges)."""
    p, t = _scored(outputs, targets)
    g = _xyxy(t)
    cx, cy = p[:, 0], p[:, 1]
    return (cx >= g[:, 0]) & (cx <= g[:, 2]) & (cy >= g[:, 1]) & (cy <= g[:, 3])


def iou_mean(outputs, targets, protocol, split_meta) -> float:
    return float_or_raise(ious(outputs, targets).mean(), "iou_mean")


def acc_at_05(outputs, targets, protocol, split_meta) -> float:
    """``acc@0.5``: fraction of scored rows with IoU ≥ 0.5."""
    return float((ious(outputs, targets) >= 0.5).mean())


def acc_at_03(outputs, targets, protocol, split_meta) -> float:
    """``acc@0.3``: fraction of scored rows with IoU ≥ 0.3."""
    return float((ious(outputs, targets) >= 0.3).mean())


def center_in_gt(outputs, targets, protocol, split_meta) -> float:
    """Separates "points at the right object but sizes it badly" from "points elsewhere"."""
    return float(center_inside(outputs, targets).mean())


def n_excluded_degenerate(outputs, targets, protocol, split_meta) -> float:
    """Disclosure: degenerate eval boxes excluded (``split_meta["n_excluded_degenerate"]`` when the
    driver dropped them while parsing, else the NaN rows still present in ``targets``)."""
    n = meta(split_meta, "n_excluded_degenerate")
    if n is None:
        n = int((~valid_rows(targets)).sum())
    return float(int(n))


def mean_train_box(train_targets: Any) -> np.ndarray:
    """Coordinate-wise mean of the valid TRAIN boxes ``float64 [4]`` — the constant predictor."""
    t = _boxes(train_targets, "train_targets")
    v = valid_rows(t)
    if not v.any():
        raise ValueError("mean_train_box: no valid train box")
    return t[v].mean(axis=0)


def iou_floor(outputs, targets, protocol, split_meta) -> float:
    """``iou_mean`` of the mean train box (VGG-SS ≈ 0.38–0.44, a strong floor)."""
    if outputs is None:
        outputs = mean_train_box(require_meta(split_meta, "train_targets", "floors"))[None, :]
    return iou_mean(outputs, targets, protocol, split_meta)


def _shuffled_outputs(split_meta: Any) -> Any:
    sh = require_meta(split_meta, "shuffled", "shuffled control")
    return sh["outputs"] if isinstance(sh, dict) and "outputs" in sh else sh


def iou_shuffled(outputs, targets, protocol, split_meta) -> float:
    """``iou_mean`` of the head trained with the conditioning stream deranged (``split_meta["shuffled"]``)."""
    return iou_mean(_shuffled_outputs(split_meta), targets, protocol, split_meta)


def mcnemar_z(success_a: Any, success_b: Any) -> float:
    """McNemar ``z = (b − c) / sqrt(b + c)`` over paired successes (``b``: a hits, b misses; ``c``: the
    reverse); 0 when no clip is discordant. The paired test is what detects a 1–2 pt gain between two
    arms whose per-clip successes correlate ≈ 0.97."""
    a = as_array(success_a).astype(bool).ravel()
    b_ = as_array(success_b).astype(bool).ravel()
    if a.shape != b_.shape:
        raise ValueError(f"paired arms must cover the same clips, got {a.shape} vs {b_.shape}")
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    return (b - c) / math.sqrt(b + c) if (b + c) else 0.0


def audio_blind(outputs, targets, protocol, split_meta) -> bool:
    """``not (McNemar z over paired acc@0.5 successes > 3)``: True when the shuffled arm is
    not significantly worse, i.e. the box is explained by the visual stream alone. Two indistinguishable
    arms (``z = 0``) are flagged, a control that beats the headline a fortiori."""
    sup = ious(outputs, targets) >= 0.5
    shuf = ious(_shuffled_outputs(split_meta), targets) >= 0.5
    return not (mcnemar_z(sup, shuf) > AUDIO_BLIND_SIGMAS)
