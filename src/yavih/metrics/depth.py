"""grid (depth) metrics: per image then mean, ``pred * valid`` first, validity
``0 < d ≤ max_depth``: ``d1``, ``d2``, ``d3``, ``absrel``, ``rmse``; ``d1_floor`` from the per-pixel mean train
log-depth map. As in Probe3D, zeroing the prediction at
invalid pixels BEFORE forming the ratio makes an invalid pixel blow up instead of contributing a
plausible value, and the mask then removes it from the denominator; each image's number is a mean
over its own valid pixels and the reported number the mean over scored images (never one pool over
all pixels, which would weight densely-labelled images more). Outputs and targets are depth in
metres, ``[n, S, S]`` (a singleton channel axis is squeezed)."""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.metrics._common import as_array, float_or_raise, require_meta

__all__ = ["d1", "d2", "d3", "absrel", "rmse", "d1_floor", "depth_valid_mask", "depth_per_image",
           "mean_log_depth_map"]

_EPS = 1e-8


def depth_valid_mask(targets, protocol):
    """``0 < d ≤ protocol["max_depth"]`` (and finite) — invalid pixels are stored as 0 in the targets."""
    t = as_array(targets, np.float64)
    return np.isfinite(t) & (t > 0) & (t <= float(protocol["max_depth"]))


def _squeeze(a: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if a.shape == target_shape:
        return a
    if a.ndim == len(target_shape) + 1:
        for axis in (1, a.ndim - 1):
            if a.shape[axis] == 1 and np.squeeze(a, axis=axis).shape == target_shape:
                return np.squeeze(a, axis=axis)
    raise ValueError(f"depth outputs {a.shape} do not match targets {target_shape}")


def depth_per_image(outputs: Any, targets: Any, protocol: Any) -> dict[str, np.ndarray]:
    """Per-image ``d1, d2, d3, absrel, rmse`` (``float64 [n]``, NaN for images without a valid pixel) and
    the ``scored`` mask — the quantities the scalar metrics average and ``bootstrap_clip`` resamples."""
    t = as_array(targets, np.float64)
    if t.ndim != 3:
        raise ValueError(f"depth targets must be [n, S, S], got {t.shape}")
    size = protocol.get("target_size") if hasattr(protocol, "get") else None
    if size is not None and t.shape[1:] != (int(size), int(size)):
        raise ValueError(f"depth targets are {t.shape[1:]} but protocol.target_size = {size}")
    p = _squeeze(as_array(outputs, np.float64), t.shape)
    n = t.shape[0]
    valid = depth_valid_mask(t, protocol).reshape(n, -1)
    gt = t.reshape(n, -1)
    pr = p.reshape(n, -1) * valid  # Probe3D: zero the prediction at invalid pixels FIRST
    count = valid.sum(axis=1).astype(np.float64)
    scored = count > 0
    safe_gt = np.where(valid, gt, 1.0)
    safe_pr = np.where(valid, pr, 1.0)
    ratio = np.maximum(safe_gt / np.maximum(safe_pr, _EPS), safe_pr / np.maximum(safe_gt, _EPS))

    def per_image(values: np.ndarray) -> np.ndarray:
        s = (values * valid).sum(axis=1) / np.maximum(count, 1.0)
        return np.where(scored, s, np.nan)

    out: dict[str, np.ndarray] = {f"d{k}": per_image((ratio < 1.25**k).astype(np.float64)) for k in (1, 2, 3)}
    out["absrel"] = per_image(np.abs(safe_pr - safe_gt) / np.maximum(safe_gt, _EPS))
    out["rmse"] = np.sqrt(per_image((safe_pr - safe_gt) ** 2))
    out["scored"] = scored
    return out


def _mean_over_scored(outputs: Any, targets: Any, protocol: Any, key: str) -> float:
    per = depth_per_image(outputs, targets, protocol)
    if not per["scored"].any():
        raise ValueError(f"{key}: no valid pixel in any image")
    return float_or_raise(np.nanmean(per[key]), key)


def d1(outputs, targets, protocol, split_meta) -> float:
    """Fraction of valid pixels with ``max(gt/pred, pred/gt) < 1.25`` per image, averaged."""
    return _mean_over_scored(outputs, targets, protocol, "d1")


def d2(outputs, targets, protocol, split_meta) -> float:
    return _mean_over_scored(outputs, targets, protocol, "d2")


def d3(outputs, targets, protocol, split_meta) -> float:
    return _mean_over_scored(outputs, targets, protocol, "d3")


def absrel(outputs, targets, protocol, split_meta) -> float:
    """Mean ``|pred − gt| / gt`` over valid pixels per image, averaged."""
    return _mean_over_scored(outputs, targets, protocol, "absrel")


def rmse(outputs, targets, protocol, split_meta) -> float:
    """Per-image RMSE over valid pixels, averaged over images."""
    return _mean_over_scored(outputs, targets, protocol, "rmse")


def mean_log_depth_map(train_targets: Any, protocol: Any, chunk: int = 512) -> np.ndarray:
    """Per-pixel mean TRAIN log-depth over the images where the pixel is valid (one streaming pass
    over the memmap in ``chunk`` slices, float64 accumulators); a pixel valid in no image falls back
    to the global mean so the floor map never carries a NaN.
    Returns the map in metres (``exp`` of the mean log-depth), ``float64 [S, S]``."""
    # A memmap / tensor is sliced in place (never materialized whole); anything else is converted once.
    src = train_targets if hasattr(train_targets, "shape") else as_array(train_targets)
    n = int(src.shape[0])
    if n == 0:
        raise ValueError("mean_log_depth_map: no train targets")
    total: np.ndarray | None = None
    count: np.ndarray | None = None
    for i in range(0, n, chunk):
        t = as_array(src[i : i + chunk], np.float64)
        valid = depth_valid_mask(t, protocol)
        logs = np.where(valid, np.log(np.where(valid, t, 1.0)), 0.0)
        if total is None:
            total = np.zeros(t.shape[1:], dtype=np.float64)
            count = np.zeros(t.shape[1:], dtype=np.float64)
        total += logs.sum(axis=0)
        count += valid.sum(axis=0)
    assert total is not None and count is not None
    if count.sum() == 0:
        raise ValueError("mean_log_depth_map: no valid train pixel")
    global_mean = total.sum() / count.sum()
    mean_log = np.where(count > 0, total / np.maximum(count, 1.0), global_mean)
    return np.exp(mean_log)


def d1_floor(outputs, targets, protocol, split_meta) -> float:
    """``d1`` of the per-pixel mean train log-depth map (NYUv2 ≈ 0.3–0.4). ``outputs`` is the
    constant map (broadcast ``[n, S, S]`` or ``[S, S]``) when the driver provides it, else it is computed
    from ``split_meta["train_targets"]``."""
    t = as_array(targets, np.float64)
    if outputs is None:
        floor_map = mean_log_depth_map(require_meta(split_meta, "train_targets", "floors"), protocol)
    else:
        floor_map = as_array(outputs, np.float64)
    if floor_map.ndim == 2:
        floor_map = np.broadcast_to(floor_map, t.shape)
    return d1(floor_map, t, protocol, split_meta)
