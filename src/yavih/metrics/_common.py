"""Private helpers shared by the metric modules (registry signature
``fn(outputs, targets, protocol, split_meta)``). numpy + stdlib only.

``split_meta`` is the driver-supplied side channel. The keys the metric
functions read, all optional unless a function says otherwise:

- ``n_train``, ``n_eval``: row counts (disclosures fall back to the targets' length).
- ``train_targets``: TRAIN targets in the readout's target shape; floors derive their constant
  prediction from it when ``outputs`` is ``None`` (floors are computed from TRAIN
  targets only). When ``outputs`` is given (``probes.floors.floor_outputs``) it is scored as is.
- ``folds``: ``{fold: {"outputs", "targets"}}`` in ``splits.folds`` order (clf folds).
- ``floor``: ``{"outputs", "targets"}`` of the floor role (avsync).
- ``shuffled``: ``{"outputs"}`` of the shuffled-conditioning control head (box).
- ``sets``: ``{set_name: targets}`` for ``multi_sets`` when ``targets`` is not already a mapping.
- ``event_start``, ``event_end`` (or ``boundary``): per eval row, for ``seg_acc_boundary``.
- ``chance``: ``benchmark.yaml[metrics.chance]`` (``floor_violated`` reads ``chance["acc"]``).
- ``bootstrap``: ``benchmark.yaml[metrics.bootstrap]`` = ``{"samples", "seed"}`` (intervals).
- ``n_excluded_degenerate``: the count the driver dropped while parsing boxes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    "as_array",
    "predictions",
    "meta",
    "require_meta",
    "set_targets",
    "count_rows",
    "float_or_raise",
]


def as_array(x: Any, dtype: Any = None) -> np.ndarray:
    """``np.asarray`` that also accepts torch tensors (``.detach().cpu().numpy()``) and memmaps
    without importing torch (import boundary)."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def predictions(outputs: Any, *, ndim_scores: int) -> np.ndarray:
    """Class predictions from raw head outputs: scores ``[..., C]`` (``ndim_scores`` axes) are
    argmaxed over the last axis, already-integer predictions (one axis fewer) pass through. Lets a
    floor hand a constant one-hot or a constant index vector to the same metric function."""
    arr = as_array(outputs)
    if arr.ndim == ndim_scores:
        return np.argmax(arr, axis=-1).astype(np.int64)
    if arr.ndim == ndim_scores - 1:
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"predictions must be integer class indices, got dtype {arr.dtype}"
            )
        return arr.astype(np.int64)
    raise ValueError(
        f"outputs must have {ndim_scores} (scores) or {ndim_scores - 1} (indices) axes, got shape {arr.shape}"
    )


def meta(split_meta: Any, key: str, default: Any = None) -> Any:
    """``split_meta[key]`` for a mapping (or attribute for an object), else ``default``."""
    if split_meta is None:
        return default
    if isinstance(split_meta, Mapping):
        return split_meta.get(key, default)
    return getattr(split_meta, key, default)


def require_meta(split_meta: Any, key: str, section: str) -> Any:
    val = meta(split_meta, key)
    if val is None:
        raise ValueError(f"split_meta[{key!r}] is required by this metric ({section})")
    return val


def set_targets(targets: Any, split_meta: Any, set_name: str) -> np.ndarray:
    """The ``[n, C]`` target matrix of one ``label.sets`` entry (``multi_sets``): ``targets``
    may be ``{set_name: matrix}``, or the matrix itself, else ``split_meta["sets"][set_name]``."""
    if isinstance(targets, Mapping):
        if set_name not in targets:
            raise KeyError(f"targets has no label set {set_name!r} (have {sorted(targets)})")
        return as_array(targets[set_name])
    sets = meta(split_meta, "sets")
    if isinstance(sets, Mapping) and set_name in sets:
        return as_array(sets[set_name])
    if targets is None:
        raise ValueError(f"no targets for label set {set_name!r} (multi_sets)")
    return as_array(targets)


def count_rows(targets: Any, split_meta: Any, key: str = "n_eval") -> int:
    """Row count of ``targets`` (first axis; a per-set mapping counts its first matrix), else
    the sum over ``split_meta["folds"]``, else ``split_meta[key]``."""
    if isinstance(targets, Mapping):
        for v in targets.values():
            return int(as_array(v).shape[0])
    elif targets is not None:
        return int(as_array(targets).shape[0])
    folds = meta(split_meta, "folds")
    if isinstance(folds, Mapping) and folds:
        return int(sum(as_array(f["targets"]).shape[0] for f in folds.values()))
    val = meta(split_meta, key)
    if val is None:
        raise ValueError(f"cannot count rows: no targets and no split_meta[{key!r}]")
    return int(val)


def float_or_raise(value: Any, what: str) -> float:
    """Metrics are JSON numbers (``allow_nan=False``): never return NaN/inf."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{what} is not finite ({out}); the population is empty or degenerate")
    return out
