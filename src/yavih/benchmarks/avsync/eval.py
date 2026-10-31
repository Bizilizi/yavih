"""``avsync`` probe hooks: the offset metrics and the floor-arm controls.

Only this benchmark scores synchronization offsets, so its metrics live here, not in ``yavih.metrics``:
``acc_tol1``, ``mae_seconds``, ``acc_coarse5`` / ``n_coarse5``, and the floor-role controls ``acc_floor_arm`` /
``floor_violated``. ``METRICS`` overlays the registry for this benchmark; each spec carries its per-clip quantity
and the interval kinds it admits (``MetricSpec.per_clip`` / ``MetricSpec.ci``), so ``metrics.ci`` may name them.
``PROTOCOL_KEYS`` are the protocol facts no metric reads: ``nominal_window_start``
(= ``input_contract.materialize.pivot``) and the frozen ``curation`` record the floor control's provenance rests on.

Classes are the offset bins ``protocol["offsets_seconds"]`` (sorted ascending, ``C`` of them). The coarse grid
(``coarse_grid_seconds = g``) is the subset of bins whose offset is a multiple of ``g`` (``round(o / g) * g ==
o``); a mass-preserving collapse is used for the PREDICTION (``p_coarse(b) = p(b) + ½ p(b − 1) +
½ p(b + 1)`` over the non-grid neighbours, softmax first), and only rows whose TARGET lies on the grid are scored
— a half-step target sits exactly between two coarse bins and has no coarse answer.

Module scope imports stdlib, numpy and yavih core only. Its normalized AST (comments and docstrings excluded) enters
the scoring hash and so ``protocol_hash`` = H(input_hash, scoring_hash); an edit here is scoring-side and never
invalidates features.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from yavih.metrics._common import as_array, float_or_raise, meta, predictions, require_meta
from yavih.metrics.classification import _single_targets, acc, correct
from yavih.metrics.registry import MetricSpec

__all__ = [
    "METRICS", "PROTOCOL_KEYS",
    "acc_tol1", "mae_seconds", "acc_coarse5", "n_coarse5", "acc_floor_arm", "floor_violated",
    "offsets", "coarse_lookup", "coarse_probabilities", "coarse_mask", "coarse_correct", "tol1_correct",
    "floor_threshold",
]

PROTOCOL_KEYS = frozenset({"nominal_window_start", "curation"})

_GRID_TOL = 1e-6


def offsets(protocol: Any) -> np.ndarray:
    """``protocol["offsets_seconds"]`` as ``float64 [C]``, required ascending (bins are ordered)."""
    o = as_array(protocol["offsets_seconds"], np.float64)
    if o.ndim != 1 or o.size < 2:
        raise ValueError("offsets_seconds must be a list of ≥ 2 offsets")
    if np.any(np.diff(o) <= 0):
        raise ValueError("offsets_seconds must be strictly ascending")
    return o


def _check_bins(preds: np.ndarray, t: np.ndarray, grid: np.ndarray) -> None:
    if preds.shape != t.shape:
        raise ValueError(f"outputs rows {preds.shape} do not match targets {t.shape}")
    if t.size == 0:
        raise ValueError("avsync metric: empty eval population")
    if np.any(t < 0) or np.any(t >= grid.size) or np.any(preds >= grid.size):
        raise ValueError(f"bin indices must lie in [0, {grid.size}) (len(offsets_seconds))")


def _scores_match(outputs: Any, grid: np.ndarray) -> None:
    o = as_array(outputs)
    if o.ndim == 2 and o.shape[1] != grid.size:
        raise ValueError(
            f"outputs have {o.shape[1]} bins but offsets_seconds has {grid.size} (never score a truncated grid)"
        )


def tol1_correct(outputs: Any, targets: Any) -> np.ndarray:
    """``bool [n]``: ``|argmax − target| ≤ 1`` in class index."""
    preds = predictions(outputs, ndim_scores=2)
    t = _single_targets(targets)
    if preds.shape != t.shape:
        raise ValueError(f"outputs rows {preds.shape} do not match targets {t.shape}")
    return np.abs(preds - t) <= 1


def acc_tol1(outputs, targets, protocol, split_meta) -> float:
    """``|argmax − target| ≤ 1`` in class index (chance ``(3C − 2) / C²``)."""
    _scores_match(outputs, offsets(protocol))
    c = tol1_correct(outputs, targets)
    if c.size == 0:
        raise ValueError("acc_tol1: empty eval population")
    return float(c.mean())


def mae_seconds(outputs, targets, protocol, split_meta) -> float:
    """mean ``|offsets_seconds[argmax] − offsets_seconds[target]|``."""
    grid = offsets(protocol)
    _scores_match(outputs, grid)
    preds = predictions(outputs, ndim_scores=2)
    t = _single_targets(targets)
    _check_bins(preds, t, grid)
    return float(np.mean(np.abs(grid[preds] - grid[t])))


def coarse_lookup(protocol: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(coarse_bins, column)``: the fine bins on the ``coarse_grid_seconds`` grid (ascending) and,
    per fine bin, its column in the collapsed vector or ``-1``."""
    grid = offsets(protocol)
    g = float(protocol["coarse_grid_seconds"])
    if not g > 0:
        raise ValueError("coarse_grid_seconds must be > 0")
    q = grid / g
    on_grid = np.abs(q - np.round(q)) < _GRID_TOL
    coarse = np.flatnonzero(on_grid)
    if coarse.size == 0:
        raise ValueError("no offset lies on the coarse_grid_seconds grid")
    column = np.full(grid.size, -1, dtype=np.int64)
    column[coarse] = np.arange(coarse.size)
    return coarse, column


def coarse_probabilities(outputs: Any, protocol: Any) -> np.ndarray:
    """Softmax over the fine bins collapsed onto the coarse grid, ``[n, n_coarse]``: every grid bin
    keeps its own mass and takes half the mass of each adjacent non-grid bin (mass-preserving on the
    pinned 0.5 s grid)."""
    o = as_array(outputs, np.float64)
    coarse, column = coarse_lookup(protocol)
    if o.ndim != 2 or o.shape[1] != column.size:
        raise ValueError(f"acc_coarse5 needs raw scores [n, {column.size}], got shape {o.shape}")
    z = o - o.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    cols = []
    for b in coarse:
        col = p[:, b].copy()
        for nb in (b - 1, b + 1):
            if 0 <= nb < column.size and column[nb] < 0:
                col += 0.5 * p[:, nb]
        cols.append(col)
    return np.stack(cols, axis=1)


def coarse_mask(targets: Any, protocol: Any) -> np.ndarray:
    """``bool [n]``: rows whose TRUE offset lies on the coarse grid (the scored population)."""
    _, column = coarse_lookup(protocol)
    t = _single_targets(targets)
    if np.any(t < 0) or np.any(t >= column.size):
        raise ValueError(f"bin indices must lie in [0, {column.size})")
    return column[t] >= 0


def coarse_correct(outputs: Any, targets: Any, protocol: Any) -> np.ndarray:
    """``bool [n_scored]`` over the grid-target rows: collapsed argmax == target's coarse column."""
    _, column = coarse_lookup(protocol)
    t = _single_targets(targets)
    mask = coarse_mask(t, protocol)
    pred_col = np.argmax(coarse_probabilities(outputs, protocol), axis=1)
    return pred_col[mask] == column[t][mask]


def acc_coarse5(outputs, targets, protocol, split_meta) -> float:
    """Accuracy on the ``coarse_grid_seconds`` grid, scored only on rows whose target lies on it
    (chance ``1 / n_coarse``). Raises when no row has a grid target."""
    c = coarse_correct(outputs, targets, protocol)
    if c.size == 0:
        raise ValueError("acc_coarse5: no evaluated row has a target on the coarse grid")
    return float(c.mean())


def n_coarse5(outputs, targets, protocol, split_meta) -> float:
    """Disclosure: the number of rows scored by ``acc_coarse5``."""
    return float(int(coarse_mask(targets, protocol).sum()))


# ----------------------------------------------------------------------------- floor role (control)


def _floor_arm(split_meta: Any) -> tuple[Any, np.ndarray]:
    fl = require_meta(split_meta, "floor", "floor role")
    return fl["outputs"], _single_targets(fl["targets"])


def acc_floor_arm(outputs, targets, protocol, split_meta) -> float:
    """The trained probe's exact-bin accuracy on the floor rows (``split_meta["floor"]``)."""
    fo, ft = _floor_arm(split_meta)
    _scores_match(fo, offsets(protocol))
    return acc(fo, ft, protocol, split_meta)


def floor_threshold(chance: float, n_floor: int, sigmas: float) -> float:
    """``chance + sigmas · sqrt(chance (1 − chance) / n_floor)`` (avsync: 0.1233 at n = 5 997, 3 σ)."""
    if n_floor < 1:
        raise ValueError("floor population must have ≥ 1 row")
    if not 0.0 < chance < 1.0:
        raise ValueError("chance must lie in (0, 1)")
    return chance + sigmas * math.sqrt(chance * (1.0 - chance) / n_floor)


def floor_violated(outputs, targets, protocol, split_meta) -> bool:
    """``acc_floor_arm > chance + floor_sigmas · sqrt(chance(1−chance)/n_floor)``. ``chance``
    is ``metrics.chance["acc"]`` when the driver passes it (``split_meta["chance"]``), else ``1 / C``.
    True voids the headline (README.md "### Protocol in yavih")."""
    fo, ft = _floor_arm(split_meta)
    grid = offsets(protocol)
    chance_map = meta(split_meta, "chance") or {}
    chance = float(chance_map.get("acc", 1.0 / grid.size))
    thr = floor_threshold(chance, int(ft.size), float(protocol["floor_sigmas"]))
    arm = float_or_raise(acc(fo, ft, protocol, split_meta), "acc_floor_arm")
    return bool(arm > thr)


# ----------------------------------------------------------------------------- the overlay


def _f64(mask: np.ndarray) -> np.ndarray:
    return as_array(mask).astype(np.float64)


def _pc_tol1(outputs, targets, protocol, split_meta) -> np.ndarray:
    return _f64(tol1_correct(outputs, targets))


def _pc_coarse5(outputs, targets, protocol, split_meta) -> np.ndarray:
    return _f64(coarse_correct(outputs, targets, protocol))


def _pc_floor_arm(outputs, targets, protocol, split_meta) -> np.ndarray:
    return _f64(correct(split_meta["floor"]["outputs"], split_meta["floor"]["targets"]))


_OFF = frozenset({"offsets_seconds"})
_COARSE = frozenset({"offsets_seconds", "coarse_grid_seconds"})
_FLOOR = frozenset({"floor_sigmas"})
_BINARY_CI = frozenset({"binomial", "bootstrap_clip", "pooled_binomial"})
_CLF = frozenset({"clf"})

#: Overlays ``yavih.metrics.REGISTRY`` for this benchmark.
METRICS: dict[str, MetricSpec] = {
    "acc_tol1": MetricSpec(name="acc_tol1", fn=acc_tol1, readouts=_CLF, protocol_keys=_OFF, per_clip=_pc_tol1, ci=_BINARY_CI),
    "mae_seconds": MetricSpec(name="mae_seconds", fn=mae_seconds, readouts=_CLF, protocol_keys=_OFF, higher_is_better=False),
    "acc_coarse5": MetricSpec(name="acc_coarse5", fn=acc_coarse5, readouts=_CLF, protocol_keys=_COARSE, per_clip=_pc_coarse5, ci=_BINARY_CI),
    "n_coarse5": MetricSpec(name="n_coarse5", fn=n_coarse5, readouts=_CLF, protocol_keys=_COARSE, kind="disclosure", higher_is_better=None),
    "acc_floor_arm": MetricSpec(name="acc_floor_arm", fn=acc_floor_arm, readouts=_CLF, protocol_keys=_FLOOR, kind="control",
                                higher_is_better=None, per_clip=_pc_floor_arm, ci=_BINARY_CI),
    "floor_violated": MetricSpec(name="floor_violated", fn=floor_violated, readouts=_CLF, protocol_keys=_FLOOR, kind="control",
                                 higher_is_better=None),
}
