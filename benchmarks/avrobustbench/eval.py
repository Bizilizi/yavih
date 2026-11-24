"""``avrobustbench`` metrics: top-1 accuracy per condition and its aggregates over the 15 corruptions.

One probe is trained on clean VGGSound train clips; the eval split holds every VGGSOUND-2C clip under 16 conditions
(``protocol.conditions``: ``clean`` first, then the 15 corruptions at ``protocol.severity``). A row's condition is the
suffix of its clip id (``<clip>#<condition>``, the manifest's ``condition`` column) and its source clip the prefix;
the metrics read both from ``split_meta["eval_clip_ids"]`` and never from the outputs.

- ``acc_<condition>``: exact-argmax (top-1) accuracy over that condition's rows (``yavih.metrics.classification
  .correct``, the core's clf accuracy);
- ``macc`` (headline): the unweighted mean of the 15 corruptions' accuracies, the paper's "mean accuracy at severity
  5" (Table 4, VGGSOUND-2C; Maharana et al. 2025, Sec. 4); ``macc_digital`` / ``macc_environmental`` /
  ``macc_human``: the same mean over the paper's corruption groups (``protocol.corruption_groups``);
- ``alpha`` / ``rho``: absolute and relative robustness, ``1 - (A_clean - mAcc)`` and ``1 - (A_clean - mAcc) /
  A_clean`` in fractions (the paper's ``alpha = 1 - dA/100``, ``rho = 1 - dA/A_cl`` in percent). The formulas are
  copied from the implementation the paper cites for them, Schiappa et al. (NeurIPS 2022 D&B)
  ``robustness_scores.py`` ``get_robustness_scores`` (https://github.com/Maddy12/VideoLanguageModelRobustness @
  005fd427b7a393adbc050a8ac7f32a48f4388aa6, licence: none stated)::

      df[f'{ret}$\\gamma^a$'] = 1 - (df[f'{ret}_GT'] - df[ret])
      df[f'{ret}$\\gamma^r$'] = 1 - (df[f'{ret}_GT'] - df[ret]) / df[f'{ret}_GT']

  Changes: pandas dropped; applied once to the mean over corruptions (both are linear in the score, so this equals
  the mean of the per-corruption values, as Table 4 reports them); ``rho`` is 0.0 when ``A_clean`` is 0 (upstream
  divides by zero);
- ``macc_floor``: ``macc`` of the train-majority constant prediction (the clf floor outputs); ``n_clips``: the
  distinct source clips scored.

``macc`` declares ``bootstrap_clip`` over SOURCE clips (``[n_clips, 15]`` per-clip correctness): the 15 rows of a
clip share its content, so resampling rows would understate the interval. ``acc_<condition>`` declare ``binomial``.

Module scope imports the standard library, numpy and yavih core only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.metrics._common import float_or_raise, require_meta
from yavih.metrics.classification import correct
from yavih.metrics.registry import MetricSpec

__all__ = ["METRICS", "PROTOCOL_KEYS", "CORRUPTIONS", "conditions_of", "sources_of", "per_condition", "macc",
           "alpha", "rho", "macc_per_clip", "n_clips"]

#: the 15 corruptions of protocol.conditions (after ``clean``), in the paper's order
CORRUPTIONS = ("gaussian", "impulse", "shot", "speckle", "compression", "snow", "frost", "spatter", "wind", "rain",
               "underwater", "concert", "smoke", "crowd", "interference")
GROUPS = ("digital", "environmental", "human")
PROTOCOL_KEYS = frozenset({"conditions", "severity", "corruption_groups"})
_CLF = frozenset({"clf"})


def _corruptions(protocol: Any) -> list[str]:
    conds = list(protocol["conditions"])
    if not conds or conds[0] != "clean" or len(set(conds)) != len(conds):
        raise ValueError(f"protocol.conditions must start with 'clean' and be distinct, got {conds}")
    return conds[1:]


def conditions_of(split_meta: Any) -> np.ndarray:
    """The condition of every eval row: the clip id's suffix after the last ``#``."""
    ids = require_meta(split_meta, "eval_clip_ids", "avrobustbench")
    out = []
    for c in ids:
        head, sep, cond = str(c).rpartition("#")
        if not sep or not head:
            raise ValueError(f"eval clip id {c!r} carries no '#<condition>' suffix")
        out.append(cond)
    return np.asarray(out, dtype=object)


def sources_of(split_meta: Any) -> np.ndarray:
    """The source clip of every eval row: the clip id before the last ``#``."""
    ids = require_meta(split_meta, "eval_clip_ids", "avrobustbench")
    return np.asarray([str(c).rpartition("#")[0] for c in ids], dtype=object)


def _correct(outputs: Any, targets: Any, split_meta: Any) -> tuple[np.ndarray, np.ndarray]:
    c = correct(outputs, targets)
    conds = conditions_of(split_meta)
    if conds.shape[0] != c.shape[0]:
        raise ValueError(f"{c.shape[0]} scored rows but {conds.shape[0]} eval clip ids")
    return c, conds


def per_condition(outputs: Any, targets: Any, protocol: Any, split_meta: Any) -> dict[str, float]:
    """Top-1 accuracy per condition of ``protocol.conditions``; every condition must have a scored row."""
    c, conds = _correct(outputs, targets, split_meta)
    unknown = sorted(set(conds) - set(protocol["conditions"]))
    if unknown:
        raise ValueError(f"eval rows of conditions outside protocol.conditions: {unknown}")
    out = {}
    for cond in protocol["conditions"]:
        m = conds == cond
        if not m.any():
            raise ValueError(f"no scored eval row under condition {cond!r}")
        out[cond] = float(c[m].mean())
    return out


def _mean_corrupted(acc: dict[str, float], protocol: Any) -> float:
    return float(np.mean([acc[k] for k in _corruptions(protocol)]))


def macc(outputs, targets, protocol, split_meta) -> float:
    """Mean top-1 accuracy over the 15 corruptions (each condition weighted equally)."""
    return float_or_raise(_mean_corrupted(per_condition(outputs, targets, protocol, split_meta), protocol), "macc")


def _group(name: str):
    def fn(outputs, targets, protocol, split_meta) -> float:
        acc = per_condition(outputs, targets, protocol, split_meta)
        members = list(protocol["corruption_groups"][name])
        return float_or_raise(np.mean([acc[k] for k in members]), f"macc_{name}")

    fn.__name__ = f"macc_{name}"
    return fn


def _cond(cond: str):
    def fn(outputs, targets, protocol, split_meta) -> float:
        return per_condition(outputs, targets, protocol, split_meta)[cond]

    def pc(outputs, targets, protocol, split_meta) -> np.ndarray:
        c, conds = _correct(outputs, targets, split_meta)
        return np.where(conds == cond, c.astype(np.float64), np.nan)

    fn.__name__, pc.__name__ = f"acc_{cond}", f"acc_{cond}_per_clip"
    return fn, pc


def alpha(outputs, targets, protocol, split_meta) -> float:
    """Absolute robustness ``1 - (A_clean - mAcc)`` (Schiappa et al.'s gamma^a on the corruption mean)."""
    acc = per_condition(outputs, targets, protocol, split_meta)
    return float_or_raise(1 - (acc["clean"] - _mean_corrupted(acc, protocol)), "alpha")


def rho(outputs, targets, protocol, split_meta) -> float:
    """Relative robustness ``1 - (A_clean - mAcc) / A_clean`` (Schiappa et al.'s gamma^r); 0.0 when A_clean = 0."""
    acc = per_condition(outputs, targets, protocol, split_meta)
    if acc["clean"] == 0.0:
        return 0.0
    return float_or_raise(1 - (acc["clean"] - _mean_corrupted(acc, protocol)) / acc["clean"], "rho")


def macc_per_clip(outputs, targets, protocol, split_meta) -> np.ndarray:
    """``[n_clips, 15]``: per source clip, its correctness under each corruption (NaN where that row is absent); the
    pooled mean over clips equals ``macc`` when every clip has every corruption."""
    c, conds = _correct(outputs, targets, split_meta)
    corr = _corruptions(protocol)
    col = {k: j for j, k in enumerate(corr)}
    uniq, inv = np.unique(sources_of(split_meta).astype(str), return_inverse=True)
    out = np.full((uniq.shape[0], len(corr)), np.nan)
    for i, (cond, ok) in enumerate(zip(conds, c, strict=True)):
        j = col.get(cond)
        if j is not None:
            out[inv[i], j] = float(ok)
    return out[~np.all(np.isnan(out), axis=1)]


def n_clips(outputs, targets, protocol, split_meta) -> float:
    """Disclosure: distinct source clips among the scored eval rows."""
    return float(np.unique(sources_of(split_meta).astype(str)).shape[0])


_ACC = {k: _cond(k) for k in ("clean", *CORRUPTIONS)}
_KEYS = frozenset({"conditions"})

METRICS: dict[str, MetricSpec] = {
    "macc": MetricSpec(name="macc", fn=macc, readouts=_CLF, protocol_keys=_KEYS, per_clip=macc_per_clip,
                       ci=frozenset({"bootstrap_clip"}),
                       description="mean top-1 accuracy over the 15 corruptions at severity 5 (paper Table 4 mAcc)"),
    "alpha": MetricSpec(name="alpha", fn=alpha, readouts=_CLF, protocol_keys=_KEYS,
                        description="absolute robustness 1 - (acc_clean - macc) (Schiappa et al. gamma^a)"),
    "rho": MetricSpec(name="rho", fn=rho, readouts=_CLF, protocol_keys=_KEYS,
                      description="relative robustness 1 - (acc_clean - macc) / acc_clean (Schiappa et al. gamma^r)"),
    **{f"macc_{g}": MetricSpec(name=f"macc_{g}", fn=_group(g), readouts=_CLF,
                               protocol_keys=frozenset({"conditions", "corruption_groups"}),
                               description=f"mean top-1 accuracy over the paper's {g} corruptions")
       for g in GROUPS},
    **{f"acc_{k}": MetricSpec(name=f"acc_{k}", fn=fn, readouts=_CLF, protocol_keys=_KEYS, per_clip=pc,
                              ci=frozenset({"binomial", "bootstrap_clip"}),
                              description=f"top-1 accuracy under condition {k}")
       for k, (fn, pc) in _ACC.items()},
    "n_clips": MetricSpec(name="n_clips", fn=n_clips, readouts=_CLF, kind="disclosure", higher_is_better=None,
                          description="distinct VGGSound source clips among the scored eval rows"),
    "macc_floor": MetricSpec(name="macc_floor", fn=macc, readouts=_CLF, protocol_keys=_KEYS, kind="floor",
                             floor_of=("macc",), description="macc of the train-majority constant prediction"),
}
