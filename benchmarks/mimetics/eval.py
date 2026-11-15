"""``mimetics`` scoring: the reference's top-1 / top-5 and the paper's class-averaged metrics.

The headline ``acc`` is the core ``clf`` accuracy (argmax == label over every scored clip), which is the Top-1 of
the reference implementation below. This file adds:

- ``top5`` — the reference's Top-5: the label is among the five highest-scored classes, over every scored clip.
- ``mean_top1``, ``mean_top5``, ``mean_ap`` — the metrics of the Mimetics paper (Weinzaepfel & Rogez, IJCV 2021,
  Tables 6 and 9: "mean top-k accuracies and mean average-precision"). No code was released for them, so they are
  written from the paper's text (grounding rung 4): top-k accuracy per class, averaged over the classes that have a
  scored clip; and, "as each video has a single label, average-precision computes for each class the inverse of the
  rank of the ground-truth label, averaged over all videos of this class", averaged over those classes. The rank of
  the label is ``1 +`` the number of classes scored strictly higher (ties in the label's favour; the paper does not
  say).

Reference (grounding rung 3; no official evaluation code exists): IntegralAction (Moon et al., CVPR Workshops 2021),
https://github.com/mks0601/IntegralAction_RELEASE at commit 50bc02ae8ad30d55628d6aecbd46f2dfa6173117,
``data/Mimetics/Mimetics.py`` ``Mimetics.evaluate`` (licence: none stated in the repository). The measuring loop of
``evaluate`` is copied in :func:`integralaction_topk` below. Changes: the per-video dicts become arrays (one row per
clip, a yavih feature set has one output per clip, so the reference's averaging of ``cfg.test_infer_num`` = 10
temporal views per video is the bridge's business, and a yavih bridge declares one view); the counts are returned
as fractions in [0, 1] instead of printed as percentages; the result file is not written; the per-clip success
vectors are exposed for the binomial intervals. The reference also skips at load time every video with
``frame_num <= (frame_per_seg - 1) * frame_interval`` = 56 frames (its own 8 x 8 frame sampling, cfg); yavih scores
every staged clip, see ``metrics.notes``.

Module scope imports the standard library, numpy and yavih core only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.metrics._common import as_array
from yavih.metrics.classification import _single_targets
from yavih.metrics.registry import MetricSpec

__all__ = [
    "METRICS", "integralaction_topk", "top5", "mean_top1", "mean_top5", "mean_ap", "label_rank",
    "per_class_mean",
]


# ----------------------------------------------------------------------------- copied: IntegralAction evaluate()
# Copied from IntegralAction_RELEASE (https://github.com/mks0601/IntegralAction_RELEASE), commit
# 50bc02ae8ad30d55628d6aecbd46f2dfa6173117, data/Mimetics/Mimetics.py, Mimetics.evaluate, lines "# measure accuracy"
# to the two prints. Licence: none stated (the repository has no LICENSE file). Changes: `action_label_gt` /
# `action_label_out` (dicts keyed by video id, outputs already averaged over views) become `gts` [n] and `scores`
# [n, C]; the success of each clip is kept (`hit1`, `hit5`); fractions are returned instead of printed percentages.


def integralaction_topk(scores: Any, gts: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-clip Top-1 and Top-5 success (``bool [n]`` each), exactly as ``Mimetics.evaluate`` decides them."""
    action_label_out = as_array(scores)
    action_label_gt = _single_targets(gts)
    if action_label_out.ndim != 2 or action_label_out.shape[0] != action_label_gt.shape[0]:
        raise ValueError(f"top-k needs scores [n, C] for {action_label_gt.shape[0]} rows, got shape {action_label_out.shape}")
    sample_num = len(action_label_gt)
    hit1 = np.zeros(sample_num, dtype=bool)
    hit5 = np.zeros(sample_num, dtype=bool)
    for k in range(sample_num):
        if action_label_gt[k] == np.argmax(action_label_out[k]):
            hit1[k] = True
        if action_label_gt[k] in np.argsort(action_label_out[k])[-5:][::-1]:
            hit5[k] = True
    return hit1, hit5

# ----------------------------------------------------------------------------- end of the copied code


def _scores(outputs: Any, targets: Any) -> tuple[np.ndarray, np.ndarray]:
    o = as_array(outputs, np.float64)
    t = _single_targets(targets)
    if o.ndim != 2 or o.shape[0] != t.shape[0]:
        raise ValueError(f"mimetics metrics need raw scores [n, C] for {t.shape[0]} rows, got shape {o.shape}")
    if t.size == 0:
        raise ValueError("mimetics metrics: empty eval population")
    return o, t


def top5(outputs, targets, protocol, split_meta) -> float:
    """IntegralAction's Top-5 accuracy over every scored clip."""
    _, hit5 = integralaction_topk(*_scores(outputs, targets))
    return float(hit5.mean())


def label_rank(outputs: Any, targets: Any) -> np.ndarray:
    """``int64 [n]``: 1 + the number of classes scored strictly higher than the label."""
    o, t = _scores(outputs, targets)
    gt = o[np.arange(t.size), t]
    return 1 + np.sum(o > gt[:, None], axis=1).astype(np.int64)


def per_class_mean(values: np.ndarray, targets: Any) -> float:
    """Mean over the classes present in ``targets`` of the per-class mean of ``values``."""
    t = _single_targets(targets)
    return float(np.mean([values[t == c].mean() for c in np.unique(t)]))


def mean_top1(outputs, targets, protocol, split_meta) -> float:
    """Top-1 accuracy per class, averaged over the classes with a scored clip (the paper's "mean top-1")."""
    return per_class_mean((label_rank(outputs, targets) <= 1).astype(np.float64), targets)


def mean_top5(outputs, targets, protocol, split_meta) -> float:
    """Top-5 accuracy per class, averaged over the classes with a scored clip (the paper's "mean top-5")."""
    return per_class_mean((label_rank(outputs, targets) <= 5).astype(np.float64), targets)


def mean_ap(outputs, targets, protocol, split_meta) -> float:
    """The paper's mAP: per class the mean of 1 / rank of the label over its clips, averaged over the classes."""
    return per_class_mean(1.0 / label_rank(outputs, targets), targets)


def _pc_top5(outputs, targets, protocol, split_meta) -> np.ndarray:
    return integralaction_topk(*_scores(outputs, targets))[1].astype(np.float64)


_CLF = frozenset({"clf"})
_BINARY_CI = frozenset({"binomial", "bootstrap_clip", "pooled_binomial"})

#: Overlays ``yavih.metrics.REGISTRY`` for this benchmark.
METRICS: dict[str, MetricSpec] = {
    "top5": MetricSpec(name="top5", fn=top5, readouts=_CLF, per_clip=_pc_top5, ci=_BINARY_CI,
                       description="IntegralAction Mimetics.evaluate Top-5: label among the 5 highest scores, over all clips"),
    "mean_top1": MetricSpec(name="mean_top1", fn=mean_top1, readouts=_CLF,
                            description="Mimetics paper (Table 6): top-1 accuracy per class, averaged over classes"),
    "mean_top5": MetricSpec(name="mean_top5", fn=mean_top5, readouts=_CLF,
                            description="Mimetics paper (Table 6): top-5 accuracy per class, averaged over classes"),
    "mean_ap": MetricSpec(name="mean_ap", fn=mean_ap, readouts=_CLF,
                          description="Mimetics paper (Table 6): per class the mean inverse rank of the label, averaged over classes"),
}
