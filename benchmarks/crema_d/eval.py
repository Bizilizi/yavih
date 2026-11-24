"""``crema_d``: HEAR's score of the task, on the core ``clf`` probe's five held-out folds.

The probe is the core ``clf`` readout under ``splits.folds`` (one probe per held-out fold, trained on the other
four). What turns its outputs into the number is HEAR's own code, copied into ``hear_score.py`` next to this file
(hear-eval-kit ``Top1Accuracy`` and ``aggregate_test_results``):

* ``top1_acc_mean`` (headline): HEAR's ``top1_acc`` of each held-out fold (argmax of the outputs against the
  one-hot target), then the mean over the five folds, as HEAR's ``aggregate_test_results`` reports it;
* ``top1_acc_std``: the sample standard deviation (``ddof = 1``, pandas ``std``) of the five fold scores, HEAR's
  ``top1_acc_std``;
* ``top1_acc_per_fold``: the five fold scores in ``splits.folds`` order;
* ``top1_acc_floor``: the same score of the label-only floor (each fold's train-majority class), the floor of
  ``top1_acc_mean``.

Its per-row quantity (argmax correct) gives ``top1_acc_mean`` the pooled Wilson interval over the 7 438 held-out
rows (``metrics.ci: pooled_binomial``). The pooled outputs reach a metric in fold order (the probe concatenates
the folds in ``splits.folds`` order); each function slices them back with the fold sizes of ``split_meta``.
Module scope: stdlib, numpy and yavih core.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from yavih.metrics.registry import MetricSpec


def _sibling(name: str):
    """A file next to this one, imported by path (keyed on its content, like yavih loads eval.py)."""
    path = Path(__file__).resolve().with_name(name)
    data = path.read_bytes()
    mod_name = f"yavih_folder_{path.parent.name}_{path.stem}_{hashlib.sha256(data).hexdigest()[:12]}"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
    return sys.modules[mod_name]


hear_score = _sibling("hear_score.py")

__all__ = ["METRICS", "fold_scores", "top1_acc_mean", "top1_acc_std", "top1_acc_per_fold", "top1_acc_floor", "top1_correct"]


def _one_hot(outputs: Any, num_classes: int) -> np.ndarray:
    """Scores ``[n, C]`` pass through; a floor's constant class indices ``[n]`` become one-hot ``[n, C]``."""
    arr = np.asarray(outputs)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
        out = np.zeros((arr.shape[0], num_classes), dtype=np.float32)
        out[np.arange(arr.shape[0]), arr] = 1.0
        return out
    raise ValueError(f"outputs must be scores [n, C] or class indices [n], got shape {arr.shape} dtype {arr.dtype}")


def _num_classes(outputs: Any, split_meta: Any) -> int:
    n = split_meta.get("num_classes") if isinstance(split_meta, dict) else getattr(split_meta, "num_classes", None)
    if n:
        return int(n)
    arr = np.asarray(outputs)
    if arr.ndim == 2:
        return int(arr.shape[1])
    raise ValueError("split_meta['num_classes'] is needed to score class indices")


def _fold_sizes(split_meta: Any, n: int) -> list[tuple[str, int]]:
    folds = split_meta.get("folds") if isinstance(split_meta, dict) else getattr(split_meta, "folds", None)
    if not folds:
        raise ValueError("crema_d scores folds: split_meta['folds'] = {fold: {outputs, targets}} is required")
    sizes = [(str(f), int(len(np.asarray(v["targets"])))) for f, v in folds.items()]
    if sum(s for _, s in sizes) != n:
        raise ValueError(f"the folds hold {sum(s for _, s in sizes)} rows, the pooled outputs {n}")
    return sizes


def fold_scores(outputs, targets, protocol, split_meta) -> dict[str, dict[str, float]]:
    """``{fold: {"top1_acc": HEAR Top1Accuracy}}`` over the pooled rows sliced back into folds."""
    t = np.asarray(targets).astype(np.int64).ravel()
    c = _num_classes(outputs, split_meta)
    pred = _one_hot(outputs, c)
    if pred.shape[0] != t.shape[0]:
        raise ValueError(f"outputs rows {pred.shape[0]} != targets {t.shape[0]}")
    onehot = np.zeros((t.shape[0], c), dtype=np.float32)
    onehot[np.arange(t.shape[0]), t] = 1.0
    scorer = hear_score.Top1Accuracy(label_to_idx={})
    out: dict[str, dict[str, float]] = {}
    lo = 0
    for fold, size in _fold_sizes(split_meta, t.shape[0]):
        out[fold] = {"top1_acc": float(scorer(pred[lo:lo + size], onehot[lo:lo + size]))}
        lo += size
    return out


def top1_acc_mean(outputs, targets, protocol, split_meta) -> float:
    return hear_score.aggregate_test_results(fold_scores(outputs, targets, protocol, split_meta))["top1_acc_mean"]


def top1_acc_std(outputs, targets, protocol, split_meta) -> float:
    return hear_score.aggregate_test_results(fold_scores(outputs, targets, protocol, split_meta))["top1_acc_std"]


def top1_acc_per_fold(outputs, targets, protocol, split_meta) -> list[float]:
    return [v["top1_acc"] for v in fold_scores(outputs, targets, protocol, split_meta).values()]


def top1_acc_floor(outputs, targets, protocol, split_meta) -> float:
    """``top1_acc_mean`` of the floor's constant outputs (each fold's train-majority class)."""
    return top1_acc_mean(outputs, targets, protocol, split_meta)


def top1_correct(outputs, targets, protocol, split_meta) -> np.ndarray:
    """Per row: 1.0 when HEAR's argmax hits the target (the pooled-binomial interval's quantity)."""
    t = np.asarray(targets).astype(np.int64).ravel()
    pred = _one_hot(outputs, _num_classes(outputs, split_meta))
    return (np.argmax(pred, axis=1) == t).astype(np.float64)


_SRC = "hear-eval-kit heareval/score.py Top1Accuracy + predictions/task_predictions.py aggregate_test_results"

METRICS: dict[str, MetricSpec] = {
    "top1_acc_mean": MetricSpec(
        name="top1_acc_mean", fn=top1_acc_mean, readouts=frozenset({"clf"}), per_clip=top1_correct,
        ci=frozenset({"pooled_binomial"}),
        description=f"HEAR top1_acc per held-out fold, mean over the folds ({_SRC})"),
    "top1_acc_std": MetricSpec(
        name="top1_acc_std", fn=top1_acc_std, readouts=frozenset({"clf"}), higher_is_better=False,
        description=f"HEAR top1_acc_std: sample std (ddof 1) of the fold scores ({_SRC})"),
    "top1_acc_per_fold": MetricSpec(
        name="top1_acc_per_fold", fn=top1_acc_per_fold, readouts=frozenset({"clf"}), shape="list",
        description=f"HEAR top1_acc of each held-out fold, splits.folds order ({_SRC})"),
    "top1_acc_floor": MetricSpec(
        name="top1_acc_floor", fn=top1_acc_floor, readouts=frozenset({"clf"}), kind="floor", higher_is_better=True,
        floor_of=("top1_acc_mean",),
        description="top1_acc_mean of each fold's train-majority class (label-only floor)"),
}
