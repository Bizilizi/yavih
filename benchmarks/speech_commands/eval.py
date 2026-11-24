"""``speech_commands`` scoring hooks: HEAR's ``top1_acc``, computed by HEAR's own scorer.

The benchmark is the HEAR 2021 task ``speech_commands-v0.0.2-5h`` (Turian et al., PMLR 176, 2022). HEAR scores it
with ``heareval.score.Top1Accuracy`` (``top1_acc``): the argmax of the model's softmax prediction against the argmax
of the one-hot target, averaged over the test clips. That class is copied, unchanged, from hear-eval-kit into the
sibling ``hear_score.py`` (grounding rung 2: the ``heareval`` package on PyPI pins ``numpy==1.19.2`` and
``tensorflow>=2.0``, and its ``heareval.score`` module imports ``torch``, ``sed_eval`` and ``dcase_util`` at module
scope). ``hear_inputs`` is the glue that hands it what ``heareval.predictions.task_predictions`` hands it: a softmax
over the head's raw scores and one-hot targets.

The probe is yavih's ``clf`` readout (core), not HEAR's grid-searched MLP; see ``metrics.notes``.

``METRICS`` overlays the registry for this benchmark:

* ``top1_acc`` (headline): ``Top1Accuracy`` on the probe's test outputs; its per-clip 0/1 success gives the
  binomial interval;
* ``top1_acc_floor``: the same scorer on the label-only floor outputs (one-hot of the train-majority class).

Module scope imports the standard library, numpy and yavih core only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from yavih.metrics._common import as_array
from yavih.metrics.classification import _single_targets
from yavih.metrics.registry import MetricSpec

__all__ = ["METRICS", "Top1Accuracy", "hear_inputs", "top1_acc", "top1_acc_floor", "top1_correct"]


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


#: hear-eval-kit's ``heareval.score.Top1Accuracy``, copied verbatim into ``hear_score.py``.
Top1Accuracy = _sibling("hear_score.py").Top1Accuracy



# ----------------------------------------------------------------------------- glue (yavih)


def hear_inputs(outputs: Any, targets: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(predictions [n, C], one-hot targets [n, C])`` as HEAR hands them to its scorer.

    Glue transcribed from hear-eval-kit @ 855964977238e89dfc76394aa11c37010edb6f20: the prediction head of
    ``heareval/predictions/task_predictions.py`` (lines 179-193) applies ``torch.nn.Softmax()`` to the logits of a
    ``multiclass`` task and ``ScenePredictionModel._score_epoch_end`` (lines 357-386) scores that prediction against
    the one-hot target (``heareval.score.label_to_binary_vector``). Here: a float64 softmax over the probe's raw
    scores (subtract the row maximum, exponentiate, normalise) and a one-hot of the integer targets.
    """
    o = as_array(outputs, np.float64)
    if o.ndim != 2:
        raise ValueError(f"top1_acc needs raw scores [n, C], got shape {o.shape}")
    t = _single_targets(targets)
    if t.shape[0] != o.shape[0]:
        raise ValueError(f"outputs rows {o.shape[0]} do not match targets {t.shape[0]}")
    if t.size == 0:
        raise ValueError("top1_acc: empty eval population")
    if np.any(t < 0) or np.any(t >= o.shape[1]):
        raise ValueError(f"target indices must lie in [0, {o.shape[1]})")
    z = np.exp(o - o.max(axis=1, keepdims=True))
    prediction = z / z.sum(axis=1, keepdims=True)
    onehot = np.zeros_like(prediction)
    onehot[np.arange(t.size), t] = 1.0
    return prediction, onehot


def _scorer(n_classes: int) -> Top1Accuracy:
    # HEAR builds its scorers with the task's label vocabulary (label -> idx); Top1Accuracy never reads it.
    return Top1Accuracy(label_to_idx={str(i): i for i in range(n_classes)})


def top1_acc(outputs, targets, protocol, split_meta) -> float:
    """HEAR ``top1_acc`` (``heareval.score.Top1Accuracy``) of the probe's eval outputs."""
    prediction, onehot = hear_inputs(outputs, targets)
    return float(_scorer(prediction.shape[1])(prediction, onehot))


def top1_acc_floor(outputs, targets, protocol, split_meta) -> float:
    """HEAR ``top1_acc`` of the label-only floor: the train-majority class (one-hot logits) on every eval clip."""
    return top1_acc(outputs, targets, protocol, split_meta)


def top1_correct(outputs: Any, targets: Any) -> np.ndarray:
    """``float64 [n]`` 0/1: the per-clip success ``Top1Accuracy`` averages (``argmax prediction == argmax target``,
    first maximum on ties, as ``np.argmax``); its mean equals ``top1_acc``. For the binomial interval."""
    prediction, onehot = hear_inputs(outputs, targets)
    return (np.argmax(prediction, axis=1) == np.argmax(onehot, axis=1)).astype(np.float64)


def _pc_top1(outputs, targets, protocol, split_meta) -> np.ndarray:
    return top1_correct(outputs, targets)


_CLF = frozenset({"clf"})
_BINARY_CI = frozenset({"binomial", "bootstrap_clip"})
_SOURCE = "hear-eval-kit @ 855964977238e89dfc76394aa11c37010edb6f20 heareval/score.py Top1Accuracy (copied)"

#: Overlays ``yavih.metrics.REGISTRY`` for this benchmark.
METRICS: dict[str, MetricSpec] = {
    "top1_acc": MetricSpec(
        name="top1_acc", fn=top1_acc, readouts=_CLF, per_clip=_pc_top1, ci=_BINARY_CI,
        description=f"HEAR top1_acc: argmax of the softmax prediction == argmax of the one-hot target, mean over clips; {_SOURCE}",
    ),
    "top1_acc_floor": MetricSpec(
        name="top1_acc_floor", fn=top1_acc_floor, readouts=_CLF, kind="floor", floor_of=("top1_acc",), higher_is_better=None,
        description=f"HEAR top1_acc of the train-majority class predicted for every clip; {_SOURCE}",
    ),
}
