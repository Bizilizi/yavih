# ruff: noqa: E721, B905, I001, UP006, UP007, UP035, UP045  -- the copied block below is kept byte-identical to upstream
"""HEAR's ``top1_acc`` scorer, copied from hear-eval-kit (no edits to the copied code).

Source: hear-eval-kit, the HEAR 2021 benchmark's evaluation kit (Turian et al., "HEAR: Holistic Evaluation of
Audio Representations", PMLR 176, 2022), https://github.com/hearbenchmark/hear-eval-kit
@ 855964977238e89dfc76394aa11c37010edb6f20 (tag v2021.1.3, the code of ``heareval==2021.1.3`` on PyPI),
file ``heareval/score.py``, lines 57-155: ``validate_score_return_type``, ``ScoreFunction``, ``Top1Accuracy``.

Licence: Apache License 2.0 (https://github.com/hearbenchmark/hear-eval-kit/blob/main/LICENSE), as found upstream.

Changes: the upstream module's imports are reduced to the two these three objects need (``typing``, ``numpy``);
upstream ``heareval/score.py`` also imports ``torch``, ``pandas``, ``sed_eval``, ``dcase_util``, ``sklearn`` and
``scipy`` at module scope for its other scorers. The copied lines below are byte-identical to the upstream lines
(``tests/test_folder.py`` pins their sha256). Why copied, not imported: ``heareval`` on PyPI pins
``numpy==1.19.2``, ``numba==0.48`` and ``tensorflow>=2.0``, and ``heareval.score`` imports ``sed_eval`` and
``dcase_util`` at module scope.

Loaded by ``eval.py`` as a sibling module (by path); module scope imports only ``typing`` and numpy.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# ----- begin copied: heareval/score.py lines 57-155 @ 855964977238e89dfc76394aa11c37010edb6f20 -----
def validate_score_return_type(ret: Union[Tuple[Tuple[str, float], ...], float]):
    """
    Valid return types for the metric are
        - tuple(tuple(string: name of the subtype, float: the value)): This is the
            case with sed eval metrics. They can return (("f_measure", value),
            ("precision", value), ...), depending on the scores
            the metric should is supposed to return. This is set as `scores`
            attribute in the metric.
        - float: Standard metric behaviour

    The downstream prediction pipeline is able to handle these two types.
    In case of the tuple return type, the value of the first entry in the
    tuple will be used as an optimisation criterion wherever required.
    For instance, if the return is (("f_measure", value), ("precision", value)),
    the value corresponding to the f_measure will be used ( for instance in
    early stopping if this metric is the primary score for the task )
    """
    if isinstance(ret, tuple):
        assert all(
            type(s) == tuple and type(s[0]) == str and type(s[1]) == float for s in ret
        ), (
            "If the return type of the score is a tuple, all the elements "
            "in the tuple should be tuple of type (string, float)"
        )
    elif isinstance(ret, float):
        pass
    else:
        raise ValueError(
            f"Return type {type(ret)} is unexpected. Return type of "
            "the score function should either be a "
            "tuple(tuple) or float. "
        )


class ScoreFunction:
    """
    A simple abstract base class for score functions
    """

    # TODO: Remove label_to_idx?
    def __init__(
        self,
        label_to_idx: Dict[str, int],
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        """
        :param label_to_idx: Map from label string to integer index.
        :param name: Override the name of this scoring function.
        :param maximize: Maximize this score? (Otherwise, it's a loss or energy
            we want to minimize, and I guess technically isn't a score.)
        """
        self.label_to_idx = label_to_idx
        if name:
            self.name = name
        self.maximize = maximize

    def __call__(self, *args, **kwargs) -> Union[Tuple[Tuple[str, float], ...], float]:
        """
        Calls the compute function of the metric, and after validating the output,
        returns the metric score
        """
        ret = self._compute(*args, **kwargs)
        validate_score_return_type(ret)
        return ret

    def _compute(
        self, predictions: Any, targets: Any, **kwargs
    ) -> Union[Tuple[Tuple[str, float], ...], float]:
        """
        Compute the score based on the predictions and targets.
        This is a private function and the metric should be used as a functor
        by calling the `__call__` method which calls this and also validates
        the return type
        """
        raise NotImplementedError("Inheriting classes must implement this function")

    def __str__(self):
        return self.name


class Top1Accuracy(ScoreFunction):
    name = "top1_acc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            if predicted_class == target_class:
                correct += 1

        return correct / len(targets)
# ----- end copied -----
