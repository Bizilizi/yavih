# ruff: noqa: UP006, UP007, UP035, UP045, B905  -- the copied upstream code keeps its own style (typing aliases, format calls, zip)
"""HEAR's score of a multiclass scene task: ``top1_acc`` per test fold, then the mean and the std over folds.

Copied from hear-eval-kit (https://github.com/hearbenchmark/hear-eval-kit @ ``855964977238e89dfc76394aa11c37010edb6f20``,
tag ``v2021.1.3``, the ``heareval`` 2021.1.3 wheel on PyPI, Apache-2.0), the code that scores HEAR's ``tfds_crema_d`` task (``evaluation: [top1_acc, mAP,
d_prime, aucroc]``, the first being the task's primary score):

* ``heareval/score.py``: ``validate_score_return_type``, ``ScoreFunction``, ``Top1Accuracy`` (unchanged but for
  the typing imports and a ``noqa: E721`` comment for ruff; ``label_to_idx`` stays a constructor argument as upstream);
* ``heareval/predictions/task_predictions.py``: ``data_splits_from_folds`` (unchanged; which fold HEAR holds out
  for test and validation, kept for reference: yavih's core ``clf`` readout trains on the four other folds and
  carves its validation rows from them) and ``aggregate_test_results`` (change: pandas replaced by numpy;
  ``DataFrame.mean`` and ``DataFrame.std`` are the mean and the sample standard deviation, ``ddof = 1``).

Not copied: ``MeanAveragePrecision``, ``DPrime`` and ``AUCROC`` (HEAR's secondary scores of the task; they need
scikit-learn and SciPy, which the probe environment does not have) and HEAR's downstream MLP and its grid search
(the readout here is yavih's core ``clf`` probe). heareval itself is not imported: its wheel pins ``numpy==1.19.2``
and TensorFlow, and ``score.py`` imports torch, ``sed_eval`` and ``dcase_util`` at module scope.

Module scope imports the standard library and numpy only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------- heareval/score.py


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
            type(s) == tuple and type(s[0]) == str and type(s[1]) == float for s in ret  # noqa: E721
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


# --------------------------------------------------------------------------- heareval/predictions/task_predictions.py


def data_splits_from_folds(folds: List[str]) -> List[Dict[str, List[str]]]:
    """
    Create data splits by using Leave One Out Cross Validation strategy.

    folds is a list of dataset partitions created during pre-processing. For example,
    for 5-fold cross val: ["fold00", "fold01", ..., "fold04"]. This function will create
    k test, validation, and train splits using these folds. Each fold is successively
    treated as test split, the next split as validation, and the remaining as train.
    Folds will be sorted before applying the above strategy.

    With 5-fold, for example, we would have:
    test=fold00, val=fold01, train=fold02..04,
    test=fold01, val=fold02, train=fold03,04,01
    ...
    test=fold04, val=fold00, train=01..03
    """
    sorted_folds = tuple(sorted(folds))
    assert len(sorted_folds) == len(set(sorted_folds)), "Folds are not unique"
    num_folds = len(sorted_folds)
    all_data_splits: List[Dict[str, List[str]]] = []
    for fold_idx in range(num_folds):
        test_fold = sorted_folds[fold_idx]
        valid_fold = sorted_folds[(fold_idx + 1) % num_folds]
        train_folds = [f for f in sorted_folds if f not in (test_fold, valid_fold)]
        all_data_splits.append(
            {
                "train": train_folds,
                "valid": [valid_fold],
                "test": [test_fold],
            }
        )
        assert not set(train_folds).intersection(
            {test_fold, valid_fold}
        ), "Train folds are not distinct from the dev and the test folds"

    return all_data_splits


def aggregate_test_results(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregates test results over folds by mean and standard deviation
    """
    scores = sorted({score for fold in results.values() for score in fold})  # change: no pandas.DataFrame
    aggregate_results = {}
    for score in scores:
        values = np.asarray([fold[score] for fold in results.values()], dtype=np.float64)
        aggregate_results[f"{score}_mean"] = float(values.mean())
        aggregate_results[f"{score}_std"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")

    return aggregate_results
