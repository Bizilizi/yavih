"""Clf metrics: hand-computed values, sklearn parity for AP / ROC-AUC, floors, folds."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from yavih.metrics import classification as C

P: dict = {}


def test_acc_hand_case_and_int_predictions():
    logits = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4], [0.6, 0.2, 0.2]], dtype=np.float32)
    targets = np.array([0, 1, 2, 1], dtype=np.int64)
    assert C.acc(logits, targets, P, {}) == pytest.approx(0.75)
    assert C.acc(np.array([0, 1, 2, 0]), targets, P, {}) == pytest.approx(0.75)
    assert C.correct(logits, targets).tolist() == [True, True, True, False]
    with pytest.raises(ValueError):
        C.acc(logits[:3], targets, P, {})
    with pytest.raises(ValueError):
        C.acc(np.zeros((0, 3)), np.zeros(0, dtype=np.int64), P, {})


def test_acc_accepts_one_hot_targets_and_float_predictions_refused():
    one_hot = np.eye(3)[[0, 1, 2]]
    assert C.acc(np.array([0, 1, 1]), one_hot, P, {}) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        C.acc(np.array([0.0, 1.0, 1.0]), np.array([0, 1, 2]), P, {})


def test_disclosures():
    targets = np.arange(6)
    assert C.n_eval(None, targets, P, {}) == 6.0
    assert C.n_eval(None, None, P, {"n_eval": 9}) == 9.0
    assert C.n_eval(None, {"a": np.zeros((4, 2))}, P, {}) == 4.0
    assert C.n_train(None, targets, P, {"n_train": 12}) == 12.0
    assert C.n_train(None, targets, P, {"train_targets": np.zeros(7, dtype=np.int64)}) == 7.0
    with pytest.raises(ValueError):
        C.n_train(None, targets, P, {})


def test_acc_floor_from_train_targets_and_from_outputs():
    train = np.array([0, 0, 1, 2, 2, 2])
    ev = np.array([2, 2, 0, 1])
    assert C.majority_class(train) == 2
    assert C.majority_class(np.array([1, 1, 0, 0])) == 0  # ties → lowest index
    assert C.acc_floor(None, ev, P, {"train_targets": train}) == pytest.approx(0.5)
    constant = np.tile(np.eye(3)[2], (4, 1))  # the driver's floor_outputs: constant one-hot
    assert C.acc_floor(constant, ev, P, {}) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        C.acc_floor(None, ev, P, {})


@pytest.mark.parametrize("seed", range(8))
def test_average_precision_matches_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = 60
    y = rng.random(n) < 0.3
    y[0] = True
    s = np.round(rng.random(n), 1)  # one decimal → many tied scores
    assert C.average_precision(s, y) == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert C.roc_auc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    # distinct scores too
    s2 = rng.random(n)
    assert C.average_precision(s2, y) == pytest.approx(average_precision_score(y, s2), abs=1e-12)
    assert C.roc_auc(s2, y) == pytest.approx(roc_auc_score(y, s2), abs=1e-12)


def test_average_precision_edge_cases():
    assert C.average_precision([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0
    assert C.average_precision([0.1, 0.8, 0.9], [1, 0, 0]) == pytest.approx(1 / 3)
    assert C.average_precision([0.5, 0.5, 0.5], [1, 0, 0]) == pytest.approx(1 / 3)  # all tied = prevalence
    with pytest.raises(ValueError):
        C.average_precision([0.1, 0.2], [0, 0])
    assert C.roc_auc([0.5, 0.5], [1, 0]) == 0.5
    with pytest.raises(ValueError):
        C.roc_auc([0.1, 0.2], [1, 1])


def test_map_hand_case_zero_positive_class_counts_zero():
    outputs = np.array([[0.9, 0.2], [0.8, 0.9], [0.1, 0.4]])
    targets = np.array([[1, 0], [0, 0], [1, 0]], dtype=np.float32)
    # class 0: ranked (1, 0, 1) → AP = (1·1 + 2/3·1)/2 = 0.8333; class 1: no positive → 0
    assert C.map(outputs, targets, P, {}) == pytest.approx((5 / 6 + 0.0) / 2)
    assert C.effective_pos_classes(outputs, targets, P, {}) == 1.0
    # sigmoid is monotone: logits and probabilities agree
    assert C.map(1 / (1 + np.exp(-outputs)), targets, P, {}) == pytest.approx(C.map(outputs, targets, P, {}))


def test_map_matches_sklearn_macro_when_every_class_has_a_positive():
    rng = np.random.default_rng(3)
    o = rng.standard_normal((80, 6))
    t = (rng.random((80, 6)) < 0.3).astype(np.float32)
    t[0] = 1
    assert C.map(o, t, P, {}) == pytest.approx(average_precision_score(t, o, average="macro"), abs=1e-12)
    with pytest.raises(ValueError):
        C.map(o[:, :5], t, P, {})  # fixed C: shapes must agree


def test_map_floor_is_eval_prevalence_averaged_over_classes():
    train = np.array([[1, 0], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
    ev = np.array([[1, 0], [0, 0], [1, 1]], dtype=np.float32)
    # a constant per-class score ties every row: AP = eval prevalence of the class
    assert C.map_floor(None, ev, P, {"train_targets": train}) == pytest.approx((2 / 3 + 1 / 3) / 2)
    prev = C.train_prevalence(train, 2)
    assert prev.tolist() == pytest.approx([0.75, 0.25])
    assert C.train_prevalence(np.array([0, 0, 1]), 3).tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])
    outputs = np.broadcast_to(prev, ev.shape)
    assert C.map_floor(outputs, ev, P, {}) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        C.map_floor(None, ev, P, {})


def test_per_set_metrics_with_mapping_targets_and_one_hot_train():
    rng = np.random.default_rng(0)
    o = rng.standard_normal((50, 4))
    sets = {
        "audible": (rng.random((50, 4)) < 0.4).astype(np.float32),
        "visible": (rng.random((50, 4)) < 0.4).astype(np.float32),
    }
    for s in sets.values():
        s[0] = 1
        s[1] = 0
    train = rng.integers(0, 4, size=200)  # VGGSounder: single train label, one-hot
    for name, t in sets.items():
        assert C.auc_pr(o, sets, P, {}, set_name=name) == pytest.approx(
            average_precision_score(t, o, average="macro"), abs=1e-12
        )
        assert C.auc_roc(o, sets, P, {}, set_name=name) == pytest.approx(
            roc_auc_score(t, o, average="macro"), abs=1e-12
        )
        # floors: outputs=None derives the constant train-prevalence score (a given outputs is scored as is)
        floor = C.auc_pr_floor(None, sets, P, {"train_targets": train}, set_name=name)
        assert floor == pytest.approx(t.mean(axis=0).mean())
        assert C.auc_pr_floor(o, sets, P, {}, set_name=name) == C.auc_pr(o, sets, P, {}, set_name=name)
        # the set matrix may also come via split_meta["sets"] or be passed directly
        assert C.auc_pr(o, None, P, {"sets": sets}, set_name=name) == C.auc_pr(o, t, P, {}, set_name=name)
    with pytest.raises(KeyError):
        C.auc_pr(o, sets, P, {}, set_name="all")


def test_macro_roc_auc_excludes_undefined_classes():
    o = np.array([[0.9, 0.1, 0.3], [0.2, 0.8, 0.3], [0.7, 0.4, 0.3], [0.1, 0.6, 0.3]])
    t = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)  # class 2 has no positive
    expect = np.mean([roc_auc_score(t[:, 0], o[:, 0]), roc_auc_score(t[:, 1], o[:, 1])])
    assert C.macro_roc_auc(o, t) == pytest.approx(expect)
    with pytest.raises(ValueError):
        C.macro_roc_auc(o, np.ones_like(t))


def test_folds_aggregates():
    folds = {
        "1": {"outputs": np.eye(3)[[0, 1, 2, 0]], "targets": np.array([0, 1, 2, 1])},  # 0.75
        "2": {"outputs": np.eye(3)[[0, 0]], "targets": np.array([1, 2])},  # 0.0
        "3": {"outputs": np.array([2, 2, 2, 2]), "targets": np.array([2, 2, 2, 2])},  # 1.0
    }
    meta = {"folds": folds}
    per = C.acc_per_fold(None, None, P, meta)
    assert per == pytest.approx([0.75, 0.0, 1.0])
    assert C.acc_mean(None, None, P, meta) == pytest.approx(np.mean(per))
    assert C.acc_pstd(None, None, P, meta) == pytest.approx(np.std(per, ddof=0))
    assert C.n_folds(None, None, P, meta) == 3.0
    assert C.n_eval(None, None, P, meta) == 10.0
    with pytest.raises(ValueError):
        C.acc_mean(None, None, P, {})
    with pytest.raises(ValueError):
        C.n_folds(None, None, P, {"folds": {}})
