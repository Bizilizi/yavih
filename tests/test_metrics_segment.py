"""Seq metrics: segment targets, pooled vs per-clip accuracy, boundary subset, floors."""

from __future__ import annotations

import numpy as np
import pytest

from yavih.metrics import segment as S

P = {"num_segments": 10, "segment_seconds": 1.0, "segments_scored": 8, "segment_offset": 1, "background_index": 28}
BG = 28


def test_segment_targets_ave_window():
    assert S.segment_targets(3, 7, 5, P) == [BG, BG, 5, 5, 5, 5, BG, BG]
    assert S.segment_targets(0, 10, 2, P) == [2] * 8
    assert S.segment_targets(0, 1, 2, P) == [BG] * 8  # event entirely before the scored window
    assert S.segment_targets(8, 10, 4, P) == [BG] * 7 + [4]


def scores(preds, c=29):
    return np.eye(c)[np.asarray(preds)] * 3.0


def test_seg_acc_and_clip_acc_diverge_with_uneven_valid_counts():
    targets = np.array([[1, 1, -1, -1], [2, 2, 2, 2]])
    preds = np.array([[1, 1, 0, 0], [2, 0, 0, 0]])
    # clip A: 2/2 valid correct; clip B: 1/4 → pooled 3/6, per-clip mean (1 + 0.25)/2
    assert S.seg_acc(scores(preds), targets, P, {}) == pytest.approx(0.5)
    assert S.seg_acc(preds, targets, P, {}) == pytest.approx(0.5)
    assert S.clip_acc(scores(preds), targets, P, {}) == pytest.approx(0.625)
    c = S.segment_correct(preds, targets)
    assert np.isnan(c[0, 2]) and np.isnan(c[0, 3]) and c[1].tolist() == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        S.seg_acc(preds, np.full_like(targets, -1), P, {})
    with pytest.raises(ValueError):
        S.seg_acc(preds[:, :3], targets, P, {})


def test_boundary_from_event_times_and_precomputed():
    targets = np.array([[2] * 8, S.segment_targets(3, 7, 5, P), S.segment_targets(1, 9, 7, P)])
    meta = {"event_start": [0, 3, 1], "event_end": [10, 7, 9]}
    # [1, 9) covers the whole scored window but is shorter than the clip → boundary row
    assert S.boundary_mask(targets, P, meta).tolist() == [False, True, True]
    assert S.n_boundary(None, targets, P, meta) == 2.0
    assert S.boundary_mask(targets, P, {"boundary": [True, False, False]}).tolist() == [True, False, False]
    with pytest.raises(ValueError, match="event_start"):
        S.boundary_mask(targets, P, {})
    with pytest.raises(ValueError):
        S.boundary_mask(targets, P, {"boundary": [True]})
    preds = np.array([[2] * 8, [BG] * 8, [7] * 8])
    # boundary rows: row1 4/8 (bg segments right), row2 8/8 → pooled 12/16
    assert S.seg_acc_boundary(scores(preds), targets, P, meta) == pytest.approx(0.75)
    assert S.seg_acc(preds, targets, P, meta) == pytest.approx(20 / 24)
    with pytest.raises(ValueError):
        S.seg_acc_boundary(preds, targets, P, {"boundary": [False, False, False]})


def test_floors_from_train_targets_and_outputs():
    train = np.array([S.segment_targets(3, 7, 5, P), S.segment_targets(0, 10, 9, P), S.segment_targets(4, 5, 9, P)])
    assert S.constant_segment_class(train) == BG  # 4 + 0 + 7 = 11 of 24 segments are background
    assert S.constant_segment_class(np.array([[1, 1, 0, 0]])) == 0  # tie → lowest index
    ev = np.array([S.segment_targets(2, 6, 3, P), S.segment_targets(0, 10, 1, P)])
    meta = {"train_targets": train, "event_start": [2, 0], "event_end": [6, 10]}
    # constant background: row0 4/8 background, row1 0/8 → 4/16
    assert S.seg_acc_floor(None, ev, P, meta) == pytest.approx(0.25)
    assert S.seg_acc_boundary_floor(None, ev, P, meta) == pytest.approx(0.5)
    const = np.full((2, 8), BG)
    assert S.seg_acc_floor(const, ev, P, {}) == pytest.approx(0.25)
    assert S.seg_acc_floor(scores(const), ev, P, {}) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        S.seg_acc_floor(None, ev, P, {})
