"""Box metrics: parsing, IoU, thresholds, centre test, floor, shuffled control, McNemar."""

from __future__ import annotations

import math

import numpy as np
import pytest

from yavih.metrics import box as B

P: dict = {}


def test_parse_box():
    assert B.parse_box("10|20|30|60") == (20.0, 40.0, 20.0, 40.0)
    assert B.parse_box("5|5|5|9") is None
    assert B.parse_box("5|9|8|9") is None
    assert B.parse_box("-5|0|5|10") == (2.5, 5.0, 5.0, 10.0)  # max(x, 0) first
    assert B.parse_box("-5|0|-1|10") is None  # sanitized to x1 = x2 = 0 → degenerate
    with pytest.raises(ValueError):
        B.parse_box("1|2|3")
    boxes, excluded = B.parse_boxes(["0|0|1|1", "3|3|3|4", "0|0|2|4"])
    assert excluded == 1 and boxes.dtype == np.float32 and boxes.shape == (3, 4)
    assert np.isnan(boxes[1]).all() and boxes[2].tolist() == [1.0, 2.0, 2.0, 4.0]
    assert B.valid_rows(boxes).tolist() == [True, False, True]


def test_box_iou_hand_cases():
    unit = np.array([[0.5, 0.5, 1.0, 1.0]])
    assert B.box_iou(unit, unit).tolist() == [1.0]
    shifted = np.array([[1.0, 0.5, 1.0, 1.0]])  # x ∈ [0.5, 1.5]: inter 0.5, union 1.5
    assert B.box_iou(unit, shifted)[0] == pytest.approx(1 / 3)
    far = np.array([[5.0, 5.0, 1.0, 1.0]])
    assert B.box_iou(unit, far).tolist() == [0.0]
    empty = np.array([[0.0, 0.0, 0.0, 0.0]])
    assert B.box_iou(empty, empty).tolist() == [0.0]
    with pytest.raises(ValueError):
        B.box_iou(np.zeros((2, 4)), np.zeros((3, 4)))


def test_metrics_drop_degenerate_rows_and_count_them():
    targets = np.array([[0.5, 0.5, 1.0, 1.0], [np.nan] * 4, [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
    outputs = np.array([
        [0.5, 0.5, 1.0, 1.0],  # iou 1
        [0.0, 0.0, 1.0, 1.0],  # excluded row
        [0.5, 0.5, 1.0, 1.0],  # iou 0.25 (gt area .25 inside pred area 1), centre inside
        [5.0, 5.0, 1.0, 1.0],  # iou 0, centre outside
    ])
    assert B.ious(outputs, targets).tolist() == pytest.approx([1.0, 0.25, 0.0])
    assert B.iou_mean(outputs, targets, P, {}) == pytest.approx(1.25 / 3)
    assert B.acc_at_05(outputs, targets, P, {}) == pytest.approx(1 / 3)
    assert B.acc_at_03(outputs, targets, P, {}) == pytest.approx(1 / 3)
    assert B.center_in_gt(outputs, targets, P, {}) == pytest.approx(2 / 3)
    assert B.n_excluded_degenerate(outputs, targets, P, {}) == 1.0
    assert B.n_excluded_degenerate(outputs, targets, P, {"n_excluded_degenerate": 7}) == 7.0
    with pytest.raises(ValueError):
        B.iou_mean(outputs, np.full((4, 4), np.nan), P, {})
    with pytest.raises(ValueError):
        B.iou_mean(outputs[:2], targets, P, {})


def test_iou_floor_mean_train_box():
    train = np.array([[0.2, 0.2, 0.4, 0.4], [0.6, 0.6, 0.4, 0.4], [np.nan] * 4])
    assert B.mean_train_box(train).tolist() == pytest.approx([0.4, 0.4, 0.4, 0.4])
    ev = np.array([[0.4, 0.4, 0.4, 0.4], [0.4, 0.4, 0.8, 0.8]])
    expect = (1.0 + 0.16 / 0.64) / 2
    assert B.iou_floor(None, ev, P, {"train_targets": train}) == pytest.approx(expect)
    assert B.iou_floor(np.array([[0.4, 0.4, 0.4, 0.4]]), ev, P, {}) == pytest.approx(expect)  # broadcast [1, 4]
    assert B.iou_floor(np.tile([0.4, 0.4, 0.4, 0.4], (2, 1)), ev, P, {}) == pytest.approx(expect)
    with pytest.raises(ValueError):
        B.iou_floor(None, ev, P, {})


def test_mcnemar_z():
    a = np.array([1] * 9 + [0] * 1 + [1] * 10, dtype=bool)
    b = np.array([0] * 9 + [1] * 1 + [1] * 10, dtype=bool)
    assert B.mcnemar_z(a, b) == pytest.approx(8 / math.sqrt(10))
    assert B.mcnemar_z(a, a) == 0.0
    assert B.mcnemar_z(b, a) == pytest.approx(-8 / math.sqrt(10))
    with pytest.raises(ValueError):
        B.mcnemar_z(a, a[:5])


def test_shuffled_control_and_audio_blind():
    n = 40
    targets = np.tile([0.5, 0.5, 0.4, 0.4], (n, 1))
    hit = np.tile([0.5, 0.5, 0.4, 0.4], (n, 1))
    miss = np.tile([0.1, 0.1, 0.1, 0.1], (n, 1))
    meta_blind = {"shuffled": {"outputs": hit}}
    meta_driven = {"shuffled": {"outputs": miss}}
    assert B.iou_shuffled(hit, targets, P, meta_driven) == pytest.approx(0.0)
    assert B.iou_shuffled(hit, targets, P, meta_blind) == pytest.approx(1.0)
    # identical arms → z = 0 → blind; 40 discordant hits → z = sqrt(40) > 3 → audio-driven
    assert B.audio_blind(hit, targets, P, meta_blind) is True
    assert B.audio_blind(hit, targets, P, meta_driven) is False
    # a control that beats the headline is blind a fortiori
    assert B.audio_blind(miss, targets, P, meta_blind) is True
    # exactly 9 discordant → z = 3 → not > 3 → blind; 10 → z = sqrt(10) > 3 → audio-driven
    mixed = hit.copy()
    mixed[:9] = miss[:9]
    assert B.mcnemar_z(B.ious(hit, targets) >= 0.5, B.ious(mixed, targets) >= 0.5) == pytest.approx(3.0)
    assert B.audio_blind(hit, targets, P, {"shuffled": {"outputs": mixed}}) is True
    mixed[9] = miss[9]
    assert B.mcnemar_z(B.ious(hit, targets) >= 0.5, B.ious(mixed, targets) >= 0.5) == pytest.approx(math.sqrt(10))
    assert B.audio_blind(hit, targets, P, {"shuffled": {"outputs": mixed}}) is False
    with pytest.raises(ValueError):
        B.audio_blind(hit, targets, P, {})
