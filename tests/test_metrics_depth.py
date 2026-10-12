"""Grid metrics (Probe3D definitions): per image then mean, pred * valid first, floor map."""

from __future__ import annotations

import math

import numpy as np
import pytest

from yavih.metrics import depth as D

S = 2
P = {"max_depth": 10.0, "target_size": S}


def img(values):
    return np.asarray(values, dtype=np.float32).reshape(S, S)


def test_perfect_prediction():
    gt = np.stack([img([1, 2, 3, 4]), img([5, 6, 0, 8])])
    assert D.d1(gt, gt, P, {}) == 1.0
    assert D.d2(gt, gt, P, {}) == 1.0
    assert D.d3(gt, gt, P, {}) == 1.0
    assert D.absrel(gt, gt, P, {}) == 0.0
    assert D.rmse(gt, gt, P, {}) == 0.0
    assert D.depth_valid_mask(gt, P).sum() == 7


def test_hand_case_single_image_with_invalid_pixels():
    gt = np.stack([img([2.0, 4.0, 0.0, 12.0])])  # 0 and > max_depth are invalid
    pred = np.stack([img([2.6, 4.0, 100.0, 100.0])])  # the invalid pixels' predictions must not matter
    # ratios: 1.3 (not < 1.25), 1.0 → d1 = 0.5; d2: 1.3 < 1.5625 → 1
    assert D.d1(pred, gt, P, {}) == pytest.approx(0.5)
    assert D.d2(pred, gt, P, {}) == pytest.approx(1.0)
    assert D.d3(pred, gt, P, {}) == pytest.approx(1.0)
    assert D.absrel(pred, gt, P, {}) == pytest.approx((0.6 / 2.0 + 0.0) / 2)
    assert D.rmse(pred, gt, P, {}) == pytest.approx(math.sqrt((0.36 + 0.0) / 2))


def test_per_image_then_mean_not_pooled():
    # image A: 1 valid pixel, correct; image B: 3 valid pixels, all wrong (ratio 2)
    gt = np.stack([img([1.0, 0, 0, 0]), img([1.0, 1.0, 1.0, 0])])
    pred = np.stack([img([1.0, 0, 0, 0]), img([2.0, 2.0, 2.0, 0])])
    assert D.d1(pred, gt, P, {}) == pytest.approx(0.5)  # (1 + 0) / 2, not 1/4 pooled
    per = D.depth_per_image(pred, gt, P)
    assert per["d1"].tolist() == [1.0, 0.0] and per["scored"].tolist() == [True, True]
    assert per["absrel"].tolist() == pytest.approx([0.0, 1.0])
    assert per["rmse"].tolist() == pytest.approx([0.0, 1.0])


def test_images_without_valid_pixels_are_skipped_and_all_invalid_raises():
    gt = np.stack([img([0, 0, 0, 0]), img([2.0, 2.0, 2.0, 2.0])])
    pred = np.stack([img([1, 1, 1, 1]), img([2.0, 2.0, 2.0, 2.0])])
    assert D.d1(pred, gt, P, {}) == 1.0
    assert np.isnan(D.depth_per_image(pred, gt, P)["d1"][0])
    with pytest.raises(ValueError):
        D.d1(pred, np.zeros_like(gt), P, {})


def test_pred_times_valid_applied_first_blows_up_invalid_not_valid_pixels():
    # a NaN prediction at an INVALID pixel must not leak; at a VALID pixel it fails that pixel only
    gt = np.stack([img([2.0, 2.0, 0.0, 2.0])])
    pred = np.stack([img([2.0, 2.0, np.nan, 2.0])])
    assert D.d1(pred, gt, P, {}) == 1.0


def test_channel_axis_squeeze_and_shape_checks():
    gt = np.stack([img([1, 2, 3, 4])])
    assert D.d1(gt[:, None], gt, P, {}) == 1.0
    assert D.d1(gt[..., None], gt, P, {}) == 1.0
    with pytest.raises(ValueError):
        D.d1(np.zeros((1, 3, 3)), gt, P, {})
    with pytest.raises(ValueError, match="target_size"):
        D.d1(gt, gt, {"max_depth": 10.0, "target_size": 4}, {})


def test_mean_log_depth_map_streaming_and_fallback():
    # pixel 0 valid in both (1, 4 → geometric mean 2); pixel 1 valid once (3); pixel 2 never → global mean
    train = np.stack([img([1.0, 0.0, 0.0, 2.0]), img([4.0, 3.0, 0.0, 2.0])])
    m = D.mean_log_depth_map(train, P)
    global_mean = math.exp(np.mean(np.log([1.0, 4.0, 3.0, 2.0, 2.0])))
    assert m.ravel().tolist() == pytest.approx([2.0, 3.0, global_mean, 2.0])
    np.testing.assert_allclose(D.mean_log_depth_map(train, P, chunk=1), m)
    with pytest.raises(ValueError):
        D.mean_log_depth_map(np.zeros((2, S, S)), P)


def test_d1_floor_from_train_targets_and_from_constant_map():
    train = np.stack([img([1.0, 0.0, 0.0, 2.0]), img([4.0, 3.0, 0.0, 2.0])])
    ev = np.stack([img([2.0, 3.0, 5.0, 2.5])])
    m = D.mean_log_depth_map(train, P)
    expect = D.d1(np.broadcast_to(m, ev.shape), ev, P, {})
    assert D.d1_floor(None, ev, P, {"train_targets": train}) == pytest.approx(expect)
    assert D.d1_floor(m, ev, P, {}) == pytest.approx(expect)
    assert D.d1_floor(np.broadcast_to(m, ev.shape), ev, P, {}) == pytest.approx(expect)
    # hand value: ratios 1.0, 1.0, global/5, 1.25 → pixel 3 ratio exactly 1.25 is NOT < 1.25
    assert expect == pytest.approx(2 / 4)
    with pytest.raises(ValueError):
        D.d1_floor(None, ev, P, {})
