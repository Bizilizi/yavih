"""Intervals: Wilson, pooled binomial, seeded clip bootstrap, and the metrics.ci dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from yavih.metrics import classification, registry, segment
from yavih.metrics import intervals as I  # noqa: E741


def test_wilson_known_values():
    lo, hi = I.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=5e-5)
    lo, hi = I.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=5e-5)
    lo, hi = I.wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=5e-5)
    lo, hi = I.wilson(1700, 2000)
    assert lo < 0.85 < hi and hi - lo < 0.04
    with pytest.raises(ValueError):
        I.wilson(1, 0)
    with pytest.raises(ValueError):
        I.wilson(11, 10)


def test_pooled_binomial_is_wilson_over_the_pooled_counts():
    assert I.pooled_binomial([40, 45], [50, 50]) == I.wilson(85, 100)
    assert I.pooled_binomial(np.array([400] * 5), np.array([400] * 5)) == (I.wilson(2000, 2000))


def test_bootstrap_clip_is_seeded_and_covers_the_point_estimate():
    rng = np.random.default_rng(1)
    v = (rng.random(200) < 0.6).astype(float)
    a = I.bootstrap_clip(v, samples=500, seed=0)
    b = I.bootstrap_clip(v, samples=500, seed=0)
    c = I.bootstrap_clip(v, samples=500, seed=1)
    assert a == b and a != c
    assert a[0] < v.mean() < a[1]
    assert I.bootstrap_clip(np.ones(10), samples=50, seed=0) == (1.0, 1.0)
    with pytest.raises(ValueError):
        I.bootstrap_clip(np.zeros(0), samples=10, seed=0)
    with pytest.raises(ValueError):
        I.bootstrap_clip(np.ones(3), samples=0, seed=0)


def test_bootstrap_clip_pools_cells_by_clip():
    rng = np.random.default_rng(2)
    v = (rng.random((50, 8)) < 0.5).astype(float)
    v[:10, 4:] = np.nan  # unscored segments
    default = I.bootstrap_clip(v, samples=300, seed=3)
    explicit = I.bootstrap_clip(v, samples=300, seed=3, statistic=lambda rows: float(np.nanmean(rows)))
    assert default == pytest.approx(explicit)
    # resampling clips, not cells: a clip-level all-or-nothing population has a wide interval
    block = np.repeat((np.arange(20) % 2).astype(float)[:, None], 8, axis=1)
    lo, hi = I.bootstrap_clip(block, samples=400, seed=0)
    assert hi - lo > 0.3


def test_interval_dispatch_binomial_bootstrap_pooled():
    targets = np.array([0, 1, 2, 1, 0, 2, 2, 1])
    outputs = np.eye(3)[[0, 1, 2, 0, 0, 2, 1, 1]]  # 6/8 correct
    assert I.interval("binomial", "acc", outputs, targets, {}, {}) == list(I.wilson(6, 8))
    P = {"num_segments": 10, "segments_scored": 8, "segment_offset": 1, "background_index": 28}
    seg_t = np.array([segment.segment_targets(3, 7, 5, P), segment.segment_targets(0, 10, 2, P)])
    seg_o = np.array([[28] * 8, [2] * 8])
    meta = {"event_start": [3, 0], "event_end": [7, 10], "bootstrap": {"samples": 100, "seed": 0}}
    ci = I.interval("bootstrap_clip", "seg_acc", seg_o, seg_t, P, meta)
    assert len(ci) == 2 and 0.0 <= ci[0] <= ci[1] <= 1.0
    cib = I.interval("bootstrap_clip", "seg_acc_boundary", seg_o, seg_t, P, meta)
    assert cib == [0.5, 0.5]  # the single boundary row scores 4/8 in every resample
    folds = {
        "1": {"outputs": np.eye(3)[[0, 1]], "targets": np.array([0, 1])},
        "2": {"outputs": np.eye(3)[[0, 0]], "targets": np.array([1, 2])},
    }
    assert I.interval("pooled_binomial", "acc_mean", None, None, {}, {"folds": folds}) == list(I.wilson(2, 4))
    with pytest.raises(ValueError):
        I.interval("pooled_binomial", "acc_mean", None, None, {}, {})
    with pytest.raises(ValueError):
        I.interval("binomial", "iou_mean", outputs, targets, {}, {})  # not a 0/1 metric
    with pytest.raises(ValueError):
        I.interval("bootstrap_clip", "map", outputs, targets, {}, {})  # map needs multi-hot [n, C] targets
    with pytest.raises(ValueError):
        I.interval("bootstrap_clip", "effective_pos_classes", outputs, targets, {}, {})  # a disclosure has none
    with pytest.raises(ValueError):
        I.interval("jackknife", "acc", outputs, targets, {}, {})


def test_per_clip_values_cover_every_table_entry():
    assert set(I.PER_CLIP) >= {"acc", "seg_acc", "seg_acc_boundary", "clip_acc",
                               "iou_mean", "acc@0.5", "acc@0.3", "center_in_gt", "d1", "d2", "d3", "absrel", "rmse"}
    boxes_t = np.tile([0.5, 0.5, 0.4, 0.4], (3, 1))
    boxes_o = np.array([[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.8, 0.8]])
    assert I.per_clip_values("acc@0.5", boxes_o, boxes_t, {}, {}).tolist() == [1.0, 0.0, 0.0]
    assert I.per_clip_values("center_in_gt", boxes_o, boxes_t, {}, {}).tolist() == [1.0, 0.0, 1.0]
    assert I.interval("binomial", "acc@0.5", boxes_o, boxes_t, {}, {}) == list(I.wilson(1, 3))


def _multilabel(rng, n=60, c=7, ties=True):
    t = (rng.random((n, c)) < 0.25).astype(np.float32)
    t[:, -1] = 0.0  # a zero-positive class: AP = 0 in the point estimate and in every resample
    o = rng.normal(size=(n, c)) + 1.5 * t
    if ties:
        o = np.round(o, 1)  # many tied scores: thresholds must be grouped exactly as average_precision does
    return o, t


def test_map_resample_stats_equal_macro_ap_on_the_materialised_resample():
    """The vectorised weighted AP == ``macro_average_precision(o[draw], t[draw])`` for every resample
    (duplicates, zero-weight clips, ties, a class whose positives all drop out of a resample)."""
    rng = np.random.default_rng(7)
    for ties in (True, False):
        o, t = _multilabel(rng, ties=ties)
        draws = I.bootstrap_draws(o.shape[0], samples=40, seed=3)
        fast = I.map_resample_stats(o, t, draws)
        slow = np.array([classification.macro_average_precision(o[d], t[d]) for d in draws])
        np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-12)
    # the identity resample reproduces the point estimate
    ident = np.arange(o.shape[0])[None, :]
    assert I.map_resample_stats(o, t, ident)[0] == pytest.approx(classification.macro_average_precision(o, t), abs=1e-12)


def test_map_bootstrap_interval_is_seeded_and_brackets_the_point_estimate():
    """fsd50k ships ``ci: {map: bootstrap_clip}``: the driver must produce it, not refuse."""
    rng = np.random.default_rng(11)
    o, t = _multilabel(rng, n=200, c=12, ties=False)
    meta = {"bootstrap": {"samples": 300, "seed": 0}}
    a = I.interval("bootstrap_clip", "map", o, t, {}, meta)
    assert a == I.interval("bootstrap_clip", "map", o, t, {}, meta)
    assert a != I.interval("bootstrap_clip", "map", o, t, {}, {"bootstrap": {"samples": 300, "seed": 1}})
    point = classification.macro_average_precision(o, t)
    assert a[0] < point < a[1] and 0.0 <= a[0] <= a[1] <= 1.0
    with pytest.raises(ValueError):
        I.interval("binomial", "map", o, t, {}, meta)  # not a 0/1 metric


def test_ci_admissible_table_matches_what_intervals_computes():
    """``registry.CI_ADMISSIBLE`` (what ``Benchmark.load`` accepts) == what the driver can compute."""
    assert set(registry.CI_ADMISSIBLE) == set(registry.CI_KINDS)
    for kind, names in registry.CI_ADMISSIBLE.items():
        assert names == I.admissible(kind), kind
        assert names <= set(registry.REGISTRY), sorted(names - set(registry.REGISTRY))
