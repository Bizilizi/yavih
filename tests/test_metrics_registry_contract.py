"""Metric registry contract (docs/readouts.md): every registered name (and every pattern expansion) is callable through its
``LazyFn`` with the driver signature ``fn(outputs, targets, protocol, split_meta)`` on synthetic inputs of
its readout, returns a JSON number / bool / list (``hashing.canon`` accepts it), and floors work both
with the driver's constant outputs and with ``outputs=None``."""

from __future__ import annotations

import numpy as np
import pytest

from yavih import hashing
from yavih.metrics import registry as R
from yavih.metrics import segment

rng = np.random.default_rng(0)

AVSYNC_P = {
    "offsets_seconds": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
    "coarse_grid_seconds": 1.0,
    "floor_sigmas": 3.0,
}
SEQ_P = {"num_segments": 10, "segments_scored": 8, "segment_offset": 1, "background_index": 28}
GRID_P = {"max_depth": 10.0, "target_size": 4}
BOX_P: dict = {}


def clf_case(c=9, n=40):
    targets = rng.integers(0, c, size=n)
    outputs = rng.standard_normal((n, c)).astype(np.float32)
    outputs[np.arange(n), targets] += 2.0
    train = rng.integers(0, c, size=100)
    folds = {str(k): {"outputs": rng.standard_normal((10, c)), "targets": rng.integers(0, c, size=10)} for k in range(3)}
    floor = {"outputs": rng.standard_normal((30, c)), "targets": rng.integers(0, c, size=30)}
    meta = {"n_train": 100, "n_eval": n, "train_targets": train, "folds": folds, "floor": floor,
            "chance": {"acc": 1 / 9}}
    return outputs, targets, meta


def multi_case(c=6, n=40):
    targets = (rng.random((n, c)) < 0.3).astype(np.float32)
    targets[0] = 1
    outputs = rng.standard_normal((n, c)).astype(np.float32)
    train = (rng.random((80, c)) < 0.3).astype(np.float32)
    sets = {"audible": targets, "visible": (rng.random((n, c)) < 0.3).astype(np.float32)}
    sets["visible"][0] = 1
    return outputs, sets, {"train_targets": train, "n_train": 80}


def seq_case(n=30, c=29):
    starts = rng.integers(0, 6, size=n)
    ends = starts + rng.integers(1, 5, size=n)
    labels = rng.integers(0, 28, size=n)
    targets = np.array([segment.segment_targets(s, e, lab, SEQ_P) for s, e, lab in zip(starts, ends, labels, strict=True)])
    outputs = rng.standard_normal((n, 8, c)).astype(np.float32)
    train = np.array([segment.segment_targets(s, e, lab, SEQ_P) for s, e, lab in zip(starts, ends, labels, strict=True)])
    meta = {"train_targets": train, "event_start": starts, "event_end": ends, "n_train": n}
    return outputs, targets, meta


def grid_case(n=5, s=4):
    targets = rng.uniform(0.5, 9.0, size=(n, s, s)).astype(np.float32)
    targets[:, 0, 0] = 0.0  # invalid pixel
    outputs = (targets * rng.uniform(0.8, 1.3, size=targets.shape)).astype(np.float32)
    train = rng.uniform(0.5, 9.0, size=(12, s, s)).astype(np.float32)
    return outputs, targets, {"train_targets": train}


def box_case(n=30):
    cx = rng.uniform(0.3, 0.7, size=n)
    cy = rng.uniform(0.3, 0.7, size=n)
    w = rng.uniform(0.2, 0.5, size=n)
    h = rng.uniform(0.2, 0.5, size=n)
    targets = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
    targets[3] = np.nan
    outputs = (targets + rng.normal(0, 0.03, size=targets.shape)).astype(np.float32)
    outputs[3] = 0.5
    train = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
    shuffled = rng.uniform(0.1, 0.9, size=targets.shape).astype(np.float32)
    return outputs, targets, {"train_targets": train, "shuffled": {"outputs": shuffled}, "n_excluded_degenerate": 1}


def inputs_for(name: str, spec: R.MetricSpec):
    if "seq" in spec.readouts and spec.readouts == {"seq"}:
        return (*seq_case(), SEQ_P)
    if spec.readouts == {"grid"}:
        return (*grid_case(), GRID_P)
    if spec.readouts == {"box"}:
        return (*box_case(), BOX_P)
    if name in {"map", "effective_pos_classes", "map_floor"}:
        o, sets, m = multi_case()
        return o, sets["audible"], m, {}
    o, t, m = clf_case()
    return o, t, m, AVSYNC_P


@pytest.mark.parametrize("name", sorted(R.REGISTRY))
def test_every_registered_fn_runs_and_returns_json_scalar(name):
    spec = R.REGISTRY[name]
    outputs, targets, meta, protocol = inputs_for(name, spec)
    value = spec.fn(outputs, targets, protocol, meta)
    if spec.shape == "list":
        assert isinstance(value, list) and all(isinstance(v, float) for v in value)
    else:
        assert isinstance(value, float | bool), (name, type(value))
    hashing.canon({name: value})  # JSON-safe: never NaN / numpy scalar
    if spec.kind == "floor":
        # outputs=None derives the constant prediction from split_meta["train_targets"]
        derived = spec.fn(None, targets, protocol, meta)
        assert isinstance(derived, float) and 0.0 <= derived <= 1.0
        hashing.canon({name: derived})


def test_pattern_expansions_run_with_bound_set_name():
    outputs, sets, meta = multi_case()
    specs = R.expand_sets(["audible", "visible"])
    for name, spec in specs.items():
        value = spec.fn(outputs, sets, {}, meta)
        assert isinstance(value, float) and 0.0 <= value <= 1.0, name
        hashing.canon({name: value})
    assert specs["auc_pr_audible_floor"].fn(None, sets, {}, meta) == pytest.approx(sets["audible"].mean())


def test_disclosure_counts_are_integral_floats():
    o, t, m = clf_case()
    for name in ("n_eval", "n_train", "n_folds"):
        v = R.REGISTRY[name].fn(o, t, AVSYNC_P, m)
        assert isinstance(v, float) and v == int(v)
    o, t, m = seq_case()
    assert R.REGISTRY["n_boundary"].fn(o, t, SEQ_P, m) == float(sum((m["event_end"] - m["event_start"]) < 10))
    o, t, m = box_case()
    assert R.REGISTRY["n_excluded_degenerate"].fn(o, t, BOX_P, m) == 1.0
    assert R.REGISTRY["n_excluded_degenerate"].fn(o, t, BOX_P, {}) == 1.0  # counted from the NaN row


def test_oracle_beats_floor_for_every_readout():
    """An oracle clears every readout's floor, at the metric level."""
    o, t, m = clf_case()
    oracle = np.eye(9)[t]
    assert R.REGISTRY["acc"].fn(oracle, t, {}, m) == 1.0 > R.REGISTRY["acc_floor"].fn(None, t, {}, m)
    o, sets, m = multi_case()
    assert R.REGISTRY["map"].fn(sets["audible"], sets["audible"], {}, m) == 1.0 > R.REGISTRY["map_floor"].fn(None, sets["audible"], {}, m)
    o, t, m = seq_case()
    assert R.REGISTRY["seg_acc"].fn(t, t, SEQ_P, m) == 1.0 > R.REGISTRY["seg_acc_floor"].fn(None, t, SEQ_P, m)
    o, t, m = grid_case()
    assert R.REGISTRY["d1"].fn(t, t, GRID_P, m) == 1.0 > R.REGISTRY["d1_floor"].fn(None, t, GRID_P, m)
    o, t, m = box_case()
    assert R.REGISTRY["iou_mean"].fn(t, t, BOX_P, m) == 1.0 > R.REGISTRY["iou_floor"].fn(None, t, BOX_P, m)
