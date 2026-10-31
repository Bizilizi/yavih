"""avsync's metrics and the floor-role controls, which live in the folder's ``eval.py`` ``METRICS`` overlay.
``test_values_match_the_golden_file`` pins every metric, control, floor and interval against
``tests/data/avsync_metrics_golden.json`` (seeded synthetic inputs)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from yavih.benchmark import Benchmark
from yavih.readouts.base import evaluate_metrics, folder_eval, resolve_metric_specs

BENCH = Benchmark.load("avsync")
A = folder_eval(BENCH)
GOLDEN = Path(__file__).parent / "data" / "avsync_metrics_golden.json"

OFFSETS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
P = {"offsets_seconds": OFFSETS, "coarse_grid_seconds": 1.0, "floor_sigmas": 3.0}


def one_hot(idx, c=9, scale=10.0):
    return np.eye(c)[np.asarray(idx)] * scale


def test_acc_tol1_and_mae_seconds_hand_case():
    preds = [0, 1, 4, 8]
    targets = np.array([1, 3, 4, 6])
    assert A.acc_tol1(one_hot(preds), targets, P, {}) == pytest.approx(0.5)
    assert A.tol1_correct(np.array(preds), targets).tolist() == [True, False, True, False]
    assert A.mae_seconds(one_hot(preds), targets, P, {}) == pytest.approx((0.5 + 1.0 + 0.0 + 1.0) / 4)
    assert A.mae_seconds(np.array(preds), targets, P, {}) == pytest.approx(0.625)


def test_truncated_logit_vector_is_refused():
    with pytest.raises(ValueError, match="bins"):
        A.acc_tol1(np.zeros((3, 5)), np.array([0, 1, 2]), P, {})
    with pytest.raises(ValueError, match="bins"):
        A.mae_seconds(np.zeros((3, 5)), np.array([0, 1, 2]), P, {})
    with pytest.raises(ValueError):
        A.offsets({"offsets_seconds": [0.0, -1.0]})  # not ascending


def test_coarse_lookup_pinned_grid():
    coarse, column = A.coarse_lookup(P)
    assert coarse.tolist() == [0, 2, 4, 6, 8]
    assert column.tolist() == [0, -1, 1, -1, 2, -1, 3, -1, 4]
    coarse2, _ = A.coarse_lookup({**P, "coarse_grid_seconds": 2.0})
    assert coarse2.tolist() == [0, 4, 8]


def test_coarse_probabilities_mass_preserving_and_half_split():
    logits = np.array([one_hot([4])[0], one_hot([1])[0], one_hot([0])[0], one_hot([7])[0]])
    pc = A.coarse_probabilities(logits, P)
    assert pc.shape == (4, 5)
    np.testing.assert_allclose(pc.sum(axis=1), 1.0, atol=1e-12)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    # whole-second bin keeps all its mass (+ half of its two half-second neighbours)
    assert pc[0, 2] == pytest.approx(p[0, 4] + 0.5 * p[0, 3] + 0.5 * p[0, 5])
    # a −1.5 s peak splits evenly between −2 and −1 (up to the residual softmax mass of the other bins)
    assert pc[1, 0] == pytest.approx(pc[1, 1], abs=1e-4)
    assert pc[1, 0] == pytest.approx(p[1, 0] + 0.5 * p[1, 1])
    assert pc[1, 1] == pytest.approx(p[1, 2] + 0.5 * p[1, 1] + 0.5 * p[1, 3])
    # edge bin −2 only has an inward neighbour
    assert pc[2, 0] == pytest.approx(p[2, 0] + 0.5 * p[2, 1])


def test_acc_coarse5_scores_only_grid_targets():
    # targets: 4 (0 s, grid), 3 (−0.5 s, off grid), 8 (2 s), 2 (−1 s), 0 (−2 s)
    targets = np.array([4, 3, 8, 2, 0])
    preds = [4, 3, 7, 2, 1]
    logits = one_hot(preds)
    mask = A.coarse_mask(targets, P)
    assert mask.tolist() == [True, False, True, True, True]
    assert A.n_coarse5(logits, targets, P, {}) == 4.0
    c = A.coarse_correct(logits, targets, P)
    # row0 exact → correct; row2: a 1.5 s peak splits between 1 s and 2 s and the inner bin (1 s) carries
    # more residual mass → 1 s ≠ 2 s → wrong; row3 exact → correct; row4: −1.5 s → −1 s ≠ −2 s → wrong
    assert c.tolist() == [True, False, True, False]
    assert A.acc_coarse5(logits, targets, P, {}) == pytest.approx(0.5)
    # a prediction already on the grid is scored exactly as the exact-bin accuracy on the grid rows
    assert A.acc_coarse5(one_hot([4, 3, 8, 2, 0]), targets, P, {}) == 1.0
    assert A.acc_coarse5(one_hot([6, 3, 8, 4, 0]), targets, P, {}) == 0.5
    with pytest.raises(ValueError):
        A.acc_coarse5(one_hot([3, 5]), np.array([3, 5]), P, {})  # no grid target
    with pytest.raises(ValueError):
        A.acc_coarse5(np.array([4, 4]), np.array([4, 4]), P, {})  # needs scores, not indices


def test_floor_threshold_values():
    assert A.floor_threshold(1 / 9, 5997, 3.0) == pytest.approx(0.1233, abs=5e-5)
    assert A.floor_threshold(1 / 9, 3000, 3.0) == pytest.approx(0.1283, abs=5e-5)
    with pytest.raises(ValueError):
        A.floor_threshold(1 / 9, 0, 3.0)


def test_floor_arm_and_violation():
    rng = np.random.default_rng(0)
    n = 5997
    ft = rng.integers(0, 9, size=n)
    # chance-level arm: random predictions
    chance_out = one_hot(rng.integers(0, 9, size=n))
    meta = {"floor": {"outputs": chance_out, "targets": ft}}
    arm = A.acc_floor_arm(None, None, P, meta)
    assert 0.09 < arm < 0.13
    assert A.floor_violated(None, None, P, meta) is (arm > A.floor_threshold(1 / 9, n, 3.0))
    # an arm that reads the target is violated
    oracle = {"floor": {"outputs": one_hot(ft), "targets": ft}}
    assert A.acc_floor_arm(None, None, P, oracle) == 1.0
    assert A.floor_violated(None, None, P, oracle) is True
    # chance copied from metrics.chance when the driver passes it
    assert A.floor_violated(None, None, P, {**oracle, "chance": {"acc": 0.999}}) is False
    with pytest.raises(ValueError):
        A.acc_floor_arm(None, None, P, {})


# ----------------------------------------------------------------------------- golden values


def _fixtures():
    """The fixtures the golden file was computed on (seeded; keep in sync with the file)."""
    out = []
    for seed, n, nf, scale in [(0, 300, 120, 1.0), (1, 57, 31, 3.0), (2, 1000, 400, 0.3), (3, 12, 5, 8.0)]:
        rng = np.random.default_rng(seed)
        t = rng.integers(0, 9, size=n)
        o = rng.normal(size=(n, 9)) * scale
        o[np.arange(n), t] += rng.normal(1.0, 1.0, size=n)
        ft = rng.integers(0, 9, size=nf)
        fo = rng.normal(size=(nf, 9)) * scale
        if seed == 3:
            fo[np.arange(nf), ft] += 50.0  # a violated floor
        floor_outputs = np.tile(np.log(np.bincount(t, minlength=9) + 1.0), (n, 1))
        out.append((seed, o.astype(np.float32), t, fo.astype(np.float32), ft, floor_outputs))
    return out


def test_values_match_the_golden_file():
    """Exact equality (not approx): metrics resolved through the eval.py overlay and computed by the driver
    (``evaluate_metrics``, ``intervals.interval``)."""
    from yavih.metrics import intervals as I

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    specs = resolve_metric_specs(BENCH.spec, A)
    protocol = dict(BENCH.spec.protocol)
    assert set(specs["metrics"]) >= {"acc_tol1", "mae_seconds", "acc_coarse5", "n_coarse5"}
    assert set(specs["controls"]) == {"acc_floor_arm", "floor_violated"}
    for seed, o, t, fo, ft, flo in _fixtures():
        meta = {"floor": {"outputs": fo, "targets": ft, "n_floor": len(ft)}, "chance": dict(BENCH.spec.metrics.chance),
                "bootstrap": {"samples": 300, "seed": seed}, "n_eval": len(t), "n_train": 100}
        ci = {"acc": "binomial", "acc_tol1": "binomial", "acc_coarse5": "bootstrap_clip", "acc_floor_arm": "bootstrap_clip"}
        got = evaluate_metrics(specs, outputs=o, targets=t, protocol=protocol, split_meta=meta, floor_outputs=flo, ci=ci)
        got = json.loads(json.dumps(got))
        want = golden[str(seed)]
        for block in ("metrics", "floors", "controls", "ci95"):
            assert got[block] == want[block], (seed, block)
        extra = {}
        for kind, name in [("binomial", "acc_floor_arm"), ("bootstrap_clip", "acc_tol1"), ("binomial", "acc_coarse5")]:
            extra[f"{kind}:{name}"] = [float(x) for x in I.interval(kind, name, o, t, protocol, meta, spec=A.METRICS[name])]
        folds = {"f0": {"outputs": o[: len(t) // 2], "targets": t[: len(t) // 2]}, "f1": {"outputs": o[len(t) // 2:], "targets": t[len(t) // 2:]}}
        extra["pooled_binomial:acc_tol1"] = [float(x) for x in I.interval("pooled_binomial", "acc_tol1", o, t, protocol, {"folds": folds}, spec=A.METRICS["acc_tol1"])]
        assert extra == want["extra"], seed


def test_overlay_declares_intervals_and_registry_is_generic():
    from yavih.metrics import registry as R

    for name in A.METRICS:
        assert name not in R.REGISTRY, f"{name} is avsync's; it must not be a core registry metric"
    assert A.METRICS["mae_seconds"].higher_is_better is False and not A.METRICS["mae_seconds"].ci
    assert A.METRICS["acc_tol1"].ci == {"binomial", "bootstrap_clip", "pooled_binomial"}
    assert BENCH.spec.protocol_extra_keys and A.PROTOCOL_KEYS == {"nominal_window_start", "curation"}
