"""The ``knn`` scorer: a training-free classification readout in the ``clf`` family (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from yavih import testing as T
from yavih.readouts import knn, registry


def test_registered_as_a_stateless_member_of_the_clf_family():
    s = registry.get("knn")
    assert s.kind == "stateless" and s.family == "clf" and not s.gpu and s.families() == frozenset({"knn", "clf"})
    assert s.hash_config(s.pinned_config()) == registry.get("knn").hash_config(s.pinned_config())


def test_vote_weights_the_nearest_neighbours():
    train = np.eye(3, dtype=np.float32)
    votes = knn._train_matrix(np.array([0, 1, 2]), 3)
    ev = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]], dtype=np.float32)
    out = knn.vote(ev, train, votes, k=2)
    assert out.shape == (2, 3) and np.allclose(out.sum(axis=1), 1.0)
    assert out[0].argmax() == 0 and out[1].argmax() == 1 and out[1, 2] == 0.0  # only the 2 nearest vote
    multi = knn._train_matrix(np.array([[1, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32), 3)
    assert np.allclose(knn.vote(ev[:1], train, multi, k=1)[0], [1, 1, 0])
    assert knn._train_matrix(np.array([-1, 2]), 3).tolist() == [[0, 0, 0], [0, 0, 1]]  # an unlabelled row has no vote


def test_ties_resolve_to_the_lower_train_index():
    train = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out = knn.vote(np.array([[1.0, 0.0]], dtype=np.float32), train, knn._train_matrix(np.array([0, 1, 1]), 2), k=1)
    assert out.tolist() == [[1.0, 0.0]]


def test_pooled_features_are_mean_pooled_and_normalized():
    class Reader:
        D = 2

        def rows(self, idx):
            return np.array([[[3.0, 0.0], [3.0, 8.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32)[np.asarray(idx)]

    x = knn.pooled(Reader(), np.array([0, 1]))
    assert np.allclose(x, [[0.6, 0.8], [0.0, 0.0]])


def test_knn_scores_the_synthetic_clf_fixture(tmp_path, yavih_home):
    from yavih.benchmark import Benchmark
    from yavih.extract import run_extract
    from yavih.io import validate_features
    from yavih.readouts import probe

    folder = T.make_synthetic_benchmark(yavih_home / "benchmarks", "knn", n_train=24, n_eval=12, media="tiny")
    bench = Benchmark.from_dir(folder)
    assert bench.spec.readout == "knn" and bench.spec.metrics.headline == "acc"
    headline = {}
    for kind, make in (("oracle", T.oracle_bridge), ("random", T.random_bridge)):
        run = tmp_path / kind
        for split in ("train", "val"):
            assert run_extract(bench=bench, bridge=make(bench, D=16), split=split, shard=(0, 1), out=run, canaries=0)["status"] == "ok"
            assert validate_features(run / "features" / split, bench=bench).ok
        res = probe(run, seeds=(0,), write_results=False, bench=str(folder))
        assert res[0].status == "ok", res[0].reason
        rec = res[0].result
        assert rec["probe"]["config"] == {"scorer": "knn", "version": knn.VERSION, "kind": "stateless"}
        assert set(rec["metrics"]) == {"acc", "n_eval"} and "acc_floor" in rec["floors"]
        headline[kind] = rec["metrics"]["acc"]
    assert headline["oracle"] >= knn.CONFORMANCE["oracle_min"] and headline["random"] <= knn.CONFORMANCE["random_max"], headline
