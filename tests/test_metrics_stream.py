"""``yavih.metrics.stream`` against the reference evaluators of the online tasks.

The reference functions below are transcribed from LSTR (amazon-research/long-short-term-transformer,
``evaluation/metrics.py`` and ``postprocessing.py``, Apache-2.0), on sklearn."""

from __future__ import annotations

import numpy as np
import pytest

from yavih.metrics import stream as M

sk = pytest.importorskip("sklearn.metrics")


def ref_cap(y_true, y_score):
    y_true_sorted = y_true[np.argsort(-y_score, kind="stable")]
    tp = y_true_sorted.astype(float)
    fp = np.abs(y_true_sorted.astype(float) - 1)
    tps, fps = np.cumsum(tp), np.cumsum(fp)
    ratio = np.sum(tp == 0) / np.sum(tp)
    cprec = tps / (tps + fps / (ratio + np.finfo(float).eps) + np.finfo(float).eps)
    return np.sum(cprec[tp == 1]) / np.sum(tp)


def ref_perframe(gt, pred, ignore_index, metric, ambiguous=None):
    if ambiguous is not None:
        valid = np.where(gt[:, ambiguous] != 1)[0]
        gt, pred = gt[valid], pred[valid]
    fn = sk.average_precision_score if metric == "AP" else ref_cap
    aps = [fn(gt[:, c], pred[:, c]) for c in range(gt.shape[1]) if c not in {0, ignore_index} and np.any(gt[:, c])]
    return float(np.mean(aps))


def _data(seed=0, m=600, c=6):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, c, size=m)
    gt = np.eye(c)[lab]
    pred = rng.random((m, c)) + 0.8 * gt
    pred[::7] = np.round(pred[::7], 1)  # ties
    return gt, pred


def test_weighted_ap_is_sklearn_and_repeats_rows():
    gt, pred = _data()
    for c in range(1, gt.shape[1]):
        assert M.weighted_ap(pred[:, c], gt[:, c]) == pytest.approx(sk.average_precision_score(gt[:, c], pred[:, c]), abs=1e-12)
    w = np.random.default_rng(1).integers(0, 4, size=gt.shape[0])
    rep = np.repeat(np.arange(gt.shape[0]), w)
    assert M.weighted_ap(pred[:, 2], gt[:, 2], w) == pytest.approx(sk.average_precision_score(gt[rep, 2], pred[rep, 2]), abs=1e-12)


def test_calibrated_ap_is_the_reference_and_repeats_rows():
    gt, pred = _data(2)
    pred = pred + np.random.default_rng(3).random(pred.shape) * 1e-6  # no ties: the reference's order is then unique
    for c in range(1, gt.shape[1]):
        assert M.weighted_calibrated_ap(pred[:, c], gt[:, c]) == pytest.approx(ref_cap(gt[:, c], pred[:, c]), abs=1e-12)
    w = np.random.default_rng(4).integers(0, 4, size=gt.shape[0])
    rep = np.repeat(np.arange(gt.shape[0]), w)
    assert M.weighted_calibrated_ap(pred[:, 1], gt[:, 1], w) == pytest.approx(ref_cap(gt[rep, 1], pred[rep, 1]), abs=1e-9)


def test_perframe_map_matches_the_reference_with_ambiguous_frames():
    gt, pred = _data(5, c=8)
    amb = 7  # the ambiguous column (THUMOS: 21)
    assert M.perframe_map(pred, gt, ignore=(0, amb), valid=gt[:, amb] != 1) == pytest.approx(ref_perframe(gt, pred, amb, "AP", ambiguous=amb), abs=1e-12)
    p2 = pred + np.random.default_rng(6).random(pred.shape) * 1e-6
    assert M.calibrated_ap(p2, gt, ignore=(0,)) == pytest.approx(ref_perframe(gt, p2, 0, "cAP"), abs=1e-12)


def test_topk_recall_and_accuracy():
    s = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.2, 0.8], [0.5, 0.4, 0.1]])
    y = np.array([0, 0, 2, 1])
    assert M.class_mean_topk_recall(s, y, k=1) == pytest.approx(np.mean([0.5, 1.0, 0.0]))
    assert M.class_mean_topk_recall(s, y, k=2) == pytest.approx(np.mean([1.0, 1.0, 1.0]))
    assert M.class_mean_topk_recall(s, y, k=1, classes=[0, 2]) == pytest.approx(0.75)
    assert M.segment_accuracy(s, np.array([0, 1, 2, -1])) == 1.0


def test_perstage_and_bin_map_and_recording_interval():
    gt, pred = _data(7)
    rec = np.repeat(np.arange(30), 20)
    st = M.perstage_map(pred, gt, rec, calibrated=False)
    assert set(st) == {f"{k * 10}-{(k + 1) * 10}" for k in range(10)} and all(0 <= v <= 1 for v in st.values())
    v = M.bin_macro_ap(pred, gt, classes=[1, 2, 3])
    lo, hi = M.recording_interval(lambda w: M.bin_macro_ap(pred, gt, classes=[1, 2, 3], w=w), rec, samples=200, seed=0)
    assert lo <= v <= hi and hi - lo < 0.2


# ----------------------------------------------------------------------------- EPIC-KITCHENS-100 anticipation
# The reference below is epic-kitchens/C3-Action-Anticipation @ 73b8a00 evaluate_anticipation_json_ek100.py
# (compute_action_scores, _scores_dict_to_ranks, mean_topk_recall), reproduced on numpy for the comparison.


def _ref_softmax(xs):
    m = np.max(xs, axis=1).reshape((-1, 1))
    e = np.exp(xs - m)
    return e / np.sum(e, axis=1).reshape((-1, 1))


def _ref_top_scores(scores, top_n=100):
    idx = np.argsort(scores)[:, ::-1][:, :top_n]
    return idx, scores[np.arange(0, len(scores)).reshape(-1, 1), idx]


def _ref_action_ranks(verb_scores, noun_scores, top_n=100):
    top_verbs, top_verb_scores = _ref_top_scores(verb_scores, top_n)
    top_nouns, top_noun_scores = _ref_top_scores(noun_scores, top_n)
    mat = _ref_softmax(top_verb_scores)[:, :, np.newaxis] * _ref_softmax(top_noun_scores)[:, np.newaxis, :]
    n = mat.shape[0]
    ranks = mat.reshape(n, -1).argsort(axis=-1)[:, ::-1]
    vi, ni = np.unravel_index(ranks[:, :top_n], shape=mat.shape[1:])
    seg = np.arange(0, n).reshape(-1, 1)
    verbs, nouns = top_verbs[seg, vi], top_nouns[seg, ni]
    scores = mat.reshape(n, -1)[seg, ranks[:, :top_n]]
    out = []
    for v, nn, s in zip(verbs, nouns, scores, strict=True):  # _scores_dict_to_ranks of the {verb*1000+noun: score} dicts
        ids = v * 1000 + nn
        out.append(ids[np.argsort(s)[::-1]])
    return np.array(out)


def _ref_mean_topk_recall(rankings, labels, k=5):
    recalls = []
    for c in np.unique(labels):
        idx = labels == c
        recalls.append((rankings[idx][:, :k] == labels[idx].reshape(-1, 1)).max(1).mean())
    return np.mean(recalls)


def test_epic_anticipation_matches_the_official_evaluator():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "src" / "yavih" / "benchmarks" / "epic100_anticipation_stream" / "eval.py"
    spec = importlib.util.spec_from_file_location("epic_eval_ref_test", p)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    rng = np.random.default_rng(0)
    n = 300
    yv, yn = rng.integers(0, 97, n), rng.integers(0, 300, n)
    pv = rng.dirichlet(np.ones(97) * 0.3, n)
    pn = rng.dirichlet(np.ones(300) * 0.3, n)
    pv[np.arange(0, n, 3), yv[::3]] += 1.0  # partly informative
    pn[np.arange(0, n, 2), yn[::2]] += 1.0
    pv, pn = pv / pv.sum(1, keepdims=True), pn / pn.sum(1, keepdims=True)
    lv, ln = np.log(pv), np.log(pn)
    meta = {"unseen": np.zeros(n, bool), "tail_verb": np.zeros(n, bool), "tail_noun": np.zeros(n, bool)}
    out, tg, sm = {"verb": pv, "noun": pn}, {"verb": yv, "noun": yn}, {"meta": meta}
    assert E.METRICS["mt5r_verb_gru"].fn(out, tg, {}, sm) == pytest.approx(_ref_mean_topk_recall(lv.argsort(1)[:, ::-1], yv))
    assert E.METRICS["mt5r_noun_gru"].fn(out, tg, {}, sm) == pytest.approx(_ref_mean_topk_recall(ln.argsort(1)[:, ::-1], yn))
    want = _ref_mean_topk_recall(_ref_action_ranks(lv, ln), yv * 1000 + yn)
    assert E.METRICS["mt5r_action_linear"].fn(out, tg, {}, sm) == pytest.approx(want)
