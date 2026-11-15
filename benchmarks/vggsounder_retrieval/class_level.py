"""``vggsounder_retrieval``: the folder's scorer, instance-level retrieval (``retrieval.py``) plus class-level metrics.

Instance level (the headline) is ``retrieval.py`` unchanged (byte-identical to ``vggsound_retrieval``'s copy): every
row is a query of each direction against the gallery of all rows of the other modality, by cosine similarity; the
true pair is the query's own clip; ``r@1`` (mean of both directions), ``r@{1,5,10}_{a2v,v2a}``, ``medr_{a2v,v2a}``.

Class level (alongside): the same cosine similarities, the query's own clip removed from the gallery; a retrieved
clip is relevant when it shares at least one audible-and-visible class (the manifest column named by
``readout_params.classes_column``, ``;``-separated) with the query. Two measures, each copied from a reference
implementation (the blocks below say from where and what changed):

- ``cls_map_{a2v,v2a}``: mean average precision over the whole gallery, ``calc_map`` of DCMH (Jiang & Li, CVPR
  2017), the multi-label cross-modal retrieval mAP whose relevance is "shares at least one label". A query without
  any relevant gallery clip counts as AP 0 (DCMH skips it but keeps it in the denominator); ``cls_n_no_relevant``
  counts such queries.
- ``cls_r@{1,5,10}_{a2v,v2a}``: the fraction of queries with at least one relevant clip among the top ``k``, the
  ``c_recall@k`` of Roth et al. (ICML 2020; the deep-metric-learning Recall@k over cosine similarities), with
  "the query's class is among the top-k classes" read as "a top-k clip shares a class with the query".

Order within ties: the instance ranks take the average tie rank (``retrieval.py``); the class-level measures take
``numpy.argsort``'s order, as both references do. Ties are measure-zero for real embeddings.

The class-level quantities reach the metrics as ``split_meta["class_level"] = {"ap": float64 [n, 2], "hits": {k:
float64 [n, 2]}, "n_relevant": int64 [n]}`` (columns ``a2v, v2a``); the scorer's outputs stay the instance ranks
``[n, 2]``. Module scope: stdlib, numpy, yavih core.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from yavih.metrics.registry import MetricSpec
from yavih.readouts.base import ProbeRefused, evaluate_metrics
from yavih.readouts.registry import Scorer

__all__ = ["VERSION", "RECALL_KS", "DIRECTIONS", "SEP", "calc_map", "c_recall_hits", "parse_labels", "class_level", "make_scorer", "make_metrics"]

#: The scorer version (enters ``eval_hash``); a change of the instance or class-level computation bumps it.
VERSION: int = 1
RECALL_KS: tuple[int, ...] = (1, 5, 10)
DIRECTIONS: tuple[str, ...] = ("a2v", "v2a")
SEP = ";"
#: queries per similarity chunk
CHUNK = 512
#: eval rows of the conformance population and the dimension of its generated embeddings
POPULATION_EVAL = 256
POPULATION_DIM = 16


# ----------------------------------------------------------------------------- copied: DCMH calc_map
# Source: DCMH, "Deep Cross-Modal Hashing" (Q.-Y. Jiang and W.-J. Li, CVPR 2017),
#   https://github.com/jiangqy/DCMH-CVPR2017 @ 67d0e84c0425fdac3fad30d67d5a2beb5e345cea,
#   DCMH_tensorflow/DCMH_tensorflow/utils/calc_hammingranking.py :: calc_map. Licence: none stated in the repository.
# Changes: the Hamming distance of binary codes (calc_hammingDist(qB[iter, :], rB)) is replaced by the negative
#   similarity, a row of `sim` (cosine here; the caller removes the query's own clip: similarity -inf and relevance 0,
#   so it sorts last and never counts); the per-query APs are returned besides their mean (0 for a query without a
#   relevant item, which the upstream skips while still dividing by num_query); Python 3 (range for xrange) and
#   np.linspace's sample count cast to int (the upstream passes a float, refused by numpy >= 1.18); 4-space indent.
def calc_map(sim, query_L, retrieval_L):
    # sim: [m, n] similarity of each query to each retrieval item (higher = closer)
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    aps = np.zeros(num_query, dtype=np.float64)
    for iter in range(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = -sim[iter, :]
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, int(tsum))

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        aps[iter] = np.mean(count / (tindex))
        map = map + aps[iter]
    map = map / num_query
    return map, aps


# ----------------------------------------------------------------------------- copied: c_recall@k
# Source: "Revisiting Training Strategies and Generalization Performance in Deep Metric Learning" (K. Roth,
#   T. Milbich, S. Sinha, P. Gupta, B. Ommer, J. P. Cohen, ICML 2020),
#   https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch @ efddbf23ccbe267f055867b4e1c7c6693e2447c9,
#   metrics/c_recall.py :: Metric.__call__ (its neighbours: metrics/__init__.py, the k nearest by cosine similarity).
#   Licence: MIT, Copyright (c) 2020 Karsten Roth.
# Changes: multi-label: a target is the query's set of classes and a recalled prediction a gallery clip's set, so
#   "target in recalled_predictions[:k]" becomes "target shares a class with one of recalled_predictions[:k]"; the
#   per-query hits are returned instead of their mean (recall_at_k = hits.mean()); the neighbours come from the
#   caller's argsort of the cosine similarities with the query's own clip removed (upstream: faiss IndexFlatIP over
#   normalised features, dropping the first neighbour, the query itself in a single-modality gallery).
def c_recall_hits(k, target_labels, k_closest_classes_cosine):
    hits = np.asarray([1 if any(target & recalled for recalled in recalled_predictions[:k]) else 0
                       for target, recalled_predictions in zip(target_labels, k_closest_classes_cosine)], dtype=np.float64)
    return hits


# ----------------------------------------------------------------------------- the class-level computation


def _normalized(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def parse_labels(cells: list[str]) -> tuple[np.ndarray, list[frozenset[str]]]:
    """``(multi-hot uint8 [n, C] over the sorted class names, [frozenset of names] per row)`` of ``;``-joined cells."""
    sets = [frozenset(t for t in str(c).split(SEP) if t) for c in cells]
    names = sorted(set().union(*sets)) if sets else []
    col = {name: j for j, name in enumerate(names)}
    L = np.zeros((len(sets), len(names)), dtype=np.uint8)
    for i, s in enumerate(sets):
        L[i, [col[name] for name in s]] = 1
    return L, sets


def class_level(audio: Any, video: Any, labels: list[str]) -> dict[str, Any]:
    """``{"ap": float64 [n, 2], "hits": {k: float64 [n, 2]}, "n_relevant": int64 [n]}``, columns ``(a2v, v2a)``:
    the class-level quantities of every query over the gallery of the other modality minus the query's own clip."""
    a, v = _normalized(audio), _normalized(video)
    if a.shape != v.shape:
        raise ValueError(f"class_level: audio {a.shape} and video {v.shape} must hold the same rows and dimension")
    L, sets = parse_labels(labels)
    n = a.shape[0]
    if len(sets) != n:
        raise ValueError(f"class_level: {len(sets)} label cells for {n} rows")
    if any(not s for s in sets):
        raise ValueError("class_level: a row without a class")
    Lf = L.astype(np.float32)
    n_rel = ((Lf @ Lf.T) > 0).sum(axis=1).astype(np.int64) - 1
    ap = np.zeros((n, 2), dtype=np.float64)
    hits = {k: np.zeros((n, 2), dtype=np.float64) for k in RECALL_KS}
    kmax = max(RECALL_KS)
    gallery_L = L.copy()
    for j, (q, g) in enumerate(((a, v), (v, a))):
        for s in range(0, n, CHUNK):
            e = min(n, s + CHUNK)
            sim = q[s:e] @ g.T
            sim[np.arange(e - s), np.arange(s, e)] = -np.inf  # the query's own clip sorts last ...
            for i in range(s, e):
                gallery_L[i] = 0  # ... and is never relevant
                _, ap_i = calc_map(sim[i - s:i - s + 1], L[i:i + 1], gallery_L)
                gallery_L[i] = L[i]
                ap[i, j] = ap_i[0]
            closest = [[sets[t] for t in row] for row in np.argsort(-sim, axis=1)[:, :kmax]]
            for k in RECALL_KS:
                hits[k][s:e, j] = c_recall_hits(k, sets[s:e], closest)
    return {"ap": ap, "hits": hits, "n_relevant": n_rel}


# ----------------------------------------------------------------------------- the driver


def _labels(ctx: Any, split: str, rows: np.ndarray) -> list[str]:
    from yavih.readouts.base import _manifest_index

    column = ctx.spec.readout_params["classes_column"]
    man = ctx.manifest(split)
    try:
        return [str(man.extra[i][column]) for i in _manifest_index(man, rows)]
    except KeyError as e:
        raise ProbeRefused("E_LABELS", f"class_level: readout_params.classes_column names {column!r}, absent from the manifest") from e


def _validate_spec(readout_params: dict, requires: Any, protocol: dict) -> None:
    subsets = readout_params.get("subsets")
    if not isinstance(subsets, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in subsets.items()):
        raise ValueError(".subsets: {name: flag column} (possibly empty)")
    if not isinstance(readout_params.get("classes_column"), str) or not readout_params["classes_column"]:
        raise ValueError(".classes_column: the manifest column of each row's ';'-joined classes")


def make_scorer(name: str, *, module: str, retrieval: Any, version: int = VERSION) -> Scorer:
    """The folder's stateless scorer: ``retrieval.py``'s instance ranks as outputs, the class-level quantities in
    ``split_meta["class_level"]``; registered under the folder's ``name``."""

    def run(ctx: Any) -> dict:
        if ctx.roles["folds"]:
            raise ProbeRefused("E_FOLDS", "the retrieval scorer scores one eval gallery; splits.folds is not supported")
        t0 = time.time()
        audio, video, rows, ids, masks = retrieval._embeddings(ctx, ctx.roles["eval"])
        labels: list[str] = []
        for s in ctx.roles["eval"]:
            r = ctx.present_rows(s)
            if len(r):
                labels.extend(_labels(ctx, s, r))
        ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)
        t1 = time.time()
        outputs = retrieval.rank_pairs(audio, video)
        subsets: dict[str, dict[str, Any]] = {}
        for sname, m in masks.items():
            sub = np.zeros_like(outputs)
            if m.sum() >= 2:
                sub[m] = retrieval.rank_pairs(audio[m], video[m])
            subsets[sname] = {"mask": m, "ranks": sub, "n": int(m.sum())}
        try:
            cl = class_level(audio, video, labels)
        except ValueError as e:
            raise ProbeRefused("E_LABELS", str(e)) from e
        ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t1)
        targets = np.arange(len(rows), dtype=np.int64)
        split_meta = {**retrieval._meta(ctx, rows, ids, subsets, ctx.seed), "class_level": cl}
        blocks = evaluate_metrics(ctx.metric_specs, outputs=outputs, targets=targets, protocol=ctx.protocol, split_meta=split_meta,
                                  floor_outputs=None, ci=ctx.ci, scorer=ctx.scorer(ctx.roles["eval"], rows, targets))
        extra = {f"subset_ranks_{k}": v["ranks"] for k, v in subsets.items()}
        extra.update({"class_ap": cl["ap"], "class_n_relevant": cl["n_relevant"], **{f"class_hits_{k}": cl["hits"][k] for k in RECALL_KS}})
        return {**blocks, "probe": {}, "fit": {"similarity": "cosine", "rank": "1 + #(strictly higher) + #(equal, others) / 2", "n_gallery": int(len(rows)),
                                               "class_level": {"relevance": "shares >= 1 class of the classes column", "gallery": "n - 1 (own clip removed)",
                                                               "order": "numpy.argsort (quicksort) of the negative similarity"},
                                               "subsets": {k: v["n"] for k, v in subsets.items()}},
                "eval_outputs": outputs.astype(np.float32), "eval_rows": rows, "extra_arrays": extra, "backend": "ram",
                "timing": {k: ctx.timing[k] for k in ("materialize_s", "score_s") if k in ctx.timing},
                "samples": ([{"role": "eval", "outputs": outputs, "targets": targets, "rows": rows, "clip_ids": ids}] if ctx.log_samples else None)}

    def population(ctx: Any, tag: str) -> dict[str, Any]:
        """The conformance population: a seeded selection of eval rows with their real classes, generated audio and
        video embeddings (each video a noisy copy of its audio, so ranks spread from 1 to n), scored by the real
        instance and class-level computation."""
        from yavih.conformance import _values, select_rows

        rows, ids, labels = [], [], []
        for s in ctx.roles["eval"]:
            r = select_rows(ctx.bench.manifest_hash, s, ctx.bench.n_rows(s), POPULATION_EVAL, salt="eval")
            rows.append(r)
            ids.extend(ctx.clip_ids(s, r))
            labels.extend(_labels(ctx, s, r))
        rows_all = np.concatenate(rows)
        n = len(rows_all)
        audio = _values(f"{tag}:class_level:audio", (n, POPULATION_DIM)) - 0.5
        video = audio + 0.6 * (_values(f"{tag}:class_level:video", (n, POPULATION_DIM)) - 0.5)
        outputs = retrieval.rank_pairs(audio, video)
        split_meta = {**retrieval._meta(ctx, rows_all, ids, {}, 0), "class_level": class_level(audio, video, labels)}
        return {"outputs": outputs, "targets": np.arange(n), "split_meta": split_meta, "floor_outputs": None,
                "eval_splits": list(ctx.roles["eval"]), "eval_rows": rows_all}

    return Scorer(
        name=name, version=version, module=module, kind="stateless", label_kinds=frozenset({"none"}), param_keys=frozenset({"subsets", "classes_column"}),
        requires_t_bins=False, requires_grid=False, dense=False, score=run, floor_outputs=None, needs_gpu=False,
        validate_spec=_validate_spec, output_keys=frozenset(retrieval.KEYS), conformance_population=population,
    )


# ----------------------------------------------------------------------------- metric specs


def make_metrics(name: str) -> dict[str, MetricSpec]:
    """``cls_map_{a2v,v2a}`` (clip bootstrap), ``cls_r@{1,5,10}_{a2v,v2a}`` (Wilson or clip bootstrap) and
    ``cls_n_no_relevant`` from ``split_meta["class_level"]``."""
    ro = frozenset({name})
    col = {"a2v": 0, "v2a": 1}
    out: dict[str, MetricSpec] = {}
    for d in DIRECTIONS:
        def ap_mean(o, t, p, m, d=d):
            return float(np.asarray(m["class_level"]["ap"])[:, col[d]].mean())

        def ap_per_clip(o, t, p, m, d=d):
            return np.asarray(m["class_level"]["ap"], dtype=np.float64)[:, col[d]]

        out[f"cls_map_{d}"] = MetricSpec(f"cls_map_{d}", ap_mean, ro, per_clip=ap_per_clip, ci=frozenset({"bootstrap_clip"}),
                                         description=f"class-level mAP, {d}: relevant = shares an audible-and-visible class (DCMH calc_map)")
        for k in RECALL_KS:
            def hit_mean(o, t, p, m, k=k, d=d):
                return float(np.asarray(m["class_level"]["hits"][k])[:, col[d]].mean())

            def hit_per_clip(o, t, p, m, k=k, d=d):
                return np.asarray(m["class_level"]["hits"][k], dtype=np.float64)[:, col[d]]

            out[f"cls_r@{k}_{d}"] = MetricSpec(f"cls_r@{k}_{d}", hit_mean, ro, per_clip=hit_per_clip, ci=frozenset({"binomial", "bootstrap_clip"}),
                                               description=f"class-level R@{k}, {d}: a top-{k} clip shares a class with the query (Roth et al. c_recall@k)")
    out["cls_n_no_relevant"] = MetricSpec("cls_n_no_relevant", lambda o, t, p, m: float((np.asarray(m["class_level"]["n_relevant"]) == 0).sum()), ro,
                                          higher_is_better=False, description="queries without a relevant gallery clip (AP 0 in cls_map_*)")
    return out
