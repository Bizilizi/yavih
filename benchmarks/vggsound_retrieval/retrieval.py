"""Training-free cross-modal retrieval over a shared embedding space: the folder's own scorer and its metrics.

The bridge returns per row one ``audio_emb [D]`` and one ``video_emb [D]`` in the model's own shared space (a
contrastive model's projections, say; the folder's ``output_contract.isolation`` makes each depend on its own stream
only, ``bridge check`` ``E_MODALITY_LEAK``). Every eval row is a query of each direction against the gallery of all
eval rows of the other modality, by cosine similarity; the output of a row is the rank of its true pair in both
directions (:func:`ranks`, the average tie rank, ``float64 [n, 2]`` = ``(a2v, v2a)``). Metrics: ``r@1_a2v`` …
``medr_v2a`` and the headline ``r@1`` (the mean of both directions' R@1); intervals resample queries with the gallery
fixed. No floor: chance is ``k / n`` (``metrics.chance``).

Rank convention: ``1 + #(candidates scored strictly higher) + #(other candidates scored equal) / 2``. Ties are
measure-zero for real embeddings, but a degenerate query (a zero embedding of a silent clip, a constant output) ties
with the whole gallery and then ranks in the middle of it rather than first.

``readout_params.subsets: {name: column}`` names manifest flag columns (``0`` / ``1``); for each, the flagged rows
form a query set and gallery of their own, ranked separately and handed to the metrics as
``split_meta["subsets"][name] = {"mask": bool [n], "ranks": float64 [n, 2] (0 outside the subset)}``; the subset
metrics are named ``<metric>_<subset>``. A model without a shared space has no bridge for this scorer: it is not
applicable, not scored. Module scope: stdlib, numpy, yavih core (a candidate for the core: ``yavih bench extensions``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from yavih.metrics._common import as_array
from yavih.metrics.registry import MetricSpec
from yavih.readouts.base import ProbeRefused, evaluate_metrics
from yavih.readouts.registry import Scorer

__all__ = ["VERSION", "KEYS", "DIRECTIONS", "RECALL_KS", "POPULATION_EVAL", "normalized", "ranks", "hits", "recall_at",
           "median_rank", "r1_mean_per_clip", "rank_pairs", "subset_masks", "run", "make_scorer", "make_metrics"]

#: The scorer version (enters ``eval_hash``); a change of the similarity, the rank convention or the subset treatment
#: bumps it.
VERSION: int = 1
KEYS: tuple[str, str] = ("audio_emb", "video_emb")
DIRECTIONS: tuple[str, ...] = ("a2v", "v2a")
RECALL_KS: tuple[int, ...] = (1, 5, 10)
#: queries per similarity chunk (``CHUNK x n`` similarities in memory at a time)
CHUNK = 1024
#: eval rows of the conformance population
POPULATION_EVAL = 256
_COL = {"a2v": 0, "v2a": 1}


# ----------------------------------------------------------------------------- ranks and metric functions


def normalized(x: Any) -> np.ndarray:
    """``float64 [n, D]`` rows scaled to unit L2 norm (a zero row stays zero)."""
    a = as_array(x, np.float64).reshape(len(x), -1)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-12)


def ranks(query: Any, gallery: Any) -> np.ndarray:
    """``float64 [n]``: for query ``i`` the rank of gallery item ``i`` by cosine similarity among the ``n`` gallery
    items (the average tie rank). Inputs are L2-normalized here."""
    q, g = normalized(query), normalized(gallery)
    if q.shape != g.shape:
        raise ValueError(f"ranks: query {q.shape} and gallery {g.shape} must hold the same rows and dimension")
    n = q.shape[0]
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, CHUNK):
        sim = q[s:s + CHUNK] @ g.T
        true = sim[np.arange(sim.shape[0]), np.arange(s, s + sim.shape[0])]
        out[s:s + CHUNK] = 1 + (sim > true[:, None]).sum(axis=1) + ((sim == true[:, None]).sum(axis=1) - 1) / 2.0
    return out


def _col(outputs: Any, direction: str) -> np.ndarray:
    r = as_array(outputs)
    if r.ndim != 2 or r.shape[1] != 2:
        raise ValueError(f"retrieval outputs must be [n, 2] ranks (a2v, v2a), got {r.shape}")
    return r[:, _COL[direction]].astype(np.float64)


def hits(outputs: Any, k: int, direction: str) -> np.ndarray:
    """``float64 [n]``: 1 where the true pair is within the top ``k``."""
    return (_col(outputs, direction) <= int(k)).astype(np.float64)


def recall_at(outputs: Any, k: int, direction: str) -> float:
    return float(hits(outputs, k, direction).mean())


def median_rank(outputs: Any, direction: str) -> float:
    return float(np.median(_col(outputs, direction)))


def r1_mean_per_clip(outputs: Any) -> np.ndarray:
    """``float64 [n]``: the mean of a query's two R@1 hits (``r@1``'s per-clip quantity)."""
    return 0.5 * (hits(outputs, 1, "a2v") + hits(outputs, 1, "v2a"))


def rank_pairs(audio: np.ndarray, video: np.ndarray) -> np.ndarray:
    """``float64 [n, 2]``: ``(a2v, v2a)`` ranks of each row's true pair."""
    return np.stack([ranks(audio, video), ranks(video, audio)], axis=1)


# ----------------------------------------------------------------------------- the driver


def _flag(v: Any) -> bool:
    if isinstance(v, str):
        if v.strip() in ("0", "1"):
            return v.strip() == "1"
        raise ProbeRefused("E_LABELS", f"retrieval: a subset flag must be '0' or '1', got {v!r}")
    return bool(int(v))


def subset_masks(ctx: Any, split: str, rows: np.ndarray) -> dict[str, np.ndarray]:
    """``{name: bool [n]}`` of the ``readout_params.subsets`` flag columns over ``rows`` of ``split``."""
    from yavih.readouts.base import _manifest_index

    subsets = dict(ctx.spec.readout_params.get("subsets") or {})
    if not subsets:
        return {}
    man = ctx.manifest(split)
    idx = _manifest_index(man, rows)
    out = {}
    for name, column in subsets.items():
        try:
            out[str(name)] = np.asarray([_flag(man.extra[i][column]) for i in idx], dtype=bool)
        except KeyError as e:
            raise ProbeRefused("E_LABELS", f"retrieval: subset {name!r} names column {column!r}, absent from the manifest") from e
    return out


def _embeddings(ctx: Any, splits: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    a, v, rows, ids, masks = [], [], [], [], {}
    for s in splits:
        r = ctx.present_rows(s)
        if len(r) == 0:
            continue
        reader = ctx.reader(s)
        a.append(np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in reader.values(KEYS[0], r, codecs=ctx.bench.codecs)]))
        v.append(np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in reader.values(KEYS[1], r, codecs=ctx.bench.codecs)]))
        rows.append(r)
        ids.extend(ctx.clip_ids(s, r))
        for name, m in subset_masks(ctx, s, r).items():
            masks.setdefault(name, []).append(m)
    if not rows:
        raise ProbeRefused("E_SPLIT_MISSING", f"no present eval rows in {splits}")
    A, V = np.concatenate(a, axis=0), np.concatenate(v, axis=0)
    if A.shape != V.shape:
        raise ProbeRefused("E_TARGETS", f"retrieval: audio_emb {A.shape} and video_emb {V.shape} differ in shape")
    return A, V, np.concatenate(rows), ids, {k: np.concatenate(m) for k, m in masks.items()}


def _meta(ctx: Any, rows: np.ndarray, ids: list[str], subsets: Mapping[str, Any], seed: int) -> dict[str, Any]:
    return {"readout": ctx.spec.readout, "n_train": 0, "n_eval": int(len(rows)), "n_val": 0, "n_gallery": int(len(rows)), "eval_rows": rows,
            "eval_clip_ids": ids, "subsets": dict(subsets), "chance": dict(ctx.chance), "bootstrap": dict(ctx.bootstrap or {}), "seed": seed}


def run(ctx: Any) -> dict:
    """Rank and score (the result does not depend on the seed); the ``result.json`` blocks."""
    if ctx.roles["folds"]:
        raise ProbeRefused("E_FOLDS", "the retrieval scorer scores one eval gallery; splits.folds is not supported")
    t0 = time.time()
    audio, video, rows, ids, masks = _embeddings(ctx, ctx.roles["eval"])
    ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)
    t1 = time.time()
    outputs = rank_pairs(audio, video)
    subsets: dict[str, dict[str, Any]] = {}
    for name, m in masks.items():
        sub = np.zeros_like(outputs)
        if m.sum() >= 2:
            sub[m] = rank_pairs(audio[m], video[m])
        subsets[name] = {"mask": m, "ranks": sub, "n": int(m.sum())}
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t1)
    targets = np.arange(len(rows), dtype=np.int64)  # the true pair of query i is gallery item i
    split_meta = _meta(ctx, rows, ids, subsets, ctx.seed)
    blocks = evaluate_metrics(ctx.metric_specs, outputs=outputs, targets=targets, protocol=ctx.protocol, split_meta=split_meta,
                              floor_outputs=None, ci=ctx.ci, scorer=ctx.scorer(ctx.roles["eval"], rows, targets))
    return {**blocks, "probe": {}, "fit": {"similarity": "cosine", "rank": "1 + #(strictly higher) + #(equal, others) / 2", "n_gallery": int(len(rows)),
                                           "subsets": {k: v["n"] for k, v in subsets.items()}},
            "eval_outputs": outputs.astype(np.float32), "eval_rows": rows,
            "extra_arrays": {f"subset_ranks_{k}": v["ranks"] for k, v in subsets.items()}, "backend": "ram",
            "timing": {k: ctx.timing[k] for k in ("materialize_s", "score_s") if k in ctx.timing},
            "samples": ([{"role": "eval", "outputs": outputs, "targets": targets, "rows": rows, "clip_ids": ids}] if ctx.log_samples else None)}


def _validate_spec(readout_params: dict, requires: Any, protocol: dict) -> None:
    subsets = readout_params.get("subsets")
    if not isinstance(subsets, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in subsets.items()):
        raise ValueError(".subsets: {name: flag column} (possibly empty)")


def _population(ctx: Any, tag: str) -> dict[str, Any]:
    """The conformance population: a seeded selection of eval rows with fabricated ranks (skewed towards 1), the
    subset masks of the manifest and their own fabricated ranks."""
    from yavih.conformance import _values, select_rows

    rows, ids, masks = [], [], {}
    for s in ctx.roles["eval"]:
        r = select_rows(ctx.bench.manifest_hash, s, ctx.bench.n_rows(s), POPULATION_EVAL, salt="eval")
        rows.append(r)
        ids.extend(ctx.clip_ids(s, r))
        for name, m in subset_masks(ctx, s, r).items():
            masks.setdefault(name, []).append(m)
    rows_all = np.concatenate(rows)
    n = len(rows_all)

    def fab(salt: str, m: int) -> np.ndarray:
        u = _values(f"{tag}:retrieval:{salt}", (m, 2))
        return 1 + np.floor(u ** 3 * m)

    outputs = fab("ranks", n)
    subsets = {}
    for name, parts in masks.items():
        m = np.concatenate(parts)
        sub = np.zeros_like(outputs)
        if m.sum() >= 2:
            sub[m] = fab(f"subset:{name}", int(m.sum()))
        subsets[name] = {"mask": m, "ranks": sub, "n": int(m.sum())}
    return {"outputs": outputs, "targets": np.arange(n), "split_meta": _meta(ctx, rows_all, ids, subsets, 0), "floor_outputs": None,
            "eval_splits": list(ctx.roles["eval"]), "eval_rows": rows_all}


def make_scorer(name: str, *, module: str, version: int = VERSION) -> Scorer:
    """The folder's stateless retrieval scorer, registered under the folder's ``name``."""
    return Scorer(
        name=name, version=version, module=module, kind="stateless", label_kinds=frozenset({"none"}), param_keys=frozenset({"subsets"}),
        requires_t_bins=False, requires_grid=False, dense=False, score=run, floor_outputs=None, needs_gpu=False,
        validate_spec=_validate_spec, output_keys=frozenset(KEYS), conformance_population=_population,
    )


# ----------------------------------------------------------------------------- metric specs


def _subset(split_meta: Mapping[str, Any], subset: str) -> Mapping[str, Any]:
    return split_meta["subsets"][subset]


def make_metrics(name: str, *, subsets: tuple[str, ...] = ()) -> dict[str, MetricSpec]:
    """``r@1`` (headline: the mean of both directions' R@1, clip bootstrap), ``r@{1,5,10}_{a2v,v2a}`` (Wilson or clip
    bootstrap), ``medr_{a2v,v2a}`` (lower is better) over the full eval gallery, and the same per subset of
    ``subsets`` (``<metric>_<subset>``: the subset's query set and gallery; ``split_meta["subsets"][<subset>]``)."""
    ro = frozenset({name})
    out: dict[str, MetricSpec] = {
        "r@1": MetricSpec("r@1", lambda o, t, p, m: float(r1_mean_per_clip(o).mean()), ro, per_clip=lambda o, t, p, m: r1_mean_per_clip(o),
                          ci=frozenset({"bootstrap_clip"}), description="the mean of r@1_a2v and r@1_v2a"),
    }
    for d in DIRECTIONS:
        for k in RECALL_KS:
            def recall(o, t, p, m, k=k, d=d):
                return recall_at(o, k, d)

            def recall_per_clip(o, t, p, m, k=k, d=d):
                return hits(o, k, d)

            out[f"r@{k}_{d}"] = MetricSpec(f"r@{k}_{d}", recall, ro, per_clip=recall_per_clip, ci=frozenset({"binomial", "bootstrap_clip"}),
                                           description=f"the fraction of {d} queries whose true pair is within the top {k}")

        def medr_full(o, t, p, m, d=d):
            return median_rank(o, d)

        out[f"medr_{d}"] = MetricSpec(f"medr_{d}", medr_full, ro, higher_is_better=False, description=f"the median rank of the true pair, {d}")
    for sub in subsets:
        for d in DIRECTIONS:
            for k in RECALL_KS:
                def fn(o, t, p, m, k=k, d=d, sub=sub):
                    s = _subset(m, sub)
                    return float(hits(s["ranks"], k, d)[s["mask"]].mean())

                def per_clip(o, t, p, m, k=k, d=d, sub=sub):
                    s = _subset(m, sub)
                    return np.where(s["mask"], hits(s["ranks"], k, d), np.nan)

                n = f"r@{k}_{d}_{sub}"
                out[n] = MetricSpec(n, fn, ro, per_clip=per_clip, ci=frozenset({"binomial", "bootstrap_clip"}),
                                    description=f"R@{k} {d} within the {sub} subset (its own query set and gallery)")

            def medr(o, t, p, m, d=d, sub=sub):
                s = _subset(m, sub)
                return median_rank(s["ranks"][s["mask"]], d)

            n = f"medr_{d}_{sub}"
            out[n] = MetricSpec(n, medr, ro, higher_is_better=False, description=f"median rank of the true pair, {d}, within the {sub} subset")
    return out
