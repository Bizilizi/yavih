"""``knn`` readout: a training-free classification readout in the ``clf`` family. numpy only.

Every clip's ``[N, D]`` features are mean-pooled over the tokens and L2-normalized; an eval clip's class scores are
the votes of its ``K`` nearest train clips by cosine similarity, each weighted by ``exp(similarity / TEMPERATURE)``
and normalized to sum to 1 (the weighted kNN of Wu et al. 2018 and DINO's evaluation, Caron et al. 2021): a one-hot
vote for single-label targets, the neighbours' multi-hot rows for multi-label ones (``multi_sets`` train targets as
``clf`` builds them). The scores are ``[n_eval, C]``, so every ``clf`` metric, floor, interval, control and sample
format applies unchanged (``family = "clf"``); ``splits.folds`` runs one vote per held-out fold.

Nothing is trained: the result depends only on the features, which makes ``knn`` a cheap first look at a bridge and a
check on whether a probe's gain comes from the head. ``K``, ``TEMPERATURE`` and the pooling are part of ``VERSION``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from yavih.readouts._targets import clf_targets
from yavih.readouts.base import ProbeRefused, ReadoutContext, evaluate_metrics
from yavih.readouts.registry import Scorer

__all__ = ["SCORER", "VERSION", "K", "TEMPERATURE", "run_knn", "pooled", "vote"]

#: The scorer version (enters ``eval_hash`` through ``scorer_config_hash``); a change of ``K``, ``TEMPERATURE``, the
#: pooling or the vote bumps it.
VERSION: int = 1
K: int = 20
TEMPERATURE: float = 0.07
#: rows per chunk when pooling and voting (bounds memory to ``CHUNK x n_train`` similarities)
CHUNK: int = 1024
#: the synthetic end-to-end band (``tests/test_e2e_synthetic.py``): oracle headline >= oracle_min, random <= random_max
CONFORMANCE: dict[str, float] = {"oracle_min": 0.9, "random_max": 0.5}


def pooled(reader: Any, rows: np.ndarray) -> np.ndarray:
    """``float32 [n, D]``: the mean over tokens of each row's features, L2-normalized (a zero row stays zero)."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((rows.size, int(reader.D)), dtype=np.float32)
    for s in range(0, rows.size, CHUNK):
        x = np.asarray(reader.rows(rows[s:s + CHUNK]), dtype=np.float32).mean(axis=1)
        n = np.linalg.norm(x, axis=1, keepdims=True)
        out[s:s + CHUNK] = x / np.maximum(n, 1e-12)
    return out


def _train_matrix(train_t: Any, num_classes: int) -> np.ndarray:
    """``float32 [n_train, C]`` vote rows: one-hot for single-label targets (``-1`` = no vote), the multi-hot rows
    otherwise (a ``{set: matrix}`` is the union of its sets)."""
    if isinstance(train_t, dict):
        return np.maximum.reduce([np.asarray(m, dtype=np.float32) for m in train_t.values()])
    t = np.asarray(train_t)
    if t.ndim == 2:
        return t.astype(np.float32)
    out = np.zeros((t.shape[0], num_classes), dtype=np.float32)
    ok = t >= 0
    out[np.flatnonzero(ok), t[ok].astype(np.int64)] = 1.0
    return out


def vote(eval_x: np.ndarray, train_x: np.ndarray, train_votes: np.ndarray, *, k: int = K, temperature: float = TEMPERATURE) -> np.ndarray:
    """``float32 [n_eval, C]``: the temperature-weighted vote of each eval row's ``k`` most similar train rows.
    Ties in similarity resolve to the lower train index (a stable sort), so the result is deterministic."""
    n_train = train_x.shape[0]
    if n_train == 0:
        raise ProbeRefused("E_SPLIT_MISSING", "knn: no train rows to vote")
    k = min(int(k), n_train)
    out = np.empty((eval_x.shape[0], train_votes.shape[1]), dtype=np.float32)
    for s in range(0, eval_x.shape[0], CHUNK):
        sim = eval_x[s:s + CHUNK] @ train_x.T
        idx = np.argsort(-sim, axis=1, kind="stable")[:, :k]
        w = np.exp((np.take_along_axis(sim, idx, axis=1) - 1.0) / temperature)  # shifted by the max cosine: no overflow
        scores = np.einsum("bk,bkc->bc", w, train_votes[idx])
        out[s:s + CHUNK] = scores / np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
    return out


def _rows(ctx: ReadoutContext, splits: list[str], *, num_classes: int, role: str) -> tuple[np.ndarray, Any, np.ndarray, list[str]]:
    from yavih.readouts.clf import _concat_targets

    xs, targets, rows, ids = [], [], [], []
    for s in splits:
        r = ctx.present_rows(s)
        xs.append(pooled(ctx.reader(s), r))
        hooked = ctx.hook_targets(s, r)
        targets.append(hooked if hooked is not None else clf_targets(ctx, s, r, num_classes=num_classes, role=role))
        rows.append(r)
        ids.extend(ctx.clip_ids(s, r))
    x = np.concatenate(xs, axis=0) if xs else np.empty((0, 0), np.float32)
    return x, _concat_targets(targets), (np.concatenate(rows) if rows else np.empty((0,), np.int64)), ids


def _arm(ctx: ReadoutContext, train_splits: list[str], eval_splits: list[str], *, num_classes: int) -> dict[str, Any]:
    from yavih.readouts.clf import _floor_outputs

    t0 = time.time()
    train_x, train_t, _, _ = _rows(ctx, train_splits, num_classes=num_classes, role="train")
    ev_x, eval_t, eval_rows, eval_ids = _rows(ctx, eval_splits, num_classes=num_classes, role="eval")
    if len(ev_x) == 0:
        raise ProbeRefused("E_SPLIT_MISSING", f"no present eval rows in {eval_splits}")
    ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)
    t1 = time.time()
    train_votes = _train_matrix(train_t, num_classes)
    outputs = vote(ev_x, train_x, train_votes)
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t1)
    floor = _floor_outputs(train_t, len(ev_x), protocol=ctx.protocol, label_kind=ctx.label_kind, num_classes=num_classes)
    return {"train_targets": train_t, "train_x": train_x, "train_votes": train_votes, "n_train": int(len(train_x)), "outputs": outputs, "targets": eval_t, "rows": eval_rows,
            "clip_ids": eval_ids, "floor_outputs": floor, "eval_splits": list(eval_splits)}


def _record() -> dict[str, Any]:
    return {"k": K, "temperature": TEMPERATURE, "pooling": "mean over tokens", "similarity": "cosine"}


def _meta(ctx: ReadoutContext, num_classes: int) -> dict[str, Any]:
    from yavih.readouts.clf import _base_meta

    return {**_base_meta(ctx, num_classes), "readout": "knn"}


def run_knn(ctx: ReadoutContext) -> dict:
    """Vote + score one seed (the result does not depend on the seed); the ``clf`` blocks of ``result.json``."""
    num_classes = len(ctx.bench.classes())
    if ctx.roles["folds"]:
        return _run_folds(ctx, num_classes)
    arm = _arm(ctx, ctx.roles["train"], ctx.roles["eval"], num_classes=num_classes)
    split_meta = _meta(ctx, num_classes)
    split_meta.update(n_train=arm["n_train"], n_eval=int(len(arm["rows"])), n_val=0, train_targets=arm["train_targets"],
                      eval_rows=arm["rows"], eval_clip_ids=arm["clip_ids"])
    if isinstance(arm["targets"], dict):
        split_meta["sets"] = arm["targets"]
    samples = [{"role": "eval", "outputs": arm["outputs"], "targets": arm["targets"], "rows": arm["rows"], "clip_ids": arm["clip_ids"]}]
    if ctx.roles["floor"]:
        fl_x, fl_t, fl_rows, fl_ids = _rows(ctx, ctx.roles["floor"], num_classes=num_classes, role="floor")
        fl_out = vote(fl_x, arm["train_x"], arm["train_votes"]) if len(fl_x) else np.empty((0, num_classes), np.float32)
        split_meta["floor"] = {"outputs": fl_out, "targets": fl_t, "n_floor": int(len(fl_x)), "rows": fl_rows, "clip_ids": fl_ids}
        samples.append({"role": "floor", "outputs": fl_out, "targets": fl_t, "rows": fl_rows, "clip_ids": fl_ids})
    blocks = evaluate_metrics(ctx.metric_specs, outputs=arm["outputs"], targets=arm["targets"], protocol=ctx.protocol,
                              split_meta=split_meta, floor_outputs=arm["floor_outputs"], ci=ctx.ci,
                              scorer=ctx.scorer(arm["eval_splits"], arm["rows"], arm["targets"]))
    rec = _record()
    return {**blocks, "probe": {}, "fit": {**rec, "n_train": arm["n_train"]}, "eval_outputs": arm["outputs"], "eval_rows": arm["rows"],
            "extra_arrays": {}, "backend": "ram", "timing": {k: ctx.timing[k] for k in ("materialize_s", "score_s") if k in ctx.timing},
            "samples": samples if ctx.log_samples else None}


def _run_folds(ctx: ReadoutContext, num_classes: int) -> dict:
    """One vote per held-out fold (train = every other fold); outputs pooled in fold order, as ``clf`` pools them."""
    from yavih.readouts.clf import _concat_targets

    folds = list(ctx.roles["folds"])
    per_fold: dict[str, dict[str, Any]] = {}
    outs, tgts, rows, fold_idx, floors, train_ts = [], [], [], [], [], []
    for k, f in enumerate(folds):
        ctx.check_interrupt()
        arm = _arm(ctx, [g for g in folds if g != f], [f], num_classes=num_classes)
        per_fold[f] = {"outputs": arm["outputs"], "targets": arm["targets"], "rows": arm["rows"], "clip_ids": arm["clip_ids"], "n_train": arm["n_train"]}
        outs.append(arm["outputs"])
        tgts.append(arm["targets"])
        rows.append(arm["rows"])
        fold_idx.append(np.full(len(arm["rows"]), k, dtype=np.int64))
        floors.append(np.asarray(arm["floor_outputs"]))
        train_ts.append(arm["train_targets"])
    outputs, targets, all_rows = np.concatenate(outs, axis=0), _concat_targets(tgts), np.concatenate(rows)
    split_meta = _meta(ctx, num_classes)
    split_meta.update(n_train=int(round(float(np.mean([p["n_train"] for p in per_fold.values()])))), n_eval=int(len(all_rows)), n_val=0,
                      train_targets=np.concatenate(train_ts, axis=0), eval_rows=all_rows,
                      eval_clip_ids=[c for p in per_fold.values() for c in p["clip_ids"]], folds=per_fold, n_folds=len(folds))
    blocks = evaluate_metrics(ctx.metric_specs, outputs=outputs, targets=targets, protocol=ctx.protocol, split_meta=split_meta,
                              floor_outputs=np.concatenate(floors, axis=0), ci=ctx.ci, scorer=ctx.scorer(folds, all_rows, targets))
    folds_used = np.concatenate(fold_idx)
    return {**blocks, "probe": {}, "fit": {**_record(), "folds": folds}, "eval_outputs": outputs, "eval_rows": all_rows,
            "extra_arrays": {"eval_fold_index": folds_used}, "backend": "ram",
            "timing": {k: ctx.timing[k] for k in ("materialize_s", "score_s") if k in ctx.timing},
            "samples": ([{"role": "eval", "outputs": outputs, "targets": targets, "rows": all_rows, "clip_ids": split_meta["eval_clip_ids"],
                          "folds": [folds[int(i)] for i in folds_used]}] if ctx.log_samples else None)}


def _floor_outputs(train_targets: Any, n_eval: int, *, protocol: dict, label_kind: str, num_classes: int | None = None) -> np.ndarray:
    from yavih.readouts.clf import _floor_outputs as clf_floor

    return clf_floor(train_targets, n_eval, protocol=protocol, label_kind=label_kind, num_classes=num_classes)


SCORER = Scorer(
    name="knn", version=VERSION, module=__name__, kind="stateless", family="clf", label_kinds=frozenset({"single", "multi", "multi_sets"}),
    requires_t_bins=False, requires_grid=False, dense=False, score=run_knn, floor_outputs=_floor_outputs, needs_gpu=False,
)
