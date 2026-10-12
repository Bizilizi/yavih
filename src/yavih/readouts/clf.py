"""``clf`` readout: single / multi / multi_sets / folds / floor role.

Head ``attentive_probe_v1`` (``yavih.probes.attentive.build_attentive_probe``) on ``[n, N, D]`` features; targets
``int64 [n]`` or ``float32 [n, C]``. The driver calls every registered fn in ``metrics.list/floors/
controls`` with ``split_meta`` carrying what the clf metric families read
(``yavih.metrics._common``): ``n_train``, ``n_eval``, ``train_targets``, ``chance``, ``bootstrap``, ``sets``
(multi_sets eval matrices), ``floor`` (``{outputs, targets}`` of the floor role) and ``folds``
(``{fold: {outputs, targets}}`` in ``splits.folds`` order).

The folder ``probe.py`` may replace the target construction (``targets`` hook) and compute
the metrics whose spec names ``external`` (``score`` hook, e.g. the official ``vggsounder`` package).
With ``log_samples`` the scored rows are returned as ``samples`` parts.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from yavih.readouts._targets import clf_targets
from yavih.readouts.base import ProbeRefused, ReadoutContext, evaluate_metrics, lazy_module
from yavih.readouts.registry import Readout

_fit = lazy_module("yavih.readouts._fit")  # torch: imported on first use, never by the descriptor

__all__ = ["READOUT", "VERSION", "run_clf"]

#: The readout version: enters ``protocol_hash``; ``yavih.versions.READOUT_VERSION`` reads it here.
VERSION: int = 1


def _build_head(ctx: ReadoutContext, num_classes: int):
    def build(dim: int | None = None):
        from yavih.probes.attentive import build_attentive_probe

        return build_attentive_probe(int(dim or ctx.D), num_classes, dict(ctx.cfg.head))

    return build


def _concat_targets(parts: list[Any]) -> Any:
    if not parts:
        return np.empty((0,), dtype=np.int64)
    if isinstance(parts[0], dict):
        return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    return np.concatenate(parts, axis=0)


def _role_rows(ctx: ReadoutContext, splits: list[str], *, num_classes: int, role: str) -> tuple[_fit.FeatureRows, Any, np.ndarray, list[str]]:
    """FeatureRows + targets + within-split rows + clip_ids for the union of ``splits``."""
    segments, targets, rows, ids = [], [], [], []
    for s in splits:
        r = ctx.present_rows(s)
        segments.append((ctx.reader(s), r))
        hooked = ctx.hook_targets(s, r)
        targets.append(hooked if hooked is not None else clf_targets(ctx, s, r, num_classes=num_classes, role=role))
        rows.append(r)
        ids.extend(ctx.clip_ids(s, r))
    feats = _fit.FeatureRows(segments)
    return feats, _concat_targets(targets), (np.concatenate(rows) if rows else np.empty((0,), np.int64)), ids


def _fit_and_score(ctx: ReadoutContext, train_splits: list[str], eval_splits: list[str], *, num_classes: int, tag: str = "") -> dict[str, Any]:
    """One fit + score; ``tag`` names the fit's checkpoint (``fold<k>`` under folds) so two fits of one
    seed directory never share a resumable checkpoint (``_fit.fit``)."""
    t0 = time.time()
    train, train_t, _, _ = _role_rows(ctx, train_splits, num_classes=num_classes, role="train")
    ev, eval_t, eval_rows, eval_ids = _role_rows(ctx, eval_splits, num_classes=num_classes, role="eval")
    if len(train) == 0:
        raise ProbeRefused("E_SPLIT_MISSING", f"no present train rows in {train_splits}")
    if len(ev) == 0:
        raise ProbeRefused("E_SPLIT_MISSING", f"no present eval rows in {eval_splits}")
    ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)
    res = _fit.fit(ctx, train, train_t, build_head=_build_head(ctx, num_classes), loss=ctx.cfg.loss, tag=tag)
    t1 = time.time()
    logits = _fit.score(ctx, res.head, ev)
    floor_out = _fit.floor_outputs(ctx, train_t, len(ev), num_classes=num_classes)
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t1)
    return {
        "head": res.head, "fit": res, "train_targets": train_t, "n_train": len(train), "backend": train.backend,
        "outputs": logits, "targets": eval_t, "rows": eval_rows, "clip_ids": eval_ids, "floor_outputs": floor_out,
        "eval_splits": list(eval_splits),
    }


def _base_meta(ctx: ReadoutContext, num_classes: int) -> dict[str, Any]:
    return {
        "readout": "clf",
        "label_kind": ctx.label_kind,
        "num_classes": num_classes,
        "chance": dict(ctx.chance),
        "bootstrap": dict(ctx.bootstrap or {}),
        "seed": ctx.seed,
        "floor": None,
        "folds": None,
        "sets": None,
    }


def _timing_block(ctx: ReadoutContext, s_per_epoch: float | None) -> dict:
    return {k: ctx.timing[k] for k in ("materialize_s", "fit_s", "score_s") if k in ctx.timing} | {"s_per_epoch": s_per_epoch}


def run_clf(ctx: ReadoutContext) -> dict:
    """Fit + score one seed; returns the metric/floor/control/ci blocks of ``result.json`` plus the side outputs."""
    _fit.prepare(ctx)
    num_classes = _fit.num_classes(ctx)
    if ctx.roles["folds"]:
        return _run_folds(ctx, num_classes)

    arm = _fit_and_score(ctx, ctx.roles["train"], ctx.roles["eval"], num_classes=num_classes)
    split_meta = _base_meta(ctx, num_classes)
    split_meta.update(
        n_train=int(arm["n_train"]), n_eval=int(len(arm["rows"])), n_val=int(arm["fit"].n_val),
        train_targets=arm["train_targets"], eval_rows=arm["rows"], eval_clip_ids=arm["clip_ids"],
    )
    if isinstance(arm["targets"], dict):
        split_meta["sets"] = arm["targets"]

    # Floor role (avsync): the SAME trained probe scores the floor rows → controls.
    extra: dict[str, np.ndarray] = {}
    samples = [{"role": "eval", "outputs": arm["outputs"], "targets": arm["targets"], "rows": arm["rows"], "clip_ids": arm["clip_ids"]}]
    if ctx.roles["floor"]:
        t1 = time.time()
        fl, fl_t, fl_rows, fl_ids = _role_rows(ctx, ctx.roles["floor"], num_classes=num_classes, role="floor")
        fl_out = _fit.score(ctx, arm["head"], fl) if len(fl) else np.empty((0, num_classes), np.float32)
        ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t1)
        split_meta["floor"] = {"outputs": fl_out, "targets": fl_t, "n_floor": int(len(fl)), "rows": fl_rows, "clip_ids": fl_ids}
        extra["floor_logits"] = fl_out
        extra["floor_rows"] = fl_rows
        samples.append({"role": "floor", "outputs": fl_out, "targets": fl_t, "rows": fl_rows, "clip_ids": fl_ids})

    blocks = evaluate_metrics(
        ctx.metric_specs, outputs=arm["outputs"], targets=arm["targets"], protocol=ctx.protocol,
        split_meta=split_meta, floor_outputs=arm["floor_outputs"], ci=ctx.ci,
        scorer=ctx.scorer(arm["eval_splits"], arm["rows"], arm["targets"]),
    )
    fit_rec = _fit.fit_record(arm["fit"])
    return {
        **blocks,
        "probe": fit_rec,
        "fit": {**fit_rec, "lr_grid": list(ctx.cfg.lr_grid), "device": ctx.device},
        "eval_outputs": arm["outputs"],
        "eval_rows": arm["rows"],
        "extra_arrays": extra,
        "backend": arm["backend"],
        "timing": _timing_block(ctx, fit_rec["s_per_epoch"]),
        "samples": samples if ctx.log_samples else None,
    }


def _run_folds(ctx: ReadoutContext, num_classes: int) -> dict:
    """Folds: one probe per held-out fold (train = all other folds); outputs pooled in fold
    order for ``acc_mean/acc_pstd/acc_per_fold/n_folds`` and ``ci: pooled_binomial``. The floor of
    each fold is its own train majority; the pooled floor outputs line up with the pooled eval rows."""
    folds = list(ctx.roles["folds"])
    per_fold: dict[str, dict[str, Any]] = {}
    outs, tgts, rows, fold_idx, floor_outs, fits, train_ts = [], [], [], [], [], [], []
    backend = None
    for k, f in enumerate(folds):
        ctx.check_interrupt()
        arm = _fit_and_score(ctx, [g for g in folds if g != f], [f], num_classes=num_classes, tag=f"fold{k}")
        per_fold[f] = {"outputs": arm["outputs"], "targets": arm["targets"], "rows": arm["rows"], "clip_ids": arm["clip_ids"], "n_train": int(arm["n_train"])}
        outs.append(arm["outputs"])
        tgts.append(arm["targets"])
        rows.append(arm["rows"])
        fold_idx.append(np.full(len(arm["rows"]), k, dtype=np.int64))
        floor_outs.append(np.asarray(arm["floor_outputs"]))
        fits.append({"fold": f, **_fit.fit_record(arm["fit"])})
        train_ts.append(arm["train_targets"])
        backend = backend or arm["backend"]
    outputs = np.concatenate(outs, axis=0)
    targets = _concat_targets(tgts)
    all_rows = np.concatenate(rows)
    floor_pooled = np.concatenate(floor_outs, axis=0)
    split_meta = _base_meta(ctx, num_classes)
    split_meta.update(
        n_train=int(round(float(np.mean([p["n_train"] for p in per_fold.values()])))),
        n_eval=int(len(all_rows)), n_val=int(round(float(np.mean([f["n_val"] for f in fits])))),
        train_targets=np.concatenate(train_ts, axis=0),
        eval_rows=all_rows, eval_clip_ids=[c for p in per_fold.values() for c in p["clip_ids"]],
        folds=per_fold, n_folds=len(folds),
    )
    blocks = evaluate_metrics(
        ctx.metric_specs, outputs=outputs, targets=targets, protocol=ctx.protocol,
        split_meta=split_meta, floor_outputs=floor_pooled, ci=ctx.ci,
        scorer=ctx.scorer(folds, all_rows, targets),
    )
    # the probe block summarizes the folds: epochs/best_epoch are means, selected_lr from the first fold
    rec = {
        "selected_lr": fits[0]["selected_lr"],
        "epochs_run": int(round(float(np.mean([f["epochs_run"] for f in fits])))),
        "best_epoch": int(round(float(np.mean([f["best_epoch"] for f in fits])))),
        "stop": ",".join(sorted({f["stop"] for f in fits})),
        "n_train": int(split_meta["n_train"]),
        "n_val": int(split_meta["n_val"]),
        "s_per_epoch": float(np.mean([f["s_per_epoch"] for f in fits])),
    }
    return {
        **blocks,
        "probe": rec,
        "fit": {**rec, "folds": fits, "lr_grid": list(ctx.cfg.lr_grid), "device": ctx.device},
        "eval_outputs": outputs,
        "eval_rows": all_rows,
        "extra_arrays": {"eval_fold_index": np.concatenate(fold_idx)},
        "backend": backend,
        "timing": _timing_block(ctx, rec["s_per_epoch"]),
        "samples": ([{"role": "eval", "outputs": outputs, "targets": targets, "rows": all_rows,
                      "clip_ids": split_meta["eval_clip_ids"], "folds": [folds[int(k)] for k in np.concatenate(fold_idx)]}]
                    if ctx.log_samples else None),
    }


# ----------------------------------------------------------------------------- the descriptor (yavih.readouts.registry)

_CLF_TRAIN = dict(lr=1e-4, lr_grid=(), weight_decay=1e-4, max_epochs=30, batch_size=256, patience=5, warmup_fraction=0.0)


def _head(*, num_classes: int | None, T: int | None = None, grid: Any = None, target_size: int | None = None) -> dict:
    return {"family": "attentive_probe_v1", "self_attn_depth": 3, "cross_attn_depth": 1, "num_heads": 16, "mlp_ratio": 4,
            "memory_pos": "none", "layout": "point", "out": num_classes}


def _loss(label_kind: str | None) -> str:
    return "bce" if label_kind in {"multi", "multi_sets"} else "ce"


def _geometry(cfg: Any) -> dict:
    return {"num_classes": cfg.head.get("out"), "label_kind": "multi" if cfg.loss == "bce" else "single"}


def _floor_outputs(train_targets: Any, n_eval: int, *, protocol: dict, label_kind: str, num_classes: int | None = None) -> np.ndarray:
    """single: one-hot logits of the train majority class; multi / multi_sets: the train prevalence per class."""
    from yavih.probes.floors import class_prevalence, majority_class

    if label_kind in {"multi", "multi_sets"}:
        prev = class_prevalence(train_targets)
        return np.broadcast_to(prev, (n_eval, prev.shape[0])).astype(np.float32).copy()
    lab = np.asarray(train_targets)
    c = int(num_classes) if num_classes is not None else int(lab[lab >= 0].max()) + 1
    out = np.zeros((n_eval, c), dtype=np.float32)
    out[:, majority_class(lab, c)] = 1.0
    return out


READOUT = Readout(
    name="clf", version=VERSION, module=__name__, label_kinds=frozenset({"single", "multi", "multi_sets"}),
    requires_t_bins=False, requires_grid=False, dense=False,
    train=_CLF_TRAIN, head=_head, loss=_loss, geometry=_geometry,
    head_families=frozenset({"attentive_probe_v1"}), losses=frozenset({"ce", "bce"}), floor_outputs=_floor_outputs,
)
