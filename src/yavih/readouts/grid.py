"""``grid`` readout: features + ``pos.(y, x)`` (full-frame fractions of the ORIGINAL frame)
of the stream named ``readout_params.visual_stream``; targets from ``Benchmark.targets(split)``
(``float32 [n, S, S]`` = the raster of ``protocol.target_region`` of the full frame, ``S =
protocol.target_size``, invalid = 0, validity ``0 < d ≤ max_depth``). The ``h × w`` query cells tile the
target region; memory tokens carry their region-relative ``(y, x)`` (``readouts.coords.region_task_pos``),
tokens outside the region included with their out-of-range coordinates. Head ``dense_query_v1``
with layout ``grid:HxW``, ``out = 1`` (log-depth), bilinear upsample to ``S×S`` + ``exp`` inside the
head (``head.upsample``/``head.activation`` of the pinned descriptor), so the trainer's ``l1_log``
loss and the depth metrics both see metric depth. Invalid train pixels are NaN for the loss.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from yavih.readouts.base import ProbeRefused, ReadoutContext, evaluate_metrics, lazy_module
from yavih.readouts.coords import region_task_pos, target_region, task_pos
from yavih.readouts.registry import Readout

_fit = lazy_module("yavih.readouts._fit")  # torch: imported on first use, never by the descriptor

__all__ = ["READOUT", "VERSION", "run_grid"]

#: The readout version: enters ``protocol_hash``; ``yavih.versions.READOUT_VERSION`` reads it here.
VERSION: int = 2


def _rows(ctx: ReadoutContext, splits: list[str], *, nan_invalid: bool) -> tuple[_fit.FeatureRows, _fit.TargetRows, np.ndarray, list[str]]:
    if len(splits) != 1:
        raise ProbeRefused("E_SPLIT_MISSING", f"grid readout expects exactly one split per role, got {splits}")
    s = splits[0]
    r = ctx.present_rows(s)
    feats = _fit.FeatureRows([(ctx.reader(s), r)])
    hooked = ctx.hook_targets(s, r)
    source = _ArrayTargets(np.asarray(hooked, dtype=np.float32), r) if hooked is not None else ctx.bench.targets(s)
    targets = _fit.TargetRows(source, r, max_depth=float(ctx.protocol["max_depth"]), nan_invalid=nan_invalid)
    return feats, targets, r, ctx.clip_ids(s, r)


class _ArrayTargets:
    """A ``TargetReader``-shaped view over the array a folder ``targets`` hook returned."""

    def __init__(self, arr: np.ndarray, rows: np.ndarray) -> None:
        self.arr = arr
        self.shape = arr.shape
        self._pos = {int(r): i for i, r in enumerate(np.asarray(rows, dtype=np.int64))}

    def rows(self, idx) -> np.ndarray:
        return self.arr[[self._pos[int(i)] for i in np.asarray(idx, dtype=np.int64)]]


def media_to_task(ctx: ReadoutContext, feats: _fit.FeatureRows) -> object:
    """Region-relative task pos over ``feats``; ``None`` without pos."""
    region = target_region(ctx.protocol)
    return task_pos(feats.pos_gather(), lambda p, _d: region_task_pos(p, region), n=len(feats), N=feats.N)


def run_grid(ctx: ReadoutContext) -> dict:
    _fit.prepare(ctx)
    visual = _fit.stream_index(ctx.describe, str(ctx.spec.readout_params["visual_stream"]))
    t0 = time.time()
    train, train_t, train_rows, _ = _rows(ctx, ctx.roles["train"], nan_invalid=True)
    ev, eval_tr, eval_rows, eval_ids = _rows(ctx, ctx.roles["eval"], nan_invalid=False)
    if len(train) == 0 or len(ev) == 0:
        raise ProbeRefused("E_SPLIT_MISSING", "grid readout needs present train and eval rows")
    pos_train, pos_eval = media_to_task(ctx, train), media_to_task(ctx, ev)
    if pos_train is None or pos_eval is None:
        raise ProbeRefused("E_POS_MISSING", "grid readout needs pos (validate should have recorded eligible: false)")
    eval_t = eval_tr.all()  # float32 [n, S, S], 0 = invalid (NYUv2 eval: 654 × 224² × 4 B fits in RAM)
    ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)

    def build(dim: int | None = None):
        from yavih.probes.dense import build_dense_query_head

        return build_dense_query_head(int(dim or ctx.D), int(ctx.cfg.head.get("out", 1)), dict(ctx.cfg.head), pos_stream_tag=visual)

    res = _fit.fit(ctx, train, train_t, build_head=build, loss=ctx.cfg.loss, pos=pos_train)
    t1 = time.time()
    pred = _fit.score(ctx, res.head, ev, pos=pos_eval)  # depth [n, S, S]
    # the floor streams the RAW train targets (invalid = 0, not NaN): floors.mean_log_depth_map masks by range
    floor_out = _fit.floor_outputs(ctx, _fit.TargetRows(train_t.tr, train_rows), len(ev))
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t1)

    split_meta: dict[str, Any] = {
        "readout": "grid",
        "n_train": int(len(train)),
        "n_eval": int(len(ev)),
        "n_val": int(res.n_val),
        "train_targets": _fit.TargetRows(train_t.tr, train_rows),
        "eval_rows": eval_rows,
        "eval_clip_ids": eval_ids,
        "chance": dict(ctx.chance),
        "bootstrap": dict(ctx.bootstrap or {}),
        "seed": ctx.seed,
    }
    blocks = evaluate_metrics(
        ctx.metric_specs, outputs=pred, targets=eval_t, protocol=ctx.protocol,
        split_meta=split_meta, floor_outputs=floor_out, ci=ctx.ci,
        scorer=ctx.scorer(ctx.roles["eval"], eval_rows, eval_t),
    )
    rec = _fit.fit_record(res)
    return {
        **blocks,
        "probe": rec,
        "fit": {**rec, "lr_grid": list(ctx.cfg.lr_grid), "device": ctx.device},
        "eval_outputs": pred,
        "eval_rows": eval_rows,
        "extra_arrays": {},
        "backend": train.backend,
        "timing": {k: ctx.timing[k] for k in ("materialize_s", "fit_s", "score_s") if k in ctx.timing} | {"s_per_epoch": rec["s_per_epoch"]},
        "samples": ([{"role": "eval", "outputs": pred, "targets": eval_t, "rows": eval_rows, "clip_ids": eval_ids}]
                    if ctx.log_samples else None),
    }


# ----------------------------------------------------------------------------- the descriptor (yavih.readouts.registry)

_GRID_TRAIN = dict(lr=3e-4, lr_grid=(1e-4, 3e-4, 1e-3), weight_decay=1e-4, max_epochs=50, batch_size=64, patience=10,
                   warmup_fraction=0.05)


def _head(*, num_classes: int | None = None, T: int | None = None, grid: Any = None, target_size: int | None = None) -> dict:
    if grid is None:
        layout = "grid:?x?"
    else:
        h, w = (grid, grid) if isinstance(grid, int) else (int(grid[0]), int(grid[1]))
        layout = f"grid:{h}x{w}"
    return {"family": "dense_query_v1", "self_attn_depth": 3, "cross_attn_depth": 1, "num_heads": 16, "mlp_ratio": 4,
            "memory_pos": "sincos(y,x)+null_embed", "layout": layout, "out": 1,
            "upsample": target_size,  # bilinear to protocol.target_size, then exp (log-depth → depth)
            "activation": "exp"}


def _task_sizes(bench: Any) -> dict:
    spec = bench.spec
    rp = dict(spec.readout_params)
    return {"grid": (int(rp["h"]), int(rp["w"])), "target_size": int(spec.protocol["target_size"])}


def _geometry(cfg: Any) -> dict:
    from yavih.probes.config import parse_layout

    kw: dict[str, Any] = {"target_size": cfg.head.get("upsample")}
    layout = str(cfg.head.get("layout", ""))
    if "?" not in layout:
        kind, shape = parse_layout(layout)
        if kind == "grid":
            kw["grid"] = shape
    return kw


def _validate_spec(params: dict, requires: Any, protocol: Any) -> None:
    if requires.grid is not None and (params["h"], params["w"]) != (requires.grid["h"], requires.grid["w"]):
        raise ValueError(": h/w must equal output_contract.requires.grid h/w")


def _floor_outputs(train_targets: Any, n_eval: int, *, protocol: dict, label_kind: str, num_classes: int | None = None) -> np.ndarray:
    """``exp`` of the per-pixel mean train log-depth map (a depth, like the head)."""
    from yavih.probes.floors import mean_log_depth_map

    mean_log = mean_log_depth_map(train_targets, max_depth=protocol.get("max_depth"))
    depth = np.exp(mean_log).astype(np.float32)
    return np.broadcast_to(depth, (n_eval, *depth.shape)).copy()


READOUT = Readout(
    name="grid", version=VERSION, module=__name__, label_kinds=frozenset({"none"}),
    protocol_keys=frozenset({"depth_param", "source_size", "target_region"}), param_keys=frozenset({"h", "w", "out", "visual_stream"}),
    targets=True, full_frame=True, requires_grid=True, validate_spec=_validate_spec, dense=True,
    train=_GRID_TRAIN, head=_head, loss=lambda label_kind: "l1_log", task_sizes=_task_sizes, geometry=_geometry,
    head_families=frozenset({"dense_query_v1"}), losses=frozenset({"l1_log"}), floor_outputs=_floor_outputs,
)
