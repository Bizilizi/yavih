"""``box`` readout: features + ``pos`` with the two named stream tags; targets ``float32
[n, 4]`` ``(cx, cy, w, h)`` in FULL-FRAME fractions from ``x1|y1|x2|y2`` (degenerate boxes excluded and
counted). Visual tokens are binned by their full-frame ``(y, x)`` into the ``G × G`` cells over the target
region (``protocol.target_region``, the whole frame for vggss_box; ``readouts.coords.region_task_pos``):
a bridge that cropped the frame simply leaves the cells it did not see empty, and IoU is computed in
full-frame fractions. Media coordinates make per-sample pos the common case (frame sizes differ), so
both constant and per-sample pos are supported. Head ``dense_av_v1`` (``yavih.probes.dense.build_dense_av_head``) built
with the per-model stream indices of ``readout_params.visual_stream`` / ``conditioning_stream`` resolved
through ``describe()["streams"]``; shuffled-conditioning control under a seeded derangement
(``readout_params.shuffle_seed``) → a second head whose eval outputs reach ``iou_shuffled``/``audio_blind``
via ``split_meta["shuffled"]``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from yavih.readouts._targets import box_targets
from yavih.readouts.base import ProbeRefused, ReadoutContext, evaluate_metrics, lazy_module
from yavih.readouts.coords import region_task_pos, target_region, task_pos
from yavih.readouts.registry import Readout

_fit = lazy_module("yavih.readouts._fit")  # torch: imported on first use, never by the descriptor

__all__ = ["READOUT", "VERSION", "run_box", "ShuffledRows"]

#: The readout version: enters ``protocol_hash``; ``yavih.versions.READOUT_VERSION`` reads it here.
#: The readout's conformance band on the synthetic end-to-end run (``tests/test_e2e_synthetic.py``): the oracle bridge's
#: headline stays at or above ``oracle_min`` and the random bridge's at or below ``random_max``. Probe training is not
#: bit-reproducible across torch builds, so this is a band, not a point; a change that leaves it bumps ``VERSION``.
CONFORMANCE: dict[str, float] = {"oracle_min": 0.5, "random_max": 0.45}
VERSION: int = 2


class ShuffledRows:
    """``FeatureRows`` with the conditioning stream's tokens replaced by another clip's under a fixed
    derangement (the shuffled control). ``pos`` is the media pos: a constant ``[N, 4]`` (the conditioning
    slots are the same for every clip: a masked copy) or a per-sample gather (each row's own slots take
    its partner's conditioning tokens in order; both must hold the same number of them)."""

    def __init__(self, base: _fit.FeatureRows, perm: np.ndarray, pos: Any, cond_tag: int) -> None:
        self.base = base
        self.perm = np.asarray(perm, dtype=np.int64)
        self.pos = pos
        self.cond_tag = float(cond_tag)
        self.N, self.D = base.N, base.D
        self._const_mask = np.asarray(pos)[:, 0] == self.cond_tag if isinstance(pos, np.ndarray) and pos.ndim == 2 else None

    @property
    def backend(self) -> str | None:
        return self.base.backend

    def __len__(self) -> int:
        return len(self.base)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.base.shape

    def __getitem__(self, idx: Any) -> np.ndarray:
        ids, scalar = _fit._index(idx, len(self.base))
        own = np.array(self.base[ids], copy=True)
        other = np.asarray(self.base[self.perm[ids]])
        if self._const_mask is not None:
            own[:, self._const_mask] = other[:, self._const_mask]
        else:
            p_own = np.asarray(self.pos[ids])
            p_other = np.asarray(self.pos[self.perm[ids]])
            for j in range(ids.size):
                m_own = p_own[j, :, 0] == self.cond_tag
                m_other = p_other[j, :, 0] == self.cond_tag
                if int(m_own.sum()) != int(m_other.sum()):
                    raise ProbeRefused(
                        "E_POS_SHUFFLE",
                        f"shuffled control: row {int(ids[j])} has {int(m_own.sum())} conditioning tokens, its partner "
                        f"{int(self.perm[ids[j]])} has {int(m_other.sum())} (the shuffled control needs equal counts)",
                    )
                own[j, m_own] = other[j, m_other]
        return own[0] if scalar else own


def _rows(ctx: ReadoutContext, splits: list[str]) -> tuple[_fit.FeatureRows, np.ndarray, np.ndarray, list[str], int]:
    segments, tg, rows, ids = [], [], [], []
    excluded = 0
    for s in splits:
        r = ctx.present_rows(s)
        hooked = ctx.hook_targets(s, r)
        if hooked is not None:  # folder hook: (cx, cy, w, h) [n, 4]; degenerate = w <= 0 or h <= 0
            arr = np.asarray(hooked, dtype=np.float32).reshape(len(r), 4)
            ok = (arr[:, 2] > 0) & (arr[:, 3] > 0) & np.isfinite(arr).all(axis=1)
            t, kept, n_ex = arr[ok], r[ok], int((~ok).sum())
        else:
            t, kept, n_ex = box_targets(ctx, s, r)
        excluded += n_ex
        segments.append((ctx.reader(s), kept))
        tg.append(t)
        rows.append(kept)
        ids.extend(ctx.clip_ids(s, kept))
    feats = _fit.FeatureRows(segments)

    def cat(xs, shape, dt):
        return np.concatenate(xs, axis=0) if xs else np.empty(shape, dtype=dt)

    return feats, cat(tg, (0, 4), np.float32), cat(rows, (0,), np.int64), ids, excluded


def media_to_task(ctx: ReadoutContext, feats: Any, media_pos: Any) -> Any:
    """Region-relative task pos; ``None`` without pos."""
    region = target_region(ctx.protocol)
    return task_pos(media_pos, lambda p, _d: region_task_pos(p, region), n=len(feats), N=feats.N)


def run_box(ctx: ReadoutContext) -> dict:
    _fit.prepare(ctx)
    rp = dict(ctx.spec.readout_params)
    vis = _fit.stream_index(ctx.describe, str(rp["visual_stream"]))
    cond = _fit.stream_index(ctx.describe, str(rp["conditioning_stream"]))
    t0 = time.time()
    train, train_t, _, _, ex_train = _rows(ctx, ctx.roles["train"])
    ev, eval_t, eval_rows, eval_ids, ex_eval = _rows(ctx, ctx.roles["eval"])
    if len(train) == 0 or len(ev) == 0:
        raise ProbeRefused("E_SPLIT_MISSING", "box readout needs present train and eval rows")
    media_train, media_eval = train.pos_gather(), ev.pos_gather()
    if media_train is None or media_eval is None:
        raise ProbeRefused("E_POS_MISSING", "box readout needs pos (validate should have recorded eligible: false)")
    pos_train, pos_eval = media_to_task(ctx, train, media_train), media_to_task(ctx, ev, media_eval)
    ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)

    def build(dim: int | None = None):
        from yavih.probes.dense import build_dense_av_head

        return build_dense_av_head(int(dim or ctx.D), dict(ctx.cfg.head), visual_tag=vis, conditioning_tag=cond)

    res = _fit.fit(ctx, train, train_t, build_head=build, loss=ctx.cfg.loss, pos=pos_train)
    t1 = time.time()
    pred = _fit.score(ctx, res.head, ev, pos=pos_eval)  # [n, 4]
    floor_out = _fit.floor_outputs(ctx, train_t, len(ev))
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t1)

    split_meta: dict[str, Any] = {
        "readout": "box",
        "n_train": int(len(train)),
        "n_eval": int(len(ev)),
        "n_val": int(res.n_val),
        "train_targets": train_t,
        "eval_rows": eval_rows,
        "eval_clip_ids": eval_ids,
        "n_excluded_degenerate": int(ex_eval),
        "n_excluded_degenerate_train": int(ex_train),
        "chance": dict(ctx.chance),
        "bootstrap": dict(ctx.bootstrap or {}),
        "seed": ctx.seed,
        "shuffled": None,
    }
    fits: dict[str, Any] = {"main": _fit.fit_record(res)}
    extra: dict[str, np.ndarray] = {}
    if rp.get("shuffled_control"):
        seed = int(rp.get("shuffle_seed", 0))
        sh_train = ShuffledRows(train, _fit.derangement(len(train), seed), media_train, cond)
        sh_eval = ShuffledRows(ev, _fit.derangement(len(ev), seed + 1), media_eval, cond)
        res_sh = _fit.fit(ctx, sh_train, train_t, build_head=build, loss=ctx.cfg.loss, pos=pos_train, tag="shuffled")
        t2 = time.time()
        pred_sh = _fit.score(ctx, res_sh.head, sh_eval, pos=pos_eval)
        ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t2)
        split_meta["shuffled"] = {"outputs": pred_sh, "perm_train": sh_train.perm, "perm_eval": sh_eval.perm, "seed": seed}
        fits["shuffled"] = _fit.fit_record(res_sh)
        extra["eval_outputs_shuffled"] = pred_sh

    blocks = evaluate_metrics(
        ctx.metric_specs, outputs=pred, targets=eval_t, protocol=ctx.protocol,
        split_meta=split_meta, floor_outputs=floor_out, ci=ctx.ci,
        scorer=ctx.scorer(ctx.roles["eval"], eval_rows, eval_t),
    )
    rec = fits["main"]
    return {
        **blocks,
        "probe": rec,
        "fit": {**rec, "arms": fits, "n_excluded_degenerate_train": int(ex_train), "lr_grid": list(ctx.cfg.lr_grid), "device": ctx.device},
        "eval_outputs": pred,
        "eval_rows": eval_rows,
        "extra_arrays": extra,
        "backend": train.backend,
        "timing": {k: ctx.timing[k] for k in ("materialize_s", "fit_s", "score_s") if k in ctx.timing} | {"s_per_epoch": rec["s_per_epoch"]},
        "samples": ([{"role": "eval", "outputs": pred, "targets": eval_t, "rows": eval_rows, "clip_ids": eval_ids}]
                    if ctx.log_samples else None),
    }


# ----------------------------------------------------------------------------- the descriptor (yavih.readouts.registry)

_BOX_TRAIN = dict(lr=3e-4, lr_grid=(1e-4, 3e-4, 1e-3), weight_decay=1e-4, max_epochs=50, batch_size=64, patience=10,
                  warmup_fraction=0.05)


def _head(*, num_classes: int | None = None, T: int | None = None, grid: Any = None, target_size: int | None = None) -> dict:
    from yavih.probes.config import DEFAULT_BOX_GRID

    g = DEFAULT_BOX_GRID if grid is None else (grid if isinstance(grid, int) else int(grid[0]))
    return {"family": "dense_av_v1", "self_attn_depth": 2, "cross_attn_depth": 2, "num_heads": 16, "mlp_ratio": 4,
            "memory_pos": "sincos(y,x) on binned grid", "layout": "box", "out": 4, "grid": int(g), "activation": "sigmoid"}


def _task_sizes(bench: Any) -> dict:
    from yavih.probes.config import DEFAULT_BOX_GRID

    return {"grid": int(dict(bench.spec.readout_params).get("grid", DEFAULT_BOX_GRID))}


def _validate_spec(params: dict, requires: Any, protocol: Any) -> None:
    if requires.grid is not None and (params["grid"] != requires.grid["h"] or params["grid"] != requires.grid["w"]):
        raise ValueError(".grid: must equal output_contract.requires.grid h and w")
    if params["visual_stream"] == params["conditioning_stream"]:
        raise ValueError(": visual_stream and conditioning_stream must differ")


def _floor_outputs(train_targets: Any, n_eval: int, *, protocol: dict, label_kind: str, num_classes: int | None = None) -> np.ndarray:
    """The mean train box."""
    from yavih.probes.floors import mean_box

    box = mean_box(train_targets)
    return np.broadcast_to(box, (n_eval, 4)).astype(np.float32).copy()


READOUT = Readout(
    name="box", version=VERSION, module=__name__, label_kinds=frozenset({"box"}),
    protocol_keys=frozenset({"box_population", "gt_geometry", "heatmap_grid", "annotated_second", "target_region"}),
    param_keys=frozenset({"grid", "visual_stream", "conditioning_stream", "shuffled_control", "shuffle_seed"}),
    full_frame=True, requires_grid=True, validate_spec=_validate_spec, dense=True,
    train=_BOX_TRAIN, head=_head, loss=lambda label_kind: "box_l1_giou", task_sizes=_task_sizes,
    geometry=lambda cfg: {"grid": cfg.head.get("grid")}, head_families=frozenset({"dense_av_v1"}),
    losses=frozenset({"box_l1_giou"}), floor_outputs=_floor_outputs,
)
