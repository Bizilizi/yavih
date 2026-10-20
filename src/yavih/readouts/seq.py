"""``seq`` readout: features + ``pos.t`` in MEDIA coordinates + each row's
``media.duration_s`` per stream (``FeatureReader.durations``). A token's absolute time is
``τ = t · duration_s`` of its own stream; it is embedded at ``u = (τ − segment_offset ·
segment_seconds) / (T · segment_seconds)`` (``readouts.coords.seq_task_pos``), so a bridge covering
``[0, 10]`` with 20 frames and one covering it with 8 tokens see the same targets and are read at the
same seconds. Targets ``int64 [n, T]`` with ``T = protocol.segments_scored``: scored index ``k`` is media
second ``s = segment_offset + k`` and takes the row's class iff ``event_start ≤ s < event_end``, else
``background_index``. Head ``dense_query_v1`` with layout ``time:T`` (``yavih.probes.dense.build_dense_query_head``),
loss ``ce_seq`` (NaN-masked). ``split_meta`` carries ``event_start``/``event_end`` for
``seg_acc_boundary``/``n_boundary``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from yavih.readouts._targets import seq_targets
from yavih.readouts.base import ProbeRefused, ReadoutContext, evaluate_metrics, lazy_module
from yavih.readouts.coords import seq_params, seq_task_pos, task_pos
from yavih.readouts.registry import Readout

_fit = lazy_module("yavih.readouts._fit")  # torch: imported on first use, never by the descriptor

__all__ = ["READOUT", "VERSION", "run_seq"]

#: The readout version: enters ``protocol_hash``; ``yavih.versions.READOUT_VERSION`` reads it here.
#: The readout's conformance band on the synthetic end-to-end run (``tests/test_e2e_synthetic.py``): the oracle bridge's
#: headline stays at or above ``oracle_min`` and the random bridge's at or below ``random_max``. Probe training is not
#: bit-reproducible across torch builds, so this is a band, not a point; a change that leaves it bumps ``VERSION``.
CONFORMANCE: dict[str, float] = {"oracle_min": 0.75, "random_max": 0.6}
VERSION: int = 2


def _rows(ctx: ReadoutContext, splits: list[str]) -> tuple[_fit.FeatureRows, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    segments, tg, st, en, rows, ids, durs = [], [], [], [], [], [], []
    T = int(ctx.protocol["segments_scored"])
    for s in splits:
        r = ctx.present_rows(s)
        segments.append((ctx.reader(s), r))
        t, a, b = seq_targets(ctx, s, r)
        hooked = ctx.hook_targets(s, r)
        if hooked is not None:
            t = np.asarray(hooked, dtype=np.int64).reshape(len(r), T)
        tg.append(t)
        st.append(a)
        en.append(b)
        rows.append(r)
        ids.extend(ctx.clip_ids(s, r))
        durs.append(ctx.durations(s, r))
    feats = _fit.FeatureRows(segments)
    n_streams = max(1, len(ctx.stream_names()))

    def cat(xs, shape, dt):
        return np.concatenate(xs, axis=0) if xs else np.empty(shape, dtype=dt)

    return (feats, cat(tg, (0, T), np.int64), cat(st, (0,), np.int64), cat(en, (0,), np.int64), cat(rows, (0,), np.int64), ids,
            cat(durs, (0, n_streams), np.float64))


def _is_const(p: Any) -> bool:
    return isinstance(p, np.ndarray) and p.ndim == 2


def media_to_task(ctx: ReadoutContext, feats: _fit.FeatureRows, durations: np.ndarray) -> Any:
    """The seq task-frame pos gather over ``feats`` (``None`` without pos)."""
    params = seq_params(ctx.protocol)
    base = feats.pos_gather()
    if base is None:
        return None
    out = task_pos(base, lambda p, d: seq_task_pos(p, d, **params), n=len(feats), N=feats.N, per_row=durations)
    # A timed token whose stream has no media.duration_s cannot be placed in seconds: refuse rather than
    # silently reading it as untimed (checked on the first rows; durations come from the sidecars).
    k = min(len(feats), 64)
    if k:
        idx = np.arange(k)
        # only a 2-D [N, 4] array is constant pos; anything else is a per-row gather indexed by view row
        media = np.broadcast_to(base, (k, *base.shape)) if _is_const(base) else np.asarray(base[idx])
        task = np.broadcast_to(out, (k, *out.shape)) if _is_const(out) else np.asarray(out[idx])
        timed = np.isfinite(media[..., 1]) & (media[..., 1] != -1.0) & (media[..., 0] >= 0)
        lost = timed & ~np.isfinite(task[..., 1])
        if lost.any():
            names = ctx.stream_names()
            tags = sorted({int(t) for t in media[..., 0][lost]})
            raise ProbeRefused(
                "E_DURATIONS",
                f"tokens of stream(s) {[names[t] if t < len(names) else t for t in tags]} carry pos.t but their media.duration_s is "
                "unknown (sidecar members[].duration_s): the seq readout cannot place them in seconds",
            )
    return out


def run_seq(ctx: ReadoutContext) -> dict:
    _fit.prepare(ctx)
    num_classes = _fit.num_classes(ctx)
    t0 = time.time()
    train, train_t, _, _, _, _, train_d = _rows(ctx, ctx.roles["train"])
    ev, eval_t, starts, ends, eval_rows, eval_ids, eval_d = _rows(ctx, ctx.roles["eval"])
    if len(train) == 0 or len(ev) == 0:
        raise ProbeRefused("E_SPLIT_MISSING", "seq readout needs present train and eval rows")
    pos_train, pos_eval = media_to_task(ctx, train, train_d), media_to_task(ctx, ev, eval_d)
    if pos_train is None or pos_eval is None:
        raise ProbeRefused("E_POS_MISSING", "seq readout needs pos (validate should have recorded eligible: false)")
    ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)

    def build(dim: int | None = None):
        from yavih.probes.dense import build_dense_query_head

        return build_dense_query_head(int(dim or ctx.D), num_classes, dict(ctx.cfg.head))

    res = _fit.fit(ctx, train, train_t, build_head=build, loss=ctx.cfg.loss, pos=pos_train)
    t1 = time.time()
    logits = _fit.score(ctx, res.head, ev, pos=pos_eval)  # [n, T, C]
    floor_out = _fit.floor_outputs(ctx, train_t, len(ev), num_classes=num_classes)
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t1)

    num_segments = int(ctx.protocol["num_segments"])
    split_meta: dict[str, Any] = {
        "readout": "seq",
        "label_kind": ctx.label_kind,
        "num_classes": num_classes,
        "n_train": int(len(train)),
        "n_eval": int(len(ev)),
        "n_val": int(res.n_val),
        "train_targets": train_t,
        "eval_rows": eval_rows,
        "eval_clip_ids": eval_ids,
        "event_start": starts,
        "event_end": ends,
        "boundary": (ends - starts) < num_segments,
        "chance": dict(ctx.chance),
        "bootstrap": dict(ctx.bootstrap or {}),
        "seed": ctx.seed,
    }
    blocks = evaluate_metrics(
        ctx.metric_specs, outputs=logits, targets=eval_t, protocol=ctx.protocol,
        split_meta=split_meta, floor_outputs=floor_out, ci=ctx.ci,
        scorer=ctx.scorer(ctx.roles["eval"], eval_rows, eval_t),
    )
    rec = _fit.fit_record(res)
    return {
        **blocks,
        "probe": rec,
        "fit": {**rec, "lr_grid": list(ctx.cfg.lr_grid), "device": ctx.device},
        "eval_outputs": logits,
        "eval_rows": eval_rows,
        "extra_arrays": {},
        "backend": train.backend,
        "timing": {k: ctx.timing[k] for k in ("materialize_s", "fit_s", "score_s") if k in ctx.timing} | {"s_per_epoch": rec["s_per_epoch"]},
        "samples": ([{"role": "eval", "outputs": logits, "targets": eval_t, "rows": eval_rows, "clip_ids": eval_ids}]
                    if ctx.log_samples else None),
    }


# ----------------------------------------------------------------------------- the descriptor (yavih.readouts.registry)

_SEQ_TRAIN = dict(lr=3e-4, lr_grid=(1e-4, 3e-4, 1e-3), weight_decay=1e-4, max_epochs=50, batch_size=64, patience=10,
                  warmup_fraction=0.05)


def _head(*, num_classes: int | None, T: int | None = None, grid: Any = None, target_size: int | None = None) -> dict:
    return {"family": "dense_query_v1", "self_attn_depth": 3, "cross_attn_depth": 1, "num_heads": 16, "mlp_ratio": 4,
            "memory_pos": "sincos(t)+null_embed", "layout": f"time:{T if T is not None else '?'}", "out": num_classes}


def _task_sizes(bench: Any) -> dict:
    spec = bench.spec
    rp = dict(spec.readout_params)
    return {"T": int(rp["T"]) if "T" in rp else int(spec.protocol.get("segments_scored"))}


def _geometry(cfg: Any) -> dict:
    from yavih.probes.config import parse_layout

    kw: dict[str, Any] = {"num_classes": cfg.head.get("out")}
    layout = str(cfg.head.get("layout", ""))
    if "?" not in layout:
        kind, shape = parse_layout(layout)
        if kind == "time":
            kw["T"] = shape[0]
    return kw


def _validate_spec(params: dict, requires: Any, protocol: Any) -> None:
    """T = protocol.segments_scored = readout_params.T; eligibility needs t_bins of them timed."""
    if requires.t_bins is not None and requires.t_bins > params["T"]:
        raise ValueError(f".T: output_contract.requires.t_bins ({requires.t_bins}) cannot exceed T ({params['T']})")
    scored = (protocol or {}).get("segments_scored")
    if scored is not None and params["T"] != scored:
        raise ValueError(f".T: must equal protocol.segments_scored ({scored})")


def _floor_outputs(train_targets: Any, n_eval: int, *, protocol: dict, label_kind: str, num_classes: int | None = None) -> np.ndarray:
    """One-hot logits of the most frequent train segment label (pooled over all (clip, segment) cells)."""
    from yavih.probes.floors import majority_class

    t = np.asarray(train_targets)
    if t.ndim != 2:
        raise ValueError(f"floor: seq train targets must be [n, T], got {t.shape}")
    T = t.shape[1]
    flat = t.reshape(-1)
    if flat.dtype.kind == "f":
        flat = flat[np.isfinite(flat)]
    flat = flat[flat >= 0].astype(np.int64)
    c = int(num_classes) if num_classes is not None else int(flat.max()) + 1
    out = np.zeros((n_eval, T, c), dtype=np.float32)
    out[:, :, majority_class(flat, c)] = 1.0
    return out


READOUT = Readout(
    name="seq", version=VERSION, module=__name__, label_kinds=frozenset({"segments"}),
    protocol_keys=frozenset({"segment_seconds"}), param_keys=frozenset({"T"}), requires_t_bins=True,
    validate_spec=_validate_spec, dense=True, train=_SEQ_TRAIN, head=_head, loss=lambda label_kind: "ce_seq",
    task_sizes=_task_sizes, geometry=_geometry, head_families=frozenset({"dense_query_v1"}), losses=frozenset({"ce_seq"}),
    floor_outputs=_floor_outputs,
)
