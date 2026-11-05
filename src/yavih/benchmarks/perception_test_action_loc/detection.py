"""Temporal detection of ``(class, start, end)`` events, owned by the benchmark folders that use it.

This file is byte-identical in every folder that ships it (``unav100``, ``perception_test_sound_loc``,
``perception_test_action_loc``, ``epic_sounds_det``): a candidate for the core once more folders share it, recorded
as such in each folder's maintainer notes. The folder's ``eval.py`` loads it by path and builds its scorer with
:func:`make_scorer` and its metrics with :func:`make_metrics`. Module scope imports the standard library, numpy and
yavih core only; torch is imported inside the loss and the driver.

**Labels.** ``manifest.label.kind: multi`` (the classes occurring in the row) plus the events column
(``protocol.events_column``, default ``events``): ``;``-joined event tokens in the folder's ``events_format``
(:data:`EVENT_FORMATS`). The time frame of a row is its annotated duration: the ``duration_s`` manifest column when
the folder declares one, else the longest stream's ``media.duration_s`` (features sidecars).

**Task frame.** ``T = readout_params.T`` bins tile ``u ∈ [0, 1]`` of the annotated duration; a token at media ``t``
of its own stream sits at ``u = t · media.duration_s / duration_s``, whatever frames or windows the bridge chose.

**Head and loss.** The core ``dense_query_v1`` head, layout ``time:T``, ``out = C + 2``: per bin ``C`` class logits
and two values whose ``softplus`` is the distance (in bins) from the bin centre to the segment start and end.
Targets ``float32 [n, T, C + 2]``: bin ``k`` is positive for class ``c`` when its centre lies in an event of class
``c`` or holds that event's midpoint; its regression target is the distance to the start and end of the SHORTEST
event covering it (NaN where none does). Loss ``<folder>_focal_giou`` (:func:`focal_giou`): sigmoid focal loss
(α 0.25, γ 2) over bins and classes plus ``1 − GIoU`` of the 1-D extents on positive bins, each divided by the row's
positive bins.

**Scoring.** Raw eval outputs are decoded (:data:`DECODE`: threshold, top-k, per-class NMS, ``max_per_row``) and
scored with the ActivityNet detection AP over ``protocol.tiou_thresholds`` (:func:`ap_matrix`: per class present in
the ground truth, greedy matching by decreasing score to the unmatched segment of the same row with the highest tIoU,
VOC-2011 interpolated AP; a class without predictions scores 0; predictions of a class absent from the ground truth
are dropped). The floor is the label-only constant output (per bin and class the train frequency, per bin the median
train extents), decoded and scored the same way.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["EVENT_FORMATS", "DECODE", "TRAIN", "parse_events", "dense_targets", "task_pos_fn", "decode", "decode_row", "ap_matrix",
           "resampled_map", "focal_giou", "floor_outputs", "make_scorer", "make_metrics", "row_events"]

#: Event-token encodings of the events column: ``(class, start, end)`` order and time unit.
EVENT_FORMATS: dict[str, tuple[tuple[str, str, str], float]] = {
    "class:start_us:end_us": (("class", "start", "end"), 1e-6),
    "class:start_ms:end_ms": (("class", "start", "end"), 1e-3),
    "start_s:end_s:class": (("start", "end", "class"), 1.0),
}
#: The pinned decoding of raw outputs into scored segments: candidate (bin, class) pairs with ``sigmoid(logit) >=
#: score_thresh``, the ``pre_nms_topk`` best per row, per-class greedy NMS at ``nms_iou`` tIoU, the ``max_per_row``
#: best kept. Segments are clipped to ``[0, duration_s]``.
DECODE: dict[str, float] = {"score_thresh": 0.001, "pre_nms_topk": 2000, "nms_iou": 0.5, "max_per_row": 100}
#: The pinned training block (the ``seq`` probe's schedule).
TRAIN: dict[str, Any] = dict(lr=3e-4, lr_grid=(1e-4, 3e-4, 1e-3), weight_decay=1e-4, max_epochs=50, batch_size=64, patience=10,
                             warmup_fraction=0.05)
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
_EPS = 1e-6


# ----------------------------------------------------------------------------- events


def parse_events(cell: Any, fmt: str) -> np.ndarray:
    """One events cell → ``float64 [k, 3]`` ``(class, start_s, end_s)`` in cell order; ``""`` / ``None`` → ``[0, 3]``."""
    if fmt not in EVENT_FORMATS:
        raise ValueError(f"events format {fmt!r} is not one of {sorted(EVENT_FORMATS)}")
    order, unit = EVENT_FORMATS[fmt]
    s = "" if cell is None else str(cell).strip()
    if not s:
        return np.zeros((0, 3), dtype=np.float64)
    out = []
    for tok in s.split(";"):
        parts = tok.split(":")
        if len(parts) != 3:
            raise ValueError(f"event token {tok!r} is not {fmt}")
        v = dict(zip(order, parts, strict=True))
        c, a, b = int(v["class"]), float(v["start"]) * unit, float(v["end"]) * unit
        if c < 0 or a < 0 or b < a:
            raise ValueError(f"event token {tok!r}: need class >= 0 and 0 <= start <= end")
        out.append((float(c), a, b))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def row_events(ctx: Any, split: str, rows: np.ndarray, fmt: str, *, media_durations: np.ndarray | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    """``(events, duration_s)`` of within-split ``rows``: the annotated duration from the ``duration_s`` column when
    declared, else the longest stream of ``media_durations [n, S]`` (features), else (no features: the conformance
    population) the last event end plus one second."""
    from yavih.readouts.base import ProbeRefused, _manifest_index

    manifest = ctx.manifest(split)
    idx = _manifest_index(manifest, np.asarray(rows, dtype=np.int64))
    col = str((ctx.protocol or {}).get("events_column", "events"))
    num_classes = len(ctx.bench.classes())
    events, dur = [], np.empty(len(idx), dtype=np.float64)
    for j, i in enumerate(idx):
        extra = manifest.extra[i] if manifest.extra is not None else {}
        try:
            ev = parse_events(extra.get(col), fmt)
        except ValueError as e:
            raise ProbeRefused("E_LABELS", f"row {manifest.clip_id[i]}: {e}") from e
        if ev.size and ev[:, 0].max() >= num_classes:
            raise ProbeRefused("E_LABELS", f"row {manifest.clip_id[i]}: event class outside classes.csv [0, {num_classes})")
        d = extra.get("duration_s")
        if d is None and media_durations is not None:
            md = np.asarray(media_durations[j], dtype=np.float64)
            d = float(np.nanmax(md)) if np.isfinite(md).any() else None
        if d is None:
            d = (float(ev[:, 2].max()) if ev.size else 0.0) + 1.0
        if not np.isfinite(float(d)) or float(d) <= 0:
            raise ProbeRefused("E_LABELS", f"row {manifest.clip_id[i]}: the annotated duration must be > 0, got {d!r}")
        events.append(ev)
        dur[j] = float(d)
    return events, dur


def dense_targets(events: Sequence[np.ndarray], durations: np.ndarray, *, T: int, num_classes: int) -> np.ndarray:
    """``float32 [n, T, C + 2]``: class indicators per bin, then the start/end distances in bins (NaN = no event)."""
    n = len(events)
    out = np.zeros((n, T, num_classes + 2), dtype=np.float32)
    out[:, :, num_classes:] = np.nan
    centre = (np.arange(T, dtype=np.float64) + 0.5) / float(T)
    for i, ev in enumerate(events):
        if len(ev) == 0:
            continue
        d = float(durations[i])
        us = np.clip(ev[:, 1] / d, 0.0, 1.0)
        ue = np.clip(ev[:, 2] / d, 0.0, 1.0)
        best_len = np.full(T, np.inf)
        for c, a, b in zip(ev[:, 0].astype(np.int64), us, ue, strict=True):
            inside = (centre >= a) & (centre <= b)
            inside[min(T - 1, int(np.floor(0.5 * (a + b) * T)))] = True
            out[i, inside, c] = 1.0
            take = inside & ((b - a) < best_len)
            if take.any():
                best_len[take] = b - a
                out[i, take, num_classes] = np.maximum(centre[take] - a, 0.0) * T
                out[i, take, num_classes + 1] = np.maximum(b - centre[take], 0.0) * T
    return out


def task_pos_fn(pos: np.ndarray, per_row: np.ndarray) -> np.ndarray:
    """``pos [k, N, 4]`` media coordinates + ``per_row [k, S + 1]`` (each stream's ``media.duration_s``, then the
    annotated duration) → task pos with ``u = t · duration_s(stream) / annotated duration`` in column 1."""
    from yavih.readouts.coords import media_missing

    p = np.asarray(pos, dtype=np.float32)
    pr = np.asarray(per_row, dtype=np.float64)
    k = p.shape[0]
    dur, annot = pr[:, :-1], pr[:, -1]
    tag = p[..., 0]
    ok = np.isfinite(tag) & (tag >= 0) & (tag < dur.shape[1]) & (tag == np.round(tag))
    tag_i = np.where(ok, tag, 0).astype(np.int64)
    tok_dur = np.where(ok, dur[np.arange(k)[:, None], tag_i], np.nan)
    u = p[..., 1].astype(np.float64) * tok_dur / annot[:, None]
    u = np.where(media_missing(p[..., 1]) | ~np.isfinite(tok_dur), np.nan, u)
    out = p.copy()
    out[..., 1] = u.astype(np.float32)
    out[..., 2] = np.nan
    out[..., 3] = np.nan
    return out


# ----------------------------------------------------------------------------- decoding


def softplus(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def inv_softplus(y: Any) -> np.ndarray:
    y = np.maximum(np.asarray(y, dtype=np.float64), 1e-6)
    return np.where(y > 20.0, y, np.log(np.expm1(np.minimum(y, 20.0))))


def _nms_classwise(cls: np.ndarray, start: np.ndarray, end: np.ndarray, score: np.ndarray, iou: float, max_keep: int) -> np.ndarray:
    """Per-class greedy NMS then the ``max_keep`` best survivors, in one pass over segments shifted by ``2 · class``
    (segments in ``[0, 1]`` of different classes never overlap)."""
    order = np.argsort(-score, kind="stable")
    s = start[order] + 2.0 * cls[order]
    e = end[order] + 2.0 * cls[order]
    length = e - s
    suppressed = np.zeros(order.size, dtype=bool)
    keep: list[int] = []
    for i in range(order.size):
        if suppressed[i]:
            continue
        keep.append(i)
        if len(keep) >= max_keep:
            break
        inter = np.clip(np.minimum(e[i], e) - np.maximum(s[i], s), 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            suppressed |= (inter / (length[i] + length - inter)) > iou
    return order[np.asarray(keep, dtype=np.int64)]


def decode_row(raw: np.ndarray, duration_s: float) -> np.ndarray:
    """One row's raw outputs ``[T, C + 2]`` → ``float64 [m, 4]`` ``(class, start_s, end_s, score)``."""
    raw = np.asarray(raw, dtype=np.float64)
    T, C = raw.shape[0], raw.shape[1] - 2
    if C < 1:
        raise ValueError(f"detection outputs need C + 2 >= 3 channels, got {raw.shape}")
    prob = 0.5 * (1.0 + np.tanh(0.5 * raw[:, :C]))
    dist = softplus(raw[:, C:]) / float(T)
    centre = (np.arange(T, dtype=np.float64) + 0.5) / float(T)
    flat = prob.reshape(-1)
    cand = np.flatnonzero(flat >= float(DECODE["score_thresh"]))
    top = int(DECODE["pre_nms_topk"])
    if cand.size > top:
        cand = np.sort(cand[np.argpartition(-flat[cand], top - 1)[:top]])
    if cand.size == 0:
        return np.zeros((0, 4))
    k, c = np.divmod(cand, C)
    start = np.clip(centre[k] - dist[k, 0], 0.0, 1.0)
    end = np.clip(centre[k] + dist[k, 1], 0.0, 1.0)
    score = flat[cand]
    ok = end > start
    c, start, end, score = c[ok], start[ok], end[ok], score[ok]
    if score.size == 0:
        return np.zeros((0, 4))
    keep = _nms_classwise(c, start, end, score, float(DECODE["nms_iou"]), int(DECODE["max_per_row"]))
    d = float(duration_s)
    return np.stack([c[keep].astype(np.float64), start[keep] * d, end[keep] * d, score[keep]], axis=1)


def decode(outputs: Any, durations: Any) -> list[np.ndarray]:
    """Raw outputs ``[n, T, C + 2]`` + annotated durations ``[n]`` → per row ``[m, 4]`` scored segments in seconds."""
    raw = np.asarray(outputs)
    dur = np.asarray(durations, dtype=np.float64).reshape(-1)
    if raw.ndim != 3 or raw.shape[0] != dur.shape[0]:
        raise ValueError(f"detection decode: outputs {raw.shape} vs durations {dur.shape}")
    return [decode_row(raw[i], float(dur[i])) for i in range(raw.shape[0])]


# ----------------------------------------------------------------------------- ActivityNet detection AP


def segment_iou(target: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    tt1 = np.maximum(target[0], candidates[:, 0])
    tt2 = np.minimum(target[1], candidates[:, 1])
    inter = (tt2 - tt1).clip(0)
    union = (candidates[:, 1] - candidates[:, 0]) + (target[1] - target[0]) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return inter.astype(float) / union


def interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray) -> float:
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    mprec = np.maximum.accumulate(mprec[::-1])[::-1]
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def match_flags(gt_row: np.ndarray, gt_seg: np.ndarray, pred_row: np.ndarray, pred_seg: np.ndarray, pred_score: np.ndarray,
                thresholds: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy matching of ONE class: ``(order [p], tp [n_thr, p], fp [n_thr, p])``, predictions by decreasing score."""
    thr = np.asarray(thresholds, dtype=np.float64)
    order = np.argsort(-np.asarray(pred_score, dtype=np.float64), kind="stable")
    pv, ps = pred_row[order], pred_seg[order]
    lock = -np.ones((len(thr), len(gt_row)), dtype=np.int64)
    tp = np.zeros((len(thr), len(pv)))
    fp = np.zeros((len(thr), len(pv)))
    by_row = {int(v): np.flatnonzero(gt_row == v) for v in np.unique(gt_row)}
    for idx in range(len(pv)):
        g = by_row.get(int(pv[idx]))
        if g is None:
            fp[:, idx] = 1
            continue
        tiou = segment_iou(ps[idx], gt_seg[g])
        sorted_idx = np.argsort(tiou, kind="stable")[::-1]
        for t, tt in enumerate(thr):
            for j in sorted_idx:
                if tiou[j] < tt:
                    fp[t, idx] = 1
                    break
                if lock[t, g[j]] >= 0:
                    continue
                tp[t, idx] = 1
                lock[t, g[j]] = idx
                break
            if fp[t, idx] == 0 and tp[t, idx] == 0:
                fp[t, idx] = 1
    return order, tp, fp


def ap_from_flags(tp: np.ndarray, fp: np.ndarray, npos: float, weight: np.ndarray | None = None) -> np.ndarray:
    ap = np.zeros(tp.shape[0])
    if tp.shape[1] == 0 or npos <= 0:
        return ap
    if weight is not None:
        keep = weight > 0
        tp, fp, w = tp[:, keep], fp[:, keep], weight[keep].astype(np.float64)
        if tp.shape[1] == 0:
            return ap
        tp, fp = tp * w, fp * w
    tp_c = np.cumsum(tp, axis=1).astype(float)
    fp_c = np.cumsum(fp, axis=1).astype(float)
    rec, prec = tp_c / npos, tp_c / (tp_c + fp_c)
    for t in range(tp.shape[0]):
        ap[t] = interpolated_prec_rec(prec[t], rec[t])
    return ap


def _flatten(gt: Sequence[np.ndarray], preds: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(gt) != len(preds):
        raise ValueError(f"ap_matrix: {len(gt)} ground-truth rows vs {len(preds)} prediction rows")
    g_rows = [np.asarray(g, dtype=np.float64).reshape(-1, 3) for g in gt]
    p_rows = [np.asarray(p, dtype=np.float64).reshape(-1, 4) for p in preds]
    gv = np.concatenate([np.full(len(g), i, dtype=np.int64) for i, g in enumerate(g_rows)]) if g_rows else np.zeros(0, np.int64)
    pv = np.concatenate([np.full(len(p), i, dtype=np.int64) for i, p in enumerate(p_rows)]) if p_rows else np.zeros(0, np.int64)
    G = np.concatenate(g_rows, axis=0) if g_rows else np.zeros((0, 3))
    P = np.concatenate(p_rows, axis=0) if p_rows else np.zeros((0, 4))
    return gv, G, pv, P


def ap_matrix(gt: Sequence[np.ndarray], preds: Sequence[np.ndarray], thresholds: Sequence[float],
              excluded: Sequence[int] = ()) -> tuple[np.ndarray, np.ndarray]:
    """``(ap float64 [n_thr, K], classes [K])`` over the ``K`` classes present in the ground truth (``excluded`` classes
    are removed from the ground truth first). ``gt[i]`` ``[k, 3]`` ``(class, start, end)``; ``preds[i]`` ``[m, 4]``
    ``(class, start, end, score)``."""
    gv, G, pv, P = _flatten(gt, preds)
    if len(excluded):
        keep = ~np.isin(G[:, 0].astype(np.int64), np.asarray(list(excluded), dtype=np.int64))
        gv, G = gv[keep], G[keep]
    classes = np.unique(G[:, 0].astype(np.int64))
    ap = np.zeros((len(thresholds), len(classes)))
    gcls, pcls = G[:, 0].astype(np.int64), P[:, 0].astype(np.int64)
    for k, c in enumerate(classes):
        gm, pm = gcls == c, pcls == c
        if pm.any():
            _, tp, fp = match_flags(gv[gm], G[gm, 1:3], pv[pm], P[pm, 1:3], P[pm, 3], thresholds)
            ap[:, k] = ap_from_flags(tp, fp, float(gm.sum()))
    return ap, classes


def resampled_map(gt: Sequence[np.ndarray], preds: Sequence[np.ndarray], thresholds: Sequence[float], draws: np.ndarray,
                  excluded: Sequence[int] = ()) -> np.ndarray:
    """The mean AP (over thresholds and classes) of every clip-bootstrap resample in ``draws [samples, n]``. Matching
    never crosses rows, so it runs once; each resample weights predictions and ground truth by the clip multiplicities."""
    gv, G, pv, P = _flatten(gt, preds)
    if len(excluded):
        keep = ~np.isin(G[:, 0].astype(np.int64), np.asarray(list(excluded), dtype=np.int64))
        gv, G = gv[keep], G[keep]
    n = len(gt)
    gcls, pcls = G[:, 0].astype(np.int64), P[:, 0].astype(np.int64)
    per_class = []
    for c in np.unique(gcls):
        gm, pm = gcls == c, pcls == c
        order, tp, fp = match_flags(gv[gm], G[gm, 1:3], pv[pm], P[pm, 1:3], P[pm, 3], thresholds)
        per_class.append((np.bincount(gv[gm], minlength=n).astype(np.float64), pv[pm][order], tp, fp))
    out = np.empty(len(draws), dtype=np.float64)
    for b, draw in enumerate(np.asarray(draws)):
        w = np.bincount(draw, minlength=n).astype(np.float64)
        aps = [ap_from_flags(tp, fp, float((w * gc).sum()), w[pvid]) for gc, pvid, tp, fp in per_class if (w * gc).sum() > 0]
        out[b] = float(np.mean(aps, axis=0).mean()) if aps else np.nan
    return out


# ----------------------------------------------------------------------------- loss (torch inside)


def focal_giou(pred: Any, target: Any) -> Any:
    """``pred``, ``target`` ``[B, T, C + 2]`` → per-row loss ``[B]`` (module docstring)."""
    import torch
    from torch.nn import functional as F

    if pred.dim() != 3 or pred.shape != target.shape or pred.shape[-1] < 3:
        raise ValueError(f"focal_giou expects [B, T, C + 2] prediction and target, got {tuple(pred.shape)} / {tuple(target.shape)}")
    c = pred.shape[-1] - 2
    logits, cls_t = pred[..., :c], target[..., :c].to(pred.dtype)
    cls_t = torch.where(torch.isfinite(cls_t), cls_t, torch.zeros_like(cls_t))
    p = torch.sigmoid(logits)
    ce_ = F.binary_cross_entropy_with_logits(logits, cls_t, reduction="none")
    p_t = p * cls_t + (1 - p) * (1 - cls_t)
    alpha_t = FOCAL_ALPHA * cls_t + (1 - FOCAL_ALPHA) * (1 - cls_t)
    focal = alpha_t * ce_ * (1 - p_t) ** FOCAL_GAMMA
    ext_t = target[..., c:]
    pos = torch.isfinite(ext_t).all(dim=-1)
    safe_t = torch.where(pos.unsqueeze(-1), ext_t, torch.ones_like(ext_t)).to(pred.dtype)
    ext_p = F.softplus(pred[..., c:])
    inter = torch.minimum(ext_p[..., 0], safe_t[..., 0]) + torch.minimum(ext_p[..., 1], safe_t[..., 1])
    union = ext_p.sum(-1) + safe_t.sum(-1) - inter
    enclose = torch.maximum(ext_p[..., 0], safe_t[..., 0]) + torch.maximum(ext_p[..., 1], safe_t[..., 1])
    giou = inter / union.clamp_min(_EPS) - (enclose - union) / enclose.clamp_min(_EPS)
    reg = (1.0 - giou) * pos.to(pred.dtype)
    n_pos = pos.to(pred.dtype).sum(dim=1).clamp(min=1.0)
    b = pred.shape[0]
    return focal.reshape(b, -1).sum(dim=1) / n_pos + reg.reshape(b, -1).sum(dim=1) / n_pos


# ----------------------------------------------------------------------------- floor


def floor_outputs(train_targets: Any, n_eval: int, *, protocol: Mapping[str, Any] | None = None, label_kind: str | None = None,
                  num_classes: int | None = None) -> np.ndarray:
    """Constant raw outputs ``[n_eval, T, C + 2]``: per bin the logit of each class's train frequency and the inverse
    softplus of the median train start/end distances (1 bin where no train event covers it)."""
    import warnings

    t = np.asarray(train_targets, dtype=np.float64)
    if t.ndim != 3:
        raise ValueError(f"floor: detection train targets must be [n, T, C + 2], got {t.shape}")
    C = t.shape[2] - 2
    freq = np.clip(t[:, :, :C].mean(axis=0), 1e-6, 1 - 1e-6)
    logit = np.log(freq) - np.log1p(-freq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ext = np.nanmedian(t[:, :, C:], axis=0)
    ext = np.where(np.isfinite(ext), ext, 1.0)
    row = np.concatenate([logit, inv_softplus(np.maximum(ext, 1e-3))], axis=1).astype(np.float32)
    return np.broadcast_to(row, (int(n_eval), *row.shape)).copy()


# ----------------------------------------------------------------------------- the driver


def _inputs(ctx: Any, splits: Sequence[str], rows_of: Callable[[str], np.ndarray], fmt: str, *, T: int, num_classes: int,
            with_media: bool) -> dict[str, Any]:
    events: list[np.ndarray] = []
    durs, media, rows, ids = [], [], [], []
    for s in splits:
        r = np.asarray(rows_of(s), dtype=np.int64)
        md = ctx.durations(s, r) if with_media else None
        ev, d = row_events(ctx, s, r, fmt, media_durations=md)
        events.extend(ev)
        durs.append(d)
        if md is not None:
            media.append(np.asarray(md, dtype=np.float64))
        rows.append(r)
        ids.extend(ctx.clip_ids(s, r))
    ev_arr = np.empty(len(events), dtype=object)
    for i, e in enumerate(events):
        ev_arr[i] = e
    dur = np.concatenate(durs) if durs else np.zeros(0)
    return {"events": ev_arr, "duration_s": dur, "rows": np.concatenate(rows) if rows else np.zeros(0, np.int64), "clip_ids": ids,
            "media_durations": np.concatenate(media, axis=0) if media else None,
            "targets": dense_targets(events, dur, T=T, num_classes=num_classes)}


def _to_task(ctx: Any, feats: Any, media_durations: np.ndarray, annot: np.ndarray) -> Any:
    from yavih.readouts.base import ProbeRefused
    from yavih.readouts.coords import media_missing, task_pos

    base = feats.pos_gather()
    if base is None:
        return None
    out = task_pos(base, task_pos_fn, n=len(feats), N=feats.N, per_row=np.concatenate([media_durations, annot[:, None]], axis=1))
    k = min(len(feats), 64)
    if k:
        idx = np.arange(k)
        media = np.broadcast_to(base, (k, *base.shape)) if isinstance(base, np.ndarray) and base.ndim == 2 else np.asarray(base[idx])
        task = np.broadcast_to(out, (k, *out.shape)) if isinstance(out, np.ndarray) and out.ndim == 2 else np.asarray(out[idx])
        timed = ~media_missing(media[..., 1]) & (media[..., 0] >= 0)
        if (timed & ~np.isfinite(task[..., 1])).any():
            raise ProbeRefused("E_DURATIONS", "tokens carry pos.t but their stream's media.duration_s is unknown: the detection "
                                              "scorer cannot place them in time")
    return out


def run(ctx: Any, fmt: str) -> dict:
    """The scorer's driver: fit the pinned head on the train role, decode and score the eval role."""
    from yavih.readouts.base import ProbeRefused, evaluate_metrics, lazy_module

    _fit = lazy_module("yavih.readouts._fit")
    _fit.prepare(ctx)
    num_classes = _fit.num_classes(ctx)
    T = int(dict(ctx.spec.readout_params)["T"])
    t0 = time.time()

    def side(role: str) -> tuple[Any, dict[str, Any]]:
        splits = list(ctx.roles[role])
        feats = _fit.FeatureRows([(ctx.reader(s), ctx.present_rows(s)) for s in splits])
        return feats, _inputs(ctx, splits, ctx.present_rows, fmt, T=T, num_classes=num_classes, with_media=True)

    train, tr = side("train")
    ev, va = side("eval")
    if len(train) == 0 or len(ev) == 0:
        raise ProbeRefused("E_SPLIT_MISSING", "the detection scorer needs present train and eval rows")
    pos_train = _to_task(ctx, train, tr["media_durations"], tr["duration_s"])
    pos_eval = _to_task(ctx, ev, va["media_durations"], va["duration_s"])
    if pos_train is None or pos_eval is None:
        raise ProbeRefused("E_POS_MISSING", "the detection scorer needs pos.t (validate should have recorded eligible: false)")
    ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)

    def build(dim: int | None = None):
        from yavih.probes.dense import build_dense_query_head

        return build_dense_query_head(int(dim or ctx.D), num_classes + 2, dict(ctx.cfg.head))

    res = _fit.fit(ctx, train, tr["targets"], build_head=build, loss=ctx.cfg.loss, pos=pos_train)
    t1 = time.time()
    outputs = _fit.score(ctx, res.head, ev, pos=pos_eval)
    floor_out = floor_outputs(tr["targets"], len(ev))
    split_meta = {"readout": ctx.readout, "label_kind": ctx.label_kind, "num_classes": num_classes, "n_train": int(len(train)),
                  "n_eval": int(len(ev)), "n_val": int(res.n_val), "train_targets": tr["targets"], "eval_rows": va["rows"],
                  "eval_clip_ids": va["clip_ids"], "duration_s": va["duration_s"], "chance": dict(ctx.chance),
                  "bootstrap": dict(ctx.bootstrap or {}), "seed": ctx.seed}
    blocks = evaluate_metrics(ctx.metric_specs, outputs=outputs, targets=va["events"], protocol=ctx.protocol, split_meta=split_meta,
                              floor_outputs=floor_out, ci=ctx.ci, scorer=ctx.scorer(ctx.roles["eval"], va["rows"], va["events"]))
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t1)
    rec = _fit.fit_record(res)
    return {**blocks, "probe": rec, "fit": {**rec, "lr_grid": list(ctx.cfg.lr_grid), "device": ctx.device}, "eval_outputs": outputs,
            "eval_rows": va["rows"], "extra_arrays": {}, "backend": train.backend,
            "timing": {k: ctx.timing[k] for k in ("materialize_s", "fit_s", "score_s") if k in ctx.timing} | {"s_per_epoch": rec["s_per_epoch"]},
            "samples": ([{"role": "eval", "outputs": outputs, "targets": va["events"], "rows": va["rows"], "clip_ids": va["clip_ids"],
                          "duration_s": va["duration_s"]}] if ctx.log_samples else None)}


def sample_lines(part: Mapping[str, Any], *, seed: int, protocol: Mapping[str, Any]) -> list[dict]:
    """Samples-file lines: ``prediction`` the top-10 decoded ``[class, start_s, end_s, score]``, ``target`` the events."""
    dets = decode(part["outputs"], part["duration_s"])
    out = []
    for i, (row, cid) in enumerate(zip(np.asarray(part["rows"]), part["clip_ids"], strict=True)):
        out.append({"clip_id": str(cid), "row": int(row), "seed": int(seed), "role": str(part.get("role", "eval")),
                    "prediction": [[int(c), round(float(s), 3), round(float(e), 3), round(float(p), 4)] for c, s, e, p in dets[i][:10]],
                    "target": [[int(c), round(float(s), 3), round(float(e), 3)] for c, s, e in np.asarray(part["targets"][i]).reshape(-1, 3)]})
    return out


# ----------------------------------------------------------------------------- conformance population

#: Rows drawn per role for the conformance fixture.
POPULATION: dict[str, int] = {"train": 64, "eval": 64}


def conformance_population(ctx: Any, tag: str, fmt: str) -> dict[str, Any]:
    """The fixture population: real rows' events and annotated durations (``yavih.conformance.select_rows``), raw
    outputs ``[n, T, C + 2]`` from the stable stream: logits uniform in ``[-3, 3)``, extents whose softplus spans
    ``(0, 3]`` bins."""
    from yavih.conformance import select_rows, stream

    T = int(dict(ctx.spec.readout_params)["T"])
    C = int(ctx.bench.num_classes)

    def draw(role: str) -> dict[str, Any]:
        rows_of = (lambda s: select_rows(ctx.bench.manifest_hash, s, ctx.bench.n_rows(s), POPULATION[role], salt=role))
        return _inputs(ctx, list(ctx.roles[role]), rows_of, fmt, T=T, num_classes=C, with_media=False)

    tr, ev = draw("train"), draw("eval")
    n = len(ev["rows"])
    outputs = ((stream(f"{tag}:outputs", n * T * (C + 2)).reshape(n, T, C + 2) - 0.5) * 6.0).astype(np.float32)
    outputs[:, :, C:] = (stream(f"{tag}:extents", n * T * 2).reshape(n, T, 2) * 3.0).astype(np.float32)
    meta = {"readout": ctx.readout, "label_kind": ctx.label_kind, "num_classes": C, "n_train": int(len(tr["rows"])), "n_eval": int(n),
            "n_val": 0, "train_targets": tr["targets"], "eval_rows": ev["rows"], "eval_clip_ids": ev["clip_ids"],
            "duration_s": ev["duration_s"], "chance": dict(ctx.chance), "bootstrap": dict(ctx.bootstrap or {}), "seed": 0}
    return {"outputs": outputs, "targets": ev["events"], "split_meta": meta, "floor_outputs": floor_outputs(tr["targets"], n),
            "eval_splits": list(ctx.roles["eval"]), "eval_rows": ev["rows"]}


# ----------------------------------------------------------------------------- the descriptor and the metrics


def _head(*, num_classes: int | None, T: int | None = None, grid: Any = None, target_size: int | None = None) -> dict:
    return {"family": "dense_query_v1", "self_attn_depth": 3, "cross_attn_depth": 1, "num_heads": 16, "mlp_ratio": 4,
            "memory_pos": "sincos(t)+null_embed", "layout": f"time:{T if T is not None else '?'}",
            "out": (int(num_classes) + 2) if num_classes is not None else None}


def _geometry(cfg: Any) -> dict:
    from yavih.probes.config import parse_layout

    out = cfg.head.get("out")
    kw: dict[str, Any] = {"num_classes": (int(out) - 2) if isinstance(out, int) else None}
    layout = str(cfg.head.get("layout", ""))
    if "?" not in layout:
        kind, shape = parse_layout(layout)
        if kind == "time":
            kw["T"] = shape[0]
    return kw


def _validate_spec(params: dict, requires: Any, protocol: Any) -> None:
    T = params["T"]
    if not isinstance(T, int) or isinstance(T, bool) or not 8 <= T <= 1024:
        raise ValueError(f".T: an int in [8, 1024], got {T!r}")
    if "t" not in tuple(requires.pos or ()):
        raise ValueError(".requires.pos: the detection scorer needs [t]")
    thr = (protocol or {}).get("tiou_thresholds")
    if not isinstance(thr, (list, tuple)) or not thr or any(not isinstance(t, (int, float)) or not 0 < float(t) <= 1 for t in thr):
        raise ValueError("protocol.tiou_thresholds: a non-empty list of numbers in (0, 1]")


def make_scorer(name: str, *, module: str, events_format: str, version: int = 1) -> Any:
    """The folder's scorer, named after it: kind ``probe`` (the core ``dense_query_v1`` head, the folder's own loss
    ``<name>_focal_giou``), driver :func:`run`, its own floor, conformance population and samples lines."""
    from yavih.readouts.registry import Scorer

    if events_format not in EVENT_FORMATS:
        raise ValueError(f"events_format {events_format!r} is not one of {sorted(EVENT_FORMATS)}")
    loss = f"{name}_focal_giou"
    return Scorer(
        name=name, version=int(version), module=module, kind="probe", label_kinds=frozenset({"multi"}),
        protocol_keys=frozenset({"tiou_thresholds"}), param_keys=frozenset({"T"}), requires_t_bins=False, requires_grid=False,
        validate_spec=_validate_spec, dense=True, train=TRAIN, head=_head, loss=lambda label_kind: loss,
        task_sizes=lambda bench: {"T": int(dict(bench.spec.readout_params)["T"])}, geometry=_geometry,
        head_families=frozenset({"dense_query_v1"}), losses=frozenset({loss}), loss_fn=lambda _name: focal_giou,
        floor_outputs=floor_outputs, score=lambda ctx: run(ctx, events_format),
        conformance_population=lambda ctx, tag: conformance_population(ctx, tag, events_format), sample_lines=sample_lines,
    )


def _detections(outputs: Any, split_meta: Mapping[str, Any]) -> list[np.ndarray]:
    cache = split_meta.get("_det_cache") if isinstance(split_meta, dict) else None
    if cache is None and isinstance(split_meta, dict):
        cache = split_meta["_det_cache"] = {}
    if cache is not None and id(outputs) in cache and cache[id(outputs)][0] is outputs:
        return cache[id(outputs)][1]
    dets = decode(outputs, split_meta["duration_s"])
    if cache is not None:
        cache[id(outputs)] = (outputs, dets)
    return dets


def _floor_or(outputs: Any, targets: Any, split_meta: Mapping[str, Any]) -> Any:
    return floor_outputs(split_meta["train_targets"], len(list(targets))) if outputs is None else outputs


def make_metrics(name: str, *, mean: str, per_tiou: str | None = None, at: Mapping[str, int] | None = None, floor: str,
                 n_events: str | None = None, excluded_key: str | None = None, scale: float = 1.0) -> dict[str, Any]:
    """The folder's metrics, under the folder's own names: ``mean`` (the mean AP over ``protocol.tiou_thresholds`` and
    the classes present in the ground truth, with a clip-bootstrap interval), ``per_tiou`` (the list per threshold),
    ``at`` (``{metric: threshold index}``), ``floor`` (``mean`` of the floor output) and ``n_events`` (a disclosure).
    ``excluded_key`` names a protocol list of classes removed from the ground truth; ``scale`` multiplies every AP
    (100 for a percentage)."""
    from yavih.metrics.intervals import bootstrap_draws
    from yavih.metrics.registry import MetricSpec

    keys = frozenset({"tiou_thresholds"} | ({excluded_key} if excluded_key else set()))

    def thr(protocol: Mapping[str, Any]) -> list[float]:
        return [float(t) for t in protocol["tiou_thresholds"]]

    def excl(protocol: Mapping[str, Any]) -> list[int]:
        return [int(c) for c in protocol.get(excluded_key, [])] if excluded_key else []

    def ap(outputs, targets, protocol, split_meta) -> np.ndarray:
        gt = list(targets)
        if not any(len(np.asarray(g).reshape(-1, 3)) for g in gt):
            raise ValueError("detection: the eval rows hold no ground-truth event")
        return ap_matrix(gt, _detections(outputs, split_meta), thr(protocol), excl(protocol))[0] * scale

    def f_mean(outputs, targets, protocol, split_meta) -> float:
        return float(ap(outputs, targets, protocol, split_meta).mean(axis=1).mean())

    def f_interval(kind, outputs, targets, protocol, split_meta) -> tuple[float, float]:
        cfg = {"samples": 2000, "seed": 0, **dict(split_meta.get("bootstrap") or {})}
        draws = bootstrap_draws(len(list(targets)), samples=int(cfg["samples"]), seed=int(cfg["seed"]))
        stats = resampled_map(list(targets), _detections(outputs, split_meta), thr(protocol), draws, excl(protocol)) * scale
        stats = stats[np.isfinite(stats)]
        lo, hi = np.percentile(stats, [2.5, 97.5])
        return float(lo), float(hi)

    ro = frozenset({name})
    out: dict[str, Any] = {
        mean: MetricSpec(name=mean, fn=f_mean, readouts=ro, protocol_keys=keys, ci=frozenset({"bootstrap_clip"}), interval=f_interval,
                         description="mean ActivityNet detection AP over protocol.tiou_thresholds and the classes in the ground truth"),
        floor: MetricSpec(name=floor, fn=lambda o, t, p, sm: f_mean(_floor_or(o, t, sm), t, p, sm), readouts=ro, protocol_keys=keys,
                          kind="floor", floor_of=(mean,), description=f"{mean} of the label-only constant output"),
    }
    if per_tiou:
        out[per_tiou] = MetricSpec(name=per_tiou, fn=lambda o, t, p, sm: [float(v) for v in ap(o, t, p, sm).mean(axis=1)], readouts=ro,
                                   protocol_keys=keys, shape="list", description="mean AP at each of protocol.tiou_thresholds")
    for metric, k in (at or {}).items():
        out[metric] = MetricSpec(name=metric, fn=(lambda k: lambda o, t, p, sm: float(ap(o, t, p, sm).mean(axis=1)[k]))(int(k)), readouts=ro,
                                 protocol_keys=keys, description=f"mean AP at protocol.tiou_thresholds[{int(k)}]")
    if n_events:
        out[n_events] = MetricSpec(name=n_events, fn=lambda o, t, p, sm: float(sum(len(np.asarray(g).reshape(-1, 3)) for g in t)),
                                   readouts=ro, kind="disclosure", higher_is_better=None, description="ground-truth events scored")
    return out
