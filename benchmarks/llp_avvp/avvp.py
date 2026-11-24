"""``llp_avvp``: weakly-supervised audio-visual video parsing, the folder's own scorer.

The protocol is the LLP authors' (Tian, Li & Xu, ECCV 2020; https://github.com/YapengTian/AVVP-ECCV20 at commit
``fde56118f2c2e7c19188dfefa2569a37b70054fe``, GPL-3.0 as its README states). Every part is theirs:

* **Scorer** (``avvp_eval_metrics.py``, their ``utils/eval_metrics.py`` byte for byte, and :func:`video_scores` /
  :func:`summary`, transcribed from ``main_avvp.py::eval``): per test video, the video-level prediction is
  ``global_prob >= 0.5``; the per-second audio and visual predictions are ``frame_prob >= 0.5`` masked by it; the
  ground truth is the 25 x 10 matrix of each modality's annotated seconds (``GT[c, onset:offset] = 1``), the
  audio-visual one their product. ``segment_level`` and ``event_level`` give four F-scores per video (audio, visual,
  audio and visual events pooled, audio-visual); each is averaged over videos and printed in percent; Type@AV is the
  mean of the audio, visual and audio-visual averages, Event@AV the pooled one.
* **Readout** (``han.py``, their ``nets/net_audiovisual.py``, and :func:`han_loss` / :func:`_fit_han`, transcribed
  from ``main_avvp.py::train`` and ``main``): the Hybrid Attention Network with attentive MMIL pooling, trained on
  the frozen features with the video-level labels only: Adam (lr 3e-4), StepLR (10 epochs, x0.1), 40 epochs,
  batch 16, BCE on the video-level prediction plus BCE on the audio and visual video-level predictions against the
  label smoothed by 1.0 (audio) and 0.9 (visual); the epoch kept is the one with the best segment-level Type@AV on
  the official val split (649 videos with dense labels, never trained on), ties to the later epoch.

What yavih adds (the bridge, not the protocol, chooses the features): each bridge token is placed in the second
``floor(t * duration_s / 1 s)`` of its own stream, and the tokens of a stream are mean-pooled per second (no
token: a zero vector), giving the ``[10, D]`` audio and visual inputs HAN reads where the authors use one VGGish
and eight ResNet-152 frame features (avg-pooled) plus one R(2+1)D feature per second. Audio tokens feed the audio
branch, video tokens the visual one; a reading that hands over one stream only feeds it to both branches.

Raw outputs ``float32 [n, 23, 25]`` (probabilities): rows 0-9 the audio ``frame_prob`` of seconds 0-9, rows 10-19
the visual ones, row 20 ``global_prob``, rows 21-22 the audio and visual video-level ``a_prob``, ``v_prob``.
Targets of a scored video ``float32 [2, 25, 10]``: ``GT_a``, ``GT_v``.

Module scope: stdlib, numpy, yavih core; torch is imported in the head, the loss and the driver.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["T", "C", "OUT_ROWS", "THRESHOLD", "TRAIN", "HEAD_FAMILY", "LOSS", "FIT_SPLIT", "SELECTION_SPLIT", "METRIC_NAMES",
           "parse_events", "gt_matrices", "video_scores", "summary", "floor_outputs", "han_loss", "make_scorer", "make_metrics"]

T = 10  # seconds per clip, one segment each (eval_metrics.py hard-codes 10)
C = 25  # classes (eval_metrics.py hard-codes 25)
OUT_ROWS = 2 * T + 3
THRESHOLD = 0.5  # main_avvp.py::eval
HEAD_FAMILY = "llp_avvp_han_v1"
LOSS = "llp_avvp_mmil_bce"
FIT_SPLIT = "train"  # the official train split: weak labels, fitted
SELECTION_SPLIT = "val"  # the official val split: dense labels, epoch selection only (main_avvp.py::main)
#: main_avvp.py's argparse defaults (--lr 3e-4, --batch-size 16, --epochs 40) and optimiser (Adam, no weight decay);
#: no early stopping (patience = epochs); the val carve, warmup and cosine of the core trainer are not used.
TRAIN: dict[str, Any] = dict(lr=3e-4, lr_grid=(), weight_decay=0.0, max_epochs=40, batch_size=16, patience=40, warmup_fraction=0.0)
#: what main_avvp.py fixes beyond TRAIN (hashed through the head descriptor)
RECIPE: dict[str, Any] = {"optimizer": "adam", "lr_step": {"step_size": 10, "gamma": 0.1}, "label_smoothing": {"a": 1.0, "v": 0.9},
                          "clamp": 1e-7, "selection": "val:seg_type_av:>="}
#: per-video columns of :func:`video_scores`
COLUMNS = ("seg_a", "seg_v", "seg_all", "seg_av", "evt_a", "evt_v", "evt_all", "evt_av")
#: the folder's metric names (percent, as main_avvp.py prints them) → how each is formed from the per-video columns
METRIC_NAMES: dict[str, tuple[str, ...]] = {
    "seg_a": ("seg_a",), "seg_v": ("seg_v",), "seg_av": ("seg_av",), "seg_type_av": ("seg_av", "seg_a", "seg_v"), "seg_event_av": ("seg_all",),
    "evt_a": ("evt_a",), "evt_v": ("evt_v",), "evt_av": ("evt_av",), "evt_type_av": ("evt_av", "evt_a", "evt_v"), "evt_event_av": ("evt_all",),
}
DESCRIPTIONS = {
    "seg_a": "segment-level audio event F-score (%)", "seg_v": "segment-level visual event F-score (%)",
    "seg_av": "segment-level audio-visual event F-score (%)", "seg_type_av": "segment-level Type@AV: mean of seg_a, seg_v, seg_av (%)",
    "seg_event_av": "segment-level Event@AV: audio and visual events pooled (%)", "evt_a": "event-level audio F-score, IoU 0.5 (%)",
    "evt_v": "event-level visual F-score, IoU 0.5 (%)", "evt_av": "event-level audio-visual F-score, IoU 0.5 (%)",
    "evt_type_av": "event-level Type@AV: mean of evt_a, evt_v, evt_av (%)", "evt_event_av": "event-level Event@AV: audio and visual pooled (%)",
}
FLOORS = ("seg_type_av", "seg_event_av", "evt_type_av", "evt_event_av")


def _sibling(name: str):
    """A file next to this one, imported by path (keyed on its content, like yavih loads eval.py)."""
    path = Path(__file__).resolve().with_name(name)
    data = path.read_bytes()
    mod_name = f"yavih_folder_{path.parent.name}_{path.stem}_{hashlib.sha256(data).hexdigest()[:12]}"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
    return sys.modules[mod_name]


# ----------------------------------------------------------------------------- labels


def parse_events(cell: Any) -> list[tuple[int, int, int]]:
    """An events cell (``onset:offset:class`` tokens, ``;``-joined; ``""`` / ``None`` = none) → ``[(onset, offset,
    class)]`` in cell order. Offsets at or before the onset are kept (they mark nothing, as upstream)."""
    s = "" if cell is None else str(cell).strip()
    out = []
    for tok in (s.split(";") if s else []):
        parts = tok.split(":")
        if len(parts) != 3:
            raise ValueError(f"event token {tok!r} is not onset:offset:class")
        on, off, c = (int(p) for p in parts)
        if not (0 <= on <= T and 0 <= off <= T and 0 <= c < C):
            raise ValueError(f"event token {tok!r} is outside [0, {T}] s or the {C} classes")
        out.append((on, off, c))
    return out


def gt_matrices(audio_cell: Any, visual_cell: Any) -> np.ndarray:
    """``float32 [2, C, T]``: ``GT_a``, ``GT_v`` as main_avvp.py::eval builds them (``GT[idx, x1:x2] = 1``)."""
    out = np.zeros((2, C, T), dtype=np.float32)
    for m, cell in enumerate((audio_cell, visual_cell)):
        for x1, x2, idx in parse_events(cell):
            out[m, idx, x1:x2] = 1
    return out


def _rows_index(manifest: Any, rows: np.ndarray) -> np.ndarray:
    from yavih.readouts.base import _manifest_index

    return _manifest_index(manifest, np.asarray(rows, dtype=np.int64))


def gt_targets(bench: Any, split: str, rows: np.ndarray) -> np.ndarray:
    """``float32 [n, 2, C, T]`` ground truth of within-split ``rows`` from the ``audio_events`` / ``visual_events`` cells."""
    from yavih.readouts.base import ProbeRefused

    m = bench.manifest(split)
    out = np.zeros((len(rows), 2, C, T), dtype=np.float32)
    for j, i in enumerate(_rows_index(m, rows)):
        ex = m.extra[i] or {}
        try:
            out[j] = gt_matrices(ex.get("audio_events"), ex.get("visual_events"))
        except ValueError as e:
            raise ProbeRefused("E_LABELS", f"row {m.clip_id[i]}: {e}") from e
    return out


def weak_targets(bench: Any, split: str, rows: np.ndarray) -> np.ndarray:
    """``float32 [n, C]`` video-level (weak) multi-hot labels of within-split ``rows`` (``labels``)."""
    m = bench.manifest(split)
    out = np.zeros((len(rows), C), dtype=np.float32)
    for j, i in enumerate(_rows_index(m, rows)):
        for c in m.label_ids[i] or []:
            out[j, int(c)] = 1.0
    return out


# ----------------------------------------------------------------------------- the authors' evaluation


def video_scores(outputs: Any, gt: Any) -> np.ndarray:
    """``float64 [n, 8]`` (:data:`COLUMNS`): the four segment-level then the four event-level F-scores of each video,
    as main_avvp.py::eval computes them (one video at a time, batch size 1)."""
    em = _sibling("avvp_eval_metrics.py")
    out = np.asarray(outputs, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if out.ndim != 3 or out.shape[1:] != (OUT_ROWS, C) or gt.shape != (out.shape[0], 2, C, T):
        raise ValueError(f"llp_avvp: outputs [n, {OUT_ROWS}, {C}] and targets [n, 2, {C}, {T}] expected, got {out.shape} / {gt.shape}")
    res = np.zeros((out.shape[0], len(COLUMNS)), dtype=np.float64)
    for i in range(out.shape[0]):
        o = (out[i, 2 * T][None] >= THRESHOLD).astype(np.int_)  # main_avvp.py: output [1, 25] >= 0.5
        Pa = out[i, 0:T]
        Pv = out[i, T:2 * T]
        # filter out false positive events with predicted weak labels
        Pa = (Pa >= THRESHOLD).astype(np.int_) * np.repeat(o, repeats=T, axis=0)
        Pv = (Pv >= THRESHOLD).astype(np.int_) * np.repeat(o, repeats=T, axis=0)
        GT_a = gt[i, 0].astype(np.float64)
        GT_v = gt[i, 1].astype(np.float64)
        GT_av = GT_a * GT_v
        # obtain prediction matrices
        SO_a = np.transpose(Pa)
        SO_v = np.transpose(Pv)
        SO_av = SO_a * SO_v
        res[i, 0:4] = em.segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)  # f_a, f_v, f, f_av
        res[i, 4:8] = em.event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
    return res


def summary(per_video: np.ndarray) -> dict[str, float]:
    """The ten numbers main_avvp.py::eval prints (percent) from :func:`video_scores`."""
    col = {name: i for i, name in enumerate(COLUMNS)}
    pv = np.asarray(per_video, dtype=np.float64)
    out = {}
    for metric, parts in METRIC_NAMES.items():
        out[metric] = float(sum(100 * np.mean(pv[:, col[p]]) for p in parts) / len(parts))
    return out


def per_clip(per_video: np.ndarray, metric: str) -> np.ndarray:
    """``[n]`` per-video values whose mean is ``metric`` (the clip bootstrap resamples them)."""
    col = {name: i for i, name in enumerate(COLUMNS)}
    parts = METRIC_NAMES[metric]
    return 100.0 * np.mean(np.asarray(per_video, dtype=np.float64)[:, [col[p] for p in parts]], axis=1)


# ----------------------------------------------------------------------------- floor


def floor_outputs(train_targets: Any, n_eval: int, *, protocol: Mapping[str, Any] | None = None, label_kind: str | None = None,
                  num_classes: int | None = None) -> np.ndarray:
    """The label-only constant output ``[n_eval, OUT_ROWS, C]``: every row (each second of both modalities, the
    video level) the train prevalence of each class's weak label. No class is in at least half the train videos
    (``Speech``: 4 988 of 10 000), so at the 0.5 threshold it predicts nothing: the floor is the score of an empty
    parse, which the authors' F-score counts as 1 on a video without ground truth in that modality and 0 elsewhere."""
    t = np.asarray(train_targets, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != C or t.shape[0] == 0:
        raise ValueError(f"llp_avvp floor: train targets must be [n > 0, {C}] weak labels, got {t.shape}")
    row = np.broadcast_to(t.mean(axis=0).astype(np.float32), (OUT_ROWS, C))
    return np.broadcast_to(row, (int(n_eval), OUT_ROWS, C)).copy()


# ----------------------------------------------------------------------------- loss (torch inside)


def han_loss(pred: Any, target: Any) -> Any:
    """Per-sample loss ``[B]`` of main_avvp.py::train: ``BCE(a_prob, Pa) + BCE(v_prob, Pv) + BCE(global_prob, y)``
    (each a mean over the classes), probabilities clamped to ``[1e-7, 1 - 1e-7]``, ``Pa = 1.0 y + 0.0 · 0.5`` and
    ``Pv = 0.9 y + 0.1 · 0.5`` (label smoothing). ``pred`` ``[B, OUT_ROWS, C]``, ``target`` ``[B, C]`` weak labels."""
    import torch
    from torch.nn import functional as F

    eps = float(RECIPE["clamp"])
    output = pred[:, 2 * T].clamp(min=eps, max=1 - eps)
    a_prob = pred[:, 2 * T + 1].clamp(min=eps, max=1 - eps)
    v_prob = pred[:, 2 * T + 2].clamp(min=eps, max=1 - eps)
    target = target.to(pred.dtype)
    # label smoothing
    a = float(RECIPE["label_smoothing"]["a"])
    v = float(RECIPE["label_smoothing"]["v"])
    Pa = a * target + (1 - a) * 0.5
    Pv = v * target + (1 - v) * 0.5
    # individual guided learning
    per = (F.binary_cross_entropy(a_prob, Pa, reduction="none").mean(dim=1) + F.binary_cross_entropy(v_prob, Pv, reduction="none").mean(dim=1)
           + F.binary_cross_entropy(output, target, reduction="none").mean(dim=1))
    return per if isinstance(per, torch.Tensor) else torch.as_tensor(per)


# ----------------------------------------------------------------------------- head (torch inside)


def build_net(dim: int) -> Any:
    """``han.py``'s ``MMIL_Net(dim)`` wrapped to take ``[B, 2, T, dim]`` (audio, visual per second) and return the packed
    ``[B, OUT_ROWS, C]`` probabilities."""
    import warnings

    import torch
    from torch import nn

    han = _sibling("han.py")

    class LLPHead(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            with warnings.catch_warnings():  # nn.TransformerEncoder's nested-tensor note for the unused upstream encoders
                warnings.simplefilter("ignore", UserWarning)
                self.net = han.MMIL_Net(int(d))

        def forward(self, x: Any) -> Any:
            global_prob, a_prob, v_prob, frame_prob = self.net(x[:, 0], x[:, 1])
            return torch.cat([frame_prob[:, :, 0, :], frame_prob[:, :, 1, :], global_prob[:, None], a_prob[:, None], v_prob[:, None]], dim=1)

    return LLPHead(dim)


def _build_head(cfg: Any, *, visual_tag: int | None = None, conditioning_tag: int | None = None):
    return lambda dim: build_net(int(dim))


# ----------------------------------------------------------------------------- per-second pooling of the bridge's tokens


def stream_branches(names: Sequence[str], kinds: Mapping[str, str]) -> np.ndarray:
    """``int64 [S]``: the HAN branch of each tag of ``describe()["streams"]``: 0 audio, 1 visual, 2 both (the only
    one of the two kinds the features carry), -1 neither."""
    k = [str(kinds.get(n, "")) for n in names]
    has_a, has_v = "audio" in k, "video" in k
    out = np.full(len(k), -1, dtype=np.int64)
    for i, kind in enumerate(k):
        if kind == "audio":
            out[i] = 0 if has_v else 2
        elif kind == "video":
            out[i] = 1 if has_a else 2
    return out


def token_slots(pos: np.ndarray, durations: np.ndarray, branches: np.ndarray, *, segment_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """``pos [k, N, 4]`` media coordinates, ``durations [k, S]`` per stream → ``(branch [k, N], second [k, N])`` int64:
    the token's branch (:func:`stream_branches`, -1 = unused) and second ``floor(t · duration_s / segment_seconds)``
    of its own stream (-1 = no time, outside ``[0, T)`` or unknown duration)."""
    from yavih.readouts.coords import media_missing

    p = np.asarray(pos, dtype=np.float64)
    dur = np.asarray(durations, dtype=np.float64)
    tag = p[..., 0]
    ok = np.isfinite(tag) & (tag >= 0) & (tag < len(branches)) & (tag == np.round(tag))
    tag_i = np.where(ok, tag, 0).astype(np.int64)
    br = np.where(ok, branches[tag_i], -1)
    td = np.where(ok, dur[np.arange(p.shape[0])[:, None], tag_i], np.nan)
    t = p[..., 1]
    good = ok & ~media_missing(t) & (t >= 0) & np.isfinite(td)
    with np.errstate(invalid="ignore"):
        sec = np.floor(np.where(good, t * td / float(segment_seconds), -1.0) + 1e-9)
    sec = np.where(good & (sec >= 0) & (sec < T), sec, -1).astype(np.int64)
    return br.astype(np.int64), sec


def pool_seconds(x: np.ndarray, branch: np.ndarray, second: np.ndarray) -> np.ndarray:
    """``x [k, N, D]`` tokens → ``float32 [k, 2, T, D]``: per branch (audio, visual) and second the mean of its tokens
    (a ``both`` token counts in both branches); no token → zeros."""
    x = np.asarray(x, dtype=np.float32)
    k, n, d = x.shape
    out = np.zeros((k, 2, T, d), dtype=np.float32)
    for b in (0, 1):
        sel = ((branch == b) | (branch == 2)) & (second >= 0)
        w = np.zeros((k, T, n), dtype=np.float32)
        ii, jj = np.nonzero(sel)
        w[ii, second[ii, jj], jj] = 1.0
        cnt = w.sum(axis=2, keepdims=True)
        out[:, b] = np.einsum("ktn,knd->ktd", w, x) / np.maximum(cnt, 1.0)
    return out


def pooled_inputs(ctx: Any, split: str, rows: np.ndarray, *, chunk: int = 64) -> np.ndarray:
    """``float32 [n, 2, T, D]`` per-second inputs of the present ``rows`` of ``split``, read in chunks."""
    from yavih.readouts._fit import FeatureRows
    from yavih.readouts.base import ProbeRefused

    feats = FeatureRows([(ctx.reader(split), rows)])
    base = feats.pos_gather()
    if base is None:
        raise ProbeRefused("E_POS_MISSING", "llp_avvp needs pos.t (validate should have recorded eligible: false)")
    names = ctx.stream_names()
    kinds = {s: st.kind for s, st in ctx.bench.spec.streams.items()}
    branches = stream_branches(names, kinds)
    if not (branches >= 0).any():
        raise ProbeRefused("E_STREAM", f"llp_avvp: describe()['streams'] = {names} holds neither an audio nor a video stream")
    dur = ctx.durations(split, rows)
    seg_s = float(ctx.protocol.get("segment_seconds", 1.0))
    out = np.zeros((len(rows), 2, T, feats.D), dtype=np.float32)
    timed = 0
    for i in range(0, len(rows), chunk):
        ctx.check_interrupt()
        ids = np.arange(i, min(len(rows), i + chunk))
        x = feats[ids]
        p = np.broadcast_to(base, (len(ids), *np.asarray(base).shape)) if isinstance(base, np.ndarray) and base.ndim == 2 else np.asarray(base[ids])
        br, sec = token_slots(p, dur[ids], branches, segment_seconds=seg_s)
        timed += int((sec >= 0).any(axis=1).sum())
        out[ids] = pool_seconds(x, br, sec)
    if len(rows) and timed == 0:
        raise ProbeRefused("E_DURATIONS", "no token of any row falls in a second of the clip (pos.t, media.duration_s): llp_avvp cannot "
                                          "place them in time")
    return out


# ----------------------------------------------------------------------------- the driver


def _predict(model: Any, X: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    import torch

    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(device).float()).float().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, OUT_ROWS, C), dtype=np.float32)


def _fit_han(ctx: Any, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, gtva: np.ndarray) -> tuple[Any, dict]:
    """main_avvp.py::main's train mode: train on the fit split, evaluate the selection split after every epoch, keep
    the best segment-level Type@AV (``>=``). Seeded (``fork_rng`` + ``manual_seed``) under the core's precision scope;
    the stop flag and the deadline are checked between batches, the last completed epoch is checkpointed."""
    import torch

    from yavih.hashing import probe_hash
    from yavih.probes.ckpt import load_ckpt, rng_state, save_ckpt, set_rng_state
    from yavih.probes.trainer import ckpt_every_s, precision_scope
    from yavih.readouts._fit import ckpt_path_for
    from yavih.readouts.base import ProbeInterrupted

    cfg, device, seed = ctx.cfg, ctx.device, int(ctx.seed)
    n = int(len(Xtr))
    bs = min(int(cfg.batch_size), n)
    step = dict(RECIPE["lr_step"])
    ckpt = ckpt_path_for(ctx, "han")
    hashes = {"fit": "han", "probe_hash": probe_hash(cfg), "seed": seed, "features_fingerprint": ctx.features_fingerprint, "n_rows": n}
    rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    every_s = ckpt_every_s()
    with torch.random.fork_rng(devices=rng_devices), precision_scope():
        torch.manual_seed(seed)
        model = build_net(int(Xtr.shape[-1])).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(step["step_size"]), gamma=float(step["gamma"]))
        gen = torch.Generator().manual_seed(seed)
        st: dict[str, Any] = {"epoch": 0, "best": -math.inf, "best_epoch": 0, "best_state": None, "history": [], "s_total": 0.0}
        resumed = load_ckpt(str(ckpt), hashes=hashes)
        if resumed is not None:
            model.load_state_dict(resumed["state_dict"])
            optimizer.load_state_dict(resumed["optimizer"])
            scheduler.load_state_dict(resumed["scheduler"])
            set_rng_state(resumed["rng"])
            gen.set_state(resumed["loader_rng"])
            st = dict(resumed["st"])
        Xt = torch.from_numpy(Xtr)
        yt = torch.from_numpy(np.asarray(ytr, dtype=np.float32))
        pending: dict[str, Any] = {"snap": None, "written": time.monotonic()}

        def snapshot() -> dict:
            return {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(), "loader_rng": gen.get_state(),
                    "st": dict(st)}

        try:
            while st["epoch"] < int(cfg.max_epochs):
                t0 = time.monotonic()
                model.train()
                perm = torch.randperm(n, generator=gen)
                loss_sum = 0.0
                for i in range(0, n, bs):
                    ctx.check_interrupt()
                    idx = perm[i:i + bs]
                    x = Xt[idx].to(device).float()
                    y = yt[idx].to(device)
                    optimizer.zero_grad()
                    loss = han_loss(model(x), y).mean()
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.item()) * int(len(idx))
                scheduler.step()
                f_val = summary(video_scores(_predict(model, Xva, device), gtva))["seg_type_av"]
                st["epoch"] += 1
                dt = time.monotonic() - t0
                st["s_total"] += dt
                st["history"].append({"epoch": st["epoch"], "train_loss": loss_sum / max(1, n), "val_seg_type_av": f_val,
                                      "lr": float(optimizer.param_groups[0]["lr"]), "s": dt})
                if f_val >= st["best"]:
                    st["best"], st["best_epoch"] = f_val, st["epoch"]
                    st["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                pending["snap"] = snapshot()
                if every_s <= 0 or time.monotonic() - pending["written"] >= every_s:
                    save_ckpt(str(ckpt), pending["snap"], hashes=hashes)
                    pending["written"] = time.monotonic()
        except ProbeInterrupted:
            if pending["snap"] is not None:
                save_ckpt(str(ckpt), pending["snap"], hashes=hashes)
            raise
        if st["best_state"] is not None:
            model.load_state_dict(st["best_state"])
    model.eval()
    ckpt.unlink(missing_ok=True)
    rec = {"selected_lr": float(cfg.lr), "epochs_run": int(st["epoch"]), "best_epoch": int(st["best_epoch"]), "stop": "max_epochs",
           "n_train": n, "n_val": int(len(Xva)), "s_per_epoch": float(st["s_total"] / max(1, st["epoch"])),
           "best_val": None, "best_val_seg_type_av": float(st["best"]) if np.isfinite(st["best"]) else None, "arms": [],
           "history": list(st["history"]), "selection": f"{SELECTION_SPLIT}: seg_type_av"}
    return model, rec


def _present(ctx: Any, split: str) -> np.ndarray:
    try:
        return ctx.present_rows(split)
    except Exception:  # noqa: BLE001 - a split without features
        return np.zeros(0, dtype=np.int64)


def run(ctx: Any) -> dict:
    """The scorer's driver: pool the bridge's tokens per second, fit HAN on the fit split selecting on the selection
    split, score the eval role."""
    from yavih.readouts.base import ProbeRefused, evaluate_metrics, lazy_module

    _fit = lazy_module("yavih.readouts._fit")
    _fit.prepare(ctx)
    if FIT_SPLIT not in ctx.roles["train"] or SELECTION_SPLIT not in ctx.roles["train"]:
        raise ProbeRefused("E_SPLIT_MISSING", f"llp_avvp trains on split {FIT_SPLIT!r} and selects on {SELECTION_SPLIT!r}: both must be train-role splits")
    t0 = time.time()
    r_tr, r_va = _present(ctx, FIT_SPLIT), _present(ctx, SELECTION_SPLIT)
    if len(r_tr) == 0 or len(r_va) == 0:
        raise ProbeRefused("E_SPLIT_MISSING", f"llp_avvp needs present rows of {FIT_SPLIT!r} and {SELECTION_SPLIT!r}")
    Xtr, Xva = pooled_inputs(ctx, FIT_SPLIT, r_tr), pooled_inputs(ctx, SELECTION_SPLIT, r_va)
    ytr = weak_targets(ctx.bench, FIT_SPLIT, r_tr)
    gtva = gt_targets(ctx.bench, SELECTION_SPLIT, r_va)
    ev_splits = list(ctx.roles["eval"])
    ev_rows = [(s, _present(ctx, s)) for s in ev_splits]
    if not sum(len(r) for _, r in ev_rows):
        raise ProbeRefused("E_SPLIT_MISSING", "llp_avvp needs present eval rows")
    Xev = np.concatenate([pooled_inputs(ctx, s, r) for s, r in ev_rows], axis=0)
    gtev = np.concatenate([gt_targets(ctx.bench, s, r) for s, r in ev_rows], axis=0)
    rows_ev = np.concatenate([r for _, r in ev_rows])
    ids_ev = [c for s, r in ev_rows for c in ctx.clip_ids(s, r)]
    ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)

    t1 = time.time()
    model, rec = _fit_han(ctx, Xtr, ytr, Xva, gtva)
    ctx.timing["fit_s"] = ctx.timing.get("fit_s", 0.0) + (time.time() - t1)
    t2 = time.time()
    outputs = _predict(model, Xev, ctx.device)
    floor_out = floor_outputs(ytr, len(outputs))
    split_meta = {"readout": ctx.readout, "label_kind": ctx.label_kind, "num_classes": C, "n_train": int(len(Xtr)), "n_eval": int(len(outputs)),
                  "n_val": int(len(Xva)), "train_targets": ytr, "eval_rows": rows_ev, "eval_clip_ids": ids_ev, "chance": dict(ctx.chance),
                  "bootstrap": dict(ctx.bootstrap or {}), "seed": ctx.seed}
    blocks = evaluate_metrics(ctx.metric_specs, outputs=outputs, targets=gtev, protocol=ctx.protocol, split_meta=split_meta,
                              floor_outputs=floor_out, ci=ctx.ci, scorer=ctx.scorer(ev_splits, rows_ev, gtev))
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t2)
    return {**blocks, "probe": rec, "fit": {**rec, "device": ctx.device}, "eval_outputs": outputs, "eval_rows": rows_ev, "extra_arrays": {},
            "backend": None, "timing": {k: ctx.timing[k] for k in ("materialize_s", "fit_s", "score_s") if k in ctx.timing} | {"s_per_epoch": rec["s_per_epoch"]},
            "samples": ([{"role": "eval", "outputs": outputs, "targets": gtev, "rows": rows_ev, "clip_ids": ids_ev}] if ctx.log_samples else None)}


def _events_of(so: np.ndarray) -> list[list[int]]:
    """``[C, T]`` 0/1 → ``[[class, start_s, end_s], ...]`` runs of consecutive seconds, by class then time."""
    out = []
    for c in range(so.shape[0]):
        s = None
        for k in range(so.shape[1] + 1):
            on = k < so.shape[1] and so[c, k] > 0
            if on and s is None:
                s = k
            elif not on and s is not None:
                out.append([c, s, k])
                s = None
    return out


def sample_lines(part: Mapping[str, Any], *, seed: int, protocol: Mapping[str, Any]) -> list[dict]:
    """Samples-file lines: ``prediction`` the parsed audio and visual events ``[class, start_s, end_s]`` (thresholded
    and masked as scored) with the video's eight F-scores, ``target`` the annotated ones."""
    outs = np.asarray(part["outputs"], dtype=np.float32)
    gts = np.asarray(part["targets"], dtype=np.float32)
    per = video_scores(outs, gts)
    lines = []
    for i, (row, cid) in enumerate(zip(np.asarray(part["rows"]), part["clip_ids"], strict=True)):
        o = (outs[i, 2 * T] >= THRESHOLD).astype(np.float32)
        pa = ((outs[i, 0:T] >= THRESHOLD) * o).T
        pv = ((outs[i, T:2 * T] >= THRESHOLD) * o).T
        lines.append({"clip_id": str(cid), "row": int(row), "seed": int(seed), "role": str(part.get("role", "eval")),
                      "prediction": {"audio": _events_of(pa), "visual": _events_of(pv),
                                     "f": {k: round(float(v), 4) for k, v in zip(COLUMNS, per[i], strict=True)}},
                      "target": {"audio": _events_of(gts[i, 0]), "visual": _events_of(gts[i, 1])}})
    return lines


# ----------------------------------------------------------------------------- conformance population

#: Rows drawn per role for the conformance fixture.
POPULATION: dict[str, int] = {"train": 64, "eval": 64}


def conformance_population(ctx: Any, tag: str) -> dict[str, Any]:
    """The fixture population: the weak labels of drawn train-role rows (the floor), the ground truth of drawn eval
    rows, and raw outputs ``[n, OUT_ROWS, C]`` uniform in ``[0, 1)`` from the stable stream."""
    from yavih.conformance import select_rows, stream

    bench = ctx.bench

    def draw(role: str) -> list[tuple[str, np.ndarray]]:
        return [(s, select_rows(bench.manifest_hash, s, bench.n_rows(s), POPULATION[role], salt=role)) for s in ctx.roles[role]]

    tr, ev = draw("train"), draw("eval")
    ytr = np.concatenate([weak_targets(bench, s, r) for s, r in tr], axis=0)
    gt = np.concatenate([gt_targets(bench, s, r) for s, r in ev], axis=0)
    rows = np.concatenate([r for _, r in ev])
    n = len(rows)
    outputs = stream(f"{tag}:outputs", n * OUT_ROWS * C).reshape(n, OUT_ROWS, C).astype(np.float32)
    meta = {"readout": ctx.readout, "label_kind": ctx.label_kind, "num_classes": C, "n_train": int(len(ytr)), "n_eval": int(n), "n_val": 0,
            "train_targets": ytr, "eval_rows": rows, "eval_clip_ids": [bench.manifest(s).clip_id[i] for s, r in ev for i in _rows_index(bench.manifest(s), r)],
            "chance": dict(ctx.chance), "bootstrap": dict(ctx.bootstrap or {}), "seed": 0}
    return {"outputs": outputs, "targets": gt, "split_meta": meta, "floor_outputs": floor_outputs(ytr, n), "eval_splits": list(ctx.roles["eval"]),
            "eval_rows": rows}


# ----------------------------------------------------------------------------- the descriptor and the metrics


def _head(*, num_classes: int | None, T: int | None = None, grid: Any = None, target_size: int | None = None) -> dict:  # noqa: N803
    return {"family": HEAD_FAMILY, "upstream": "AVVP-ECCV20@fde5611:nets/net_audiovisual.py:MMIL_Net", "d_model": 512, "nhead": 1,
            "dim_feedforward": 512, "dropout": 0.1, "num_layers": 1, "input": "per-second token mean per stream; Linear(D, 512) each",
            "layout": f"time:{T if T is not None else '?'}", "out": int(num_classes) if num_classes is not None else None, "recipe": dict(RECIPE)}


def _geometry(cfg: Any) -> dict:
    out = cfg.head.get("out")
    kw: dict[str, Any] = {"num_classes": int(out) if isinstance(out, int) else None}
    layout = str(cfg.head.get("layout", ""))
    if layout.startswith("time:") and layout[5:].isdigit():
        kw["T"] = int(layout[5:])
    return kw


def _validate_spec(params: dict, requires: Any, protocol: Any) -> None:
    proto = dict(protocol or {})
    if proto.get("num_segments") != T or float(proto.get("segment_seconds", 0)) != 1.0:
        raise ValueError(f"protocol: the authors' evaluation is fixed to num_segments {T} of segment_seconds 1.0")
    if "t" not in tuple(requires.pos or ()):
        raise ValueError(".requires.pos: llp_avvp places tokens in seconds and needs [t]")


def make_scorer(name: str, *, module: str, version: int = 1) -> Any:
    """The folder's scorer: kind ``probe``, its own head (``han.py``), loss (:func:`han_loss`), driver (:func:`run`,
    its own training loop), floor, conformance population and samples lines."""
    from yavih.readouts.registry import Scorer

    return Scorer(
        name=name, version=int(version), module=module, kind="probe", label_kinds=frozenset({"multi"}),
        protocol_keys=frozenset({"num_segments", "segment_seconds"}), param_keys=frozenset(), requires_t_bins=True, requires_grid=False,
        validate_spec=_validate_spec, dense=True, train=TRAIN, head=_head, loss=lambda label_kind: LOSS,
        task_sizes=lambda bench: {"T": int(bench.spec.protocol.get("num_segments", T))}, geometry=_geometry,
        head_families=frozenset({HEAD_FAMILY}), losses=frozenset({LOSS}), build_head=_build_head, loss_fn=lambda _name: han_loss,
        floor_outputs=floor_outputs, score=run, conformance_population=conformance_population, sample_lines=sample_lines,
    )


def _per_video(outputs: Any, targets: Any, split_meta: Mapping[str, Any]) -> np.ndarray:
    cache = split_meta.get("_avvp_cache") if isinstance(split_meta, dict) else None
    if cache is None and isinstance(split_meta, dict):
        cache = split_meta["_avvp_cache"] = {}
    key = (id(outputs), id(targets))
    if cache is not None and key in cache and cache[key][0] is outputs and cache[key][1] is targets:
        return cache[key][2]
    per = video_scores(outputs, targets)
    if cache is not None:
        cache[key] = (outputs, targets, per)
    return per


def make_metrics(name: str) -> dict[str, Any]:
    """The folder's metrics under the names of :data:`METRIC_NAMES` (percent), a clip-bootstrap interval for each, and
    a floor ``<metric>_floor`` for each of :data:`FLOORS` (the metric on :func:`floor_outputs`)."""
    from yavih.metrics.registry import MetricSpec

    ro = frozenset({name})
    out: dict[str, Any] = {}

    def value(metric: str):
        return lambda o, t, p, sm: summary(_per_video(o, t, sm))[metric]

    def clips(metric: str):
        return lambda o, t, p, sm: per_clip(_per_video(o, t, sm), metric)

    def floor_value(metric: str):
        def fn(o, t, p, sm):
            if o is None:
                o = floor_outputs(sm["train_targets"], len(t))
            return summary(_per_video(o, t, sm))[metric]
        return fn

    for metric in METRIC_NAMES:
        out[metric] = MetricSpec(name=metric, fn=value(metric), readouts=ro, per_clip=clips(metric), ci=frozenset({"bootstrap_clip"}),
                                 description=DESCRIPTIONS[metric] + "; AVVP-ECCV20 utils/eval_metrics.py, per video, mean over videos")
    for metric in FLOORS:
        out[f"{metric}_floor"] = MetricSpec(name=f"{metric}_floor", fn=floor_value(metric), readouts=ro, kind="floor", floor_of=(metric,),
                                            description=f"{metric} of the label-only constant output (train weak-label prevalence)")
    return out
