#!/usr/bin/env python
"""Freeze the ``avsync`` manifest from the public VGGSound release.

Inputs (read only, nothing is downloaded):

- ``<vggsound-root>/metadata/vggsound.csv`` (the official listing, sha256 ``ed9fdeed…``, 199 176 rows): the clip
  universe and its train/test split;
- the VGGSound media ``<vggsound-root>/audio/<clip>.wav`` (16 kHz mono) and ``<vggsound-root>/video/<clip>.mp4``,
  read by the ``score`` step below.

The freeze has two steps.

**1. ``score``: model-free per-clip statistics** (hand-written signal statistics, never a learned model, so the
selection favours no encoder), written as JSON lines under a work directory (default
``<curation_root>/scores``; one file per shard, resumable):

- *audio* (every clip): duration; a Kaldi-compatible 128-bin log-mel filterbank (25 ms Hamming frames every 10 ms,
  pre-emphasis 0.97, 512-point FFT); its positive spectral flux ``o[t] = Σ_f max(0, S[f,t] − S[f,t−1])`` divided
  by its mean; ``flux_peakiness = p99(o) / median(o)``; ``onset_rate`` = local maxima above ``median + 3·MAD``
  per second; ``env_stationarity`` = std / mean of the RMS in 0.5 s bins; and ``o`` averaged in 0.25 s bins
  (``flux_bins``, the audio side of ``av_corr``).
- *video* (the audio-selected candidates only): the clip's frames as a full packet scan counts them
  (``video_frames``) and ``video_fps`` = frames / stream span; ``grid_ok`` = for every offset δ, the 16 frame
  indices ``round((1 + δ/2 + (k + 0.5)·0.5)·fps)`` lie inside the clip; 16 frames of the δ = 0 window (each the
  first frame at or after its index's pts, decoded from the preceding key frame), resized (short side 224,
  bilinear, antialiased), centre-cropped to 224, scaled to [−1, 1] and averaged over RGB; ``motion`` = mean
  absolute difference of consecutive frames; ``av_corr`` = Pearson r between those 15 frame-pair differences and
  the flux averaged over the same 0.5 s intervals.

**2. the freeze** (this script without a sub-command) applies the population gates of ``protocol.curation`` to
the scores: audio ≥ ``required_seconds`` (10.0 s: the δ = ±2 windows need it); the *transient* pool = onset rate
≥ 0.5/s and peakiness in the top 40 % of the long-enough clips (cut 3.564906); the *stationary* pool = the 6 000
VGGSound-test clips at the bottom of the peakiness ranking outside the transient pool; video scored and
``grid_ok``; on the transient pool a motion gate at its 0.25 quantile (0.065111). Then eval = the 3 000
VGGSound-test transient clips with the highest ``av_corr``, train = the 12 000 VGGSound-train transient clips
with the highest ``av_corr``, floor = the first 6 000 of the stationary pool ranked by ``av_corr`` ascending
(minus eval clips; 5 997 survive the video gate); ties break on the clip id. Every clip gets offset bins from
:func:`assign_offset_bins` (one per eval / floor clip, two distinct ones per train clip, seed 0).

Manifest: columns ``clip_id,label,split,rel_video,rel_audio,offset_bin,offset_seconds,arm,rel_media``,
``clip_id = <clip>#<bin>``, rows sorted by (arm, VGGSound split, clip_id); the stationary arm has split
``stationary`` (role ``floor``). ``rel_media`` = :func:`materialized_relpath`, the row's materialized mp4 under
``derived_root`` (``prepare/03_materialize.py``): a hash, so the path carries neither the offset bin (the label)
nor the split. ``classes.csv`` = ``index,name`` with ``name`` = the offset in seconds.

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (gzip mtime 0, no file name), ``classes.csv``, and a printed
JSON report with every value ``benchmark.yaml`` carries. ``--check`` compares instead of writing (exit 1 on drift),
including ``benchmark.yaml``'s hashes and curation block. Without a complete score directory ``--check`` re-derives
the rows from the committed manifest's clip selection (offset draw, columns, split audit) and says so.

Usage (``FOLDER=$(yavih bench path avsync)``)::

    python "$FOLDER/make_manifest.py" score --workers 16                    # audio, then video (≈ 2 core-days)
    python "$FOLDER/make_manifest.py" score --stage audio --shard 3/64     # or as an array: audio shards first,
    python "$FOLDER/make_manifest.py" score --stage video --shard 3/64     # then video shards
    python "$FOLDER/make_manifest.py" --check                              # re-derive and compare
    python "$FOLDER/make_manifest.py"                                       # write manifest.csv.gz + classes.csv

Scoring needs torch, PyAV, soundfile and numpy. Re-scoring other media bytes (a fresh download) changes the scores
and therefore the selection: that is a new ``protocol_version``, never a silent re-freeze.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent
NAME = "avsync"

VGGSOUND_CSV = "metadata/vggsound.csv"
VGGSOUND_CSV_SHA = "ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8"
TOTAL_UPSTREAM_CLIPS = 199176  # rows (= distinct clips) of metadata/vggsound.csv

# The offset grid, the window pivot and the manifest columns.
OFFSETS_SECONDS: tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
NUM_CLASSES = len(OFFSETS_SECONDS)
NOMINAL_START_SECONDS = 1.0
ROW_COLUMNS = ["clip_id", "label", "split", "rel_video", "rel_audio", "offset_bin", "offset_seconds", "arm"]
MANIFEST_COLUMNS = [*ROW_COLUMNS, "rel_media"]
MATERIALIZED_SALT = "avsync/v2:"

# protocol.curation (benchmark.yaml): the gates; the two cuts are re-derived from the scores and asserted.
PROTOCOL_CURATION = {
    "required_seconds": 10.0,
    "min_onset_rate": 0.5,
    "top_frac_peakiness": 0.4,
    "peakiness_cut": 3.564906,
    "motion_quantile": 0.25,
    "motion_cut": 0.065111,
    "stationary_pool_size": 6000,
    "offset_seed": 0,
}
EVAL_CLIPS, TRAIN_CLIPS, TRAIN_DRAWS = 3000, 12000, 2
# (arm, VGGSound split) -> manifest split value. Roles: train / eval / floor (benchmark.yaml splits).
SPLIT_MAP = {("transient", "train"): "train", ("transient", "test"): "test", ("stationary", "test"): "stationary"}
EXPECTED_ROWS = {"train": 24000, "test": 3000, "stationary": 5997}
EXPECTED_CLIPS = {"train": 12000, "test": 3000, "stationary": 5997}

# Scoring recipe (fixed: changing a number changes the selection).
SAMPLE_RATE = 16000
N_MELS = 128
WIN_LENGTH = 400  # samples (25 ms)
HOP_LENGTH = 160  # samples (10 ms)
ENV_BIN_SECONDS = 0.5
FLUX_BIN_SECONDS = 0.25
NUM_BUCKETS = 16  # video frames over the 8 s window: 2 fps, the geometry of the grid_ok gate
BUCKET_SECONDS = 0.5
CROP = 224
SCORE_FIELDS_AUDIO = ("duration_seconds", "flux_peakiness", "onset_rate", "env_stationarity", "flux_bins")
SCORE_FIELDS_VIDEO = ("video_frames", "video_fps", "grid_ok", "motion", "av_corr")

_VGG_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}_\d{6}$")
_SHARD_RE = re.compile(r"^(\d+)/(\d+)$")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- grid, windows, draw


def materialized_relpath(clip_id: str) -> str:
    """``<h[:2]>/<h>.mp4``, ``h = sha256("avsync/v2:" + clip_id)[:24]``: the materialized mp4 of a row, relative to
    ``derived_root``. Carries neither the offset bin nor the split (both are visible in ``clip_id``)."""
    h = hashlib.sha256((MATERIALIZED_SALT + clip_id).encode("utf-8")).hexdigest()[:24]
    return f"{h[:2]}/{h}.mp4"


def window_starts(offset_seconds: float, nominal_start: float = NOMINAL_START_SECONDS) -> tuple[float, float]:
    """``(audio_start, video_start) = nominal ∓ δ/2``: the shift is split between the two streams."""
    delta = float(offset_seconds)
    return (float(nominal_start) - delta / 2.0, float(nominal_start) + delta / 2.0)


def required_seconds(num_buckets: int = NUM_BUCKETS, bucket_seconds: float = BUCKET_SECONDS) -> float:
    """Shortest clip whose audio window fits at every offset (audio binds): 10.0 s for 8 s at pivot 1.0."""
    return max(window_starts(d)[0] for d in OFFSETS_SECONDS) + num_buckets * float(bucket_seconds)


def bucket_indices(fps: float, start: float) -> list[int]:
    """``round((start + (k + 0.5) · 0.5) · fps)`` for the 16 half-second buckets (no clamping)."""
    return [int(round((start + (b + 0.5) * BUCKET_SECONDS) * fps)) for b in range(NUM_BUCKETS)]


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an unsorted list."""
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def assign_offset_bins(clips: list[str], draws: int, seed: int, num_bins: int = NUM_CLASSES) -> list[tuple[str, int, int]]:
    """``[(clip_id, draw_index, offset_bin), ...]`` in sorted clip order.

    Exactly ``floor(N / num_bins)`` rows per bin from a ``numpy.random.default_rng(seed)`` shuffle; the
    ``N mod num_bins`` surplus rows take bins ``0 .. surplus-1`` in clip-id order (never the RNG); with ``draws > 1``
    the bins of one clip are forced distinct by multiset-preserving swaps.
    """
    import numpy as np

    if draws > num_bins:
        raise ValueError(f"cannot draw {draws} DISTINCT bins from {num_bins}")
    clips = sorted(clips)
    n_rows = len(clips) * draws
    base = n_rows // num_bins
    surplus = n_rows - base * num_bins

    rng = np.random.default_rng(seed)
    balanced = np.array([b for b in range(num_bins) for _ in range(base)], dtype=np.int64)
    rng.shuffle(balanced)
    pool = [int(b) for b in balanced] + list(range(surplus))

    if draws > 1:
        for i in range(0, n_rows, draws):
            for j in range(1, draws):
                if pool[i + j] not in pool[i : i + j]:
                    continue
                for k in [*range(i + draws, n_rows), *range(0, i)]:
                    other = k - (k % draws)
                    partner_group = [pool[m] for m in range(other, min(other + draws, n_rows)) if m != k]
                    if pool[k] in pool[i : i + draws]:
                        continue
                    if pool[i + j] in partner_group:
                        continue
                    pool[i + j], pool[k] = pool[k], pool[i + j]
                    break
                else:
                    raise ValueError(f"no swap partner for clip {clips[i // draws]!r}: the pool is too small for distinct draws")
    return [(clips[i], d, pool[i * draws + d]) for i in range(len(clips)) for d in range(draws)]


def rows_for(eval_clips: list[str], train_clips: list[str], stationary_clips: list[str], seed: int) -> list[dict[str, str]]:
    """The manifest rows (``ROW_COLUMNS`` with VGGSound split values) of a clip selection, sorted (arm, split, clip_id)."""
    rows: list[dict[str, str]] = []
    for group, split, arm, draws in ((eval_clips, "test", "transient", 1), (train_clips, "train", "transient", TRAIN_DRAWS), (stationary_clips, "test", "stationary", 1)):
        for clip, _d, b in assign_offset_bins(group, draws, seed):
            rows.append({"clip_id": f"{clip}#{b}", "label": str(b), "split": split, "rel_video": f"video/{clip}.mp4",
                         "rel_audio": f"audio/{clip}.wav", "offset_bin": str(b), "offset_seconds": repr(OFFSETS_SECONDS[b]), "arm": arm})
    rows.sort(key=lambda r: (r["arm"], r["split"], r["clip_id"]))
    return rows


# ----------------------------------------------------------------------------- audio scores


def _mel_banks(num_bins: int, padded: int, sample_rate: float, low: float = 20.0):
    """Triangular mel filters ``[num_bins, padded // 2]`` over ``[low, Nyquist]`` (Kaldi's mel scale)."""
    import torch

    width = sample_rate / padded
    mel = lambda f: 1127.0 * math.log(1.0 + f / 700.0)  # noqa: E731
    lo, hi = mel(low), mel(0.5 * sample_rate)
    delta = (hi - lo) / (num_bins + 1)
    b = torch.arange(num_bins).unsqueeze(1)
    left, center, right = lo + b * delta, lo + (b + 1.0) * delta, lo + (b + 2.0) * delta
    m = (1127.0 * (1.0 + (width * torch.arange(padded / 2)) / 700.0).log()).unsqueeze(0)
    return torch.max(torch.zeros(1), torch.min((m - left) / (center - left), (right - m) / (right - center)))


def log_mel(wav):
    """Kaldi-compatible log-mel filterbank ``[T, 128]`` of a mono float32 waveform ``[n]``: 25 ms frames every 10 ms
    (only whole frames), per-frame DC removal, pre-emphasis 0.97, Hamming window, zero-padded 512-point power
    spectrum, 128 mel bins, natural log floored at float32 eps."""
    import torch

    eps = torch.tensor(torch.finfo(torch.float).eps)
    padded = 1 << (WIN_LENGTH - 1).bit_length()
    if wav.size(0) < WIN_LENGTH:
        return torch.empty((0, N_MELS))
    m = 1 + (wav.size(0) - WIN_LENGTH) // HOP_LENGTH
    x = wav.as_strided((m, WIN_LENGTH), (HOP_LENGTH * wav.stride(0), wav.stride(0)))
    x = x - torch.mean(x, dim=1).unsqueeze(1)
    prev = torch.nn.functional.pad(x.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
    x = x - 0.97 * prev[:, :-1]
    x = x * torch.hamming_window(WIN_LENGTH, periodic=False, alpha=0.54, beta=0.46, dtype=x.dtype).unsqueeze(0)
    x = torch.nn.functional.pad(x.unsqueeze(0), (0, padded - WIN_LENGTH), mode="constant", value=0).squeeze(0)
    spec = torch.fft.rfft(x).abs().pow(2.0)
    banks = torch.nn.functional.pad(_mel_banks(N_MELS, padded, float(SAMPLE_RATE)).to(dtype=x.dtype), (0, 1), mode="constant", value=0)
    return torch.max(torch.mm(spec, banks.T), eps).log()


def _median(x) -> float:
    return float(x.median()) if x.numel() else 0.0


def audio_scores(path: str) -> dict:
    """The audio statistics of one clip (module docstring, step 1), rounded as recorded."""
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{sr} Hz, expected {SAMPLE_RATE}")
    wav = torch.from_numpy(np.ascontiguousarray(data.T)).mean(dim=0, keepdim=True)
    length = wav.shape[1]
    duration = length / float(SAMPLE_RATE)
    mel = log_mel(wav[0])
    if mel.shape[0] < 2:
        return {"duration_seconds": duration, "flux_peakiness": 0.0, "onset_rate": 0.0, "env_stationarity": 0.0, "flux_bins": []}
    flux = (mel[1:] - mel[:-1]).clamp(min=0.0).sum(dim=1)
    mean = float(flux.mean())
    if mean > 1e-12:
        flux = flux / mean
    med = _median(flux)
    mad = _median((flux - med).abs())
    peakiness = float(torch.quantile(flux, 0.99)) / max(med, 1e-6)
    if flux.numel() >= 3:
        inner = flux[1:-1]
        onsets = int(((inner > flux[:-2]) & (inner >= flux[2:]) & (inner > med + 3.0 * mad)).sum())
    else:
        onsets = 0
    bin_len = int(ENV_BIN_SECONDS * SAMPLE_RATE)
    nbins = length // bin_len
    if nbins >= 2:
        rms = wav[0, : nbins * bin_len].reshape(nbins, bin_len).pow(2).mean(dim=1).sqrt()
        env = float(rms.std() / rms.mean()) if float(rms.mean()) > 1e-12 else 0.0
    else:
        env = 0.0
    # flux binned at 0.25 s by each flux sample's frame-centre time
    times = ((torch.arange(flux.numel(), dtype=torch.float64) + 1) * HOP_LENGTH + WIN_LENGTH / 2.0) / SAMPLE_RATE
    n_bins = int(duration / FLUX_BIN_SECONDS)
    idx = (times / FLUX_BIN_SECONDS).floor().long()
    keep = idx < n_bins
    sums = torch.zeros(n_bins, dtype=torch.float64)
    counts = torch.zeros(n_bins, dtype=torch.float64)
    sums.index_add_(0, idx[keep], flux[keep].double())
    counts.index_add_(0, idx[keep], torch.ones(int(keep.sum()), dtype=torch.float64))
    return {
        "duration_seconds": round(duration, 6),
        "flux_peakiness": round(peakiness, 6),
        "onset_rate": round(onsets / max(duration, 1e-6), 6),
        "env_stationarity": round(env, 6),
        "flux_bins": [round(v, 5) for v in (sums / counts.clamp(min=1.0)).tolist()],
    }


# ----------------------------------------------------------------------------- video scores


def _scan(container, stream) -> tuple[list[int], list[int], float]:
    """Every non-discarded packet of ``stream``: ``(frame pts sorted, key-frame pts sorted, fps)`` with ``fps`` =
    frames / (max(pts + duration) − min(pts)) in seconds (the header's average rate for an empty span)."""
    pts: list[int] = []
    keys: list[int] = []
    begin = end = None
    for pkt in container.demux(stream):
        if (pkt.pts is None and pkt.dts is None) or pkt.is_discard:
            continue
        p = pkt.pts if pkt.pts is not None else pkt.dts
        pts.append(p)
        if pkt.is_keyframe:
            keys.append(p)
        begin = p if begin is None else min(begin, p)
        end = p + pkt.duration if end is None else max(end, p + pkt.duration)
    pts.sort()
    keys.sort()
    if not pts:
        begin = end = 0
    num, den = stream.time_base.numerator, stream.time_base.denominator
    span = float(end) * num / den - float(begin) * num / den  # seconds as pts * num / den (multiply, then divide)
    return pts, keys, (len(pts) / span if span != 0 else (float(stream.average_rate) if stream.average_rate else 0.0))


_SWS_TABLES = {1, 4, 5, 6, 7, 9, 10}  # colorspaces with their own YUV→RGB coefficients in swscale; others: BT.601


def _rgb(frame):
    """``uint8 [H, W, 3]`` with the YUV→RGB coefficients of the frame's colorspace (BT.601 when it names none) and
    the range of its pixel format (full for ``yuvj*``); no scaling, so no chroma interpolation choice applies."""
    cs = frame.colorspace if frame.colorspace in _SWS_TABLES else 5
    rng = "JPEG" if frame.format.name.startswith("yuvj") else "MPEG"
    # a target colorspace different from the source makes swscale load the source coefficients explicitly
    return frame.reformat(format="rgb24", src_colorspace=cs, dst_colorspace=1 if cs != 1 else 5, interpolation="POINT",
                          src_color_range=rng, dst_color_range="JPEG").to_ndarray()


def grab_frames(container, stream, pts: list[int], keys: list[int], idx: list[int]) -> dict:
    """``{index: rgb}`` for the frame indices ``idx``: per target pts (ascending), keep decoding when the target lies
    after the last returned frame in the same key-frame interval, else seek to the key frame at or before it (the
    first key frame when none precedes it); return the first decoded frame whose pts is ≥ the target. A run of
    consecutive indices returns consecutive decoded frames."""

    def decoded():
        return (fr for pkt in container.demux(stream) for fr in pkt.decode())

    container.seek(0, backward=True, any_frame=False)
    it = decoded()
    out: dict = {}
    last_pts, last_idx, cursor = 0, -2, 0
    for i in sorted(set(idx)):
        if i != last_idx + 1:
            target = pts[i]
            ka, kb = bisect.bisect_right(keys, last_pts) - 1, bisect.bisect_right(keys, target) - 1
            if not (target > last_pts and ka >= 0 and kb >= 0 and ka == kb):
                container.seek(keys[max(kb, 0)] if keys else target, stream=stream, backward=True, any_frame=False)
                it = decoded()
            cursor = target
        for fr in it:
            fp = fr.pts if fr.pts is not None else fr.dts
            if fp >= cursor:
                out[i] = _rgb(fr)
                last_pts = fp
                break
        else:
            raise ValueError(f"frame index {i} not decodable")
        last_idx = i
    return out


def _pearson(a, b) -> float:
    """Pearson r, 0.0 when either series is constant."""
    if a.numel() < 2 or b.numel() != a.numel():
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(a.norm() * b.norm())
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


def video_scores(path: str, flux_bins: list[float]) -> dict:
    """The video statistics of one candidate clip (module docstring, step 1), rounded as recorded."""
    import av
    import numpy as np
    import torch

    with av.open(path) as f:
        st = f.streams.video[0]
        st.codec_context.thread_count = 1
        pts, keys, fps = _scan(f, st)
        total = len(pts)
        fps = fps or 25.0
        grid_ok = all(w[0] >= 0 and w[-1] <= total - 1 for w in (bucket_indices(fps, window_starts(d)[1]) for d in OFFSETS_SECONDS))
        idx = [min(total - 1, max(0, i)) for i in bucket_indices(fps, NOMINAL_START_SECONDS)]
        got = grab_frames(f, st, pts, keys, idx)
    x = torch.from_numpy(np.stack([got[i] for i in idx])).permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]
    h, w = x.shape[-2:]
    size = (CROP, int(CROP * w / h)) if h <= w else (int(CROP * h / w), CROP)
    x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    hh, ww = x.shape[-2:]
    top, left = int(round((hh - CROP) / 2.0)), int(round((ww - CROP) / 2.0))
    gray = (x[..., top : top + CROP, left : left + CROP] * 2.0 - 1.0).permute(1, 0, 2, 3).mean(dim=0)  # [T, 224, 224]
    motion = (gray[1:] - gray[:-1]).abs().mean(dim=(1, 2))
    per = int(round(BUCKET_SECONDS / FLUX_BIN_SECONDS))
    first = int(round((NOMINAL_START_SECONDS + 0.5 * BUCKET_SECONDS) / FLUX_BIN_SECONDS))
    pair_flux = []
    for k in range(motion.numel()):
        win = flux_bins[first + k * per : first + (k + 1) * per]
        pair_flux.append(sum(win) / len(win) if win else 0.0)
    return {
        "video_frames": int(total),
        "video_fps": round(float(fps), 4),
        "grid_ok": bool(grid_ok),
        "motion": round(float(motion.mean()), 6),
        "av_corr": round(_pearson(torch.tensor(pair_flux, dtype=torch.float32), motion), 6),
    }


# ----------------------------------------------------------------------------- scores on disk


def read_vggsound(path: Path) -> dict[str, str]:
    """``{clip: split}`` from the official listing (``ytid,start,label,split``, no header)."""
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 4:
                out[f"{row[0].strip()}_{int(row[1]):06d}"] = row[3].strip()
    return out


def parse_shard(shard: str | None) -> tuple[int, int]:
    if shard is None:
        return 0, 1
    m = _SHARD_RE.match(shard)
    if not m or not 0 <= int(m.group(1)) < int(m.group(2)):
        raise SystemExit(f"--shard {shard!r}: expected i/n with 0 <= i < n")
    return int(m.group(1)), int(m.group(2))


def read_scores(work: Path, stage: str) -> dict[str, dict]:
    """``{clip: record}`` merged over ``<work>/<stage>/*.jsonl`` (later lines win; a torn line is skipped)."""
    out: dict[str, dict] = {}
    d = work / stage
    for p in sorted(d.glob("*.jsonl")) if d.is_dir() else []:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("clip"), str):
                    out[rec["clip"]] = rec
    return out


def select_candidates(audio: dict[str, dict], splits: dict[str, str], curation: dict = PROTOCOL_CURATION) -> dict:
    """The audio gates: ``{transient, stationary_pool, peakiness_cut, too_short, failed}`` (pools sorted)."""
    req = curation["required_seconds"]
    usable = {c: r for c, r in audio.items() if "error" not in r and r["duration_seconds"] >= req}
    cut = _quantile([r["flux_peakiness"] for r in usable.values()], 1.0 - curation["top_frac_peakiness"])
    transient = sorted(c for c, r in usable.items() if r["onset_rate"] >= curation["min_onset_rate"] and r["flux_peakiness"] >= cut)
    tset = set(transient)
    floor_ranked = sorted((c for c in usable if splits.get(c) == "test" and c not in tset), key=lambda c: (usable[c]["flux_peakiness"], c))
    return {
        "transient": transient,
        "stationary_pool": sorted(floor_ranked[: curation["stationary_pool_size"]]),
        "peakiness_cut": cut,
        "too_short": sum(1 for r in audio.values() if "error" not in r and r["duration_seconds"] < req),
        "failed": sum(1 for r in audio.values() if "error" in r),
    }


def _score_audio_task(task: tuple[str, str]) -> dict:
    clip, path = task
    try:
        return {"clip": clip, **audio_scores(path)}
    except Exception as exc:  # noqa: BLE001 — a failing clip is recorded, never fatal
        return {"clip": clip, "error": f"{type(exc).__name__}: {exc}"}


def _score_video_task(task: tuple[str, str, list[float]]) -> dict:
    clip, path, flux_bins = task
    try:
        return {"clip": clip, **video_scores(path, flux_bins)}
    except Exception as exc:  # noqa: BLE001
        return {"clip": clip, "error": f"{type(exc).__name__}: {exc}"}


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def run_scores(work: Path, stage: str, tasks: list[tuple], shard: tuple[int, int], workers: int) -> dict:
    """Score ``tasks[i::n]`` not yet recorded into ``<work>/<stage>/shard-i-of-n.jsonl`` (append, flushed per line)."""
    i, n = shard
    out = work / stage / f"shard-{i:05d}-of-{n:05d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(read_scores(work, stage))
    todo = [t for t in tasks[i::n] if t[0] not in done]
    fn = _score_audio_task if stage == "audio" else _score_video_task
    failed = 0
    with open(out, "a", encoding="utf-8") as fh:
        if workers <= 1:
            _init_worker()
            results = map(fn, todo)
            for rec in results:
                failed += "error" in rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        else:
            with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
                for rec in ex.map(fn, todo, chunksize=16):
                    failed += "error" in rec
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
    return {"stage": stage, "shard": f"{i}/{n}", "scored": len(todo), "failed": failed, "file": str(out)}


def cmd_score(ns: argparse.Namespace) -> int:
    vroot = ns.vggsound_root or default_vggsound_root()
    work = ns.work or default_work_dir()
    splits = read_vggsound(vroot / VGGSOUND_CSV)
    clips = sorted(splits)
    if ns.limit:
        clips = clips[: ns.limit]
    shard = parse_shard(ns.shard)
    report = []
    if ns.stage in ("audio", "all"):
        report.append(run_scores(work, "audio", [(c, str(vroot / "audio" / f"{c}.wav")) for c in clips], shard, ns.workers))
    if ns.stage in ("video", "all"):
        audio = read_scores(work, "audio")
        missing = [c for c in clips if c not in audio]
        if missing:
            raise SystemExit(f"{len(missing)} clips have no audio score yet (first {missing[:3]}): finish every audio shard first")
        cand = select_candidates({c: audio[c] for c in clips}, splits)
        todo = sorted(set(cand["transient"]) | set(cand["stationary_pool"]))
        report.append(run_scores(work, "video", [(c, str(vroot / "video" / f"{c}.mp4"), audio[c]["flux_bins"]) for c in todo], shard, ns.workers))
    print(json.dumps(report, indent=1))
    return 0


# ----------------------------------------------------------------------------- selection


def select(audio: dict[str, dict], video: dict[str, dict], splits: dict[str, str], *, curation: dict = PROTOCOL_CURATION,
           sizes: tuple[int, int] = (EVAL_CLIPS, TRAIN_CLIPS), pinned: bool = True) -> tuple[list[str], list[str], list[str], dict]:
    """The freeze from complete scores: ``(eval, train, stationary clips, accounting)``. ``pinned``: the derived cuts
    must equal ``curation``'s recorded ``peakiness_cut`` / ``motion_cut`` (6 decimals)."""
    cand = select_candidates(audio, splits, curation)
    if pinned and round(cand["peakiness_cut"], 6) != curation["peakiness_cut"]:
        raise SystemExit(f"derived peakiness_cut {cand['peakiness_cut']!r} != protocol.curation {curation['peakiness_cut']}")
    transient, floor_pool = set(cand["transient"]), set(cand["stationary_pool"])
    unscored = sorted((transient | floor_pool) - set(video))
    if unscored:
        raise SystemExit(f"{len(unscored)} candidate clips have no video score (first {unscored[:3]}): run `score --stage video`")
    req = curation["required_seconds"]
    scored = {c: {**audio[c], **v} for c, v in video.items() if c in audio and "error" not in v and v.get("grid_ok")
              and "error" not in audio[c] and audio[c]["duration_seconds"] >= req}
    motion_cut = _quantile([r["motion"] for c, r in scored.items() if c in transient], curation["motion_quantile"])
    if pinned and round(motion_cut, 6) != curation["motion_cut"]:
        raise SystemExit(f"derived motion_cut {motion_cut!r} != protocol.curation {curation['motion_cut']}")

    def ranked(pool: set[str], split: str, descending: bool, gate: bool) -> list[str]:
        cs = [c for c, r in scored.items() if c in pool and splits.get(c) == split and (not gate or r["motion"] >= motion_cut)]
        return sorted(cs, key=lambda c: (-scored[c]["av_corr"] if descending else scored[c]["av_corr"], c))

    eval_clips = ranked(transient, "test", True, True)[: sizes[0]]
    train_clips = ranked(transient, "train", True, True)[: sizes[1]]
    taken = set(eval_clips)
    stationary = [c for c in ranked(floor_pool, "test", False, False) if c not in taken][: curation["stationary_pool_size"]]

    # one exclusion reason per clip, in funnel order
    selected = taken | set(train_clips) | set(stationary)
    reasons: Counter = Counter()
    for c, r in audio.items():
        if c in selected:
            continue
        v = video.get(c, {})
        if "error" in r:
            reasons["audio_failed"] += 1
        elif r["duration_seconds"] < req:
            reasons["short_for_window"] += 1
        elif c not in transient and c not in floor_pool:
            reasons["onset_peakiness_gate"] += 1
        elif "error" in v:
            reasons["video_failed"] += 1
        elif not v.get("grid_ok"):
            reasons["not_grid_ok"] += 1
        elif c in transient and v["motion"] < motion_cut:
            reasons["motion_gate"] += 1
        elif c in transient:
            reasons["rank_remainder_transient"] += 1
        else:
            reasons["rank_remainder_stationary"] += 1
    acc = {"excluded": dict(sorted(reasons.items())), "peakiness_cut": round(cand["peakiness_cut"], 6), "motion_cut": round(motion_cut, 6),
           "transient_pool": len(transient), "stationary_pool": len(floor_pool), "video_scored": len(video)}
    return eval_clips, train_clips, stationary, acc


def frame_grid_audit(video: dict[str, dict], rows: list[dict[str, str]]) -> dict:
    """Rows where a ``uniform`` rule with ``n`` frames over ``[0, 8]`` asks for a time past the clip's last frame
    (the reader would serve a duplicated last frame), per ``n``; the curation's ``grid_ok`` covers ``n = 16``."""
    out: dict[str, dict] = {}
    for n in (8, 16, 24, 32, 48, 64):
        bad: Counter = Counter()
        for r in rows:
            rec = video[r["clip_id"].split("#", 1)[0]]
            t_last = window_starts(float(r["offset_seconds"]))[1] + (n - 0.5) * 8.0 / n
            if t_last * float(rec["video_fps"]) > int(rec["video_frames"]) - 0.5 + 1e-9:
                bad[r["split"]] += 1
        out[str(n)] = {"rows_past_last_frame": sum(bad.values()), "by_split": dict(sorted(bad.items()))}
    fps = [float(video[r["clip_id"].split("#", 1)[0]]["video_fps"]) for r in rows]
    return {"uniform_n": out, "fps_min": min(fps), "fps_max": max(fps), "rows_fps_30": sum(1 for v in fps if abs(v - 30.0) < 1e-6)}


# ----------------------------------------------------------------------------- freeze


def default_vggsound_root() -> Path:
    """The folder's ``media_root`` (``--data``, ``YAVIH_DATA__AVSYNC__MEDIA_ROOT``, config, else ``${datasets.vggsound}``)."""
    try:
        return config.folder_root(HERE, "media_root")
    except config.ConfigError as exc:
        raise SystemExit(f"{exc} (or pass --vggsound-root)") from None


def default_work_dir() -> Path:
    """``<curation_root>/scores`` (``YAVIH_DATA__AVSYNC__CURATION_ROOT``, config, else ``${data.root}/avsync``)."""
    try:
        return config.folder_root(HERE, "curation_root") / "scores"
    except config.ConfigError as exc:
        raise SystemExit(f"{exc} (or pass --work)") from None


def classes_csv_bytes() -> bytes:
    return csv_bytes(["index", "name"], [[i, repr(o)] for i, o in enumerate(OFFSETS_SECONDS)])


def read_committed_rows(folder: Path) -> list[dict[str, str]]:
    import gzip

    with gzip.open(folder / "manifest.csv.gz", "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def selection_of(rows: list[dict[str, str]]) -> tuple[list[str], list[str], list[str]]:
    """The (eval, train, stationary) clip sets of manifest rows (manifest split values)."""
    by = {s: sorted({r["clip_id"].split("#", 1)[0] for r in rows if r["split"] == s}) for s in EXPECTED_ROWS}
    return by["test"], by["train"], by["stationary"]


def freeze(vggsound_root: Path | None, work: Path | None, seed: int, committed: Path = HERE) -> dict:
    """The manifest from complete scores under ``work`` (``source: scores``), else from the clip selection of the
    committed manifest (``source: committed selection``: re-checks the draw, the columns and the split audit)."""
    upstream_files: dict[str, str] = {}
    splits: dict[str, str] | None = None
    csv_path = vggsound_root / VGGSOUND_CSV if vggsound_root is not None else None
    if csv_path is not None and csv_path.is_file():
        upstream_files["vggsound/" + VGGSOUND_CSV] = sha256_file(csv_path)
        if upstream_files["vggsound/" + VGGSOUND_CSV] != VGGSOUND_CSV_SHA:
            raise SystemExit(f"{csv_path}: sha256 differs from the pinned {VGGSOUND_CSV_SHA}")
        splits = read_vggsound(csv_path)
        if len(splits) != TOTAL_UPSTREAM_CLIPS:
            raise SystemExit(f"{csv_path}: {len(splits)} clips, expected {TOTAL_UPSTREAM_CLIPS}")

    audio = read_scores(work, "audio") if work is not None else {}
    video = read_scores(work, "video") if work is not None else {}
    complete = splits is not None and bool(audio) and set(splits) <= set(audio)
    accounting = None
    if complete:
        audio = {c: audio[c] for c in splits}
        eval_clips, train_clips, stationary, accounting = select(audio, video, splits)
        source = "scores"
    else:
        eval_clips, train_clips, stationary = selection_of(read_committed_rows(committed))
        source = "committed selection"
    rows = [{**r, "split": SPLIT_MAP[(r["arm"], r["split"])], "rel_media": materialized_relpath(r["clip_id"])}
            for r in rows_for(eval_clips, train_clips, stationary, seed)]

    # invariants
    if len({r["rel_media"] for r in rows}) != len(rows):
        raise SystemExit("materialized_relpath collision")
    counts = Counter(r["split"] for r in rows)
    if dict(counts) != EXPECTED_ROWS:
        raise SystemExit(f"rows per split {dict(counts)} != {EXPECTED_ROWS}")
    clips_by_split = {"test": set(eval_clips), "train": set(train_clips), "stationary": set(stationary)}
    if {s: len(v) for s, v in clips_by_split.items()} != EXPECTED_CLIPS:
        raise SystemExit(f"clips per split {({s: len(v) for s, v in clips_by_split.items()})} != {EXPECTED_CLIPS}")
    if clips_by_split["test"] & clips_by_split["stationary"] or clips_by_split["train"] & (clips_by_split["test"] | clips_by_split["stationary"]):
        raise SystemExit("arms overlap at clip level")
    for r in rows:
        if not _VGG_ID_RE.match(r["clip_id"].split("#", 1)[0]):
            raise SystemExit(f"bad clip id {r['clip_id']!r}")

    # split audit against the official listing, presence of the source media (both optional)
    leakage_audit = None
    missing = dict.fromkeys(EXPECTED_ROWS, 0)
    if splits is not None:
        want = {"train": "train", "test": "test", "stationary": "test"}
        wrong = [c for s, cs in clips_by_split.items() for c in cs if splits.get(c) != want[s]]
        if wrong:
            raise SystemExit(f"split audit failed: {len(wrong)} clips are not in the VGGSound split their arm claims: {wrong[:5]}")
        leakage_audit = {"against": "VGGSound metadata/vggsound.csv split column (train rows ⊂ VGGSound-train; test and stationary rows ⊂ VGGSound-test)", "excluded": 0}
    if vggsound_root is not None and (vggsound_root / "video").is_dir() and (vggsound_root / "audio").is_dir():
        videos = {e.name for e in os.scandir(vggsound_root / "video")}  # one scandir per directory, never a stat per row
        audios = {e.name for e in os.scandir(vggsound_root / "audio")}
        for r in rows:
            if r["rel_video"][6:] not in videos or r["rel_audio"][6:] not in audios:
                missing[r["split"]] += 1

    manifest_bytes = csv_bytes(MANIFEST_COLUMNS, [[r[c] for c in MANIFEST_COLUMNS] for r in rows])
    classes_bytes = classes_csv_bytes()
    manifest_sha, classes_sha = sha256_bytes(manifest_bytes), sha256_bytes(classes_bytes)
    # rows are (clip, draw) pairs: every VGGSound clip is one candidate, each train clip adds a second draw
    curation_block = {
        "total_upstream": TOTAL_UPSTREAM_CLIPS + EXPECTED_CLIPS["train"] * (TRAIN_DRAWS - 1),
        "per_split": {s: {"listed": EXPECTED_ROWS[s], "staged": EXPECTED_ROWS[s] - missing[s], "missing": missing[s]} for s in ("train", "test", "stationary")},
        "excluded": accounting["excluded"] if accounting else None,
        "short_clips": 0,
        "leakage_audit": leakage_audit,
    }
    if accounting and sum(EXPECTED_ROWS.values()) + sum(accounting["excluded"].values()) != curation_block["total_upstream"]:
        raise SystemExit("curation sums do not balance")
    report = {
        "source": source,
        "manifest": {"file": "manifest.csv.gz", "sha256": manifest_sha, "classes": "classes.csv", "classes_sha256": classes_sha},
        "manifest_hash": sha256_bytes(b"manifest:" + manifest_sha.encode() + b"\nclasses:" + classes_sha.encode()),
        "rows": {**dict(sorted(counts.items())), "total": len(rows)},
        "upstream": {"source": "VGGSound metadata/vggsound.csv + model-free scores of the VGGSound media (make_manifest.py score)",
                     "sha256": VGGSOUND_CSV_SHA, "rows": len(rows), "upstream_files": dict(sorted(upstream_files.items())), "curation": curation_block},
        "protocol_curation": dict(PROTOCOL_CURATION),
        "selection": accounting,
        "frame_grid_audit": frame_grid_audit(video, rows) if accounting else None,
        "rows_per_bin": {s: {str(b): sum(1 for r in rows if r["split"] == s and int(r["offset_bin"]) == b) for b in range(NUM_CLASSES)} for s in EXPECTED_ROWS},
        "chance": {"acc": round(1 / NUM_CLASSES, 4), "acc_tol1": round((3 * NUM_CLASSES - 2) / NUM_CLASSES**2, 4),
                   "mae_seconds": round(0.5 * (NUM_CLASSES**2 - 1) / (3 * NUM_CLASSES), 4), "acc_coarse5": 0.2},
        "n_coarse5_eval": sum(1 for r in rows if r["split"] == "test" and float(r["offset_seconds"]).is_integer()),
        "floor_threshold_3sigma": round(1 / 9 + 3.0 * math.sqrt((1 / 9) * (8 / 9) / EXPECTED_ROWS["stationary"]), 4),
    }
    return {"manifest_bytes": manifest_bytes, "classes_bytes": classes_bytes, "report": report}


def yaml_drift(folder: Path, report: dict) -> list[str]:
    """``--check``: ``benchmark.yaml`` must carry the derived hashes and curation values (the exclusion counts only
    when they were derived from scores)."""
    import yaml

    p = folder / "benchmark.yaml"
    if not p.is_file():
        return [f"{p} missing"]
    y = yaml.safe_load(p.read_text(encoding="utf-8"))
    drift = [f"benchmark.yaml manifest.{k}" for k in ("sha256", "classes_sha256") if y["manifest"][k] != report["manifest"][k]]
    if y["protocol"]["curation"] != report["protocol_curation"]:
        drift.append("benchmark.yaml protocol.curation")
    up = y["manifest"]["upstream"]
    if up["sha256"] != report["upstream"]["sha256"] or up["rows"] != report["upstream"]["rows"]:
        drift.append("benchmark.yaml manifest.upstream.sha256/rows")
    derived = report["upstream"]["curation"]
    for k, v in derived.items():
        if v is None:
            continue  # not derivable in this run (no scores / no listing)
        if up["curation"].get(k) != v:
            drift.append(f"benchmark.yaml manifest.upstream.curation.{k}")
    return drift


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["score"]:
        p = argparse.ArgumentParser(prog="make_manifest.py score", description="model-free per-clip scores of the VGGSound media")
        p.add_argument("--stage", choices=("audio", "video", "all"), default="all", help="audio (every clip), video (the audio-selected candidates; needs every audio score), or both")
        p.add_argument("--shard", default=None, help="i/n: score clips i::n (one output file per shard)")
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
        p.add_argument("--vggsound-root", type=Path, default=None, help="VGGSound root with metadata/vggsound.csv, audio/, video/ (default: the folder's media_root)")
        p.add_argument("--work", type=Path, default=None, help="score directory (default: <curation_root>/scores)")
        p.add_argument("--limit", type=int, default=0, help="first N clips only (testing)")
        return cmd_score(p.parse_args(argv[1:]))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--vggsound-root", type=Path, default=None, help="VGGSound root (default: the folder's media_root)")
    p.add_argument("--work", type=Path, default=None, help="score directory of `score` (default: <curation_root>/scores)")
    p.add_argument("--no-media", action="store_true", help="no VGGSound root: re-derive from the committed selection only")
    p.add_argument("--out", type=Path, default=HERE, help="folder to write into or compare with (default: this folder)")
    p.add_argument("--seed", type=int, default=PROTOCOL_CURATION["offset_seed"])
    p.add_argument("--check", action="store_true", help="compare with the files in --out instead of writing (exit 1 on drift)")
    ns = p.parse_args(argv)
    vroot = None if ns.no_media else (ns.vggsound_root or default_vggsound_root())
    work = ns.work
    if work is None and not ns.no_media:
        try:
            work = config.folder_root(HERE, "curation_root") / "scores"
        except config.ConfigError:
            work = None
    result = freeze(vroot, work, ns.seed, committed=ns.out)
    targets = {ns.out / "manifest.csv.gz": gzip_bytes(result["manifest_bytes"]), ns.out / "classes.csv": result["classes_bytes"]}
    if ns.check:
        drift = [str(path) for path, data in targets.items() if not path.is_file() or path.read_bytes() != data]
        drift += yaml_drift(ns.out, result["report"])
        print(json.dumps({**result["report"], "check": "ok" if not drift else "drift", "drift": drift}, indent=1))
        return 1 if drift else 0
    if result["report"]["source"] != "scores":
        raise SystemExit("no complete scores under --work: run `make_manifest.py score` first (or use --check)")
    ns.out.mkdir(parents=True, exist_ok=True)
    for path, data in targets.items():
        write_if_changed(path, data)
    print(json.dumps(result["report"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
