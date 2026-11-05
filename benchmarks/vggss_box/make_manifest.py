#!/usr/bin/env python
"""Freeze the ``vggss_box`` manifest from the public VGG-SS annotations and the facts of the VGGSound media.

Inputs (read only, nothing is downloaded; every file is sha256-pinned below, a different file is refused):

- ``<staging-dir>/vggss.json``: the VGG-SS annotations (5 158 VGGSound-test clips, 221 classes, 1–7 boxes each);
- ``<vggsound-root>/metadata/vggsound.csv``: the official VGGSound listing (leakage rule), and
  ``<vggsound-root>/{video,audio}/`` for presence (one ``scandir`` per directory);
- the two geometry sidecars shipped in ``prepare/``: ``frame_sizes.json`` (``{clip_id: [width, height]}``, ffprobe)
  and ``video_geometry.json`` (``{clip_id: [num_frames, avg_fps]}``, torchcodec), one entry per non-leaked clip;
- ``prepare/audio_shorter_than_clip_seconds.txt``: the rows whose wav is shorter than ``protocol.clip_seconds``.

The rules:

1. **Leakage.** Every entry must be a VGGSound test clip; an entry whose YouTube source also has a clip in the
   VGGSound train split is excluded (source level).
2. **Split.** One seeded (``random.Random(20260913)``), source-level, class-stratified 80/20 draw (``build_split``)
   over the non-leaked entries (multi-box included) whose video holds a frame at every centre of the 1 s and
   0.5 s bucket grids shifted by 0.5 s (``offset_grid_ok`` on ``video_geometry.json``). An entry outside the draw
   (a video too short for that grid) is an eval row; its source must have no drawn entry.
3. **Rows.** Every non-leaked entry with exactly one box that is not degenerate after the official ``max(x, 0)``
   sanitisation (``test.py``), in sorted ``clip_id`` order: 4 614 = 3 700 train / 914 eval. Excluded and counted:
   4 leakage, 538 multi-box, 2 degenerate.
4. **Columns** ``clip_id,split,box,class_name,rel_video,rel_audio,frame_w,frame_h``: ``box`` = ``x1|y1|x2|y2`` in
   fractions of the original frame, 6 decimals (the upstream values carry at most 6, so this is exact);
   ``class_name`` = the entry's ``class``; ``frame_w``/``frame_h`` = the native frame size (``frame_sizes.json``).
   No ``classes.csv``: the loader requires ``classes: null`` for ``label.kind: box``.
5. Writes ``manifest.csv.gz`` (byte-identical re-runs) and prints a JSON report with every value
   ``benchmark.yaml`` carries (sha256s, ``manifest.upstream``, ``curation``, presence), the constant-box floor and
   the crop-coverage facts behind ``output_contract.requires.grid.min_cell_coverage``. ``curation.short_clips`` =
   rows whose audio (the pinned list) or video (``video_geometry.json``) is shorter than ``clip_seconds``;
   ``--census`` re-reads every wav header and refuses on any difference from the pinned list.

``--check`` compares the would-be manifest and the yaml's ``budget.clips`` / ``manifest.upstream.curation`` with the
committed ones instead of writing (exit 1 on drift). numpy is needed; torch is not.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent
NAME = "vggss_box"

MANIFEST_COLUMNS = ["clip_id", "split", "box", "class_name", "rel_video", "rel_audio", "frame_w", "frame_h"]
SPLITS = ("train", "eval")

# Pinned inputs.
VGGSS_JSON_SHA = "ae300b4a259267e6fe839970d495862edd70bfe3da63d925bcdb34054ed3b15e"
VGGSOUND_CSV_SHA = "ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8"
FRAME_SIZES_FILE = HERE / "prepare" / "frame_sizes.json"
FRAME_SIZES_SHA = "4ac9cf7250149b2fd39c59c0f01156c59e4c7fa2474099f55c25a21131edb37f"
VIDEO_GEOMETRY_FILE = HERE / "prepare" / "video_geometry.json"
VIDEO_GEOMETRY_SHA = "48cbfb8933225b2c18bce3b6c2b214d75ab062b7a01f6771a478f7feb6896fad"
VGGSS_JSON_URL = "https://raw.githubusercontent.com/hche11/Localizing-Visual-Sounds-the-Hard-Way/master/metadata/vggss.json"

# ----------------------------------------------------------------------------- freeze constants

TOTAL_UPSTREAM = 5158
NUM_CLASSES = 221
_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}_\d{6}$")
RGB_OFFSET_SECONDS = 0.5
# (num_buckets, bucket_seconds, alignment) of the duration gate's bucket grids
OFFSET_GRID_GEOMETRIES: tuple[tuple[int, float, str], ...] = ((8, 1.0, "start"), (16, 0.5, "start"))
MIN_RESIZE = 256
CROP_SIZE = 224
BOX_DECIMALS = 6
SPLIT_SEED = 20260913
SPLIT_FRACTION = 0.2
MIN_EVAL_PER_CLASS = 2

# benchmark.yaml protocol.clip_seconds: the nominal media duration (a fact; nothing is cut or padded).
CLIP_SECONDS = 10.0

# Frozen rows whose wav is shorter than CLIP_SECONDS (487 rows = 384 train / 103 eval, min 4.94 s, all 16 kHz
# mono), pinned (sorted ids, one per line) so `curation.short_clips` needs no wav-header reads; `--census`
# re-reads them and refuses on any difference.
SHORT_AUDIO_FILE = HERE / "prepare" / "audio_shorter_than_clip_seconds.txt"
SHORT_AUDIO_SHA = "8a06c63d606827f9e82a718d9a562fa7b2b6ad3441a4c185e2b5f4114f60cbe1"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _pinned(path: Path, expected: str) -> bytes:
    if not path.is_file():
        raise SystemExit(f"missing input {path} (see setup.md 'Obtain the data')")
    raw = path.read_bytes()
    got = sha256_bytes(raw)
    if got != expected:
        raise SystemExit(f"{path}: sha256 {got} != pinned {expected}; a different input is refused (re-freezing would be a new protocol_version)")
    return raw


def short_audio_clips() -> frozenset[str]:
    """The pinned ids of ``SHORT_AUDIO_FILE`` (sha256-checked)."""
    return frozenset(_pinned(SHORT_AUDIO_FILE, SHORT_AUDIO_SHA).decode("utf-8").split())


def frame_sizes() -> dict[str, tuple[int, int]]:
    """``{clip_id: (width, height)}`` of the shipped ``frame_sizes.json`` (sha256-checked)."""
    return {k: (int(v[0]), int(v[1])) for k, v in json.loads(_pinned(FRAME_SIZES_FILE, FRAME_SIZES_SHA)).items()}


def video_geometry() -> dict[str, tuple[int, float]]:
    """``{clip_id: (num_frames, avg_fps)}`` of the shipped ``video_geometry.json`` (sha256-checked)."""
    return {k: (int(v[0]), float(v[1])) for k, v in json.loads(_pinned(VIDEO_GEOMETRY_FILE, VIDEO_GEOMETRY_SHA)).items()}


# ----------------------------------------------------------------------------- freeze rules


def load_annotations(raw_json: bytes) -> list[dict[str, Any]]:
    """The ``vggss.json`` entries as validated ``{clip_id, source, label, bbox}`` records."""
    raw = json.loads(raw_json.decode("utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"vggss.json: expected a JSON list, got {type(raw).__name__}")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        clip_id = str(item["file"])
        if not _CLIP_ID_RE.fullmatch(clip_id):
            raise ValueError(f"vggss.json: entry file {clip_id!r} does not match '<ytid>_<6-digit-start>'")
        if clip_id in seen:
            raise ValueError(f"vggss.json: duplicate entry for {clip_id!r}")
        seen.add(clip_id)
        boxes = item["bbox"]
        if not boxes:
            raise ValueError(f"vggss.json: entry {clip_id!r} has no boxes")
        for box in boxes:
            if len(box) != 4:
                raise ValueError(f"vggss.json: entry {clip_id!r} has a {len(box)}-element box")
        entries.append(
            {
                "clip_id": clip_id,
                "source": clip_id.rsplit("_", 1)[0],
                "label": str(item["class"]),
                "bbox": [[float(v) for v in box] for box in boxes],
            }
        )
    return entries


def sanitise_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    """The official scorer's entire sanitisation (``test.py``): per-coordinate ``max(x, 0)``."""
    x1, y1, x2, y2 = (max(float(v), 0.0) for v in box)
    return x1, y1, x2, y2


def format_boxes(boxes: Sequence[Sequence[float]], decimals: int = BOX_DECIMALS) -> str:
    """``x1|y1|x2|y2`` quadruples ``;``-joined, fixed precision, ``-0.0`` normalised."""
    return ";".join("|".join(f"{float(v) + 0.0:.{decimals}f}" for v in box) for box in boxes)


def resized_size(width: int, height: int, min_resize: int = MIN_RESIZE) -> tuple[int, int]:
    """torchvision ``Resize(min_resize)`` output size (short side to ``min_resize``, long side truncated)."""
    short, long_side = (width, height) if width <= height else (height, width)
    new_long = int(min_resize * long_side / short)
    return (min_resize, new_long) if width <= height else (new_long, min_resize)


def crop_window(width: int, height: int, min_resize: int = MIN_RESIZE, crop_size: int = CROP_SIZE) -> tuple[int, int, int, int]:
    """``(W, H, left, top)`` of torchvision ``Resize(min_resize)`` + ``CenterCrop(crop_size)``
    (``int(round((n - c) / 2.0))``)."""
    resized_w, resized_h = resized_size(width, height, min_resize)
    if resized_w < crop_size or resized_h < crop_size:
        raise ValueError(f"resized frame {resized_w}x{resized_h} is smaller than the {crop_size}x{crop_size} crop")
    return resized_w, resized_h, int(round((resized_w - crop_size) / 2.0)), int(round((resized_h - crop_size) / 2.0))


def load_vggsound_splits(vggsound_csv: Path) -> dict[tuple[str, int], str]:
    """``(ytid, start_seconds) -> split`` from the official headerless ``vggsound.csv``."""
    splits: dict[tuple[str, int], str] = {}
    with open(vggsound_csv, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 4:
                splits[(row[0].strip(), int(float(row[1])))] = row[3].strip()
    return splits


def train_source_ids(splits: Mapping[tuple[str, int], str]) -> set[str]:
    """Every ytid with at least one segment in the official train split (source level)."""
    return {ytid for (ytid, _start), split in splits.items() if split == "train"}


def bucket_frame_indices_unclamped(fps: float, num_buckets: int, bucket_seconds: float = 1.0, offset_seconds: float = 0.0) -> list[int]:
    """The frame index at each bucket centre ``offset + (b + 0.5) · bucket_seconds``."""
    return [int(round((offset_seconds + (b + 0.5) * bucket_seconds) * fps)) for b in range(num_buckets)]


def bucket_frame_indices(total: int, fps: float, num_buckets: int, bucket_seconds: float = 1.0, offset_seconds: float = 0.0) -> list[int]:
    """:func:`bucket_frame_indices_unclamped` clamped to ``[0, total - 1]``."""
    return [min(total - 1, max(0, i)) for i in bucket_frame_indices_unclamped(fps, num_buckets, bucket_seconds, offset_seconds)]


def offset_grid_ok(num_frames: int, fps: float, *, offset_seconds: float = RGB_OFFSET_SECONDS, geometries: Sequence[tuple[int, float, str]] = OFFSET_GRID_GEOMETRIES) -> bool:
    """The draw's duration gate: a real frame (no clamping) at every shifted bucket centre of every grid."""
    total = int(num_frames)
    rate = float(fps) or 25.0
    if total <= 0:
        return False
    for num_buckets, bucket_seconds, window_align in geometries:
        if window_align != "start":
            raise ValueError("only 'start'-aligned geometries are supported")
        wanted = bucket_frame_indices_unclamped(rate, num_buckets, bucket_seconds, offset_seconds)
        if bucket_frame_indices(total, rate, num_buckets, bucket_seconds, offset_seconds) != wanted:
            return False
    return True


def build_split(entries: Sequence[Mapping[str, Any]], seed: int = SPLIT_SEED, fraction: float = SPLIT_FRACTION, min_eval_per_class: int = MIN_EVAL_PER_CLASS) -> dict[str, str]:
    """Seeded, source-level, class-stratified 80/20 split ``{clip_id: "train" | "eval"}``.

    Classes rarest-first (ties by name), each class's undecided sources sorted then shuffled by one
    ``random.Random(seed)`` until the class has ``max(min_eval_per_class, round(fraction · n))`` eval entries, then a
    top-up pass over the sorted+shuffled undecided sources until the eval share reaches ``fraction``. Whole source
    videos are assigned, never single entries.
    """
    rng = random.Random(seed)
    by_clip = {entry["clip_id"]: entry for entry in entries}
    source_of = {cid: entry["source"] for cid, entry in by_clip.items()}
    class_clips: dict[str, list[str]] = defaultdict(list)
    for cid, entry in sorted(by_clip.items()):
        class_clips[entry["label"]].append(cid)

    eval_sources: set[str] = set()

    def class_eval_count(cls: str) -> int:
        return sum(1 for cid in class_clips[cls] if source_of[cid] in eval_sources)

    for cls in sorted(class_clips, key=lambda c: (len(class_clips[c]), c)):
        n_entries = len(class_clips[cls])
        target = min(n_entries, max(min_eval_per_class, int(round(fraction * n_entries))))
        candidates = sorted({source_of[cid] for cid in class_clips[cls]} - eval_sources)
        rng.shuffle(candidates)
        for source in candidates:
            if class_eval_count(cls) >= target:
                break
            eval_sources.add(source)

    total_target = int(round(fraction * len(by_clip)))
    current = sum(1 for cid in by_clip if source_of[cid] in eval_sources)
    if current < total_target:
        undecided = sorted({s for s in source_of.values() if s not in eval_sources})
        rng.shuffle(undecided)
        entries_per_source: dict[str, int] = defaultdict(int)
        for source in source_of.values():
            entries_per_source[source] += 1
        for source in undecided:
            if current >= total_target:
                break
            eval_sources.add(source)
            current += entries_per_source[source]

    return {cid: ("eval" if source_of[cid] in eval_sources else "train") for cid in by_clip}


def derive(entries: Sequence[Mapping[str, Any]], vggsound_splits: Mapping[tuple[str, int], str],
           sizes: Mapping[str, Sequence[int]], geometry: Mapping[str, Sequence[float]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Rules 1–4 of the module docstring as a pure function: ``(rows, stats)``."""
    non_test = [e["clip_id"] for e in entries if vggsound_splits.get((e["source"], int(e["clip_id"].rsplit("_", 1)[1]))) != "test"]
    if non_test:
        raise SystemExit(f"{len(non_test)} VGG-SS entries are not in the official VGGSound test split, e.g. {non_test[:5]}")
    train_sources = train_source_ids(vggsound_splits)
    leaked_sources = sorted({e["source"] for e in entries if e["source"] in train_sources})
    kept = [e for e in entries if e["source"] not in train_sources]

    # --- 2. the seeded draw over the entries that pass the duration gate
    drawn = [e for e in kept if offset_grid_ok(*geometry[e["clip_id"]])]
    split = build_split(drawn)
    drawn_sources = {e["source"] for e in drawn}

    # --- 3/4. single-box, non-degenerate rows in full-frame fractions
    multi = degenerate = 0
    rows: list[dict[str, str]] = []
    for e in sorted(kept, key=lambda x: x["clip_id"]):
        cid, bbox = e["clip_id"], e["bbox"]
        if len(bbox) != 1:
            multi += 1
            continue
        full = sanitise_box(bbox[0])
        if max(full) > 1.0:
            raise SystemExit(f"{cid}: upstream box {bbox[0]} exceeds the frame")
        x1, y1, x2, y2 = full
        if not (x2 > x1 and y2 > y1):
            degenerate += 1  # zero-extent box: excluded and counted
            continue
        if cid in split:
            row_split = split[cid]
        else:  # outside the draw: its source must have no drawn entry → eval
            if e["source"] in drawn_sources:
                raise SystemExit(f"{cid}: undrawn entry shares source {e['source']} with a drawn entry (source-level split)")
            row_split = "eval"
        w, h = sizes[cid]
        rows.append({"clip_id": cid, "split": row_split, "box": format_boxes([full]), "class_name": e["label"],
                     "rel_video": f"video/{cid}.mp4", "rel_audio": f"audio/{cid}.wav", "frame_w": str(w), "frame_h": str(h)})

    # leakage check on the frozen rows: no source straddles train/eval, none is a VGGSound-train source
    src_split: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        src_split[r["clip_id"].rsplit("_", 1)[0]].add(r["split"])
    straddling = sorted(s for s, v in src_split.items() if len(v) > 1)
    in_train = sorted(s for s in src_split if s in train_sources)
    if straddling or in_train:
        raise SystemExit(f"leakage check failed: straddling={straddling[:5]} train_sources={in_train[:5]}")
    stats = {
        "leaked_sources": leaked_sources,
        "excluded": {"leakage": len(entries) - len(kept), "multi_box": multi, "degenerate_box": degenerate},
        "draw": {"entries": len(drawn), "outside_draw": len(kept) - len(drawn),
                 "rows_outside_draw": sum(1 for r in rows if r["clip_id"] not in split)},
    }
    return rows, stats


# ----------------------------------------------------------------------------- report helpers


def listing(directory: Path) -> set[str]:
    """One ``scandir`` per directory (avoids one stat per row on network filesystems)."""
    if not directory.is_dir():
        raise SystemExit(f"media directory missing: {directory} (stage VGGSound first; setup.md 'Obtain the data')")
    return {e.name for e in os.scandir(directory)}


def default_staging_dir() -> Path:
    """The folder's ``annotations_root`` (``vggss.json``) via yavih's resolver."""
    return config.folder_root(HERE, "annotations_root")


def default_root() -> Path:
    """The folder's ``media_root`` via yavih's resolver (``--data``, ``YAVIH_DATA__VGGSS_BOX__MEDIA_ROOT``,
    config ``[data.vggss_box] media_root``, else the yaml default ``${datasets.vggsound}``)."""
    return config.folder_root(HERE, "media_root")


def _cxcywh(cell: str) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (max(float(v), 0.0) for v in cell.split("|"))
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def _iou(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    def xyxy(b: np.ndarray) -> np.ndarray:
        return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)

    p, g = xyxy(p), xyxy(g)
    iw = np.clip(np.minimum(p[:, 2], g[:, 2]) - np.maximum(p[:, 0], g[:, 0]), 0, None)
    ih = np.clip(np.minimum(p[:, 3], g[:, 3]) - np.maximum(p[:, 1], g[:, 1]), 0, None)
    inter = iw * ih
    union = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]) + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def floor_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    """The constant-box floor (mean TRAIN cxcywh box scored on eval, ``iou_floor``) and the population facts
    README.md quotes. Pure numpy, the same arithmetic as ``yavih.metrics.box``."""
    tr = np.asarray([_cxcywh(r["box"]) for r in rows if r["split"] == "train"], dtype=np.float64)
    ev = np.asarray([_cxcywh(r["box"]) for r in rows if r["split"] == "eval"], dtype=np.float64)
    const = tr.mean(axis=0)
    iou = _iou(np.broadcast_to(const, ev.shape), ev)
    g = np.stack([ev[:, 0] - ev[:, 2] / 2, ev[:, 1] - ev[:, 3] / 2, ev[:, 0] + ev[:, 2] / 2, ev[:, 1] + ev[:, 3] / 2], axis=1)
    centre_in = (const[0] >= g[:, 0]) & (const[0] <= g[:, 2]) & (const[1] >= g[:, 1]) & (const[1] <= g[:, 3])
    areas = np.concatenate([tr[:, 2] * tr[:, 3], ev[:, 2] * ev[:, 3]])
    edge = sum(1 for r in rows if (lambda b: min(b[0], b[1]) == 0.0 or max(b[2], b[3]) == 1.0)([float(v) for v in r["box"].split("|")]))
    return {
        "mean_train_box_cxcywh": [round(float(x), 6) for x in const],
        "iou_floor": round(float(iou.mean()), 4),
        "acc@0.5_floor": round(float((iou >= 0.5).mean()), 4),
        "acc@0.3_floor": round(float((iou >= 0.3).mean()), 4),
        "center_in_gt_floor": round(float(centre_in.mean()), 4),
        "median_box_area": round(float(np.median(areas)), 4),
        "rows_with_edge_coordinate": edge,
    }


def crop_region_resize_centre_crop(width: int, height: int) -> tuple[float, float, float, float]:
    """``(y0, x0, y1, x1)`` of the ORIGINAL frame a Resize(short 256) + CenterCrop(224) bridge sees, as fractions —
    what ``yavih.media.rules.crop_region`` returns for ``{"resize": {"short": 256}, "crop": {"kind": "center",
    "size": [224, 224]}}``."""
    resized_w, resized_h, left, top = crop_window(int(width), int(height))
    return top / resized_h, left / resized_w, (top + CROP_SIZE) / resized_h, (left + CROP_SIZE) / resized_w


def cell_coverage(region: Sequence[float], tokens: int = 14, grid: int = 14) -> float:
    """Fraction of the ``grid × grid`` full-frame cells holding ≥ 1 centre of a ``tokens × tokens`` patch grid
    laid over ``region`` (``N_ELIGIBILITY`` for box: target region = the full frame)."""
    y0, x0, y1, x1 = region
    c = (np.arange(tokens) + 0.5) / tokens
    ys = np.minimum((grid * (y0 + c * (y1 - y0))).astype(int), grid - 1)
    xs = np.minimum((grid * (x0 + c * (x1 - x0))).astype(int), grid - 1)
    return len(set(ys.tolist())) * len(set(xs.tolist())) / grid**2


def crop_coverage(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Grid-cell coverage of a ViT/16-at-224 Resize(256) + CenterCrop(224) bridge per row (setup.md "Output
    contract"): the facts behind ``min_cell_coverage: 0.25``."""
    cov = np.asarray([cell_coverage(crop_region_resize_centre_crop(int(r["frame_w"]), int(r["frame_h"]))) for r in rows])
    aspect = Counter("16:9" if abs(int(r["frame_w"]) / int(r["frame_h"]) - 16 / 9) < 0.01 else "4:3" if abs(int(r["frame_w"]) / int(r["frame_h"]) - 4 / 3) < 0.01 else "other" for r in rows)
    return {
        "median": round(float(np.median(cov)), 4),
        "min": round(float(cov.min()), 4),
        "rows_below_0.25": int((cov < 0.25).sum()),
        "rows_below_0.5": int((cov < 0.5).sum()),
        "aspect_rows": dict(sorted(aspect.items())),
    }


def audio_census(root: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    """Wav-header census (soundfile; freeze-time only): rows whose audio is shorter than ``CLIP_SECONDS``."""
    from concurrent.futures import ThreadPoolExecutor

    import soundfile as sf  # noqa: PLC0415

    def info(r: dict[str, str]) -> tuple[str, float]:
        i = sf.info(str(root / r["rel_audio"]))
        return r["split"], i.frames / i.samplerate

    with ThreadPoolExecutor(16) as ex:
        durs = list(ex.map(info, rows))
    out: dict[str, Any] = {"short_audio_ids": sorted(r["clip_id"] for r, (_, d) in zip(rows, durs, strict=True) if d < CLIP_SECONDS)}
    for thr in (8.5, 9.0, CLIP_SECONDS):
        c = Counter(s for s, d in durs if d < thr)
        out[f"audio_lt_{thr}"] = {s: c.get(s, 0) for s in SPLITS}
    out["audio_min_seconds"] = round(min(d for _, d in durs), 4)
    return out


# ----------------------------------------------------------------------------- freeze


def freeze(staging: Path, vggsound_root: Path, *, census: bool = False) -> dict[str, Any]:
    vggss_raw = _pinned(staging / "vggss.json", VGGSS_JSON_SHA)
    vggsound_csv = vggsound_root / "metadata" / "vggsound.csv"
    _pinned(vggsound_csv, VGGSOUND_CSV_SHA)
    sizes, geometry = frame_sizes(), video_geometry()

    entries = load_annotations(vggss_raw)
    if len(entries) != TOTAL_UPSTREAM or len({e["label"] for e in entries}) != NUM_CLASSES:
        raise SystemExit(f"vggss.json: {len(entries)} entries / {len({e['label'] for e in entries})} classes, expected {TOTAL_UPSTREAM} / {NUM_CLASSES}")
    rows, stats = derive(entries, load_vggsound_splits(vggsound_csv), sizes, geometry)

    # presence (disclosure only: presence never filters rows)
    vids = listing(vggsound_root / "video")
    auds = listing(vggsound_root / "audio")
    per_split: dict[str, dict[str, int]] = {}
    for s in SPLITS:
        srows = [r for r in rows if r["split"] == s]
        present = sum(1 for r in srows if r["rel_video"].split("/", 1)[1] in vids and r["rel_audio"].split("/", 1)[1] in auds)
        per_split[s] = {"listed": len(srows), "staged": present, "missing": len(srows) - present}

    manifest_bytes = csv_bytes(MANIFEST_COLUMNS, ([r[c] for c in MANIFEST_COLUMNS] for r in rows))
    row_ids = {r["clip_id"] for r in rows}
    short_audio = short_audio_clips()
    if not short_audio <= row_ids:
        raise SystemExit(f"pinned short-audio ids not in the frozen rows: {sorted(short_audio - row_ids)[:5]}")

    def _video_seconds(cid: str) -> float:
        n, fps = geometry[cid]
        return n / (fps or 25.0)

    video_short_ids = {cid for cid in row_ids if _video_seconds(cid) < CLIP_SECONDS}
    short_union = video_short_ids | short_audio
    report: dict[str, Any] = {
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "rows": {s: per_split[s]["listed"] for s in SPLITS},
        "classes_in_population": len({r["class_name"] for r in rows}),
        "classes_in_eval": len({r["class_name"] for r in rows if r["split"] == "eval"}),
        "upstream": {
            "source": "VGG-SS vggss.json + VGGSound vggsound.csv: every non-leaked single-box non-degenerate entry; seeded source-level class-stratified split (entries outside the draw: eval)",
            "sha256": VGGSS_JSON_SHA,
            "rows": len(entries),
            "upstream_files": {
                "vggss/vggss.json": VGGSS_JSON_SHA,
                "vggsound/metadata/vggsound.csv": VGGSOUND_CSV_SHA,
                "prepare/frame_sizes.json": FRAME_SIZES_SHA,
                "prepare/video_geometry.json": VIDEO_GEOMETRY_SHA,
            },
            "curation": {
                "total_upstream": TOTAL_UPSTREAM,
                "per_split": per_split,
                "excluded": stats["excluded"],
                # rows with audio (pinned short-audio list) or video (video_geometry) shorter than clip_seconds
                "short_clips": len(short_union),
                "leakage_audit": {"against": "VGGSound train ytids (metadata/vggsound.csv split==train), source level", "excluded": stats["excluded"]["leakage"]},
            },
        },
        "leaked_sources": stats["leaked_sources"],
        "draw": stats["draw"],
        "video_shorter_than_clip_seconds": {s: sum(1 for r in rows if r["split"] == s and r["clip_id"] in video_short_ids) for s in SPLITS},
        "audio_shorter_than_clip_seconds": {s: sum(1 for r in rows if r["split"] == s and r["clip_id"] in short_audio) for s in SPLITS},
        "short_clips_per_split": {s: sum(1 for r in rows if r["split"] == s and r["clip_id"] in short_union) for s in SPLITS},
        "frame_sizes_top": Counter(f"{r['frame_w']}x{r['frame_h']}" for r in rows).most_common(5),
        "portrait_rows": sum(1 for r in rows if int(r["frame_h"]) > int(r["frame_w"])),
        "floor": floor_stats(rows),
        "crop_coverage": crop_coverage(rows),
    }
    cur = report["upstream"]["curation"]
    if sum(v["listed"] for v in cur["per_split"].values()) + sum(cur["excluded"].values()) != cur["total_upstream"]:
        raise SystemExit(f"curation invariant violated: {cur}")
    if census:
        report["audio_census"] = audio_census(vggsound_root, rows)
        if set(report["audio_census"]["short_audio_ids"]) != short_audio:
            raise SystemExit(f"wav-header census disagrees with the pinned {SHORT_AUDIO_FILE.name} — refusing to freeze")
    return {"manifest_bytes": manifest_bytes, "report": report}


def yaml_drift(report: Mapping[str, Any], folder: Path = HERE) -> list[str]:
    """What ``benchmark.yaml`` records differently from ``report`` (``budget.clips``, ``manifest.sha256``,
    ``manifest.upstream`` source/sha256/rows/upstream_files/curation)."""
    import yaml

    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    up, ry = y["manifest"]["upstream"], report["upstream"]
    out = [f"budget.clips {y['budget']['clips']} != {report['rows']}"] if y["budget"]["clips"] != report["rows"] else []
    if y["manifest"]["sha256"] != report["manifest_sha256"]:
        out.append(f"manifest.sha256 {y['manifest']['sha256']} != {report['manifest_sha256']}")
    out += [f"manifest.upstream.{k} differs" for k in ("source", "sha256", "rows", "upstream_files", "curation") if up.get(k) != ry[k]]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--staging-dir", type=Path, default=None, help="dir with vggss.json (default: the folder's annotations_root)")
    p.add_argument("--vggsound-root", type=Path, default=None, help="VGGSound root with video/, audio/, metadata/ (default: the folder's media_root)")
    p.add_argument("--out", type=Path, default=HERE, help="folder to write manifest.csv.gz into (default: this folder)")
    p.add_argument("--census", action="store_true", help="also read every wav header and check the pinned short-audio list")
    p.add_argument("--check", action="store_true", help="compare against the committed manifest.csv.gz and benchmark.yaml instead of writing (exit 1 on drift)")
    ns = p.parse_args(argv)
    try:
        staging = ns.staging_dir or default_staging_dir()
        root = ns.vggsound_root or default_root()
    except config.ConfigError as exc:
        raise SystemExit(f"{exc} (or pass --staging-dir / --vggsound-root)") from None
    result = freeze(staging, root, census=ns.census)
    target = ns.out / "manifest.csv.gz"
    gz = gzip_bytes(result["manifest_bytes"])
    if ns.check:
        drift = [] if target.is_file() and target.read_bytes() == gz else ["manifest.csv.gz bytes"]
        drift += yaml_drift(result["report"])
        print(json.dumps({**result["report"], "check": "ok" if not drift else "drift", "drift": drift}, indent=1))
        return 1 if drift else 0
    ns.out.mkdir(parents=True, exist_ok=True)
    write_if_changed(target, gz)
    print(json.dumps(result["report"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
