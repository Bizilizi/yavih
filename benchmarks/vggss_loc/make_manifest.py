#!/usr/bin/env python
"""Freeze the ``vggss_loc`` manifest from the public VGG-SS annotations and the VGGSound listing.

Inputs (read only, nothing is downloaded; every file is sha256-pinned below, a different file is refused):

- ``<annotations-root>/vggss.json``: the VGG-SS annotations (5 158 VGGSound-test clips, 221 classes, 1–7 boxes each);
- ``<vggsound-root>/metadata/vggsound.csv``: the official VGGSound listing (the leakage rule), and
  ``<vggsound-root>/{video,audio}/`` for presence (one ``scandir`` per directory; a disclosure, never a filter).

The rules:

1. **Leakage.** Every entry must be a VGGSound test clip; an entry whose YouTube source also has a clip in the
   VGGSound train split is excluded (source level): 4 entries.
2. **Rows.** Every other entry, in sorted ``clip_id`` order, one split ``test``: 5 154 rows. Multi-box entries are
   kept (the scorer rasterises the union, as the published evaluation does); the 2 entries whose only box has no
   extent after the official ``max(x, 0)`` sanitisation and the 12 corrupt coordinates (``-2.8823e16``) are kept too
   (an empty ground truth scores 0 and is counted by ``n_empty_gt``).
3. **Columns** ``clip_id,split,boxes,class_name,rel_video,rel_audio``: ``boxes`` = the entry's boxes as ``x1|y1|x2|y2``
   quadruples ``;``-joined, in fractions of the original frame after ``max(x, 0)``, 6 decimals (the upstream values
   carry at most 6, so this is exact); ``class_name`` = the entry's ``class``. No ``classes.csv`` (``label.kind: none``).
4. Writes ``manifest.csv.gz`` (byte-identical re-runs) and prints a JSON report with every value ``benchmark.yaml``
   carries (sha256s, ``manifest.upstream``, ``curation``, presence, the ground-truth area facts the README quotes).

``--check`` compares the would-be manifest and the yaml's ``budget.clips`` / ``manifest.upstream.curation`` with the
committed ones instead of writing (exit 1 on drift). numpy is needed; torch is not.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent
NAME = "vggss_loc"

MANIFEST_COLUMNS = ["clip_id", "split", "boxes", "class_name", "rel_video", "rel_audio"]
SPLIT = "test"

# Pinned inputs.
VGGSS_JSON_SHA = "ae300b4a259267e6fe839970d495862edd70bfe3da63d925bcdb34054ed3b15e"
VGGSOUND_CSV_SHA = "ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8"
VGGSS_JSON_URL = "https://raw.githubusercontent.com/hche11/Localizing-Visual-Sounds-the-Hard-Way/master/metadata/vggss.json"

TOTAL_UPSTREAM = 5158
NUM_CLASSES = 221
BOX_DECIMALS = 6
GT_SIZE = 224
_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}_\d{6}$")
# benchmark.yaml protocol.clip_seconds: the nominal media duration (a fact; nothing is cut or padded).
CLIP_SECONDS = 10.0


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
        entries.append({"clip_id": clip_id, "source": clip_id.rsplit("_", 1)[0], "label": str(item["class"]),
                        "bbox": [[float(v) for v in box] for box in boxes]})
    return entries


def sanitise_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    """The official scorer's entire sanitisation (``test.py``): per-coordinate ``max(x, 0)``."""
    x1, y1, x2, y2 = (max(float(v), 0.0) for v in box)
    return x1, y1, x2, y2


def format_boxes(boxes: Sequence[Sequence[float]], decimals: int = BOX_DECIMALS) -> str:
    """``x1|y1|x2|y2`` quadruples ``;``-joined, fixed precision, ``-0.0`` normalised."""
    return ";".join("|".join(f"{float(v) + 0.0:.{decimals}f}" for v in box) for box in boxes)


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


def raster_union(boxes: Sequence[Sequence[float]], size: int = GT_SIZE) -> np.ndarray:
    """The published rasteriser: ``int(size * x)`` of the sanitised coordinates as slice bounds, boxes OR-ed."""
    gt = np.zeros((size, size), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = (int(size * v) for v in sanitise_box(box))
        gt[y1:y2, x1:x2] = True
    return gt


def derive(entries: Sequence[Mapping[str, Any]], vggsound_splits: Mapping[tuple[str, int], str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Rules 1–3 of the module docstring as a pure function: ``(rows, stats)``."""
    non_test = [e["clip_id"] for e in entries if vggsound_splits.get((e["source"], int(e["clip_id"].rsplit("_", 1)[1]))) != "test"]
    if non_test:
        raise SystemExit(f"{len(non_test)} VGG-SS entries are not in the official VGGSound test split, e.g. {non_test[:5]}")
    train_sources = train_source_ids(vggsound_splits)
    leaked = sorted(e["clip_id"] for e in entries if e["source"] in train_sources)
    kept = sorted((e for e in entries if e["source"] not in train_sources), key=lambda e: e["clip_id"])
    rows: list[dict[str, str]] = []
    boxes_per_entry: Counter = Counter()
    corrupt = empty = 0
    areas = []
    for e in kept:
        full = [sanitise_box(b) for b in e["bbox"]]
        if any(max(b) > 1.0 for b in full):
            raise SystemExit(f"{e['clip_id']}: a box exceeds the frame: {e['bbox']}")
        corrupt += sum(1 for b in e["bbox"] if min(b) < 0)
        gt = raster_union(e["bbox"])
        areas.append(gt.mean())
        empty += int(not gt.any())
        boxes_per_entry[len(full)] += 1
        cid = e["clip_id"]
        rows.append({"clip_id": cid, "split": SPLIT, "boxes": format_boxes(full), "class_name": e["label"],
                     "rel_video": f"video/{cid}.mp4", "rel_audio": f"audio/{cid}.wav"})
    area = np.asarray(areas)
    stats = {
        "leaked_entries": leaked,
        "excluded": {"leakage": len(entries) - len(kept)},
        "boxes_per_entry": {str(k): v for k, v in sorted(boxes_per_entry.items())},
        "multi_box_entries": sum(v for k, v in boxes_per_entry.items() if k > 1),
        "corrupt_coordinates": corrupt,
        "empty_gt_entries": empty,
        "classes": len({r["class_name"] for r in rows}),
        "gt_area": {"median": round(float(np.median(area)), 4), "mean": round(float(area.mean()), 4),
                    "entries_at_least_0.25": int((area >= 0.25).sum()), "entries_above_0.5": int((area > 0.5).sum())},
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
    """The folder's ``media_root`` via yavih's resolver (``--data``, ``YAVIH_DATA__VGGSS_LOC__MEDIA_ROOT``,
    config ``[data.vggss_loc] media_root``, else the yaml default ``${datasets.vggsound}``)."""
    return config.folder_root(HERE, "media_root")


def floor_stats(rows: list[dict[str, str]], grid: int = 14) -> dict[str, Any]:
    """The leave-one-out mean-mask floor through the folder's own pipeline (``loc.py``), for the README."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vggss_loc_scorer", HERE / "loc.py")
    loc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loc)
    ciou, mean_mask_maps, median_mask, upsample_bilinear = loc.ciou, loc.mean_mask_maps, loc.median_mask, loc.upsample_bilinear

    targets = [np.asarray([[float(v) for v in q.split("|")] for q in r["boxes"].split(";")]) for r in rows]
    floor = mean_mask_maps(targets, len(rows), grid=grid, size=GT_SIZE, leave_one_out=True)
    c = np.concatenate([ciou(median_mask(upsample_bilinear(floor[s:s + 256], GT_SIZE)), np.stack([raster_union(t) for t in targets[s:s + 256]]))
                        for s in range(0, len(rows), 256)])
    return {"ciou@0.5_floor": round(float((c >= 0.5).mean()), 4), "ciou_mean_floor": round(float(c.mean()), 4)}


# ----------------------------------------------------------------------------- freeze


def freeze(staging: Path, vggsound_root: Path) -> dict[str, Any]:
    vggss_raw = _pinned(staging / "vggss.json", VGGSS_JSON_SHA)
    vggsound_csv = vggsound_root / "metadata" / "vggsound.csv"
    _pinned(vggsound_csv, VGGSOUND_CSV_SHA)
    entries = load_annotations(vggss_raw)
    if len(entries) != TOTAL_UPSTREAM or len({e["label"] for e in entries}) != NUM_CLASSES:
        raise SystemExit(f"vggss.json: {len(entries)} entries / {len({e['label'] for e in entries})} classes, expected {TOTAL_UPSTREAM} / {NUM_CLASSES}")
    rows, stats = derive(entries, load_vggsound_splits(vggsound_csv))
    vids, auds = listing(vggsound_root / "video"), listing(vggsound_root / "audio")
    present = sum(1 for r in rows if r["rel_video"].split("/", 1)[1] in vids and r["rel_audio"].split("/", 1)[1] in auds)
    manifest_bytes = csv_bytes(MANIFEST_COLUMNS, ([r[c] for c in MANIFEST_COLUMNS] for r in rows))
    return {
        "manifest_bytes": manifest_bytes,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "rows": len(rows),
        "budget_clips": {SPLIT: len(rows)},
        "upstream": {
            "source": "VGG-SS vggss.json + VGGSound vggsound.csv: every entry not leaked into VGGSound train (source level), multi-box included, one split",
            "sha256": VGGSS_JSON_SHA, "rows": TOTAL_UPSTREAM,
            "upstream_files": {"vggss/vggss.json": VGGSS_JSON_SHA, "vggsound/metadata/vggsound.csv": VGGSOUND_CSV_SHA},
            "curation": {"total_upstream": TOTAL_UPSTREAM,
                         "per_split": {SPLIT: {"listed": len(rows), "staged": present, "missing": len(rows) - present}},
                         "excluded": stats["excluded"],
                         "leakage_audit": {"against": "VGGSound train ytids (metadata/vggsound.csv split==train), source level", "excluded": stats["excluded"]["leakage"]}},
        },
        "stats": {k: v for k, v in stats.items() if k != "excluded"},
        "floor": floor_stats(rows),
    }


def _yaml_facts() -> dict[str, Any]:
    import yaml

    y = yaml.safe_load((HERE / "benchmark.yaml").read_text(encoding="utf-8"))
    return {"budget_clips": y["budget"]["clips"], "curation": y["manifest"]["upstream"]["curation"], "sha256": y["manifest"]["sha256"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--staging-dir", type=Path, default=None, help="directory holding vggss.json (default: the folder's annotations_root)")
    ap.add_argument("--vggsound-root", type=Path, default=None, help="VGGSound root: metadata/vggsound.csv, video/, audio/ (default: the folder's media_root)")
    ap.add_argument("--check", action="store_true", help="compare with the committed manifest and benchmark.yaml facts; exit 1 on drift")
    ns = ap.parse_args(argv)
    staging = ns.staging_dir or default_staging_dir()
    root = ns.vggsound_root or default_root()
    out = freeze(staging, root)
    report = {k: v for k, v in out.items() if k != "manifest_bytes"}
    if ns.check:
        committed = gzip.decompress((HERE / "manifest.csv.gz").read_bytes())
        facts = _yaml_facts()
        drift = []
        if committed != out["manifest_bytes"]:
            drift.append("manifest.csv.gz differs")
        if facts["sha256"] != out["manifest_sha256"]:
            drift.append("benchmark.yaml manifest.sha256 differs")
        if facts["budget_clips"] != out["budget_clips"]:
            drift.append(f"budget.clips {facts['budget_clips']} != {out['budget_clips']}")
        if facts["curation"] != out["upstream"]["curation"]:
            drift.append(f"manifest.upstream.curation differs: {facts['curation']} != {out['upstream']['curation']}")
        print("manifest matches" if not drift else "DRIFT: " + "; ".join(drift), f"({out['manifest_sha256']})")
        return 0 if not drift else 1
    write_if_changed(HERE / "manifest.csv.gz", gzip_bytes(out["manifest_bytes"]))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
