#!/usr/bin/env python
"""Freeze the ``vggsound_retrieval`` manifest from the official VGGSound listing and the pinned CAV-MAE Sync subset.

Inputs (read only, nothing is downloaded; sha256-pinned, a different file is refused):

- ``<vggsound-root>/metadata/vggsound.csv``: the official headerless listing (``ytid, start, label, split``);
  ``<vggsound-root>/{video,audio}/`` for presence (one ``scandir`` per directory; a disclosure, never a filter);
- ``cavmae_sync_subset.csv`` (shipped here): the 1 545 clip ids of the retrieval subset the CAV-MAE Sync authors
  evaluate on (5 per class × 309 classes; ``datafiles/vgg_test_5_per_class_for_retrieval_template.json`` of
  edsonroteia/cav-mae-sync @ 0add2e7, sha256 ``09c577a6…``; only its ``video_id`` values).

The rules:

1. **Rows.** Every ``split == test`` row of the listing (15 446), minus the clips whose YouTube source also has a clip
   in the train split (6, source level), in sorted ``clip_id`` order (``<ytid>_<start:06d>``), one split ``test``:
   15 440 rows.
2. **Columns** ``clip_id,split,class_name,rel_video,rel_audio,cavmae_sync_5pc``: ``class_name`` = the listing's label;
   ``cavmae_sync_5pc`` = ``1`` for the 1 545 subset clips (all of them are rows), else ``0``.
3. Writes ``manifest.csv.gz`` (byte-identical re-runs) and prints a JSON report with every value ``benchmark.yaml``
   carries. ``--check`` compares with the committed manifest and yaml facts instead of writing (exit 1 on drift).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent
NAME = "vggsound_retrieval"
MANIFEST_COLUMNS = ["clip_id", "split", "class_name", "rel_video", "rel_audio", "cavmae_sync_5pc"]
SPLIT = "test"
VGGSOUND_CSV_SHA = "ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8"
SUBSET_FILE = HERE / "cavmae_sync_subset.csv"
SUBSET_SHA = "77d83114deff63f551278526fb76845a9e69c95e9d61ede99dbc760ebc96847f"
SUBSET_SIZE = 1545
TOTAL_TEST = 15446
NUM_CLASSES = 309
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


def load_listing(raw: bytes) -> list[dict[str, Any]]:
    """``[{ytid, start, label, split}]`` of the headerless ``vggsound.csv``."""
    out = []
    for row in csv.reader(raw.decode("utf-8").splitlines()):
        if len(row) >= 4:
            out.append({"ytid": row[0].strip(), "start": int(float(row[1])), "label": row[2].strip(), "split": row[3].strip()})
    return out


def subset_ids() -> list[str]:
    """The pinned subset's clip ids (``cavmae_sync_subset.csv``, column ``clip_id``)."""
    text = _pinned(SUBSET_FILE, SUBSET_SHA).decode("utf-8")
    ids = [r["clip_id"] for r in csv.DictReader(text.splitlines())]
    if len(ids) != SUBSET_SIZE or len(set(ids)) != SUBSET_SIZE:
        raise SystemExit(f"{SUBSET_FILE}: {len(ids)} ids ({len(set(ids))} distinct), expected {SUBSET_SIZE}")
    return ids


def derive(listing: list[dict[str, Any]], subset: list[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    train_sources = {r["ytid"] for r in listing if r["split"] == "train"}
    test = [r for r in listing if r["split"] == "test"]
    if len(test) != TOTAL_TEST:
        raise SystemExit(f"vggsound.csv: {len(test)} test rows, expected {TOTAL_TEST}")
    leaked = sorted(f"{r['ytid']}_{r['start']:06d}" for r in test if r["ytid"] in train_sources)
    kept = {f"{r['ytid']}_{r['start']:06d}": r for r in test if r["ytid"] not in train_sources}
    if len(kept) != len(test) - len(leaked):
        raise SystemExit("vggsound.csv: duplicate test clip ids")
    sub = set(subset)
    outside = sorted(sub - set(kept))
    if outside:
        raise SystemExit(f"{len(outside)} subset clips are not rows (not test, or leaked), e.g. {outside[:5]}")
    rows = [{"clip_id": cid, "split": SPLIT, "class_name": r["label"], "rel_video": f"video/{cid}.mp4", "rel_audio": f"audio/{cid}.wav",
             "cavmae_sync_5pc": "1" if cid in sub else "0"} for cid, r in sorted(kept.items())]
    classes = Counter(r["class_name"] for r in rows)
    sub_classes = Counter(r["class_name"] for r in rows if r["cavmae_sync_5pc"] == "1")
    stats = {"leaked_clips": leaked, "excluded": {"leakage": len(leaked)}, "classes": len(classes),
             "clips_per_class": {"min": min(classes.values()), "median": sorted(classes.values())[len(classes) // 2], "max": max(classes.values())},
             "subset": {"rows": len(sub), "classes": len(sub_classes), "per_class": sorted(set(sub_classes.values()))}}
    return rows, stats


def listing_dir(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise SystemExit(f"media directory missing: {directory} (stage VGGSound first; setup.md 'Obtain the data')")
    return {e.name for e in os.scandir(directory)}


def default_root() -> Path:
    """The folder's ``media_root`` via yavih's resolver (``--data``, ``YAVIH_DATA__VGGSOUND_RETRIEVAL__MEDIA_ROOT``,
    config ``[data.vggsound_retrieval] media_root``, else the yaml default ``${datasets.vggsound}``)."""
    return config.folder_root(HERE, "media_root")


def freeze(vggsound_root: Path) -> dict[str, Any]:
    raw = _pinned(vggsound_root / "metadata" / "vggsound.csv", VGGSOUND_CSV_SHA)
    rows, stats = derive(load_listing(raw), subset_ids())
    if len({r["class_name"] for r in rows}) != NUM_CLASSES:
        raise SystemExit(f"{len({r['class_name'] for r in rows})} classes, expected {NUM_CLASSES}")
    vids, auds = listing_dir(vggsound_root / "video"), listing_dir(vggsound_root / "audio")
    present = sum(1 for r in rows if r["rel_video"].split("/", 1)[1] in vids and r["rel_audio"].split("/", 1)[1] in auds)
    manifest_bytes = csv_bytes(MANIFEST_COLUMNS, ([r[c] for c in MANIFEST_COLUMNS] for r in rows))
    return {
        "manifest_bytes": manifest_bytes, "manifest_sha256": sha256_bytes(manifest_bytes), "rows": len(rows), "budget_clips": {SPLIT: len(rows)},
        "upstream": {"source": "VGGSound vggsound.csv split == test, minus the clips whose source has a train clip; cavmae_sync_subset.csv flags the CAV-MAE Sync retrieval subset",
                     "sha256": VGGSOUND_CSV_SHA, "rows": TOTAL_TEST,
                     "upstream_files": {"vggsound/metadata/vggsound.csv": VGGSOUND_CSV_SHA, "cavmae_sync_subset.csv": SUBSET_SHA},
                     "curation": {"total_upstream": TOTAL_TEST, "per_split": {SPLIT: {"listed": len(rows), "staged": present, "missing": len(rows) - present}},
                                  "excluded": stats["excluded"],
                                  "leakage_audit": {"against": "VGGSound train ytids (metadata/vggsound.csv split==train), source level", "excluded": stats["excluded"]["leakage"]}}},
        "stats": {k: v for k, v in stats.items() if k != "excluded"},
    }


def _yaml_facts() -> dict[str, Any]:
    import yaml

    y = yaml.safe_load((HERE / "benchmark.yaml").read_text(encoding="utf-8"))
    return {"budget_clips": y["budget"]["clips"], "curation": y["manifest"]["upstream"]["curation"], "sha256": y["manifest"]["sha256"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vggsound-root", type=Path, default=None, help="VGGSound root: metadata/vggsound.csv, video/, audio/ (default: the folder's media_root)")
    ap.add_argument("--check", action="store_true", help="compare with the committed manifest and benchmark.yaml facts; exit 1 on drift")
    ns = ap.parse_args(argv)
    out = freeze(ns.vggsound_root or default_root())
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
    print(json.dumps({k: v for k, v in out.items() if k != "manifest_bytes"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
