#!/usr/bin/env python
"""Freeze the ``vggsounder_retrieval`` manifest from the VGGSounder labels and the official VGGSound listing.

Inputs (read only, nothing is downloaded; sha256-pinned, a different file is refused):

- ``data/vggsounder.csv`` of the ``vggsounder`` 1.0 package on PyPI (``pip download --no-deps vggsounder==1.0``;
  ``video_id,label,modality,background_music,static_image,voice_over``, one line per (video, class); the default
  variant without background-music clips, 12 014 videos; sha256 ``400979ea…``). The same file the ``vggsounder``
  folder freezes its test rows from;
- the official VGGSound listing ``<media_root>/metadata/vggsound.csv`` (``ytid,start,label,split``, no header,
  199 176 rows, sha256 ``ed9fdeed…``), for the leakage audit only;
- optionally the staged media ``<media_root>/{video,audio}/<clip_id>.{mp4,wav}``, for the presence counts under
  ``curation.per_split`` (presence never changes the rows).

The rules:

1. **Clips.** The package's videos in file order (12 014), minus the 6 whose YouTube id is a VGGSound-train id
   (the ``vggsounder`` folder's 12 008 test rows). Each clip's classes are split by the package's ``modality``
   column into audible-only (``A``), visible-only (``V``) and audible-and-visible (``AV``).
2. **Rows.** The clips with at least one ``AV`` class and no ``A`` or ``V`` class: everything that sounds is seen
   and everything seen sounds, so the clip's audio and its video describe the same events. 5 294 rows over 308
   classes, one split ``test``. Excluded and counted: ``no_av_class`` (no ``AV`` class at all) and
   ``a_or_v_only_class`` (an ``AV`` class next to an audible-only or visible-only one). No further curation:
   same-source clips, class imbalance and many-class clips are kept.
3. **Columns** ``clip_id,split,labels_av,rel_video,rel_audio``: ``labels_av`` = the clip's ``AV`` classes,
   ``;``-joined in file order (the class-level relevance of the folder's ``cls_*`` metrics; never handed over).
4. Writes ``manifest.csv.gz`` (gzip mtime 0, no file name; byte-identical re-runs) and prints a JSON report with
   every value ``benchmark.yaml`` carries (``manifest.sha256``, ``manifest.upstream``, ``budget.clips``, the class
   chance levels under ``chance_class``). ``--check`` compares with the committed manifest and yaml facts instead
   of writing (exit 1 on drift).

Usage::

    pip download --no-deps vggsounder==1.0 -d /tmp/vs && unzip -o /tmp/vs/vggsounder-1.0-py3-none-any.whl -d /tmp/vs
    python "$(yavih bench path vggsounder_retrieval)/make_manifest.py" --metadata /tmp/vs/vggsounder/data/vggsounder.csv --check
    python "$(yavih bench path vggsounder_retrieval)/make_manifest.py" --metadata /tmp/vs/vggsounder/data/vggsounder.csv   # re-freeze (git checkout)
    python "$(yavih bench path vggsounder_retrieval)/make_manifest.py" --no-media --metadata vggsounder.csv --vggsound-csv vggsound.csv --check

``--metadata`` defaults to the installed ``vggsounder`` package's ``data/vggsounder.csv`` (``pip install --no-deps
vggsounder==1.0``); ``--media-root`` to the folder's ``media_root`` (``yavih config get vggsounder_retrieval.media_root``);
``--vggsound-csv`` to ``<media_root>/metadata/vggsound.csv``. ``--no-media`` skips the presence counts (recorded as
staged = listed, the committed value).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent
NAME = "vggsounder_retrieval"
SPLIT = "test"
MANIFEST_COLUMNS = ["clip_id", "split", "labels_av", "rel_video", "rel_audio"]
_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}_\d{6}$")

VGGSOUNDER_CSV_SHA = "400979ea3dee34bed145d3fd96966e16689352df63b645c4730caf8eb2cacb06"
VGGSOUND_CSV_SHA = "ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8"
PACKAGE_VIDEOS = 12014
LEAKS = 6
ROWS = 5294
NUM_CLASSES = 308
RECALL_KS = (1, 5, 10)
UPSTREAM_SOURCE = ("vggsounder 1.0 package data/vggsounder.csv (the default variant): the clips whose classes are all audible and "
                   "visible (modality AV only), minus VGGSound-train ytids (vggsound.csv)")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _pinned(path: Path, want: str) -> bytes:
    if not path.is_file():
        raise SystemExit(f"missing input {path} (setup.md 'Obtain the data')")
    raw = path.read_bytes()
    got = sha256_bytes(raw)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got} != pinned {want}; a different input is refused (re-freezing would be a new protocol_version)")
    return raw


# ----------------------------------------------------------------------------- inputs


def read_package(raw: bytes) -> dict[str, dict[str, list[str]]]:
    """``vggsounder.csv`` -> ``{video_id: {"A": [...], "V": [...], "AV": [...]}}`` in file order (tokens in file order)."""
    videos: dict[str, dict[str, list[str]]] = {}
    for r in csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")):
        vid = r["video_id"]
        if not _CLIP_ID_RE.match(vid):
            raise SystemExit(f"vggsounder.csv: bad video_id {vid!r}")
        if r["background_music"].strip().lower() != "false":
            raise SystemExit(f"vggsounder.csv: {vid} is a background-music clip (expected the default variant)")
        if r["modality"] not in ("A", "V", "AV"):
            raise SystemExit(f"vggsounder.csv: {vid}: unknown modality {r['modality']!r}")
        if ";" in r["label"]:
            raise SystemExit(f"vggsounder.csv: {vid}: a class name holds ';' (the cell separator): {r['label']!r}")
        cells = videos.setdefault(vid, {"A": [], "V": [], "AV": []})
        if any(r["label"] in c for c in cells.values()):
            raise SystemExit(f"vggsounder.csv: {vid}: class {r['label']!r} listed twice")
        cells[r["modality"]].append(r["label"])
    return videos


def train_ytids(raw: bytes) -> set[str]:
    """The ytids of the ``split == train`` rows of ``vggsound.csv``."""
    out = set()
    for row in csv.reader(io.StringIO(raw.decode("utf-8"), newline="")):
        if len(row) != 4:
            raise SystemExit(f"vggsound.csv: expected 4 columns, got {row}")
        if row[3].strip() == "train":
            out.add(row[0].strip())
    return out


def listing(directory: Path) -> set[str]:
    """One ``scandir`` per directory (never one ``isfile`` per row on NFS)."""
    if not directory.is_dir():
        raise SystemExit(f"media directory missing: {directory} (stage VGGSound first, or pass --no-media)")
    return {e.name for e in os.scandir(directory)}


def package_csv() -> Path:
    """``data/vggsounder.csv`` of the installed ``vggsounder`` package."""
    try:
        import vggsounder
    except ImportError:
        raise SystemExit("vggsounder is not installed: pip install --no-deps vggsounder==1.0, or pass --metadata") from None
    return Path(vggsounder.__file__).resolve().parent / "data" / "vggsounder.csv"


# ----------------------------------------------------------------------------- chance levels


def relevant_counts(labels: list[set[str]]) -> list[int]:
    """Per clip, the other clips that share at least one class with it (the class-level relevant set's size)."""
    index: dict[str, set[int]] = {}
    for i, s in enumerate(labels):
        for c in s:
            index.setdefault(c, set()).add(i)
    return [len(set().union(*(index[c] for c in s))) - 1 for s in labels]


def class_chance(labels: list[set[str]]) -> dict[str, float]:
    """Expected ``cls_r@k_*`` of a uniformly random ranking: the mean over queries of ``1 - C(n-1-r, k) / C(n-1, k)``,
    ``r`` = the gallery clips (own clip excluded) sharing a class with the query. Both directions have the same value."""
    n = len(labels)
    rel = relevant_counts(labels)
    out = {}
    for k in RECALL_KS:
        p = sum(1 - comb(n - 1 - r, k) / comb(n - 1, k) for r in rel) / n
        for d in ("a2v", "v2a"):
            out[f"cls_r@{k}_{d}"] = round(p, 4)
    return out


# ----------------------------------------------------------------------------- freeze


def freeze(package_raw: bytes, listing_raw: bytes, media_root: Path | None) -> dict[str, Any]:
    videos = read_package(package_raw)
    if len(videos) != PACKAGE_VIDEOS:
        raise SystemExit(f"vggsounder.csv: {len(videos)} videos, expected {PACKAGE_VIDEOS}")
    train = train_ytids(listing_raw)
    clips = {vid: c for vid, c in videos.items() if vid.rsplit("_", 1)[0] not in train}
    leaks = len(videos) - len(clips)
    if leaks != LEAKS:
        raise SystemExit(f"leakage audit: {leaks} package videos carry a VGGSound-train ytid, expected {LEAKS}")
    no_av = [vid for vid, c in clips.items() if not c["AV"]]
    mixed = [vid for vid, c in clips.items() if c["AV"] and (c["A"] or c["V"])]
    kept = [(vid, c["AV"]) for vid, c in clips.items() if c["AV"] and not c["A"] and not c["V"]]
    if len(kept) != ROWS:
        raise SystemExit(f"{len(kept)} rows, expected {ROWS}")
    classes = Counter(lab for _, labs in kept for lab in labs)
    if len(classes) != NUM_CLASSES:
        raise SystemExit(f"{len(classes)} classes, expected {NUM_CLASSES}")
    rows = [{"clip_id": vid, "split": SPLIT, "labels_av": ";".join(labs), "rel_video": f"video/{vid}.mp4", "rel_audio": f"audio/{vid}.wav"}
            for vid, labs in kept]
    staged = len(rows)
    if media_root is not None:
        vids, auds = listing(media_root / "video"), listing(media_root / "audio")
        staged = sum(1 for r in rows if r["rel_video"].split("/", 1)[1] in vids and r["rel_audio"].split("/", 1)[1] in auds)
    curation = {"total_upstream": len(videos), "per_split": {SPLIT: {"listed": len(rows), "staged": staged, "missing": len(rows) - staged}},
                "excluded": {"leakage": leaks, "no_av_class": len(no_av), "a_or_v_only_class": len(mixed)},
                "leakage_audit": {"against": "VGGSound train ytids (vggsound.csv split==train)", "excluded": leaks}}
    assert len(rows) + sum(curation["excluded"].values()) == curation["total_upstream"], curation
    manifest_bytes = csv_bytes(MANIFEST_COLUMNS, ([r[c] for c in MANIFEST_COLUMNS] for r in rows))
    per_clip = Counter(len(labs) for _, labs in kept)
    ytid = Counter(vid.rsplit("_", 1)[0] for vid, _ in kept)
    rel = relevant_counts([set(labs) for _, labs in kept])
    return {
        "manifest_bytes": manifest_bytes, "manifest_sha256": sha256_bytes(manifest_bytes), "rows": len(rows), "budget_clips": {SPLIT: len(rows)},
        "upstream": {"source": UPSTREAM_SOURCE, "sha256": VGGSOUNDER_CSV_SHA, "rows": len(videos),
                     "upstream_files": {"vggsounder/data/vggsounder.csv": VGGSOUNDER_CSV_SHA, "vggsound/metadata/vggsound.csv": VGGSOUND_CSV_SHA},
                     "curation": curation},
        "chance_class": class_chance([set(labs) for _, labs in kept]),
        "stats": {"classes": len(classes), "clips_per_class": {"min": min(classes.values()), "max": max(classes.values()),
                                                               "under_10": sum(1 for v in classes.values() if v < 10)},
                  "largest_classes": classes.most_common(5), "classes_per_clip": dict(sorted(per_clip.items())),
                  "queries_without_relevant": sum(1 for r in rel if r == 0), "mean_relevant_fraction": round(sum(rel) / len(rel) / (len(rel) - 1), 4),
                  "same_source": {"ytids_with_several_clips": sum(1 for v in ytid.values() if v > 1), "clips": sum(v for v in ytid.values() if v > 1)}},
    }


def _yaml_facts() -> dict[str, Any]:
    import yaml

    y = yaml.safe_load((HERE / "benchmark.yaml").read_text(encoding="utf-8"))
    up = {k: v for k, v in y["manifest"]["upstream"].items() if k != "note"}
    return {"budget_clips": y["budget"]["clips"], "upstream": up, "sha256": y["manifest"]["sha256"],
            "chance_class": {k: v for k, v in y["metrics"]["chance"].items() if k.startswith("cls_")}}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=None, help="the vggsounder 1.0 package's data/vggsounder.csv (default: the installed package's)")
    ap.add_argument("--vggsound-csv", type=Path, default=None, help="the official vggsound.csv (default: <media_root>/metadata/vggsound.csv)")
    ap.add_argument("--media-root", type=Path, default=None, help="VGGSound root with video/, audio/ (default: the folder's media_root)")
    ap.add_argument("--no-media", action="store_true", help="skip the presence counts (no media root needed with --vggsound-csv)")
    ap.add_argument("--out", type=Path, default=HERE, help="folder to write manifest.csv.gz into (default: this folder)")
    ap.add_argument("--check", action="store_true", help="re-derive and compare with the committed manifest and yaml facts; write nothing; exit 1 on drift")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    media_root = None
    if not args.no_media or args.vggsound_csv is None:
        try:
            media_root = config.folder_root(HERE, "media_root", args.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root (or --no-media --vggsound-csv) or configure vggsounder_retrieval.media_root: {e}") from None
    listing_csv = args.vggsound_csv or media_root / "metadata" / "vggsound.csv"
    package_raw = _pinned(args.metadata or package_csv(), VGGSOUNDER_CSV_SHA)
    listing_raw = _pinned(listing_csv, VGGSOUND_CSV_SHA)
    out = freeze(package_raw, listing_raw, None if args.no_media else media_root)
    report = {k: v for k, v in out.items() if k != "manifest_bytes"}
    if args.check:
        committed = gzip.decompress((HERE / "manifest.csv.gz").read_bytes())
        facts = _yaml_facts()
        drift = []
        if committed != out["manifest_bytes"]:
            drift.append("manifest.csv.gz differs")
        if facts["sha256"] != out["manifest_sha256"]:
            drift.append("benchmark.yaml manifest.sha256 differs")
        if facts["budget_clips"] != out["budget_clips"]:
            drift.append(f"budget.clips {facts['budget_clips']} != {out['budget_clips']}")
        if facts["upstream"] != out["upstream"]:
            drift.append("manifest.upstream differs")
        if facts["chance_class"] != out["chance_class"]:
            drift.append(f"metrics.chance cls_* {facts['chance_class']} != {out['chance_class']}")
        print(json.dumps({**report, "check": "ok" if not drift else "drift", "drift": drift}, indent=1))
        return 0 if not drift else 1
    write_if_changed(args.out / "manifest.csv.gz", gzip_bytes(out["manifest_bytes"]))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
