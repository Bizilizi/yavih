"""Freeze epic100_anticipation_stream's manifest from the pinned EPIC-KITCHENS-100 annotation files.

Upstream: github.com/epic-kitchens/epic-kitchens-100-annotations at commit ``ANNOTATIONS_COMMIT`` (prepare/01 fetches
the files below into ``<media_root>/annotations/``; sha256-pinned). Rows: every video of ``EPIC_100_train.csv`` and
``EPIC_100_validation.csv`` (the test labels are not public), one row per video: its participant, its file
``EPIC-KITCHENS/<participant>/videos/<video_id>.MP4`` (the official downloader's layout), its duration (from
``EPIC_100_video_info.csv``) and its narrated action segments ``[[start_s, verb_class, noun_class], ...]`` in start
order, every one of them (the official anticipation evaluation scores every validation segment).

    python make_manifest.py --annotations <dir>            # rewrite manifest.csv.gz
    python make_manifest.py --annotations <dir> --check    # re-derive and compare byte for byte
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANNOTATIONS_COMMIT = "ea8b40457a400c3fffa1c7f406ef3dc169cc2522"
FILES_SHA256 = {
    "EPIC_100_train.csv": "a3a7ef2e397bd8af12bd0496b09d61b1f2c1f3f6e1726fa86af50a27f4855e87",
    "EPIC_100_validation.csv": "35f7932ba0a1127a96cac215a98d35398946f343e3cea9ad6688ed17eee9d75d",
    "EPIC_100_video_info.csv": "75fd040f6662cb407b4ca2eed4811ca94df280a1aba00b6bf8a5cd286deff382",
}
HEADER = ["clip_id", "split", "participant_id", "rel_media", "duration_s", "segments"]


def _seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def build(ann: Path) -> tuple[bytes, dict]:
    for name, sha in FILES_SHA256.items():
        if hashlib.sha256((ann / name).read_bytes()).hexdigest() != sha:
            sys.exit(f"{ann / name}: not the pinned annotation file (commit {ANNOTATIONS_COMMIT})")
    with open(ann / "EPIC_100_video_info.csv", newline="") as f:
        duration = {r["video_id"]: float(r["duration"]) for r in csv.DictReader(f)}
    rows, segs = [], 0
    for split, name in (("train", "EPIC_100_train.csv"), ("validation", "EPIC_100_validation.csv")):
        by_video: dict[str, list] = defaultdict(list)
        part: dict[str, str] = {}
        with open(ann / name, newline="") as f:
            for r in csv.DictReader(f):
                by_video[r["video_id"]].append([round(_seconds(r["start_timestamp"]), 3), int(r["verb_class"]), int(r["noun_class"])])
                part[r["video_id"]] = r["participant_id"]
        for vid in sorted(by_video):
            s = sorted(by_video[vid])
            segs += len(s)
            rows.append([vid, split, part[vid], f"EPIC-KITCHENS/{part[vid]}/videos/{vid}.MP4", f"{duration[vid]:.6f}",
                         json.dumps(s, separators=(",", ":"))])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HEADER)
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    return data, {"rows": len(rows), "per_split": dict(Counter(r[1] for r in rows)), "segments": segs,
                  "sha256": hashlib.sha256(data).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--annotations", type=Path, required=True, help="the directory holding the pinned annotation csvs")
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args(argv)
    data, info = build(ns.annotations)
    out = HERE / "manifest.csv.gz"
    if ns.check:
        ok = gzip.decompress(out.read_bytes()) == data
        print(f"manifest {'matches' if ok else 'DIFFERS'} ({info['sha256']})")
        return 0 if ok else 1
    out.write_bytes(gzip.compress(data, mtime=0))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
