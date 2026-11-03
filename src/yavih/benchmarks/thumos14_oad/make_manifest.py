"""Freeze thumos14_oad's manifest from the official THUMOS'14 temporal annotations and the downloaded videos.

Upstream (prepare/01, sha256-pinned there): ``TH14_Temporal_annotations_{validation,test}.zip`` (one text file per
class, rows ``<video> <start s> <end s>``, plus ``Ambiguous_*.txt``) and ``Class Index_Detection.txt`` (the 20
detection classes). Rows: every video an annotation file names, the online-action-detection convention (200
validation videos for training, 213 test videos for evaluation); per video its file, its recording duration (the
video track's ``duration_s`` from yavih's header probe) and its intervals ``[[start, end, class], ...]`` with class
1 ... 20 in ``Class Index_Detection.txt`` order and 21 for ``Ambiguous``.

    python make_manifest.py --root <media_root>            # rewrite manifest.csv.gz
    python make_manifest.py --root <media_root> --check    # re-derive and compare byte for byte
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASS_INDEX_SHA256 = "d0a1711d537c6837c43ddbde943aaa8c648e524ed48dd86e6561b5d3d280e98f"
ANN_DIR = {"validation": "annotation", "test": "TH14_Temporal_Annotations_Test/annotations/annotation"}
AMBIGUOUS = 21
HEADER = ["clip_id", "split", "rel_media", "duration_s", "events"]


def classes(root: Path) -> list[str]:
    p = root / "annotations" / "Class_Index_Detection.txt"
    if hashlib.sha256(p.read_bytes()).hexdigest() != CLASS_INDEX_SHA256:
        sys.exit(f"{p}: not the pinned class index")
    return [line.split()[1] for line in p.read_text().splitlines() if line.strip()]


def events(root: Path, split: str, names: list[str]) -> dict[str, list]:
    index = {n: k + 1 for k, n in enumerate(names)}
    suffix = "val" if split == "validation" else "test"
    out: dict[str, list] = {}
    d = root / "annotations" / split / ANN_DIR[split]
    for name, c in [*index.items(), ("Ambiguous", AMBIGUOUS)]:
        for line in (d / f"{name}_{suffix}.txt").read_text().splitlines():
            if line.strip():
                vid, s, e = line.split()
                out.setdefault(vid, []).append([round(float(s), 4), round(float(e), 4), c])
    return out


def _duration(path: Path) -> float | None:
    from yavih.media.info import probe_info

    try:
        d = probe_info(str(path), "video", ext="mp4").get("duration_s")
    except Exception:  # noqa: BLE001 - an unreadable file is an exclusion
        return None
    return float(d) if d else None


def build(root: Path, workers: int = 8) -> tuple[bytes, dict]:
    names = classes(root)
    rows, excluded = [], Counter()
    for split in ("validation", "test"):
        ev = events(root, split, names)
        vids = sorted(ev)
        with ThreadPoolExecutor(workers) as ex:
            durs = list(ex.map(lambda v, s=split: _duration(root / "videos" / s / f"{v}.mp4"), vids))
        for vid, d in zip(vids, durs, strict=True):
            if d is None:
                excluded["missing_or_unreadable"] += 1
                continue
            rows.append([vid, split, f"videos/{split}/{vid}.mp4", f"{d:.6f}", json.dumps(sorted(ev[vid]), separators=(",", ":"))])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HEADER)
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    return data, {"rows": len(rows), "per_split": dict(Counter(r[1] for r in rows)), "excluded": dict(excluded),
                  "sha256": hashlib.sha256(data).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ns = ap.parse_args(argv)
    data, info = build(ns.root, ns.workers)
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
