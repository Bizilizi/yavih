"""Freeze unav100_causal's manifest from the pinned Hugging Face revision of UnAV-100.

Upstream: ``ttgeng233/UnAV-100`` at the revision in ``benchmark.yaml`` (``data.roots.media_root.hf``): the annotation
parquets ``data/{train,validation,test}.parquet`` (one row per event: ``video_id``, ``duration``, ``timestamp``
``[start, end]`` in seconds, ``event_label``; identical to the official ``unav100_annotations.json``) and the videos
extracted from ``unav_raw_videos.tar.gz.part_*`` into ``raw_videos/<video_id>.mp4`` (prepare/02).

Per video the manifest keeps its split, its file, its recording duration (the video track's ``duration_s`` from
yavih's own header probe, the duration the stream helpers schedule on) and its events ``[[start, end, class], ...]``
(class index in ``event_classes.csv``, the official ``label_id`` order). Excluded: a video whose recording duration differs
from the annotated one by more than ``MAX_DURATION_GAP_S`` (the annotation grid does not match the file) or whose file
is missing or unreadable.

    python make_manifest.py --root <media_root>            # rewrite manifest.csv.gz; print the exclusions
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
PARQUET_SHA256 = {
    "train": "d8b1302fc12dc22a62ca23b592392cb000d00d2a487a70aa40876dde52742d55",
    "validation": "551e1781c8a8fd77cb7a1daf0ad9b71346c615864de7bba6fb5dba3565dadc6b",
    "test": "36a7d528772c1be736c105bfc65955255c822e99b7ceded64ef6e1b9c969df45",
}
MAX_DURATION_GAP_S = 1.0
HEADER = ["clip_id", "split", "rel_media", "duration_s", "events"]


def _classes() -> dict[str, int]:
    with open(HERE / "event_classes.csv", newline="", encoding="utf-8") as f:
        return {r["name"]: int(r["index"]) for r in csv.DictReader(f)}


def _events(root: Path) -> dict[str, dict]:
    import pyarrow.parquet as pq

    out: dict[str, dict] = {}
    for split, sha in PARQUET_SHA256.items():
        p = root / "data" / f"{split}.parquet"
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            sys.exit(f"{p}: not the pinned annotation file (sha256 {sha})")
        for r in pq.read_table(p).to_pylist():
            v = out.setdefault(r["video_id"], {"split": split, "duration": float(r["duration"]), "events": []})
            s, e = r["timestamp"]
            v["events"].append((float(s), float(e), r["event_label"]))
    return out


def _duration(path: Path) -> float | None:
    from yavih.media.info import probe_info

    try:
        info = probe_info(str(path), "video", ext="mp4")  # header only
    except Exception:  # noqa: BLE001 - an unreadable file is an exclusion, named below
        return None
    d = info.get("duration_s")
    return float(d) if d else None


def build(root: Path, workers: int = 16) -> tuple[bytes, dict]:
    cls = _classes()
    videos = _events(root)
    ids = sorted(videos)
    with ThreadPoolExecutor(workers) as ex:
        durs = dict(zip(ids, ex.map(lambda v: _duration(root / "raw_videos" / f"{v}.mp4"), ids), strict=True))
    excluded: Counter = Counter()
    rows = []
    order = {"train": 0, "validation": 1, "test": 2}
    for vid in sorted(ids, key=lambda v: (order[videos[v]["split"]], v)):
        v, d = videos[vid], durs[vid]
        if d is None:
            excluded["missing_or_unreadable"] += 1
            continue
        if abs(d - v["duration"]) > MAX_DURATION_GAP_S:
            excluded["duration_mismatch"] += 1
            continue
        ev = sorted((round(s, 4), round(e, 4), cls[lab]) for s, e, lab in v["events"])
        rows.append([vid, v["split"], f"raw_videos/{vid}.mp4", f"{d:.6f}", json.dumps([list(x) for x in ev], separators=(",", ":"))])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HEADER)
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    counts = Counter(r[1] for r in rows)
    return data, {"rows": len(rows), "per_split": dict(counts), "excluded": dict(excluded), "upstream_videos": len(ids),
                  "sha256": hashlib.sha256(data).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="the media root: the HF snapshot (data/*.parquet) + raw_videos/")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
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
