"""Freeze ave_causal's manifest from the frozen ave manifest: one row per video, its ten one-second labels.

The upstream is the packaged ``ave`` folder's ``manifest.csv.gz`` (sha256-pinned below): one row per annotated event,
leakage-filtered against VGGSound train and across AVE splits (see ``ave``'s README). This script keeps the train and
test rows, groups them by video, and writes per video the label of each second ``s = 0 ... 9``: the event's class
index where ``event_start <= s < event_end``, else ``background`` (28). The 16 train videos that carry two events over
the same seconds (two classes for one second: not a single-label target) are excluded. ``val`` stays unused, as in ave.

    python make_manifest.py            # rewrite manifest.csv.gz (gzip mtime 0) and print the manifest.upstream block
    python make_manifest.py --check    # re-derive and compare byte for byte
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
AVE_MANIFEST_SHA256 = "4ff303259694f0f7537438ac017976defe3ebdde92786b4616a46d5a22bb5911"  # ave/manifest.csv.gz, decompressed
AVE_CLASSES_SHA256 = "e664fd38da87b56f94114fe67455c521da3a126ae29e1d0b390c478b464295b6"
SECONDS = 10
BACKGROUND = 28
HEADER = ["clip_id", "split", "rel_media", "seconds"]


def _ave_folder() -> Path:
    try:
        from yavih.benchmark import resolve_folder

        return Path(resolve_folder("ave"))
    except Exception:  # noqa: BLE001 - outside an installed yavih: the sibling folder
        return HERE.parent / "ave"


def build(ave: Path) -> tuple[bytes, dict]:
    raw = gzip.decompress((ave / "manifest.csv.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != AVE_MANIFEST_SHA256:
        sys.exit(f"{ave}/manifest.csv.gz is not the pinned ave manifest (sha256 {AVE_MANIFEST_SHA256})")
    classes = (ave / "classes.csv").read_bytes()
    if hashlib.sha256(classes).hexdigest() != AVE_CLASSES_SHA256:
        sys.exit(f"{ave}/classes.csv is not the pinned ave class list")
    index = {r["name"]: int(r["index"]) for r in csv.DictReader(io.StringIO(classes.decode("utf-8")))}
    by_video: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        if r["split"] in ("train", "test"):
            by_video[(r["split"], r["rel_media"])].append(r)
    rows, excluded = [], 0
    for (split, rel), events in sorted(by_video.items(), key=lambda kv: (kv[0][0] != "train", kv[0][1])):
        if len(events) > 1:
            excluded += 1
            continue
        e = events[0]
        lab = [index[e["label"]] if int(e["event_start"]) <= s < int(e["event_end"]) else BACKGROUND for s in range(SECONDS)]
        rows.append([Path(rel).stem, split, rel, ";".join(str(x) for x in lab)])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(HEADER)
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    counts = {s: sum(1 for r in rows if r[1] == s) for s in ("train", "test")}
    upstream = {"source": "the packaged ave folder's manifest.csv.gz (one row per event), grouped by video",
                "sha256": AVE_MANIFEST_SHA256, "rows": len(rows),
                "note": f"train and test videos of ave; {excluded} train videos with two events over the same seconds excluded",
                "excluded": {"two_events_same_seconds": excluded}, "counts": counts}
    return data, upstream


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ave", type=Path, default=None, help="the ave folder (default: resolved by yavih)")
    ns = ap.parse_args(argv)
    data, upstream = build(ns.ave or _ave_folder())
    out = HERE / "manifest.csv.gz"
    if ns.check:
        have = gzip.decompress(out.read_bytes())
        ok = have == data
        print(f"manifest {'matches' if ok else 'DIFFERS'} ({hashlib.sha256(data).hexdigest()})")
        return 0 if ok else 1
    out.write_bytes(gzip.compress(data, mtime=0))
    print(f"wrote {out} sha256 {hashlib.sha256(data).hexdigest()}")
    print(upstream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
