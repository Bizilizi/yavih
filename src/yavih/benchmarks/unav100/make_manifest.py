#!/usr/bin/env python
"""Freeze the yavih ``unav100`` manifest from the official UnAV-100 annotations.

Inputs (all read-only, public, sha256-pinned; either one gives the same bytes):

* the HF release ``ttgeng233/UnAV-100`` at revision ``daeb88bc14eeff48f8a4ac2836b67f6c350f847b``:
  ``data/train.parquet`` (sha256 ``d8b1302f…``, 18 123 events), ``data/validation.parquet`` (``551e1781…``,
  6 036 events), ``data/test.parquet`` (``36a7d528…``, 5 900 events); columns ``video_id, duration, timestamp
  [start, end], event_label``, one row per event. Reading them needs ``pyarrow`` (a prepare-time tool);
* or the authors' ``unav100_annotations.json`` (``{"database": {ytid: {duration, subset, annotations: [{segment,
  label, label_id}]}}}``, sha256 ``2fdfc587…``, the Google Drive file the project page links): the same 10 790
  videos, 30 059 events, subsets, durations and event order (checked when it was pinned).

Derivation (no curation: every upstream video is listed, the official splits are kept): one row per video,
ordered by split (``train``, ``validation``, ``test``) then ``video_id`` (bytewise); ``clip_id`` = ``video_id``
(the YouTube id); ``labels`` = the video's distinct event classes, sorted, ``;``-joined (label kind ``multi``);
``rel_media`` = ``video/<ytid>.mp4`` (the key ``prepare/03_index_tar.py`` gives the tar member);
``duration_s`` = the annotated duration; ``events`` = every annotated event in upstream order as
``start:end:class_index`` (seconds as Python's shortest float repr, the class index of ``classes.csv``),
``;``-joined. ``classes.csv`` = ``index,name`` over the 100 sorted class names (the authors' ``label_id`` order is
not used: mAP averages over classes, so the order changes no number).

With a staged media root (``--media-root``, default the configured ``unav100.media_root``), presence per split is
read from ``<media_root>/index.json`` (the tar index of step 03); ``--no-media`` skips it (staged == listed).

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (gzip ``mtime=0``), ``classes.csv``, and the
``manifest.upstream`` block printed as JSON on stdout (paste into ``benchmark.yaml``).

Usage::

    python "$(yavih bench path unav100)/make_manifest.py"                       # media_root from the config
    python "$(yavih bench path unav100)/make_manifest.py" --check               # re-derive, compare, write nothing
    python "$(yavih bench path unav100)/make_manifest.py" --no-media --metadata DIR --check     # DIR/{train,validation,test}.parquet
    python "$(yavih bench path unav100)/make_manifest.py" --no-media --annotations-json unav100_annotations.json --check
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

HF_REPO = "ttgeng233/UnAV-100"
HF_REVISION = "daeb88bc14eeff48f8a4ac2836b67f6c350f847b"
PARQUET_SHA256 = {
    "train": "d8b1302fc12dc22a62ca23b592392cb000d00d2a487a70aa40876dde52742d55",
    "validation": "551e1781c8a8fd77cb7a1daf0ad9b71346c615864de7bba6fb5dba3565dadc6b",
    "test": "36a7d528772c1be736c105bfc65955255c822e99b7ceded64ef6e1b9c969df45",
}
JSON_SHA256 = "2fdfc587ad36fc405496b5520d967b5df54b2d82027d16d69e968b5cf04d3dba"
SPLITS = ("train", "validation", "test")
TOTAL_VIDEOS = 10790
TOTAL_EVENTS = 30059
VIDEOS_PER_SPLIT = {"train": 6489, "validation": 2134, "test": 2167}
NUM_CLASSES = 100

MANIFEST_COLUMNS = ["clip_id", "split", "labels", "rel_media", "duration_s", "events"]
CLASSES_COLUMNS = ["index", "name"]

#: one upstream event: (video_id, split, duration, start, end, label)
Event = tuple[str, str, float, float, float, str]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_sha(path: Path, want: str) -> str:
    got = sha256_file(path)
    if got != want:
        raise SystemExit(f"{path} has sha256 {got}, expected {want} (HF {HF_REPO} @ {HF_REVISION[:8]} / the authors' json)")
    return got


def read_parquets(meta_dir: Path) -> tuple[list[Event], dict[str, str]]:
    """Events from ``<meta_dir>/{train,validation,test}.parquet`` (sha256-pinned), file order."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("reading the HF parquets needs pyarrow (pip install pyarrow), or pass --annotations-json") from None
    events: list[Event] = []
    shas: dict[str, str] = {}
    for split in SPLITS:
        path = meta_dir / f"{split}.parquet"
        if not path.is_file():
            raise SystemExit(f"{path} not found (HF {HF_REPO} data/{split}.parquet)")
        shas[f"{HF_REPO}@{HF_REVISION[:8]}/data/{split}.parquet"] = _check_sha(path, PARQUET_SHA256[split])
        for r in pq.read_table(path).to_pylist():
            ts = list(r["timestamp"])
            if len(ts) != 2:
                raise SystemExit(f"{path}: {r['video_id']} timestamp {ts} is not [start, end]")
            events.append((str(r["video_id"]), split, float(r["duration"]), float(ts[0]), float(ts[1]), str(r["event_label"])))
    return events, shas


def read_json(path: Path) -> tuple[list[Event], dict[str, str]]:
    """Events from the authors' ``unav100_annotations.json`` (sha256-pinned), database order."""
    sha = _check_sha(path, JSON_SHA256)
    db = json.loads(path.read_text(encoding="utf-8"))["database"]
    events: list[Event] = []
    for vid, v in db.items():
        for a in v["annotations"]:
            events.append((str(vid), str(v["subset"]), float(v["duration"]), float(a["segment"][0]), float(a["segment"][1]), str(a["label"])))
    return events, {"unav100_annotations.json (authors' Google Drive file)": sha}


def derive(events: Iterable[Event]) -> tuple[list[list[str]], list[str], Counter]:
    """``(manifest rows, class names, videos per split)`` from upstream events (any order of videos; the event
    order within a video is kept)."""
    videos: dict[str, dict] = {}
    for vid, split, dur, s, e, label in events:
        v = videos.setdefault(vid, {"split": split, "duration": dur, "events": []})
        if v["split"] != split or v["duration"] != dur:
            raise SystemExit(f"{vid}: inconsistent split/duration across its events")
        if not (0.0 <= s < e <= dur + 1e-6):
            raise SystemExit(f"{vid}: event [{s}, {e}] outside [0, {dur}]")
        v["events"].append((s, e, label))
    names = sorted({label for v in videos.values() for _, _, label in v["events"]})
    index = {n: i for i, n in enumerate(names)}
    order = {s: k for k, s in enumerate(SPLITS)}
    unknown = sorted({v["split"] for v in videos.values()} - set(order))
    if unknown:
        raise SystemExit(f"unknown subsets {unknown}")
    rows = []
    for vid in sorted(videos, key=lambda k: (order[videos[k]["split"]], k)):
        v = videos[vid]
        rows.append([
            vid, v["split"], ";".join(sorted({label for _, _, label in v["events"]})), f"video/{vid}.mp4", repr(v["duration"]),
            ";".join(f"{s!r}:{e!r}:{index[label]}" for s, e, label in v["events"]),
        ])
    return rows, names, Counter(v["split"] for v in videos.values())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--media-root", type=Path, default=None,
                   help="the staged root (upstream/data/*.parquet, index.json); default the configured unav100.media_root")
    p.add_argument("--metadata", type=Path, default=None, help="directory holding {train,validation,test}.parquet (default <media-root>/upstream/data)")
    p.add_argument("--annotations-json", type=Path, default=None, help="the authors' unav100_annotations.json instead of the parquets")
    p.add_argument("--no-media", action="store_true", help="skip the presence check (re-freeze from the annotations alone)")
    p.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write (default: this folder)")
    p.add_argument("--check", action="store_true", help="re-derive and compare with the committed files; write nothing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = lambda m: print(f"[make_manifest unav100] {m}", file=sys.stderr)  # noqa: E731
    if args.no_media and args.metadata is None and args.annotations_json is None:
        raise SystemExit("--no-media needs --metadata DIR or --annotations-json FILE")
    media_root = None
    if not args.no_media:
        try:
            media_root = config.folder_root(HERE, "media_root", args.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root or configure unav100.media_root: {e}") from None

    # 1. the upstream annotations (sha256-pinned).
    if args.annotations_json is not None:
        events, upstream_files = read_json(args.annotations_json)
        source = "the authors' unav100_annotations.json (Google Drive)"
    else:
        events, upstream_files = read_parquets(args.metadata or (media_root / "upstream" / "data"))
        source = f"HF {HF_REPO} @ {HF_REVISION[:8]} data/{{train,validation,test}}.parquet"
    rows, names, per_split = derive(events)
    n_events = sum(1 for _ in events)
    log(f"{len(rows)} videos, {n_events} events, {len(names)} classes from {source}")

    # 2. invariants the protocol rests on (the paper's counts).
    if len(rows) != TOTAL_VIDEOS or n_events != TOTAL_EVENTS or len(names) != NUM_CLASSES or dict(per_split) != VIDEOS_PER_SPLIT:
        raise SystemExit(f"counts changed: {len(rows)} videos / {n_events} events / {len(names)} classes / {dict(per_split)}")

    # 3. presence from the tar index.
    missing: Counter = Counter()
    if args.no_media:
        log("--no-media: presence assumed (staged == listed)")
    else:
        index_path = media_root / "index.json"
        members = set(json.loads(index_path.read_text(encoding="utf-8"))["members"]) if index_path.is_file() else set()
        if not members:
            log(f"WARNING: {index_path} missing or empty (run prepare steps 01-03)")
        for r in rows:
            if r[3] not in members:
                missing[r[1]] += 1
        log(f"presence: {sum(missing.values())} missing {dict(missing)}")

    # 4. manifest.csv.gz + classes.csv (fixed column and row order).
    manifest_data = csv_bytes(MANIFEST_COLUMNS, rows)
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), n] for i, n in enumerate(names)])
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    classes_sha = hashlib.sha256(classes_data).hexdigest()
    if args.check:
        same_m = gzip.decompress((HERE / "manifest.csv.gz").read_bytes()) == manifest_data
        same_c = (HERE / "classes.csv").read_bytes() == classes_data
        log(f"--check: manifest.csv.gz {'identical' if same_m else 'DIFFERS'}, classes.csv {'identical' if same_c else 'DIFFERS'}")
        if not (same_m and same_c):
            return 1
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        cm = write_if_changed(args.out / "manifest.csv.gz", gzip_bytes(manifest_data))
        cc = write_if_changed(args.out / "classes.csv", classes_data)
        log(f"manifest.csv.gz {'written' if cm else 'unchanged'} ({len(rows)} rows); classes.csv {'written' if cc else 'unchanged'}")

    # 5. the manifest.upstream block.
    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(rows)},
        "upstream": {
            "source": f"HF {HF_REPO} @ {HF_REVISION[:8]} data/{{train,validation,test}}.parquet (= the authors' unav100_annotations.json)",
            "sha256": PARQUET_SHA256["test"],
            "rows": TOTAL_VIDEOS,
            "upstream_files": {
                **{f"{HF_REPO}@{HF_REVISION[:8]}/data/{s}.parquet": PARQUET_SHA256[s] for s in SPLITS},
                "unav100_annotations.json (authors' Google Drive file)": JSON_SHA256,
            },
            "curation": {
                "total_upstream": TOTAL_VIDEOS,
                "per_split": {s: {"listed": per_split[s], "staged": per_split[s] - missing[s], "missing": missing[s]} for s in SPLITS},
                "excluded": {},
            },
        },
        "derived": {"events": n_events, "events_per_split": dict(Counter(ev[1] for ev in events))},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
