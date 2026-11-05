#!/usr/bin/env python
"""Freeze the yavih ``ave`` manifest from the public AVE annotation and split files.

Inputs (all read-only):

* ``<media_root>/upstream/Annotations.txt``: the HuggingFace-root copy (4 143 rows, 28 categories; NOT the stale
  2 757-row copy inside the tar);
* ``<media_root>/upstream/{train,val,test}_order.h5``: the official AVE-ECCV18 splits (``order`` indexes
  Annotations.txt rows, 3 339 / 402 / 402);
* ``<media_root>/index.json``: the tar member set written by ``prepare/03_index_tar.py`` (pinned by the sha256 of its
  sorted ``{member: [offset, size]}`` map, so any shard path reproduces it);
* VGGSound ``metadata/vggsound.csv`` (the official metadata; its ``train`` ytids are a leakage source).

Rules: one row per Annotations.txt event whose ``video/<ytid>.mp4`` is a tar member, in Annotations.txt order,
``clip_id = <ytid>#<row>``; drop every row whose ytid is in VGGSound train (all splits), then every val/test row
whose ytid also appears in AVE train (AVE's official splits are not video-disjoint). There is NO train draw: AVE is
uncapped (``protocol.train_draw: null``); all 2 624 train rows are used.

Outputs:

* ``manifest.csv.gz`` with columns ``clip_id,split,label,rel_media,event_start,event_end`` (val rows kept,
  ``splits.unused``), gzip ``mtime=0``;
* ``classes.csv`` (``index,name``): the 28 sorted categories, ``background`` at index 28;
* on stdout a JSON report with the hashes and the ``manifest.upstream`` block (curation sums, upstream file
  sha256s) for ``benchmark.yaml``; ``tests/test_folder.py`` re-checks the sums.

Re-runs are byte-identical. ``curation.short_clips`` is the value measured by ``prepare/04_media_census.py``
(``SHORT_CLIPS``); pass ``--short-clips`` after re-running the census.

Usage::

    python "$(yavih bench path ave)/make_manifest.py" --check \
        --media-root "$(yavih config get ave.media_root)" \
        --vggsound-csv "$(yavih config get datasets.vggsound)/metadata/vggsound.csv"

Every flag defaults to the same yavih config resolution (the folder's ``media_root``, ``[datasets] vggsound``).
``--check`` writes nothing and exits 1 if the committed files differ; ``--out DIR`` writes the files elsewhere.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- pinned upstream facts
# Public upstream files (sha256).
ANNOTATIONS_SHA256 = "b0aabcea3281abb4c884e19ad4ff90314b199bea4742efe8556585a18f1396e7"
SPLIT_ORDER_SHA256 = {
    "train": "a216fcdd02782ba9d10c9ea79116120eb3be467740e58290e1b340d42a590456",
    "val": "5692aec0efb30e4efa49a867a902c1993b4c88a7d3ec19879349e540e7a5d9ca",
    "test": "f9bb79defc1f784ca5dda9cdd3ee52af6d24ccb399a1cfab6bf0d373f5681076",
}
# sha256 of the sorted {member: [offset, size]} map of AVE_Datset.tar (``member_set_sha256``): independent of the
# shard path stored in index.json.
INDEX_MEMBERS_SHA256 = "988ac22660af1f4235d61818f005bd22926db301a64135fa8f4ca626a7f06726"
VGGSOUND_CSV_SHA256 = "ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8"

TOTAL_UPSTREAM = 4143
EXPECTED_SPLIT_ROWS = {"train": 3339, "val": 402, "test": 402}
EXPECTED_CATEGORIES = 28
EXPECTED_YTIDS = 4097
NUM_SEGMENTS = 10
BACKGROUND = "background"
BACKGROUND_INDEX = 28
# Post-exclusion populations.
EXPECTED_ROWS = {"train": 2624, "val": 308, "test": 316}
EXPECTED_LEAK = {"vggsound_train": 886, "ave_cross_split": 9}
EXPECTED_TEST_SHORT_EVENT_ROWS = 104
# prepare/04_media_census.py: 74 rows have a stream shorter than the nominal 10 s (audio 68, 9.73–9.99 s;
# video 6, 9.96–9.99 s).
SHORT_CLIPS = 74

SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = ["clip_id", "split", "label", "rel_media", "event_start", "event_end"]
CLASSES_COLUMNS = ["index", "name"]
UPSTREAM_SOURCE = ("AVE Annotations.txt (HF root) split by AVE-ECCV18 {train,val,test}_order.h5; "
                   "leakage-filtered against VGGSound metadata/vggsound.csv")
UPSTREAM_KEYS = {
    "annotations": "AVE/Annotations.txt (HF root)",
    "index": "AVE_Datset.tar members (index.json, sorted {member: [offset, size]})",
    "vggsound": "vggsound/metadata/vggsound.csv",
    **{s: f"AVE-ECCV18/data/{s}_order.h5" for s in SPLITS},
}


def default_media_root() -> Path | None:
    """The folder's ``media_root`` via yavih's resolver (``YAVIH_DATA__AVE__MEDIA_ROOT``, config ``[data.ave]
    media_root``, else the yaml default ``${data.root}/ave``); ``None`` when unresolvable."""
    try:
        return config.folder_root(HERE, "media_root")
    except config.ConfigError:
        return None


def default_vggsound_csv() -> Path | None:
    """``$([datasets] vggsound)/metadata/vggsound.csv`` (default ``${data.root}/vggsound``); ``None`` when unresolvable."""
    try:
        root = config.dataset("vggsound")
    except config.ConfigError:
        return None
    return Path(root) / "metadata" / "vggsound.csv" if root else None


def log(msg: str) -> None:
    print(f"[make_manifest ave] {msg}", file=sys.stderr)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def member_set_sha256(members: dict[str, list[int]]) -> str:
    """sha256 of ``{member: [offset, size]}`` (sorted keys, compact JSON): the tar's member set, without the shard."""
    canon = {k: [int(v[-2]), int(v[-1])] for k, v in members.items()}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def clip_ytid(clip_id: str) -> str:
    """``<ytid>#<upstream row>`` → ``<ytid>`` (split on the LAST ``#``)."""
    return clip_id.rsplit("#", 1)[0]


# ----------------------------------------------------------------------------- row derivation from the public files


def read_annotations(path: Path) -> list[dict]:
    """HF-root Annotations.txt (``Category&VideoID&Quality&StartTime&EndTime``), guarded on rows/categories/ytids."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n")
        if header.split("&") != ["Category", "VideoID", "Quality", "StartTime", "EndTime"]:
            raise SystemExit(f"unexpected Annotations.txt header: {header!r}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = [p.strip() for p in line.split("&")]
            if len(parts) != 5:
                raise SystemExit(f"malformed Annotations.txt row: {line!r}")
            category, ytid, _quality, start, end = parts
            rows.append({"row": len(rows), "category": category, "ytid": ytid, "event_start": int(start), "event_end": int(end)})
    if len(rows) != TOTAL_UPSTREAM:
        raise SystemExit(
            f"{path}: {len(rows)} rows, expected {TOTAL_UPSTREAM} — the copy bundled inside AVE_Datset.tar is a "
            "stale 2 757-row / 19-category subset; use the HuggingFace-root Annotations.txt (setup.md, Obtain the data)"
        )
    if len({r["category"] for r in rows}) != EXPECTED_CATEGORIES or len({r["ytid"] for r in rows}) != EXPECTED_YTIDS:
        raise SystemExit(f"{path}: category/ytid counts differ from 28 / 4 097")
    for r in rows:
        if not 0 <= r["event_start"] <= r["event_end"] <= NUM_SEGMENTS:
            raise SystemExit(f"row {r['row']}: event range out of [0, 10]")
    return rows


def read_splits(upstream: Path) -> dict[int, str]:
    """The three ``order`` arrays (int64, Annotations.txt row indices) partition range(4143)."""
    import h5py  # freeze-time only

    split_of: dict[int, str] = {}
    seen: list[int] = []
    for split in SPLITS:
        with h5py.File(upstream / f"{split}_order.h5", "r") as fh:
            order = [int(v) for v in fh["order"][()]]
        if len(order) != EXPECTED_SPLIT_ROWS[split] or len(set(order)) != len(order):
            raise SystemExit(f"{split}_order.h5: {len(order)} rows (expected {EXPECTED_SPLIT_ROWS[split]}) or duplicates")
        for i in order:
            split_of[i] = split
        seen.extend(order)
    if sorted(seen) != list(range(TOTAL_UPSTREAM)):
        raise SystemExit("the three order.h5 files do not partition range(4143)")
    return split_of


def load_vggsound_train_ids(path: Path) -> set[str]:
    """Official vggsound.csv (no header; ``ytid,start,label,split``): the ytids with split == train."""
    ids: set[str] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 4 and row[3].strip() == "train":
                ids.add(row[0].strip())
    return ids


def rederive(media_root: Path, vggsound_csv: Path) -> tuple[list[dict[str, str]], list[str], dict[str, int], dict[str, str], dict[str, dict[str, int]]]:
    """Rows, class names, exclusion counts, upstream file sha256s and per-split exclusions from the public files."""
    upstream = media_root / "upstream"
    ann = read_annotations(upstream / "Annotations.txt")
    split_of = read_splits(upstream)
    index_members = json.loads((media_root / "index.json").read_text(encoding="utf-8"))["members"]
    members = set(index_members)
    staged: list[dict[str, str]] = []
    missing = 0
    for a in ann:
        rel = f"video/{a['ytid']}.mp4"
        if rel not in members:
            missing += 1
            continue
        staged.append({"clip_id": f"{a['ytid']}#{a['row']}", "label": a["category"], "split": split_of[a["row"]],
                       "rel_media": rel, "event_start": str(a["event_start"]), "event_end": str(a["event_end"])})
    # VGGSound-train leaks first (all splits), then eval rows of videos AVE train also holds (train keeps its copy).
    vgg_train = load_vggsound_train_ids(vggsound_csv)
    ave_train = {clip_ytid(r["clip_id"]) for r in staged if r["split"] == "train"}
    kept: list[dict[str, str]] = []
    counts = {"vggsound_train": 0, "ave_cross_split": 0}
    per_split = {s: {"vggsound_train": 0, "ave_cross_split": 0} for s in SPLITS}
    for r in staged:
        y = clip_ytid(r["clip_id"])
        if y in vgg_train:
            counts["vggsound_train"] += 1
            per_split[r["split"]]["vggsound_train"] += 1
            continue
        if r["split"] in ("val", "test") and y in ave_train:
            counts["ave_cross_split"] += 1
            per_split[r["split"]]["ave_cross_split"] += 1
            continue
        kept.append(r)
    counts["skipped_missing"] = missing
    names = sorted({a["category"] for a in ann}) + [BACKGROUND]
    files = {
        UPSTREAM_KEYS["annotations"]: sha256_file(upstream / "Annotations.txt"),
        **{UPSTREAM_KEYS[s]: sha256_file(upstream / f"{s}_order.h5") for s in SPLITS},
        UPSTREAM_KEYS["index"]: member_set_sha256(index_members),
        UPSTREAM_KEYS["vggsound"]: sha256_file(vggsound_csv),
    }
    return kept, names, counts, files, per_split


def check_rows(rows: list[dict[str, str]], names: list[str]) -> collections.Counter:
    """Invariants of the derived rows; returns the short-event rows per split."""
    by_split = collections.Counter(r["split"] for r in rows)
    if dict(by_split) != EXPECTED_ROWS:
        raise SystemExit(f"rows by split {dict(by_split)} != {EXPECTED_ROWS}")
    if len({r["clip_id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate clip_id")
    if len(names) != EXPECTED_CATEGORIES + 1 or names[BACKGROUND_INDEX] != BACKGROUND or BACKGROUND in names[:BACKGROUND_INDEX]:
        raise SystemExit("classes: expected 28 categories with background pinned at index 28")
    label_set = set(names[:BACKGROUND_INDEX])
    for r in rows:
        if r["label"] not in label_set:
            raise SystemExit(f"{r['clip_id']}: category {r['label']!r} not in the 28 categories")
        a, b = int(r["event_start"]), int(r["event_end"])
        if not 0 <= a < b <= NUM_SEGMENTS:  # the seq readout needs a non-empty event
            raise SystemExit(f"{r['clip_id']}: event [{a}, {b}) is empty or outside [0, 10]")
    short_event = collections.Counter(r["split"] for r in rows if int(r["event_end"]) - int(r["event_start"]) < NUM_SEGMENTS)
    if short_event["test"] != EXPECTED_TEST_SHORT_EVENT_ROWS:
        raise SystemExit(f"short-event test rows {short_event['test']} != {EXPECTED_TEST_SHORT_EVENT_ROWS}")
    train_y = {clip_ytid(r["clip_id"]) for r in rows if r["split"] == "train"}
    for s in ("val", "test"):
        if train_y & {clip_ytid(r["clip_id"]) for r in rows if r["split"] == s}:
            raise SystemExit(f"AVE train/{s} ytid overlap survived the leakage filter")
    return short_event


# ----------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze the yavih ave manifest from the public AVE annotation and split files")
    p.add_argument("--media-root", type=Path, default=None, help="staged AVE root (upstream/, index.json); default: the folder's media_root")
    p.add_argument("--vggsound-csv", type=Path, default=None,
                   help="official VGGSound metadata/vggsound.csv; default $([datasets] vggsound)/metadata/vggsound.csv")
    p.add_argument("--short-clips", type=int, default=SHORT_CLIPS, help="curation.short_clips (from prepare/04_media_census.py)")
    p.add_argument("--no-pins", action="store_true", help="skip the upstream sha256 pins (for a synthetic fixture)")
    p.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write or check (default: this folder)")
    p.add_argument("--check", action="store_true", help="write nothing; exit 1 if the files under --out differ from a re-freeze")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    media_root = args.media_root or default_media_root()
    vggsound_csv = args.vggsound_csv or default_vggsound_csv()
    if media_root is None or not (media_root / "upstream" / "Annotations.txt").is_file() or not (media_root / "index.json").is_file():
        raise SystemExit(f"upstream/Annotations.txt or index.json not found under {media_root} (pass --media-root; setup.md, Obtain the data)")
    if vggsound_csv is None or not vggsound_csv.is_file():
        raise SystemExit(f"VGGSound metadata {vggsound_csv} not found (pass --vggsound-csv)")

    rows, names, counts, files, per_split_leak = rederive(media_root, vggsound_csv)
    if not args.no_pins:
        pinned = {UPSTREAM_KEYS["annotations"]: ANNOTATIONS_SHA256, UPSTREAM_KEYS["index"]: INDEX_MEMBERS_SHA256,
                  UPSTREAM_KEYS["vggsound"]: VGGSOUND_CSV_SHA256, **{UPSTREAM_KEYS[s]: SPLIT_ORDER_SHA256[s] for s in SPLITS}}
        for k, v in files.items():
            if v != pinned[k]:
                raise SystemExit(f"upstream file {k} has sha256 {v}, expected {pinned[k]}")
        want = {**EXPECTED_LEAK, "skipped_missing": 0}
        if counts != want:
            raise SystemExit(f"exclusion counts {counts} != {want}")
        short_event = check_rows(rows, names)
    else:
        short_event = collections.Counter(r["split"] for r in rows if int(r["event_end"]) - int(r["event_start"]) < NUM_SEGMENTS)
    log(f"{len(rows)} rows from {TOTAL_UPSTREAM} upstream events ({counts['vggsound_train']} VGGSound-train + "
        f"{counts['ave_cross_split']} cross-split rows excluded, {counts['skipped_missing']} missing from the tar)")

    manifest_rows = [[r[c] for c in MANIFEST_COLUMNS] for r in rows]
    manifest_data = csv_bytes(MANIFEST_COLUMNS, manifest_rows)
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), n] for i, n in enumerate(names)])
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    classes_sha = hashlib.sha256(classes_data).hexdigest()
    out: Path = args.out
    if args.check:
        same_m = (out / "manifest.csv.gz").is_file() and (out / "manifest.csv.gz").read_bytes() == gzip_bytes(manifest_data)
        same_c = (out / "classes.csv").is_file() and (out / "classes.csv").read_bytes() == classes_data
        log(f"--check: manifest.csv.gz {'identical' if same_m else 'DIFFERS'}, classes.csv {'identical' if same_c else 'DIFFERS'}")
        if not (same_m and same_c):
            return 1
    else:
        out.mkdir(parents=True, exist_ok=True)
        cm = write_if_changed(out / "manifest.csv.gz", gzip_bytes(manifest_data))
        cc = write_if_changed(out / "classes.csv", classes_data)
        log(f"manifest.csv.gz {'written' if cm else 'unchanged'} ({len(manifest_rows)} rows, sha256 {manifest_sha[:16]}…); classes.csv {'written' if cc else 'unchanged'}")

    # Curation (manifest.upstream.curation; invariant sum(listed) + sum(excluded) == total_upstream).
    by_split = collections.Counter(r["split"] for r in rows)
    per_split = {s: {"listed": by_split[s], "staged": by_split[s], "missing": 0} for s in SPLITS}
    excluded = {"leakage_vggsound_train": counts["vggsound_train"], "leakage_ave_cross_split": counts["ave_cross_split"]}
    curation = {
        "total_upstream": TOTAL_UPSTREAM,
        "per_split": per_split,
        "excluded": excluded,
        "short_clips": args.short_clips,
        "leakage_audit": {
            "against": "VGGSound train ytids (metadata/vggsound.csv split==train, all AVE splits) + AVE train ytids (val/test rows)",
            "excluded": sum(excluded.values()),
        },
    }
    shared = collections.Counter(r["rel_media"] for r in rows)
    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(manifest_rows), "rows_by_split": dict(by_split)},
        "upstream": {
            "source": UPSTREAM_SOURCE,
            "sha256": files[UPSTREAM_KEYS["annotations"]],
            "rows": len(rows),
            "upstream_files": files,
            "curation": curation,
        },
        "derived": {
            "official_split_rows": dict(EXPECTED_SPLIT_ROWS),
            "excluded_per_split": per_split_leak,
            "videos_by_split": {s: len({clip_ytid(r["clip_id"]) for r in rows if r["split"] == s}) for s in SPLITS},
            "rows_sharing_media_by_split": {s: sum(1 for r in rows if r["split"] == s and shared[r["rel_media"]] > 1) for s in SPLITS},
            "short_event_rows_by_split": {s: short_event[s] for s in SPLITS},
            "full_clip_event_rows": sum(1 for r in rows if int(r["event_end"]) - int(r["event_start"]) == NUM_SEGMENTS),
            "distinct_videos": len(shared),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
