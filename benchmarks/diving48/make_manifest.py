#!/usr/bin/env python
"""Freeze the yavih ``diving48`` manifest from the official Diving48 V2 annotation files.

Inputs (all read-only; each sha256 asserted before anything is written), from the official Diving48 page
http://www.svcl.ucsd.edu/projects/resound/dataset.html (Nextcloud share links; the V2 files of the
2020-10-30 update, the vocabulary of the 2018 release):

* ``Diving48_V2_train.json`` (1 586 934 B, sha256 ``62bcff15…``): 15 027 ``{vid_name, label, start_frame,
  end_frame}`` entries;
* ``Diving48_V2_test.json`` (207 545 B, sha256 ``658fe0d8…``): 1 970 entries of the same shape;
* ``Diving48_vocab.json`` (3 094 B, sha256 ``cc423890…``): the 48 classes, each ``[takeoff, somersaults,
  twists, position]`` (e.g. ``["Back", "15som", "05Twis", "FREE"]``), list position = label index;
* with media, the staged ``<media_root>/rgb/`` (the ``rgb/`` directory of ``Diving48_rgb.tar.gz``) for
  presence only (one ``os.scandir``).

The V2 files are the protocol because VideoGLUE (Yuan et al., the reference suite; ``tensorflow/models``
``official/projects/videoglue/configs/yaml/vmae/frozen/vc_vmae_vit3d_diving48.yaml``: 48 classes,
``num_examples: 15027``, ``validation_steps: 122  # 1970 // 16``) trains and scores on them. The V1 files
(``Diving48_train.json`` 15 943 / ``Diving48_test.json`` 2 096 rows) are not read.

Each entry becomes one row, file order, train file first: ``clip_id`` = ``vid_name`` (YouTube id + ``_`` +
segment number; ids may start with ``-``), ``rel_media`` = ``rgb/<vid_name>.mp4``, ``label`` = the vocabulary
entry joined with single spaces (``Back 15som 05Twis FREE``), ``split`` = ``train`` / ``test``, and
``end_frame`` (the annotation's frame count; ``role: extra``). Every upstream ``start_frame`` is 0 and is not
kept (asserted). ``classes.csv`` is ``index,name`` in vocabulary order (``index`` = the upstream label id).

Curation: none. All 16 997 V2 rows are listed, nothing is excluded, the train split is not capped
(``protocol.train_draw: null``; VideoGLUE trains on all 15 027). Class 30 (``Forward Dive NoTwis STR``)
has no row in either V2 file (it has rows in V1) and is kept in ``classes.csv``: the label space is the
48-way vocabulary, as in VideoGLUE (``num_classes: 48``). Asserted: no clip in both splits, no duplicate
``vid_name``, 47 classes present in each split.

Outputs, byte-identical on re-run (``yavih.freeze``: gzip ``mtime=0``, no file name, ``\\n`` line ends):
``manifest.csv.gz`` (``clip_id,rel_media,label,split,end_frame``) and ``classes.csv``; the
``manifest.upstream`` block (``source``, ``sha256``, ``rows``, ``upstream_files``, ``curation``) and the
manifest digests are printed as JSON on stdout for ``benchmark.yaml``.

Usage::

    python "$(yavih bench path diving48)/make_manifest.py" --media-root "$(yavih config get diving48.media_root)"
    python "$(yavih bench path diving48)/make_manifest.py" --check     # re-derive, compare, write nothing
    python "$(yavih bench path diving48)/make_manifest.py" --no-media --metadata DIR --check

``--media-root`` defaults to the configured ``[data.diving48] media_root``
(``YAVIH_DATA__DIVING48__MEDIA_ROOT``); the three JSON files are read from ``--metadata DIR`` (default: the
media root). ``--no-media`` re-freezes from the JSON files alone (no presence check). ``--check`` exits 1 when
the re-derived ``manifest.csv.gz`` or ``classes.csv`` differs from the committed files. Writing rewrites the
folder, so run it in a git checkout of https://github.com/Bizilizi/yavih; ``yavih bench prepare diving48 03``
runs it with the checks, then ``yavih bench verify``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

# Pinned upstream files (official page, Nextcloud shares; see setup.md "Obtain the data").
UPSTREAM_FILES = {
    "Diving48_V2_train.json": "62bcff1537b24162b1c9be49f376b8f9b6fccd222734f10d4a5d545b38c00e93",
    "Diving48_V2_test.json": "658fe0d84ebea61414548bb69a996d668d9f249a82919e276ffc0f424d9371a3",
    "Diving48_vocab.json": "cc4238904e3ab274ae3ab36d26a09afd41e8c7891aa1f7c773d43fecb9388696",
}
SPLIT_FILES = (("train", "Diving48_V2_train.json"), ("test", "Diving48_V2_test.json"))
UPSTREAM_ROWS = {"train": 15027, "test": 1970}
TOTAL_UPSTREAM = sum(UPSTREAM_ROWS.values())
NUM_CLASSES = 48
ABSENT_CLASSES = (30,)  # no row in either V2 file
MEDIA_DIR = "rgb"
UPSTREAM_SOURCE = "official Diving48 V2 annotations (2020-10-30 update): Diving48_V2_{train,test}.json + Diving48_vocab.json"

MANIFEST_COLUMNS = ["clip_id", "rel_media", "label", "split", "end_frame"]
CLASSES_COLUMNS = ["index", "name"]
ENTRY_KEYS = {"vid_name", "label", "start_frame", "end_frame"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def class_name(entry: list[str]) -> str:
    """``["Back", "15som", "05Twis", "FREE"]`` -> ``"Back 15som 05Twis FREE"``."""
    if len(entry) != 4 or not all(isinstance(p, str) and p and " " not in p for p in entry):
        raise SystemExit(f"vocabulary entry {entry!r} is not four space-free strings")
    return " ".join(entry)


def read_inputs(metadata: Path) -> dict[str, bytes]:
    blobs = {}
    for name, want in UPSTREAM_FILES.items():
        p = metadata / name
        if not p.is_file():
            raise SystemExit(f"{p} not found: pass --metadata <dir holding {', '.join(UPSTREAM_FILES)}>")
        blobs[name] = p.read_bytes()
        got = sha256_bytes(blobs[name])
        if got != want:
            raise SystemExit(f"{p} has sha256 {got}, expected {want}; refusing to freeze from a different file")
    return blobs


def derive(blobs: dict[str, bytes]) -> dict:
    """Rows and classes from the parsed upstream files (file order), with the invariants asserted."""
    vocab = json.loads(blobs["Diving48_vocab.json"])
    names = [class_name(v) for v in vocab]
    if len(names) != NUM_CLASSES or len(set(names)) != NUM_CLASSES:
        raise SystemExit(f"vocabulary has {len(names)} entries ({len(set(names))} distinct), expected {NUM_CLASSES}")
    rows: list[list[str]] = []
    per_split_classes: dict[str, Counter] = {}
    for split, fname in SPLIT_FILES:
        entries = json.loads(blobs[fname])
        if len(entries) != UPSTREAM_ROWS[split]:
            raise SystemExit(f"{fname}: {len(entries)} entries, expected {UPSTREAM_ROWS[split]}")
        counts: Counter = Counter()
        for e in entries:
            if set(e) != ENTRY_KEYS:
                raise SystemExit(f"{fname}: entry keys {sorted(e)} != {sorted(ENTRY_KEYS)}")
            if e["start_frame"] != 0:
                raise SystemExit(f"{fname}: {e['vid_name']} has start_frame {e['start_frame']} (every V2 entry is 0)")
            lab = int(e["label"])
            if not 0 <= lab < NUM_CLASSES:
                raise SystemExit(f"{fname}: {e['vid_name']} has label {lab} outside 0..{NUM_CLASSES - 1}")
            counts[lab] += 1
            vid = str(e["vid_name"])
            rows.append([vid, f"{MEDIA_DIR}/{vid}.mp4", names[lab], split, str(int(e["end_frame"]))])
        per_split_classes[split] = counts
        absent = tuple(i for i in range(NUM_CLASSES) if i not in counts)
        if absent != ABSENT_CLASSES:
            raise SystemExit(f"{fname}: classes without rows {absent}, expected {ABSENT_CLASSES}")
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("a vid_name appears twice across the V2 train and test files")
    return {"rows": rows, "names": names, "per_split_classes": per_split_classes}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--media-root", type=Path, default=None, help="the staged Diving48 root (rgb/, the JSON files); default the configured [data.diving48] media_root")
    p.add_argument("--metadata", type=Path, default=None, help="directory holding the three JSON files (default: the media root)")
    p.add_argument("--no-media", action="store_true", help="skip the presence check (re-freeze from --metadata alone)")
    p.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write (default: this folder)")
    p.add_argument("--check", action="store_true", help="re-derive and compare with the committed manifest.csv.gz and classes.csv; write nothing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_media and args.metadata is None:
        raise SystemExit("--no-media needs --metadata (a directory holding the three JSON files)")
    media_root = None
    if not args.no_media:
        try:
            media_root = config.folder_root(HERE, "media_root", args.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root or configure the Diving48 root ([data.diving48] media_root): {e}") from None
    metadata: Path = args.metadata or media_root
    log = lambda m: print(f"[make_manifest diving48] {m}", file=sys.stderr)  # noqa: E731

    blobs = read_inputs(metadata)
    d = derive(blobs)
    rows, names = d["rows"], d["names"]
    log(f"V2 annotations: {len(rows)} rows, {NUM_CLASSES} classes (absent from V2: {list(ABSENT_CLASSES)})")

    missing: Counter = Counter()
    if args.no_media:
        log("--no-media: presence assumed (staged == listed)")
    else:
        media_dir = media_root / MEDIA_DIR
        present = {e.name for e in os.scandir(media_dir)} if media_dir.is_dir() else set()
        for r in rows:
            if r[1].split("/", 1)[1] not in present:
                missing[r[3]] += 1
        log(f"presence: {len(present)} files in {media_dir}; missing per split {dict(missing) or 0}")

    manifest_data = csv_bytes(MANIFEST_COLUMNS, rows)
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), n] for i, n in enumerate(names)])
    manifest_sha, classes_sha = sha256_bytes(manifest_data), sha256_bytes(classes_data)
    if args.check:
        same_m = gzip.decompress((args.out / "manifest.csv.gz").read_bytes()) == manifest_data
        same_c = (args.out / "classes.csv").read_bytes() == classes_data
        log(f"--check: manifest.csv {'identical' if same_m else 'DIFFERS'} (sha256 {manifest_sha[:16]}…); "
            f"classes.csv {'identical' if same_c else 'DIFFERS'} (sha256 {classes_sha[:16]}…)")
        if not (same_m and same_c):
            return 1
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        cm = write_if_changed(args.out / "manifest.csv.gz", gzip_bytes(manifest_data))
        cc = write_if_changed(args.out / "classes.csv", classes_data)
        log(f"manifest.csv.gz {'written' if cm else 'unchanged'} ({len(rows)} rows); classes.csv {'written' if cc else 'unchanged'}")

    n = {s: sum(1 for r in rows if r[3] == s) for s in UPSTREAM_ROWS}
    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(rows)},
        "upstream": {
            "source": UPSTREAM_SOURCE,
            "sha256": UPSTREAM_FILES["Diving48_V2_train.json"],
            "rows": TOTAL_UPSTREAM,
            "upstream_files": dict(UPSTREAM_FILES),
            "curation": {
                "total_upstream": TOTAL_UPSTREAM,
                "per_split": {s: {"listed": n[s], "staged": n[s] - missing[s], "missing": missing[s]} for s in UPSTREAM_ROWS},
                "excluded": {},
                "short_clips": 0,
                "leakage_audit": None,
            },
        },
        "derived": {
            "classes_present": {s: len(c) for s, c in d["per_split_classes"].items()},
            "train_per_class": {"min": min(d["per_split_classes"]["train"].values()), "max": max(d["per_split_classes"]["train"].values())},
            "test_majority_class_share": round(max(d["per_split_classes"]["test"].values()) / n["test"], 6),
            "chance_acc": round(1.0 / NUM_CLASSES, 6),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
