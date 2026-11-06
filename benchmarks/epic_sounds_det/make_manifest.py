#!/usr/bin/env python
"""Freeze the yavih ``epic_sounds_det`` manifest from the public EPIC-SOUNDS annotation files.

Inputs (all read-only, ``epic-kitchens/epic-sounds-annotations`` at commit ``57a922f``):

* ``EPIC_Sounds_train.csv`` (sha256 ``225427ab…``, 60 055 labelled segments over 495 recordings) and
  ``EPIC_Sounds_validation.csv`` (sha256 ``fff66485…``, 8 035 segments over 137 recordings): one manifest row per
  RECORDING (``video_id``), in order of first appearance in the file (train first). The detection task is defined
  over the whole untrimmed recording, so the row is the recording, never a segment or a window.
* ``EPIC_Sounds_recognition_test_timestamps.csv`` (sha256 ``4912b017…``, 44 recordings) and
  ``EPIC_Sounds_detection_test_videos.csv`` (sha256 ``b77ec524…``, 23 recordings): read only to count the
  recordings the freeze excludes (their labels are held by the challenge server).

Row columns: ``clip_id`` = ``video_id``; ``split`` = ``train`` | ``validation``; ``labels`` = the distinct classes
with at least one segment in the recording (``;``-joined, class-index order; a ``multi`` label, so the probe gets
``classes.csv`` and the class count); ``rel_audio`` = ``audio/<video_id>.m4a`` (the recording's AAC track, see
``prepare/02_audio.py``); ``events`` (``label_aux``) = every labelled segment of the recording in upstream file order
as ``<class_id>:<start_ms>:<stop_ms>`` joined by ``;``, with the times taken from ``start_timestamp`` /
``stop_timestamp`` (``HH:MM:SS.mmm``, millisecond-exact, the columns the official scorer reads). Background
segments (class 13) stay in ``events``: they are training targets; the official scorer drops them from the ground
truth (``eval.py``). ``classes.csv`` = ``index,name`` from ``class_id`` / ``class`` (44 classes, upstream ids).

No train draw, no cap: every labelled recording is listed.

Usage::

    python "$(yavih bench path epic_sounds_det)/make_manifest.py" --media-root "$(yavih config get epic_sounds_det.media_root)"
    python "$(yavih bench path epic_sounds_det)/make_manifest.py" --check
    python "$(yavih bench path epic_sounds_det)/make_manifest.py" --no-media --metadata <annotations dir> --check

``--media-root`` defaults to the configured ``[data.epic_sounds_det] media_root``; ``--metadata`` (the directory
holding the four CSVs) defaults to ``<media-root>/annotations``. ``--no-media`` skips the presence check of
``audio/*.m4a``. ``--check`` exits 1 when the re-derived files differ from the committed ones.
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

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

UPSTREAM_REPO = "epic-kitchens/epic-sounds-annotations"
UPSTREAM_COMMIT = "57a922f0d352e9429f1ef8a37eee21758dd3a33c"
#: file -> (sha256, rows, distinct recordings)
UPSTREAM_FILES: dict[str, tuple[str, int, int]] = {
    "EPIC_Sounds_train.csv": ("225427ab288cc5c72c1680abb29f5ab8ccd3608c3f4cf409541d2c5cf720dd32", 60055, 495),
    "EPIC_Sounds_validation.csv": ("fff66485d8478762fd9cde1a829e6d91713a5f2f7f917e3af746cf891dd62e5a", 8035, 137),
    "EPIC_Sounds_recognition_test_timestamps.csv": ("4912b017f96e371b3d05ef6194fb2796ecffff3782eacf8909a3cbaf2af4ef3a", 5131, 44),
    "EPIC_Sounds_detection_test_videos.csv": ("b77ec524fdd64be5b01db6bc8472a983d58acb767c854e3d40ac98cdf4ec9bc3", 23, 23),
}
SPLITS = {"train": "EPIC_Sounds_train.csv", "validation": "EPIC_Sounds_validation.csv"}
LABELLED_HEADER = ["annotation_id", "participant_id", "video_id", "start_timestamp", "stop_timestamp", "start_sample",
                   "stop_sample", "description", "class", "class_id"]
NUM_CLASSES = 44
BACKGROUND = (13, "background")
TOTAL_UPSTREAM = 495 + 137 + 44 + 23   # recordings across the four public files

MANIFEST_COLUMNS = ["clip_id", "split", "labels", "rel_audio", "events"]
CLASSES_COLUMNS = ["index", "name"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ts_ms(ts: str) -> int:
    """``HH:MM:SS.mmm`` -> integer milliseconds (exact: upstream writes three decimals)."""
    hh, mm, rest = ts.split(":")
    sec, _, frac = rest.partition(".")
    if len(frac) != 3:
        raise SystemExit(f"timestamp {ts!r} does not carry exactly three decimals")
    return ((int(hh) * 60 + int(mm)) * 60 + int(sec)) * 1000 + int(frac)


def read_rows(path: Path, header: list[str] | None) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if header is not None and reader.fieldnames != header:
            raise SystemExit(f"{path}: unexpected header {reader.fieldnames}, expected {header}")
        return list(reader)


def check_inputs(meta_dir: Path) -> dict[str, str]:
    digests = {}
    for name, (sha, _, _) in UPSTREAM_FILES.items():
        p = meta_dir / name
        if not p.is_file():
            raise SystemExit(f"{p} not found: run `yavih bench prepare epic_sounds_det 01` (or pass --metadata DIR)")
        got = sha256_file(p)
        if got != sha:
            raise SystemExit(f"{p} has sha256 {got}, expected {sha} ({UPSTREAM_REPO} @ {UPSTREAM_COMMIT[:7]})")
        digests[name] = got
    return digests


def derive(meta_dir: Path) -> tuple[list[dict[str, str]], dict[int, str], dict]:
    classes: dict[int, str] = {}
    rows: list[dict[str, str]] = []
    facts: dict = {"segments": {}, "background_segments": {}, "zero_length_segments": 0}
    for split, fname in SPLITS.items():
        segs = read_rows(meta_dir / fname, LABELLED_HEADER)
        sha, n_rows, n_videos = UPSTREAM_FILES[fname]
        if len(segs) != n_rows:
            raise SystemExit(f"{fname}: {len(segs)} rows, expected {n_rows}")
        order: list[str] = []
        per_video: dict[str, list[tuple[int, int, int]]] = {}
        for r in segs:
            cid, name = int(r["class_id"]), r["class"].strip()
            if classes.setdefault(cid, name) != name:
                raise SystemExit(f"class_id {cid} names both {classes[cid]!r} and {name!r}")
            s, e = ts_ms(r["start_timestamp"]), ts_ms(r["stop_timestamp"])
            if e <= s:
                facts["zero_length_segments"] += 1
            vid = r["video_id"].strip()
            if vid not in per_video:
                order.append(vid)
                per_video[vid] = []
            per_video[vid].append((cid, s, e))
        if len(order) != n_videos:
            raise SystemExit(f"{fname}: {len(order)} recordings, expected {n_videos}")
        facts["segments"][split] = len(segs)
        facts["background_segments"][split] = sum(1 for r in segs if int(r["class_id"]) == BACKGROUND[0])
        for vid in order:
            evs = per_video[vid]
            present = sorted({c for c, _, _ in evs})
            rows.append({
                "clip_id": vid, "split": split, "labels": ";".join(str(c) for c in present),  # names filled below
                "rel_audio": f"audio/{vid}.m4a", "events": ";".join(f"{c}:{s}:{e}" for c, s, e in evs),
            })
    if sorted(classes) != list(range(NUM_CLASSES)) or classes.get(BACKGROUND[0]) != BACKGROUND[1]:
        raise SystemExit(f"class ids {sorted(classes)} are not 0..{NUM_CLASSES - 1} with {BACKGROUND} (upstream changed?)")
    for r in rows:
        r["labels"] = ";".join(classes[int(c)] for c in r["labels"].split(";"))
    ids = [r["clip_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("a recording appears in train and validation")
    return rows, classes, facts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--media-root", type=Path, default=None, help="the staged root (audio/, annotations/); default the configured [data.epic_sounds_det] media_root")
    p.add_argument("--metadata", type=Path, default=None, help="directory holding the four EPIC-SOUNDS CSVs (default <media-root>/annotations)")
    p.add_argument("--no-media", action="store_true", help="skip the presence check (re-freeze from --metadata alone)")
    p.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write (default: this folder)")
    p.add_argument("--check", action="store_true", help="re-derive and compare with the committed files; write nothing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_media and args.metadata is None:
        raise SystemExit("--no-media needs --metadata (the directory holding the four EPIC-SOUNDS CSVs)")
    media_root = None
    if not args.no_media:
        try:
            media_root = config.folder_root(HERE, "media_root", args.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root or configure [data.epic_sounds_det] media_root: {e}") from None
    meta_dir = args.metadata or (media_root / "annotations")
    log = lambda m: print(f"[make_manifest epic_sounds_det] {m}", file=sys.stderr)  # noqa: E731

    digests = check_inputs(meta_dir)
    rows, classes, facts = derive(meta_dir)
    n_split = Counter(r["split"] for r in rows)
    log(f"{len(rows)} recordings ({dict(n_split)}), {sum(facts['segments'].values())} segments, {len(classes)} classes")

    missing: Counter = Counter()
    if args.no_media:
        log("--no-media: presence assumed (staged == listed)")
    else:
        audio_dir = media_root / "audio"
        present = {e.name for e in os.scandir(audio_dir)} if audio_dir.is_dir() else set()
        for r in rows:
            if r["rel_audio"].split("/", 1)[1] not in present:
                missing[r["split"]] += 1
        log(f"presence: {sum(missing.values())} of {len(rows)} audio files missing {dict(missing)}")

    manifest_data = csv_bytes(MANIFEST_COLUMNS, [[r[c] for c in MANIFEST_COLUMNS] for r in rows])
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), classes[i]] for i in range(NUM_CLASSES)])
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    classes_sha = hashlib.sha256(classes_data).hexdigest()
    if args.check:
        same_m = gzip.decompress((HERE / "manifest.csv.gz").read_bytes()) == manifest_data
        same_c = (HERE / "classes.csv").read_bytes() == classes_data
        log(f"--check: manifest.csv {'identical' if same_m else 'DIFFERS'} ({manifest_sha[:16]}…); classes.csv {'identical' if same_c else 'DIFFERS'} ({classes_sha[:16]}…)")
        if not (same_m and same_c):
            return 1
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        cm = write_if_changed(args.out / "manifest.csv.gz", gzip_bytes(manifest_data))
        cc = write_if_changed(args.out / "classes.csv", classes_data)
        log(f"manifest.csv.gz {'written' if cm else 'unchanged'}; classes.csv {'written' if cc else 'unchanged'}")

    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(rows)},
        "upstream": {
            "source": f"{UPSTREAM_REPO} @ {UPSTREAM_COMMIT[:7]} EPIC_Sounds_{{train,validation}}.csv (one row per recording)",
            "sha256": digests["EPIC_Sounds_validation.csv"],
            "rows": TOTAL_UPSTREAM,
            "upstream_files": {f"epic-sounds-annotations/{k}": v for k, v in digests.items()},
            "curation": {
                "total_upstream": TOTAL_UPSTREAM,
                "per_split": {s: {"listed": n_split[s], "staged": n_split[s] - missing[s], "missing": missing[s]} for s in SPLITS},
                "excluded": {"recognition_test_labels_withheld": 44, "detection_test_labels_withheld": 23},
            },
        },
        "derived": facts,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
