#!/usr/bin/env python
"""Freeze the yavih ``perception_test_sound_loc`` manifest from the official Perception Test annotations.

Inputs (all read-only, public, pinned by sha256 below):

* ``challenge_sound_localisation_train_annotations.zip`` (one JSON, 2 009 videos) and
  ``challenge_sound_localisation_valid_annotations.zip`` (one JSON, 5 359 videos) from
  ``https://storage.googleapis.com/dm-perception-test/zip_data/``: the train and validation files of the Perception
  Test temporal sound localisation challenge, the ones the official baseline notebook
  (``baselines/temporal_sound_localisation.ipynb`` of google-deepmind/perception_test) trains and evaluates on. The
  challenge test labels are not public. Each JSON maps ``video_id`` to ``{"metadata": {split, frame_rate,
  num_frames, ...}, "sound_localisation": [{label, label_id, timestamps: [start_us, end_us], ...}]}``.

Rows (one per video; train file order, then valid file order): ``clip_id`` = ``video_id``; ``split`` =
``metadata.split`` (``train`` / ``valid``); ``label`` = the distinct classes of the video's events joined by ``;``
(classes.csv order); ``events`` = ``class:start_us:end_us`` per event (class = classes.csv index, times the
upstream integer microseconds) joined by ``;`` in upstream order; ``duration_s`` = ``num_frames / frame_rate`` (the
annotated duration, 6 decimals); ``rel_video`` = ``videos/<video_id>.mp4``, ``rel_audio`` = ``audios/<video_id>.wav``.
classes.csv: ``index,name`` = the distinct upstream labels of both files ordered by the upstream ``label_id``, names
stripped of surrounding whitespace (upstream writes ``" Human:Speech"``). Nothing is excluded or capped.

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (gzip ``mtime=0``), ``classes.csv``, and the
``manifest.upstream`` block printed as JSON on stdout (paste into ``benchmark.yaml``; ``tests/test_folder.py``
re-checks it).

Usage::

    python "$(yavih bench path perception_test_sound_loc)/make_manifest.py" --metadata "$(yavih config get perception_test_sound_loc.media_root)/annotations"
    python "$(yavih bench path perception_test_sound_loc)/make_manifest.py" --check       # re-derive, compare, write nothing

``--metadata DIR`` holds the two zips (``yavih bench prepare perception_test_sound_loc 01`` downloads them to
``<media_root>/annotations``, the default). Without ``--no-media`` the staged media under ``--media-root`` are
counted for ``curation.per_split.*.staged`` (presence only). ``--check`` exits 1 when the re-derived files differ.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

TASK = "sound_localisation"
BASE_URL = "https://storage.googleapis.com/dm-perception-test/zip_data"
#: split value -> (zip file name, member JSON, sha256 of the zip, videos)
ANNOTATIONS = {
    "train": ("challenge_sound_localisation_train_annotations.zip", "challenge_sound_localisation_train.json",
              "2f94960db329d18f70ffdbfebc0a72e47476bd8ac42c81d7fecae2d7877f0a9e", 2009),
    "valid": ("challenge_sound_localisation_valid_annotations.zip", "challenge_sound_localisation_valid.json",
              "4f0e602ba8f97313aca7d84805334a73edd94fa6f56d80b945954cd32327c141", 5359),
}
NUM_CLASSES = 11
UPSTREAM_SOURCE = "Perception Test challenge_sound_localisation_{train,valid}_annotations.zip (storage.googleapis.com/dm-perception-test)"

MANIFEST_COLUMNS = ["clip_id", "split", "label", "events", "duration_s", "rel_video", "rel_audio"]
CLASSES_COLUMNS = ["index", "name"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_annotations(meta_dir: Path) -> dict[str, dict]:
    """``{split: parsed JSON}`` from the pinned zips in ``meta_dir`` (a zip with another sha256 is refused)."""
    out = {}
    for split, (zname, member, want, n) in ANNOTATIONS.items():
        path = meta_dir / zname
        if not path.is_file():
            raise SystemExit(f"{path} not found: download {BASE_URL}/{zname} (yavih bench prepare perception_test_sound_loc 01)")
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{path} has sha256 {got}, expected {want}")
        with zipfile.ZipFile(path) as zf:
            db = json.loads(zf.read(member).decode("utf-8"))
        if len(db) != n:
            raise SystemExit(f"{zname}: {len(db)} videos, expected {n}")
        out[split] = db
    return out


def class_index(dbs: dict[str, dict]) -> list[str]:
    """Distinct labels of every file, ordered by the upstream ``label_id`` (one label per id, checked)."""
    by_id: dict[int, str] = {}
    for db in dbs.values():
        for v in db.values():
            for a in v[TASK]:
                name = str(a["label"]).strip()
                prev = by_id.setdefault(int(a["label_id"]), name)
                if prev != name:
                    raise SystemExit(f"label_id {a['label_id']} names both {prev!r} and {name!r}")
    names = [by_id[i] for i in sorted(by_id)]
    if len(set(names)) != len(names):
        raise SystemExit("two label_ids share a name")
    return names


def derive(dbs: dict[str, dict]) -> tuple[list[list[str]], list[str], dict]:
    """``(manifest rows, class names, facts)`` from the parsed upstream files."""
    names = class_index(dbs)
    index = {n: i for i, n in enumerate(names)}
    rows: list[list[str]] = []
    n_events: Counter = Counter()
    for split in ANNOTATIONS:
        for vid, v in dbs[split].items():
            md = v["metadata"]
            if md["split"] != split or md["video_id"] != vid:
                raise SystemExit(f"{vid}: metadata split/video_id {md['split']}/{md['video_id']} disagree with the file")
            events = []
            for a in v[TASK]:
                s, e = (int(x) for x in a["timestamps"])
                if not 0 <= s <= e:
                    raise SystemExit(f"{vid}: event timestamps {a['timestamps']} are not 0 <= start <= end")
                events.append((index[str(a["label"]).strip()], s, e))
            if not events:
                raise SystemExit(f"{vid}: no {TASK} event (every upstream video has one)")
            present = sorted({c for c, _, _ in events})
            duration = float(md["num_frames"]) / float(md["frame_rate"])
            rows.append([vid, split, ";".join(names[c] for c in present), ";".join(f"{c}:{s}:{e}" for c, s, e in events),
                         f"{duration:.6f}", f"videos/{vid}.mp4", f"audios/{vid}.wav"])
            n_events[split] += len(events)
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("a video_id appears in two files")
    return rows, names, {"events": dict(n_events)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--media-root", type=Path, default=None, help="the staged root (videos/, audios/, annotations/); default the configured media_root")
    p.add_argument("--metadata", type=Path, default=None, help="directory holding the two annotation zips (default <media-root>/annotations)")
    p.add_argument("--no-media", action="store_true", help="skip the presence count (staged == listed)")
    p.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write (default: this folder)")
    p.add_argument("--check", action="store_true", help="re-derive and compare against the committed files; write nothing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_media and args.metadata is None:
        raise SystemExit("--no-media needs --metadata (the directory holding the two annotation zips)")
    media_root = None
    if not args.no_media or args.metadata is None:
        try:
            media_root = config.folder_root(HERE, "media_root", args.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root (or --no-media --metadata DIR) or configure the Perception Test root: {e}") from None
    meta_dir = args.metadata or (media_root / "annotations")
    log = lambda m: print(f"[make_manifest perception_test_sound_loc] {m}", file=sys.stderr)  # noqa: E731

    dbs = read_annotations(meta_dir)
    rows, names, facts = derive(dbs)
    if len(names) != NUM_CLASSES:
        raise SystemExit(f"{len(names)} classes, expected {NUM_CLASSES}")
    per_split = Counter(r[1] for r in rows)
    missing: Counter = Counter()
    if args.no_media:
        log("--no-media: presence assumed (staged == listed)")
    else:
        present = {}
        for sub in ("videos", "audios"):
            d = media_root / sub
            present[sub] = {e.name for e in os.scandir(d)} if d.is_dir() else set()
        for r in rows:
            if r[5].split("/", 1)[1] not in present["videos"] or r[6].split("/", 1)[1] not in present["audios"]:
                missing[r[1]] += 1
        log(f"presence: {sum(missing.values())} rows miss a video or a wav: {dict(missing)}")

    manifest_data = csv_bytes(MANIFEST_COLUMNS, rows)
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), n] for i, n in enumerate(names)])
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    classes_sha = hashlib.sha256(classes_data).hexdigest()
    if args.check:
        same_m = gzip.decompress((HERE / "manifest.csv.gz").read_bytes()) == manifest_data
        same_c = (HERE / "classes.csv").read_bytes() == classes_data
        log(f"--check: manifest.csv {'identical' if same_m else 'DIFFERS'}; classes.csv {'identical' if same_c else 'DIFFERS'}")
        if not (same_m and same_c):
            return 1
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        cm = write_if_changed(args.out / "manifest.csv.gz", gzip_bytes(manifest_data))
        cc = write_if_changed(args.out / "classes.csv", classes_data)
        log(f"manifest.csv.gz {'written' if cm else 'unchanged'} ({len(rows)} rows); classes.csv {'written' if cc else 'unchanged'}")

    zips = {ANNOTATIONS[s][0]: ANNOTATIONS[s][2] for s in ANNOTATIONS}
    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(rows)},
        "upstream": {
            "source": UPSTREAM_SOURCE,
            "sha256": ANNOTATIONS["valid"][2],
            "rows": len(rows),
            "upstream_files": zips,
            "curation": {
                "total_upstream": len(rows),
                "per_split": {s: {"listed": per_split[s], "staged": per_split[s] - missing[s], "missing": missing[s]} for s in ANNOTATIONS},
                "excluded": {},
                "short_clips": 0,
                "leakage_audit": None,
            },
        },
        "derived": {"classes": names, "events_per_split": facts["events"]},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
