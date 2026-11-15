#!/usr/bin/env python
"""Freeze the yavih ``llp_avvp`` manifest from the official LLP (Look, Listen and Parse) annotation files.

Inputs (all read-only, public, sha256-pinned): the six tab-separated files of the authors' repository
``YapengTian/AVVP-ECCV20`` at commit ``fde56118f2c2e7c19188dfefa2569a37b70054fe`` (``data/``; ``yavih bench prepare
llp_avvp 01`` fetches them into the ``annotations_root``):

* ``AVVP_train.csv`` (sha256 ``bbeb83e8…``, 10 000 videos), ``AVVP_val_pd.csv`` (``1811813b…``, 649) and
  ``AVVP_test_pd.csv`` (``3f7a2b04…``, 1 200): ``filename`` (``<ytid>_<start>_<end>``) and the weak video-level
  ``event_labels`` (comma-separated class names);
* ``AVVP_eval_audio.csv`` (``d5435664…``, 4 131 rows) and ``AVVP_eval_visual.csv`` (``e2274af6…``, 2 495 rows):
  ``filename, onset, offset, event_labels``, the dense second-level audio and visual events of the val and test
  videos (the pinned commit is the authors' 2024 fix of three rows of the audio file);
* ``AVVP_dataset_full.csv`` (``89f112ff…``): the 11 849 videos with their weak labels; asserted equal to the union
  of the three split files.

Derivation (no curation: every upstream video is listed, the official splits are kept): one row per video, ordered
by split (``train``, ``val``, ``test``) then upstream file order; ``clip_id`` = the upstream ``filename``;
``labels`` = the weak labels, de-duplicated (one train row lists ``Clapping`` twice) and ordered by class index,
``;``-joined (label kind ``multi``); ``rel_media`` = ``llp_videos/<ytid>.mp4`` (``ytid`` = the first 11
characters, as the authors' dataloader takes them); ``ytid``; ``start_s`` = the segment start in the YouTube video;
``audio_events`` / ``visual_events`` = every row of the dense file whose ``filename`` EQUALS the video's, in file
order, as ``onset:offset:class_index``, ``;``-joined (empty for train). That exact match is how the authors'
evaluation (``main_avvp.py::eval``) reads the files, so their defects are kept as they are: 14 audio and 11 visual
rows name no val/test video (garbled ids such as ``19jWnmYnlD8_30_400`` for ``19jWnmYnlD8_30_40``, which therefore
has no ground truth) and are ignored; 6 audio rows have ``offset <= onset`` and mark nothing; 8 audio rows are
exact duplicates. ``classes.csv`` = ``index,name`` in the authors' category order (``dataloader.py``), the logit
order of their model.

With a staged media root (``--media-root``, default the configured ``llp_avvp.media_root``) every clip's presence
and container facts are read (PyAV, threads): ``curation.short_clips`` counts clips whose container is shorter than
9.9 s; a clip without an audio or video track is listed in the printed ``derived`` block. ``--no-media`` skips it
(staged == listed).

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (gzip ``mtime=0``), ``classes.csv``, and the
``manifest.upstream`` block printed as JSON on stdout (paste into ``benchmark.yaml``).

Usage::

    yavih bench prepare llp_avvp 01                                              # the six annotation files
    python "$(yavih bench path llp_avvp)/make_manifest.py"                       # roots from the config; writes
    python "$(yavih bench path llp_avvp)/make_manifest.py" --check               # re-derive, compare, write nothing
    python "$(yavih bench path llp_avvp)/make_manifest.py" --no-media --metadata DIR --check   # DIR holds the six files

Writing rewrites the folder: run it in a git checkout of https://github.com/Bizilizi/yavih.
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
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

UPSTREAM_REPO = "YapengTian/AVVP-ECCV20"
UPSTREAM_COMMIT = "fde56118f2c2e7c19188dfefa2569a37b70054fe"
FILES_SHA256 = {
    "AVVP_train.csv": "bbeb83e8af46a85075216b213d3464f92dd4ab71468c852ae454b82cc745c0a4",
    "AVVP_val_pd.csv": "1811813b54c5f43c3be0a97bd0971cfb8a93818ca7b3ab96cdc7f4ac37ef37a6",
    "AVVP_test_pd.csv": "3f7a2b040e3b67c15d08efef25002bf8c5b2320f11049eb99ae0eb2d62f8eb9b",
    "AVVP_eval_audio.csv": "d5435664668bec6e7eafb39059d2f424ed2682f6b55395262fc3cb691b7d0778",
    "AVVP_eval_visual.csv": "e2274af694f3f27e38b840e0cd91a9d8d9ac85280f193e90828d26b22db69abe",
    "AVVP_dataset_full.csv": "89f112ffc602279c87cc2b2040c4cd42eaf163a9d7e61cf4e74250a343d21e56",
}
SPLIT_FILES = {"train": "AVVP_train.csv", "val": "AVVP_val_pd.csv", "test": "AVVP_test_pd.csv"}
EVENT_FILES = {"audio": "AVVP_eval_audio.csv", "visual": "AVVP_eval_visual.csv"}
VIDEOS_PER_SPLIT = {"train": 10000, "val": 649, "test": 1200}
TOTAL_VIDEOS = 11849
#: the authors' category order (``dataloader.py::ids_to_multinomial`` and ``main_avvp.py::eval``): the logit order
CATEGORIES = ("Speech", "Car", "Cheering", "Dog", "Cat", "Frying_(food)", "Basketball_bounce", "Fire_alarm", "Chainsaw", "Cello", "Banjo",
              "Singing", "Chicken_rooster", "Violin_fiddle", "Vacuum_cleaner", "Baby_laughter", "Accordion", "Lawn_mower", "Motorcycle",
              "Helicopter", "Acoustic_guitar", "Telephone_bell_ringing", "Baby_cry_infant_cry", "Blender", "Clapping")
SHORT_CLIP_S = 9.9

MANIFEST_COLUMNS = ["clip_id", "split", "labels", "rel_media", "ytid", "start_s", "audio_events", "visual_events"]
CLASSES_COLUMNS = ["index", "name"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path, header: Sequence[str]) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames != list(header):
            raise SystemExit(f"{path}: unexpected header {reader.fieldnames}, expected {list(header)}")
        return list(reader)


def read_upstream(meta_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """The six pinned files of ``meta_dir`` → ``({"splits": {split: [(filename, labels)]}, "events": {modality:
    [(filename, onset, offset, label)]}, "full": [(filename, labels)]}, {file: sha256})``; another sha256 is refused."""
    shas: dict[str, str] = {}
    for name, want in FILES_SHA256.items():
        path = meta_dir / name
        if not path.is_file():
            raise SystemExit(f"{path} not found: run `yavih bench prepare llp_avvp 01` or pass --metadata DIR")
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{path} has sha256 {got}, expected {want} ({UPSTREAM_REPO} @ {UPSTREAM_COMMIT[:7]})")
        shas[name] = got
    up: dict[str, Any] = {"splits": {}, "events": {}}
    for split, name in SPLIT_FILES.items():
        up["splits"][split] = [(r["filename"], r["event_labels"]) for r in read_tsv(meta_dir / name, ["filename", "event_labels"])]
    for modality, name in EVENT_FILES.items():
        up["events"][modality] = [(r["filename"], r["onset"], r["offset"], r["event_labels"])
                                  for r in read_tsv(meta_dir / name, ["filename", "onset", "offset", "event_labels"])]
    up["full"] = [(r["filename"], r["event_labels"]) for r in read_tsv(meta_dir / "AVVP_dataset_full.csv", ["filename", "event_labels"])]
    return up, shas


def split_filename(filename: str) -> tuple[str, str]:
    """``<ytid>_<start>_<end>`` → ``(ytid, start)``: the id is the first 11 characters (it may hold ``_`` and ``-``),
    as ``dataloader.py`` takes it; the start may be fractional (``ApKgnUPayqU_8.5_18.5``)."""
    ytid, rest = filename[:11], filename[11:]
    parts = rest.split("_")
    if len(parts) != 3 or parts[0] != "":
        raise SystemExit(f"filename {filename!r} is not <ytid>_<start>_<end>")
    start, end = float(parts[1]), float(parts[2])
    if abs(end - start - 10.0) > 1e-6:
        raise SystemExit(f"filename {filename!r}: the segment is not 10 s")
    return ytid, parts[1]


def derive(up: Mapping[str, Any]) -> tuple[list[list[str]], dict[str, Any]]:
    """The manifest rows (``MANIFEST_COLUMNS`` order) and the derivation facts from the parsed upstream files
    (``up["full"]`` may be ``None``: the union check is skipped)."""
    index = {c: i for i, c in enumerate(CATEGORIES)}
    splits = up["splits"]
    counts = {s: len(splits[s]) for s in SPLIT_FILES}
    if counts != VIDEOS_PER_SPLIT:
        raise SystemExit(f"videos per split {counts}, expected {VIDEOS_PER_SPLIT}")
    names = [f for s in SPLIT_FILES for f, _ in splits[s]]
    if len(set(names)) != TOTAL_VIDEOS:
        raise SystemExit(f"{len(set(names))} distinct filenames, expected {TOTAL_VIDEOS}")
    ytids = {s: {split_filename(f)[0] for f, _ in splits[s]} for s in SPLIT_FILES}
    overlap = {f"{a}/{b}": len(ytids[a] & ytids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlap.values()):
        raise SystemExit(f"a YouTube id is in two splits: {overlap}")
    if up.get("full") is not None and sorted(up["full"]) != sorted((f, lab) for s in SPLIT_FILES for f, lab in splits[s]):
        raise SystemExit("AVVP_dataset_full.csv is not the union of the three split files")
    scored = {f for s in ("val", "test") for f, _ in splits[s]}
    events: dict[str, dict[str, list[str]]] = {m: {} for m in EVENT_FILES}
    facts: dict[str, Any] = {"event_rows": {}, "orphan_event_rows": {}, "empty_event_rows": {}, "duplicate_event_rows": {}}
    for modality in EVENT_FILES:
        seen: Counter = Counter()
        orphan = empty = 0
        for f, on, off, lab in up["events"][modality]:
            if lab not in index:
                raise SystemExit(f"{EVENT_FILES[modality]}: unknown class {lab!r}")
            if not (on.isdigit() and off.isdigit() and 0 <= int(on) <= 10 and 0 <= int(off) <= 10):
                raise SystemExit(f"{EVENT_FILES[modality]}: onset/offset {on!r}/{off!r} are not whole seconds in [0, 10]")
            seen[(f, on, off, lab)] += 1
            if f not in scored:
                orphan += 1
                continue
            empty += int(int(off) <= int(on))
            events[modality].setdefault(f, []).append(f"{int(on)}:{int(off)}:{index[lab]}")
        facts["event_rows"][modality] = len(up["events"][modality])
        facts["orphan_event_rows"][modality] = orphan
        facts["empty_event_rows"][modality] = empty
        facts["duplicate_event_rows"][modality] = sum(c - 1 for c in seen.values() if c > 1)
    rows: list[list[str]] = []
    dup_labels = 0
    for split in SPLIT_FILES:
        for f, labels in splits[split]:
            toks = labels.split(",")
            bad = [t for t in toks if t not in index]
            if bad:
                raise SystemExit(f"{f}: unknown weak labels {bad}")
            dup_labels += int(len(set(toks)) != len(toks))
            weak = ";".join(CATEGORIES[i] for i in sorted({index[t] for t in toks}))
            ytid, start = split_filename(f)
            rows.append([f, split, weak, f"llp_videos/{ytid}.mp4", ytid, start, ";".join(events["audio"].get(f, [])),
                         ";".join(events["visual"].get(f, []))])
    facts["weak_label_cells_with_duplicates"] = dup_labels
    facts["ytid_overlap"] = overlap
    for modality in EVENT_FILES:
        facts[f"scored_videos_without_{modality}_events"] = {s: sum(1 for f, _ in splits[s] if f not in events[modality]) for s in ("val", "test")}
    return rows, facts


def probe_media(path: str) -> dict[str, Any]:
    """Container facts of one clip (PyAV): duration and whether it has a video and an audio track."""
    import av

    try:
        with av.open(path) as c:
            dur = float(c.duration) / 1e6 if c.duration is not None else None
            return {"ok": True, "duration_s": dur, "video": len(c.streams.video) > 0, "audio": len(c.streams.audio) > 0}
    except Exception as e:  # noqa: BLE001 - recorded per clip
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=None, help="directory holding the six annotation files (default: the configured annotations_root)")
    p.add_argument("--media-root", type=Path, default=None, help="the staged media root (llp_videos/<ytid>.mp4); default the configured llp_avvp.media_root")
    p.add_argument("--no-media", action="store_true", help="skip the presence and container checks (re-freeze from the annotation files alone)")
    p.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write (default: this folder)")
    p.add_argument("--check", action="store_true", help="re-derive and compare with the committed manifest.csv.gz and classes.csv; write nothing")
    p.add_argument("--workers", type=int, default=4, help="threads for the container reads")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = lambda m: print(f"[make_manifest llp_avvp] {m}", file=sys.stderr)  # noqa: E731
    meta_dir = args.metadata
    if meta_dir is None:
        try:
            meta_dir = config.folder_root(HERE, "annotations_root")
        except config.ConfigError as e:
            raise SystemExit(f"pass --metadata DIR or configure llp_avvp.annotations_root: {e}") from None
    media_root = None
    if not args.no_media:
        try:
            media_root = config.folder_root(HERE, "media_root", args.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root or configure llp_avvp.media_root: {e}") from None

    up, shas = read_upstream(Path(meta_dir))
    rows, facts = derive(up)
    log(f"{len(rows)} videos; event rows {facts['event_rows']}, ignored (no val/test video) {facts['orphan_event_rows']}")

    per_split = Counter(r[1] for r in rows)
    missing: Counter = Counter()
    short = 0
    derived_media: dict[str, Any] = {}
    if args.no_media:
        log("--no-media: presence assumed (staged == listed), short_clips 0")
    else:
        vdir = Path(media_root) / "llp_videos"
        present = {e.name for e in os.scandir(vdir)} if vdir.is_dir() else set()
        todo = []
        for r in rows:
            if r[3].split("/", 1)[1] in present:
                todo.append(r)
            else:
                missing[r[1]] += 1
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            facts_media = list(pool.map(probe_media, [str(Path(media_root) / r[3]) for r in todo]))
        unreadable = [r[0] for r, m in zip(todo, facts_media, strict=True) if not m["ok"]]
        no_audio = [r[0] for r, m in zip(todo, facts_media, strict=True) if m["ok"] and not m["audio"]]
        no_video = [r[0] for r, m in zip(todo, facts_media, strict=True) if m["ok"] and not m["video"]]
        durs = [m["duration_s"] for m in facts_media if m["ok"] and m["duration_s"] is not None]
        short = sum(1 for d in durs if d < SHORT_CLIP_S)
        derived_media = {"unreadable": unreadable, "no_audio_track": no_audio, "no_video_track": no_video,
                         "duration_s": {"min": min(durs), "max": max(durs)} if durs else None,
                         "media_bytes": sum(os.path.getsize(Path(media_root) / r[3]) for r in todo)}
        log(f"media: {len(todo)} present, missing {dict(missing)}, unreadable {len(unreadable)}, no audio {len(no_audio)}, "
            f"no video {len(no_video)}, {short} shorter than {SHORT_CLIP_S} s")

    manifest_data = csv_bytes(MANIFEST_COLUMNS, rows)
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), n] for i, n in enumerate(CATEGORIES)])
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    classes_sha = hashlib.sha256(classes_data).hexdigest()
    if args.check:
        same_m = gzip.decompress((HERE / "manifest.csv.gz").read_bytes()) == manifest_data
        same_c = (HERE / "classes.csv").read_bytes() == classes_data
        log(f"--check: manifest.csv {'identical' if same_m else 'DIFFERS'} (sha256 {manifest_sha[:16]}…); "
            f"classes.csv {'identical' if same_c else 'DIFFERS'} (sha256 {classes_sha[:16]}…)")
        if not (same_m and same_c):
            return 1
    else:
        out: Path = args.out
        out.mkdir(parents=True, exist_ok=True)
        changed_m = write_if_changed(out / "manifest.csv.gz", gzip_bytes(manifest_data))
        changed_c = write_if_changed(out / "classes.csv", classes_data)
        log(f"manifest.csv.gz {'written' if changed_m else 'unchanged'} ({len(rows)} rows, sha256 {manifest_sha[:16]}…); "
            f"classes.csv {'written' if changed_c else 'unchanged'}")

    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(rows)},
        "upstream": {
            "source": f"{UPSTREAM_REPO} @ {UPSTREAM_COMMIT[:7]} data/AVVP_{{train,val_pd,test_pd,eval_audio,eval_visual,dataset_full}}.csv",
            "sha256": FILES_SHA256["AVVP_test_pd.csv"],
            "rows": TOTAL_VIDEOS,
            "upstream_files": {f"{UPSTREAM_REPO}@{UPSTREAM_COMMIT[:7]}/data/{k}": v for k, v in shas.items()},
            "curation": {
                "total_upstream": TOTAL_VIDEOS,
                "per_split": {s: {"listed": per_split[s], "staged": per_split[s] - missing[s], "missing": missing[s]} for s in SPLIT_FILES},
                "excluded": {},
                "short_clips": short,
                "leakage_audit": {"against": "YouTube-id overlap between the train, val and test files (asserted empty)", "excluded": 0},
            },
        },
        "derived": {**facts, **derived_media},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
