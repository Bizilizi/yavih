#!/usr/bin/env python
"""Freeze the yavih ``audioset20k`` manifest from the published dataset's public release files.

Input: the Hugging Face dataset ``11hu83/audioset20k`` at revision ``REVISION``, staged as downloaded
(setup.md "## Obtain the data"). Read (all read-only, under ``--media-root``):

* ``clips.csv`` (sha256 ``CLIPS_SHA256``): one row per published clip, ``key, split, ytid, start_s, end_s,
  labels, bytes, duration_s, width, height, vcodec, shard``; train rows then eval rows, each sorted by key;
  ``labels`` = comma-separated AudioSet mids from the official segment CSVs. THE row source.
* ``MISSING.csv`` (``MISSING_SHA256``): ``split, key`` of the listed segments the dataset could not provide.
* ``index.json`` (``INDEX_SHA256``): ``{"version": 1, "shards": [...], "members": {"<split>/<key>.mp4":
  [shard, offset, size]}}``; every frozen row's ``rel_media`` must be a member (and its size = ``bytes``).
* ``SHA256SUMS`` (``SHA256SUMS_SHA256``): must list the three files above with the pinned hashes.

What it does (stdlib only; byte-identical on re-run):

1. Every ``clips.csv`` row becomes a manifest row, in file order: no clip is excluded. The clips whose mp4 has
   no audio or no video track (``NO_AUDIO_TRACK`` / ``NO_VIDEO_TRACK``, measured by a header probe of every
   member of the pinned copy; ``--rescan`` re-measures with PyAV and refuses on any difference) are a recorded fact, printed
   as ``track_facts``: the reader hands their mp4 over like any other, the absent track's ``RawStream.info``
   says ``has_track: false``, and what to feed for it is the bridge's declared choice (setup.md "## Input contract").
2. Columns ``clip_id,split,labels,rel_media,ytid,start_s,duration_s``: ``clip_id`` = the dataset key
   ``<ytid>_<start ms, 7 digits>``; ``labels`` = the clip's mids de-duplicated, sorted by ``classes.csv``
   index and joined by ``;`` (an unknown mid is a hard error); ``rel_media`` = ``<split>/<key>.mp4`` (the
   index key; the media cell of both streams); ``ytid``, ``start_s``, ``duration_s`` verbatim from
   ``clips.csv`` (``duration_s`` = the container duration as published).
3. Asserts: the pinned sha256s; 527 classes = ``classes.csv`` (``CLASSES_SHA256``); keys and ytids
   unique; no ytid in both splits (``leakage_audit``); ``clips.csv`` + ``MISSING.csv`` = the official
   22 160 + 20 371 segments; every class has train and eval positives in the frozen rows.
4. Writes ``manifest.csv.gz`` (gzip ``mtime=0``, no filename) and prints the ``manifest.sha256``,
   ``manifest.upstream`` (with ``curation``; ``excluded`` holds only the ``MISSING.csv`` segments), ``budget``
   and ``metrics.chance`` values ``benchmark.yaml`` carries, and the ``track_facts`` setup.md lists
   (``tests/test_folder.py`` re-checks them against the yaml and setup.md).

No train draw: AS-20K has no cap (``protocol.train_draw: null``).

Usage::

    FOLDER=$(yavih bench path audioset20k)
    python "$FOLDER/make_manifest.py" [--media-root /path/to/audioset20k-wds-v1]   # default: the configured media_root
    python "$FOLDER/make_manifest.py" --check    # do not write; exit 1 on drift
    python "$FOLDER/make_manifest.py" --rescan   # + PyAV track probe
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

from yavih import config
from yavih.freeze import gzip_bytes

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- pinned release (benchmark.yaml `source`)
REPO_ID = "11hu83/audioset20k"
REVISION = "4a0055b21c183257fd9b65ee8d1cd11f13362f0f"
SHA256SUMS_SHA256 = "a6f22393b4f813af63e241b4a407771e0d98ade764a8399c4dd70951861b177c"
INDEX_SHA256 = "6976d509de988883a7e00783cec5dc438c1d38517d877518212a7ec88317b441"
CLIPS_SHA256 = "96e308e4e01647dafded79207663837b82e5e124371dbcf3d752bbfeea99fbad"
MISSING_SHA256 = "bbc4f0320faa852372a9244d07dcf687bfb6d46bbfa7e154a5b81424d20ce457"
CLASSES_SHA256 = "448831bb9bcc45279894d7328e0cc6bf2e19dccccf954621511375d8c7b7a50b"  # classes.csv

CLIPS_HEADER = ["key", "split", "ytid", "start_s", "end_s", "labels", "bytes", "duration_s", "width", "height", "vcodec", "shard"]
MANIFEST_HEADER = ["clip_id", "split", "labels", "rel_media", "ytid", "start_s", "duration_s"]
SPLITS = ("train", "eval")
OFFICIAL = {"train": 22160, "eval": 20371}  # balanced_train_segments.csv / eval_segments.csv rows
PUBLISHED = {"train": 17726, "eval": 16223}
KEY_RE = re.compile(r"[A-Za-z0-9_-]{11}_[0-9]{7}")

# Facts, not exclusions: clips whose mp4 holds no audio track (header probe of all 33 949 members with PyAV).
# They are manifest rows like every other;
# their audio stream's info says has_track: false. Every member has exactly one video track; no
# member has none.
NO_AUDIO_TRACK: dict[str, tuple[str, ...]] = {
    "train": ("NlBQ4kLpdzk_0260000", "d8WgfWSf1VM_0280000"),
    "eval": ("chc33hy6iIU_0510000",),
}
NO_VIDEO_TRACK: dict[str, tuple[str, ...]] = {"train": (), "eval": ()}

SHORT_SECONDS = 9.9  # curation.short_clips: published duration_s below this (nominal clips are 10 s)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fail(msg: str) -> None:
    raise SystemExit(f"make_manifest: {msg}")


def _pinned(path: Path, want: str) -> None:
    if not path.is_file():
        _fail(f"{path} is missing (setup.md '## Obtain the data': hf download {REPO_ID} --revision {REVISION})")
    got = sha256_file(path)
    if got != want:
        _fail(f"{path.name} sha256 {got} != pinned {want} (another revision of {REPO_ID}?)")


def read_classes() -> dict[str, int]:
    path = HERE / "classes.csv"
    if sha256_file(path) != CLASSES_SHA256:
        _fail("classes.csv changed: it is frozen (527 AudioSet mids, index,name,mid)")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    idx = {r["mid"]: int(r["index"]) for r in rows}
    if len(idx) != 527 or sorted(idx.values()) != list(range(527)):
        _fail("classes.csv is not 527 contiguous mids")
    return idx


def read_sums(media_root: Path) -> dict[str, str]:
    _pinned(media_root / "SHA256SUMS", SHA256SUMS_SHA256)
    out: dict[str, str] = {}
    for line in (media_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = sha
    for name, want in (("index.json", INDEX_SHA256), ("clips.csv", CLIPS_SHA256), ("MISSING.csv", MISSING_SHA256)):
        if out.get(name) != want:
            _fail(f"SHA256SUMS lists {name} as {out.get(name)}, pinned {want}")
    return out


def rescan_tracks(media_root: Path, index: dict) -> dict[str, dict[str, list[str]]]:
    """Header-probe every member with PyAV (no decode; reads by offset) → keys without an audio / video track."""
    from concurrent.futures import ThreadPoolExecutor

    import av  # optional: only for --rescan

    shards = [media_root / s for s in index["shards"]]
    fds = [os.open(p, os.O_RDONLY) for p in shards]

    class _Member(io.RawIOBase):
        def __init__(self, fd: int, off: int, size: int) -> None:
            self.fd, self.off, self.size, self.pos = fd, off, size, 0

        def readable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return True

        def tell(self) -> int:
            return self.pos

        def seek(self, p: int, whence: int = 0) -> int:
            self.pos = p if whence == 0 else (self.pos + p if whence == 1 else self.size + p)
            return self.pos

        def readinto(self, b) -> int:  # noqa: ANN001
            n = min(len(b), self.size - self.pos)
            if n <= 0:
                return 0
            d = os.pread(self.fd, n, self.off + self.pos)
            b[: len(d)] = d
            self.pos += len(d)
            return len(d)

    def probe(key: str) -> tuple[str, int, int]:
        sh, off, size = index["members"][key]
        with av.open(io.BufferedReader(_Member(fds[sh], off, size), 1 << 16)) as c:
            return key, len(c.streams.video), len(c.streams.audio)

    try:
        with ThreadPoolExecutor(16) as ex:
            res = list(ex.map(probe, sorted(index["members"])))
    finally:
        for fd in fds:
            os.close(fd)
    out: dict[str, dict[str, list[str]]] = {"audio": {s: [] for s in SPLITS}, "video": {s: [] for s in SPLITS}}
    for key, nv, na in res:
        split, name = key.split("/", 1)
        if not na:
            out["audio"][split].append(name[: -len(".mp4")])
        if not nv:
            out["video"][split].append(name[: -len(".mp4")])
    return out


def freeze(media_root: Path, *, rescan: bool = False) -> tuple[bytes, dict]:
    sums = read_sums(media_root)
    for name, want in (("clips.csv", CLIPS_SHA256), ("MISSING.csv", MISSING_SHA256), ("index.json", INDEX_SHA256)):
        _pinned(media_root / name, want)
    classes = read_classes()
    index = json.loads((media_root / "index.json").read_text(encoding="utf-8"))
    if index.get("version") != 1 or any(os.path.isabs(s) for s in index["shards"]):
        _fail("index.json must be version 1 with relative shard names")
    members = index["members"]

    with open(media_root / "clips.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != CLIPS_HEADER:
            _fail(f"clips.csv header {reader.fieldnames} != {CLIPS_HEADER}")
        clips = list(reader)
    with open(media_root / "MISSING.csv", newline="", encoding="utf-8") as fh:
        missing = list(csv.DictReader(fh))

    # ---- published population: order, uniqueness, counts, leakage
    if [c["split"] for c in clips] != ["train"] * PUBLISHED["train"] + ["eval"] * PUBLISHED["eval"]:
        _fail("clips.csv is not train rows then eval rows with the published counts")
    for s in SPLITS:
        keys = [c["key"] for c in clips if c["split"] == s]
        if keys != sorted(keys):
            _fail(f"clips.csv {s} rows are not sorted by key")
    if len({c["key"] for c in clips}) != len(clips) or len({c["ytid"] for c in clips}) != len(clips):
        _fail("clips.csv keys / ytids are not unique")
    by_split_ytids = {s: {c["ytid"] for c in clips if c["split"] == s} for s in SPLITS}
    leak = by_split_ytids["train"] & by_split_ytids["eval"]
    if leak:
        _fail(f"{len(leak)} ytids in both train and eval")
    n_missing = {s: sum(m["split"] == s for m in missing) for s in SPLITS}
    if len({(m["split"], m["key"]) for m in missing} & {(c["split"], c["key"]) for c in clips}):
        _fail("MISSING.csv overlaps clips.csv")
    for s in SPLITS:
        if PUBLISHED[s] + n_missing[s] != OFFICIAL[s]:
            _fail(f"{s}: {PUBLISHED[s]} published + {n_missing[s]} missing != {OFFICIAL[s]} official segments")

    # ---- track-less clips (pinned; --rescan re-measures)
    if rescan:
        found = rescan_tracks(media_root, index)
        for kind, pinned in (("audio", NO_AUDIO_TRACK), ("video", NO_VIDEO_TRACK)):
            got = {s: tuple(sorted(found[kind][s])) for s in SPLITS}
            if got != {s: tuple(sorted(pinned[s])) for s in SPLITS}:
                _fail(f"--rescan: clips without a {kind} track {got} != pinned {pinned}")
    published_keys = {(c["split"], c["key"]) for c in clips}
    for kind, pinned in (("audio", NO_AUDIO_TRACK), ("video", NO_VIDEO_TRACK)):
        unknown_keys = sorted({(s, k) for s, ks in pinned.items() for k in ks} - published_keys)
        if unknown_keys:
            _fail(f"pinned clips without a {kind} track {unknown_keys} are not clips.csv rows")

    # ---- rows
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(MANIFEST_HEADER)
    n_rows = {s: 0 for s in SPLITS}
    eval_instances = 0
    pos = {s: [0] * 527 for s in SPLITS}
    short = 0
    media_bytes = 0
    for c in clips:
        s, key = c["split"], c["key"]
        if not KEY_RE.fullmatch(key) or key != f"{c['ytid']}_{round(float(c['start_s']) * 1000):07d}":
            _fail(f"key {key!r} is not <ytid>_<start ms, 7 digits>")
        rel = f"{s}/{key}.mp4"
        m = members.get(rel)
        if m is None:
            _fail(f"{rel} is not an index.json member")
        if int(m[2]) != int(c["bytes"]) or index["shards"][int(m[0])] != c["shard"]:
            _fail(f"{rel}: index.json [shard, size] disagrees with clips.csv")
        mids = [x for x in c["labels"].split(",") if x]
        unknown = sorted(set(mids) - set(classes))
        if not mids or unknown:
            _fail(f"{key}: labels {c['labels']!r} empty or unknown mids {unknown}")
        ordered = sorted(set(mids), key=classes.__getitem__)
        for mid in ordered:
            pos[s][classes[mid]] += 1
        if s == "eval":
            eval_instances += len(ordered)
        short += float(c["duration_s"]) < SHORT_SECONDS
        media_bytes += int(c["bytes"])
        n_rows[s] += 1
        w.writerow([key, s, ";".join(ordered), rel, c["ytid"], c["start_s"], c["duration_s"]])
    if len(members) != len(clips):
        _fail(f"index.json has {len(members)} members, clips.csv {len(clips)} rows")
    for s in SPLITS:
        empty = [i for i, n in enumerate(pos[s]) if n == 0]
        if empty:
            _fail(f"{s}: classes without a positive: {empty[:10]}")
    raw = out.getvalue().encode("utf-8")

    for s in SPLITS:
        if n_rows[s] != PUBLISHED[s]:
            _fail(f"{s}: {n_rows[s]} manifest rows != {PUBLISHED[s]} published clips (no clip is excluded)")
    excluded = {"train_unavailable": n_missing["train"], "eval_unavailable": n_missing["eval"]}  # MISSING.csv only
    report = {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "classes_sha256": CLASSES_SHA256,
        "rows": n_rows,
        "chance_map": round(eval_instances / (527 * n_rows["eval"]), 6),
        "eval_label_instances": eval_instances,
        "min_class_positives": {s: min(pos[s]) for s in SPLITS},
        "media_bytes": media_bytes,
        "upstream": {
            "source": f"https://huggingface.co/datasets/{REPO_ID} @ {REVISION}: clips.csv",
            "sha256": CLIPS_SHA256,
            "rows": len(clips),
            "upstream_files": {"clips.csv": CLIPS_SHA256, "MISSING.csv": MISSING_SHA256, "index.json": INDEX_SHA256,
                               "SHA256SUMS": SHA256SUMS_SHA256},
            "curation": {
                "total_upstream": OFFICIAL["train"] + OFFICIAL["eval"],
                "per_split": {s: {"listed": n_rows[s], "staged": n_rows[s], "missing": 0} for s in SPLITS},
                "excluded": excluded,
                "short_clips": short,
                "leakage_audit": {"against": "train/eval ytid overlap of clips.csv", "excluded": 0},
            },
        },
        "sha256sums_entries": len(sums),
        # recorded facts (setup.md "## Input contract"; the census records has_track: false for them), never exclusions
        "track_facts": {
            "no_audio_track": {s: [f"{s}/{k}" for k in sorted(NO_AUDIO_TRACK[s])] for s in SPLITS},
            "no_video_track": {s: [f"{s}/{k}" for k in sorted(NO_VIDEO_TRACK[s])] for s in SPLITS},
        },
    }
    return raw, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--media-root", default=None,
                    help="the downloaded dataset directory (default: the folder's media_root, `yavih config get audioset20k.media_root`)")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 unless the committed manifest is identical")
    ap.add_argument("--rescan", action="store_true", help="re-probe every member's tracks with PyAV (~2 min) and compare")
    args = ap.parse_args(argv)
    try:
        media_root = config.folder_root(HERE, "media_root", args.media_root)
    except config.ConfigError as exc:
        ap.error(f"{exc} (or pass --media-root)")
    raw, report = freeze(media_root, rescan=args.rescan)
    target = HERE / "manifest.csv.gz"
    if args.check:
        have = gzip.decompress(target.read_bytes()) if target.is_file() else b""
        same = have == raw
        print(json.dumps({"identical": same, **report}, indent=1))
        return 0 if same else 1
    data = gzip_bytes(raw)
    if not target.is_file() or target.read_bytes() != data:
        target.write_bytes(data)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
