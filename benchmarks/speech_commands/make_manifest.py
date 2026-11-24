#!/usr/bin/env python
"""Freeze the yavih ``speech_commands`` manifest: the HEAR 2021 task ``speech_commands-v0.0.2-5h``.

The rows are the ones hear-preprocess builds for this task (https://github.com/hearbenchmark/hear-preprocess
@ ``610d864b2c0f7064eaeced98ee8992d3914d58f1``, tag v2021.1.1, Apache-2.0): ``hearpreprocess/speech_commands.py``
(splits, labels, silence slices) and ``hearpreprocess/pipeline.py`` (``SubsampleSplit``: the 5 h draw). The parts of
that code the freeze needs are transcribed below and cited by file and line; every count they produce is asserted.

Inputs (all public, read-only):

* ``speech_commands_v0.02.tar.gz`` (http://download.tensorflow.org/data/, sha256 ``af14739e…``; md5 ``6b74f390…`` as
  HEAR pins it), extracted to ``<media_root>/train/``: 105 829 word wavs in 35 folders, ``validation_list.txt``
  (``VALIDATION_LIST_SHA256``), ``testing_list.txt`` (``TESTING_LIST_SHA256``) and the six ``_background_noise_``
  recordings (``BACKGROUND_NOISE``: their sample counts, pinned);
* ``speech_commands_test_set_v0.02.tar.gz`` (sha256 ``cc2a00c1…``; md5 ``854c580e…``), extracted to
  ``<media_root>/test/``: 4 890 wavs in 12 folders (the ten words, ``_unknown_``, ``_silence_``);
* the 1 s silence slices ``prepare/02_hear_silence.py`` cuts from the background recordings into
  ``<media_root>/hear_silence/`` (HEAR's ``GenerateTrainDataset``: a 16 000-sample window every 8 000 samples), whose
  names follow from the pinned sample counts.

With ``--no-media`` the archives' contents come from ``--metadata``, a listing CSV (``source,path``) that
``--write-metadata FILE`` writes from staged media: one row per word wav of each archive (``train_archive``,
``test_archive``) and one per line of the two list files (``validation_list``, ``testing_list``), each source sorted.
Every source is pinned (``LISTING_SHA256``: the sha256 of its sorted paths, one per line), and with media the two
list files also by the sha256 of their bytes; the listing's own sha256 is ``manifest.upstream.sha256``.

Rows (HEAR ``ExtractMetadata.get_split_paths``, ``speech_commands.py`` L178-235; labels L171-176):

* ``test`` = every wav of the test archive;
* HEAR ``valid`` = the ``validation_list.txt`` files + the ``running_tap`` silence slices;
* HEAR ``train`` = every word wav of the train archive minus ``testing_list.txt`` and ``validation_list.txt``, plus
  the other silence slices;
* label = the folder if it is one of the ten words or ``_silence_``, else ``_unknown_`` (12 classes, sorted as
  HEAR's ``labelvocabulary.csv``); ``clip_id`` = HEAR's ``unique_filestem`` (``<folder>_<file stem>``); ``word`` = the
  upstream folder; ``speaker`` = the 8-hex speaker id before ``_nohash_`` (empty for silence clips).

The 5 h draw (``pipeline.py`` ``SubsampleSplit.run`` L872-934; the task's ``max_task_duration_by_split``,
``speech_commands.py`` L52-60): per split, the files sorted by ``unique_filestem``, shuffled by
``DataFrame.sample(frac=1, random_state=str2int("SubsampleSplit(<split>)"))`` (pandas draws it as
``numpy.random.RandomState(seed).permutation(n)``) and the first ``int(max_duration / 1.0)`` kept: 16 000 of HEAR
train (5 h x 80/90) and 2 000 of HEAR valid (5 h x 10/90); test is not capped. The kept rows are the splits ``train``
and ``valid``; the rest of HEAR's ``full`` task stays in the manifest as the unused splits ``train_rest`` and
``valid_rest``, and the ``testing_list.txt`` files of the train archive (which HEAR never reads: it scores the
separate test archive) as the unused split ``testing_list``, whose ``clip_id`` carries the prefix ``testing_list:``
(4 074 of these files share their name, and so HEAR's filestem, with a test-archive file). So every upstream wav and every silence slice is one
row, and nothing is excluded.

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (gzip ``mtime=0``, no file name; rows in split order
``train, valid, test, train_rest, valid_rest, testing_list``, each sorted by ``clip_id``), ``classes.csv``
(``index,name``), and the ``manifest.upstream`` block printed as JSON on stdout (paste into ``benchmark.yaml``).

With media, every listed wav is checked present (one ``os.scandir`` per folder) and the headers of the read splits
are read: 16 kHz mono PCM_16, at most 16 000 samples; the clips shorter than 1.0 s are counted and must equal
``SHORT_CLIPS`` (``curation.short_clips``).

Usage::

    python "$(yavih bench path speech_commands)/make_manifest.py" --media-root "$(yavih config get speech_commands.media_root)"
    python "$(yavih bench path speech_commands)/make_manifest.py" --check     # re-derive, compare, write nothing
    python "$(yavih bench path speech_commands)/make_manifest.py" --write-metadata listing.csv   # list the staged archives
    python "$(yavih bench path speech_commands)/make_manifest.py" --no-media --metadata listing.csv --check

``--media-root`` defaults to the configured ``[data.speech_commands] media_root``
(``YAVIH_DATA__SPEECH_COMMANDS__MEDIA_ROOT``). ``--check`` exits 1 when the re-derived ``manifest.csv.gz`` or
``classes.csv`` differs from the committed files. Writing rewrites the folder, so run it in a git checkout of
https://github.com/Bizilizi/yavih installed with ``pip install -e``. The full staging and re-check::

    yavih bench prepare speech_commands 01   # download, verify, extract the two archives
    yavih bench prepare speech_commands 02   # cut and verify the HEAR silence slices
    yavih bench prepare speech_commands 03   # make_manifest.py --check against the staged media, then bench verify
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- pinned upstream

TRAIN_ARCHIVE = {"name": "speech_commands_v0.02.tar.gz", "url": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                 "sha256": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58", "md5": "6b74f3901214cb2c2934e98196829835"}
TEST_ARCHIVE = {"name": "speech_commands_test_set_v0.02.tar.gz", "url": "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz",
                "sha256": "cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df", "md5": "854c580ee90bff80c516491c84544e32"}
VALIDATION_LIST_SHA256 = "5747407275538b4056e823982f0db1fc993776ab532048196a19be701bdc87d2"   # train/validation_list.txt, the file's bytes (checked with media)
TESTING_LIST_SHA256 = "2d17c6b3faf63be43eda93cfeb0c747cfd79b7b236282039dbac65a2cb5f1df5"      # train/testing_list.txt, the file's bytes (checked with media)
#: sha256 of each listing source: its sorted paths, "<folder>/<file>\n" each (the order of the list files is not used).
LISTING_SHA256 = {
    "train_archive": "9c8e73c6ec502d61b1430d54a73d921149ec65cb2744cb8074c8b6e492c5aa10",     # 105 829 word wavs
    "test_archive": "53d945f02a69117f4a3244e6bf2610add9a497458faf5819fff35faa8b52cffc",      # 4 890 wavs
    "validation_list": "d4730d159c025513afcea0750083d82031a1f0da9101ca6f5e5bf1b5a5f9d943",   # 9 981 lines
    "testing_list": "95864d539da5a0b96b488ba4b6e5afc7d34d77cdad65c355eae92f267e02c508",      # 11 005 lines
}
N_TRAIN_ARCHIVE_WAVS = 105829
N_TEST_ARCHIVE_WAVS = 4890
#: ``train/_background_noise_/<file>``: its sample count (16 kHz mono PCM_16); the silence slice names follow from it.
BACKGROUND_NOISE: dict[str, int] = {
    "doing_the_dishes.wav": 1522930, "dude_miaowing.wav": 988891, "exercise_bike.wav": 980062,
    "pink_noise.wav": 960000, "running_tap.wav": 978488, "white_noise.wav": 960000,
}
#: clips of the read splits shorter than 1.0 s (16 000 samples), measured on the staged media.
SHORT_CLIPS = 1804

HEAR_PREPROCESS = "hearbenchmark/hear-preprocess @ 610d864 (v2021.1.1)"

# ----------------------------------------------------------------------------- HEAR task constants
# hearpreprocess/speech_commands.py L24-27 and L29-60, hearpreprocess/pipeline.py L39-45
# @ 610d864b2c0f7064eaeced98ee8992d3914d58f1 (Apache-2.0).

WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
BACKGROUND_NOISE_DIR = "_background_noise_"
UNKNOWN = "_unknown_"
SILENCE = "_silence_"
SAMPLE_DURATION = 1.0
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
TRAIN_PERCENTAGE = 100 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE
TRAINVAL_PERCENTAGE = TRAIN_PERCENTAGE + VALIDATION_PERCENTAGE
MAX_TASK_DURATION_BY_SPLIT = {  # mode "5h"
    "train": 3600 * 5 * TRAIN_PERCENTAGE / TRAINVAL_PERCENTAGE,
    "valid": 3600 * 5 * VALIDATION_PERCENTAGE / TRAINVAL_PERCENTAGE,
    "test": None,
}
SAMPLE_RATE = 16000

# ----------------------------------------------------------------------------- the yavih manifest

SPLIT_ORDER = ("train", "valid", "test", "train_rest", "valid_rest", "testing_list")
READ_SPLITS = ("train", "valid", "test")
EXPECTED_ROWS = {"train": 16000, "valid": 2000, "test": 4890}
CLASSES = sorted([*WORDS, SILENCE, UNKNOWN])   # HEAR MetadataVocabulary: the sorted labels (pipeline.py L1185-1188)
SILENCE_DIR = "hear_silence"
TESTING_LIST_PREFIX = "testing_list:"   # clip_id prefix of the unused testing_list rows (not HEAR rows)
MANIFEST_COLUMNS = ["clip_id", "split", "label", "rel_path", "word", "speaker"]
CLASSES_COLUMNS = ["index", "name"]
LISTING_COLUMNS = ["source", "path"]
LISTING_SOURCES = ("train_archive", "test_archive", "validation_list", "testing_list")


# ----------------------------------------------------------------------------- transcribed from hear-preprocess


def str2int(s: str) -> int:
    """hearpreprocess/util/luigi.py L217-222 (the body verbatim): the seed of a named shuffle."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (2 ** 32 - 1)


def relpath_to_unique_filestem(relpath: str) -> str:
    """speech_commands.py ``ExtractMetadata.relpath_to_unique_filestem`` (L146-155): ``<folder>_<stem>``."""
    name = "_".join(relpath.split("/")[-2:])
    return os.path.splitext(name)[0]


def relpath_to_label(relpath: str) -> str:
    """speech_commands.py ``ExtractMetadata.relpath_to_label`` (L171-176)."""
    label = os.path.basename(os.path.dirname(relpath))
    if label not in WORDS and label != SILENCE:
        label = UNKNOWN
    return label


def silence_starts(n_samples: int, sr: int = SAMPLE_RATE) -> list[int]:
    """speech_commands.py ``GenerateTrainDataset.run`` (L119): ``range(0, len(audio) - sr, sr // 2)``."""
    return list(range(0, n_samples - sr, sr // 2))


def silence_files() -> list[str]:
    """The slice file names ``<name>-<start>.wav`` (L121-123: ``f"{name}-{start}{ext}"``), sorted."""
    out = []
    for fname, n in sorted(BACKGROUND_NOISE.items()):
        name, ext = os.path.splitext(fname)
        out.extend(f"{name}-{start}{ext}" for start in silence_starts(n))
    return sorted(out)


def subsample(filestems: list[str], split: str) -> list[str]:
    """pipeline.py ``SubsampleSplit.run`` (L872-934): the filestems of ``split`` it keeps, in its shuffled order.

    Upstream sorts the split's ``(unique_filestem, relpath)`` pairs by ``unique_filestem``, shuffles them with
    ``DataFrame.sample(frac=1, random_state=str2int(f"SubsampleSplit({split})"))`` and keeps the first
    ``int(max_split_duration / sample_duration)``. pandas draws that shuffle as
    ``RandomState(seed).choice(n, size=n, replace=False)``, which is ``RandomState(seed).permutation(n)``.
    """
    stems = sorted(filestems)
    if len(set(stems)) != len(stems):
        raise SystemExit(f"{split}: duplicate unique_filestem")
    order = np.random.RandomState(str2int(f"SubsampleSplit({split})")).permutation(len(stems))
    shuffled = [stems[i] for i in order]
    max_split_duration = MAX_TASK_DURATION_BY_SPLIT[split]
    max_files = len(shuffled) if max_split_duration is None else int(max_split_duration / SAMPLE_DURATION)
    return shuffled[:max_files]


# ----------------------------------------------------------------------------- helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def listing_digest(paths: list[str]) -> str:
    return sha256_bytes("".join(f"{p}\n" for p in sorted(paths)).encode())


def read_listing(path: Path) -> dict[str, list[str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != LISTING_COLUMNS:
            raise SystemExit(f"{path}: header {reader.fieldnames}, expected {LISTING_COLUMNS}")
        out: dict[str, list[str]] = {s: [] for s in LISTING_SOURCES}
        for r in reader:
            if r["source"] not in out:
                raise SystemExit(f"{path}: unknown source {r['source']!r}")
            out[r["source"]].append(r["path"])
    return out


def listing_bytes(listing: dict[str, list[str]]) -> bytes:
    return csv_bytes(LISTING_COLUMNS, [[s, p] for s in LISTING_SOURCES for p in sorted(listing[s])])


def scan_archive(root: Path) -> list[str]:
    """``<folder>/<file>.wav`` of every folder of an extracted archive except ``_background_noise_``, sorted."""
    out = []
    for d in os.scandir(root):
        if d.is_dir() and d.name != BACKGROUND_NOISE_DIR:
            out.extend(f"{d.name}/{e.name}" for e in os.scandir(d.path) if e.name.endswith(".wav"))
    return sorted(out)


def listing_from_media(media_root: Path) -> dict[str, list[str]]:
    train, test = media_root / "train", media_root / "test"
    lists = {}
    for key, fname, want in (("validation_list", "validation_list.txt", VALIDATION_LIST_SHA256),
                             ("testing_list", "testing_list.txt", TESTING_LIST_SHA256)):
        p = train / fname
        if not p.is_file():
            raise SystemExit(f"{p} not found: stage the archives first (yavih bench prepare speech_commands 01)")
        data = p.read_bytes()
        if sha256_bytes(data) != want:
            raise SystemExit(f"{p}: sha256 {sha256_bytes(data)}, expected {want} ({TRAIN_ARCHIVE['name']})")
        lists[key] = sorted(data.decode().splitlines())
    if not test.is_dir():
        raise SystemExit(f"{test} not found: stage the archives first (yavih bench prepare speech_commands 01)")
    return {"train_archive": scan_archive(train), "test_archive": scan_archive(test), **lists}


def check_listing(listing: dict[str, list[str]]) -> None:
    """Every pin of the archives' contents, or SystemExit naming both digests."""
    for source in LISTING_SOURCES:
        got, want = listing_digest(listing[source]), LISTING_SHA256[source]
        if got != want:
            raise SystemExit(f"listing {source}: sha256 {got}, expected {want} ({TRAIN_ARCHIVE['name']} / {TEST_ARCHIVE['name']})")
    if len(listing["train_archive"]) != N_TRAIN_ARCHIVE_WAVS or len(listing["test_archive"]) != N_TEST_ARCHIVE_WAVS:
        raise SystemExit("archive wav counts differ from the pinned release")


_SPEAKER = re.compile(r"(?:^|_)([0-9a-f]{8})_nohash_")


def _row(rel_path: str, folder: str, filename: str, hear_split: str) -> dict[str, str]:
    stem = relpath_to_unique_filestem(f"{folder}/{filename}")
    m = _SPEAKER.search(filename)   # "<speaker>_nohash_<n>.wav"; the test archive's _unknown_ files are "<word>_<speaker>_nohash_<n>.wav.wav"
    speaker = m.group(1) if m else ""
    return {"clip_id": stem, "hear_split": hear_split, "label": relpath_to_label(f"{folder}/{filename}"),
            "rel_path": rel_path, "word": folder, "speaker": speaker}


def derive_rows(listing: dict[str, list[str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Manifest rows (HEAR ``get_split_paths``, its labels and the 5 h draw) and the HEAR ``full`` split sizes."""
    validation = set(listing["validation_list"])
    testing = set(listing["testing_list"])
    archive = set(listing["train_archive"])
    if not validation <= archive or not testing <= archive:
        raise SystemExit("validation_list.txt / testing_list.txt name files absent from the train archive")
    silence = silence_files()
    val_silence = {f for f in silence if f.startswith("running_tap")}  # L196-199: glob(f"{SILENCE}/running_tap*.wav")

    rows = []
    for p in listing["test_archive"]:
        folder, filename = p.split("/")
        rows.append(_row(f"test/{p}", folder, filename, "test"))
    for p in listing["train_archive"]:
        folder, filename = p.split("/")
        hear_split = "testing_list" if p in testing else ("valid" if p in validation else "train")
        r = _row(f"train/{p}", folder, filename, hear_split)
        if hear_split == "testing_list":   # outside HEAR's task; 4 074 share a file name with the test archive
            r["clip_id"] = f"{TESTING_LIST_PREFIX}{r['clip_id']}"
        rows.append(r)
    for f in silence:
        rows.append(_row(f"{SILENCE_DIR}/{f}", SILENCE, f, "valid" if f in val_silence else "train"))
    ids = Counter(r["clip_id"] for r in rows)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise SystemExit(f"{len(dup)} duplicate unique_filestem, e.g. {dup[:5]}")

    full = Counter(r["hear_split"] for r in rows)
    kept = {s: set(subsample([r["clip_id"] for r in rows if r["hear_split"] == s], s)) for s in ("train", "valid", "test")}
    for r in rows:
        s = r.pop("hear_split")
        r["split"] = s if s == "testing_list" or r["clip_id"] in kept[s] else f"{s}_rest"
    rank = {s: i for i, s in enumerate(SPLIT_ORDER)}
    rows.sort(key=lambda r: (rank[r["split"]], r["clip_id"]))
    return rows, dict(sorted(full.items()))


def wav_header(path: str) -> tuple[int, int, int, str] | None:
    import soundfile as sf  # header read only

    try:
        i = sf.info(path)
    except Exception:  # noqa: BLE001 - an unreadable file is counted missing, never fatal here
        return None
    return i.frames, i.samplerate, i.channels, i.subtype


# ----------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--media-root", type=Path, default=None,
                   help="the staged root (train/, test/, hear_silence/); default the configured [data.speech_commands] media_root")
    p.add_argument("--metadata", type=Path, default=None, help="a listing CSV (source,path; .csv or .csv.gz) in place of the staged archives")
    p.add_argument("--no-media", action="store_true", help="skip presence and WAV-header checks (re-freeze from --metadata alone)")
    p.add_argument("--write-metadata", type=Path, default=None, help="write the listing CSV of the staged archives (.csv or .csv.gz) and exit")
    p.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write (default: this folder)")
    p.add_argument("--check", action="store_true", help="re-derive and compare against the committed manifest.csv.gz and classes.csv; write nothing")
    p.add_argument("--workers", type=int, default=16, help="threads for WAV header reads (network file system latency bound)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_media and args.metadata is None:
        raise SystemExit("--no-media needs --metadata (a listing CSV written by --write-metadata)")
    media_root = None
    if not args.no_media:
        try:
            media_root = config.folder_root(HERE, "media_root", args.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root or configure [data.speech_commands] media_root: {e}") from None
    log = lambda m: print(f"[make_manifest speech_commands] {m}", file=sys.stderr)  # noqa: E731

    # 1. the archives' contents: the staged media, or the listing.
    listing = read_listing(args.metadata) if args.metadata is not None else listing_from_media(media_root)
    if args.write_metadata is not None:
        data = listing_bytes(listing)
        args.write_metadata.write_bytes(gzip_bytes(data) if args.write_metadata.suffix == ".gz" else data)
        log(f"listing written to {args.write_metadata} ({sum(map(len, listing.values()))} rows)")
    check_listing(listing)
    log(f"listing: {len(listing['train_archive'])} + {len(listing['test_archive'])} wavs, list files "
        f"{len(listing['validation_list'])} / {len(listing['testing_list'])} lines; every pin matches")
    if args.write_metadata is not None:
        return 0

    # 2. HEAR's rows and the 5 h draw.
    rows, full = derive_rows(listing)
    counts = Counter(r["split"] for r in rows)
    for s, n in EXPECTED_ROWS.items():
        if counts[s] != n:
            raise SystemExit(f"split {s}: {counts[s]} rows, expected {n}")
    if sorted({r["label"] for r in rows}) != CLASSES:
        raise SystemExit(f"labels {sorted({r['label'] for r in rows})} != {CLASSES}")
    for s in READ_SPLITS:
        lacking = set(CLASSES) - {r["label"] for r in rows if r["split"] == s}
        if lacking:
            raise SystemExit(f"split {s} lacks classes {sorted(lacking)} (HEAR asserts every label in every split)")
    log(f"rows: {dict(counts)}; HEAR full splits {full}")

    # 3. presence and WAV headers.
    missing: Counter = Counter()
    short = SHORT_CLIPS
    media_bytes = 0
    headers: Counter = Counter()
    if args.no_media:
        log(f"--no-media: presence assumed (staged == listed), short_clips = the pinned {SHORT_CLIPS}")
    else:
        present: dict[str, set[str]] = {}
        for r in rows:
            d = os.path.dirname(r["rel_path"])
            if d not in present:
                present[d] = {e.name for e in os.scandir(media_root / d)} if (media_root / d).is_dir() else set()
            if os.path.basename(r["rel_path"]) not in present[d]:
                missing[r["split"]] += 1
        if sum(missing.values()):
            log(f"WARNING: {sum(missing.values())} wavs missing: {dict(missing)}")
        read_rows = [r for r in rows if r["split"] in READ_SPLITS
                     and os.path.basename(r["rel_path"]) in present[os.path.dirname(r["rel_path"])]]
        short = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            got = pool.map(wav_header, [str(media_root / r["rel_path"]) for r in read_rows])
            for r, h in zip(read_rows, got, strict=True):
                if h is None:
                    missing[r["split"]] += 1
                    continue
                frames, rate, ch, subtype = h
                headers[(rate, ch, subtype)] += 1
                media_bytes += os.path.getsize(media_root / r["rel_path"])
                if frames > SAMPLE_RATE or rate != SAMPLE_RATE or ch != 1 or subtype != "PCM_16":
                    raise SystemExit(f"{r['rel_path']}: {frames} frames, {rate} Hz, {ch} ch, {subtype}; expected <= 16000 at 16 kHz mono PCM_16")
                short += frames < round(SAMPLE_DURATION * SAMPLE_RATE)
        log(f"WAV headers (read splits): {dict(headers)}; {short} shorter than 1.0 s; {media_bytes / 1e9:.3f} GB")
        if not sum(missing.values()) and short != SHORT_CLIPS:
            raise SystemExit(f"{short} clips shorter than 1.0 s, pinned SHORT_CLIPS = {SHORT_CLIPS}")

    # 4. manifest.csv.gz + classes.csv.
    rows_out = [[r[c] for c in MANIFEST_COLUMNS] for r in rows]
    manifest_data = csv_bytes(MANIFEST_COLUMNS, rows_out)
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), c] for i, c in enumerate(CLASSES)])
    manifest_sha, classes_sha = sha256_bytes(manifest_data), sha256_bytes(classes_data)
    if args.check:
        committed_m = gzip.decompress((HERE / "manifest.csv.gz").read_bytes())
        committed_c = (HERE / "classes.csv").read_bytes()
        same_m, same_c = committed_m == manifest_data, committed_c == classes_data
        log(f"--check: manifest.csv {'identical' if same_m else 'DIFFERS'} (sha256 {manifest_sha[:16]}…); "
            f"classes.csv {'identical' if same_c else 'DIFFERS'} (sha256 {classes_sha[:16]}…)")
        if not (same_m and same_c):
            return 1
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        changed_m = write_if_changed(args.out / "manifest.csv.gz", gzip_bytes(manifest_data))
        changed_c = write_if_changed(args.out / "classes.csv", classes_data)
        log(f"manifest.csv.gz {'written' if changed_m else 'unchanged'} ({len(rows_out)} rows, sha256 {manifest_sha[:16]}…); "
            f"classes.csv {'written' if changed_c else 'unchanged'} (sha256 {classes_sha[:16]}…)")

    # 5. the manifest.upstream block (paste into benchmark.yaml).
    speakers = {s: {r["speaker"] for r in rows if r["split"] == s and r["speaker"]} for s in READ_SPLITS}
    shared_tr_te = len(speakers["train"] & speakers["test"])
    shared_tr_va = len(speakers["train"] & speakers["valid"])
    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(rows_out)},
        "upstream": {
            "source": f"{TRAIN_ARCHIVE['name']} + {TEST_ARCHIVE['name']} (listing below) + the HEAR silence slices; "
                      f"HEAR task speech_commands-v0.0.2-5h of {HEAR_PREPROCESS}",
            "sha256": sha256_bytes(listing_bytes(listing)),
            "rows": len(rows),
            "upstream_files": {
                TRAIN_ARCHIVE["name"]: TRAIN_ARCHIVE["sha256"],
                TEST_ARCHIVE["name"]: TEST_ARCHIVE["sha256"],
                "train/validation_list.txt": VALIDATION_LIST_SHA256,
                "train/testing_list.txt": TESTING_LIST_SHA256,
            },
            "curation": {
                "total_upstream": len(rows),
                "per_split": {s: {"listed": counts[s], "staged": counts[s] - missing[s], "missing": missing[s]} for s in SPLIT_ORDER},
                "excluded": {},
                "short_clips": short,
                "leakage_audit": {
                    "against": f"8-hex speaker ids (before _nohash_) across train, valid and test: {shared_tr_te} train/test "
                               f"and {shared_tr_va} train/valid speakers shared",
                    "excluded": 0,
                },
            },
        },
        "derived": {
            "hear_full_splits": full,
            "labels_per_read_split": {s: dict(sorted(Counter(r["label"] for r in rows if r["split"] == s).items())) for s in READ_SPLITS},
            "silence_slices": len(silence_files()),
            "wav_headers": {f"{k[0]}Hz_{k[1]}ch_{k[2]}": v for k, v in headers.items()},
            "media_gb_read_splits": round(media_bytes / 1e9, 3) if media_bytes else None,
            "chance_top1_acc": 1.0 / len(CLASSES),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
