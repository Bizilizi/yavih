#!/usr/bin/env python
"""Freeze the yavih ``nsynth`` manifest: HEAR's ``nsynth_pitch-v2.2.3-5h`` task from the public NSynth metadata.

The task is the HEAR 2021 benchmark's NSynth Pitch task in its default ``5h`` mode (Turian et al., NeurIPS 2021
competition track), re-derived here with HEAR's own preprocessing code (``hearpreprocess`` v2021.1.1,
https://github.com/hearbenchmark/hear-preprocess at commit ``610d864b2c0f7064eaeced98ee8992d3914d58f1``,
Apache-2.0; the copied functions are marked below):

1. per official NSynth split (``train``, ``valid``, ``test``), the notes whose MIDI ``pitch`` lies in
   ``[21, 108]`` (the piano range; ``nsynth_pitch.py`` ``ExtractMetadata.get_requires_metadata``);
2. per split, the sorted note names shuffled with ``numpy.random.RandomState(str2int("SubsampleSplit(<split>)"))``
   and truncated to ``int(max_task_duration_by_split[split] / sample_duration)`` files: 4 000 train, 500 valid,
   500 test (``pipeline.py`` ``SubsampleSplit.run`` with the ``5h`` mode of ``nsynth_pitch.py``: 5 h of
   train + valid audio in 80/10 proportion, the test split capped like valid);
3. the label is the pitch; the class vocabulary is HEAR's ``labelvocabulary.csv``: the sorted distinct pitches of
   the subset, which is all 88 pitches 21–108 (asserted), so class index = pitch − 21.

Inputs (all read-only), the three ``examples.json`` files of the NSynth "json/wav" release
(https://magenta.tensorflow.org/datasets/nsynth, CC BY 4.0), each inside its archive at
``nsynth-<split>/examples.json``:

* ``nsynth-train.jsonwav.tar.gz`` (23 815 298 079 B, md5 ``fde6665a…`` as pinned by HEAR), ``examples.json``
  sha256 ``EXAMPLES_SHA256["train"]`` (289 205 notes);
* ``nsynth-valid.jsonwav.tar.gz`` (1 068 767 009 B, md5 ``87e94a00…``), ``examples.json`` 12 678 notes;
* ``nsynth-test.jsonwav.tar.gz`` (349 501 546 B, md5 ``5e6f8719…``), ``examples.json`` 4 096 notes.

A file with another sha256 is refused. The rows are written split by split (train, valid, test), sorted by note
name within a split: ``clip_id`` = ``note_str``, ``labels`` = the pitch, ``split`` = the NSynth split,
``rel_path`` = ``nsynth-<split>/audio/<note_str>.wav``, and the extras ``instrument_str``,
``instrument_family_str``, ``instrument_source_str`` and ``velocity`` from ``examples.json``. ``classes.csv`` is
``index,name`` with the names ``21`` … ``108`` in pitch order.

Curation (``manifest.upstream.curation``): of the 305 979 upstream notes, those outside the pitch range and those
not drawn by the 5h subsample are excluded, each under its own reason, so ``listed + excluded == total_upstream``.
The leakage audit checks NSynth's instrument split: no train instrument occurs in valid or test, while valid and
test share the same 53 instruments (pinned in ``VALID_TEST_INSTRUMENTS``); this is why ``valid`` is ``unused`` in
``benchmark.yaml`` and never trains the probe (HEAR only selects its MLP on it). With media
(``--media-root``) every listed wav must be present and every header must read 64 000 frames at 16 kHz, mono.

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (gzip ``mtime=0``, no file name), ``classes.csv``, and the
``manifest.upstream`` block printed as JSON on stdout (paste it into ``benchmark.yaml``).

Usage::

    python "$(yavih bench path nsynth)/make_manifest.py" --media-root "$(yavih config get nsynth.media_root)"
    python "$(yavih bench path nsynth)/make_manifest.py" --check     # re-derive, compare, write nothing
    python "$(yavih bench path nsynth)/make_manifest.py" --no-media --metadata DIR --check

``--media-root`` defaults to the configured ``[data.nsynth] media_root`` (``YAVIH_DATA__NSYNTH__MEDIA_ROOT``);
``--metadata DIR`` is a directory holding ``nsynth-{train,valid,test}/examples.json`` (default: the media root,
where ``yavih bench prepare nsynth 02`` extracts them). ``--no-media`` re-freezes from the metadata alone (no
presence or header checks). ``--check`` exits 1 when the re-derived ``manifest.csv.gz`` or ``classes.csv`` differs
from the committed files. Writing rewrites the folder, so run it in a git checkout of
https://github.com/Bizilizi/yavih. The full re-freeze with the pinned checks and ``bench verify``::

    yavih bench prepare nsynth 04
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

SPLITS = ("train", "valid", "test")
UPSTREAM_ROWS = {"train": 289205, "valid": 12678, "test": 4096}
TOTAL_UPSTREAM = sum(UPSTREAM_ROWS.values())          # 305 979
EXAMPLES_SHA256 = {
    "train": "a8182e084aa338d764f64f6a283bd1778d358e913a7074294110e9ec601e0087",
    "valid": "050e0bf55d1a87eee2abbc3d273c7f525404c0a51ac378561b905f2da31a7286",
    "test": "74df2dd960c156cd5e8757d8afb1fe30e6a233aa99043a29037e7b54cde6908f",
}
HEAR_PREPROCESS = "hearbenchmark/hear-preprocess @ 610d864 (v2021.1.1)"
VALID_TEST_INSTRUMENTS = 53                           # valid and test are the same 53 instruments; train shares none
NUM_CLASSES = 88
SAMPLE_RATE = 16000
N_SAMPLES = 64000                                     # 4.0 s at 16 kHz

MANIFEST_COLUMNS = ["clip_id", "labels", "split", "rel_path", "instrument_str", "instrument_family_str",
                    "instrument_source_str", "velocity"]
CLASSES_COLUMNS = ["index", "name"]

EXCLUDED_PITCH = "pitch outside 21-108 (HEAR nsynth_pitch pitch_range_min/max)"
EXCLUDED_DRAW = "not drawn by the HEAR 5h subsample (SubsampleSplit: train 4000, valid 500, test 500)"


# ----------------------------------------------------------------------------- copied from HEAR
# Copied from hearpreprocess (HEAR benchmark), https://github.com/hearbenchmark/hear-preprocess
# at commit 610d864b2c0f7064eaeced98ee8992d3914d58f1 (tag v2021.1.1, = the PyPI wheel hearpreprocess==2021.1.1),
# licence Apache-2.0 (Copyright the HEAR authors; https://www.apache.org/licenses/LICENSE-2.0).
#   * str2int: hearpreprocess/util/luigi.py, verbatim.
#   * the task constants: hearpreprocess/nsynth_pitch.py generic_task_config (pitch range, sample_duration, the
#     "5h" mode's max_task_duration_by_split) and hearpreprocess/pipeline.py (the 80/10/10 percentages), verbatim
#     values; the "50h" and "small" modes and the download URLs are left out.
#   * pitch_filter: nsynth_pitch.py ExtractMetadata.get_requires_metadata, the pandas
#     ``metadata["pitch"].between(min, max)`` filter rewritten over the list of examples.json records.
#   * max_files_of / subsample_split: pipeline.py SubsampleSplit.run, the selection part only (no symlinks).
#     Changes: pandas replaced by numpy. ``df.sort_values("unique_filestem")`` over unique names is ``sorted``;
#     ``df.sample(frac=1, random_state=int)`` is ``numpy.random.RandomState(int).permutation(n)`` (pandas calls
#     ``RandomState(seed).choice(n, n, replace=False)``, which is that permutation); tests/test_folder.py
#     checks the equivalence against the pandas call where pandas is installed.

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
TRAIN_PERCENTAGE = 100 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE
TRAINVAL_PERCENTAGE = TRAIN_PERCENTAGE + VALIDATION_PERCENTAGE

generic_task_config = {
    "task_name": "nsynth_pitch",
    "version": "v2.2.3",
    "sample_duration": 4.0,
    "pitch_range_min": 21,
    "pitch_range_max": 108,
    "evaluation": ["pitch_acc", "chroma_acc"],
    "default_mode": "5h",
    "modes": {
        "5h": {
            # No more than 5 hours of audio (training + validation)
            "max_task_duration_by_split": {
                "train": 3600 * 5 * TRAIN_PERCENTAGE / TRAINVAL_PERCENTAGE,
                "valid": 3600 * 5 * VALIDATION_PERCENTAGE / TRAINVAL_PERCENTAGE,
                "test": 3600 * 5 * TEST_PERCENTAGE / TRAINVAL_PERCENTAGE,
            }
        },
    },
}


def str2int(s: str) -> int:
    """
    Convert string to int using hex hashing.
    https://stackoverflow.com/a/16008760/82733
    """
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (2 ** 32 - 1)


def pitch_filter(records: list[dict], task_config: dict = generic_task_config) -> list[dict]:
    """Filter out pitches that are not within the range (inclusive at both ends, as ``Series.between``)."""
    return [r for r in records if task_config["pitch_range_min"] <= int(r["pitch"]) <= task_config["pitch_range_max"]]


def max_files_of(split: str, task_config: dict = generic_task_config, mode: str = "5h") -> int:
    """``SubsampleSplit.run``: ``max_files = int(max_split_duration / sample_duration)``."""
    max_split_duration = task_config["modes"][mode]["max_task_duration_by_split"][split]
    sample_duration = task_config["sample_duration"]
    return int(max_split_duration / sample_duration)


def subsample_split(unique_filestems: list[str], split: str, max_files: int) -> list[str]:
    """``SubsampleSplit.run``: deterministically sort the split's filestems, deterministically shuffle them,
    keep the first ``max_files``. Returns them in the shuffled (drawn) order."""
    # Get all unique_filestem in this split, deterministically sorted
    split_filestems = sorted(set(unique_filestems))
    # Deterministically shuffle the filestems
    perm = np.random.RandomState(str2int(f"SubsampleSplit({split})")).permutation(len(split_filestems))
    shuffled = [split_filestems[i] for i in perm]
    return shuffled[:max_files]


# ----------------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_examples(meta_dir: Path, split: str) -> tuple[list[dict], str]:
    """The records of ``<meta_dir>/nsynth-<split>/examples.json`` (pinned sha256), in file order."""
    path = meta_dir / f"nsynth-{split}" / "examples.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found: extract it with `yavih bench prepare nsynth 02` (or pass --metadata DIR)")
    digest = sha256_file(path)
    if digest != EXAMPLES_SHA256[split]:
        raise SystemExit(f"{path} has sha256 {digest}, expected {EXAMPLES_SHA256[split]}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    records = []
    for key, rec in data.items():
        if rec.get("note_str") != key:
            raise SystemExit(f"{path}: key {key!r} != note_str {rec.get('note_str')!r}")
        records.append(rec)
    if len(records) != UPSTREAM_ROWS[split]:
        raise SystemExit(f"{path} has {len(records)} notes, expected {UPSTREAM_ROWS[split]}")
    return records, digest


def wav_header(path: str) -> tuple[int, int, int]:
    import soundfile as sf  # header read only

    info = sf.info(path)
    return info.frames, info.samplerate, info.channels


def derive(meta_dir: Path) -> dict:
    """Everything the freeze needs, from the three examples.json files."""
    rows: list[dict[str, str]] = []
    per_split: dict[str, dict] = {}
    digests: dict[str, str] = {}
    instruments: dict[str, set[str]] = {}
    for split in SPLITS:
        records, digests[split] = read_examples(meta_dir, split)
        in_range = pitch_filter(records)
        by_name = {r["note_str"]: r for r in in_range}
        drawn = subsample_split(list(by_name), split, max_files_of(split))
        instruments[split] = {r["instrument_str"] for r in records}
        per_split[split] = {"upstream": len(records), "in_range": len(in_range), "drawn": len(drawn)}
        for name in sorted(drawn):
            r = by_name[name]
            rows.append({
                "clip_id": name, "labels": str(int(r["pitch"])), "split": split,
                "rel_path": f"nsynth-{split}/audio/{name}.wav", "instrument_str": r["instrument_str"],
                "instrument_family_str": r["instrument_family_str"], "instrument_source_str": r["instrument_source_str"],
                "velocity": str(int(r["velocity"])),
            })
    return {"rows": rows, "per_split": per_split, "digests": digests, "instruments": instruments}


# ----------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--media-root", type=Path, default=None, help="the staged NSynth root (nsynth-<split>/examples.json, nsynth-<split>/audio/); default the configured [data.nsynth] media_root")
    p.add_argument("--metadata", type=Path, default=None, help="directory holding nsynth-{train,valid,test}/examples.json (default: the media root)")
    p.add_argument("--no-media", action="store_true", help="skip presence and WAV-header checks (re-freeze from --metadata alone)")
    p.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write (default: this folder)")
    p.add_argument("--check", action="store_true", help="re-derive and compare against the committed manifest.csv.gz and classes.csv; write nothing")
    p.add_argument("--workers", type=int, default=16, help="threads for WAV header reads")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_media and args.metadata is None:
        raise SystemExit("--no-media needs --metadata DIR (holding nsynth-{train,valid,test}/examples.json)")
    media_root = None
    if not args.no_media:
        try:
            media_root = config.folder_root(HERE, "media_root", args.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root or configure the NSynth root: {e}") from None
    meta_dir: Path = args.metadata or media_root
    out: Path = args.out
    log = lambda m: print(f"[make_manifest nsynth] {m}", file=sys.stderr)  # noqa: E731

    # 1. HEAR's nsynth_pitch 5h selection from the three examples.json files.
    d = derive(meta_dir)
    rows, per_split = d["rows"], d["per_split"]
    for s in SPLITS:
        log(f"{s}: {per_split[s]['upstream']} notes, {per_split[s]['in_range']} in pitch range, {per_split[s]['drawn']} drawn")

    # 2. invariants the protocol rests on.
    ids = [r["clip_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate clip_id")
    counts = Counter(r["split"] for r in rows)
    want = {s: max_files_of(s) for s in SPLITS}
    if dict(counts) != want:
        raise SystemExit(f"split counts {dict(counts)} != {want}")
    labels = sorted({int(r["labels"]) for r in rows})
    if labels != list(range(21, 109)):
        raise SystemExit(f"the subset's pitches are not exactly 21..108 ({len(labels)} distinct): HEAR's class vocabulary would shift")
    per_split_labels = {s: {r["labels"] for r in rows if r["split"] == s} for s in SPLITS}
    missing = {s: NUM_CLASSES - len(v) for s, v in per_split_labels.items() if len(v) != NUM_CLASSES}
    if missing:  # HEAR's _diagnose_split_labels raises on a split missing a label (non-small modes)
        raise SystemExit(f"a split misses pitches: {missing}")
    inst = d["instruments"]
    overlaps = {f"{a}/{b}": sorted(inst[a] & inst[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:] if inst[a] & inst[b]}
    log(f"leakage audit: instrument overlaps between splits: {({k: len(v) for k, v in overlaps.items()}) or 'none'}")
    if set(overlaps) != {"valid/test"} or not (inst["valid"] == inst["test"] and len(inst["test"]) == VALID_TEST_INSTRUMENTS):
        raise SystemExit(f"NSynth's instrument split changed: overlaps {({k: len(v) for k, v in overlaps.items()})}")
    audit = (f"instrument_str across the NSynth splits (examples.json): train ({len(inst['train'])} instruments) shares none with "
             f"valid or test; valid and test are the same {VALID_TEST_INSTRUMENTS} instruments, so valid is unused (never trains the probe)")

    # 3. presence (one scandir per split) and WAV headers.
    missing_per_split: Counter = Counter()
    header_counts: Counter = Counter()
    short = 0
    pcm_bytes = 0
    if args.no_media:
        log("--no-media: presence assumed (staged == listed), short_clips 0")
    else:
        present: dict[str, set[str]] = {}
        for s in SPLITS:
            adir = media_root / f"nsynth-{s}" / "audio"
            present[s] = {e.name for e in os.scandir(adir)} if adir.is_dir() else set()
        here = [r for r in rows if r["rel_path"].rsplit("/", 1)[1] in present[r["split"]]]
        for r in rows:
            if r["rel_path"].rsplit("/", 1)[1] not in present[r["split"]]:
                missing_per_split[r["split"]] += 1
        if sum(missing_per_split.values()):
            log(f"WARNING: {sum(missing_per_split.values())} wavs missing: {dict(missing_per_split)} (yavih bench prepare nsynth 02)")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for frames, rate, ch in pool.map(wav_header, [str(media_root / r["rel_path"]) for r in here]):
                header_counts[(frames, rate, ch)] += 1
                pcm_bytes += 44 + 2 * frames * ch
                if frames < N_SAMPLES:
                    short += 1
        log(f"WAV headers: {dict(header_counts)}; {short} shorter than {N_SAMPLES} frames; ~{pcm_bytes / 1e9:.3f} GB PCM")

    # 4. manifest.csv.gz + classes.csv (fixed column and row order).
    manifest_data = csv_bytes(MANIFEST_COLUMNS, [[r[c] for c in MANIFEST_COLUMNS] for r in rows])
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), str(p)] for i, p in enumerate(labels)])
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    classes_sha = hashlib.sha256(classes_data).hexdigest()
    if args.check:
        committed_m = gzip.decompress((out / "manifest.csv.gz").read_bytes())
        committed_c = (out / "classes.csv").read_bytes()
        same_m, same_c = committed_m == manifest_data, committed_c == classes_data
        log(f"--check: manifest.csv {'identical' if same_m else 'DIFFERS'} (sha256 {manifest_sha[:16]}…); "
            f"classes.csv {'identical' if same_c else 'DIFFERS'} (sha256 {classes_sha[:16]}…)")
        if not (same_m and same_c):
            return 1
    else:
        out.mkdir(parents=True, exist_ok=True)
        changed_m = write_if_changed(out / "manifest.csv.gz", gzip_bytes(manifest_data))
        changed_c = write_if_changed(out / "classes.csv", classes_data)
        log(f"manifest.csv.gz {'written' if changed_m else 'unchanged'} ({len(rows)} rows, sha256 {manifest_sha[:16]}…); "
            f"classes.csv {'written' if changed_c else 'unchanged'} (sha256 {classes_sha[:16]}…)")

    # 5. the manifest.upstream block (paste into benchmark.yaml).
    n_pitch = sum(per_split[s]["upstream"] - per_split[s]["in_range"] for s in SPLITS)
    n_draw = sum(per_split[s]["in_range"] - per_split[s]["drawn"] for s in SPLITS)
    digests = d["digests"]
    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(rows)},
        "upstream": {
            "source": f"NSynth json/wav release (magenta, 2017-04) nsynth-{{train,valid,test}}/examples.json; selection {HEAR_PREPROCESS} nsynth_pitch 5h",
            "sha256": digests["test"],
            "rows": TOTAL_UPSTREAM,
            "upstream_files": {f"nsynth-{s}/examples.json": digests[s] for s in SPLITS},
            "curation": {
                "total_upstream": TOTAL_UPSTREAM,
                "per_split": {
                    s: {"listed": counts[s], "staged": counts[s] - missing_per_split[s], "missing": missing_per_split[s]} for s in SPLITS
                },
                "excluded": {EXCLUDED_PITCH: n_pitch, EXCLUDED_DRAW: n_draw},
                "short_clips": short,
                "leakage_audit": {"against": audit, "excluded": 0},
            },
        },
        "derived": {
            "per_split": per_split,
            "draw_seeds": {s: str2int(f"SubsampleSplit({s})") for s in SPLITS},
            "per_split_label_counts": {s: len(v) for s, v in per_split_labels.items()},
            "instruments": {s: len({r["instrument_str"] for r in rows if r["split"] == s}) for s in SPLITS},
            "wav_headers": {f"{k[0]}x{k[1]}Hzx{k[2]}ch": v for k, v in header_counts.items()},
            "budget_media_gb_pcm": round(pcm_bytes / 1e9, 3) if pcm_bytes else None,
            "chance_acc": 1.0 / NUM_CLASSES,
            "majority_by_split": {s: Counter(r["labels"] for r in rows if r["split"] == s).most_common(1)[0] for s in SPLITS},
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
