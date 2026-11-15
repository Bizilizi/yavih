#!/usr/bin/env python
"""Freeze the yavih ``crema_d`` manifest: CREMA-D audio in HEAR 2021's five folds.

Protocol source: HEAR 2021's ``tfds_crema_d`` task (hear-preprocess @ ``610d864``, v2021.1.1:
``split_mode: new_split_kfold``, ``nfolds: 5``, multiclass, primary score ``top1_acc``). HEAR's preprocessed
CREMA-D is not downloadable (the Zenodo release retracted the TFDS tasks; the bucket is requester-pays), so the
folds are re-derived here, step by step, from the public CREMA-D files, with the copied code of ``hear_split.py``:

1. the clip list of the TensorFlow Datasets builder ``crema_d`` 1.0.0: every ``FileName`` of
   ``processedResults/summaryTable.csv`` except the four TFDS skips as absent from GitHub (``1040_ITH_SAD_XX``,
   ``1006_TIE_NEU_XX``, ``1013_WSI_DIS_XX``, ``1017_IWW_FEA_XX``): 7 438 of 7 442;
2. TFDS's speaker-grouped split (91 actors, ``np.random.RandomState(0)``, 70/10/20: 63/9/19 actors);
3. per TFDS split, the writer's order (md5 of the file name salted with the split name), its shard count
   (``NUM_SHARDS``: train 16, validation 2, test 4, as TFDS's own ``dataset_info.json`` of ``crema_d/1.0.0``
   records them) and each clip's ``tfds_id`` (``crema_d-<split>.tfrecord-<shard>-of-<n>__<index>``). With media
   staged, the serialized size of every split is recomputed from the wavs and asserted equal, byte for byte, to
   that file's ``numBytes`` (``TFDS_NUM_BYTES``), and the shard count TFDS's writer derives from it to
   ``NUM_SHARDS``; the file itself, when ``01_download.sh`` staged it, is checked against the pinned values;
4. HEAR's file stem ``tfds_id_<slugify(tfds_id)>`` and its ``split_k_folds`` (the sorted stems shuffled with
   ``random.Random("split_k_folds")``, ``np.array_split`` into ``fold00`` … ``fold04``; every fold holds all six
   labels with the first seed).

Inputs (all read-only): ``<media_root>/processedResults/summaryTable.csv`` (CREMA-D @ ``1658cd3``, sha256
``fb0e3827…``), ``<media_root>/tfds/crema_d-1.0.0-dataset_info.json`` (TFDS's
``gs://tfds-data/dataset_info/crema_d/1.0.0/dataset_info.json``, sha256 ``5e748e0e…``; its shard lengths and
sizes are pinned below, so a ``--no-media`` re-freeze needs only summaryTable.csv) and, unless ``--no-media``, the
7 438 wavs ``<media_root>/AudioWAV/<FileName>.wav`` (16 kHz mono PCM_16, read in full for the TFDS sizes, plus
presence and the header facts).

Rows, in summaryTable.csv order: ``clip_id`` = ``FileName``; ``labels`` = the emotion code (third field of the
name; ``classes.csv`` = the six codes sorted, HEAR's ``labelvocabulary.csv`` order); ``split`` = the HEAR fold;
``rel_path`` = ``AudioWAV/<FileName>.wav``; extras ``actor``, ``sentence``, ``intensity`` (the name's fields),
``tfds_split`` (TFDS's own split) and ``hear_file`` (HEAR's file stem, to cross-check against a copy of HEAR's
preprocessed task). No train draw: every row of the four other folds trains each fold's probe.

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (gzip ``mtime=0``, no file name), ``classes.csv``, and the
``manifest.upstream`` block printed as JSON on stdout (paste into ``benchmark.yaml``; ``tests/test_folder.py``
re-checks the sums).

Usage::

    yavih bench prepare crema_d 01        # stage the wavs (the clone + 7 438 LFS downloads)
    python "$(yavih bench path crema_d)/make_manifest.py" --media-root "$(yavih config get crema_d.media_root)"
    python "$(yavih bench path crema_d)/make_manifest.py" --check     # re-derive, compare, write nothing
    python "$(yavih bench path crema_d)/make_manifest.py" --no-media --metadata summaryTable.csv --check

``--media-root`` defaults to the configured ``[data.crema_d] media_root`` (``YAVIH_DATA__CREMA_D__MEDIA_ROOT``).
``--no-media`` re-freezes from ``--metadata`` (summaryTable.csv) alone (no wav read). ``--check`` exits 1 when the re-derived ``manifest.csv.gz`` or ``classes.csv`` differs from the committed
files. Writing rewrites the folder, so run it in a git checkout of https://github.com/Bizilizi/yavih installed with
``pip install -e``; ``yavih bench prepare crema_d 03`` runs the full re-freeze with ``bench verify``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

TOTAL_UPSTREAM = 7442           # FileName rows of summaryTable.csv
LISTED = 7438                   # what TFDS crema_d 1.0.0 keeps
NUM_CLASSES = 6
FOLDS = ("fold00", "fold01", "fold02", "fold03", "fold04")
SAMPLE_RATE = 16000
HEAR_SAMPLE_SECONDS = 5.0       # HEAR's trim/pad length (a bridge default here, not applied to the media)

SUMMARY_SHA256 = "fb0e382700d32c0d9adc0c6815292215a36bcf912f2657048862c91e3bd2b317"
UPSTREAM_COMMIT = "1658cd342dff90010aa843eaeebd53610a08b1dc"
UPSTREAM_SOURCE = f"CheyneyComputerScience/CREMA-D @ {UPSTREAM_COMMIT[:7]} processedResults/summaryTable.csv"
# TFDS's dataset_info.json of crema_d 1.0.0 (gs://tfds-data/dataset_info/crema_d/1.0.0/dataset_info.json,
# last modified 2020-03-23): the shard count (len(shardLengths)) and the serialized bytes (numBytes) of each split.
TFDS_INFO_SHA256 = "5e748e0ecf01210ebcc1e2a4bcb80a59af47b89bfdee6521d18efa4ff0a7b6b0"
TFDS_INFO_REL = "tfds/crema_d-1.0.0-dataset_info.json"
NUM_SHARDS = {"train": 16, "validation": 2, "test": 4}
TFDS_NUM_BYTES = {"train": 1223332734, "validation": 181972381, "test": 365486261}
EXCLUDED_REASON = "skipped by TFDS crema_d 1.0.0 as absent from GitHub (hence absent from HEAR's task)"

MANIFEST_COLUMNS = ["clip_id", "labels", "split", "rel_path", "actor", "sentence", "intensity", "tfds_split", "hear_file"]
CLASSES_COLUMNS = ["index", "name"]


def _sibling(name: str):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(f"crema_d_make_manifest_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hear_split = _sibling("hear_split.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_facts(path: str) -> tuple[int, int, int, str, int]:
    """``(frames, sample_rate, channels, subtype, packed int64 payload bytes)`` of one wav, read in full."""
    import soundfile as sf

    with sf.SoundFile(path) as f:
        data = f.read(dtype="int16", always_2d=True)
        return f.frames, f.samplerate, f.channels, f.subtype, hear_split.packed_int64_payload(data.reshape(-1))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--media-root", type=Path, default=None, help="the CREMA-D clone (AudioWAV/, processedResults/); default the configured [data.crema_d] media_root")
    p.add_argument("--metadata", type=Path, default=None, help="summaryTable.csv (default: <media-root>/processedResults/summaryTable.csv)")
    p.add_argument("--no-media", action="store_true", help="read no wav: pinned NUM_SHARDS, no presence or header checks")
    p.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write (default: this folder)")
    p.add_argument("--check", action="store_true", help="re-derive and compare against the committed manifest.csv.gz and classes.csv; write nothing")
    p.add_argument("--workers", type=int, default=4, help="threads reading wavs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_media and args.metadata is None:
        raise SystemExit("--no-media needs --metadata (a copy of processedResults/summaryTable.csv)")
    media_root = None
    if not args.no_media:
        try:
            media_root = config.folder_root(HERE, "media_root", args.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root or configure the CREMA-D clone: {e}") from None
    meta_path = args.metadata or (media_root / "processedResults" / "summaryTable.csv")
    log = lambda m: print(f"[make_manifest crema_d] {m}", file=sys.stderr)  # noqa: E731

    # 1. summaryTable.csv -> the TFDS clip list.
    if not meta_path.is_file():
        raise SystemExit(f"{meta_path} not found: the manifest is derived from processedResults/summaryTable.csv")
    meta_sha = sha256_file(meta_path)
    if meta_sha != SUMMARY_SHA256:
        raise SystemExit(f"{meta_path} has sha256 {meta_sha}, expected {SUMMARY_SHA256} (CREMA-D @ {UPSTREAM_COMMIT[:7]})")
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    total = len(lines) - 1
    if total != TOTAL_UPSTREAM:
        raise SystemExit(f"{meta_path} has {total} rows, expected {TOTAL_UPSTREAM}")
    names = hear_split.summary_table_names(lines)
    if len(names) != LISTED or len(set(names)) != LISTED:
        raise SystemExit(f"{len(names)} names ({len(set(names))} distinct) kept, expected {LISTED}")
    log(f"summaryTable.csv: {total} rows, {len(names)} kept (TFDS), sha256 {meta_sha[:16]}…")

    # 2. the TFDS split, and (with media) the serialized size of each split -> its shard count.
    tfds = hear_split.tfds_splits(names)
    split_of = {n: s for s, keys in tfds.items() for n in keys}
    header_counts: Counter = Counter()
    pcm_bytes = 0
    over_5s = 0
    max_frames = 0
    sizes: dict[str, int] = defaultdict(int)
    num_shards = dict(NUM_SHARDS)
    margins: dict[str, dict] = {}
    if args.no_media:
        log(f"--no-media: pinned NUM_SHARDS {NUM_SHARDS}; presence assumed (staged == listed)")
    else:
        wav_dir = media_root / "AudioWAV"
        present = {e.name for e in os.scandir(wav_dir)} if wav_dir.is_dir() else set()
        absent = [n for n in names if f"{n}.wav" not in present]
        if absent:
            raise SystemExit(f"{len(absent)} wavs missing under {wav_dir} (e.g. {absent[:3]}): run `yavih bench prepare crema_d 01`")
        paths = [str(wav_dir / f"{n}.wav") for n in names]
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            facts = list(pool.map(wav_facts, paths))
        for n, (frames, rate, ch, subtype, payload) in zip(names, facts, strict=True):
            header_counts[(rate, ch, subtype)] += 1
            pcm_bytes += 44 + 2 * frames * ch
            max_frames = max(max_frames, frames)
            over_5s += int(frames > HEAR_SAMPLE_SECONDS * rate)
            label = hear_split.LABELS.index(hear_split.tfds_label(n))
            sizes[split_of[n]] += hear_split.example_size(frames * ch, payload, label, n.split("_")[0])
        if set(header_counts) != {(SAMPLE_RATE, 1, "PCM_16")}:
            raise SystemExit(f"wav headers {dict(header_counts)}: every CREMA-D wav must be 16 kHz mono PCM_16")
        for split, total_size in sorted(sizes.items()):
            n_ex = len(tfds[split])
            computed = hear_split._get_number_shards(total_size, n_ex)
            padded = total_size + n_ex * hear_split.TFRECORD_REC_OVERHEAD
            margins[split] = {"examples": n_ex, "serialized_bytes": total_size, "num_shards": computed,
                              "MiB_with_overhead": round(padded / (1 << 20), 1)}
            num_shards[split] = computed
        if num_shards != NUM_SHARDS or dict(sizes) != TFDS_NUM_BYTES:
            raise SystemExit(f"TFDS sizes {dict(sizes)} / shards {num_shards} differ from TFDS's dataset_info.json "
                             f"({TFDS_NUM_BYTES} / {NUM_SHARDS}): the staged wavs are not what TFDS serialized")
        info_path = media_root / TFDS_INFO_REL
        if info_path.is_file():
            if sha256_file(info_path) != TFDS_INFO_SHA256:
                raise SystemExit(f"{info_path} has sha256 {sha256_file(info_path)}, expected {TFDS_INFO_SHA256}")
            info = json.loads(info_path.read_text(encoding="utf-8"))
            pinned = {sp["name"]: (len(sp["shardLengths"]), int(sp["numBytes"])) for sp in info["splits"]}
            if pinned != {k: (NUM_SHARDS[k], TFDS_NUM_BYTES[k]) for k in NUM_SHARDS}:
                raise SystemExit(f"{info_path}: {pinned} differs from the pinned NUM_SHARDS / TFDS_NUM_BYTES")
        else:
            log(f"{info_path} not staged: compared with the pinned values only")
        log(f"wavs: {dict(header_counts)}; longest {max_frames / SAMPLE_RATE:.3f} s; {over_5s} longer than {HEAR_SAMPLE_SECONDS} s; ~{pcm_bytes / 1e9:.3f} GB PCM")
        log(f"TFDS shards: {margins}")

    # 3. tfds_id -> HEAR file stem -> HEAR fold.
    derived = hear_split.derive(names, num_shards)
    fold_counts = Counter(r["fold"] for r in derived)
    if sorted(fold_counts) != list(FOLDS) or sorted(fold_counts.values()) != [1487, 1487, 1488, 1488, 1488]:
        raise SystemExit(f"fold counts {dict(fold_counts)}: HEAR's comment says 1 488 or 1 487 per fold")
    classes = sorted({r["label"] for r in derived})
    if len(classes) != NUM_CLASSES:
        raise SystemExit(f"{len(classes)} labels, expected {NUM_CLASSES}")
    per_fold_labels = {f: {r["label"] for r in derived if r["fold"] == f} for f in FOLDS}
    if any(v != set(classes) for v in per_fold_labels.values()):
        raise SystemExit("a fold misses a label")
    actor_folds: dict[str, set] = defaultdict(set)
    for r in derived:
        actor_folds[r["wav_name"].split("_")[0]].add(r["fold"])
    actors_multi = sum(1 for v in actor_folds.values() if len(v) > 1)
    log(f"folds {dict(sorted(fold_counts.items()))}; {len(actor_folds)} actors, {actors_multi} in more than one fold")

    # 4. manifest.csv.gz + classes.csv.
    rows_out = []
    for r in derived:
        actor, sentence, emotion, intensity = r["wav_name"].split("_")
        rows_out.append([r["wav_name"], emotion, r["fold"], f"AudioWAV/{r['wav_name']}.wav", actor, sentence, intensity,
                         r["tfds_split"], r["hear_stem"]])
    manifest_data = csv_bytes(MANIFEST_COLUMNS, rows_out)
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), name] for i, name in enumerate(classes)])
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    classes_sha = hashlib.sha256(classes_data).hexdigest()
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
    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(rows_out)},
        "upstream": {
            "source": UPSTREAM_SOURCE,
            "sha256": meta_sha,
            "rows": TOTAL_UPSTREAM,
            "upstream_files": {"CREMA-D/processedResults/summaryTable.csv": meta_sha,
                               "tfds-data/dataset_info/crema_d/1.0.0/dataset_info.json": TFDS_INFO_SHA256},
            "curation": {
                "total_upstream": TOTAL_UPSTREAM,
                "per_split": {f: {"listed": fold_counts[f], "staged": fold_counts[f], "missing": 0} for f in FOLDS},
                "excluded": {EXCLUDED_REASON: TOTAL_UPSTREAM - LISTED},
                "short_clips": 0,
                "leakage_audit": {
                    "against": f"actors across HEAR's 5 folds (drawn over files, not actors): {actors_multi} of {len(actor_folds)} actors in several folds; kept, HEAR's protocol",
                    "excluded": 0,
                },
            },
        },
        "derived": {
            "num_shards": num_shards,
            "tfds_split_sizes": {s: len(v) for s, v in sorted(tfds.items())},
            "tfds_shard_margins": margins or None,
            "per_class": dict(sorted(Counter(r["label"] for r in derived).items())),
            "longest_s": round(max_frames / SAMPLE_RATE, 4) if max_frames else None,
            "longer_than_5s": over_5s if not args.no_media else None,
            "budget_media_gb_pcm": round(pcm_bytes / 1e9, 3) if pcm_bytes else None,
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
