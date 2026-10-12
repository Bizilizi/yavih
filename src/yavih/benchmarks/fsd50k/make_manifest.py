#!/usr/bin/env python
"""Freeze the yavih ``fsd50k`` manifest from the official FSD50K ground truth.

Inputs (all read-only):

* the official ground truth under ``<media_root>/ground_truth/{dev,eval,vocabulary}.csv`` (Zenodo 4060432,
  ``FSD50K.ground_truth.zip``; sha256 pinned below): ``vocabulary.csv`` becomes ``classes.csv``; ``dev.csv``
  (40 966 rows, official ``train``/``val`` sub-split) and ``eval.csv`` (10 231 rows) give the rows in file order,
  labels ``,``-joined upstream and ``;``-joined in the manifest (leaf label first, as upstream lists them).
* the staged wavs under ``<media_root>/{dev_audio,eval_audio}/`` for presence (one ``os.scandir`` per
  directory) and for the per-row ``duration_s`` column (WAV headers only). ``--reuse-durations`` takes
  ``duration_s`` from the committed ``manifest.csv.gz`` instead, so only the three CSVs are needed.

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (gzip ``mtime=0``), ``classes.csv``, and the
``manifest.upstream`` block + hashes + chance level printed as JSON on stdout (to copy into ``benchmark.yaml``;
``tests/test_folder.py`` re-checks the sums). ``--check`` writes nothing and exits 1 unless the re-derived files
equal the committed ones.

Train draw: ``yavih.freeze.class_balanced_subsample`` (one ``torch.Generator`` seeded once, ``torch.randperm`` per
class in sorted-label order, round robin, result sorted) over the present dev rows in ``dev.csv`` order, keyed by
each row's FIRST label, cap 20 000, seed 0. It needs torch at freeze time only; the reader and the probe never run
it.

Usage::

    python "$(yavih bench path fsd50k)/make_manifest.py" [--media-root /path/to/fsd50k] [--check]

``--media-root`` defaults to the configured ``[data.fsd50k] media_root`` (``YAVIH_DATA__FSD50K__MEDIA_ROOT``).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from yavih import config
from yavih.freeze import class_balanced_subsample, csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

# Official release counts (FSD50K paper / Zenodo 4060432).
DEV_UPSTREAM = 40966
EVAL_UPSTREAM = 10231
TOTAL_UPSTREAM = DEV_UPSTREAM + EVAL_UPSTREAM
NUM_CLASSES = 200

# Pinned sha256 of the official ground-truth files (FSD50K.ground_truth.zip).
GROUND_TRUTH_SHA256 = {
    "vocabulary.csv": "992b6e11422101ee575ab3022d37b79b98e19407b211331c72f5d9fdd16e7174",
    "dev.csv": "ce4ab5f01baa5a52b0ec892edaf137a7560743fce3a70155f8bf975c197ded1b",
    "eval.csv": "ef861a4619691783bfceb7f43e1a395dcf969150ba4ad34dcb36b93abd26b3d6",
}

# Protocol constants of the frozen train draw (benchmark.yaml protocol.train_draw).
TRAIN_CAP = 20000
TRAIN_SEED = 0
CLIP_SECONDS = 10.0

MANIFEST_COLUMNS = ["clip_id", "labels", "split", "rel_path", "dev_split", "duration_s"]
CLASSES_COLUMNS = ["index", "name", "mid"]


# ----------------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_vocabulary(path: Path) -> list[tuple[int, str, str]]:
    """The headerless official ``index,label,mid`` rows, index-contiguous."""
    rows: list[tuple[int, str, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.reader(fh):
            if len(r) != 3:
                raise ValueError(f"{path}: vocabulary row must be index,label,mid, got {r!r}")
            rows.append((int(r[0]), r[1], r[2]))
    if [i for i, _, _ in rows] != list(range(len(rows))) or len(rows) != NUM_CLASSES:
        raise ValueError(f"{path}: expected {NUM_CLASSES} contiguous indices 0..{NUM_CLASSES - 1}")
    return rows


def rows_from_ground_truth(gt_dir: Path, label_set: set[str]) -> list[dict[str, str]]:
    """The rows of the official CSVs: ``dev.csv`` (split = its ``train``/``val`` column) then ``eval.csv``
    (split ``eval``), each in file order; labels ``;``-joined in upstream order."""
    rows: list[dict[str, str]] = []
    for name, audio_dir, fixed_split in (("dev.csv", "dev_audio", None), ("eval.csv", "eval_audio", "eval")):
        with open(gt_dir / name, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                fname = r["fname"].strip()
                if not fname.isdigit():
                    raise ValueError(f"{name}: fname {fname!r} is not a numeric Freesound id")
                labels = [x.strip() for x in r["labels"].split(",") if x.strip()]
                if not labels:
                    raise ValueError(f"{name}: clip {fname} has no label")
                unknown = [x for x in labels if x not in label_set]
                if unknown:
                    raise ValueError(f"{name}: clip {fname}: labels not in the vocabulary: {unknown}")
                split = fixed_split if fixed_split is not None else r["split"].strip()
                if split not in ("train", "val", "eval"):
                    raise ValueError(f"{name}: clip {fname}: unexpected split {split!r}")
                rows.append({"clip_id": fname, "labels": ";".join(labels), "split": split, "rel_path": f"{audio_dir}/{fname}.wav"})
    return rows


def wav_duration_seconds(path: str) -> tuple[float, int]:
    """``(frames / samplerate, approx_bytes)`` from the WAV header via soundfile (no sample data read)."""
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate, 44 + 2 * info.frames * info.channels


def committed_durations(path: Path) -> dict[str, float]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return {r["clip_id"]: float(r["duration_s"]) for r in csv.DictReader(fh)}


# ----------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--media-root", type=Path, default=None, help="extracted release root (ground_truth/, dev_audio/, eval_audio/); default the configured [data.fsd50k] media_root (YAVIH_DATA__FSD50K__MEDIA_ROOT)")
    p.add_argument("--out", type=Path, default=HERE, help="directory to write manifest.csv.gz and classes.csv to (default: this folder)")
    p.add_argument("--check", action="store_true", help="write nothing; exit 1 unless the re-derived files equal the committed ones in this folder")
    p.add_argument("--cap", type=int, default=TRAIN_CAP)
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument("--workers", type=int, default=32, help="threads for WAV header reads (network file systems are latency bound)")
    p.add_argument("--reuse-durations", action="store_true", help="take duration_s from the committed manifest.csv.gz instead of reading WAV headers (re-freeze without media)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        media_root = config.folder_root(HERE, "media_root", args.media_root)
    except config.ConfigError as e:
        raise SystemExit(f"pass --media-root or configure the FSD50K release root: {e}") from None
    log = lambda m: print(f"[make_manifest fsd50k] {m}", file=sys.stderr)  # noqa: E731

    # 1. the official ground truth, sha-pinned.
    gt_dir = media_root / "ground_truth"
    upstream_files: dict[str, str] = {}
    for name, want in GROUND_TRUTH_SHA256.items():
        path = gt_dir / name
        if not path.is_file():
            raise SystemExit(f"{path} not found: extract FSD50K.ground_truth.zip into the media root (prepare step 02)")
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{path} has sha256 {got}, expected {want} (the official Zenodo 4060432 file)")
        upstream_files[f"FSD50K.ground_truth/{name}"] = got

    # 2. vocabulary → classes.csv (the index column IS the logit order, never re-derived).
    vocab = read_vocabulary(gt_dir / "vocabulary.csv")
    label_set = {name for _, name, _ in vocab}
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), name, mid] for i, name, mid in vocab])

    # 3. the upstream rows: dev.csv then eval.csv, file order.
    upstream = rows_from_ground_truth(gt_dir, label_set)
    dev_rows = [r for r in upstream if r["split"] in ("train", "val")]
    eval_rows = [r for r in upstream if r["split"] == "eval"]
    if len(dev_rows) != DEV_UPSTREAM or len(eval_rows) != EVAL_UPSTREAM:
        raise SystemExit(f"split counts dev={len(dev_rows)} eval={len(eval_rows)} differ from the release")
    log(f"ground truth: {len(vocab)} classes, {len(dev_rows)} dev + {len(eval_rows)} eval rows")

    # 4. presence (one scandir per directory).
    durations: dict[str, float] = {}
    if args.reuse_durations:
        durations = committed_durations(HERE / "manifest.csv.gz")
    present: dict[str, set[str]] = {}
    for d in ("dev_audio", "eval_audio"):
        p = media_root / d
        present[d] = {e.name for e in os.scandir(p)} if p.is_dir() else set()
        log(f"{d}/: {len(present[d])} files")
    if args.reuse_durations and not any(present.values()):
        # no media staged (a CSV-only re-freeze): every upstream wav counts as present
        log("no media staged: every upstream row counts as present")
        present = {"dev_audio": {f"{r['clip_id']}.wav" for r in dev_rows}, "eval_audio": {f"{r['clip_id']}.wav" for r in eval_rows}}

    def is_present(rel: str) -> bool:
        d, _, name = rel.partition("/")
        return name in present.get(d, ())

    dev_present = [r for r in dev_rows if is_present(r["rel_path"])]
    eval_present = [r for r in eval_rows if is_present(r["rel_path"])]
    dev_missing = len(dev_rows) - len(dev_present)
    eval_missing = len(eval_rows) - len(eval_present)
    if dev_missing > 0.5 * len(dev_rows) or eval_missing > 0.5 * len(eval_rows):
        raise SystemExit("more than half of a split's wavs are missing: the archives are not extracted (refusing)")
    if dev_missing or eval_missing:
        log(f"WARNING: {dev_missing} dev / {eval_missing} eval wavs missing — the draw population differs from the frozen one (recorded in curation)")

    # 5. the frozen train draw: first-label-balanced round robin over the PRESENT dev rows.
    first_labels = [r["labels"].split(";")[0] for r in dev_present]
    keep = class_balanced_subsample(first_labels, args.cap, args.seed)
    train_rows = [dev_present[i] for i in keep]
    draw_remainder = len(dev_present) - len(train_rows)
    log(f"train draw: {len(train_rows)} of {len(dev_present)} dev rows (seed {args.seed}, first-label balanced); remainder {draw_remainder}")

    frozen = [(r, "train") for r in train_rows] + [(r, "eval") for r in eval_present]

    # 6. durations (WAV headers) for duration_s and curation.short_clips.
    approx_bytes = 0
    if args.reuse_durations:
        missing_d = [r["clip_id"] for r, _ in frozen if r["clip_id"] not in durations]
        if missing_d:
            raise SystemExit(f"--reuse-durations: {len(missing_d)} frozen rows have no duration in the committed manifest")
    else:
        paths = [(r["clip_id"], str(media_root / r["rel_path"])) for r, _ in frozen]
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for (cid, _), (dur, nbytes) in zip(paths, pool.map(lambda cp: wav_duration_seconds(cp[1]), paths), strict=True):
                durations[cid] = dur
                approx_bytes += nbytes
        log(f"durations read for {len(paths)} wavs (~{approx_bytes / 1e9:.1f} GB of PCM)")

    # 7. manifest.csv.gz + classes.csv (fixed column and row order).
    rows_out = [
        [r["clip_id"], r["labels"], split, r["rel_path"], r["split"] if split == "train" else "", f"{durations[r['clip_id']]:.3f}"]
        for r, split in frozen
    ]
    manifest_data = csv_bytes(MANIFEST_COLUMNS, rows_out)
    manifest_sha = hashlib.sha256(manifest_data).hexdigest()
    classes_sha = hashlib.sha256(classes_data).hexdigest()
    if args.check:
        with gzip.open(HERE / "manifest.csv.gz", "rb") as fh:
            same_m = fh.read() == manifest_data
        same_c = (HERE / "classes.csv").read_bytes() == classes_data
        log(f"--check: manifest.csv.gz {'identical' if same_m else 'DIFFERS'}, classes.csv {'identical' if same_c else 'DIFFERS'}")
        if not (same_m and same_c):
            return 1
    else:
        out: Path = args.out
        out.mkdir(parents=True, exist_ok=True)
        changed_m = write_if_changed(out / "manifest.csv.gz", gzip_bytes(manifest_data))
        changed_c = write_if_changed(out / "classes.csv", classes_data)
        log(f"manifest.csv.gz {'written' if changed_m else 'unchanged'} ({len(rows_out)} rows, sha256 {manifest_sha[:16]}…); classes.csv {'written' if changed_c else 'unchanged'}")

    # 8. the upstream block and the derived constants (copy into benchmark.yaml).
    n_train, n_eval = len(train_rows), len(eval_present)
    # Counted on the 3-decimal value the manifest carries, so tests can re-derive short_clips from the
    # column (6 clips within 1 ms of 10 s count as full; they get <= 44 pad samples).
    shown = {r["clip_id"]: float(f"{durations[r['clip_id']]:.3f}") for r, _ in frozen}
    short = sum(1 for d in shown.values() if d < CLIP_SECONDS)
    long_ = sum(1 for d in shown.values() if d > CLIP_SECONDS)
    eval_pos = [0] * NUM_CLASSES
    name_to_idx = {name: i for i, name, _ in vocab}
    for r in eval_present:
        for lab in set(r["labels"].split(";")):
            eval_pos[name_to_idx[lab]] += 1
    chance_map = sum(eval_pos) / (NUM_CLASSES * n_eval)  # macro AP of a random scorer ≈ mean class prevalence
    train_first = defaultdict(int)
    for r in train_rows:
        train_first[r["labels"].split(";")[0]] += 1
    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(rows_out)},
        "upstream": {
            "source": "FSD50K.ground_truth/dev.csv + eval.csv (Zenodo 4060432)",
            "sha256": upstream_files["FSD50K.ground_truth/dev.csv"],
            "rows": TOTAL_UPSTREAM,
            "upstream_files": upstream_files,
            "curation": {
                "total_upstream": TOTAL_UPSTREAM,
                "per_split": {
                    "train": {"listed": n_train, "staged": n_train, "missing": 0},
                    "eval": {"listed": n_eval, "staged": n_eval, "missing": eval_missing},
                },
                "excluded": {"draw_remainder": draw_remainder, **({"dev_missing": dev_missing} if dev_missing else {})},
                "short_clips": short,
                "leakage_audit": None,
            },
        },
        "derived": {
            "long_clips": long_,
            "budget_media_gb_pcm_estimate": round(approx_bytes / 1e9, 2) if approx_bytes else None,
            "chance_map_mean_eval_prevalence": round(chance_map, 6),
            "effective_pos_classes_eval": sum(1 for c in eval_pos if c > 0),
            "min_eval_positives": min(eval_pos),
            "train_first_label_classes": len(train_first),
            "train_first_label_min_max": [min(train_first.values()), max(train_first.values())],
            "train_dev_split_counts": {s: sum(1 for r in train_rows if r["split"] == s) for s in ("train", "val")},
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
