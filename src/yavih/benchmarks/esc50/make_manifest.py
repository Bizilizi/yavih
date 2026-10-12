#!/usr/bin/env python
"""Freeze the yavih ``esc50`` manifest from the official ESC-50 metadata.

Inputs (all read-only):

* the official metadata ``<media_root>/meta/esc50.csv`` (``karolpiczak/ESC-50`` at commit ``33c8ce9``,
  sha256 ``ca660da6…``). The rows are derived from it in file order (fold-major, filename order within a
  fold): ``clip_id`` = filename stem, ``labels`` = ``category``, ``split`` = ``fold`` (the official fold
  1–5), ``rel_path`` = ``audio/<filename>``, plus the three columns ``src_file``, ``take``, ``esc10``
  (``role: extra``). The same file drives the fold-leakage audit: ``src_file`` groups across folds
  (4 Freesound sources / 9 clips sit in two folds, always under two different classes; pinned in
  ``KNOWN_CROSS_FOLD_SOURCES`` and kept, since the official folds are the protocol).
* the staged wavs under ``<media_root>/audio/`` for presence (one ``os.scandir``) and, from the WAV
  headers only, ``curation.short_clips`` (clips with fewer than 5.0 s × 44 100 = 220 500 frames; every
  ESC-50 clip is exactly 220 500 frames, mono, PCM_16).

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (gzip ``mtime=0``, no file name), ``classes.csv``
(``index,name`` = the sorted distinct labels), and the ``manifest.upstream`` block (hashes,
``upstream_files``, ``curation``) printed as JSON on stdout (paste into ``benchmark.yaml``;
``tests/test_folder.py`` re-checks the sums).

No train draw: ESC-50 is 5-fold cross-validation with every clip of the four training folds used, so
``protocol.train_draw`` is ``null``.

Usage::

    python "$(yavih bench path esc50)/make_manifest.py" --media-root "$(yavih config get esc50.media_root)"
    python "$(yavih bench path esc50)/make_manifest.py" --check     # re-derive, compare, write nothing

``--media-root`` defaults to the configured ``[data.esc50] media_root`` (``YAVIH_DATA__ESC50__MEDIA_ROOT``).
``--no-media`` re-freezes from ``--metadata`` alone (no presence or header checks). ``--check`` exits 1 when
the re-derived ``manifest.csv.gz`` or ``classes.csv`` differs from the committed files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from yavih import config
from yavih.freeze import csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

TOTAL_UPSTREAM = 2000           # ESC-50 release: 2 000 clips
NUM_CLASSES = 50
FOLDS = ("1", "2", "3", "4", "5")
PER_FOLD = 400
CLIP_SECONDS = 5.0
SAMPLE_RATE = 44100

# Pinned upstream hashes.
METADATA_SHA256 = "ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246"
UPSTREAM_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
UPSTREAM_SOURCE = f"karolpiczak/ESC-50 @ {UPSTREAM_COMMIT[:7]} meta/esc50.csv"
# Four Freesound source ids occur in two folds under different classes (clock_tick/clock_alarm, hen/rooster;
# 9 clips). Kept: the official folds are the protocol (setup.md "Gotchas", curation.leakage_audit).
KNOWN_CROSS_FOLD_SOURCES = ("131943", "134049", "209698", "234879")


MANIFEST_COLUMNS = ["clip_id", "labels", "split", "rel_path", "src_file", "take", "esc10"]
CLASSES_COLUMNS = ["index", "name"]
METADATA_COLUMNS = ["filename", "fold", "target", "category", "esc10", "src_file", "take"]


# ----------------------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path, header: list[str]) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != header:
            raise SystemExit(f"{path}: unexpected header {reader.fieldnames}, expected {header}")
        return list(reader)


def build_class_index(labels: list[str]) -> dict[str, int]:
    """Sorted distinct labels of the full manifest (all folds) → contiguous indices."""
    return {label: i for i, label in enumerate(sorted(set(labels)))}


def derive_from_metadata(meta: list[dict[str, str]]) -> list[dict[str, str]]:
    """Manifest rows from ``meta/esc50.csv``, file order."""
    rows = []
    for r in meta:
        filename = r["filename"].strip()
        stem = filename[:-4] if filename.endswith(".wav") else filename
        rows.append({
            "clip_id": stem, "labels": r["category"].strip(), "split": r["fold"].strip(), "rel_path": f"audio/{filename}",
            "src_file": r["src_file"].strip(), "take": r["take"].strip(), "esc10": r["esc10"].strip(),
        })
    return rows


def wav_frames(path: str) -> tuple[int, int, int]:
    import soundfile as sf  # header read only

    info = sf.info(path)
    return info.frames, info.samplerate, info.channels


# ----------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--media-root", type=Path, default=None, help="the ESC-50 clone (audio/, meta/esc50.csv); default the configured [data.esc50] media_root (YAVIH_DATA__ESC50__MEDIA_ROOT)")
    p.add_argument("--metadata", type=Path, default=None, help="meta/esc50.csv (default: <media-root>/meta/esc50.csv)")
    p.add_argument("--no-media", action="store_true", help="skip presence and WAV-header checks (re-freeze from --metadata alone)")
    p.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write (default: this folder)")
    p.add_argument("--check", action="store_true", help="re-derive and compare against the committed manifest.csv.gz and classes.csv; write nothing")
    p.add_argument("--workers", type=int, default=32, help="threads for WAV header reads (NFS latency bound)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_media and args.metadata is None:
        raise SystemExit("--no-media needs --metadata (a copy of meta/esc50.csv)")
    media_root = None
    if not args.no_media:
        try:
            media_root = config.folder_root(HERE, "media_root", args.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"pass --media-root or configure the ESC-50 clone: {e}") from None
    meta_path = args.metadata or (media_root / "meta" / "esc50.csv")
    out: Path = args.out
    log = lambda m: print(f"[make_manifest esc50] {m}", file=sys.stderr)  # noqa: E731

    # 1. the official metadata, file order.
    if not meta_path.is_file():
        raise SystemExit(f"{meta_path} not found: the manifest is derived from meta/esc50.csv")
    meta_sha = sha256_file(meta_path)
    if meta_sha != METADATA_SHA256:
        raise SystemExit(f"{meta_path} has sha256 {meta_sha}, expected {METADATA_SHA256} (upstream commit {UPSTREAM_COMMIT[:7]})")
    meta = read_csv(meta_path, METADATA_COLUMNS)
    if len(meta) != TOTAL_UPSTREAM:
        raise SystemExit(f"{meta_path} has {len(meta)} rows, expected {TOTAL_UPSTREAM}")
    upstream = derive_from_metadata(meta)
    log(f"metadata: {len(upstream)} rows, sha256 {meta_sha[:16]}…")

    # 2. invariants the protocol rests on.
    ids = [r["clip_id"] for r in upstream]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate clip_id")
    fold_counts = Counter(r["split"] for r in upstream)
    if sorted(fold_counts) != list(FOLDS) or set(fold_counts.values()) != {PER_FOLD}:
        raise SystemExit(f"fold counts {dict(fold_counts)} are not 5 × {PER_FOLD}")
    class_index = build_class_index([r["labels"] for r in upstream])
    if len(class_index) != NUM_CLASSES:
        raise SystemExit(f"{len(class_index)} distinct labels, expected {NUM_CLASSES}")
    per_fold_class = Counter((r["split"], r["labels"]) for r in upstream)
    if set(per_fold_class.values()) != {PER_FOLD // NUM_CLASSES}:
        raise SystemExit("ESC-50 is balanced 8 clips per class per fold; this manifest is not")
    # category <-> upstream `target` is a bijection, and NOT the sorted order (setup.md "Gotchas").
    cat_target = {(r["category"], r["target"]) for r in meta}
    if len(cat_target) != NUM_CLASSES or len({c for c, _ in cat_target}) != NUM_CLASSES:
        raise SystemExit("category/target is not a bijection")
    target_equals_index = all(int(t) == class_index[c] for c, t in cat_target)
    # Fold leakage audit: the folds are grouped by source recording (src_file); verify it.
    src_folds: dict[str, set[str]] = defaultdict(set)
    for r in meta:
        src_folds[r["src_file"]].add(r["fold"])
    leaking_sources = sorted(s for s, f in src_folds.items() if len(f) > 1)
    if tuple(leaking_sources) != KNOWN_CROSS_FOLD_SOURCES:
        raise SystemExit(f"src_file groups spanning several folds changed: {leaking_sources} (expected {list(KNOWN_CROSS_FOLD_SOURCES)})")
    leak_clips = [r for r in meta if r["src_file"] in KNOWN_CROSS_FOLD_SOURCES]
    leak_same_class = [s for s in leaking_sources if len({r["category"] for r in meta if r["src_file"] == s}) == 1]
    log(f"leakage audit: {len(src_folds)} src_file groups; {len(leaking_sources)} span two folds ({len(leak_clips)} clips, same-class cross-fold groups: {len(leak_same_class)}); kept, official folds")

    # 3. presence (one scandir) and WAV headers.
    missing_per_fold: Counter = Counter()
    short = 0
    pcm_bytes = 0
    header_counts: Counter = Counter()
    if args.no_media:
        log("--no-media: presence assumed (staged == listed), short_clips 0")
    else:
        audio_dir = media_root / "audio"
        present = {e.name for e in os.scandir(audio_dir)} if audio_dir.is_dir() else set()
        for r in upstream:
            if r["rel_path"].split("/", 1)[1] not in present:
                missing_per_fold[r["split"]] += 1
        if sum(missing_per_fold.values()):
            log(f"WARNING: {sum(missing_per_fold.values())} wavs missing: {dict(missing_per_fold)} (re-clone)")
        paths = [str(media_root / r["rel_path"]) for r in upstream if r["rel_path"].split("/", 1)[1] in present]
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for frames, rate, ch in pool.map(wav_frames, paths):
                header_counts[(frames, rate, ch)] += 1
                pcm_bytes += 44 + 2 * frames * ch
                if frames < round(CLIP_SECONDS * rate):
                    short += 1
        log(f"WAV headers: {dict(header_counts)}; {short} shorter than {CLIP_SECONDS} s; ~{pcm_bytes / 1e9:.3f} GB PCM")

    # 4. manifest.csv.gz + classes.csv (fixed column and row order).
    rows_out = [[r[c] for c in MANIFEST_COLUMNS] for r in upstream]
    manifest_data = csv_bytes(MANIFEST_COLUMNS, rows_out)
    classes_data = csv_bytes(CLASSES_COLUMNS, [[str(i), name] for name, i in sorted(class_index.items(), key=lambda kv: kv[1])])
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
        out.mkdir(parents=True, exist_ok=True)
        changed_m = write_if_changed(out / "manifest.csv.gz", gzip_bytes(manifest_data))
        changed_c = write_if_changed(out / "classes.csv", classes_data)
        log(f"manifest.csv.gz {'written' if changed_m else 'unchanged'} ({len(rows_out)} rows, sha256 {manifest_sha[:16]}…); classes.csv {'written' if changed_c else 'unchanged'} (sha256 {classes_sha[:16]}…)")

    # 5. the manifest.upstream block (paste into benchmark.yaml).
    report = {
        "manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(rows_out)},
        "upstream": {
            "source": UPSTREAM_SOURCE,
            "sha256": meta_sha,
            "rows": TOTAL_UPSTREAM,
            "upstream_files": {"ESC-50/meta/esc50.csv": meta_sha},
            "curation": {
                "total_upstream": TOTAL_UPSTREAM,
                "per_split": {
                    f: {"listed": fold_counts[f], "staged": fold_counts[f] - missing_per_fold[f], "missing": missing_per_fold[f]} for f in FOLDS
                },
                "excluded": {},
                "short_clips": short,
                "leakage_audit": {"against": "src_file source-recording groups across the 5 official folds (meta/esc50.csv): 4 groups / 9 clips span two folds, always under different classes; kept", "excluded": 0},
            },
        },
        "derived": {
            "src_file_groups": len(src_folds),
            "cross_fold_sources": {s: sorted({f"{r['fold']}:{r['category']}" for r in meta if r["src_file"] == s}) for s in leaking_sources},
            "upstream_target_equals_sorted_index": target_equals_index,
            "wav_headers": {f"{k[0]}x{k[1]}Hzx{k[2]}ch": v for k, v in header_counts.items()},
            "budget_media_gb_pcm": round(pcm_bytes / 1e9, 3) if pcm_bytes else None,
            "esc10_rows": sum(1 for r in meta if r["esc10"] == "True"),
            "chance_acc": 1.0 / NUM_CLASSES,
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
