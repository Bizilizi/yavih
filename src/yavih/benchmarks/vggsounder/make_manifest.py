#!/usr/bin/env python
"""Freeze the ``vggsounder`` manifest from public files (README.md, "Re-freezing the manifest").

Inputs (read only, nothing is downloaded):

- the ``vggsounder`` 1.0 package's bundled ``data/vggsounder.csv`` (``video_id,label,modality,background_music,
  static_image,voice_over``; the default variant without background-music clips, 12 014 videos) and
  ``data/classes.csv`` (``index,mid,display_name``, the 309 VGGSound classes), both sha256-pinned;
- the official VGGSound listing ``vggsound.csv`` (``ytid,start_seconds,label,split``, no header, 199 176 rows,
  sha256-pinned), the source of the probe-train draw and of the leakage audit;
- optionally the staged media tree ``<vggsound-root>/{video,audio}/<clip_id>.{mp4,wav}``, for the presence
  counts printed under ``curation.per_split`` (presence never changes the rows).

Eval rows = the package's videos in file order, minus the 6 whose YouTube id is a VGGSound-train id; each row's
labels are split by the package's ``modality`` column into the disjoint ``labels_a`` (A), ``labels_v`` (V) and
``labels_av`` (AV) cells, tokens in file order.

Train rows = ``yavih.freeze.class_balanced_subsample`` (``torch.Generator`` + ``torch.randperm`` per class in
sorted class order, round-robin, cap 20 000, seed 0) over the ``split == train`` rows of ``vggsound.csv`` in file
order (183 730 rows, clip id ``<ytid>_<start zero-padded to 6 digits>``). The draw depends on the row order of
``vggsound.csv``, hence its sha256 pin. torch is needed here (freeze time only; never by the reader or the probe).

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (gzip mtime 0, no filename; train rows then eval rows),
``classes.csv`` (``index,name,mid``: the package's ``index,mid,display_name`` normalized, order untouched) and a
printed JSON block with every value ``benchmark.yaml`` needs (sha256s, ``manifest.upstream``, ``curation``).
``--check`` compares against the committed files instead of writing (exit 1 on drift).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from yavih import config
from yavih.freeze import class_balanced_subsample, gzip_bytes

HERE = Path(__file__).resolve().parent
NAME = "vggsounder"
CAP = 20000
SEED = 0
NUM_CLASSES = 309
# clip id = <ytid>_<start6>: the join key and a path-component guard.
_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}_\d{6}$")
MANIFEST_COLUMNS = ["clip_id", "split", "label", "labels_a", "labels_v", "labels_av", "rel_video", "rel_audio"]
MODALITY_COLUMN = {"A": "labels_a", "V": "labels_v", "AV": "labels_av"}
# Rows whose stored video or audio is shorter than protocol.clip_seconds (10.0 s), measured over all 32 008 frozen
# rows by prepare/05_media_facts.py. Nothing is cut or padded for them (curation.short_clips); make_manifest.py does
# not probe media, so the census value is carried as a constant.
SHORT_CLIPS_MEASURED = 2350

# Pinned public inputs; a different file is refused rather than silently re-frozen.
VGGSOUNDER_CSV_SHA = "400979ea3dee34bed145d3fd96966e16689352df63b645c4730caf8eb2cacb06"
CLASSES_CSV_SHA = "90a5cd19345019f50d0c3540629c1f4cde33266f00c57ce0cf67cd69835993c2"
VGGSOUND_CSV_SHA = "ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8"
#: package videos (default variant, no background music) and the VGGSound-train ytid leaks among them.
PACKAGE_VIDEOS = 12014
LEAKS = 6
UPSTREAM_SOURCE = (
    "vggsounder 1.0 package data/vggsounder.csv (eval, VGGSounder(background_music=False)) "
    "+ VGGSound vggsound.csv split == train (train draw)"
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _pinned(path: Path, want: str) -> bytes:
    if not path.is_file():
        raise SystemExit(f"missing {path} (setup.md, Obtain the data)")
    raw = path.read_bytes()
    got = sha256_bytes(raw)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got} != pinned {want}")
    return raw


# ----------------------------------------------------------------------------- inputs


def read_classes(raw: bytes) -> list[tuple[int, str, str]]:
    """The package's ``classes.csv`` (``index,mid,display_name``) → ``[(index, name, mid)]`` in file order."""
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    out = [(int(r["index"]), r["display_name"], r["mid"]) for r in rows]
    if [i for i, _, _ in out] != list(range(len(out))) or len(out) != NUM_CLASSES:
        raise SystemExit(f"classes.csv: expected contiguous indices 0..{NUM_CLASSES - 1}, got {len(out)} rows")
    if len({n for _, n, _ in out}) != len(out) or len({m for _, _, m in out}) != len(out):
        raise SystemExit("classes.csv: duplicate class names or mids")
    return out


def classes_csv_bytes(classes: list[tuple[int, str, str]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["index", "name", "mid"])
    for i, name, mid in classes:
        w.writerow([i, name, mid])
    return buf.getvalue().encode("utf-8")


def read_package_rows(raw: bytes, class_names: set[str]) -> list[dict[str, str]]:
    """``vggsounder.csv`` → one row per video in file order with disjoint ``labels_a`` / ``labels_v`` /
    ``labels_av`` cells (tokens in file order, ``;``-joined)."""
    videos: dict[str, dict[str, list[str]]] = {}
    for r in csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")):
        vid = r["video_id"]
        if not _CLIP_ID_RE.match(vid):
            raise SystemExit(f"vggsounder.csv: bad video_id {vid!r}")
        if r["background_music"].strip().lower() != "false":
            raise SystemExit(f"vggsounder.csv: {vid} is a background-music clip (expected the default variant)")
        if r["modality"] not in MODALITY_COLUMN:
            raise SystemExit(f"vggsounder.csv: {vid}: unknown modality {r['modality']!r}")
        if r["label"] not in class_names:
            raise SystemExit(f"vggsounder.csv: {vid}: label {r['label']!r} is not in classes.csv")
        cells = videos.setdefault(vid, {c: [] for c in MODALITY_COLUMN.values()})
        cells[MODALITY_COLUMN[r["modality"]]].append(r["label"])
    out = []
    for vid, cells in videos.items():
        toks = [t for c in cells.values() for t in c]
        if len(set(toks)) != len(toks):
            raise SystemExit(f"vggsounder.csv: {vid}: a label carries two modalities: {toks}")
        out.append({"clip_id": vid, **{c: ";".join(v) for c, v in cells.items()}})
    return out


def read_vggsound_csv(raw: bytes) -> list[tuple[str, str, str]]:
    """``vggsound.csv`` → ``[(clip_id, label, split)]`` in file order."""
    out = []
    for row in csv.reader(io.StringIO(raw.decode("utf-8"), newline="")):
        if len(row) != 4:
            raise SystemExit(f"vggsound.csv: expected 4 columns, got {row}")
        ytid, start, label, split = (x.strip() for x in row)
        cid = f"{ytid}_{int(start):06d}"
        if not _CLIP_ID_RE.match(cid):
            raise SystemExit(f"vggsound.csv: clip id {cid!r} violates <ytid>_<start6>")
        out.append((cid, label, split))
    return out


def listing(directory: Path) -> set[str]:
    """One ``scandir`` per directory (never one ``isfile`` per row on NFS)."""
    return {e.name for e in os.scandir(directory)}


def package_data_dir() -> Path:
    """``data/`` of the installed ``vggsounder`` package."""
    try:
        import vggsounder
    except ImportError:
        raise SystemExit("vggsounder is not installed: pip install --no-deps vggsounder==1.0 (or pass --vggsounder-data)") from None
    return Path(vggsounder.__file__).resolve().parent / "data"


def default_root() -> Path | None:
    """The folder's ``media_root`` via yavih's resolver (``YAVIH_DATA__VGGSOUNDER__MEDIA_ROOT``, config
    ``[data.vggsounder] media_root``, else the yaml default ``${datasets.vggsound}``); ``None`` when unset."""
    try:
        return config.folder_root(HERE, "media_root")
    except config.ConfigError:
        return None


# ----------------------------------------------------------------------------- freeze


def freeze(vggsounder_data: Path, vggsound_csv: Path, media_root: Path | None, cap: int, seed: int) -> dict:
    classes_raw = _pinned(vggsounder_data / "classes.csv", CLASSES_CSV_SHA)
    package_raw = _pinned(vggsounder_data / "vggsounder.csv", VGGSOUNDER_CSV_SHA)
    listing_raw = _pinned(vggsound_csv, VGGSOUND_CSV_SHA)
    classes = read_classes(classes_raw)
    class_names = {n for _, n, _ in classes}

    package_rows = read_package_rows(package_raw, class_names)
    if len(package_rows) != PACKAGE_VIDEOS:
        raise SystemExit(f"vggsounder.csv: {len(package_rows)} videos, expected {PACKAGE_VIDEOS}")
    vggsound = read_vggsound_csv(listing_raw)

    # Leakage audit: no eval row may carry a VGGSound-train ytid.
    train_ytids = {cid.rsplit("_", 1)[0] for cid, _, split in vggsound if split == "train"}
    eval_rows = [r for r in package_rows if r["clip_id"].rsplit("_", 1)[0] not in train_ytids]
    leaks = len(package_rows) - len(eval_rows)
    if leaks != LEAKS:
        raise SystemExit(f"leakage audit: {leaks} package videos carry a VGGSound-train ytid, expected {LEAKS}")

    # Train: the split == train rows of vggsound.csv in file order → seeded class-balanced draw.
    train_rows = [(cid, label) for cid, label, split in vggsound if split == "train"]
    unknown = sum(1 for _, label in train_rows if label not in class_names)
    known = [(cid, label) for cid, label in train_rows if label in class_names]
    keep = class_balanced_subsample([lab for _, lab in known], cap, seed)
    drawn = [known[i] for i in keep]
    if len(drawn) != cap:
        raise SystemExit(f"draw produced {len(drawn)} rows, expected {cap} (eligible {len(known)})")
    drawn_ids = {s for s, _ in drawn}
    eval_ids = {r["clip_id"] for r in eval_rows}
    if drawn_ids & eval_ids or {s.rsplit("_", 1)[0] for s in drawn_ids} & {c.rsplit("_", 1)[0] for c in eval_ids}:
        raise SystemExit("train draw overlaps the eval population (clip or ytid)")

    # Presence (informational: rows never depend on it).
    staged: dict[str, int | None] = {"train": None, "test": None}
    if media_root is not None and (media_root / "video").is_dir() and (media_root / "audio").is_dir():
        videos, audios = listing(media_root / "video"), listing(media_root / "audio")

        def _present(ids: list[str]) -> int:
            return sum(1 for c in ids if f"{c}.mp4" in videos and f"{c}.wav" in audios)

        staged = {"train": _present([s for s, _ in drawn]), "test": _present([r["clip_id"] for r in eval_rows])}

    # Manifest bytes.
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(MANIFEST_COLUMNS)
    for stem, label in drawn:
        w.writerow([stem, "train", label, "", "", "", f"video/{stem}.mp4", f"audio/{stem}.wav"])
    for r in eval_rows:
        cid = r["clip_id"]
        w.writerow([cid, "test", "", r["labels_a"], r["labels_v"], r["labels_av"], f"video/{cid}.mp4", f"audio/{cid}.wav"])
    manifest_bytes = buf.getvalue().encode("utf-8")
    classes_bytes = classes_csv_bytes(classes)

    listed = {"train": len(drawn), "test": len(eval_rows)}
    curation = {
        "total_upstream": len(package_rows) + len(train_rows),
        "per_split": {
            s: {"listed": n, "staged": staged[s], "missing": None if staged[s] is None else n - staged[s]}
            for s, n in listed.items()
        },
        "excluded": {"leakage": leaks, "draw_remainder": len(known) - len(drawn), "train_unknown_label": unknown},
        "short_clips": SHORT_CLIPS_MEASURED,
        "leakage_audit": {"against": "VGGSound train ytids (vggsound.csv split==train)", "excluded": leaks},
    }
    assert sum(listed.values()) + sum(curation["excluded"].values()) == curation["total_upstream"], curation
    manifest_sha = sha256_bytes(manifest_bytes)
    classes_sha = sha256_bytes(classes_bytes)
    manifest_hash = sha256_bytes(b"manifest:" + manifest_sha.encode() + b"\nclasses:" + classes_sha.encode())
    return {
        "manifest_bytes": manifest_bytes,
        "classes_bytes": classes_bytes,
        "report": {
            "manifest": {"file": "manifest.csv.gz", "sha256": manifest_sha, "classes": "classes.csv", "classes_sha256": classes_sha},
            "manifest_hash": manifest_hash,
            "rows": {**listed, "total": sum(listed.values())},
            "upstream": {
                "source": UPSTREAM_SOURCE,
                "sha256": VGGSOUNDER_CSV_SHA,
                "rows": len(eval_rows) + len(train_rows),
                "upstream_files": {
                    "vggsounder/data/vggsounder.csv": VGGSOUNDER_CSV_SHA,
                    "vggsounder/data/classes.csv": CLASSES_CSV_SHA,
                    "vggsound/metadata/vggsound.csv": VGGSOUND_CSV_SHA,
                },
                "curation": curation,
            },
            "train_draw": {"cap": cap, "seed": seed, "balance": "class", "eligible_train_rows": len(known), "per_class_min_max": _min_max(drawn)},
        },
    }


def _min_max(drawn: list[tuple[str, str]]) -> list[int]:
    counts: dict[str, int] = defaultdict(int)
    for _, lab in drawn:
        counts[lab] += 1
    return [min(counts.values()), max(counts.values())]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--vggsounder-data", type=Path, default=None,
                   help="directory holding the package's vggsounder.csv and classes.csv (default: the installed package's data/)")
    p.add_argument("--vggsound-csv", type=Path, default=None,
                   help="the official vggsound.csv (default: <media_root>/metadata/vggsound.csv)")
    p.add_argument("--vggsound-root", type=Path, default=None,
                   help="VGGSound root with video/, audio/ for the presence counts (default: yavih data root resolution; optional)")
    p.add_argument("--out", type=Path, default=HERE, help="folder to write (default: this folder)")
    p.add_argument("--cap", type=int, default=CAP)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--check", action="store_true", help="compare against the committed files instead of writing (exit 1 on drift)")
    ns = p.parse_args(argv)
    data = ns.vggsounder_data or package_data_dir()
    root = ns.vggsound_root or default_root()
    listing_csv = ns.vggsound_csv or (root / "metadata" / "vggsound.csv" if root is not None else None)
    if listing_csv is None:
        raise SystemExit("no vggsound.csv: pass --vggsound-csv or configure the vggsounder media_root")
    result = freeze(data, listing_csv, root, ns.cap, ns.seed)
    gz = gzip_bytes(result["manifest_bytes"])
    if ns.check:
        targets = {HERE / "manifest.csv.gz": gz, HERE / "classes.csv": result["classes_bytes"]}
        drift = [str(path) for path, b in targets.items() if not path.is_file() or path.read_bytes() != b]
        print(json.dumps({**result["report"], "check": "ok" if not drift else "drift", "drift": drift}, indent=1))
        return 1 if drift else 0
    out = ns.out
    out.mkdir(parents=True, exist_ok=True)
    for path, b in {out / "manifest.csv.gz": gz, out / "classes.csv": result["classes_bytes"]}.items():
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(b)
        os.replace(tmp, path)
    print(json.dumps(result["report"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
