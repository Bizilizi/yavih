#!/usr/bin/env python
"""Freeze the yavih ``in1k`` manifest from the public ILSVRC2012 files.

Inputs (read-only; sha256 asserted before anything is written):

- ``ILSVRC2012_devkit_t12.tar.gz`` (image-net.org): ``data/meta.mat`` maps ``ILSVRC2012_ID`` 1..1000 to the
  1 000 leaf WordNet ids (wnids), and ``data/ILSVRC2012_validation_ground_truth.txt`` gives the ``ILSVRC2012_ID``
  of val image 1..50 000. Val rows are ``val/<wnid>/ILSVRC2012_val_<8d>.JPEG``: the layout
  ``torchvision.datasets.ImageNet`` builds from the same two files.
- The train file list, 1 281 167 names ``train/<wnid>/<wnid>_<n>.JPEG``, read either from an extracted
  torchvision ImageNet root (``<media_root>/train/<wnid>/``, names only) or from the member names of
  ``ILSVRC2012_img_train.tar`` (``--train-tar``: 1 000 per-class tars, streamed once, ~148 GB read). Its sha256
  over the sorted lines is pinned, so both routes must yield the same list.

Rows: ``clip_id`` = file stem, ``label`` = wnid, ``rel_media`` relative to the torchvision root; each split is
sorted by (wnid, ``rel_media``).

Outputs (byte-identical on re-run; gzip mtime pinned to 0):

- ``manifest.csv.gz`` — ``clip_id,rel_media,label,split``: the 50 000 drawn train rows (in listing order)
  followed by all 50 000 val rows.
- ``classes.csv`` — ``index,name``: the 1 000 wnids sorted, i.e. the torchvision ``ImageFolder`` order.
- ``curation.json`` — the ``manifest.upstream.curation`` block and the input digests.

The train draw is ``yavih.freeze.class_balanced_subsample(train labels, 50000, 0)``: one seeded
``torch.Generator`` shuffles each class bucket in sorted-class order, then the buckets are round-robined until
50 000 rows (50 per class) are kept. It runs over the full sorted train list. ``torch`` and ``scipy`` (for
``meta.mat``) are needed only by this offline freezer.

Usage (offline; then commit the outputs)::

    python "$(yavih bench path in1k)/make_manifest.py" [--media-root ROOT] [--train-tar TAR] [--devkit TGZ]
        [--out DIR] [--check]

``--media-root`` defaults to the configured ``[data.in1k] media_root``; the devkit defaults to
``<media_root>/ILSVRC2012_devkit_t12.tar.gz``. ``--check`` re-derives everything in memory and exits 1 if the
committed files differ (nothing is written).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

from yavih import config
from yavih.freeze import class_balanced_subsample, csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent

# Public inputs.
DEVKIT_NAME = "ILSVRC2012_devkit_t12.tar.gz"
DEVKIT_SHA256 = "b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953"
TRAIN_LIST_SHA256 = "858b30e30ec1016a900c9e2e03bba81d7de9e049f6f9e7c8d84afe31a43d6737"  # sorted "train/<wnid>/<file>\n" lines
UPSTREAM_TRAIN = 1_281_167
UPSTREAM_VAL = 50_000
NUM_CLASSES = 1000

# Protocol constants (benchmark.yaml protocol.train_draw).
TRAIN_CAP = 50_000
TRAIN_SEED = 0
DRAWN_FROM = (
    "ILSVRC2012 train file list (1 281 167 images, sorted by wnid then file name), class-balanced draw "
    "(yavih.freeze.class_balanced_subsample), cap 50 000, seed 0"
)

COLUMNS = ["clip_id", "rel_media", "label", "split"]
_TRAIN_RE = re.compile(r"train/(n\d{8})/(n\d{8}_\d+)\.JPEG")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------------------------- inputs
def parse_devkit(devkit: bytes) -> tuple[dict[int, str], list[int]]:
    """``(ILSVRC2012_ID -> wnid for the 1 000 leaf synsets, val ground truth as ILSVRC2012_IDs)`` from the devkit."""
    from scipy.io import loadmat  # freeze-time only

    with tarfile.open(fileobj=io.BytesIO(devkit), mode="r:gz") as tf:
        meta = tf.extractfile("ILSVRC2012_devkit_t12/data/meta.mat").read()  # type: ignore[union-attr]
        gt = tf.extractfile("ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt").read()  # type: ignore[union-attr]
    synsets = loadmat(io.BytesIO(meta), squeeze_me=True)["synsets"]
    id_to_wnid = {int(s["ILSVRC2012_ID"]): str(s["WNID"]) for s in synsets if int(s["num_children"]) == 0}
    return id_to_wnid, [int(x) for x in gt.decode("ascii").split()]


def val_rel_media(id_to_wnid: dict[int, str], ground_truth: list[int]) -> list[str]:
    """``val/<wnid>/ILSVRC2012_val_<8d>.JPEG`` for val image 1..n, in file-number order."""
    return [f"val/{id_to_wnid[g]}/ILSVRC2012_val_{i:08d}.JPEG" for i, g in enumerate(ground_truth, start=1)]


def list_train_root(root: Path) -> list[str]:
    """The train file names of an extracted torchvision root (one directory listing per class; nothing is read)."""
    out: list[str] = []
    with os.scandir(root / "train") as it:
        wnids = sorted(e.name for e in it if e.is_dir())
    for w in wnids:
        with os.scandir(root / "train" / w) as it:
            out.extend(f"train/{w}/{e.name}" for e in it)
    return out


def list_train_tar(tar_path: Path) -> list[str]:
    """The train file names from ``ILSVRC2012_img_train.tar`` (outer tar of ``<wnid>.tar`` members), streamed."""
    out: list[str] = []
    with tarfile.open(tar_path, mode="r|") as outer:
        for m in outer:
            if not m.isfile() or not m.name.endswith(".tar"):
                continue
            wnid = Path(m.name).stem
            with tarfile.open(fileobj=outer.extractfile(m), mode="r|") as inner:
                out.extend(f"train/{wnid}/{Path(n.name).name}" for n in inner if n.isfile())
    return out


def train_list_sha256(train_rel: list[str]) -> str:
    return _sha256("".join(r + "\n" for r in sorted(train_rel)).encode("utf-8"))


# --------------------------------------------------------------------------------------------- freeze
def derive(train_rel: list[str], id_to_wnid: dict[int, str], ground_truth: list[int], *,
           cap: int = TRAIN_CAP, seed: int = TRAIN_SEED) -> dict[str, object]:
    """Everything the folder freezes, in memory: ``{manifest_csv, classes_csv, curation, per_class}``.

    Count and digest pins are checked by :func:`main`; this function checks only the structure, so a small
    synthetic layout runs through it too."""
    wnids = sorted(id_to_wnid.values())
    known = set(wnids)
    if len(known) != len(id_to_wnid):
        raise SystemExit("devkit maps two ILSVRC2012_IDs to one wnid")
    train: list[list[str]] = []
    for rel in sorted(train_rel):
        m = _TRAIN_RE.fullmatch(rel)
        if m is None or m.group(1) not in known or not m.group(2).startswith(m.group(1) + "_"):
            raise SystemExit(f"unexpected train file name {rel!r}")
        train.append([m.group(2), rel, m.group(1), "train"])
    if {r[2] for r in train} != known:
        raise SystemExit(f"train list covers {len({r[2] for r in train})} of {len(known)} classes")
    val = sorted(([Path(rel).stem, rel, rel.split("/")[1], "val"] for rel in val_rel_media(id_to_wnid, ground_truth)),
                 key=lambda r: (r[2], r[1]))

    keep = class_balanced_subsample([r[2] for r in train], cap, seed)
    if keep != sorted(keep):
        raise SystemExit("the draw returned unsorted indices")
    per_class: dict[str, int] = defaultdict(int)
    for i in keep:
        per_class[train[i][2]] += 1
    out_rows = [train[i] for i in keep] + val
    ids = [r[0] for r in out_rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("clip_id collision between the drawn train rows and val rows")
    curation = {
        "total_upstream": len(train) + len(val),
        "per_split": {
            "train": {"listed": len(keep), "staged": len(keep), "missing": 0},
            "val": {"listed": len(val), "staged": len(val), "missing": 0},
        },
        "excluded": {"draw_remainder": len(train) - len(keep)},
        "short_clips": 0,
        "leakage_audit": None,
    }
    return {
        "manifest_csv": csv_bytes(COLUMNS, out_rows),
        "classes_csv": csv_bytes(["index", "name"], [[str(i), w] for i, w in enumerate(wnids)]),
        "curation": curation,
        "per_class": dict(per_class),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media-root", type=Path, default=None, help="torchvision ImageNet root (read-only); default [data.in1k] media_root")
    ap.add_argument("--train-tar", type=Path, default=None, help="list train from ILSVRC2012_img_train.tar instead of <media_root>/train")
    ap.add_argument("--devkit", type=Path, default=None, help=f"{DEVKIT_NAME}; default <media_root>/{DEVKIT_NAME}")
    ap.add_argument("--out", type=Path, default=HERE, help="benchmark folder to write into (or compare with)")
    ap.add_argument("--check", action="store_true", help="compare with the files in --out instead of writing")
    ns = ap.parse_args(argv)

    root = None
    if ns.train_tar is None or ns.devkit is None:
        try:
            root = config.folder_root(HERE, "media_root", ns.media_root)
        except config.ConfigError as e:
            raise SystemExit(f"no ImageNet root: pass --media-root (or --train-tar and --devkit) or set [data.in1k] media_root ({e})") from e
    devkit = (ns.devkit or root / DEVKIT_NAME).read_bytes()  # type: ignore[operator]
    if _sha256(devkit) != DEVKIT_SHA256:
        raise SystemExit(f"{DEVKIT_NAME} sha256 {_sha256(devkit)} != pinned {DEVKIT_SHA256}")
    id_to_wnid, gt = parse_devkit(devkit)
    if (len(id_to_wnid), len(gt)) != (NUM_CLASSES, UPSTREAM_VAL):
        raise SystemExit(f"devkit lists {len(id_to_wnid)} leaf synsets and {len(gt)} val labels, expected {NUM_CLASSES} / {UPSTREAM_VAL}")
    train_rel = list_train_tar(ns.train_tar) if ns.train_tar is not None else list_train_root(root)  # type: ignore[arg-type]
    listing_sha = train_list_sha256(train_rel)
    if len(train_rel) != UPSTREAM_TRAIN or listing_sha != TRAIN_LIST_SHA256:
        raise SystemExit(f"train list: {len(train_rel)} files, sha256 {listing_sha}; expected {UPSTREAM_TRAIN}, {TRAIN_LIST_SHA256}")

    d = derive(train_rel, id_to_wnid, gt)
    per_class: dict[str, int] = d["per_class"]  # type: ignore[assignment]
    if len(per_class) != NUM_CLASSES or set(per_class.values()) != {TRAIN_CAP // NUM_CLASSES}:
        raise SystemExit(f"the draw is not 50 per class: {min(per_class.values())}..{max(per_class.values())}")
    manifest_csv: bytes = d["manifest_csv"]  # type: ignore[assignment]
    classes_csv: bytes = d["classes_csv"]  # type: ignore[assignment]
    sidecar = {
        "manifest_sha256": _sha256(manifest_csv),
        "classes_sha256": _sha256(classes_csv),
        "rows": {"train": TRAIN_CAP, "val": UPSTREAM_VAL},
        "train_draw": {"cap": TRAIN_CAP, "seed": TRAIN_SEED, "balance": "class", "drawn_from": DRAWN_FROM},
        "curation": d["curation"],
        "upstream_files": {DEVKIT_NAME: DEVKIT_SHA256, "train file list": TRAIN_LIST_SHA256},
    }
    targets = {
        "manifest.csv.gz": gzip_bytes(manifest_csv),
        "classes.csv": classes_csv,
        "curation.json": (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    }
    import torch  # reported only: the draw's generator

    report = {**sidecar, "torch": torch.__version__}
    if ns.check:
        bad = [n for n, b in targets.items() if not (ns.out / n).is_file() or (ns.out / n).read_bytes() != b]
        print(json.dumps({**report, "check": "identical" if not bad else f"differs: {bad}"}, indent=2))
        return 1 if bad else 0
    ns.out.mkdir(parents=True, exist_ok=True)
    for name, data in targets.items():
        write_if_changed(ns.out / name, data)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
