#!/usr/bin/env python
"""Freeze the ``ssv2`` manifest from the official Something-Something-v2 label files.

Inputs (read-only; sha256 asserted before anything is written): the release's ``labels/`` directory (the labels
zip extracted), ``--labels`` or by default ``<[data.ssv2] media_root>/labels``:

- ``labels.json`` — ``{template: "index"}`` for the 174 classes (index order is also Python-sorted order:
  0 = "Approaching something with your camera" … 173 = "Wiping something off of something");
- ``train.json`` — 168 913 ``{id, label, template, placeholders}`` entries;
- ``validation.json`` — 24 777 entries of the same shape.

Each entry becomes one row ``clip_id = id``, ``rel_media = 20bn-something-something-v2/<id>.webm``,
``label = template`` with ``[``/``]`` stripped (the official convention; it equals a ``labels.json`` key).

Outputs (byte-identical on re-run; gzip mtime pinned to 0; ``\\n`` line ends, minimal CSV quoting — 12 class
names contain a comma and are quoted):

- ``manifest.csv.gz`` — ``clip_id,rel_media,label,split``: the 20 000 frozen train rows in ascending
  ``train.json`` order, followed by all 24 777 val rows in ``validation.json`` order.
- ``classes.csv`` — ``index,name`` in ``labels.json`` index order.
- ``curation.json`` — the machine copy of ``manifest.upstream.curation``, the train draw and the input hashes
  (the yaml carries the same values; the folder test compares them).

The train draw is ``yavih.freeze.class_balanced_subsample(labels, 20000, seed=0)`` over the train rows in
``train.json`` order: one ``torch.Generator`` seeded with 0 shuffles each class's rows in sorted-label order,
then one row per class is taken round-robin (sorted-label order) until 20 000 are drawn; the kept indices are
returned sorted. ``torch`` is needed only here (the folder itself is torch-free).

Usage (offline; commit the outputs)::

    python "$(yavih bench path ssv2)/make_manifest.py" [--labels ROOT/labels] [--out DIR] [--check]

``--check`` re-derives everything in memory and exits 1 if the files in ``--out`` (default: this folder) differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from yavih import config
from yavih.freeze import class_balanced_subsample, csv_bytes, gzip_bytes

HERE = Path(__file__).resolve().parent

# Official label files (Qualcomm release, labels zip extracted).
LABEL_FILES_SHA256 = {
    "labels/labels.json": "d98d8b657a5d418d7c05dbf630dedd34763025b3d764ceef19bbfd1dc654f54f",
    "labels/train.json": "b24e46090960a49d732a0d3633fd939030b4eb717e16728871a3180b8704192b",
    "labels/validation.json": "fbdd69ac72b40b9e3ed3aaf484e1575777152565caed693b0ced2d4959fd3958",
}
UPSTREAM_TRAIN = 168_913
UPSTREAM_VAL = 24_777
UPSTREAM_ROWS = UPSTREAM_TRAIN + UPSTREAM_VAL
NUM_CLASSES = 174
MEDIA_SUBDIR = "20bn-something-something-v2"

# Protocol constants (benchmark.yaml protocol.train_draw).
TRAIN_CAP = 20_000
TRAIN_SEED = 0
DRAWN_FROM = (f"official labels/train.json sha256 {LABEL_FILES_SHA256['labels/train.json']}, class-balanced draw "
              f"(yavih.freeze.class_balanced_subsample), cap {TRAIN_CAP}, seed {TRAIN_SEED}")

COLUMNS = ["clip_id", "rel_media", "label", "split"]


def default_labels() -> Path | None:
    """``<[data.ssv2] media_root>/labels``; None when the root is not configured."""
    try:
        return config.folder_root(HERE, "media_root") / "labels"
    except config.ConfigError:
        return None


def template_to_label(template: str) -> str:
    """Official SSv2 convention: the class name is the template with ``[``/``]`` stripped."""
    return template.replace("[", "").replace("]", "")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(label_map: dict[str, str], train: list[dict], val: list[dict], cap: int = TRAIN_CAP,
          seed: int = TRAIN_SEED) -> dict[str, object]:
    """Rows, classes and the draw from the parsed label JSONs (no file access, no pinned counts)."""
    names = [k for k, _ in sorted(label_map.items(), key=lambda kv: int(kv[1]))]
    if [int(label_map[n]) for n in names] != list(range(len(names))):
        raise SystemExit("labels.json indices are not contiguous 0..n-1")
    known = set(names)
    split_rows: dict[str, list[list[str]]] = {}
    for split, entries in (("train", train), ("val", val)):
        out = []
        for e in entries:
            cid, label = str(e["id"]), template_to_label(e["template"])
            if not cid.isdigit():
                raise SystemExit(f"{split}: non-numeric id {cid!r}")
            if label not in known:
                raise SystemExit(f"{split}: clip {cid} has label {label!r} outside labels.json")
            out.append([cid, f"{MEDIA_SUBDIR}/{cid}.webm", label, split])
        split_rows[split] = out
    ids = [r[0] for rows in split_rows.values() for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("a clip id appears twice across train.json / validation.json")

    train_rows = split_rows["train"]
    keep = class_balanced_subsample([r[2] for r in train_rows], cap, seed)
    if keep != sorted(keep):
        raise SystemExit("the draw did not return sorted indices")
    per_class = Counter(train_rows[i][2] for i in keep)
    out_rows = [train_rows[i] for i in keep] + split_rows["val"]
    return {
        "manifest_csv": csv_bytes(COLUMNS, out_rows),
        "classes_csv": csv_bytes(["index", "name"], [[str(i), n] for i, n in enumerate(names)]),
        "n_train": len(train_rows),
        "n_val": len(split_rows["val"]),
        "n_drawn": len(keep),
        "per_class": {n: per_class[n] for n in names},
    }


def derive(labels_dir: Path) -> dict[str, object]:
    """Everything the folder freezes, in memory, from the official ``labels/`` directory (sha256 asserted)."""
    blobs = {}
    for key, want in LABEL_FILES_SHA256.items():
        p = labels_dir / Path(key).name
        if not p.is_file():
            raise SystemExit(f"{p} not found: pass --labels <the release's labels/ dir>")
        blobs[key] = p.read_bytes()
        got = _sha256(blobs[key])
        if got != want:
            raise SystemExit(f"{p} sha256 {got} != pinned {want}; refusing to freeze from a different file")
    d = build(json.loads(blobs["labels/labels.json"]), json.loads(blobs["labels/train.json"]),
              json.loads(blobs["labels/validation.json"]))
    if (d["n_train"], d["n_val"], d["n_drawn"]) != (UPSTREAM_TRAIN, UPSTREAM_VAL, TRAIN_CAP):
        raise SystemExit(f"train/val/drawn {d['n_train']}/{d['n_val']}/{d['n_drawn']} differ from the pinned "
                         f"{UPSTREAM_TRAIN}/{UPSTREAM_VAL}/{TRAIN_CAP}")
    if len(d["per_class"]) != NUM_CLASSES or min(d["per_class"].values()) == 0:  # type: ignore[arg-type]
        raise SystemExit(f"the draw does not cover all {NUM_CLASSES} classes")
    d["curation"] = {
        "total_upstream": UPSTREAM_ROWS,
        "per_split": {
            "train": {"listed": TRAIN_CAP, "staged": TRAIN_CAP, "missing": 0},
            "val": {"listed": UPSTREAM_VAL, "staged": UPSTREAM_VAL, "missing": 0},
        },
        "excluded": {"draw_remainder": UPSTREAM_TRAIN - TRAIN_CAP},
        "short_clips": 0,
        "leakage_audit": None,
    }
    d["upstream_files"] = dict(LABEL_FILES_SHA256)
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=None, help="the release's labels/ dir; default <[data.ssv2] media_root>/labels")
    ap.add_argument("--out", type=Path, default=HERE, help="directory to write into (or compare with, --check); default this folder")
    ap.add_argument("--check", action="store_true", help="compare with the files in --out instead of writing")
    ns = ap.parse_args(argv)

    labels_dir = ns.labels or default_labels()
    if labels_dir is None:
        raise SystemExit("no ssv2 media root configured: pass --labels <the release's labels/ dir>")
    d = derive(labels_dir)
    manifest_csv: bytes = d["manifest_csv"]  # type: ignore[assignment]
    classes_csv: bytes = d["classes_csv"]  # type: ignore[assignment]
    import torch  # version reported for the draw's provenance (stdout only; not in the committed files)

    counts = sorted(d["per_class"].values())  # type: ignore[union-attr]
    committed = {
        "manifest_sha256": _sha256(manifest_csv),
        "classes_sha256": _sha256(classes_csv),
        "rows": {"train": TRAIN_CAP, "val": UPSTREAM_VAL},
        "train_draw": {"cap": TRAIN_CAP, "seed": TRAIN_SEED, "balance": "class", "drawn_from": DRAWN_FROM},
        "train_per_class": {"min": counts[0], "max": counts[-1]},
        "curation": d["curation"],
        "upstream_files": d["upstream_files"],
    }
    report = {**committed, "torch": torch.__version__}
    targets = {
        "manifest.csv.gz": gzip_bytes(manifest_csv),
        "classes.csv": classes_csv,
        "curation.json": (json.dumps(committed, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    }
    if ns.check:
        bad = [n for n, b in targets.items() if not (ns.out / n).is_file() or (ns.out / n).read_bytes() != b]
        print(json.dumps({**report, "check": "identical" if not bad else f"differs: {bad}"}, indent=2))
        return 1 if bad else 0
    ns.out.mkdir(parents=True, exist_ok=True)
    for name, data in targets.items():
        tmp = ns.out / (name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(ns.out / name)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
