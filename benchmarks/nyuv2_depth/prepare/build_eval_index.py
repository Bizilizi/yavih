#!/usr/bin/env python
"""Build the derived NYUv2 eval index (``eval_rgb.npy``, ``eval_depth.npy``, ``eval_meta.json``).

Deterministic: the same reads, axis transposition and row order every run, so the arrays are byte-identical to
the pinned digests the frozen ``eval_row`` column refers to:

- reads ONLY ``images`` and ``rawDepths`` from the MATLAB v7.3 (HDF5) ``nyu_depth_v2_labeled.mat``, one image
  at a time (never the whole 3 GB array); ``images`` is stored ``[N, 3, 640, 480]`` → ``transpose(2, 1, 0)``
  → ``[480, 640, 3]``; ``rawDepths`` ``[N, 640, 480]`` → ``.T`` → ``[480, 640]``. RAW Kinect depth, holes = 0
  — NEVER the inpainted ``depths`` field (scoring against it measures the inpainter);
- rows = the 654 ``test_SN40.json`` images in ASCENDING labelled index (``000003_rgb.jpg`` → index 2);
- ``.npy`` (memmap-able), not ``.npz``.

Usage::

    python build_eval_index.py [--nyuv2-dir <geonet_root>/nyuv2] --out <derived_root>

``--nyuv2-dir`` defaults to ``<geonet_root>/nyuv2`` (the configured ``geonet_root``, yavih's config resolver).

Needs numpy + h5py. Writes only under ``--out`` (``*.tmp`` then rename). Expected digests (setup.md "## Obtain the data"):
eval_rgb.npy 019589449649ab4a1e5602854f65746a70b326a4ea84f72a3e6a2dc08d732a39,
eval_depth.npy fdbc9659f04c11274b2a72bf8e79ac8069df2559603b181103d237583a401fe5.
``eval_meta.json`` additionally records ``source_mat`` (the absolute path), so its own file digest differs per
site; ``make_manifest.py`` pins its content without that field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parents[1]  # the benchmark folder (benchmark.yaml)

SOURCE_HEIGHT, SOURCE_WIDTH = 480, 640
EVAL_DEPTH_FIELD = "rawDepths"
EXPECTED_PARTS = {
    "eval_rgb.npy": "019589449649ab4a1e5602854f65746a70b326a4ea84f72a3e6a2dc08d732a39",
    "eval_depth.npy": "fdbc9659f04c11274b2a72bf8e79ac8069df2559603b181103d237583a401fe5",
}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def eval_index_digest(parts: dict[str, str]) -> str:
    """``eval_index_digest``: sha256 over sorted ``"<name>:<sha256>\\n"`` lines."""
    return hashlib.sha256("".join(f"{n}:{parts[n]}\n" for n in sorted(parts)).encode("utf-8")).hexdigest()


def read_h5_strings(h5file, key: str) -> list[str]:
    """MATLAB v7.3 cell array of strings → list."""
    return ["".join(chr(int(c)) for c in h5file[ref][:].flatten()) for ref in h5file[key][0]]


def labeled_test_indices(nyuv2_dir: Path) -> list[int]:
    """``test_SN40.json`` entries → 0-based labelled indices (``000003_rgb.jpg`` → 2)."""
    entries = json.loads((nyuv2_dir / "test_SN40.json").read_text(encoding="utf-8"))
    return [int(str(e["img"]).split("_", 1)[0]) - 1 for e in entries]


def build(nyuv2_dir: Path, out_dir: Path) -> dict:
    import h5py

    mat_path = nyuv2_dir / "nyu_depth_v2_labeled.mat"
    order = sorted(labeled_test_indices(nyuv2_dir))
    n = len(order)
    if n != 654:
        raise SystemExit(f"test_SN40.json lists {n} images, expected 654")
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path, depth_path = out_dir / "eval_rgb.npy", out_dir / "eval_depth.npy"
    rgb_tmp, depth_tmp = out_dir / "eval_rgb.npy.tmp", out_dir / "eval_depth.npy.tmp"
    try:
        rgb = np.lib.format.open_memmap(rgb_tmp, mode="w+", dtype=np.uint8, shape=(n, SOURCE_HEIGHT, SOURCE_WIDTH, 3))
        depth = np.lib.format.open_memmap(depth_tmp, mode="w+", dtype=np.float32, shape=(n, SOURCE_HEIGHT, SOURCE_WIDTH))
        with h5py.File(mat_path, "r") as fh:
            images_ds, depths_ds = fh["images"], fh[EVAL_DEPTH_FIELD]
            if images_ds.shape[1:] != (3, SOURCE_WIDTH, SOURCE_HEIGHT):
                raise SystemExit(f"unexpected 'images' shape {images_ds.shape}")
            raw_filenames = read_h5_strings(fh, "rawRgbFilenames")
            for row, index in enumerate(order):
                rgb[row] = np.transpose(images_ds[index], (2, 1, 0))
                depth[row] = np.asarray(depths_ds[index], dtype=np.float32).T
        rgb.flush()
        depth.flush()
        del rgb, depth
        os.replace(rgb_tmp, rgb_path)
        os.replace(depth_tmp, depth_path)
    finally:
        rgb_tmp.unlink(missing_ok=True)
        depth_tmp.unlink(missing_ok=True)
    parts = {"eval_rgb.npy": sha256_file(rgb_path), "eval_depth.npy": sha256_file(depth_path)}
    meta = {
        "labeled_indices": order,
        "clip_ids": [f"labeled_{i + 1:06d}" for i in order],
        "sessions": [raw_filenames[i].split("/", 1)[0] for i in order],
        "raw_rgb_filenames": [raw_filenames[i] for i in order],
        "rgb_shape": [n, SOURCE_HEIGHT, SOURCE_WIDTH, 3],
        "depth_shape": [n, SOURCE_HEIGHT, SOURCE_WIDTH],
        "depth_field": EVAL_DEPTH_FIELD,
        "depth_units": "metres",
        "depth_invalid_value": 0.0,
        "channel_order": "RGB",
        "parts": parts,
        "eval_index_sha256": eval_index_digest(parts),
        "source_mat": str(mat_path),
    }
    meta_path = out_dir / "eval_meta.json"
    tmp = out_dir / "eval_meta.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, meta_path)
    return {"parts": parts, "expected_parts_match": parts == EXPECTED_PARTS, "eval_index_sha256": meta["eval_index_sha256"],
            "eval_meta_sha256": sha256_file(meta_path), "rows": n}


def default_nyuv2_dir() -> Path:
    """``<geonet_root>/nyuv2`` from yavih's config resolver (``[data.nyuv2_depth] geonet_root``,
    ``YAVIH_DATA__NYUV2_DEPTH__GEONET_ROOT``, sites, else the benchmark.yaml default)."""
    from yavih import config  # lazy: with --nyuv2-dir given the script needs no yavih

    try:
        return config.folder_root(FOLDER, "geonet_root") / "nyuv2"
    except config.ConfigError as e:
        raise SystemExit(f"pass --nyuv2-dir or configure [data.nyuv2_depth] geonet_root: {e}") from None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nyuv2-dir", type=Path, default=None, help="dir with nyu_depth_v2_labeled.mat + test_SN40.json; default <geonet_root>/nyuv2")
    ap.add_argument("--out", type=Path, required=True)
    ns = ap.parse_args(argv)
    rep = build(ns.nyuv2_dir or default_nyuv2_dir(), ns.out)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if rep["expected_parts_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
