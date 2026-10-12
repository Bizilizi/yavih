#!/usr/bin/env python
"""Derive the dense depth targets of ``nyuv2_depth`` (``targets``; setup.md).

Writes, under ``--out`` (the benchmark's ``derived_root``):

- ``train_depth_224.npy``  float32 ``[12000, 224, 224]`` in manifest TRAIN row order (from the GeoNet ``.mat``
  ``depth`` field under ``--geonet-root``),
- ``eval_depth_224.npy``   float32 ``[654, 224, 224]`` in manifest EVAL row order (from the raw-Kinect
  ``eval_depth.npy`` of the derived eval index, row = manifest ``eval_row``),
- ``derive_targets.<split>.json`` — shape, sha256, wall time, valid-pixel statistics, source digests.

Per image, with ``invalid = 0`` (``targets``: "invalid = 0"; the grid readout re-derives validity as
``0 < d <= max_depth``):

1. truncate to ``[:480, :640]`` (GeoNet arrays are 481×641; the extra row/column is interpolated padding);
2. ``valid = 0 < d <= max_depth`` on the SOURCE (NaN → invalid);
3. keep ``protocol.target_region`` = ``[0.0, 0.125, 1.0, 0.875]`` of the full frame, i.e. columns ``[80, 560)``
   → 480×480 (Probe3D's centre square; crop THEN resize). The target raster lives in full-frame coordinates
   through that declaration (``targets``): target pixel ``(i, j)`` covers frame rows
   ``[i, i+1)·480/224`` and columns ``80 + [j, j+1)·480/224``;
4. ``depth_small = nearest-exact(d)`` to 224 and ``mask_small = nearest(valid)`` to 224 — torch's two
   DIFFERENT nearest rules (``nearest-exact``: ``src = floor((dst + 0.5) * s)``, ``nearest``:
   ``src = floor(dst * s)``, ``s = float32(480) / float32(224)``, all in float32, clamped), replicated in
   numpy and cross-checked against ``torch.nn.functional.interpolate`` with ``--check-torch``;
5. ``final = mask_small & (0 < depth_small <= max_depth)``; ``target = where(final, depth_small, 0)``.

Never bilinear: every kept value is a verbatim copy of a valid source pixel.

Usage::

    python derive_targets.py --manifest <folder>/manifest.csv.gz --split all \
        [--geonet-root <geonet_root>] [--eval-index <derived_root>] --out <derived_root> [--workers 16] [--check-torch 32]
    # a bounded sample (writes <split>_depth_224.first<N>.npy, never the pinned name):
    python derive_targets.py ... --split train --limit 500

``--geonet-root`` defaults to the configured ``geonet_root`` and ``--eval-index`` to the configured
``derived_root`` (``$(yavih config get nyuv2_depth.geonet_root)`` / ``... derived_root``).

Needs numpy + scipy (train); torch only for ``--check-torch``. Reads only; writes only under ``--out``
(``*.tmp`` then atomic rename).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SOURCE_HW = (480, 640)
CROP_MARGIN = 80
TARGET_REGION = (0.0, 0.125, 1.0, 0.875)  # benchmark.yaml protocol.target_region, full-frame fractions [y0, x0, y1, x1]
assert (TARGET_REGION[1] * SOURCE_HW[1], (1 - TARGET_REGION[3]) * SOURCE_HW[1]) == (CROP_MARGIN, CROP_MARGIN)
TARGET_SIZE = 224
MAX_DEPTH = 10.0
HERE = Path(__file__).resolve().parent


def nearest_index(n_in: int, n_out: int, *, exact: bool) -> np.ndarray:
    """torch CPU ``interpolate(mode='nearest' | 'nearest-exact')`` source indices along one axis (float32 math)."""
    scale = np.float32(n_in) / np.float32(n_out)
    d = np.arange(n_out).astype(np.float32)
    src = (d + np.float32(0.5)) * scale if exact else d * scale
    return np.minimum(np.floor(src.astype(np.float32)).astype(np.int64), n_in - 1)


IDX_EXACT = nearest_index(SOURCE_HW[0], TARGET_SIZE, exact=True)
IDX_NEAREST = nearest_index(SOURCE_HW[0], TARGET_SIZE, exact=False)


def validity(d: np.ndarray, max_depth: float = MAX_DEPTH) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (d > 0) & (d <= max_depth)


def depth_target(depth: np.ndarray, max_depth: float = MAX_DEPTH) -> np.ndarray:
    """float32 ``[H>=480, W>=640]`` metres → float32 ``[224, 224]``, invalid = 0 (module docstring steps 1-5)."""
    d = np.asarray(depth, dtype=np.float32)[: SOURCE_HW[0], : SOURCE_HW[1]]
    if d.shape != SOURCE_HW:
        raise ValueError(f"depth truncates to {d.shape}, expected {SOURCE_HW}")
    valid = validity(d, max_depth)
    d = d[:, CROP_MARGIN:-CROP_MARGIN]
    valid = valid[:, CROP_MARGIN:-CROP_MARGIN]
    small = d[IDX_EXACT][:, IDX_EXACT]
    mask = valid[IDX_NEAREST][:, IDX_NEAREST] & validity(small, max_depth)
    return np.where(mask, small, np.float32(0.0)).astype(np.float32)


def depth_target_torch(depth: np.ndarray, max_depth: float = MAX_DEPTH) -> np.ndarray:
    """Reference implementation with ``torch.nn.functional.interpolate`` (nearest-exact depth, nearest mask), invalid = 0."""
    import torch

    d = torch.from_numpy(np.array(depth, dtype=np.float32)[: SOURCE_HW[0], : SOURCE_HW[1]].copy())
    valid = (d > 0) & (d <= max_depth)
    x = d[..., CROP_MARGIN:-CROP_MARGIN]
    small = torch.nn.functional.interpolate(x[None, None].float(), size=(TARGET_SIZE, TARGET_SIZE), mode="nearest-exact")[0]
    m = valid[..., CROP_MARGIN:-CROP_MARGIN]
    mask = torch.nn.functional.interpolate(m[None, None].float(), size=(TARGET_SIZE, TARGET_SIZE), mode="nearest")[0] > 0.5
    mask = mask & (small > 0) & (small <= max_depth)
    return torch.where(mask, small, torch.zeros_like(small))[0].numpy()


def _sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def configured_root(key: str, flag: str) -> Path:
    """``[data.nyuv2_depth] <key>`` from yavih's config resolver (``--data``, ``YAVIH_DATA__NYUV2_DEPTH__<KEY>``,
    config files, sites, else the benchmark.yaml default); consulted only when ``flag`` is omitted."""
    from yavih import config  # lazy: with every path flag given the script needs no yavih

    try:
        return config.folder_root(HERE.parent, key)
    except config.ConfigError as e:
        raise SystemExit(f"pass {flag} or configure [data.nyuv2_depth] {key}: {e}") from None


def read_manifest(path: Path) -> list[dict[str, str]]:
    raw = gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))


def _load_geonet_depth(path: str) -> np.ndarray:
    import scipy.io

    return np.asarray(scipy.io.loadmat(path, variable_names=["depth"])["depth"], dtype=np.float32)


def _train_one(path: str) -> np.ndarray:
    return depth_target(_load_geonet_depth(path))


def derive_split(split: str, rows: list[dict[str, str]], ns: argparse.Namespace) -> dict:
    n = len(rows)
    name = f"{split}_depth_{TARGET_SIZE}.npy" if ns.limit is None else f"{split}_depth_{TARGET_SIZE}.first{n}.npy"
    out = ns.out / name
    tmp = ns.out / (name + ".tmp")
    t0 = time.time()
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, TARGET_SIZE, TARGET_SIZE))
    sources: dict[str, str] = {}
    try:
        if split == "eval":
            src_path = ns.eval_index / "eval_depth.npy"
            src = np.load(src_path, mmap_mode="r")
            if src.shape[1:] != SOURCE_HW or src.dtype != np.float32:
                raise SystemExit(f"{src_path}: {src.shape} {src.dtype}, expected [654, 480, 640] float32")
            sources["eval_depth.npy"] = _sha256_file(src_path)
            for k, r in enumerate(rows):
                arr[k] = depth_target(np.asarray(src[int(r["eval_row"])]))
        else:
            paths = [str(ns.geonet_root / r["source_mat"]) for r in rows]
            if ns.workers > 1:
                from multiprocessing import get_context

                with get_context("fork").Pool(ns.workers) as pool:
                    for k, t in enumerate(pool.imap(_train_one, paths, chunksize=8)):
                        arr[k] = t
                        if ns.progress and (k + 1) % 1000 == 0:
                            print(f"[derive_targets] train {k + 1}/{n} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
            else:
                for k, p in enumerate(paths):
                    arr[k] = _train_one(p)
        arr.flush()
        valid = np.empty(n, dtype=np.float64)
        dmax = 0.0
        for lo in range(0, n, 512):
            blk = np.asarray(arr[lo : lo + 512])
            valid[lo : lo + len(blk)] = (blk > 0).mean(axis=(1, 2))
            dmax = max(dmax, float(blk.max()))
        del arr
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    rep = {
        "file": name,
        "split": split,
        "shape": [n, TARGET_SIZE, TARGET_SIZE],
        "dtype": "float32",
        "sha256": _sha256_file(out),
        "bytes": out.stat().st_size,
        "seconds": round(time.time() - t0, 1),
        "workers": ns.workers if split == "train" else 1,
        "valid_fraction": {"mean": round(float(valid.mean()), 6), "min": round(float(valid.min()), 6), "max": round(float(valid.max()), 6)},
        "images_without_valid_pixel": int((valid == 0).sum()),
        "max_value": dmax,
        "rule": "truncate[:480,:640] -> target_region [0,0.125,1,0.875] = [:,80:560] -> nearest-exact depth / nearest mask to 224; valid 0<d<=10; invalid=0",
        "sources": sources,
        "manifest_sha256": ns.manifest_sha256,
    }
    (ns.out / f"derive_targets.{split}{'' if ns.limit is None else f'.first{n}'}.json").write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    return rep


def check_torch(rows_by_split: dict[str, list[dict[str, str]]], ns: argparse.Namespace, k: int) -> dict:
    """Bit-compare the numpy path with the torch reference on the first ``k`` rows of each split."""
    out = {}
    for split, rows in rows_by_split.items():
        n_eq = 0
        for r in rows[:k]:
            if split == "eval":
                d = np.asarray(np.load(ns.eval_index / "eval_depth.npy", mmap_mode="r")[int(r["eval_row"])])
            else:
                d = _load_geonet_depth(str(ns.geonet_root / r["source_mat"]))
            n_eq += bool(np.array_equal(depth_target(d), depth_target_torch(d)))
        out[split] = {"checked": min(k, len(rows)), "bit_identical": n_eq}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=HERE.parent / "manifest.csv.gz")
    ap.add_argument("--split", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--geonet-root", type=Path, default=None, help="Probe3D layout root (nyu_geonet/); default the configured geonet_root")
    ap.add_argument("--eval-index", type=Path, default=None, help="dir with eval_depth.npy; default the configured derived_root")
    ap.add_argument("--out", type=Path, required=True, help="derived_root to write into")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (writes a .first<N>.npy sample, never the pinned name)")
    ap.add_argument("--check-torch", type=int, default=0, help="bit-compare against the torch reference on N rows per split")
    ap.add_argument("--progress", action="store_true")
    ns = ap.parse_args(argv)

    if not np.array_equal(IDX_EXACT[:4], [1, 3, 5, 7]) or not np.array_equal(IDX_NEAREST[:4], [0, 2, 4, 6]):
        raise SystemExit("nearest index tables drifted")  # pinned: 480 → 224
    ns.manifest_sha256 = hashlib.sha256(gzip.open(ns.manifest, "rb").read()).hexdigest()
    rows = read_manifest(ns.manifest)
    splits = ["train", "eval"] if ns.split == "all" else [ns.split]
    if ns.geonet_root is None and "train" in splits:
        ns.geonet_root = configured_root("geonet_root", "--geonet-root")
    if ns.eval_index is None and "eval" in splits:
        ns.eval_index = configured_root("derived_root", "--eval-index")
    by_split = {s: [r for r in rows if r["split"] == s] for s in splits}
    if ns.limit is not None:
        by_split = {s: v[: ns.limit] for s, v in by_split.items()}
    ns.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"manifest_sha256": ns.manifest_sha256}
    if ns.check_torch:
        report["torch_check"] = check_torch(by_split, ns, ns.check_torch)
        bad = {s: v for s, v in report["torch_check"].items() if v["bit_identical"] != v["checked"]}
        if bad:
            print(json.dumps(report, indent=2))
            raise SystemExit(f"numpy targets differ from the torch reference: {bad}")
    for s in splits:
        report[s] = derive_split(s, by_split[s], ns)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
