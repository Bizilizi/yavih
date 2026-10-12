#!/usr/bin/env python
"""Re-container the ``nyuv2_depth`` RGB frames as lossless PNG (``input_contract``; setup.md).

Why (the raw-input contract): both upstream containers hold the depth LABELS next to the pixels —
one GeoNet ``.mat`` per train frame (``img``, ``depth``, ``mask``, ``norm``, ``grid``) and the labelled set's single
``nyu_depth_v2_labeled.mat`` (``images``, ``depths``, ``rawDepths``, …) — so neither can be handed to a bridge.
This script writes the frame's RGB, and nothing else, into a PNG. The pixel values are EXACTLY the upstream
ones (no crop, no resize, no colour conversion):

- train (GeoNet ``img``, 481×641×3 float32, mean-subtracted): truncate ``[:480, :640]`` (the extra row and column
  are interpolation padding), add back the channel offsets ``(244.350, 232.338, 207.016)`` in float32, ``rint``,
  clip to ``[0, 255]``, uint8 — Probe3D's recovery; every pixel lands on an exact integer;
- eval (labelled set): row ``eval_row`` of the derived eval index ``eval_rgb.npy`` (uint8 ``[654, 480, 640, 3]`` =
  ``images[i]`` of ``nyu_depth_v2_labeled.mat`` in ascending labelled index; ``prepare/build_eval_index.py``).

Output under ``--out`` (the benchmark's ``derived_root``): ``rgb/<split>/<clip_id>.png`` (= the manifest's
``rel_media``), ``rgb_png.sha256.csv`` (``clip_id,split,rel_media,bytes,sha256,pixels_sha256``, manifest order;
its sha256 is pinned in ``benchmark.yaml`` ``data.derived`` and a gzip copy ships in the folder) and
``recontainer_png.json`` (timing, zlib version, digests).

The PNG encoder is this file (stdlib ``zlib`` + numpy; no pillow): 8-bit RGB, no ancillary chunks, every scanline
Paeth-filtered, one IDAT, ``zlib.compress(level=6)``. The file bytes therefore depend only on this script and
the zlib build (recorded); ``pixels_sha256`` = sha256 of the decoded uint8 ``[480, 640, 3]`` C-order bytes does
not depend on either and is what ``--verify`` falls back to when a different zlib produced different bytes.

Usage::

    python recontainer_png.py --split all --geonet-root <GEONET_ROOT> --eval-index <dir with eval_rgb.npy> \
        --out <DERIVED_ROOT> [--workers 16] [--limit N] [--progress]
    python recontainer_png.py --verify --out <DERIVED_ROOT> [--listing <folder>/rgb_png.sha256.csv.gz] [--limit N]

``--geonet-root`` defaults to the configured ``geonet_root`` and ``--eval-index`` to the configured
``derived_root`` (``$(yavih config get nyuv2_depth.geonet_root)`` / ``... derived_root``).

Needs numpy + scipy (train). Reads only; writes only under ``--out`` (``*.tmp`` then atomic rename).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE_HW = (480, 640)
GEONET_CHANNEL_OFFSETS = (244.350, 232.338, 207.016)
ZLIB_LEVEL = 6
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
LISTING = "rgb_png.sha256.csv"
LISTING_COLUMNS = ["clip_id", "split", "rel_media", "bytes", "sha256", "pixels_sha256"]


# ----------------------------------------------------------------------------- pixels


def geonet_rgb(img: np.ndarray) -> np.ndarray:
    """GeoNet ``img`` (481×641×3 float32, mean-subtracted) → uint8 ``[480, 640, 3]``."""
    a = np.asarray(img, dtype=np.float32)[: SOURCE_HW[0], : SOURCE_HW[1]]
    if a.shape != (*SOURCE_HW, 3):
        raise ValueError(f"GeoNet img truncates to {a.shape}, expected {(*SOURCE_HW, 3)}")
    return np.rint(a + np.asarray(GEONET_CHANNEL_OFFSETS, dtype=np.float32)).clip(0, 255).astype(np.uint8)


def load_geonet_rgb(path: str) -> np.ndarray:
    import scipy.io

    return geonet_rgb(scipy.io.loadmat(path, variable_names=["img"])["img"])


def pixels_sha256(rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- PNG codec (8-bit RGB only)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _paeth(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    a16, b16, c16 = (x.astype(np.int16) for x in (a, b, c))
    p = a16 + b16 - c16
    pa, pb, pc = np.abs(p - a16), np.abs(p - b16), np.abs(p - c16)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c)).astype(np.uint8)


def encode_png(rgb: np.ndarray, level: int = ZLIB_LEVEL) -> bytes:
    """uint8 ``[H, W, 3]`` → PNG bytes: IHDR (8-bit RGB, no interlace), one IDAT of Paeth-filtered scanlines, IEND."""
    x = np.ascontiguousarray(rgb, dtype=np.uint8)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"encode_png wants uint8 [H, W, 3], got {x.shape}")
    h, w, _ = x.shape
    raw = x.reshape(h, w * 3)
    left = np.zeros_like(raw)
    left[:, 3:] = raw[:, :-3]
    up = np.zeros_like(raw)
    up[1:] = raw[:-1]
    upleft = np.zeros_like(raw)
    upleft[1:, 3:] = raw[:-1, :-3]
    filt = (raw - _paeth(left, up, upleft)).astype(np.uint8)  # uint8 wrap-around = mod 256
    scan = np.concatenate([np.full((h, 1), 4, np.uint8), filt], axis=1)  # filter type 4 (Paeth) on every row
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(scan.tobytes(), level)) + _chunk(b"IEND", b"")


def decode_png(data: bytes) -> np.ndarray:
    """PNG bytes (8-bit RGB, non-interlaced, any of the five filter types) → uint8 ``[H, W, 3]``. Verification only."""
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG")
    pos, idat, hdr = 8, [], None
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + n]
        pos += 12 + n
        if kind == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if hdr is None or hdr[2:] != (8, 2, 0, 0, 0):
        raise ValueError(f"unsupported PNG header {hdr}")
    w, h = hdr[0], hdr[1]
    stride = w * 3
    buf = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, stride + 1)
    out = np.zeros((h, stride), np.uint8)
    prev = np.zeros(stride, np.uint8)
    for i in range(h):
        f, line = buf[i, 0], buf[i, 1:]
        if f == 0:
            row = line.copy()
        elif f == 2:
            row = (line + prev).astype(np.uint8)
        else:  # 1 Sub, 3 Average, 4 Paeth: sequential in x (bpp = 3)
            row = np.zeros(stride, np.uint8)
            li, pv = line.astype(np.int32), prev.astype(np.int32)
            for x in range(stride):
                a = int(row[x - 3]) if x >= 3 else 0
                c = int(pv[x - 3]) if x >= 3 else 0
                b = int(pv[x])
                if f == 1:
                    pred = a
                elif f == 3:
                    pred = (a + b) // 2
                else:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                row[x] = (li[x] + pred) & 0xFF
        out[i] = row
        prev = row
    return out.reshape(h, w, 3)


def decode_png_fast(data: bytes) -> np.ndarray:
    """Any 8-bit RGB PNG → uint8 ``[H, W, 3]``: pillow when importable (fast), else the pure ``decode_png``."""
    try:
        from PIL import Image  # optional: pillow decodes any PNG and is much faster

        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    except ImportError:
        return decode_png(data)


# ----------------------------------------------------------------------------- manifest / jobs


def manifest_bytes(path: Path) -> bytes:
    return gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()


def configured_root(key: str, flag: str) -> Path:
    """``[data.nyuv2_depth] <key>`` from yavih's config resolver (``--data``, ``YAVIH_DATA__NYUV2_DEPTH__<KEY>``,
    config files, sites, else the benchmark.yaml default); consulted only when ``flag`` is omitted."""
    from yavih import config  # lazy: with every path flag given the script needs no yavih

    try:
        return config.folder_root(HERE.parent, key)
    except config.ConfigError as e:
        raise SystemExit(f"pass {flag} or configure [data.nyuv2_depth] {key}: {e}") from None


def read_manifest(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(manifest_bytes(path).decode("utf-8"), newline="")))


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _train_job(args: tuple[str, str, str]) -> tuple[str, int, str, str]:
    src, out, clip_id = args
    rgb = load_geonet_rgb(src)
    data = encode_png(rgb)
    _write_atomic(Path(out), data)
    return clip_id, len(data), hashlib.sha256(data).hexdigest(), pixels_sha256(rgb)


def recontainer(ns: argparse.Namespace) -> int:
    rows = read_manifest(ns.manifest)
    splits = ["train", "eval"] if ns.split == "all" else [ns.split]
    t_all = time.time()
    report: dict = {"manifest_sha256": hashlib.sha256(manifest_bytes(ns.manifest)).hexdigest(),
                    "zlib": {"runtime": zlib.ZLIB_RUNTIME_VERSION, "compile": zlib.ZLIB_VERSION, "level": ZLIB_LEVEL},
                    "numpy": np.__version__, "splits": {}}
    listing: dict[str, list[str]] = {}
    for split in splits:
        todo = [r for r in rows if r["split"] == split]
        if ns.limit is not None:
            todo = todo[: ns.limit]
        t0 = time.time()
        if split == "eval":
            src = np.load(ns.eval_index / "eval_rgb.npy", mmap_mode="r")
            if src.shape[1:] != (*SOURCE_HW, 3) or src.dtype != np.uint8:
                raise SystemExit(f"eval_rgb.npy is {src.shape} {src.dtype}, expected [654, 480, 640, 3] uint8")
            for r in todo:
                rgb = np.asarray(src[int(r["eval_row"])])
                data = encode_png(rgb)
                _write_atomic(ns.out / r["rel_media"], data)
                listing[r["clip_id"]] = [r["clip_id"], split, r["rel_media"], str(len(data)), hashlib.sha256(data).hexdigest(), pixels_sha256(rgb)]
        else:
            jobs = [(str(ns.geonet_root / r["source_mat"]), str(ns.out / r["rel_media"]), r["clip_id"]) for r in todo]
            by_id = {r["clip_id"]: r for r in todo}
            if ns.workers > 1:
                from multiprocessing import get_context

                with get_context("fork").Pool(ns.workers) as pool:
                    results = pool.imap(_train_job, jobs, chunksize=4)
                    for k, (cid, n, sha, psha) in enumerate(results):
                        listing[cid] = [cid, split, by_id[cid]["rel_media"], str(n), sha, psha]
                        if ns.progress and (k + 1) % 1000 == 0:
                            print(f"[recontainer_png] train {k + 1}/{len(jobs)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
            else:
                for j in jobs:
                    cid, n, sha, psha = _train_job(j)
                    listing[cid] = [cid, split, by_id[cid]["rel_media"], str(n), sha, psha]
        sizes = [int(listing[r["clip_id"]][3]) for r in todo]
        report["splits"][split] = {"files": len(todo), "bytes": int(sum(sizes)), "seconds": round(time.time() - t0, 1),
                                   "workers": ns.workers if split == "train" else 1}
    order = [r["clip_id"] for r in rows if r["clip_id"] in listing]
    buf = io.StringIO(newline="")
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(LISTING_COLUMNS)
    for cid in order:
        wr.writerow(listing[cid])
    data = buf.getvalue().encode("utf-8")
    name = LISTING if ns.limit is None and ns.split == "all" else f"rgb_png.{ns.split}{'' if ns.limit is None else f'.first{ns.limit}'}.sha256.csv"
    _write_atomic(ns.out / name, data)
    report.update({"listing": name, "listing_sha256": hashlib.sha256(data).hexdigest(), "rows": len(order),
                   "seconds": round(time.time() - t_all, 1)})
    _write_atomic(ns.out / "recontainer_png.json" if name == LISTING else ns.out / (name + ".json"),
                  (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def verify(ns: argparse.Namespace) -> int:
    """Re-read every listed PNG: file sha256 must match, or (another zlib build) its decoded pixels' sha256."""
    lp = ns.listing or (ns.out / LISTING)
    raw = gzip.open(lp, "rb").read() if str(lp).endswith(".gz") else Path(lp).read_bytes()
    items = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    if ns.limit is not None:
        items = items[: ns.limit]
    t0 = time.time()
    res = {"checked": 0, "file_sha_ok": 0, "pixels_only_ok": 0, "missing": [], "bad": []}
    for it in items:
        p = ns.out / it["rel_media"]
        if not p.is_file():
            res["missing"].append(it["rel_media"])
            continue
        data = p.read_bytes()
        res["checked"] += 1
        if hashlib.sha256(data).hexdigest() == it["sha256"]:
            res["file_sha_ok"] += 1
        elif pixels_sha256(decode_png_fast(data)) == it["pixels_sha256"]:
            res["pixels_only_ok"] += 1
        else:
            res["bad"].append(it["rel_media"])
    res["listing_sha256"] = hashlib.sha256(raw).hexdigest()
    res["seconds"] = round(time.time() - t0, 1)
    missing = res.pop("missing")
    res["missing"] = len(missing)
    res["missing_first10"] = missing[:10]
    print(json.dumps(res, indent=2))
    return 0 if not res["bad"] and not missing else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=HERE.parent / "manifest.csv.gz")
    ap.add_argument("--split", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--geonet-root", type=Path, default=None, help="Probe3D layout root (nyu_geonet/); default the configured geonet_root")
    ap.add_argument("--eval-index", type=Path, default=None, help="dir with eval_rgb.npy; default the configured derived_root")
    ap.add_argument("--out", type=Path, required=True, help="derived_root to write into (rgb/<split>/<clip_id>.png)")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=None, help="first N rows per split only (listing gets a .first<N> name)")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--verify", action="store_true", help="check the written PNGs against the listing instead of writing")
    ap.add_argument("--listing", type=Path, default=None, help="--verify: listing to check against (default <out>/rgb_png.sha256.csv)")
    ns = ap.parse_args(argv)
    if not ns.verify:
        splits = ["train", "eval"] if ns.split == "all" else [ns.split]
        if ns.geonet_root is None and "train" in splits:
            ns.geonet_root = configured_root("geonet_root", "--geonet-root")
        if ns.eval_index is None and "eval" in splits:
            ns.eval_index = configured_root("derived_root", "--eval-index")
    return verify(ns) if ns.verify else recontainer(ns)


if __name__ == "__main__":
    sys.exit(main())
