#!/usr/bin/env python
"""Full-read pre-flight of the frozen GeoNet train rows.

Why a FULL ``scipy.io.loadmat``: ``recontainer_png.py`` and ``derive_targets.py`` read with ``variable_names=[...]``,
which seeks past unrequested elements — a file truncated after ``img``/``depth`` reads clean that way (GeoNet's
``library_0006_r-1300708259.539496-3335480784.mat`` is such a file; it is not among the frozen rows). Only a full load
inflates every element and reaches a truncated tail. Do not narrow this call.

Asserts ``img (481, 641, 3) float32``, ``depth (481, 641) float32``, ``mask (481, 641) float32``.

Usage (contiguous shard ``i`` of ``n`` over the manifest's train rows; one JSON per shard)::

    python verify_readable.py shard --geonet-root <root> --shard 0 --num-shards 32 --out-dir <dir>
    python verify_readable.py merge --out-dir <dir> --num-shards 32
    python verify_readable.py shard ... --limit 200      # bounded sample (setup.md "## Prepare & verify")

Expected: 12 000 checked, 0 failures (e.g. 32 shards × 375 files, 2 CPUs each, ~71 s per shard). A damaged file is a FINDING (recorded, exit 0), not a job error; ``merge`` exits 1 on any
failure or missing shard.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED_ARRAYS = {"img": ((481, 641, 3), "float32"), "depth": ((481, 641), "float32"), "mask": ((481, 641), "float32")}


def train_rel_media(manifest: Path) -> list[str]:
    rows = csv.DictReader(io.StringIO(gzip.open(manifest, "rb").read().decode("utf-8"), newline=""))
    return [r["source_mat"] for r in rows if r["split"] == "train"]


def shard_of(items: list[str], shard: int, num_shards: int) -> list[str]:
    n = len(items)
    return items[(n * shard) // num_shards : (n * (shard + 1)) // num_shards]


def verify_mat_file(path: str) -> None:
    import scipy.io

    mat = scipy.io.loadmat(path)  # FULL load, on purpose (module docstring)
    for name, (shape, dtype) in REQUIRED_ARRAYS.items():
        if name not in mat:
            raise ValueError(f"missing array {name!r}")
        arr = mat[name]
        if tuple(arr.shape) != shape or str(arr.dtype) != dtype:
            raise ValueError(f"{name}: {tuple(arr.shape)} {arr.dtype} != {shape} {dtype}")


def cmd_shard(ns: argparse.Namespace) -> int:
    items = shard_of(train_rel_media(ns.manifest), ns.shard, ns.num_shards)
    if ns.limit is not None:
        items = items[: ns.limit]
    t0 = time.time()
    failures = []
    for rel in items:
        try:
            verify_mat_file(str(ns.geonet_root / rel))
        except Exception as e:  # noqa: BLE001 — recorded as a finding
            failures.append({"rel_media": rel, "error": f"{type(e).__name__}: {e}"})
    rep = {"shard": ns.shard, "num_shards": ns.num_shards, "checked": len(items), "passed": len(items) - len(failures), "failed": len(failures),
           "failures": failures, "seconds": round(time.time() - t0, 1), "host": socket.gethostname(), "geonet_root": str(ns.geonet_root),
           "limit": ns.limit}
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    (ns.out_dir / f"shard_{ns.shard}.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "failures"} | {"failures": failures[:5]}))
    return 0


def cmd_merge(ns: argparse.Namespace) -> int:
    missing, checked, failures = [], 0, []
    for i in range(ns.num_shards):
        p = ns.out_dir / f"shard_{i}.json"
        if not p.is_file():
            missing.append(i)
            continue
        d = json.loads(p.read_text())
        checked += d["checked"]
        failures += d["failures"]
    rep = {"num_shards": ns.num_shards, "checked": checked, "failed": len(failures), "missing_shards": missing, "failures": failures}
    print(json.dumps(rep, indent=2))
    return 1 if (missing or failures) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--manifest", type=Path, default=HERE.parent / "manifest.csv.gz")
    s.add_argument("--geonet-root", type=Path, required=True)
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--limit", type=int, default=None)
    m = sub.add_parser("merge")
    m.add_argument("--out-dir", type=Path, required=True)
    m.add_argument("--num-shards", type=int, required=True)
    ns = ap.parse_args(argv)
    return cmd_shard(ns) if ns.cmd == "shard" else cmd_merge(ns)


if __name__ == "__main__":
    sys.exit(main())
