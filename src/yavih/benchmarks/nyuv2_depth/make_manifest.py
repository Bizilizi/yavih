#!/usr/bin/env python
"""Freeze the ``nyuv2_depth`` manifest from the GeoNet listing and the derived eval index.

Inputs (READ-ONLY, never modified; digests asserted before anything is written):

- the GeoNet ``nyu_geonet/`` directory of Probe3D's data layout: only its LISTING is read (the sorted names of the
  30 916 ``.mat`` files; no file is opened). Pinned by ``GEONET_LISTING_SHA256`` = sha256 over the sorted names,
  one per line (``"<name>\\n"``).
- the eval index's ``eval_meta.json`` (``<derived_root>/eval_meta.json``, written by ``prepare/build_eval_index.py``
  from the public ``nyu_depth_v2_labeled.mat`` and ``test_SN40.json``): ``labeled_indices``, ``clip_ids`` and
  ``sessions`` of the 654 test images in ``eval_rgb.npy`` row order. The file also records the absolute path of the
  labelled ``.mat``, so it is pinned by ``EVAL_META_SHA256`` = sha256 over its canonical JSON WITHOUT
  ``source_mat`` (``json.dumps(..., sort_keys=True, separators=(",", ":"))``), the same at every site.

Output (byte-identical on re-run; gzip mtime pinned to 0):

- ``manifest.csv.gz`` — ``clip_id,split,rel_media,source_mat,eval_row,labeled_index,session,source``:

  =============  =================================================================================
  column         meaning
  =============  =================================================================================
  rel_media      the ONE media column (both splits): ``rgb/<split>/<clip_id>.png`` under ``derived_root``, the
                 lossless PNG re-containering written by ``prepare/recontainer_png.py`` (layout ``dir``)
  source_mat     train: ``nyu_geonet/<stem>.mat`` under ``geonet_root`` (prepare-time source of the PNG and of
                 the depth target; never handed to a bridge: it holds the depth labels); eval: empty
  eval_row       eval: 0-based row in ``eval_rgb.npy`` / ``eval_depth.npy`` / ``eval_depth_224.npy``; train: empty
  labeled_index  eval: 0-based index into ``nyu_depth_v2_labeled.mat``; train: empty
  session        NYU capture session (``rawRgbFilenames`` dir / GeoNet stem prefix): the leakage unit
  source         ``geonet`` | ``labeled``
  =============  =================================================================================

  Row order: train as the draw returns it (ascending stem), then eval by ascending labelled index.
- no ``classes.csv``: a ``grid`` readout has none (``classes: null``).
- ``curation.json`` — the ``manifest.upstream.curation`` block + digests, also printed.

The train draw: sorted listing of 30 916 → minus the 847 files of the 6 leakage-excluded sessions
(``LEAKAGE_SESSIONS``: label-less GeoNet sessions sharing a base name with a test session) → minus Probe3D's two
positional deletes (asserted to resolve to the pinned file names; both files are unreadable in full) → a pool of
30 067 files → ``yavih.freeze.class_balanced_subsample(sessions, 12000, seed=0)`` with the capture session as the
class (342 sessions drawn). The draw needs ``torch``.

Usage (offline, once; then commit the outputs)::

    python "$(yavih bench path nyuv2_depth)/make_manifest.py" \
        [--geonet-dir "$(yavih config get nyuv2_depth.geonet_root)/nyu_geonet"] \
        [--eval-meta "$(yavih config get nyuv2_depth.derived_root)/eval_meta.json"] \
        [--out <dir>] [--check]

Defaults (yavih's config resolver): ``--geonet-dir`` = ``<geonet_root>/nyu_geonet``, ``--eval-meta`` =
``<derived_root>/eval_meta.json``, ``--out`` = this folder.

``--check`` re-derives everything in memory and exits 1 if the files in ``--out`` differ (no write).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from yavih import config
from yavih.freeze import class_balanced_subsample, csv_bytes, gzip_bytes

HERE = Path(__file__).resolve().parent

#: sha256 over ``"".join(f"{name}\n" for name in sorted(nyu_geonet/*.mat))``.
GEONET_LISTING_SHA256 = "247d4ceb29fb39ef55f1dcce2a5739ddab57971787368e55255908a1ccf33ff3"
#: sha256 over the canonical JSON of ``eval_meta.json`` without ``source_mat``.
EVAL_META_SHA256 = "ff040b99a494dd7eadc874d45bc77b348d39f863fe616e02fe7dcf3f16a260ad"

# Protocol constants.
TRAIN_CAP = 12_000
TRAIN_SEED = 0
N_EVAL = 654
GEONET_TOTAL = 30_916
#: Label-less GeoNet capture sessions whose base name (``bookstore_0001``) is shared with a test session: excluded
#: wholesale although no exact session is shared (a conservative over-exclusion).
LEAKAGE_SESSIONS = ("bookstore_0001l", "bookstore_0001m", "classroom_0001b", "dining_room_0026b",
                    "living_room_0043c", "living_room_0064b")
LEAKAGE_EXCLUDED = 847
#: Probe3D's ``evals/datasets/nyu.py`` deletes these positions of ``sorted(os.listdir(nyu_geonet))``.
PROBE3D_DELETED_INDICES = {
    6919: "bedroom_0129_r-1315328302.785180-1089784518.mat",
    21181: "library_0006_r-1300708259.539496-3335480784.mat",
}
POOL = 30_067
SESSIONS_DRAWN = 342
GEONET_SUBDIR = "nyu_geonet"
_GEONET_STEM_RE = re.compile(r"^(?P<session>.+?)_r-[0-9.]+-[0-9]+$")
_SESSION_BASE_RE = re.compile(r"^(?P<base>.+_\d+)")

COLUMNS = ["clip_id", "split", "rel_media", "source_mat", "eval_row", "labeled_index", "session", "source"]
PNG_DIR = "rgb"  # rel_media = rgb/<split>/<clip_id>.png (prepare/recontainer_png.py)
LEAKAGE_AUDIT = ("the 654 labeled NYUv2 test images (test_SN40.json), compared by capture session (rawRgbFilenames), "
                 "plus a conservative base-name exclusion of 6 label-less GeoNet sessions")


def drawn_from(listing_sha256: str = GEONET_LISTING_SHA256) -> str:
    """``protocol.train_draw.drawn_from`` (hashed in benchmark.yaml; curation.json carries the same string)."""
    return (f"GeoNet nyu_geonet listing (30 916 .mat, sorted names sha256 {listing_sha256}) minus 847 "
            "leakage-excluded and 2 corrupt files = 30 067-file pool; session-balanced draw "
            "(yavih.freeze.class_balanced_subsample), cap 12 000, seed 0")


def default_root(key: str) -> Path | None:
    """The configured ``[data.nyuv2_depth] <key>``; None when unresolvable."""
    try:
        return config.folder_root(HERE, key)
    except config.ConfigError:
        return None


# --------------------------------------------------------------------------------------------- digests
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def listing_sha256(names: list[str]) -> str:
    return _sha256("".join(f"{n}\n" for n in sorted(names)).encode("utf-8"))


def eval_meta_sha256(meta: dict) -> str:
    canon = {k: v for k, v in meta.items() if k != "source_mat"}
    return _sha256(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8"))


# --------------------------------------------------------------------------------------------- the draw
def geonet_session(filename: str) -> str:
    """``basement_0001a_r-1316653580.484909-1316500621.mat`` → ``basement_0001a``."""
    stem = os.path.splitext(filename)[0]
    m = _GEONET_STEM_RE.match(stem)
    if m is None:
        raise SystemExit(f"GeoNet filename {filename!r} does not match '<session>_r-<float>-<int>.mat'")
    return m.group("session")


def session_base(session: str) -> str:
    """``bookstore_0001m`` → ``bookstore_0001``."""
    m = _SESSION_BASE_RE.match(session)
    return m.group("base") if m else session


def replay_draw(files: list[str]) -> tuple[list[str], dict[str, int]]:
    """Pool construction + draw from the sorted ``nyu_geonet`` listing; ``(drawn file names, counts)``."""
    if len(files) != GEONET_TOTAL:
        raise SystemExit(f"{len(files)} .mat files in the listing, expected {GEONET_TOTAL} (the listing differs from the frozen one)")
    for idx, name in sorted(PROBE3D_DELETED_INDICES.items()):
        if files[idx] != name:
            raise SystemExit(f"Probe3D positional delete {idx} resolves to {files[idx]!r}, not {name!r} (the listing differs from the frozen one)")
    excl = set(LEAKAGE_SESSIONS)
    usable = [n for n in files if geonet_session(n) not in excl]
    dropped = set(PROBE3D_DELETED_INDICES.values())
    pool = [n for n in usable if n not in dropped]
    sessions = [geonet_session(n) for n in pool]
    keep = class_balanced_subsample(sessions, TRAIN_CAP, TRAIN_SEED)
    counts = {"geonet_files": len(files), "leakage_excluded": len(files) - len(usable), "corrupt": len(usable) - len(pool),
              "pool": len(pool), "sessions_drawn": len({sessions[i] for i in keep})}
    want = {"geonet_files": GEONET_TOTAL, "leakage_excluded": LEAKAGE_EXCLUDED, "corrupt": len(PROBE3D_DELETED_INDICES),
            "pool": POOL, "sessions_drawn": SESSIONS_DRAWN}
    if counts != want:
        raise SystemExit(f"replayed pool counts {counts} != frozen {want}")
    return [pool[i] for i in keep], counts


# --------------------------------------------------------------------------------------------- derivation
def derive(files: list[str], meta: dict) -> dict[str, object]:
    """Everything the folder freezes, in memory. ``files``: the ``nyu_geonet`` ``.mat`` names; ``meta``: the parsed
    ``eval_meta.json``."""
    files = sorted(files)
    shas = {"nyu_geonet listing": listing_sha256(files), "eval_meta.json (without source_mat)": eval_meta_sha256(meta)}
    for name, got, want in (("nyu_geonet listing", shas["nyu_geonet listing"], GEONET_LISTING_SHA256),
                            ("eval_meta.json", shas["eval_meta.json (without source_mat)"], EVAL_META_SHA256)):
        if got != want:
            raise SystemExit(f"{name} sha256 {got} != pinned {want}; refusing to freeze from different inputs")

    drawn, counts = replay_draw(files)
    train = [(os.path.splitext(n)[0], geonet_session(n)) for n in drawn]

    # eval: eval_meta lists the 654 test images in eval_rgb.npy row order (ascending labelled index)
    lab = [int(i) for i in meta["labeled_indices"]]
    clip_ids, sessions = list(meta["clip_ids"]), list(meta["sessions"])
    if len(lab) != N_EVAL or lab != sorted(lab) or len(set(lab)) != N_EVAL or len(clip_ids) != N_EVAL or len(sessions) != N_EVAL:
        raise SystemExit("eval_meta.json does not list 654 test images in ascending labelled index")
    for li, cid in zip(lab, clip_ids, strict=True):
        if cid != f"labeled_{li + 1:06d}":
            raise SystemExit(f"eval clip_id {cid!r} != labeled_{li + 1:06d}")

    # leakage (freeze-time facts): no capture session shared by train and eval; the excluded sessions are label-less
    # namesakes of test sessions
    shared = sorted({s for _, s in train} & set(sessions))
    if shared:
        raise SystemExit(f"train/eval share capture sessions {shared[:5]}")
    test_bases = {session_base(s) for s in sessions}
    if any(session_base(s) not in test_bases or s in sessions for s in LEAKAGE_SESSIONS):
        raise SystemExit("a leakage-excluded session is not a label-less namesake of a test session")

    out_rows = [[cid, "train", f"{PNG_DIR}/train/{cid}.png", f"{GEONET_SUBDIR}/{cid}.mat", "", "", s, "geonet"] for cid, s in train]
    out_rows += [[cid, "eval", f"{PNG_DIR}/eval/{cid}.png", "", str(k), str(li), s, "labeled"]
                 for k, (cid, li, s) in enumerate(zip(clip_ids, lab, sessions, strict=True))]
    manifest_csv = csv_bytes(COLUMNS, out_rows)

    curation = {
        "total_upstream": GEONET_TOTAL + N_EVAL,  # 30 916 GeoNet files + 654 labelled test images
        "per_split": {
            "train": {"listed": TRAIN_CAP, "staged": TRAIN_CAP, "missing": 0},
            "eval": {"listed": N_EVAL, "staged": N_EVAL, "missing": 0},
        },
        "excluded": {
            "leakage": counts["leakage_excluded"],
            "corrupt": counts["corrupt"],
            "draw_remainder": counts["pool"] - TRAIN_CAP,
        },
        "short_clips": 0,
        "leakage_audit": {"against": LEAKAGE_AUDIT, "excluded": counts["leakage_excluded"]},
    }
    total = sum(v["listed"] for v in curation["per_split"].values()) + sum(curation["excluded"].values())
    if total != curation["total_upstream"]:
        raise SystemExit(f"curation sums do not hold: {total} != {curation['total_upstream']}")
    return {"manifest_csv": manifest_csv, "curation": curation, "upstream_files": shas, "counts": counts}


def read_listing(geonet_dir: Path) -> list[str]:
    return sorted(n for n in os.listdir(geonet_dir) if n.endswith(".mat"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geonet-dir", type=Path, default=None, help="the nyu_geonet directory (listing only); default <geonet_root>/nyu_geonet")
    ap.add_argument("--eval-meta", type=Path, default=None, help="eval_meta.json of the eval index; default <derived_root>/eval_meta.json")
    ap.add_argument("--out", type=Path, default=HERE, help="directory to write into (or to compare with under --check)")
    ap.add_argument("--check", action="store_true", help="compare with the files in --out instead of writing")
    ns = ap.parse_args(argv)

    geonet_dir = ns.geonet_dir
    if geonet_dir is None and (root := default_root("geonet_root")) is not None:
        geonet_dir = root / GEONET_SUBDIR
    if geonet_dir is None or not geonet_dir.is_dir():
        raise SystemExit(f"no GeoNet listing at {geonet_dir}: pass --geonet-dir or configure [data.nyuv2_depth] geonet_root")
    eval_meta = ns.eval_meta
    if eval_meta is None and (root := default_root("derived_root")) is not None:
        eval_meta = root / "eval_meta.json"
    if eval_meta is None or not eval_meta.is_file():
        raise SystemExit(f"no eval index at {eval_meta}: pass --eval-meta or configure [data.nyuv2_depth] derived_root (prepare step 02)")

    d = derive(read_listing(geonet_dir), json.loads(eval_meta.read_bytes()))
    manifest_csv: bytes = d["manifest_csv"]  # type: ignore[assignment]
    report = {
        "manifest_sha256": _sha256(manifest_csv),
        "rows": {"train": TRAIN_CAP, "eval": N_EVAL},
        "train_draw": {"cap": TRAIN_CAP, "seed": TRAIN_SEED, "balance": "session", "sessions_drawn": SESSIONS_DRAWN,
                       "drawn_from": drawn_from()},
        "curation": d["curation"],
        "upstream_files": d["upstream_files"],
    }
    targets = {"manifest.csv.gz": gzip_bytes(manifest_csv), "curation.json": (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")}
    shown = {**report, "counts": d["counts"]}
    if ns.check:
        bad = [n for n, b in targets.items() if not (ns.out / n).is_file() or (ns.out / n).read_bytes() != b]
        print(json.dumps({**shown, "check": "identical" if not bad else f"differs: {bad}"}, indent=2))
        return 1 if bad else 0
    ns.out.mkdir(parents=True, exist_ok=True)
    for name, data in targets.items():
        tmp = ns.out / (name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(ns.out / name)
    print(json.dumps(shown, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
