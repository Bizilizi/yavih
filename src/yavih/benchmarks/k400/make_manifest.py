#!/usr/bin/env python
"""Freeze the ``k400`` manifest from the public Academic-Torrents Kinetics-400 ``_256`` release.

Inputs (read only, nothing is downloaded):

- the file listing of the release, either the staged tree ``<media_root>/{train,val}/<class>/<file>``
  (``prepare/02_stage.sh``; default: the configured ``[data.k400] media_root``) or, with ``--zips DIR``, the two
  torrent zips themselves (``kinetics400.zip``: ``kinetics400/train_256/<class>/<file>``;
  ``kinetics400_validation.zip``: ``val_256/<class>/<file>``; only their central directories are read). The
  listing of all 260 022 train + val files is pinned by sha256 (``TREE_SHA256``) and by its container census
  (``DISK_CENSUS``): a different listing is a different media set and is refused, not silently re-frozen.
- ``classes.csv`` of this folder (``index,name``, committed): the class order, i.e. the en_US-locale
  ``ls val | sort`` order of the release's class directories (15 indices differ from Python ``sorted``). Its
  class set must equal the listing's class directories.
- optionally (``--annotations DIR``) the Kinetics-400 annotation CSVs ``train.csv`` and ``val.csv``
  (https://s3.amazonaws.com/kinetics/400/annotations/, 246 534 + 19 906 rows; sha256-pinned in
  ``ANNOTATION_SHA``): every frozen row must be an annotated clip of its split carrying the same label.

Rows. Every ``*.mp4`` of ``train/`` and ``val/`` is a candidate row with ``clip_id`` = file stem and ``label`` =
class directory; within a split the order is class (``classes.csv`` order), then file name (Python ``sorted``). The
5 912 ``.mkv``/``.webm`` files of ``val/`` are not frozen and are recorded as excluded. Train = the seeded
class-balanced draw ``yavih.freeze.class_balanced_subsample(labels, 20000, seed 0)`` over all 240 618 train rows in
that order (one ``torch.Generator``, per-class ``randperm`` in sorted label order, round-robin, kept indices
ascending: exactly 50 per class); val = all 13 492 val rows. The frozen manifest is the drawn train rows, then the
val rows. torch is a freeze-time dependency of this script only.

Output, byte-identical on re-run: ``manifest.csv.gz`` (gzip mtime 0, no file name, level 9). A JSON block with the
values ``benchmark.yaml`` records (sha256s, ``manifest.upstream`` incl. ``curation``, ``train_draw``) is printed.
``--check`` compares with the committed file instead of writing (exit 1 on drift). ``--durations CSV`` probes every
frozen file's header (PyAV) for ``curation.short_clips`` and ``budget.media_gb``; it needs the staged tree.
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
import zipfile
from collections import Counter
from pathlib import Path

from yavih import config
from yavih.freeze import class_balanced_subsample, gzip_bytes

HERE = Path(__file__).resolve().parent
NAME = "k400"
CAP = 20000
SEED = 0
NUM_CLASSES = 400
SPLITS = ("train", "val")
MANIFEST_COLUMNS = ["clip_id", "split", "label", "rel_media"]

TORRENT = "184d11318372f70018cf9a72ef867e2fb9ce1d26"
#: Where each split sits inside the torrent zips.
ZIP_LAYOUT = {
    "train": ("kinetics400.zip", "kinetics400/train_256/"),
    "val": ("kinetics400_validation.zip", "val_256/"),
}
#: sha256 of the sorted listing ``<split>/<class>/<file>\n`` of every file under train/ and val/ (260 022 lines).
TREE_SHA256 = "c9a7dfd4e9119aa4a033a28e14e2be2bfa22a73b4bcbc6bdce5321bb3baf39a5"
#: Container census of that listing. Only ``*.mp4`` are frozen.
DISK_CENSUS = {
    "train": {".mp4": 240618},
    "val": {".mkv": 5402, ".mp4": 13492, ".webm": 510},
}
#: Kinetics-400 annotation CSVs used by ``--annotations`` (https://s3.amazonaws.com/kinetics/400/annotations/).
ANNOTATION_URL = "https://s3.amazonaws.com/kinetics/400/annotations/"
ANNOTATION_SHA = {
    "train.csv": "a5f585f790f35c080aa13964f14de63171e909575feb843efc1c29bb5cfc6236",
    "val.csv": "358eaf47e7f80ebf9b17d49eb0635ad5e0fdab98a9cbd75ffdd2ee5d5e5b6944",
}
# Duration census of the 33 492 frozen files (PyAV header, video-stream duration, ``--durations``): clips shorter than
# protocol.clip_seconds = 10.0 s. Recorded as ``curation.short_clips`` (train + val); they are kept, and how a bridge
# serves times past a short clip's end is its declared ``edge`` rule.
SHORT_CLIPS = {"train": 2756, "val": 1799}

_SFX = re.compile(r"^(.*)_(\d{6})_(\d{6})$")  # train: <ytid>_<start6>_<end6>
_YT = re.compile(r"^youtube-(.+)-(\d+)-(\d+)$")  # 36 files: youtube-<ytid>-<start>-<end>
_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_.#:-]+$")
_CLASS_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------- inputs


def read_classes(path: Path) -> list[str]:
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"), newline="")))
    if not rows or list(rows[0].keys()) != ["index", "name"]:
        raise SystemExit(f"{path}: expected header index,name")
    names = [r["name"] for r in rows]
    if [int(r["index"]) for r in rows] != list(range(NUM_CLASSES)) or len(set(names)) != NUM_CLASSES:
        raise SystemExit(f"{path}: expected {NUM_CLASSES} unique names with contiguous indices")
    return names


Listing = dict[str, dict[str, list[str]]]  # split -> class dir -> sorted file names


def list_tree(media_root: Path) -> Listing:
    """One scandir per class dir (NFS-friendly): ``{split: {class: sorted file names}}``."""
    out: Listing = {}
    for split in SPLITS:
        base = media_root / split
        if not base.is_dir():
            raise SystemExit(f"media root lacks {base} (stage K400 first: yavih bench prepare k400 02)")
        d: dict[str, list[str]] = {}
        for e in os.scandir(base):
            if not e.is_dir():
                raise SystemExit(f"{base}: unexpected non-directory {e.name!r}")
            d[e.name] = sorted(x.name for x in os.scandir(e.path))
        out[split] = d
    return out


def list_zips(zip_dir: Path) -> Listing:
    """The same listing from the torrent zips' central directories (no extraction)."""
    out: Listing = {s: {} for s in SPLITS}
    for split, (zname, prefix) in ZIP_LAYOUT.items():
        p = zip_dir / zname
        if not p.is_file():
            raise SystemExit(f"missing {p} (download it first: yavih bench prepare k400 01)")
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if not name.startswith(prefix) or name.endswith("/"):
                    continue
                parts = name[len(prefix):].split("/")
                if len(parts) != 2:
                    raise SystemExit(f"{zname}: unexpected member {name!r}")
                out[split].setdefault(parts[0], []).append(parts[1])
    return {s: {c: sorted(fs) for c, fs in d.items()} for s, d in out.items()}


def listing_sha256(listing: Listing) -> str:
    lines = sorted(f"{s}/{c}/{f}" for s, d in listing.items() for c, fs in d.items() for f in fs)
    return sha256_bytes("".join(line + "\n" for line in lines).encode("utf-8"))


def census(listing: Listing) -> dict[str, dict[str, int]]:
    return {s: dict(sorted(Counter(os.path.splitext(f)[1] for fs in d.values() for f in fs).items())) for s, d in listing.items()}


def ytid(clip_id: str) -> tuple[str, int | None, int | None]:
    """``(ytid, start, end)`` of a release file stem: ``<ytid>_<start6>_<end6>`` (train), ``youtube-<ytid>-<s>-<e>``
    (36 files) or the bare ytid (val)."""
    m = _YT.match(clip_id) or _SFX.match(clip_id)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (clip_id, None, None)


def norm_ytid(y: str) -> str:
    """The release's file-name normalization of YouTube ids: leading ``-``/``_`` stripped, separator runs collapsed."""
    return re.sub(r"[-_]+", "_", y.strip("-_"))


def class_dir(label: str) -> str:
    """Annotation label → release class directory: space → ``_``; ``(``, ``)``, ``'`` → ``-``."""
    return re.sub(r"[ ()']", lambda m: "_" if m.group() == " " else "-", label)


def read_annotations(ann_dir: Path) -> dict[str, dict]:
    """``{split: {"clip": {(norm ytid, start, end): {labels}}, "ytid": {norm ytid: {labels}}}}`` of the pinned CSVs."""
    out: dict[str, dict] = {}
    for split, fname in (("train", "train.csv"), ("val", "val.csv")):
        p = ann_dir / fname
        if not p.is_file():
            raise SystemExit(f"missing {p} (download {ANNOTATION_URL}{fname})")
        b = p.read_bytes()
        if sha256_bytes(b) != ANNOTATION_SHA[fname]:
            raise SystemExit(f"{p}: sha256 {sha256_bytes(b)} != pinned {ANNOTATION_SHA[fname]}")
        clip: dict = {}
        yt: dict = {}
        for r in csv.DictReader(io.StringIO(b.decode("utf-8"), newline="")):
            y, lab = norm_ytid(r["youtube_id"]), class_dir(r["label"])
            clip.setdefault((y, int(r["time_start"]), int(r["time_end"])), set()).add(lab)
            yt.setdefault(y, set()).add(lab)
        out[split] = {"clip": clip, "ytid": yt}
    return out


def annotation_check(rows: list[dict[str, str]], ann: dict[str, dict]) -> dict:
    """Every frozen row is an annotated clip of its split with the same label (train: by ytid + times)."""
    unmatched: list[str] = []
    mislabelled: list[str] = []
    for r in rows:
        y, s, e = ytid(r["clip_id"])
        a = ann[r["split"]]
        labs = a["clip"].get((norm_ytid(y), s, e)) if s is not None else None
        if labs is None and r["split"] == "val":
            labs = a["ytid"].get(norm_ytid(y))
        if labs is None:
            unmatched.append(r["rel_media"])
        elif r["label"] not in labs:
            mislabelled.append(r["rel_media"])
    return {"rows": len(rows), "unmatched": len(unmatched), "label_mismatch": len(mislabelled),
            "unmatched_first5": unmatched[:5], "label_mismatch_first5": mislabelled[:5]}


# ----------------------------------------------------------------------------- freeze


def population(listing: Listing, names: list[str], split: str) -> list[dict[str, str]]:
    rows = []
    for c in names:
        for f in listing[split][c]:
            if not f.endswith(".mp4"):
                continue
            cid = f[: -len(".mp4")]
            if not _CLIP_ID_RE.match(cid):
                raise SystemExit(f"unexpected file name {split}/{c}/{f}")
            rows.append({"clip_id": cid, "split": split, "label": c, "rel_media": f"{split}/{c}/{f}"})
    return rows


def freeze(listing: Listing, names: list[str], cap: int = CAP, seed: int = SEED, *, pins: bool = True,
           annotations: Path | None = None) -> dict:
    """The frozen rows and the report from a release listing. ``pins=False`` skips the ``TREE_SHA256`` /
    ``DISK_CENSUS`` pins (synthetic fixtures)."""
    for split in SPLITS:
        if set(listing[split]) != set(names) or not all(_CLASS_RE.match(c) for c in names):
            raise SystemExit(f"{split}/ class dirs != classes.csv names: "
                             f"{sorted(set(listing[split]) ^ set(names))[:5]}")
    disk = census(listing)
    tree_sha = listing_sha256(listing)
    if pins and (disk != DISK_CENSUS or tree_sha != TREE_SHA256):
        raise SystemExit(f"listing census {disk} / sha256 {tree_sha} != pinned {DISK_CENSUS} / {TREE_SHA256}: "
                         "a different media set (a new protocol_version, not a re-freeze)")
    train = population(listing, names, "train")
    val = population(listing, names, "val")
    if len({r["clip_id"] for r in train + val}) != len(train) + len(val):
        raise SystemExit("duplicate clip_id across the release")

    # Leakage audit at ytid level (raw and normalized); K400's official splits are disjoint.
    tr_y = {ytid(r["clip_id"])[0] for r in train}
    va_y = {ytid(r["clip_id"])[0] for r in val}
    leaks = (tr_y & va_y) | ({norm_ytid(y) for y in tr_y} & {norm_ytid(y) for y in va_y})
    if leaks:
        raise SystemExit(f"train/val ytid overlap: {sorted(leaks)[:5]}")

    keep = class_balanced_subsample([r["label"] for r in train], cap, seed)
    drawn = [train[i] for i in keep]
    if len(drawn) != min(cap, len(train)):
        raise SystemExit(f"draw produced {len(drawn)} rows, expected {cap}")
    per_class = Counter(r["label"] for r in drawn)
    val_per_class = Counter(r["label"] for r in val)

    rows = drawn + val
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(MANIFEST_COLUMNS)
    for r in rows:
        w.writerow([r[k] for k in MANIFEST_COLUMNS])
    manifest_bytes = buf.getvalue().encode("utf-8")

    non_mp4 = {s: sum(n for ext, n in disk[s].items() if ext != ".mp4") for s in SPLITS}
    curation = {
        "total_upstream": sum(sum(v.values()) for v in disk.values()),
        "per_split": {
            "train": {"listed": len(drawn), "staged": len(drawn), "missing": 0},
            "val": {"listed": len(val), "staged": len(val), "missing": 0},
        },
        "excluded": {
            "draw_remainder": len(train) - len(drawn),
            "val_non_mp4_container": non_mp4["val"],
            "train_non_mp4_container": non_mp4["train"],
        },
        "short_clips": sum(SHORT_CLIPS.values()),  # pinned duration census; --durations re-measures and compares
        "leakage_audit": {"against": "train ytids of the staged K400 train split (raw and torrent-normalized)", "excluded": 0},
    }
    listed = sum(v["listed"] for v in curation["per_split"].values())
    assert listed + sum(curation["excluded"].values()) == curation["total_upstream"], curation
    manifest_sha = sha256_bytes(manifest_bytes)
    report = {
        "manifest": {"file": "manifest.csv.gz", "sha256": manifest_sha},
        "rows": {"train": len(drawn), "val": len(val), "total": len(rows)},
        "disk_census": disk,
        "upstream": {
            "source": f"Academic Torrents {TORRENT} (Kinetics-400 _256 release): file listing of train_256 + val_256",
            "sha256": tree_sha,
            "rows": curation["total_upstream"],
            "curation": curation,
        },
        "train_draw": {
            "cap": cap,
            "seed": seed,
            "balance": "class",
            "population": len(train),
            "per_class_min_max": [min(per_class.values()), max(per_class.values())],
            "n_classes": len(per_class),
        },
        "val_per_class_min_max": [min(val_per_class.values()), max(val_per_class.values())],
    }
    if annotations is not None:
        chk = annotation_check(rows, read_annotations(annotations))
        report["annotation_check"] = chk
        if chk["unmatched"] or chk["label_mismatch"]:
            raise SystemExit(f"annotation check failed: {chk}")
    return {"manifest_bytes": manifest_bytes, "drawn": drawn, "val": val, "report": report}


def duration_census(media_root: Path, rels: list[str], workers: int) -> list[tuple[str, float | None, int]]:
    """Video-stream duration (container duration as fallback; = ``RawStream.info.duration_s``) and bytes of every
    frozen file (PyAV header probe, threaded): fills ``curation.short_clips`` and ``budget.media_gb``."""
    from concurrent.futures import ThreadPoolExecutor

    import av  # noqa: PLC0415 — [media] extra, freeze-time only

    def one(rel: str) -> tuple[str, float | None, int]:
        p = media_root / rel
        try:
            size = p.stat().st_size
            with av.open(str(p)) as c:
                vs = c.streams.video[0]
                d = float(vs.duration * vs.time_base) if vs.duration is not None else (c.duration / 1e6 if c.duration else None)
            return rel, d, size
        except Exception:  # noqa: BLE001 — a probe failure is recorded as duration None
            return rel, None, (p.stat().st_size if p.exists() else 0)

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rels, chunksize=64))


def durations_report(cen: list[tuple[str, float | None, int]]) -> dict:
    durs: dict[str, list[float | None]] = {"train": [], "val": []}
    for rel, d, _ in cen:
        durs[rel.split("/", 1)[0]].append(d)
    out = {}
    for split, v in durs.items():
        ok = sorted(d for d in v if d is not None)
        out[split] = {"n": len(v), "unprobed": len(v) - len(ok), "short_lt_10s": sum(d < 10.0 for d in ok),
                      "lt_5s": sum(d < 5.0 for d in ok), "min": ok[0] if ok else None,
                      "median": ok[len(ok) // 2] if ok else None, "max": ok[-1] if ok else None}
    return out


def default_root() -> Path | None:
    """The configured ``[data.k400] media_root`` (``--data``, ``YAVIH_DATA__K400__MEDIA_ROOT``, config files, else the
    yaml default ``${data.root}/kinetics400``); None if unresolvable or absent."""
    try:
        p = config.folder_root(HERE, "media_root")
    except config.ConfigError:
        return None
    return p if p.is_dir() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--media-root", type=Path, default=None, help="staged K400 root with train/ and val/ (default: the configured media_root)")
    p.add_argument("--zips", type=Path, default=None, help="read the listing from the two torrent zips in this directory instead of the staged tree")
    p.add_argument("--annotations", type=Path, default=None, help=f"directory with train.csv and val.csv from {ANNOTATION_URL}: cross-check every frozen row")
    p.add_argument("--durations", type=Path, default=None, help="write a per-file duration/bytes CSV here (PyAV header probe of all frozen rows; needs the staged tree)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path, default=HERE, help="folder to write manifest.csv.gz into (default: this folder)")
    p.add_argument("--check", action="store_true", help="compare against the committed manifest.csv.gz instead of writing (exit 1 on drift)")
    ns = p.parse_args(argv)

    names = read_classes(HERE / "classes.csv")
    root = None
    if ns.zips is not None:
        listing = list_zips(ns.zips)
        root = ns.media_root
    else:
        root = ns.media_root or default_root()
        if root is None:
            raise SystemExit("no staged K400 tree: pass --media-root (or --zips DIR) or configure [data.k400] media_root")
        listing = list_tree(root)
    result = freeze(listing, names, CAP, SEED, annotations=ns.annotations)
    report = result["report"]
    if ns.durations is not None:
        if root is None:
            raise SystemExit("--durations needs the staged tree (--media-root)")
        cen = duration_census(root, [r["rel_media"] for r in result["drawn"] + result["val"]], ns.workers)
        with open(ns.durations, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["rel_media", "duration_s", "bytes"])
            for rel, d, size in cen:
                w.writerow([rel, "" if d is None else f"{d:.3f}", size])
        report["duration_census"] = durations_report(cen)
        report["media_bytes"] = sum(s for _, _, s in cen)
        measured = {k: v["short_lt_10s"] for k, v in report["duration_census"].items()}
        if measured != SHORT_CLIPS:
            print(f"WARNING: duration census {measured} != pinned SHORT_CLIPS {SHORT_CLIPS}", file=sys.stderr)
    gz = gzip_bytes(result["manifest_bytes"])
    target = ns.out / "manifest.csv.gz"
    if ns.check:
        drift = not target.is_file() or target.read_bytes() != gz
        print(json.dumps({**report, "check": "drift" if drift else "ok"}, indent=1))
        return 1 if drift else 0
    ns.out.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(gz)
    os.replace(tmp, target)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
