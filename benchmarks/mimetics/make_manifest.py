#!/usr/bin/env python
"""Freeze the ``mimetics`` manifest: Kinetics-400 train clips of the 50 Mimetics classes, then the 713 Mimetics clips.

Mimetics (Weinzaepfel & Rogez, IJCV 2021) is test-only: 713 YouTube clips of mimed actions of 50 Kinetics-400
classes. A probe is trained on Kinetics-400 clips of the same 50 classes (the "Kinetics50" training set of the
reference implementation, IntegralAction) and scored on Mimetics.

Inputs (read only, nothing is downloaded):

- the official release ``Mimetics_release_v1.0.zip``
  (https://europe.naverlabs.com/wp-content/uploads/2019/12/Mimetics_release_v1.0.zip, sha256 ``6ddc9c98…``),
  unpacked by ``prepare/01_release.sh`` into ``<media_root>/release/``: ``mimetics_v1.0.csv`` (713 rows
  ``label,youtube_id,time_start,time_end,split,is_cc``) gives the test rows; ``mimetics_v1.0_objectsceneannot.txt``
  (per clip: a relevant object present, a relevant scene, a mime artist) gives the ``object``, ``scene`` and ``mime``
  columns; ``mimetics_v1.0_clsannot.txt`` gives the class list and its order. All three are sha256-pinned
  (``RELEASE_SHA``). ``--release DIR`` (default ``<media_root>/release``), or with ``--no-media`` the file
  ``--metadata mimetics_v1.0.csv`` with the two ``.txt`` files next to it.
- the listing of the 50 class directories of the Academic-Torrents Kinetics-400 ``_256`` release (infohash
  ``184d11318372f70018cf9a72ef867e2fb9ce1d26``, the media of the ``k400`` folder), either the staged tree
  ``<k400_root>/train/<class>/<file>`` (``--k400-root``, default the configured ``mimetics.k400_root`` =
  ``${datasets.kinetics400}``) or a text file of ``train/<class>/<file>`` lines (``--k400-listing``; write one with
  ``--write-listing FILE``). The sorted listing of its 34 106 files is pinned by sha256 (``K400_LISTING_SHA256``):
  another listing is another media set and is refused.
- optionally ``--media-root`` (default the configured ``mimetics.media_root``): ``clips/<clip_id>.mp4`` as
  ``prepare/02_clips.py`` writes them, counted for the site's coverage line (never written into the manifest).

Rows. Classes (``classes.csv``, ``index,name``): the 50 Mimetics labels in the release's order (that of
``mimetics_v1.0_clsannot.txt``, which is sorted), each named as its Academic-Torrents class directory (space ->
``_``; ``(``, ``)``, ``'`` -> ``-``, e.g. ``punching_person_-boxing-``). Train: every ``*.mp4`` of those 50
directories (``clip_id`` = file stem, ``rel_media`` = ``train/<class>/<file>`` under ``k400_root``), ordered by class
then file name, then the seeded class-balanced draw ``yavih.freeze.class_balanced_subsample(labels, 10000, seed 0)``
(exactly 200 per class; torch is a freeze-time dependency of this script only). Test: the 713 release rows,
``clip_id`` = ``<youtube_id>_<time_start:06d>_<time_end:06d>`` (the crawler's file name, and the name the release's
own ``objectsceneannot`` file uses), ``rel_media`` = ``clips/<clip_id>.mp4`` under ``media_root``, ordered by class
then ``clip_id``. All 713 are frozen whether or not their YouTube video still exists: staging decays, and
``splits.min_eval_coverage`` says how much may be missing. Leakage audit (when the K400 tree is at hand): no
Mimetics YouTube id is a Kinetics-400 train or val id of the release, raw or with the release's id normalisation.

Output, byte-identical on re-run: ``manifest.csv.gz`` (gzip mtime 0, no file name, level 9) and ``classes.csv``, and
a JSON block with what ``benchmark.yaml`` records (``manifest.sha256``, ``classes_sha256``, ``manifest.upstream``).
The ``curation.per_split.test`` staged / missing counts are the census at freeze time (``STAGED_AT_FREEZE``); the
site's live count is logged separately.

Usage::

    yavih bench prepare mimetics 01 && yavih bench prepare mimetics 02      # the release, then the clips
    python "$(yavih bench path mimetics)/make_manifest.py"                    # freeze (staged release + K400 tree)
    python "$(yavih bench path mimetics)/make_manifest.py" --check            # re-derive, compare, write nothing
    python make_manifest.py --write-listing k400_50.txt                       # the K400 listing as a text file
    python make_manifest.py --no-media --metadata release/mimetics_v1.0.csv --k400-listing k400_50.txt --check

``--check`` exits 1 when the re-derived ``manifest.csv.gz`` (decompressed) or ``classes.csv`` differs from the
committed files. ``--out DIR`` (default this folder) writes elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from yavih import config
from yavih.freeze import class_balanced_subsample, csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent
NAME = "mimetics"
CAP = 10000
SEED = 0
NUM_CLASSES = 50
N_TEST = 713
MANIFEST_COLUMNS = ["clip_id", "split", "label", "rel_media", "object", "scene", "mime"]
CLASSES_COLUMNS = ["index", "name"]

RELEASE_URL = "https://europe.naverlabs.com/wp-content/uploads/2019/12/Mimetics_release_v1.0.zip"
RELEASE_ZIP_SHA256 = "6ddc9c98c3f720baa5c70f1dc6cc05b80f1a16deb679a728082a698afb8680ab"
RELEASE_SHA = {
    "mimetics_v1.0.csv": "78b32467a248c24b7082d9a29d9eb0bf6eeb4462ed8abda290d2036018115044",
    "mimetics_v1.0_objectsceneannot.txt": "a4db94284f22b777442c40ca28318602c8cadb253721e3a1016ba4f1df4b9f63",
    "mimetics_v1.0_clsannot.txt": "cb18b3435545c247fcef83d90ad25d1aa9225a621eb71ec6d4c7d6cfd0ebfe62",
}
TORRENT = "184d11318372f70018cf9a72ef867e2fb9ce1d26"
#: sha256 of the sorted listing ``train/<class>/<file>\n`` of the 50 Mimetics class directories of the
#: Academic-Torrents Kinetics-400 ``_256`` release (34 106 lines, all ``*.mp4``).
K400_LISTING_SHA256 = "9b2af343a935b67dd2b676637eb121bb34a53e1e18bd5b5bcc8593182d617a67"
K400_LISTING_ROWS = 34106
#: Census of the staged clips at freeze time (prepare/02_clips.py on that date): how many of the 713 Mimetics
#: clips could still be downloaded from YouTube. Recorded in ``curation.per_split.test``.
STAGED_AT_FREEZE = {"date": "2026-11-15", "staged": 422}

_SFX = re.compile(r"^(.*)_(\d{6})_(\d{6})$")  # K400 train file stem: <ytid>_<start6>_<end6>
_YT = re.compile(r"^youtube-(.+)-(\d+)-(\d+)$")  # 36 K400 files: youtube-<ytid>-<start>-<end>


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def class_dir(label: str) -> str:
    """Kinetics label -> Academic-Torrents class directory: space -> ``_``; ``(``, ``)``, ``'`` -> ``-``."""
    return re.sub(r"[ ()']", lambda m: "_" if m.group() == " " else "-", label)


def norm_ytid(y: str) -> str:
    """The Academic-Torrents release's file-name normalisation of YouTube ids (leading ``-``/``_`` stripped,
    separator runs collapsed); used by the leakage audit only."""
    return re.sub(r"[-_]+", "_", y.strip("-_"))


def k400_ytid(stem: str) -> str:
    """The YouTube id of a release file stem (``<ytid>_<start6>_<end6>``, ``youtube-<ytid>-<s>-<e>`` or the bare id)."""
    m = _YT.match(stem) or _SFX.match(stem)
    return m.group(1) if m else stem


# ----------------------------------------------------------------------------- the release


def _pinned(path: Path) -> bytes:
    if not path.is_file():
        raise SystemExit(f"missing {path} (yavih bench prepare mimetics 01 unpacks the release)")
    b = path.read_bytes()
    want = RELEASE_SHA[path.name]
    if sha256_bytes(b) != want:
        raise SystemExit(f"{path}: sha256 {sha256_bytes(b)} != pinned {want} (not the Mimetics v1.0 release)")
    return b


def read_release(csv_path: Path, *, pins: bool = True) -> dict:
    """``{"labels": [the 50 labels, release order], "rows": [713 csv rows], "annot": {clip_id: (object, scene, mime)}}``."""
    d = csv_path.parent
    read = _pinned if pins else (lambda p: p.read_bytes())
    rows = list(csv.DictReader(io.StringIO(read(csv_path).decode("utf-8"), newline="")))
    labels = []
    for line in read(d / "mimetics_v1.0_clsannot.txt").decode("utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            labels.append(line.split()[0].replace("_", " "))
    annot = {}
    for line in read(d / "mimetics_v1.0_objectsceneannot.txt").decode("utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            video, obj, scene, mime = (c.strip() for c in line.split("\t"))
            if not video.endswith(".mp4") or {obj, scene, mime} - {"yes", "no"}:
                raise SystemExit(f"objectsceneannot: unexpected line {line!r}")
            annot[video[: -len(".mp4")]] = (obj, scene, mime)
    return {"labels": labels, "rows": rows, "annot": annot}


def test_rows(release: dict, names: list[str]) -> list[dict[str, str]]:
    """The Mimetics rows, ordered by class then clip_id."""
    labels = release["labels"]
    if sorted(labels) != labels or len(set(labels)) != len(labels):
        raise SystemExit("clsannot: the class list is expected sorted and unique")
    index = {lab: i for i, lab in enumerate(labels)}
    out = []
    for r in release["rows"]:
        if r["split"] != "test" or r["label"] not in index or len(r["youtube_id"]) != 11:
            raise SystemExit(f"mimetics_v1.0.csv: unexpected row {r}")
        cid = f"{r['youtube_id']}_{int(r['time_start']):06d}_{int(r['time_end']):06d}"
        if cid not in release["annot"]:
            raise SystemExit(f"objectsceneannot lacks {cid}")
        obj, scene, mime = release["annot"][cid]
        out.append({"clip_id": cid, "split": "test", "label": names[index[r["label"]]], "rel_media": f"clips/{cid}.mp4",
                    "object": obj, "scene": scene, "mime": mime})
    if len({r["clip_id"] for r in out}) != len(out) or set(release["annot"]) != {r["clip_id"] for r in out}:
        raise SystemExit("release: clip ids are not unique, or objectsceneannot names other clips")
    order = {n: i for i, n in enumerate(names)}
    return sorted(out, key=lambda r: (order[r["label"]], r["clip_id"]))


# ----------------------------------------------------------------------------- the K400 listing


def list_k400(k400_root: Path, names: list[str]) -> list[str]:
    """``train/<class>/<file>`` of the 50 class directories, one scandir each (NFS-friendly), sorted."""
    out = []
    for c in names:
        d = k400_root / "train" / c
        if not d.is_dir():
            raise SystemExit(f"{d} is missing (stage Kinetics-400 as the k400 folder does: yavih bench prepare k400)")
        out.extend(f"train/{c}/{e.name}" for e in os.scandir(d))
    return sorted(out)


def read_listing(path: Path) -> list[str]:
    return sorted(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def listing_sha256(lines: list[str]) -> str:
    return sha256_bytes("".join(line + "\n" for line in sorted(lines)).encode("utf-8"))


def train_rows(lines: list[str], names: list[str], cap: int, seed: int) -> tuple[list[dict[str, str]], int]:
    """The class-balanced draw over every ``*.mp4`` of the listing: ``(rows, candidates)``."""
    order = {n: i for i, n in enumerate(names)}
    cands = []
    for line in lines:
        split, c, f = line.split("/")
        if split != "train" or c not in order:
            raise SystemExit(f"listing: unexpected line {line!r}")
        if not f.endswith(".mp4"):
            raise SystemExit(f"listing: non-mp4 train file {line!r}")
        cands.append({"clip_id": f[: -len(".mp4")], "split": "train", "label": c, "rel_media": line,
                      "object": "", "scene": "", "mime": ""})
    cands.sort(key=lambda r: (order[r["label"]], r["clip_id"]))
    keep = class_balanced_subsample([r["label"] for r in cands], cap, seed)
    return [cands[i] for i in keep], len(cands)


def leakage_audit(k400_root: Path, rows: list[dict[str, str]]) -> int:
    """How many Mimetics YouTube ids are Kinetics-400 train or val ids of the staged release (raw or normalised)."""
    raw, norm = set(), set()
    for split in ("train", "val"):
        for d in os.scandir(k400_root / split):
            for e in os.scandir(d.path):
                y = k400_ytid(os.path.splitext(e.name)[0])
                raw.add(y)
                norm.add(norm_ytid(y))
    ids = {r["clip_id"][:11] for r in rows if r["split"] == "test"}
    return sum(1 for y in ids if y in raw or norm_ytid(y) in norm)


# ----------------------------------------------------------------------------- freeze


def freeze(release: dict, lines: list[str], *, cap: int = CAP, seed: int = SEED, pins: bool = True,
           leaked: int | None = None) -> dict:
    """The manifest and classes bytes and the ``manifest.upstream`` block."""
    names = [class_dir(lab) for lab in release["labels"]]
    if pins and len(names) != NUM_CLASSES:
        raise SystemExit(f"{len(names)} classes, expected {NUM_CLASSES}")
    sha = listing_sha256(lines)
    if pins and (sha != K400_LISTING_SHA256 or len(lines) != K400_LISTING_ROWS):
        raise SystemExit(f"K400 listing: {len(lines)} files, sha256 {sha} != pinned {K400_LISTING_SHA256} "
                         f"({K400_LISTING_ROWS} files): a different media set, refusing to re-freeze")
    test = test_rows(release, names)
    if pins and len(test) != N_TEST:
        raise SystemExit(f"{len(test)} Mimetics rows, expected {N_TEST}")
    train, n_cand = train_rows(lines, names, cap, seed)
    rows = train + test
    manifest = csv_bytes(MANIFEST_COLUMNS, [[r[c] for c in MANIFEST_COLUMNS] for r in rows])
    classes = csv_bytes(CLASSES_COLUMNS, [[str(i), n] for i, n in enumerate(names)])
    staged = min(int(STAGED_AT_FREEZE["staged"]), len(test))
    upstream = {
        "source": (f"Mimetics v1.0 release ({RELEASE_URL}) + the 50 Mimetics class directories of the "
                   f"Academic-Torrents Kinetics-400 _256 release ({TORRENT})"),
        "sha256": RELEASE_SHA["mimetics_v1.0.csv"],
        "rows": len(test) + n_cand,
        "draw": (f"all {n_cand} train mp4 of the 50 Mimetics classes in the Academic-Torrents Kinetics-400 _256 release "
                 f"(classes.csv order, then file name), class-balanced draw (yavih.freeze.class_balanced_subsample), "
                 f"cap {cap}, seed {seed}"),
        "upstream_files": {
            "Mimetics_release_v1.0.zip": RELEASE_ZIP_SHA256,
            **{f"release/{k}": v for k, v in RELEASE_SHA.items()},
            "k400_train_listing_50_classes": sha,
        },
        "curation": {
            "total_upstream": len(test) + n_cand,
            "per_split": {
                "train": {"listed": len(train), "staged": len(train), "missing": 0},
                "test": {"listed": len(test), "staged": staged, "missing": len(test) - staged},
            },
            "excluded": {"draw_remainder": n_cand - len(train)},
            "leakage_audit": {
                "against": "Kinetics-400 train and val YouTube ids of the Academic-Torrents release (raw and normalised)",
                "excluded": 0 if leaked is None else leaked,
            },
        },
    }
    return {"manifest_bytes": manifest, "classes_bytes": classes, "rows": rows, "upstream": upstream,
            "train_per_class": dict(Counter(r["label"] for r in train))}


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media-root", type=Path, default=None, help="Mimetics staging root (default: mimetics.media_root)")
    ap.add_argument("--k400-root", type=Path, default=None, help="staged Kinetics-400 release (default: mimetics.k400_root)")
    ap.add_argument("--release", type=Path, default=None, help="unpacked release dir (default: <media_root>/release)")
    ap.add_argument("--k400-listing", type=Path, default=None, help="text listing train/<class>/<file> instead of the tree")
    ap.add_argument("--no-media", action="store_true", help="freeze from --metadata and --k400-listing alone")
    ap.add_argument("--metadata", type=Path, default=None, help="with --no-media: mimetics_v1.0.csv (and the two .txt next to it)")
    ap.add_argument("--write-listing", type=Path, default=None, help="write the K400 listing to this file and exit")
    ap.add_argument("--check", action="store_true", help="re-derive and compare with the committed files; write nothing")
    ap.add_argument("--out", type=Path, default=HERE, help="output directory (default: this folder)")
    return ap.parse_args(argv)


def _root(key: str, override: Path | None) -> Path:
    try:
        return config.folder_root(HERE, key, override)
    except config.ConfigError as e:
        raise SystemExit(f"pass --{key.replace('_', '-')} or configure mimetics.{key}: {e}") from None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = lambda m: print(f"[make_manifest mimetics] {m}", file=sys.stderr)  # noqa: E731
    media_root = None
    if args.no_media:
        if args.metadata is None or args.k400_listing is None:
            raise SystemExit("--no-media needs --metadata (mimetics_v1.0.csv) and --k400-listing")
        csv_path = args.metadata
    else:
        media_root = _root("media_root", args.media_root)
        csv_path = (args.release or media_root / "release") / "mimetics_v1.0.csv"
    release = read_release(csv_path)
    names = [class_dir(lab) for lab in release["labels"]]

    k400_root = None
    if args.k400_listing is not None:
        lines = read_listing(args.k400_listing)
    else:
        k400_root = _root("k400_root", args.k400_root)
        lines = list_k400(k400_root, names)
    if args.write_listing is not None:
        args.write_listing.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        log(f"wrote {len(lines)} lines to {args.write_listing} (sha256 {listing_sha256(lines)})")
        return 0

    leaked = None
    if k400_root is not None:
        leaked = leakage_audit(k400_root, test_rows(release, names))
        if leaked:
            raise SystemExit(f"leakage audit: {leaked} Mimetics YouTube ids are Kinetics-400 ids of the release")
        log("leakage audit: none of the 560 Mimetics YouTube ids is a Kinetics-400 train/val id of the release")
    res = freeze(release, lines, leaked=leaked)

    if media_root is not None:
        clips = media_root / "clips"
        present = {e.name for e in os.scandir(clips)} if clips.is_dir() else set()
        n = sum(1 for r in res["rows"] if r["split"] == "test" and r["rel_media"].split("/", 1)[1] in present)
        log(f"site: {n}/{N_TEST} Mimetics clips staged under {clips} (census at freeze: {STAGED_AT_FREEZE})")

    manifest_sha = sha256_bytes(res["manifest_bytes"])
    classes_sha = sha256_bytes(res["classes_bytes"])
    if args.check:
        same_m = gzip.decompress((HERE / "manifest.csv.gz").read_bytes()) == res["manifest_bytes"]
        same_c = (HERE / "classes.csv").read_bytes() == res["classes_bytes"]
        log(f"--check: manifest.csv {'identical' if same_m else 'DIFFERS'} (sha256 {manifest_sha[:16]}…); "
            f"classes.csv {'identical' if same_c else 'DIFFERS'} (sha256 {classes_sha[:16]}…)")
        if not (same_m and same_c):
            return 1
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        wm = write_if_changed(args.out / "manifest.csv.gz", gzip_bytes(res["manifest_bytes"]))
        wc = write_if_changed(args.out / "classes.csv", res["classes_bytes"])
        log(f"manifest.csv.gz {'written' if wm else 'unchanged'} ({len(res['rows'])} rows); "
            f"classes.csv {'written' if wc else 'unchanged'}")
    print(json.dumps({"manifest": {"sha256": manifest_sha, "classes_sha256": classes_sha, "rows": len(res["rows"])},
                      "upstream": res["upstream"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
