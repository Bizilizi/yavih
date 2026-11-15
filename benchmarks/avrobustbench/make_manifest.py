#!/usr/bin/env python
"""Re-freeze ``manifest.csv.gz`` / ``classes.csv`` of avrobustbench (VGGSOUND-2C) byte-identically from public files.

Inputs (read only, nothing is downloaded here):

- the official VGGSound listing ``vggsound.csv`` (``ytid,start_seconds,label,split``, no header, 199 176 rows,
  sha256 ``ed9fdeed…``; ``<media_root>/metadata/vggsound.csv``): labels, the 309 classes, the train population;
- the VGGSOUND-2C clip list: the member names of ``vgg_data/clean_vgg.zip`` of the authors' Hugging Face release
  ``sakshamsingh1/av_robust_data`` at revision ``42b5e078…`` (14 046 ``clean_vgg/<ytid>_<start>.mp4``), as
  prepare step 2 writes them to ``<derived_root>/meta/clean_vgg.members.txt`` (sha256 ``d8a5754e…``);
- optionally the materialized media under ``derived_root`` (prepare step 4), for the presence counts printed
  under ``curation.per_split`` (presence never changes the rows).

Classes = the 309 distinct labels of ``vggsound.csv``, sorted (``index,name``).

Eval rows (split ``test``) = the 14 046 VGGSOUND-2C clips (every one is a ``split == test`` row of
``vggsound.csv``; clip id ``<ytid>_<start zero-padded to 6 digits>``, sorted) under each of the 16 conditions of
``protocol.conditions`` (``clean``, then the 15 corruptions in the paper's order), condition-major: 224 736 rows,
``clip_id = <clip>#<condition>``, ``rel_video`` / ``rel_audio = media/test/<condition>/<clip>.{mp4,wav}`` (written
by prepare step 4 under ``derived_root``), ``src_video`` / ``src_audio = {video,audio}/<clip>.{mp4,wav}`` (the
VGGSound source under ``media_root``). The 1 400 VGGSound test clips the authors did not release are excluded
(``not_in_vggsound2c``, counted under all 16 conditions).

Train rows (split ``train``, condition ``clean``) = ``yavih.freeze.class_balanced_subsample`` (cap 20 000, seed 0)
over the ``split == train`` rows of ``vggsound.csv`` in file order, minus the rows whose YouTube id is the id of an
eval clip (``leakage_train_ytid``). The draw depends on the row order of ``vggsound.csv``, hence its sha256 pin;
torch is needed here (freeze time only).

``total_upstream`` counts candidate rows: the 183 730 train rows of ``vggsound.csv`` plus 16 conditions x the
15 446 test rows, so listed + excluded = 183 730 + 247 136 = 430 866.

Outputs, byte-identical on re-run: ``manifest.csv.gz`` (``yavih.freeze.gzip_bytes``), ``classes.csv``
(``yavih.freeze.csv_bytes``), written with ``yavih.freeze.write_if_changed``, and a printed JSON block with every
value ``benchmark.yaml`` needs (``manifest`` digests, ``manifest.upstream``, ``budget.clips``).

Usage::

    python "$(yavih bench path avrobustbench)/make_manifest.py" --check     # re-derive and compare, write nothing
    python "$(yavih bench path avrobustbench)/make_manifest.py"             # re-freeze (in a git checkout)
    python make_manifest.py --no-media --metadata DIR --check  # DIR holds vggsound.csv + clean_vgg.members.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

from yavih import config
from yavih.freeze import class_balanced_subsample, csv_bytes, gzip_bytes, write_if_changed

HERE = Path(__file__).resolve().parent
NAME = "avrobustbench"
CAP = 20000
SEED = 0
NUM_CLASSES = 309
CONDITIONS = ["clean", "gaussian", "impulse", "shot", "speckle", "compression", "snow", "frost", "spatter", "wind",
              "rain", "underwater", "concert", "smoke", "crowd", "interference"]
VGGSOUND_CSV_SHA = "ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8"
MEMBERS_SHA = "d8a5754e7a2824848209986f363dc92bc5802ea10c4b44d431f5e5a2583c244a"
CLEAN_ZIP_SHA = "74ca5d49efa3dc340695a16f2176d81082e868e48ba2ca24ad493ab863d48b3e"
HF_REVISION = "42b5e0789ccd1a1545a9e2a4c68e6ecb499541bb"
MEMBERS_NAME = "clean_vgg.members.txt"
RELEASE_CLIPS = 14046
COLUMNS = ["clip_id", "split", "label", "condition", "rel_video", "rel_audio", "src_video", "src_audio"]
_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}_\d{6}$")
_MEMBER_RE = re.compile(r"^clean_vgg/(?P<ytid>[A-Za-z0-9_-]{11})_(?P<start>\d+)\.mp4$")
UPSTREAM_SOURCE = ("VGGSound vggsound.csv (labels, train draw) + VGGSOUND-2C clip list "
                   f"(sakshamsingh1/av_robust_data@{HF_REVISION[:8]} vgg_data/clean_vgg.zip members)")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _pinned(path: Path, want: str, what: str) -> bytes:
    if not path.is_file():
        raise SystemExit(f"missing {what} {path} (setup.md, Obtain the data)")
    raw = path.read_bytes()
    got = sha256_bytes(raw)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got} != pinned {want}")
    return raw


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


def read_members(raw: bytes) -> list[str]:
    """``clean_vgg.members.txt`` → the released clip ids ``<ytid>_<start6>`` (the directory entry skipped)."""
    out = []
    for line in raw.decode("utf-8").splitlines():
        if line == "clean_vgg/":
            continue
        m = _MEMBER_RE.match(line)
        if not m:
            raise SystemExit(f"{MEMBERS_NAME}: unexpected member {line!r}")
        out.append(f"{m['ytid']}_{int(m['start']):06d}")
    if len(set(out)) != len(out):
        raise SystemExit(f"{MEMBERS_NAME}: duplicate clips")
    return out


def _root(name: str, flag: Path | None) -> Path | None:
    if flag is not None:
        return flag
    try:
        return Path(config.folder_root(HERE, name))
    except config.ConfigError:
        return None


def freeze(vggsound_csv: Path, members: Path, derived_root: Path | None, cap: int, seed: int) -> dict:
    vgg = read_vggsound_csv(_pinned(vggsound_csv, VGGSOUND_CSV_SHA, "vggsound.csv"))
    released = read_members(_pinned(members, MEMBERS_SHA, MEMBERS_NAME))
    if len(released) != RELEASE_CLIPS:
        raise SystemExit(f"{MEMBERS_NAME}: {len(released)} clips, expected {RELEASE_CLIPS}")
    classes = sorted({label for _, label, _ in vgg})
    if len(classes) != NUM_CLASSES:
        raise SystemExit(f"vggsound.csv: {len(classes)} classes, expected {NUM_CLASSES}")

    test = {cid: label for cid, label, split in vgg if split == "test"}
    missing = sorted(set(released) - set(test))
    if missing:
        raise SystemExit(f"{len(missing)} released clips are not split == test rows of vggsound.csv: {missing[:5]}")
    eval_clips = sorted(released)
    eval_ytids = {c.rsplit("_", 1)[0] for c in eval_clips}

    train_all = [(cid, label) for cid, label, split in vgg if split == "train"]
    leak = [(cid, label) for cid, label in train_all if cid.rsplit("_", 1)[0] in eval_ytids]
    pool = [(cid, label) for cid, label in train_all if cid.rsplit("_", 1)[0] not in eval_ytids]
    keep = class_balanced_subsample([lab for _, lab in pool], cap, seed)
    drawn = [pool[i] for i in keep]
    if len(drawn) != cap:
        raise SystemExit(f"draw produced {len(drawn)} rows, expected {cap} (eligible {len(pool)})")

    rows: list[list[str]] = []
    for cid, label in drawn:
        rows.append([f"{cid}#clean", "train", label, "clean", f"media/train/clean/{cid}.mp4",
                     f"media/train/clean/{cid}.wav", f"video/{cid}.mp4", f"audio/{cid}.wav"])
    for cond in CONDITIONS:
        for cid in eval_clips:
            rows.append([f"{cid}#{cond}", "test", test[cid], cond, f"media/test/{cond}/{cid}.mp4",
                         f"media/test/{cond}/{cid}.wav", f"video/{cid}.mp4", f"audio/{cid}.wav"])
    manifest_bytes = csv_bytes(COLUMNS, rows)
    classes_bytes = csv_bytes(["index", "name"], [[i, n] for i, n in enumerate(classes)])

    listed = {"train": len(drawn), "test": len(eval_clips) * len(CONDITIONS)}
    staged: dict[str, int | None] = {"train": None, "test": None}
    if derived_root is not None and (derived_root / "media").is_dir():
        present = {"train": 0, "test": 0}
        for r in rows:
            if (derived_root / r[4]).is_file() and (derived_root / r[5]).is_file():
                present[r[1]] += 1
        staged = dict(present)
    curation = {
        "total_upstream": len(train_all) + len(test) * len(CONDITIONS),
        "per_split": {s: {"listed": n, "staged": staged[s], "missing": None if staged[s] is None else n - staged[s]}
                      for s, n in listed.items()},
        "excluded": {
            "not_in_vggsound2c": (len(test) - len(eval_clips)) * len(CONDITIONS),
            "leakage_train_ytid": len(leak),
            "draw_remainder": len(pool) - len(drawn),
        },
        "leakage_audit": {"against": "VGGSOUND-2C eval ytids (train rows sharing a YouTube id with an eval clip)",
                          "excluded": len(leak)},
    }
    assert sum(listed.values()) + sum(curation["excluded"].values()) == curation["total_upstream"], curation
    manifest_sha = sha256_bytes(manifest_bytes)
    classes_sha = sha256_bytes(classes_bytes)
    return {
        "manifest_bytes": manifest_bytes,
        "classes_bytes": classes_bytes,
        "report": {
            "manifest": {"file": "manifest.csv.gz", "sha256": manifest_sha, "classes": "classes.csv",
                         "classes_sha256": classes_sha},
            "budget_clips": listed,
            "upstream": {
                "source": UPSTREAM_SOURCE,
                "sha256": MEMBERS_SHA,
                "rows": len(eval_clips) + len(train_all),
                "upstream_files": {
                    "vggsound/metadata/vggsound.csv": VGGSOUND_CSV_SHA,
                    f"av_robust_data/{MEMBERS_NAME}": MEMBERS_SHA,
                    "av_robust_data/vgg_data/clean_vgg.zip": CLEAN_ZIP_SHA,
                },
                "curation": curation,
            },
            "eval_clips": len(eval_clips),
            "train_draw": {"cap": cap, "seed": seed, "eligible_train_rows": len(pool),
                           "per_class_min_max": _min_max(drawn)},
        },
    }


def _min_max(drawn: list[tuple[str, str]]) -> list[int]:
    counts: dict[str, int] = {}
    for _, lab in drawn:
        counts[lab] = counts.get(lab, 0) + 1
    return [min(counts.values()), max(counts.values())]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vggsound-csv", type=Path, default=None, help="default: <media_root>/metadata/vggsound.csv")
    ap.add_argument("--members", type=Path, default=None, help=f"default: <derived_root>/meta/{MEMBERS_NAME}")
    ap.add_argument("--metadata", type=Path, default=None, help=f"a directory holding vggsound.csv and {MEMBERS_NAME}")
    ap.add_argument("--media-root", type=Path, default=None, help="the VGGSound tree (default: the folder's media_root)")
    ap.add_argument("--derived-root", type=Path, default=None, help="default: the folder's derived_root")
    ap.add_argument("--no-media", action="store_true", help="no presence counts (staged / missing are null)")
    ap.add_argument("--out", type=Path, default=HERE, help="folder to write (default: this folder)")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--check", action="store_true",
                    help="compare with the committed files instead of writing (exit 1 on drift)")
    ns = ap.parse_args(argv)
    if ns.metadata is not None:
        vgg_csv, members = ns.metadata / "vggsound.csv", ns.metadata / MEMBERS_NAME
        derived = None if ns.no_media else _root("derived_root", ns.derived_root)
    else:
        media = _root("media_root", ns.media_root)
        derived = _root("derived_root", ns.derived_root)
        vgg_csv = ns.vggsound_csv or (media / "metadata" / "vggsound.csv" if media is not None else None)
        members = ns.members or (derived / "meta" / MEMBERS_NAME if derived is not None else None)
        if vgg_csv is None or members is None:
            raise SystemExit(f"pass --metadata DIR (or --vggsound-csv and --members) or configure the media_root and "
                             f"derived_root of {NAME}")
    result = freeze(vgg_csv, members, None if ns.no_media else derived, ns.cap, ns.seed)
    gz = gzip_bytes(result["manifest_bytes"])
    if ns.check:
        targets = {HERE / "manifest.csv.gz": gz, HERE / "classes.csv": result["classes_bytes"]}
        drift = [str(p) for p, b in targets.items() if not p.is_file() or p.read_bytes() != b]
        print(json.dumps({**result["report"], "check": "ok" if not drift else "drift", "drift": drift}, indent=1))
        return 1 if drift else 0
    ns.out.mkdir(parents=True, exist_ok=True)
    written = [str(p) for p, b in {ns.out / "manifest.csv.gz": gz, ns.out / "classes.csv": result["classes_bytes"]}.items()
               if write_if_changed(p, b)]
    print(json.dumps({**result["report"], "written": written}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
