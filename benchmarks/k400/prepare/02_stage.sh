#!/usr/bin/env bash
# k400 step 2 — unzip the two torrent zips into the layout benchmark.yaml reads (media_root):
#   <ROOT>/train/<class>/<ytid>_<start6>_<end6>.mp4   400 dirs, 240 618 files (all mp4)
#   <ROOT>/val/<class>/<ytid>.{mp4,mkv,webm}          400 dirs, 19 404 files (13 492 mp4 + 5 402 mkv + 510 webm)
# test_256/ is deliberately NOT extracted (the protocol never reads it).
#
# The frozen manifest reads only the 13 492 val .mp4 files; the .mkv/.webm are on disk but excluded
# (benchmark.yaml manifest.upstream.curation). Keep the zips: they are the provenance record of the media bytes.
#
# Inodes: 260 022 files + 800 dirs. On an inode-capped filesystem do not stage loose files; pack the frozen rows
# with 04_pack_webdataset_optional.sh on a machine that has the staged tree, or copy its tars.
#
# Usage:  yavih bench prepare k400 02   (or: FOLDER=$(yavih bench path k400) SRC=... ROOT=... sbatch prepare/02_stage.sh)
#         defaults: SRC=$(yavih config get data.root)/k400_download/kinetics400_dataset, ROOT=$(yavih config get k400.media_root)
#SBATCH --job-name=stage_k400
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --output=stage_k400-%j.log
set -euo pipefail

# Benchmark folder (holds classes.csv): FOLDER (set by `yavih bench prepare`), else K400_DIR, else next to this script.
HERE="${FOLDER:-${K400_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
[ -f "$HERE/classes.csv" ] || { echo "[k400/02] classes.csv not found under $HERE; set FOLDER=\$(yavih bench path k400)"; exit 2; }
CLS_TMP="/tmp/k400_classes.$$.txt"
trap 'rm -f "$CLS_TMP"' EXIT

SRC="${SRC:-$(yavih config get data.root)/k400_download/kinetics400_dataset}"
ROOT="${ROOT:-$(yavih config get k400.media_root)}"
mkdir -p "$ROOT"

# val: zip root is val_256/<class>/*
if [ ! -d "$ROOT/val" ]; then
  unzip -q -n "$SRC/kinetics400_validation.zip" -d "$ROOT/.val-tmp"
  mv "$ROOT/.val-tmp/val_256" "$ROOT/val"
  rmdir "$ROOT/.val-tmp"
fi
echo "[k400/02] val class dirs: $(ls "$ROOT/val" | wc -l) (expect 400)"

# train: zip root is kinetics400/train_256/<class>/*  (only that subtree is extracted)
if [ ! -d "$ROOT/train" ]; then
  unzip -q -n "$SRC/kinetics400.zip" 'kinetics400/train_256/*' -d "$ROOT/.train-tmp"
  mv "$ROOT/.train-tmp/kinetics400/train_256" "$ROOT/train"
  rm -rf "$ROOT/.train-tmp"
fi
echo "[k400/02] train class dirs: $(ls "$ROOT/train" | wc -l) (expect 400)"

# Class vocabulary. The frozen classes.csv is `ls val | sort` under the en_US.UTF-8 locale
# (its collation ignores '_'), NOT byte order: 15 indices differ from Python sorted() / LC_ALL=C sort
# (brushing_hair < brush_painting, pushing_car < push_up, skiing_* < ski_jumping, spraying < spray_painting).
# Never re-derive the vocabulary from a sort — classes.csv is frozen and is the only logit order.
ls "$ROOT/val" | LC_ALL=en_US.UTF-8 sort > "$CLS_TMP"
python3 - "$HERE/classes.csv" "$CLS_TMP" <<'PY'
import csv, sys
frozen = [r["name"] for r in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))]
staged = open(sys.argv[2], encoding="utf-8").read().splitlines()
print(f"[k400/02] staged class dirs {len(staged)}; equal to frozen classes.csv (as a set): {set(staged) == set(frozen)}; same order: {staged == frozen}")
PY
echo "next: yavih bench prepare k400 03"
