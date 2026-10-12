#!/usr/bin/env bash
# nyuv2_depth step 5 — re-container every row's RGB as a lossless PNG into derived_root (setup.md "## Input
# contract", "## Prepare & verify"). Required by the raw-input contract: the upstream containers (GeoNet .mat,
# labelled .mat) also hold the depth labels, so the reader hands over these PNGs instead. Pixel values are exactly
# the upstream ones (train: GeoNet img truncated to 480x640 + channel offsets + rint → uint8; eval: eval_rgb.npy rows
# = images[i] of the labelled .mat); no crop, no resize.
#
# Writes <DERIVED_ROOT>/rgb/{train,eval}/<clip_id>.png (= manifest rel_media), rgb_png.sha256.csv, recontainer_png.json.
# Expected (zlib 1.2.13): train 12000 files 5407357983 B, eval 654 files 245690741 B; listing rgb_png.sha256.csv sha256
# 02639980f22166fea9092decf943e6a6f9e0380e4cc68c381286f9a7a4c4f95e (pinned in benchmark.yaml data.derived; a gzip
# copy ships in the folder). A different zlib build may write different bytes for the same pixels: --verify then
# accepts the decoded-pixel sha256 of the listing instead (and the listing's own sha256 will differ from the pin).
# Needs numpy + scipy; derived_root must hold eval_rgb.npy (step 2, or a symlink).
#
# Usage: bash 05_recontainer_png.sh <GEONET_ROOT> <DERIVED_ROOT> [--workers N]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GEONET="${1:?usage: 05_recontainer_png.sh <geonet_root> <derived_root>}"; OUT="${2:?usage}"; shift 2
python -I "$HERE/recontainer_png.py" --split all --geonet-root "$GEONET" --eval-index "$OUT" --out "$OUT" --progress "$@"
python -I "$HERE/recontainer_png.py" --verify --out "$OUT" --listing "$HERE/../rgb_png.sha256.csv.gz"
