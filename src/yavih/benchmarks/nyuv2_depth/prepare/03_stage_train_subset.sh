#!/usr/bin/env bash
# nyuv2_depth step 3 — optional: copy exactly the 12 000 frozen GeoNet .mat files (source_mat of the train rows)
# out of a full layout (setup.md "## Obtain the data"). 55 163 911 114 B, 12 000 files + 1 dir. The other ~19k
# GeoNet files and the rest of the layout are never read by the benchmark.
#
# Usage: bash 03_stage_train_subset.sh <SRC_ROOT (contains nyu_geonet/)> <DEST_ROOT> [rsync opts…]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:?usage: 03_stage_train_subset.sh <src_root> <dest_root>}"; DEST="${2:?usage}"; shift 2
LIST="$(mktemp)"; trap 'rm -f "$LIST"' EXIT
python -c "import csv,gzip,io,sys; [print(r['source_mat']) for r in csv.DictReader(io.StringIO(gzip.open(sys.argv[1]).read().decode())) if r['split']=='train']" \
  "$HERE/../manifest.csv.gz" > "$LIST"
echo "[nyuv2/03] $(wc -l < "$LIST") files → $DEST/nyu_geonet/ (expected 12000, 55163911114 B)"
mkdir -p "$DEST"
rsync -a --files-from="$LIST" "$@" "$SRC/" "$DEST/"
find "$DEST/nyu_geonet" -maxdepth 1 -name '*.mat' | wc -l
echo "[nyuv2/03] then: export YAVIH_DATA__NYUV2_DEPTH__GEONET_ROOT=$DEST ; copy the eval index (step 2 outputs) into derived_root"
