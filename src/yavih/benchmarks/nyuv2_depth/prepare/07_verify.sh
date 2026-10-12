#!/usr/bin/env bash
# nyuv2_depth step 7 — re-freeze check + PNG listing check + yavih data verification (setup.md "## Prepare &
# verify"). Writes nothing outside the yavih tree, $YAVIH_HOME/data_status and $YAVIH_HOME/media_info.
#
# Usage: bash 07_verify.sh [GEONET_ROOT] [DERIVED_ROOT]   (defaults: the configured nyuv2_depth roots,
#        $(yavih config get nyuv2_depth.geonet_root) / $(yavih config get nyuv2_depth.derived_root))
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GEONET="${1:-$(yavih config get nyuv2_depth.geonet_root 2>/dev/null || true)}"   # optional: only the re-freeze check reads it
DER="${2:-$(yavih config get nyuv2_depth.derived_root)}"
export YAVIH_DATA__NYUV2_DEPTH__DERIVED_ROOT="$DER"
if [ -n "$GEONET" ]; then export YAVIH_DATA__NYUV2_DEPTH__GEONET_ROOT="$GEONET"; fi
# 1. committed manifest == re-derivation from the nyu_geonet listing + eval_meta.json (the draw needs torch);
#    skipped LOUDLY when an input or torch is absent
if [ -n "$GEONET" ] && [ -d "$GEONET/nyu_geonet" ] && [ -f "$DER/eval_meta.json" ] && python -c 'import torch' 2>/dev/null; then
  if python "$HERE/../make_manifest.py" --check --geonet-dir "$GEONET/nyu_geonet" --eval-meta "$DER/eval_meta.json" >/dev/null; then
    echo "[nyuv2/07] make_manifest --check: identical"
  else
    echo "[nyuv2/07] make_manifest --check FAILED (see the message above)" >&2; exit 1
  fi
else
  echo "[nyuv2/07] GeoNet listing, $DER/eval_meta.json or torch absent: re-freeze check SKIPPED"
fi
# 2. every PNG against the shipped listing (file sha256, else decoded-pixel sha256)
python -I "$HERE/recontainer_png.py" --verify --out "$DER" --listing "$HERE/../rgb_png.sha256.csv.gz"
# 3. presence of 12 654 PNGs, read sample, both target sha256, derived sha256, curation sums; census of the PNG headers
yavih bench verify nyuv2_depth --census --json
# Expected: manifest_sha_ok, labels_ok, targets_sha_ok, derived_sha_ok, curation_invariants_ok all true;
# train 12000/12000 and eval 654/654 present; media_info rows 12654 (every PNG 640x480).
