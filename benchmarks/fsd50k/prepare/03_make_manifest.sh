#!/usr/bin/env bash
# FSD50K step 3/3 — re-derive manifest.csv.gz + classes.csv from the staged official ground truth and check
# them against the committed files (make_manifest.py --check, writes nothing), then verify the staged media
# with yavih and write the media-info census.
#
#   bash 03_make_manifest.sh   # media root = $(yavih config get fsd50k.media_root)
#   YAVIH_DATA__FSD50K__MEDIA_ROOT=/path/to/datasets/fsd50k bash 03_make_manifest.sh
#
# Needs: the extracted tree of step 2 (ground_truth/*.csv, presence + WAV headers of the 30 231 frozen rows).
# The re-derivation needs torch (the train draw uses torch.randperm) and soundfile; without them it is skipped
# and only the checks of `yavih bench verify` run. Extra arguments go to make_manifest.py.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
MEDIA_ROOT="$(yavih config get fsd50k.media_root)"

if "$PY" -c "import torch, soundfile" 2>/dev/null; then
  "$PY" "$HERE/../make_manifest.py" --media-root "$MEDIA_ROOT" --check "$@" > /dev/null
else
  echo "torch or soundfile not importable with $PY: skipping the manifest re-derivation" >&2
fi
echo "e90a44a6aac640ca51ba66ac42454e952fbb40caf27fa07aa9725944c117338b  -" | (zcat "$HERE/../manifest.csv.gz" | sha256sum -c -)
echo "9f27f91e44383f191451fcb17034b53c670fe9e3a07e0281cc810d819ca22d06  $HERE/../classes.csv" | sha256sum -c -
# --census writes the media-info table (header facts handed to bridges as RawStream.info)
YAVIH_DATA__FSD50K__MEDIA_ROOT="$MEDIA_ROOT" yavih bench verify fsd50k --census --json
