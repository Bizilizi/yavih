#!/usr/bin/env bash
# CREMA-D step 3/3 — re-freeze manifest.csv.gz + classes.csv (byte-identical to the committed files), with the
# TFDS serialized sizes recomputed from the wavs and asserted equal to TFDS's dataset_info.json, then verify the
# staged media with yavih.
#
#   yavih bench prepare crema_d 03               # media root = $(yavih config get crema_d.media_root)
#   yavih bench prepare crema_d 03 -- --check    # re-derive and compare, write nothing
#
# Needs the clone of step 01 (processedResults/summaryTable.csv, sha256 fb0e3827…; AudioWAV/ for the sizes,
# presence and headers). `make_manifest.py --no-media --metadata <summaryTable.csv>` re-freezes from the table alone.
set -euo pipefail
FOLDER="${FOLDER:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=${PY:-python}
MEDIA_ROOT="${ROOT:-$(yavih config get crema_d.media_root)}"

"$PY" "$FOLDER/make_manifest.py" --media-root "$MEDIA_ROOT" "$@"
# sha256 is over the DECOMPRESSED manifest; compare the digest of the zcat stream.
got=$(zcat "$FOLDER/manifest.csv.gz" | sha256sum | cut -d' ' -f1)
[ "$got" = 9844ba2ee5fbefac529645d714f8f922cce9b8a366824ca5c16450f7de8cf183 ] || { echo "FAIL: manifest sha256 $got" >&2; exit 1; }
echo "manifest.csv (decompressed): OK"
echo "85bbf18ea6375dba383ba4a536b7ce6dbe9d1e17f4603788d01de8dda3cd247f  $FOLDER/classes.csv" | sha256sum -c -
YAVIH_DATA__CREMA_D__MEDIA_ROOT="$MEDIA_ROOT" yavih bench verify crema_d --json
