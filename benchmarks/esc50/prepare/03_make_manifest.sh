#!/usr/bin/env bash
# ESC-50 step 3/3 — re-freeze manifest.csv.gz + classes.csv (byte-identical to the committed files) and
# verify the staged media with yavih.
#
#   bash 03_make_manifest.sh   # media root = $(yavih config get esc50.media_root)
#   YAVIH_DATA__ESC50__MEDIA_ROOT=/path/to/datasets/esc50 bash 03_make_manifest.sh
#   bash 03_make_manifest.sh --check   # re-derive and compare, write nothing
#
# Needs the clone of step 1: meta/esc50.csv (sha256 ca660da6…) and audio/ for presence + WAV headers.
# `--no-media --metadata <meta/esc50.csv>` re-freezes from the metadata alone.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
MEDIA_ROOT="$(yavih config get esc50.media_root)"

"$PY" "$HERE/../make_manifest.py" --media-root "$MEDIA_ROOT" "$@"
# sha256 is over the DECOMPRESSED manifest; compare the digest of the zcat stream.
got=$(zcat "$HERE/../manifest.csv.gz" | sha256sum | cut -d' ' -f1)
[ "$got" = 4846c6d167ff53e1caf060ce7e9a45c80fff508e05492ef0073639082ee4079c ] || { echo "FAIL: manifest sha256 $got" >&2; exit 1; }
echo "manifest.csv (decompressed): OK"
echo "a614223f2aba50003f58a7a2e8ce95579e493b4635b41bc9e19ff0945e9d06ff  $HERE/../classes.csv" | sha256sum -c -
YAVIH_DATA__ESC50__MEDIA_ROOT="$MEDIA_ROOT" yavih bench verify esc50 --json
