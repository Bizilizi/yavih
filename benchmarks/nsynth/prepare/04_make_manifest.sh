#!/usr/bin/env bash
# nsynth step 4/4 — re-derive manifest.csv.gz + classes.csv (HEAR's nsynth_pitch 5h selection, byte-identical to
# the committed files) from the staged examples.json files, check the staged wavs, and verify with yavih.
#
#   yavih bench prepare nsynth 04             # --check: re-derive and compare, write nothing (the default here)
#   yavih bench prepare nsynth 04 -- --write  # re-freeze in a git checkout (rewrites the folder's files)
#
# Needs steps 01-02 (the three examples.json and the 5 000 listed wavs under the media root).
set -euo pipefail
FOLDER="${FOLDER:-$(yavih bench path nsynth)}"
PY=${PY:-python}
MEDIA_ROOT="${ROOT:-$(yavih config get nsynth.media_root)}"
if [ "${1:-}" = "--write" ]; then shift; else set -- --check "$@"; fi
"$PY" "$FOLDER/make_manifest.py" --media-root "$MEDIA_ROOT" "$@" > /dev/null
got=$(zcat "$FOLDER/manifest.csv.gz" | sha256sum | cut -d' ' -f1)   # benchmark.yaml pins the DECOMPRESSED bytes
[ "$got" = 949e49b1030370a188f538c71b3150d38017263fc1314cb5f81a47f806f0b7c0 ] || { echo "FAIL: manifest sha256 $got" >&2; exit 1; }
echo "manifest.csv (decompressed): OK"
echo "1cd9e961bb471b9a7da1adf8dc9b8c952f84eba99d77a476bb356857c3eb1470  $FOLDER/classes.csv" | sha256sum -c -
YAVIH_DATA__NSYNTH__MEDIA_ROOT="$MEDIA_ROOT" yavih bench verify nsynth --json
