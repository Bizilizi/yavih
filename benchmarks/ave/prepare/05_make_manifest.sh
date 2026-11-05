#!/usr/bin/env bash
# AVE step 5/5 — check manifest.csv.gz + classes.csv against the public upstream files and run `bench verify`.
# make_manifest.py --check re-derives the 3 248 rows from the upstream files (steps 1-3) and VGGSound's
# metadata/vggsound.csv (886 VGGSound-train rows + 9 cross-split eval rows dropped) and compares them byte for byte
# with the committed files; it writes nothing.
#
#   ROOT=/path/to/datasets/ave VGGSOUND_CSV=/path/to/vggsound/metadata/vggsound.csv yavih bench prepare ave 05
# Both variables are optional overrides. Unset, make_manifest.py / bench verify resolve them through yavih's config:
# ROOT = the folder's media_root, VGGSOUND_CSV = $([datasets] vggsound)/metadata/vggsound.csv.
set -euo pipefail
FOLDER="${FOLDER:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=${PY:-python}
ARGS=()
[ -n "${ROOT:-}" ] && ARGS+=(--media-root "$ROOT")
[ -n "${VGGSOUND_CSV:-}" ] && ARGS+=(--vggsound-csv "$VGGSOUND_CSV")
"$PY" "$FOLDER/make_manifest.py" --check ${ARGS[@]+"${ARGS[@]}"}
# expected stderr: "3248 rows from 4143 upstream events …" and "--check: manifest.csv.gz identical, classes.csv identical"
if [ -n "${ROOT:-}" ]; then export YAVIH_DATA__AVE__MEDIA_ROOT="$ROOT"; fi
yavih bench verify ave --census
# expected: status verified; train 2624/2624, test 316/316 present; read sample ok; media-info table written
# (header facts of every member)
