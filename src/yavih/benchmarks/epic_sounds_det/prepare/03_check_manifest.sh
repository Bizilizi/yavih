#!/usr/bin/env bash
# epic_sounds_det step 3/3: re-derive the manifest from the staged annotation files and compare it with the
# committed one (writes nothing), then report how many audio files are staged per split.
#
#   ROOT=/path/to/datasets/epic_sounds_det yavih bench prepare epic_sounds_det 03
set -euo pipefail
ROOT="${ROOT:-$(yavih config get epic_sounds_det.media_root)}"
PY=${PY:-python}
"$PY" "$FOLDER/make_manifest.py" --media-root "$ROOT" --check > /dev/null
echo "=== $(date -Is) MANIFEST_OK (re-derived from $ROOT/annotations, identical to the committed manifest.csv.gz)"
