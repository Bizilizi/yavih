#!/usr/bin/env bash
# ssv2 step 4 — re-freeze check and data verification (setup.md "Prepare & verify"). Nothing is downloaded or
# written outside $YAVIH_HOME/data_status.
#
# Usage: bash 04_verify.sh [ROOT]     (ROOT defaults to the resolved ssv2.media_root)
#   step 1 (maintainers) needs torch and the official ROOT/labels/; it is skipped when either is missing.
#   PYTHON / YAVIH select the interpreter and CLI: PYTHON=/path/to/python YAVIH=/path/to/yavih yavih bench prepare ssv2 04 -- [ROOT]
#   A user only needs step 2: yavih bench verify ssv2 --census
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
YAVIH="${YAVIH:-yavih}"
if [ -n "${1:-}" ]; then export YAVIH_DATA__SSV2__MEDIA_ROOT="$1"; fi
# 1. the committed manifest, classes and curation.json are byte-identical to a re-derivation from the official
#    labels/{labels,train,validation}.json (sha256 asserted; torch for the class-balanced draw)
ROOT="$("$YAVIH" config get ssv2.media_root)"
LABELS="$ROOT/labels"
if [ -f "$LABELS/train.json" ] && "$PY" -c "import torch" 2>/dev/null; then
  "$PY" "$HERE/../make_manifest.py" --check --labels "$LABELS" | grep -E '"check"'
else
  echo "[ssv2/04] make_manifest --check skipped (needs torch and $LABELS)"
fi
# 2. presence of all 44 777 webm (one scandir of the flat dir), read + decode sample, curation sums, and
#    (--census) the site media-info table: webm headers carry no frame count, so the guaranteed
#    info key n_frames exists only through the census's packet count. Run it once per site.
"$YAVIH" bench verify ssv2 --census --json
# Expected (setup.md "Prepare & verify"): manifest_sha_ok true, per_split train 20000/20000 and
# val 24777/24777 present, read failed 0, decode_sample failed 0, labels_ok true, curation_invariants_ok
# true, media_info rows 44777, external {}.
