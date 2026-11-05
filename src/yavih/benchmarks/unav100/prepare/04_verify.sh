#!/usr/bin/env bash
# UnAV-100 step 4/4 — check manifest.csv.gz + classes.csv against the downloaded annotations, then `bench verify`.
# make_manifest.py --check re-derives the 10 790 rows from $ROOT/upstream/data/*.parquet (sha256-pinned; reading them
# needs pyarrow) and compares them byte for byte with the committed files; it writes nothing. Presence comes from
# $ROOT/index.json (step 03).
#
#   yavih bench prepare unav100 04                     # ROOT defaults to `yavih config get unav100.media_root`
set -euo pipefail
FOLDER="${FOLDER:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=${PY:-python}
ARGS=()
[ -n "${ROOT:-}" ] && ARGS+=(--media-root "$ROOT")
"$PY" "$FOLDER/make_manifest.py" --check ${ARGS[@]+"${ARGS[@]}"} > /dev/null
# expected stderr: "10790 videos, 30059 events, 100 classes …", "presence: 0 missing", "--check: … identical"
if [ -n "${ROOT:-}" ]; then export YAVIH_DATA__UNAV100__MEDIA_ROOT="$ROOT"; fi
yavih bench verify unav100 --census
# expected: train 6489/6489, test 2167/2167 present; read sample ok; media-info table written
