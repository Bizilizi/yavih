#!/usr/bin/env bash
# diving48 step 3/3 — re-derive manifest.csv.gz + classes.csv from the staged V2 JSON files and compare them with the
# committed files (`--check`: writes nothing), with the presence of every listed rgb/<vid_name>.mp4, then verify the
# staged media with yavih and write the site's media-info census.
#
#   yavih bench prepare diving48 03            # media root = $(yavih config get diving48.media_root)
#   yavih bench prepare diving48 03 -- --write # re-freeze instead of --check (a git checkout of the repository only)
#
# Success: make_manifest.py prints `--check: manifest.csv identical ... classes.csv identical` and "missing per split
# 0"; `yavih bench verify diving48 --census` reports status verified, train 15027/15027 and test 1970/1970 present.
set -euo pipefail
HERE="${FOLDER:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=${PY:-python}
MEDIA_ROOT="$(yavih config get diving48.media_root)"
MODE=--check
[ "${1:-}" = --write ] && MODE=

"$PY" "$HERE/make_manifest.py" --media-root "$MEDIA_ROOT" $MODE
got=$(zcat "$HERE/manifest.csv.gz" | sha256sum | cut -d' ' -f1)   # benchmark.yaml manifest.sha256 is over the decompressed bytes
[ "$got" = cab70696095ce4f7c12feebbb13678828610786cbf6f50690b8cc720979e3d1a ] || { echo "FAIL: manifest sha256 $got" >&2; exit 1; }
echo "be976e05a7e4d67b8ff772eace7f599bb70dfe77c674189fabf03cbd257d3b99  $HERE/classes.csv" | sha256sum -c -
YAVIH_DATA__DIVING48__MEDIA_ROOT="$MEDIA_ROOT" yavih bench verify diving48 --census
