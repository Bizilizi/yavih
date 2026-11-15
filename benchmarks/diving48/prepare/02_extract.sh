#!/usr/bin/env bash
# diving48 step 2/3 — extract rgb/ from Diving48_rgb.tar.gz into the media root and check the extracted tree.
#
#   yavih bench prepare diving48 02                         # in $(yavih config get diving48.media_root)
#   ROOT=/path/to/datasets/diving48 yavih bench prepare diving48 02
#   KEEP_ARCHIVE=0 yavih bench prepare diving48 02          # delete the 10.3 GB archive once the tree checks out
#
# Layout after this step (what benchmark.yaml `data.roots.media_root` describes):
#   <ROOT>/rgb/<vid_name>.mp4                     18 404 files (10 313 555 478 B), one flat directory
#   <ROOT>/Diving48_V2_{train,test}.json, Diving48_vocab.json    (step 01)
# The archive holds every clip of the 2018 release; the frozen manifest reads the 16 997 V2 clips among them.
# The check: the number of mp4 files and the digest of the tree, sha256 of
# `cd rgb && export LC_ALL=C && sha256sum *.mp4` (one line per file in C-locale order), which reads ~10 GB.
# Set VERIFY_DIGEST=0 to skip the digest (the count and step 03's presence check still run).
set -euo pipefail
ROOT="${ROOT:-$(yavih config get diving48.media_root)}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-1}"
VERIFY_DIGEST="${VERIFY_DIGEST:-1}"
WANT_FILES=18404
WANT_DIGEST=c5aaecb78e29f96d81749fd7c36a7e5a79887cf106814ecd51f523b2d9bfd536
cd "$ROOT"

if [ -d rgb ] && [ "$(find rgb -maxdepth 1 -name '*.mp4' | wc -l)" -eq "$WANT_FILES" ]; then
  echo "=== rgb/ already holds $WANT_FILES mp4 files; not extracting again"
else
  [ -f Diving48_rgb.tar.gz ] || { echo "FAIL: $ROOT/Diving48_rgb.tar.gz missing: run step 01 first" >&2; exit 1; }
  echo "=== $(date -Is) extracting Diving48_rgb.tar.gz -> $ROOT/rgb"
  tar -xzf Diving48_rgb.tar.gz --no-same-owner rgb/
fi
n=$(find rgb -maxdepth 1 -name '*.mp4' | wc -l)
[ "$n" -eq "$WANT_FILES" ] || { echo "FAIL: $n mp4 files in $ROOT/rgb, expected $WANT_FILES" >&2; exit 1; }
echo "=== $n mp4 files"
if [ "$VERIFY_DIGEST" = 1 ]; then
  digest=$(cd rgb && export LC_ALL=C && sha256sum -- *.mp4 | sha256sum | cut -d' ' -f1)
  [ "$digest" = "$WANT_DIGEST" ] || { echo "FAIL: rgb/ tree digest $digest != $WANT_DIGEST" >&2; exit 1; }
  echo "=== rgb/ tree digest ok ($digest)"
fi
if [ "$KEEP_ARCHIVE" = 0 ]; then rm -f Diving48_rgb.tar.gz; echo "=== archive deleted (KEEP_ARCHIVE=0)"; fi
echo "=== $(date -Is) OK: $ROOT/rgb staged; next: yavih bench prepare diving48 03"
