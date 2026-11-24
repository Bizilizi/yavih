#!/usr/bin/env bash
# nsynth step 1/4 — download the three NSynth "json/wav" archives (public, CC BY 4.0, no account) into
# <media_root>/archives/ and check each against the md5 HEAR pins (hearpreprocess nsynth_pitch.py) and its sha256.
#
#   yavih bench prepare nsynth 01                       # media root = $(yavih config get nsynth.media_root)
#   ROOT=/path/to/datasets/nsynth yavih bench prepare nsynth 01
#   SPLITS="valid test" yavih bench prepare nsynth 01  # a subset of the archives
#
# Sizes: train 23 815 298 079 B, valid 1 068 767 009 B, test 349 501 546 B (25.2 GB in all; the train archive is
# needed in full because its examples.json sits at its end and the 4 000 frozen train notes are spread through
# it). curl resumes a partial download (-C -). An archive already present with the right sha256 is skipped.
set -euo pipefail
ROOT="${ROOT:-$(yavih config get nsynth.media_root)}"
SPLITS="${SPLITS:-train valid test}"
BASE=http://download.magenta.tensorflow.org/datasets/nsynth
declare -A MD5=(
  [train]=fde6665a93865503ba598b9fac388660
  [valid]=87e94a00a19b6dbc99cf6d4c0c0cae87
  [test]=5e6f8719bf7e16ad0a00d518b78af77d
)
declare -A SHA256=(
  [train]=15c5100fc40c3a262f9b42707c15ffed30b8fec58c0d9b88d4136f6cb1fddefe
  [valid]=00dea2645fbe0069258567da30807a90825e0bab54c077d6481f253096c4e2a0
  [test]=0f9ba5d62beba9ec4612f918d19f5e87a681822f1c566124f05fe8b27a51934c
)
mkdir -p "$ROOT/archives"
for s in $SPLITS; do
  f="$ROOT/archives/nsynth-$s.jsonwav.tar.gz"
  if [ -f "$f" ] && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "${SHA256[$s]}" ]; then
    echo "=== $f present, sha256 ok ==="
    continue
  fi
  echo "=== $(date -Is) downloading $BASE/nsynth-$s.jsonwav.tar.gz -> $f ==="
  curl -fL --retry 5 -C - -o "$f.part" "$BASE/nsynth-$s.jsonwav.tar.gz"
  mv "$f.part" "$f"
  echo "${MD5[$s]}  $f" | md5sum -c -
  echo "${SHA256[$s]}  $f" | sha256sum -c -
done
echo "=== OK: archives in $ROOT/archives ==="
