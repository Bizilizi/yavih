#!/usr/bin/env bash
# speech_commands step 1/3 — download the two Speech Commands v0.02 archives HEAR builds the task from, verify
# them (sha256, and the md5 hear-preprocess pins), and extract them: the train archive to <media_root>/train/,
# the test archive to <media_root>/test/ (the directory names of HEAR's ExtractArchive tasks).
#
#   yavih bench prepare speech_commands 01                                  # media root = $(yavih config get speech_commands.media_root)
#   ROOT=/path/to/datasets/speech_commands yavih bench prepare speech_commands 01
#   DOWNLOADS=/path/with/the/archives ROOT=... yavih bench prepare speech_commands 01   # archives already downloaded
#
# Public, no account: http://download.tensorflow.org/data/ (Google, CC BY 4.0). Sizes: 2 428 923 189 B and
# 112 563 277 B. Extracting ~110 000 small files takes minutes on a local disk and up to an hour on a network file
# system. Re-running skips what is done (a `.extracted` marker per archive).
set -euo pipefail
ROOT="${ROOT:-$(yavih config get speech_commands.media_root)}"
DOWNLOADS="${DOWNLOADS:-$ROOT/_downloads}"
mkdir -p "$ROOT" "$DOWNLOADS"

fetch() {  # name url sha256 md5
  local name=$1 url=$2 sha=$3 md5=$4 f="$DOWNLOADS/$1"
  if [ ! -f "$f" ] || ! echo "$sha  $f" | sha256sum -c --status -; then
    echo "=== $(date -Is) downloading $url ==="
    curl -fL --retry 5 -o "$f.part" "$url"
    mv "$f.part" "$f"
  fi
  echo "$sha  $f" | sha256sum -c -
  echo "$md5  $f" | md5sum -c -
}

extract() {  # name dir
  local f="$DOWNLOADS/$1" out="$ROOT/$2"
  if [ -f "$out/.extracted" ]; then
    echo "=== $out already extracted ==="
    return
  fi
  echo "=== $(date -Is) extracting $1 -> $out ==="
  mkdir -p "$out"
  tar -xzf "$f" -C "$out"
  date -Is > "$out/.extracted"
}

fetch speech_commands_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz \
  af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58 6b74f3901214cb2c2934e98196829835
fetch speech_commands_test_set_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz \
  cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df 854c580ee90bff80c516491c84544e32
extract speech_commands_v0.02.tar.gz train
extract speech_commands_test_set_v0.02.tar.gz test

n_train=$(find "$ROOT/train" -mindepth 2 -maxdepth 2 -name '*.wav' -not -path '*/_background_noise_/*' | wc -l)
n_test=$(find "$ROOT/test" -mindepth 2 -maxdepth 2 -name '*.wav' | wc -l)
[ "$n_train" -eq 105829 ] || { echo "FAIL: $n_train word wavs under $ROOT/train, expected 105829" >&2; exit 1; }
[ "$n_test" -eq 4890 ] || { echo "FAIL: $n_test wavs under $ROOT/test, expected 4890" >&2; exit 1; }
echo "=== OK: $n_train train-archive and $n_test test-archive wavs under $ROOT ==="
