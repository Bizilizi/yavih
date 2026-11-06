#!/usr/bin/env bash
# epic_sounds_det step 1/3: the public annotation files the manifest is frozen from, plus the EPIC-KITCHENS
# checksum list step 02 verifies every video against. All on raw.githubusercontent.com, pinned commits, no account.
#
#   ROOT=/path/to/datasets/epic_sounds_det yavih bench prepare epic_sounds_det 01   # default: `yavih config get epic_sounds_det.media_root`
#
# Disk: 35.5 MB, 6 files under $ROOT/annotations/.
set -euo pipefail
ROOT="${ROOT:-$(yavih config get epic_sounds_det.media_root)}"
ES='https://raw.githubusercontent.com/epic-kitchens/epic-sounds-annotations/57a922f0d352e9429f1ef8a37eee21758dd3a33c'
C10='https://raw.githubusercontent.com/epic-kitchens/C10-epic-sounds-detection/196e7cf2c4f17946279fb57d9ff24516b7403344'
DL='https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-download-scripts/4f11fb2b579833f360c3c7bb917bf1e24a9787b5'
mkdir -p "$ROOT/annotations"

get() { # get <url> <dest> <bytes> <sha256>
  local url=$1 dest=$2 bytes=$3 sha=$4
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$bytes" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    echo "=== OK (already present) $dest"; return 0
  fi
  echo "=== $(date -Is) downloading $url"
  curl -fsSL --retry 5 --retry-delay 10 -o "$dest.part" "$url"
  local got_b got_s; got_b=$(stat -c %s "$dest.part"); got_s=$(sha256sum "$dest.part" | cut -d' ' -f1)
  if [ "$got_b" != "$bytes" ] || [ "$got_s" != "$sha" ]; then
    echo "FAIL $dest bytes=$got_b (want $bytes) sha256=$got_s (want $sha)" >&2; return 1
  fi
  mv "$dest.part" "$dest"
  echo "=== $(date -Is) OK $dest"
}

A="$ROOT/annotations"
get "$ES/EPIC_Sounds_train.csv"                       "$A/EPIC_Sounds_train.csv"                       6184177 225427ab288cc5c72c1680abb29f5ab8ccd3608c3f4cf409541d2c5cf720dd32
get "$ES/EPIC_Sounds_validation.csv"                  "$A/EPIC_Sounds_validation.csv"                  802681  fff66485d8478762fd9cde1a829e6d91713a5f2f7f917e3af746cf891dd62e5a
get "$ES/EPIC_Sounds_recognition_test_timestamps.csv" "$A/EPIC_Sounds_recognition_test_timestamps.csv" 338247  4912b017f96e371b3d05ef6194fb2796ecffff3782eacf8909a3cbaf2af4ef3a
get "$ES/EPIC_Sounds_detection_test_videos.csv"       "$A/EPIC_Sounds_detection_test_videos.csv"       193     b77ec524fdd64be5b01db6bc8472a983d58acb767c854e3d40ac98cdf4ec9bc3
# EPIC-KITCHENS md5 list: one row per video (videos/<split>/<P>/<id>.MP4 for EPIC-55, <P>/videos/<id>.MP4 for the extension)
get "$DL/data/md5.csv"                                "$A/epic_kitchens_md5.csv"                       236630  3f9c31899ff5807c8f0fed5d26a8fe62300f3f4c40dc3eea767345212c2e83c7
# the C10 baseline detections on validation: scored by tests/test_folder.py to pin probe.py to the published Table V row
get "$C10/EvaluationCode/actionformer_baseline_validation.json" "$A/actionformer_baseline_validation.json" 27866254 5a71da70f0e098b201a64551264fe8043d93a45c3f8956c1aac5ac5c7ee6f389
echo "=== $(date -Is) ANNOTATIONS_DONE in $A"
