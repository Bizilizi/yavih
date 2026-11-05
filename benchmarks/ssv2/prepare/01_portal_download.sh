#!/usr/bin/env bash
# ssv2 step 1 — obtain the official release (setup.md "Obtain the data"). MANUAL STEP: nothing here downloads.
#
# Something-Something-v2 is distributed by Qualcomm (formerly TwentyBN) behind a free registration +
# licence acceptance: https://developer.qualcomm.com/software/ai-datasets/something-something
# A human must log in, accept the licence and download the files below into ONE new directory you own
# (DL_DIR). Do NOT try Hugging Face: HuggingFaceM4/something_something_v2 hosts no media (loading script
# only, manual_download_instructions → the Qualcomm portal).
# If an extracted copy of the release already exists, skip to 03_stage_from_mirror.sh.
#
# Expected files of the official release (sizes in bytes, sha256):
#   20bn-something-something-v2-00                         10000000000  aab5bf3badcdf58aed2ac4b6c219a910ea97069b346bcb3ab19202b42dc8f0aa
#   20bn-something-something-v2-01                          9444975475  65b505fd26e7e72b745421d33508a5b5fcb7544c6b5510e39aa754282bad8d24
#   20bn-something-something-download-package-labels.zip       5140376  8e9859f2862c3626dd1cd013dbc0bccccc12ab48d7c18812e3f302988d6f2c0c
# The parts are byte slices of ONE gzip'd tar (magic 1f8b at offset 0 of -00): concatenate, then untar
# (02_extract.sh). Older 20BN mirrors split the same tar into more, smaller parts (00..19); the
# concatenation is what matters — check the extracted count (220 847 webm), not the part count.
#
# Usage: bash 01_portal_download.sh DL_DIR      # verifies what the human downloaded
set -euo pipefail
DL="${1:?usage: 01_portal_download.sh DL_DIR (the directory holding the downloaded parts + labels zip)}"
declare -A SIZE=(
  [20bn-something-something-v2-00]=10000000000
  [20bn-something-something-v2-01]=9444975475
  [20bn-something-something-download-package-labels.zip]=5140376
)
declare -A SHA=(
  [20bn-something-something-v2-00]=aab5bf3badcdf58aed2ac4b6c219a910ea97069b346bcb3ab19202b42dc8f0aa
  [20bn-something-something-v2-01]=65b505fd26e7e72b745421d33508a5b5fcb7544c6b5510e39aa754282bad8d24
  [20bn-something-something-download-package-labels.zip]=8e9859f2862c3626dd1cd013dbc0bccccc12ab48d7c18812e3f302988d6f2c0c
)
rc=0
for f in "${!SIZE[@]}"; do
  p="$DL/$f"
  if [ ! -f "$p" ]; then echo "[ssv2/01] MISSING $p — download it from the Qualcomm portal (manual step)"; rc=1; continue; fi
  s=$(stat -c %s "$p")
  [ "$s" = "${SIZE[$f]}" ] || { echo "[ssv2/01] SIZE $f = $s, expected ${SIZE[$f]} (a different split of the same tar is OK only if 02's count check passes)"; rc=1; continue; }
  h=$(sha256sum "$p" | cut -d' ' -f1)
  [ "$h" = "${SHA[$f]}" ] && echo "[ssv2/01] ok $f" || { echo "[ssv2/01] SHA256 $f = $h, expected ${SHA[$f]}"; rc=1; }
done
exit $rc
