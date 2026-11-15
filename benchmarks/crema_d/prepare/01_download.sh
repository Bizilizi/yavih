#!/usr/bin/env bash
# CREMA-D step 1/3 — clone the public GitHub repository at the pinned commit WITHOUT its Git LFS media (the
# working tree holds the LFS pointer files: `oid sha256:<hex>` and `size <bytes>` of every wav), then download
# the 7 438 wavs the manifest lists from GitHub's LFS media endpoint at that commit and check each one against
# its pointer. No account, no agreement, no git-lfs needed.
#
#   yavih bench prepare crema_d 01                 # root = $(yavih config get crema_d.media_root)
#   ROOT=/path/to/datasets/crema_d yavih bench prepare crema_d 01
#   JOBS=4 yavih bench prepare crema_d 01          # parallel downloads (default 4)
#
# Size: the sparse clone (AudioWAV pointers, processedResults/, LICENSE.txt) is a few MB; the 7 438 wavs are
# 605 MB (16 kHz mono PCM_16). Re-running skips every wav already present with the right sha256.
# Layout written: $ROOT/AudioWAV/<name>.wav (the pointer replaced by the wav), $ROOT/processedResults/summaryTable.csv,
# $ROOT/LICENSE.txt.
set -euo pipefail
ROOT="${ROOT:-$(yavih config get crema_d.media_root)}"
JOBS="${JOBS:-4}"
REPO=https://github.com/CheyneyComputerScience/CREMA-D.git
COMMIT=1658cd342dff90010aa843eaeebd53610a08b1dc   # 2025-03-21; AudioWAV unchanged since 2017 for the 7 438 wavs listed
MEDIA=https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/$COMMIT/AudioWAV

if [ -d "$ROOT/.git" ]; then
  echo "=== $ROOT is already a clone ($(git -C "$ROOT" rev-parse HEAD)); skipping the clone ==="
else
  mkdir -p "$ROOT"
  echo "=== $(date -Is) sparse clone of $REPO -> $ROOT (no LFS smudge) ==="
  GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --no-checkout --filter=blob:none "$REPO" "$ROOT"
  git -C "$ROOT" sparse-checkout set --no-cone /AudioWAV/ /processedResults/ /LICENSE.txt
  GIT_LFS_SKIP_SMUDGE=1 git -C "$ROOT" -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false \
    checkout --quiet --detach "$COMMIT"
fi
[ "$(git -C "$ROOT" rev-parse HEAD)" = "$COMMIT" ] || { echo "FAIL: $ROOT is not at $COMMIT" >&2; exit 1; }
echo "fb0e382700d32c0d9adc0c6815292215a36bcf912f2657048862c91e3bd2b317  $ROOT/processedResults/summaryTable.csv" | sha256sum -c -
echo "e76747a4efa4b338914bb3f930b5d3d3c3ac15b886bbabf897f6a19731f987f8  $ROOT/LICENSE.txt" | sha256sum -c -

# The wav names the manifest lists: column 2 (FileName) of summaryTable.csv minus the four names TFDS's crema_d
# builder skips as absent from GitHub (1040_ITH_SAD_XX 1006_TIE_NEU_XX 1013_WSI_DIS_XX 1017_IWW_FEA_XX).
names=$(mktemp)
trap 'rm -f "$names"' EXIT
tail -n +2 "$ROOT/processedResults/summaryTable.csv" | cut -d, -f2 | tr -d '"\r' \
  | grep -v -x -e 1040_ITH_SAD_XX -e 1006_TIE_NEU_XX -e 1013_WSI_DIS_XX -e 1017_IWW_FEA_XX > "$names"
[ "$(wc -l < "$names")" -eq 7438 ] || { echo "FAIL: $(wc -l < "$names") names, expected 7438" >&2; exit 1; }
echo "=== $(date -Is) $(wc -l < "$names") wavs to fetch with $JOBS parallel downloads ==="

fetch_one() {   # $1 = wav name; downloads $MEDIA/$1.wav unless the file already has the pointer's sha256
  local name=$1 dst="$ROOT/AudioWAV/$1.wav" ptr oid size got
  ptr=$(git -C "$ROOT" show "$COMMIT:AudioWAV/$name.wav")
  oid=$(printf '%s\n' "$ptr" | sed -n 's/^oid sha256:\([0-9a-f]\{64\}\)$/\1/p')
  size=$(printf '%s\n' "$ptr" | sed -n 's/^size \([0-9]*\)$/\1/p')
  [ -n "$oid" ] && [ -n "$size" ] || { echo "FAIL: no LFS pointer for $name" >&2; return 1; }
  if [ "$(stat -c %s "$dst" 2>/dev/null || echo 0)" = "$size" ] && [ "$(sha256sum "$dst" | cut -d' ' -f1)" = "$oid" ]; then
    return 0
  fi
  curl -fsSL --retry 5 --retry-delay 3 -o "$dst.part" "$MEDIA/$name.wav"
  got=$(sha256sum "$dst.part" | cut -d' ' -f1)
  [ "$got" = "$oid" ] || { echo "FAIL: $name sha256 $got != pointer $oid" >&2; rm -f "$dst.part"; return 1; }
  mv -f "$dst.part" "$dst"
}
export -f fetch_one
export ROOT COMMIT MEDIA
xargs -P "$JOBS" -I{} bash -c 'fetch_one "$@"' _ {} < "$names"
echo "=== $(date -Is) DONE: $(wc -l < "$names") wavs match their LFS pointers; $(du -sh "$ROOT/AudioWAV" | cut -f1) in AudioWAV ==="
