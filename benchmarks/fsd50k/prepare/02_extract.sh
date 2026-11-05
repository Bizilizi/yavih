#!/usr/bin/env bash
# FSD50K step 2/3 — merge the multi-part zips (`zip -s 0`) and extract into the media root with the
# SHORT directory names the frozen manifest uses (dev_audio/, eval_audio/, ground_truth/, metadata/).
#
#   DL=/path/to/fsd50k_download OUT=/path/to/datasets/fsd50k bash 02_extract.sh
#   (defaults: DL=$(yavih config get data.root)/fsd50k_download, OUT=$(yavih config get fsd50k.media_root))
#
# Expected result: dev_audio/ 40 966 wav (24 GB), eval_audio/
# 10 231 wav (8.3 GB), ground_truth/{dev.csv,eval.csv,vocabulary.csv}, metadata/, FSD50K.doc/ — 33 GB,
# 51 210 files. Needs `zip` and `unzip` (Info-ZIP). The merged *.unsplit.zip is a TEMPORARY copy of the
# largest archive (18.4 GB for dev = 5 x 3 221 225 472 + 2 306 663 327 bytes): plan 43 GB free in $DL
# during this step (24.7 GB of archives stay + the merged copy). Never extract in place
# inside $DL: the frozen rel_paths are relative to $OUT.
set -euo pipefail
DL="${DL:-$(yavih config get data.root)/fsd50k_download}"
OUT="${OUT:-$(yavih config get fsd50k.media_root)}"
mkdir -p "$OUT"

extract() { # extract <stem> <targetname>   e.g. extract FSD50K.eval_audio eval_audio
  local stem=$1 tgt=$2
  echo "=== $(date -Is) merging+extracting $stem ==="
  rm -f "$DL/${stem}.unsplit.zip"
  zip -q -s 0 "$DL/${stem}.zip" --out "$DL/${stem}.unsplit.zip"
  unzip -q -o "$DL/${stem}.unsplit.zip" -d "$OUT"
  rm -f "$DL/${stem}.unsplit.zip"
  if [ -d "$OUT/$stem" ] && [ ! -d "$OUT/$tgt" ]; then mv "$OUT/$stem" "$OUT/$tgt"; fi
  echo "=== $(date -Is) EXTRACTED $tgt: $(find "$OUT/$tgt" -type f | wc -l) files ==="
}

unzip -q -o "$DL/FSD50K.ground_truth.zip" -d "$OUT"
[ -d "$OUT/FSD50K.ground_truth" ] && [ ! -d "$OUT/ground_truth" ] && mv "$OUT/FSD50K.ground_truth" "$OUT/ground_truth"
unzip -q -o "$DL/FSD50K.metadata.zip" -d "$OUT"
[ -d "$OUT/FSD50K.metadata" ] && [ ! -d "$OUT/metadata" ] && mv "$OUT/FSD50K.metadata" "$OUT/metadata"
unzip -q -o "$DL/FSD50K.doc.zip" -d "$OUT" || true      # FSD50K.doc/ keeps its official name (README.md + LICENSE-DATASET)

# Vocabulary must be the official file byte for byte (classes.csv is derived from it).
echo "992b6e11422101ee575ab3022d37b79b98e19407b211331c72f5d9fdd16e7174  $OUT/ground_truth/vocabulary.csv" | sha256sum -c -
echo "ce4ab5f01baa5a52b0ec892edaf137a7560743fce3a70155f8bf975c197ded1b  $OUT/ground_truth/dev.csv" | sha256sum -c -
echo "ef861a4619691783bfceb7f43e1a395dcf969150ba4ad34dcb36b93abd26b3d6  $OUT/ground_truth/eval.csv" | sha256sum -c -

extract FSD50K.eval_audio eval_audio
extract FSD50K.dev_audio dev_audio
dev=$(find "$OUT/dev_audio" -name '*.wav' | wc -l); ev=$(find "$OUT/eval_audio" -name '*.wav' | wc -l)
echo "=== $(date -Is) ALL_DONE dev_wavs=$dev eval_wavs=$ev du=$(du -sh "$OUT" | cut -f1) ==="
[ "$dev" -eq 40966 ] && [ "$ev" -eq 10231 ] || { echo "FAIL: expected 40966 dev / 10231 eval wavs" >&2; exit 1; }
