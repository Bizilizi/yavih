#!/usr/bin/env bash
# Extract the seven verified parts into <media_root>/AV_SyncBench/ (about 70 GB: 3 494 mp4, 37 743 wav and the two
# metadata.jsonl files), streaming them into tar (no concatenated copy). Refuses a non-empty AV_SyncBench/.
# DELETE_PARTS=1 removes raw/ afterwards (68 GB). Usage: yavih bench prepare av_syncbench 02
set -euo pipefail
ROOT="${ROOT:-$(yavih config get av_syncbench.media_root)}"
RAW="$ROOT/raw"
for p in aa ab ac ad ae af ag; do
  [ -f "$RAW/AV_SyncBench.tar.gz.part_$p.ok" ] || { echo "part_$p is not verified: run yavih bench prepare av_syncbench 01" >&2; exit 3; }
done
if [ -d "$ROOT/AV_SyncBench" ] && [ -n "$(ls -A "$ROOT/AV_SyncBench")" ]; then
  echo "$ROOT/AV_SyncBench is not empty: refusing to extract over it" >&2; exit 2
fi
cat "$RAW"/AV_SyncBench.tar.gz.part_a[a-g] | tar --no-same-owner --no-same-permissions -xzf - -C "$ROOT"
for d in hf_temporal_challenge hf_timbre_edit; do
  echo "$d: $(ls "$ROOT/AV_SyncBench/$d/original_video" | wc -l) mp4, $(ls "$ROOT/AV_SyncBench/$d/edited_audio" | wc -l) wav"
done   # expected: 2676 mp4 + 36925 wav; 818 mp4 + 818 wav
if [ "${DELETE_PARTS:-0}" = "1" ]; then rm -rf "$RAW"; fi
echo "next: yavih bench prepare av_syncbench 03"
