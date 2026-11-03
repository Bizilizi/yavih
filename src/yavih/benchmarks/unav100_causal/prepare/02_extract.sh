#!/usr/bin/env bash
# unav100_causal — step 2: the videos as raw_videos/<video_id>.mp4 under the media root.
# Extracts the concatenated archive (its members sit under ssd3/gengtiantian/audio_visual_data/unav100/raw_videos/;
# --strip-components 4 drops the prefix). An extraction made elsewhere: LINK=<dir holding the mp4s> links it instead.
set -euo pipefail
ROOT="$(yavih config get unav100_causal.media_root)"
if [[ -n "${LINK:-}" ]]; then
  ln -sfn "$LINK" "$ROOT/raw_videos"
else
  mkdir -p "$ROOT"
  cat "$ROOT"/unav_raw_videos.tar.gz.part_* | tar -xz -C "$ROOT" --strip-components 4 \
    ssd3/gengtiantian/audio_visual_data/unav100/raw_videos
fi
n=$(find -L "$ROOT/raw_videos" -maxdepth 1 -name '*.mp4' | wc -l)
echo "raw_videos: $n mp4 (10 790 expected)"
[[ "$n" -eq 10790 ]]
