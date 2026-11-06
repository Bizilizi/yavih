#!/usr/bin/env bash
# epic100_anticipation_stream — step 2: the train and validation videos with the official downloader (pinned commit),
# into <media_root>/EPIC-KITCHENS/<participant>/videos/<video_id>.MP4; the downloader checks its md5 list.
# A partial or torrent download made elsewhere can be linked into that layout instead (setup.md "Gotchas").
# The downloader md5-checks files it finds but not the ones it has just written: a dropped connection leaves a
# truncated MP4 (no moov atom; step 3's census reports the cell as failed). Delete such files and re-run this step
# with EPIC_VIDEOS=P05_03,P15_09 (the downloader's --specific-videos) to fetch only those.
set -euo pipefail
ROOT="$(yavih config get epic100_anticipation_stream.media_root)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone -q https://github.com/epic-kitchens/epic-kitchens-download-scripts.git "$TMP/dl"
git -C "$TMP/dl" checkout -q 4f11fb2b579833f360c3c7bb917bf1e24a9787b5
# --action-recognition: its train + val splits are exactly the 633 manifest videos (all challenges would add 13 more)
cd "$TMP/dl"   # the downloader reads its data/*.csv relative to its own directory
python epic_downloader.py --videos --action-recognition --train --val --output-path "$ROOT" \
  ${EPIC_VIDEOS:+--specific-videos "$EPIC_VIDEOS"}
