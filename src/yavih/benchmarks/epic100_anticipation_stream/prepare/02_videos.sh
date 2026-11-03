#!/usr/bin/env bash
# epic100_anticipation_stream — step 2: the train and validation videos with the official downloader (pinned commit),
# into <media_root>/EPIC-KITCHENS/<participant>/videos/<video_id>.MP4; the downloader checks its md5 list.
# A partial or torrent download made elsewhere can be linked into that layout instead (setup.md "Gotchas").
set -euo pipefail
ROOT="$(yavih config get epic100_anticipation_stream.media_root)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone -q https://github.com/epic-kitchens/epic-kitchens-download-scripts.git "$TMP/dl"
git -C "$TMP/dl" checkout -q 4f11fb2b579833f360c3c7bb917bf1e24a9787b5
python "$TMP/dl/epic_downloader.py" --videos --train --val --output-path "$ROOT"
