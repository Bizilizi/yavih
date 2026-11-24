#!/usr/bin/env bash
# Step 02: the LLP clips. The authors distribute no videos (their scripts/download_dataset.py fetches each clip from
# YouTube with youtube-dl and cuts [start, start + 10 s] with ffmpeg, libx264 + aac); the complete set of 11 849
# clips cut that way is the public, ungated Hugging Face dataset CserDu123/LLP-dataset. This step downloads its
# llp_video.zip (13 438 121 116 bytes, sha256-checked) at the pinned revision into <media_root>/download/ and
# extracts it into <media_root>/llp_videos/<ytid>.mp4 (13.5 GB); the zip is deleted afterwards unless KEEP_ZIP=1.
# One connection; resumable (curl -C -). Needs curl, sha256sum and unzip (or python3's zipfile module).
#   yavih bench prepare llp_avvp 02
set -euo pipefail
ROOT="${ROOT:-$(yavih config get llp_avvp.media_root)}"
REPO=CserDu123/LLP-dataset
REVISION=31727951b0ea2a573501cda490005c68de6822d0
ZIP_SHA256=36093363b2d6cc540d1da665fd78af73d339cc6e3eeeb0993778b967e75211d3
ZIP_BYTES=13438121116
N_CLIPS=11849
mkdir -p "$ROOT/download"
ZIP="$ROOT/download/llp_video.zip"
have=$(find "$ROOT/llp_videos" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l || true)
if [ "$have" -ge "$N_CLIPS" ]; then
  echo "[02] $have clips already in $ROOT/llp_videos; nothing to do"
  exit 0
fi
if [ ! -f "$ZIP" ] || [ "$(stat -c %s "$ZIP")" -ne "$ZIP_BYTES" ]; then
  echo "[02] downloading $REPO@${REVISION:0:8} llp_video.zip (13.4 GB)"
  curl -fL --retry 5 -C - -o "$ZIP" "https://huggingface.co/datasets/$REPO/resolve/$REVISION/llp_video.zip"
fi
echo "$ZIP_SHA256  $ZIP" | sha256sum -c --quiet || { echo "[02] llp_video.zip: sha256 mismatch (expected $ZIP_SHA256)" >&2; exit 1; }
echo "[02] extracting into $ROOT/llp_videos"
if command -v unzip > /dev/null; then
  unzip -q -n "$ZIP" -d "$ROOT"
else
  python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$ZIP" "$ROOT"
fi
have=$(find "$ROOT/llp_videos" -maxdepth 1 -name '*.mp4' | wc -l)
[ "$have" -eq "$N_CLIPS" ] || { echo "[02] expected $N_CLIPS mp4 in $ROOT/llp_videos, found $have" >&2; exit 1; }
if [ "${KEEP_ZIP:-0}" != "1" ]; then rm -f "$ZIP"; fi
echo "[02] ok: $have clips in $ROOT/llp_videos"
