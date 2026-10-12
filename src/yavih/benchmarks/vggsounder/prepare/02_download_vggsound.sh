#!/usr/bin/env bash
# vggsounder — step 2: the VGGSound listing and media (YouTube; clips decay).
#
#   - the official listing metadata/vggsound.csv (ytid,start_seconds,label,split; no header), 199 176 rows,
#     sha256 ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8, from
#     https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv (fetched below when missing). The frozen
#     train draw is a function of its row order, hence the pin;
#   - clip id = <ytid>_<start zero-padded to 6 digits>, e.g. ---g-f_I2yQ_000001; each clip is the 10 s
#     window [start, start+10) of the YouTube video;
#   - audio is 16 kHz mono PCM-16 wav; video is H.264 mp4 (with its own 44.1 kHz AAC track) at 6–60 fps
#     (mostly 30 / 29.97 / 25), native resolution, MIXED: about half 720x1280, the rest 360x{480,640},
#     240x320 and ~5 % portrait.
#
# MANUAL STEP: a fresh download misses decayed clips, so prefer an existing copy of the corpus if you have
# access to one. The yt-dlp loop below is illustrative: a bare yt-dlp loop is rate-limited within minutes;
# use a downloader with a proxy pool for a real run.
# The VGGSounder package can also fetch TEST-split media from HF `11hu83/vggsound`
# (VGGSounder(download_samples=True)); not used by this benchmark (it does not cover the train split).
set -euo pipefail

ROOT="${VGGSOUND_ROOT:-$(yavih config get vggsounder.media_root)}"
CSV="$ROOT/metadata/vggsound.csv"
URL="https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv"
mkdir -p "$ROOT/video" "$ROOT/audio" "$ROOT/metadata"

if [ ! -s "$CSV" ]; then
  echo "fetching $URL"
  curl -fsSL -o "$CSV.tmp" "$URL" && mv "$CSV.tmp" "$CSV"
fi
(cd "$ROOT/metadata" && sha256sum -c --strict <<'EOF'
ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8  vggsound.csv
EOF
) || { echo "metadata/vggsound.csv differs from the pinned official listing (the train draw depends on its row order)" >&2; exit 3; }

# ---- download loop (ILLUSTRATIVE; use a proxy-pooled downloader for a real run) -------------------------
# One clip = 10 s from `start`; video mp4 (H.264), audio 16 kHz mono wav. Idempotent: existing files are
# skipped. Expect yt-dlp>=2026.7.4. The format filter caps video at 360p to bound disk and decode cost; the
# frozen media are native resolution (about half 720x1280): drop `[height<=360]` to match. Either way a
# re-download is new media (YouTube re-encodes), never byte-equal to an earlier copy.
if [ "${VGGSOUND_DOWNLOAD:-0}" != "1" ]; then
  echo "set VGGSOUND_DOWNLOAD=1 to run the illustrative yt-dlp loop (see the MANUAL STEP note above)"
  exit 0
fi
while IFS=, read -r ytid start label split; do
  id="${ytid}_$(printf '%06d' "$start")"
  [ -s "$ROOT/video/$id.mp4" ] && [ -s "$ROOT/audio/$id.wav" ] && continue
  yt-dlp -q --no-warnings -f "bv*[height<=360]+ba/b[height<=360]" \
    --download-sections "*${start}-$((start + 10))" --force-keyframes-at-cuts \
    --merge-output-format mp4 -o "$ROOT/video/$id.mp4" "https://www.youtube.com/watch?v=$ytid" || { echo "FAIL $id" >> "$ROOT/metadata/download_failures.txt"; continue; }
  ffmpeg -nostdin -loglevel error -y -i "$ROOT/video/$id.mp4" -vn -ac 1 -ar 16000 -sample_fmt s16 "$ROOT/audio/$id.wav"
done < "$CSV"
