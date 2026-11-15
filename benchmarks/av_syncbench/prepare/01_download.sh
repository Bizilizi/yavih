#!/usr/bin/env bash
# Download the seven parts of the AV-SyncBench release (ModelScope coming245/AVSyncBench at the pinned revision,
# anonymous HTTPS, no account) into <media_root>/raw/, resumably, and check each part's size and sha256.
# 68 191 839 983 bytes in all. Usage: yavih bench prepare av_syncbench 01   (ROOT=<dir> overrides the media_root)
set -euo pipefail
ROOT="${ROOT:-$(yavih config get av_syncbench.media_root)}"
REVISION="86c06579529a6e7b2cafb0dc386a50152a37fb98"
BASE="https://modelscope.cn/api/v1/datasets/coming245/AVSyncBench/repo?Revision=${REVISION}&FilePath="
RAW="$ROOT/raw"
mkdir -p "$RAW"
# part  bytes  sha256 (the ModelScope tree API at the revision; also pinned by the authors' scripts/download_dataset.sh)
PARTS="
aa 10737418240 4e88e575b3c5e202cd2addbc1789bfed6a4cf036d85f0f70d5ca56be05b96d7d
ab 10737418240 00e4b50c2b4b32d539d8c75a8e6f9308afa83983c3c0b221445935740354fdbd
ac 10737418240 078c1d9a70beb6f9fa6d6b0c6afb21e3527d932c04f76cec2b726a388baf4d53
ad 10737418240 7ef96fb3dce0b6f241e4d7aa19d873e518176ddf33e3b7a16bc5feb5a32eac01
ae 10737418240 4dfbbd97087b025a0b16f46def669a953de566bbae275489dd5ef90330b4d712
af 10737418240 35ec67bc890589289f1644ecbba1a30337119db65fec99072e8bcc2a06196630
ag 3767330543 3d71987d812814950fb940862fff1648d127f4cc15bbfe68c34e7ad4c6bcd2f1
"
echo "$PARTS" | while read -r part bytes sha; do
  [ -z "$part" ] && continue
  f="$RAW/AV_SyncBench.tar.gz.part_$part"
  if [ -f "$f.ok" ] && [ "$(stat -c %s "$f")" = "$bytes" ]; then
    echo "part_$part: verified earlier"; continue
  fi
  if [ ! -f "$f" ] || [ "$(stat -c %s "$f")" -lt "$bytes" ]; then
    echo "part_$part: downloading ($bytes bytes)"
    curl -fL --retry 10 --retry-delay 30 -C - -o "$f" "${BASE}AV_SyncBench.tar.gz.part_$part"
  fi
  got="$(stat -c %s "$f")"
  [ "$got" = "$bytes" ] || { echo "part_$part: $got bytes, expected $bytes (delete it and rerun)" >&2; exit 4; }
  echo "$sha  $f" | sha256sum -c --quiet - || { echo "part_$part: sha256 mismatch (delete it and rerun)" >&2; exit 4; }
  touch "$f.ok"
  echo "part_$part: ok"
done
echo "all seven parts in $RAW; next: yavih bench prepare av_syncbench 02"
