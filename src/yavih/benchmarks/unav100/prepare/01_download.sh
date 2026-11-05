#!/usr/bin/env bash
# UnAV-100 step 1/4 — download the official annotations and the raw-video tar.gz parts from the authors' HF dataset
# ttgeng233/UnAV-100 at the pinned revision (public, ungated: no token, no licence click). Resumable; every file is
# checked against its size and sha256 (benchmark.yaml source.sha256).
#
#   yavih bench prepare unav100 01                     # ROOT defaults to `yavih config get unav100.media_root`
#   ROOT=/path/to/datasets/unav100 yavih bench prepare unav100 01
#   PARTS="00 01" yavih bench prepare unav100 01       # a subset of the 13 parts (testing; step 02 needs all)
#
# Disk: 65.78 GB of parts + 0.56 MB of parquets under $ROOT/upstream/. Time: bound by the HF CDN (~1 h at 20 MB/s).
set -euo pipefail
ROOT="${ROOT:-$(yavih config get unav100.media_root)}"
REV=daeb88bc14eeff48f8a4ac2836b67f6c350f847b
HF="https://huggingface.co/datasets/ttgeng233/UnAV-100/resolve/$REV"
mkdir -p "$ROOT/upstream/data" "$ROOT/upstream/parts"

get() { # get <url> <dest> <bytes> <sha256>
  local url=$1 dest=$2 bytes=$3 sha=$4
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$bytes" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    echo "=== OK (already present) $dest"; return 0
  fi
  echo "=== $(date -Is) downloading $url"
  curl -fL --retry 20 --retry-delay 30 --retry-all-errors -C - -o "$dest" "$url"
  local got_b got_s; got_b=$(stat -c %s "$dest"); got_s=$(sha256sum "$dest" | cut -d' ' -f1)
  if [ "$got_b" != "$bytes" ] || [ "$got_s" != "$sha" ]; then
    echo "FAIL $dest bytes=$got_b (want $bytes) sha256=$got_s (want $sha); delete it and rerun" >&2; return 1
  fi
  echo "=== $(date -Is) OK $dest"
}

# annotations: one row per event (video_id, duration, timestamp [start, end], event_label)
get "$HF/data/train.parquet"      "$ROOT/upstream/data/train.parquet"      323290 d8b1302fc12dc22a62ca23b592392cb000d00d2a487a70aa40876dde52742d55
get "$HF/data/validation.parquet" "$ROOT/upstream/data/validation.parquet" 116962 551e1781c8a8fd77cb7a1daf0ad9b71346c615864de7bba6fb5dba3565dadc6b
get "$HF/data/test.parquet"       "$ROOT/upstream/data/test.parquet"       115568 36a7d528772c1be736c105bfc65955255c822e99b7ceded64ef6e1b9c969df45

# raw videos: one gzip tar split into 13 byte-ranges (concatenate in order, then gunzip: step 02)
declare -A SHA=(
  [00]=aa8dc6798fc8b99d69533b4ebd068a0f659532d13ff82a21a527db5fbdd37f65 [01]=d549709123379d9c9f3b63d2aabd31a09f72c0004c50178e0a8b509896122af0
  [02]=2f1fae736a5d567519bb6edfa2c79f97571923a6dbb679fb0350adf8df59aeb0 [03]=021ef07d239e735af94a76e46421350788850b2a3b5bfb44b8124b1205cfd6ba
  [04]=89c2097a6bab34441302d2f2db65d4e504468a9f7ade713bef4698c6d6223546 [05]=f56ad30d4e4bee762f8c9577689a909e93a267785f7e4452fafe5d987e106a1a
  [06]=25cea1db67f92036241fcb84a4da430be3b79d9a49a620252cab20f18d141d37 [07]=feb5c3f1268c8125d3d9012ef0c37f4e8dc6ddff0493922e3d6dbbc71d7c7677
  [08]=064ca3707ed4b234ae935413d4f95dc7bc9e55d716df8f39a00576f40dd142d1 [09]=6b5f72f5892e36aa808c357a365cabe826bafa28438cd308e7596977265f0a9a
  [10]=5ddca8ddc839cdf71a6f0d0c8e67bc14400bf8aaf1e9a8f646f7bfba9eefcc69 [11]=815520a0b00649d77f22e6a4abc18b1e971691d6da355ba7bd951e6cbc45c600
  [12]=d38371218b5f1dcf2babd039b882608ecf4cd24a6863b4f6c6a39dc03eb788d3
)
for p in ${PARTS:-00 01 02 03 04 05 06 07 08 09 10 11 12}; do
  bytes=5368709120; [ "$p" = 12 ] && bytes=1351094873
  get "$HF/unav_raw_videos.tar.gz.part_$p" "$ROOT/upstream/parts/unav_raw_videos.tar.gz.part_$p" "$bytes" "${SHA[$p]}"
done
echo "=== $(date -Is) DOWNLOAD_DONE $(du -sh "$ROOT/upstream" | cut -f1) in $ROOT/upstream"
