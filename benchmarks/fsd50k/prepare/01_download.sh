#!/usr/bin/env bash
# FSD50K step 1/3 — download the 12 Zenodo files (record 4060432) with md5 verification.
# Plain HTTP, no account, no agreement; resumable. ~45 min for 24.7 GB on a fast link.
#
#   DL=/path/to/fsd50k_download bash 01_download.sh          # archives land in $DL (24.7 GB, 12 files)
#   (default DL: $(yavih config get data.root)/fsd50k_download)
#
# Disk: 24.7 GB for the archives (keep them: they are the reproducible source; the extracted tree in
# step 2 needs another 33 GB / 51 210 inodes). sha256 of every archive is pinned in benchmark.yaml
# source.sha256; Zenodo publishes md5s, which is what is verified here.
set -euo pipefail
DL="${DL:-$(yavih config get data.root)/fsd50k_download}"
BASE='https://zenodo.org/records/4060432/files'
mkdir -p "$DL"
cd "$DL"

get() { # get <filename> <md5>
  local f=$1 md5=$2
  echo "=== $(date -Is) downloading $f ==="
  wget --continue --tries=0 --retry-connrefused --waitretry=30 -q --show-progress --progress=dot:giga \
    -O "$DL/$f" "$BASE/$f?download=1"
  local got; got=$(md5sum "$DL/$f" | cut -d' ' -f1)
  if [ "$got" != "$md5" ]; then echo "FAIL md5 $f got=$got want=$md5" >&2; return 1; fi
  echo "=== $(date -Is) OK $f md5=$got size=$(stat -c %s "$DL/$f") ==="
}

# Phase 1: ground truth + metadata + doc + eval (2 parts) — everything needed to validate first.
get FSD50K.ground_truth.zip ca27382c195e37d2269c4c866dd73485   #     334 701 B
get FSD50K.metadata.zip     b9ea0c829a411c1d42adb9da539ed237   #   6 700 838 B
get FSD50K.doc.zip          3516162b82dc2945d3e7feba0904e800   #       6 984 B
get FSD50K.eval_audio.z01   3090670eaeecc013ca1ff84fe4442aeb   # 3 221 225 472 B
get FSD50K.eval_audio.zip   6fa47636c3a3ad5c7dfeba99f2637982   # 3 037 675 767 B
# Phase 2: dev audio (6 parts).
get FSD50K.dev_audio.z01 faa7cf4cc076fc34a44a479a5ed862a3      # 3 221 225 472 B
get FSD50K.dev_audio.z02 8f9b66153e68571164fb1315d00bc7bc      # 3 221 225 472 B
get FSD50K.dev_audio.z03 1196ef47d267a993d30fa98af54b7159      # 3 221 225 472 B
get FSD50K.dev_audio.z04 d088ac4e11ba53daf9f7574c11cccac9      # 3 221 225 472 B
get FSD50K.dev_audio.z05 81356521aa159accd3c35de22da28c7f      # 3 221 225 472 B
get FSD50K.dev_audio.zip c480d119b8f7a7e32fdb58f3ea4d6c5a      # 2 306 663 327 B
echo "=== $(date -Is) DOWNLOAD_DONE $(du -sh "$DL" | cut -f1) in $DL ==="
