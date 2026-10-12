#!/usr/bin/env bash
# AVE step 1/5 — download the media tar + the upstream annotation/split files (all public, ungated).
# URLs, sizes and sha256s as in benchmark.yaml source.sha256. HF needs no token for this dataset.
#
#   ROOT=/path/to/datasets/ave yavih bench prepare ave 01     # default ROOT: `yavih config get ave.media_root`
#
# Disk: 13.36 GB, 5 files (+1 index.json in step 3). The tar is NEVER extracted (4 097 mp4 + 4 776 other
# members stay inside it; every read goes through index.json byte offsets).
set -euo pipefail
ROOT="${ROOT:-$(yavih config get ave.media_root)}"
HF='https://huggingface.co/datasets/Helios1208/AVE_Dataset/resolve/main'
GH='https://raw.githubusercontent.com/YapengTian/AVE-ECCV18/master/data'
mkdir -p "$ROOT/upstream"

get() { # get <url> <dest> <bytes> <sha256>
  local url=$1 dest=$2 bytes=$3 sha=$4
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$bytes" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
    echo "=== OK (already present) $dest"; return 0
  fi
  echo "=== $(date -Is) downloading $url"
  wget --continue --tries=0 --retry-connrefused --waitretry=30 -q --show-progress --progress=dot:giga -O "$dest" "$url"
  local got_b got_s; got_b=$(stat -c %s "$dest"); got_s=$(sha256sum "$dest" | cut -d' ' -f1)
  if [ "$got_b" != "$bytes" ] || [ "$got_s" != "$sha" ]; then
    echo "FAIL $dest bytes=$got_b (want $bytes) sha256=$got_s (want $sha)" >&2; return 1
  fi
  echo "=== $(date -Is) OK $dest"
}

# The upstream filename typo `AVE_Datset.tar` is real — do not "fix" it.
get "$HF/AVE_Datset.tar"  "$ROOT/AVE_Datset.tar"           13353553920 479905dc043ebae0e445c6ed1a109e5295a8f72b65ae959511db0a2bfc3a02c3
# The HF-ROOT Annotations.txt (4 143 rows / 28 categories). NOT AVE_Dataset/Annotations.txt inside the tar:
# that one is a stale 2 757-row / 19-category subset the official split files cannot index.
get "$HF/Annotations.txt" "$ROOT/upstream/Annotations.txt" 144330      b0aabcea3281abb4c884e19ad4ff90314b199bea4742efe8556585a18f1396e7
# Official splits (AVE-ECCV18 data/): int64 `order` datasets indexing Annotations.txt rows, 3 339 / 402 / 402.
get "$GH/train_order.h5"  "$ROOT/upstream/train_order.h5"  28856       a216fcdd02782ba9d10c9ea79116120eb3be467740e58290e1b340d42a590456
get "$GH/val_order.h5"    "$ROOT/upstream/val_order.h5"    5360        5692aec0efb30e4efa49a867a902c1993b4c88a7d3ec19879349e540e7a5d9ca
get "$GH/test_order.h5"   "$ROOT/upstream/test_order.h5"   5360        f9bb79defc1f784ca5dda9cdd3ee52af6d24ccb399a1cfab6bf0d373f5681076
echo "=== $(date -Is) DOWNLOAD_DONE $(du -sh "$ROOT" | cut -f1) in $ROOT"
