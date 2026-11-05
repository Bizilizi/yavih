#!/usr/bin/env bash
# AVE step 2/5 — verify sizes and sha256 of everything step 1 fetched (and of the VGGSound metadata the
# leakage audit reads). Hashing the 13.35 GB tar takes minutes on NFS; SKIP_TAR=1 skips it.
#
#   ROOT=/path/to/datasets/ave VGGSOUND_CSV=/path/to/vggsound/metadata/vggsound.csv yavih bench prepare ave 02
# defaults: `yavih config get ave.media_root`, `$(yavih config get datasets.vggsound)/metadata/vggsound.csv`
set -euo pipefail
ROOT="${ROOT:-$(yavih config get ave.media_root)}"
VGGSOUND_CSV="${VGGSOUND_CSV:-$(yavih config get datasets.vggsound)/metadata/vggsound.csv}"
fail=0
check() { # check <path> <bytes> <sha256>
  local p=$1 b=$2 s=$3
  if [ ! -f "$p" ]; then echo "MISSING $p"; fail=1; return; fi
  local gb; gb=$(stat -c %s "$p")
  if [ "$gb" != "$b" ]; then echo "BAD SIZE $p $gb (want $b)"; fail=1; return; fi
  local gs; gs=$(sha256sum "$p" | cut -d' ' -f1)
  if [ "$gs" != "$s" ]; then echo "BAD SHA256 $p $gs (want $s)"; fail=1; return; fi
  echo "ok $p"
}
if [ "${SKIP_TAR:-0}" = 1 ]; then
  [ "$(stat -c %s "$ROOT/AVE_Datset.tar")" = 13353553920 ] && echo "ok (size only) $ROOT/AVE_Datset.tar" || { echo "BAD SIZE tar"; fail=1; }
else
  check "$ROOT/AVE_Datset.tar" 13353553920 479905dc043ebae0e445c6ed1a109e5295a8f72b65ae959511db0a2bfc3a02c3
fi
check "$ROOT/upstream/Annotations.txt" 144330 b0aabcea3281abb4c884e19ad4ff90314b199bea4742efe8556585a18f1396e7
check "$ROOT/upstream/train_order.h5"  28856  a216fcdd02782ba9d10c9ea79116120eb3be467740e58290e1b340d42a590456
check "$ROOT/upstream/val_order.h5"    5360   5692aec0efb30e4efa49a867a902c1993b4c88a7d3ec19879349e540e7a5d9ca
check "$ROOT/upstream/test_order.h5"   5360   f9bb79defc1f784ca5dda9cdd3ee52af6d24ccb399a1cfab6bf0d373f5681076
# VGGSound official metadata (the leakage audit's train-id list; staged with vggsounder, not downloaded here).
if [ -f "$VGGSOUND_CSV" ]; then
  gs=$(sha256sum "$VGGSOUND_CSV" | cut -d' ' -f1)
  [ "$gs" = ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8 ] && echo "ok $VGGSOUND_CSV" || { echo "BAD SHA256 $VGGSOUND_CSV $gs"; fail=1; }
else
  echo "note: $VGGSOUND_CSV absent — make_manifest.py needs it"
fi
# The stale in-tar copy must never be used: its row count is the tell.
head -1 "$ROOT/upstream/Annotations.txt" | grep -q '^Category&VideoID&Quality&StartTime&EndTime' || { echo "BAD header in Annotations.txt"; fail=1; }
n=$(($(wc -l < "$ROOT/upstream/Annotations.txt")))
echo "Annotations.txt lines: $n (4 144 = header + 4 143 rows for the HF-root copy; ~2 758 means the stale in-tar copy)"
exit $fail
