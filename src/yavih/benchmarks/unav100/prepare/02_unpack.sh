#!/usr/bin/env bash
# UnAV-100 step 2/4 — join the 13 parts and decompress them into ONE uncompressed tar, $ROOT/unav_raw_videos.tar
# (10 790 mp4 members). The tar is never extracted: step 03 indexes it in place and yavih reads members by offset,
# so the staged media cost 2 inodes. mp4 does not compress, so the tar is about as large as the parts (~66 GB).
#
#   yavih bench prepare unav100 02                     # ROOT defaults to `yavih config get unav100.media_root`
#   KEEP_PARTS=0 yavih bench prepare unav100 02        # delete the parts after a successful unpack (peak disk ~132 GB)
#
# Time: one sequential pass, ~10-20 min on local disk. Writes to <tar>.partial first, renames when complete.
set -euo pipefail
ROOT="${ROOT:-$(yavih config get unav100.media_root)}"
PARTS_DIR="$ROOT/upstream/parts"
OUT="$ROOT/unav_raw_videos.tar"
if [ -f "$OUT" ]; then echo "=== OK (already present) $OUT ($(stat -c %s "$OUT") B)"; exit 0; fi
parts=()
for p in 00 01 02 03 04 05 06 07 08 09 10 11 12; do
  f="$PARTS_DIR/unav_raw_videos.tar.gz.part_$p"
  [ -f "$f" ] || { echo "missing $f: run step 01 first" >&2; exit 3; }
  parts+=("$f")
done
echo "=== $(date -Is) unpacking ${#parts[@]} parts -> $OUT"
cat "${parts[@]}" | gzip -dc > "$OUT.partial"
tar -tf "$OUT.partial" > /dev/null   # a truncated or corrupt stream fails here, before the rename
mv "$OUT.partial" "$OUT"
echo "=== $(date -Is) OK $OUT ($(stat -c %s "$OUT") B)"
if [ "${KEEP_PARTS:-1}" = 0 ]; then rm -f "${parts[@]}"; echo "=== removed the parts"; fi
