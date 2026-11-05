#!/usr/bin/env bash
# in1k step 3 (optional: a subset root) — stage ONLY the 100 000 files the frozen manifest names (50 000 drawn
# train + 50 000 val; 12 432 228 705 B = 12.4 GB, 100 000 inodes) into a directory with the same relative
# layout, instead of extracting the 1.33 M-file root. The manifest freezes the draw, so the subset is exactly its
# rel_media column. The subset is protocol-bound: a protocol_version change that alters train_draw makes it wrong.
#
# Usage: bash 03_stage_probe_subset.sh SRC_ROOT DEST_ROOT            (local rsync; for a remote DEST use
#        DEST_ROOT=host:/path and your site's transport, e.g. rsync over ssh)
set -euo pipefail
SRC="${1:?SRC_ROOT (extracted torchvision root)}"; DEST="${2:?DEST_ROOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LIST="$(mktemp)"
# rel_media column of the frozen manifest (100 000 lines), header dropped.
python - "$HERE/../manifest.csv.gz" > "$LIST" <<'PY'
import csv, gzip, io, sys
rows = csv.DictReader(io.TextIOWrapper(gzip.open(sys.argv[1], "rb"), encoding="utf-8", newline=""))
for r in rows:
    print(r["rel_media"])
PY
n=$(wc -l < "$LIST"); echo "[in1k/03] $n files listed (expect 100000)"; [ "$n" -eq 100000 ]
# --partial-dir: an in-flight file never sits under its final name (a half-written JPEG would pass a presence check).
rsync -a --files-from="$LIST" --partial-dir=.rsync-partial --info=stats2 "$SRC/" "$DEST/"
rm -f "$LIST"
# Verify: byte total, then yavih's own check.
echo "[in1k/03] bytes staged: $(du -sb "$DEST" | cut -f1)   (expect 12432228705 + directory overhead)"
echo "[in1k/03] now: YAVIH_DATA__IN1K__MEDIA_ROOT=$DEST yavih bench verify in1k --census   (expects train 50000/50000, val 50000/50000)"
