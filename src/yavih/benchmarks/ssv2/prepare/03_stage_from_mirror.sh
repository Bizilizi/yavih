#!/usr/bin/env bash
# ssv2 step 3 — stage a copy you own from an already-extracted release (setup.md "Obtain the data"):
#   SRC  = an extracted copy of the official release (20bn-something-something-v2/ + labels/); ask its owner
#          before relying on a copy you do not own, and do not symlink it: stage your own.
#   DEST = $(yavih config get ssv2.media_root)       (the yaml default is ${data.root}/ssv2)
#
# Copies into DEST/.20bn-tmp, checks the count, then renames atomically (no half-copied tree is ever at the
# final path). Labels are copied flat into DEST/labels/ (a copy that nests them as labels/labels/ is handled).
# A full copy (220 847 files / 19 GB) takes ~2 h over NFS with 4 CPUs: run it as a CPU batch job.
#
# Usage: bash 03_stage_from_mirror.sh SRC DEST
#   or only the 44 777 files the frozen manifest names (4.0 GB):
#   bash 03_stage_from_mirror.sh SRC DEST --frozen-only
set -euo pipefail
SRC="${1:?usage: 03_stage_from_mirror.sh SRC DEST [--frozen-only]}"; DEST="${2:?usage}"; MODE="${3:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LINK="$DEST/20bn-something-something-v2"
[ -e "$LINK" ] && { echo "[ssv2/03] $LINK exists; nothing to do (remove it first to re-stage)" >&2; exit 1; }
mkdir -p "$DEST/labels"
LBL="$SRC/labels"; [ -f "$LBL/labels/labels.json" ] && LBL="$SRC/labels/labels"
cp -p "$LBL"/{labels.json,train.json,validation.json,test.json,test-answers.csv} "$DEST/labels/"
TMP="$DEST/.20bn-tmp"; mkdir -p "$TMP"
echo "=== $(date -Is) rsync media -> $TMP on $(hostname)"
if [ "$MODE" = "--frozen-only" ]; then
  # exactly the 44 777 webm the frozen manifest names (rel_media column), 4 020 523 417 B
  python -c "import csv,gzip,io,sys; [print(r['rel_media'].split('/',1)[1]) for r in csv.DictReader(io.StringIO(gzip.open(sys.argv[1]).read().decode()))]" \
    "$HERE/../manifest.csv.gz" > "$DEST/.frozen_files.txt"
  rsync -a --files-from="$DEST/.frozen_files.txt" "$SRC/20bn-something-something-v2/" "$TMP/"
  want=44777
else
  rsync -a "$SRC/20bn-something-something-v2/" "$TMP/"
  want=220847
fi
n=$(find "$TMP" -maxdepth 1 -name '*.webm' | wc -l)
echo "=== $(date -Is) rsync done: $n files (expect $want)"
[ "$n" -eq "$want" ] || { echo "COUNT MISMATCH: $n" >&2; exit 1; }
mv "$TMP" "$LINK"
for id in 41775 850 194233 103716 218927; do stat -c '%n %s' "$LINK/$id.webm" 2>/dev/null || true; done
# expected sizes (bytes): 41775 94042, 850 86990, 194233 101349, 103716 49404, 218927 75923
echo "=== $(date -Is) DONE: $LINK"
