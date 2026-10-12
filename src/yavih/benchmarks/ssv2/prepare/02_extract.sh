#!/usr/bin/env bash
# ssv2 step 2 — build the release root from the downloaded parts (setup.md "Obtain the data"). Only needed after
# 01_portal_download.sh (03_stage_from_mirror.sh instead copies an already-extracted release).
#
#   ROOT/20bn-something-something-v2/<id>.webm   220 847 files, 19.4 GB, ONE flat directory (inode budget:
#                                                 220 849 inodes; a flat dir of 220k entries is slow to `ls`
#                                                 over NFS — never `ls | wc -l` in a loop, use one `find`)
#   ROOT/labels/{labels,train,validation,test}.json, test-answers.csv   (the zip carries the labels/ prefix)
#
# Extracts into a NEW temp dir next to ROOT and renames atomically, so a half-extracted tree is never
# mistaken for the release (count check before the swap). Never extract inside a directory you do not own.
#
# Usage: bash 02_extract.sh DL_DIR ROOT
set -euo pipefail
DL="${1:?usage: 02_extract.sh DL_DIR ROOT}"; ROOT="${2:?usage: 02_extract.sh DL_DIR ROOT}"
MEDIA="$ROOT/20bn-something-something-v2"
[ -e "$MEDIA" ] && { echo "[ssv2/02] $MEDIA exists; refusing to overwrite" >&2; exit 1; }
mkdir -p "$ROOT"
TMP=$(mktemp -d "$ROOT/.extract-XXXX")
echo "[ssv2/02] $(date -Is) cat parts | tar -xz → $TMP"
cat "$DL"/20bn-something-something-v2-?? | tar -xz -C "$TMP"
n=$(find "$TMP/20bn-something-something-v2" -maxdepth 1 -name '*.webm' | wc -l)
echo "[ssv2/02] extracted $n webm (expect 220847)"
[ "$n" -eq 220847 ] || { echo "[ssv2/02] COUNT MISMATCH: $n" >&2; exit 1; }
mv "$TMP/20bn-something-something-v2" "$MEDIA"; rmdir "$TMP"
if [ ! -d "$ROOT/labels" ]; then
  python - "$DL/20bn-something-something-download-package-labels.zip" "$ROOT" <<'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])  # members are labels/<file>
PY
fi
sha256sum "$ROOT"/labels/*
# expected (setup.md "Obtain the data"):
#   d98d8b657a5d418d7c05dbf630dedd34763025b3d764ceef19bbfd1dc654f54f  labels/labels.json
#   b24e46090960a49d732a0d3633fd939030b4eb717e16728871a3180b8704192b  labels/train.json
#   fbdd69ac72b40b9e3ed3aaf484e1575777152565caed693b0ced2d4959fd3958  labels/validation.json
#   60702636202db04cf60c76c76905d821b8fb281376fc1ee4e140087afe108560  labels/test.json
#   82fe3127545b172d5e1a92641e5192bccfbea689d4b793d79f091239ce5cd2ad  labels/test-answers.csv
echo "[ssv2/02] $(date -Is) done: $MEDIA"
