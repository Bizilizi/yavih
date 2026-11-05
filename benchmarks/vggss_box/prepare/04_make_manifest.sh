#!/usr/bin/env bash
# vggss_box — step 4 (optional): re-derive the frozen manifest and check it is byte-identical to the committed one.
#
# Inputs (all read-only): $STAGING/vggss.json, $ROOT/metadata/vggsound.csv and the media listing, plus the sidecars
# shipped in prepare/ (frame_sizes.json, video_geometry.json, audio_shorter_than_clip_seconds.txt).
# make_manifest.py --check re-derives manifest.csv.gz (decompressed sha256
# fbf93c348d424ed48dc598ce80f512bd1aa8b529c60748624c61011d429898db) and the yaml's budget / curation facts and
# compares them with the committed ones (exit 1 on drift); --census adds a read of every wav header.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# Overrides: VGGSS_STAGING, VGGSOUND_ROOT. Unset, make_manifest.py resolves the folder's annotations_root and
# media_root through yavih's config (`yavih config get vggss_box.annotations_root` / `vggss_box.media_root`).
ARGS=()
[ -n "${VGGSS_STAGING:-}" ] && ARGS+=(--staging-dir "$VGGSS_STAGING")
[ -n "${VGGSOUND_ROOT:-}" ] && ARGS+=(--vggsound-root "$VGGSOUND_ROOT")

"$PYTHON" "$HERE/make_manifest.py" ${ARGS[@]+"${ARGS[@]}"} --check \
  | "$PYTHON" -c 'import json,sys; d=json.load(sys.stdin); print("check", d["check"], d["drift"], "sha256", d["manifest_sha256"], "rows", d["rows"], "draw", d["draw"]); sys.exit(d["check"] != "ok")'
