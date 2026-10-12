#!/usr/bin/env bash
# vggss_box — step 1: the VGG-SS annotations (public, one JSON, no auth, no rate limit).
#
# Upstream facts:
#   5 158 entries, 221 classes, 1–7 boxes per entry (normalised x1,y1,x2,y2 on the frame at t = 5.0 s),
#   12 boxes with a corrupt coordinate -2.8823e16 (sanitised by the official max(x, 0) rule downstream).
#
# Expected:
#   vggss.json   620 306 bytes   sha256 ae300b4a259267e6fe839970d495862edd70bfe3da63d925bcdb34054ed3b15e
#
# If the sha differs, the upstream file moved: do NOT re-freeze silently — make_manifest.py refuses it, and a
# different annotation file is a new protocol_version.
set -euo pipefail

STAGING="${VGGSS_STAGING:-$(yavih config get vggss_box.annotations_root)}"
URL="https://raw.githubusercontent.com/hche11/Localizing-Visual-Sounds-the-Hard-Way/master/metadata/vggss.json"

mkdir -p "$STAGING"
if [ ! -f "$STAGING/vggss.json" ]; then
  curl -fL --retry 3 -o "$STAGING/vggss.json.tmp" "$URL"
  mv "$STAGING/vggss.json.tmp" "$STAGING/vggss.json"
fi
(cd "$STAGING" && sha256sum -c --strict <<'SHA'
ae300b4a259267e6fe839970d495862edd70bfe3da63d925bcdb34054ed3b15e  vggss.json
SHA
)
test "$(stat -c %s "$STAGING/vggss.json")" = 620306
python - "$STAGING/vggss.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"entries {len(d)} (expect 5158)  classes {len({e['class'] for e in d})} (expect 221)")
PY
echo "ok: $STAGING/vggss.json"
