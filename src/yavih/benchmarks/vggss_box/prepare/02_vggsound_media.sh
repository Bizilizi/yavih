#!/usr/bin/env bash
# vggss_box — step 2: the media (manual: an existing VGGSound copy).
#
# VGG-SS has NO media download of its own: every one of the 5 158 entries is a clip of the VGGSound TEST
# split, read in place from the VGGSound tree (layout "dir", same rel paths as vggsounder):
#   <ROOT>/video/<ytid>_<start6>.mp4   (native resolution: 2 231 of the 4 614 frozen rows are 1280×720, the rest
#                                       mostly 480×360, 640×360, 320×240, and 182 portrait clips)
#   <ROOT>/audio/<ytid>_<start6>.wav   (16 kHz mono)
#   <ROOT>/metadata/vggsound.csv       (official listing; make_manifest.py uses it for the leakage rule)
# The 5 158 clips are 10.54 GB (mp4 + wav + json); the 4 614 frozen rows touch 9.35 GB / 9 228 files.
#
# Use an existing VGGSound copy; otherwise see ../../vggsounder/prepare/02_download_vggsound.sh (YouTube
# re-download; clips disappear over time, and a re-download's frame sizes may differ — the full-frame boxes are
# fractions and survive a uniform rescale, but not a cut or re-framed clip; see step 3). Never extract the bulk
# VGGSound tars in place next to the tree.
#
# This script only CHECKS that the 5 158 VGG-SS clips are present (one scandir per directory).
set -euo pipefail

ROOT="${VGGSOUND_ROOT:-$(yavih config get vggss_box.media_root)}"
STAGING="${VGGSS_STAGING:-$(yavih config get vggss_box.annotations_root)}"

(cd "$ROOT/metadata" && sha256sum -c --strict <<'SHA'
ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8  vggsound.csv
SHA
)
python - "$ROOT" "$STAGING/vggss.json" <<'PY'
import json, os, sys
root, js = sys.argv[1], sys.argv[2]
ids = [e["file"] for e in json.load(open(js))]
v = {e.name for e in os.scandir(f"{root}/video")}
a = {e.name for e in os.scandir(f"{root}/audio")}
mv = [i for i in ids if f"{i}.mp4" not in v]
ma = [i for i in ids if f"{i}.wav" not in a]
print(f"vggss entries {len(ids)}  mp4 present {len(ids)-len(mv)}  wav present {len(ids)-len(ma)}  (expect 5158 / 5158 / 5158)")
if mv or ma:
    print("missing mp4 first10", mv[:10]); print("missing wav first10", ma[:10])
PY
