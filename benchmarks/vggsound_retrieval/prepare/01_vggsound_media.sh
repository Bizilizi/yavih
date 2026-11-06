#!/usr/bin/env bash
# vggsound_retrieval — step 1: the media (manual: an existing VGGSound copy).
#
# Every row is a clip of the VGGSound TEST split, read in place from the VGGSound tree (layout "dir", the same
# rel paths as vggsounder):
#   <ROOT>/video/<ytid>_<start6>.mp4   (native resolution, mixed; H.264 with its own AAC track)
#   <ROOT>/audio/<ytid>_<start6>.wav   (16 kHz mono)
#   <ROOT>/metadata/vggsound.csv       (the official listing: the rows and the leakage rule)
# The 15 440 frozen rows touch 30 880 files (budget.media_gb in benchmark.yaml).
#
# Use an existing VGGSound copy; otherwise see ../../vggsounder/prepare/02_download_vggsound.sh (YouTube
# re-download; clips disappear over time). Never extract the bulk VGGSound tars in place next to the tree.
#
# This script only CHECKS the listing's sha256 and that the test clips are present (one scandir per directory).
set -euo pipefail

ROOT="${VGGSOUND_ROOT:-$(yavih config get vggsound_retrieval.media_root)}"

(cd "$ROOT/metadata" && sha256sum -c --strict <<'SHA'
ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8  vggsound.csv
SHA
)
python - "$ROOT" <<'PY'
import csv, os, sys
root = sys.argv[1]
ids = [f"{r[0]}_{int(float(r[1])):06d}" for r in csv.reader(open(f"{root}/metadata/vggsound.csv")) if len(r) >= 4 and r[3].strip() == "test"]
v = {e.name for e in os.scandir(f"{root}/video")}
a = {e.name for e in os.scandir(f"{root}/audio")}
mv = [i for i in ids if f"{i}.mp4" not in v]
ma = [i for i in ids if f"{i}.wav" not in a]
print(f"test clips listed {len(ids)}  mp4 present {len(ids)-len(mv)}  wav present {len(ids)-len(ma)}  (expect 15446 / 15446 / 15446)")
if mv or ma:
    print("missing mp4 first10", mv[:10]); print("missing wav first10", ma[:10])
PY
