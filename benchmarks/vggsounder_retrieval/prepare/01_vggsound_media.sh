#!/usr/bin/env bash
# vggsounder_retrieval — step 1: the media (manual: an existing VGGSound copy, the tree the vggsounder folder reads).
#
# Every row is a VGGSound TEST clip, read in place (layout "dir", the same rel paths as vggsounder):
#   <ROOT>/video/<ytid>_<start6>.mp4   (native resolution, mixed; H.264 with its own AAC track)
#   <ROOT>/audio/<ytid>_<start6>.wav   (16 kHz mono)
#   <ROOT>/metadata/vggsound.csv       (the official listing: the leakage audit of make_manifest.py)
# The 5 294 frozen rows touch 10 588 files, 10.72 GB (budget.media_gb in benchmark.yaml).
#
# Without a copy, the vggsounder folder's prepare/02_download_vggsound.sh re-downloads from YouTube (clips disappear
# over time). This script only CHECKS the listing's sha256 and that the frozen rows' files exist (one scandir per
# directory); it writes nothing.
set -euo pipefail

ROOT="${VGGSOUND_ROOT:-$(yavih config get vggsounder_retrieval.media_root)}"
FOLDER="${FOLDER:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

(cd "$ROOT/metadata" && sha256sum -c --strict <<'SHA'
ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8  vggsound.csv
SHA
)
python - "$ROOT" "$FOLDER/manifest.csv.gz" <<'PY'
import csv, gzip, io, os, sys
root, man = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(io.StringIO(gzip.decompress(open(man, "rb").read()).decode("utf-8"))))
v = {e.name for e in os.scandir(f"{root}/video")}
a = {e.name for e in os.scandir(f"{root}/audio")}
mv = [r["clip_id"] for r in rows if r["rel_video"].split("/", 1)[1] not in v]
ma = [r["clip_id"] for r in rows if r["rel_audio"].split("/", 1)[1] not in a]
print(f"rows {len(rows)}  mp4 present {len(rows)-len(mv)}  wav present {len(rows)-len(ma)}  (expect 5294 / 5294 / 5294)")
if mv or ma:
    print("missing mp4 first10", mv[:10]); print("missing wav first10", ma[:10])
    sys.exit(1)
PY
