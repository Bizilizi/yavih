#!/usr/bin/env bash
# avrobustbench — step 1: the VGGSound SOURCE media. Every row of this benchmark is a VGGSound clip (clean, or under
# one AVROBUSTBENCH corruption), read from the tree the vggsounder benchmark reads (root media_root =
# ${datasets.vggsound}). These files are read only by step 4 (04_materialize.py), which writes the per-row mp4 + wav
# the reader hands over (derived_root), and by make_manifest.py (metadata/vggsound.csv).
#
#   <ROOT>/video/<clip>.mp4        (H.264, native resolution)
#   <ROOT>/audio/<clip>.wav        (16 kHz mono PCM-16)
#   <ROOT>/metadata/vggsound.csv   (official listing; ytid,start_seconds,label,split; no header; 199 176 rows)
#
# The frozen manifest touches 34 046 source clips (14 046 eval + 20 000 train) = 68 092 files.
#
# Getting the media (manual): VGGSound media are YouTube clips, and clips decay. Prefer an existing copy of the
# corpus; otherwise download it as the vggsounder folder's setup.md describes (same corpus, same layout), restricted
# to the clips this script lists with LIST_ONLY=1. A re-downloaded clip is new media: its corrupted rows differ.
#
# Usage: yavih bench prepare avrobustbench 01            (VGGSOUND_ROOT=<dir> overrides the media_root)
#        LIST_ONLY=1 yavih bench prepare avrobustbench 01 > clips.txt
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "${LIST_ONLY:-0}" = "1" ]; then
  # The distinct VGGSound source clips the frozen manifest needs, one per line (34 046).
  python - "$HERE/../manifest.csv.gz" <<'PYEOF'
import csv, gzip, sys
for c in dict.fromkeys(r["clip_id"].rpartition("#")[0] for r in csv.DictReader(gzip.open(sys.argv[1], "rt", encoding="utf-8", newline=""))):
    print(c)
PYEOF
  exit 0
fi

ROOT="${VGGSOUND_ROOT:-$(yavih config get avrobustbench.media_root)}"
test -s "$ROOT/metadata/vggsound.csv" || { echo "missing $ROOT/metadata/vggsound.csv (official listing; see header)" >&2; exit 2; }
(cd "$ROOT/metadata" && sha256sum -c --strict <<'SHA'
ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8  vggsound.csv
SHA
)
MISSING=$(python - "$HERE/../manifest.csv.gz" "$ROOT" <<'PYEOF'
import csv, gzip, os, sys
root = sys.argv[2]
videos, audios = set(os.listdir(os.path.join(root, "video"))), set(os.listdir(os.path.join(root, "audio")))
srcs = {(r["src_video"], r["src_audio"]) for r in csv.DictReader(gzip.open(sys.argv[1], "rt", encoding="utf-8", newline=""))}
print(sum(1 for v, a in srcs if v.split("/", 1)[1] not in videos or a.split("/", 1)[1] not in audios))
PYEOF
)
echo "VGGSound tree at $ROOT: metadata ok; $MISSING of the manifest's source clips lack their mp4 or wav"
echo "next: yavih bench prepare avrobustbench 03, then 04"
