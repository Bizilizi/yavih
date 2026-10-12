#!/usr/bin/env bash
# avsync — step 1: the VGGSound SOURCE media. avsync has no media of its own: every row is a window of a VGGSound
# clip, read from the same tree the vggsounder benchmark reads (root media_root = ${datasets.vggsound}). These files
# are read only by step 3 (03_materialize.py), which writes the per-row mp4s the reader hands over (derived_root).
#
#   <ROOT>/video/<clip>.mp4   (H.264, native resolution, mixed ≈ half 720×1280; 6–60 fps)
#   <ROOT>/audio/<clip>.wav   (16 kHz mono; the frozen clips all measure ≥ 10.0 s, max 10.47 s)
#   <ROOT>/metadata/vggsound.csv   (official listing; ytid,start_seconds,label,split; no header; 199 176 rows)
#
# The frozen manifest touches 20 997 distinct clips = 41 994 files, 46.3 GB (39.6 GB mp4 + 6.7 GB wav).
#
# Getting the media (manual): the VGGSound media are YouTube clips, and clips decay. Prefer an existing copy of
# the corpus; otherwise download it as the vggsounder folder's setup.md describes (same corpus, same layout),
# restricted to the 20 997 clips this script lists with LIST_ONLY=1. A re-download is new media: a clip that is
# shorter than 10.0 s of audio makes 03_materialize.py refuse that row (no padding), and re-downloaded bytes change
# the materialized files.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "${LIST_ONLY:-0}" = "1" ]; then
  # The distinct VGGSound clips the frozen manifest needs, one per line (20 997).
  python - "$HERE/../manifest.csv.gz" <<'PYEOF'
import csv, gzip, sys
for c in dict.fromkeys(r["clip_id"].split("#", 1)[0] for r in csv.DictReader(gzip.open(sys.argv[1], "rt", encoding="utf-8", newline=""))):
    print(c)
PYEOF
  exit 0
fi

ROOT="${VGGSOUND_ROOT:-$(yavih config get avsync.media_root)}"
test -s "$ROOT/metadata/vggsound.csv" || { echo "missing $ROOT/metadata/vggsound.csv (official listing; see header)" >&2; exit 2; }
(cd "$ROOT/metadata" && sha256sum -c --strict <<'SHA'
ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8  vggsound.csv
SHA
)
echo "VGGSound tree at $ROOT looks staged; next: 03_materialize.py"
