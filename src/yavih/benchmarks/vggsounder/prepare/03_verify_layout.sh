#!/usr/bin/env bash
# vggsounder — step 3: verify the staged VGGSound tree the benchmark reads (layout "dir").
#
#   <ROOT>/video/<clip_id>.mp4   199 176 files, 318 GiB   (frozen manifest touches 32 008)
#   <ROOT>/audio/<clip_id>.wav   199 176 files,  61 GiB   (16 kHz mono)
#   <ROOT>/metadata/vggsound.csv
#
# Inode budget: 398 352 loose files. Never extract bulk tars in place next to this tree.
# One directory listing per dir: never stat() 400 k files one by one on NFS.
set -euo pipefail

ROOT="${VGGSOUND_ROOT:-$(yavih config get vggsounder.media_root)}"

(cd "$ROOT/metadata" && sha256sum -c --strict <<'EOF'
ed9fdeed1b4a4ee34bb1f3722af8690354815b27cd55daea53576be79283f1f8  vggsound.csv
EOF
)

python - "$ROOT" <<'EOF'
import csv, os, sys
root = sys.argv[1]
videos = {e.name for e in os.scandir(f"{root}/video")}
audios = {e.name for e in os.scandir(f"{root}/audio")}
ids = [f"{r[0]}_{int(r[1]):06d}" for r in csv.reader(open(f"{root}/metadata/vggsound.csv", newline="", encoding="utf-8"))]
mv = [i for i in ids if f"{i}.mp4" not in videos]
ma = [i for i in ids if f"{i}.wav" not in audios]
print(f"csv rows {len(ids)}  mp4 present {len(ids) - len(mv)}  wav present {len(ids) - len(ma)}  (a complete copy: 199176 / 199176 / 199176)")
if mv or ma:
    print("missing mp4 first10", mv[:10]); print("missing wav first10", ma[:10])
EOF

echo "next: yavih bench verify vggsounder   (expects train 20000/20000, test 12008/12008, decode sample 32/32 ok)"
