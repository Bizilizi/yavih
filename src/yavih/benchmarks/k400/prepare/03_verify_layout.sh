#!/usr/bin/env bash
# k400 step 3 — verify the staged tree: container census per split, presence of every frozen row,
# then `yavih bench verify k400`. One scandir per directory (never 260 k stat() calls on NFS);
# minutes cold on network storage, seconds warm. Usage: yavih bench prepare k400 03 [-- ROOT]
# (default ROOT: $(yavih config get k400.media_root)).
set -euo pipefail
HERE="${FOLDER:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="${1:-$(yavih config get k400.media_root)}"
PY="${PYTHON:-python3}"

"$PY" - "$ROOT" "$HERE/manifest.csv.gz" <<'PY'
import collections, csv, gzip, io, os, sys
root, man = sys.argv[1], sys.argv[2]
present, census = set(), {}
for split in ("train", "val"):
    c = collections.Counter()
    for d in os.scandir(f"{root}/{split}"):
        for e in os.scandir(d.path):
            c[os.path.splitext(e.name)[1]] += 1
            present.add(f"{split}/{d.name}/{e.name}")
    census[split] = dict(sorted(c.items()))
print("census", census)
print("expect  {'train': {'.mp4': 240618}, 'val': {'.mkv': 5402, '.mp4': 13492, '.webm': 510}}")
rows = list(csv.DictReader(io.StringIO(gzip.decompress(open(man, "rb").read()).decode("utf-8"), newline="")))
for split in ("train", "val"):
    rs = [r for r in rows if r["split"] == split]
    miss = [r["rel_media"] for r in rs if r["rel_media"] not in present]
    print(f"{split}: frozen rows present {len(rs) - len(miss)}/{len(rs)}  missing_first5={miss[:5]}")
PY

echo "[k400/03] yavih bench verify k400 (expects train 20000/20000, val 13492/13492, decode sample ok):"
yavih bench verify k400 || true
