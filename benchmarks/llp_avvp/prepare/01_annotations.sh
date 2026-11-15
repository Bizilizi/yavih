#!/usr/bin/env bash
# Step 01: the official LLP annotation files (public, no account; 1.0 MB) from the authors' GitHub repository
# YapengTian/AVVP-ECCV20 at the pinned commit, each checked against its sha256, into the prepare-only
# annotations_root. make_manifest.py reads them; the frozen manifest already carries every label.
#   yavih bench prepare llp_avvp 01
set -euo pipefail
ROOT="${ROOT:-$(yavih config get llp_avvp.annotations_root)}"
COMMIT=fde56118f2c2e7c19188dfefa2569a37b70054fe
BASE="https://raw.githubusercontent.com/YapengTian/AVVP-ECCV20/${COMMIT}/data"
mkdir -p "$ROOT"
while read -r sha name; do
  dst="$ROOT/$name"
  if [ ! -f "$dst" ] || ! echo "$sha  $dst" | sha256sum -c --status; then
    echo "[01] fetching $name"
    curl -fsSL --retry 3 -o "$dst.part" "$BASE/$name"
    mv "$dst.part" "$dst"
  fi
  echo "$sha  $dst" | sha256sum -c --quiet || { echo "[01] $name: sha256 mismatch (expected $sha)" >&2; exit 1; }
done <<'SUMS'
bbeb83e8af46a85075216b213d3464f92dd4ab71468c852ae454b82cc745c0a4 AVVP_train.csv
1811813b54c5f43c3be0a97bd0971cfb8a93818ca7b3ab96cdc7f4ac37ef37a6 AVVP_val_pd.csv
3f7a2b040e3b67c15d08efef25002bf8c5b2320f11049eb99ae0eb2d62f8eb9b AVVP_test_pd.csv
d5435664668bec6e7eafb39059d2f424ed2682f6b55395262fc3cb691b7d0778 AVVP_eval_audio.csv
e2274af694f3f27e38b840e0cd91a9d8d9ac85280f193e90828d26b22db69abe AVVP_eval_visual.csv
89f112ffc602279c87cc2b2040c4cd42eaf163a9d7e61cf4e74250a343d21e56 AVVP_dataset_full.csv
SUMS
echo "[01] ok: 6 annotation files in $ROOT"
