#!/usr/bin/env bash
# ESC-50 step 1/3 — clone the public GitHub repository at the pinned commit (`git clone --depth 1`).
# No account or agreement needed.
#
#   OUT=/path/to/datasets/esc50 bash 01_download.sh   # default: $(yavih config get esc50.media_root)
#
# Size: the shallow pack is 645 520 453 B; the clone is 1.53 GB / 2 038 files (audio/ = 2 000 wav,
# 882 MB). Alternative without git: https://github.com/karoldvl/ESC-50/archive/master.zip (~600 MB;
# unzip and rename ESC-50-master/ -> esc50/), then run 02_verify.sh — the pinned per-file digest is the
# arbiter, not the transport.
set -euo pipefail
OUT="${OUT:-$(yavih config get esc50.media_root)}"
REPO=https://github.com/karolpiczak/ESC-50.git
COMMIT=33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6   # commit date 2023-10-13

if [ -f "$OUT/meta/esc50.csv" ]; then
  echo "=== $OUT already holds meta/esc50.csv; skipping the clone (run 02_verify.sh) ==="
else
  mkdir -p "$(dirname "$OUT")"
  echo "=== $(date -Is) cloning $REPO -> $OUT ==="
  git clone --depth 1 "$REPO" "$OUT"
  head=$(git -C "$OUT" rev-parse HEAD)
  if [ "$head" != "$COMMIT" ]; then
    # Upstream master moved: check out the pinned commit (GitHub serves reachable SHAs).
    echo "=== HEAD is $head, fetching the pinned $COMMIT ==="
    git -C "$OUT" fetch --depth 1 origin "$COMMIT"
    git -C "$OUT" checkout --detach "$COMMIT"
  fi
  echo "=== $(date -Is) CLONED $(git -C "$OUT" rev-parse HEAD) $(du -sh "$OUT" | cut -f1) ==="
fi

echo "ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246  $OUT/meta/esc50.csv" | sha256sum -c -
echo "f6aec3a3283a2efdcbb1feff334adad7ab17c097bdec0cbb4eebdb9e16886074  $OUT/LICENSE" | sha256sum -c -
