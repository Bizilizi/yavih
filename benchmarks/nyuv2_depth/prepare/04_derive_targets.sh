#!/usr/bin/env bash
# nyuv2_depth step 4 — derive the dense depth targets into derived_root (setup.md "## Prepare & verify").
#
# train_depth_224.npy  float32 [12000, 224, 224]  2408448128 B  sha256 af4988cbc3f5eefb1e93e5d5a6f6d59793208b49ba2ef8c2156c760df3933b49
# eval_depth_224.npy   float32 [  654, 224, 224]   131260544 B  sha256 08a16cbafc34647b60f30c10669cc6ae507699161bcba108862f3533162a6888
# Rule: truncate [:480,:640] → keep target_region [0, 0.125, 1, 0.875] = columns [80, 560) → nearest-exact depth / nearest mask to 224 → 0 < d <= 10, invalid = 0
# (--check-torch N compares against torch.nn.functional.interpolate).
# Needs numpy + scipy; the 16-row torch cross-check runs only when torch imports (skipped otherwise).
# ~10 min train, ~25 s eval with 16 workers.
#
# Usage: bash 04_derive_targets.sh <GEONET_ROOT> <DERIVED_ROOT (holds eval_depth.npy; outputs land here)> [--workers N]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GEONET="${1:?usage: 04_derive_targets.sh <geonet_root> <derived_root>}"; OUT="${2:?usage}"; shift 2
CT=()
if python -c 'import torch' 2>/dev/null; then CT=(--check-torch 16); else echo "[nyuv2/04] torch not importable: --check-torch cross-check skipped"; fi
python "$HERE/derive_targets.py" --split all --geonet-root "$GEONET" --eval-index "$OUT" --out "$OUT" "${CT[@]}" --progress "$@"
sha256sum "$OUT/train_depth_224.npy" "$OUT/eval_depth_224.npy"
