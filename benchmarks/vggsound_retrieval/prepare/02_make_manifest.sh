#!/usr/bin/env bash
# vggsound_retrieval — step 2 (optional): make_manifest.py --check re-derives the frozen manifest from the VGGSound
# listing and the shipped subset list and compares it byte for byte with the committed one.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$HERE/make_manifest.py" --check
