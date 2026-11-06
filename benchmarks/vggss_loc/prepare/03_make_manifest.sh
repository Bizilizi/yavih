#!/usr/bin/env bash
# vggss_loc — step 3 (optional): make_manifest.py --check re-derives the frozen manifest from vggss.json and the
# VGGSound listing and compares it byte for byte with the committed one (README.md "Re-freezing the manifest").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$HERE/make_manifest.py" --check
