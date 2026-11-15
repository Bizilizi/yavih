#!/usr/bin/env bash
# vggsounder_retrieval — step 2 (optional): make_manifest.py --check re-derives the frozen manifest from the vggsounder
# 1.0 wheel's data/vggsounder.csv and the VGGSound listing and compares it byte for byte with the committed one.
# The wheel is downloaded (no install, no dependencies) into a temporary directory unless VGGSOUNDER_CSV points at
# the file already. vggsounder 1.0 declares Requires-Python >=3.11; only the wheel's data file is read, so the download
# asks for the 3.11 wheel explicitly (`--python-version 3.11 --only-binary=:all:`) and works from any interpreter
# yavih supports (3.10 included).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV="${VGGSOUNDER_CSV:-}"
if [ -z "$CSV" ]; then
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
  python -m pip download --no-deps --python-version 3.11 --only-binary=:all: vggsounder==1.0 -d "$TMP" -q
  python -m zipfile -e "$TMP"/vggsounder-1.0-py3-none-any.whl "$TMP/x"
  CSV="$TMP/x/vggsounder/data/vggsounder.csv"
fi
python "$HERE/make_manifest.py" --metadata "$CSV" --check > /dev/null && echo "manifest matches (make_manifest.py --check)"
