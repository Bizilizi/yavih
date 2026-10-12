#!/usr/bin/env bash
# in1k step 4 — data verification (setup.md "## Prepare & verify"). Nothing is downloaded or written outside
# $YAVIH_HOME/data_status.
#
# Usage: bash 04_verify.sh [ROOT]      (default ROOT: $(yavih config get in1k.media_root))
set -euo pipefail
ROOT="${1:-$(yavih config get in1k.media_root)}"
export YAVIH_DATA__IN1K__MEDIA_ROOT="$ROOT"
# Presence of all 100 000 files (one scandir per class dir), 32 decoded JPEGs, curation sums, fingerprint;
# --census writes the media-info table (width, height, container, bytes).
yavih bench verify in1k --census
# Expected: train present 50000/50000, val present 50000/50000, decode sample ok=32 failed=0, status verified.
# val present 0/50000 means the root's val/ is unpopulated (setup.md "## Gotchas" 1).
# Re-freezing the manifest (make_manifest.py --check) is a maintainer step: README.md "Re-freezing the manifest".
