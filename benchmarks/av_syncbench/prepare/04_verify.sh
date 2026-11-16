#!/usr/bin/env bash
# av_syncbench, step 4: yavih's own presence and read check; --census writes the site media-info table from which
# RawStream.info is served. Expected: status verified; test 37743/37743 (setup.md "Prepare & verify").
set -euo pipefail
YAVIH="${YAVIH:-yavih}"
"$YAVIH" bench verify av_syncbench --census
