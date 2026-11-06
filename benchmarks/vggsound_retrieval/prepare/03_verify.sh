#!/usr/bin/env bash
# vggsound_retrieval — step 3: yavih's own presence + read check; --census writes the site media-info table
# from which RawStream.info is served.
#   expected: status verified; test 15440/15440; curation ok (setup.md "Prepare & verify").
# Site override of the media root: YAVIH_DATA__VGGSOUND_RETRIEVAL__MEDIA_ROOT=/path/to/vggsound
set -euo pipefail
YAVIH="${YAVIH:-yavih}"
"$YAVIH" bench verify vggsound_retrieval --census
