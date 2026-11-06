#!/usr/bin/env bash
# vggss_loc — step 4: yavih's own presence + read check; --census writes the site media-info table
# from which RawStream.info is served.
#   expected: status verified; test 5154/5154; curation ok (setup.md "Prepare & verify").
# Site override of the media root (never edit a shared config): YAVIH_DATA__VGGSS_LOC__MEDIA_ROOT=/path/to/vggsound
set -euo pipefail
YAVIH="${YAVIH:-yavih}"
"$YAVIH" bench verify vggss_loc --census
