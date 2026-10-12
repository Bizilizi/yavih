#!/usr/bin/env bash
# vggss_box — step 5: yavih's own presence + read check; --census writes the site media-info table
# from which RawStream.info is served.
#   expected: status verified; train 3700/3700, eval 914/914; curation ok (setup.md "Prepare & verify").
# Site override of the media root (never edit a shared config): YAVIH_DATA__VGGSS_BOX__MEDIA_ROOT=/path/to/vggsound
set -euo pipefail
YAVIH="${YAVIH:-yavih}"
"$YAVIH" bench verify vggss_box --census
