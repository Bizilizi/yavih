#!/usr/bin/env bash
# speech_commands step 3/3 — re-derive the manifest from the staged media (make_manifest.py --check: every pin of
# the archives' contents, presence of every listed wav, the WAV headers of the read splits), then verify the
# staged data with yavih (manifest hash, presence per split, the silence SHA256SUMS of data.derived, a decode
# sample, the media-info census).
#
#   yavih bench prepare speech_commands 03                                  # media root = $(yavih config get speech_commands.media_root)
#   ROOT=/path/to/datasets/speech_commands yavih bench prepare speech_commands 03
#
# Needs steps 01 and 02. Reading ~23 000 WAV headers takes a minute or two on a network file system.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
MEDIA_ROOT="${ROOT:-$(yavih config get speech_commands.media_root)}"

"$PY" "$HERE/../make_manifest.py" --media-root "$MEDIA_ROOT" --check
YAVIH_DATA__SPEECH_COMMANDS__MEDIA_ROOT="$MEDIA_ROOT" yavih bench verify speech_commands --census
