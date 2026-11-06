#!/usr/bin/env bash
# Verify the staged Perception Test media for this benchmark: manifest hash, presence per split, a read sample and the
# media-info census (needs yavih[media]). Run as `yavih bench prepare <benchmark> 02 [-- extra verify args]`.
#   YAVIH selects the CLI: YAVIH="python -m yavih" yavih bench prepare <benchmark> 02
set -euo pipefail
YAVIH="${YAVIH:-yavih}"
$YAVIH bench verify "${YAVIH_BENCH:?run through yavih bench prepare}" --census "$@"
