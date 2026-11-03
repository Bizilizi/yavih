#!/usr/bin/env bash
# unav100_causal — step 3: every frozen video present and decodable; the media-info census.
set -euo pipefail
yavih bench verify unav100_causal --census
