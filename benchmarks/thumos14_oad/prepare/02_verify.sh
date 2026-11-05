#!/usr/bin/env bash
# thumos14_oad — step 2: every annotated video present and decodable; the media-info census.
set -euo pipefail
yavih bench verify thumos14_oad --census
