#!/usr/bin/env bash
# ave_causal — step 1: the ave folder's staged data (AVE_Datset.tar + index.json) serves every frozen row; the census.
# Stage the data with the ave folder first: yavih bench prepare ave 01 / 02 / 03.
set -euo pipefail
yavih bench verify ave_causal --census
