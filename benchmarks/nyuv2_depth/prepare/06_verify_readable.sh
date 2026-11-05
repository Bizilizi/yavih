#!/usr/bin/env bash
# nyuv2_depth step 6 — FULL-read pre-flight of the 12 000 train .mat files (setup.md "## Gotchas"). The re-container
# and target scripts read with variable_names=[...] (fast), which a truncated file survives; this full
# scipy.io.loadmat does not. ~56 GB of reads: shard it (e.g. 32 shards × 375 files, 2 CPUs, 8 GB, ~71 s each).
# Do not scan the whole 30k pool (~143 GB).
#
# Usage, one shard:   bash 06_verify_readable.sh <GEONET_ROOT> <OUT_DIR> <SHARD> <NUM_SHARDS> [--limit N]
#        merge:       python verify_readable.py merge --out-dir <OUT_DIR> --num-shards <NUM_SHARDS>
#        SLURM:       sbatch --array=0-31%16 --cpus-per-task=2 --mem=8G --time=02:00:00 \
#                       --wrap 'bash 06_verify_readable.sh <GEONET_ROOT> <OUT_DIR> $SLURM_ARRAY_TASK_ID 32'
# Expected: 12000 checked, 0 failed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GEONET="${1:?usage}"; OUT="${2:?usage}"; SHARD="${3:-0}"; N="${4:-1}"; shift $(( $# < 4 ? $# : 4 ))
python "$HERE/verify_readable.py" shard --geonet-root "$GEONET" --out-dir "$OUT" --shard "$SHARD" --num-shards "$N" "$@"
