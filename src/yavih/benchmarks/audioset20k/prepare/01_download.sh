#!/usr/bin/env bash
# audioset20k step 1/2 — download the published dataset at the pinned revision (setup.md "## Obtain the data").
# Public, ungated: no token needed. 62 tars + index.json + clips.csv + MISSING.csv + SHA256SUMS + README.md,
# 66.1 GB in 67 files (inode-free: the tars are read in place, never extracted).
#
#   DEST=/path/to/datasets/audioset20k-wds-v1 bash 01_download.sh   # default: `yavih config get audioset20k.media_root`
#
# Needs the Hugging Face CLI (`pip install -U huggingface_hub`, which provides `hf`). Re-running resumes.
set -euo pipefail
DEST="${DEST:-$(yavih config get audioset20k.media_root)}"
REVISION=4a0055b21c183257fd9b65ee8d1cd11f13362f0f
echo "=== $(date -Is) hf download 11hu83/audioset20k @ $REVISION -> $DEST"
hf download 11hu83/audioset20k --repo-type dataset --revision "$REVISION" --local-dir "$DEST"
echo "=== $(date -Is) DOWNLOAD_DONE $(du -sh "$DEST" | cut -f1) in $DEST"
