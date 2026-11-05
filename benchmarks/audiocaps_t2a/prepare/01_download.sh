#!/usr/bin/env bash
# audiocaps_t2a step 1/3 — download the official test captions and the OpenSound audio mirror at pinned revisions
# (setup.md "## Obtain the data"). Public and ungated: no token. 402 KB + 41 parquet files, 3.4 GB.
#
#   DEST=/path/to/datasets/audiocaps bash 01_download.sh   # default: `yavih config get audiocaps_t2a.source_root`
#
# Needs curl and the Hugging Face CLI (`pip install -U huggingface_hub`, which provides `hf`). Re-running resumes.
set -euo pipefail
DEST="${DEST:-$(yavih config get audiocaps_t2a.source_root)}"
CAPTIONS_COMMIT=80fb3ade6d400d4311354a360db25da051a48cc1
CAPTIONS_SHA256=b91c4b7ded2f4f6e7db5b9c4983dc1e1dca3d556f505b61e3fd65cac7e1c638a
MIRROR_REVISION=b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded
mkdir -p "$DEST/upstream"
echo "=== $(date -Is) cdjkim/audiocaps@$CAPTIONS_COMMIT dataset/test.csv -> $DEST/upstream/test.csv"
curl -fsSL "https://raw.githubusercontent.com/cdjkim/audiocaps/$CAPTIONS_COMMIT/dataset/test.csv" -o "$DEST/upstream/test.csv.part"
echo "$CAPTIONS_SHA256  $DEST/upstream/test.csv.part" | sha256sum -c -
mv "$DEST/upstream/test.csv.part" "$DEST/upstream/test.csv"
echo "=== $(date -Is) hf download OpenSound/AudioCaps @ $MIRROR_REVISION (test split) -> $DEST/opensound-b29b324"
hf download OpenSound/AudioCaps --repo-type dataset --revision "$MIRROR_REVISION" --include "data/test-*" --include README.md \
  --local-dir "$DEST/opensound-b29b324"
echo "=== $(date -Is) DOWNLOAD_DONE $(du -sh "$DEST" | cut -f1) in $DEST"
