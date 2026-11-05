#!/usr/bin/env bash
# in1k step 2 (only with the ILSVRC2012 tars and no extracted train/ val/) — build a torchvision ImageNet root,
# as torchvision.datasets.ImageNet does it. Never extract in place next to tars you do not own; write into a
# NEW directory you own.
#
# Budget: 1 281 167 + 50 000 JPEGs = 1 331 167 inodes + 2 000 dirs, ~155 GB. On inode-capped filesystems, stage
# the 100k frozen files instead (03_stage_probe_subset.sh).
#
# Usage: bash 02_extract_torchvision_root.sh TARS_DIR NEW_ROOT
#   TARS_DIR must hold ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar, ILSVRC2012_devkit_t12.tar.gz.
set -euo pipefail
TARS="${1:?TARS_DIR}"; ROOT="${2:?NEW_ROOT}"
[ -e "$ROOT/train" ] && { echo "[in1k/02] $ROOT/train exists; refusing to extract over it"; exit 1; }
mkdir -p "$ROOT"
# Expected inputs (train tar: torchvision md5 only):
echo "c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0  $TARS/ILSVRC2012_img_val.tar" | sha256sum -c -
echo "b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953  $TARS/ILSVRC2012_devkit_t12.tar.gz" | sha256sum -c -
echo "1d675b47d978889d74fa0da5fadfb00e  $TARS/ILSVRC2012_img_train.tar" | md5sum -c -   # 147 897 477 120 B

# Symlink (do not copy) the tars into the new root so torchvision finds them, then let torchvision do the
# extraction: train tar = 1000 per-class tars -> train/<wnid>/; val tar + devkit -> val/<wnid>/ via the
# ground-truth file; meta.bin is written from the devkit. This is exactly the layout the frozen manifest names.
for f in ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar ILSVRC2012_devkit_t12.tar.gz; do ln -sfn "$TARS/$f" "$ROOT/$f"; done
python - "$ROOT" <<'PY'
import sys
from torchvision.datasets import ImageNet  # torchvision >= 0.9; pillow
root = sys.argv[1]
for split in ("val", "train"):            # val first: small, and it validates the devkit before the 147 GB pass
    ds = ImageNet(root, split=split)      # extracts on first use (idempotent: skips when the split dir exists)
    print(f"[in1k/02] {split}: {len(ds)} images, {len(ds.classes)} classes")
    assert len(ds.wnids) == 1000
PY
# Expected: val 50000 / train 1281167. Then: yavih bench prepare in1k 01 -- "$ROOT" && yavih bench verify in1k
echo "[in1k/02] done: $ROOT"
