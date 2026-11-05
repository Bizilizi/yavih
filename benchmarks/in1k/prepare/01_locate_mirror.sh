#!/usr/bin/env bash
# in1k step 1 — locate (never download) the ImageNet-1K torchvision root. setup.md "## Obtain the data".
#
# yavih does not download ImageNet (image-net.org terms of access: account + signed agreement). This step checks
# an existing torchvision.datasets.ImageNet root (tars + meta.bin + extracted train/ and val/). If only the tars
# exist, build the root (02) or stage just the 100k frozen files from a machine that has one (03).
#
# Usage: bash 01_locate_mirror.sh [ROOT]      (exit 0 = usable extracted root; 2 = tars only; 1 = nothing)
#        default ROOT: $(yavih config get in1k.media_root)
set -euo pipefail
ROOT="${1:-$(yavih config get in1k.media_root)}"
echo "[in1k/01] root: $ROOT"
[ -d "$ROOT" ] || { echo "[in1k/01] not a directory; see setup.md \"## Obtain the data\""; exit 1; }

# Expected files (sizes in bytes; sha256, train tar: torchvision md5):
#   ILSVRC2012_img_train.tar   147897477120   md5 1d675b47d978889d74fa0da5fadfb00e
#   ILSVRC2012_img_val.tar       6744924160   sha256 c7e06a6c0baccf06d8dbeb6577d71efff84673a5dbdd50633ab44f8ea0456ae0
#   ILSVRC2012_devkit_t12.tar.gz    2568145   sha256 b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953
#   meta.bin                         303087   sha256 c0c241b17643e8cd3a3f5028d8bd93be950aec0c84fa9bf42079d2c1199c3db3
for f in ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar ILSVRC2012_devkit_t12.tar.gz meta.bin; do
  if [ -e "$ROOT/$f" ]; then printf '[in1k/01]   %-30s %15d B\n' "$f" "$(stat -c %s "$ROOT/$f")"; else echo "[in1k/01]   $f: MISSING"; fi
done

# The extracted torchvision layout the frozen manifest's rel_media paths point into:
#   train/<wnid>/<wnid>_<n>.JPEG  (1000 dirs, 1 281 167 files)     val/<wnid>/ILSVRC2012_val_<8d>.JPEG (1000 dirs x 50)
ok=1
for split in train val; do
  if [ -d "$ROOT/$split" ]; then
    n=$(find "$ROOT/$split" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "[in1k/01]   $split/: $n class dirs (expect 1000)"
    [ "$n" -eq 1000 ] || ok=0
  else
    echo "[in1k/01]   $split/: MISSING"; ok=0
  fi
done
# A root with an empty val/ makes the eval split 0 % present (setup.md "## Gotchas" 1): check one val class dir has its 50 files.
if [ -d "$ROOT/val/n01440764" ]; then
  nv=$(ls "$ROOT/val/n01440764" | wc -l); echo "[in1k/01]   val/n01440764: $nv files (expect 50)"; [ "$nv" -eq 50 ] || ok=0
fi
if [ "$ok" -eq 1 ]; then
  echo "[in1k/01] usable torchvision root. Configure it: export YAVIH_DATA__IN1K__MEDIA_ROOT=$ROOT"
  echo "[in1k/01]   or in ~/.config/yavih/config.toml:  [data.in1k]  media_root = \"$ROOT\""
  exit 0
fi
if [ -e "$ROOT/ILSVRC2012_img_val.tar" ]; then
  echo "[in1k/01] tars only: run 02_extract_torchvision_root.sh into a NEW directory (needs 1.33 M inodes, 155 GB) or stage the 100k subset (03)."
  exit 2
fi
exit 1
