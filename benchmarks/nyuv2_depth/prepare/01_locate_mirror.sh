#!/usr/bin/env bash
# nyuv2_depth step 1 — locate the Probe3D NYUv2 data layout (setup.md "## Obtain the data"). NOTHING is downloaded.
#
# MANUAL STEP: the GeoNet train frames are not fetchable by script. Build the layout by hand as Probe3D's data
# README describes (request access from the dataset authors if the GeoNet links are unavailable):
#   GeoNet: download data1.zip + data2.zip from the links in https://github.com/xjqi/GeoNet (Google Drive /
#           Baidu; interactive), unzip both, mv data1/* data2/* nyu_geonet/  → 30 916 MATLAB v7 .mat, 142 791 955 406 B
#   labelled set: wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat  (2 972 037 809 B)
#                 wget https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/nyuv2_surfacenormal_metadata.zip
#                 unzip; mv surfacenormal_metadata/* nyuv2/   (test_SN40.json, train_SN40.json, vocab40.mat, all_normals.pklz)
# yavih never runs those commands; they are listed so a human can.
#
# Expected (sha256 also in benchmark.yaml source.sha256):
#   nyu_geonet/                    30916 *.mat files (sorted listing index 6919 = bedroom_0129_r-1315328302.785180-1089784518.mat,
#                                                     21181 = library_0006_r-1300708259.539496-3335480784.mat)
#   nyuv2/nyu_depth_v2_labeled.mat 2972037809 B  sha256 2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa
#   nyuv2/test_SN40.json                54282 B  sha256 2aee4e157046cc7991161ddfea1a8e8f76b5f9883f519615fcb900b168de6b21
#   nyuv2/train_SN40.json               65985 B  sha256 694c3ba6a1a2ba712c3a9df59a6fc29e93f39438e197442d474fbb3f81d341e3
#
# Usage: bash 01_locate_mirror.sh [ROOT]     exit 0 usable, 1 not found / incomplete
#        default ROOT: $(yavih config get nyuv2_depth.geonet_root)
set -euo pipefail
ROOT="${1:-$(yavih config get nyuv2_depth.geonet_root)}"
ok=0
[ -d "$ROOT/nyu_geonet" ] || { echo "[nyuv2/01] $ROOT/nyu_geonet missing — build it as this script's header describes (setup.md "## Obtain the data")"; exit 1; }
n=$(find "$ROOT/nyu_geonet" -maxdepth 1 -name '*.mat' | wc -l)
echo "[nyuv2/01] nyu_geonet: $n .mat (expected 30916; a 12000-file staged subset is also usable for this benchmark)"
[ "$n" -eq 30916 ] || [ "$n" -eq 12000 ] || ok=1
if [ -f "$ROOT/nyuv2/nyu_depth_v2_labeled.mat" ]; then
  echo "[nyuv2/01] labelled .mat: $(stat -c %s "$ROOT/nyuv2/nyu_depth_v2_labeled.mat") B (expected 2972037809; only needed to rebuild the eval index, step 2)"
else
  echo "[nyuv2/01] no nyuv2/nyu_depth_v2_labeled.mat here — fine if the eval index (step 2) already exists in derived_root"
fi
echo "[nyuv2/01] configure:  export YAVIH_DATA__NYUV2_DEPTH__GEONET_ROOT=$ROOT"
echo "           or ~/.config/yavih/config.toml:  [data.nyuv2_depth]  geonet_root = \"$ROOT\""
exit $ok
