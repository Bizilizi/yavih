#!/usr/bin/env bash
# nyuv2_depth step 2 — derive the eval index (eval_rgb.npy, eval_depth.npy, eval_meta.json) from the public
# labelled .mat (setup.md "## Obtain the data"). Skip when derived_root already holds it with the digests below.
#
# Reads only `images` + `rawDepths` (RAW Kinect, holes = 0; never the inpainted `depths`), 654 test_SN40 rows
# in ascending labelled index; needs numpy + h5py; ~1.4 GB written, 3 files.
# Expected:
#   eval_rgb.npy    602726528 B  sha256 019589449649ab4a1e5602854f65746a70b326a4ea84f72a3e6a2dc08d732a39
#   eval_depth.npy  803635328 B  sha256 fdbc9659f04c11274b2a72bf8e79ac8069df2559603b181103d237583a401fe5
#   eval_meta.json  (records the absolute source path, so its file digest differs per site; make_manifest.py pins it
#                    without that path)
#
# Usage: bash 02_build_eval_index.sh [NYUV2_DIR (<geonet_root>/nyuv2)] [DERIVED_ROOT]
#        defaults: $(yavih config get nyuv2_depth.geonet_root)/nyuv2 and $(yavih config get nyuv2_depth.derived_root)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NYUV2_DIR="${1:-$(yavih config get nyuv2_depth.geonet_root)/nyuv2}"
OUT="${2:-$(yavih config get nyuv2_depth.derived_root)}"
python -I "$HERE/build_eval_index.py" --nyuv2-dir "$NYUV2_DIR" --out "$OUT"
