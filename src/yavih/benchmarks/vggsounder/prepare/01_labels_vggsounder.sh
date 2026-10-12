#!/usr/bin/env bash
# vggsounder — step 1: the VGGSounder annotations (public, PyPI). Offline after the install: the
# package bundles its label CSVs and the 309-class vocabulary; nothing else is fetched
# (`download_samples=False` everywhere).
#
# The frozen eval rows are VGGSounder(background_music=False) (12 014 videos, the "default without
# background music" variant) minus 6 videos whose YouTube id is in VGGSound-train. Every vggsounder metric
# and floor is computed by this package (benchmark.yaml `external`, probe.py), so the PROBE env needs it;
# the model env (bridges) does not. It is not a yavih dependency: install it independently, exactly as
# below (setup.md "## Dependencies").
#
# Expected sha256 of the bundled data files (vggsounder 1.0):
#   classes.csv                           90a5cd19345019f50d0c3540629c1f4cde33266f00c57ce0cf67cd69835993c2
#   vggsounder.csv                        400979ea3dee34bed145d3fd96966e16689352df63b645c4730caf8eb2cacb06
#   vggsounder+background-music.csv       6279875ac93df11d90b1a9d60da7dee89073ce233b0843759856bc72e00396ba
set -euo pipefail

PY="${PYTHON:-python}"

# --no-deps is load-bearing: the package's `datasets` dependency can upgrade numpy past what the installed
# torch supports. The scorer (vggsounder.benchmark) imports pandas, scikit-learn and tqdm, which --no-deps
# does not install: take them from the probe env (`pip install pandas scikit-learn tqdm` if absent).
# The pin is the PyPI release "1.0" (importlib.metadata reports "1.0"); a source checkout of the authors'
# repo reports another version (e.g. "1.0.0") and renamed `auc_pr` to `mAP`, so probe.py refuses it.
"$PY" -m pip install --no-deps "vggsounder==1.0"

PKG_DATA="$("$PY" -c 'import pathlib, vggsounder; print(pathlib.Path(vggsounder.__file__).parent / "data")')"
echo "vggsounder data dir: $PKG_DATA"
(cd "$PKG_DATA" && sha256sum -c --strict <<'EOF'
90a5cd19345019f50d0c3540629c1f4cde33266f00c57ce0cf67cd69835993c2  classes.csv
400979ea3dee34bed145d3fd96966e16689352df63b645c4730caf8eb2cacb06  vggsounder.csv
6279875ac93df11d90b1a9d60da7dee89073ce233b0843759856bc72e00396ba  vggsounder+background-music.csv
EOF
)
"$PY" -c 'import importlib.metadata as m, pandas, sklearn, tqdm; v = m.version("vggsounder"); assert v == "1.0", f"vggsounder {v} installed, the protocol pins 1.0"' \
  || { echo "probe env incomplete: need vggsounder==1.0 (PyPI) + pandas, scikit-learn, tqdm" >&2; exit 3; }
echo "ok: vggsounder==1.0 data files verified; scorer imports (pandas, scikit-learn, tqdm) present"
