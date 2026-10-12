#!/usr/bin/env bash
# avsync — step 4: the manifest re-derives byte-identically from its clip selection (offset draw, columns, split
# audit against vggsound.csv, presence of the 20 997 source clips: one scandir per directory), and yavih sees every
# materialized row (bench verify: presence per split, read sample, media-info census).
#   PYTHON / YAVIH select the interpreter and CLI: PYTHON=/path/to/python YAVIH=/path/to/yavih yavih bench prepare avsync 04
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"
YAVIH="${YAVIH:-yavih}"

OUT="$(mktemp)"; trap 'rm -f "$OUT"' EXIT
"$PY" "$HERE/../make_manifest.py" --check > "$OUT" || { echo "make_manifest --check: DRIFT (see $OUT)" >&2; trap - EXIT; exit 1; }
"$PY" - "$OUT" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
c = r["upstream"]["curation"]["per_split"]
print("make_manifest --check:", r["check"], "| from", r["source"], "| rows", r["rows"], "| source clips missing", {s: v["missing"] for s, v in c.items()})
PYEOF

"$YAVIH" bench verify avsync --census
