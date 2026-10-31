#!/usr/bin/env bash
# vggsounder_v2a — step 2: every frozen row's muted video and reference audio are staged; the media-info census.
set -euo pipefail
yavih bench verify vggsounder_v2a --census
python - <<'PY'
import csv, gzip, io, subprocess
from pathlib import Path
root = Path(subprocess.run(["yavih", "config", "get", "vggsounder_v2a.media_root"], capture_output=True, text=True, check=True).stdout.strip())
folder = Path(subprocess.run(["yavih", "bench", "path", "vggsounder_v2a"], capture_output=True, text=True, check=True).stdout.strip())
rows = list(csv.DictReader(io.StringIO(gzip.decompress((folder / "manifest.csv.gz").read_bytes()).decode())))
missing = [r["clip_id"] for r in rows if not (root / r["rel_audio"]).is_file()]
print(f"reference audio: {len(rows) - len(missing)}/{len(rows)} present" + (f"; first missing {missing[:5]}" if missing else ""))
raise SystemExit(1 if missing else 0)
PY
