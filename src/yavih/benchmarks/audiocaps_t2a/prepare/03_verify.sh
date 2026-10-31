#!/usr/bin/env bash
# audiocaps_t2a step 3/3 — every frozen row's reference wav is staged with its frozen sha256; the bench verify record.
set -euo pipefail
yavih bench verify audiocaps_t2a
python - <<'PY'
import csv, gzip, hashlib, io, subprocess
from pathlib import Path
root = Path(subprocess.run(["yavih", "config", "get", "audiocaps_t2a.derived_root"], capture_output=True, text=True, check=True).stdout.strip())
folder = Path(subprocess.run(["yavih", "bench", "path", "audiocaps_t2a"], capture_output=True, text=True, check=True).stdout.strip())
rows = list(csv.DictReader(io.StringIO(gzip.decompress((folder / "manifest.csv.gz").read_bytes()).decode())))
bad = [r["clip_id"] for r in rows if not (root / r["rel_audio"]).is_file()
       or hashlib.sha256((root / r["rel_audio"]).read_bytes()).hexdigest() != r["audio_sha256"]]
print(f"reference audio: {len(rows) - len(bad)}/{len(rows)} staged with the frozen sha256" + (f"; first bad {bad[:5]}" if bad else ""))
raise SystemExit(1 if bad else 0)
PY
