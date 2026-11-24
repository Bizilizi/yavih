#!/usr/bin/env bash
# nsynth step 2/4 — extract from each archive its examples.json and only the wavs the frozen manifest lists
# (4 000 train, 500 valid, 500 test), into <media_root>/nsynth-<split>/{examples.json,audio/<note_str>.wav}, the
# layout of the archives themselves. Nothing is decoded or converted: the wavs are the stored NSynth files.
#
#   yavih bench prepare nsynth 02                       # media root = $(yavih config get nsynth.media_root)
#   ROOT=/path/to/datasets/nsynth yavih bench prepare nsynth 02
#
# tar reads each archive once from start to end (the train archive is 23.8 GB of gzip: about 5 to 15 minutes);
# 5 000 wavs of 128 044 B each (0.64 GB) and three examples.json (train 270 MB) are written. A split whose
# examples.json and wavs are all present is skipped. Needs GNU tar, gzip and python (for the list).
set -euo pipefail
ROOT="${ROOT:-$(yavih config get nsynth.media_root)}"
FOLDER="${FOLDER:-$(yavih bench path nsynth)}"
PY=${PY:-python}
for s in train valid test; do
  arc="$ROOT/archives/nsynth-$s.jsonwav.tar.gz"
  list="$ROOT/archives/members-$s.txt"
  "$PY" -I - "$FOLDER/manifest.csv.gz" "$s" > "$list" <<'PY'
import csv, gzip, sys
with gzip.open(sys.argv[1], "rt", encoding="utf-8", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["split"] == sys.argv[2]]
print(f"nsynth-{sys.argv[2]}/examples.json")
for r in rows:
    print(r["rel_path"])
PY
  want=$(($(wc -l < "$list")))
  have=0
  while read -r m; do [ -f "$ROOT/$m" ] && have=$((have + 1)); done < "$list"
  if [ "$have" -eq "$want" ]; then
    echo "=== $s: $have/$want members present, skipping ==="
    continue
  fi
  [ -f "$arc" ] || { echo "FAIL: $arc missing (yavih bench prepare nsynth 01)" >&2; exit 1; }
  echo "=== $(date -Is) $s: extracting $want members of $arc ==="
  tar -xzf "$arc" -C "$ROOT" -T "$list"
  have=0
  while read -r m; do [ -f "$ROOT/$m" ] && have=$((have + 1)); done < "$list"
  [ "$have" -eq "$want" ] || { echo "FAIL: $s: $have/$want members extracted" >&2; exit 1; }
  echo "=== $(date -Is) $s: $have members ==="
done
echo "=== OK: examples.json and the manifest's wavs under $ROOT/nsynth-{train,valid,test} ==="
