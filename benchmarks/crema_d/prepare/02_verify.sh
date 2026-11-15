#!/usr/bin/env bash
# CREMA-D step 2/3 — verify the staged wavs: the 7 438 the manifest lists are present, every one is a 16 kHz mono
# PCM_16 RIFF wav (the facts input_contract.info promises: sample_rate 16000, channels 1; the length varies), every
# one has the sha256 that TensorFlow Datasets recorded for it when it built crema_d 1.0.0 (its public manual
# checksums, staged by step 01 as tfds/crema_d-manual_checksums.txt), and the digest of the whole audio tree matches.
#
#   yavih bench prepare crema_d 02     # the clone at $(yavih config get crema_d.media_root); ROOT=/path overrides
#
# The digest is the sha256 of `cd AudioWAV && export LC_ALL=C && sha256sum <the 7 438 names, sorted>.wav`
# (benchmark.yaml source.sha256["AudioWAV.sha256sum"]); the file list is the manifest's, so the four wavs TFDS
# skips never enter it. Needs python with numpy and soundfile (the yavih [media] extra). Reading 0.6 GB takes a
# few minutes on a network file system.
set -euo pipefail
ROOT="${ROOT:-$(yavih config get crema_d.media_root)}"
PY=${PY:-python}
FOLDER="${FOLDER:-$(cd "$(dirname "$0")/.." && pwd)}"
CHECKSUMS="$ROOT/tfds/crema_d-manual_checksums.txt"
echo "66b5d959dcee997321baf9080c80022a6937279419446db3a9d4827d839b0abb  $CHECKSUMS" | sha256sum -c -
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
zcat "$FOLDER/manifest.csv.gz" | tail -n +2 | cut -d, -f1 | LC_ALL=C sort > "$work/names"
[ "$(wc -l < "$work/names")" -eq 7438 ] || { echo "FAIL: the manifest lists $(wc -l < "$work/names") clips, expected 7438" >&2; exit 1; }
"$PY" -I - "$ROOT/AudioWAV" "$work/names" <<'PY'
import collections, os, sys
import soundfile as sf
d, names = sys.argv[1], [n.strip() for n in open(sys.argv[2]) if n.strip()]
c, frames = collections.Counter(), []
for n in names:
    i = sf.info(os.path.join(d, f"{n}.wav"))
    c[(i.samplerate, i.channels, i.subtype, i.format)] += 1
    frames.append(i.frames)
print(dict(c), f"frames min {min(frames)} max {max(frames)} total {sum(frames)}")
assert c == {(16000, 1, "PCM_16", "WAV"): 7438}, "every CREMA-D wav must be a 16 kHz mono PCM_16 WAV"
PY
(cd "$ROOT/AudioWAV" && export LC_ALL=C && sed 's/$/.wav/' "$work/names" | xargs sha256sum) > "$work/sums"
# TFDS's list: "<url ending in AudioWAV/<name>.wav> <size> <sha256>"; ours: "<sha256>  <name>.wav"
awk '$1 ~ /AudioWAV/ {n = $1; sub(/.*\/AudioWAV\//, "", n); print $3 "  " n}' "$CHECKSUMS" | LC_ALL=C sort -k2 > "$work/tfds"
if ! diff -q "$work/sums" "$work/tfds" > /dev/null; then
  echo "FAIL: $(diff "$work/sums" "$work/tfds" | grep -c '^<') wavs differ from TFDS's checksums (or the lists differ)" >&2; exit 1
fi
digest=$(sha256sum < "$work/sums" | cut -d' ' -f1)
want=e183e6db23361d1d13a307b87e08b1d31ceba20e652ab5825ae769789a2b23c3
[ "$digest" = "$want" ] || { echo "FAIL: audio tree digest $digest != $want" >&2; exit 1; }
echo "=== OK: 7438 wavs, headers, TFDS checksums and audio digest match ==="
