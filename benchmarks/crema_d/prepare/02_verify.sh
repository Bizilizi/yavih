#!/usr/bin/env bash
# CREMA-D step 2/3 — verify the staged wavs: the 7 438 the manifest lists are present, every one is a 16 kHz mono
# PCM_16 RIFF wav (the facts input_contract.info promises: sample_rate 16000, channels 1; the length varies), and
# the digest of the whole audio tree matches.
#
#   yavih bench prepare crema_d 02     # the clone at $(yavih config get crema_d.media_root); ROOT=/path overrides
#
# The digest is the sha256 of `cd AudioWAV && export LC_ALL=C && sha256sum <the 7 438 names, sorted>.wav`
# (benchmark.yaml source.sha256["AudioWAV.sha256sum"]); the file list is the manifest's, so the four wavs TFDS
# skips (one of them, 1040_ITH_SAD_XX, was added upstream in 2022) never enter it. Needs python with numpy and
# soundfile (the yavih [media] extra). Reading 0.6 GB takes a few minutes on a network file system.
set -euo pipefail
ROOT="${ROOT:-$(yavih config get crema_d.media_root)}"
PY=${PY:-python}
FOLDER="${FOLDER:-$(cd "$(dirname "$0")/.." && pwd)}"
names=$(mktemp)
trap 'rm -f "$names"' EXIT
zcat "$FOLDER/manifest.csv.gz" | tail -n +2 | cut -d, -f1 | LC_ALL=C sort > "$names"
[ "$(wc -l < "$names")" -eq 7438 ] || { echo "FAIL: the manifest lists $(wc -l < "$names") clips, expected 7438" >&2; exit 1; }
"$PY" -I - "$ROOT/AudioWAV" "$names" <<'PY'
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
digest=$(cd "$ROOT/AudioWAV" && export LC_ALL=C && sed 's/$/.wav/' "$names" | xargs sha256sum | sha256sum | cut -d' ' -f1)
want=TODO_AUDIO_DIGEST
[ "$digest" = "$want" ] || { echo "FAIL: audio tree digest $digest != $want" >&2; exit 1; }
echo "=== OK: 7438 wavs, headers and audio digest match ==="
