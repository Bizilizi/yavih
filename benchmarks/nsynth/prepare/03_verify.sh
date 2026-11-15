#!/usr/bin/env bash
# nsynth step 3/4 — verify the staged media: the three examples.json files (pinned sha256), the 5 000 wavs the
# manifest lists (4 000 train, 500 valid, 500 test), every one 16 kHz mono PCM_16 of exactly 64 000 frames
# (4.000 s; the facts input_contract.info promises), and the pinned digest of the whole staged wav set.
#
#   yavih bench prepare nsynth 03                       # media root = $(yavih config get nsynth.media_root)
#   ROOT=/path/to/datasets/nsynth PY=python yavih bench prepare nsynth 03
#
# The digest is sha256 of `cd <root> && export LC_ALL=C && sha256sum nsynth-*/audio/*.wav` (5 000 lines, C-locale
# glob order; benchmark.yaml source.sha256["audio.sha256sum"]). Needs python with soundfile (the yavih [media] extra).
set -euo pipefail
ROOT="${ROOT:-$(yavih config get nsynth.media_root)}"
PY=${PY:-python}
cd "$ROOT"
sha256sum -c - <<SUMS
a8182e084aa338d764f64f6a283bd1778d358e913a7074294110e9ec601e0087  nsynth-train/examples.json
050e0bf55d1a87eee2abbc3d273c7f525404c0a51ac378561b905f2da31a7286  nsynth-valid/examples.json
74df2dd960c156cd5e8757d8afb1fe30e6a233aa99043a29037e7b54cde6908f  nsynth-test/examples.json
SUMS
for s in train:4000 valid:500 test:500; do
  n=$(find "nsynth-${s%%:*}/audio" -maxdepth 1 -name '*.wav' | wc -l)
  [ "$n" -eq "${s##*:}" ] || { echo "FAIL: $n wavs in nsynth-${s%%:*}/audio, expected ${s##*:}" >&2; exit 1; }
done
"$PY" -I - "$ROOT" <<'PY'
import collections, glob, os, sys
import soundfile as sf
c = collections.Counter()
for p in sorted(glob.glob(os.path.join(sys.argv[1], "nsynth-*", "audio", "*.wav"))):
    i = sf.info(p); c[(i.frames, i.samplerate, i.channels, i.subtype)] += 1
print(dict(c))
assert c == {(64000, 16000, 1, "PCM_16"): 5000}, "every staged NSynth wav must be 64000 frames, 16 kHz, mono, PCM_16"
PY
digest=$(export LC_ALL=C && sha256sum nsynth-*/audio/*.wav | sha256sum | cut -d' ' -f1)
want=d4e890c1dc76f7dbecc13ebfe7f7274cabcde8578c4c15b892d6a60a0794703a
[ "$digest" = "$want" ] || { echo "FAIL: audio digest $digest != $want" >&2; exit 1; }
echo "=== OK: examples.json, 5000 wavs, headers and audio digest match ==="
