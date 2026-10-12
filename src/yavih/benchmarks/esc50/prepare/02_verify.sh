#!/usr/bin/env bash
# ESC-50 step 2/3 — verify the staged clone: 2 000 wavs, every one 44.1 kHz mono PCM_16 of exactly
# 220 500 frames (5.000 s; the facts input_contract.info promises: duration_s 5.0, sample_rate 44100,
# n_samples 220500, channels 1), and the pinned digest of the whole audio tree.
#
#   bash 02_verify.sh   # the clone at $(yavih config get esc50.media_root)
#   YAVIH_DATA__ESC50__MEDIA_ROOT=/path/to/datasets/esc50 bash 02_verify.sh
#
# The digest is sha256 of `cd audio && export LC_ALL=C && sha256sum *.wav` (2 000 lines, C-locale glob
# order; the export must precede the glob, which the calling shell expands under its own LC_COLLATE;
# benchmark.yaml source.sha256["audio.sha256sum"]). Reading 882 MB takes a few minutes on NFS. Needs python
# with soundfile for the header check (the yavih [media] extra has it).
set -euo pipefail
ROOT="$(yavih config get esc50.media_root)"
PY=${PY:-python}
n=$(find "$ROOT/audio" -maxdepth 1 -name '*.wav' | wc -l)
[ "$n" -eq 2000 ] || { echo "FAIL: $n wavs in $ROOT/audio, expected 2000" >&2; exit 1; }
"$PY" -I - "$ROOT/audio" <<'PY'
import os, sys, collections
import soundfile as sf
d = sys.argv[1]
c = collections.Counter()
for n in sorted(os.listdir(d)):
    if n.endswith(".wav"):
        i = sf.info(os.path.join(d, n)); c[(i.frames, i.samplerate, i.channels, i.subtype)] += 1
print(dict(c))
assert c == {(220500, 44100, 1, "PCM_16"): 2000}, "every ESC-50 wav must be 220500 frames, 44.1 kHz, mono, PCM_16"
PY
digest=$(cd "$ROOT/audio" && export LC_ALL=C && sha256sum *.wav | sha256sum | cut -d' ' -f1)
want=52754b9099c1c48a9fec379bd7dd75dbc6d78387faecc2a9318fc0c43332d398
[ "$digest" = "$want" ] || { echo "FAIL: audio tree digest $digest != $want" >&2; exit 1; }
echo "=== OK: 2000 wavs, headers and audio digest match ==="
