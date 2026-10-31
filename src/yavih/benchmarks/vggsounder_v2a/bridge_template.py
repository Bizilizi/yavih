"""Bridge skeleton for **vggsounder_v2a** (video-to-audio generation). Copy, rename, fill the TODOs.

Read ``setup.md`` first: its input contract is what ``prepare`` and ``encode`` receive, its output contract what must
come back. ``yavih bridge check <file> --bench vggsounder_v2a --synthetic`` smoke-tests it before any data is staged;
an unfilled copy fails bridge check (E_TEMPLATE_UNFILLED).

Preprocessing source (paste the model's own inference pipeline here, verbatim, with repo, file, function and
commit; ``describe()["input"]`` must match it; ``bridge check`` warns W_NO_PREPROCESSING_SOURCE while it is empty):

    TODO

What ``prepare`` and ``encode`` receive per Sample:
    s.streams["video"]  RawStream of the MUTED mp4 (the stored VGGSound video stream, copied without re-encoding;
                        no audio track). ``.path`` / ``.open()`` / ``.bytes()``; ``.info`` = {duration_s, fps, width,
                        height, container, bytes}. 8 to ~10 s at 25-30 fps.
    s.meta["facts"]     {"audio_length": 8.0}: the seconds of audio scored per clip.

What comes back (``encode`` returns a dict, one entry per key of ``describe()["outputs"]``):
    "audio"   one wav file's bytes per sample (``yavih.codecs.wav_bytes(samples, sample_rate)``): MONO, the first
              ``audio_length`` seconds of the clip (8.0 to 8.05 s; every clip is at least that long). Any sample
              rate: the scorer resamples for every encoder.
    "used"    per sample, the frame times actually fed (``{"video": {"times": [...]}}``).

Generation is part of the protocol: ``describe()["sampling"]`` declares the seed, the number of steps, the guidance
scale, the candidates per clip and the rule that picks one (``"first"`` for one candidate). The generator must be a
pure function of the clip and that seed: ``bridge check`` reruns every clip and compares the audio (exact when
``precision.deterministic`` is true), and re-encodes clips alone and in shuffled batches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih import codecs
from yavih.bridge import Bridge

SAMPLE_RATE = 16000  # TODO: the generator's output rate
SAMPLING = {"seed": 0, "steps": 25, "guidance": 4.5, "n_candidates": 1, "selection": "first"}  # TODO: the evaluated budget


class TemplateBridge(Bridge):
    batch_size = 4
    outputs = ("audio",)

    def describe(self) -> dict:
        return {
            "model": "TODO-model",  # the model's NAME, shared by its bridges
            "reading": "v",  # the model sees the (muted) video
            "weights_sha256": None,  # TODO: yavih.bridge.fingerprint_state_dict(model)
            "weights_ref": "TODO",
            "batch_size": self.batch_size,
            "views": 1,
            "input": {
                "video": {
                    "temporal": {"rule": "custom", "description": "TODO: the frames the generator reads (rate, window)"},
                    "spatial": {"rule": "custom", "description": "TODO: resize / crop"},
                    "norm": {"source": "TODO"},
                    "decoder": {"lib": "TODO", "version": "TODO"},
                }
            },
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),
            "outputs": list(self.outputs),
            "sampling": dict(SAMPLING),
        }

    def encode(self, samples: list[Any]) -> dict[str, Any]:
        audio, used = [], []
        for s in samples:
            length = float(s.meta["facts"]["audio_length"])
            # TODO: decode the frames, run the generator with SAMPLING["seed"], downmix to mono
            x = np.zeros(int(round(length * SAMPLE_RATE)), dtype=np.float32)
            audio.append(codecs.wav_bytes(x, SAMPLE_RATE))
            used.append({"video": {"times": []}})
        return {"audio": audio, "used": used}
