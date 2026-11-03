"""Bridge skeleton for **audiocaps_t2a** (text-to-audio generation). Copy, rename, fill the TODOs.

Read ``setup.md`` first: its input contract is what ``encode`` receives, its output contract what must come back.
``yavih bridge check <file> --bench audiocaps_t2a --synthetic`` smoke-tests it before any data is staged; an unfilled
copy fails bridge check (E_TEMPLATE_UNFILLED).

Preprocessing source (paste the model's own inference pipeline here, verbatim, with repo, file, function and
commit: the text encoder, its tokenizer and any prompt template):

    TODO

What ``encode`` receives per Sample:
    s.cells["caption"]  str: one human caption of the clip, verbatim (no file is handed over).
    s.meta["facts"]     {"audio_length": 10.0}: the seconds of audio generated and scored per caption.

What comes back (``encode`` returns a dict, one entry per key of ``describe()["outputs"]``):
    "audio"   one wav file's bytes per sample (``yavih.codecs.wav_bytes(samples, sample_rate)``): MONO, exactly
              ``audio_length`` seconds (10.0 to 10.05 s). Any sample rate: the scorer resamples for every encoder.
    "used"    per sample, ``{}`` (a caption has no frames or samples to record).

Generation is part of the protocol: ``describe()["sampling"]`` declares the seed, the number of steps, the guidance
scale, the candidates per caption and the rule that picks one (``"first"`` for one candidate). The generator must be a
pure function of the caption and that seed: ``bridge check`` reruns every caption and compares the audio (exact when
``precision.deterministic`` is true), and re-encodes captions alone and in shuffled batches.
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
    streams = ()

    def describe(self) -> dict:
        return {
            "model": "TODO-model",  # the model's NAME, shared by its bridges
            "reading": "t",  # the model reads text only
            "weights_sha256": None,  # TODO: yavih.bridge.fingerprint_state_dict(model)
            "weights_ref": "TODO",
            "batch_size": self.batch_size,
            "views": 1,
            "input": {"caption": {"note": "verbatim, no prompt template"}},  # TODO: the prompt template, if any
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),
            "outputs": list(self.outputs),
            "sampling": dict(SAMPLING),
        }

    def encode(self, samples: list[Any]) -> dict[str, Any]:
        audio = []
        for s in samples:
            length = float(s.meta["facts"]["audio_length"])
            # TODO: run the generator on s.cells["caption"] with SAMPLING["seed"], downmix to mono
            x = np.zeros(int(round(length * SAMPLE_RATE)), dtype=np.float32)
            audio.append(codecs.wav_bytes(x, SAMPLE_RATE))
        return {"audio": audio, "used": [{} for _ in samples]}
