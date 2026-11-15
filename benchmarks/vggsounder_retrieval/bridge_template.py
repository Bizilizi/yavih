"""Bridge skeleton for **vggsound_retrieval**. Copy, rename, fill the TODOs.

Read ``setup.md`` first (input and output contracts). ``yavih bridge check <file> --bench vggsound_retrieval
--synthetic`` smoke-tests it before any data is staged; an unfilled copy fails bridge check: E_TEMPLATE_UNFILLED /
E_CONSTANT_FEATURES. A bridge file serves one benchmark; each matrix entry (file + args) is one reading of one model.
``vggsound_retrieval`` is TRAINING-FREE cross-modal retrieval: the folder's scorer (``retrieval.py``) ranks every clip's audio
against every clip's video (and back) by cosine similarity in the model's own SHARED space — the vectors a
contrastive (or other audio-visual-correspondence) objective aligned, after their projection heads. A model without
such a space has nothing to put here: do not write a bridge for it (two unrelated spaces score at chance).
The only reading is ``"av"`` (both streams are read, each by its own key).

MODALITY ISOLATION (``output_contract.isolation``): ``audio_emb`` may depend on the wav only, ``video_emb`` on the
mp4's frames only. ``bridge check`` swaps one stream for another clip's and blanks the mp4's own audio track; a key
that changes fails ``E_MODALITY_LEAK``. So: no fused forward, no normalization over both, never decode the mp4's
audio track; the two embeddings of a clip come from two independent single-modality forwards.

PREPROCESSING SOURCE — paste the model's OWN evaluation preprocessing here, verbatim, with repo, file, function
and commit; ``describe()["input"]`` below must match it line by line (``bridge check`` warns
``W_NO_PREPROCESSING_SOURCE`` while this slot is empty):

    TODO

What ``prepare`` and ``encode`` receive per Sample (setup.md "Input contract"):
    s.streams["video"]: RawStream over the stored VGGSound mp4 — ``open()`` / ``bytes()`` / ``path``; ``info`` =
        {duration_s, fps, width, height, container, bytes}. Native size, mixed; ~10 s. NOTHING is decoded for you.
    s.streams["audio"]: RawStream over the stored wav, 16 kHz mono PCM; ``info`` = {duration_s, sample_rate,
        n_samples, channels, container, bytes}; some files are shorter than 10 s.
    Nothing else (no class, no row identity you may use: ``row``/``clip_id``/``split`` are scrambled by ``bridge check``).

What ``encode`` returns (``describe()["outputs"]``), per sample: ``audio_emb [D]`` and ``video_emb [D]``, the two
projections of one clip (un-normalized is fine: the scorer L2-normalizes); ``used[i]`` reports the frames fed and the
audio range.

Gotchas (setup.md "Gotchas"): take the vectors the model's OWN retrieval evaluation uses (the projection heads'
outputs, pooled as the model pools them for its contrastive loss; CAV-MAE's are its post-projection means, not the
encoder tokens); one deterministic view (the model's own evaluation frames and audio window); normalization constants
from the model's own recipe; keep the model in ``.eval()``; declare precision honestly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge

# TODO: frames the model consumes. Default: 10 frames 1 s apart (0.5 … 9.5 s); edge "clamp" for the shorter clips.
N_FRAMES = 10
FRAME_TIMES = [0.5 + k for k in range(N_FRAMES)]
# TODO: the model's own eval transform, in the declared-rule vocabulary (yavih.media.rules.crop_region evaluates it).
SPATIAL = {"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}
# TODO: the audio span and front end of the model. Default: the whole wav, right-padded with zeros to 10 s.
AUDIO_TEMPORAL = {"rule": "full", "max_s": 10.0, "pad_to_s": 10.0}
MODEL_SAMPLE_RATE = 16000


class VGGSoundRetrievalBridge(Bridge):
    name = "TODO-model-vggsound_retrieval-av"  # optional label of this file, recorded only; the model's name is describe()["model"]
    batch_size = 8
    dtype = "float32"  # "float16" | "float32"
    outputs = ("audio_emb", "video_emb")
    ARGS = {
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap before the projection head (-1 = last)"),
    }

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)
        from yavih.media import (
            decode as ymd,  # [media] extra; or decode with the model's own stack and declare it
        )

        self._decode = ymd
        self._decoders = {"video": ymd.decoder_info("video"), "audio": ymd.decoder_info("audio")}  # {lib, version, libav}
        # TODO: build the model, load weights BEFORE the first sample (no network during encode), call .eval():
        # import torch
        # from yavih.bridge import fingerprint_state_dict
        # self.model = build_model(...).eval().to("cuda")
        # self._weights_sha256 = fingerprint_state_dict(self.model)
        self.model = None
        self._weights_sha256 = None  # TODO: real fingerprint; None is recorded as W_WEIGHTS_UNFINGERPRINTED

    def describe(self) -> dict:
        # Every key below is REQUIRED; extra keys are allowed and recorded (never identity).
        return {
            "weights_sha256": self._weights_sha256,
            "weights_ref": "TODO: path or hub id (provenance only, not hashed)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "reading": "av",
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.args["layer"],
            "views": 1,
            "outputs": list(self.outputs),
            "input": {
                "video": {
                    "temporal": {"rule": "explicit", "times": list(FRAME_TIMES), "edge": "clamp"},
                    "spatial": SPATIAL,
                    "norm": {"scale": 1 / 255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],  # TODO
                             "source": "TODO: <repo>@<commit> <file> (the model's pretraining recipe)"},
                    "decoder": self._decoders["video"],
                },
                "audio": {
                    "temporal": AUDIO_TEMPORAL,
                    "front_end": {"sample_rate": MODEL_SAMPLE_RATE, "resample": "none (stored at 16 kHz)",  # TODO
                                  "features": "kaldi_fbank", "n_mels": 128, "win_ms": 25, "hop_ms": 10, "dither": 0.0,
                                  "channels": "mono"},
                    "norm": {"mean": -4.268, "std": 4.569, "source": "TODO: <repo>@<commit> <file>"},  # TODO
                    "decoder": self._decoders["audio"],
                },
            },
            # DECLARE what you run with; run_extract enforces it (E_PRECISION_UNDECLARED on a null).
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),  # echoes every consumed --arg with defaults filled
        }

    def prepare(self, sample: Any) -> dict:
        """One clip's CPU work, ahead of ``encode``: what this returns arrives as ``sample.prepared``."""
        from yavih.media import rules

        v, a = sample.streams["video"], sample.streams["audio"]
        frames, pts = self._decode.frames_at(v, FRAME_TIMES)  # uint8 [N_FRAMES, H, W, 3], pts float64 [N_FRAMES]
        ranges, pad = rules.audio_ranges(AUDIO_TEMPORAL, a.info)  # int64 [k, 2] native-rate samples, (left, right) padding
        wav, sr = self._decode.audio(a, ranges)  # float32 [L, C], 16000
        # TODO video: the pasted transform exactly, /255, the model's FIXED mean/std → [T, 3, 224, 224].
        # TODO audio: mono, zero-pad by `pad`, fbank exactly as upstream, the model's FIXED norm.
        return {"frames": frames, "pts": pts, "wav": wav, "sr": sr, "ranges": ranges, "pad": pad}

    def encode(self, samples: list) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = {"audio_emb": [], "video_emb": [], "used": []}
        for s in samples:  # instance-wise: nothing below may depend on the other samples of the batch
            p = s.prepared
            # TODO: frames → visual encoder → the visual projection, pooled as for the contrastive loss → [D];
            #       wav → audio encoder → the audio projection, pooled the same way → [D]. Two separate forwards.
            _ = (p["frames"], p["wav"], p["sr"])
            out["video_emb"].append(np.zeros((1,), dtype=self.dtype))  # placeholder
            out["audio_emb"].append(np.zeros((1,), dtype=self.dtype))  # placeholder
            out["used"].append({"video": {"times": [float(t) for t in p["pts"]]},
                                "audio": {"samples": [[int(lo), int(hi)] for lo, hi in p["ranges"]], "pad": [int(p["pad"][0]), int(p["pad"][1])]}})
        return out

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
