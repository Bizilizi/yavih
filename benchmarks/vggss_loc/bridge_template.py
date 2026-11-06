"""Bridge skeleton for **vggss_loc**. Copy, rename, fill the TODOs.

Read ``setup.md`` first (input and output contracts). ``yavih bridge check <file> --bench vggss_loc --synthetic``
smoke-tests it before any data is staged; an unfilled copy fails bridge check: E_TEMPLATE_UNFILLED /
E_CONSTANT_FEATURES. A bridge file serves one benchmark; each matrix entry (file + args) is one reading of one model.
``vggss_loc`` is TRAINING-FREE sound-source localization: the folder's scorer (``loc.py``) takes the cosine similarity between the
clip's audio embedding and each visual token of the annotated frame, both in the model's own SHARED space (the
projections a contrastive or audio-visual-correspondence objective aligned), averages it per cell of a 14 × 14 grid
over the FULL original frame and scores the map with the published VGG-SS cIoU. A model without a shared audio–visual
space has nothing to put here: do not write a bridge for it (an arbitrary pair of spaces scores at the floor).
The only reading is ``"av"`` (both streams are read, each by its own key).

MODALITY ISOLATION (``output_contract.isolation``): ``audio_emb`` may depend on the wav only, ``video_tokens`` /
``video_pos`` on the mp4's frames only. ``bridge check`` swaps one stream for another clip's and blanks the mp4's own
audio track; a key that changes fails ``E_MODALITY_LEAK``. So: no fused forward whose visual tokens attended to the
audio, no normalization over both, never decode the mp4's audio track.

PREPROCESSING SOURCE — paste the model's OWN evaluation preprocessing here, verbatim, with repo, file, function
and commit; ``describe()["input"]`` below must match it line by line (``bridge check`` warns
``W_NO_PREPROCESSING_SOURCE`` while this slot is empty):

    TODO

What ``prepare`` and ``encode`` receive per Sample (setup.md "Input contract"):
    s.streams["video"]: RawStream over the stored VGGSound mp4 — ``open()`` / ``bytes()`` / ``path``; ``info`` =
        {duration_s, fps, width, height, container, bytes}. Native size (1280×720, 480×360, 640×360, 320×240,
        some portrait clips …), ~10 s, some shorter. NOTHING is decoded, cut or cropped for you.
    s.streams["audio"]: RawStream over the stored wav, 16 kHz mono PCM; ``info`` = {duration_s, sample_rate,
        n_samples, channels, container, bytes}; some files are shorter than 10 s.
    s.meta["facts"] == {"annotated_second": 5.0}: the boxes mark the sounding object in the frame at 5.0 s.
    Nothing else (no boxes, no class, no row identity you may use: ``row``/``clip_id``/``split`` are scrambled by
    ``bridge check``).

What ``encode`` returns (``describe()["outputs"]``), per sample:
    audio_emb    [D]      the audio embedding in the shared space (the audio projection the model retrieves with);
    video_tokens [N, D]   the visual tokens of the frame at 5.0 s projected into the SAME space (one per patch);
    video_pos    [N, 2]   (y, x) centre of each token in fractions of the ORIGINAL frame: map your crop back with
                          ``yavih.media.rules.crop_region`` (``grid_pos(..., region=…)[:, 2:]``); CLS / register
                          tokens get ``-1, -1`` (dropped by the scorer) or are left out.
    used[i] = {"video": {"times": [pts of the frame fed]}, "audio": {"samples": [[lo, hi]], "pad": [l, r]}}.

Gotchas (setup.md "Gotchas"): the 14 × 14 cells tile the FULL frame — a Resize(256) + CenterCrop(224) ViT/16 covers
about half the cells on 16:9 clips (the cells it did not see score lowest; boxes outside the crop are simply hard);
the literature's Resize((224, 224)) squash covers every cell and is what the published numbers use. The scorer
refuses a feature set whose median cell coverage is below 0.25. Project BOTH sides into the space the model's
own objective aligned (the audio projection head and the visual projection head applied per token, not the
encoder's raw tokens); normalization constants from the model's own recipe; one deterministic view; keep the
model in ``.eval()``; declare precision honestly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, grid_pos

# TODO: the model's own eval transform, in the declared-rule vocabulary (yavih.media.rules.crop_region evaluates it).
# Default: the literature's VGG-SS squash (full-frame coverage, distorted aspect). Alternative: {"resize": {"short": 256},
# "crop": {"kind": "center", "size": [224, 224]}, ...}. Declare exactly what the pasted preprocessing does.
SPATIAL = {"resize": {"size": [224, 224]}, "crop": None, "interp": "bilinear", "antialias": True}
# TODO: the model's visual token grid for a 224 input (ViT-B/16 → 14 × 14).
GRID_H, GRID_W = 14, 14
# TODO: the audio span and front end of the model. Default: the whole wav, right-padded with zeros to 10 s.
AUDIO_TEMPORAL = {"rule": "full", "max_s": 10.0, "pad_to_s": 10.0}
MODEL_SAMPLE_RATE = 16000


class VGGSSLocBridge(Bridge):
    name = "TODO-model-vggss_loc-av"  # optional label of this file, recorded only; the model's name is describe()["model"]
    batch_size = 8
    dtype = "float32"  # "float16" | "float32"
    outputs = ("audio_emb", "video_tokens", "video_pos")
    ARGS = {
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap before the projection head (-1 = last)"),
    }

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)
        self.annotated = float(bench.facts["annotated_second"])  # facts come from the BenchmarkView, never the manifest
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
                    "temporal": {"rule": "explicit", "times": [self.annotated], "edge": "clamp"},
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
        frames, pts = self._decode.frames_at(v, [self.annotated])  # uint8 [1, H, W, 3], pts float64 [1]
        ranges, pad = rules.audio_ranges(AUDIO_TEMPORAL, a.info)  # int64 [k, 2] native-rate samples, (left, right) padding
        wav, sr = self._decode.audio(a, ranges)  # float32 [L, C], 16000
        # TODO video: the pasted transform exactly, /255, the model's FIXED mean/std → [3, 224, 224].
        # TODO audio: mono, zero-pad by `pad`, fbank exactly as upstream, the model's FIXED norm.
        return {"frames": frames, "pts": pts, "wav": wav, "sr": sr, "ranges": ranges, "pad": pad}

    def encode(self, samples: list) -> dict[str, list[Any]]:
        from yavih.media import rules

        out: dict[str, list[Any]] = {"audio_emb": [], "video_tokens": [], "video_pos": [], "used": []}
        for s in samples:  # instance-wise: nothing below may depend on the other samples of the batch
            v = s.streams["video"]
            p = s.prepared
            region = rules.crop_region(SPATIAL, v.info["width"], v.info["height"])  # (y0, x0, y1, x1) of the ORIGINAL frame
            # TODO: frame → visual encoder → patch tokens [GRID_H * GRID_W, D] row-major (y, x) → the visual projection
            #   head PER TOKEN (the shared space); drop CLS / registers. wav → audio encoder → pooled → the audio projection.
            _ = (p["frames"], p["wav"], p["sr"])
            out["video_tokens"].append(np.zeros((GRID_H * GRID_W, 1), dtype=self.dtype))  # placeholder
            out["audio_emb"].append(np.zeros((1,), dtype=self.dtype))  # placeholder
            out["video_pos"].append(grid_pos(1, GRID_H, GRID_W, stream=0, t_centres=[0.0], region=tuple(region))[:, 2:].astype(np.float64))
            out["used"].append({"video": {"times": [float(t) for t in p["pts"]]},
                                "audio": {"samples": [[int(lo), int(hi)] for lo, hi in p["ranges"]], "pad": [int(p["pad"][0]), int(p["pad"][1])]}})
        return out

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
