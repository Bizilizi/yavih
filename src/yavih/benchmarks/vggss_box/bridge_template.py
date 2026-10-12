"""Bridge skeleton for **vggss_box**. Copy, rename, fill the TODOs.

Read ``setup.md`` first (input and output contracts). ``yavih bridge check <file> --bench vggss_box --synthetic``
smoke-tests it before any data is staged; an unfilled copy fails bridge check: E_TEMPLATE_UNFILLED /
E_CONSTANT_FEATURES. A bridge file serves one benchmark; each matrix entry (file + args) is one reading of one model.
``vggss_box`` is audio-CONDITIONED box regression: the ``box`` readout bins the VIDEO tokens onto a
14 × 14 grid over the FULL original frame by their ``pos.(y, x)`` and cross-attends those cells into the AUDIO
tokens, so the only meaningful reading is ``"av"`` with BOTH streams tagged and spatial positions on the video
tokens. A video-only, audio-only or pooled (``N == 1``) bridge is recorded ``not_applicable`` by ``yavih features
validate`` (eligibility), not failed — there is no point writing one.

PREPROCESSING SOURCE — paste the model's OWN evaluation preprocessing here, verbatim, with repo, file, function
and commit; ``describe()["input"]`` below must match it line by line (``bridge check`` warns
``W_NO_PREPROCESSING_SOURCE`` while this slot is empty):

    TODO

What ``encode`` receives per Sample (setup.md "Input contract"):
    s.streams["video"]: RawStream over the stored VGGSound mp4 — ``open()`` / ``bytes()`` / ``path``; ``info`` =
        {duration_s, fps, width, height, container, bytes}. Native size (1280×720, 480×360, 640×360, 320×240,
        182 portrait clips …), ~10 s (221 rows shorter, min 5.09 s). NOTHING is decoded, cut or cropped for you.
    s.streams["audio"]: RawStream over the stored wav, 16 kHz mono PCM; ``info`` = {duration_s, sample_rate,
        n_samples, channels, container, bytes}; 487 rows are shorter than 10 s (min 4.94 s).
    s.meta["facts"] == {"annotated_second": 5.0}: the box marks the sounding object in the frame at 5.0 s.
    Nothing else (no box, no class, no row identity you may use: ``row``/``clip_id``/``split`` are scrambled by
    ``bridge check``).

Every model-side choice is yours and is declared in ``describe()["input"]`` (recorded): which frames (here
``explicit`` times with one frame on the annotated second), spatial rule (here the common Resize(256) +
CenterCrop(224)), audio span and front end, normalization constants with their source, decoder.

What ``encode`` returns: ``(feats [B, N, D], pos [B, N, 4], used)``.
    pos is in MEDIA coordinates: ``t`` = media seconds / ``info.duration_s``; ``(y, x)`` = fractions of
    the ORIGINAL frame — map your crop back with ``crop_region`` (``grid_pos(..., region=…)`` does it). Frame
    sizes and durations differ per clip, hence ``pos_mode = "per_sample"``.
    used[i] = {"video": {"times": [pts of the frames fed]}, "audio": {"samples": [[lo, hi]], "pad": [l, r]}};
    validate re-evaluates the declared rules on the member's media facts (``E_RULE_MISMATCH``,
    ``E_TIMES_OUT_OF_MEDIA``).

Gotchas (setup.md "Gotchas"): the 14 × 14 cells tile the FULL frame — a Resize(256)+CenterCrop(224) ViT/16 covers
96 / 196 cells on the 61 % 16:9 clips (0.49) and 120 on 4:3 (0.61), above ``min_cell_coverage 0.25``; a 7 × 7
token grid inside a centre crop is not; boxes outside what the model saw are simply hard for it. Drop CLS /
register tokens or give them ``y = x = -1``. Normalization constants from the model's own pretraining recipe
(per-clip statistics only when they ARE the model's own recipe, declared in ``norm`` with their ``source``; never
batch or dataset statistics); never pad to the batch maximum; one deterministic view; keep the model in ``.eval()``; declare precision honestly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, grid_pos

# TODO: frames the model consumes. Default: 8 frames 1 s apart with frame 4 on the annotated second
# (5 + (k − 4) · 1 s → 1.0 … 8.0 s); 16 eval clips are shorter than 8.0 s (min 5.09 s), hence edge "clamp".
N_FRAMES = 8
FRAME_STEP_S = 1.0
# TODO: the model's own eval transform, in the declared-rule vocabulary (yavih.media.rules.crop_region evaluates it).
# Alternative: {"resize": {"size": [224, 224]}, "crop": None, ...} (the literature's VGG-SS squash: full-frame
# coverage, distorted aspect). Declare exactly what the pasted preprocessing does.
SPATIAL = {"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}
# TODO: the model's visual token grid per frame (ViT-B/16 at 224 → 14 × 14) and temporal tubelet (1 or 2).
GRID_H, GRID_W = 14, 14
TUBELET = 1
# TODO: the audio span and front end of the model. Default: the whole wav, right-padded with zeros to 10 s.
AUDIO_TEMPORAL = {"rule": "full", "max_s": 10.0, "pad_to_s": 10.0}
AUDIO_SECONDS = 10.0
MODEL_SAMPLE_RATE = 16000
# TODO: the audio token layout the model emits (T_a time bins over AUDIO_SECONDS; frequency bins pooled or kept).
AUDIO_T = 8


class VGGSSBoxBridge(Bridge):
    name = "TODO-model-vggss_box-av"  # optional label of this file, recorded only; the model's name is describe()["model"]
    batch_size = 8
    dtype = "float16"  # "float16" | "float32"
    pos_mode = "per_sample"  # frame sizes and durations differ per clip → positions differ per clip
    streams = ("video", "audio")  # tag order = stream index in pos[:, 0]; BOTH are required (output_contract.requires.streams)
    ARGS = {
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap (-1 = last)"),
    }

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)
        # Facts come from the BenchmarkView, never from the manifest.
        annotated = float(bench.facts["annotated_second"])
        self.frame_times = [annotated + (k - N_FRAMES // 2) * FRAME_STEP_S for k in range(N_FRAMES)]
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
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": "av",
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.args["layer"],
            "pooling": "none",  # tokens, never pooled: the head needs the spatial grid
            "views": 1,
            "input": {
                "video": {
                    "temporal": {"rule": "explicit", "times": list(self.frame_times), "edge": "clamp"},
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

    def encode(self, samples: list) -> tuple[Any, Any, list[dict]]:
        from yavih.media import rules

        feats, pos, used = [], [], []
        for s in samples:  # instance-wise: nothing below may depend on the other samples of the batch
            v, a = s.streams["video"], s.streams["audio"]
            frames, pts = self._decode.frames_at(v, self.frame_times)  # uint8 [N_FRAMES, H, W, 3], pts float64 [N_FRAMES]
            region = rules.crop_region(SPATIAL, v.info["width"], v.info["height"])  # (y0, x0, y1, x1) of the ORIGINAL frame
            ranges, pad = rules.audio_ranges(AUDIO_TEMPORAL, a.info)  # int64 [k, 2] native-rate samples, (left, right) padding
            wav, sr = self._decode.audio(a, ranges)  # float32 [L, C], 16000
            # TODO video: the pasted transform exactly (e.g. torchvision Resize(256, antialias=True) +
            #   CenterCrop(224)), /255, the model's FIXED mean/std, [T, 3, 224, 224] → model → patch tokens
            #   [T // TUBELET, GRID_H, GRID_W, D] flattened row-major (t, y, x); drop CLS/registers.
            # TODO audio: mono, zero-pad by `pad`, fbank exactly as upstream, the model's FIXED norm → audio tokens [AUDIO_T, D].
            _ = (frames, wav, sr)
            vis = np.zeros(((N_FRAMES // TUBELET) * GRID_H * GRID_W, 1), dtype=np.float32)  # placeholder
            aud = np.zeros((AUDIO_T, 1), dtype=np.float32)  # placeholder
            feats.append(np.concatenate([vis, aud], axis=0))
            pos.append(self.positions(pts, v.info, a.info, region))
            used.append({
                "video": {"times": [float(t) for t in pts]},
                "audio": {"samples": [[int(lo), int(hi)] for lo, hi in ranges], "pad": [int(pad[0]), int(pad[1])]},
            })
        return np.stack(feats).astype(self.dtype), np.stack(pos), used  # [B, N, D] with the same N for every sample

    @staticmethod
    def positions(pts: Any, video_info: dict, audio_info: dict, region: tuple[float, float, float, float]) -> np.ndarray:
        """Token positions ``[N, 4] = (stream, t, y, x)`` in media coordinates.

        Video (stream 0): ``t`` = each token slab's media time / ``duration_s`` (the mean over a tubelet's frames);
        ``(y, x)`` = patch centres mapped into ``region``, the part of the ORIGINAL frame the model input shows.
        Audio (stream 1): ``AUDIO_T`` bins over ``AUDIO_SECONDS`` of fed signal; ``t`` = bin centre / ``duration_s``,
        ``-1`` for a bin centred in the zero padding beyond the file; ``y = x = -1`` (audio tokens are never binned
        into cells, every audio-tagged token is a conditioning row).
        """
        frac = np.asarray(pts, dtype=np.float64) / float(video_info["duration_s"])
        if TUBELET > 1:
            frac = frac.reshape(-1, TUBELET).mean(axis=1)
        video = grid_pos(len(frac), GRID_H, GRID_W, stream=0, t_centres=frac.tolist(), region=tuple(region))
        centres = (np.arange(AUDIO_T) + 0.5) * AUDIO_SECONDS / AUDIO_T
        dur = float(audio_info["duration_s"])
        t_a = np.where(centres <= dur, centres / dur, -1.0)
        audio = np.stack([np.ones(AUDIO_T), t_a, -np.ones(AUDIO_T), -np.ones(AUDIO_T)], axis=1)
        return np.concatenate([video, audio], axis=0).astype(np.float32)

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
