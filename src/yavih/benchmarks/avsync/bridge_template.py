"""Bridge skeleton for **avsync**. Copy, rename, fill the TODOs. Read ``setup.md`` ("Input contract", "Output
contract", "Gotchas") first. ``yavih bridge check <file> --bench avsync --synthetic`` smoke-tests it before any data
is staged; an unfilled copy fails bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES. A bridge file serves one
benchmark; each matrix entry (file + args) is one reading of one model.

The label of a row is the temporal offset δ ∈ {-2.0, -1.5, …, +2.0} s between the two tracks of ONE mp4: at
prepare time the benchmark cut the VGGSound video over ``[1 + δ/2, 9 + δ/2]`` and the audio over ``[1 − δ/2,
9 − δ/2]`` and muxed them (``input_contract.materialize``), so inside the file "video leads audio by δ". δ is never
handed to you. The benchmark is defined for the JOINT reading (``reading: "av"``: both streams, one feature set
that sees both). Audio-only / video-only readings are legal and are CONTROLS: they must sit at chance (0.111);
above chance they found a cue that is not synchronization.

PREPROCESSING SOURCE (paste the model's own eval-time preprocessing here, verbatim; describe() must match it line
by line — ``bridge check`` warns W_NO_PREPROCESSING_SOURCE while this slot holds only TODO):
    TODO: repo / file / function / commit of the video transform (frame sampling, resize, crop, normalization)
    TODO: repo / file / function / commit of the audio front end (sample rate, resampler, fbank/mel, normalization)

What ``encode`` receives per Sample: ``s.streams["video"]`` and ``s.streams["audio"]``, two RawStreams
of the same materialized mp4 (``path``, ``bytes()``, ``open()``), and ``s.meta == {"facts": {}}``.
    video ``info``: duration_s (≈ 8.0), fps (6–60, mostly 30), width, height (native, mixed: ≈ half 1280x720, the
        rest 640x360 / 320x240, a few portrait), container "mp4", bytes. H.264, yuv420p.
    audio ``info``: duration_s 8.0, sample_rate 16000, n_samples 128000, channels 1, container "mp4", bytes. ALAC
        (lossless; decodes to int16, bit-identical to the source wav).
Decode with the model's own stack (PyAV / torchaudio / decord on ``open()`` or ``path``) or with the torch-free
helpers ``yavih.media.decode`` (``frames_at``, ``audio``, ``decoder_info``); both are fine, declare which.

Gotchas (setup.md "Gotchas"), each passes validation silently:
  1. Take frame TIMES from your declared rule, not from the decoded pts, when you build ``pos``: the first frame of
     the materialized video sits at (source pts − window start), i.e. 0 or a fraction of a frame depending on
     δ mod (1/fps) — at 30 fps, whole-second vs half-second offsets. Rule times (``rule_times``) are identical for
     every row; pts-derived ``t`` would carry that parity into the features. (``used`` reports the decoded pts.)
  2. Use ``span: [0.0, 8.0]`` (absolute) rather than ``span: "full"``: a few source clips end before 10 s, so a
     δ = +2 row can have a video a frame shorter than 8 s; "full" scales your times by that duration.
  3. Normalization constants from the model's own pretraining recipe; per-clip statistics only when they ARE the
     model's own recipe (then declare them in ``norm`` with their ``source``); never batch or dataset statistics.
  4. No caches keyed on path, clip_id or content: every row is its own file, and ``bridge check`` scrambles
     ``row``/``clip_id``/``split``. ``clip_id`` ends in ``#<bin>`` = the label: never read it.
  5. Never pad, trim or re-centre one stream against the other; read both tracks over the same span.
  6. One feature set for "av": concatenate the video and audio token sets and tag streams in ``pos``; two pooled
     vectors averaged into one lose the cross-stream timing the probe needs. Same ``N`` for every sample.
  7. 1 fps (n = 8 frames over 8 s) cannot resolve a 0.5 s grid; n = 16 (2 fps) is the reference geometry.
     A 1-fps model reports ``acc_coarse5`` as its comparable number.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, grid_pos

# TODO: frames the model consumes over the 8 s file. 16 = 2 fps (0.5 s bucket centres: the only geometry that
# resolves the 0.5 s offset grid); 8 = 1 fps.
N_FRAMES = 16
TUBELET = 2  # TODO: temporal patch size of the video encoder (1 for frame-wise encoders)
VIDEO_H, VIDEO_W = 14, 14  # TODO: patch grid of the model input (224 / 16)
AUDIO_T, AUDIO_F = 16, 8  # TODO: the audio encoder's token grid over the 8 s (time bins x frequency bins)
SPAN = [0.0, 8.0]  # absolute media seconds; every materialized file holds [0, 8)

VIDEO_TEMPORAL = {"rule": "uniform", "n": N_FRAMES, "span": SPAN, "edge": "error"}
VIDEO_SPATIAL = {"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}
AUDIO_TEMPORAL = {"rule": "span", "start_s": 0.0, "length_s": 8.0, "pad": "none"}  # every file is exactly 128 000 samples


def _rule_times(temporal: dict, info: dict) -> np.ndarray:
    """The declared rule's times. ``yavih.media.rules.rule_times`` when available; the inline fallback
    implements ``uniform`` over an absolute span, which is all this template declares."""
    try:
        from yavih.media.rules import rule_times

        return np.asarray(rule_times(temporal, info), dtype=np.float64)
    except ImportError:
        a, b = temporal["span"]
        n = temporal["n"]
        return a + (np.arange(n) + 0.5) * (b - a) / n


def _crop_region(spatial: dict, width: int, height: int) -> tuple[float, float, float, float]:
    """The part of the ORIGINAL frame the model input shows, ``(y0, x0, y1, x1)`` in fractions."""
    try:
        from yavih.media.rules import crop_region

        return tuple(float(v) for v in crop_region(spatial, width, height))
    except ImportError:  # resize-short + centre-crop, inline
        s = spatial["resize"]["short"] / min(width, height)
        ch, cw = spatial["crop"]["size"]
        rh, rw = ch / (height * s), cw / (width * s)
        return ((1 - rh) / 2, (1 - rw) / 2, (1 + rh) / 2, (1 + rw) / 2)


class AVSyncBridge(Bridge):
    name = "TODO-model-avsync-av"  # optional label of this file, recorded only; the model's name is describe()["model"]
    batch_size = 8
    dtype = "float16"  # "float16" | "float32"
    pos_mode = "per_sample"  # frame sizes differ between rows, so the crop region (y, x) differs: per_sample
    streams = ("video", "audio")  # tag order = stream index in pos[:, 0]
    ARGS = {
        "reading": ArgSpec(type=str, default="av", choices=("av", "a", "v"), help="av = the benchmark; a / v = controls"),
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap (-1 = last)"),
    }

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)  # bench is a BenchmarkView: no manifest, labels or splits
        self.reading = self.args["reading"]
        self.streams = {"av": ("video", "audio"), "a": ("audio",), "v": ("video",)}[self.reading]
        # TODO: build the model, load weights BEFORE the first sample (no network during encode), call .eval():
        # import torch
        # self.model = build_model(...).eval().to("cuda")
        # self._weights_sha256 = fingerprint_state_dict(self.model)   # yavih.bridge.fingerprint_state_dict
        self.model = None
        self._weights_sha256 = None  # TODO: the real fingerprint; None is recorded as W_WEIGHTS_UNFINGERPRINTED

    def _decoder(self) -> dict:
        try:
            from yavih.media.decode import decoder_info

            return dict(decoder_info())
        except ImportError:
            return {"lib": "TODO", "version": "TODO"}  # TODO: the library you decode with, as it reports itself

    def describe(self) -> dict:
        # Every key below is REQUIRED; extra keys are allowed and recorded (never identity).
        inp: dict = {}
        if "video" in self.streams:
            inp["video"] = {
                "temporal": VIDEO_TEMPORAL,
                "spatial": VIDEO_SPATIAL,
                "norm": {
                    "scale": 1 / 255,
                    "mean": [0.485, 0.456, 0.406],  # TODO: the model's own constants
                    "std": [0.229, 0.224, 0.225],
                    "source": "TODO: <org>/<repo>@<commit> <file> (the model's pretraining recipe)",
                },
                "decoder": self._decoder(),
            }
        if "audio" in self.streams:
            inp["audio"] = {
                "temporal": AUDIO_TEMPORAL,
                "front_end": {  # TODO: exactly the model's front end
                    "sample_rate": 16000,
                    "resample": "none (native 16 kHz)",
                    "features": "kaldi_fbank",
                    "n_mels": 128,
                    "win_ms": 25,
                    "hop_ms": 10,
                    "dither": 0.0,
                    "channels": "mono",
                },
                "norm": {"mean": -4.268, "std": 4.569, "source": "TODO: <org>/<repo>@<commit> <file> (pretraining recipe constants)"},
                "decoder": self._decoder(),
            }
        return {
            "weights_sha256": self._weights_sha256,
            "weights_ref": "TODO: path or hub id (provenance only)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": self.reading,  # "av" is the benchmark; "a" / "v" are controls that must sit at chance
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.args["layer"],
            "pooling": "none",  # "none" keeps tokens (+ pos) for the attentive probe; "mean"/"cls" collapse to N = 1
            "views": 1,  # one deterministic view per clip
            "input": inp,
            # DECLARE what you run with; run_extract enforces it (E_PRECISION_UNDECLARED on a null).
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),  # echoes every consumed --arg with defaults filled
        }

    # ------------------------------------------------------------------ per clip, instance-wise

    def _video(self, s) -> tuple[np.ndarray, np.ndarray, dict]:
        """``(tokens [T·H·W, D], pos [T·H·W, 4], used)`` for one clip."""
        v = s.streams["video"]
        times = _rule_times(VIDEO_TEMPORAL, v.info)  # (k + 0.5) · 0.5 s for n = 16: identical for every row
        # or PyAV on v.open(): your own stack, declared in "decoder"
        from yavih.media.decode import frames_at

        frames, pts = frames_at(v, times)  # uint8 [n, H, W, 3] (native size), pts float64 [n] of the frames fed
        # TODO: resize short side → 256 (bilinear, antialias), centre-crop 224, * 1/255, normalize, → model → tokens
        _ = frames
        tokens = np.zeros((N_FRAMES // TUBELET * VIDEO_H * VIDEO_W, 1), dtype=np.float32)  # placeholder [T*H*W, D]
        region = _crop_region(VIDEO_SPATIAL, int(v.info["width"]), int(v.info["height"]))
        # t of a tubelet = mean of its frames' RULE times (gotcha 1), as a fraction of the video's duration_s
        t_centres = times.reshape(-1, TUBELET).mean(axis=1) / float(v.info["duration_s"])
        pos = grid_pos(N_FRAMES // TUBELET, VIDEO_H, VIDEO_W, stream=self.streams.index("video"), t_centres=t_centres.tolist(), region=region)
        return tokens, pos, {"times": [float(t) for t in pts]}

    def _audio(self, s) -> tuple[np.ndarray, np.ndarray, dict]:
        a = s.streams["audio"]
        sr, n = int(a.info["sample_rate"]), int(a.info["n_samples"])
        lo, hi = 0, min(n, round(AUDIO_TEMPORAL["length_s"] * sr))
        # or soundfile / torchaudio on a.open()
        from yavih.media.decode import audio as decode_audio

        wav, rate = decode_audio(a, np.array([[lo, hi]], dtype=np.int64))  # float32 [L, C], native rate
        # TODO: mono (the file is mono), fbank/mel with the model's front end, normalize with the RECIPE constants
        _ = (wav, rate)
        tokens = np.zeros((AUDIO_T * AUDIO_F, 1), dtype=np.float32)  # placeholder [T*F, D]
        # audio tokens carry time only: t = bin centre / duration_s, y = x = -1 (no spatial coordinate)
        t = (np.arange(AUDIO_T) + 0.5) * (hi - lo) / sr / AUDIO_T / float(a.info["duration_s"])
        pos = np.full((AUDIO_T * AUDIO_F, 4), -1.0, dtype=np.float32)
        pos[:, 0] = self.streams.index("audio")
        pos[:, 1] = np.repeat(t, AUDIO_F)
        return tokens, pos, {"samples": [[lo, hi]], "pad": [0, 0]}

    def encode(self, samples: list) -> tuple[Any, Any, list[dict]]:
        feats, poss, used = [], [], []
        for s in samples:  # instance-wise: nothing computed across the batch
            toks, ps, u = [], [], {}
            try:
                if "video" in self.streams:
                    t, p, u["video"] = self._video(s)
                    toks.append(t)
                    ps.append(p)
                if "audio" in self.streams:
                    t, p, u["audio"] = self._audio(s)
                    toks.append(t)
                    ps.append(p)
            except (OSError, ValueError) as exc:  # a corrupt file becomes a missing row, not an aborted shard
                from yavih import MediaUndecodable

                raise MediaUndecodable([s.clip_id], f"{type(exc).__name__}: {exc}") from exc
            feats.append(np.concatenate(toks, axis=0))
            poss.append(np.concatenate(ps, axis=0))
            used.append(u)
        out = np.stack(feats).astype(self.dtype)  # [B, N, D] — the same N for every sample
        return out, np.stack(poss).astype(np.float32), used

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
