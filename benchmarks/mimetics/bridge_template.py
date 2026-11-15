"""Bridge skeleton for **k400** — Kinetics-400 single-label action classification, video only. Read
``setup.md`` first (input and output contracts). Copy, rename, fill the TODOs, then
``yavih bridge check my_k400_bridge.py --bench k400 --out k400.profile.json`` (``--synthetic`` instead of
``--out`` smoke-tests it before any data is staged). An unfilled copy fails
bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES.

A bridge file serves one benchmark; each matrix entry (file + args) is one reading of one model. K400 hands over one stream, ``video`` (no audio
stream is part of the protocol), so ``describe()["reading"]`` is ``"v"``; an audio-only model raises
``yavih.NotApplicable("k400 has no audio stream")`` from ``__init__``.

PREPROCESSING SOURCE — paste the model's own EVAL input pipeline here, verbatim, with repo, file, function and
commit; ``describe()["input"]`` must match it line by line (``bridge check`` warns W_NO_PREPROCESSING_SOURCE
while this slot is empty):
    TODO

What ``prepare`` and ``encode`` receive per Sample ``s`` (setup.md "## Input contract"):
    s.streams["video"]  RawStream, kind "video": the stored mp4, undecoded. ``.path`` (layout dir) or
        ``.open()`` (works for every layout, incl. tar_index) or ``.bytes()``. Decode it yourself — with the
        model's own stack or ``yavih.media.decode`` (``[media]`` extra) — and declare the decoder.
    s.streams["video"].info  {"duration_s", "fps", "width", "height", "container", "bytes"} non-null on every
        row ("n_frames" may be null). This release (census of the 33 492 frozen files): h264, height 256, width
        138-1424 (mostly 454 or 340; 4 013 portrait, mostly 144 wide x 256 high), 6-60 fps (mostly 29.97/30);
        train ~10 s, val ~12 s, 4 555 frozen clips < 10 s (min 0.53 s). duration_s is the VIDEO stream's (the
        container runs up to 2.6 s longer on 1 415 files).
    s.meta == {"facts": {}}  (no facts; never read row / clip_id / split — bridge check scrambles them)

What ``encode`` returns (setup.md "## Output contract"): ``(feats [B, N, D], pos, used)``
    feats  the same N, D for every clip of every split;
    pos    ``None`` with pooled features (``pos_mode = "none"``), else per-clip ``[B, N, 4]`` = (stream, t, y, x)
           in MEDIA coordinates: t = seconds / info["duration_s"], (y, x) = fractions of the ORIGINAL frame
           (``grid_pos(..., t_centres=..., region=crop_region(...))``). Durations and frame sizes differ between
           clips, so token positions are ``per_sample``. clf needs no positions (``requires.pos: []``), but
           correct ones cost nothing and keep the features usable by any probe;
    used   one dict per clip: ``{"video": {"times": [pts of the frames fed, in feed order]}}``; validate
           re-evaluates the declared temporal rule on the clip's facts and refuses a mismatch (E_RULE_MISMATCH).

Decisions that are yours (declare them in ``describe()["input"]["video"]``; recorded with every number):
    temporal  default below: ``uniform`` n=16 over the whole clip (``span: "full"``: never out of media, the val
              clips' extra 2 s included). Alternative: ``span: [0.0, 10.0], edge: "clamp"`` (late times on the
              4 555 short clips get the last frame). ``edge: "error"`` with a
              fixed span fails on those clips (E_TIMES_OUT_OF_MEDIA).
    spatial   default below: resize short side to 256, centre-crop 224 (portrait clips are UPSCALED from 144).
    norm      the model's pretraining constants with their source (per-clip statistics only when they ARE its
              recipe; never batch or dataset statistics); e.g. V-JEPA / VideoMAE use ImageNet mean/std, some
              models x*2-1 ([-1, 1]).
Gotchas (setup.md "## Gotchas"): one clip, one view (no 16x8x3 multi-view averaging, ``views`` MUST be 1);
no batch statistics or padding to the batch maximum; the model in ``.eval()``; declare precision honestly;
labels and class order never reach a bridge.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, grid_pos  # , fingerprint_state_dict

# TODO: the frame count and token grid of the model (16 frames, tubelet 2 -> 8 temporal tokens; 224/16 = 14).
N_FRAMES = 16
TUBELET = 2
T_TOK, H_TOK, W_TOK = N_FRAMES // TUBELET, 14, 14

# Declared rules. Edit these to match the PREPROCESSING SOURCE; describe() and encode() both read them,
# so what is declared is what runs.
TEMPORAL = {"rule": "uniform", "n": N_FRAMES, "span": "full"}
# a fixed-rate window instead, e.g.
#   {"rule": "fps", "fps": 4, "n": N_FRAMES, "start": "centre_clamped", "phase": 0, "edge": "clamp"}
# = a 4 s window centred on the clip (start = max(0, (duration_s - 4) / 2)), frames at start + k/4
# (phase 0 = window start; the default phase 0.5 = bin centres start + (k + 0.5)/4).
SPATIAL = {"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}
NORM = {
    "scale": 1 / 255,
    "mean": [0.485, 0.456, 0.406],  # TODO: the model's own constants
    "std": [0.229, 0.224, 0.225],
    "source": "TODO: <org>/<repo>@<commit> <file>::<function>",
}


class K400Bridge(Bridge):
    name = "TODO-model-k400"  # optional label of this file, recorded only
    batch_size = 8
    dtype = "float16"  # "float16" | "float32" (switch to float32 when bridge check warns about range)
    pos_mode = "per_sample"  # "none" for pooled features (pooling mean/cls -> N = 1, return pos None)
    streams = ("video",)  # tag index 0 in pos[:, 0]
    ARGS = {
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap (-1 = last)"),
    }

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)  # bench is a BenchmarkView: no manifest, labels, splits or roots
        # TODO: build the model, load weights (BEFORE the first sample; no network during encode) and call .eval()
        # yourself (run_extract refuses a module in training mode, E_TRAIN_MODE):
        # import torch
        # self.model = build_model(...).eval().to("cuda")
        # self._weights_sha256 = fingerprint_state_dict(self.model)
        self.model = None
        self._weights_sha256 = None  # TODO: real fingerprint; None is recorded as W_WEIGHTS_UNFINGERPRINTED
        from yavih.media.decode import (
            decoder_info,  # [media] extra; or your own decoder and its version
        )

        self._decoder = decoder_info()

    def describe(self) -> dict:
        # Every key below is REQUIRED; extra keys are allowed and recorded (never identity).
        return {
            "weights_sha256": self._weights_sha256,
            "weights_ref": "TODO: path or hub id (provenance only, not hashed)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": "v",  # a | v | av: one vocabulary across benchmarks (run dir, results key and report row: model + reading)
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.args["layer"],
            "pooling": "none",  # "none" keeps tokens for the attentive probe; "mean"/"cls" collapse to N = 1
            "views": 1,  # MUST be 1 (no multi-crop / multi-clip / TTA averaging)
            "input": {"video": {"temporal": TEMPORAL, "spatial": SPATIAL, "norm": NORM, "decoder": self._decoder}},
            # DECLARE what you run with; run_extract enforces it (E_PRECISION_UNDECLARED on a null).
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),  # echoes every consumed --arg with defaults filled
        }

    def prepare(self, sample: Any) -> dict:
        """One clip's CPU work, ahead of ``encode``: what this returns arrives as ``sample.prepared``."""
        # Runs in decode_workers threads ahead of encode(): per-sample decoding only, no model, no state across samples.
        from yavih.media.decode import (
            frames_at,  # nearest stored frame by pts; reads through RawStream.open()
        )
        from yavih.media.rules import rule_times

        v = sample.streams["video"]
        times = rule_times(TEMPORAL, v.info)  # float64 [N_FRAMES] media seconds, the declared rule on this clip
        try:
            frames, pts = frames_at(v, times)  # uint8 [N_FRAMES, H, W, 3] at the STORED size, pts float64 [N_FRAMES]
        except Exception as e:  # noqa: BLE001 — a corrupt file is a missing row, not a bridge bug
            from yavih import MediaUndecodable

            raise MediaUndecodable([sample.clip_id], f"k400 video decode failed: {e!r}") from e
        # TODO: resize short side -> 256 (bilinear, antialias), centre-crop 224, * scale, (x - mean) / std here too (CPU,
        #       per clip); the stored bytes are released after prepare(), so encode() works from this dict only.
        return {"frames": frames, "pts": pts, "used": {"video": {"times": [float(t) for t in pts]}}}

    def encode(self, samples: list) -> tuple[Any, Any | None, list[dict]]:
        feats, pos, used = [], [], []
        for s in samples:  # instance-wise: each clip is processed exactly as it would be alone
            p = s.prepared  # what prepare() returned for this sample
            # TODO: [C, T, 224, 224] -> model; tap self.args["layer"]; tokens [T_TOK*H_TOK*W_TOK, D] (drop CLS or add a
            #       CLS pos row with grid_pos(..., cls=True)).
            _ = p["frames"]
            feats.append(np.zeros((T_TOK * H_TOK * W_TOK, 1), dtype=np.float32))  # placeholder [N, D]
            pos.append(self._pos(p["pts"], s.streams["video"].info))
            used.append(p["used"])
        out = np.stack(feats).astype(self.dtype)  # [B, N, D]; the same N for every clip
        return out, (np.stack(pos) if self.pos_mode != "none" else None), used

    @staticmethod
    def _pos(pts: np.ndarray, info: dict) -> np.ndarray:
        """Token centres ``[T_TOK*H_TOK*W_TOK, 4]`` = (stream, t, y, x) in media coordinates, row-major (t, y, x).

        t: the mean pts of each tubelet's frames over the clip's duration; (y, x): patch centres mapped into the part
        of the ORIGINAL frame the crop keeps (``crop_region`` of the declared spatial rule on this clip's size).
        """
        from yavih.media.rules import crop_region

        duration = float(info["duration_s"])
        t_centres = np.asarray(pts, dtype=np.float64).reshape(T_TOK, TUBELET).mean(axis=1) / duration
        region = crop_region(SPATIAL, int(info["width"]), int(info["height"]))
        return grid_pos(T_TOK, H_TOK, W_TOK, stream=0, t_centres=np.clip(t_centres, 0.0, 1.0), region=region).astype(np.float32)

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
