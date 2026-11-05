"""Bridge template for ``ssv2`` — Something-Something-v2 temporal action classification.

Read ``setup.md`` first (``yavih bench info ssv2 --setup``). Copy this file next to your model code, paste the
model's own preprocessing source below, fill every ``TODO``, then run
``yavih bridge check my_ssv2_bridge.py --bench ssv2 --out ssv2.profile.json``; before any data
is staged, ``yavih bridge check my_ssv2_bridge.py --bench ssv2 --synthetic`` smoke-tests it on tiny generated
media. An unfilled copy fails bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES. A bridge file serves one
benchmark; each matrix entry (file + args) is one reading of one model.

INPUT CONTRACT (what ``prepare`` and ``encode`` receive; setup.md "Input contract"). One stream per row::

    st = sample.streams["video"]          # RawStream, kind "video": the stored SSv2 webm, byte for byte
    st.path / st.open() / st.bytes()      # the file; nothing decoded, cut, re-timed, resized or cropped
    st.info = {"duration_s": D,           # container duration, 0.83-14.17 s, median 4.0 s; = last pts + 1/12
               "fps": 12.0,               # avg_frame_rate; NOT the true rate of every clip (gaps, gotcha 2)
               "n_frames": n,             # packet count from the site census (webm headers have none), 5-142
               "width": W, "height": 240, # W from 240 to 483 (427 for 59 %, 320 for 20 %): widths vary per clip
               "container": "webm", "bytes": ...}
    sample.meta == {"facts": {}}          # no label facts on ssv2; labels, split and manifest never reach you

OUTPUT CONTRACT (setup.md "Output contract"). ``encode`` returns ``(feats, pos, used)``:
  feats  [B, N, D], one fixed (N, D) for every row of both splits (clf: any N, pooled N = 1 is fine);
  pos    optional on ssv2 (``requires.pos: []``); when written, MEDIA coordinates: t = token time / D,
         (y, x) = fractions of the ORIGINAL 240p frame. Widths vary, so a centre crop covers a different part
         of each clip: ``pos_mode = "per_sample"`` with ``region = crop_region(...)`` per clip (below);
  used   per clip ``{"video": {"times": [pts of the frames fed, in feed order]}}``; validate re-evaluates the
         declared temporal rule on ``st.info`` and compares (E_RULE_MISMATCH, E_TIMES_OUT_OF_MEDIA).

Everything model-side is YOUR declared decision, recorded in ``describe()["input"]``:
frame count and times (``uniform`` bin centres here; ``index_linspace`` is the frame-index alternative), resize, crop,
interpolation, normalisation constants with their source, decoder. Instance-wise only: no batch statistics,
no padding to the batch maximum, one deterministic view, no cache keyed on clip id (``bridge check`` hard-checks
rerun, batch and order invariance).

Preprocessing source of the model (paste VERBATIM — the transform pipeline / processor config / video data
config the checkpoint was evaluated with, with repo and commit; ``describe()["input"]`` must match it line by
line):

    TODO

Gotchas (setup.md "Gotchas"), in the order they bite:
  1. SSv2 is 240p: short side 224 + centre crop 224 keeps the middle 56 % of a 427-wide clip's width; a
     short side of 256 (an upscale) keeps 49 % of the width and 87.5 % of the height. Declare it, do not "fix" it
     per clip.
  2. Frame times are not k / fps. 1 315 of the 44 777 clips start at pts > 0 (up to 1.417 s) and 2 762 have
     holes inside (largest 3.08 s); 3 769 clips hold fewer frames than D * 12. Report the pts you actually
     fed; never synthesize times from indices. ``index_linspace`` (t = idx / fps) is wrong on exactly those
     3 769 clips (setup.md "Gotchas").
  3. Short clips (n_frames down to 5): 16 uniform times on a 0.83-s clip repeat frames (nearest pts). Allowed;
     do not pad, loop or resample to a fixed duration, and never pad to the batch maximum.
  4. Normalisation constants come from the model's own pretraining recipe (ImageNet mean/std, or x*2-1);
     per-clip statistics only when they ARE the model's own recipe (then declare them in ``norm`` with their
     ``source``); never batch or dataset statistics.
  5. ``.eval()`` yourself; declare precision honestly (``run_extract`` enforces it); ``weights_sha256`` must be
     the real ``fingerprint_state_dict`` (a null one is recorded as W_WEIGHTS_UNFINGERPRINTED).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, grid_pos  # , fingerprint_state_dict
from yavih.media.rules import crop_region, rule_times

# TODO: the model's clip geometry.
N_FRAMES = 16  # frames per clip fed to the model (SSv2 median clip: 48 frames at 12 fps)
TUBELET = 2  # temporal patch size (1 for per-frame image encoders); token time bins = N_FRAMES // TUBELET
RESIZE = 224  # short side after resize (the source is 240p: anything > 240 is an upscale)
CROP = 224  # centre crop side
PATCH = 16  # spatial patch size
GRID = CROP // PATCH  # token grid side (14)


class Ssv2Bridge(Bridge):
    name = "TODO-model-ssv2"  # optional label of this file, recorded only
    batch_size = 8  # TODO: what fits; describe()["batch_size"] must equal this
    dtype = "float16"  # switch to "float32" when bridge check warns about range
    pos_mode = "per_sample"  # widths vary → the crop region differs per clip; "none" for pooled features
    streams = ("video",)  # stream tags used in pos (tag index 0 = video)
    ARGS = {
        # every --arg k=v the bridge consumes, with its type and default; echoed in describe()["args"]
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap (-1 = last)"),  # TODO: or remove
    }

    # Declared rules. validate re-evaluates TEMPORAL on every row's info (E_RULE_MISMATCH) and checks
    # that every token centre lies inside crop_region(SPATIAL, width, height).
    TEMPORAL = {"rule": "uniform", "n": N_FRAMES, "span": "full", "edge": "error"}  # bin centres (k + 0.5) D / n
    # frame-index alternative: {"rule": "index_linspace", "n": N_FRAMES} (see gotcha 2 for its caveat)
    # a fixed-rate window instead, e.g.
    #   {"rule": "fps", "fps": 4, "n": 16, "start": "centre_clamped", "phase": 0, "edge": "clamp"}
    # = a 4 s window centred on the clip (start = max(0, (duration_s - 4) / 2)), frames at start + k/4
    # (phase 0 = window start; the default phase 0.5 = bin centres start + (k + 0.5)/4). Add
    # "duration_from": "frame_count" to centre on n_frames / fps instead of the webm's duration_s (the two
    # differ on ~7% of SSv2 clips, whose webm timestamps have gaps).
    SPATIAL = {"resize": {"short": RESIZE}, "crop": {"kind": "center", "size": [CROP, CROP]}, "interp": "bilinear", "antialias": True}
    NORM = {
        "scale": 1 / 255,
        "mean": [0.485, 0.456, 0.406],  # TODO: the model's own constants
        "std": [0.229, 0.224, 0.225],  # TODO
        "source": "TODO: <repo>@<commit> <file> (the model's pretraining recipe)",
    }

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)
        # Decoding with yavih's helpers ([media] extra, PyAV). Your model's own decoder (decord, torchvision.io,
        # PyAV directly through st.open()) is equally fine: then declare it in describe()["input"]["video"]["decoder"].
        from yavih.media.decode import decoder_info, frames_at

        self._frames_at = frames_at
        self._decoder = decoder_info()
        # TODO: build the model from pinned weights BEFORE the first sample; call .eval() YOURSELF (run_extract
        # refuses a module in training mode, E_TRAIN_MODE); move it to the device; no network during encode.
        # import torch
        # self.model = build_model(...).eval().to("cuda")
        # self._weights_sha256 = fingerprint_state_dict(self.model)
        self.model = None
        self._weights_sha256: str | None = None  # TODO: real fingerprint; None is recorded as W_WEIGHTS_UNFINGERPRINTED

    def describe(self) -> dict:
        # Every key below is REQUIRED; extra keys are allowed and recorded (never identity).
        return {
            "weights_sha256": self._weights_sha256,
            "weights_ref": "TODO: path or hub id (provenance only, not hashed)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": "v",  # SSv2 has no audio track (a | v | av across benchmarks)
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.args["layer"],
            "pooling": "none",  # "none" keeps tokens for the attentive probe; "mean"/"cls" collapse to N = 1
            "views": 1,  # MUST be 1: one deterministic view per clip
            "input": {
                "video": {
                    "temporal": dict(self.TEMPORAL),
                    "spatial": dict(self.SPATIAL),
                    "norm": dict(self.NORM),
                    "decoder": dict(self._decoder),
                },
            },
            "tubelet": TUBELET,  # further top-level keys are allowed and recorded (the input entries are closed)
            # DECLARE what you run with; run_extract enforces it (E_PRECISION_UNDECLARED on a null).
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),  # echoes every consumed --arg with defaults filled
        }

    # ------------------------------------------------------------------ one clip, instance-wise

    def _clip(self, st: Any) -> tuple[np.ndarray, np.ndarray]:
        """The declared frames of one clip: uint8 [N_FRAMES, 240, W, 3] RGB and their pts (media seconds)."""
        times = rule_times(self.TEMPORAL, st.info)  # (k + 0.5) * D / n from this clip's facts
        frames, pts = self._frames_at(st, times)  # nearest stored frame by pts; pts are what you REPORT
        return frames, np.asarray(pts, dtype=np.float64)

    def _tokens(self, frames: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
        """One clip → float32 [N, D] with N = (N_FRAMES // TUBELET) * GRID * GRID in row-major (t, y, x)."""
        if self.model is not None:
            # TODO: the model's own transform, exactly as pasted above, then the forward pass, e.g.
            # import torch, torch.nn.functional as F
            # x = torch.from_numpy(frames).to("cuda").permute(0, 3, 1, 2).float()          # [T, 3, 240, W]
            # h, w = x.shape[-2:]; s = RESIZE / min(h, w)
            # x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", antialias=True, align_corners=False)
            # top, left = (x.shape[-2] - CROP) // 2, (x.shape[-1] - CROP) // 2
            # x = x[..., top:top + CROP, left:left + CROP] * self.NORM["scale"]
            # x = (x - mean[:, None, None]) / std[:, None, None]                            # FIXED constants
            # tokens = self.model(x.permute(1, 0, 2, 3)[None], layer=self.args["layer"])   # [1, N, D]
            # return tokens[0].float().cpu().numpy()
            raise NotImplementedError("TODO: run the model")
        # Placeholder (no model): the mean RGB of each (tubelet, cell) of the crop region of the ORIGINAL frame.
        # A pure function of this clip's pixels, so it passes bridge check; it carries no model signal.
        h, w = frames.shape[1:3]
        y0, x0, y1, x1 = region
        ys = np.round(np.linspace(y0 * h, y1 * h, GRID + 1)).astype(int)
        xs = np.round(np.linspace(x0 * w, x1 * w, GRID + 1)).astype(int)
        f = frames.astype(np.float32).reshape(N_FRAMES // TUBELET, TUBELET, h, w, 3).mean(axis=1) / 255.0
        cells = [f[:, ys[i] : max(ys[i + 1], ys[i] + 1), xs[j] : max(xs[j + 1], xs[j] + 1)].mean(axis=(1, 2)) for i in range(GRID) for j in range(GRID)]
        return np.stack(cells, axis=1).reshape(-1, 3)  # [T' * GRID * GRID, 3], row-major (t, y, x)

    def prepare(self, sample: Any) -> dict:
        """One clip's CPU work, ahead of ``encode``: what this returns arrives as ``sample.prepared``."""
        # Runs in decode_workers threads ahead of encode(): per-sample decoding only, no model, no state across samples.
        # The stored bytes are released after prepare(), so encode() works from this dict only.
        frames, pts = self._clip(sample.streams["video"])
        return {"frames": frames, "pts": pts}

    def encode(self, samples: list) -> tuple[Any, Any | None, list[dict]]:
        feats, pos, used = [], [], []
        for s in samples:  # instance-wise: nothing crosses clip boundaries
            st = s.streams["video"]
            frames, pts = s.prepared["frames"], s.prepared["pts"]  # what prepare() returned for this sample
            region = crop_region(self.SPATIAL, st.info["width"], st.info["height"])  # the crop, in ORIGINAL-frame fractions
            feats.append(self._tokens(frames, region))
            # media coordinates: a tubelet's time = mean pts of its frames / D; (y, x) = cell centres inside region
            t_centres = np.clip(pts.reshape(-1, TUBELET).mean(axis=1) / st.info["duration_s"], 0.0, 1.0)
            pos.append(grid_pos(N_FRAMES // TUBELET, GRID, GRID, stream=0, t_centres=t_centres, region=region))
            used.append({"video": {"times": [float(t) for t in pts]}})
        out = np.stack(feats).astype(self.dtype)  # [B, N, D], the same N for every sample
        return out, (np.stack(pos) if self.pos_mode == "per_sample" else None), used

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
