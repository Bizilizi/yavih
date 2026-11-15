"""Bridge template for ``diving48`` — Diving48 fine-grained dive classification (V2 splits, 48 classes).

Read ``setup.md`` first (``yavih bench show diving48 setup``). Copy this file next to your model code, paste the
model's own preprocessing source below, fill every ``TODO``, then run
``yavih bridge check my_diving48_bridge.py --bench diving48 --out diving48.profile.json``; before any data is
staged, ``yavih bridge check my_diving48_bridge.py --bench diving48 --synthetic`` smoke-tests it on tiny
generated media. An unfilled copy fails bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES. A bridge file
serves one benchmark; each matrix entry (file + args) is one reading of one model.

INPUT CONTRACT (what ``prepare`` and ``encode`` receive; setup.md "Input contract"). One stream per row::

    st = sample.streams["video"]          # RawStream, kind "video": the stored Diving48 mp4, byte for byte
    st.path / st.open() / st.bytes()      # the file; nothing decoded, cut, re-timed, resized or cropped
    st.info = {"duration_s": D,           # container duration (setup.md "Input contract" has the range)
               "fps": f,                  # avg_frame_rate of THIS clip: 25, 30, 29.97, 60, ... (varies per clip)
               "n_frames": n,             # frames of the video track
               "width": 640, "height": 480,
               "container": "mp4", "bytes": ...}
    sample.meta == {"facts": {}}          # no label facts; labels, split and manifest never reach you

OUTPUT CONTRACT (setup.md "Output contract"). ``encode`` returns ``(feats, pos, used)``:
  feats  [B, N, D], one fixed (N, D) for every row of both splits (clf: any N, pooled N = 1 is fine);
  pos    optional on diving48 (``requires.pos: []``); when written, MEDIA coordinates: t = token time / D,
         (y, x) = fractions of the ORIGINAL 640x480 frame inside the crop region (below);
  used   per clip ``{"video": {"times": [pts of the frames fed, in feed order]}}``; validate re-evaluates the
         declared temporal rule on ``st.info`` and compares (E_RULE_MISMATCH, E_TIMES_OUT_OF_MEDIA).

Everything model-side is YOUR declared decision, recorded in ``describe()["input"]``: frame count and times,
resize, crop, interpolation, normalisation constants with their source, decoder. Instance-wise only: no batch
statistics, no padding to the batch maximum, one deterministic view, no cache keyed on clip id (``bridge check``
hard-checks rerun, batch and order invariance).

DEFAULTS = the reference's (VideoGLUE, Yuan et al.; tensorflow/models official/projects/videoglue @
19f287775ae950e22a9b5fb82727e72cff2e3664, Apache-2.0; configs/yaml/vmae/frozen/vc_vmae_vit3d_diving48.yaml and
datasets/video_classification.py ``_build``): 16 frames at a temporal stride of 4 decoded frames from one clip in
the middle of the video (VideoGLUE's ``num_test_clips: 1`` path), short side resized to 256 (``min_resize``), a
224 centre crop, pixels scaled to [0, 1] (``zero_centering_image: false``). Stated in seconds so it holds for every
frame rate: 16 frames at 6.25 Hz (stride 4 at 25 fps, the most common rate) from a 2.56 s window centred on the
clip. VideoGLUE's own scores average 4 clips x 3 crops; a yavih bridge feeds one view. These are the reference's
choices for ITS models, not protocol: replace them with your model's own recipe and declare it.

Preprocessing source of the model (paste VERBATIM — the transform pipeline / processor config / video data
config the checkpoint was evaluated with, with repo and commit; ``describe()["input"]`` must match it line by
line):

    TODO

Gotchas (setup.md "Gotchas"), in the order they bite:
  1. Frame rates differ per clip (25, 30, 29.97, 60, ... fps): a stride counted in frames covers a different span
     of the dive on each clip. Declare times in seconds (the ``fps`` rule below) or an index rule you mean; never
     assume 25 fps. Report the pts you actually fed.
  2. A dive lasts longer than the 2.56 s default window on many clips (takeoff, flight and entry; median clip
     about 5.8 s): the centred window can miss the takeoff or the entry. ``{"rule": "uniform", "n": 16,
     "span": "full"}`` covers the whole clip; choose what your model was evaluated with and declare it.
  3. Clips shorter than the window (down to about 1 s): ``edge: "clamp"`` repeats the last frame. Allowed; do not
     loop or pad to the batch maximum.
  4. Every clip is 640x480 (4:3): short side 256 + centre crop 224 keeps 66 % of the width and 87.5 % of the
     height; the diver is often near the frame's edge on takeoff. Declare it, do not "fix" it per clip.
  5. Normalisation constants come from the model's own pretraining recipe; never batch or dataset statistics.
  6. ``.eval()`` yourself; declare precision honestly (``run_extract`` enforces it); ``weights_sha256`` must be
     the real ``fingerprint_state_dict`` (a null one is recorded as W_WEIGHTS_UNFINGERPRINTED).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, grid_pos  # , fingerprint_state_dict
from yavih.media.rules import crop_region, rule_times

# TODO: the model's clip geometry (defaults: VideoGLUE's ViT-B/16 VideoMAE setting).
N_FRAMES = 16  # frames per clip fed to the model
SAMPLE_HZ = 6.25  # frames per second of media: VideoGLUE's temporal stride 4 at 25 fps
TUBELET = 2  # temporal patch size (1 for per-frame image encoders); token time bins = N_FRAMES // TUBELET
RESIZE = 256  # short side after resize (VideoGLUE min_resize)
CROP = 224  # centre crop side
PATCH = 16  # spatial patch size
GRID = CROP // PATCH  # token grid side (14)


class Diving48Bridge(Bridge):
    name = "TODO-model-diving48"  # optional label of this file, recorded only
    batch_size = 8  # TODO: what fits; describe()["batch_size"] must equal this
    dtype = "float16"  # switch to "float32" when bridge check warns about range
    pos_mode = "per_sample"  # (y, x) of the crop are the same for every clip, t depends on the clip; "none" for pooled features
    streams = ("video",)  # stream tags used in pos (tag index 0 = video)
    ARGS = {
        # every --arg k=v the bridge consumes, with its type and default; echoed in describe()["args"]
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap (-1 = last)"),  # TODO: or remove
    }

    # Declared rules. validate re-evaluates TEMPORAL on every row's info (E_RULE_MISMATCH) and checks
    # that every token centre lies inside crop_region(SPATIAL, width, height).
    # 16 frames at start + k / 6.25 s, k = 0..15, start = max(0, (duration_s - 2.56) / 2): VideoGLUE's middle clip.
    TEMPORAL = {"rule": "fps", "fps": SAMPLE_HZ, "n": N_FRAMES, "start": "centre_clamped", "phase": 0, "edge": "clamp"}
    # the whole clip instead (gotcha 2): {"rule": "uniform", "n": N_FRAMES, "span": "full", "edge": "error"}
    SPATIAL = {"resize": {"short": RESIZE}, "crop": {"kind": "center", "size": [CROP, CROP]}, "interp": "bilinear", "antialias": True}
    NORM = {
        "scale": 1 / 255,
        "mean": [0.485, 0.456, 0.406],  # TODO: the model's own constants (VideoGLUE feeds [0, 1] pixels)
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
            "reading": "v",  # Diving48 clips have no audio track (a | v | av across benchmarks)
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
        """The declared frames of one clip: uint8 [N_FRAMES, 480, 640, 3] RGB and their pts (media seconds)."""
        times = rule_times(self.TEMPORAL, st.info)  # start + k / 6.25 from this clip's facts (clamped to the clip)
        frames, pts = self._frames_at(st, times)  # nearest stored frame by pts; pts are what you REPORT
        return frames, np.asarray(pts, dtype=np.float64)

    def _tokens(self, frames: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
        """One clip → float32 [N, D] with N = (N_FRAMES // TUBELET) * GRID * GRID in row-major (t, y, x)."""
        if self.model is not None:
            # TODO: the model's own transform, exactly as pasted above, then the forward pass, e.g.
            # import torch, torch.nn.functional as F
            # x = torch.from_numpy(frames).to("cuda").permute(0, 3, 1, 2).float()          # [T, 3, 480, 640]
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
