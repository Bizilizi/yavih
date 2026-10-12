"""Bridge template for ``nyuv2_depth`` — NYUv2 dense monocular depth, 14×14 grid readout.

Copy this file next to your model code, fill every ``TODO``, then run
``yavih bridge check my_nyuv2_bridge.py --bench nyuv2_depth --out nyuv2_depth.profile.json`` (SKILL step 5);
before any data is staged, ``--synthetic`` instead of ``--out`` smoke-tests it on tiny generated images. An
unfilled copy fails bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES. A bridge file serves one benchmark;
each matrix entry (file + args) is one reading of one model. Read ``setup.md`` ("## Input contract",
"## Output contract") first.

INPUT (what the reader hands over, setup.md "## Input contract"): ONE stream per row::

    sample.streams["image"]      RawStream, kind "image"
        .path                    <derived_root>/rgb/<split>/<clip_id>.png — a lossless 8-bit RGB PNG, 640×480,
                                 exactly the upstream pixels (no crop, no resize, no normalisation)
        .open() / .bytes()       the stored PNG bytes; decode them yourself (pillow, torchvision.io, cv2, …)
                                 or with yavih.media.decode.image(stream)
        .info                    {"width": 640, "height": 480, "container": "png", "bytes": …} (guaranteed keys)
    sample.meta["facts"]         {"target_region": [0.0, 0.125, 1.0, 0.875]} — the part of the frame the depth
                                 targets cover (Probe3D's 480×480 centre square), in full-frame fractions

There is NO audio stream (a silent still). An audio-visual model reads its visual pathway (``reading: "v"``);
what it feeds its audio branch (zeros, a mask, nothing) is a model-side decision: declare it in ``describe()``
(e.g. ``"audio_input": "zeros"``). An audio-only model raises ``NotApplicable`` from ``__init__``.

OUTPUT (what the probe needs, setup.md "## Output contract"): ``[N, D]`` tokens per image, fixed ``N``, ``D``, and
per-token positions in MEDIA COORDINATES: ``pos = (stream, t, y, x)`` with ``y, x`` = fractions of the ORIGINAL
640×480 frame (before your resize/crop), ``t = -1`` (an image has no time). ``features validate`` bins the tokens
tagged ``image`` into 14×14 cells tiling ``target_region`` and records ``eligible: false`` (→ ``not_applicable``)
unless the median image fills ≥ 50 % of them. A pooled ``N = 1`` bridge is therefore ineligible. Resize(224) +
CenterCrop(224) on the 640×480 frame shows ``target_region`` (to one resized pixel): all 196 cells filled; that
is the ``SPATIAL`` example below. Feeding the whole frame (squashed or letterboxed) is equally legal — the tokens
outside the region simply inform no cell.

EVERYTHING ELSE IS YOURS and MUST be declared in ``describe()["input"]["image"]``: the decoder, the
resize/crop (``spatial``), the normalisation constants and their source (the model's own pretraining recipe;
per-image statistics only when they ARE that recipe, declared in ``norm`` with their ``source``; never batch or
dataset statistics), and frame replication for a video tower (declare it, e.g. ``"replicate_frames": 16``).

Preprocessing source of the model (paste VERBATIM — the transform pipeline / processor config / timm
data_config the checkpoint was evaluated with; ``describe()["input"]`` must match it line by line):

    TODO
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih import NotApplicable  # noqa: F401  (raise it from __init__ for an audio-only model)
from yavih.bridge import ArgSpec, Bridge, grid_pos  # , fingerprint_state_dict
from yavih.media.rules import crop_region

GRID_H, GRID_W = 14, 14  # TODO: the model's patch grid at its input size (224 / 16 = 14 for a ViT-B/16)
SIZE = 224  # TODO: the model's input side


class Nyuv2DepthBridge(Bridge):
    name = "TODO-model-nyuv2_depth"  # optional label of this file, recorded only
    batch_size = 32  # TODO: what fits; describe()["batch_size"] must equal this
    dtype = "float16"  # switch to "float32" when bridge check warns about range
    pos_mode = "per_sample"  # region derived from each row's info (all rows are 640×480, so "constant" also holds)
    streams = ("image",)  # stream tags used in pos: index 0 == "image"
    ARGS = {
        # every --arg k=v the bridge consumes, with its type and default; echoed in describe()["args"]
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap (-1 = last)"),  # TODO: the tap
    }

    # The declared spatial rule; crop_region() maps it to the part of the ORIGINAL frame the model sees.
    SPATIAL = {"resize": {"short": SIZE}, "crop": {"kind": "center", "size": [SIZE, SIZE]}, "interp": "bilinear", "antialias": True}
    NORM = {"scale": 1 / 255, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],  # TODO: the model's own constants
            "source": "TODO <repo>@<commit> <file>: the model's pretraining transform"}

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)
        # TODO: build the model from pinned weights (loaded HERE, before the first sample; no network later);
        # call .eval() YOURSELF (run_extract refuses a module in training mode, E_TRAIN_MODE); move to the device.
        # self.model = ...; self.model.eval()
        # if the model has no visual pathway: raise NotApplicable("audio-only model: nyuv2_depth is a still image")
        self.has_cls = False  # TODO: True when the token sequence starts with a CLS/register token you keep
        import PIL

        self._decoder = {"lib": "pillow", "version": PIL.__version__}

    def describe(self) -> dict:
        # Every key below is REQUIRED and must be TRUE; extra keys are allowed and recorded (never identity).
        return {
            "weights_sha256": None,  # TODO: fingerprint_state_dict(self.model); null is recorded as W_WEIGHTS_UNFINGERPRINTED
            "weights_ref": "TODO hub id or path (provenance only)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": "v",  # the visual pathway; nyuv2 has no audio (a | v | av across benchmarks)
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.args["layer"],  # TODO: the tap, as int or str
            "pooling": "none",  # a token grid
            "views": 1,  # one deterministic view; no flips, crops or TTA averaging
            "input": {
                "image": {
                    "spatial": self.SPATIAL,
                    "norm": self.NORM,
                    "decoder": self._decoder,
                    # TODO (video tower): "replicate_frames": T; (AV model): "audio_input": "zeros" | "masked" | …
                },
            },
            "precision": {
                # DECLARED here, ENFORCED by run_extract; no null except autocast (E_PRECISION_UNDECLARED)
                "autocast": None,  # None | "bf16" | "fp16"
                "matmul_precision": "highest",
                "allow_tf32": False,
                "deterministic": True,
            },
            "args": dict(self.args),  # MUST echo every consumed --arg with defaults filled in
        }

    def _preprocess(self, stream) -> np.ndarray:
        """PNG → float32 [3, SIZE, SIZE]: exactly the declared SPATIAL + NORM, nothing else (instance-wise)."""
        from PIL import Image

        with Image.open(stream.open()) as im:
            im = im.convert("RGB")  # already RGB; a no-op kept for safety
            w, h = im.size
            # torchvision's Resize(int) size rule (= what yavih.media.rules.crop_region assumes): short side → SIZE,
            # long side → int(SIZE * long / short); 640×480 → 298×224.
            if w >= h:
                size = (int(SIZE * w / h), SIZE)
            else:
                size = (SIZE, int(SIZE * h / w))
            im = im.resize(size, Image.BILINEAR, reducing_gap=None)  # TODO: match the model's resize implementation
            W, H = im.size
            left, top = int(round((W - SIZE) / 2.0)), int(round((H - SIZE) / 2.0))  # torchvision center_crop
            im = im.crop((left, top, left + SIZE, top + SIZE))
            x = np.asarray(im, dtype=np.float32) * self.NORM["scale"]
        x = (x - np.asarray(self.NORM["mean"], np.float32)) / np.asarray(self.NORM["std"], np.float32)
        return x.transpose(2, 0, 1)

    def encode(self, samples: list) -> tuple[Any, Any, list[dict]]:
        batch, pos, used = [], [], []
        for s in samples:
            st = s.streams["image"]
            batch.append(self._preprocess(st))
            # positions of the 14×14 patch centres, mapped back into the ORIGINAL frame through the declared crop
            region = crop_region(self.SPATIAL, st.info["width"], st.info["height"])
            # an image has no time axis: t = -1 (validate refuses t >= 0 on image tokens)
            pos.append(grid_pos(1, GRID_H, GRID_W, stream=0, t_centres=[-1.0], region=region, cls=self.has_cls))
            used.append({"image": {}})  # image streams report nothing
        # TODO: x = torch.from_numpy(np.stack(batch)).to(device); tok = self.model(x)  (run_extract wraps encode in
        # torch.inference_mode()); video tower: x[:, :, None].expand(-1, -1, T, -1, -1) and average the T' tubelet
        # slices to one 14×14 grid (or pass grid_pos(T', …) with t = -1 handled by you).
        # tok: [B, (1 +) GRID_H*GRID_W, D] in ROW-MAJOR (y, x) order — check the model's flatten order: a transposed
        # (x, y) grid passes validation and silently scores a mirrored depth map (setup.md "## Gotchas").
        n = pos[0].shape[0]
        feats = np.zeros((len(samples), n, 1), dtype=np.float32)  # TODO: replace with the real tokens
        p = np.stack(pos).astype(np.float32)
        p[:, :, 1] = -1.0  # an image has no time axis: t = -1 for every token
        return feats, p, used
