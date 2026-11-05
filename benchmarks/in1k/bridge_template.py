"""Bridge template for ``in1k`` — ImageNet-1K single-label image classification.

Copy this file next to your model code, fill every ``TODO``, then run
``yavih bridge check my_in1k_bridge.py --bench in1k --out in1k.profile.json`` (SKILL step 5); before any data is
staged, ``--synthetic`` instead of ``--out`` smoke-tests it on tiny generated images. An unfilled copy fails
bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES. A bridge file serves one benchmark; each matrix entry
(file + args) is one reading of one model. Read ``setup.md`` ("## Input contract", "## Output contract",
"## Gotchas") first.

INPUT (setup.md "## Input contract"): one RawStream per row, ``sample.streams["image"]``::

    kind "image"; path = the stored ILSVRC2012 JPEG; bytes() / open() = that file byte for byte;
    info = {width, height, container, bytes} (guaranteed), the ORIGINAL stored size; meta["facts"] == {}

Nothing is decoded for you. Decoder, colour conversion (a few files are greyscale or CMYK), resize, crop,
interpolation, normalisation constants and — for a video tower — how many identical frames you feed are
YOUR decisions; declare every one in ``describe()["input"]["image"]``. That declaration is recorded
with every number (``describe_sha256``), never identity: to show two crops side by side, give them two ``model``
names; one name re-run with another crop is one report row, newest shown (``W_BRIDGE_CHANGED``).

OUTPUT (setup.md "## Output contract"): ``encode`` returns ``(feats, pos, used)``: feats ``[B, N, D]`` with
the same ``N`` and ``D`` for every row; ``pos`` optional for ``clf`` (``pos_mode: "none"`` is eligible); if you
emit a token grid, positions are in MEDIA coordinates: ``t = -1`` (an image has no time), ``(y, x)`` = fraction
of the ORIGINAL image, i.e. patch centres mapped back through your crop with
``yavih.media.rules.crop_region`` (validate checks every centre lies in the declared crop, E_RULE_MISMATCH).
Image sizes differ per row, so a grid needs ``pos_mode: "per_sample"``. ``used[i] == {"image": {}}``.

GUARDRAILS (``bridge check`` hard-checks rerun bit-identity and batch-/order-
invariance at rel_l2 1e-5): instance-wise preprocessing (no batch statistics, no padding to the batch max,
normalisation constants from the model's own pretraining recipe; per-sample statistics only when they ARE that
recipe, declared in ``norm`` with their ``source``; never batch or dataset statistics); one deterministic view; no state across
samples; never read row/clip_id/split; ``.eval()`` + frozen weights; weights loaded in ``__init__``, no
network and no writes outside the run directory inside ``encode``.

Preprocessing source of the model (paste VERBATIM — the transform pipeline / processor config / timm
data_config the checkpoint was evaluated with, with repo and commit; ``describe()["input"]`` must match it
line by line):

    TODO

Example recipe (declare whatever your checkpoint was evaluated with): ``Resize(256) -> CenterCrop(224) ->`` the
model's normalisation; a video tower repeats the image to T identical frames (``"image_as_video"`` below).
yavih does none of it for you.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, grid_pos  # , fingerprint_state_dict

# The spatial rule (the declared-rule vocabulary of the skill's references/bridge.md). This one is torchvision's Resize(256) + CenterCrop(224); validate
# re-evaluates it with yavih.media.rules.crop_region on every row's width/height. TODO: your model's own.
SPATIAL = {
    "resize": {"short": 256},
    "crop": {"kind": "center", "size": [224, 224]},
    "interp": "bilinear",
    "antialias": True,
}
# TODO: the model's own constants and their source (repo@commit file); mean [0.5]*3, std [0.5]*3 is the [-1, 1] convention.
NORM = {
    "scale": 1 / 255,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "source": "TODO repo@commit path/to/transforms.py (the checkpoint's own eval recipe)",
}
PATCH = 16  # TODO: the model's patch size; 224 / 16 = 14 x 14 tokens


class In1kBridge(Bridge):
    name = "TODO-model-in1k"  # optional label of this file, recorded only; the model's name is describe()["model"]
    batch_size = 32  # TODO: what fits; describe()["batch_size"] must equal this (recorded)
    dtype = "float16"  # switch to "float32" when bridge check warns about range
    pos_mode = "none"  # clf needs no positions; "per_sample" if you emit a token grid (image sizes vary)
    streams = ()  # ("image",) when pos_mode != "none"
    ARGS = {
        # every --arg k=v the bridge consumes, with its type and default; echoed in describe()["args"]
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap (-1 = last)"),  # TODO: the tap
    }

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)  # bench is a BenchmarkView: no manifest, labels or splits
        # TODO: build the model from pinned LOCAL weights (no network after this point); call .eval() YOURSELF
        # (run_extract refuses a module in training mode, E_TRAIN_MODE); move it to the device.
        # self.model = ...; self.model.eval()
        self.n_frames = 1  # TODO: 1 for an image model; T for a video tower fed T identical frames

    def describe(self) -> dict:
        import PIL

        # Every key is REQUIRED and must be TRUE; extra keys are allowed and recorded (never identity).
        return {
            "weights_sha256": None,  # TODO: fingerprint_state_dict(self.model); null is recorded as W_WEIGHTS_UNFINGERPRINTED
            "weights_ref": "TODO hub id or path (provenance only)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": "v",  # the visual pathway; in1k has no audio (a | v | av across benchmarks)
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.args["layer"],
            "pooling": "mean",  # TODO: "none" (token grid) | "mean" | "cls" | ...
            "views": 1,  # MUST be 1: one deterministic view (no multi-crop / flip / TTA averaging)
            "input": {
                "image": {  # an image stream has no temporal rule
                    "spatial": SPATIAL,
                    "norm": NORM,
                    "decoder": {"lib": "pillow", "version": PIL.__version__, "convert": "RGB"},
                },
            },
            # Not in the input-rule vocabulary, so declared at top level (recorded): how a video tower sees a still.
            "image_as_video": {"frames": self.n_frames, "how": "repeat"},
            "precision": {
                # DECLARED here, ENFORCED by run_extract; no null except autocast (E_PRECISION_UNDECLARED)
                "autocast": None,  # None | "bf16" | "fp16"
                "matmul_precision": "highest",
                "allow_tf32": False,
                "deterministic": True,
            },
            "args": dict(self.args),  # MUST echo every consumed --arg with defaults filled in
        }

    # ------------------------------------------------------------------ preprocessing (instance-wise)

    @staticmethod
    def preprocess(stream: Any) -> np.ndarray:
        """Decode the stored file and apply SPATIAL + NORM to ONE image -> float32 [3, 224, 224].

        TODO: replace by the model's own transform, verbatim (e.g. its torchvision / timm pipeline applied to
        the PIL image); keep it a pure function of this one file. This placeholder is torchvision's
        Resize(256) (long side = int(256 * long / short)) + CenterCrop(224) on pillow's bilinear resize.
        """
        from PIL import Image

        with stream.open() as fh, Image.open(fh) as im:
            img = im.convert("RGB")  # greyscale and CMYK files exist: convert explicitly (declared: decoder.convert)
        w, h = img.size
        short = SPATIAL["resize"]["short"]
        if w <= h:
            nw, nh = short, int(short * h / w)
        else:
            nw, nh = int(short * w / h), short
        img = img.resize((nw, nh), Image.BILINEAR)  # pillow antialiases on downscale (declared: antialias)
        ch, cw = SPATIAL["crop"]["size"]
        top, left = int(round((nh - ch) / 2.0)), int(round((nw - cw) / 2.0))
        img = img.crop((left, top, left + cw, top + ch))
        x = np.asarray(img, dtype=np.float32) * NORM["scale"]
        x = (x - np.asarray(NORM["mean"], np.float32)) / np.asarray(NORM["std"], np.float32)
        return np.ascontiguousarray(x.transpose(2, 0, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """[B, 3, 224, 224] -> [B, N, D] with the same N for every row.

        TODO: torch.from_numpy(x).to(device, dtype); for a video tower x[:, :, None].repeat(T) to
        [B, 3, T, H, W]; tokens = self.model(...) at the tap; pool per self.describe()["pooling"].
        run_extract already wraps encode in torch.inference_mode().
        """
        raise NotImplementedError("TODO: run the model")

    def prepare(self, sample: Any) -> dict:
        """One image's CPU work, ahead of ``encode``: what this returns arrives as ``sample.prepared``."""
        # Runs in decode_workers threads ahead of encode(): per-sample decoding only, no model, no state across samples.
        # The stored bytes are released after prepare(), so encode() works from this dict only.
        return {"x": self.preprocess(sample.streams["image"])}

    def encode(self, samples: list) -> tuple[Any, Any | None, list[dict]]:
        x = np.stack([s.prepared["x"] for s in samples])  # what prepare() returned, per sample
        feats = self.forward(x)
        pos = None
        if self.pos_mode == "per_sample":
            from yavih.media.rules import crop_region

            g = SPATIAL["crop"]["size"][0] // PATCH
            rows = []
            for s in samples:
                info = s.streams["image"].info
                region = crop_region(SPATIAL, info["width"], info["height"])  # the part of the ORIGINAL image shown
                # t_centres=[-1.0]: an image has no time axis
                rows.append(grid_pos(1, g, g, stream=0, t_centres=[-1.0], region=region, cls=False))  # TODO: cls=True if a CLS token leads
            pos = np.stack(rows)
        used = [{"image": {}} for _ in samples]  # image streams report {}
        return feats, pos, used
