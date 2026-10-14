"""Bridge skeleton for **vggsounder**. Copy, rename, fill the TODOs.

Read ``setup.md`` first: its input contract is what ``prepare`` and ``encode`` receive, its output contract what must
come back. ``yavih bridge check <file> --bench vggsounder --synthetic`` smoke-tests it before any data is
staged; an unfilled copy fails bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES.

A bridge file serves one benchmark; each matrix entry (file + args) is one reading of one model. VGGSounder
scores three readings — audio only (``a``), video only (``v``), both (``av``) — so register this file three
times in the matrix with ``--arg reading=a|v|av`` (or write three files) and give them ONE ``model`` name. Each reading is probed and scored by
the package on its own ground-truth sets (``a`` + ``a only``, ``v`` + ``v only``, ``av``), so ``describe()["reading"]``
must be ``a``, ``v`` or ``av``; ``yavih report build --join vggsounder a=RUN v=RUN av=RUN`` then scores the three readings
together with the official package (``mu``).

Preprocessing source (paste the model's own eval-time input pipeline here, verbatim, with repo, file,
function and commit; ``describe()["input"]`` must match it line by line; ``bridge check`` warns
W_NO_PREPROCESSING_SOURCE while this slot is empty):

    TODO

What ``prepare`` and ``encode`` receive per Sample (``s.streams``), nothing decoded:
    s.streams["video"]  RawStream of the stored mp4 (H.264, native size — about half 720x1280, the rest
                        360x480 / 360x640 / 240x320 and ~5 % portrait —, mostly 25–30 fps, ~10 s; on eval down to
                        1.0 s). ``.path`` / ``.open()`` / ``.bytes()``; ``.info`` = {duration_s, fps,
                        width, height, container, bytes, …}. The mp4 ALSO carries a 44.1 kHz AAC track:
                        never take the audio from it — the benchmark's audio is the ``audio`` stream.
    s.streams["audio"]  RawStream of the stored wav (16 kHz mono PCM-16, nominal 10 s; ~11 % of eval files
                        are shorter, down to 1.02 s). ``.info`` = {duration_s, sample_rate, n_samples,
                        channels, container, bytes}.
    s.meta["facts"]     {} (no label facts for this benchmark).
Everything model-side is decided HERE and declared in ``describe()["input"]``: which frames (rule over
each clip's own duration), resize/crop, the audio window and its padding, the front end, the
normalization constants (from the model's own pretraining recipe; per-clip statistics only when they ARE the
model's own recipe, declared in ``norm`` with their ``source``; never batch or dataset statistics), the decoder.

Output contract: feats ``[B, N, D]`` with one ``(N, D)`` for every row; ``pos`` optional (``clf`` ignores
it) but when given it is in MEDIA coordinates (t = fraction of that stream's duration, y/x = fraction of
the ORIGINAL frame, ``-1`` = none; tokens centred in padding carry ``t = -1``); ``used`` per clip reports
the frame times / sample ranges actually fed, which ``validate`` checks against the declared rules
(E_RULE_MISMATCH, E_TIMES_OUT_OF_MEDIA). Instance-wise only: ``bridge check`` hard-checks rerun,
batch- and order-invariance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, MediaUndecodable, grid_pos

# TODO: the model's geometry. A 16-frame ViT-B/16 with tubelet 2 at 224 px gives 8 x 14 x 14 video tokens.
N_FRAMES = 16
TUBELET = 2
IMG = 224
PATCH = 16
AUDIO_SECONDS = (
    10.0  # the audio window the model was trained on (from the start; zeros after a short file)
)
AUDIO_T, AUDIO_F = 8, 8  # TODO: the audio encoder's token grid (time x frequency)
D = 8  # TODO: the model's feature width (placeholder model below)

VIDEO_RULES = {
    "temporal": {
        "rule": "uniform",
        "n": N_FRAMES,
        "span": "full",
    },  # bin centres over EACH clip's duration
    # a fixed-rate window instead, e.g.
    #   {"rule": "fps", "fps": 4, "n": 16, "start": "centre_clamped", "phase": 0, "edge": "clamp"}
    # = a 4 s window centred on the clip (start = max(0, (duration_s - 4) / 2)), frames at start + k/4
    # (phase 0 = window start; the default phase 0.5 = bin centres start + (k + 0.5)/4). Centring on
    # n_frames / fps ("duration_from": "frame_count") needs n_frames, which this benchmark's input_contract.info
    # does not guarantee; duration_s and n_frames / fps differ on most VGGSound mp4s.
    "spatial": {
        "resize": {"short": 256},
        "crop": {"kind": "center", "size": [IMG, IMG]},
        "interp": "bilinear",
        "antialias": True,
    },
    # TODO: constants and their source = the model's pretraining recipe (file + commit), never data statistics
    "norm": {
        "scale": 1 / 255,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "source": "TODO: repo@commit file",
    },
}
AUDIO_RULES = {
    "temporal": {"rule": "span", "start_s": 0.0, "length_s": AUDIO_SECONDS, "pad": "zeros"},
    # TODO: the model's front end exactly (rate, resampler, features, n_mels, window, hop, dither, channels)
    "front_end": {
        "sample_rate": 16000,
        "resample": "none (stored at 16 kHz)",
        "features": "TODO (placeholder: log-power STFT, 400/160)",
        "channels": "mono as stored",
    },
    "norm": {
        "mean": 0.0,
        "std": 1.0,
        "source": "TODO: repo@commit file (the model's fixed constants)",
    },
}


class VGGSounderBridge(Bridge):
    name = "TODO-model-vggsounder"  # optional label, recorded only; the reading is an arg, so one file serves a/v/av
    batch_size = 8
    dtype = "float16"
    pos_mode = "per_sample"  # durations and frame sizes differ per clip, so pos differs per clip
    streams = ("video", "audio")  # pos tag order; reset per reading in __init__
    ARGS = {
        "reading": ArgSpec(
            type=str, default="av", choices=("a", "v", "av"), help="which streams the model reads"
        ),
        "layer": ArgSpec(type=int, default=-1, help="block to tap (-1 = last)"),
    }

    def __init__(self, bench: Any, args: dict | None = None) -> None:
        super().__init__(bench, args)
        self.reading = self.args["reading"]
        self.streams = {"a": ("audio",), "v": ("video",), "av": ("video", "audio")}[self.reading]
        # TODO: build the model, load weights BEFORE the first sample, call .eval() yourself:
        # self.model = build_model(...).eval().to("cuda")
        # self._weights_sha256 = yavih.bridge.fingerprint_state_dict(self.model)
        rng = np.random.default_rng(
            0
        )  # placeholder "model": fixed projections, so the skeleton runs end to end
        self._proj_v = rng.standard_normal((3, D)).astype(np.float32)
        self._proj_a = rng.standard_normal((1, D)).astype(np.float32)
        self._weights_sha256 = None  # TODO: the real fingerprint; None is recorded as W_WEIGHTS_UNFINGERPRINTED

    # ------------------------------------------------------------------ declared configuration
    def describe(self) -> dict:
        import av
        import soundfile

        inputs = {}
        if "video" in self.streams:
            libav = ".".join(str(v) for v in av.library_versions.get("libavcodec", ()))
            inputs["video"] = {
                **VIDEO_RULES,
                "decoder": {"lib": "pyav", "version": av.__version__, "libavcodec": libav},
            }
        if "audio" in self.streams:
            inputs["audio"] = {
                **AUDIO_RULES,
                "decoder": {"lib": "soundfile", "version": soundfile.__version__},
            }
        return {
            "weights_sha256": self._weights_sha256,
            "weights_ref": "TODO: path or hub id (provenance only)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": self.reading,  # a | v | av — report --join pairs the three readings of one model
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.args["layer"],
            "pooling": "none",
            "views": 1,
            "input": inputs,
            "precision": {
                "autocast": None,
                "matmul_precision": "highest",
                "allow_tf32": False,
                "deterministic": True,
            },
            "args": dict(self.args),
        }

    # ------------------------------------------------------------------ extraction (instance-wise)
    def prepare(self, sample: Any) -> dict:
        """One clip's CPU work, ahead of ``encode``: decode and preprocess every stream READ; what this returns arrives
        as ``sample.prepared``."""
        # Runs in decode_workers threads ahead of encode(): per-sample decoding only, no model, no state across samples.
        # The stored bytes are released after prepare(), so encode() works from this dict only.
        out: dict[str, Any] = {}
        try:
            if "video" in self.streams:
                out["video"] = self._decode_video(sample.streams["video"])
            if "audio" in self.streams:
                out["audio"] = self._decode_audio(sample.streams["audio"])
        except _Undecodable as exc:  # yavih records the row as missing and encodes the rest of the batch
            raise MediaUndecodable([sample.clip_id], f"{sample.clip_id}: {exc}") from exc
        return out

    def encode(self, samples: list) -> tuple[Any, Any | None, list[dict]]:
        feats, pos, used = [], [], []
        for s in samples:
            f, p, u = self._one(s)
            feats.append(f)
            pos.append(p)
            used.append(u)
        return np.stack(feats).astype(self.dtype), np.stack(pos).astype(np.float32), used

    def _one(self, s: Any) -> tuple[np.ndarray, np.ndarray, dict]:
        toks, pos, used = [], [], {}
        if "video" in self.streams:
            t, p, u = self._video(s.streams["video"], self.streams.index("video"), s.prepared["video"])
            toks.append(t)
            pos.append(p)
            used["video"] = u
        if "audio" in self.streams:
            t, p, u = self._audio(s.streams["audio"], self.streams.index("audio"), s.prepared["audio"])
            toks.append(t)
            pos.append(p)
            used["audio"] = u
        return np.concatenate(toks), np.concatenate(pos), used

    def _decode_video(self, rs: Any) -> dict:
        # validate evaluates the same rules on the same facts
        from yavih.media.rules import rule_times

        times = rule_times(
            VIDEO_RULES["temporal"], rs.info
        )  # uniform bin centres over [0, duration_s]
        frames, pts = _decode_nearest(rs, times)  # uint8 [n, H, W, 3], the decoded frames' pts
        clip = np.stack([_resize_short_center_crop(f, 256, IMG) for f in frames]).astype(np.float32)
        n = VIDEO_RULES["norm"]
        clip = (clip * n["scale"] - np.asarray(n["mean"], np.float32)) / np.asarray(
            n["std"], np.float32
        )
        return {"clip": clip, "pts": pts}

    def _video(self, rs: Any, tag: int, prepared: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        from yavih.media.rules import crop_region

        info = rs.info
        clip, pts = prepared["clip"], prepared["pts"]
        # TODO: replace the placeholder with the model: clip [T, 224, 224, 3] → tokens [T/tubelet * 14 * 14, D]
        g = IMG // PATCH
        x = clip.reshape(N_FRAMES // TUBELET, TUBELET, g, PATCH, g, PATCH, 3).mean(axis=(1, 3, 5))
        tokens = x.reshape(-1, 3) @ self._proj_v
        # token centres in media coordinates: a tubelet's t is the mean of its frames' pts; y/x map the
        # 14 x 14 patch grid into the part of the ORIGINAL frame the centre crop kept
        t_c = pts.reshape(-1, TUBELET).mean(axis=1) / info["duration_s"]
        region = crop_region(VIDEO_RULES["spatial"], info["width"], info["height"])
        p = grid_pos(
            N_FRAMES // TUBELET, g, g, stream=tag, t_centres=np.clip(t_c, 0.0, 1.0), region=region
        )
        return tokens, p, {"times": [float(t) for t in pts]}

    def _decode_audio(self, rs: Any) -> dict:
        import soundfile

        from yavih.media.rules import audio_ranges

        ranges, (pad_l, pad_r) = audio_ranges(
            AUDIO_RULES["temporal"], rs.info
        )  # [[lo, hi]], zeros after a short file
        lo, hi = (int(v) for v in ranges[0])
        with rs.open() as fh:
            wav, sr = soundfile.read(fh, start=lo, stop=hi, dtype="float32", always_2d=True)
        if sr != AUDIO_RULES["front_end"]["sample_rate"]:
            raise _Undecodable(f"sample rate {sr}")
        wav = np.concatenate([np.zeros(pad_l, np.float32), wav[:, 0], np.zeros(pad_r, np.float32)])
        return {"wav": wav, "sr": sr, "lo": lo, "hi": hi, "pad": (pad_l, pad_r)}

    def _audio(self, rs: Any, tag: int, prepared: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        info = rs.info
        wav, sr, lo, hi = prepared["wav"], prepared["sr"], prepared["lo"], prepared["hi"]
        pad_l, pad_r = prepared["pad"]
        # TODO: replace with the model's front end + encoder: wav [160000] → tokens [AUDIO_T * AUDIO_F, D]
        spec = _log_power_stft(wav, 400, 160)  # [frames, 201]
        tf = (
            spec[: spec.shape[0] // AUDIO_T * AUDIO_T, :200]
            .reshape(AUDIO_T, -1, AUDIO_F, 200 // AUDIO_F)
            .mean(axis=(1, 3))
        )
        tf = (tf - AUDIO_RULES["norm"]["mean"]) / AUDIO_RULES["norm"]["std"]
        tokens = tf.reshape(-1, 1).astype(np.float32) @ self._proj_a
        # time-bin centres in media seconds over the fed window [start, start + length); padding → t = -1
        start = lo / sr - pad_l / sr
        centres = start + (np.arange(AUDIO_T) + 0.5) * AUDIO_SECONDS / AUDIO_T
        t = np.where(centres <= info["duration_s"], centres / info["duration_s"], -1.0)
        p = np.repeat(
            np.stack([np.full(AUDIO_T, float(tag)), t, -np.ones(AUDIO_T), -np.ones(AUDIO_T)], 1),
            AUDIO_F,
            axis=0,
        )
        return tokens, p, {"samples": [[lo, hi]], "pad": [int(pad_l), int(pad_r)]}


class _Undecodable(Exception):
    pass


def _decode_nearest(rs: Any, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode the stored mp4 and keep, for each requested time, the frame whose pts is nearest (exact ties →
    the earlier frame), streaming so at most ``len(times)`` frames are held. Returns RGB uint8 + their pts."""
    import av

    best: list[Any] = [None] * len(times)
    best_d = [float("inf")] * len(times)
    try:
        with rs.open() as fh, av.open(fh) as c:
            for fr in c.decode(video=0):
                t = float(fr.time)
                for k, target in enumerate(times):
                    d = round(abs(float(target) - t), 9)
                    if d < best_d[k]:
                        best_d[k], best[k] = d, (t, fr)
    except av.error.FFmpegError as exc:
        raise _Undecodable(f"{type(exc).__name__}: {exc}") from exc
    if any(b is None for b in best):
        raise _Undecodable("no decodable video frame")
    return np.stack([fr.to_ndarray(format="rgb24") for _, fr in best]), np.asarray(
        [t for t, _ in best], np.float64
    )


def _resize_short_center_crop(frame: np.ndarray, short: int, size: int) -> np.ndarray:
    """torchvision's ``Resize(short)`` + ``CenterCrop(size)`` size rules (the ones
    ``yavih.media.rules.crop_region`` assumes): long side ``int(short * long / short_side)``, crop
    offset ``int(round((resized - size) / 2))``. TODO: use the model's own transform instead."""
    from PIL import Image

    h, w = frame.shape[:2]
    if h <= w:
        nh, nw = short, int(short * w / h)
    else:
        nh, nw = int(short * h / w), short
    im = Image.fromarray(frame).resize((nw, nh), Image.BILINEAR)
    top, left = int(round((nh - size) / 2.0)), int(round((nw - size) / 2.0))
    return np.asarray(im)[top : top + size, left : left + size]


def _log_power_stft(wav: np.ndarray, win: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(wav) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = wav[idx] * np.hanning(win).astype(np.float32)
    return np.log(np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-10).astype(np.float32)
