"""Bridge skeleton for **avrobustbench** (AVROBUSTBENCH VGGSOUND-2C). Copy, rename, fill the TODOs.

Read ``setup.md`` first: its input contract is what ``prepare`` and ``encode`` receive, its output contract what must
come back. ``yavih bridge check <file> --bench avrobustbench --synthetic`` smoke-tests it before any data is
staged; an unfilled copy fails bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES.

A bridge file serves one benchmark; each matrix entry (file + args) is one reading of one model. The benchmark is
defined for the joint reading (``reading: "av"``: both streams, one feature set); audio-only (``a``) and video-only
(``v``) readings are legal (``--arg reading=a|v``) and show which modality carries the robustness. One probe is
trained on clean train clips and scored on the clean and the 15 corrupted versions of every eval clip, so the bridge
must treat every row alike: a row's condition is never handed over and must not be inferred from anything but the
media.

Preprocessing source (paste the model's own eval-time input pipeline here, verbatim, with repo, file,
function and commit; ``describe()["input"]`` must match it line by line; ``bridge check`` warns
W_NO_PREPROCESSING_SOURCE while this slot is empty):

    TODO

What ``prepare`` and ``encode`` receive per Sample (``s.streams``), nothing decoded:
    s.streams["video"]  RawStream of the row's materialized mp4: exactly 10 frames of 224 x 224 at 1 fps (pts 0..9 s,
                        duration 10.0 s), H.264 yuv420p, no audio track. The frames are CAV-MAE's (10 frames spread
                        over the VGGSound clip, resize-224 + centre crop), already corrupted for a corrupted row.
                        ``.path`` / ``.open()`` / ``.bytes()``; ``.info`` = {duration_s, fps, n_frames, width,
                        height, container, bytes, ...}.
    s.streams["audio"]  RawStream of the row's wav (16 kHz PCM-16, the VGGSound clip's length: nominal 10 s, ~11 %
                        shorter, down to ~1 s), already corrupted for a corrupted row. Mono, except most ``crowd``
                        rows: their noise recording is stereo, so the wav has 2 channels (each = the clip + one noise
                        channel; read channel 0, as CAV-MAE's Kaldi fbank does). ``.info`` = {duration_s,
                        sample_rate, n_samples, channels, container, bytes}.
    s.meta["facts"]     {} (no label facts).
Everything model-side is decided HERE and declared in ``describe()["input"]``: which of the 10 frames (an image model
takes one or several; the AVROBUSTBENCH reference models CAV-MAE, EquiAV and ImageBind use one frame, UAVM all of
them), the resize to the model's input size (the frames are already 224 px), the audio window and its padding, the
front end (CAV-MAE: 128-bin Kaldi fbank over 1024 frames), the normalization constants (from the model's own
pretraining recipe), the decoder.

Output contract: feats ``[B, N, D]`` with one ``(N, D)`` for every row; ``pos`` optional (``clf`` ignores it) but when
given it is in MEDIA coordinates; ``used`` per clip reports the frame times / sample ranges actually fed, which
``validate`` checks against the declared rules. Instance-wise only: ``bridge check`` hard-checks rerun, batch- and
order-invariance.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, MediaUndecodable, grid_pos

# TODO: the model's geometry. The placeholder reads all 10 frames (tubelet 2) at 224 px with 16 px patches.
N_FRAMES = 10
TUBELET = 2
IMG = 224
PATCH = 16
AUDIO_SECONDS = 10.0  # the audio window the model was trained on (from the start; zeros after a short file)
AUDIO_T, AUDIO_F = 8, 8  # TODO: the audio encoder's token grid (time x frequency)
D = 8  # TODO: the model's feature width (placeholder model below)

VIDEO_RULES = {
    # the 10 stored frames sit at t = 0, 1, ..., 9 s. An image model takes n = 1 at the frame it was evaluated with
    # (AVROBUSTBENCH's dataset.py default frame_num=4: start 4.0).
    "temporal": {"rule": "fps", "fps": 1.0, "n": N_FRAMES, "start": 0.0, "phase": 0.0, "edge": "clamp"},
    "spatial": {
        "resize": {"short": IMG},  # the stored frames are 224 x 224: the identity at IMG = 224
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
        "channels": "channel 0 (2-channel crowd rows: the first)",
    },
    "norm": {
        "mean": 0.0,
        "std": 1.0,
        "source": "TODO: repo@commit file (the model's fixed constants)",
    },
}


class AVRobustBenchBridge(Bridge):
    name = "TODO-model-avrobustbench"  # optional label, recorded only; the reading is an arg, so one file serves a/v/av
    batch_size = 8
    dtype = "float16"
    pos_mode = "per_sample"  # audio durations differ per clip, so pos differs per clip
    streams = ("video", "audio")  # pos tag order; reset per reading in __init__
    ARGS = {
        "reading": ArgSpec(type=str, default="av", choices=("a", "v", "av"), help="which streams the model reads"),
        "layer": ArgSpec(type=int, default=-1, help="block to tap (-1 = last)"),
    }

    def __init__(self, bench: Any, args: dict | None = None) -> None:
        super().__init__(bench, args)
        self.reading = self.args["reading"]
        self.streams = {"a": ("audio",), "v": ("video",), "av": ("video", "audio")}[self.reading]
        # TODO: build the model, load weights BEFORE the first sample, call .eval() yourself:
        # self.model = build_model(...).eval().to("cuda")
        # self._weights_sha256 = yavih.bridge.fingerprint_state_dict(self.model)
        rng = np.random.default_rng(0)  # placeholder "model": fixed projections, so the skeleton runs end to end
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
            inputs["video"] = {**VIDEO_RULES, "decoder": {"lib": "pyav", "version": av.__version__, "libavcodec": libav}}
        if "audio" in self.streams:
            inputs["audio"] = {**AUDIO_RULES, "decoder": {"lib": "soundfile", "version": soundfile.__version__}}
        return {
            "weights_sha256": self._weights_sha256,
            "weights_ref": "TODO: path or hub id (provenance only)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": self.reading,  # a | v | av
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.args["layer"],
            "pooling": "none",
            "views": 1,
            "input": inputs,
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),
        }

    # ------------------------------------------------------------------ extraction (instance-wise)
    def prepare(self, sample: Any) -> dict:
        """One clip's CPU work, ahead of ``encode``: decode and preprocess every stream READ; what this returns arrives
        as ``sample.prepared``."""
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
        from yavih.media.rules import rule_times

        times = rule_times(VIDEO_RULES["temporal"], rs.info)  # 0, 1, ..., 9 s: the 10 stored frames
        frames, pts = _decode_nearest(rs, times)  # uint8 [n, H, W, 3], the decoded frames' pts
        clip = np.stack([_resize_short_center_crop(f, IMG, IMG) for f in frames]).astype(np.float32)
        n = VIDEO_RULES["norm"]
        clip = (clip * n["scale"] - np.asarray(n["mean"], np.float32)) / np.asarray(n["std"], np.float32)
        return {"clip": clip, "pts": pts}

    def _video(self, rs: Any, tag: int, prepared: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        from yavih.media.rules import crop_region

        info = rs.info
        clip, pts = prepared["clip"], prepared["pts"]
        # TODO: replace the placeholder with the model: clip [T, 224, 224, 3] → tokens [T/tubelet * 14 * 14, D]
        g = IMG // PATCH
        x = clip.reshape(N_FRAMES // TUBELET, TUBELET, g, PATCH, g, PATCH, 3).mean(axis=(1, 3, 5))
        tokens = x.reshape(-1, 3) @ self._proj_v
        # token centres in media coordinates: a tubelet's t is the mean of its frames' pts; y/x map the patch grid into
        # the part of the stored frame the crop kept (all of it at IMG = 224)
        t_c = pts.reshape(-1, TUBELET).mean(axis=1) / info["duration_s"]
        region = crop_region(VIDEO_RULES["spatial"], info["width"], info["height"])
        p = grid_pos(N_FRAMES // TUBELET, g, g, stream=tag, t_centres=np.clip(t_c, 0.0, 1.0), region=region)
        return tokens, p, {"times": [float(t) for t in pts]}

    def _decode_audio(self, rs: Any) -> dict:
        import soundfile

        from yavih.media.rules import audio_ranges

        ranges, (pad_l, pad_r) = audio_ranges(AUDIO_RULES["temporal"], rs.info)  # [[lo, hi]], zeros after a short file
        lo, hi = (int(v) for v in ranges[0])
        with rs.open() as fh:
            wav, sr = soundfile.read(fh, start=lo, stop=hi, dtype="float32", always_2d=True)
        # every row of this benchmark is 16 kHz; the stand-in of `bridge check --synthetic` is 8 kHz. A model whose
        # front end needs 16 kHz resamples or refuses here (raise _Undecodable) and declares it in front_end.
        wav = np.concatenate([np.zeros(pad_l, np.float32), wav[:, 0], np.zeros(pad_r, np.float32)])  # channel 0
        return {"wav": wav, "sr": sr, "lo": lo, "hi": hi, "pad": (pad_l, pad_r)}

    def _audio(self, rs: Any, tag: int, prepared: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        info = rs.info
        wav, sr, lo, hi = prepared["wav"], prepared["sr"], prepared["lo"], prepared["hi"]
        pad_l, pad_r = prepared["pad"]
        # TODO: replace with the model's front end + encoder: wav [160000] → tokens [AUDIO_T * AUDIO_F, D]
        spec = _log_power_stft(wav, 400, 160)  # [frames, 201]
        tf = spec[: spec.shape[0] // AUDIO_T * AUDIO_T, :200].reshape(AUDIO_T, -1, AUDIO_F, 200 // AUDIO_F).mean(axis=(1, 3))
        tf = (tf - AUDIO_RULES["norm"]["mean"]) / AUDIO_RULES["norm"]["std"]
        tokens = tf.reshape(-1, 1).astype(np.float32) @ self._proj_a
        # time-bin centres in media seconds over the fed window [start, start + length); padding → t = -1
        start = lo / sr - pad_l / sr
        centres = start + (np.arange(AUDIO_T) + 0.5) * AUDIO_SECONDS / AUDIO_T
        t = np.where(centres <= info["duration_s"], centres / info["duration_s"], -1.0)
        p = np.repeat(np.stack([np.full(AUDIO_T, float(tag)), t, -np.ones(AUDIO_T), -np.ones(AUDIO_T)], 1), AUDIO_F, axis=0)
        return tokens, p, {"samples": [[lo, hi]], "pad": [int(pad_l), int(pad_r)]}


class _Undecodable(Exception):
    pass


def _decode_nearest(rs: Any, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode the stored mp4 and keep, for each requested time, the frame whose pts is nearest (exact ties → the
    earlier frame). Returns RGB uint8 + their pts."""
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
                        best_d[k], best[k] = d, (t, fr.to_ndarray(format="rgb24"))
    except av.error.FFmpegError as exc:
        raise _Undecodable(f"{type(exc).__name__}: {exc}") from exc
    if any(b is None for b in best):
        raise _Undecodable("no decodable video frame")
    return np.stack([f for _, f in best]), np.asarray([t for t, _ in best], np.float64)


def _resize_short_center_crop(frame: np.ndarray, short: int, size: int) -> np.ndarray:
    """torchvision's ``Resize(short)`` + ``CenterCrop(size)`` size rules (the ones ``yavih.media.rules.crop_region``
    assumes). TODO: use the model's own transform instead."""
    from PIL import Image

    h, w = frame.shape[:2]
    if h <= w:
        nh, nw = short, int(short * w / h)
    else:
        nh, nw = int(short * h / w), short
    if (nh, nw) != (h, w):
        frame = np.asarray(Image.fromarray(frame).resize((nw, nh), Image.BILINEAR))
    top, left = int(round((nh - size) / 2.0)), int(round((nw - size) / 2.0))
    return frame[top : top + size, left : left + size]


def _log_power_stft(wav: np.ndarray, win: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(wav) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = wav[idx] * np.hanning(win).astype(np.float32)
    return np.log(np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-10).astype(np.float32)
