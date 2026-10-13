"""Bridge skeleton for **audioset20k** (setup.md). Copy, rename, fill the TODOs, then
``yavih bridge check <file> --bench audioset20k --n 32 --arg reading=av`` before any extraction
(``--synthetic`` smoke-tests it before any data is staged). An unfilled copy fails bridge check:
E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES.

A bridge file serves one benchmark; each matrix entry (file + args) is one reading of one model. AS-20K is
usually read three ways — audio only (``a``), video only (``v``), both (``av``) — here three matrix entries of
this one file (``--arg reading=…``) under one ``model`` name.
Every reading is scored against the SAME 527 audio-event labels (a video reading predicts sound events).

Preprocessing source: TODO <repo>@<commit> <file>::<function>, e.g. the model's build_eval_transform
    TODO paste the source lines: frame sampling (how many frames, at which times, over which span),
    TODO resize / crop / interpolation, value range and normalization constants (the pretraining recipe's,
    TODO never dataset statistics computed here), audio resampler, fbank/mel parameters, audio length and
    TODO padding, audio normalization statistics.

What ``prepare`` and ``encode`` receive per Sample (setup.md "## Input contract"; nothing is decoded for you): two RawStreams
over the SAME published mp4 (one member of a dataset tar, read in place; ``RawStream.open()`` gives a seekable
file object, so PyAV/torchaudio decode it without extracting anything):

    s.streams["video"].info: duration_s (nominal 10 s; 0.73-63.73 s), fps (6-60), n_frames, width x height
                             (128x120 ... 3840x2160; mostly 4:3 and 16:9, some portrait and square),
                             container ("mp4"), bytes.
                             Codecs: h264 (most), av1, vp9 — your decoder must handle all three.
    s.streams["audio"].info: duration_s (0.02-10.02 s; about 2 500 clips shorter than 9.9 s), sample_rate (48 000,
                             44 100 or 32 000), n_samples, channels (2; some 1, five 6), container, bytes.
                             Codecs: AAC, Opus, E-AC-3.
    s.meta == {"facts": {}}  ``row`` / ``clip_id`` / ``split`` MUST NOT change what you compute.

THREE CLIPS HAVE NO AUDIO TRACK (train/NlBQ4kLpdzk_0260000, train/d8WgfWSf1VM_0280000,
eval/chc33hy6iIU_0510000; setup.md "## Input contract"). They are handed over like every other row (the same
mp4, video track only): ``s.streams["audio"].info["has_track"] is False`` and its track facts (duration_s,
sample_rate, n_samples, channels) are None. An audio or av reading must still return one fixed-shape feature
row for them; what it feeds is YOUR choice, declared in ``describe()["no_track"]`` (below: the declared span
of zeros through the same front end, ``used`` = ``{"samples": [], "pad": [0, 0]}``, audio tokens at t = -1).
Do not decode them (``yavih.media.decode.audio`` raises DecodeError) and do not raise MediaUndecodable for
them (that would drop the row).

The video track can run far longer than the audio track (up to 63.7 s against at most 10.02 s): "the whole
clip" and "the first 10 s" are different declared rules. Everything model-side is yours and is declared in
``describe()["input"]``: the temporal rule per stream, resize + crop, the audio front end
(resampler, features, channels), the normalization constants with their source, and the decoder. The defaults
below are an 8 × 1 s geometry: 8 frames at 0.5, 1.5, …, 7.5 s (clamped to the clip for the few
shorter than 8 s) and the first 8 s of audio, zero-padded at the end for short clips. When the vocabulary does
not fit your model's recipe, declare ``{"rule": "custom", "description": "<one line>"}``: that is a legitimate
opt-out (validate records the note N_RULE_CUSTOM; out-of-media and position checks still apply).

``encode`` returns ``(feats, pos, used)``:

    feats  [B, N, D]  the same N and D for every clip of both splits
    pos    [B, N, 4]  (stream, t, y, x) in MEDIA coordinates: t = token centre seconds / that stream's
                      info.duration_s; (y, x) = token centre as a fraction of the ORIGINAL frame (before your
                      resize/crop: grid_pos(region=crop_region(...)) does the mapping); -1 = no axis
    used   B dicts    what you actually fed: {"video": {"times": [pts, ...]},
                                              "audio": {"samples": [[lo, hi], ...], "pad": [left, right]}}

``clf`` needs no positions (``requires.pos: []``); positions you do write are checked (E_POS_RANGE,
E_RULE_MISMATCH). Guardrails: one deterministic view; normalization constants from the model's own pretraining
recipe (per-clip statistics only when they ARE the model's own recipe: then declare them in ``norm`` with their
``source``; never batch or dataset statistics); no cross-sample state; ``.eval()`` the model yourself; declare precision. Record the
real ``weights_sha256`` (a null one is only the warning W_WEIGHTS_UNFINGERPRINTED, but it is what tells two
checkpoints apart in the record).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, grid_pos

# ---- TODO: the model's choices (placeholders: a ViT-B/16-like video encoder at 224 with tubelet 1 and an audio
#      encoder with 1 s time patches). Every value here is declared in describe()["input"].
N_FRAMES = 8  # frames fed per clip
VIDEO_TEMPORAL = {"rule": "uniform", "n": N_FRAMES, "span": [0.0, 8.0], "edge": "clamp"}  # 0.5, 1.5, …, 7.5 s
SIZE = 224  # model input side
VIDEO_SPATIAL = {"resize": {"short": SIZE}, "crop": {"kind": "center", "size": [SIZE, SIZE]},
                 "interp": "nearest", "antialias": False}  # TODO: the model's transform (see _preprocess_video)
TUBELET = 1  # frames per temporal token (2 for a tubelet-2 video ViT)
T_VIDEO, H_PATCH, W_PATCH = N_FRAMES // TUBELET, 14, 14  # temporal x spatial patch grid of the video tokens
AUDIO_TEMPORAL = {"rule": "span", "start_s": 0.0, "length_s": 8.0, "pad": "zeros"}  # the first 8 s, zero-padded
SAMPLE_RATE = 16000  # the model's rate; the clips are 48 / 44.1 / 32 kHz, so the bridge resamples (declared)
T_AUDIO = 8  # audio time patches over the 8 s input (placeholder: one per second)
D_PLACEHOLDER = 4  # TODO: the model's width (768 for ViT-B); the same D for video and audio tokens
NO_TRACK = {  # what is fed for a stream whose file holds no such track (info["has_track"] is False), declared in describe()
    "audio": "length_s of zeros at the model's sample_rate through the same front end; used samples [] pad [0, 0]; audio tokens t = -1",
    "video": "N_FRAMES black frames (zeros) of SIZE x SIZE through the same transform; used times []; video tokens t = -1",
}


class AudioSet20kBridge(Bridge):
    name = "TODO-model-audioset20k"  # optional label of this file (recorded only); the model is describe()["model"]
    batch_size = 8
    dtype = "float16"  # "float16" | "float32" (switch to float32 if bridge check warns about range)
    pos_mode = "per_sample"  # durations and frame sizes differ between clips → per-sample t and region
    streams = ("video", "audio")  # tag order = stream index in pos[:, 0]
    ARGS = {
        "reading": ArgSpec(type=str, default="av", choices=("a", "v", "av"), help="which modality the model sees"),
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap (-1 = last)"),
    }

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)  # bench is a BenchmarkView: no manifest, labels or splits
        self.reading = self.args["reading"]
        self.streams = {"a": ("audio",), "v": ("video",), "av": ("video", "audio")}[self.reading]
        # TODO: build the model, load weights BEFORE the first sample (no network during encode), call .eval()
        # yourself (run_extract refuses E_TRAIN_MODE):
        #   self.model = build_model(...).eval().to("cuda")
        #   self._weights_sha256 = yavih.bridge.fingerprint_state_dict(self.model)
        self.model = None
        self._weights_sha256 = None  # TODO: the real fingerprint (null is recorded as W_WEIGHTS_UNFINGERPRINTED)

    # ------------------------------------------------------------------ declaration

    def _input(self) -> dict:
        """The declared rules, one entry per stream READ. They are recorded with the features and validate
        re-evaluates them on every member's media facts (E_RULE_MISMATCH)."""
        out: dict[str, dict] = {}
        if "video" in self.streams:
            out["video"] = {
                "temporal": VIDEO_TEMPORAL,
                "spatial": VIDEO_SPATIAL,
                "norm": {"scale": 1 / 255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                         "source": "TODO: <repo>@<commit> <file> (the model's pretraining recipe)"},
                "decoder": _decoder_info("video"),
            }
        if "audio" in self.streams:
            out["audio"] = {
                "temporal": AUDIO_TEMPORAL,
                "front_end": {"sample_rate": SAMPLE_RATE, "resample": "numpy.interp linear (TODO: the model's resampler)",
                              "features": "TODO (placeholder: log mean power per 1 s patch)", "channels": "mean",
                              "fit": "decoded samples cut / zero-filled to the declared ranges (AAC/Opus priming)"},
                "norm": {"mean": 0.0, "std": 1.0, "source": "TODO: <repo>@<commit> <file> (pretraining recipe)"},
                "decoder": _decoder_info("audio"),
            }
        return out

    def describe(self) -> dict:
        # Every key below is REQUIRED; extra keys are allowed and recorded.
        return {
            "model": "TODO-model",  # the name all of this model's bridges use ([A-Za-z0-9_.-]{1,64}); results are filed under it
            "reading": self.reading,  # "a" | "v" | "av": the run subdirectory and the results key <benchmark>:<reading>
            "weights_sha256": self._weights_sha256,
            "weights_ref": "TODO: path or hub id (provenance only)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "layer": self.args["layer"],
            "pooling": "none",  # "none" keeps [N, D] tokens for the attentive probe; "mean"/"cls" → N = 1
            "views": 1,  # one deterministic view per clip (MUST be 1)
            "input": self._input(),
            # TODO: your choice for a clip whose file holds no track of a stream you read (info["has_track"] False;
            # 3 AS-20K clips have no audio track). Recorded, covered by describe_sha256.
            "no_track": {name: NO_TRACK[name] for name in self.streams},
            # DECLARE what you run with; run_extract enforces it (E_PRECISION_UNDECLARED on a null).
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),  # echoes every consumed --arg with defaults filled
        }

    # ------------------------------------------------------------------ extraction (instance-wise)

    def prepare(self, sample: Any) -> dict:
        """One clip's CPU work, ahead of ``encode``: decode and fit every stream READ; what this returns arrives as
        ``sample.prepared`` (``None`` for a stream whose file holds no such track)."""
        # Runs in decode_workers threads ahead of encode(): per-sample decoding only, no model, no state across samples.
        # The stored bytes are released after prepare(), so encode() works from this dict only.
        from yavih.media import decode, rules  # yavih[media]; or decode with the model's own stack

        rules_in = self._input()
        out: dict[str, Any] = {}
        try:
            if "video" in self.streams and sample.streams["video"].info.get("has_track") is False:  # no video track
                out["video"] = None
            elif "video" in self.streams:
                v = sample.streams["video"]
                times = rules.rule_times(rules_in["video"]["temporal"], v.info)
                frames, pts = decode.frames_at(v, times)  # uint8 [n, H, W, 3] native size, decoded pts
                out["video"] = {"x": _preprocess_video(frames), "pts": pts}  # TODO: the model's transform → model input
            if "audio" in self.streams and sample.streams["audio"].info.get("has_track") is False:  # no audio track
                out["audio"] = None
            elif "audio" in self.streams:
                a = sample.streams["audio"]
                ranges, (pad_l, pad_r) = rules.audio_ranges(rules_in["audio"]["temporal"], a.info)
                wav, sr = decode.audio(a, ranges)  # float32 [L, C] at the native rate
                wav = wav.mean(axis=1)  # channels "mean" (declared)
                # AAC/Opus decode to a few hundred samples more or fewer than the header count in info
                # (priming/padding). Fit the decoded audio to the declared ranges so what is fed matches
                # `used` (declared in front_end["fit"]).
                want = int(sum(int(hi) - int(lo) for lo, hi in ranges))
                wav = np.concatenate([wav[:want], np.zeros(max(0, want - len(wav)), np.float32)])
                wav = np.concatenate([np.zeros(int(pad_l), np.float32), wav, np.zeros(int(pad_r), np.float32)])
                out["audio"] = {
                    "x": _resample(wav, sr, SAMPLE_RATE),  # the model's input rate
                    "fed_s": len(wav) / sr,  # seconds fed, padding included
                    "start": (int(ranges[0][0]) - int(pad_l)) / sr,  # media second of the first fed sample
                    "used": {"samples": [[int(lo), int(hi)] for lo, hi in ranges], "pad": [int(pad_l), int(pad_r)]},
                }
        except Exception as exc:  # noqa: BLE001 — TODO: narrow to your decoder's error types
            if _is_decode_error(exc):  # run_extract records the row as missing and encodes the rest
                from yavih import MediaUndecodable

                raise MediaUndecodable([sample.clip_id], "decoder failed") from exc
            raise
        return out

    def encode(self, samples: list) -> tuple[Any, Any, list[dict]]:
        """One clip at a time: nothing in a clip's features may depend on the other clips of the batch
        (bridge check: rerun bit-identity, batch- and order-invariance)."""
        from yavih.media import rules

        rules_in = self._input()
        feats, pos, used = [], [], []
        for s in samples:
            p = s.prepared  # what prepare() returned for this sample
            toks, tpos, u = [], [], {}
            if "video" in self.streams and p["video"] is None:  # no video track
                toks.append(self._video_model(_preprocess_video(np.zeros((N_FRAMES, SIZE, SIZE, 3), np.uint8))))
                vp = grid_pos(T_VIDEO, H_PATCH, W_PATCH, stream=self.streams.index("video"))
                vp[:, 1] = -1.0  # no media time to point at
                tpos.append(vp)
                u["video"] = {"times": []}
            elif "video" in self.streams:
                v, pts = s.streams["video"], p["video"]["pts"]
                toks.append(self._video_model(p["video"]["x"]))  # [T_VIDEO * H_PATCH * W_PATCH, D]
                region = rules.crop_region(rules_in["video"]["spatial"], v.info["width"], v.info["height"])
                t_c = pts.reshape(T_VIDEO, TUBELET).mean(axis=1) / float(v.info["duration_s"])
                tpos.append(grid_pos(T_VIDEO, H_PATCH, W_PATCH, stream=self.streams.index("video"),
                                     t_centres=np.clip(t_c, 0.0, 1.0), region=region))
                u["video"] = {"times": [float(t) for t in pts]}
            if "audio" in self.streams and p["audio"] is None:  # no audio track
                silence = np.zeros(int(round(float(AUDIO_TEMPORAL["length_s"]) * SAMPLE_RATE)), np.float32)
                toks.append(self._audio_model(silence))  # [T_AUDIO, D], the declared NO_TRACK["audio"]
                ap = grid_pos(T_AUDIO, 1, 1, stream=self.streams.index("audio"))
                ap[:, 1:] = -1.0  # no media time, no place in the frame
                tpos.append(ap)
                u["audio"] = {"samples": [], "pad": [0, 0]}
            elif "audio" in self.streams:
                q = p["audio"]
                toks.append(self._audio_model(q["x"]))  # [T_AUDIO, D]
                centres = q["start"] + (np.arange(T_AUDIO) + 0.5) * q["fed_s"] / T_AUDIO  # media seconds
                dur = float(s.streams["audio"].info["duration_s"])
                ap = grid_pos(T_AUDIO, 1, 1, stream=self.streams.index("audio"))
                ap[:, 1] = np.where((centres >= 0) & (centres < dur), centres / dur, -1.0)  # padding → -1
                ap[:, 2:] = -1.0  # a spectrogram patch has no place in the frame
                tpos.append(ap)
                u["audio"] = q["used"]
            feats.append(np.concatenate(toks, axis=0))
            pos.append(np.concatenate(tpos, axis=0).astype(np.float32))
            used.append(u)
        return np.stack(feats).astype(self.dtype), np.stack(pos), used

    # ---- TODO: replace both with the model (frozen, under the inference_mode run_extract sets up).

    def _video_model(self, x: np.ndarray) -> np.ndarray:
        """Placeholder: per-patch mean colour (+ zero columns) → ``[T_VIDEO * H_PATCH * W_PATCH, D]``."""
        n, h, w, c = x.shape
        t = x.reshape(T_VIDEO, TUBELET, H_PATCH, h // H_PATCH, W_PATCH, w // W_PATCH, c).mean(axis=(1, 3, 5))
        t = t.reshape(-1, c)
        return np.concatenate([t, np.zeros((t.shape[0], D_PLACEHOLDER - c), np.float32)], axis=1)

    def _audio_model(self, wav: np.ndarray) -> np.ndarray:
        """Placeholder: log mean power per time patch → ``[T_AUDIO, D]``."""
        chunks = np.array_split(wav.astype(np.float64), T_AUDIO)
        lp = np.array([np.log(np.mean(ch**2) + 1e-10) for ch in chunks], np.float32)[:, None]
        return np.concatenate([lp, np.zeros((T_AUDIO, D_PLACEHOLDER - 1), np.float32)], axis=1)

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}


def _preprocess_video(frames: np.ndarray) -> np.ndarray:
    """Placeholder for the declared spatial rule: resize short side → SIZE (nearest), centre crop SIZE², /255,
    mean/std. TODO: replace with the model's own transform (and update VIDEO_SPATIAL)."""
    n, h, w, _ = frames.shape
    scale = SIZE / min(h, w)
    nh, nw = max(SIZE, round(h * scale)), max(SIZE, round(w * scale))
    y0, x0 = (nh - SIZE) // 2, (nw - SIZE) // 2
    yi = np.minimum(((np.arange(y0, y0 + SIZE) + 0.5) / scale).astype(np.int64), h - 1)
    xi = np.minimum(((np.arange(x0, x0 + SIZE) + 0.5) / scale).astype(np.int64), w - 1)
    x = frames[:, yi][:, :, xi].astype(np.float32) / 255.0
    return (x - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])


def _resample(wav: np.ndarray, sr: int, target: int) -> np.ndarray:
    """Placeholder resampler (linear interpolation, deterministic). TODO: the model's own resampler."""
    if sr == target:
        return wav
    n = int(round(len(wav) * target / sr))
    return np.interp(np.arange(n) * (sr / target), np.arange(len(wav)), wav).astype(np.float32)


def _decoder_info(kind: str) -> dict:
    try:
        from yavih.media.decode import decoder_info

        return decoder_info(kind)
    except Exception:  # noqa: BLE001 — TODO: {"lib": ..., "version": ...} of the decoder you use
        return {"lib": "TODO", "version": "TODO"}


def _is_decode_error(exc: Exception) -> bool:
    """TODO: True only for corrupt/undecodable media (e.g. av.error.InvalidDataError); a bug must raise."""
    from yavih.media import DecodeError

    return isinstance(exc, DecodeError) or type(exc).__module__.startswith("av.")
