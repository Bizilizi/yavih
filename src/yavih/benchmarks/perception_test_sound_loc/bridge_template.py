"""Bridge skeleton for **perception_test_sound_loc** (read setup.md first). Copy, rename, fill the TODOs.

``yavih bridge check <file> --bench perception_test_sound_loc --synthetic`` smoke-tests a copy before any data is
staged; an unfilled copy fails bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES. A bridge file serves one
benchmark; each matrix entry (file + args) is one reading of one model.

The "Preprocessing source" slot below is REQUIRED: paste the model's own input pipeline into it, and make
``describe()["input"]`` match it line by line. ``bridge check`` warns W_NO_PREPROCESSING_SOURCE while every
line of the slot still says TODO.

Preprocessing source: TODO <repo>@<commit> <file>::<function>, e.g. the model's build_eval_transform
    TODO paste the source lines: frame sampling (how many frames, at which times, over which span),
    TODO resize / crop / interpolation, value range and normalization constants (the pretraining recipe's;
    TODO never dataset statistics computed here), audio resampler, fbank/mel parameters, audio length and
    TODO padding, audio normalization statistics.

What Perception Test hands ``encode`` (setup.md "Input contract"): per Sample, two RawStreams over two stored files:

    s.streams["video"].info: duration_s (6.7-36 s, median ~23 s), fps (~25-30, varies), width x height (up to
                             1920x1080, portrait and landscape), container, bytes      -- videos/<id>.mp4
    s.streams["audio"].info: duration_s, sample_rate (44 100 or 48 000: it varies), n_samples, channels (1 or 2),
                             container, bytes
                                                                                         -- audios/<id>.wav
    s.meta["facts"]       == {}

Nothing is decoded, cut, resized or padded for you. Every model-side choice is yours and is declared in
``describe()["input"]``. ``encode`` returns ``(feats, pos, used)``:

    feats  [B, N, D]  the same N and D for every clip (clips last 6.7-36 s: a FIXED token count, spread over each clip)
    pos    [B, N, 4]  (stream, t, y, x) in MEDIA coordinates: t = token centre seconds / that stream's
                      info.duration_s; (y, x) = token centre as a fraction of the ORIGINAL frame; -1 = no axis
    used   B dicts    what you actually fed: {"video": {"times": [pts, ...]},
                                              "audio": {"samples": [[lo, hi], ...], "pad": [left, right]}}

The folder's detection scorer reads ``t``: each token is placed at its absolute second (t * duration_s) relative to the annotated
duration, and the probe predicts, on 128 bins over the clip, which sound classes are active and where each event
starts and ends. Output contract: ``requires: {pos: [t], streams: [audio]}`` — tokens with ``t`` tagged ``audio``
must exist, else ``features validate`` records ``eligible: false`` and the probe writes ``not_applicable`` (a ``v``
reading of this benchmark is therefore not applicable by design: the official task localises SOUNDS).

Time resolution is the point of this benchmark: sound events last 0.13 s (5th percentile) to 5.8 s (95th), median
0.52 s. A bridge that pools time, or reads one token per several seconds, cannot localise them. Spread the tokens
over the WHOLE clip (fixed count), keep their time axis, and report each token's real time.

Gotchas (setup.md "Gotchas"; all pass validation silently):
    * A time-pooled token set (CLS only, learned latent tokens, mean over time) carries no time: give such
      tokens t = -1 and expose per-frame / per-patch tokens.
    * Report t from the frames / samples you actually decoded, not from the times you asked for; ``used`` must
      carry the same pts / sample ranges (E_RULE_MISMATCH tolerance: half a frame + 1 ms).
    * Clips are 1080p: decode at reduced size if the model's transform allows it (PyAV thread_type / scale), and
      give the decoder workers (``--decode-workers``); video decoding dominates extraction time.
    * Normalization constants come from the model's pretraining recipe (``norm.source``); batch statistics break
      batch invariance (bridge check E_BATCH_INVARIANCE).

Readings: ``a`` (the headline, as the official audio-only baseline) and ``av`` (audio + video tokens, as the official
multimodal baseline) are matrix entries of this file under one ``model`` name (``--arg reading=av``); ``v`` has no
audio tokens and is recorded ``not_applicable``. A video-only MODEL raises ``yavih.NotApplicable`` from ``__init__``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, grid_pos

# ---- TODO: the model's choices (placeholders). Every value here is declared in describe()["input"].
N_FRAMES = 32  # frames fed per clip: uniform bin centres over the whole clip (~0.7 s apart on a 23 s clip)
TUBELET = 1  # frames per temporal token (2 for a tubelet-2 video ViT: T_VIDEO = N_FRAMES // 2)
T_VIDEO, H_PATCH, W_PATCH = N_FRAMES // TUBELET, 2, 2  # temporal x spatial patch grid of the video tokens
SIZE = 224  # model input side
SAMPLE_RATE = 16000  # the model's audio rate: every wav is resampled to it (upstream wavs are 44.1 or 48 kHz)
T_AUDIO = 128  # audio tokens over the whole clip (placeholder: ~0.18 s each on a 23 s clip)
D_PLACEHOLDER = 16  # TODO: the model's width (768 for ViT-B); the same D for video and audio tokens. The probe heads need D divisible by 16


class PerceptionTestSoundLocBridge(Bridge):
    name = "TODO-model-perception_test_sound_loc"  # optional label of this file, recorded only; the model's name is describe()["model"]
    batch_size = 8
    dtype = "float16"  # "float16" | "float32" (switch to float32 if bridge check warns about range)
    pos_mode = "per_sample"  # durations and frame sizes differ between clips → per-sample t and region
    streams = ("video", "audio")  # tag order = stream index in pos[:, 0]
    ARGS = {
        "reading": ArgSpec(type=str, default="a", choices=("a", "v", "av"), help="which modality the model sees"),
        "layer": ArgSpec(type=int, default=-1, help="transformer block to tap (-1 = last)"),
    }

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)  # bench is a BenchmarkView: no manifest, labels or splits
        self.reading = self.args["reading"]
        self.streams = {"a": ("audio",), "v": ("video",), "av": ("video", "audio")}[self.reading]
        # TODO: build the model, load weights BEFORE the first sample (no network during encode), call
        # .eval() yourself (run_extract refuses E_TRAIN_MODE):
        #   self.model = build_model(...).eval().to("cuda")
        #   self._weights_sha256 = yavih.bridge.fingerprint_state_dict(self.model)
        self.model = None
        self._weights_sha256 = None  # TODO: the real fingerprint; None is recorded as W_WEIGHTS_UNFINGERPRINTED

    # ------------------------------------------------------------------ declared configuration

    def _input(self) -> dict:
        """The declared rules, one entry per stream READ (recorded; validate re-evaluates them: E_RULE_MISMATCH)."""
        decoder = _decoder_info()
        out: dict[str, dict] = {}
        if "video" in self.streams:
            out["video"] = {
                # TODO: the model's frame sampling. "uniform" over span "full" = bin centres of [0, duration_s].
                "temporal": {"rule": "uniform", "n": N_FRAMES, "span": "full", "edge": "clamp"},
                # TODO: the model's resize/crop (the placeholder is what _preprocess_video implements).
                "spatial": {"resize": {"short": SIZE}, "crop": {"kind": "center", "size": [SIZE, SIZE]},
                            "interp": "nearest", "antialias": False},
                "norm": {"scale": 1 / 255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                         "source": "TODO: <repo>@<commit> <file> (the model's pretraining recipe)"},
                "decoder": decoder,
            }
        if "audio" in self.streams:
            out["audio"] = {
                # the whole clip, no cap and no padding: the token count is fixed, each token spans duration / T_AUDIO
                "temporal": {"rule": "full", "max_s": None, "pad_to_s": None},
                "front_end": {"sample_rate": SAMPLE_RATE, "resample": "TODO: the model's resampler (placeholder: linear, np.interp)",
                              "features": "TODO (placeholder: log mean power per time patch)", "channels": "mean"},
                "norm": {"mean": 0.0, "std": 1.0, "source": "TODO: <repo>@<commit> <file> (pretraining recipe)"},
                "decoder": decoder,
            }
        return out

    def describe(self) -> dict:
        # Every key below is REQUIRED; extra keys are allowed and recorded (never identity).
        return {
            "weights_sha256": self._weights_sha256,
            "weights_ref": "TODO: path or hub id (provenance only, not hashed)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": self.reading,
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.args["layer"],
            "pooling": "none",  # keep the time axis: pooling over time makes the set useless for detection
            "views": 1,  # MUST be 1: one deterministic view per clip
            "input": self._input(),
            # DECLARE what you run with; run_extract enforces it (E_PRECISION_UNDECLARED on a null).
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),  # echoes every consumed --arg with defaults filled
        }

    # ------------------------------------------------------------------ extraction (instance-wise)

    def prepare(self, sample: Any) -> dict:
        """One clip's CPU work, ahead of ``encode``: decode every stream READ; the result arrives as ``sample.prepared``."""
        from yavih.media import decode, rules  # yavih[media]; or decode with the model's own stack

        rules_in = self._input()
        out: dict[str, Any] = {}
        try:
            if "video" in self.streams:
                v = sample.streams["video"]
                times = rules.rule_times(rules_in["video"]["temporal"], v.info)
                frames, pts = decode.frames_at(v, times)  # uint8 [n, H, W, 3] native size, decoded pts
                out["video"] = {"x": _preprocess_video(frames), "pts": pts}  # TODO: the model's transform
            if "audio" in self.streams:
                a = sample.streams["audio"]
                ranges, (pad_l, pad_r) = rules.audio_ranges(rules_in["audio"]["temporal"], a.info)
                wav, sr = decode.audio(a, ranges)  # float32 [L, C] at the native rate (PCM wav: exactly n_samples)
                wav = wav.mean(axis=1)  # channels "mean" (declared)
                wav, sr = _resample(wav, sr, SAMPLE_RATE)  # TODO: the model's resampler (declared in front_end)
                out["audio"] = {
                    "wav": wav,
                    "sr": sr,
                    "used": {"samples": [[int(lo), int(hi)] for lo, hi in ranges], "pad": [int(pad_l), int(pad_r)]},
                }
        except Exception as exc:  # noqa: BLE001 — TODO: narrow to your decoder's error types
            if _is_decode_error(exc):  # run_extract records the row as missing and encodes the rest
                from yavih import MediaUndecodable

                raise MediaUndecodable([sample.clip_id], "decoder failed") from exc
            raise
        return out

    def encode(self, samples: list) -> tuple[Any, Any, list[dict]]:
        """One clip at a time: nothing in a clip's features may depend on the other clips of the batch."""
        from yavih.media import rules

        rules_in = self._input()
        feats, pos, used = [], [], []
        for s in samples:
            p = s.prepared
            toks, tpos, u = [], [], {}
            if "video" in self.streams:
                v, pts = s.streams["video"], p["video"]["pts"]
                toks.append(self._video_model(p["video"]["x"]))  # [T_VIDEO * H_PATCH * W_PATCH, D]
                region = rules.crop_region(rules_in["video"]["spatial"], v.info["width"], v.info["height"])
                t_c = pts.reshape(T_VIDEO, TUBELET).mean(axis=1) / float(v.info["duration_s"])
                tpos.append(grid_pos(T_VIDEO, H_PATCH, W_PATCH, stream=self.streams.index("video"),
                                     t_centres=np.clip(t_c, 0.0, 1.0), region=region))
                u["video"] = {"times": [float(t) for t in pts]}
            if "audio" in self.streams:
                q = p["audio"]
                wav, sr = q["wav"], q["sr"]
                toks.append(self._audio_model(wav, sr))  # [T_AUDIO, D]
                dur = float(s.streams["audio"].info["duration_s"])
                centres = (np.arange(T_AUDIO) + 0.5) * (len(wav) / sr) / T_AUDIO  # media seconds
                ap = grid_pos(T_AUDIO, 1, 1, stream=self.streams.index("audio"))
                ap[:, 1] = np.clip(centres / dur, 0.0, 1.0)
                ap[:, 2:] = -1.0  # a spectrogram patch has no place in the frame
                tpos.append(ap)
                u["audio"] = q["used"]
            feats.append(np.concatenate(toks, axis=0))
            pos.append(np.concatenate(tpos, axis=0).astype(np.float32))
            used.append(u)
        return np.stack(feats).astype(self.dtype), np.stack(pos), used

    # ---- TODO: replace both with the model (frozen, under the inference_mode run_extract sets up).

    def _video_model(self, x: np.ndarray) -> np.ndarray:
        """Placeholder: per-patch mean colour (+ a zero column) → ``[T_VIDEO * H_PATCH * W_PATCH, D]``."""
        n, h, w, c = x.shape
        t = x.reshape(T_VIDEO, TUBELET, H_PATCH, h // H_PATCH, W_PATCH, w // W_PATCH, c).mean(axis=(1, 3, 5))
        t = t.reshape(-1, c)
        return np.concatenate([t, np.zeros((t.shape[0], D_PLACEHOLDER - c), np.float32)], axis=1)

    def _audio_model(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Placeholder: log mean power per time patch → ``[T_AUDIO, D]``."""
        chunks = np.array_split(wav.astype(np.float64), T_AUDIO)
        lp = np.array([np.log(np.mean(ch**2) + 1e-10) if ch.size else -23.0 for ch in chunks], np.float32)[:, None]
        return np.concatenate([lp, np.zeros((T_AUDIO, D_PLACEHOLDER - 1), np.float32)], axis=1)

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}


def _preprocess_video(frames: np.ndarray) -> np.ndarray:
    """Placeholder for the declared spatial rule: resize short side → SIZE (nearest), centre crop SIZE², /255,
    mean/std. TODO: replace with the model's own transform (and update describe()["input"])."""
    n, h, w, _ = frames.shape
    scale = SIZE / min(h, w)
    nh, nw = max(SIZE, round(h * scale)), max(SIZE, round(w * scale))
    y0, x0 = (nh - SIZE) // 2, (nw - SIZE) // 2
    yi = np.minimum(((np.arange(y0, y0 + SIZE) + 0.5) / scale).astype(np.int64), h - 1)
    xi = np.minimum(((np.arange(x0, x0 + SIZE) + 0.5) / scale).astype(np.int64), w - 1)
    x = frames[:, yi][:, :, xi].astype(np.float32) / 255.0
    return (x - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])


def _resample(wav: np.ndarray, sr: int, target: int) -> tuple[np.ndarray, int]:
    """Placeholder resampler (linear interpolation onto the target grid). TODO: the model's own."""
    if int(sr) == int(target) or wav.size == 0:
        return wav.astype(np.float32), int(sr)
    n_out = int(round(wav.size * target / sr))
    t_out = (np.arange(n_out) + 0.5) / target
    t_in = (np.arange(wav.size) + 0.5) / sr
    return np.interp(t_out, t_in, wav).astype(np.float32), int(target)


def _decoder_info() -> dict:
    try:
        from yavih.media.decode import decoder_info

        return decoder_info()
    except Exception:  # noqa: BLE001 — TODO: {"lib": ..., "version": ...} of the decoder you use
        return {"lib": "TODO", "version": "TODO"}


def _is_decode_error(exc: Exception) -> bool:
    """TODO: True only for corrupt/undecodable media (e.g. av.error.InvalidDataError); a bug must raise."""
    return type(exc).__module__.startswith("av.")
