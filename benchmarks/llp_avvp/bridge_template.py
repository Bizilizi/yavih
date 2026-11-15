"""Bridge skeleton for **llp_avvp** (read setup.md first). Copy, rename, fill the TODOs.

``yavih bridge check <file> --bench llp_avvp`` checks a copy on the staged clips (``--synthetic`` is refused,
E_SYNTHETIC: the core's generated stand-in has no family for this folder's own scorer); an unfilled copy fails
bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES. A bridge file serves one
benchmark; each matrix entry (file + args) is one reading of one model.

The "Preprocessing source" slot below is REQUIRED: paste the model's own input pipeline into it, and make
``describe()["input"]`` match it line by line. ``bridge check`` warns W_NO_PREPROCESSING_SOURCE while every
line of the slot still says TODO.

Preprocessing source: TODO <repo>@<commit> <file>::<function>, e.g. the model's build_eval_transform
    TODO paste the source lines: frame sampling (how many frames, at which times, over which span),
    TODO resize / crop / interpolation, value range and normalization constants (the pretraining recipe's;
    TODO per-clip statistics only if they ARE that recipe; never dataset statistics computed here), audio
    TODO resampler, fbank/mel parameters, audio length and padding, audio normalization statistics.

What LLP hands ``encode`` (setup.md "Input contract"): per Sample, two RawStreams over the SAME stored mp4
(llp_videos/<ytid>.mp4; ``RawStream.open()`` gives a seekable file object, so PyAV/torchaudio decode it in place):

    s.streams["video"].info: duration_s (6.8-10.2 s; 683 clips are shorter than 9.9 s), fps, width x height
                             (mixed resolutions and aspect ratios), container, bytes
    s.streams["audio"].info: duration_s, sample_rate, n_samples, channels, container, bytes (AAC)
    s.meta["facts"]       == {"num_segments": 10, "segment_seconds": 1.0}   (labels are per media second)

Nothing is decoded, cut, resized or padded for you. Every model-side choice is yours and is declared in
``describe()["input"]``: the temporal rule per stream (which frames / which samples), the
spatial rule (resize + crop), the audio front end, the normalization constants with their source, and the
decoder. ``encode`` returns ``(feats, pos, used)``:

    feats  [B, N, D]  the same N and D for every clip
    pos    [B, N, 4]  (stream, t, y, x) in MEDIA coordinates: t = token centre seconds / that stream's
                      info.duration_s; (y, x) = token centre as a fraction of the ORIGINAL frame, or -1;
                      -1 = no axis
    used   B dicts    what you actually fed: {"video": {"times": [pts, ...]},
                                              "audio": {"samples": [[lo, hi], ...], "pad": [left, right]}}

How the scorer reads them: every token goes to the media second ``floor(t * duration_s)`` of its own stream; the
tokens of a stream are averaged per second (seconds 0-9), and the ten audio and ten video means are what the LLP
authors' HAN readout sees (it is trained here, on the train clips' video-level labels). A second with no token of a
stream reads zeros; ``(y, x)`` is not used. Output contract: ``requires: {pos: [t], streams: [], t_bins: 8}``: on
the median clip at least 8 of the 10 seconds must hold a timed token (of any stream), else ``features validate``
records ``eligible: false``.

The defaults below are the LLP authors' own features (scripts/extract_frames.py, extract_rgb_feat.py,
extract_audio.py of YapengTian/AVVP-ECCV20): 80 frames per clip (8 per second) at 224 x 224 (short side 256, centre
crop, ImageNet mean/std), one pooled feature per frame (ResNet-152's global pool), and the audio at 16 kHz mono in
ten one-second windows (VGGish's 0.96 s examples). They are a reading of the reference, not protocol: use the
model's own.

Gotchas (setup.md "Gotchas"; all pass validation silently):
    * A time-pooled token set (CLS only, learned memory/latent tokens, mean over time) carries no per-second
      information: give such tokens t = -1 and expose per-frame / per-window tokens.
    * Report t from the frames you actually decoded (pts / duration_s), not from the times you asked for;
      ``used`` must carry the same pts (E_RULE_MISMATCH tolerance: half a frame + 1 ms).
    * 683 clips last less than 9.9 s (6.8 s at the least) and the two tracks differ in length: never assume
      10 s. Audio tokens centred in padding you added carry t = -1.
    * Mixed resolutions and aspect ratios: a centre crop keeps a different region per clip (pos is per_sample).
    * Normalization constants come from the model's pretraining recipe; per-clip statistics only when they
      ARE that recipe (declare them in ``norm`` with their ``source``); batch statistics break batch
      invariance (bridge check E_BATCH_INVARIANCE).
    * Every LLP clip is an AudioSet segment: a model pretrained on AudioSet may have seen the test clips.
      Disclose it next to any number.

LLP parses audio AND visual events: write the ``av`` reading as the headline; ``a`` and ``v`` are further matrix
entries of this file under one ``model`` name (``--arg reading=a`` / ``v``). All three are eligible: a reading that
hands over one stream feeds it to both of HAN's branches, which then guess the other modality's events from it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, grid_pos

# ---- TODO: the model's choices (placeholders: the LLP authors' feature recipe, a frame encoder pooled per frame and
#      an audio encoder over 1 s windows). Every value here is declared in describe()["input"].
N_FRAMES = 80  # frames fed per clip: uniform bin centres over the whole clip, 8 per second (the authors' ffmpeg -r 8)
TUBELET = 1  # frames per temporal token (2 for a tubelet-2 video ViT: T_VIDEO = N_FRAMES // 2)
T_VIDEO, H_PATCH, W_PATCH = N_FRAMES // TUBELET, 14, 14  # temporal x spatial patch grid of the video tokens
RESIZE_SHORT, SIZE = 256, 224  # short side, then the centre crop the model sees (the authors' ResNet-152 transform)
AUDIO_RATE = 16000  # the audio front end's rate (the authors' VGGish input: moviepy write_audiofile(fps=16000))
AUDIO_SECONDS = 10.0  # audio fed: [0, min(duration, 10 s)) right-padded with zeros to 10 s
T_AUDIO = 10  # audio tokens: one per 1 s window (VGGish: 0.96 s examples)
D_PLACEHOLDER = 4  # TODO: the model's width (2048 for ResNet-152); the same D for video and audio tokens


class LLPAVVPBridge(Bridge):
    name = "TODO-model-llp_avvp"  # optional label of this file, recorded only; the model's name is describe()["model"]
    batch_size = 8
    dtype = "float16"  # "float16" | "float32" (switch to float32 if bridge check warns about range)
    pos_mode = "per_sample"  # durations differ between LLP clips → per-sample t
    streams = ("video", "audio")  # tag order = stream index in pos[:, 0]
    ARGS = {
        "reading": ArgSpec(type=str, default="av", choices=("a", "v", "av"), help="which modality the model sees"),
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
        """The declared rules, one entry per stream READ. They are recorded and validate re-evaluates them
        on every member's media facts (E_RULE_MISMATCH)."""
        decoder = _decoder_info()
        out: dict[str, dict] = {}
        if "video" in self.streams:
            out["video"] = {
                # TODO: the model's frame sampling. "uniform" over span "full" = bin centres of [0, duration_s];
                # a fixed-rate window: {"rule": "fps", "fps": F, "n": N, "start": "centre_clamped", "phase": 0, "edge": "clamp"}.
                # index_linspace needs n_frames, which this benchmark's input_contract.info does not guarantee (E_INPUT).
                "temporal": {"rule": "uniform", "n": N_FRAMES, "span": "full", "edge": "error"},
                # TODO: the model's resize/crop. The placeholder below is what _preprocess_video implements
                # (nearest-neighbour index sampling); replace both with the model's own transform.
                "spatial": {"resize": {"short": RESIZE_SHORT}, "crop": {"kind": "center", "size": [SIZE, SIZE]},
                            "interp": "nearest", "antialias": False},
                "norm": {"scale": 1 / 255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                         "source": "TODO: <repo>@<commit> <file> (the model's pretraining recipe)"},
                "decoder": decoder,
            }
        if "audio" in self.streams:
            out["audio"] = {
                "temporal": {"rule": "full", "max_s": AUDIO_SECONDS, "pad_to_s": AUDIO_SECONDS},
                "front_end": {"sample_rate": AUDIO_RATE, "resample": "linear interpolation (TODO: the model's resampler)",
                              "features": f"TODO (placeholder: log mean power of {T_AUDIO} one-second windows)", "channels": "mean",
                              "fit": "decoded samples cut / zero-filled to the declared ranges (AAC priming)"},
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
            # keep the time axis (pooling over time leaves the probe nothing to localize); space is pooled per frame
            "pooling": "video: mean over the 14 x 14 patches of each frame (time kept); audio: none",  # time must stay
            "views": 1,  # MUST be 1: one deterministic view per clip
            "input": self._input(),
            # DECLARE what you run with; run_extract enforces it (E_PRECISION_UNDECLARED on a null).
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),  # echoes every consumed --arg with defaults filled
        }

    # ------------------------------------------------------------------ extraction (instance-wise)

    def prepare(self, sample: Any) -> dict:
        """One clip's CPU work, ahead of ``encode``: decode and fit every stream READ; what this returns arrives as
        ``sample.prepared``."""
        # Runs in decode_workers threads ahead of encode(): per-sample decoding only, no model, no state across samples.
        # The stored bytes are released after prepare(), so encode() works from this dict only.
        from yavih.media import decode, rules  # yavih[media]; or decode with the model's own stack

        rules_in = self._input()
        out: dict[str, Any] = {}
        try:
            if "video" in self.streams:
                v = sample.streams["video"]
                times = rules.rule_times(rules_in["video"]["temporal"], v.info)
                frames, pts = decode.frames_at(v, times)  # uint8 [n, H, W, 3] native size, decoded pts
                out["video"] = {"x": _preprocess_video(frames), "pts": pts}  # TODO: the model's transform → model input
            if "audio" in self.streams:
                a = sample.streams["audio"]
                ranges, (pad_l, pad_r) = rules.audio_ranges(rules_in["audio"]["temporal"], a.info)
                wav, sr = decode.audio(a, ranges)  # float32 [L, C] at the native rate
                wav = wav.mean(axis=1)  # channels "mean" (declared)
                # AAC decodes to a few hundred samples more or fewer than the header count in info
                # (priming/padding). Fit the decoded audio to the declared
                # ranges so what is fed matches `used` (declared in front_end["fit"]).
                want = int(sum(int(hi) - int(lo) for lo, hi in ranges))
                wav = np.concatenate([wav[:want], np.zeros(max(0, want - len(wav)), np.float32)])
                wav = np.concatenate([np.zeros(int(pad_l), np.float32), wav, np.zeros(int(pad_r), np.float32)])
                wav = _resample(wav, sr, AUDIO_RATE)  # TODO: the model's resampler (declared in front_end)
                out["audio"] = {
                    "wav": wav,
                    "sr": AUDIO_RATE,
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
        feats, pos, used = [], [], []
        for s in samples:
            p = s.prepared  # what prepare() returned for this sample
            toks, tpos, u = [], [], {}
            if "video" in self.streams:
                v, pts = s.streams["video"], p["video"]["pts"]
                patches = self._video_model(p["video"]["x"])  # [T_VIDEO * H_PATCH * W_PATCH, D]
                toks.append(patches.reshape(T_VIDEO, H_PATCH * W_PATCH, -1).mean(axis=1))  # [T_VIDEO, D] (pooling)
                t_c = pts.reshape(T_VIDEO, TUBELET).mean(axis=1) / float(v.info["duration_s"])
                vp = grid_pos(T_VIDEO, 1, 1, stream=self.streams.index("video"), t_centres=np.clip(t_c, 0.0, 1.0))
                vp[:, 2:] = -1.0  # a frame-pooled token has no place in the frame
                tpos.append(vp)
                u["video"] = {"times": [float(t) for t in pts]}
            if "audio" in self.streams:
                q = p["audio"]
                wav, sr = q["wav"], q["sr"]
                toks.append(self._audio_model(wav, sr))  # [T_AUDIO, D]
                centres = q["start"] + (np.arange(T_AUDIO) + 0.5) * (len(wav) / sr) / T_AUDIO  # media seconds
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
        """Placeholder: per-patch mean colour (+ a zero column) → ``[T_VIDEO * H_PATCH * W_PATCH, D]``."""
        n, h, w, c = x.shape
        t = x.reshape(T_VIDEO, TUBELET, H_PATCH, h // H_PATCH, W_PATCH, w // W_PATCH, c).mean(axis=(1, 3, 5))
        t = t.reshape(-1, c)
        return np.concatenate([t, np.zeros((t.shape[0], D_PLACEHOLDER - c), np.float32)], axis=1)

    def _audio_model(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Placeholder: log mean power per time patch → ``[T_AUDIO, D]``."""
        chunks = np.array_split(wav.astype(np.float64), T_AUDIO)
        lp = np.array([np.log(np.mean(ch**2) + 1e-10) for ch in chunks], np.float32)[:, None]
        return np.concatenate([lp, np.zeros((T_AUDIO, D_PLACEHOLDER - 1), np.float32)], axis=1)

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}


def _resample(wav: np.ndarray, sr: int, rate: int) -> np.ndarray:
    """Placeholder resampler: linear interpolation from ``sr`` to ``rate``. TODO: the model's own."""
    if int(sr) == int(rate) or len(wav) == 0:
        return wav.astype(np.float32)
    n = int(round(len(wav) * rate / sr))
    return np.interp(np.arange(n) * (sr / rate), np.arange(len(wav)), wav).astype(np.float32)


def _preprocess_video(frames: np.ndarray) -> np.ndarray:
    """Placeholder for the declared spatial rule: resize short side → RESIZE_SHORT (nearest), centre crop SIZE²,
    /255, mean/std. TODO: replace with the model's own transform (and update describe()["input"])."""
    n, h, w, _ = frames.shape
    scale = RESIZE_SHORT / min(h, w)
    nh, nw = max(SIZE, round(h * scale)), max(SIZE, round(w * scale))
    y0, x0 = (nh - SIZE) // 2, (nw - SIZE) // 2
    yi = np.minimum(((np.arange(y0, y0 + SIZE) + 0.5) / scale).astype(np.int64), h - 1)
    xi = np.minimum(((np.arange(x0, x0 + SIZE) + 0.5) / scale).astype(np.int64), w - 1)
    x = frames[:, yi][:, :, xi].astype(np.float32) / 255.0
    return (x - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])


def _decoder_info() -> dict:
    try:
        from yavih.media.decode import decoder_info

        return decoder_info()
    except Exception:  # noqa: BLE001 — TODO: {"lib": ..., "version": ...} of the decoder you use
        return {"lib": "TODO", "version": "TODO"}


def _is_decode_error(exc: Exception) -> bool:
    """TODO: True only for corrupt/undecodable media (e.g. av.error.InvalidDataError); a bug must raise."""
    return type(exc).__module__.startswith("av.")
