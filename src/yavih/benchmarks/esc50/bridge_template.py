"""Bridge skeleton for the ``esc50`` benchmark.

Copy this file next to your model code, fill every ``TODO``, then run
``yavih bridge check <file> --bench esc50 --out esc50.profile.json`` before any extraction
(``--synthetic`` instead of ``--out`` smoke-tests it before any data is staged). An unfilled copy fails
bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES. A bridge file serves one benchmark; each matrix
entry (file + args) is one reading of one model. Read ``setup.md`` ("Input contract", "Output contract",
"Gotchas") first.

Preprocessing source of the model (paste the model's own audio front end VERBATIM here, with the
repository, commit and file:line it came from; ``describe()["input"]["audio"]`` must match it line by line):

    TODO (e.g. `torchaudio.compliance.kaldi.fbank(wav16k, htk_compat=True, sample_frequency=16000,
    use_energy=False, window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)`, then the
    model's PRETRAINING-DATASET mean/std, then how the model's input length is reached: AudioMAE sizes
    ESC input to 512 frames; AST/BEATs/CAV-MAE pad to 1024 frames …)

What the reader hands you per sample (setup.md "Input contract"):

* ``sample.streams["audio"]`` — the ONLY stream, a ``RawStream`` over the stored wav: ``path``,
  ``bytes()``, ``open()`` (seekable; soundfile / torchaudio / PyAV accept it). Nothing is decoded,
  resampled, trimmed or padded for you.
* ``stream.info`` (guaranteed keys): ``duration_s`` 5.0, ``sample_rate`` 44100, ``n_samples`` 220500,
  ``channels`` 1, ``container`` "wav", ``bytes``. Use these facts, never a hard-coded constant, so a
  re-staged corpus cannot silently change what you feed.
* ``sample.meta == {"facts": {}}``: ESC-50 has no label facts. Never let ``row``, ``clip_id`` or
  ``split`` (the fold) change what you compute: ``bridge check`` scrambles them.
* Folds: the extractor runs this bridge once per fold split ``"1"…"5"`` and five probes reuse the
  features; your features must not depend on the fold or on the batch a clip lands in.

Everything model-side is YOUR decision and is declared in ``describe()["input"]`` (recorded with every
number, never part of what makes numbers comparable): which samples (temporal rule), resampling, padding,
front end, normalization constants, decoder.

Guardrails: instance-wise extraction (no batch or dataset statistics, no padding to the batch maximum,
normalization constants from the model's own pretraining recipe; per-clip statistics only when they ARE
the model's own recipe, then declared in ``norm`` with their ``source``); one deterministic view; no cross-sample state; frozen weights, ``.eval()``;
fixed ``N`` and ``D``; no network and no writes outside the run directory inside ``encode``.

Gotchas (setup.md "Gotchas") — each passes validation and silently changes the number:

* **Padding to a longer model input.** Zero-padding the 5 s clip to 8–10 s makes >= 3/8 of the input
  digital silence and skews per-clip normalization. If your model needs a longer input, choose the fill DELIBERATELY: ``{"rule": "full", "pad_to_s": 10.0}`` declares zero
  padding (checked); any other fill (repeat, reflect) is a ``custom`` rule with a description.
  Normalize as the model was pretrained: its own constants; per-clip statistics (even over the padded
  buffer) only when that IS its recipe, declared in ``norm``; never batch or dataset statistics.
* Resample 44 100 -> the model rate yourself and declare the resampler; never assume 16 kHz input.
* Do not re-window unless that IS the model's documented ESC recipe (then declare a ``span`` rule).
* Return the same ``N`` for every sample (fixed-length input -> fixed token count).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge


class Esc50Bridge(Bridge):
    name = "TODO-model-esc50"       # optional label of this file, recorded only
    batch_size = 32
    dtype = "float16"               # "float16" | "float32" — what encode() returns
    pos_mode = "none"               # clf needs no pos (output_contract.requires.pos == []); "constant" + _time_pos() is allowed
    streams = ("audio",)            # stream tags used in pos; the benchmark's only stream
    ARGS = {"layer": ArgSpec(type=int, default=-1, help="the tap (-1 = last block)")}  # every --arg k=v; echoed in describe()["args"]

    # audio temporal rule (closed vocabulary; validate re-evaluates it on every row's info):
    #   {"rule": "full", "max_s": None, "pad_to_s": None}   the whole stored clip, nothing added (default here)
    #   {"rule": "full", "max_s": 10.0, "pad_to_s": 10.0}   whole clip, right zero-padded to 10 s (AST/BEATs style)
    #   {"rule": "span", "start_s": "centre", "length_s": 4.0, "pad": "none"}   a centred 4 s cut
    #   {"rule": "custom", "description": "..."}            anything else (repeat padding): a legitimate opt-out,
    #                                                       noted by validate (N_RULE_CUSTOM), never penalized
    TEMPORAL: dict = {"rule": "full", "max_s": None, "pad_to_s": None}   # TODO

    def __init__(self, bench: Any, args: dict | None = None) -> None:
        super().__init__(bench, args)
        self.layer = int(self.args["layer"])   # parsed + defaulted by Bridge.__init__ from ARGS
        # TODO: load the model and its weights BEFORE the first sample; call .eval() YOURSELF (run_extract
        # refuses any torch.nn.Module left in training mode: E_TRAIN_MODE). Raise
        # yavih.NotApplicable("...") here if the model cannot take audio (a video-only model).
        self.model = None
        self.model_sr = 16000            # TODO: the model's sample rate (resample 44100 -> model_sr below)
        self.n_t, self.n_f = 1, 1        # TODO: the model's time × frequency token grid (n_t * n_f == N)

    def describe(self) -> dict:
        return {
            "weights_sha256": None,  # TODO: yavih.bridge.fingerprint_state_dict(self.model); null is recorded as W_WEIGHTS_UNFINGERPRINTED
            "weights_ref": "TODO: hub id or path (provenance only)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it ([A-Za-z0-9_.-]{1,64})
            "reading": "a",  # a | v | av: one vocabulary across benchmarks (run dir, results key and report row: model + reading)
            "layer": self.layer,
            "pooling": "none",       # "none" keeps [N, D] tokens for the attentive probe; "mean"/"cls" give N = 1
            "views": 1,              # MUST be 1: one deterministic view per clip
            "input": {
                "audio": {
                    "temporal": dict(self.TEMPORAL),
                    "front_end": {   # TODO: match the pasted preprocessing source line by line
                        "sample_rate": self.model_sr,
                        "resample": "TODO e.g. torchaudio.functional.resample(sinc_interp_hann, lowpass_filter_width 6)",
                        "features": "TODO e.g. kaldi_fbank",
                        "n_mels": None, "win_ms": None, "hop_ms": None, "dither": 0.0,
                        "channels": "mean",          # ESC-50 is mono; say what you would do with more channels
                    },
                    "norm": {"mean": None, "std": None,   # TODO: the model's pretraining constants (per-clip stats only if that IS its recipe)
                             "source": "TODO repo@commit file:line"},
                    "decoder": self._decoder_info(),
                },
            },
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": {"layer": self.layer},
        }

    @staticmethod
    def _decoder_info() -> dict:
        import soundfile as sf  # or yavih.media.decode.decoder_info() when you decode with yavih's helpers

        return {"lib": "soundfile", "version": sf.__version__, "libsndfile": sf.__libsndfile_version__}

    def _read(self, stream: Any) -> tuple[np.ndarray, int, dict]:
        """Decode exactly what TEMPORAL declares, at the native rate, and report it as ``used``."""
        import soundfile as sf

        # core (numpy only): the same evaluation of the declared rule that validate uses
        from yavih.media.rules import audio_ranges

        ranges, (pad_l, pad_r) = audio_ranges(self.TEMPORAL, stream.info)
        sr = int(stream.info["sample_rate"])
        parts = []
        with stream.open() as fh, sf.SoundFile(fh) as f:
            assert f.samplerate == sr, (f.samplerate, sr)
            for lo, hi in ranges.tolist():
                f.seek(lo)
                parts.append(f.read(hi - lo, dtype="float32", always_2d=True))
        wav = np.concatenate(parts, axis=0).mean(axis=1)            # [L] mono
        wav = np.pad(wav, (int(pad_l), int(pad_r)))                  # zeros, as the rule declares
        used = {"samples": [[int(lo), int(hi)] for lo, hi in ranges.tolist()], "pad": [int(pad_l), int(pad_r)]}
        return wav.astype(np.float32, copy=False), sr, used

    def prepare(self, sample: Any) -> dict:
        """One clip's CPU work, ahead of ``encode``: what this returns arrives as ``sample.prepared``."""
        # Runs in decode_workers threads ahead of encode(): per-sample decoding only, no model, no state across samples.
        wav, sr, u = self._read(sample.streams["audio"])
        # TODO: resample sr (44100) -> self.model_sr here too (per clip, deterministic); the stored bytes are released
        # after prepare(), so encode() works from this dict only.
        return {"wav": wav, "sr": sr, "used": {"audio": u}}

    def encode(self, samples: list) -> tuple[Any, Any | None, list[dict]]:
        waves = [s.prepared["wav"] for s in samples]   # what prepare() returned, per sample
        used = [s.prepared["used"] for s in samples]
        # Every clip has the same length here because every stored clip is 220 500 samples; never pad
        # to the batch maximum (instance-wise extraction).
        # TODO: front end, forward (run_extract wraps encode in torch.inference_mode()), take the tap
        # `self.layer`, return tokens [B, N, D] as np.ndarray or a torch tensor; N fixed for every clip.
        _ = waves
        feats = np.zeros((len(samples), self.n_t * self.n_f, 1), dtype=np.float32)  # TODO replace
        pos = None
        if self.pos_mode == "constant":
            info = samples[0].streams["audio"].info   # constant pos is legal: every ESC-50 clip is 5.0 s
            pos = self._time_pos(float(info["duration_s"]))
        return feats, pos, used

    def _time_pos(self, duration_s: float) -> np.ndarray:
        """``(stream, t, y, x)`` for time-major tokens (all n_f frequency patches of step 0, then step 1, …)
        in media coordinates: t = token centre / duration_s of the stored clip, -1 for a token
        centred in padding you added; audio has no frame, so y = x = -1. Assumes the model input starts
        at sample 0 (rule ``full``); adapt the centres for a ``span`` rule."""
        model_seconds = duration_s  # TODO: the seconds the model input covers (pad_to_s when you pad)
        centres = (np.arange(self.n_t) + 0.5) * model_seconds / self.n_t
        t = np.where(centres < duration_s, centres / duration_s, -1.0)
        pos = np.full((self.n_t * self.n_f, 4), -1.0, dtype=np.float32)
        pos[:, 0] = 0.0
        pos[:, 1] = np.repeat(t, self.n_f)
        return pos

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": float}
