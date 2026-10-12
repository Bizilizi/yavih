"""The one shipped reference bridge: a public HF audio model on ESC-50, run by ``yavih doctor --selftest`` on 32
clips.

torch / transformers / scipy are imported at call time (``__init__``), never at module scope, so
``import yavih.bridges.examples.hf_reference`` is cheap and the ``[media]`` + torch requirement
surfaces as a clear error only when the bridge is actually constructed.

The reader hands over the stored wav; this bridge decodes it itself (``yavih.media.decode.audio``),
declares its rules in ``describe()["input"]`` (the whole track, ``full``; polyphase resampling to the model's
rate; the HF feature extractor's per-clip normalization), reports ``used`` (the native-rate sample range it
fed), and runs ONE clip per forward pass, so no clip is ever padded to the batch maximum (instance-wise
extraction; ``bridge check`` enforces batch- and order-invariance).

``REFERENCE_MODEL`` is a public hub id that needs no torchaudio (its feature extractor only normalizes the
waveform), so the bridge is runnable in any ``[media]`` + torch + transformers env. ``REFERENCE_ACC`` is not pinned
(``None``): ``doctor --selftest`` reports the measured number without a pass/fail verdict; a pinned value (5-fold
mean acc under ``ProbeConfig.pinned("clf")``) would be checked within ``REFERENCE_TOL``.

Preprocessing source:
    transformers Wav2Vec2FeatureExtractor (preprocessor_config.json of facebook/wav2vec2-base):
    do_normalize=True -> x = (x - mean(x)) / sqrt(var(x) + 1e-7) per clip; sampling_rate=16000; no padding here
    (one clip per call).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge, NotApplicable, fingerprint_state_dict, stream_kinds

__all__ = [
    "HFReferenceBridge",
    "SELFTEST_BENCH",
    "SELFTEST_N",
    "REFERENCE_MODEL",
    "REFERENCE_ACC",
    "REFERENCE_TOL",
    "TARGET_SAMPLE_RATE",
    "BRIDGE",
    "model_name",
]

SELFTEST_BENCH = "esc50"
SELFTEST_N = 32
#: Public HF hub id used as the reference.
REFERENCE_MODEL = "facebook/wav2vec2-base"
#: 5-fold ESC-50 acc of this bridge under ``ProbeConfig.pinned("clf")``; ``None``: not pinned, no verdict.
REFERENCE_ACC: float | None = None
REFERENCE_TOL: float = 0.05
TARGET_SAMPLE_RATE = 16000  # wav2vec2-family models are trained at 16 kHz
_HF_IMPORT_HINT = "pip install 'yavih[media,probe]' transformers"


def model_name(model_ref: str) -> str:
    """``describe()["model"]`` of this bridge (the name results are filed under,
    ``[A-Za-z0-9_.-]{1,64}``): ``hf-`` + the hub id with every other character as ``_``, so two checkpoints are
    two names (two report rows) and a layer change under one checkpoint is one name (``W_BRIDGE_CHANGED``)."""
    import re

    return ("hf-" + re.sub(r"[^A-Za-z0-9_.-]", "_", str(model_ref)))[:64]


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling (scipy, part of ``[media]``); identity when the rates agree.
    The bridge resamples, the reader never does."""
    if sr_in == sr_out:
        return wav.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    out = resample_poly(wav.astype(np.float64), int(sr_out) // g, int(sr_in) // g)
    return np.ascontiguousarray(out, dtype=np.float32)


class HFReferenceBridge(Bridge):
    """Mean-pooled last hidden state of a public HF speech/audio encoder over the ESC-50 clip
    (``N = 1``, ``pos_mode = none``): the smallest honest audio reading, enough for a clf probe."""

    name = "hf-reference-esc50"
    batch_size = 8
    dtype = "float32"
    pos_mode = "none"
    streams = ("audio",)
    ARGS = {
        "model": ArgSpec(type=str, default=REFERENCE_MODEL, help="HF hub id of the reference audio model"),
        "layer": ArgSpec(type=str, default="last_hidden_state", help="'last_hidden_state' or an int index into hidden_states"),
        "device": ArgSpec(type=str, default="auto", help="'auto' | 'cpu' | 'cuda'"),
    }
    #: The ``full`` temporal rule: the whole stored track, never cut or padded (ESC-50 clips are all 5.0 s).
    TEMPORAL = {"rule": "full", "max_s": None, "pad_to_s": None}

    def __init__(self, bench: Any, args: dict | None = None) -> None:
        super().__init__(bench, args)
        if stream_kinds(bench).get("audio") != "audio":
            # one bridge serves one benchmark: without an audio stream named "audio" it is not applicable
            # (yavih records NOT_APPLICABLE.json)
            raise NotApplicable(f"audio-only bridge: benchmark {getattr(bench, 'name', '?')!r} has no stream 'audio' of kind audio")
        self.model_ref: str = self.args["model"]
        self.torch, self.model, self.extractor = self._load_model(self.model_ref, self.args["device"])
        self.model.eval()  # the bridge, never yavih, puts the model in eval mode (else E_TRAIN_MODE)
        self.weights_sha256 = fingerprint_state_dict(self.model)
        self.device = next(self.model.parameters()).device if any(True for _ in self.model.parameters()) else self.torch.device("cpu")

    # Separated so a test can monkeypatch it with a tiny in-memory model: no download in tests.
    @staticmethod
    def _load_model(model_ref: str, device: str = "auto") -> tuple[Any, Any, Any]:
        try:
            import torch
        except ImportError as e:  # pragma: no cover - torch is part of [probe]
            raise ImportError(f"hf_reference needs torch: {_HF_IMPORT_HINT}") from e
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as e:
            raise ImportError(f"hf_reference needs transformers: {_HF_IMPORT_HINT}") from e
        model = AutoModel.from_pretrained(model_ref)
        extractor = AutoFeatureExtractor.from_pretrained(model_ref)
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        model.to(dev)
        return torch, model, extractor

    def describe(self) -> dict:
        sr = int(getattr(self.extractor, "sampling_rate", TARGET_SAMPLE_RATE))
        do_norm = bool(getattr(self.extractor, "do_normalize", False))
        try:
            from yavih.media.decode import decoder_info

            decoder = decoder_info("audio")
        except Exception:  # noqa: BLE001 - [media] absent: encode would fail anyway; describe stays declarable
            decoder = {"lib": "soundfile", "version": "unavailable"}
        return {
            "weights_sha256": self.weights_sha256,
            "weights_ref": self.model_ref,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "model": model_name(self.model_ref),  # the name results are filed under
            "reading": "a",  # a | v | av
            "layer": self.args["layer"],
            "pooling": "mean",
            "views": 1,
            "input": {
                "audio": {
                    "temporal": dict(self.TEMPORAL),
                    "front_end": {"sample_rate": sr, "resample": "scipy.signal.resample_poly (polyphase, gcd ratio)",
                                  "features": "raw waveform -> HF feature extractor", "channels": "mean", "padding": "none (one clip per forward)"},
                    "norm": {"per_clip_zero_mean_unit_var": do_norm, "eps": 1e-7,
                             "source": f"{self.model_ref} preprocessor_config.json (transformers Wav2Vec2FeatureExtractor, do_normalize)"},
                    "decoder": decoder,
                }
            },
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),
        }

    def _embed(self, wav: np.ndarray, sr_target: int) -> Any:
        """One clip → ``[1, D]`` (no batch padding: instance-wise extraction)."""
        inputs = self.extractor([wav], sampling_rate=sr_target, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        layer = self.args["layer"]
        want_hidden = layer != "last_hidden_state"
        out = self.model(**inputs, output_hidden_states=want_hidden)
        hidden = out.hidden_states[int(layer)] if want_hidden else out.last_hidden_state  # [1, T, D]
        return hidden.float().mean(dim=1)  # [1, D]

    def encode(self, samples: list) -> tuple[Any, Any | None, list[dict]]:
        from yavih.media.decode import audio
        from yavih.media.rules import audio_ranges

        sr_target = int(getattr(self.extractor, "sampling_rate", TARGET_SAMPLE_RATE))
        feats, used = [], []
        for s in samples:
            st = s.streams["audio"]
            ranges, (pad_l, pad_r) = audio_ranges(self.TEMPORAL, st.info)
            wav, sr = audio(st, ranges)  # float32 [L, C] at the native rate
            mono = np.asarray(wav, dtype=np.float32).mean(axis=1)
            feats.append(self._embed(_resample(mono, int(sr), sr_target), sr_target))
            used.append({"audio": {"samples": [[int(a), int(b)] for a, b in ranges], "pad": [int(pad_l), int(pad_r)]}})
        out = self.torch.stack(feats, dim=0)  # [B, 1, D]
        return out.detach().cpu(), None, used

    def profile(self) -> dict | None:
        torch = self.torch
        if torch.cuda.is_available():
            return {"peak_gpu_gb": float(torch.cuda.max_memory_allocated()) / 1e9}
        return None


BRIDGE = HFReferenceBridge
