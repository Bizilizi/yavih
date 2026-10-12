"""Bridge skeleton for the ``fsd50k`` benchmark.

Copy this file next to your model code, fill every ``TODO``, then run
``yavih bridge check <file> --bench fsd50k`` before any extraction (``--synthetic`` smoke-tests it
before any data is staged). An unfilled copy fails bridge check: E_TEMPLATE_UNFILLED /
E_CONSTANT_FEATURES. A bridge file serves one benchmark; each matrix entry (file + args) is one reading
of one model. Read ``setup.md`` ("Input contract", "Output contract", "Gotchas") first.

Preprocessing source of the model (paste the model's own audio front end VERBATIM here, with the
repository, commit and file:line it came from; ``describe()["input"]["audio"]`` must match it line by
line — ``bridge check`` warns ``W_NO_PREPROCESSING_SOURCE`` while this slot is empty):

    TODO (e.g. `torchaudio.compliance.kaldi.fbank(waveform16k, htk_compat=True, sample_frequency=16000,
    use_energy=False, window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)` then
    `(fbank - mean) / (2 * std)` with the model's own pretraining constants …)

What the reader hands you per sample (setup.md "Input contract"):

* ``sample.streams["audio"]`` — the ONLY stream, a ``RawStream``: ``open()`` / ``bytes()`` are the
  stored FSD50K wav byte for byte (RIFF WAV, PCM16, 44 100 Hz, mono). ``info`` holds the header facts
  ``duration_s``, ``sample_rate`` (44100), ``n_samples``, ``channels`` (1), ``container`` ("wav"),
  ``bytes``. Durations run 0.300–64.133 s: 67 % of the rows are shorter than 10 s, 33 % longer.
* ``sample.meta == {"facts": {}}`` — FSD50K has no label facts for the bridge.
* Nothing is decoded, resampled, cut or padded for you. Which samples you read, the resampler, the
  padding of short clips, the front end and the normalization constants are YOUR decisions; declare
  every one of them in ``describe()["input"]["audio"]`` (recorded) and report what you
  actually read in ``used`` (validate re-evaluates the declared ``temporal`` rule on each row's facts:
  ``E_RULE_MISMATCH`` when they differ by more than one sample).

Output (setup.md "Output contract"): ``feats [B, N, D]`` with ONE ``(N, D)`` for every row of both
splits; ``clf`` ignores ``pos``, so ``pos_mode = "none"`` is the default. If you emit time positions
(``pos_mode = "per_sample"``: durations differ, so ``constant`` is wrong), they are media coordinates:
``t = token centre in seconds / info["duration_s"]``, ``-1`` for a token centred in padding you added,
``y = x = -1`` (audio has no space; ``yavih.grid_pos`` / ``crop_region`` are for video and image
streams and are not used here).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge

# The declared temporal rule (``span``): read [START_S, START_S + LENGTH_S) of the stored
# file and zero-pad the shortfall at the end. TODO: set it to what the model was pretrained on (AST /
# PaSST: 10 s). Alternatives: ``{"rule": "full", "max_s": …, "pad_to_s": …}`` or
# ``{"rule": "segments", …}`` — whatever you pick, report the same in ``used``.
START_S = 0.0
LENGTH_S = 10.0
N_TOKENS = 10  # TODO: the model's token count over LENGTH_S (placeholder: one token per second)


class Fsd50kBridge(Bridge):
    name = "TODO-model-fsd50k"      # optional label of this file, recorded only
    batch_size = 16
    dtype = "float16"               # "float16" | "float32" — what encode() returns
    pos_mode = "none"               # clf needs no pos (output_contract.requires.pos == []); "per_sample" if you keep time tokens
    streams = ("audio",)            # stream tags used in pos; the benchmark's only stream
    ARGS = {"layer": ArgSpec(type=int, default=-1, help="the tap (-1 = last block)")}  # every --arg k=v; echoed in describe()["args"]

    def __init__(self, bench: Any, args: dict) -> None:
        # ``bench`` is a BenchmarkView: name, streams, contracts, facts — no manifest, labels or splits.
        super().__init__(bench, args)
        self.layer = int(self.args.get("layer", -1))
        # TODO: load the model and its weights HERE (no network, no writes outside the run dir during
        # encode); call .eval() YOURSELF (run_extract refuses a torch.nn.Module left in
        # training mode: E_TRAIN_MODE). A video-only model raises yavih.NotApplicable("...") here.
        self.model = None
        self.model_sr = 16000   # TODO: the model's sample rate (you resample 44100 -> model_sr in encode)

    # ------------------------------------------------------------------ declared configuration
    def describe(self) -> dict:
        import soundfile as sf  # the decoder this template uses; replace with your own stack if you prefer

        return {
            "weights_sha256": None,  # TODO: yavih.bridge.fingerprint_state_dict(self.model); null is recorded as W_WEIGHTS_UNFINGERPRINTED
            "weights_ref": "TODO: hub id or path (provenance only)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": "a",  # a | v | av: one vocabulary across benchmarks (run dir, results key and report row: model + reading)
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "layer": self.layer,
            "pooling": "none",       # "none" keeps [N, D] tokens for the attentive probe; "mean"/"cls" give N = 1
            "views": 1,              # MUST be 1: one deterministic view, no multi-crop / TTA averaging
            "input": {
                "audio": {
                    "temporal": {"rule": "span", "start_s": START_S, "length_s": LENGTH_S, "pad": "zeros"},
                    "front_end": {
                        "sample_rate": self.model_sr,
                        "resample": "TODO e.g. torchaudio.functional.resample(sinc_interp_hann, lowpass_filter_width=6)",
                        "features": "TODO e.g. kaldi_fbank",
                        "n_mels": None,          # TODO
                        "win_ms": None,          # TODO
                        "hop_ms": None,          # TODO
                        "dither": 0.0,           # MUST be 0 (deterministic, rerun bit-identity)
                        "channels": "as stored (FSD50K is mono)",
                        "value_range": "PCM16 / 32768 -> float32 [-1, 1)",
                    },
                    "norm": {
                        "mean": None,            # TODO the model's pretraining constants (per-clip stats only if that IS its recipe; never FSD50K statistics)
                        "std": None,
                        "source": "TODO repo@commit path:line of the model's recipe",
                    },
                    "decoder": {"lib": "soundfile", "version": sf.__version__, "libsndfile": sf.__libsndfile_version__},
                },
            },
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": {"layer": self.layer},
        }

    # ------------------------------------------------------------------ extraction
    def _read(self, sample: Any) -> tuple[np.ndarray, dict]:
        """Decode exactly the declared ``span`` of one stored wav: float32 ``[round(LENGTH_S·sr)]`` at the
        NATIVE rate (zero-padded at the end) and its ``used`` entry. Reads only the needed samples."""
        import soundfile as sf

        st = sample.streams["audio"]
        sr, n = int(st.info["sample_rate"]), int(st.info["n_samples"])
        lo = int(round(START_S * sr))
        length = int(round(LENGTH_S * sr))
        hi = min(lo + length, n)
        try:
            with st.open() as fh:
                wav, got_sr = sf.read(fh, start=lo, stop=hi, dtype="float32", always_2d=True)
        except Exception as exc:  # corrupt/truncated media: a missing row, not a crashed shard
            from yavih import MediaUndecodable

            raise MediaUndecodable([sample.clip_id], f"soundfile: {exc}") from exc
        if got_sr != sr or wav.shape[0] != hi - lo:
            from yavih import MediaUndecodable

            raise MediaUndecodable([sample.clip_id], f"header says {sr} Hz/{n} samples, decoded {got_sr} Hz/{wav.shape[0]} in [{lo}, {hi})")
        mono = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]  # FSD50K is mono; declared under front_end.channels
        out = np.zeros(length, dtype=np.float32)
        out[: hi - lo] = mono
        return out, {"audio": {"samples": [[lo, hi]], "pad": [0, length - (hi - lo)]}}

    def encode(self, samples: list) -> tuple[Any, Any | None, list[dict]]:
        # Decode in parallel within the CPU budget run_extract grants; soundfile releases the GIL.
        workers = max(1, int(getattr(self, "decode_workers", 1)))
        if workers > 1 and len(samples) > 1:
            with ThreadPoolExecutor(min(workers, len(samples))) as ex:
                decoded = list(ex.map(self._read, samples))
        else:
            decoded = [self._read(s) for s in samples]
        waves = np.stack([w for w, _ in decoded])          # [B, round(LENGTH_S * 44100)] at 44.1 kHz
        used = [u for _, u in decoded]
        # Gotchas (setup.md):
        #  - resample 44100 -> self.model_sr with a proper filter; never decimate by slicing;
        #  - normalise as the model was pretrained: its own constants; per-clip statistics only when they ARE
        #    its recipe (declared in norm with their source) — over a mostly-zero padded clip (2 515 train clips
        #    are < 1 s) they rescale silence into signal; batch statistics fail bridge check (E_BATCH_INVARIANCE);
        #  - every op per clip: never pad to the longest clip of the batch (N would change with the batch).
        feats = self._forward(waves)                       # [B, N_TOKENS, D]
        pos = None
        if self.pos_mode == "per_sample":
            centres = START_S + (np.arange(N_TOKENS) + 0.5) * (LENGTH_S / N_TOKENS)   # seconds of the stored file
            pos = np.stack([_audio_pos(centres, float(s.streams["audio"].info["duration_s"])) for s in samples])
        return feats, pos, used

    def _forward(self, waves: np.ndarray) -> np.ndarray:
        # TODO replace: resample, front end, forward under torch.inference_mode() (run_extract wraps encode
        # in it), take the tap `self.layer`, return [B, N, D] (np.ndarray or torch tensor). The placeholder
        # below (log-RMS and peak of N_TOKENS equal chunks) only lets the template run end to end, so that
        # `bridge check` measures decode throughput before the model is wired in.
        chunks = waves.reshape(waves.shape[0], N_TOKENS, -1)
        rms = np.sqrt(np.mean(chunks.astype(np.float64) ** 2, axis=-1) + 1e-10)
        peak = np.abs(chunks).max(axis=-1)
        return np.stack([np.log(rms), peak], axis=-1).astype(np.float32)

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": float}


def _audio_pos(centres_s: np.ndarray, duration_s: float, stream: int = 0) -> np.ndarray:
    """float32 ``[len(centres_s), 4]`` audio token positions in media coordinates:
    ``(stream, t = centre / duration_s, -1, -1)``; a token centred in added padding gets ``t = -1``."""
    t = np.where(centres_s < duration_s, centres_s / duration_s, -1.0)
    out = np.full((len(centres_s), 4), -1.0, dtype=np.float32)
    out[:, 0] = float(stream)
    out[:, 1] = t
    return out
