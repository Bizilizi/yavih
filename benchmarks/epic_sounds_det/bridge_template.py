"""Bridge skeleton for **epic_sounds_det** (read setup.md first). Copy, rename, fill the TODOs.

``yavih bridge check <file> --bench epic_sounds_det --synthetic`` smoke-tests a copy before any data is staged;
an unfilled copy fails bridge check: E_TEMPLATE_UNFILLED / E_CONSTANT_FEATURES. A bridge file serves one
benchmark; each matrix entry (file + args) is one reading of one model.

The "Preprocessing source" slot below is REQUIRED: paste the model's own input pipeline into it, and make
``describe()["input"]`` match it line by line. ``bridge check`` warns W_NO_PREPROCESSING_SOURCE while every
line of the slot still says TODO.

Preprocessing source: TODO <repo>@<commit> <file>::<function>, e.g. the model's audio feature extractor
    TODO paste the source lines: resampler and target rate, mono downmix, the front end (fbank / mel:
    TODO n_mels, window, hop), the window length the encoder sees and the hop between windows, padding,
    TODO normalization constants (the pretraining recipe's; never dataset statistics computed here).

What epic_sounds_det hands ``prepare`` / ``encode`` (setup.md "Input contract"): per Sample, ONE RawStream,
``audio``: the WHOLE untrimmed recording's AAC track (an m4a holding the packets of the EPIC-KITCHENS-100 MP4,
never re-encoded), 10 s to 62 min long:

    s.streams["audio"].info: duration_s (10.5-3708 s), sample_rate (48 000), n_samples (header count incl.
                             AAC priming), channels (2), container ("m4a"), bytes (0.2-90 MB)
    s.meta["facts"]       == {}

Nothing is decoded, resampled, cut, windowed or padded for you. How a minutes-long recording becomes a FIXED
``[N, D]`` token set is your decision, declared in ``describe()["input"]``. This template reads every sample of
the recording (``full`` rule), stream-decoding it with PyAV and reducing DECODE_CHUNK_S at a time (memory stays
bounded: one hour of 48 kHz stereo float32 is 1.4 GB), emits one token per HOP_S of media, and pads to a fixed
N = ceil(MAX_S / HOP_S) with t = -1 tokens.

    feats  [B, N, D]  the same N and D for every recording
    pos    [B, N, 4]  (stream, t, y, x): t = token centre seconds / info.duration_s, -1 for padding; y = x = -1
    used   B dicts    {"audio": {"samples": [[0, n]], "pad": [0, pad]}} (what the full rule reads)

The folder's detection scorer reads ``t``: a token at media second t * duration_s informs the time bins around it, and
detections are scored in seconds against the whole recording. Output contract:
``requires: {pos: [t], streams: [audio]}``; a time-pooled bridge (N = 1, or every t = -1) is ineligible.

Gotchas (setup.md "Gotchas"):
    * Fixed N over recordings of 10 s to 62 min: either a fixed rate padded to the longest recording (this
      template: 18 600 tokens at 0.2 s, most of them t = -1 on a median 5-min recording; tokens past the end of
      every recording in a sample are identical zeros, hence the declared E_DEAD_ROWS waiver) or a fixed count
      spread over each recording (time resolution then varies 360x between recordings). Both are legal; declare it.
    * Stream-decode. ``yavih.media.decode.audio(stream, ranges)`` decodes the WHOLE track and then slices it:
      calling it once per chunk is quadratic in the recording length, and one call holds 1.4 GB per hour.
    * AAC decodes to a few hundred samples more or fewer than ``n_samples`` (priming): fit the decoded audio to
      the declared range (what ``used`` reports), as the template does.
    * Labelled segments run from 0.2 s to 38 min (median 1.0 s): a coarse hop cannot localize most of them.
    * Report ``t`` per token from the samples it covers, never from its index alone.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from yavih.bridge import ArgSpec, Bridge

# ---- TODO: the model's choices (placeholders). Every value here is declared in describe()["input"].
HOP_S = 0.2  # one token per 0.2 s of media (the EPIC-SOUNDS baselines' feature stride)
MAX_S = 3720.0  # longest recording in the manifest is 3 708 s; N = ceil(MAX_S / HOP_S)
N_TOKENS = int(math.ceil(MAX_S / HOP_S))
DECODE_CHUNK_S = 60.0  # reduce this much decoded audio at a time
D_PLACEHOLDER = 4  # TODO: the model's width


class EpicSoundsDetBridge(Bridge):
    name = "TODO-model-epic_sounds_det"  # optional label of this file, recorded only; the model's name is describe()["model"]
    batch_size = 2  # long rows: a few recordings per batch
    dtype = "float16"
    pos_mode = "per_sample"  # durations differ → per-recording t
    streams = ("audio",)
    ARGS = {"layer": ArgSpec(type=int, default=-1, help="transformer block to tap (-1 = last)")}

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)  # bench is a BenchmarkView: no manifest, labels or splits
        # TODO: build the model, load weights BEFORE the first sample (no network during encode), .eval():
        #   self.model = build_model(...).eval().to("cuda")
        #   self._weights_sha256 = yavih.bridge.fingerprint_state_dict(self.model)
        self.model = None
        self._weights_sha256 = None

    def _input(self) -> dict:
        return {
            "audio": {
                "temporal": {"rule": "full", "max_s": MAX_S, "pad_to_s": MAX_S},
                "front_end": {"sample_rate": 48000, "resample": "none (TODO: the model's resampler)",
                              "features": f"TODO (placeholder: log mean power per {HOP_S} s hop)", "channels": "mean",
                              "window_s": HOP_S, "hop_s": HOP_S, "decode_chunk_s": DECODE_CHUNK_S,
                              "fit": "decoded samples cut / zero-filled to the declared range (AAC priming)"},
                "norm": {"mean": None, "std": None, "source": "TODO: <repo>@<commit> <file> (pretraining recipe)"},
                "decoder": _decoder_info(),
            }
        }

    def describe(self) -> dict:
        return {
            "weights_sha256": self._weights_sha256,
            "weights_ref": "TODO: path or hub id (provenance only, not hashed)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "streams": list(self.streams),
            "pos_mode": self.pos_mode,
            "reading": "a",
            "model": "TODO-model",
            "layer": self.args["layer"],
            "pooling": "none",  # keep the time axis: a pooled set is ineligible for detection
            "views": 1,
            "input": self._input(),
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),
            # fixed N over 10 s .. 62 min recordings: the tail tokens of the shorter recordings are padding
            "waive": {"E_DEAD_ROWS": "fixed N = MAX_S / HOP_S: tokens past the end of every recording in the sample are zero padding (t = -1)"},
        }

    def prepare(self, sample: Any) -> dict:
        """Stream-decode the whole recording with PyAV and reduce it to per-hop model input as it arrives.

        Not ``yavih.media.decode.audio(stream, ranges)`` in a loop over chunks: it decodes the WHOLE track on every
        call and slices afterwards."""
        import av  # yavih[media]; or the model's own decoder

        from yavih.media import rules

        a = sample.streams["audio"]
        ranges, (pad_l, pad_r) = rules.audio_ranges(self._input()["audio"]["temporal"], a.info)
        lo, hi = int(ranges[0][0]), int(ranges[0][1])  # the full rule: lo == 0
        sr = int(a.info["sample_rate"])
        hop = int(round(HOP_S * sr))
        block = int(round(DECODE_CHUNK_S * sr)) // hop * hop  # reduce this many samples at a time
        per_hop: list[np.ndarray] = []
        buf: list[np.ndarray] = []
        n_buf = done = 0  # samples waiting in buf; samples of [lo, hi) already reduced
        try:
            with av.open(a.open()) as c:
                for frame in c.decode(c.streams.audio[0]):
                    x = frame.to_ndarray().astype(np.float32)
                    x = x.mean(axis=0) if frame.format.is_planar else x.reshape(-1, len(frame.layout.channels)).mean(axis=1)
                    x = x[: max(0, hi - done - n_buf)]  # fit: never past the declared range (AAC priming)
                    buf.append(x)
                    n_buf += len(x)
                    while n_buf >= block:
                        w = np.concatenate(buf)
                        per_hop.append(self._front_end(w[:block], hop))  # TODO: the model's front end per window
                        buf, n_buf, done = [w[block:]], n_buf - block, done + block
                    if done + n_buf >= hi:
                        break
        except av.error.FFmpegError as exc:  # TODO: your decoder's error types
            from yavih import MediaUndecodable

            raise MediaUndecodable([sample.clip_id], f"decoder failed: {exc}") from exc
        tail = np.concatenate(buf) if buf else np.zeros(0, np.float32)
        tail = np.concatenate([tail, np.zeros(max(0, hi - lo - done - len(tail)), np.float32)])  # fit: zero-fill
        if len(tail):
            per_hop.append(self._front_end(tail, hop))
        x = np.concatenate(per_hop, axis=0)  # [n_hops, F]
        return {"x": x, "sr": sr, "used": {"samples": [[lo, hi]], "pad": [int(pad_l), int(pad_r)]}}

    def encode(self, samples: list) -> tuple[Any, Any, list[dict]]:
        feats, pos, used = [], [], []
        for s in samples:
            p = s.prepared
            tok = self._model(p["x"])  # [n_hops, D]
            n = min(len(tok), N_TOKENS)
            dur = float(s.streams["audio"].info["duration_s"])
            f = np.zeros((N_TOKENS, tok.shape[1]), np.float32)
            f[:n] = tok[:n]
            t = np.full(N_TOKENS, -1.0, np.float32)
            (lo, hi), sr = p["used"]["samples"][0], p["sr"]
            hop = int(round(HOP_S * sr))
            starts = lo + np.arange(n) * hop
            centres = (starts + np.minimum(starts + hop, hi)) / 2 / sr  # centre of the samples each token covers
            t[:n] = np.clip(centres / dur, 0.0, 1.0)
            ps = np.full((N_TOKENS, 4), -1.0, np.float32)
            ps[:, 0] = 0.0  # stream index of "audio"
            ps[:, 1] = t
            feats.append(f)
            pos.append(ps)
            used.append({"audio": p["used"]})
        return np.stack(feats).astype(self.dtype), np.stack(pos), used

    # ---- TODO: replace both with the model (frozen, under the inference_mode run_extract sets up).

    @staticmethod
    def _front_end(wav: np.ndarray, hop: int) -> np.ndarray:
        """Placeholder: log mean power per hop → ``[n_hops, 1]`` (the last partial hop included)."""
        n = int(math.ceil(len(wav) / hop))
        w = np.zeros(n * hop, np.float64)
        w[: len(wav)] = wav
        return np.log(np.mean(w.reshape(n, hop) ** 2, axis=1) + 1e-10).astype(np.float32)[:, None]

    def _model(self, x: np.ndarray) -> np.ndarray:
        """Placeholder: the front-end value and its first difference, zero-padded to ``D_PLACEHOLDER``."""
        d = np.diff(x[:, 0], prepend=x[0, 0])[:, None]
        return np.concatenate([x, d, np.zeros((len(x), D_PLACEHOLDER - 2), np.float32)], axis=1)

    def profile(self) -> dict | None:
        return None


def _decoder_info() -> dict:
    try:
        from yavih.media.decode import decoder_info

        return decoder_info("audio_container")  # PyAV; decoder_info("audio") names soundfile, which cannot read AAC
    except Exception:  # noqa: BLE001 — TODO: {"lib": ..., "version": ...} of the decoder you use
        return {"lib": "TODO", "version": "TODO"}
