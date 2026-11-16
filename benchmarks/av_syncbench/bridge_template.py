"""Bridge skeleton for **av_syncbench**. Copy, rename, fill the TODOs.

Read ``setup.md`` first (input and output contracts). ``yavih bridge check <file> --bench av_syncbench --synthetic``
is refused with E_SYNTHETIC: the core's synthetic check covers the core scorers only, not this folder's own pairwise
scorer (setup.md "Gotchas"). Check the bridge on the staged data (``yavih bridge check <file> --bench av_syncbench
--out <profile>``, the line ``plan init`` prints); an unfilled copy fails bridge check: E_TEMPLATE_UNFILLED /
E_CONSTANT_FEATURES. A bridge file serves one benchmark; each matrix entry (file + args) is one reading of one model.
``av_syncbench`` is TRAINING-FREE: the folder's scorer (``pairwise.py``) compares, for every pair, the mean cosine
of the video's segment embeddings with the original audio's and with the perturbed audio's, segment by segment, in
the model's own SHARED audio-visual space (the vectors a contrastive or other correspondence objective aligned, after
their projection heads). A model without such a space has nothing to put here. The only reading is ``"av"``.

THE SEGMENT GRID (protocol, ``segment_seconds`` = 0.64, handed over in ``self.bench.facts``): segment ``k`` of a stream
covers ``[0.64 k, 0.64 (k + 1))`` s of THAT stream; embed every COMPLETE segment, in time order, one row each (``n =
floor((duration + 1e-9) / 0.64)``, the authors' ``non_overlapping_starts``). The scorer keeps the segments the three
streams share and caps them at 18; never cut, align or pad one stream to another yourself.

MODALITY ISOLATION (``output_contract.isolation``): ``video_seg`` may depend on the mp4's frames only, ``audio_seg`` on
the mp4's own audio track only, ``edited_seg`` on the wav only. ``bridge check`` swaps the wav for another row's and
hands over lossless copies of the mp4 with one track blanked (frames re-encoded losslessly, the audio track as 16-bit
PCM); a key that changes fails ``E_MODALITY_LEAK``. So: three independent single-modality forwards, no shared
normalization, and decode the mp4's audio track to 16-bit PCM at its own rate and layout FIRST (``_pcm16`` below): a
float decode of the AAC track differs from the gate's 16-bit copy in the last bit.

PREPROCESSING SOURCE — paste the model's OWN evaluation preprocessing here, verbatim, with repo, file, function
and commit; ``describe()["input"]`` below must match it line by line (``bridge check`` warns
``W_NO_PREPROCESSING_SOURCE`` while this slot is empty):

    TODO

The defaults below are those of the reference implementation, the authors' ImageBind adapter (``paper_legacy`` mode
of ``avsyncbench/evaluators/imagebind.py``, https://github.com/fgt7t6g/AV-SyncBench @ 82e1353): per segment the first
and last frame (``round(start * fps)``, ``round((start + 0.64 - 1e-3) * fps)``) warped to 224 x 224 with CLIP's
mean/std, and a log-mel spectrogram 128 x 204 (n_fft 400, hop 160, 16 kHz mono) of the segment's audio. They are the
reference's choices for ImageBind, not protocol: replace them with your model's own.

What ``prepare`` and ``encode`` receive per Sample (setup.md "Input contract"):
    s.streams["video"]: RawStream over the stored original mp4 — ``open()`` / ``bytes()`` / ``path``; ``info`` =
        {duration_s, fps, width, height, container, bytes}. Native size and rate, mixed; 2.5–13 s.
    s.streams["audio"]: the SAME mp4, for its own AAC track (44.1 or 48 kHz stereo); ``info`` = {duration_s,
        sample_rate, channels, container, bytes}.
    s.streams["edited_audio"]: RawStream over the perturbed wav (16-bit PCM mono, 16 kHz or 22.05 kHz); ``info`` =
        {duration_s, sample_rate, n_samples, channels, container, bytes}. Starts with the video; its length differs.
    Nothing else (no task, no level, no row identity you may use: ``row``/``clip_id``/``split`` are scrambled by
    ``bridge check``).

What ``encode`` returns (``describe()["outputs"]``), per sample: ``video_seg [Tv, D]``, ``audio_seg [Ta, D]``,
``edited_seg [Te, D]`` (un-normalized is fine: the scorer L2-normalizes each row); ``used[i]`` reports the frames fed
and the audio ranges, per stream.
"""

from __future__ import annotations

import io
import math
from typing import Any

import numpy as np

from yavih.bridge import Bridge

MODEL_SAMPLE_RATE = 16000
# TODO: the model's own per-segment input. Defaults: the reference's paper_legacy ImageBind transforms.
FRAME_SIZE = (224, 224)
MEAN, STD = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
N_MELS, N_FFT, HOP, MEL_FRAMES = 128, 400, 160, 204


def n_segments(duration_s: float, segment_s: float) -> int:
    """Complete segments of a stream (the authors' ``non_overlapping_starts``)."""
    return int(math.floor((float(duration_s) + 1e-9) / segment_s))


def _pcm16(data: bytes) -> tuple[np.ndarray, int]:
    """``(float32 [L], native rate)``: a wav, or an mp4's audio track (the frames are never decoded), as 16-bit PCM at
    its own rate and layout first, then the channel mean. The 16-bit step keeps the decode bit-identical on the
    lossless copies ``bridge check``'s isolation gate hands over."""
    import av

    with av.open(io.BytesIO(data)) as c:
        s = c.streams.audio[0]
        rate = int(s.codec_context.sample_rate or s.rate)
        channels = s.codec_context.channels or 1
        res = av.AudioResampler(format="s16", layout=s.codec_context.layout.name, rate=rate)
        parts = [f.to_ndarray().reshape(-1) for frame in c.decode(s) for f in res.resample(frame)]
        parts += [f.to_ndarray().reshape(-1) for f in res.resample(None)]
    pcm = np.concatenate(parts).astype(np.float32) / 32768.0 if parts else np.zeros(0, np.float32)
    return pcm.reshape(-1, channels).mean(axis=1), rate


class AVSyncBenchBridge(Bridge):
    name = "TODO-model-av_syncbench-av"  # optional label of this file, recorded only; the model's name is describe()["model"]
    batch_size = 4
    dtype = "float32"  # "float16" | "float32"
    outputs = ("video_seg", "audio_seg", "edited_seg")
    ARGS: dict = {}

    def __init__(self, bench: Any, args: dict) -> None:
        super().__init__(bench, args)
        from yavih.media import (
            decode as ymd,  # [media] extra; or decode with the model's own stack and declare it
        )

        self._decode = ymd
        self._decoders = {"video": ymd.decoder_info("video"), "audio": ymd.decoder_info("audio")}  # {lib, version, libav}
        self.segment = float(bench.facts["segment_seconds"])
        # TODO: build the model, load weights BEFORE the first sample (no network during encode), call .eval():
        # import torch
        # from yavih.bridge import fingerprint_state_dict
        # self.model = build_model(...).eval().to("cuda")
        # self._weights_sha256 = fingerprint_state_dict(self.model)
        self.model = None
        self._weights_sha256 = None  # TODO: real fingerprint; None is recorded as W_WEIGHTS_UNFINGERPRINTED

    def describe(self) -> dict:
        # Every key below is REQUIRED; extra keys are allowed and recorded (never identity).
        audio = {
            "temporal": {"rule": "custom", "description": "every complete 0.64 s segment of the track, from 0 s, one model input each"},
            "front_end": {"sample_rate": MODEL_SAMPLE_RATE, "resample": "TODO: e.g. torchaudio.functional.resample from the native rate",
                          "features": "log mel (torchaudio MelSpectrogram n_fft 400, hop 160, 128 mels, power 2, log(x + 1e-6)), 204 frames",
                          "channels": "mean", "pcm": "16-bit at the native rate first"},
            "norm": {"source": "TODO: <repo>@<commit> <file> (none in the reference's paper_legacy mode)"},
            "decoder": self._decoders["audio"],
        }
        return {
            "weights_sha256": self._weights_sha256,
            "weights_ref": "TODO: path or hub id (provenance only, not hashed)",
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "reading": "av",
            "model": "TODO-model",  # the name all of this model's bridges use; results are filed under it
            "views": 1,
            "outputs": list(self.outputs),
            "input": {
                "video": {
                    "temporal": {"rule": "custom", "description": "per complete 0.64 s segment its first and last frame (round(start*fps), round((start+0.64-1e-3)*fps))"},
                    "spatial": {"resize": {"size": list(FRAME_SIZE)}, "crop": None, "interp": "bilinear", "antialias": False},
                    "norm": {"scale": 1 / 255, "mean": MEAN, "std": STD,  # TODO
                             "source": "TODO: <repo>@<commit> <file> (the model's pretraining recipe)"},
                    "decoder": self._decoders["video"],
                },
                "audio": audio,
                "edited_audio": dict(audio),
            },
            # DECLARE what you run with; run_extract enforces it (E_PRECISION_UNDECLARED on a null).
            "precision": {"autocast": None, "matmul_precision": "highest", "allow_tf32": False, "deterministic": True},
            "args": dict(self.args),  # echoes every consumed --arg with defaults filled
        }

    def _video(self, v: Any) -> tuple[np.ndarray, list[float]]:
        fps = float(v.info["fps"])
        n_frames = int(round(float(v.info["duration_s"]) * fps))
        starts = [k * self.segment for k in range(n_segments(n_frames / fps, self.segment))]
        idx = []
        for s in starts:  # the reference's paper_legacy frame pair of each segment
            first = max(0, min(n_frames - 1, int(round(s * fps))))
            last = max(0, min(n_frames - 1, int(round((s + self.segment - 1e-3) * fps))))
            idx += [first, last]
        frames, pts = self._decode.frames_at(v, [i / fps for i in idx])  # uint8 [2 n, H, W, 3]
        return frames, [float(t) for t in pts]

    def _audio(self, a: Any) -> tuple[np.ndarray, int, list[list[int]]]:
        wav, rate = _pcm16(a.bytes())
        n = n_segments(len(wav) / rate, self.segment)
        return wav, rate, [[0, int(round(n * self.segment * rate))]]

    def prepare(self, sample: Any) -> dict:
        """One pair's CPU work, ahead of ``encode``: what this returns arrives as ``sample.prepared``."""
        frames, times = self._video(sample.streams["video"])
        wav, rate, used_a = self._audio(sample.streams["audio"])
        ewav, erate, used_e = self._audio(sample.streams["edited_audio"])
        # TODO video: per segment the frame pair, /255, warp to FRAME_SIZE, the model's FIXED mean/std → [n, 3, 2, 224, 224].
        # TODO audio: resample to 16 kHz, cut each complete 0.64 s segment, the model's own front end → [n, 1, 128, 204].
        return {"frames": frames, "times": times, "wav": wav, "rate": rate, "used_a": used_a, "ewav": ewav, "erate": erate, "used_e": used_e}

    def encode(self, samples: list) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = {"video_seg": [], "audio_seg": [], "edited_seg": [], "used": []}
        for s in samples:  # instance-wise: nothing below may depend on the other samples of the batch
            p = s.prepared
            n_v = len(p["frames"]) // 2
            n_a = n_segments(len(p["wav"]) / p["rate"], self.segment)
            n_e = n_segments(len(p["ewav"]) / p["erate"], self.segment)
            # TODO: frames → the visual encoder → its projection into the shared space → [n_v, D];
            #       each audio's segments → the audio encoder → its projection → [n_a, D] and [n_e, D]. Three separate forwards.
            out["video_seg"].append(np.zeros((max(n_v, 1), 1), dtype=self.dtype))  # placeholder
            out["audio_seg"].append(np.zeros((max(n_a, 1), 1), dtype=self.dtype))  # placeholder
            out["edited_seg"].append(np.zeros((max(n_e, 1), 1), dtype=self.dtype))  # placeholder
            out["used"].append({"video": {"times": p["times"]},
                                "audio": {"samples": p["used_a"], "pad": [0, 0]},
                                "edited_audio": {"samples": p["used_e"], "pad": [0, 0]}})
        return out

    def profile(self) -> dict | None:
        return None  # optional: {"peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30}
