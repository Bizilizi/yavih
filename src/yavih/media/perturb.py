"""Lossless copies of a recording with its future replaced: the input of ``bridge check``'s future-invariance gate.

:func:`future_pair` decodes a stored recording once (PyAV, the ``media`` extra) and writes Matroska copies with
lossless codecs (``ffv1`` video in ``bgr0``, ``flac`` audio as 16-bit PCM), all cut at ``until_s``: ``A`` is the
decoded content; for each cutoff ``c`` in ``cutoffs``, ``B_c`` is the same content with every video frame after ``c``
black and every audio sample after ``c`` silent. A causal bridge returns the same updates for ``A`` and ``B_c`` up to
cutoff ``c``; an update that differs read input after its cutoff (a centered window, a look-ahead audio frontend, a
normalization over the whole recording). The copies share every lossy step (decode, pixel and sample conversion), so
only the future differs between them.
"""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["future_pair"]


def _av() -> Any:
    try:
        import av
    except ImportError as e:  # pragma: no cover - the media extra
        from yavih.media import DecoderMissing

        raise DecoderMissing("the future-invariance gate needs PyAV: pip install 'yavih[media]'") from e
    return av


def future_pair(data: bytes, out_dir: str | os.PathLike, cutoffs: Sequence[float], *, until_s: float, name: str = "rec") -> dict[str, str]:
    """Write ``A`` and one ``B_c`` per cutoff under ``out_dir``; returns ``{"A": path, "<c>": path, ...}`` (keys of
    the cutoffs formatted ``f"{c:.6f}"``). Video frames and audio samples at times ``> c`` are blanked in ``B_c``;
    everything at or after ``until_s`` is dropped from every copy."""
    av = _av()
    os.makedirs(out_dir, exist_ok=True)
    cut = [float(c) for c in cutoffs]
    names = {"A": os.path.join(os.fspath(out_dir), f"{name}.A.mkv")}
    names.update({f"{c:.6f}": os.path.join(os.fspath(out_dir), f"{name}.B{k}.mkv") for k, c in enumerate(cut)})
    blank_after = {"A": None, **{f"{c:.6f}": c for c in cut}}
    src = av.open(io.BytesIO(data))
    try:
        vin = next(iter(src.streams.video), None)
        ain = next(iter(src.streams.audio), None)
        if vin is None and ain is None:
            raise ValueError("future_pair: the recording has neither a video nor an audio stream")
        outs = {k: av.open(p, mode="w", format="matroska") for k, p in names.items()}
        vouts: dict[str, Any] = {}
        aouts: dict[str, Any] = {}
        resampler = None
        try:
            if vin is not None:
                rate = vin.average_rate or vin.guessed_rate or 25
                for k, o in outs.items():
                    s = o.add_stream("ffv1", rate=rate)
                    s.width, s.height, s.pix_fmt = vin.codec_context.width, vin.codec_context.height, "bgr0"
                    s.time_base = vin.time_base
                    vouts[k] = s
            if ain is not None:
                sr = int(ain.codec_context.sample_rate or ain.rate)
                layout = ain.codec_context.layout.name if ain.codec_context.layout is not None else "mono"
                resampler = av.AudioResampler(format="s16", layout=layout, rate=sr)
                for k, o in outs.items():
                    s = o.add_stream("flac", rate=sr)
                    s.layout = layout
                    aouts[k] = s
            streams = [s for s in (vin, ain) if s is not None]
            for frame in src.decode(*streams):
                t = float(frame.time) if frame.time is not None else 0.0
                if t >= until_s:
                    continue
                if frame.__class__.__name__ == "VideoFrame":
                    rgb = frame.to_ndarray(format="rgb24")
                    for k, o in outs.items():
                        c = blank_after[k]
                        img = np.zeros_like(rgb) if c is not None and t > c + 1e-9 else rgb
                        vf = av.VideoFrame.from_ndarray(img, format="rgb24")
                        vf.pts, vf.time_base = frame.pts, frame.time_base
                        for p in vouts[k].encode(vf):
                            o.mux(p)
                else:
                    assert resampler is not None
                    for rf in resampler.resample(frame):
                        pcm = rf.to_ndarray()  # (channels, n) for planar, (1, n * channels) for packed s16
                        sr = int(rf.sample_rate)
                        t0 = float(rf.time) if rf.time is not None else t
                        n = rf.samples
                        times = t0 + np.arange(n) / sr
                        keep = times < until_s
                        if not keep.any():
                            continue
                        for k, o in outs.items():
                            c = blank_after[k]
                            x = pcm.copy()
                            if c is not None:
                                mute = times > c + 1e-9
                                if x.shape[0] == 1 and x.shape[1] != n:  # packed: interleaved channels
                                    ch = x.shape[1] // n
                                    x = x.reshape(n, ch)
                                    x[mute] = 0
                                    x = x.reshape(1, n * ch)
                                else:
                                    x[:, mute] = 0
                            af = av.AudioFrame.from_ndarray(x, format=rf.format.name, layout=rf.layout.name)
                            af.sample_rate, af.pts, af.time_base = sr, rf.pts, rf.time_base
                            for p in aouts[k].encode(af):
                                o.mux(p)
            for k, o in outs.items():
                for s in (vouts.get(k), aouts.get(k)):
                    if s is not None:
                        for p in s.encode(None):
                            o.mux(p)
        finally:
            for o in outs.values():
                o.close()
    finally:
        src.close()
    return names
