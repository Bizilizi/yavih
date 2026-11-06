"""Drawing helpers for the README figures (``yavih.previews``). NO protocol semantics.

numpy at module scope; scipy and Pillow inside the functions (the ``[media]`` extra), like ``yavih.media.decode``. No
plotting library: a waveform is an inline SVG path, a spectrogram or a depth map a PNG data URI coloured with the design
palette, and boxes, bands and timelines are positioned ``<div>`` s the design CSS styles; Chrome rasterises the page.

- ``mel_spectrogram(y, rate) -> float [n_mels, T]`` in dB (scipy STFT, a numpy Slaney mel filterbank), low bands first.
- ``envelope(y, columns) -> (min, max)``: the per-column extremes of a waveform.
- ``colormap(values, stops) -> uint8 [H, W, 3]``: values in [0, 1] through five colour stops.
- ``png_data_uri(rgb, max_side=None) -> "data:image/png;base64,..."``.
- ``svg_wave(y, width, height) -> "<svg ...>"``: the envelope as one filled path.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence

import numpy as np

from yavih.media import DecoderMissing

__all__ = ["DEPTH_STOPS", "SPEC_STOPS", "colormap", "envelope", "mel_filterbank", "mel_spectrogram", "png_data_uri",
           "svg_wave"]

#: Spectrogram colours, quiet to loud: the page background, deep violet, violet (--b), pink (--c), a warm white.
SPEC_STOPS: tuple[str, ...] = ("#070a12", "#2e1065", "#a78bfa", "#f472b6", "#fdf2f8")
#: Depth colours, near to far: a warm white, cyan (--a), violet (--b), deep violet, the page background.
DEPTH_STOPS: tuple[str, ...] = ("#fdf2f8", "#22d3ee", "#a78bfa", "#3b1d6e", "#070a12")


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep, f / f_sp)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank(rate: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """float64 ``[n_mels, n_fft // 2 + 1]``: triangular Slaney filters, area-normalised (librosa's default)."""
    fmax = rate / 2.0 if fmax is None else fmax
    freqs = np.linspace(0.0, rate / 2.0, n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax)), n_mels + 2))
    widths = np.diff(edges)
    ramps = edges[:, None] - freqs[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    fb = np.maximum(0.0, np.minimum(lower, upper))
    return fb * (2.0 / (edges[2:n_mels + 2] - edges[:n_mels]))[:, None]


def mel_spectrogram(y: np.ndarray, rate: int, *, n_fft: int = 1024, hop: int = 256, n_mels: int = 64,
                    top_db: float = 80.0) -> np.ndarray:
    """float32 ``[n_mels, T]``: the power mel spectrogram of mono ``y`` in dB re its maximum, floored at ``-top_db``."""
    try:
        from scipy.signal import stft
    except ImportError as exc:
        raise DecoderMissing("scipy") from exc
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    _, _, z = stft(y, fs=rate, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary="even", padded=True)
    power = np.abs(z) ** 2
    mel = mel_filterbank(int(rate), n_fft, n_mels) @ power
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db -= db.max()
    return np.maximum(db, -float(top_db)).astype(np.float32)


def envelope(y: np.ndarray, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """``(min, max)`` float64 ``[columns]`` of mono ``y`` split into ``columns`` near-equal runs (zeros past the end)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    columns = max(1, int(columns))
    if y.size == 0:
        z = np.zeros(columns)
        return z, z.copy()
    edges = np.linspace(0, y.size, columns + 1).astype(np.int64)
    lo, hi = np.zeros(columns), np.zeros(columns)
    for k in range(columns):
        a, b = edges[k], max(edges[k + 1], edges[k] + 1)
        seg = y[a:min(b, y.size)]
        if seg.size:
            lo[k], hi[k] = seg.min(), seg.max()
    return lo, hi


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def colormap(values: np.ndarray, stops: Sequence[str] = SPEC_STOPS) -> np.ndarray:
    """uint8 ``[..., 3]``: ``values`` (clipped to [0, 1]; NaN reads 0) mapped linearly through ``stops`` (hex)."""
    v = np.nan_to_num(np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0), nan=0.0)
    cols = np.asarray([_rgb(s) for s in stops], dtype=np.float64)
    pos = v * (len(stops) - 1)
    i = np.minimum(pos.astype(np.int64), len(stops) - 2)
    t = (pos - i)[..., None]
    return np.clip(np.round(cols[i] * (1 - t) + cols[i + 1] * t), 0, 255).astype(np.uint8)


def png_data_uri(rgb: np.ndarray, *, max_side: int | None = None) -> str:
    """A ``data:image/png;base64`` URI of uint8 ``[H, W, 3]`` (or ``[H, W]``), downscaled (Lanczos) so its longer
    side is at most ``max_side``."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise DecoderMissing("pillow") from exc
    im = Image.fromarray(np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)))
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def svg_wave(y: np.ndarray, width: int, height: int, *, peak: float | None = None, css_class: str = "wave") -> str:
    """An inline ``<svg>`` (``viewBox 0 0 width height``) of the min/max envelope of mono ``y`` as one filled path,
    scaled to ``peak`` (default: the clip's own absolute peak), over a centre line."""
    w, h = max(1, int(width)), max(2, int(height))
    lo, hi = envelope(y, w)
    p = float(peak) if peak else float(max(np.abs(lo).max(), np.abs(hi).max(), 1e-6))
    mid = h / 2.0
    top = mid - np.clip(hi / p, -1, 1) * (mid - 1)
    bot = mid - np.clip(lo / p, -1, 1) * (mid - 1)
    top = np.minimum(top, mid - 0.5)
    bot = np.maximum(bot, mid + 0.5)
    xs = np.arange(w) + 0.5
    pts = [f"{x:.1f},{t:.1f}" for x, t in zip(xs, top, strict=True)] + [f"{x:.1f},{b:.1f}" for x, b in zip(xs[::-1], bot[::-1], strict=True)]
    return (f'<svg class="{css_class}" viewBox="0 0 {w} {h}" preserveAspectRatio="none" width="100%" height="{h}" '
            f'xmlns="http://www.w3.org/2000/svg"><line x1="0" y1="{mid:.1f}" x2="{w}" y2="{mid:.1f}"/>'
            f'<path d="M{" L".join(pts)} Z"/></svg>')
