"""Media helpers. Nothing here carries protocol semantics.

- ``yavih.media.rules`` (core, numpy): evaluate a bridge's declared rule on a row's facts.
- ``yavih.media.info`` (``[media]`` extra): header probes (PyAV, soundfile, pillow; ffprobe CLI fallback)
  and the site census ``bench verify --census`` writes.
- ``yavih.media.decode`` (``[media]`` extra): optional decode helpers a bridge may use instead of its own
  stack (``frames_at``, ``stored_frames``, ``audio``, ``image``, ``decoder_info``).
- ``yavih.media.tarindex`` (stdlib): ``index.json`` members read in place with ``os.pread``.
- ``yavih.media.ffmpeg``: ffprobe / ffmpeg CLI fallbacks for files on disk.

This package itself imports nothing optional (numpy + stdlib only); every backend is
imported inside a function and a missing one raises ``DecoderMissing(package, pip_hint)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "DecoderMissing",
    "MediaMissing",
    "DecodeError",
    "decoder_info",
    "nearest_ties_earlier",
    "TIE_DECIMALS",
    "PIP_HINTS",
]

#: ``pip`` hints per optional backend (``DecoderMissing(package, pip_hint)``).
PIP_HINTS: dict[str, str] = {
    "av": "pip install 'yavih[media]' (or: pip install av)",
    "soundfile": "pip install 'yavih[media]' (or: pip install soundfile)",
    "pillow": "pip install 'yavih[media]' (or: pip install pillow)",
    "scipy": "pip install 'yavih[media]' (or: pip install scipy)",
    "ffmpeg": "install the ffmpeg CLI (apt install ffmpeg) or pip install av",
}

from yavih.media.rules import TIE_DECIMALS  # noqa: E402  (re-export; rules is numpy-only)


class DecoderMissing(ImportError):
    """An optional decoder package is absent: ``DecoderMissing(package, pip_hint)``. Exit 3."""

    def __init__(self, package: str, pip_hint: str | None = None) -> None:
        pip_hint = pip_hint or PIP_HINTS.get(package, f"pip install {package}")
        super().__init__(f"decoder package {package!r} is not installed; {pip_hint}")
        self.package = package
        self.pip_hint = pip_hint


class MediaMissing(FileNotFoundError):
    """The stored media of ``(clip_id, stream)`` is missing or unreadable: absent file, absent index entry,
    short read (``MediaMissing(clip_id, stream, reason)``). Raised by ``RawStream.bytes()`` and the
    byte read-ahead; ``run_extract`` records the row as missing and never hands it to the bridge."""

    def __init__(self, clip_id: str, stream: str, reason: str | None = None) -> None:
        super().__init__(f"MediaMissing: {stream} of {clip_id}" + (f" ({reason})" if reason else ""))
        self.clip_id = clip_id
        self.stream = stream
        self.reason = reason


class DecodeError(RuntimeError):
    """A backend failed to probe or decode media that exists (a helper error, never a protocol step)."""


def decoder_info(kind: str | None = None) -> dict | None:
    """``{"lib", "version", "libav"}`` of the library ``yavih.media.decode`` uses for stream ``kind`` on this
    host (``None``: the video decoder), or ``None`` when none is importable. See ``decode.decoder_info``."""
    from yavih.media.decode import decoder_info as _decoder_info

    try:
        return _decoder_info(kind)
    except DecoderMissing:
        return None


def nearest_ties_earlier(times: Sequence[float] | np.ndarray, pts: Sequence[float] | np.ndarray) -> np.ndarray:
    """``nearest_ties_earlier`` with argument order ``(times, pts)``: the same function as
    ``yavih.media.rules.select_nearest(pts, times)``."""
    from yavih.media.rules import select_nearest

    return select_nearest(pts, times)
