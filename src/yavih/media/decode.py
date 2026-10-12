"""Optional decode helpers (``[media]`` extra). NO protocol semantics.

A bridge may decode with these or with its own stack and loses nothing either way. They accept a
``RawStream`` (read through ``open()``, so tar members decode in place), a path or bytes:

- ``frames_at(stream, times) -> (uint8 [n, H, W, 3], pts float64 [n])``: per requested time the decoded
  frame nearest by pts (``nearest_ties_earlier``). A time
  outside the media simply gets the first / last frame: the bridge's declared ``edge`` decides whether that
  is legal, not this helper. ``pts`` are what the bridge reports as ``used[stream]["times"]``.
- ``stored_frames(stream, idx) -> uint8 [n, H, W, 3]``: frames by index in decode order.
- ``audio(stream, ranges=None) -> (float32 [L, C], rate)``: the native-rate samples of the given half-open
  ranges, concatenated (all samples when ``None``); no resampling, no downmix, no padding.
- ``image(stream) -> uint8 [H, W, 3]``.
- ``decoder_info(kind=None) -> {lib, version, libav}`` for ``describe()["input"][s]["decoder"]``.
"""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np

from yavih.media import TIE_DECIMALS, DecodeError, DecoderMissing

__all__ = ["frames_at", "stored_frames", "audio", "image", "all_pts", "decoder_info"]


# ----------------------------------------------------------------------------- backends


def _av():
    try:
        import av
    except ImportError as exc:
        raise DecoderMissing("av") from exc
    return av


def _sf():
    try:
        import soundfile
    except (ImportError, OSError):
        return None
    return soundfile


def _pil():
    try:
        import PIL
        from PIL import Image
    except ImportError as exc:
        raise DecoderMissing("pillow") from exc
    return PIL, Image


def _pyav_info() -> dict:
    av = _av()
    libav = None
    versions = getattr(av, "library_versions", None)
    if isinstance(versions, dict) and "libavcodec" in versions:
        libav = ".".join(str(x) for x in versions["libavcodec"])
    return {"lib": "pyav", "version": str(av.__version__), "libav": libav}


def decoder_info(kind: str | None = None) -> dict:
    """``{"lib", "version", "libav"}`` of the library these helpers use for stream ``kind`` on this host:
    video (default) → PyAV (else the ffmpeg CLI); audio → soundfile for wav/flac (PyAV for container audio:
    pass ``kind="audio_container"``); image → pillow. ``libav`` is the libavcodec build where one
    decides the decoded frames, else ``None``. ``DecoderMissing`` when nothing is importable."""
    kind = kind or "video"
    if kind == "video":
        try:
            return _pyav_info()
        except DecoderMissing:
            from yavih.media.ffmpeg import ffmpeg_info

            info = ffmpeg_info()
            if info is None:
                raise
            return {"lib": "ffmpeg", "version": info["version"], "libav": info["libav"]}
    if kind == "audio":
        sf = _sf()
        if sf is not None:
            return {"lib": "soundfile", "version": str(sf.__version__), "libav": None}
        return _pyav_info()
    if kind == "audio_container":
        return _pyav_info()
    if kind == "image":
        PIL, _ = _pil()
        return {"lib": "pillow", "version": str(PIL.__version__), "libav": None}
    raise ValueError(f"decoder_info: unknown kind {kind!r}")


def _source(stream: Any) -> Any:
    """A path or a seekable file object for any supported input."""
    if hasattr(stream, "open") and hasattr(stream, "bytes"):
        path = getattr(stream, "path", None)
        return path if path else stream.open()
    if isinstance(stream, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(stream))
    if isinstance(stream, (str, os.PathLike)):
        return os.fspath(stream)
    if hasattr(stream, "read"):
        return stream
    raise TypeError(f"unsupported media input {type(stream).__name__}")


# ----------------------------------------------------------------------------- video


def _frame_time(frame: Any) -> float | None:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    return None


def _select_streaming(times: np.ndarray, frames_iter: Any) -> list[tuple[float, Any]]:
    """One-pass ``nearest_ties_earlier`` over ascending decoded pts, holding at most one undecided frame.
    For ascending pts the minimum of ``|t − pts|`` is attained at the last pts ``≤ t`` or the first ``> t``;
    an exact (rounded) tie goes to the earlier one, as in the full scan (``rules.select_nearest``)."""
    order = np.argsort(times, kind="stable")
    ts = [float(times[i]) for i in order]
    chosen: list[tuple[float, Any] | None] = [None] * len(ts)
    k = 0
    prev: tuple[float, Any] | None = None
    for frame in frames_iter:
        pts = _frame_time(frame)
        if pts is None:
            continue
        cur = (pts, frame)
        while k < len(ts) and ts[k] <= pts:
            if prev is None:
                chosen[order[k]] = cur
            else:
                d_prev = round(abs(ts[k] - prev[0]), TIE_DECIMALS)
                d_cur = round(abs(pts - ts[k]), TIE_DECIMALS)
                chosen[order[k]] = prev if d_prev <= d_cur else cur
            k += 1
        prev = cur
        if k >= len(ts):
            break
    if prev is None:
        raise DecodeError("the video stream decoded no frame")
    while k < len(ts):
        chosen[order[k]] = prev
        k += 1
    return chosen  # type: ignore[return-value]


def all_pts(stream: Any) -> np.ndarray:
    """float64 pts (media seconds) of every decoded frame, presentation order (diagnostics, prepare checks)."""
    av = _av()
    try:
        with av.open(_source(stream), mode="r") as c:
            vs = c.streams.video[0]
            return np.asarray([t for t in (_frame_time(f) for f in c.decode(vs)) if t is not None], dtype=np.float64)
    except av.error.FFmpegError as exc:  # type: ignore[attr-defined]
        raise DecodeError(f"PyAV failed to decode: {exc}") from exc


def frames_at(stream: Any, times: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(uint8 [n, H, W, 3], pts float64 [n])``: for each requested media time the frame nearest by pts,
    ties to the earlier frame. Works on video streams (PyAV; the ffmpeg CLI for a file on disk when PyAV is
    absent)."""
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if t.size < 1:
        raise ValueError("frames_at: times must be non-empty")
    try:
        av = _av()
    except DecoderMissing:
        src = _source(stream)
        if not isinstance(src, str):
            raise
        return _ffmpeg_frames_at(src, t)
    try:
        with av.open(_source(stream), mode="r") as c:
            if not c.streams.video:
                raise DecodeError("no video track in the file (RawStream.info has_track: false): nothing to decode; what to feed for a track-less clip is the bridge's declared choice")
            vs = c.streams.video[0]
            vs.thread_type = "AUTO"
            chosen = _select_streaming(t, c.decode(vs))
            out = np.stack([f.to_ndarray(format="rgb24") for _, f in chosen])
            return out, np.asarray([p for p, _ in chosen], dtype=np.float64)
    except av.error.FFmpegError as exc:  # type: ignore[attr-defined]
        raise DecodeError(f"PyAV failed to decode: {exc}") from exc


def _ffmpeg_frames_at(path: str, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from yavih.media.ffmpeg import ffmpeg_frames_by_index, ffprobe_pts
    from yavih.media.rules import select_nearest

    pts = ffprobe_pts(path)
    if pts.size == 0:
        raise DecodeError(f"ffprobe found no video frames in {path!r}")
    idx = select_nearest(pts, times)
    return ffmpeg_frames_by_index(path, idx), pts[idx]


def stored_frames(stream: Any, idx: Any) -> np.ndarray:
    """uint8 ``[n, H, W, 3]``: frames by index into the decode (presentation) order of a video."""
    ix = np.asarray(idx, dtype=np.int64).reshape(-1)
    av = _av()
    want = sorted(set(int(i) for i in ix))
    got: dict[int, np.ndarray] = {}
    try:
        with av.open(_source(stream), mode="r") as c:
            vs = c.streams.video[0]
            for k, f in enumerate(c.decode(vs)):
                if k in want:
                    got[k] = f.to_ndarray(format="rgb24")
                    if len(got) == len(want):
                        break
    except av.error.FFmpegError as exc:  # type: ignore[attr-defined]
        raise DecodeError(f"PyAV failed to decode: {exc}") from exc
    absent = [i for i in want if i not in got]
    if absent:
        raise IndexError(f"frame index {absent[0]} beyond the {len(got)} decoded frames")
    return np.stack([got[int(i)] for i in ix])


# ----------------------------------------------------------------------------- audio / image


def _audio_av(src: Any) -> tuple[np.ndarray, int]:
    av = _av()
    try:
        with av.open(src, mode="r") as c:
            if not c.streams.audio:
                raise DecodeError("no audio track in the file (RawStream.info has_track: false): nothing to decode; what to feed for a track-less clip is the bridge's declared choice")
            st = c.streams.audio[0]
            rate = int(st.codec_context.sample_rate or st.rate)
            resampler = av.AudioResampler(format="fltp", layout=st.codec_context.layout, rate=rate)
            chunks = []
            for frame in c.decode(st):
                for rf in resampler.resample(frame):
                    chunks.append(rf.to_ndarray().astype(np.float32, copy=False))
            for rf in resampler.resample(None):
                chunks.append(rf.to_ndarray().astype(np.float32, copy=False))
    except av.error.FFmpegError as exc:  # type: ignore[attr-defined]
        raise DecodeError(f"PyAV failed to decode audio: {exc}") from exc
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32), rate
    return np.ascontiguousarray(np.concatenate(chunks, axis=1).T, dtype=np.float32), rate


def audio(stream: Any, ranges: Any = None) -> tuple[np.ndarray, int]:
    """``(float32 [L, C], native rate)``: the samples of the half-open ranges ``int [k, 2]`` concatenated in
    order (the whole track when ``None``), cut at the end of the media. soundfile first (wav/flac), PyAV for
    container audio (mp4). No resampling, downmix or padding: those are the bridge's declared front end."""
    src = _source(stream)
    data: np.ndarray | None = None
    sf = _sf()
    if sf is not None:
        try:
            if hasattr(src, "seek"):
                src.seek(0)
            data, rate = sf.read(src, dtype="float32", always_2d=True)
        except Exception:  # noqa: BLE001 - libsndfile does not read mp4; PyAV does
            data = None
    if data is None:
        if hasattr(src, "seek"):
            src.seek(0)
        try:
            data, rate = _audio_av(src)
        except DecoderMissing:
            if sf is None:
                raise DecoderMissing("soundfile", "pip install 'yavih[media]' (soundfile for wav/flac, av for container audio)") from None
            raise
    if ranges is None:
        return np.ascontiguousarray(data, dtype=np.float32), int(rate)
    r = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
    parts = [data[max(0, int(lo)) : max(0, int(hi))] for lo, hi in r]
    out = np.concatenate(parts, axis=0) if parts else np.zeros((0, data.shape[1]), dtype=np.float32)
    return np.ascontiguousarray(out, dtype=np.float32), int(rate)


def _rgb(src: Any) -> np.ndarray:
    _, Image = _pil()
    try:
        with Image.open(src) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, SyntaxError, ValueError) as exc:
        raise DecodeError(f"pillow failed to decode image: {exc}") from exc


def image(stream: Any) -> np.ndarray:
    """uint8 ``[H, W, 3]`` RGB of an image stream, the whole stored frame."""
    return _rgb(_source(stream))
