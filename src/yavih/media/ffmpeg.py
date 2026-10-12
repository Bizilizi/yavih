"""``ffmpeg`` / ``ffprobe`` CLI fallbacks for a video FILE on disk, used when PyAV is
absent: header facts (``ffprobe_video``), decoded frame pts (``ffprobe_pts``) and frames by index
(``ffmpeg_frames_by_index``). No protocol semantics. ``ffmpeg_info`` reports ``{backend: "ffmpeg", version,
libav}`` from ``ffmpeg -version``."""

from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache

import numpy as np

from yavih.media import DecodeError, DecoderMissing

__all__ = ["ffmpeg_available", "ffmpeg_info", "ffmpeg_frames_by_index", "ffprobe_video", "ffprobe_pts"]

_VERSION_RE = re.compile(r"^ffmpeg version (\S+)")
_LIBAV_RE = re.compile(r"^\s*libavcodec\s+(\d+)\.\s*(\d+)\.\s*(\d+)")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@lru_cache(maxsize=1)
def ffmpeg_info() -> dict | None:
    """``{"backend": "ffmpeg", "version": "<ffmpeg -version>", "libav": "<libavcodec a.b.c>"}`` or ``None``."""
    if not ffmpeg_available():
        return None
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    version, libav = None, None
    for line in out.splitlines():
        m = _VERSION_RE.match(line)
        if m and version is None:
            version = m.group(1)
        m = _LIBAV_RE.match(line)
        if m and libav is None:
            libav = ".".join(m.groups())
    return {"backend": "ffmpeg", "version": version or "unknown", "libav": libav}


def _run(cmd: list[str], *, timeout: float = 600.0) -> bytes:
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=timeout)
    except OSError as exc:
        raise DecoderMissing("ffmpeg") from exc
    if proc.returncode != 0:
        raise DecodeError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.decode('utf-8', 'replace')[-500:]}")
    return proc.stdout


def ffprobe_video(path: str) -> dict:
    """``{"duration", "fps", "n_frames", "width", "height"}`` from the stream header (no decode)."""
    if not ffmpeg_available():
        raise DecoderMissing("ffmpeg")
    out = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames:format=duration",
            "-of", "default=noprint_wrappers=1", path,
        ]
    ).decode("utf-8", "replace")
    kv: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv.setdefault(k.strip(), v.strip())  # stream duration first, format duration as fallback

    def _rate(s: str | None) -> float | None:
        if not s or s in ("0/0", "N/A"):
            return None
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b) if float(b) else None
        return float(s)

    def _float(s: str | None) -> float | None:
        try:
            return None if s in (None, "N/A", "") else float(s)
        except ValueError:
            return None

    fps = _rate(kv.get("avg_frame_rate")) or _rate(kv.get("r_frame_rate"))
    n_frames = _float(kv.get("nb_frames"))
    return {
        "duration": _float(kv.get("duration")),
        "fps": fps,
        "n_frames": int(n_frames) if n_frames is not None else None,
        "width": int(kv["width"]) if kv.get("width") else None,
        "height": int(kv["height"]) if kv.get("height") else None,
    }


def ffprobe_pts(path: str) -> np.ndarray:
    """float64 pts (seconds) of every decoded frame of the first video stream, in presentation order."""
    if not ffmpeg_available():
        raise DecoderMissing("ffmpeg")
    out = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=pts_time,pkt_pts_time,best_effort_timestamp_time", "-of", "csv=p=0", path,
        ]
    ).decode("utf-8", "replace")
    pts: list[float] = []
    for line in out.splitlines():
        for tok in line.split(","):
            tok = tok.strip()
            if tok and tok != "N/A":
                pts.append(float(tok))
                break
    return np.asarray(pts, dtype=np.float64)


def ffmpeg_frames_by_index(path: str, idx: np.ndarray) -> np.ndarray:
    """uint8 ``[n, H, W, 3]``: the decoded frames with presentation index ``idx`` (repeats allowed), pulled in
    one ``ffmpeg -vf select`` call. No selection rule here: ``yavih.media.decode.frames_at`` picks the indices."""
    ix = [int(i) for i in np.asarray(idx, dtype=np.int64).reshape(-1)]
    uniq = sorted(set(ix))
    expr = "+".join(f"eq(n\\,{i})" for i in uniq)
    info = ffprobe_video(path)
    w, h = info["width"], info["height"]
    if not w or not h:
        raise DecodeError(f"ffprobe gave no geometry for {path!r}")
    raw = _run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vf", f"select='{expr}'", "-vsync", "0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
    )
    frame_bytes = w * h * 3
    if len(raw) != frame_bytes * len(uniq):
        raise DecodeError(f"ffmpeg returned {len(raw)} bytes for {len(uniq)} frames of {w}x{h} from {path!r}")
    decoded = np.frombuffer(raw, dtype=np.uint8).reshape(len(uniq), h, w, 3)
    by_index = {i: k for k, i in enumerate(uniq)}
    return np.stack([decoded[by_index[i]] for i in ix]).copy()
