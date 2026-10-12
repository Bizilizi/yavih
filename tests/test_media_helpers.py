"""Synthetic media builders shared by ``test_media*.py`` / ``test_sample*.py``. No tests here.

Everything is tiny and deterministic: a lossless x264 video whose frame ``i`` is the flat grey
``GREY_STEP * i`` (so a decoded frame identifies its index), an mp4 holding such a video plus a lossless
ALAC audio track (AVE / avsync shape: two streams, one file), a stereo wav with a known sample count, RGB
images, a GeoNet-style ``.mat`` and a plain tar + ``index.json`` in the tar-index format.
"""

from __future__ import annotations

import io
import json
import tarfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

GREY_STEP = 12  # frame i of a synthetic video has every pixel == GREY_STEP * i (lossless x264)


def have(mod: str) -> bool:
    try:
        __import__(mod)
    except ImportError:
        return False
    return True


needs_av = pytest.mark.skipif(not have("av"), reason="PyAV not installed")
needs_sf = pytest.mark.skipif(not have("soundfile"), reason="soundfile not installed")
needs_pil = pytest.mark.skipif(not have("PIL"), reason="pillow not installed")
needs_scipy = pytest.mark.skipif(not have("scipy"), reason="scipy not installed")


def frame_value(i: int) -> int:
    return GREY_STEP * i


def frame_index_of(frames: np.ndarray) -> list[int]:
    """Frame indices decoded frames encode (``[n, H, W, 3]`` or ``[H, W, 3]``); yuv420p round-trips are
    within +-2 of the flat grey written, so round to the nearest step."""
    f = frames if frames.ndim == 4 else frames[None]
    return [int(round(float(x[..., 0].mean()) / GREY_STEP)) for x in f]


def write_video(path: Path, *, fps: int = 10, n_frames: int = 20, width: int = 64, height: int = 48, start_pts: int = 0) -> np.ndarray:
    """Lossless x264 mp4; returns the frame pts (seconds). Frame i is flat grey ``frame_value(i)``."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = width, height, "yuv420p"
        s.time_base = Fraction(1, fps)
        s.options = {"crf": "0", "preset": "ultrafast"}
        for i in range(n_frames):
            img = np.full((height, width, 3), frame_value(i), dtype=np.uint8)
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts = start_pts + i
            for pk in s.encode(f):
                c.mux(pk)
        for pk in s.encode(None):
            c.mux(pk)
    return (np.arange(n_frames) + start_pts) / float(fps)


def write_av_mp4(path: Path, *, fps: int = 10, n_frames: int = 20, width: int = 64, height: int = 48, rate: int = 16000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """mp4 with a lossless x264 video (frame i = flat grey ``frame_value(i)``) and a mono ALAC audio track of
    ``n_frames / fps`` seconds. Returns ``(video pts seconds, int16 audio samples written)``."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(n_frames / fps * rate))
    audio = (np.random.default_rng(seed).uniform(-0.5, 0.5, n) * 32767).astype(np.int16)
    with av.open(str(path), "w") as c:
        v = c.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = width, height, "yuv420p"
        v.time_base = Fraction(1, fps)
        v.options = {"crf": "0", "preset": "ultrafast"}
        a = c.add_stream("alac", rate=rate)
        a.layout = "mono"
        for i in range(n_frames):
            f = av.VideoFrame.from_ndarray(np.full((height, width, 3), frame_value(i), dtype=np.uint8), format="rgb24")
            f.pts = i
            for pk in v.encode(f):
                c.mux(pk)
        step = 4096
        for pos in range(0, n, step):
            af = av.AudioFrame.from_ndarray(audio[pos : pos + step][None], format="s16p", layout="mono")
            af.sample_rate, af.pts = rate, pos
            for pk in a.encode(af):
                c.mux(pk)
        for pk in a.encode(None):
            c.mux(pk)
        for pk in v.encode(None):
            c.mux(pk)
    return np.arange(n_frames) / float(fps), audio


def write_wav(path: Path, *, seconds: float = 2.0, rate: int = 16000, channels: int = 2, seed: int = 0) -> np.ndarray:
    """Stereo float wav with a known sample count; returns the ``[L, C]`` data written."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(round(seconds * rate))
    data = np.random.default_rng(seed).uniform(-0.5, 0.5, (n, channels)).astype(np.float32)
    sf.write(str(path), data, rate, subtype="FLOAT")
    return data


def write_image(path: Path, *, height: int = 40, width: int = 60, seed: int = 0) -> np.ndarray:
    """PNG (lossless) RGB image with a deterministic gradient; returns the array."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.stack([(xx * 255 // max(1, width - 1)), (yy * 255 // max(1, height - 1)), np.full_like(xx, seed % 256)], axis=-1).astype(np.uint8)
    Image.fromarray(img).save(str(path))
    return img


def jpeg_bytes(img: np.ndarray, quality: int = 95) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def png_bytes(img: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def write_geonet_mat(path: Path, *, offsets=(244.350, 232.338, 207.016), height: int = 481, width: int = 641, seed: int = 0) -> np.ndarray:
    """A GeoNet-like ``.mat``: ``img`` float32 ``[481, 641, 3]`` mean-subtracted; returns the uint8 image
    (``[480, 640, 3]``) the offsets recover."""
    import scipy.io

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = rgb.astype(np.float32) - np.asarray(offsets, dtype=np.float32)
    scipy.io.savemat(str(path), {"img": img, "depth": np.zeros((height, width), np.float32)})
    return rgb[:480, :640]


def write_tar_with_index(tar_path: Path, members: dict[str, bytes], index_path: Path | None = None, *, shard_name: str | None = None) -> Path:
    """Plain tar of ``members`` plus a tar-index ``index.json`` (``[shard_idx, offset_data, size]`` from
    tarfile's authoritative ``offset_data``). Returns the index path."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "w") as tf:
        for key, data in members.items():
            ti = tarfile.TarInfo(key)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    index: dict[str, list[int]] = {}
    with tarfile.open(str(tar_path), "r") as tf:
        for ti in tf.getmembers():
            index[ti.name] = [0, int(ti.offset_data), int(ti.size)]
    index_path = index_path or tar_path.with_name("index.json")
    index_path.write_text(json.dumps({"version": 1, "shards": [shard_name or tar_path.name], "members": index}))
    return index_path
