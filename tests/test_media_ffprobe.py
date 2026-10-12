"""Media facts without the ``[media]`` extra.

Without PyAV / soundfile / pillow and without a site census, ``probe_info`` falls back to the ``ffprobe`` CLI (same
libav, same facts), ``doctor --bench`` warns ``W_HEADER_BACKEND`` when a stream kind has no backend at all, and
``extract`` / ``bridge check`` refuse at start (exit 3) instead of per row.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

import yavih.media.info as I
from test_media_helpers import (
    jpeg_bytes,
    needs_av,
    needs_pil,
    needs_sf,
    write_av_mp4,
    write_image,
    write_video,
    write_wav,
)
from yavih.media import DecoderMissing

needs_ffprobe = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe CLI not on PATH")


@pytest.fixture
def no_py_backends(monkeypatch):
    """Block PyAV, soundfile and pillow (``import`` raises ImportError) for the duration of a test."""
    for mod in ("av", "soundfile", "PIL", "PIL.Image"):
        monkeypatch.setitem(sys.modules, mod, None)


def _both(path_or_bytes, kind, monkeypatch, **kw):
    """``(facts with the Python backend, facts with ffprobe only)``."""
    with_py = I.probe_info(path_or_bytes, kind, **kw)
    with monkeypatch.context() as m:
        for mod in ("av", "soundfile", "PIL", "PIL.Image"):
            m.setitem(sys.modules, mod, None)
        ff = I.probe_info(path_or_bytes, kind, **kw)
    return with_py, ff


@needs_ffprobe
@needs_av
def test_video_facts_match_pyav_on_path_and_bytes(tmp_path: Path, monkeypatch):
    p = tmp_path / "v.mp4"
    write_video(p, fps=10, n_frames=20, width=64, height=48)
    a, b = _both(p, "video", monkeypatch)
    assert a == b and b["n_frames"] == 20 and b["fps"] == 10.0 and b["has_track"] is True
    a, b = _both(p.read_bytes(), "video", monkeypatch, ext="mp4")
    assert a == b
    a, b = _both(p, "video", monkeypatch, demux=True)
    assert a == b


@needs_ffprobe
@needs_av
def test_container_audio_and_track_less_file_match_pyav(tmp_path: Path, monkeypatch):
    p = tmp_path / "av.mp4"
    write_av_mp4(p, fps=10, n_frames=20, rate=16000)
    a, b = _both(p, "audio", monkeypatch, demux=True)
    assert a == b and b["sample_rate"] == 16000 and b["channels"] == 1
    v = tmp_path / "v.mp4"
    write_video(v)
    a, b = _both(v, "audio", monkeypatch)  # an mp4 without an audio track: a fact, not an error
    assert a == b and b["has_track"] is False and b["container"] == "mp4"


@needs_ffprobe
@needs_sf
def test_wav_facts_match_soundfile(tmp_path: Path, monkeypatch):
    p = tmp_path / "a.wav"
    data = write_wav(p, seconds=1.5, rate=16000, channels=2)
    a, b = _both(p, "audio", monkeypatch)
    assert a == b and b["n_samples"] == data.shape[0]


@needs_ffprobe
@needs_pil
def test_image_facts_match_pillow(tmp_path: Path, monkeypatch):
    p = tmp_path / "i.png"
    write_image(p, height=40, width=60)
    a, b = _both(p, "image", monkeypatch)
    assert a == b and (b["width"], b["height"], b["container"]) == (60, 40, "png")
    a, b = _both(jpeg_bytes(np.zeros((8, 10, 3), np.uint8)), "image", monkeypatch, ext="jpg")
    assert a == b and b["container"] == "jpeg"


@needs_ffprobe
def test_unreadable_header_is_decode_error_under_ffprobe(no_py_backends):
    from yavih.media import DecodeError

    with pytest.raises(DecodeError):
        I.probe_info(b"not a video", "video")


def test_no_backend_at_all_is_decoder_missing(no_py_backends, monkeypatch):
    monkeypatch.setattr(I, "_ffprobe_available", lambda: False)
    assert I.header_backends("video") == [] and I.header_backends("image") == []
    assert I.missing_header_backends(["video", "audio", "image", "video"]) == ["video", "audio", "image"]
    with pytest.raises(DecoderMissing):
        I.probe_info(b"\x00" * 64, "video")


def test_header_backends_lists_ffprobe(no_py_backends, monkeypatch):
    monkeypatch.setattr(I, "_ffprobe_available", lambda: True)
    assert I.header_backends("audio") == ["ffprobe"] and I.missing_header_backends(["audio", "image"]) == []


# ----------------------------------------------------------------------------- doctor / extract start


@pytest.fixture
def bench_dir(tmp_path):
    from yavih import testing as T

    return T.make_synthetic_benchmark(tmp_path, "clf", n_train=8, n_eval=4, media="tiny")


def test_doctor_warns_when_no_census_and_no_backend(bench_dir, no_py_backends, monkeypatch, capsys):
    from yavih import cli

    monkeypatch.setattr(I, "_ffprobe_available", lambda: False)
    rc = cli.main(["--json", "doctor", "--bench", str(bench_dir)])
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0
    hb = env["data"]["media_info"]["header_backends"]
    assert hb and all(v == [] for v in hb.values())
    w = [x for x in env["warnings"] if x["code"] == "W_HEADER_BACKEND"]
    assert w and "--census" in w[0]["message"] and "ffprobe" in w[0]["message"]


def test_doctor_quiet_when_ffprobe_is_there(bench_dir, no_py_backends, monkeypatch, capsys):
    from yavih import cli

    monkeypatch.setattr(I, "_ffprobe_available", lambda: True)
    cli.main(["--json", "doctor", "--bench", str(bench_dir)])
    env = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert not [x for x in env["warnings"] if x["code"] == "W_HEADER_BACKEND"]


def test_extract_and_bridge_check_refuse_at_start(bench_dir, no_py_backends, monkeypatch, tmp_path):
    from yavih import extract
    from yavih.bridge import Bridge

    monkeypatch.setattr(I, "_ffprobe_available", lambda: False)

    class Never(Bridge):  # never constructed: the refusal comes before the model load
        def __init__(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("bridge constructed before the header-backend check")

    with pytest.raises(DecoderMissing, match="census"):
        extract.run_extract(bench=str(bench_dir), bridge=Never, split="eval", shard=(0, 1), out=tmp_path / "run")
    with pytest.raises(DecoderMissing, match="census"):
        extract.bridge_check(bridge=Never, bench=str(bench_dir))
    from yavih.cli import exit_code_for

    assert exit_code_for(DecoderMissing("av"))[0] == 3
