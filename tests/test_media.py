"""Media helpers (docs/bridge.md): ``yavih.media.rules`` (every declared rule, ``nearest_ties_earlier``
pins, ``crop_region`` geometry), header probes (``yavih.media.info``), the optional decode helpers
(``yavih.media.decode``) on files, bytes and tar members, the ffmpeg CLI fallback, ``.mat`` recovery, the tar
index and backend reporting. Nothing here has protocol semantics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from test_media_helpers import (
    frame_index_of,
    jpeg_bytes,
    needs_av,
    needs_pil,
    needs_scipy,
    needs_sf,
    png_bytes,
    write_av_mp4,
    write_geonet_mat,
    write_image,
    write_tar_with_index,
    write_video,
    write_wav,
)
from yavih.media import (
    DecodeError,
    DecoderMissing,
    MediaMissing,
    decoder_info,
    nearest_ties_earlier,
)
from yavih.media.rules import (
    TEMPORAL_RULES,
    OutOfMedia,
    audio_ranges,
    check_spatial,
    check_temporal,
    crop_region,
    rule_times,
    select_nearest,
)

AS20K_TRAIN_GRID = [float(i) for i in range(10)]
AS20K_EVAL_GRID = [i * 0.625 for i in range(16)]
BUCKET_CENTRES_8 = [k + 0.5 for k in range(8)]  # 8 x 1 s start-aligned bucket centres


# ----------------------------------------------------------------------------- nearest_ties_earlier


def test_select_nearest_pins_as20k_selections():
    """8 ``uniform`` times over ``[0, 8]`` → eval ``[1, 2, 4, 6, 7, 9, 10, 12]``, train ``[0..7]``."""
    times = rule_times({"rule": "uniform", "n": 8, "span": [0, 8]}, {"duration_s": 10.0})
    assert times.tolist() == BUCKET_CENTRES_8
    assert select_nearest(AS20K_EVAL_GRID, times).tolist() == [1, 2, 4, 6, 7, 9, 10, 12]
    assert select_nearest(AS20K_TRAIN_GRID, times).tolist() == list(range(8))
    # the package-level helper takes (times, pts)
    assert nearest_ties_earlier(BUCKET_CENTRES_8, AS20K_EVAL_GRID).tolist() == [1, 2, 4, 6, 7, 9, 10, 12]


def test_select_nearest_is_not_floor_and_breaks_exact_ties_earlier():
    assert select_nearest(AS20K_EVAL_GRID, BUCKET_CENTRES_8).tolist() != [0, 2, 4, 5, 7, 8, 10, 12]  # a floor rule
    assert select_nearest([0.0, 1.0], [0.5]).tolist() == [0]
    assert select_nearest([0.0, 1.0], [0.5 + 1e-12]).tolist() == [0]  # noise below the 1e-9 quantum
    assert select_nearest([0.0, 1.0], [0.6]).tolist() == [1]
    assert select_nearest([0.0, 1.0], [-5.0, 50.0]).tolist() == [0, 1]
    assert select_nearest([0.3, 0.3, 0.3], [0.3, 0.3]).tolist() == [0, 0]
    out = select_nearest(np.array([0.0, 1.0]), np.array([0.1]))
    assert out.dtype == np.int64 and out.shape == (1,)
    with pytest.raises(ValueError):
        select_nearest([], [0.1])


class _F:
    def __init__(self, t):
        self.time = t
        self.pts = None
        self.time_base = None


def test_streaming_selection_matches_full_scan():
    """``decode.frames_at``'s one-pass selector equals the full ``select_nearest`` scan on random ascending
    pts, exact pts and exact midpoints included."""
    from yavih.media.decode import _select_streaming

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 40))
        pts = np.round(np.cumsum(rng.choice([0.04, 0.05, 0.1], size=n)), 6)
        k = int(rng.integers(1, 12))
        times = np.round(
            np.concatenate([rng.uniform(pts[0] - 0.2, pts[-1] + 0.2, size=k), rng.choice(pts, size=min(3, n)), (pts[:-1] + pts[1:]) / 2 if n > 1 else np.array([pts[0]])]),
            6,
        )
        want = select_nearest(pts, times)
        got = _select_streaming(times, (_F(float(p)) for p in pts))
        assert [p for p, _ in got] == [float(pts[i]) for i in want]


# ----------------------------------------------------------------------------- rule_times


VIDEO = {"duration_s": 10.0, "fps": 25.0, "n_frames": 250}


def test_rule_times_uniform_and_edges():
    assert rule_times({"rule": "uniform", "n": 4, "span": "full"}, VIDEO).tolist() == [1.25, 3.75, 6.25, 8.75]
    assert rule_times({"rule": "uniform", "n": 2, "span": [2.0, 4.0]}, VIDEO).tolist() == [2.5, 3.5]
    short = {"duration_s": 3.0}
    with pytest.raises(OutOfMedia, match="edge: error") as ei:
        rule_times({"rule": "uniform", "n": 4, "span": [0.0, 8.0]}, short)
    assert ei.value.values == [5.0, 7.0]
    assert rule_times({"rule": "uniform", "n": 4, "span": [0.0, 8.0], "edge": "clamp"}, short).tolist() == [1.0, 3.0, 3.0, 3.0]
    with pytest.raises(ValueError, match="duration_s"):
        rule_times({"rule": "uniform", "n": 4, "span": "full"}, {"duration_s": None})


def test_rule_times_fps_index_linspace_explicit_all():
    assert rule_times({"rule": "fps", "fps": 2.0, "n": 4, "start": 1.0}, VIDEO).tolist() == [1.25, 1.75, 2.25, 2.75]
    centre = rule_times({"rule": "fps", "fps": 2.0, "n": 4, "start": "centre"}, VIDEO)
    assert centre.tolist() == [4.25, 4.75, 5.25, 5.75]  # (10 − 2)/2 = 4
    with pytest.raises(OutOfMedia):
        rule_times({"rule": "fps", "fps": 1.0, "n": 20, "start": 0.0}, VIDEO)
    assert rule_times({"rule": "fps", "fps": 1.0, "n": 20, "start": 0.0, "edge": "clamp"}, VIDEO)[-1] == 10.0
    # the index_linspace rule: round(linspace(0, n_frames − 1, n)) / fps (numpy round: half to even)
    ssv2 = {"duration_s": 4.0, "fps": 12.0, "n_frames": 48}
    idx = np.round(np.linspace(0, 47, 16))
    assert np.allclose(rule_times({"rule": "index_linspace", "n": 16}, ssv2), idx / 12.0)
    assert rule_times({"rule": "index_linspace", "n": 3}, {"n_frames": 5, "fps": 1.0}).tolist() == [0.0, 2.0, 4.0]
    with pytest.raises(ValueError, match="n_frames"):
        rule_times({"rule": "index_linspace", "n": 3}, {"fps": 1.0})
    assert rule_times({"rule": "explicit", "times": [1.0, 5.0, 8.0]}, VIDEO).tolist() == [1.0, 5.0, 8.0]
    with pytest.raises(OutOfMedia):
        rule_times({"rule": "explicit", "times": [11.0]}, VIDEO)
    assert rule_times({"rule": "explicit", "times": [-1.0, 11.0], "edge": "clamp"}, VIDEO).tolist() == [0.0, 10.0]
    assert rule_times({"rule": "all"}, {"n_frames": 4, "fps": 2.0}).tolist() == [0.0, 0.5, 1.0, 1.5]
    assert rule_times({"rule": "custom", "description": "first 16 frames"}, VIDEO) is None
    with pytest.raises(ValueError, match="audio_ranges"):
        rule_times({"rule": "span", "start_s": 0.0, "length_s": 1.0, "pad": "zeros"}, VIDEO)
    with pytest.raises(ValueError, match="unknown temporal rule"):
        rule_times({"rule": "random"}, VIDEO)


# ----------------------------------------------------------------------------- audio_ranges


AUDIO = {"sample_rate": 16000, "n_samples": 160000}  # 10 s


def test_audio_ranges_span():
    r, pad = audio_ranges({"rule": "span", "start_s": 1.0, "length_s": 8.0, "pad": "zeros"}, AUDIO)
    assert r.tolist() == [[16000, 144000]] and pad == (0, 0) and r.dtype == np.int64
    short = {"sample_rate": 16000, "n_samples": 100000}  # 6.25 s
    r, pad = audio_ranges({"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "zeros"}, short)
    assert r.tolist() == [[0, 100000]] and pad == (0, 60000)
    with pytest.raises(OutOfMedia, match="pad: none"):
        audio_ranges({"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "none"}, short)
    r, pad = audio_ranges({"rule": "span", "start_s": "centre", "length_s": 4.0, "pad": "none"}, AUDIO)
    assert r.tolist() == [[48000, 112000]] and pad == (0, 0)
    r, pad = audio_ranges({"rule": "span", "start_s": "centre", "length_s": 10.0, "pad": "zeros"}, short)
    assert r.tolist() == [[0, 100000]] and pad == (0, 60000)  # centred on a shorter file: start 0, right pad
    r, pad = audio_ranges({"rule": "span", "start_s": 9.0, "length_s": 4.0, "pad": "zeros"}, short)
    assert r.tolist() == [[100000, 100000]] and pad == (0, 64000)  # starts past the end: all padding


def test_audio_ranges_full_segments_custom():
    r, pad = audio_ranges({"rule": "full", "max_s": None, "pad_to_s": None}, AUDIO)
    assert r.tolist() == [[0, 160000]] and pad == (0, 0)
    r, pad = audio_ranges({"rule": "full", "max_s": 8.0, "pad_to_s": None}, AUDIO)
    assert r.tolist() == [[0, 128000]] and pad == (0, 0)
    r, pad = audio_ranges({"rule": "full", "max_s": None, "pad_to_s": 12.0}, AUDIO)
    assert r.tolist() == [[0, 160000]] and pad == (0, 32000)
    r, pad = audio_ranges({"rule": "segments", "n": 5, "length_s": 1.0, "span": "full", "pad": "zeros"}, AUDIO)
    assert r.tolist() == [[8000, 24000], [40000, 56000], [72000, 88000], [104000, 120000], [136000, 152000]] and pad == (0, 0)  # centres 1, 3, …, 9 s
    r, pad = audio_ranges({"rule": "segments", "n": 2, "length_s": 6.0, "span": "full", "pad": "zeros"}, AUDIO)
    assert r.tolist() == [[0, 88000], [72000, 160000]] and pad == (8000, 8000)  # centres 2.5 s and 7.5 s, ±3 s
    r, _ = audio_ranges({"rule": "segments", "n": 2, "length_s": 1.0, "span": [2.0, 4.0], "pad": "none"}, AUDIO)
    assert r.tolist() == [[32000, 48000], [48000, 64000]]
    with pytest.raises(OutOfMedia):
        audio_ranges({"rule": "segments", "n": 2, "length_s": 6.0, "span": "full", "pad": "none"}, AUDIO)
    # a segment wholly outside the media pads exactly its own length, never more
    one_s = {"sample_rate": 16000, "n_samples": 16000}
    r, pad = audio_ranges({"rule": "segments", "n": 2, "length_s": 1.0, "span": [0, 4], "pad": "zeros"}, one_s)
    assert r.tolist() == [[8000, 16000], [16000, 16000]] and pad == (0, 8000 + 16000)
    r, pad = audio_ranges({"rule": "segments", "n": 2, "length_s": 1.0, "span": [-3, 1], "pad": "zeros"}, one_s)
    assert r.tolist() == [[0, 0], [0, 8000]] and pad == (16000 + 8000, 0)
    assert audio_ranges({"rule": "custom", "description": "x"}, AUDIO) is None
    with pytest.raises(ValueError, match="rule_times"):
        audio_ranges({"rule": "uniform", "n": 2, "span": "full"}, AUDIO)
    with pytest.raises(ValueError, match="sample_rate"):
        audio_ranges({"rule": "full", "max_s": None, "pad_to_s": None}, {"n_samples": 10})


# ----------------------------------------------------------------------------- crop_region


RESIZE_CROP = {"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}


def test_crop_region_resize_short_centre_crop_4_3_and_16_9():
    """The part of the ORIGINAL frame a Resize(256) + CenterCrop(224) bridge sees."""
    y0, x0, y1, x1 = crop_region(RESIZE_CROP, 640, 480)  # 4:3 → 341 × 256, top 16, left round(58.5) = 58
    assert (y0, y1) == (16 / 256, 240 / 256) and (x0, x1) == (58 / 341, 282 / 341)
    y0, x0, y1, x1 = crop_region(RESIZE_CROP, 1280, 720)  # 16:9 → 455 × 256, top 16, left round(115.5) = 116
    assert (y0, y1) == (16 / 256, 240 / 256) and (x0, x1) == (116 / 455, 340 / 455)
    # the same geometry on a portrait frame (short side = width)
    y0, x0, y1, x1 = crop_region(RESIZE_CROP, 480, 640)
    assert (x0, x1) == (16 / 256, 240 / 256) and (y0, y1) == (58 / 341, 282 / 341)
    # vggss decision: 14×14 token centres of this bridge fall in 8 × 12 = 96 of 196 full-frame cells on 16:9
    y0, x0, y1, x1 = crop_region(RESIZE_CROP, 1280, 720)
    ys = y0 + (np.arange(14) + 0.5) * (y1 - y0) / 14
    xs = x0 + (np.arange(14) + 0.5) * (x1 - x0) / 14
    cells = {(int(y * 14), int(x * 14)) for y in ys for x in xs}
    assert len(cells) == 96


def test_crop_region_other_forms():
    assert crop_region({"resize": None, "crop": None, "interp": "bilinear", "antialias": True}, 640, 480) == (0.0, 0.0, 1.0, 1.0)
    assert crop_region({"resize": {"size": [224, 224]}, "crop": None, "interp": "bicubic", "antialias": False}, 640, 480) == (0.0, 0.0, 1.0, 1.0)
    region = {"resize": None, "crop": {"kind": "region", "region": [0.0, 0.125, 1.0, 0.875]}, "interp": "x", "antialias": True}
    assert crop_region(region, 640, 480) == (0.0, 0.125, 1.0, 0.875)
    longside = {"resize": {"long": 320}, "crop": {"kind": "center", "size": [240, 240]}, "interp": "x", "antialias": True}
    assert crop_region(longside, 640, 480) == (0.0, 40 / 320, 1.0, 280 / 320)  # 640×480 → 320×240
    big = {"resize": None, "crop": {"kind": "center", "size": [600, 600]}, "interp": "x", "antialias": True}
    assert crop_region(big, 640, 480) == (0.0, 20 / 640, 1.0, 620 / 640)  # padded rows clip to the frame
    assert crop_region({"rule": "custom", "description": "letterbox"}, 640, 480) is None
    with pytest.raises(ValueError, match="width and height"):
        crop_region(RESIZE_CROP, None, 480)


# ----------------------------------------------------------------------------- vocabulary


@pytest.mark.parametrize(
    "temporal,kind",
    [
        ({"rule": "uniform", "n": 16, "span": "full"}, "video"),
        ({"rule": "uniform", "n": 8, "span": [0, 8], "edge": "clamp"}, "video"),
        ({"rule": "fps", "fps": 2.0, "n": 16, "start": "centre"}, "video"),
        ({"rule": "index_linspace", "n": 16}, "video"),
        ({"rule": "explicit", "times": [5.0]}, "video"),
        ({"rule": "all"}, "video"),
        ({"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "zeros"}, "audio"),
        ({"rule": "full", "max_s": None, "pad_to_s": 10.0}, "audio"),
        ({"rule": "segments", "n": 4, "length_s": 2.0, "span": "full", "pad": "none"}, "audio"),
        ({"rule": "custom", "description": "the model's own loader"}, "image"),
    ],
)
def test_check_temporal_accepts_the_vocabulary(temporal, kind):
    check_temporal(temporal, kind)


@pytest.mark.parametrize(
    "temporal,kind,match",
    [
        ({"rule": "uniform", "n": 16}, "video", "span"),
        ({"rule": "uniform", "n": 0, "span": "full"}, "video", "n"),
        ({"rule": "uniform", "n": 4, "span": [5, 1]}, "video", "span"),
        ({"rule": "uniform", "n": 4, "span": "full", "edge": "wrap"}, "video", "edge"),
        ({"rule": "uniform", "n": 4, "span": "full", "crop": 1}, "video", "does not take"),
        ({"rule": "span", "start_s": 0.0, "length_s": 1.0, "pad": "zeros"}, "video", "not legal"),
        ({"rule": "fps", "fps": 0, "n": 4, "start": 0.0}, "video", "fps"),
        ({"rule": "explicit", "times": []}, "video", "times"),
        ({"rule": "custom", "description": " "}, "video", "description"),
        ({"rule": "random"}, "video", "not one of"),
        ("uniform", "video", "mapping"),
    ],
)
def test_check_temporal_refuses(temporal, kind, match):
    with pytest.raises(ValueError, match=match):
        check_temporal(temporal, kind)


def test_check_spatial():
    check_spatial(RESIZE_CROP)
    check_spatial({"resize": None, "crop": {"kind": "region", "region": [0, 0.125, 1, 0.875]}, "interp": "nearest", "antialias": False})
    check_spatial({"rule": "custom", "description": "pad to square then resize"})
    for bad, match in [
        ({"resize": {"short": 256}, "crop": None, "interp": "bilinear"}, "exactly"),
        ({**RESIZE_CROP, "resize": {"short": 0}}, "short"),
        ({**RESIZE_CROP, "resize": {"short": 256, "long": 300}}, "resize"),
        ({**RESIZE_CROP, "crop": {"kind": "random", "size": [1, 1]}}, "crop"),
        ({**RESIZE_CROP, "crop": {"kind": "center", "size": [224]}}, "size"),
        ({**RESIZE_CROP, "crop": {"kind": "region", "region": [0, 0, 2, 1]}}, "region"),
        ({**RESIZE_CROP, "antialias": "yes"}, "antialias"),
        ({"rule": "custom"}, "custom"),
    ]:
        with pytest.raises(ValueError, match=match):
            check_spatial(bad)
    assert set(TEMPORAL_RULES) == {"uniform", "fps", "index_linspace", "explicit", "all", "span", "full", "segments", "custom"}


# ----------------------------------------------------------------------------- backend reporting


def test_decoder_info_shape():
    for kind in ("video", "audio", "image"):
        info = decoder_info(kind)
        assert info is None or (set(info) == {"lib", "version", "libav"} and isinstance(info["version"], str))
    from yavih.media.decode import decoder_info as dinfo

    with pytest.raises(ValueError):
        dinfo("text")


@needs_av
def test_video_decoder_reports_pyav_and_libav():
    info = decoder_info("video")
    assert info["lib"] == "pyav" and info["libav"] and info["libav"].count(".") == 2
    assert decoder_info() == info  # default kind is video


@needs_sf
@needs_pil
def test_audio_and_image_decoders():
    assert decoder_info("audio")["lib"] == "soundfile" and decoder_info("audio")["libav"] is None
    assert decoder_info("image")["lib"] == "pillow"


def test_decoder_missing_message_has_pip_hint(monkeypatch):
    import yavih.media.decode as dec

    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(DecoderMissing) as ei:
        dec._av()
    assert ei.value.package == "av" and "pip install" in str(ei.value)
    assert isinstance(ei.value, ImportError)
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(DecoderMissing, match="pillow"):
        dec.decoder_info("image")
    assert decoder_info("image") is None  # the package-level helper reports absence as None


def test_media_missing_and_removed_window_error():
    e = MediaMissing("clip1", "video", "absent file /x.mp4")
    assert isinstance(e, FileNotFoundError) and e.reason == "absent file /x.mp4" and "clip1" in str(e) and "video" in str(e)
    import yavih.media as m

    with pytest.raises(AttributeError):  # WindowError does not exist (no reader windows)
        m.WindowError  # noqa: B018
    with pytest.raises(AttributeError):
        m.not_a_thing  # noqa: B018


# ----------------------------------------------------------------------------- header probes (yavih.media.info)


@needs_av
def test_probe_info_video_from_path_bytes_and_file(tmp_path: Path):
    from yavih.media.info import probe_info
    from yavih.sample import MEDIA_INFO_KEYS

    p = tmp_path / "v.mp4"
    write_video(p, fps=10, n_frames=20, width=64, height=48)
    info = probe_info(p, "video")
    assert set(MEDIA_INFO_KEYS) <= set(info)
    assert info["duration_s"] == pytest.approx(2.0, abs=0.11) and info["fps"] == 10.0 and info["n_frames"] == 20
    assert (info["width"], info["height"], info["container"], info["bytes"]) == (64, 48, "mp4", p.stat().st_size)
    assert info["sample_rate"] is None and info["n_samples"] is None
    assert probe_info(p.read_bytes(), "video", ext="mp4") == info
    with open(p, "rb") as fh:
        assert probe_info(fh, "video", ext="mp4") == info
    with pytest.raises(DecodeError):
        probe_info(b"not a video", "video")
    with pytest.raises(ValueError):
        probe_info(p, "text")


@needs_av
def test_probe_info_webm_frame_count_by_demux(tmp_path: Path):
    """webm headers carry no frame count: the census demuxes packets (nothing decoded), as for ssv2."""
    import av

    from yavih.media.info import probe_info

    p = tmp_path / "v.webm"
    try:
        with av.open(str(p), "w") as c:
            s = c.add_stream("libvpx", rate=12)
            s.width, s.height, s.pix_fmt = 32, 24, "yuv420p"
            for i in range(12):
                f = av.VideoFrame.from_ndarray(np.full((24, 32, 3), i * 10, np.uint8), format="rgb24")
                f.pts = i
                for pk in s.encode(f):
                    c.mux(pk)
            for pk in s.encode(None):
                c.mux(pk)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no libvpx encoder: {e}")
    plain = probe_info(p, "video")
    assert plain["container"] == "webm" and plain["n_frames"] in (None, 12)
    assert probe_info(p, "video", demux=True)["n_frames"] == 12


@needs_sf
def test_probe_info_wav(tmp_path: Path):
    from yavih.media.info import probe_info

    p = tmp_path / "a.wav"
    data = write_wav(p, seconds=1.5, rate=16000, channels=2)
    info = probe_info(p, "audio")
    assert (info["sample_rate"], info["n_samples"], info["channels"], info["container"]) == (16000, data.shape[0], 2, "wav")
    assert info["duration_s"] == data.shape[0] / 16000 and info["width"] is None and info["fps"] is None


@needs_av
def test_probe_info_container_audio_and_video_tracks_of_one_mp4(tmp_path: Path):
    from yavih.media.info import probe_info

    p = tmp_path / "av.mp4"
    _, audio = write_av_mp4(p, fps=10, n_frames=20, rate=16000)
    a = probe_info(p, "audio", demux=True)
    v = probe_info(p, "video")
    assert (a["sample_rate"], a["n_samples"], a["channels"], a["container"]) == (16000, audio.shape[0], 1, "mp4")
    assert a["duration_s"] == 2.0 and a["fps"] is None
    assert (v["fps"], v["n_frames"], v["width"], v["sample_rate"]) == (10.0, 20, 64, None)


@needs_pil
def test_probe_info_images_and_container_names(tmp_path: Path):
    from yavih.media.info import container_name, probe_info

    p = tmp_path / "i.png"
    write_image(p, height=40, width=60)
    info = probe_info(p, "image")
    assert (info["width"], info["height"], info["container"], info["bytes"]) == (60, 40, "png", p.stat().st_size)
    j = probe_info(jpeg_bytes(np.zeros((8, 10, 3), np.uint8)), "image", ext="JPEG")
    assert (j["width"], j["height"], j["container"]) == (10, 8, "jpeg") and j["duration_s"] is None
    assert container_name("mov,mp4,m4a,3gp,3g2,mj2", None) == "mp4"
    assert container_name("mov,mp4,m4a,3gp,3g2,mj2", "mov") == "mov"
    assert container_name("matroska,webm", "mkv") == "mkv" and container_name("matroska,webm", None) == "webm"
    assert container_name("wav", None) == "wav" and container_name(None, "jpg") == "jpeg"


# ----------------------------------------------------------------------------- decode helpers (yavih.media.decode)


@needs_av
def test_frames_at_nearest_by_pts_on_path_bytes_and_tar_member(tmp_path: Path):
    from yavih.media.decode import all_pts, frames_at, stored_frames

    p = tmp_path / "v.mp4"
    pts = write_video(p, fps=10, n_frames=20)
    assert np.allclose(all_pts(p), pts)
    times = rule_times({"rule": "uniform", "n": 4, "span": "full"}, {"duration_s": 2.0})  # 0.25 0.75 1.25 1.75
    frames, got = frames_at(p, times)
    assert frames.shape == (4, 48, 64, 3) and frames.dtype == np.uint8
    assert frame_index_of(frames) == [2, 7, 12, 17] and got.tolist() == [0.2, 0.7, 1.2, 1.7]  # 0.25 ties → 0.2
    frames_b, got_b = frames_at(p.read_bytes(), times)
    assert np.array_equal(frames_b, frames) and np.array_equal(got_b, got)
    # no protocol semantics: an out-of-media time gets the first / last frame; the bridge's edge decides legality
    _, edge = frames_at(p, [-3.0, 99.0])
    assert edge.tolist() == [0.0, 1.9]
    # order of requested times is kept, repeats allowed
    _, rep = frames_at(p, [1.0, 0.0, 1.0])
    assert rep.tolist() == [1.0, 0.0, 1.0]
    assert frame_index_of(stored_frames(p, [3, 0, 3])) == [3, 0, 3]
    with pytest.raises(IndexError):
        stored_frames(p, [25])
    with pytest.raises(DecodeError):
        frames_at(b"junk", [0.0])
    with pytest.raises(ValueError):
        frames_at(p, [])


@needs_av
@needs_pil
def test_decode_helpers_accept_raw_streams_read_in_place(tmp_path: Path):
    """A RawStream on a tar member decodes through ``open()`` (no temp file): video by time, image whole."""
    from yavih.media.decode import frames_at, image
    from yavih.sample import RawStream, _Source

    v = tmp_path / "v.mp4"
    write_video(v, fps=10, n_frames=10)
    imgs = [np.full((8, 8, 3), 40 * i, dtype=np.uint8) for i in range(3)]
    idx_path = write_tar_with_index(tmp_path / "s.tar", {"v.mp4": v.read_bytes(), **{f"f/{i}.png": png_bytes(im) for i, im in enumerate(imgs)}})
    from yavih.media.tarindex import TarIndex

    idx = TarIndex(str(idx_path))
    st = RawStream("video", "video", src=_Source(("member", *idx.locate("v.mp4"))))
    fr, pts = frames_at(st, [0.5])
    assert frame_index_of(fr) == [5] and pts.tolist() == [0.5]
    im = RawStream("image", "image", src=_Source(("member", *idx.locate("f/2.png"))), ext="png")
    assert np.array_equal(image(im), imgs[2])


@needs_sf
def test_audio_whole_track_and_ranges(tmp_path: Path):
    from yavih.media.decode import audio

    p = tmp_path / "a.wav"
    data = write_wav(p, seconds=1.0, rate=8000, channels=2)
    wav, rate = audio(p)
    assert rate == 8000 and wav.shape == (8000, 2) and wav.dtype == np.float32 and np.array_equal(wav, data)
    ranges, pad = audio_ranges({"rule": "span", "start_s": 0.5, "length_s": 1.0, "pad": "zeros"}, {"sample_rate": 8000, "n_samples": 8000})
    seg, _ = audio(p.read_bytes(), ranges)
    assert np.array_equal(seg, data[4000:8000]) and pad == (0, 4000)  # the helper never pads; the bridge does
    two, _ = audio(p, [[0, 10], [100, 110]])
    assert np.array_equal(two, np.concatenate([data[:10], data[100:110]]))
    with pytest.raises(DecodeError):
        audio(b"junk")


@needs_av
def test_audio_from_mp4_falls_back_to_pyav_bit_exact(tmp_path: Path):
    """AVE / avsync: the audio track of an mp4 (ALAC, lossless) decodes through PyAV, sample-exact."""
    from yavih.media.decode import audio

    p = tmp_path / "av.mp4"
    _, written = write_av_mp4(p, fps=10, n_frames=10, rate=8000)
    wav, rate = audio(p)
    assert rate == 8000 and wav.shape == (written.shape[0], 1)
    assert np.array_equal(wav[:, 0], written.astype(np.float32) / np.float32(2**15))


@needs_pil
def test_image_helper(tmp_path: Path):
    from yavih.media.decode import image

    p = tmp_path / "i.png"
    want = write_image(p)
    assert np.array_equal(image(p), want) and np.array_equal(image(p.read_bytes()), want)
    out = image(jpeg_bytes(np.full((16, 16, 3), 128, np.uint8)))
    assert out.shape == (16, 16, 3) and abs(int(out.mean()) - 128) <= 2
    with pytest.raises(DecodeError):
        image(b"junk")


@pytest.mark.skipif(not __import__("yavih.media.ffmpeg", fromlist=["x"]).ffmpeg_available(), reason="ffmpeg CLI not on PATH")
@needs_av
def test_ffmpeg_cli_fallback_matches_pyav(tmp_path: Path, monkeypatch):
    from yavih.media.decode import frames_at
    from yavih.media.ffmpeg import ffmpeg_info, ffprobe_pts, ffprobe_video

    p = tmp_path / "v.mp4"
    pts = write_video(p, fps=10, n_frames=12)
    assert np.allclose(ffprobe_pts(str(p)), pts)
    hdr = ffprobe_video(str(p))
    assert (hdr["width"], hdr["height"], hdr["n_frames"]) == (64, 48, 12) and hdr["fps"] == 10.0
    assert ffmpeg_info()["backend"] == "ffmpeg"
    want, want_pts = frames_at(p, [0.05, 0.6, 1.1])
    monkeypatch.setitem(sys.modules, "av", None)
    got, got_pts = frames_at(p, [0.05, 0.6, 1.1])
    assert np.array_equal(got, want) and np.array_equal(got_pts, want_pts)
    with pytest.raises(DecoderMissing):
        frames_at(p.read_bytes(), [0.0])  # bytes need PyAV; only a file on disk has a CLI fallback


# ----------------------------------------------------------------------------- .mat (prepare-time helper)


@needs_scipy
def test_geonet_mat_recovery_lives_in_the_nyuv2_folder(tmp_path: Path):
    """The GeoNet ``.mat`` recovery is nyuv2_depth's prepare-time step (``prepare/recontainer_png.py``): truncate to
    480×640, add the channel offsets in float32, rint, clip, uint8."""
    import importlib.util

    from yavih.benchmarks import packaged_folder

    path = packaged_folder("nyuv2_depth") / "prepare" / "recontainer_png.py"
    spec = importlib.util.spec_from_file_location("nyuv2_recontainer_png_media_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = tmp_path / "x.mat"
    want = write_geonet_mat(p)
    got = mod.load_geonet_rgb(str(p))
    assert got.shape == (480, 640, 3) and got.dtype == np.uint8
    assert np.array_equal(got, want)
    with pytest.raises(ValueError):
        mod.geonet_rgb(np.zeros((10, 10, 3), np.float32))


# ----------------------------------------------------------------------------- tar index


def test_tar_index_reads_members_with_pread(tmp_path: Path):
    from yavih.media.tarindex import TAR_INDEX_VERSION, TarIndex, open_tar_index, pread_member

    members = {"frames/train/clipA/0.jpg": b"A0" * 10, "frames/train/clipA/1.jpg": b"A1" * 700, "audio/clipA.wav": os.urandom(1234)}
    idx_path = write_tar_with_index(tmp_path / "shard.tar", members)
    idx = TarIndex(str(idx_path))
    assert TAR_INDEX_VERSION == 1 and len(idx) == 3 and sorted(idx.keys()) == sorted(members)
    for k, v in members.items():
        assert idx.has(k) and idx.read(k) == v
        shard, off, size = idx.locate(k)
        assert shard == str(tmp_path / "shard.tar") and pread_member(shard, off, size) == v
    assert not idx.has("missing") and idx.locate("missing") is None
    with pytest.raises(KeyError):
        idx.read("missing")
    with pytest.raises(EOFError, match="short read"):
        pread_member(str(tmp_path / "shard.tar"), 0, 10**8)
    assert open_tar_index(str(idx_path)) is open_tar_index(str(idx_path))
    import pickle

    clone = pickle.loads(pickle.dumps(idx))
    assert clone.read("audio/clipA.wav") == members["audio/clipA.wav"]
    idx.close()


def test_tar_index_validates_version_and_shape(tmp_path: Path):
    from yavih.media.tarindex import TarIndex

    bad = tmp_path / "index.json"
    bad.write_text('{"version": 2, "shards": [], "members": {}}')
    with pytest.raises(ValueError, match="version"):
        TarIndex(str(bad))
    bad.write_text('{"version": 1, "members": {}}')
    with pytest.raises(ValueError, match="shards"):
        TarIndex(str(bad))
    with pytest.raises(FileNotFoundError):
        TarIndex(str(tmp_path / "nope.json"))


def test_tar_index_absolute_shard_path_ignores_root(tmp_path: Path):
    """An index with an ABSOLUTE shard path ignores shard_root (such an index is not relocatable); a relative one
    resolves against it."""
    from yavih.media.tarindex import TarIndex

    members = {"k": b"xyz"}
    tar = tmp_path / "data" / "s.tar"
    idx_rel = write_tar_with_index(tar, members, tmp_path / "data" / "index.json")
    idx_abs = write_tar_with_index(tar, members, tmp_path / "abs_index.json", shard_name=str(tar))
    assert TarIndex(str(idx_rel)).read("k") == b"xyz"
    assert TarIndex(str(idx_abs), shard_root=str(tmp_path / "elsewhere")).shard_path(0) == str(tar)
    assert TarIndex(str(idx_rel), shard_root=str(tmp_path / "data")).read("k") == b"xyz"
    with pytest.raises(FileNotFoundError):
        TarIndex(str(idx_rel), shard_root=str(tmp_path / "elsewhere")).read("k")


def test_tar_index_short_read_is_an_error(tmp_path: Path):
    import json

    from yavih.media.tarindex import TarIndex

    idx_path = write_tar_with_index(tmp_path / "s.tar", {"k": b"x" * 4096})
    d = json.loads(idx_path.read_text())
    d["members"]["k"][2] = 10**7  # stale size
    idx_path.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="short read"):
        TarIndex(str(idx_path)).read("k")


def test_media_package_imports_nothing_optional():
    """``yavih.media`` and ``yavih.media.rules`` import numpy + stdlib only."""
    import subprocess

    code = (
        "import sys\n"
        "for m in ('av', 'soundfile', 'PIL', 'scipy', 'torch'): sys.modules[m] = None\n"
        "import yavih.media, yavih.media.rules, yavih.media.tarindex, yavih.media.info, yavih.media.decode, yavih.media.ffmpeg\n"
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


@needs_av
@needs_sf
def test_source_prefers_cached_bytes_and_frames_identical_to_path(tmp_path: Path):
    """A RawStream decodes from its stored path until the read-ahead cached its bytes, then from memory (no second
    read of the store); both give the same frames, pts and samples."""
    import io

    from yavih.media import decode
    from yavih.sample import RawStream, _Source

    vid, wav = tmp_path / "v.mp4", tmp_path / "a.wav"
    write_video(vid, fps=10, n_frames=20, width=64, height=48)
    write_wav(wav, seconds=1.0, rate=16000, channels=2)
    for p, kind in ((vid, "video"), (wav, "audio")):
        st = RawStream(kind, kind, src=_Source(("file", str(p))), clip_id="c", ext=p.suffix[1:])

        def dec(st=st, kind=kind):
            return decode.frames_at(st, [0.05, 0.92, 1.5]) if kind == "video" else decode.audio(st, [[100, 900]])

        assert isinstance(decode._source(st), str)  # nothing cached yet: by path
        by_path = dec()
        st.bytes()
        assert isinstance(decode._source(st), io.BytesIO)  # cached by the read-ahead: from memory
        p.unlink()  # the store is gone: a decode that read it again would fail
        cached = dec()
        for x, y in zip(by_path, cached, strict=True):
            assert np.asarray(x).dtype == np.asarray(y).dtype and np.asarray(x).tobytes() == np.asarray(y).tobytes()
        st._src.release()
        assert isinstance(decode._source(st), str)  # released: back to the path
