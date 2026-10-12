"""The rule evaluators validate uses to re-check a bridge's ``used`` report
(``select_nearest``, ``rule_times``, ``audio_ranges``, ``crop_region``), pinned value by value as validate
relies on them. They ARE ``yavih.media.rules`` (one semantics for bridges and validate); the module has its
own tests (test_media.py)."""

from __future__ import annotations

import numpy as np
import pytest

import yavih.media.rules as media_rules
from yavih.io import validate as v
from yavih.io.validate import audio_ranges, crop_region, rule_times, select_nearest
from yavih.media.rules import OutOfMedia

VIDEO = {"duration_s": 10.0, "fps": 25.0, "n_frames": 250}
SSV2 = {"duration_s": 4.0, "fps": 12.0, "n_frames": 48}
AUDIO = {"duration_s": 10.0, "sample_rate": 16000, "n_samples": 160000}
SHORT_AUDIO = {"duration_s": 9.375, "sample_rate": 16000, "n_samples": 150000}
EVAL_PTS = [k * 0.625 for k in range(16)]  # 16 frames over 10 s
TRAIN_PTS = [float(k) for k in range(10)]  # 10 frames over 10 s


def test_select_nearest_pinned_selection():
    times = rule_times({"rule": "uniform", "n": 8, "span": [0.0, 8.0]}, {"duration_s": 10.0})
    np.testing.assert_allclose(times, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    assert select_nearest(EVAL_PTS, times).tolist() == [1, 2, 4, 6, 7, 9, 10, 12]
    assert select_nearest(TRAIN_PTS, times).tolist() == list(range(8))  # exact ties go to the earlier frame


@pytest.mark.parametrize(
    "temporal, info, expected",
    [
        ({"rule": "uniform", "n": 4, "span": "full"}, VIDEO, [1.25, 3.75, 6.25, 8.75]),
        ({"rule": "uniform", "n": 2, "span": [2.0, 6.0]}, VIDEO, [3.0, 5.0]),
        ({"rule": "uniform", "n": 2, "span": [2.0, 14.0], "edge": "clamp"}, VIDEO, [5.0, 10.0]),
        ({"rule": "fps", "fps": 2.0, "n": 4, "start": 1.0}, VIDEO, [1.25, 1.75, 2.25, 2.75]),
        ({"rule": "fps", "fps": 1.0, "n": 4, "start": "centre"}, VIDEO, [3.5, 4.5, 5.5, 6.5]),
        ({"rule": "index_linspace", "n": 8}, SSV2, [i / 12 for i in (0, 7, 13, 20, 27, 34, 40, 47)]),
        ({"rule": "explicit", "times": [5.0]}, VIDEO, [5.0]),
        ({"rule": "explicit", "times": [12.0], "edge": "clamp"}, VIDEO, [10.0]),
        ({"rule": "all"}, {"duration_s": 0.2, "fps": 25.0, "n_frames": 5}, [0.0, 0.04, 0.08, 0.12, 0.16]),
    ],
)
def test_rule_times(temporal, info, expected):
    np.testing.assert_allclose(rule_times(temporal, info), expected, atol=1e-12)


def test_rule_times_edge_error_and_missing_facts():
    with pytest.raises(OutOfMedia):  # validate's E_TIMES_OUT_OF_MEDIA
        rule_times({"rule": "uniform", "n": 2, "span": [2.0, 14.0]}, VIDEO)
    with pytest.raises(ValueError):  # validate treats a missing fact as unchecked
        rule_times({"rule": "index_linspace", "n": 4}, {"duration_s": 4.0})
    assert rule_times({"rule": "custom", "description": "x"}, VIDEO) is None


@pytest.mark.parametrize(
    "temporal, info, ranges, pad",
    [
        ({"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "zeros"}, AUDIO, [[0, 160000]], (0, 0)),
        ({"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "zeros"}, SHORT_AUDIO, [[0, 150000]], (0, 10000)),
        ({"rule": "span", "start_s": 1.0, "length_s": 8.0, "pad": "none"}, AUDIO, [[16000, 144000]], (0, 0)),
        ({"rule": "span", "start_s": "centre", "length_s": 8.0, "pad": "zeros"}, AUDIO, [[16000, 144000]], (0, 0)),
        ({"rule": "span", "start_s": "centre", "length_s": 10.0, "pad": "zeros"}, SHORT_AUDIO, [[0, 150000]], (0, 10000)),
        ({"rule": "full", "max_s": None, "pad_to_s": None}, SHORT_AUDIO, [[0, 150000]], (0, 0)),
        ({"rule": "full", "max_s": 5.0, "pad_to_s": 10.0}, AUDIO, [[0, 80000]], (0, 80000)),
        ({"rule": "full", "max_s": None, "pad_to_s": 10.0}, SHORT_AUDIO, [[0, 150000]], (0, 10000)),
        ({"rule": "segments", "n": 2, "length_s": 2.0, "span": "full", "pad": "zeros"}, AUDIO, [[24000, 56000], [104000, 136000]], (0, 0)),
        ({"rule": "segments", "n": 2, "length_s": 6.0, "span": "full", "pad": "zeros"}, AUDIO, [[0, 88000], [72000, 160000]], (8000, 8000)),
    ],
)
def test_audio_ranges(temporal, info, ranges, pad):
    got, got_pad = audio_ranges(temporal, info)
    assert got.tolist() == ranges and tuple(got_pad) == pad


def test_audio_ranges_pad_none_and_missing_facts():
    with pytest.raises(OutOfMedia):
        audio_ranges({"rule": "span", "start_s": 0.0, "length_s": 10.0, "pad": "none"}, SHORT_AUDIO)
    with pytest.raises(ValueError):
        audio_ranges({"rule": "full", "max_s": None, "pad_to_s": None}, {"duration_s": 1.0})
    assert audio_ranges({"rule": "custom", "description": "x"}, AUDIO) is None


@pytest.mark.parametrize(
    "spatial, w, h, region",
    [
        # torchvision: Resize(224) of 320x240 is 298x224, CenterCrop(224) keeps columns [37, 261)
        ({"resize": {"short": 224}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}, 320, 240, (0.0, 37 / 298, 1.0, 261 / 298)),
        ({"resize": {"short": 224}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}, 1280, 720, (0.0, 87 / 398, 1.0, 311 / 398)),
        ({"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True}, 640, 480, (16 / 256, 58 / 341, 240 / 256, 282 / 341)),
        ({"resize": None, "crop": None}, 640, 480, (0.0, 0.0, 1.0, 1.0)),
        ({"resize": {"size": [224, 224]}, "crop": None}, 640, 480, (0.0, 0.0, 1.0, 1.0)),
        ({"resize": {"long": 224}, "crop": {"kind": "region", "region": [0.0, 0.125, 1.0, 0.875]}}, 640, 480, (0.0, 0.125, 1.0, 0.875)),
        ({"resize": None, "crop": {"kind": "center", "size": [1000, 1000]}}, 640, 480, (0.0, 0.0, 1.0, 1.0)),  # crop larger than the frame
    ],
)
def test_crop_region(spatial, w, h, region):
    np.testing.assert_allclose(crop_region(spatial, w, h), region, atol=1e-12)


def test_crop_region_custom_and_degenerate():
    assert crop_region({"rule": "custom", "description": "letterbox"}, 640, 480) is None
    with pytest.raises(ValueError):
        crop_region({"resize": None, "crop": {"kind": "center", "size": [2, 2]}}, 0, 480)


# ----------------------------------------------------------------------------- one semantics


def test_validate_uses_yavih_media_rules():
    """A bridge that picks its times with ``yavih.media.rules`` is judged by the very same functions."""
    for name in ("select_nearest", "rule_times", "audio_ranges", "crop_region"):
        assert getattr(v, name) is getattr(media_rules, name)
