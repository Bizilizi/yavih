"""The ``fps`` rule's options: ``phase`` (default 0.5 = bin centres; 0 = window start, ``start + k/fps``),
``start: "centre_clamped"`` (``max(0, (duration_s - n/fps)/2)``) and ``duration_from``.
"""

from __future__ import annotations

import numpy as np
import pytest

from yavih import hashing
from yavih.bridge import DescribeError, validate_input
from yavih.io.validate import _check_times, _Findings
from yavih.media.rules import (
    FPS_DEFAULT_PHASE,
    TEMPORAL_PARAMS,
    OutOfMedia,
    check_temporal,
    rule_facts,
    rule_times,
)

INFO_12S = {"duration_s": 12.0, "fps": 25.0, "n_frames": 300}
INFO_SHORT = {"duration_s": 0.83, "fps": 12.0, "n_frames": 10}  # an SSv2-length clip


def _centred_window_times(total: int, native: float, n: int = 16, frame_fps: float = 4.0) -> np.ndarray:
    """A centred-window recipe: ``n`` frames at ``frame_fps`` around the clip centre, the media seconds of the
    frames it decodes (nearest index ``round(t * native)`` clamped to ``[0, total - 1]``)."""
    start = max(0.0, (total / native - n / frame_fps) / 2.0)
    t = start + np.arange(n, dtype=np.float64) / frame_fps
    idx = np.clip(np.round(t * native), 0, total - 1)
    return idx / native


# ----------------------------------------------------------------------------- defaults


def test_fps_default_phase_is_bin_centres():
    rule = {"rule": "fps", "fps": 4, "n": 16, "start": "centre"}
    want = (12.0 - 4.0) / 2.0 + (np.arange(16) + 0.5) / 4.0
    assert FPS_DEFAULT_PHASE == 0.5
    np.testing.assert_allclose(rule_times(rule, INFO_12S), want)
    np.testing.assert_allclose(rule_times(dict(rule, phase=0.5), INFO_12S), want)
    np.testing.assert_allclose(rule_times({"rule": "fps", "fps": 2, "n": 3, "start": 1.0}, INFO_12S), [1.25, 1.75, 2.25])


def test_fps_required_params_and_optional_phase():
    req, opt = TEMPORAL_PARAMS["fps"]
    assert req == frozenset({"fps", "n", "start"})
    assert {"edge", "phase"} <= opt
    assert rule_facts({"rule": "fps", "fps": 4, "n": 16, "start": "centre_clamped", "phase": 0}, "video") == frozenset({"duration_s"})


def test_absent_optional_key_keeps_the_describe_hash():
    """Adding an optional key must not change the describe hash of a declaration that does not use it."""
    d = {"input": {"video": {"temporal": {"rule": "fps", "fps": 4, "n": 16, "start": "centre", "edge": "clamp"}}}}
    before = hashing.describe_sha256(d)
    check_temporal(d["input"]["video"]["temporal"], "video")
    assert hashing.describe_sha256(d) == before
    assert hashing.describe_sha256({"input": {"video": {"temporal": dict(d["input"]["video"]["temporal"], phase=0)}}}) != before


# ----------------------------------------------------------------------------- phase 0 and centre_clamped


def test_fps_phase_zero_is_window_start():
    rule = {"rule": "fps", "fps": 4, "n": 16, "start": "centre", "phase": 0}
    np.testing.assert_allclose(rule_times(rule, INFO_12S), 4.0 + np.arange(16) / 4.0)
    rule = {"rule": "fps", "fps": 4, "n": 4, "start": 2.0, "phase": 0.25}
    np.testing.assert_allclose(rule_times(rule, INFO_12S), [2.0625, 2.3125, 2.5625, 2.8125])


def test_fps_centre_clamped_on_short_clip():
    clamped = {"rule": "fps", "fps": 4, "n": 16, "start": "centre_clamped", "phase": 0, "edge": "clamp"}
    t = rule_times(clamped, INFO_SHORT)
    # start = max(0, (0.83 - 4)/2) = 0: frames from 0 at 4 fps, those past the end clamped to duration_s
    np.testing.assert_allclose(t, np.minimum(np.arange(16) / 4.0, 0.83))
    assert t[0] == 0.0
    # "centre" on the same clip starts before 0: clamping gives a different (front-loaded) set
    centre = dict(clamped, start="centre")
    assert not np.allclose(rule_times(centre, INFO_SHORT), t)
    # on a long clip both agree
    np.testing.assert_allclose(rule_times(centre, INFO_12S), rule_times(clamped, INFO_12S))
    with pytest.raises(OutOfMedia):
        rule_times(dict(clamped, edge="error"), INFO_SHORT)


@pytest.mark.parametrize(
    "bad",
    [
        {"phase": 1.0},
        {"phase": -0.1},
        {"phase": "0"},
        {"phase": True},
        {"phase": None},
        {"start": "middle"},
        {"start": "Centre_Clamped"},
        {"start": [0.0]},
    ],
)
def test_check_temporal_refuses_bad_fps_phase_or_start(bad):
    rule = dict({"rule": "fps", "fps": 4, "n": 16, "start": "centre_clamped", "phase": 0}, **bad)
    with pytest.raises(ValueError):
        check_temporal(rule, "video")


def test_validate_input_accepts_window_start_fps_rule():
    entry = {
        "temporal": {"rule": "fps", "fps": 4, "n": 16, "start": "centre_clamped", "phase": 0, "edge": "clamp"},
        "spatial": {"resize": {"short": 256}, "crop": {"kind": "center", "size": [224, 224]}, "interp": "bilinear", "antialias": True},
        "norm": {"mean": [0.5] * 3, "std": [0.5] * 3, "source": "example/model@abc eval.py"},
        "decoder": {"lib": "torchcodec", "version": "0.1"},
    }
    validate_input({"video": entry}, {"video": "video"}, ["video"])
    bad = dict(entry, temporal=dict(entry["temporal"], phase=2))
    with pytest.raises(DescribeError, match="phase"):
        validate_input({"video": bad}, {"video": "video"}, ["video"])


# ----------------------------------------------------------------------------- validate checks a centred-window recipe


@pytest.mark.parametrize(
    "info",
    [
        {"duration_s": 12.0, "fps": 25.0, "n_frames": 300},  # k400 val-like
        {"duration_s": 300 / 29.97, "fps": 29.97, "n_frames": 300},
        {"duration_s": 0.83, "fps": 12.0, "n_frames": 10},  # SSv2's shortest
        {"duration_s": 2.5, "fps": 30.0, "n_frames": 75},
    ],
)
def test_validate_checks_centred_window_with_phase_zero(info):
    """A centred-window recipe passes E_RULE_MISMATCH under phase 0 + centre_clamped and fails under the
    bin-centre default (0.125 s off at 4 fps, far beyond 0.5/fps + 1 ms). The recipe centres on
    ``n_frames / fps``, the rule on ``duration_s``: the media facts here agree, as they do for a clip whose
    container duration is its video track's."""
    used = {"times": _centred_window_times(int(info["n_frames"]), float(info["fps"])).tolist()}
    rule = {"rule": "fps", "fps": 4, "n": 16, "start": "centre_clamped", "phase": 0, "edge": "clamp"}
    fx = _Findings()
    assert _check_times("video", rule, used, info, "row0", fx) is True
    assert fx.ok, [f.message for f in fx.errors]

    fx_default = _Findings()
    _check_times("video", {k: v for k, v in rule.items() if k != "phase"}, used, info, "row0", fx_default)
    if info["duration_s"] >= 4.0:  # on clips shorter than the window clamping hides part of the phase shift
        assert fx_default.has("E_RULE_MISMATCH")


# ----------------------------------------------------------------------------- duration_from: frame_count


def test_fps_duration_from_frame_count_centres_on_the_video_track():
    # container says 10.4 s (audio runs longer), the video track holds 250 frames at 25 fps = 10.0 s
    info = {"duration_s": 10.4, "fps": 25.0, "n_frames": 250}
    base = {"rule": "fps", "fps": 4, "n": 16, "start": "centre_clamped", "phase": 0, "edge": "clamp"}
    np.testing.assert_allclose(rule_times(base, info), 3.2 + np.arange(16) / 4.0)
    np.testing.assert_allclose(rule_times(dict(base, duration_from="duration_s"), info), 3.2 + np.arange(16) / 4.0)
    np.testing.assert_allclose(rule_times(dict(base, duration_from="frame_count"), info), 3.0 + np.arange(16) / 4.0)
    assert rule_facts(dict(base, duration_from="frame_count"), "video") == frozenset({"n_frames", "fps"})
    assert rule_facts(dict(base, duration_from="duration_s"), "video") == frozenset({"duration_s"})


def test_fps_duration_from_frame_count_clamps_to_the_last_frame():
    """A frame index clamped to n_frames - 1 gives the LAST frame's time, not n_frames / fps
    (SSv2 webm: 30 frames at 12 fps, container duration 3.916 s because of gaps)."""
    info = {"duration_s": 3.916, "fps": 12.0, "n_frames": 30}
    rule = {"rule": "fps", "fps": 4, "n": 16, "start": "centre_clamped", "phase": 0, "edge": "clamp", "duration_from": "frame_count"}
    t = rule_times(rule, info)
    assert t.max() == pytest.approx(29 / 12)
    np.testing.assert_allclose(t[:10], np.arange(10) / 4.0)
    used = {"times": _centred_window_times(30, 12.0).tolist()}
    fx = _Findings()
    _check_times("video", rule, used, info, "row0", fx)
    assert fx.ok, [f.message for f in fx.errors]
    with pytest.raises(OutOfMedia):
        rule_times(dict(rule, edge="error"), info)


@pytest.mark.parametrize("bad", ["frames", "n_frames", None, 1, ["frame_count"]])
def test_check_temporal_refuses_bad_duration_from(bad):
    with pytest.raises(ValueError, match="duration_from"):
        check_temporal({"rule": "fps", "fps": 4, "n": 16, "start": "centre", "duration_from": bad}, "video")


def test_frame_count_needs_guaranteed_n_frames():
    """duration_from frame_count needs n_frames + fps, so a benchmark that does not guarantee n_frames
    (k400, vggsounder, vggss_box, ave, avsync) refuses it with E_INPUT; ssv2 accepts it."""
    from types import SimpleNamespace

    from yavih.bridge import E_INPUT, _check_rule_facts

    temporal = {"rule": "fps", "fps": 4, "n": 16, "start": "centre_clamped", "phase": 0, "edge": "clamp", "duration_from": "frame_count"}
    inp = {"video": {"temporal": temporal}}

    def bench(info):
        return SimpleNamespace(spec=SimpleNamespace(input_contract=SimpleNamespace(info={"video": info})))

    _check_rule_facts(inp, {"video": "video"}, bench(["duration_s", "fps", "n_frames", "width", "height", "container", "bytes"]))
    with pytest.raises(DescribeError) as ei:
        _check_rule_facts(inp, {"video": "video"}, bench(["duration_s", "fps", "width", "height", "container", "bytes"]))
    assert ei.value.code == E_INPUT and "n_frames" in str(ei.value)
    # the duration_s form stays legal there
    _check_rule_facts({"video": {"temporal": dict(temporal, duration_from="duration_s")}}, {"video": "video"},
                      bench(["duration_s", "fps", "width", "height", "container", "bytes"]))
