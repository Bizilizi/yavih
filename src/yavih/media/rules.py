"""Declared bridge rules: evaluate a ``describe()["input"]`` rule on one row's media facts.

Core module (numpy + stdlib): ``validate`` recomputes from a bridge's declared rule and a row's
``RawStream.info`` what the bridge should have consumed, and compares it with the reported ``used``.
These are helpers WITHOUT protocol semantics: nothing in a benchmark depends on them; a bridge
that declares a rule may use them to compute the times it then feeds (and reports).

- ``rule_times(temporal, info) -> float64 [n] | None``: the media seconds of the frames a
  ``video`` rule feeds (``None`` for ``custom``). ``fps``: ``t_k = s + (k + phase)/fps`` with the
  optional ``phase`` (default 0.5 = bin centres; 0 = window start, ``start + k/fps``), ``start`` a number,
  ``"centre"`` or ``"centre_clamped"`` (``max(0, (D − n/fps)/2)``) and the optional ``duration_from``: ``D`` =
  ``duration_s`` (default, also the ``edge`` bound) or ``"frame_count"`` = ``n_frames / fps`` of the video track
  (``edge`` bound: the last frame, ``(n_frames − 1) / fps``).
- ``audio_ranges(temporal, info) -> (int64 [k, 2], (pad_left, pad_right)) | None``: the half-open
  native-rate sample ranges an ``audio`` rule feeds and the padding it adds (``None`` for ``custom``).
- ``crop_region(spatial, width, height) -> (y0, x0, y1, x1) | None``: the part of the ORIGINAL frame a
  spatial rule keeps, in fractions (``None`` for ``custom``).
- ``select_nearest(pts, times) -> int64 [n]``: ``nearest_ties_earlier`` (the nearest frame, ties to the earlier).
- ``check_temporal`` / ``check_spatial``: the closed rule vocabulary, for ``validate_describe``.

A time outside ``[0, duration_s]`` under ``edge: "error"`` (or an audio span past the end under
``pad: "none"``) raises ``OutOfMedia`` — validate's ``E_TIMES_OUT_OF_MEDIA``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "TIE_DECIMALS",
    "TEMPORAL_RULES",
    "TEMPORAL_PARAMS",
    "EDGES",
    "FPS_STARTS",
    "FPS_DEFAULT_PHASE",
    "FPS_DURATION_FROM",
    "OutOfMedia",
    "select_nearest",
    "rule_times",
    "audio_ranges",
    "crop_region",
    "check_temporal",
    "check_spatial",
    "rule_facts",
]

#: Distances are rounded to this many decimals before the nearest comparison: float noise far below the quantum
#: cannot flip an exact tie.
TIE_DECIMALS = 9

#: rule -> stream kinds it is legal for.
TEMPORAL_RULES: Mapping[str, frozenset[str]] = {
    "uniform": frozenset({"video"}),
    "fps": frozenset({"video"}),
    "index_linspace": frozenset({"video"}),
    "explicit": frozenset({"video"}),
    "all": frozenset({"video"}),
    "span": frozenset({"audio"}),
    "full": frozenset({"audio"}),
    "segments": frozenset({"audio"}),
    "custom": frozenset({"video", "audio", "image"}),
}
#: rule -> (required params, optional params), ``rule`` itself excluded.
TEMPORAL_PARAMS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "uniform": (frozenset({"n", "span"}), frozenset({"edge"})),
    "fps": (frozenset({"fps", "n", "start"}), frozenset({"edge", "phase", "duration_from"})),
    "index_linspace": (frozenset({"n"}), frozenset()),
    "explicit": (frozenset({"times"}), frozenset({"edge"})),
    "all": (frozenset(), frozenset()),
    "span": (frozenset({"start_s", "length_s", "pad"}), frozenset()),
    "full": (frozenset({"max_s", "pad_to_s"}), frozenset()),
    "segments": (frozenset({"n", "length_s", "span", "pad"}), frozenset()),
    "custom": (frozenset({"description"}), frozenset()),
}
EDGES = frozenset({"error", "clamp"})
#: ``fps`` rule ``start`` keywords: ``centre`` = ``(duration_s − n/fps)/2`` (may be negative on a clip shorter than
#: the window), ``centre_clamped`` = ``max(0, (duration_s − n/fps)/2)``.
FPS_STARTS = frozenset({"centre", "centre_clamped"})
#: ``fps`` rule default ``phase``: ``t_k = s + (k + phase)/fps``; 0.5 = bin centres (default), 0 = window start.
FPS_DEFAULT_PHASE = 0.5
#: ``fps`` rule ``duration_from``: the clip length the window is centred on and clamped to. ``duration_s`` (default)
#: is the container's duration; ``frame_count`` is ``n_frames / fps`` of the video track (for a model
#: whose decoder centres on ``num_frames / average_fps``; on vggsounder / vggss_box the two differ by a median
#: 0.13-0.33 s).
FPS_DURATION_FROM = frozenset({"duration_s", "frame_count"})
_PADS = frozenset({"zeros", "none"})
_SPATIAL_KEYS = frozenset({"resize", "crop", "interp", "antialias"})


class OutOfMedia(ValueError):
    """A declared rule asks for media outside ``[0, duration_s]`` (``edge: error``) or past the end of the
    audio (``pad: none``): validate's ``E_TIMES_OUT_OF_MEDIA``."""

    def __init__(self, message: str, values: Sequence[float] | None = None) -> None:
        super().__init__(message)
        self.values = list(values) if values is not None else []


# ----------------------------------------------------------------------------- nearest_ties_earlier


def select_nearest(pts: Sequence[float] | np.ndarray, times: Sequence[float] | np.ndarray) -> np.ndarray:
    """``nearest_ties_earlier``: for each time choose
    the index minimising ``round(|t − pts[i]|, 9)`` with a strict ``<`` scan in ascending ``i``, so an exact
    tie goes to the EARLIER frame. int64 ``[len(times)]``; indices may repeat."""
    pts_f = [float(p) for p in np.asarray(pts, dtype=np.float64).reshape(-1)]
    if not pts_f:
        raise ValueError("select_nearest: empty pts")
    times_f = np.asarray(times, dtype=np.float64).reshape(-1)
    out = np.empty(times_f.shape[0], dtype=np.int64)
    for k, t in enumerate(times_f.tolist()):
        best_i, best_d = 0, None
        for i, p in enumerate(pts_f):
            d = round(abs(t - p), TIE_DECIMALS)
            if best_d is None or d < best_d:  # strict: ties keep the earlier i
                best_i, best_d = i, d
        out[k] = best_i
    return out


# ----------------------------------------------------------------------------- vocabulary checks


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _check_span(v: Any, where: str) -> None:
    if v == "full":
        return
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_num(x) for x in v) and float(v[0]) < float(v[1])):
        raise ValueError(f"{where}: span must be \"full\" or [t0, t1] with t0 < t1, got {v!r}")


def check_temporal(temporal: Any, kind: str, where: str = "temporal") -> None:
    """Raise ``ValueError`` unless ``temporal`` is a temporal rule of the vocabulary legal for stream ``kind``."""
    if not isinstance(temporal, Mapping):
        raise ValueError(f"{where}: expected a mapping with a 'rule' key, got {type(temporal).__name__}")
    rule = temporal.get("rule")
    if rule not in TEMPORAL_RULES:
        raise ValueError(f"{where}.rule: {rule!r} is not one of {sorted(TEMPORAL_RULES)}")
    if kind not in TEMPORAL_RULES[rule]:
        raise ValueError(f"{where}.rule: {rule!r} is not legal for a {kind!r} stream (legal for {sorted(TEMPORAL_RULES[rule])})")
    req, opt = TEMPORAL_PARAMS[rule]
    keys = set(temporal) - {"rule"}
    missing = sorted(req - keys)
    unknown = sorted(keys - req - opt)
    if missing:
        raise ValueError(f"{where}: rule {rule!r} needs {missing}")
    if unknown:
        raise ValueError(f"{where}: rule {rule!r} does not take {unknown}")
    if "edge" in temporal and temporal["edge"] not in EDGES:
        raise ValueError(f"{where}.edge: {temporal['edge']!r} is not one of {sorted(EDGES)}")
    if "n" in temporal and not _is_pos_int(temporal["n"]):
        raise ValueError(f"{where}.n: expected an int >= 1, got {temporal['n']!r}")
    if "span" in temporal:
        _check_span(temporal["span"], f"{where}.span")
    if "pad" in temporal and temporal["pad"] not in _PADS:
        raise ValueError(f"{where}.pad: {temporal['pad']!r} is not one of {sorted(_PADS)}")
    if rule == "fps":
        if not (_is_num(temporal["fps"]) and float(temporal["fps"]) > 0):
            raise ValueError(f"{where}.fps: expected a number > 0")
        if not ((isinstance(temporal["start"], str) and temporal["start"] in FPS_STARTS) or _is_num(temporal["start"])):
            raise ValueError(f"{where}.start: expected a number, \"centre\" or \"centre_clamped\", got {temporal['start']!r}")
        if "phase" in temporal and not (_is_num(temporal["phase"]) and 0.0 <= float(temporal["phase"]) < 1.0):
            raise ValueError(f"{where}.phase: expected a number in [0, 1) (0.5 = bin centres, the default; 0 = window start), "
                             f"got {temporal['phase']!r}")
        if "duration_from" in temporal and not (isinstance(temporal["duration_from"], str) and temporal["duration_from"] in FPS_DURATION_FROM):
            raise ValueError(f"{where}.duration_from: expected one of {sorted(FPS_DURATION_FROM)}, got {temporal['duration_from']!r}")
    elif rule == "explicit":
        t = temporal["times"]
        if not (isinstance(t, (list, tuple)) and t and all(_is_num(x) for x in t)):
            raise ValueError(f"{where}.times: expected a non-empty list of numbers")
    elif rule == "span":
        if not (temporal["start_s"] == "centre" or _is_num(temporal["start_s"])):
            raise ValueError(f"{where}.start_s: expected a number or \"centre\"")
        if not (_is_num(temporal["length_s"]) and float(temporal["length_s"]) > 0):
            raise ValueError(f"{where}.length_s: expected a number > 0")
    elif rule == "full":
        for k in ("max_s", "pad_to_s"):
            v = temporal[k]
            if v is not None and not (_is_num(v) and float(v) > 0):
                raise ValueError(f"{where}.{k}: expected null or a number > 0")
    elif rule == "segments":
        if not (_is_num(temporal["length_s"]) and float(temporal["length_s"]) > 0):
            raise ValueError(f"{where}.length_s: expected a number > 0")
    elif rule == "custom":
        if not (isinstance(temporal["description"], str) and temporal["description"].strip()):
            raise ValueError(f"{where}.description: a custom rule needs a non-empty description")


def _check_size(v: Any, where: str) -> None:
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_pos_int(x) for x in v)):
        raise ValueError(f"{where}: expected [h, w] of positive ints, got {v!r}")


def _check_region(v: Any, where: str) -> None:
    if not (isinstance(v, (list, tuple)) and len(v) == 4 and all(_is_num(x) for x in v)):
        raise ValueError(f"{where}: expected [y0, x0, y1, x1], got {v!r}")
    y0, x0, y1, x1 = (float(x) for x in v)
    if not (0.0 <= y0 < y1 <= 1.0 and 0.0 <= x0 < x1 <= 1.0):
        raise ValueError(f"{where}: need 0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1, got {list(v)}")


def check_spatial(spatial: Any, where: str = "spatial") -> None:
    """Raise ``ValueError`` unless ``spatial`` is a spatial rule of the vocabulary (or ``{"rule": "custom", "description"}``)."""
    if not isinstance(spatial, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(spatial).__name__}")
    if "rule" in spatial:
        if spatial.get("rule") != "custom" or set(spatial) != {"rule", "description"}:
            raise ValueError(f"{where}: the only rule form is {{rule: custom, description}}")
        if not (isinstance(spatial["description"], str) and spatial["description"].strip()):
            raise ValueError(f"{where}.description: a custom rule needs a non-empty description")
        return
    keys = set(spatial)
    if keys != _SPATIAL_KEYS:
        raise ValueError(f"{where}: keys must be exactly {sorted(_SPATIAL_KEYS)}, got {sorted(keys)}")
    resize = spatial["resize"]
    if resize is not None:
        if not (isinstance(resize, Mapping) and len(resize) == 1 and next(iter(resize)) in ("short", "long", "size")):
            raise ValueError(f"{where}.resize: expected null, {{short: S}}, {{long: S}} or {{size: [h, w]}}")
        (k, v), = resize.items()
        if k == "size":
            _check_size(v, f"{where}.resize.size")
        elif not _is_pos_int(v):
            raise ValueError(f"{where}.resize.{k}: expected an int >= 1")
    crop = spatial["crop"]
    if crop is not None:
        if not isinstance(crop, Mapping) or crop.get("kind") not in ("center", "region"):
            raise ValueError(f"{where}.crop: expected null, {{kind: center, size}} or {{kind: region, region}}")
        if crop["kind"] == "center":
            if set(crop) != {"kind", "size"}:
                raise ValueError(f"{where}.crop: kind center takes exactly `size`")
            _check_size(crop["size"], f"{where}.crop.size")
        else:
            if set(crop) != {"kind", "region"}:
                raise ValueError(f"{where}.crop: kind region takes exactly `region`")
            _check_region(crop["region"], f"{where}.crop.region")
    if not isinstance(spatial["interp"], str):
        raise ValueError(f"{where}.interp: expected str")
    if not isinstance(spatial["antialias"], bool):
        raise ValueError(f"{where}.antialias: expected bool")


def rule_facts(temporal: Mapping[str, Any], kind: str) -> frozenset[str]:
    """The media facts (``MediaInfo`` keys) a temporal rule needs to be evaluated on a row, so validate can
    check the reported ``used`` against it. ``bridge.validate_describe`` refuses a rule whose facts
    the benchmark's ``input_contract.info[stream]`` does not guarantee. Video rules need ``fps`` (or ``n_frames``) for
    the nearest-frame tolerance, which validate takes from whichever the row carries."""
    rule = temporal.get("rule") if isinstance(temporal, Mapping) else None
    if rule in (None, "custom"):
        return frozenset()
    if rule in ("span", "full", "segments"):
        return frozenset({"sample_rate", "n_samples"})
    if rule == "index_linspace":
        return frozenset({"n_frames", "fps"})
    if rule == "all":
        return frozenset({"n_frames", "fps"})
    if rule == "fps" and temporal.get("duration_from") == "frame_count":
        return frozenset({"n_frames", "fps"})
    if rule in ("uniform", "fps", "explicit"):
        return frozenset({"duration_s"})
    return frozenset()


# ----------------------------------------------------------------------------- evaluation


def _fact(info: Mapping[str, Any], key: str, rule: str) -> float:
    v = info.get(key)
    if v is None:
        raise ValueError(f"rule {rule!r} needs media fact {key!r}, which is null for this row")
    return float(v)


def _edge(times: np.ndarray, duration: float, edge: str, rule: str) -> np.ndarray:
    out_of = (times < 0.0) | (times > duration)
    if not np.any(out_of):
        return times
    if edge == "clamp":
        return np.clip(times, 0.0, duration)
    raise OutOfMedia(
        f"rule {rule!r} asks for time(s) {times[out_of].tolist()} outside the media [0, {duration}] (edge: error)",
        times[out_of].tolist(),
    )


def _uniform(n: int, span: Any, duration: float) -> np.ndarray:
    a, b = (0.0, duration) if span == "full" else (float(span[0]), float(span[1]))
    k = np.arange(int(n), dtype=np.float64)
    return a + (k + 0.5) * (b - a) / int(n)


def rule_times(temporal: Mapping[str, Any], info: Mapping[str, Any]) -> np.ndarray | None:
    """float64 ``[n]`` media seconds a ``video`` rule feeds, in feed order. ``None`` for ``custom``.
    ``OutOfMedia`` under ``edge: error``."""
    rule = temporal.get("rule")
    if rule == "custom":
        return None
    if rule in ("span", "full", "segments"):
        raise ValueError(f"rule {rule!r} is an audio rule: use audio_ranges")
    edge = temporal.get("edge", "error")
    if rule == "uniform":
        return _edge(_uniform(int(temporal["n"]), temporal["span"], _fact(info, "duration_s", rule)), _fact(info, "duration_s", rule), edge, rule)
    if rule == "fps":
        if temporal.get("duration_from", "duration_s") == "frame_count":
            # the video track's own clock: n_frames frames at k / fps; nothing exists past the last frame, so
            # that frame's time bounds `edge` (a decoder clamps the frame index to n_frames - 1)
            n_frames, src_fps = _fact(info, "n_frames", rule), _fact(info, "fps", rule)
            duration, bound = n_frames / src_fps, max(0.0, (n_frames - 1.0) / src_fps)
        else:
            duration = bound = _fact(info, "duration_s", rule)
        fps, n = float(temporal["fps"]), int(temporal["n"])
        if temporal["start"] == "centre":
            start = (duration - n / fps) / 2.0
        elif temporal["start"] == "centre_clamped":
            start = max(0.0, (duration - n / fps) / 2.0)
        else:
            start = float(temporal["start"])
        phase = float(temporal.get("phase", FPS_DEFAULT_PHASE))
        times = start + (np.arange(n, dtype=np.float64) + phase) / fps
        return _edge(times, bound, edge, rule)
    if rule == "index_linspace":
        n_frames, fps = _fact(info, "n_frames", rule), _fact(info, "fps", rule)
        idx = np.round(np.linspace(0.0, n_frames - 1.0, int(temporal["n"])))
        return idx / fps
    if rule == "explicit":
        times = np.asarray([float(t) for t in temporal["times"]], dtype=np.float64)
        return _edge(times, _fact(info, "duration_s", rule), edge, rule)
    if rule == "all":
        n_frames, fps = _fact(info, "n_frames", rule), _fact(info, "fps", rule)
        return np.arange(int(n_frames), dtype=np.float64) / fps
    raise ValueError(f"unknown temporal rule {rule!r}; expected one of {sorted(TEMPORAL_RULES)}")


def audio_ranges(temporal: Mapping[str, Any], info: Mapping[str, Any]) -> tuple[np.ndarray, tuple[int, int]] | None:
    """``(int64 [k, 2], (pad_left, pad_right))``: the half-open native-rate sample ranges an audio rule feeds
    and the padding (in samples) it adds before/after. ``None`` for ``custom``.

    ``span``: one range from ``round(start·sr)`` (``"centre"``: ``max(0, round((n_samples − L)/2))``) of
    ``L = round(length·sr)`` samples, cut at ``n_samples``; the shortfall is ``pad_right`` under
    ``pad: zeros`` and ``OutOfMedia`` under ``pad: none``. ``full``: ``[0, min(n_samples, round(max_s·sr)))``,
    right-padded to ``round(pad_to_s·sr)``. ``segments``: ``n`` ranges of ``L`` samples centred on the
    ``uniform`` times of ``span``, each cut to ``[0, n_samples]``; the summed cuts before 0 / after the end
    are ``(pad_left, pad_right)``.
    """
    rule = temporal.get("rule")
    if rule == "custom":
        return None
    if rule not in ("span", "full", "segments"):
        raise ValueError(f"rule {rule!r} is not an audio rule: use rule_times")
    sr = _fact(info, "sample_rate", rule)
    ns = int(_fact(info, "n_samples", rule))
    if rule == "span":
        L = int(round(float(temporal["length_s"]) * sr))
        lo = max(0, int(round((ns - L) / 2.0))) if temporal["start_s"] == "centre" else int(round(float(temporal["start_s"]) * sr))
        if lo < 0:
            raise OutOfMedia(f"span starts before the media ({lo} samples)", [lo / sr])
        lo_c = min(lo, ns)
        hi = min(lo + L, ns)
        short = L - (hi - lo_c)
        if short and temporal["pad"] == "none":
            raise OutOfMedia(f"span needs {L} samples from {lo} but the media has {ns} (pad: none)", [(lo + L) / sr])
        return np.asarray([[lo_c, hi]], dtype=np.int64), (0, int(short))
    if rule == "full":
        max_s, pad_to = temporal["max_s"], temporal["pad_to_s"]
        hi = ns if max_s is None else min(ns, int(round(float(max_s) * sr)))
        pad = 0 if pad_to is None else max(0, int(round(float(pad_to) * sr)) - hi)
        return np.asarray([[0, hi]], dtype=np.int64), (0, pad)
    # segments
    duration = ns / sr
    centres = _uniform(int(temporal["n"]), temporal["span"], duration)
    L = int(round(float(temporal["length_s"]) * sr))
    ranges, left, right = [], 0, 0
    for c in centres.tolist():
        lo = int(round((c - float(temporal["length_s"]) / 2.0) * sr))
        hi = lo + L
        lo_c, hi_c = min(max(lo, 0), ns), max(min(hi, ns), 0)
        # only the samples of this segment that fall outside [0, ns] are padding (a segment wholly
        # past the end contributes its own length, never more)
        left += max(0, min(hi, 0) - lo)
        right += max(0, hi - max(lo, ns))
        ranges.append([lo_c, max(lo_c, hi_c)])
    if (left or right) and temporal["pad"] == "none":
        raise OutOfMedia(f"segments reach outside the media ({left} samples before, {right} after; pad: none)")
    return np.asarray(ranges, dtype=np.int64).reshape(-1, 2), (int(left), int(right))


def _resized(spatial: Mapping[str, Any], width: int, height: int) -> tuple[float, float]:
    """``(rh, rw)``: the frame size after the declared resize, torchvision's size rule."""
    resize = spatial.get("resize")
    h, w = int(height), int(width)
    if resize is None:
        return float(h), float(w)
    (k, v), = resize.items()
    if k == "size":
        return float(v[0]), float(v[1])
    short, long = (h, w) if h <= w else (w, h)
    if k == "short":
        new_short, new_long = int(v), int(int(v) * long / short)
    else:  # long
        new_long, new_short = int(v), int(int(v) * short / long)
    return (float(new_short), float(new_long)) if h <= w else (float(new_long), float(new_short))


def crop_region(spatial: Mapping[str, Any], width: int, height: int) -> tuple[float, float, float, float] | None:
    """``(y0, x0, y1, x1)``: the part of the ORIGINAL ``width × height`` frame a declared spatial rule keeps,
    in fractions. Resize maps the whole frame onto the whole resized frame, so a centre crop of
    ``[ch, cw]`` at torchvision's ``top = round((rh − ch)/2)`` covers ``[top/rh, (top + ch)/rh]``; a crop
    larger than the frame is clipped to ``[0, 1]``. ``None`` for ``custom``."""
    if spatial.get("rule") == "custom":
        return None
    if not width or not height:
        raise ValueError("crop_region needs the frame's width and height")
    crop = spatial.get("crop")
    if crop is None:
        return (0.0, 0.0, 1.0, 1.0)
    if crop["kind"] == "region":
        y0, x0, y1, x1 = (float(x) for x in crop["region"])
        return (y0, x0, y1, x1)
    rh, rw = _resized(spatial, width, height)
    ch, cw = float(crop["size"][0]), float(crop["size"][1])
    top = float(int(round((rh - ch) / 2.0)))
    left = float(int(round((rw - cw) / 2.0)))

    def _clip(v: float) -> float:
        return min(1.0, max(0.0, v))

    return (_clip(top / rh), _clip(left / rw), _clip((top + ch) / rh), _clip((left + cw) / rw))
