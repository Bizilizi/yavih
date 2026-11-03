"""Streaming benchmarks: the update grid, the two execution modes a bridge runs, and the checks yavih applies.

A streaming benchmark (``output_contract.shape: sequence``) hands a bridge one recording per row; the bridge answers
one update every ``update_interval_s`` seconds of the recording and returns the whole sequence
(:data:`yavih.codecs.STREAM_SCHEMA`: ``emb``, ``support``, ``cutoff`` and, in real time, ``timing``). An update with
cutoff ``c`` may use observations up to ``c`` only. yavih does not decode the stream; it supplies the schedule and
checks what comes back:

* :func:`causal` — the unrestricted mode: every grid cutoff ``k * update_interval_s`` (``k >= 1``, up to the
  recording's duration) in order, the stream pausing while the model computes.
* :func:`replay` — the real-time mode: a 1x wall-clock replay with the protocol's ``latest_pending_serial`` scheduler
  (one running job, one replaceable pending request; a newer request replaces the pending one, which is logged as
  skipped). ``ready_at`` is taken after the step returns its numpy result (and after ``torch.cuda.synchronize()``
  when torch is loaded). Every model calls this one helper, so every model runs under the same scheduler.

``step(cutoff, since)`` is the bridge's own function: it returns ``(emb [D], support [4])`` for the observations up to
``cutoff``. ``since`` is the cutoff of the previous completed update (a native streaming model consumes
``(since, cutoff]``), raised to ``cutoff - buffer_s`` when a finite raw buffer evicted older observations (the gap is
logged). ``support`` is ``[video_start, video_end, audio_start, audio_end]`` in seconds of the recording, NaN for a
modality the update did not read.

``bridge check`` runs a streaming bridge under :func:`forced_causal`: :func:`replay` then behaves as :func:`causal`,
so the determinism gates (rerun, shuffled, alone) compare the same code path without a clock; a separate smoke run
checks the real-time log (:func:`check_row`). numpy + stdlib.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np

__all__ = [
    "HELPER_VERSION",
    "SCHEDULERS",
    "MODES",
    "StreamError",
    "grid",
    "causal",
    "replay",
    "forced_causal",
    "is_forced_causal",
    "recording_duration",
    "check_row",
    "check_describe_stream",
]

#: The replay helper's version, recorded in every timing log.
HELPER_VERSION = 1
SCHEDULERS: tuple[str, ...] = ("latest_pending_serial",)
MODES: tuple[str, ...] = ("causal", "realtime")
#: Tolerance on times (seconds): grid arithmetic and float round trips.
EPS = 1e-6

Step = Callable[[float, float], "tuple[Any, Any]"]

_FORCED = contextvars.ContextVar("yavih_stream_forced_causal", default=False)


class StreamError(ValueError):
    """A step returned something that is not ``(emb [D], support [4])``."""


def grid(duration_s: float, update_interval_s: float) -> np.ndarray:
    """The update cutoffs of a recording: ``k * update_interval_s`` for ``k = 1, 2, ...`` up to ``duration_s``."""
    if not update_interval_s > 0:
        raise ValueError("update_interval_s must be > 0")
    n = int(math.floor(float(duration_s) / float(update_interval_s) + EPS))
    return np.arange(1, n + 1, dtype=np.float64) * float(update_interval_s)


@contextlib.contextmanager
def forced_causal() -> Iterator[None]:
    """Within the block, :func:`replay` runs the unrestricted causal mode (``bridge check``'s determinism gates)."""
    token = _FORCED.set(True)
    try:
        yield
    finally:
        _FORCED.reset(token)


def is_forced_causal() -> bool:
    return bool(_FORCED.get())


def _one(out: Any, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(out, tuple | list) or len(out) != 2:
        raise StreamError(f"step({cutoff:g}) must return (emb [D], support [4]), got {type(out).__name__}")
    emb = np.asarray(_numpy(out[0]))
    sup = np.asarray(_numpy(out[1]), dtype=np.float64)
    if emb.ndim != 1:
        raise StreamError(f"step({cutoff:g}): emb must be 1-D [D], got shape {list(emb.shape)}")
    if sup.shape != (4,):
        raise StreamError(f"step({cutoff:g}): support must be [video_start, video_end, audio_start, audio_end], got shape {list(sup.shape)}")
    return emb, sup


def _numpy(x: Any) -> Any:
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy() if str(getattr(x, "dtype", "")) == "torch.bfloat16" else x.detach().cpu().numpy()
    return x


def _stack(embs: list[np.ndarray], sups: list[np.ndarray], cutoffs: list[float], dim: int | None) -> dict[str, np.ndarray]:
    if embs:
        emb = np.stack(embs)
    else:
        emb = np.zeros((0, int(dim or 0)), dtype=np.float32)
    support = np.stack(sups) if sups else np.zeros((0, 4), dtype=np.float64)
    return {"emb": emb, "support": support, "cutoff": np.asarray(cutoffs, dtype=np.float64)}


def causal(step: Step, *, duration_s: float, update_interval_s: float, reset: Callable[[], None] | None = None,
           dim: int | None = None) -> dict[str, np.ndarray]:
    """The unrestricted causal mode: every grid cutoff in order. Returns ``{emb [n, D], support [n, 4], cutoff [n]}``
    (``dim`` sizes ``emb`` for a recording shorter than one interval)."""
    if reset is not None:
        reset()
    embs, sups, cuts = [], [], []
    since = 0.0
    for c in grid(duration_s, update_interval_s):
        e, s = _one(step(float(c), since), float(c))
        embs.append(e)
        sups.append(s)
        cuts.append(float(c))
        since = float(c)
    return _stack(embs, sups, cuts, dim)


def _sync() -> str:
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        return "cuda"
    return "none"


def replay(step: Step, *, duration_s: float, update_interval_s: float, reset: Callable[[], None] | None = None,
           scheduler: str = "latest_pending_serial", buffer_s: float | None = None, dim: int | None = None,
           clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """The real-time mode: requests arrive at the grid cutoffs on a 1x clock; one job runs at a time; a request that
    arrives while one is pending replaces it (the replaced one is ``skipped``). A job answers its request's cutoff
    (observations that arrived after it never enter the job). Returns the :func:`causal` arrays for the completed
    updates plus ``timing``: ``{helper_version, scheduler, buffer_s, sync, updates: [[request_s, started_at,
    ready_at], ...], skipped: [request_s, ...], gaps: [[from, to], ...]}`` in seconds since the stream started.
    Under :func:`forced_causal` it is :func:`causal` (no clock, no ``timing``)."""
    if is_forced_causal():
        return causal(step, duration_s=duration_s, update_interval_s=update_interval_s, reset=reset, dim=dim)
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler {scheduler!r}: one of {list(SCHEDULERS)}")
    if buffer_s is not None and not buffer_s > 0:
        raise ValueError("buffer_s must be > 0 (or None: unbounded)")
    if reset is not None:
        reset()
    requests = grid(duration_s, update_interval_s)
    embs, sups, cuts, updates, skipped, gaps = [], [], [], [], [], []
    sync = "none"
    prev = 0.0
    nxt, pending = 0, None
    t0 = clock()
    while True:
        now = clock() - t0
        while nxt < len(requests) and requests[nxt] <= now + EPS:
            if pending is not None:
                skipped.append(float(requests[pending]))
            pending = nxt
            nxt += 1
        if pending is None:
            if nxt >= len(requests):
                break
            sleep(max(0.0, float(requests[nxt]) - now))
            continue
        k, pending = pending, None
        cutoff = float(requests[k])
        since = prev
        if buffer_s is not None and cutoff - buffer_s > since + EPS:
            gaps.append([since, cutoff - float(buffer_s)])
            since = cutoff - float(buffer_s)
        started = clock() - t0
        e, s = _one(step(cutoff, since), cutoff)
        sync = _sync()
        ready = clock() - t0
        embs.append(e)
        sups.append(s)
        cuts.append(cutoff)
        updates.append([cutoff, started, ready])
        prev = cutoff
    out: dict[str, Any] = _stack(embs, sups, cuts, dim)
    out["timing"] = {"helper_version": HELPER_VERSION, "scheduler": scheduler, "buffer_s": buffer_s, "sync": sync,
                     "updates": updates, "skipped": skipped, "gaps": gaps}
    return out


def recording_duration(media: Mapping[str, Any]) -> float | None:
    """A recording's duration from its streams' facts (``{stream: info}``): the video's, else the longest audio's."""
    vids = [i.get("duration_s") for i in media.values() if isinstance(i, Mapping) and i.get("fps") is not None and i.get("duration_s") is not None]
    if vids:
        return float(max(vids))
    durs = [i.get("duration_s") for i in media.values() if isinstance(i, Mapping) and i.get("duration_s") is not None]
    return float(max(durs)) if durs else None


def check_describe_stream(block: Any) -> list[str]:
    """Problems with ``describe()["stream"]`` (empty = valid)."""
    if not isinstance(block, Mapping):
        return ["describe()['stream'] must be a dict {mode, native, video_window_s, audio_window_s, num_frames, frame_sampling, "
                "startup_policy, gap_policy, device}"]
    need = {"mode", "native", "video_window_s", "audio_window_s", "num_frames", "frame_sampling", "startup_policy", "gap_policy", "device"}
    out = []
    if set(block) != need:
        out.append(f"describe()['stream'] keys must be exactly {sorted(need)} (missing {sorted(need - set(block))}, unknown {sorted(set(block) - need)})")
    if block.get("mode") not in MODES:
        out.append(f"describe()['stream']['mode'] must be one of {list(MODES)}")
    if not isinstance(block.get("native"), bool):
        out.append("describe()['stream']['native'] must be a bool (true: the model keeps state across updates)")
    for k in ("video_window_s", "audio_window_s"):
        v = block.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int | float) or v <= 0):
            out.append(f"describe()['stream'][{k!r}] must be seconds > 0 or null (no window: native state, or modality unread)")
    for k in ("frame_sampling", "startup_policy", "gap_policy", "device"):
        if not isinstance(block.get(k), str) or not block.get(k):
            out.append(f"describe()['stream'][{k!r}] must be a non-empty string")
    return out


def check_row(values: Mapping[str, Any], *, duration_s: float | None, update_interval_s: float, mode: str,
              windows: Mapping[str, float | None] | None = None) -> list[tuple[str, str]]:
    """``[(code, message)]`` for one recording's outputs: shapes agree; cutoffs strictly increasing, on the grid and
    within the recording (the whole grid in causal mode); no support ends after its cutoff (``E_FUTURE_SUPPORT``) or
    spans more than the declared window; a real-time ``timing`` log consistent with the scheduler (``E_REPLAY_LOG``)."""
    out: list[tuple[str, str]] = []
    emb = np.asarray(values.get("emb"))
    sup = np.asarray(values.get("support"), dtype=np.float64)
    cut = np.asarray(values.get("cutoff"), dtype=np.float64)
    n = int(cut.shape[0]) if cut.ndim == 1 else -1
    if emb.ndim != 2 or sup.ndim != 2 or cut.ndim != 1 or not (emb.shape[0] == sup.shape[0] == cut.shape[0]):
        return [("E_MEMBERS", f"emb {list(emb.shape)}, support {list(sup.shape)} and cutoff {list(cut.shape)} must have one row per update")]
    timing = values.get("timing")
    if (mode == "realtime") != (timing is not None):
        out.append(("E_REPLAY_LOG", f"mode {mode!r} {'needs' if mode == 'realtime' else 'has no'} a timing log (yavih.stream.replay)"))
    if n and (np.any(np.diff(cut) <= 0) or cut[0] <= 0):
        out.append(("E_MEMBERS", "cutoffs must be strictly increasing and > 0"))
    k = cut / float(update_interval_s)
    if n and np.any(np.abs(k - np.round(k)) > EPS * max(1.0, float(np.max(np.abs(k))))):
        out.append(("E_MEMBERS", f"cutoffs off the {update_interval_s:g} s update grid"))
    if duration_s is not None:
        full = grid(duration_s, update_interval_s)
        if n and cut[-1] > float(duration_s) + EPS:
            out.append(("E_FUTURE_SUPPORT", f"cutoff {cut[-1]:g} s is after the recording's end ({duration_s:g} s)"))
        if mode == "causal" and (n != len(full) or not np.allclose(cut, full, atol=EPS)):
            out.append(("E_MEMBERS", f"the causal mode answers every grid cutoff: {len(full)} expected, {n} returned"))
    for j, (a, b) in enumerate(((0, 1), (2, 3))):
        lo, hi = sup[:, a], sup[:, b]
        nan = np.isnan(lo) | np.isnan(hi)
        if np.any(np.isnan(lo) != np.isnan(hi)):
            out.append(("E_MEMBERS", f"support {'video' if j == 0 else 'audio'} start/end must be both NaN or both set"))
        ok = ~nan
        if np.any(hi[ok] > cut[ok] + EPS):
            i = int(np.flatnonzero(ok & (hi > cut + EPS))[0])
            out.append(("E_FUTURE_SUPPORT", f"update {i}: {'video' if j == 0 else 'audio'} support ends at {hi[i]:g} s, after its cutoff {cut[i]:g} s"))
        if np.any(lo[ok] < -EPS) or np.any(lo[ok] > hi[ok] + EPS):
            out.append(("E_MEMBERS", f"{'video' if j == 0 else 'audio'} support must satisfy 0 <= start <= end"))
        w = (windows or {}).get("video_window_s" if j == 0 else "audio_window_s")
        if w is not None and np.any((hi[ok] - lo[ok]) > float(w) + EPS):
            out.append(("E_FUTURE_SUPPORT", f"{'video' if j == 0 else 'audio'} support spans more than the declared window {float(w):g} s"))
    if timing is not None:
        out.extend(_check_timing(timing, cut, duration_s, update_interval_s))
    return out


def _check_timing(timing: Any, cut: np.ndarray, duration_s: float | None, update_interval_s: float) -> list[tuple[str, str]]:
    bad = []
    if not isinstance(timing, Mapping) or timing.get("scheduler") not in SCHEDULERS or not isinstance(timing.get("updates"), list):
        return [("E_REPLAY_LOG", "timing must be the log yavih.stream.replay returns ({scheduler, updates, skipped, gaps, ...})")]
    try:
        upd = np.asarray(timing["updates"], dtype=np.float64).reshape(-1, 3)
        skipped = np.asarray(timing.get("skipped") or [], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return [("E_REPLAY_LOG", "timing.updates must be [[request_s, started_at, ready_at], ...] and skipped [request_s, ...]")]
    if upd.shape[0] != cut.shape[0] or not np.allclose(upd[:, 0], cut, atol=EPS):
        bad.append(("E_REPLAY_LOG", "timing.updates must hold one [request_s, started_at, ready_at] per update, request_s = cutoff"))
        return bad
    req, start, ready = upd[:, 0], upd[:, 1], upd[:, 2]
    if np.any(start < req - EPS):
        bad.append(("E_REPLAY_LOG", "a job started before its request arrived (started_at < request_s)"))
    if np.any(ready < start - EPS):
        bad.append(("E_REPLAY_LOG", "ready_at precedes started_at"))
    if len(start) > 1 and np.any(start[1:] < ready[:-1] - EPS):
        bad.append(("E_REPLAY_LOG", "jobs overlap: the serial scheduler starts a job only after the previous one is ready"))
    if duration_s is not None:
        full = grid(duration_s, update_interval_s)
        seen = np.sort(np.concatenate([req, skipped]))
        if seen.shape != full.shape or not np.allclose(seen, full, atol=EPS):
            bad.append(("E_REPLAY_LOG", f"every grid request is answered or skipped exactly once ({len(full)} requests, "
                                        f"{len(req)} answered + {len(skipped)} skipped)"))
    return bad
