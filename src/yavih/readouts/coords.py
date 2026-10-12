"""Media coordinates → the readout's task frame. numpy only, torch-free.

The bridge writes ``pos = (stream, t, y, x)`` in MEDIA coordinates: ``t`` is a fraction of the token's
own stream's ``media.duration_s``, ``(y, x)`` fractions of the ORIGINAL frame, ``-1`` = no coordinate.
The heads read coordinates in the readout's task frame:

- ``seq``: ``u = (τ − segment_offset · segment_seconds) / (T · segment_seconds)`` with ``τ = t ·
  duration_s`` of the token's own stream (absolute media seconds), so a token is read at the
  second it lies in whatever frame count, fps or window the bridge chose;
- ``grid`` / ``box``: ``((y − y0) / (y1 − y0), (x − x0) / (x1 − x0))`` over ``protocol.target_region``
  ``[y0, x0, y1, x1]`` (the full frame when absent), so the cells tile the region the targets
  rasterize and a bridge that cropped the frame simply leaves the cells it did not see empty.

Tokens outside the scored span or region keep their out-of-range coordinates; a missing coordinate
becomes ``NaN`` (the heads' null marker, ``yavih.probes.dense.coord_present``), so an out-of-range
task coordinate can never be confused with the media ``-1`` marker.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "FULL_FRAME",
    "target_region",
    "media_missing",
    "region_task_pos",
    "seq_task_pos",
    "seq_params",
    "TaskPos",
    "task_pos",
    "stream_names",
    "row_durations",
    "durations_from_sidecars",
]

#: ``[y0, x0, y1, x1]`` of the whole original frame.
FULL_FRAME: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


def target_region(protocol: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """``protocol.target_region`` as floats, or the full frame when the key is absent."""
    r = protocol.get("target_region")
    if r is None:
        return FULL_FRAME
    y0, x0, y1, x1 = (float(v) for v in r)
    if not (0.0 <= y0 < y1 <= 1.0 and 0.0 <= x0 < x1 <= 1.0):
        raise ValueError(f"protocol.target_region {list(r)} is not [y0, x0, y1, x1] with 0 <= y0 < y1 <= 1, 0 <= x0 < x1 <= 1")
    return (y0, x0, y1, x1)


def media_missing(a: np.ndarray) -> np.ndarray:
    """``True`` where a media coordinate is absent: ``-1`` or non-finite."""
    a = np.asarray(a)
    return ~np.isfinite(a) | (a == -1.0)


def _nan_missing(col: np.ndarray) -> np.ndarray:
    out = np.asarray(col, dtype=np.float32).copy()
    out[media_missing(col)] = np.nan
    return out


def region_task_pos(pos: np.ndarray, region: Sequence[float] = FULL_FRAME) -> np.ndarray:
    """``[..., 4]`` media pos → task pos for grid/box: ``(y, x)`` re-expressed in ``region``; ``t`` kept
    (NaN where absent); absent coordinates NaN. Values outside ``[0, 1]`` stay as they are."""
    p = np.asarray(pos, dtype=np.float32)
    y0, x0, y1, x1 = (float(v) for v in region)
    out = np.empty_like(p, dtype=np.float32)
    out[..., 0] = p[..., 0]
    out[..., 1] = _nan_missing(p[..., 1])
    y, x = p[..., 2], p[..., 3]
    out[..., 2] = np.where(media_missing(y), np.nan, (y - y0) / (y1 - y0))
    out[..., 3] = np.where(media_missing(x), np.nan, (x - x0) / (x1 - x0))
    return out


def seq_params(protocol: Mapping[str, Any]) -> dict[str, float]:
    """``{T, segment_offset, segment_seconds}`` of the seq protocol (``segment_seconds`` defaults to 1.0, the
    AVE value, when a synthetic protocol omits it)."""
    return {
        "T": int(protocol["segments_scored"]),
        "segment_offset": int(protocol.get("segment_offset", 0)),
        "segment_seconds": float(protocol.get("segment_seconds", 1.0)),
    }


def seq_task_pos(pos: np.ndarray, durations: np.ndarray, *, T: int, segment_offset: int, segment_seconds: float) -> np.ndarray:
    """``pos [k, N, 4]`` + ``durations [k, S]`` (``media.duration_s`` of each row's declared stream ``s``,
    NaN when unknown, e.g. an image stream) → task pos with ``u`` in column 1. A token whose
    ``t`` is absent, whose stream tag is ``-1``, or whose stream has no duration gets ``NaN``."""
    p = np.asarray(pos, dtype=np.float32)
    if p.ndim != 3:
        raise ValueError(f"seq_task_pos: pos must be [k, N, 4], got {p.shape}")
    dur = np.asarray(durations, dtype=np.float64)
    if dur.ndim != 2 or dur.shape[0] != p.shape[0]:
        raise ValueError(f"seq_task_pos: durations must be [k, S] with k = {p.shape[0]}, got {dur.shape}")
    k, n = p.shape[:2]
    tag = p[..., 0]
    ok_tag = np.isfinite(tag) & (tag >= 0) & (tag < dur.shape[1]) & (tag == np.round(tag))
    tag_i = np.where(ok_tag, tag, 0).astype(np.int64)
    tok_dur = np.where(ok_tag, dur[np.arange(k)[:, None], tag_i], np.nan)
    t = p[..., 1].astype(np.float64)
    tau = t * tok_dur
    u = (tau - float(segment_offset) * float(segment_seconds)) / (float(T) * float(segment_seconds))
    u = np.where(media_missing(p[..., 1]) | ~np.isfinite(tok_dur), np.nan, u)
    out = np.empty_like(p, dtype=np.float32)
    out[..., 0] = p[..., 0]
    out[..., 1] = u.astype(np.float32)
    out[..., 2] = _nan_missing(p[..., 2])
    out[..., 3] = _nan_missing(p[..., 3])
    return out


def _index(idx: Any, n: int) -> tuple[np.ndarray, bool]:
    if isinstance(idx, slice):
        return np.arange(*idx.indices(n), dtype=np.int64), False
    scalar = np.isscalar(idx) or isinstance(idx, (int, np.integer))
    ids = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    return np.where(ids < 0, ids + n, ids), scalar


class TaskPos:
    """A per-sample ``(n, N, 4)`` pos gather in the task frame (what ``train_head(pos=...)`` and
    ``_fit.score`` accept). ``base`` is the media pos: a constant ``[N, 4]`` array or a per-sample gather
    with ``__getitem__(ndarray) -> [k, N, 4]``; ``per_row`` (``[n, S]``, e.g. durations) is passed to
    ``transform(pos [k, N, 4], per_row [k, S] | None)`` with the same row indices."""

    def __init__(self, base: Any, transform: Callable[[np.ndarray, np.ndarray | None], np.ndarray], *, n: int, N: int,
                 per_row: np.ndarray | None = None) -> None:
        self.base = base
        self.transform = transform
        self.per_row = None if per_row is None else np.asarray(per_row)
        self.shape = (int(n), int(N), 4)
        self.ndim = 3
        self._const = isinstance(base, np.ndarray) and base.ndim == 2

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: Any) -> np.ndarray:
        ids, scalar = _index(idx, self.shape[0])
        if self._const:
            p = np.broadcast_to(self.base, (ids.size, *self.base.shape))
        else:
            p = np.asarray(self.base[ids], dtype=np.float32)
        out = self.transform(np.asarray(p, dtype=np.float32), None if self.per_row is None else self.per_row[ids])
        return out[0] if scalar else out


def task_pos(base: Any, transform: Callable[[np.ndarray, np.ndarray | None], np.ndarray], *, n: int, N: int,
             per_row: np.ndarray | None = None) -> Any:
    """``None`` when there is no pos; a constant task ``[N, 4]`` when the media pos is constant and the
    transform needs no per-row input; else a :class:`TaskPos` gather."""
    if base is None:
        return None
    if isinstance(base, np.ndarray) and base.ndim == 2 and per_row is None:
        return transform(base[None].astype(np.float32), None)[0]
    return TaskPos(base, transform, n=n, N=N, per_row=per_row)


def stream_names(describe: Mapping[str, Any]) -> list[str]:
    """Names of ``describe()["streams"]`` in tag order (entries may be names or ``{"name": …}``)."""
    return [str(s.get("name")) if isinstance(s, Mapping) else str(s) for s in (describe.get("streams") or [])]


def durations_from_sidecars(split_dir: str | os.PathLike, rows: np.ndarray, stream: str) -> np.ndarray:
    """``float64 [len(rows)]`` ``media.duration_s`` of ``stream`` from the sidecars' ``members[].duration_s``;
    NaN when unknown. The fallback for a ``FeatureReader`` without ``durations()``."""
    from yavih.io.reader import list_sidecars
    from yavih.io.sidecar import read_sidecar

    by_row: dict[int, float] = {}
    for p in list_sidecars(Path(split_dir)):
        sc = read_sidecar(p)
        for m in sc.get("members") or []:
            d = (m.get("duration_s") or {}).get(stream)
            by_row[int(m["row"])] = float(d) if d is not None else float("nan")
    return np.asarray([by_row.get(int(r), float("nan")) for r in np.asarray(rows, dtype=np.int64)], dtype=np.float64)


def row_durations(reader: Any, rows: np.ndarray, streams: Sequence[str], *, split_dir: str | os.PathLike | None = None) -> np.ndarray:
    """``float64 [n, S]``: per row, ``media.duration_s`` of every declared stream
    (``FeatureReader.durations(idx, stream)``, else the sidecars); NaN when null (images) or unknown."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.full((rows.shape[0], len(streams)), np.nan, dtype=np.float64)
    fn = getattr(reader, "durations", None)
    for j, s in enumerate(streams):
        if callable(fn):
            col = np.asarray(fn(rows, s), dtype=np.float64)
        elif split_dir is not None:
            col = durations_from_sidecars(split_dir, rows, s)
        else:
            continue
        out[:, j] = col.reshape(-1)
    return out
