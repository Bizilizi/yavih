"""Targets from ``Benchmark.manifest`` cells. numpy only; torch-free.

The manifest row ``r`` of a split carries ``label_ids`` (``int`` single, ``list[int]`` multi, ``None``),
``label_sets`` (multi_sets: ``{set: [ids]}``) and ``extra`` (declared non-label columns, e.g.
``event_start``/``event_end``, or the raw ``x1|y1|x2|y2`` box cell under the label column name).

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from yavih.readouts.base import ProbeRefused, ReadoutContext, _manifest_index

__all__ = [
    "single_targets",
    "multi_hot",
    "clf_targets",
    "segment_targets",
    "seq_targets",
    "parse_box",
    "box_targets",
]


def _ids(manifest: Any, rows: np.ndarray) -> np.ndarray:
    return _manifest_index(manifest, rows)


def _as_id_list(v: Any) -> list[int]:
    if v is None:
        return []
    if isinstance(v, (int, np.integer)):
        return [int(v)]
    return [int(x) for x in v]


def single_targets(manifest: Any, rows: np.ndarray) -> np.ndarray:
    """``int64 [n]`` from single-label cells (a one-element list is accepted)."""
    out = np.empty(len(rows), dtype=np.int64)
    for j, i in enumerate(_ids(manifest, rows)):
        ids = _as_id_list(manifest.label_ids[i])
        if len(ids) != 1:
            raise ProbeRefused("E_LABELS", f"row {int(rows[j])} ({manifest.clip_id[i]}) has {len(ids)} labels; single-label readout needs exactly one")
        out[j] = ids[0]
    return out


def multi_hot(id_lists: Sequence[Sequence[int]], num_classes: int) -> np.ndarray:
    out = np.zeros((len(id_lists), num_classes), dtype=np.float32)
    for j, ids in enumerate(id_lists):
        for c in ids:
            if not 0 <= int(c) < num_classes:
                raise ProbeRefused("E_LABELS", f"label index {c} outside classes.csv [0, {num_classes})")
            out[j, int(c)] = 1.0
    return out


def clf_targets(ctx: ReadoutContext, split: str, rows: np.ndarray, *, num_classes: int, role: str) -> Any:
    """clf targets. ``single`` → ``int64 [n]``; ``multi`` → ``float32 [n, C]``; ``multi_sets``: train/floor
    targets from ``label.column`` (or the union of ``label.train_set`` columns), eval → one ``float32
    [n, C]`` matrix per ``label.sets`` entry as ``{set: matrix}``."""
    manifest = ctx.manifest(split)
    label = ctx.spec.manifest.label
    kind = label.kind
    idx = _ids(manifest, rows)
    if kind == "single":
        return single_targets(manifest, rows)
    if kind == "multi":
        return multi_hot([_as_id_list(manifest.label_ids[i]) for i in idx], num_classes)
    if kind == "multi_sets":
        sets = manifest.label_sets
        if role == "eval":
            if sets is None:
                raise ProbeRefused("E_LABELS", "multi_sets benchmark without label_sets in the manifest")
            return {
                name: multi_hot([list(sets[i].get(name, [])) for i in idx], num_classes)
                for name in (label.sets or {})
            }
        if label.train_set:
            if sets is None:
                raise ProbeRefused("E_LABELS", "label.train_set set but the manifest has no label_sets")
            return multi_hot([list(sets[i].get(label.train_set, [])) for i in idx], num_classes)
        return multi_hot([_as_id_list(manifest.label_ids[i]) for i in idx], num_classes)
    raise ProbeRefused("E_LABELS", f"label.kind {kind!r} is not a clf label kind")


# ----------------------------------------------------------------------------- seq


def segment_targets(event_start: int, event_end: int, label: int, protocol: dict) -> list[int]:
    """Scored segment ``k ∈ [0, T)`` is media second ``s = segment_offset + k``; segment ``s``
    takes the row's class iff ``event_start ≤ s < event_end``, else ``background_index``."""
    try:
        from yavih.metrics import segment as _seg

        return [int(v) for v in _seg.segment_targets(event_start, event_end, label, protocol)]
    except NotImplementedError:
        pass
    T = int(protocol["segments_scored"])
    off = int(protocol["segment_offset"])
    bg = int(protocol["background_index"])
    return [int(label) if int(event_start) <= off + k < int(event_end) else bg for k in range(T)]


def _extra(manifest: Any, i: int, key: str) -> Any:
    extra = manifest.extra[i] if manifest.extra is not None else {}
    if key not in extra:
        raise ProbeRefused("E_LABELS", f"manifest row {manifest.clip_id[i]} lacks the {key!r} column")
    return extra[key]


def seq_targets(ctx: ReadoutContext, split: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(targets int64 [n, T], event_start int64 [n], event_end int64 [n])``."""
    manifest = ctx.manifest(split)
    idx = _ids(manifest, rows)
    T = int(ctx.protocol["segments_scored"])
    out = np.empty((len(rows), T), dtype=np.int64)
    starts = np.empty(len(rows), dtype=np.int64)
    ends = np.empty(len(rows), dtype=np.int64)
    segments = getattr(manifest, "segments", None)
    for j, i in enumerate(idx):
        ids = _as_id_list(manifest.label_ids[i])
        if len(ids) != 1:
            raise ProbeRefused("E_LABELS", f"row {manifest.clip_id[i]} has {len(ids)} labels; seq needs one event class")
        if segments is not None:  # Benchmark.load normalizes the cells into Manifest.segments (int64 [n, 2])
            s, e = int(segments[i][0]), int(segments[i][1])
        else:
            s, e = int(_extra(manifest, i, "event_start")), int(_extra(manifest, i, "event_end"))
        out[j] = segment_targets(s, e, ids[0], ctx.protocol)
        starts[j], ends[j] = s, e
    return out, starts, ends


# ----------------------------------------------------------------------------- box


def parse_box(cell: str) -> tuple[float, float, float, float] | None:
    """``x1|y1|x2|y2`` → ``(cx, cy, w, h)`` after ``max(x, 0)``; ``None`` when degenerate."""
    try:
        from yavih.metrics import box as _box

        return _box.parse_box(cell)
    except NotImplementedError:
        pass
    parts = str(cell).split("|")
    if len(parts) != 4:
        raise ProbeRefused("E_LABELS", f"box cell {cell!r} is not x1|y1|x2|y2")
    x1, y1, x2, y2 = (max(float(p), 0.0) for p in parts)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def box_targets(ctx: ReadoutContext, split: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """``(targets float32 [m, 4], kept rows int64 [m], n_excluded_degenerate)``."""
    manifest = ctx.manifest(split)
    idx = _ids(manifest, rows)
    column = ctx.spec.manifest.label.column or "box"
    boxes: list[tuple[float, float, float, float]] = []
    kept: list[int] = []
    excluded = 0
    mboxes = getattr(manifest, "boxes", None)
    for j, i in enumerate(idx):
        if mboxes is not None:  # Benchmark.load normalizes the cells into Manifest.boxes (float32 [n, 4], x1 y1 x2 y2)
            b = parse_box("|".join(repr(float(v)) for v in mboxes[i]))
        else:
            extra = manifest.extra[i] if manifest.extra is not None else {}
            cell = extra.get(column, extra.get("box"))
            if cell is None:
                raise ProbeRefused("E_LABELS", f"row {manifest.clip_id[i]} lacks the box cell {column!r}")
            b = parse_box(cell) if isinstance(cell, str) else parse_box("|".join(repr(float(v)) for v in cell))
        if b is None:
            excluded += 1
            continue
        boxes.append(b)
        kept.append(int(rows[j]))
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(kept, dtype=np.int64), excluded
