"""Samples-file lines (``eval run --log-samples``). numpy only, torch-free.

One JSON line per scored row, seed and fold: ``{"clip_id", "row", "seed", "role", "fold"?, "prediction",
"target"}`` with ``role`` ∈ ``eval``, ``floor``:

- clf single → class indices + ``"correct"``;
- clf multi / multi_sets → ``target`` = label indices (per set for multi_sets), ``prediction`` = the
  top-10 ``[index, score]`` pairs;
- seq → ``[T]`` class indices each;
- grid → ``prediction`` = that image's ``{d1, absrel, rmse}``, ``target`` = ``null`` (maps stay in the memmap);
- box → ``[cx, cy, w, h]`` each + ``"iou"``.

A readout hands :func:`sample_lines` a list of parts ``{"role", "outputs", "targets", "rows",
"clip_ids", "folds"?}`` (``folds`` = the fold value per row under ``splits.folds``).

A folder scorer (``eval.py`` ``SCORER``) hands parts ``{"role", "rows", "clip_ids", "values"}`` instead, ``values`` =
``{metric: [n] per-clip values}``: each line's ``prediction`` is that row's ``{metric: value}`` (``null`` where not
finite) and ``target`` is ``null``. Two systems' lines on the same rows give paired per-clip differences.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["sample_lines", "TOP_K"]

#: ``prediction`` of a multi-label row = the top ``TOP_K`` ``[index, score]`` pairs.
TOP_K: int = 10


def _num(x: Any) -> float | None:
    v = float(x)
    return v if math.isfinite(v) else None


def _indices(row: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.asarray(row) > 0.5)]


def _top_k(scores: np.ndarray) -> list[list[float]]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    k = min(TOP_K, s.shape[0])
    order = np.argsort(-s, kind="stable")[:k]
    return [[int(i), float(s[i])] for i in order]


def _value_lines(part: Mapping[str, Any], *, seed: int) -> list[dict]:
    """A folder scorer's part: per-clip metric values per row."""
    rows = np.asarray(part["rows"], dtype=np.int64)
    ids: Sequence[str] = list(part["clip_ids"])
    values = {str(k): np.asarray(v, dtype=np.float64).reshape(-1) for k, v in dict(part["values"]).items()}
    n = rows.shape[0]
    bad = {k: v.shape[0] for k, v in values.items() if v.shape[0] != n}
    if len(ids) != n or bad:
        raise ValueError(f"samples: rows {n}, clip_ids {len(ids)} and values {bad} disagree")
    return [{"clip_id": str(ids[i]), "row": int(rows[i]), "seed": int(seed), "role": str(part.get("role", "eval")),
             "prediction": {k: _num(v[i]) for k, v in values.items()}, "target": None} for i in range(n)]


def _part_lines(part: Mapping[str, Any], *, readout: str, seed: int, protocol: Mapping[str, Any]) -> list[dict]:
    if "values" in part:
        return _value_lines(part, seed=seed)
    outputs = np.asarray(part["outputs"])
    targets = part["targets"]
    rows = np.asarray(part["rows"], dtype=np.int64)
    ids: Sequence[str] = list(part["clip_ids"])
    folds = part.get("folds")
    n = rows.shape[0]
    if outputs.shape[0] != n or len(ids) != n:
        raise ValueError(f"samples: outputs {outputs.shape[0]}, rows {n} and clip_ids {len(ids)} disagree")
    lines: list[dict] = []
    preds: list[Any] = []
    tgts: list[Any] = []
    extra: list[dict] = [{} for _ in range(n)]
    if readout == "clf":
        if isinstance(targets, Mapping):  # multi_sets: {set: [n, C]}
            for i in range(n):
                preds.append(_top_k(outputs[i]))
                tgts.append({name: _indices(np.asarray(m)[i]) for name, m in targets.items()})
        else:
            t = np.asarray(targets)
            if t.ndim == 1:  # single
                pred = outputs.reshape(n, -1).argmax(axis=1)
                for i in range(n):
                    preds.append(int(pred[i]))
                    tgts.append(int(t[i]))
                    extra[i]["correct"] = bool(int(pred[i]) == int(t[i]))
            else:  # multi
                for i in range(n):
                    preds.append(_top_k(outputs[i]))
                    tgts.append(_indices(t[i]))
    elif readout == "seq":
        pred = outputs.argmax(axis=-1)  # [n, T]
        t = np.asarray(targets)
        for i in range(n):
            preds.append([int(v) for v in pred[i]])
            tgts.append([int(v) for v in t[i]])
    elif readout == "grid":
        from yavih.metrics.depth import depth_per_image

        per = depth_per_image(outputs, targets, protocol)
        for i in range(n):
            preds.append({k: _num(per[k][i]) for k in ("d1", "absrel", "rmse")})
            tgts.append(None)
    elif readout == "box":
        from yavih.metrics.box import box_iou

        t = np.asarray(targets, dtype=np.float64).reshape(n, 4)
        p = outputs.astype(np.float64).reshape(n, 4)
        iou = box_iou(p, t) if n else np.empty((0,))
        for i in range(n):
            preds.append([float(v) for v in p[i]])
            tgts.append([float(v) for v in t[i]])
            extra[i]["iou"] = float(iou[i])
    else:  # pragma: no cover - READOUT_VERSION is closed
        raise ValueError(f"samples: unknown readout {readout!r}")
    for i in range(n):
        line: dict[str, Any] = {"clip_id": str(ids[i]), "row": int(rows[i]), "seed": int(seed), "role": str(part.get("role", "eval"))}
        if folds is not None:
            line["fold"] = str(folds[i])
        line["prediction"] = preds[i]
        line["target"] = tgts[i]
        line.update(extra[i])
        lines.append(line)
    return lines


def sample_lines(parts: Sequence[Mapping[str, Any]], *, readout: str, seed: int, protocol: Mapping[str, Any] | None = None) -> list[dict]:
    """Every part's rows as sample lines, in part order then row order."""
    out: list[dict] = []
    for part in parts:
        out.extend(_part_lines(part, readout=readout, seed=seed, protocol=dict(protocol or {})))
    return out
