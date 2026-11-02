"""``yavih features inspect``: per-dim std histogram, near-constant dims, mean abs, pos
summary, dead rows.

Diagnostic only: nothing here decides a verdict (that is ``validate_features``). It reads
the same seeded sample ``validate`` uses for ``E_DEAD_ROWS``/``E_DUPLICATE_CLIPS`` so an agent
looking at "why did validate complain" sees the same clips.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from yavih import hashing
from yavih.io.validate import DEAD_ROW_REL_VAR, features_block, iter_members, list_shards

__all__ = ["inspect_features", "NEAR_CONSTANT_REL_STD", "HIST_BINS"]

#: A feature dimension whose std is below this fraction of the median per-dim std is "near-constant".
NEAR_CONSTANT_REL_STD: float = 1e-3
HIST_BINS: int = 16


def _sample_rows(split_dir: Path, sidecars: list[dict], sample_rows: int) -> tuple[str, list[int]]:
    present = sorted({int(m["row"]) for s in sidecars for m in s["members"]})
    manifest_hash = str((sidecars[0].get("benchmark") or {}).get("manifest_hash"))
    split = str(sidecars[0].get("split", split_dir.name))
    rng = np.random.default_rng(hashing.rng_seed(f"{manifest_hash}:{split}:dup"))
    k = min(int(sample_rows), len(present))
    rows = sorted(int(r) for r in rng.choice(np.asarray(present, dtype=np.int64), size=k, replace=False)) if k else []
    return split, rows


def inspect_features(features_dir: str | os.PathLike, *, sample_rows: int = 512) -> dict:
    """Summary statistics of a split directory over the seeded sample (``features inspect``)."""
    split_dir = Path(features_dir)
    shards = list_shards(split_dir)
    if not shards:
        raise FileNotFoundError(f"{split_dir}: no finished shards (sidecar + tar) to inspect")
    sidecars = []
    for _idx, sc_path, _tar in shards:
        with open(sc_path, encoding="utf-8") as fh:
            sidecars.append(json.load(fh))
    from yavih.io.finalize import generic_outputs

    outputs = generic_outputs(sidecars)
    if outputs is not None:
        return _inspect_generic(split_dir, sidecars, outputs, sample_rows)
    feats = features_block(sidecars)
    if feats.get("N") is None:
        raise FileNotFoundError(f"{split_dir}: no shard wrote any row (nothing to inspect)")
    N, D = int(feats["N"]), int(feats["D"])
    pos_mode = str(feats["pos_mode"])
    tag_names = [str(s) for s in ((sidecars[0]["bridge"].get("describe") or {}).get("streams") or [])]
    split, rows = _sample_rows(split_dir, sidecars, sample_rows)
    row_set = set(rows)

    s1 = np.zeros((N, D), dtype=np.float64)
    s2 = np.zeros((N, D), dtype=np.float64)
    abs_sum = 0.0
    n_values = 0
    n_zero = 0
    max_abs = 0.0
    n_seen = 0
    constant_clips: list[int] = []
    pos_arrays: list[np.ndarray] = []
    for rec in iter_members(split_dir, rows=row_set, with_pos=True):
        f = rec.feat.astype(np.float64)
        s1 += f
        s2 += f * f
        abs_sum += float(np.abs(f).sum())
        n_values += f.size
        n_zero += int(np.count_nonzero(f == 0))
        max_abs = max(max_abs, float(np.abs(f).max()) if f.size else 0.0)
        n_seen += 1
        if f.size and ((rec.feat.shape[0] > 1 and bool((rec.feat == rec.feat[0]).all())) or not bool(rec.feat.any())):
            constant_clips.append(rec.row)
        if rec.pos is not None and (pos_mode == "per_sample" or not pos_arrays):
            pos_arrays.append(np.asarray(rec.pos, dtype=np.float64))

    out: dict[str, Any] = {
        "split": split,
        "N": N,
        "D": D,
        "dtype": str(feats["dtype"]),
        "pos_mode": pos_mode,
        "n_present": sum(len(s["members"]) for s in sidecars),
        "n_sampled": n_seen,
        "sampled_rows_first10": rows[:10],
    }
    if n_seen == 0:
        out["note"] = "no present rows"
        return out
    n = float(n_seen)
    mean = s1 / n
    var = np.maximum(s2 / n - mean * mean, 0.0)
    per_dim_std = np.sqrt(var.mean(axis=0))  # [D]: token-averaged variance per feature dim
    median_std = float(np.median(per_dim_std))
    near_const = np.nonzero(per_dim_std < NEAR_CONSTANT_REL_STD * max(median_std, 1e-30))[0]
    hist_counts, hist_edges = np.histogram(per_dim_std, bins=HIST_BINS)
    per_token_var = var.mean(axis=1)  # [N]
    mean_var = float(per_token_var.mean())
    dead = np.nonzero(per_token_var < DEAD_ROW_REL_VAR * mean_var)[0] if n_seen >= 2 else np.array([], dtype=np.int64)

    out.update(
        {
            "mean_abs": abs_sum / max(n_values, 1),
            "max_abs": max_abs,
            "zero_fraction": n_zero / max(n_values, 1),
            "global_mean": float(mean.mean()),
            "global_std": float(np.sqrt(var.mean())),
            "per_dim_std": {
                "min": float(per_dim_std.min()),
                "median": median_std,
                "max": float(per_dim_std.max()),
                "hist_counts": [int(c) for c in hist_counts],
                "hist_edges": [float(e) for e in hist_edges],
            },
            "near_constant_dims": {
                "count": int(near_const.size),
                "threshold_rel_std": NEAR_CONSTANT_REL_STD,
                "indices_first50": [int(i) for i in near_const[:50]],
            },
            "dead_rows": {
                "count": int(dead.size),
                "positions_first50": [int(i) for i in dead[:50]],
                "threshold_rel_var": DEAD_ROW_REL_VAR,
            },
            "constant_clips": {"count": len(constant_clips), "rows_first10": constant_clips[:10]},
        }
    )
    out["pos"] = _pos_summary(pos_arrays, tag_names, pos_mode)
    return out


def _pos_summary(pos_arrays: list[np.ndarray], tag_names: list[str], pos_mode: str) -> dict:
    if pos_mode == "none" or not pos_arrays:
        return {"mode": pos_mode, "n_tokens": None, "stream_counts": {}, "axes_present": [], "grid_inferred": None, "t_distinct": None}
    p = np.concatenate(pos_arrays, axis=0)
    s = p[:, 0]
    tagged = s >= 0
    stream_counts: dict[str, int] = {}
    for tag, cnt in zip(*np.unique(s[tagged], return_counts=True), strict=True):
        name = tag_names[int(tag)] if 0 <= int(tag) < len(tag_names) else f"#{int(tag)}"
        stream_counts[name] = int(cnt) // len(pos_arrays)
    axes = [axis for col, axis in ((1, "t"), (2, "y"), (3, "x")) if bool((p[:, col] >= 0).any())]
    yx = (p[:, 2] >= 0) & (p[:, 3] >= 0)
    grid = [int(np.unique(np.round(p[yx, 2], 6)).size), int(np.unique(np.round(p[yx, 3], 6)).size)] if bool(yx.any()) else None
    t = p[:, 1]
    return {
        "mode": pos_mode,
        "n_tokens": int(pos_arrays[0].shape[0]),
        "global_tokens": int((~tagged).sum()) // len(pos_arrays),
        "stream_counts": stream_counts,
        "axes_present": axes,
        "grid_inferred": grid,
        "t_distinct": int(np.unique(np.round(t[t >= 0], 6)).size),
    }


# ----------------------------------------------------------------------------- a declared output schema


def _stats(x: list[float]) -> dict[str, float] | None:
    if not x:
        return None
    a = np.asarray(x, dtype=np.float64)
    return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def _summarize(codec_name: str, values: list[Any]) -> dict[str, Any]:
    """Per-key diagnostics by codec: wav (duration, rate, channels, level, silent / clipped clips), array (shapes,
    dtypes, magnitude, finiteness, constant values), json (value types), anything else (sizes)."""
    from collections import Counter

    from yavih import codecs as C

    if codec_name == "wav":
        dur, rms, peak, rates, chans, bad = [], [], [], Counter(), Counter(), 0
        for v in values:
            try:
                x, sr = C.read_wav(bytes(v))
            except ValueError:
                bad += 1
                continue
            dur.append(x.shape[0] / sr)
            rms.append(float(np.sqrt(np.mean(x * x))) if x.size else 0.0)
            peak.append(float(np.abs(x).max()) if x.size else 0.0)
            rates[int(sr)] += 1
            chans[int(x.shape[1]) if x.ndim == 2 else 1] += 1
        return {"codec": "wav", "duration_s": _stats(dur), "sample_rates": dict(rates), "channels": dict(chans), "rms": _stats(rms),
                "peak": _stats(peak), "silent": int(sum(r < 1e-4 for r in rms)), "clipped": int(sum(p >= 0.999 for p in peak)), "unreadable": bad}
    if codec_name == "array":
        shapes, dtypes, mabs, finite, const = Counter(), Counter(), [], 0, 0
        for v in values:
            a = np.asarray(v)
            shapes[str(tuple(a.shape))] += 1
            dtypes[str(a.dtype)] += 1
            f = a.astype(np.float64, copy=False) if a.dtype.kind in "fiub" else None
            if f is not None and f.size:
                mabs.append(float(np.abs(f).mean()))
                finite += int(np.isfinite(f).all())
                const += int(bool((f == f.reshape(-1)[0]).all()))
        return {"codec": "array", "shapes_top5": dict(shapes.most_common(5)), "dtypes": dict(dtypes), "mean_abs": _stats(mabs),
                "all_finite": finite, "constant": const}
    if codec_name == "json":
        kinds = Counter(type(v).__name__ for v in values)
        first = json.dumps(values[0], default=str)[:200] if values else None
        return {"codec": "json", "types": dict(kinds), "first": first}
    sizes = [len(v) if isinstance(v, (bytes, bytearray, str)) else len(json.dumps(v, default=str)) for v in values]
    return {"codec": codec_name, "size_bytes": _stats([float(n) for n in sizes])}


def _inspect_generic(split_dir: Path, sidecars: list[dict], outputs: dict, sample_rows: int) -> dict:
    """``features inspect`` of a set with a declared output schema: each key summarized by its codec over the seeded
    sample (a folder codec not registered here is named, not decoded)."""
    from yavih import codecs as C
    from yavih.io.reader import FeatureReader

    reader = FeatureReader(split_dir, require_meta=False)
    split, rows = _sample_rows(split_dir, sidecars, sample_rows)
    schema = outputs.get("schema") or {}
    reg = C.registry()
    keys: dict[str, Any] = {}
    for key in reader.keys():
        spec = dict(schema.get(key) or {})
        if spec.get("type") not in reg:
            keys[key] = {"codec": spec.get("type"), "note": "a folder codec not registered here: pass through the benchmark to decode"}
            continue
        keys[key] = _summarize(str(spec["type"]), reader.values(key, rows, codecs=reg))
    return {"split": split, "schema": schema, "n_present": sum(len(s["members"]) for s in sidecars), "n_sampled": len(rows),
            "sampled_rows_first10": rows[:10], "keys": keys}
