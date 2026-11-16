# ruff: noqa: UP006, UP035, UP045  (the copied upstream blocks keep their typing.Dict / List / Optional verbatim)
"""AV-SyncBench's pairwise synchronisation score: the folder's own training-free scorer and its metrics.

Every row is one pair: an original video with its own audio (the positive) and the same video with a perturbed audio
(the negative: shifted, sped up or slowed down, locally jittered, or timbre-edited). The bridge embeds each stream
segment by segment on the protocol's grid (segment ``k`` covers ``[k * segment_seconds, (k + 1) * segment_seconds)``
of its stream; one embedding per COMPLETE segment): ``video_seg [Tv, D]`` from the frames, ``audio_seg [Ta, D]``
from the mp4's own audio track, ``edited_seg [Te, D]`` from the perturbed wav, all in the model's shared space. The
scorer (:func:`pair_scores`) keeps the first ``min(Tv, Ta, Te)`` segments (the segments of the pair's common
duration), at most ``max_segments`` of them chosen uniformly, L2-normalizes, and scores each audio by its mean
diagonal cosine with the video: ``S = mean_k cos(v_k, a_k)``. The pair is correct iff ``S(positive) > S(negative)``
strictly (a tie is wrong). Per row the output is ``float64 [2]`` = ``(S_pos, S_neg)``; a row with no common
segment, or a non-finite score, is invalid (``NaN``) and is left out of every accuracy (the upstream ``accuracy``,
valid records only).

Provenance (grounding rung 2: the benchmark authors' own evaluation code, not released on PyPI, copied):
    AV-SyncBench evaluation toolkit, https://github.com/fgt7t6g/AV-SyncBench @ 82e13536074377b71a51d81b2f5559a43cf4a122
    licence: none stated (the README says the repository "does not declare a separate license for the new
    AV-SyncBench adapter code"); the dataset card says MIT.
    - ``avsyncbench/metrics.py``: ``PairwiseResult``, ``_group_summary``, ``summarize_results`` copied whole (the
      strict decision, ties incorrect, valid-only ``accuracy``); unchanged but for the removed ``asdict`` /
      ``to_dict`` serialisation helpers and the result-file writers, which yavih replaces.
    - ``avsyncbench/media.py``: ``non_overlapping_starts`` copied whole, unchanged.
    - ``avsyncbench/evaluators/imagebind.py``: the scoring part of ``ImageBindScorer.score_record`` (the common
      duration, the ``paper_legacy`` cap of ``max_segments`` segments chosen with ``linspace(...).round()``, the
      per-segment ``F.normalize`` and the mean of the diagonal dot products) transcribed to numpy as
      :func:`pair_scores`; the model-specific transforms of that file (two frames per segment warped to 224 x 224, a
      log-mel 128 x 204 per segment) are the defaults of ``bridge_template.py``, not protocol. Change: the common
      duration is the smallest segment count of the three streams (equal to the upstream
      ``floor((min duration + 1e-9) / segment_seconds)`` when each stream's count is computed the same way).
    The metrics (:func:`make_metrics`) aggregate the per-pair decision as the paper's tables do: a task's accuracy
    is the unweighted mean of its levels' accuracies (Table 2 "Overall"); the timbre accuracy pools its pairs; the
    headline ``acc_task_mean`` (the mean of the four task accuracies) is yavih's, not the paper's.
Module scope: stdlib, numpy, yavih core.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from yavih.metrics._common import as_array
from yavih.metrics.registry import MetricSpec
from yavih.readouts.base import ProbeRefused, evaluate_metrics
from yavih.readouts.registry import Scorer

__all__ = ["VERSION", "KEYS", "TASK_LEVELS", "TEMPORAL_TASKS", "TIMBRE_TASKS", "PairwiseResult", "summarize_results",
           "non_overlapping_starts", "segment_indices", "pair_scores", "decisions", "group_masks", "run", "make_scorer",
           "make_metrics", "metric_names"]

#: The scorer version (enters ``protocol_hash``); a change of the score, the decision or the aggregation bumps it.
VERSION: int = 1
KEYS: tuple[str, str, str] = ("video_seg", "audio_seg", "edited_seg")
#: the levels of the temporal tasks, as the manifest's ``level`` column spells them, and their metric suffixes
TASK_LEVELS: dict[str, dict[str, str]] = {
    "offset": {"50ms": "50ms", "100ms": "100ms", "200ms": "200ms", "300ms": "300ms", "500ms": "500ms"},
    "jitter": {"L1": "l1", "L2": "l2", "L3": "l3"},
    "speed": {lv: lv.replace(".", "").rstrip("x") for lv in ("0.80x", "0.83x", "0.87x", "0.91x", "0.95x", "1.05x", "1.10x", "1.15x", "1.20x", "1.25x")},
}
TEMPORAL_TASKS: tuple[str, ...] = ("offset", "jitter", "speed")
TIMBRE_TASKS: tuple[str, ...] = ("voice_timbre", "instrument_timbre")
PROTOCOL_KEYS: frozenset[str] = frozenset({"segment_seconds", "max_segments"})
#: rows per (task, level) group of the conformance population
POPULATION_PER_GROUP = 6


# ----------------------------------------------------------------------------- copied: avsyncbench/metrics.py @ 82e1353
# AV-SyncBench evaluation toolkit, https://github.com/fgt7t6g/AV-SyncBench, avsyncbench/metrics.py at commit
# 82e13536074377b71a51d81b2f5559a43cf4a122; licence: none stated. Copied whole; removed: ``to_dict`` (asdict
# serialisation) and ``write_jsonl`` / ``write_summary`` (yavih writes result.json). Nothing else changed.


@dataclass(frozen=True)
class PairwiseResult:
    sample_id: str
    task: str
    positive_score: Optional[float]
    negative_score: Optional[float]
    edit_type: Optional[str] = None
    offset_ms: Optional[int] = None
    direction: Optional[str] = None
    category: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.error is None
            and self.positive_score is not None
            and self.negative_score is not None
            and math.isfinite(self.positive_score)
            and math.isfinite(self.negative_score)
        )

    @property
    def is_tie(self) -> bool:
        return self.is_valid and self.positive_score == self.negative_score

    @property
    def correct(self) -> bool:
        # AV-SyncBench uses a strict comparison; ties are counted as incorrect.
        return self.is_valid and self.positive_score > self.negative_score


def _group_summary(results: List[PairwiseResult]) -> Dict[str, Any]:
    valid = [result for result in results if result.is_valid]
    correct = sum(result.correct for result in valid)
    ties = sum(result.is_tie for result in valid)
    return {
        "total": len(results),
        "valid": len(valid),
        "errors": len(results) - len(valid),
        "correct": correct,
        "ties": ties,
        # `accuracy` matches the valid-only research metric. The all-record
        # variant prevents a permissive continue-on-error run from looking
        # better simply because failures disappeared from the denominator.
        "accuracy": None if not valid else correct / len(valid),
        "accuracy_all_records": None if not results else correct / len(results),
        "error_rate": None if not results else (len(results) - len(valid)) / len(results),
    }


def summarize_results(results: Iterable[PairwiseResult]) -> Dict[str, Any]:
    results = list(results)
    by_offset: Dict[str, List[PairwiseResult]] = defaultdict(list)
    by_edit_type: Dict[str, List[PairwiseResult]] = defaultdict(list)
    by_category: Dict[str, List[PairwiseResult]] = defaultdict(list)
    by_direction: Dict[str, List[PairwiseResult]] = defaultdict(list)
    for result in results:
        if result.offset_ms is not None:
            by_offset[str(result.offset_ms)].append(result)
        if result.edit_type:
            by_edit_type[result.edit_type].append(result)
        if result.category:
            by_category[result.category].append(result)
        if result.direction:
            by_direction[result.direction].append(result)
    return {
        "overall": _group_summary(results),
        "by_offset_ms": {
            key: _group_summary(group)
            for key, group in sorted(by_offset.items(), key=lambda item: int(item[0]))
        },
        "by_edit_type": {key: _group_summary(group) for key, group in sorted(by_edit_type.items())},
        "by_category": {key: _group_summary(group) for key, group in sorted(by_category.items())},
        "by_direction": {key: _group_summary(group) for key, group in sorted(by_direction.items())},
    }


# ----------------------------------------------------------------------------- copied: avsyncbench/media.py @ 82e1353
# AV-SyncBench evaluation toolkit, avsyncbench/media.py at commit 82e13536074377b71a51d81b2f5559a43cf4a122; licence:
# none stated. Copied whole, unchanged. The bridge uses it to cut each stream into complete segments.


def non_overlapping_starts(duration: float, segment_seconds: float) -> List[float]:
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")
    count = int(math.floor((duration + 1e-9) / segment_seconds))
    return [index * segment_seconds for index in range(count)]


# ----------------------------------------------------------------------------- transcribed: imagebind.py score_record
# AV-SyncBench evaluation toolkit, avsyncbench/evaluators/imagebind.py (ImageBindScorer.score_record) at commit
# 82e13536074377b71a51d81b2f5559a43cf4a122; licence: none stated. Transcribed from torch to numpy: the segment
# selection (``paper_legacy``: at most ``max_segments``, ``torch.linspace(0, n - 1, steps=max_segments).round()``) and
# the score (``F.normalize`` each embedding, ``(visual * audio).sum(-1).mean()``). The model's forward and transforms
# stay in the bridge.


def segment_indices(n: int, max_segments: int | None) -> np.ndarray:
    """``int64 [m]``: the segments of a pair with ``n`` common segments that are scored (all of them, or
    ``max_segments`` chosen uniformly: ``round(linspace(0, n - 1, max_segments))``, round half to even as torch)."""
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    if max_segments is not None and n > int(max_segments):
        return np.rint(np.linspace(0.0, n - 1.0, int(max_segments), dtype=np.float32)).astype(np.int64)
    return np.arange(n, dtype=np.int64)


def _normalize(x: np.ndarray) -> np.ndarray:
    """``torch.nn.functional.normalize(x, dim=-1)``: ``x / max(||x||, 1e-12)``."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _segments(x: Any) -> np.ndarray:
    a = as_array(x, np.float64)
    if a.ndim == 1:
        a = a[None]
    if a.ndim != 2:
        raise ValueError(f"a segment embedding must be [T, D], got {a.shape}")
    return a


def pair_scores(video: Any, positive: Any, negative: Any, *, max_segments: int | None) -> tuple[float, float, int]:
    """``(S_pos, S_neg, m)``: the mean diagonal cosine of the video segments with the original and with the
    perturbed audio segments over the ``m`` scored segments of the pair's common duration; ``(nan, nan, 0)`` when the
    three streams share no complete segment."""
    v, p, q = _segments(video), _segments(positive), _segments(negative)
    if not (v.shape[1] == p.shape[1] == q.shape[1]):
        raise ValueError(f"segment embeddings of one pair differ in dimension: video {v.shape}, audio {p.shape}, edited {q.shape}")
    idx = segment_indices(min(len(v), len(p), len(q)), max_segments)
    if len(idx) == 0:
        return float("nan"), float("nan"), 0
    vn, pn, qn = _normalize(v[idx]), _normalize(p[idx]), _normalize(q[idx])
    pos = float((vn * pn).sum(axis=-1).mean())
    neg = float((vn * qn).sum(axis=-1).mean())
    return pos, neg, int(len(idx))


# ----------------------------------------------------------------------------- decisions and groups


def decisions(outputs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(valid, correct, tie)``, ``bool [n]`` each, from ``(S_pos, S_neg)`` rows: :class:`PairwiseResult`'s rules."""
    o = as_array(outputs, np.float64)
    if o.ndim != 2 or o.shape[1] != 2:
        raise ValueError(f"av_syncbench outputs must be [n, 2] (S_pos, S_neg), got {o.shape}")
    valid = np.isfinite(o).all(axis=1)
    pos, neg = np.where(valid, o[:, 0], 0.0), np.where(valid, o[:, 1], 0.0)
    return valid, valid & (pos > neg), valid & (pos == neg)


def group_masks(split_meta: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """``{group: bool [n]}``: ``<task>`` and ``<task>:<level>`` of every row (``split_meta["task"]`` / ``["level"]``)."""
    task, level = np.asarray(split_meta["task"], dtype=object), np.asarray(split_meta["level"], dtype=object)
    out: dict[str, np.ndarray] = {}
    for t in np.unique(task):
        m = task == t
        out[str(t)] = m
        for lv in np.unique(level[m]):
            out[f"{t}:{lv}"] = m & (level == lv)
    return out


def _mask(split_meta: Mapping[str, Any], group: str, n: int) -> np.ndarray:
    return group_masks(split_meta).get(group, np.zeros(n, dtype=bool))


def _weights(outputs: Any, split_meta: Mapping[str, Any], parts: list[list[str]]) -> np.ndarray:
    """``float64 [n]`` row weights of a mean of means: ``parts`` is a list (outer, equal weights) of lists of groups
    (inner, equal weights: one level each, or several groups pooled when a part names more than one); each part's
    weight is spread over its valid rows. Parts with no valid row drop out (the others are re-weighted)."""
    valid, _, _ = decisions(outputs)
    n = len(valid)
    parts_w = []
    for part in parts:
        inner = []
        for groups in part:
            m = np.zeros(n, dtype=bool)
            for g in groups.split("+"):
                m |= _mask(split_meta, g, n)
            m &= valid
            if m.any():
                inner.append(m / m.sum())
        if inner:
            parts_w.append(np.mean(inner, axis=0))
    if not parts_w:
        return np.zeros(n, dtype=np.float64)
    return np.mean(parts_w, axis=0)


def _weighted(outputs: Any, split_meta: Mapping[str, Any], parts: list[list[str]]) -> float:
    w = _weights(outputs, split_meta, parts)
    if w.sum() <= 0:
        return float("nan")
    _, correct, _ = decisions(outputs)
    return float((w * correct).sum() / w.sum())


def _weighted_per_clip(outputs: Any, split_meta: Mapping[str, Any], parts: list[list[str]]) -> np.ndarray:
    """Per-row values whose mean over the non-NaN rows is :func:`_weighted` (``correct * w * n_population``)."""
    w = _weights(outputs, split_meta, parts)
    _, correct, _ = decisions(outputs)
    pop = w > 0
    if not pop.any():
        return np.full(len(w), np.nan)
    return np.where(pop, correct * (w / w.sum()) * pop.sum(), np.nan)


def _weighted_interval(parts: list[list[str]]):
    """``MetricSpec.interval`` of a mean of means: the percentile interval of the metric recomputed on every
    ``bootstrap_clip`` resample of the rows (``yavih.metrics.intervals.bootstrap_draws``, the shared scheme); each
    resample re-weights the groups by how often their rows were drawn."""

    def interval(kind: str, outputs: Any, targets: Any, protocol: Any, split_meta: Mapping[str, Any]) -> tuple[float, float]:
        from yavih.metrics.intervals import _percentile_interval, bootstrap_draws

        valid, correct, _ = decisions(outputs)
        n = len(valid)
        items, owner = [], []
        for j, part in enumerate(parts):
            for groups in part:
                m = np.zeros(n, dtype=bool)
                for g in groups.split("+"):
                    m |= _mask(split_meta, g, n)
                items.append(m & valid)
                owner.append(j)
        G = np.stack(items, axis=1).astype(np.float64)
        Gc = G * correct[:, None]
        owner_a = np.asarray(owner)
        b = split_meta.get("bootstrap") or {}
        stats = []
        for d in bootstrap_draws(n, samples=int(b.get("samples", 2000)), seed=int(b.get("seed", 0))):
            c = np.bincount(d, minlength=n).astype(np.float64)
            num, den = c @ Gc, c @ G
            acc = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
            per_part = [acc[owner_a == j] for j in range(len(parts))]
            per_part = [float(np.nanmean(a)) for a in per_part if np.isfinite(a).any()]
            stats.append(float(np.mean(per_part)) if per_part else float("nan"))
        return _percentile_interval(np.asarray(stats))

    return interval


def _pooled_per_clip(outputs: Any, split_meta: Mapping[str, Any], groups: tuple[str, ...]) -> np.ndarray:
    valid, correct, _ = decisions(outputs)
    m = np.zeros(len(valid), dtype=bool)
    for g in groups:
        m |= _mask(split_meta, g, len(valid))
    m &= valid
    return np.where(m, correct.astype(np.float64), np.nan)


def _pooled(outputs: Any, split_meta: Mapping[str, Any], groups: tuple[str, ...]) -> float:
    v = _pooled_per_clip(outputs, split_meta, groups)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def _task_parts(task: str) -> list[list[str]]:
    """The inner parts of a task's accuracy: its levels (equal weights) for a temporal task, the pooled timbre pairs."""
    if task == "timbre":
        return [["voice_timbre+instrument_timbre"]]
    return [[f"{task}:{lv}" for lv in TASK_LEVELS[task]]]


def _mean_parts() -> list[list[str]]:
    return [_task_parts(t)[0] for t in TEMPORAL_TASKS] + [_task_parts("timbre")[0]]


# ----------------------------------------------------------------------------- the driver


def _protocol(protocol: Mapping[str, Any]) -> tuple[float, int | None]:
    seg = float(protocol["segment_seconds"])
    cap = protocol.get("max_segments")
    return seg, (None if cap is None else int(cap))


def _row_meta(ctx: Any, split: str, rows: np.ndarray) -> dict[str, list[str]]:
    from yavih.readouts.base import _manifest_index

    man = ctx.manifest(split)
    idx = _manifest_index(man, rows)
    out: dict[str, list[str]] = {"task": [], "level": [], "direction": [], "category": []}
    for i in idx:
        ex = man.extra[i]
        for k in out:
            out[k].append(str(ex.get(k) or ""))
    return out


def _meta(rows: np.ndarray, ids: list[str], groups: Mapping[str, list[str]], n_segments: np.ndarray, ctx: Any, seed: int) -> dict[str, Any]:
    return {"readout": ctx.spec.readout, "n_train": 0, "n_eval": int(len(rows)), "n_val": 0, "eval_rows": rows, "eval_clip_ids": ids,
            "task": list(groups["task"]), "level": list(groups["level"]), "direction": list(groups["direction"]),
            "category": list(groups["category"]), "n_segments": n_segments, "chance": dict(ctx.chance),
            "bootstrap": dict(ctx.bootstrap or {}), "seed": seed}


def run(ctx: Any) -> dict:
    """Score every present eval pair (the result does not depend on the seed); the ``result.json`` blocks."""
    if ctx.roles["folds"]:
        raise ProbeRefused("E_FOLDS", "the av_syncbench scorer scores one eval set; splits.folds is not supported")
    _, cap = _protocol(ctx.protocol)
    t0 = time.time()
    scores, nseg, all_rows, ids = [], [], [], []
    groups: dict[str, list[str]] = {"task": [], "level": [], "direction": [], "category": []}
    for s in ctx.roles["eval"]:
        r = ctx.present_rows(s)
        if len(r) == 0:
            continue
        reader = ctx.reader(s)
        vals = [list(reader.values(k, r, codecs=ctx.bench.codecs)) for k in KEYS]
        for v, a, e in zip(*vals, strict=True):
            try:
                pos, neg, m = pair_scores(v, a, e, max_segments=cap)
            except ValueError as err:
                raise ProbeRefused("E_TARGETS", f"av_syncbench: {err}") from err
            scores.append((pos, neg))
            nseg.append(m)
        all_rows.append(r)
        ids.extend(ctx.clip_ids(s, r))
        for k, v in _row_meta(ctx, s, r).items():
            groups[k].extend(v)
    if not all_rows:
        raise ProbeRefused("E_SPLIT_MISSING", f"no present eval rows in {ctx.roles['eval']}")
    rows = np.concatenate(all_rows)
    outputs = np.asarray(scores, dtype=np.float64).reshape(-1, 2)
    n_segments = np.asarray(nseg, dtype=np.int64)
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t0)
    targets = np.zeros(len(rows), dtype=np.int64)  # the label of every row is the same: the original audio wins
    split_meta = _meta(rows, ids, groups, n_segments, ctx, ctx.seed)
    blocks = evaluate_metrics(ctx.metric_specs, outputs=outputs, targets=targets, protocol=ctx.protocol, split_meta=split_meta,
                              floor_outputs=None, ci=ctx.ci, scorer=ctx.scorer(ctx.roles["eval"], rows, targets))
    valid, correct, tie = decisions(outputs)
    results = [PairwiseResult(sample_id=i, task=t, positive_score=float(o[0]), negative_score=float(o[1]), edit_type=t,
                              direction=d or None, category=c or None) for i, t, d, c, o in
               zip(ids, groups["task"], groups["direction"], groups["category"], outputs, strict=True)]
    upstream = summarize_results(results)["overall"]
    return {**blocks, "probe": {}, "fit": {"score": "mean diagonal cosine over the common complete segments", "decision": "S_pos > S_neg (ties wrong)",
                                           "max_segments": cap, "invalid": int((~valid).sum()), "ties": int(tie.sum()),
                                           "segments": {"min": int(n_segments.min()), "max": int(n_segments.max())},
                                           "upstream_overall": {k: upstream[k] for k in ("total", "valid", "correct", "ties", "accuracy")}},
            "eval_outputs": outputs.astype(np.float32), "eval_rows": rows, "extra_arrays": {"n_segments": n_segments}, "backend": "ram",
            "timing": {k: ctx.timing[k] for k in ("score_s",) if k in ctx.timing},
            "samples": ([{"role": "eval", "outputs": outputs, "targets": targets, "rows": rows, "clip_ids": ids}] if ctx.log_samples else None)}


def _validate_spec(readout_params: dict, requires: Any, protocol: dict) -> None:
    if readout_params:
        raise ValueError(f"av_syncbench takes no readout_params, got {sorted(readout_params)}")
    seg = protocol.get("segment_seconds")
    if not isinstance(seg, (int, float)) or isinstance(seg, bool) or not seg > 0:
        raise ValueError("protocol.segment_seconds: a number > 0")
    cap = protocol.get("max_segments")
    if cap is not None and (not isinstance(cap, int) or isinstance(cap, bool) or cap < 1):
        raise ValueError("protocol.max_segments: an int >= 1 or null")


def _population(ctx: Any, tag: str) -> dict[str, Any]:
    """The conformance population: ``POPULATION_PER_GROUP`` seeded rows of every (task, level) group of the eval
    split (every temporal level and both timbre tasks appear) with fabricated ``(S_pos, S_neg)``: uniform scores in
    ``[-1, 1]`` with a per-group bias, every fifth pair tied, one invalid row."""
    from yavih.conformance import _values, select_rows

    rows_all, ids, groups = [], [], {"task": [], "level": [], "direction": [], "category": []}
    for s in ctx.roles["eval"]:
        meta = _row_meta(ctx, s, np.arange(ctx.bench.n_rows(s), dtype=np.int64))
        keys = [(t, lv if t in TASK_LEVELS else "") for t, lv in zip(meta["task"], meta["level"], strict=True)]
        by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, k in enumerate(keys):
            by_group[k].append(i)
        picked = []
        for k in sorted(by_group):
            members = np.asarray(by_group[k], dtype=np.int64)
            sel = select_rows(ctx.bench.manifest_hash, s, len(members), POPULATION_PER_GROUP, salt=f"eval:{k[0]}:{k[1]}")
            picked.extend(members[sel].tolist())
        r = np.asarray(sorted(picked), dtype=np.int64)
        rows_all.append(r)
        ids.extend(ctx.clip_ids(s, r))
        for k, v in _row_meta(ctx, s, r).items():
            groups[k].extend(v)
    rows = np.concatenate(rows_all)
    n = len(rows)
    u = _values(f"{tag}:av_syncbench:scores", (n, 2)) * 2.0 - 1.0
    bias = np.asarray([(hash_level(t, lv) - 0.5) * 0.4 for t, lv in zip(groups["task"], groups["level"], strict=True)])
    outputs = np.stack([u[:, 0] + bias, u[:, 1]], axis=1)
    outputs[::5, 1] = outputs[::5, 0]  # ties: wrong by the strict rule
    outputs[n // 2] = np.nan  # one invalid pair: left out of every accuracy
    n_segments = np.full(n, 15, dtype=np.int64)
    return {"outputs": outputs, "targets": np.zeros(n, dtype=np.int64), "split_meta": _meta(rows, ids, groups, n_segments, ctx, 0), "floor_outputs": None,
            "eval_splits": list(ctx.roles["eval"]), "eval_rows": rows}


def hash_level(task: str, level: str) -> float:
    """A deterministic number in ``[0, 1)`` per (task, level) (the conformance population's per-group bias)."""
    import hashlib

    return int.from_bytes(hashlib.sha256(f"{task}:{level}".encode()).digest()[:4], "big") / 2**32


def make_scorer(name: str, *, module: str, version: int = VERSION) -> Scorer:
    """The folder's stateless pairwise scorer, registered under the folder's ``name``."""
    return Scorer(
        name=name, version=version, module=module, kind="stateless", label_kinds=frozenset({"none"}), param_keys=frozenset(),
        protocol_keys=PROTOCOL_KEYS, requires_t_bins=False, requires_grid=False, dense=False, score=run, floor_outputs=None,
        needs_gpu=False, validate_spec=_validate_spec, output_keys=frozenset(KEYS), conformance_population=_population,
    )


# ----------------------------------------------------------------------------- metric specs


def metric_names() -> list[str]:
    """Every folder metric, in ``metrics.list`` order (``n_eval`` is the registry's)."""
    names = ["acc_task_mean", "acc_offset", "acc_jitter", "acc_speed", "acc_timbre", "acc_voice_timbre", "acc_instrument_timbre", "acc_all"]
    for t in TEMPORAL_TASKS:
        names += [f"acc_{t}_{suffix}" for suffix in TASK_LEVELS[t].values()]
    return names + ["tie_rate", "invalid_rate"]


def make_metrics(name: str) -> dict[str, MetricSpec]:
    """``acc_task_mean`` (headline: the mean of the four task accuracies), ``acc_<task>`` (a temporal task: the unweighted
    mean of its levels; ``acc_timbre``: the timbre pairs pooled), the per-level and per-timbre-task accuracies,
    ``acc_all`` (every valid pair pooled: the upstream ``summarize_results`` overall accuracy) and the disclosures
    ``tie_rate`` and ``invalid_rate``. Every accuracy counts valid pairs only; chance is 0.5."""
    ro = frozenset({name})
    ci_w = frozenset({"bootstrap_clip"})
    ci_p = frozenset({"binomial", "bootstrap_clip"})
    out: dict[str, MetricSpec] = {}

    def weighted(metric: str, parts: list[list[str]], desc: str) -> None:
        out[metric] = MetricSpec(metric, lambda o, t, p, m, parts=parts: _weighted(o, m, parts), ro,
                                 per_clip=lambda o, t, p, m, parts=parts: _weighted_per_clip(o, m, parts), ci=ci_w,
                                 interval=_weighted_interval(parts), description=desc)

    def pooled(metric: str, groups: tuple[str, ...], desc: str) -> None:
        out[metric] = MetricSpec(metric, lambda o, t, p, m, groups=groups: _pooled(o, m, groups), ro,
                                 per_clip=lambda o, t, p, m, groups=groups: _pooled_per_clip(o, m, groups), ci=ci_p, description=desc)

    weighted("acc_task_mean", _mean_parts(), "the mean of acc_offset, acc_jitter, acc_speed and acc_timbre (yavih's aggregate)")
    for t in TEMPORAL_TASKS:
        weighted(f"acc_{t}", _task_parts(t), f"{t}: the unweighted mean of its levels' pairwise accuracies (the paper's Overall)")
        for lv, suffix in TASK_LEVELS[t].items():
            pooled(f"acc_{t}_{suffix}", (f"{t}:{lv}",), f"{t} level {lv}: the fraction of pairs with S_pos > S_neg")
    pooled("acc_timbre", TIMBRE_TASKS, "timbre editing (voice and instrument pairs pooled): the fraction with S_pos > S_neg")
    pooled("acc_voice_timbre", ("voice_timbre",), "voice timbre edits: the fraction of pairs with S_pos > S_neg")
    pooled("acc_instrument_timbre", ("instrument_timbre",), "instrument timbre edits: the fraction of pairs with S_pos > S_neg")
    pooled("acc_all", TEMPORAL_TASKS + TIMBRE_TASKS, "every valid pair pooled (the upstream summarize_results overall accuracy)")

    def tie_rate(o, t, p, m):
        valid, _, tie = decisions(o)
        return float(tie.sum() / valid.sum()) if valid.any() else float("nan")

    def invalid_rate(o, t, p, m):
        valid, _, _ = decisions(o)
        return float((~valid).mean()) if len(valid) else float("nan")

    out["tie_rate"] = MetricSpec("tie_rate", tie_rate, ro, kind="disclosure", higher_is_better=None,
                                 description="the fraction of valid pairs with S_pos == S_neg (scored wrong)")
    out["invalid_rate"] = MetricSpec("invalid_rate", invalid_rate, ro, kind="disclosure", higher_is_better=None,
                                     description="the fraction of scored rows with no common segment or a non-finite score")
    return out
