"""Training-free audio-visual source localisation heatmaps scored with the VGG-SS cIoU: the folder's own scorer and its
metrics.

The bridge returns per row one ``audio_emb [D]`` and the visual tokens of the annotated frame in the same shared
space, ``video_tokens [N, D]`` with ``video_pos [N, 2]`` = ``(y, x)`` centres in fractions of the ORIGINAL frame (a
token outside ``[0, 1]`` is dropped: CLS, registers). The heatmap of a row is the cosine similarity between the
audio embedding and each token, averaged per cell of the ``G x G`` grid tiling the full frame
(``readout_params.grid``); a cell no token falls in scores below every filled cell (a bridge that cropped leaves them
empty).

The maps go through a transcription of the VGG-SS evaluation (Chen et al., CVPR 2021, ``test.py`` /
``utils.Evaluator`` of ``hche11/Localizing-Visual-Sounds-the-Hard-Way``, the scorer EZ-VSL and its successors reuse):
the ground truth of a clip is the union of its boxes rasterised on a ``gt_size x gt_size`` grid (per coordinate
``max(x, 0)`` then ``int(gt_size * x)``, used as slice bounds); the predicted map is upsampled bilinearly to
``gt_size`` (half-pixel centres, as ``cv2.resize`` / ``F.interpolate(align_corners=False)``) and binarised at its
upper median (``sorted(pred)[n // 2]``, strictly above); ``cIoU = |pred ∧ gt| / (|gt| + |pred ∧ ¬gt|)`` (``0`` when
the denominator is ``0``: an empty ground truth that nothing was predicted on; such rows are counted by
``n_empty_gt`` and kept). Metrics: ``ciou@0.5`` (headline: the fraction of clips with ``cIoU >= 0.5``), ``auc`` (the
trapezoid area of the fraction ``cIoU >= τ`` over ``τ = 0, 0.05, …, 1``), ``ciou_mean``, ``n_empty_gt``.

Floor ``ciou_floor``: the mean ground-truth mask of the other eval rows (leave-one-out; the train masks when a train
split exists) pooled to the grid, through the same pipeline: the dataset's centre / size prior. Controls:
``ciou_shuffled`` scores the maps of every row's visual tokens against ANOTHER row's audio embedding (a seeded
derangement, ``readout_params.shuffle_seed``), and ``audio_blind`` says whether the headline fails to beat it by more
than 3 McNemar σ on paired ``cIoU >= 0.5`` successes (the map is then visual saliency, not localisation; the report
voids such a result). Targets come from the manifest column ``readout_params.boxes_column`` (``x1|y1|x2|y2``
quadruples ``;``-joined, fractions of the frame). Module scope: stdlib, numpy, yavih core (a candidate for the core:
``yavih bench extensions``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from yavih.metrics._common import as_array, require_meta
from yavih.metrics.box import AUDIO_BLIND_SIGMAS, mcnemar_z
from yavih.metrics.registry import MetricSpec
from yavih.readouts.base import ProbeRefused, evaluate_metrics
from yavih.readouts.registry import Scorer

__all__ = ["VERSION", "KEYS", "AUC_THRESHOLDS", "POPULATION_EVAL", "rasterize", "upsample_bilinear", "median_mask", "ciou", "cious",
           "pool_to_grid", "mean_mask_maps", "heatmaps", "boxes_targets", "run", "make_scorer", "make_metrics"]

#: The scorer version (enters ``eval_hash``); a change of the similarity, the cell pooling, the empty-cell rule or the
#: floor bumps it.
VERSION: int = 1
KEYS: tuple[str, str, str] = ("audio_emb", "video_tokens", "video_pos")
#: the cIoU thresholds of ``auc`` (``0.05 * i`` for ``i`` in ``0..20``)
AUC_THRESHOLDS: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(21))
#: eval rows of the conformance population
POPULATION_EVAL = 128
_CACHE: dict[int, tuple[Any, np.ndarray]] = {}
_CHUNK = 256


# ----------------------------------------------------------------------------- the published pipeline


def rasterize(boxes: Sequence[Any], size: int) -> np.ndarray:
    """``uint8 [n, size, size]``: per row the union of its boxes, each ``[x1, y1, x2, y2]`` in fractions mapped by
    ``max(x, 0)`` then ``int(size * x)`` and written as the slice ``[y1:y2, x1:x2]`` (the published rasteriser)."""
    out = np.zeros((len(boxes), size, size), dtype=np.uint8)
    for i, bb in enumerate(boxes):
        for box in np.asarray(bb, dtype=np.float64).reshape(-1, 4):
            x1, y1, x2, y2 = (int(size * max(float(v), 0.0)) for v in box)
            out[i, y1:y2, x1:x2] = 1
    return out


def upsample_bilinear(maps: Any, size: int) -> np.ndarray:
    """``float64 [n, size, size]`` from ``[n, G, G]`` by bilinear interpolation with half-pixel centres (the sampling of
    ``cv2.resize(INTER_LINEAR)`` and ``torch.nn.functional.interpolate(mode="bilinear", align_corners=False)``)."""
    m = as_array(maps, np.float64)
    n, g, _ = m.shape
    src = (np.arange(size) + 0.5) * (g / size) - 0.5
    lo = np.clip(np.floor(src), 0, g - 1).astype(np.int64)
    hi = np.clip(lo + 1, 0, g - 1)
    w = np.clip(src - lo, 0.0, 1.0)
    rows = m[:, lo, :] * (1 - w)[None, :, None] + m[:, hi, :] * w[None, :, None]  # [n, size, g]
    return rows[:, :, lo] * (1 - w)[None, None, :] + rows[:, :, hi] * w[None, None, :]


def median_mask(pred: Any) -> np.ndarray:
    """``bool [n, S, S]``: each map thresholded strictly above its upper median ``sorted(flat)[len // 2]`` (the
    ``np.sort(pred.flatten())[int(H * W / 2)]`` of the published scorer, then ``pred > threshold``)."""
    p = as_array(pred, np.float64)
    flat = p.reshape(p.shape[0], -1)
    k = flat.shape[1] // 2
    thr = np.partition(flat, k, axis=1)[:, k]
    return flat > thr[:, None]


def ciou(pred_mask: Any, gt: Any) -> np.ndarray:
    """``float64 [n]``: ``|pred ∧ gt| / (|gt| + |pred ∧ ¬gt|)``, ``0`` where the denominator is ``0``."""
    p = np.asarray(pred_mask, dtype=bool).reshape(len(pred_mask), -1)
    g = np.asarray(gt, dtype=bool).reshape(len(gt), -1)
    inter = (p & g).sum(axis=1).astype(np.float64)
    denom = g.sum(axis=1) + (p & ~g).sum(axis=1)
    return np.where(denom > 0, inter / np.maximum(denom, 1), 0.0)


def cious(outputs: Any, targets: Any, protocol: Any) -> np.ndarray:
    """Per-clip cIoU of the score maps against the rasterised targets (chunked; cached by the outputs' identity)."""
    key = id(outputs)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is outputs:
        return hit[1]
    size = int(protocol["gt_size"])
    maps = as_array(outputs, np.float32)
    if maps.ndim != 3 or maps.shape[1] != maps.shape[2]:
        raise ValueError(f"localization outputs must be [n, G, G] score maps, got {maps.shape}")
    if len(targets) != maps.shape[0]:
        raise ValueError(f"localization: {maps.shape[0]} outputs for {len(targets)} targets")
    out = np.empty(maps.shape[0], dtype=np.float64)
    for s in range(0, maps.shape[0], _CHUNK):
        gt = rasterize(targets[s:s + _CHUNK], size)
        out[s:s + _CHUNK] = ciou(median_mask(upsample_bilinear(maps[s:s + _CHUNK], size)), gt)
    if len(_CACHE) > 8:
        _CACHE.clear()
    _CACHE[key] = (outputs, out)  # the entry holds its array: the id cannot be reused while it lives
    return out


def pool_to_grid(masks: Any, grid: int) -> np.ndarray:
    """``float64 [n, grid, grid]``: the mean of each ``[n, S, S]`` map over the cells of a ``grid x grid`` tiling
    (cell ``(i, j)`` covers rows ``[i * S / grid, (i + 1) * S / grid)``)."""
    m = as_array(masks, np.float64)
    n, size, _ = m.shape
    edges = [int(round(k * size / grid)) for k in range(grid + 1)]
    out = np.empty((n, grid, grid), dtype=np.float64)
    for i in range(grid):
        for j in range(grid):
            out[:, i, j] = m[:, edges[i]:edges[i + 1], edges[j]:edges[j + 1]].mean(axis=(1, 2))
    return out


def mean_mask_maps(targets: Sequence[Any], n_eval: int, *, grid: int, size: int, leave_one_out: bool) -> np.ndarray:
    """The floor's score maps ``float32 [n_eval, grid, grid]``: the per-pixel mean of the rasterised targets pooled to
    the grid, the same map for every row (``leave_one_out`` false: the train masks), or each eval row's mean over
    the other eval rows (``leave_one_out`` true: ``targets`` are the eval targets, in row order)."""
    total = np.zeros((size, size), dtype=np.float64)
    for s in range(0, len(targets), _CHUNK):
        total += rasterize(targets[s:s + _CHUNK], size).sum(axis=0)
    if not leave_one_out:
        if len(targets) == 0:
            raise ValueError("mean_mask_maps: no train targets")
        pooled = pool_to_grid((total / len(targets))[None], grid)[0]
        return np.repeat(pooled[None].astype(np.float32), int(n_eval), axis=0)
    if len(targets) != n_eval or n_eval < 2:
        raise ValueError("mean_mask_maps: leave-one-out needs the eval targets (at least two rows)")
    out = np.empty((n_eval, grid, grid), dtype=np.float32)
    for s in range(0, n_eval, _CHUNK):
        own = rasterize(targets[s:s + _CHUNK], size).astype(np.float64)
        out[s:s + _CHUNK] = pool_to_grid((total[None] - own) / (n_eval - 1), grid)
    return out


# ----------------------------------------------------------------------------- the driver


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def heatmaps(audio: Any, tokens: list[Any], pos: list[Any], *, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """``(maps float32 [n, grid, grid], coverage float64 [n])``: per row the mean cosine of the audio embedding with
    the tokens whose ``(y, x)`` fall in each cell (``floor(y * grid)``, clipped to the last cell for ``y == 1``); an
    empty cell gets the row's minimum filled score minus one; ``coverage`` is the fraction of filled cells. A row
    without a token inside ``[0, 1]`` is refused (``E_POS_MISSING``)."""
    n = len(tokens)
    maps = np.empty((n, grid, grid), dtype=np.float32)
    cov = np.empty(n, dtype=np.float64)
    a = _unit(np.asarray(audio, dtype=np.float64).reshape(n, -1))
    for i in range(n):
        t = np.asarray(tokens[i], dtype=np.float64).reshape(-1, a.shape[1])
        p = np.asarray(pos[i], dtype=np.float64).reshape(-1, 2)
        if t.shape[0] != p.shape[0]:
            raise ProbeRefused("E_TARGETS", f"loc: row {i} has {t.shape[0]} video_tokens and {p.shape[0]} video_pos")
        ok = np.isfinite(p).all(axis=1) & (p >= 0.0).all(axis=1) & (p <= 1.0).all(axis=1)
        if not ok.any():
            raise ProbeRefused("E_POS_MISSING", f"loc: row {i} has no visual token with (y, x) inside [0, 1]")
        s = _unit(t[ok]) @ a[i]
        cell = np.minimum((p[ok] * grid).astype(np.int64), grid - 1)
        flat = cell[:, 0] * grid + cell[:, 1]
        total = np.bincount(flat, weights=s, minlength=grid * grid)
        count = np.bincount(flat, minlength=grid * grid)
        filled = count > 0
        m = np.full(grid * grid, np.nan)
        m[filled] = total[filled] / count[filled]
        m[~filled] = m[filled].min() - 1.0
        maps[i] = m.reshape(grid, grid)
        cov[i] = filled.mean()
    return maps, cov


def boxes_targets(ctx: Any, split: str, rows: np.ndarray) -> list[np.ndarray]:
    """Per row the ``float64 [k, 4]`` boxes ``(x1, y1, x2, y2)`` of the manifest column ``readout_params.boxes_column``
    (``x1|y1|x2|y2`` quadruples ``;``-joined, as written; the rasteriser sanitises). An empty cell is no box."""
    from yavih.readouts.base import _manifest_index

    manifest = ctx.manifest(split)
    column = str(ctx.spec.readout_params["boxes_column"])
    out: list[np.ndarray] = []
    for i in _manifest_index(manifest, rows):
        extra = manifest.extra[i] if manifest.extra is not None else {}
        if column not in extra:
            raise ProbeRefused("E_LABELS", f"row {manifest.clip_id[i]} lacks the boxes cell {column!r}")
        boxes = []
        for quad in (q for q in str(extra[column]).strip().split(";") if q):
            parts = quad.split("|")
            if len(parts) != 4:
                raise ProbeRefused("E_LABELS", f"row {manifest.clip_id[i]}: box {quad!r} is not x1|y1|x2|y2")
            boxes.append([float(x) for x in parts])
        out.append(np.asarray(boxes, dtype=np.float64).reshape(-1, 4))
    return out


def _targets(ctx: Any, split: str, rows: np.ndarray) -> list[Any]:
    hooked = ctx.hook_targets(split, rows)
    return list(hooked) if hooked is not None else boxes_targets(ctx, split, rows)


def _read(ctx: Any, splits: list[str]) -> tuple[np.ndarray, list[Any], list[Any], list[Any], np.ndarray, list[str]]:
    audio, tokens, pos, targets, rows, ids = [], [], [], [], [], []
    for s in splits:
        r = ctx.present_rows(s)
        if len(r) == 0:
            continue
        reader = ctx.reader(s)
        audio.extend(np.asarray(x, dtype=np.float32).reshape(-1) for x in reader.values(KEYS[0], r, codecs=ctx.bench.codecs))
        tokens.extend(reader.values(KEYS[1], r, codecs=ctx.bench.codecs))
        pos.extend(reader.values(KEYS[2], r, codecs=ctx.bench.codecs))
        targets.extend(_targets(ctx, s, r))
        rows.append(r)
        ids.extend(ctx.clip_ids(s, r))
    if not rows:
        raise ProbeRefused("E_SPLIT_MISSING", f"no present rows in {splits}")
    return np.stack(audio), tokens, pos, targets, np.concatenate(rows), ids


def run(ctx: Any) -> dict:
    """Heatmaps and score (the result does not depend on the seed); the ``result.json`` blocks."""
    from yavih.readouts._fit import derangement

    if ctx.roles["folds"]:
        raise ProbeRefused("E_FOLDS", "the loc scorer scores one eval split; splits.folds is not supported")
    rp = dict(ctx.spec.readout_params)
    grid, size = int(rp["grid"]), int(ctx.protocol["gt_size"])
    t0 = time.time()
    audio, tokens, pos, targets, rows, ids = _read(ctx, ctx.roles["eval"])
    train_targets = None
    if ctx.roles["train"] and any(len(ctx.present_rows(s)) for s in ctx.roles["train"]):
        train_targets = _read(ctx, ctx.roles["train"])[3]
    ctx.timing["materialize_s"] = ctx.timing.get("materialize_s", 0.0) + (time.time() - t0)
    t1 = time.time()
    maps, cov = heatmaps(audio, tokens, pos, grid=grid)
    min_cov = float(rp["min_cell_coverage"])
    if float(np.median(cov)) < min_cov:
        raise ProbeRefused("E_POS_MISSING", f"loc: the median fraction of the {grid} x {grid} cells holding a visual token is {np.median(cov):.3f}, "
                           f"below readout_params.min_cell_coverage {min_cov}: the tokens do not cover the frame")
    seed = int(rp["shuffle_seed"])
    perm = derangement(len(rows), seed)
    maps_sh, _ = heatmaps(audio[perm], tokens, pos, grid=grid)
    if train_targets:
        floor = mean_mask_maps(train_targets, len(rows), grid=grid, size=size, leave_one_out=False)
    else:
        floor = mean_mask_maps(targets, len(rows), grid=grid, size=size, leave_one_out=True)
    ctx.timing["score_s"] = ctx.timing.get("score_s", 0.0) + (time.time() - t1)
    split_meta: dict[str, Any] = {"readout": ctx.spec.readout, "n_train": int(len(train_targets or [])), "n_eval": int(len(rows)), "n_val": 0,
                                  "eval_rows": rows, "eval_clip_ids": ids, "chance": dict(ctx.chance), "bootstrap": dict(ctx.bootstrap or {}),
                                  "seed": ctx.seed, "shuffled": {"outputs": maps_sh, "perm": perm, "seed": seed},
                                  "coverage": {"median": float(np.median(cov)), "min": float(cov.min()), "mean": float(cov.mean())}}
    blocks = evaluate_metrics(ctx.metric_specs, outputs=maps, targets=targets, protocol=ctx.protocol, split_meta=split_meta,
                              floor_outputs=floor, ci=ctx.ci, scorer=ctx.scorer(ctx.roles["eval"], rows, targets))
    return {**blocks, "probe": {}, "fit": {"similarity": "cosine", "grid": grid, "cell": "mean over the tokens in the cell", "empty_cell": "min - 1",
                                           "floor": "leave-one-out mean mask" if not train_targets else "mean train mask",
                                           "coverage": split_meta["coverage"], "shuffle_seed": seed},
            "eval_outputs": maps, "eval_rows": rows, "extra_arrays": {"eval_outputs_shuffled": maps_sh, "eval_coverage": cov}, "backend": "ram",
            "timing": {k: ctx.timing[k] for k in ("materialize_s", "score_s") if k in ctx.timing},
            "samples": ([{"role": "eval", "outputs": maps, "targets": [np.asarray(t).tolist() for t in targets], "rows": rows, "clip_ids": ids}]
                        if ctx.log_samples else None)}


def _floor_outputs(train_targets: Any, n_eval: int, *, protocol: dict, label_kind: str, num_classes: int | None = None,
                   grid: int = 14) -> np.ndarray:
    """The mean train mask on the grid; the driver itself uses the leave-one-out eval mean when there is no train
    split."""
    return mean_mask_maps(list(train_targets), int(n_eval), grid=int(grid), size=int(protocol["gt_size"]), leave_one_out=False)


def _validate_spec(readout_params: dict, requires: Any, protocol: dict) -> None:
    g = readout_params.get("grid")
    if isinstance(g, bool) or not isinstance(g, int) or not 2 <= g <= 64:
        raise ValueError(".grid: an int in [2, 64]")
    c = readout_params.get("min_cell_coverage")
    if isinstance(c, bool) or not isinstance(c, int | float) or not 0 < float(c) <= 1:
        raise ValueError(".min_cell_coverage: a fraction in (0, 1]")
    if not isinstance(readout_params.get("boxes_column"), str) or not readout_params["boxes_column"]:
        raise ValueError(".boxes_column: the manifest column holding the boxes")
    if isinstance(readout_params.get("shuffle_seed"), bool) or not isinstance(readout_params.get("shuffle_seed"), int):
        raise ValueError(".shuffle_seed: an int")
    s = protocol.get("gt_size")
    if isinstance(s, bool) or not isinstance(s, int) or s < 8:
        raise ValueError("protocol.gt_size: the raster side, an int >= 8")


def _population(ctx: Any, tag: str) -> dict[str, Any]:
    """The conformance population: a seeded selection of eval rows, their real boxes, fabricated maps that are partly
    informative (the box's cells raised by a stable amount plus noise), a shuffled control and the leave-one-out
    floor, so every metric runs on real target structure without features."""
    from yavih.conformance import _derangement, _values, select_rows

    rp = dict(ctx.spec.readout_params)
    grid, size = int(rp["grid"]), int(ctx.protocol["gt_size"])
    rows, ids, targets = [], [], []
    for s in ctx.roles["eval"]:
        r = select_rows(ctx.bench.manifest_hash, s, ctx.bench.n_rows(s), POPULATION_EVAL, salt="eval")
        targets.extend(_targets(ctx, s, r))
        rows.append(r)
        ids.extend(ctx.clip_ids(s, r))
    rows_all = np.concatenate(rows)
    n = len(rows_all)
    signal = pool_to_grid(rasterize(targets, size), grid)

    def fab(salt: str, sig: np.ndarray) -> np.ndarray:
        return (sig + (_values(f"{tag}:loc:{salt}", (n, grid, grid)) - 0.5) * 0.8).astype(np.float32)

    perm = _derangement(n, f"{tag}:loc:perm")
    meta = {"readout": ctx.spec.readout, "n_train": 0, "n_eval": n, "n_val": 0, "eval_rows": rows_all, "eval_clip_ids": ids, "chance": dict(ctx.chance),
            "bootstrap": dict(ctx.bootstrap or {}), "seed": 0, "shuffled": {"outputs": fab("shuffled", signal[perm]), "perm": perm,
                                                                             "seed": int(rp["shuffle_seed"])},
            "coverage": {"median": 1.0, "min": 1.0, "mean": 1.0}}
    return {"outputs": fab("outputs", signal), "targets": targets, "split_meta": meta,
            "floor_outputs": mean_mask_maps(targets, n, grid=grid, size=size, leave_one_out=True), "eval_splits": list(ctx.roles["eval"]),
            "eval_rows": rows_all}


def make_scorer(name: str, *, module: str, version: int = VERSION) -> Scorer:
    """The folder's stateless localisation scorer, registered under the folder's ``name``."""
    return Scorer(
        name=name, version=version, module=module, kind="stateless", label_kinds=frozenset({"none"}), protocol_keys=frozenset({"gt_size"}),
        param_keys=frozenset({"grid", "min_cell_coverage", "boxes_column", "shuffle_seed"}), requires_t_bins=False, requires_grid=False, dense=False,
        score=run, floor_outputs=_floor_outputs, needs_gpu=False, validate_spec=_validate_spec, output_keys=frozenset(KEYS),
        conformance_population=_population,
    )


# ----------------------------------------------------------------------------- metric specs


def _hit(o, t, p, m) -> np.ndarray:
    return (cious(o, t, p) >= 0.5).astype(np.float64)


def _auc(o, t, p, m) -> float:
    c = cious(o, t, p)
    frac = [float((c >= x).mean()) for x in AUC_THRESHOLDS]
    return float(np.trapezoid(frac, AUC_THRESHOLDS)) if hasattr(np, "trapezoid") else float(np.trapz(frac, AUC_THRESHOLDS))


def _n_empty_gt(o, t, p, m) -> int:
    """Rows whose rasterised ground truth is empty (a degenerate box after ``max(x, 0)``); they score ``0``."""
    size = int(p["gt_size"])
    return int(sum(int((rasterize(t[s:s + _CHUNK], size).reshape(min(_CHUNK, len(t) - s), -1).sum(axis=1) == 0).sum())
                   for s in range(0, len(t), _CHUNK)))


def _floor(o, t, p, m) -> float:
    if o is None:
        raise ValueError("ciou_floor: the driver passes the floor outputs")
    return float(_hit(o, t, p, m).mean())


def _shuffled(o, t, p, m) -> float:
    sh = require_meta(m, "shuffled", "the shuffled-audio control")
    return float(_hit(sh["outputs"], t, p, m).mean())


def _audio_blind(o, t, p, m) -> bool:
    """``not (McNemar z over paired cIoU >= 0.5 successes > 3)``: True when the shuffled-audio arm is not
    significantly worse, i.e. the map is explained by the visual stream alone."""
    sh = require_meta(m, "shuffled", "the shuffled-audio control")
    return not (mcnemar_z(cious(o, t, p) >= 0.5, cious(sh["outputs"], t, p) >= 0.5) > AUDIO_BLIND_SIGMAS)


def make_metrics(name: str) -> dict[str, MetricSpec]:
    """The VGG-SS metrics, the floor and the two controls, for the folder's scorer ``name``."""
    ro, keys = frozenset({name}), frozenset({"gt_size"})
    return {
        "ciou@0.5": MetricSpec("ciou@0.5", lambda o, t, p, m: float(_hit(o, t, p, m).mean()), ro, protocol_keys=keys, per_clip=_hit,
                               ci=frozenset({"binomial", "bootstrap_clip"}), description="the fraction of clips with cIoU >= 0.5 (VGG-SS)"),
        "auc": MetricSpec("auc", _auc, ro, protocol_keys=keys, description="area under the fraction cIoU >= tau, tau in 0:0.05:1 (VGG-SS)"),
        "ciou_mean": MetricSpec("ciou_mean", lambda o, t, p, m: float(cious(o, t, p).mean()), ro, protocol_keys=keys,
                                per_clip=lambda o, t, p, m: cious(o, t, p), ci=frozenset({"bootstrap_clip"}), description="the mean cIoU"),
        "n_empty_gt": MetricSpec("n_empty_gt", _n_empty_gt, ro, protocol_keys=keys, kind="disclosure", higher_is_better=None,
                                 description="rows whose rasterised ground truth is empty (scored 0)"),
        "ciou_floor": MetricSpec("ciou_floor", _floor, ro, protocol_keys=keys, kind="floor", floor_of=("ciou@0.5",),
                                 description="ciou@0.5 of the leave-one-out mean ground-truth mask (the centre / size prior)"),
        "ciou_shuffled": MetricSpec("ciou_shuffled", _shuffled, ro, protocol_keys=keys, kind="control", higher_is_better=None,
                                    description="ciou@0.5 with every row's audio embedding taken from another row"),
        "audio_blind": MetricSpec("audio_blind", _audio_blind, ro, protocol_keys=keys, kind="control", higher_is_better=None,
                                  description="the headline does not beat the shuffled-audio control by 3 McNemar sigma"),
    }
