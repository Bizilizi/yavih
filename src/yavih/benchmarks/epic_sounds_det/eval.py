"""``epic_sounds_det``: the folder's own scorer and the EPIC-SOUNDS audio-based interaction detection mAP.

The scorer (head, loss, targets, decoding, the floor, the conformance population) is ``detection.py``, next to this
file and byte-identical in every folder that detects timed events (a candidate for the core); this
file names the event encoding (``class:start_ms:end_ms``, the EPIC-SOUNDS annotation in milliseconds) and scores the
decoded segments with the official scorer, transcribed below.

The official scorer is ``EvaluationCode/audio_based_interaction_detection.py`` of
https://github.com/epic-kitchens/C10-epic-sounds-detection (commit ``196e7cf2c4f17946279fb57d9ff24516b7403344``,
sha256 of the file in ``SCORER_SHA256``), itself adapted from ActivityNet's ``eval_detection.py``. It is not an
installable package (no ``setup.py`` / ``pyproject.toml``), it pins numpy 1.19 / pandas 1.1 / joblib 0.17 and it
uses ``np.float``, which numpy >= 1.24 removed, so it cannot be reached through ``yavih.external.require``. This
file is a line-by-line numpy transcription of its scoring path (``ANETdetection`` with
``compute_average_precision_detection``, ``segment_iou`` and ``interpolated_prec_rec``), without pandas:

* ground truth: every labelled segment of the eval recordings with ``class != background`` (the challenge v0.1
  rule: ``groundtruth_df[groundtruth_df["class"] != "background"]``), times in seconds;
* predictions whose class does not occur in that ground truth are dropped;
* per class present in the ground truth, AP at each tIoU threshold (``np.linspace(0.1, 0.5, 5)``): predictions of
  the class sorted by ``score.argsort()[::-1]`` (numpy's default quicksort, reversed: the same tie order), each
  matched greedily to the unmatched ground-truth segment of the same recording with the highest tIoU
  (``tiou.argsort()[::-1]``), interpolated VOC-2011 precision-recall area; a class without predictions scores 0;
* ``mAP@k`` = mean over classes, ``map_avg`` = mean of the five, reported in percent like the official script.

The transcription is pinned by the folder test against the AP of the C10 baseline file
``actionformer_baseline_validation.json``. Module scope imports stdlib, numpy and yavih core only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from yavih.metrics.registry import MetricSpec


def _sibling(name: str):
    """A file next to this one, imported by path (keyed on its content, like yavih loads eval.py)."""
    path = Path(__file__).resolve().with_name(name)
    data = path.read_bytes()
    mod_name = f"yavih_folder_{path.parent.name}_{path.stem}_{hashlib.sha256(data).hexdigest()[:12]}"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
    return sys.modules[mod_name]


detection = _sibling("detection.py")
EVENTS_FORMAT = "class:start_ms:end_ms"

SCORER_REPO = "epic-kitchens/C10-epic-sounds-detection"
SCORER_COMMIT = "196e7cf2c4f17946279fb57d9ff24516b7403344"
SCORER_FILE = "EvaluationCode/audio_based_interaction_detection.py"
SCORER_SHA256 = "7c2c2509bc162e2067993def4acb392aa20077fd618a97cee2b9ecc0f5ad4338"

PROTOCOL_KEYS = frozenset({"tiou_thresholds", "eval_excluded_classes", "events_column"})
SCORER = detection.make_scorer("epic_sounds_det", module=__name__, events_format=EVENTS_FORMAT)


# ----------------------------------------------------------------------------- the transcription


def segment_iou(target_segment: np.ndarray, candidate_segments: np.ndarray) -> np.ndarray:
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    inter = (tt2 - tt1).clip(0)
    union = (candidate_segments[:, 1] - candidate_segments[:, 0]) + (target_segment[1] - target_segment[0]) - inter
    return inter.astype(float) / union


def interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray) -> float:
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def average_precision_detection(gt_video: np.ndarray, gt_seg: np.ndarray, pr_video: np.ndarray, pr_seg: np.ndarray,
                                pr_score: np.ndarray, tiou_thresholds: np.ndarray) -> np.ndarray:
    """``compute_average_precision_detection`` for one class: ``gt_*`` in ground-truth order, ``pr_*`` in
    submission order (both already restricted to the class)."""
    ap = np.zeros(len(tiou_thresholds))
    if len(pr_score) == 0:
        return ap
    npos = float(len(gt_video))
    lock_gt = np.ones((len(tiou_thresholds), len(gt_video))) * -1
    sort_idx = pr_score.argsort()[::-1]
    pr_video, pr_seg = pr_video[sort_idx], pr_seg[sort_idx]
    tp = np.zeros((len(tiou_thresholds), len(sort_idx)))
    fp = np.zeros((len(tiou_thresholds), len(sort_idx)))
    by_video: dict[str, np.ndarray] = {}
    for j, v in enumerate(gt_video):
        by_video.setdefault(v, []).append(j)  # type: ignore[arg-type]
    by_video = {v: np.asarray(j, dtype=np.int64) for v, j in by_video.items()}
    for idx in range(len(sort_idx)):
        gidx = by_video.get(pr_video[idx])
        if gidx is None:
            fp[:, idx] = 1
            continue
        tiou_arr = segment_iou(pr_seg[idx], gt_seg[gidx])
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, gidx[jdx]] >= 0:
                    continue
                tp[tidx, idx] = 1
                lock_gt[tidx, gidx[jdx]] = idx
                break
            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1
    tp_c = np.cumsum(tp, axis=1).astype(float)
    fp_c = np.cumsum(fp, axis=1).astype(float)
    rec = tp_c / npos
    prec = tp_c / (tp_c + fp_c)
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(prec[tidx, :], rec[tidx, :])
    return ap


def official_map(gt: Sequence[tuple[str, int, float, float]], pred: Sequence[tuple[str, int, float, float, float]],
                 tiou_thresholds: Sequence[float], excluded: Sequence[int] = (13,)) -> tuple[np.ndarray, float]:
    """``ANETdetection(...).evaluate()``: ``gt`` = (video, class, t_start, t_end) in annotation-file order,
    ``pred`` = (video, class, t_start, t_end, score) in submission order. Returns (mAP per threshold, average)
    as fractions (the script prints them x 100)."""
    thr = np.asarray(tiou_thresholds, dtype=float)
    gt = [g for g in gt if int(g[1]) not in set(excluded)]
    labels = sorted({int(g[1]) for g in gt})
    pred = [p for p in pred if int(p[1]) in set(labels)]
    ap = np.zeros((len(thr), len(labels)))
    g_lab = np.asarray([int(g[1]) for g in gt], dtype=np.int64)
    g_vid = np.asarray([g[0] for g in gt], dtype=object)
    g_seg = np.asarray([[g[2], g[3]] for g in gt], dtype=float).reshape(-1, 2)
    p_lab = np.asarray([int(p[1]) for p in pred], dtype=np.int64)
    p_vid = np.asarray([p[0] for p in pred], dtype=object)
    p_seg = np.asarray([[p[2], p[3]] for p in pred], dtype=float).reshape(-1, 2)
    p_sc = np.asarray([p[4] for p in pred], dtype=float)
    for ci, c in enumerate(labels):
        gm, pm = g_lab == c, p_lab == c
        ap[:, ci] = average_precision_detection(g_vid[gm], g_seg[gm], p_vid[pm], p_seg[pm], p_sc[pm], thr)
    m = ap.mean(axis=1)
    return m, float(m.mean())


# ----------------------------------------------------------------------------- the yavih metric interface
#
# outputs: the scorer's raw eval outputs [n, T, C + 2], decoded by detection.py into per-row [k, 4] =
# (class, t_start_s, t_end_s, score); targets: one float array [m, 3] per eval row = (class, t_start_s, t_end_s) in
# manifest (= upstream) order; split_meta["eval_clip_ids"]: the recording ids of the eval rows.


def _flatten(outputs, targets, split_meta):
    if outputs is None:  # the floor: the label-only constant output from the TRAIN targets
        outputs = detection.floor_outputs(split_meta["train_targets"], len(list(targets)))
    ids = list(split_meta["eval_clip_ids"])
    dets = detection._detections(outputs, split_meta)
    gt = [(v, int(c), float(s), float(e)) for v, t in zip(ids, targets, strict=True) for c, s, e in np.asarray(t).reshape(-1, 3)]
    pr = [(v, int(c), float(s), float(e), float(sc)) for v, o in zip(ids, dets, strict=True) for c, s, e, sc in np.asarray(o).reshape(-1, 4)]
    return gt, pr


def _maps(outputs, targets, protocol, split_meta):
    gt, pr = _flatten(outputs, targets, split_meta)
    m, avg = official_map(gt, pr, protocol["tiou_thresholds"], protocol["eval_excluded_classes"])
    return m * 100.0, avg * 100.0


def map_avg(outputs, targets, protocol, split_meta) -> float:
    return float(_maps(outputs, targets, protocol, split_meta)[1])


def map_at(outputs, targets, protocol, split_meta, *, k: int) -> float:
    return float(_maps(outputs, targets, protocol, split_meta)[0][k])


def map_avg_paper(outputs, targets, protocol, split_meta) -> float:
    """The TPAMI 2025 Table V convention: background segments KEPT in the ground truth (44 classes)."""
    gt, pr = _flatten(outputs, targets, split_meta)
    return float(official_map(gt, pr, protocol["tiou_thresholds"], excluded=())[1] * 100.0)


_RO = frozenset({"epic_sounds_det"})
METRICS: dict[str, MetricSpec] = {
    "map_avg": MetricSpec(
        name="map_avg", fn=map_avg, readouts=_RO, protocol_keys=frozenset({"tiou_thresholds", "eval_excluded_classes"}),
        description="interaction_map_avg of C10 audio_based_interaction_detection.py (mean AP over classes and tIoU 0.1:0.1:0.5, %, background excluded)"),
    **{f"map@{t}": MetricSpec(
        name=f"map@{t}", fn=(lambda k: (lambda o, tg, p, sm: map_at(o, tg, p, sm, k=k)))(i), readouts=_RO,
        protocol_keys=frozenset({"tiou_thresholds", "eval_excluded_classes"}), description=f"interaction_map_at_{i + 1:02d} of the C10 scorer (tIoU {t}, %)")
       for i, t in enumerate(("0.1", "0.2", "0.3", "0.4", "0.5"))},
    "map_avg_paper": MetricSpec(
        name="map_avg_paper", fn=map_avg_paper, readouts=_RO, protocol_keys=frozenset({"tiou_thresholds"}),
        description="map_avg with background kept in the ground truth (Huh et al., TPAMI 2025, Table V convention)"),
    "map_avg_floor": MetricSpec(
        name="map_avg_floor", fn=map_avg, readouts=_RO, protocol_keys=frozenset({"tiou_thresholds", "eval_excluded_classes"}), kind="floor",
        floor_of=("map_avg",), description="map_avg of the label-only constant output (detection.py floor_outputs)"),
}

__all__ = ["SCORER", "METRICS", "PROTOCOL_KEYS", "EVENTS_FORMAT", "official_map", "average_precision_detection", "segment_iou",
           "interpolated_prec_rec", "detection"]
