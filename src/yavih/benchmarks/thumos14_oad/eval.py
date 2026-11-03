"""thumos14_oad: THUMOS'14 online action detection targets and the per-frame mAP (scorer ``stream``).

The evaluation grid is the online-detection convention of LSTR / TeSTra: chunks of a quarter second (6 frames at the
24 fps the reference features are extracted at), ``[k / 4, (k + 1) / 4)``, scored at their end from the latest
prediction available then (``yavih.readouts.stream``). A chunk's ground truth is the multi-hot of the classes whose
annotated intervals cover its centre, over 22 columns in the reference order: background (0, no action), the 20
detection classes in ``Class Index_Detection.txt`` order (1 ... 20) and ``Ambiguous`` (21).

Training targets (``action``, ``ce`` over 21 classes): background or the lowest-index action at the chunk centre;
an ``Ambiguous`` chunk is not trained on (``-1``).

``frame_map`` (per head): the reference ``perframe_average_precision`` with THUMOS post-processing: chunks whose ground
truth is ``Ambiguous`` removed, AP per class over every remaining chunk of every test video (columns 0 and 21
skipped), classes without a positive left out of the mean; no smoothing, no class switch. Interval: bootstrap over
test videos. Floor ``frame_map_floor``: the train class frequencies on every chunk.
"""

from __future__ import annotations

import math

import numpy as np

from yavih.metrics.registry import MetricSpec
from yavih.metrics.stream import perframe_map, recording_interval

#: The protocol keys this file reads (``benchmark.yaml`` ``protocol``).
PROTOCOL_KEYS = frozenset({"chunk_fps", "ambiguous_index"})
N_ACTIONS = 20


def chunks_of(duration: float, events, *, fps: float = 4.0, ambiguous: int = 21):
    """``(times [m], action int [m], gt uint8 [m, 22])`` of one video: the chunks that lie inside it."""
    n = int(math.floor(duration * fps + 1e-9))
    centre = (np.arange(n) + 0.5) / fps
    gt = np.zeros((n, ambiguous + 1), np.uint8)
    for s, e, c in events:
        gt[(centre >= float(s)) & (centre < float(e)), int(c)] = 1
    acts = gt[:, 1:ambiguous]
    gt[:, 0] = (acts.sum(axis=1) == 0) & (gt[:, ambiguous] == 0)
    action = np.where(acts.any(axis=1), acts.argmax(axis=1) + 1, 0).astype(np.int64)
    action[gt[:, ambiguous] == 1] = -1
    return (np.arange(n) + 1.0) / fps, action, gt


def targets(bench, split, rows):
    man = bench.manifest(split)
    proto = bench.spec.protocol
    out = []
    for r in np.asarray(rows, dtype=np.int64):
        x = man.extra[int(r)]
        times, action, gt = chunks_of(float(x["duration_s"]), x["events"], fps=float(proto["chunk_fps"]),
                                      ambiguous=int(proto["ambiguous_index"]))
        out.append({"times": times, "labels": {"action": action}, "meta": {"gt": gt}})
    return out


def _frame_map(outputs, targets, protocol, split_meta, w=None):
    amb = int(protocol["ambiguous_index"])
    gt = np.asarray(split_meta["meta"]["gt"])
    s = np.asarray(outputs["action"])
    scores = np.zeros((s.shape[0], amb + 1), np.float32)
    scores[:, : s.shape[1]] = s
    return perframe_map(scores, gt, ignore=(0, amb), valid=gt[:, amb] != 1, w=w)


def _interval(kind, outputs, targets, protocol, split_meta):
    b = split_meta.get("bootstrap") or {}
    return recording_interval(lambda w: _frame_map(outputs, targets, protocol, split_meta, w=w), split_meta["recording"],
                              samples=int(b.get("samples", 2000)), seed=int(b.get("seed", 0)))


def _metric(head):
    return MetricSpec(f"frame_map_{head}", _frame_map, frozenset({"stream"}), protocol_keys=frozenset({"ambiguous_index"}),
                      ci=frozenset({"bootstrap_clip"}), interval=_interval,
                      description=f"per-frame mAP, THUMOS post-processing, LSTR reference ({head} head)")


METRICS = {
    "frame_map_linear": _metric("linear"),
    "frame_map_gru": _metric("gru"),
    "frame_map_floor": MetricSpec("frame_map_floor", _frame_map, frozenset({"stream"}), protocol_keys=frozenset({"ambiguous_index"}),
                                  kind="floor", floor_of=("frame_map_linear", "frame_map_gru"),
                                  description="per-frame mAP of the train class frequencies"),
}
