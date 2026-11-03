"""ave_causal: causal per-second targets and metrics (scorer ``stream``).

Targets: every one-second segment ``[s, s + 1)`` of a video, ``s = 0 ... 9``, is scored at its end ``t = s + 1`` from
the prediction available by then (``yavih.readouts.stream``); its label is the event class or ``background`` (28),
from the manifest's ``seconds`` column. Metrics, per head (``_linear`` / ``_gru``):

* ``sec_acc`` — accuracy of the arg-max over all ten segments of every eval video, background included (AVE's
  ``compute_acc`` on causal predictions);
* ``event_map`` — mean over the event classes with a positive segment of the one-vs-rest average precision over
  all segments (background excluded; the online-detection convention, ``yavih.metrics.stream.perframe_map``).

Floors: the train prior on every segment (``sec_acc_floor``, ``event_map_floor``). Intervals: 95 % percentile
bootstrap over eval VIDEOS (``metrics.bootstrap``), every segment of a drawn video kept.
"""

from __future__ import annotations

import numpy as np

from yavih.metrics.registry import MetricSpec
from yavih.metrics.stream import perframe_map, recording_interval, segment_accuracy

#: The protocol keys this file reads (``benchmark.yaml`` ``protocol``).
PROTOCOL_KEYS = frozenset({"segment_seconds", "num_segments", "background_index"})


def targets(bench, split, rows):
    """Per row: the ten segment-end times and their labels (``{"event": int [10]}``)."""
    man = bench.manifest(split)
    proto = bench.spec.protocol
    seg, n = float(proto["segment_seconds"]), int(proto["num_segments"])
    times = (np.arange(n, dtype=np.float64) + 1.0) * seg
    out = []
    for r in np.asarray(rows, dtype=np.int64):
        lab = np.asarray([int(x) for x in str(man.extra[int(r)]["seconds"]).split(";")], dtype=np.int64)
        if lab.size != n:
            raise ValueError(f"row {int(r)}: {lab.size} segment labels, expected {n}")
        out.append({"times": times, "labels": {"event": lab}})
    return out


def _onehot(labels, c):
    y = np.zeros((len(labels), c), dtype=np.int8)
    ok = labels >= 0
    y[np.flatnonzero(ok), labels[ok]] = 1
    return y


def _sec_acc(outputs, targets, protocol, split_meta, w=None):
    return segment_accuracy(outputs["event"], targets["event"], w=w)


def _event_map(outputs, targets, protocol, split_meta, w=None):
    s = np.asarray(outputs["event"])
    return perframe_map(s, _onehot(np.asarray(targets["event"]), s.shape[1]), ignore=(int(protocol["background_index"]),), w=w)


def _interval(fn):
    def interval(kind, outputs, targets, protocol, split_meta):
        b = split_meta.get("bootstrap") or {}
        return recording_interval(lambda w: fn(outputs, targets, protocol, split_meta, w=w), split_meta["recording"],
                                  samples=int(b.get("samples", 2000)), seed=int(b.get("seed", 0)))

    return interval


def _metric(name, fn, description):
    return MetricSpec(name, fn, frozenset({"stream"}), protocol_keys=frozenset({"background_index"}), ci=frozenset({"bootstrap_clip"}),
                      interval=_interval(fn), description=description)


METRICS = {
    "sec_acc_linear": _metric("sec_acc_linear", _sec_acc, "segment accuracy over all ten seconds, background included (linear head)"),
    "sec_acc_gru": _metric("sec_acc_gru", _sec_acc, "segment accuracy over all ten seconds, background included (causal GRU head)"),
    "event_map_linear": _metric("event_map_linear", _event_map, "mean one-vs-rest AP over the event classes with a positive segment (linear head)"),
    "event_map_gru": _metric("event_map_gru", _event_map, "mean one-vs-rest AP over the event classes with a positive segment (causal GRU head)"),
    "sec_acc_floor": MetricSpec("sec_acc_floor", _sec_acc, frozenset({"stream"}), kind="floor", floor_of=("sec_acc_linear", "sec_acc_gru"),
                                description="segment accuracy of the train prior"),
    "event_map_floor": MetricSpec("event_map_floor", _event_map, frozenset({"stream"}), protocol_keys=frozenset({"background_index"}),
                                  kind="floor", floor_of=("event_map_linear", "event_map_gru"), description="event mAP of the train prior"),
}
