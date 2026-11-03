"""unav100_causal: dense one-second bins from UnAV-100's event intervals, and their metrics (scorer ``stream``).

Targets (``unav100_causal_recognition_v1``): a video of recording duration ``D`` is cut into one-second bins
``[b, min(b + 1, D))``; a class is positive in a bin when the union of its annotated intervals covers at least half of
the bin's actual length (the last bin is clipped to ``D``). Each bin is scored at its end from the prediction available
then (``yavih.readouts.stream``): bin recognition, not an instantaneous label at the bin's start. Multilabel: concurrent
events stay concurrent. Per bin and class the hook also records the merged same-class interval that overlaps the bin
most (``ev_start`` / ``ev_end``, NaN where none), which the slices and the onset diagnostic read.

Metrics, per head (``_linear`` / ``_gru``):

* ``map`` — per-class AP over every scored bin, macro-averaged over the classes with a positive test bin (all 100);
  intervals by bootstrap over test videos;
* ``map_short`` / ``map_medium`` / ``map_long`` — the same with the positives restricted to bins of merged events
  shorter than 3 s / 3 to 10 s / 10 s and longer (negatives unchanged): short events are diluted by one-second bins;
* ``onset_recall``, ``onset_delay_s``, ``false_alerts_per_h`` — the onset diagnostic at one operating point: a global
  threshold chosen on the scorer's held-out train recordings (``split_meta["val"]``) as the lowest that keeps their
  false alerts per hour at or below ``protocol.onset.false_alerts_per_hour``. An alert is a rising edge of a class's
  held score through the threshold at a bin end; it detects the first unmatched merged event of its class active at
  that time (``start <= t <= end + tolerance_s``), with delay ``t - start``; another alert within an already detected
  event is a duplicate (neither a detection nor a false alert); any other alert is false. Recall is over every merged
  test event; the delay is the median over detected events.

Floor: ``bin_map_floor``, the train prevalence of each class on every bin.
"""

from __future__ import annotations

import math

import numpy as np

from yavih.metrics.registry import MetricSpec
from yavih.metrics.stream import recording_interval, weighted_ap

#: The protocol keys this file reads (``benchmark.yaml`` ``protocol``).
PROTOCOL_KEYS = frozenset({"bin_seconds", "positive_coverage", "onset", "length_slices"})
N_CLASSES = 100


def _merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def bins_of(duration: float, events, *, bin_s: float = 1.0, coverage: float = 0.5, n_classes: int = N_CLASSES):
    """``(times [m], labels float32 [m, C], ev_start [m, C], ev_end [m, C])`` of one video."""
    n = max(1, int(math.ceil(duration / bin_s - 1e-9)))
    lo = np.arange(n) * bin_s
    hi = np.minimum(lo + bin_s, duration)
    labels = np.zeros((n, n_classes), np.float32)
    ev_s = np.full((n, n_classes), np.nan, np.float32)
    ev_e = np.full((n, n_classes), np.nan, np.float32)
    by_class: dict[int, list] = {}
    for s, e, c in events:
        by_class.setdefault(int(c), []).append((float(s), float(e)))
    for c, iv in by_class.items():
        merged = _merge(iv)
        best = np.zeros(n)
        for s, e in merged:
            ov = np.clip(np.minimum(hi, e) - np.maximum(lo, s), 0.0, None)
            labels[:, c] += ov.astype(np.float32)
            take = ov > best
            ev_s[take, c], ev_e[take, c] = s, e
            best = np.where(take, ov, best)
        labels[:, c] = (labels[:, c] >= coverage * (hi - lo) - 1e-9).astype(np.float32)
    return hi.astype(np.float64), labels, ev_s, ev_e


def targets(bench, split, rows):
    man = bench.manifest(split)
    proto = bench.spec.protocol
    out = []
    for r in np.asarray(rows, dtype=np.int64):
        x = man.extra[int(r)]
        times, labels, s, e = bins_of(float(x["duration_s"]), x["events"], bin_s=float(proto["bin_seconds"]),
                                      coverage=float(proto["positive_coverage"]))
        out.append({"times": times, "labels": {"event": labels}, "meta": {"ev_start": s, "ev_end": e}})
    return out


# ----------------------------------------------------------------------------- AP metrics


def _ap_mean(scores, labels, w=None, mask=None):
    """Macro AP over the classes with a positive (``mask`` restricts the positives, negatives kept)."""
    aps = []
    for c in range(scores.shape[1]):
        y = labels[:, c] > 0.5
        keep = np.ones_like(y) if mask is None else (~y | (y & mask[:, c]))
        pos = y & keep
        if not pos.any() or (w is not None and not (np.asarray(w)[pos] > 0).any()):
            continue
        aps.append(weighted_ap(scores[keep, c], y[keep], None if w is None else np.asarray(w)[keep]))
    return float(np.mean(aps)) if aps else float("nan")


def _map(outputs, targets, protocol, split_meta, w=None):
    return _ap_mean(np.asarray(outputs["event"]), np.asarray(targets["event"]), w)


def _slice(bucket):
    def fn(outputs, targets, protocol, split_meta, w=None):
        lo, hi = protocol["length_slices"][bucket]
        meta = split_meta["meta"]
        length = np.asarray(meta["ev_end"]) - np.asarray(meta["ev_start"])
        mask = (length >= float(lo)) & (length < (math.inf if hi is None else float(hi)))
        return _ap_mean(np.asarray(outputs["event"]), np.asarray(targets["event"]), w, mask)

    return fn


def _interval(fn):
    def interval(kind, outputs, targets, protocol, split_meta):
        b = split_meta.get("bootstrap") or {}
        return recording_interval(lambda w: fn(outputs, targets, protocol, split_meta, w=w), split_meta["recording"],
                                  samples=int(b.get("samples", 2000)), seed=int(b.get("seed", 0)))

    return interval


# ----------------------------------------------------------------------------- onset diagnostic


def _events_of(recording, meta):
    """``{(rec, class): [(start, end), ...]}``: the merged events that overlap at least one bin."""
    rec = np.asarray(recording)
    s, e = np.asarray(meta["ev_start"]), np.asarray(meta["ev_end"])
    out: dict[tuple[int, int], set] = {}
    for i, c in zip(*np.nonzero(np.isfinite(s)), strict=True):
        out.setdefault((int(rec[i]), int(c)), set()).add((float(s[i, c]), float(e[i, c])))
    return {k: sorted(v) for k, v in out.items()}


def onset(scores, recording, time, meta, theta, tolerance_s, events=None):
    """``(recall, median delay s, false alerts per hour, n events)`` at threshold ``theta`` (``events``: a precomputed
    :func:`_events_of`)."""
    scores = np.asarray(scores)
    rec = np.asarray(recording)
    t = np.asarray(time, dtype=np.float64)
    events = _events_of(rec, meta) if events is None else events
    n_events = sum(len(v) for v in events.values())
    hours = sum(float(t[rec == r].max()) for r in np.unique(rec)) / 3600.0
    above = scores >= theta
    prev = np.zeros_like(above)
    same = np.r_[False, rec[1:] == rec[:-1]]
    prev[1:] = above[:-1] & same[1:, None]
    rising = above & ~prev
    matched: set = set()
    delays, false = [], 0
    for i, c in zip(*np.nonzero(rising), strict=True):
        key = (int(rec[i]), int(c))
        ti = float(t[i])
        active = [ev for ev in events.get(key, []) if ev[0] <= ti <= ev[1] + tolerance_s]
        fresh = [ev for ev in active if (key, ev) not in matched]
        if fresh:
            ev = fresh[0]
            matched.add((key, ev))
            delays.append(ti - ev[0])
        elif not active:
            false += 1
    recall = len(matched) / n_events if n_events else float("nan")
    return recall, (float(np.median(delays)) if delays else float("nan")), false / hours if hours else float("nan"), n_events


def operating_point(val, protocol):
    """The lowest threshold of a 200-quantile grid of the validation scores whose false alerts per hour on the
    validation recordings stay at or below the target (bisection: false alerts fall as the threshold rises)."""
    o = protocol["onset"]
    s = np.asarray(val["outputs"]["event"])
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.99999, 200)))
    events = _events_of(val["recording"], val["meta"])

    def ok(theta):
        _, _, fa, _ = onset(s, val["recording"], val["time"], val["meta"], float(theta), float(o["tolerance_s"]), events)
        return fa <= float(o["false_alerts_per_hour"])

    lo, hi = 0, len(grid) - 1
    if not ok(grid[hi]):
        return float(grid[hi])
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(grid[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(grid[lo])


def _onset_value(which):
    def fn(outputs, targets, protocol, split_meta):
        val = split_meta.get("val")
        if not val:
            return None
        theta = operating_point(val, protocol)
        r = onset(outputs["event"], split_meta["recording"], split_meta["time"], split_meta["meta"], theta,
                  float(protocol["onset"]["tolerance_s"]))
        return {"recall": r[0], "delay": r[1], "fa": r[2]}[which]

    return fn


# ----------------------------------------------------------------------------- the registry


def _metrics(head):
    lk = frozenset({"length_slices"})
    return {
        f"map_{head}": MetricSpec(f"map_{head}", _map, frozenset({"stream"}), ci=frozenset({"bootstrap_clip"}), interval=_interval(_map),
                                  description=f"macro per-class AP over the one-second bins ({head} head)"),
        **{f"map_{b}_{head}": MetricSpec(f"map_{b}_{head}", _slice(b), frozenset({"stream"}), protocol_keys=lk,
                                         description=f"macro AP, positives of {b} merged events only ({head} head)")
           for b in ("short", "medium", "long")},
        f"onset_recall_{head}": MetricSpec(f"onset_recall_{head}", _onset_value("recall"), frozenset({"stream"}),
                                           protocol_keys=frozenset({"onset"}), description=f"events detected at the operating point ({head} head)"),
        f"onset_delay_s_{head}": MetricSpec(f"onset_delay_s_{head}", _onset_value("delay"), frozenset({"stream"}),
                                            protocol_keys=frozenset({"onset"}), higher_is_better=False,
                                            description=f"median seconds from event start to its detection ({head} head)"),
        f"false_alerts_per_h_{head}": MetricSpec(f"false_alerts_per_h_{head}", _onset_value("fa"), frozenset({"stream"}),
                                                 protocol_keys=frozenset({"onset"}), higher_is_better=False,
                                                 description=f"false alerts per hour of test video at the operating point ({head} head)"),
    }


METRICS = {
    **_metrics("linear"),
    **_metrics("gru"),
    "bin_map_floor": MetricSpec("bin_map_floor", _map, frozenset({"stream"}), kind="floor", floor_of=("map_linear", "map_gru"),
                            description="macro AP of the train prevalence"),
}
