"""epic100_anticipation_stream: EPIC-KITCHENS-100 action anticipation scored on the stream (scorer ``stream``).

Targets: every narrated action segment of a video, starting at ``s``, is an anchor scored at ``s - protocol.
anticipation_s`` (``tau_a`` = 1 s) from the latest prediction available then (``yavih.readouts.stream``): the
observation ends at ``s - 1`` (the unrestricted causal mode) or the prediction had to be ready by ``s - 1`` (a ``-rt``
real-time reading). A segment that starts within the first second is still scored (from the train prior: no evidence
yet), as the official evaluation scores every validation segment. Label groups ``verb`` (97 classes) and ``noun``
(300), ``ce``.

Metrics, per head: the official ``evaluate_anticipation_json_ek100.py`` (epic-kitchens/C3-Action-Anticipation @
73b8a00) transcribed: class-mean top-5 recall (``mean_topk_recall``: per class present in the ground truth, the fraction
of its segments whose top-5 contains it; their mean) for ``verb``, ``noun`` and ``action``, where action scores are
derived from the verb and noun scores as the evaluator does when a submission carries none (``compute_action_scores``:
the softmax of the top-100 verb scores times that of the top-100 noun scores; here the scores are the heads'
log-probabilities, so the products rank by probability), ``action = verb * 1000 + noun``. Subsets: ``unseen`` (the
segments of participants absent from train, ``unseen_participants.csv``) and ``tail`` (``tail_verbs.csv`` /
``tail_nouns.csv``; an action is tail when its verb or its noun is). Interval: bootstrap over validation videos.
Floors: the train prior on every anchor.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from yavih.metrics.registry import MetricSpec
from yavih.metrics.stream import recording_interval

HERE = Path(__file__).resolve().parent
#: The protocol keys this file reads (``benchmark.yaml`` ``protocol``).
PROTOCOL_KEYS = frozenset({"anticipation_s"})
TOP_K = 5
TOP_N_ACTIONS = 100
_ACTION_VERB_MULTIPLIER = 1000
TASKS = ("verb", "noun", "action")
SUBSETS = ("overall", "unseen", "tail")


@lru_cache(maxsize=None)
def _list(name: str, column: str) -> frozenset:
    with open(HERE / name, newline="", encoding="utf-8") as f:
        return frozenset(r[column] for r in csv.DictReader(f))


def tail_verbs() -> frozenset[int]:
    return frozenset(int(x) for x in _list("tail_verbs.csv", "verb"))


def tail_nouns() -> frozenset[int]:
    return frozenset(int(x) for x in _list("tail_nouns.csv", "noun"))


def unseen_participants() -> frozenset[str]:
    return _list("unseen_participants.csv", "participant_id")


def targets(bench, split, rows):
    man = bench.manifest(split)
    tau = float(bench.spec.protocol["anticipation_s"])
    tv, tn, unseen = tail_verbs(), tail_nouns(), unseen_participants()
    out = []
    for r in np.asarray(rows, dtype=np.int64):
        x = man.extra[int(r)]
        seg = np.asarray(x["segments"], dtype=np.float64).reshape(-1, 3)
        verb, noun = seg[:, 1].astype(np.int64), seg[:, 2].astype(np.int64)
        out.append({"times": seg[:, 0] - tau, "labels": {"verb": verb, "noun": noun},
                    "meta": {"unseen": np.full(len(seg), x["participant_id"] in unseen),
                             "tail_verb": np.isin(verb, list(tv)), "tail_noun": np.isin(noun, list(tn))}})
    return out


# ----------------------------------------------------------------------------- the official evaluator, transcribed


def _softmax(xs):
    max_x = np.max(xs, axis=1).reshape((-1, 1))
    exp_x = np.exp(xs - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def _top_scores(scores, top_n):
    idx = np.argsort(scores)[:, ::-1][:, :top_n]
    return idx, scores[np.arange(len(scores)).reshape(-1, 1), idx]


def action_ranks(verb_scores, noun_scores, top_n=TOP_N_ACTIONS, chunk=256):
    """``compute_action_scores`` then ``_scores_dict_to_ranks``: per segment the action ids ``verb * 1000 + noun`` of
    the top ``top_n`` products, best first (in chunks of segments: the product matrix is ``top_n x top_n`` each)."""
    out = []
    for a in range(0, len(verb_scores), chunk):
        tv, tvs = _top_scores(verb_scores[a:a + chunk], top_n)
        tn, tns = _top_scores(noun_scores[a:a + chunk], top_n)
        m = _softmax(tvs)[:, :, None] * _softmax(tns)[:, None, :]
        n = m.shape[0]
        order = m.reshape(n, -1).argsort(axis=-1)[:, ::-1][:, :top_n]
        vi, ni = np.unravel_index(order, m.shape[1:])
        seg = np.arange(n).reshape(-1, 1)
        out.append(tv[seg, vi] * _ACTION_VERB_MULTIPLIER + tn[seg, ni])
    return np.concatenate(out, axis=0) if out else np.zeros((0, top_n), np.int64)


_HITS: dict = {}


def _hits(outputs, targets):
    """Per task, whether each segment's label is in its top-5 (the evaluator's ranks); cached per output array (the
    bootstrap re-weights segments, the hits stay)."""
    refs = (outputs["verb"], outputs["noun"], targets["verb"], targets["noun"])
    key = tuple(id(x) for x in refs)
    hit = _HITS.get(key)
    if hit is not None and all(a is b for a, b in zip(hit[0], refs, strict=True)):
        return hit[1]
    if len(_HITS) > 8:
        _HITS.clear()
    value = _hits_uncached(outputs, targets)
    _HITS[key] = (refs, value)  # the entry holds its arrays: their ids cannot be reused while it lives
    return value


def _hits_uncached(outputs, targets):
    v = np.log(np.clip(np.asarray(outputs["verb"], dtype=np.float64), 1e-12, None))
    n = np.log(np.clip(np.asarray(outputs["noun"], dtype=np.float64), 1e-12, None))
    yv, yn = np.asarray(targets["verb"]), np.asarray(targets["noun"])
    ya = yv * _ACTION_VERB_MULTIPLIER + yn
    rv, rn = v.argsort(axis=-1)[:, ::-1][:, :TOP_K], n.argsort(axis=-1)[:, ::-1][:, :TOP_K]
    ra = action_ranks(v, n)[:, :TOP_K]
    return ({"verb": (rv == yv[:, None]).any(1), "noun": (rn == yn[:, None]).any(1), "action": (ra == ya[:, None]).any(1)},
            {"verb": yv, "noun": yn, "action": ya})


def mean_topk_recall(hit, labels, w=None):
    """The evaluator's ``mean_topk_recall`` from per-segment top-k hits (integer weights ``w`` repeat segments)."""
    w = np.ones(len(labels)) if w is None else np.asarray(w, dtype=np.float64)
    keep = w > 0
    hit, labels, w = hit[keep], labels[keep], w[keep]
    classes = np.unique(labels)
    if classes.size == 0:
        return float("nan")
    return float(np.mean([np.sum(w[labels == c] * hit[labels == c]) / np.sum(w[labels == c]) for c in classes]))


def _subset(meta, task, subset):
    m = len(np.asarray(meta["unseen"]))
    if subset == "overall":
        return np.ones(m, bool)
    if subset == "unseen":
        return np.asarray(meta["unseen"], bool)
    tv, tn = np.asarray(meta["tail_verb"], bool), np.asarray(meta["tail_noun"], bool)
    return {"verb": tv, "noun": tn, "action": tv | tn}[task]


def _fn(task, subset):
    def fn(outputs, targets, protocol, split_meta, w=None):
        hits, labels = _hits(outputs, targets)
        sel = _subset(split_meta["meta"], task, subset)
        return mean_topk_recall(hits[task][sel], labels[task][sel], None if w is None else np.asarray(w)[sel])

    return fn


def _interval(fn):
    def interval(kind, outputs, targets, protocol, split_meta):
        b = split_meta.get("bootstrap") or {}
        return recording_interval(lambda w: fn(outputs, targets, protocol, split_meta, w=w), split_meta["recording"],
                                  samples=int(b.get("samples", 2000)), seed=int(b.get("seed", 0)))

    return interval


def _name(task, subset, head):
    return f"mt5r_{task}" + ("" if subset == "overall" else f"_{subset}") + f"_{head}"


METRICS: dict[str, MetricSpec] = {}
for _head in ("linear", "gru"):
    for _task in TASKS:
        for _sub in SUBSETS:
            _f = _fn(_task, _sub)
            _n = _name(_task, _sub, _head)
            _ci = _sub == "overall"
            METRICS[_n] = MetricSpec(_n, _f, frozenset({"stream"}), ci=frozenset({"bootstrap_clip"}) if _ci else frozenset(),
                                     interval=_interval(_f) if _ci else None,
                                     description=f"class-mean top-5 recall, {_task}, {_sub} ({_head} head; EK100 evaluator)")
for _task in TASKS:
    _n = f"mt5r_{_task}_floor"
    METRICS[_n] = MetricSpec(_n, _fn(_task, "overall"), frozenset({"stream"}), kind="floor",
                             floor_of=(_name(_task, "overall", "linear"), _name(_task, "overall", "gru")),
                             description=f"class-mean top-5 recall of the train prior, {_task}")
