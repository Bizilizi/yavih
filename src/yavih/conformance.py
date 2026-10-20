"""Conformance fixtures: the numbers a benchmark folder's scoring path produces on a fixed, generated population,
recorded next to the folder at its ``protocol_version`` and checked against tolerance.

Identity is the declaration (``yavih-protocol-5``, :mod:`yavih.hashing`): no source file enters ``protocol_hash``.
What pins the behaviour of the code behind a declared version — the folder's ``probe.py`` and the readouts and
metrics of yavih alike — is this fixture. Every folder carries ``conformance/expected.json``; a change that moves
its numbers beyond tolerance is ``E_CONFORMANCE_DRIFT`` until ``protocol_version`` is bumped (or the readout's
``VERSION``, which moves every folder on it) and the fixture is re-recorded.

The fixture holds no inputs. At check time a stable generator (sha256 in counter mode, never numpy's distribution
methods, which NEP 19 leaves free to change between releases) draws a small population of the folder's REAL
manifest rows per role and fabricates probe outputs (and, for ``grid``, dense targets) in the readout's output
shape; the folder's whole scoring path then runs on them: the readout's target construction, the metrics registry
with the folder's protocol values, the folder's ``probe.py`` ``METRICS`` / ``score`` and the pinned external
package where present. Only ``expected.json`` (a few KB) is stored. Population sizes keep ``1 / n`` above the
default tolerance, so a single row scored differently still fails while an interpolation-level difference passes.

Tolerance: absolute, per metric on its reported scale, ``metrics.tolerance`` in ``benchmark.yaml`` (default
:data:`DEFAULT_TOLERANCE`); not hashed (it changes no number, it bounds what "same protocol" accepts), recorded in
``expected.json`` and in every registry entry. Intervals (``ci95``) are recorded, never compared.

stdlib + numpy + yavih core: importable without torch (the scoring path is torch-free).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from yavih.versions import __version__

__all__ = [
    "CONFORMANCE_SCHEMA",
    "GENERATOR_VERSION",
    "DEFAULT_TOLERANCE",
    "EXPECTED_FILE",
    "POPULATION",
    "ConformanceError",
    "Finding",
    "ConformanceReport",
    "stream",
    "select_rows",
    "tolerances",
    "compute",
    "check",
    "record",
    "expected_path",
    "read_expected",
]

#: The ``expected.json`` schema tag.
CONFORMANCE_SCHEMA = "yavih-conformance-1"
#: Bumped when the population draw or the fabricated outputs change (every folder then re-records at its version).
GENERATOR_VERSION = 1
#: Absolute tolerance per metric on its reported scale when ``metrics.tolerance`` names none: a tenth of a point.
DEFAULT_TOLERANCE = 0.001
EXPECTED_FILE = "conformance/expected.json"
#: Population per role and readout: ``1 / n`` stays above :data:`DEFAULT_TOLERANCE`.
POPULATION: dict[str, dict[str, int]] = {
    "clf": {"eval": 256, "train": 256, "floor": 64, "fold": 64},
    "seq": {"eval": 64, "train": 64},
    "grid": {"eval": 16, "train": 16},
    "box": {"eval": 128, "train": 128},
}

E_MISSING = "E_CONFORMANCE_MISSING"
E_STALE = "E_CONFORMANCE_STALE"
E_DRIFT = "E_CONFORMANCE_DRIFT"
E_SCHEMA = "E_CONFORMANCE_SCHEMA"
E_UNSUPPORTED = "E_CONFORMANCE_UNSUPPORTED"
W_SKIPPED = "W_CONFORMANCE_SKIPPED"


class ConformanceError(Exception):
    """A fixture cannot be computed or recorded (``code``, ``message``)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Finding:
    severity: str  # error | warning
    code: str
    message: str


@dataclass
class ConformanceReport:
    """``check``'s outcome for one folder: ``ok`` iff no error finding; ``skipped`` when the scoring path could
    not run here (an absent external package) and nothing was compared."""

    name: str
    folder: str
    protocol_version: int
    expected_version: int | None
    ok: bool
    skipped: bool
    findings: list[Finding] = field(default_factory=list)
    compared: int = 0
    drifted: list[dict[str, Any]] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "folder": self.folder, "protocol_version": self.protocol_version,
            "expected_version": self.expected_version, "ok": self.ok, "skipped": self.skipped, "compared": self.compared,
            "drifted": list(self.drifted),
            "errors": [{"code": f.code, "message": f.message} for f in self.errors],
            "warnings": [{"code": f.code, "message": f.message} for f in self.warnings],
        }


# --------------------------------------------------------------------------- the stable generator


def stream(tag: str, n: int) -> np.ndarray:
    """``float64 [n]`` uniform in ``[0, 1)`` from sha256 in counter mode over ``tag``: the same values on every
    platform, numpy and Python version (no numpy RNG method is involved). Eight values per digest."""
    n = int(n)
    if n <= 0:
        return np.empty((0,), dtype=np.float64)
    blocks = (n + 3) // 4
    buf = bytearray()
    for i in range(blocks):
        buf += hashlib.sha256(f"{tag}:{i}".encode()).digest()
    u64 = np.frombuffer(bytes(buf), dtype=">u8")[:n]
    return (u64 >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))


def _values(tag: str, shape: Sequence[int]) -> np.ndarray:
    size = int(np.prod([int(s) for s in shape])) if len(shape) else 1
    return stream(tag, size).reshape(tuple(int(s) for s in shape))


def select_rows(manifest_hash: str, split: str, n_rows: int, n: int, *, salt: str = "") -> np.ndarray:
    """``int64 [min(n, n_rows)]`` within-split row indices: the ``n`` rows with the smallest
    ``sha256(f"{manifest_hash}:conformance:{split}:{salt}:{row}")``, ascending by row. A deterministic uniform draw
    over the folder's real rows, so real label kinds, empty labels and exclusive sets appear at their own rate."""
    n_rows, n = int(n_rows), int(n)
    if n_rows <= 0 or n <= 0:
        return np.empty((0,), dtype=np.int64)
    if n >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    keys = [hashlib.sha256(f"{manifest_hash}:conformance:{split}:{salt}:{r}".encode()).digest() for r in range(n_rows)]
    order = sorted(range(n_rows), key=keys.__getitem__)[:n]
    return np.asarray(sorted(order), dtype=np.int64)


def _derangement(n: int, tag: str) -> np.ndarray:
    """A fixed-point-free permutation from the stable stream (Fisher–Yates with a final fix-up)."""
    n = int(n)
    if n < 2:
        return np.arange(n, dtype=np.int64)
    u = stream(f"{tag}:derangement", n)
    perm = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = int(u[i] * (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    fixed = np.flatnonzero(perm == np.arange(n))
    for k in fixed:
        other = (int(k) + 1) % n
        perm[k], perm[other] = perm[other], perm[k]
    if (perm == np.arange(n)).any():  # pragma: no cover - two adjacent fixed points swapped twice
        perm = np.roll(np.arange(n, dtype=np.int64), 1)
    return perm


# --------------------------------------------------------------------------- tolerance


def tolerances(bench: Any) -> dict[str, float]:
    """``{metric: absolute tolerance}`` for every name in ``metrics.list`` / ``floors`` / ``controls``:
    ``metrics.tolerance`` where set, else :data:`DEFAULT_TOLERANCE`."""
    m = bench.spec.metrics
    declared = dict(getattr(m, "tolerance", None) or {})
    return {name: float(declared.get(name, DEFAULT_TOLERANCE)) for name in (*m.list, *m.floors, *m.controls)}


# --------------------------------------------------------------------------- the population and its outputs


class _Ctx:
    """What the target builders and the scoring step read from a ``ReadoutContext``: the folder, its protocol and
    manifests; never features. Duck-typed so that this module imports no torch."""

    def __init__(self, bench: Any, probe_mod: Any, reading: str | None) -> None:
        from yavih.readouts.base import _json_native, resolve_metric_specs

        self.bench = bench
        self.spec = bench.spec
        self.readout = bench.spec.readout
        self.protocol = _json_native(dict(bench.spec.protocol))
        self.chance = _json_native(dict(bench.spec.metrics.chance))
        self.ci = dict(bench.spec.metrics.ci)
        self.bootstrap = dict(bench.spec.metrics.bootstrap) if bench.spec.metrics.bootstrap else None
        self.roles = bench.roles()
        self.metric_specs = resolve_metric_specs(bench.spec, probe_mod)
        self.probe_mod = probe_mod
        self.reading = reading
        self.seed = 0
        self._manifests: dict[str, Any] = {}

    @property
    def label_kind(self) -> str:
        return self.spec.manifest.label.kind

    def manifest(self, split: str) -> Any:
        if split not in self._manifests:
            self._manifests[split] = self.bench.manifest(split)
        return self._manifests[split]

    def clip_ids(self, split: str, rows: np.ndarray) -> list[str]:
        from yavih.readouts.base import _manifest_index

        m = self.manifest(split)
        idx = _manifest_index(m, rows)
        return [m.clip_ids[i] for i in idx] if hasattr(m, "clip_ids") else [m.clip_id[i] for i in idx]

    def hook_targets(self, split: str, rows: np.ndarray) -> Any | None:
        fn = getattr(self.probe_mod, "targets", None)
        if fn is None:
            return None
        return fn(self.bench, split, np.asarray(rows, dtype=np.int64))

    def scorer(self, splits: Sequence[str], rows: np.ndarray, targets: Any) -> Any:
        from yavih.readouts.base import _accepts_keyword

        fn = getattr(self.probe_mod, "score", None)
        if fn is None:
            return None
        split = "+".join(splits)
        rows = np.asarray(rows, dtype=np.int64)
        kw = {"reading": self.reading} if _accepts_keyword(fn, "reading") else {}
        return lambda outputs: fn(self.bench, split, rows, outputs, targets, **kw)


def _draw(ctx: _Ctx, split: str, n: int, *, salt: str = "") -> np.ndarray:
    return select_rows(ctx.bench.manifest_hash, split, ctx.bench.n_rows(split), n, salt=salt)


def _floor(ctx: _Ctx, train_targets: Any, n_eval: int, *, num_classes: int | None = None) -> Any:
    from yavih.probes import floors

    kw: dict[str, Any] = {"protocol": ctx.protocol, "label_kind": ctx.label_kind}
    if num_classes is not None:
        kw["num_classes"] = int(num_classes)
    return floors.floor_outputs(ctx.readout, train_targets, int(n_eval), **kw)


def _concat(parts: list[Any]) -> Any:
    if not parts:
        return np.empty((0,), dtype=np.int64)
    if isinstance(parts[0], dict):
        return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    return np.concatenate(parts, axis=0)


def _logits(tag: str, shape: Sequence[int]) -> np.ndarray:
    return ((_values(tag, shape) - 0.5) * 6.0).astype(np.float32)


def _clf_role(ctx: _Ctx, splits: Sequence[str], n: int, *, num_classes: int, role: str, tag: str) -> tuple[Any, np.ndarray, list[str]]:
    from yavih.readouts._targets import clf_targets

    targets, rows, ids = [], [], []
    for s in splits:
        r = _draw(ctx, s, n, salt=role)
        hooked = ctx.hook_targets(s, r)
        targets.append(hooked if hooked is not None else clf_targets(ctx, s, r, num_classes=num_classes, role=role))
        rows.append(r)
        ids.extend(ctx.clip_ids(s, r))
    return _concat(targets), (np.concatenate(rows) if rows else np.empty((0,), np.int64)), ids


def _population_clf(ctx: _Ctx, tag: str) -> dict[str, Any]:
    pop = POPULATION["clf"]
    num_classes = ctx.bench.num_classes
    if num_classes <= 0:
        raise ConformanceError(E_UNSUPPORTED, f"{ctx.bench.name}: clf conformance needs classes.csv")
    base = {"readout": "clf", "label_kind": ctx.label_kind, "num_classes": num_classes, "chance": dict(ctx.chance),
            "bootstrap": dict(ctx.bootstrap or {}), "seed": 0, "floor": None, "folds": None, "sets": None}
    if ctx.roles["folds"]:
        folds = list(ctx.roles["folds"])
        per_fold: dict[str, dict[str, Any]] = {}
        outs, tgts, rows, floors_, train_ts, ids_all = [], [], [], [], [], []
        for k, f in enumerate(folds):
            train_t, _, _ = _clf_role(ctx, [g for g in folds if g != f], pop["fold"], num_classes=num_classes, role="train", tag=tag)
            eval_t, eval_rows, ids = _clf_role(ctx, [f], pop["fold"], num_classes=num_classes, role="eval", tag=tag)
            out = _logits(f"{tag}:fold{k}:outputs", (len(eval_rows), num_classes))
            per_fold[f] = {"outputs": out, "targets": eval_t, "rows": eval_rows, "clip_ids": ids, "n_train": int(len(train_t))}
            outs.append(out)
            tgts.append(eval_t)
            rows.append(eval_rows)
            floors_.append(np.asarray(_floor(ctx, train_t, len(eval_rows), num_classes=num_classes)))
            train_ts.append(train_t)
            ids_all.extend(ids)
        all_rows = np.concatenate(rows)
        targets = _concat(tgts)
        meta = dict(base, n_train=int(round(float(np.mean([p["n_train"] for p in per_fold.values()])))), n_eval=int(len(all_rows)),
                    n_val=0, train_targets=np.concatenate(train_ts, axis=0), eval_rows=all_rows, eval_clip_ids=ids_all,
                    folds=per_fold, n_folds=len(folds))
        if isinstance(targets, dict):
            meta["sets"] = targets
        return {"outputs": np.concatenate(outs, axis=0), "targets": targets, "split_meta": meta,
                "floor_outputs": np.concatenate(floors_, axis=0), "eval_splits": folds, "eval_rows": all_rows}
    train_t, _, _ = _clf_role(ctx, ctx.roles["train"], pop["train"], num_classes=num_classes, role="train", tag=tag)
    eval_t, eval_rows, ids = _clf_role(ctx, ctx.roles["eval"], pop["eval"], num_classes=num_classes, role="eval", tag=tag)
    outputs = _logits(f"{tag}:outputs", (len(eval_rows), num_classes))
    meta = dict(base, n_train=int(len(train_t)), n_eval=int(len(eval_rows)), n_val=0, train_targets=train_t,
                eval_rows=eval_rows, eval_clip_ids=ids)
    if isinstance(eval_t, dict):
        meta["sets"] = eval_t
    if ctx.roles["floor"]:
        fl_t, fl_rows, fl_ids = _clf_role(ctx, ctx.roles["floor"], pop["floor"], num_classes=num_classes, role="floor", tag=tag)
        fl_out = _logits(f"{tag}:floor:outputs", (len(fl_rows), num_classes))
        meta["floor"] = {"outputs": fl_out, "targets": fl_t, "n_floor": int(len(fl_rows)), "rows": fl_rows, "clip_ids": fl_ids}
    return {"outputs": outputs, "targets": eval_t, "split_meta": meta, "floor_outputs": _floor(ctx, train_t, len(eval_rows), num_classes=num_classes),
            "eval_splits": list(ctx.roles["eval"]), "eval_rows": eval_rows}


def _seq_role(ctx: _Ctx, splits: Sequence[str], n: int, *, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    from yavih.readouts._targets import seq_targets

    T = int(ctx.protocol["segments_scored"])
    tg, st, en, rows, ids = [], [], [], [], []
    for s in splits:
        r = _draw(ctx, s, n, salt=role)
        t, a, b = seq_targets(ctx, s, r)
        hooked = ctx.hook_targets(s, r)
        if hooked is not None:
            t = np.asarray(hooked, dtype=np.int64).reshape(len(r), T)
        tg.append(t)
        st.append(a)
        en.append(b)
        rows.append(r)
        ids.extend(ctx.clip_ids(s, r))

    def cat(xs, shape, dt):
        return np.concatenate(xs, axis=0) if xs else np.empty(shape, dtype=dt)

    return cat(tg, (0, T), np.int64), cat(st, (0,), np.int64), cat(en, (0,), np.int64), cat(rows, (0,), np.int64), ids


def _population_seq(ctx: _Ctx, tag: str) -> dict[str, Any]:
    pop = POPULATION["seq"]
    num_classes = ctx.bench.num_classes
    if num_classes <= 0:
        raise ConformanceError(E_UNSUPPORTED, f"{ctx.bench.name}: seq conformance needs classes.csv")
    T = int(ctx.protocol["segments_scored"])
    train_t, _, _, _, _ = _seq_role(ctx, ctx.roles["train"], pop["train"], role="train")
    eval_t, starts, ends, eval_rows, ids = _seq_role(ctx, ctx.roles["eval"], pop["eval"], role="eval")
    outputs = _logits(f"{tag}:outputs", (len(eval_rows), T, num_classes))
    num_segments = int(ctx.protocol["num_segments"])
    meta = {"readout": "seq", "label_kind": ctx.label_kind, "num_classes": num_classes, "n_train": int(len(train_t)),
            "n_eval": int(len(eval_rows)), "n_val": 0, "train_targets": train_t, "eval_rows": eval_rows, "eval_clip_ids": ids,
            "event_start": starts, "event_end": ends, "boundary": (ends - starts) < num_segments, "chance": dict(ctx.chance),
            "bootstrap": dict(ctx.bootstrap or {}), "seed": 0}
    return {"outputs": outputs, "targets": eval_t, "split_meta": meta, "floor_outputs": _floor(ctx, train_t, len(eval_rows), num_classes=num_classes),
            "eval_splits": list(ctx.roles["eval"]), "eval_rows": eval_rows}


def _depth(tag: str, n: int, size: int, max_depth: float) -> np.ndarray:
    """Positive depths in ``(0, max_depth]`` with ~5 % invalid (0) pixels, the way a sensor map carries holes."""
    u = _values(f"{tag}:depth", (n, size, size))
    d = (0.02 + 0.98 * u) * float(max_depth)
    holes = _values(f"{tag}:holes", (n, size, size)) < 0.05
    return np.where(holes, 0.0, d).astype(np.float32)


def _population_grid(ctx: _Ctx, tag: str) -> dict[str, Any]:
    pop = POPULATION["grid"]
    size = ctx.protocol.get("target_size")
    max_depth = ctx.protocol.get("max_depth")
    if not isinstance(size, int) or max_depth is None:
        raise ConformanceError(E_UNSUPPORTED, f"{ctx.bench.name}: grid conformance needs protocol.target_size and protocol.max_depth")
    (train_split,), (eval_split,) = ctx.roles["train"], ctx.roles["eval"]
    train_rows = _draw(ctx, train_split, pop["train"], salt="train")
    eval_rows = _draw(ctx, eval_split, pop["eval"], salt="eval")
    train_arr = _depth(f"{tag}:train", len(train_rows), int(size), float(max_depth))
    eval_t = _depth(f"{tag}:eval", len(eval_rows), int(size), float(max_depth))
    outputs = (0.05 + 0.95 * _values(f"{tag}:outputs", (len(eval_rows), int(size), int(size)))).astype(np.float32) * float(max_depth)
    # the floor streams the raw train maps positionally (``floors._iter_chunks`` slices an array), as ``run_grid`` does
    meta = {"readout": "grid", "n_train": int(len(train_rows)), "n_eval": int(len(eval_rows)), "n_val": 0,
            "train_targets": train_arr, "eval_rows": eval_rows, "eval_clip_ids": ctx.clip_ids(eval_split, eval_rows),
            "chance": dict(ctx.chance), "bootstrap": dict(ctx.bootstrap or {}), "seed": 0}
    return {"outputs": outputs, "targets": eval_t, "split_meta": meta, "floor_outputs": _floor(ctx, train_arr, len(eval_rows)),
            "eval_splits": [eval_split], "eval_rows": eval_rows}


def _box_role(ctx: _Ctx, splits: Sequence[str], n: int, *, role: str) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    from yavih.readouts._targets import box_targets

    tg, rows, ids = [], [], []
    excluded = 0
    for s in splits:
        r = _draw(ctx, s, n, salt=role)
        hooked = ctx.hook_targets(s, r)
        if hooked is not None:
            arr = np.asarray(hooked, dtype=np.float32).reshape(len(r), 4)
            ok = (arr[:, 2] > 0) & (arr[:, 3] > 0) & np.isfinite(arr).all(axis=1)
            t, kept, n_ex = arr[ok], r[ok], int((~ok).sum())
        else:
            t, kept, n_ex = box_targets(ctx, s, r)
        excluded += n_ex
        tg.append(t)
        rows.append(kept)
        ids.extend(ctx.clip_ids(s, kept))

    def cat(xs, shape, dt):
        return np.concatenate(xs, axis=0) if xs else np.empty(shape, dtype=dt)

    return cat(tg, (0, 4), np.float32), cat(rows, (0,), np.int64), ids, excluded


def _boxes(tag: str, n: int) -> np.ndarray:
    u = _values(tag, (n, 4))
    out = np.empty((n, 4), dtype=np.float32)
    out[:, 0] = 0.2 + 0.6 * u[:, 0]
    out[:, 1] = 0.2 + 0.6 * u[:, 1]
    out[:, 2] = 0.1 + 0.3 * u[:, 2]
    out[:, 3] = 0.1 + 0.3 * u[:, 3]
    return out


def _population_box(ctx: _Ctx, tag: str) -> dict[str, Any]:
    pop = POPULATION["box"]
    train_t, _, _, ex_train = _box_role(ctx, ctx.roles["train"], pop["train"], role="train")
    eval_t, eval_rows, ids, ex_eval = _box_role(ctx, ctx.roles["eval"], pop["eval"], role="eval")
    outputs = _boxes(f"{tag}:outputs", len(eval_rows))
    meta: dict[str, Any] = {"readout": "box", "n_train": int(len(train_t)), "n_eval": int(len(eval_rows)), "n_val": 0,
                            "train_targets": train_t, "eval_rows": eval_rows, "eval_clip_ids": ids,
                            "n_excluded_degenerate": int(ex_eval), "n_excluded_degenerate_train": int(ex_train),
                            "chance": dict(ctx.chance), "bootstrap": dict(ctx.bootstrap or {}), "seed": 0, "shuffled": None}
    rp = dict(ctx.spec.readout_params)
    if rp.get("shuffled_control"):
        seed = int(rp.get("shuffle_seed", 0))
        meta["shuffled"] = {"outputs": _boxes(f"{tag}:shuffled", len(eval_rows)), "perm_train": _derangement(len(train_t), f"{tag}:train"),
                            "perm_eval": _derangement(len(eval_rows), f"{tag}:eval"), "seed": seed}
    return {"outputs": outputs, "targets": eval_t, "split_meta": meta, "floor_outputs": _floor(ctx, train_t, len(eval_rows)),
            "eval_splits": list(ctx.roles["eval"]), "eval_rows": eval_rows}


_POPULATIONS = {"clf": _population_clf, "seq": _population_seq, "grid": _population_grid, "box": _population_box}


# --------------------------------------------------------------------------- compute


def _readings(probe_mod: Any) -> list[str | None]:
    """The readings a reading-routed ``score`` hook is checked under (``probe.py`` ``READINGS``), else ``[None]``."""
    from yavih.readouts.base import _accepts_keyword

    fn = getattr(probe_mod, "score", None)
    if fn is None or not _accepts_keyword(fn, "reading"):
        return [None]
    readings = getattr(probe_mod, "READINGS", None)
    if not readings:
        raise ConformanceError(E_UNSUPPORTED, "probe.py score() takes a reading but declares no READINGS tuple")
    return [str(r) for r in readings]


def _externals(bench: Any) -> dict[str, str]:
    """``{package: version}`` of the pinned externals, or ``ConformanceError(W_SKIPPED)`` when one is absent here."""
    from yavih.readouts.base import ProbeRefused, check_externals

    try:
        return check_externals(bench.spec)
    except ProbeRefused as e:
        raise ConformanceError(W_SKIPPED, f"{e.code}: {e.message}") from e


def compute(bench: Any) -> dict[str, Any]:
    """Run the folder's scoring path on the generated population and return the fixture values:
    ``{"generator", "population", "externals", "values": {reading or "*": {metrics, floors, controls, ci95}}}``.
    Raises ``ConformanceError`` (``W_CONFORMANCE_SKIPPED`` when an external package is absent,
    ``E_CONFORMANCE_UNSUPPORTED`` for a readout without a population)."""
    from yavih.readouts.base import ProbeRefused, evaluate_metrics, folder_probe

    readout = bench.spec.readout
    build = _POPULATIONS.get(readout)
    if build is None:
        raise ConformanceError(E_UNSUPPORTED, f"{bench.name}: no conformance population for readout {readout!r}")
    externals = _externals(bench)
    probe_mod = folder_probe(bench)
    tag = f"{bench.manifest_hash}:conformance:{GENERATOR_VERSION}"
    values: dict[str, Any] = {}
    population: dict[str, Any] | None = None
    for reading in _readings(probe_mod):
        ctx = _Ctx(bench, probe_mod, reading)
        pop = build(ctx, tag)
        try:
            blocks = evaluate_metrics(ctx.metric_specs, outputs=pop["outputs"], targets=pop["targets"], protocol=ctx.protocol,
                                      split_meta=pop["split_meta"], floor_outputs=pop["floor_outputs"], ci=ctx.ci,
                                      scorer=ctx.scorer(pop["eval_splits"], pop["eval_rows"], pop["targets"]))
        except ProbeRefused as e:
            first = e.message.split("; ")[0]
            if "E_EXTERNAL_" in e.message:  # the folder's own guard on the external package (a wrong build): not checkable here
                raise ConformanceError(W_SKIPPED, f"{bench.name}: {first}") from e
            raise ConformanceError(e.code if e.code.startswith("E_") else E_UNSUPPORTED, f"{bench.name}: {first}") from e
        values[reading or "*"] = {k: _plain(blocks.get(k) or {}) for k in ("metrics", "floors", "controls", "ci95")}
        meta = pop["split_meta"]
        population = {"eval": int(meta.get("n_eval", 0)), "train": int(meta.get("n_train", 0)),
                      "floor": int((meta.get("floor") or {}).get("n_floor", 0)) if isinstance(meta.get("floor"), dict) else 0,
                      "folds": int(meta.get("n_folds", 0) or 0)}
    return {"generator": GENERATOR_VERSION, "population": population or {}, "externals": externals, "values": values}


def _plain(block: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in block.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple, np.ndarray)):
            out[k] = [None if x is None else float(x) for x in v]
        else:
            out[k] = float(v)
    return out


# --------------------------------------------------------------------------- expected.json


def expected_path(bench: Any) -> Path:
    return Path(bench.folder) / EXPECTED_FILE


def read_expected(bench: Any) -> dict[str, Any] | None:
    """The folder's ``expected.json`` as a dict, ``None`` when absent; ``ConformanceError(E_CONFORMANCE_SCHEMA)``
    when it does not parse or lacks the keys this version writes."""
    p = expected_path(bench)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConformanceError(E_SCHEMA, f"{EXPECTED_FILE}: {e}") from e
    need = {"schema", "benchmark", "protocol_version", "generator", "tolerance", "values", "recorded_with"}
    if not isinstance(d, dict) or not need <= set(d):
        raise ConformanceError(E_SCHEMA, f"{EXPECTED_FILE}: not a {CONFORMANCE_SCHEMA} record (keys {sorted(need)})")
    if d["schema"] != CONFORMANCE_SCHEMA:
        raise ConformanceError(E_SCHEMA, f"{EXPECTED_FILE}: schema {d['schema']!r}, this yavih writes {CONFORMANCE_SCHEMA!r}")
    if d["benchmark"] != bench.name:
        raise ConformanceError(E_SCHEMA, f"{EXPECTED_FILE}: records benchmark {d['benchmark']!r}, the folder is {bench.name!r}")
    return d


def _record_dict(bench: Any, computed: Mapping[str, Any], *, reason: str | None) -> dict[str, Any]:
    from yavih.versions import writer_stamp

    stamp = writer_stamp()
    rec: dict[str, Any] = {
        "schema": CONFORMANCE_SCHEMA,
        "benchmark": bench.name,
        "protocol_version": int(bench.spec.protocol_version),
        "generator": int(computed["generator"]),
        "population": dict(computed["population"]),
        "tolerance": tolerances(bench),
        "values": computed["values"],
        "recorded_with": {
            "yavih_version": stamp.get("version", __version__), "commit": stamp.get("commit"),
            "protocol_hash": bench.protocol_hash, "scoring_hash": bench.scoring_hash,
            "folder_source": dict(getattr(bench, "folder_source", None) or {}),
            "externals": dict(computed.get("externals") or {}),
        },
    }
    if reason:
        rec["same_version_reason"] = str(reason)
    return rec


def _compare(expected: Mapping[str, Any], got: Mapping[str, Any], tol: Mapping[str, float]) -> tuple[int, list[dict[str, Any]]]:
    """Every metric / floor / control of every reading within tolerance; ``ci95`` is recorded, never compared."""
    compared, drifted = 0, []
    exp_vals, got_vals = expected["values"], got["values"]
    for reading in sorted(set(exp_vals) | set(got_vals)):
        if reading not in exp_vals or reading not in got_vals:
            drifted.append({"reading": reading, "block": None, "metric": None, "expected": reading in exp_vals, "got": reading in got_vals,
                            "tolerance": None, "message": "reading recorded on one side only"})
            continue
        for block in ("metrics", "floors", "controls"):
            e, g = dict(exp_vals[reading].get(block) or {}), dict(got_vals[reading].get(block) or {})
            for name in sorted(set(e) | set(g)):
                compared += 1
                t = float(tol.get(name, DEFAULT_TOLERANCE))
                ev, gv = e.get(name, "<absent>"), g.get(name, "<absent>")
                if not _within(ev, gv, t):
                    drifted.append({"reading": reading, "block": block, "metric": name, "expected": ev, "got": gv, "tolerance": t})
    return compared, drifted


def _within(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != len(b):
            return False
        return all(_within(x, y, tol) for x, y in zip(a, b, strict=True))
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return abs(fa - fb) <= tol


def check(bench: Any) -> ConformanceReport:
    """Compare the folder's current numbers with ``conformance/expected.json``.

    - no file: ``E_CONFORMANCE_MISSING`` (record one: ``yavih bench conformance <name> --record``);
    - the file's ``protocol_version`` (or generator) differs from the folder's: ``E_CONFORMANCE_STALE`` (re-record);
    - a value outside its tolerance at the same version: ``E_CONFORMANCE_DRIFT`` (bump ``protocol_version`` with
      ``invalidates: true``, re-record; or ``--same-version --reason`` when the protocol did not change);
    - an external package absent here: ``W_CONFORMANCE_SKIPPED`` (nothing compared).
    """
    pv = int(bench.spec.protocol_version)
    rep = ConformanceReport(name=bench.name, folder=str(bench.folder), protocol_version=pv, expected_version=None, ok=True, skipped=False)
    try:
        expected = read_expected(bench)
    except ConformanceError as e:
        rep.findings.append(Finding("error", e.code, e.message))
        rep.ok = False
        return rep
    if expected is None:
        rep.findings.append(Finding("error", E_MISSING, f"{bench.name}: no {EXPECTED_FILE}; record it with `yavih bench conformance {bench.name} --record`"))
        rep.ok = False
        return rep
    rep.expected_version = int(expected["protocol_version"])
    if rep.expected_version != pv or int(expected.get("generator", -1)) != GENERATOR_VERSION:
        rep.findings.append(Finding("error", E_STALE, f"{bench.name}: {EXPECTED_FILE} was recorded at protocol_version {rep.expected_version} "
                                    f"(generator {expected.get('generator')}), the folder is at {pv} (generator {GENERATOR_VERSION}); re-record it with "
                                    f"`yavih bench conformance {bench.name} --record`"))
        rep.ok = False
        return rep
    try:
        got = compute(bench)
    except ConformanceError as e:
        if e.code == W_SKIPPED:
            rep.findings.append(Finding("warning", W_SKIPPED, f"{bench.name}: not checked here: {e.message}"))
            rep.skipped = True
            return rep
        rep.findings.append(Finding("error", e.code, e.message))
        rep.ok = False
        return rep
    tol = tolerances(bench)
    rep.compared, rep.drifted = _compare(expected, got, tol)
    if rep.drifted:
        rep.ok = False
        shown = "; ".join(f"{d['metric'] or d['reading']}: expected {_fmt(d['expected'])}, got {_fmt(d['got'])} (tol {d['tolerance']})"
                          for d in rep.drifted[:6])
        more = f" (+{len(rep.drifted) - 6} more)" if len(rep.drifted) > 6 else ""
        rep.findings.append(Finding("error", E_DRIFT, f"{bench.name}: {len(rep.drifted)} of {rep.compared} values moved at protocol_version {pv}: {shown}{more}. "
                                    "If the protocol changed, bump protocol_version (changelog entry with invalidates: true) and re-record; if the hash "
                                    "moved but these numbers should not have, the code changed: fix it, or bump the version; if only the fixture "
                                    "was wrong, `--record --same-version --reason TEXT`"))
    return rep


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def record(bench: Any, *, same_version: bool = False, reason: str | None = None, force: bool = False) -> dict[str, Any]:
    """Compute the fixture and write ``conformance/expected.json``. Refuses (``E_CONFORMANCE_DRIFT``) to overwrite a
    record of the same ``protocol_version`` and generator whose values differ, unless ``same_version`` with a
    ``reason`` (stored) or ``force``. Returns the record written."""
    computed = compute(bench)  # W_CONFORMANCE_SKIPPED propagates: nothing can be recorded without the package
    rec = _record_dict(bench, computed, reason=reason if same_version else None)
    existing = None if force else read_expected(bench)
    if existing is not None:
        same = int(existing["protocol_version"]) == rec["protocol_version"] and int(existing.get("generator", -1)) == GENERATOR_VERSION
        if same:
            _, drifted = _compare(existing, rec, tolerances(bench))
            if drifted and not same_version:
                raise ConformanceError(E_DRIFT, f"{bench.name}: {len(drifted)} value(s) differ from the record at the same protocol_version "
                                       f"{rec['protocol_version']} ({', '.join(str(d['metric'] or d['reading']) for d in drifted[:6])}). Bump "
                                       "protocol_version and record, or pass --same-version --reason TEXT if the protocol did not change")
            if drifted and same_version and not reason:
                raise ConformanceError(E_DRIFT, "--same-version needs --reason TEXT (why the numbers moved without a protocol change)")
    p = expected_path(bench)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    p.write_text(text, encoding="utf-8")
    return rec
