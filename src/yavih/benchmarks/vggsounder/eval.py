"""vggsounder probe hooks: an external scorer (the folder's setup.md).

Every vggsounder number is a value the authors' package, ``vggsounder`` (PyPI 1.0), returns; yavih computes
none of them. The package is NOT a yavih dependency: it is installed independently (setup.md
"## Dependencies") and reached only through the assert-import of ``benchmark.yaml`` ``external``
(``yavih.external.require``). Module scope imports stdlib, numpy and yavih core only.

The package's ground-truth sets (``vggsounder.benchmark.MODALITIES``) and the pkl logits key each is scored
from (``load_test_predictions``): ``a`` (A + AV labels) ← ``logits["a"]``, ``v`` (V + AV) ← ``logits["v"]``,
``av`` (labels both audible and visible) ← ``logits["av"]``, ``a only`` ← ``logits["a"]``, ``v only`` ←
``logits["v"]``. Per set ``compare_labels_logits`` returns the 30 ``PACKAGE_METRICS``: for k in 1, 3, 5, 10
``f1@k`` (micro), ``f1_macro@k``, ``accuracy@k``, ``jaccard@k``, ``precision@k``, ``recall@k``, ``hit_rate@k``,
and ``auc_roc`` / ``auc_pr`` (macro over the classes with a positive). Each set is scored over the clips holding at
least one label of it.

Hooks:

- ``METRICS``: the 30 package names on the reading's own set and the same names with ``_only`` on its exclusive
  set; ``external: vggsounder``, the package's name and set in ``MetricSpec.description``. The ids do not depend on
  the reading, so one ``benchmark.yaml`` serves the three readings. No floors, no chance values, no yavih
  disclosure: nothing here is computed outside the package.
- ``score(bench, split, rows, outputs, targets, *, reading)``: the run's eval logits ``[n, 309]`` (rows in
  ``classes.csv`` order) are written under the pkl key of the reading ONLY (``READING_SETS``: ``a`` → sets ``a``
  and ``a only``; ``v`` → ``v`` and ``v only``; ``av`` → ``av``) and every set of that reading is scored by the
  package's own per-set step of ``compute_all_results`` (``get_gt_for_modality`` → ``load_test_predictions`` →
  ``compare_labels_logits``). A reading's logits are never scored on another reading's set. The ``av`` reading has
  no exclusive set: its ``*_only`` metrics are ``None`` (``null`` in ``result.json``: not applicable).
- ``join(bench, results)``: ``report --join vggsounder a=RUN v=RUN av=RUN`` — the three readings'
  ``eval_logits.npy`` under their own pkl keys through ``compute_all_results``: the 30 metrics on all five sets
  plus ``mu`` (modality confusion against the ``all`` ground truth) for ``a``, ``v`` and ``av``, exactly as the
  package returns them, and a compact ``table`` (rows A / V / AV; ``auc_pr``, ``hit_rate@1``, ``f1@1``, and ``mu``
  displayed A / V / A∩V for the package keys ``a`` / ``v`` / ``av``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from yavih.metrics.registry import MetricSpec

__all__ = [
    "METRICS",
    "JOIN_METRICS",
    "MU_LABELS",
    "PACKAGE_METRICS",
    "PACKAGE_SETS",
    "READING_SETS",
    "ONLY_SUFFIX",
    "PROTOCOL_KEYS",
    "JOIN_ROUTINGS",
    "JOIN_INPUTS",
    "JOIN_TABLE",
    "ExternalPackageError",
    "JoinRefused",
    "external_entry",
    "require_vggsounder",
    "official_set_scores",
    "official_scores",
    "metric_ids",
    "score",
    "join",
]

#: ``compare_labels_logits`` (``ks = [1, 3, 5, 10]``): its result keys, in the order it computes them.
PACKAGE_METRICS: tuple[str, ...] = tuple(
    f"{m}@{k}" for k in (1, 3, 5, 10) for m in ("f1", "f1_macro", "accuracy", "jaccard", "precision", "recall", "hit_rate")
) + ("auc_roc", "auc_pr")
#: the package's ground-truth sets (``vggsounder.benchmark.MODALITIES``).
PACKAGE_SETS: tuple[str, ...] = ("a", "v", "av", "a only", "v only")
#: reading → (its own set, its exclusive set or ``None``): the pkl key is the reading, and the package reads
#: ``a only`` / ``v only`` from ``logits["a"]`` / ``logits["v"]``.
READING_SETS: dict[str, tuple[str, str | None]] = {"a": ("a", "a only"), "v": ("v", "v only"), "av": ("av", None)}
#: The readings ``score`` routes (``yavih.conformance`` records the fixture under each).
READINGS: tuple[str, ...] = ("a", "v", "av")
#: suffix of the metric ids scored on the reading's exclusive set (``a only`` / ``v only``).
ONLY_SUFFIX = "_only"
#: pkl keys of ``report --join`` (the three readings of one model), the paper's routings.
JOIN_ROUTINGS: tuple[str, ...] = ("a", "v", "av")
#: the join's compact markdown table: (column label, package metric); rows A / V / AV are the sets a / v / av.
JOIN_TABLE: tuple[tuple[str, str], ...] = (("mAP", "auc_pr"), ("HIT", "hit_rate@1"), ("F1", "f1@1"), ("mu", "mu"))
#: the package's dataset variant this benchmark freezes: default, no background-music clips.
_VARIANT = {"background_music": False, "download_samples": False}
_MODEL = "yavih_probe"

PROTOCOL_KEYS: frozenset[str] = frozenset()


class ExternalPackageError(RuntimeError):
    """The official package is absent or at another version (exit 3, not ready)."""

    exit_code = 3

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class JoinRefused(RuntimeError):
    """``report --join vggsounder`` refused (exit 2): mixed hashes, populations or missing results."""

    exit_code = 2

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class _ComputedByScore:
    """``MetricSpec.fn`` placeholder: these metrics come from ``score()`` (the external scorer)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"{self.name} is computed by the vggsounder folder's eval.py score() with the official "
            "vggsounder package; it has no registry function"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"vggsounder.probe.score:{self.name}"


def _spec(name: str, description: str, *, higher_is_better: bool | None = True) -> MetricSpec:
    return MetricSpec(
        name=name,
        fn=_ComputedByScore(name),
        readouts=frozenset({"clf"}),
        protocol_keys=frozenset(),
        shape="scalar",
        kind="metric",
        higher_is_better=higher_is_better,
        external="vggsounder",
        description=description,
    )


METRICS: dict[str, MetricSpec] = {}
for _m in PACKAGE_METRICS:
    METRICS[_m] = _spec(_m, f"vggsounder 1.0 compare_labels_logits {_m!r} on the reading's own ground truth "
                            "(reading a: set 'a' = A + AV labels; v: 'v' = V + AV; av: 'av' = labels both audible and visible)")
for _m in PACKAGE_METRICS:
    METRICS[_m + ONLY_SUFFIX] = _spec(_m + ONLY_SUFFIX, f"vggsounder 1.0 compare_labels_logits {_m!r} on the reading's "
                                                        "exclusive ground truth (reading a: set 'a only'; v: 'v only'); "
                                                        "null for the av reading (no exclusive set)")
del _m

#: the join-only ``mu`` values (``compute_modality_confusion_logits`` against the ``all`` ground truth), named by
#: their package key, with the package's own definitions.
JOIN_METRICS: dict[str, MetricSpec] = {
    "mu_a": _spec("mu_a", "vggsounder 1.0 mu['a']: Audio confusion rate (audio top-1 hits when AV top-1 fails)",
                  higher_is_better=None),
    "mu_v": _spec("mu_v", "vggsounder 1.0 mu['v']: Visual confusion rate (visual top-1 hits when AV top-1 fails)",
                  higher_is_better=None),
    "mu_av": _spec("mu_av", "vggsounder 1.0 mu['av']: Combined confusion rate (both A and V top-1 hit when AV top-1 fails)",
                   higher_is_better=None),
}
#: how the report displays the three ``mu`` values (display labels only; the ids stay the package keys).
MU_LABELS: dict[str, str] = {"mu_a": "A", "mu_v": "V", "mu_av": "A∩V"}


def metric_ids(reading: str) -> dict[str, tuple[str, str]]:
    """``{metric id: (package set, package metric)}`` a reading's ``score`` returns a value for (the ``av``
    reading's ``*_only`` ids are absent here: ``score`` returns ``None`` for them)."""
    own, only = READING_SETS[reading]
    out = {m: (own, m) for m in PACKAGE_METRICS}
    if only is not None:
        out.update({m + ONLY_SUFFIX: (only, m) for m in PACKAGE_METRICS})
    return out


# ----------------------------------------------------------------------------- the assert-import


def external_entry(bench: Any = None) -> dict[str, str]:
    """The ``external`` entry for ``vggsounder`` from ``benchmark.yaml`` (the folder's yaml next to
    this file; ``bench.spec.external`` when the loaded spec exposes it)."""
    entries = None
    spec = getattr(bench, "spec", None)
    if spec is not None and getattr(spec, "external", None) is not None:
        entries = [e if isinstance(e, Mapping) else dataclasses.asdict(e) for e in spec.external]
    if entries is None:
        import yaml

        entries = yaml.safe_load(
            Path(__file__).with_name("benchmark.yaml").read_text(encoding="utf-8")
        )["external"]
    for e in entries:
        if e["name"] == "vggsounder":
            return {k: str(e[k]) for k in ("name", "pip", "version_pin", "used_for")}
    raise KeyError("benchmark.yaml external has no 'vggsounder' entry")


def require_vggsounder(bench: Any = None) -> str:
    """Assert-import the official package at its pinned version (``yavih.external.require``); return the version
    used (recorded by ``yavih.external.versions_used`` for ``result.json[external]``)."""
    from yavih.external import require, versions_used

    e = external_entry(bench)
    require(e["name"], e["pip"], e["version_pin"])
    return versions_used()[e["name"]]


def _check_package_result(r: Mapping[str, Any], bench: Any) -> None:
    if "auc_pr" not in r:
        raise ExternalPackageError(
            "E_EXTERNAL_VERSION",
            "the installed vggsounder build reports no 'auc_pr' (a development build renamed it 'mAP'); "
            f"install the pinned release: {external_entry(bench)['pip']}",
        )


def _float(v: Any) -> Any:
    return float(v) if np.ndim(v) == 0 else v


# ----------------------------------------------------------------------------- the official calls


def _write_pkl(tmp: str, payload: Mapping[str, Any]) -> str:
    path = os.path.join(tmp, f"{_MODEL}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(dict(payload), fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def official_set_scores(payload: Mapping[str, Any], subset_ids: Sequence[str], sets: Sequence[str]) -> dict:
    """The package's per-set step of ``compute_all_results`` (its logits branch) for ``sets`` only, on one model's
    pkl payload ``{video_id: {"logits": {key: [C floats]}}}``: ``get_gt_for_modality`` → ``load_test_predictions``
    (which maps ``a only`` / ``v only`` to the ``a`` / ``v`` key) → ``compare_labels_logits``. Returns ``{set:
    {metric: value}}`` as fractions. Sets outside ``sets`` are never scored, so a payload holding one reading's key
    is never read under another key (the package raises ``KeyError`` first)."""
    from vggsounder.benchmark import (
        compare_labels_logits,
        get_gt_for_modality,
        load_test_predictions,
    )
    from vggsounder.labels import VGGSounder

    vggsounder = VGGSounder(**_VARIANT)
    classes = vggsounder.get_all_labels()
    ids = set(subset_ids)  # membership only; a set keeps it O(n)
    out: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="yavih-vggsounder-") as tmp:
        pkl = _write_pkl(tmp, payload)
        for s in sets:
            gt = get_gt_for_modality(vggsounder, s, ids)
            pred = load_test_predictions(pkl, s)
            pred.drop(columns=["labels"], inplace=True)
            out[s] = {k: _float(v) for k, v in compare_labels_logits(gt, pred, classes).items()}
    return out


def official_scores(
    payload: Mapping[str, Any], subset_ids: Sequence[str], modalities: Sequence[str]
) -> dict:
    """``vggsounder.benchmark.compute_all_results`` on one model's pkl payload
    ``{video_id: {"logits": {modality: [C floats]}}}``; returns ``{modality: {metric: value}}``
    (fractions, as the package computes them before ``benchmark()``'s ×100 display scaling)."""
    from vggsounder.benchmark import compute_all_results
    from vggsounder.labels import VGGSounder

    with tempfile.TemporaryDirectory(prefix="yavih-vggsounder-") as tmp:
        _write_pkl(tmp, payload)
        res = compute_all_results(
            models_path=tmp,
            subset_ids=set(subset_ids),  # membership only; a set keeps it O(n)
            modalities=list(modalities),
            vggsounder=VGGSounder(**_VARIANT),
            verbose=False,
        )
    return {m: {k: _float(v) for k, v in d.items()} for m, d in res[_MODEL].items()}


def _eval_ids(bench: Any, split: str, rows: Any) -> list[str]:
    ids = list(bench.manifest(split).clip_id)
    return [str(ids[int(r)]) for r in np.asarray(rows).ravel()]


def _check_outputs(outputs: Any, n: int) -> np.ndarray:
    out = np.asarray(outputs)
    if out.ndim != 2 or out.shape[0] != n or out.shape[1] != 309:
        raise ValueError(f"vggsounder score: outputs must be [n_eval={n}, 309], got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError("vggsounder score: non-finite logits")
    return out


def score(bench: Any, split: str, rows: Any, outputs: Any, targets: Any = None, *, reading: str | None = None) -> dict:
    """Every ``METRICS`` id for one reading from the official package, routed as the package routes it.

    ``reading`` = the run's ``describe()["reading"]`` (one of ``a``, ``v``, ``av``; setup.md); ``outputs`` = the
    head's raw logits ``[len(rows), 309]`` in ``classes.csv`` order (identical to the package's ``classes.csv``,
    sha256 ``90a5cd19…``), written under the pkl key ``reading`` only; ``targets`` is unused: the ground truth is
    the package's own. Returns the 30 package metrics on the reading's own set and, with ``_only``, on its
    exclusive set; ``None`` for the ``_only`` ids of the ``av`` reading (not applicable).
    """
    if reading not in READING_SETS:
        raise ValueError(
            f"vggsounder score: describe()['reading'] is {reading!r}; the package routes only the readings "
            f"{sorted(READING_SETS)} (a = audio only, v = video only, av = both)"
        )
    require_vggsounder(bench)
    ids = _eval_ids(bench, split, rows)
    out = _check_outputs(outputs, len(ids))
    payload = {cid.removesuffix(".mp4"): {"logits": {reading: out[i].astype(float).tolist()}} for i, cid in enumerate(ids)}
    sets = [s for s in READING_SETS[reading] if s is not None]
    res = official_set_scores(payload, ids, sets)
    for s in sets:
        _check_package_result(res[s], bench)
    metrics: dict[str, Any] = {mid: float(res[s][m]) for mid, (s, m) in metric_ids(reading).items()}
    for m in PACKAGE_METRICS:
        metrics.setdefault(m + ONLY_SUFFIX, None)  # the av reading: no exclusive set
    return metrics


# ----------------------------------------------------------------------------- report --join


def _record(ref: Any, seed_dir: Path) -> dict:
    """The result record of one reading: ``ref.result`` when the caller handed it over (``report --join`` passes
    ``ResultRef``s whose records are upgraded in memory), else ``<seed_dir>/result.json``."""
    rec = getattr(ref, "result", None)
    if isinstance(rec, Mapping):
        return dict(rec)
    return json.loads((seed_dir / "result.json").read_text(encoding="utf-8"))


def _seed_dir(ref: Any) -> Path:
    for attr in ("seed_dir", "path", "dir"):
        if hasattr(ref, attr):
            ref = getattr(ref, attr)
            break
    if isinstance(ref, Mapping):
        ref = ref.get("seed_dir") or ref.get("path")
    p = Path(os.fspath(ref)).expanduser()
    if not (p / "result.json").is_file():
        raise JoinRefused(
            "E_JOIN_MISSING",
            f"{p}: no result.json (pass the seed directory of a DONE probe result)",
        )
    if not (p / "DONE").is_file():
        raise JoinRefused("E_JOIN_MISSING", f"{p}: no DONE marker")
    return p


#: the streams each routing's bridge must read (its ``describe()["input"]`` keys): ``mu`` and the per-set
#: numbers depend on which logits are ``a``, ``v`` and ``av``, so a swapped or impure routing is refused.
JOIN_INPUTS: dict[str, frozenset[str]] = {"a": frozenset({"audio"}), "v": frozenset({"video"}), "av": frozenset({"audio", "video"})}


def _describe(rec: Mapping[str, Any]) -> Mapping[str, Any]:
    return (rec.get("bridge") or {}).get("describe") or {}


def _model(rec: Mapping[str, Any]) -> str | None:
    """The model NAME of a result (``result.json["model"]`` = ``describe()["model"]``)."""
    d = _describe(rec)
    name = rec.get("model") or d.get("model")
    return None if name is None else str(name)


def _label(rec: Mapping[str, Any]) -> str:
    return f"{_model(rec)}/{rec.get('reading') or _describe(rec).get('reading')}"


def _check_readings(recs: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    """The three results must be the audio-only, video-only and audio-visual readings of ONE model under ONE
    standard probe (setup.md): ``E_JOIN_ROUTING`` (a reading's bridge reads other streams than its
    routing), ``E_JOIN_MODEL`` (not one ``model`` name), ``E_JOIN_PROBE`` (different
    probe hashes, or a non-standard probe config). Returns the join's warnings: ``W_JOIN_WEIGHTS`` when the
    readings carry different non-null ``weights_sha256`` (a bridge decision, recorded, never a refusal: an
    audio-only reading may fingerprint only the modules it loads)."""
    describes = {k: _describe(r) for k, r in recs.items()}
    for k in JOIN_ROUTINGS:
        got = frozenset(str(s) for s in (describes[k].get("input") or {}))
        if got != JOIN_INPUTS[k]:
            raise JoinRefused(
                "E_JOIN_ROUTING",
                f"reading {k}={_label(recs[k])} reads streams {sorted(got)}, routing {k!r} needs exactly "
                f"{sorted(JOIN_INPUTS[k])} (a = audio-only, v = video-only, av = both; check the a=/v=/av= order)",
            )
    models = {k: _model(r) for k, r in recs.items()}
    if len(set(models.values())) != 1 or None in models.values():
        raise JoinRefused(
            "E_JOIN_MODEL",
            f"the three readings are not one model: describe()['model'] {models} "
            "(every bridge of one model declares the same `model` name)",
        )
    probes = {str((r.get("probe") or {}).get("hash")) for r in recs.values()}
    nonstd = [k for k, r in recs.items() if not bool(r.get("standard"))]
    if len(probes) != 1 or nonstd:
        raise JoinRefused(
            "E_JOIN_PROBE",
            f"the three readings must share one standard probe: probe hashes {sorted(p[:12] for p in probes)}"
            + (f", non-standard readings {nonstd}" if nonstd else ""),
        )
    weights = {k: d.get("weights_sha256") for k, d in describes.items()}
    warnings: list[dict] = []
    if len({w for w in weights.values() if w is not None}) > 1:
        warnings.append({
            "code": "W_JOIN_WEIGHTS",
            "message": "the three readings of {} carry different weights_sha256 ({}); recorded, not refused: a "
            "reading may fingerprint only the modules it loads".format(
                next(iter(models.values())), ", ".join(f"{k} {str(w)[:12]}" for k, w in weights.items())),
            "where": None,
        })
    return warnings


def _join_table(scores: Mapping[str, Mapping[str, Any]]) -> dict:
    """The compact view ``report`` prints by default: rows A / V / AV (the package sets ``a`` / ``v`` / ``av``),
    columns ``JOIN_TABLE``; the ``mu`` column holds the package's ``mu`` of the row's key, displayed as
    ``MU_LABELS`` (A, V, A∩V) and defined in the notes."""
    header = ["", *(f"mu ({', '.join(MU_LABELS.values())})" if m == "mu" else label for label, m in JOIN_TABLE)]
    rows = [[k.upper(), *(scores[k].get(m) for _, m in JOIN_TABLE)] for k in JOIN_ROUTINGS]
    notes = [
        "mAP = auc_pr, HIT = hit_rate@1, F1 = f1@1 (micro), each on the row's own set (A = a, V = v, AV = av)",
        *(f"mu {MU_LABELS[name]} ({name}): {spec.description.removeprefix('vggsounder 1.0 ')}" for name, spec in JOIN_METRICS.items()),
    ]
    return {"header": header, "rows": rows, "notes": notes}


def join(bench: Any, results: Mapping[str, Any]) -> dict:
    """Score three readings of one model with the official package (setup.md).

    ``results = {"a": ref, "v": ref, "av": ref}`` where ``ref`` is a probe seed directory (or an object
    with ``seed_dir`` / ``path``) holding ``result.json``, ``DONE``, ``eval_logits.npy`` and
    ``eval_rows.npy``. Refuses (``JoinRefused``, exit 2) unless all three are ``ok``, are the audio-only,
    video-only and audio-visual readings (``describe()["input"]`` = audio / video / both) of one ``model``
    under one standard probe (one ``probe.hash``), carry one ``protocol_hash`` equal to
    ``bench.protocol_hash``, one ``eval_hash`` and one ``coverage.eval.ids_sha256``. Different non-null
    ``weights_sha256`` are the warning ``W_JOIN_WEIGHTS``, never a refusal. A ``ref`` that carries the
    (possibly upgraded) record as ``.result`` is read from it, else from ``result.json``. Returns
    ``{"scores": {set: {metric: value}}`` (``compute_all_results``' own output: the 30 ``PACKAGE_METRICS`` on
    each of the five ``PACKAGE_SETS``, plus ``mu`` on ``a`` / ``v`` / ``av``), ``"table"`` (the compact view),
    ``"n_eval", "protocol_hash", "eval_hash", "eval_ids_sha256", "model", "readings": {routing: reading},
    "describe_sha256": {routing: sha}, "warnings": [{"code", "message", "where"}], "external": {"vggsounder":
    version}}`` as fractions.
    """
    missing = [k for k in JOIN_ROUTINGS if k not in results]
    if missing:
        raise JoinRefused(
            "E_JOIN_MISSING", f"join vggsounder needs a=RUN v=RUN av=RUN; missing {missing}"
        )
    dirs = {k: _seed_dir(results[k]) for k in JOIN_ROUTINGS}
    recs = {k: _record(results[k], d) for k, d in dirs.items()}
    for k, r in recs.items():
        if r.get("status") != "ok":
            raise JoinRefused(
                "E_JOIN_STATUS", f"reading {k}: status {r.get('status')!r}, need 'ok'"
            )
    warnings = _check_readings(recs)
    hashes = {(r.get("benchmark") or {}).get("protocol_hash") for r in recs.values()}
    if len(hashes) != 1:
        raise JoinRefused(
            "E_JOIN_MIXED",
            f"the three readings carry different protocol_hash values: {sorted(map(str, hashes))}",
        )
    eval_hashes = {r.get("eval_hash") for r in recs.values()}
    if len(eval_hashes) != 1:
        raise JoinRefused(
            "E_JOIN_MIXED",
            f"the three readings carry different eval_hash values: {sorted(map(str, eval_hashes))}",
        )
    ph = hashes.pop()
    if getattr(bench, "protocol_hash", ph) != ph:
        raise JoinRefused(
            "E_JOIN_MIXED",
            f"results were scored under protocol_hash {ph}, this benchmark is {bench.protocol_hash}",
        )
    ids_sha = {
        ((r.get("coverage") or {}).get("eval") or {}).get("ids_sha256") for r in recs.values()
    }
    if len(ids_sha) != 1 or None in ids_sha:
        raise JoinRefused(
            "E_JOIN_POPULATION",
            f"eval ids_sha256 differ between readings: {sorted(map(str, ids_sha))}",
        )

    eval_splits = list(bench.roles()["eval"]) if hasattr(bench, "roles") else ["test"]
    all_ids: list[str] = []
    for s in eval_splits:
        all_ids.extend(str(c) for c in bench.manifest(s).clip_id)
    payload: dict[str, dict] = {}
    row_sets = []
    for k, d in dirs.items():
        logits = np.load(d / "eval_logits.npy", mmap_mode="r")
        rows = np.load(d / "eval_rows.npy").astype(np.int64).ravel()
        if logits.ndim != 2 or logits.shape[0] != rows.size:
            raise JoinRefused(
                "E_JOIN_SHAPE", f"reading {k}: eval_logits {logits.shape} vs {rows.size} eval rows"
            )
        row_sets.append(np.sort(rows))
        for i, row in enumerate(rows.tolist()):
            if not 0 <= row < len(all_ids):
                raise JoinRefused(
                    "E_JOIN_SHAPE",
                    f"reading {k}: eval row {row} out of range ({len(all_ids)} eval clips)",
                )
            payload.setdefault(all_ids[row].removesuffix(".mp4"), {"logits": {}})["logits"][k] = (
                np.asarray(logits[i]).astype(float).tolist()
            )
    if any(not np.array_equal(row_sets[0], r) for r in row_sets[1:]):
        raise JoinRefused("E_JOIN_POPULATION", "the three readings score different eval rows")
    version = require_vggsounder(bench)  # after the refusals above (exit 2), before scoring (exit 3)
    ids = list(payload)
    res = official_scores(payload, ids, PACKAGE_SETS)
    for s in PACKAGE_SETS:
        _check_package_result(res[s], bench)
    scores = {s: {k: (float(v) if np.ndim(v) == 0 else v) for k, v in res[s].items()} for s in PACKAGE_SETS}
    return {
        "scores": scores,
        "table": _join_table(scores),
        "n_eval": len(ids),
        "protocol_hash": ph,
        "eval_ids_sha256": ids_sha.pop(),
        "ids_sha256_scored": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "eval_hash": eval_hashes.pop(),
        "model": _model(recs["av"]),
        "readings": {k: recs[k].get("reading") or _describe(recs[k]).get("reading") for k in JOIN_ROUTINGS},
        "describe_sha256": {k: (recs[k].get("bridge") or {}).get("describe_sha256") for k in JOIN_ROUTINGS},
        "warnings": warnings,
        "runs": {k: str(d) for k, d in dirs.items()},
        "external": {"vggsounder": version},
    }
