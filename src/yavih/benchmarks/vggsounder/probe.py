"""vggsounder probe hooks: an external scorer (the folder's setup.md).

Every vggsounder number is computed by the authors' package, ``vggsounder`` (PyPI 1.0), never by a
yavih re-implementation. The package is NOT a yavih dependency: it is installed independently
(setup.md "## Dependencies") and reached only through the assert-import of ``benchmark.yaml``
``external`` (``yavih.external.require``). Module scope imports stdlib, numpy and yavih core only.

Hooks:

- ``METRICS``: ``auc_pr_<set>``, ``auc_roc_<set>`` (audible, visible, all) and ``auc_pr_<set>_floor``,
  each naming ``external: vggsounder``; they replace the registry's native multi_sets versions for
  this benchmark. ``n_eval`` stays the registry's disclosure.
- ``score(bench, split, rows, outputs, targets)``: this bridge's eval logits ``[n, 309]`` (rows in
  ``classes.csv`` order) are written under the pkl keys ``a``, ``v`` and ``all`` and scored by
  ``vggsounder.benchmark.compute_all_results`` (the function behind ``benchmark()``) against the
  package's ground truth ``a`` (A + AV labels = yavih ``audible``), ``v`` (V + AV = ``visible``) and
  ``all`` (A + V + AV = ``all``). The package's ``av`` ground truth is the AV-only label set, which is
  NOT yavih's ``all``; it is filled with the same
  logits only because the package's ``mu`` step needs it, and its numbers are discarded. The package
  scores each set over the clips holding at least one label of that set (``a``: 11 911, ``v``:
  10 182, ``all``: 12 008 of the 12 008 eval clips) and macro-averages over classes with a positive.
- Floors: ``score_floors`` scores the constant train-prevalence row (train targets only)
  through the same call; a driver may equivalently call ``score`` with that matrix and read the
  ``auc_pr_<set>`` values as ``auc_pr_<set>_floor`` (``MetricSpec.floor_of``).
- ``join(bench, results)``: ``report --join vggsounder a=RUN v=RUN av=RUN`` — the three readings'
  ``eval_logits.npy`` under their own pkl keys, scored with ``auc_pr``, ``auc_roc`` and ``mu`` per
  routing exactly as the paper's table does.
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
    "PROTOCOL_KEYS",
    "SET_TO_PACKAGE",
    "JOIN_ROUTINGS",
    "JOIN_INPUTS",
    "ExternalPackageError",
    "JoinRefused",
    "external_entry",
    "require_vggsounder",
    "official_scores",
    "score",
    "score_floors",
    "join",
]

#: yavih ``label.sets`` name → the package's ground-truth modality key (verified equal on the 12 008
#: eval clips, tests/test_folder.py::test_package_ground_truth_equals_label_sets).
SET_TO_PACKAGE: dict[str, str] = {"audible": "a", "visible": "v", "all": "all"}
#: pkl keys of ``report --join`` (the three readings of one model), the paper's routings.
JOIN_ROUTINGS: tuple[str, ...] = ("a", "v", "av")
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
            f"{self.name} is computed by the vggsounder folder's probe.py score() with the official "
            "vggsounder package; it has no registry function"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"vggsounder.probe.score:{self.name}"


def _spec(name: str, *, kind: str = "metric", floor_of: tuple[str, ...] = ()) -> MetricSpec:
    return MetricSpec(
        name=name,
        fn=_ComputedByScore(name),
        readouts=frozenset({"clf"}),
        protocol_keys=frozenset(),
        shape="scalar",
        kind=kind,
        higher_is_better=True,
        floor_of=floor_of,
        external="vggsounder",
    )


METRICS: dict[str, MetricSpec] = {}
for _set in SET_TO_PACKAGE:
    METRICS[f"auc_pr_{_set}"] = _spec(f"auc_pr_{_set}")
    METRICS[f"auc_roc_{_set}"] = _spec(f"auc_roc_{_set}")
    METRICS[f"auc_pr_{_set}_floor"] = _spec(
        f"auc_pr_{_set}_floor", kind="floor", floor_of=(f"auc_pr_{_set}",)
    )
del _set


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


# ----------------------------------------------------------------------------- the official call


def official_scores(
    payload: Mapping[str, Any], subset_ids: Sequence[str], modalities: Sequence[str]
) -> dict:
    """``vggsounder.benchmark.compute_all_results`` on one model's pkl payload
    ``{video_id: {"logits": {modality: [C floats]}}}``; returns ``{modality: {metric: value}}``
    (fractions, as the package computes them before ``benchmark()``'s ×100 display scaling)."""
    from vggsounder.benchmark import compute_all_results
    from vggsounder.labels import VGGSounder

    with tempfile.TemporaryDirectory(prefix="yavih-vggsounder-") as tmp:
        with open(os.path.join(tmp, f"{_MODEL}.pkl"), "wb") as fh:
            pickle.dump(dict(payload), fh, protocol=pickle.HIGHEST_PROTOCOL)
        res = compute_all_results(
            models_path=tmp,
            subset_ids=set(subset_ids),  # membership only; a set keeps it O(n)
            modalities=list(modalities),
            vggsounder=VGGSounder(**_VARIANT),
            verbose=False,
        )
    return {
        m: {k: (float(v) if np.ndim(v) == 0 else v) for k, v in d.items()}
        for m, d in res[_MODEL].items()
    }


def _eval_ids(bench: Any, split: str, rows: Any) -> list[str]:
    ids = list(bench.manifest(split).clip_id)
    return [str(ids[int(r)]) for r in np.asarray(rows).ravel()]


def _single_reading_payload(ids: Sequence[str], outputs: np.ndarray, keys: Sequence[str]) -> dict:
    """One reading's logits under several pkl keys; one list object per clip (pickle memoizes it)."""
    out: dict[str, dict] = {}
    for i, cid in enumerate(ids):
        vec = outputs[i].astype(float).tolist()
        out[cid.removesuffix(".mp4")] = {"logits": {k: vec for k in keys}}
    return out


def _check_outputs(outputs: Any, n: int) -> np.ndarray:
    out = np.asarray(outputs)
    if out.ndim != 2 or out.shape[0] != n or out.shape[1] != 309:
        raise ValueError(f"vggsounder score: outputs must be [n_eval={n}, 309], got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError("vggsounder score: non-finite logits")
    return out


def score(bench: Any, split: str, rows: Any, outputs: Any, targets: Any = None) -> dict[str, float]:
    """``{auc_pr_<set>, auc_roc_<set>}`` for audible / visible / all from the official package.

    ``outputs`` = the head's raw logits ``[len(rows), 309]`` in ``classes.csv`` order (identical to the
    package's ``classes.csv``, sha256 ``90a5cd19…``); ``targets`` is unused: the ground truth is the
    package's own (equal to the manifest's ``label.sets``, test-pinned).
    """
    require_vggsounder(bench)
    ids = _eval_ids(bench, split, rows)
    out = _check_outputs(outputs, len(ids))
    pkg_keys = ("a", "v", "av", "all")  # av: needed by the package's mu step only, discarded
    res = official_scores(_single_reading_payload(ids, out, pkg_keys), ids, pkg_keys)
    metrics: dict[str, float] = {}
    for set_name, key in SET_TO_PACKAGE.items():
        r = res[key]
        if "auc_pr" not in r:
            raise ExternalPackageError(
                "E_EXTERNAL_VERSION",
                "the installed vggsounder build reports no 'auc_pr' (a development build renamed it 'mAP'); "
                f"install the pinned release: {external_entry(bench)['pip']}",
            )
        metrics[f"auc_pr_{set_name}"] = float(r["auc_pr"])
        metrics[f"auc_roc_{set_name}"] = float(r["auc_roc"])
    return metrics


def score_floors(bench: Any, split: str, rows: Any, train_prevalence: Any) -> dict[str, float]:
    """``auc_pr_<set>_floor``: the constant per-class train prevalence ``[309]`` (from TRAIN targets
    only) as every eval clip's score, through the same official call."""
    prev = np.asarray(train_prevalence, dtype=np.float64).ravel()
    n = len(np.asarray(rows).ravel())
    vals = score(bench, split, rows, np.broadcast_to(prev, (n, prev.size)))
    return {f"auc_pr_{s}_floor": vals[f"auc_pr_{s}"] for s in SET_TO_PACKAGE}


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
    ``{"auc_pr": {a, v, av}, "auc_roc": {...}, "mu": {...}, "n_eval", "protocol_hash", "eval_hash",
    "eval_ids_sha256", "model", "readings": {routing: reading}, "describe_sha256": {routing: sha},
    "warnings": [{"code", "message", "where"}], "external": {"vggsounder": version}}`` as fractions.
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
    res = official_scores(payload, ids, JOIN_ROUTINGS)
    out: dict[str, Any] = {
        m: {k: float(res[k][m]) for k in JOIN_ROUTINGS} for m in ("auc_pr", "auc_roc", "mu")
    }
    out.update(
        n_eval=len(ids),
        protocol_hash=ph,
        eval_ids_sha256=ids_sha.pop(),
        ids_sha256_scored=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        eval_hash=eval_hashes.pop(),
        model=_model(recs["av"]),
        readings={k: recs[k].get("reading") or _describe(recs[k]).get("reading") for k in JOIN_ROUTINGS},
        describe_sha256={k: (recs[k].get("bridge") or {}).get("describe_sha256") for k in JOIN_ROUTINGS},
        warnings=warnings,
        runs={k: str(d) for k, d in dirs.items()},
        external={"vggsounder": version},
    )
    return out
