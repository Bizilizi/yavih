"""Readout driver: roles, eligibility (READ from ``_meta.json``, never
computed here), refusals, the folder ``probe.py`` hooks (targets, metrics, external scorer), the
external-package checks (``yavih.external.require``, before training), scratch staging, signals, the
per-seed loop, ``result.json`` + ``DONE`` + ``eval_logits.npy`` + ``eval_rows.npy`` + ``probe_fit.json``,
then the invocation's results-store file and, with ``log_samples``, its samples file.
Re-exports ``RESULT_SCHEMA_VERSION``.

Torch-free on purpose: everything that touches tensors lives in
``yavih.readouts._fit`` and the per-readout modules, imported lazily by :func:`probe` once the
refusal checks have passed. Collaborators (``Benchmark``, ``FeatureReader``, the metric registry)
are resolved at call time through small module-level hooks so tests can substitute fakes.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from yavih import hashing
from yavih.versions import (
    FEATURE_FORMAT_VERSION,
    READOUT_VERSION,
    RESULT_SCHEMA_VERSION,
    __version__,
)

if TYPE_CHECKING:  # pragma: no cover
    from yavih.probes.config import ProbeConfig

__all__ = [
    "probe",
    "ProbeResult",
    "ProbeRefused",
    "ProbeInterrupted",
    "ResultSchemaError",
    "ReadoutContext",
    "StopFlag",
    "RESULT_SCHEMA_VERSION",
    "READOUT_VERSION",
    "RESULT_STATUSES",
    "RESULT_KEYS",
    "DECIDED_BY",
    "write_result",
    "check_result",
    "result_dir",
    "read_done",
    "utc_now",
    "ids_sha256",
    "resolve_deadline",
    "plan_time_missing",
    "evaluate_metrics",
    "resolve_metric_specs",
    "stage_features",
    "sweep_scratch",
    "mirror_features",
    "resolve_results_root",
    "STALE_SCRATCH_SECONDS",
    "probe_not_applicable",
    "request_stop",
    "check_meta_fresh",
    "folder_probe",
    "check_externals",
    "output_requires",
]

RESULT_STATUSES: frozenset[str] = frozenset({"ok", "not_applicable", "failed"})
#: Top-level keys of the ``result.json`` record. ``metrics/floors/controls/ci95`` exist only for ``status == ok``
#: (metric keys for a non-ok status are ABSENT); ``warnings`` carries probe-time findings
#: such as ``W_COVERAGE``; ``sample`` (``{split: [lo, hi]}``) exists only on a result of a capped run
#: (``plan build --sample N``): never stored, every report row from it is ``partial``.
RESULT_KEYS: frozenset[str] = frozenset(
    {
        "schema_version", "status", "reason", "standard", "eval_hash", "benchmark", "model", "reading", "bridge", "features",
        "probe", "metrics", "ci95", "floors", "controls", "chance", "coverage", "not_applicable",
        "timing", "env", "created", "warnings", "external", "waived", "sample",
    }
)
_METRIC_BLOCKS: tuple[str, ...] = ("metrics", "floors", "controls", "ci95")
_BENCHMARK_KEYS = frozenset({"name", "protocol_version", "readout", "readout_version", "manifest_hash", "protocol_hash"})
#: The result's ``bridge`` block: recorded, never a key.
_BRIDGE_KEYS = frozenset({"name", "describe", "describe_sha256", "source_sha256"})
#: Charsets of ``describe()["model"]`` / ``["reading"]`` (they key paths, results files and report rows).
#: (first character a letter or digit: never ``_stale`` or a ``.``-name, which the report would not see)
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_READING_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
#: The warning a result carries when ``describe()["weights_sha256"]`` is null (recorded, nothing else).
W_WEIGHTS_UNFINGERPRINTED = "W_WEIGHTS_UNFINGERPRINTED"
#: The result's ``features`` block (decoders are recorded in ``describe()``): the declared batch tolerance and the
#: rules validate re-checked.
_FEATURES_KEYS = frozenset({"fingerprint", "dtype", "N", "D", "pos_mode", "bytes", "backend", "canary_max_rel_l2",
                            "batch_tolerance", "rules_checked"})
_PROBE_KEYS = frozenset({"hash", "config", "seed", "selected_lr", "epochs_run", "best_epoch", "stop", "n_train", "n_val"})
_COVERAGE_ROLE_KEYS = frozenset({"present", "nominal", "ids_sha256"})
_TIMING_KEYS = frozenset({"validate_s", "materialize_s", "fit_s", "score_s", "wall_s", "s_per_epoch"})
_ENV_KEYS = frozenset({"host", "gpu", "torch", "python", "yavih", "slurm_job_id"})
_NA_KEYS = frozenset({"missing", "reason", "decided_by"})
DECIDED_BY: frozenset[str] = frozenset({"validate", "plan", "bridge"})
#: A sibling scratch directory older than this is swept whatever its pid says (36 h).
STALE_SCRATCH_SECONDS: int = 36 * 3600
#: Scratch preflight factor (``scratch_gb = 1.1 * bytes_train``).
SCRATCH_FACTOR: float = 1.1
DONE_FILE = "DONE"
RESULT_FILE = "result.json"
FIT_FILE = "probe_fit.json"
LOGITS_FILE = "eval_logits.npy"
ROWS_FILE = "eval_rows.npy"
NOT_APPLICABLE_FILE = "NOT_APPLICABLE.json"
META_FILE = "_meta.json"


# ----------------------------------------------------------------------------- errors / records


class ProbeRefused(Exception):
    """A probe refusal (exit 2 unless stated): train ∩ eval ≠ ∅, low coverage, mixed hashes, missing
    ``_meta.json``, ``validation.ok == false``, ``metrics.list`` names not all produced. ``exit_code``
    3 = not ready (features absent, scratch short), 1 = usage (non-pinned cfg outside testing)."""

    def __init__(self, code: str, message: str, exit_code: int = 2) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.exit_code = exit_code


class ProbeInterrupted(TimeoutError):
    """Deadline reached or SIGTERM/SIGUSR1 received (exit 4): the
    checkpoint is on disk, the scratch copy is gone, nothing partial is left under the seed dir."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.exit_code = 4


class ResultSchemaError(ValueError):
    """``result.json`` violates its schema (a bug in the readout driver, exit 5)."""


@dataclass
class ProbeResult:
    """One ``result.json`` plus where it was written."""

    status: str  # "ok" | "not_applicable" | "failed"
    seed: int
    path: Path | None  # probe/<eval_hash12>/seed<k>/result.json
    result: dict[str, Any] = field(default_factory=dict)  # the full result.json record
    reason: str | None = None
    skipped: bool = False  # a matching DONE was already there
    #: Samples-file lines of this seed (``log_samples`` only; ``None`` for a skipped or non-ok seed)
    samples: list[dict] | None = field(default=None, repr=False)
    #: the invocation's results-store file and samples file, set after the DONE markers
    results_file: Path | None = None
    samples_file: Path | None = None

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self.result.get("metrics") or {})


class StopFlag:
    """Set by the SIGTERM/SIGUSR1 handler; the trainer checks ``is_set()`` between micro-batches."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.signal: int | None = None

    def set(self, signum: int | None = None) -> None:
        self.signal = signum
        self._event.set()

    def is_set(self) -> bool:
        # A process-wide request (``request_stop``, set by ``yavih exec``'s own signal handler for an
        # in-process probe unit) stops every live probe as if its handler had fired.
        if self._event.is_set():
            return True
        if self is not _STOP_REQUESTED and _STOP_REQUESTED._event.is_set():
            self.signal = _STOP_REQUESTED.signal
            return True
        return False

    def clear(self) -> None:
        self.signal = None
        self._event.clear()

    def __bool__(self) -> bool:
        return self.is_set()


#: The cooperative process-wide stop request (signals; ``yavih exec``'s in-process probe).
_STOP_REQUESTED: StopFlag = StopFlag()


def request_stop(signum: int | None = None) -> None:
    """Ask every running ``probe()`` in this process to stop between micro-batches (checkpoint, remove
    scratch, exit 4). ``yavih exec`` calls this from its SIGTERM/SIGINT/SIGUSR1 handler when the probe
    runs in-process; ``probe()`` also installs its own handlers when it owns the
    main thread."""
    _STOP_REQUESTED.set(signum)


# ----------------------------------------------------------------------------- small helpers


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ids_sha256(clip_ids: Iterable[str]) -> str:
    """``sha256("\\n".join(sorted present clip_ids))`` (``coverage.ids_sha256``)."""
    return hashlib.sha256("\n".join(sorted(clip_ids)).encode("utf-8")).hexdigest()


def result_dir(run_dir: str | Path, protocol_hash: str, probe_hash: str, seed: int) -> Path:
    """``<run_dir>/probe/<eval_hash12>/seed<k>/`` (``eval_hash`` folds the two
    hashes: what yavih controls). The full ``protocol_hash``, ``probe_hash`` and ``eval_hash`` are in
    ``result.json`` and DONE."""
    return Path(run_dir) / "probe" / hashing.hash12(hashing.eval_hash(protocol_hash, probe_hash)) / f"seed{int(seed)}"


def _write_json_atomic(path: Path, obj: Any) -> bytes:
    data = json.dumps(obj, indent=1, sort_keys=False, ensure_ascii=False).encode("utf-8")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return data


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_done(seed_dir: str | Path) -> dict | None:
    """The DONE marker of a seed directory, or ``None``."""
    p = Path(seed_dir) / DONE_FILE
    if not p.is_file():
        return None
    try:
        return _read_json(p)
    except (OSError, ValueError):
        return None


def resolve_deadline(deadline: float | str | None, *, env: Mapping[str, str] | None = None, margin: float = 900.0) -> float | None:
    """Unix epoch seconds or ``None``: ``yavih.deadline.parse_deadline``, shared with
    ``exec`` and ``extract``. ``None`` falls back to ``$YAVIH_DEADLINE`` (set by ``exec``); a misconfigured
    deadline (``slurm`` inside a job without ``SLURM_JOB_END_TIME``, an epoch before 2001) raises ``DeadlineError``
    (exit 1)."""
    from yavih.deadline import parse_deadline

    return parse_deadline(deadline, env=env, margin_s=margin)


def _json_native(x: Any) -> Any:
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Mapping):
        return {str(k): _json_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_native(v) for v in x]
    return x


def _cfg_dict(cfg: Any) -> dict:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(cfg) and not isinstance(cfg, type):
        return _json_native(asdict(cfg))
    if isinstance(cfg, Mapping):
        return _json_native(dict(cfg))
    return _json_native({k: getattr(cfg, k) for k in ("readout", "head", *hashing.PROBE_TRAIN_FIELDS)})


def env_block() -> dict:
    """The result's ``env`` block. torch/gpu are read from an already-imported torch only, so a
    ``not_applicable`` result never loads torch."""
    torch = sys.modules.get("torch")
    gpu = None
    tv = None
    if torch is not None:
        tv = getattr(torch, "__version__", None)
        try:
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - driver quirks
            gpu = None
    return {
        "host": socket.gethostname(),
        "gpu": gpu,
        "torch": tv,
        "python": platform.python_version(),
        "yavih": __version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


# ----------------------------------------------------------------------------- result.json schema check / writer


def _expect_keys(block: Any, keys: frozenset[str], where: str, *, exact: bool = True) -> None:
    if not isinstance(block, Mapping):
        raise ResultSchemaError(f"{where}: expected an object, got {type(block).__name__}")
    got = set(block)
    if exact and got != keys:
        raise ResultSchemaError(f"{where}: keys {sorted(got ^ keys)} differ from the result schema (missing {sorted(keys - got)}, extra {sorted(got - keys)})")
    if not exact and not keys <= got:
        raise ResultSchemaError(f"{where}: missing keys {sorted(keys - got)}")


def check_result(result: Mapping[str, Any], *, expect: Mapping[str, Any] | None = None) -> None:
    """Raise ``ResultSchemaError`` unless ``result`` is a valid ``result.json`` record. ``expect`` =
    ``{"metrics": [...], "floors": [...], "controls": [...], "ci": {name: kind|None}}`` from
    ``benchmark.yaml`` pins the exact key sets of the metric blocks for ``status == ok``."""
    if not isinstance(result, Mapping):
        raise ResultSchemaError("result must be an object")
    keys = set(result)
    unknown = keys - RESULT_KEYS
    if unknown:
        raise ResultSchemaError(f"unknown result keys {sorted(unknown)}")
    required = RESULT_KEYS - set(_METRIC_BLOCKS) - {"warnings", "chance", "waived", "sample"}
    missing = required - keys
    if missing:
        raise ResultSchemaError(f"missing result keys {sorted(missing)}")
    if result["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResultSchemaError(f"schema_version {result['schema_version']!r} != {RESULT_SCHEMA_VERSION}")
    status = result["status"]
    if status not in RESULT_STATUSES:
        raise ResultSchemaError(f"status {status!r} not in {sorted(RESULT_STATUSES)}")
    if status == "ok" and result["reason"] is not None:
        raise ResultSchemaError("reason must be null for status ok")
    if status != "ok" and not isinstance(result["reason"], str):
        raise ResultSchemaError("reason must be a string for a non-ok status")
    if not isinstance(result["standard"], bool):
        raise ResultSchemaError("standard must be a bool")
    _expect_keys(result["benchmark"], _BENCHMARK_KEYS, "benchmark")
    if result["benchmark"]["readout"] not in READOUT_VERSION:
        raise ResultSchemaError(f"benchmark.readout {result['benchmark']['readout']!r} unknown")
    if result["benchmark"]["readout_version"] != READOUT_VERSION[result["benchmark"]["readout"]]:
        raise ResultSchemaError("benchmark.readout_version differs from yavih.versions.READOUT_VERSION")
    for key, pattern in (("model", _MODEL_RE), ("reading", _READING_RE)):
        v = result[key]
        if not isinstance(v, str) or not pattern.match(v) or v in (".", ".."):
            raise ResultSchemaError(f"{key} {v!r} must match {pattern.pattern} (describe()[{key!r}])")
    eh = result["eval_hash"]
    want = hashing.eval_hash(str(result["benchmark"]["protocol_hash"]), str(result["probe"]["hash"])) if isinstance(result.get("probe"), Mapping) else None
    if not isinstance(eh, str) or not _HEX64_RE.match(eh) or eh != want:
        raise ResultSchemaError("eval_hash must equal sha256(canon({protocol_hash, probe_hash}))")
    _expect_keys(result["bridge"], _BRIDGE_KEYS, "bridge")
    _expect_keys(result["probe"], _PROBE_KEYS, "probe")
    _expect_keys(result["timing"], _TIMING_KEYS, "timing")
    _expect_keys(result["env"], _ENV_KEYS, "env")
    if not isinstance(result["created"], str):
        raise ResultSchemaError("created must be a UTC timestamp string")
    ext = result["external"]
    if not isinstance(ext, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in ext.items()):
        raise ResultSchemaError("external must be an object {package: version} ({} when none)")
    if "sample" in result:
        smp = result["sample"]
        if not isinstance(smp, Mapping) or not smp or not all(
                isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v) for v in smp.values()):
            raise ResultSchemaError("sample must be a non-empty object {split: [lo, hi]} (a capped run's rows)")

    na = result["not_applicable"]
    plan_time = False
    if status == "not_applicable":
        _expect_keys(na, _NA_KEYS, "not_applicable")
        if na["decided_by"] not in DECIDED_BY:
            raise ResultSchemaError(f"not_applicable.decided_by {na['decided_by']!r} not in {sorted(DECIDED_BY)}")
        if not isinstance(na["missing"], list):
            raise ResultSchemaError("not_applicable.missing must be a list")
        plan_time = na["decided_by"] == "plan"
    elif na is not None:
        raise ResultSchemaError("not_applicable must be null unless status == not_applicable")

    feats = result["features"]
    cov = result["coverage"]
    if (feats is None or cov is None) and not plan_time:
        # Null only for a plan-time not_applicable; every other record knows its features.
        raise ResultSchemaError("features/coverage may be null only for a plan-time not_applicable result")
    if feats is not None:
        _expect_keys(feats, _FEATURES_KEYS, "features")
    if cov is not None:
        if not isinstance(cov, Mapping) or not {"train", "eval", "floor"} <= set(cov):
            raise ResultSchemaError("coverage must have train/eval/floor")
        for role, block in cov.items():
            if block is None:
                continue
            if role == "folds":
                for f, b in block.items():
                    _expect_keys(b, _COVERAGE_ROLE_KEYS, f"coverage.folds.{f}")
            else:
                _expect_keys(block, _COVERAGE_ROLE_KEYS, f"coverage.{role}")

    if status == "ok":
        for block in _METRIC_BLOCKS:
            if not isinstance(result.get(block), Mapping):
                raise ResultSchemaError(f"{block} must be an object for status ok")
        if not isinstance(result.get("chance"), Mapping):
            raise ResultSchemaError("chance must be an object for status ok")
        if expect is not None:
            want = {
                "metrics": set(expect.get("metrics", ())),
                "floors": set(expect.get("floors", ())),
                "controls": set(expect.get("controls", ())),
                "ci95": {k for k, v in (expect.get("ci") or {}).items() if v},
            }
            for block, names in want.items():
                got = set(result[block])
                if got != names:
                    raise ResultSchemaError(
                        f"{block} keys {sorted(got)} must equal benchmark.yaml exactly {sorted(names)}"
                    )
        for name, ci in result["ci95"].items():
            if not (isinstance(ci, list) and len(ci) == 2):
                raise ResultSchemaError(f"ci95[{name}] must be [lo, hi]")
    else:
        present = [b for b in _METRIC_BLOCKS if b in result]
        if present:
            raise ResultSchemaError(f"metric blocks {present} must be ABSENT for status {status!r}")


def write_result(
    out_dir: str | Path,
    result: dict,
    *,
    features_fingerprint: str | None,
    expect: Mapping[str, Any] | None = None,
) -> Path:
    """Schema-check the result record, write ``result.json`` atomically, then ``DONE`` for
    ``status ∈ {ok, not_applicable}``: ``result_sha256``, the three hashes, ``model``, ``reading``,
    ``describe_sha256`` and ``features_fingerprint`` (what a later probe compares to decide staleness).
    A ``failed`` status never writes DONE. Returns the result path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    check_result(result, expect=expect)
    done = out / DONE_FILE
    if done.exists():
        done.unlink()
    data = _write_json_atomic(out / RESULT_FILE, result)
    if result["status"] in ("ok", "not_applicable"):
        marker = {
            "result_sha256": hashlib.sha256(data).hexdigest(),
            "protocol_hash": result["benchmark"]["protocol_hash"],
            "probe_hash": result["probe"]["hash"],
            "eval_hash": result["eval_hash"],
            "model": result["model"],
            "reading": result["reading"],
            "describe_sha256": (result.get("bridge") or {}).get("describe_sha256"),
            "features_fingerprint": features_fingerprint,
            "yavih_version": __version__,
            "created": utc_now(),
        }
        _write_json_atomic(done, marker)
    return out / RESULT_FILE


# ----------------------------------------------------------------------------- collaborator hooks


def open_reader(split_dir: str | Path) -> Any:
    """``yavih.io.FeatureReader(split_dir)``; resolved at call time so tests can fake it."""
    from yavih.io import reader as _reader

    return _reader.FeatureReader(split_dir)


def load_benchmark(ref: str) -> Any:
    """``Benchmark.load(ref)``, resolved at call time."""
    from yavih import benchmark as _bm

    return _bm.Benchmark.load(ref)


def pinned_config(readout: str, bench: Any) -> Any:
    from yavih.probes.config import ProbeConfig

    return ProbeConfig.pinned(readout, bench)


class _LazyModule:
    """A module imported on first attribute access: a readout module keeps its torch code (``_fit``) behind it, so
    importing the module for its ``READOUT`` descriptor stays torch-free (``yavih.readouts.registry``)."""

    __slots__ = ("_name", "_mod")

    def __init__(self, name: str) -> None:
        self._name = name
        self._mod: Any = None

    def __getattr__(self, attr: str) -> Any:
        if attr in _LazyModule.__slots__:
            raise AttributeError(attr)
        if self._mod is None:
            self._mod = importlib.import_module(self._name)
        return getattr(self._mod, attr)


def lazy_module(name: str) -> Any:
    return _LazyModule(name)


_FOLDER_PROBES: dict[str, Any] = {}


def folder_probe(bench: Any) -> Any | None:
    """The benchmark folder's optional ``probe.py`` (hooks ``METRICS``, ``PROTOCOL_KEYS``,
    ``targets``, ``score``, ``join``), or ``None``. A ``Benchmark`` that already loaded
    it (``bench.probe_module``) is used as is; otherwise the file is loaded by path (a folder is not a
    package) under a name keyed by its sha256, so an edited file is a new module."""
    mod = getattr(bench, "probe_module", None)
    if callable(mod) and not hasattr(mod, "__file__"):
        mod = mod()
    if mod is not None:
        return mod
    folder = getattr(bench, "folder", None)
    if folder is None:
        return None
    path = Path(folder) / "probe.py"
    if not path.is_file():
        return None
    data = path.read_bytes()
    key = hashlib.sha256(data).hexdigest()
    if key not in _FOLDER_PROBES:
        import importlib.util

        name = f"yavih_folder_probe_{getattr(bench, 'name', 'bench')}_{key[:12]}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - a file that exists always has a loader
            raise ProbeRefused("E_PROBE_PY", f"cannot load {path}", exit_code=1)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            sys.modules.pop(name, None)
            raise ProbeRefused("E_PROBE_PY", f"{path} failed to import ({type(e).__name__}: {e}); probe.py imports only stdlib, numpy and yavih core at module scope", exit_code=1) from e
        _FOLDER_PROBES[key] = module
    return _FOLDER_PROBES[key]


def resolve_metric_specs(spec: Any, probe_mod: Any | None = None) -> dict[str, dict[str, Any]]:
    """``{"metrics": {name: MetricSpec}, "floors": ..., "controls": ...}`` resolved through the registry
    overlaid by the folder ``probe.py``'s ``METRICS``: the driver calls every listed fn (or
    the folder's external scorer for a spec naming ``external``) and never hard-codes a metric name."""
    from yavih.metrics import registry

    sets = list(spec.manifest.label.sets) if getattr(spec.manifest.label, "sets", None) else None
    readout = spec.readout
    folder_metrics: Mapping[str, Any] = dict(getattr(probe_mod, "METRICS", None) or {})

    def resolve(names: Iterable[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in names:
            if name in folder_metrics:
                out[name] = folder_metrics[name]
            else:
                out.update(registry.resolve_names([name], readout, sets=sets))
        return out

    return {
        "metrics": resolve(spec.metrics.list),
        "floors": resolve(spec.metrics.floors),
        "controls": resolve(spec.metrics.controls),
    }


def output_requires(spec: Any) -> Any:
    """``output_contract.requires``."""
    oc = getattr(spec, "output_contract", None)
    req = oc.get("requires") if isinstance(oc, Mapping) else getattr(oc, "requires", None)
    if isinstance(req, Mapping):
        from types import SimpleNamespace

        req = SimpleNamespace(pos=tuple(req.get("pos") or ()), streams=tuple(req.get("streams") or ()), grid=req.get("grid"), t_bins=req.get("t_bins"))
    return req


def _entry(e: Any, key: str) -> Any:
    return e.get(key) if isinstance(e, Mapping) else getattr(e, key, None)


def _installed_version(name: str, module: Any = None) -> str:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", None) or getattr(sys.modules.get(name), "__version__", None) or "unknown")


def check_externals(spec: Any) -> dict[str, str]:
    """Assert-import every ``benchmark.yaml[external]`` entry BEFORE training: absent →
    ``E_EXTERNAL_MISSING``, a version ≠ ``version_pin`` → ``E_EXTERNAL_VERSION`` (both exit 3, "not
    ready", the message carries the ``pip`` line). Returns ``{name: installed version}``, which goes to
    ``result.json[external]`` and the results file's ``config.external``."""
    entries = list(getattr(spec, "external", None) or [])
    if not entries:
        return {}
    from yavih.external import require as _require  # the one assert-import
    from yavih.external import versions_used as _versions_used

    out: dict[str, str] = {}
    for e in entries:
        name, pip, pin = str(_entry(e, "name")), str(_entry(e, "pip")), str(_entry(e, "version_pin"))
        try:
            module = _require(name, pip, pin)
        except ProbeRefused:
            raise
        except Exception as ex:  # yavih.external.ExternalMissing / ExternalVersion (exit 3)
            code = getattr(ex, "code", None) or ("E_EXTERNAL_VERSION" if "Version" in type(ex).__name__ else "E_EXTERNAL_MISSING")
            raise ProbeRefused(str(code), str(ex), exit_code=int(getattr(ex, "exit_code", 3) or 3)) from ex
        # require() records the identity it accepted (dist version, or "<pin>+<sha8> (git checkout)" for a checkout
        # without dist metadata); fall back to metadata / __version__ only for a stand-in that records nothing.
        out[name] = _versions_used().get(name) or _installed_version(name, module)
    return out


def interval_fn() -> Any:
    from yavih.metrics import intervals

    return intervals.interval


# ----------------------------------------------------------------------------- metric evaluation


def _coerce_value(name: str, spec: Any, value: Any) -> Any:
    if spec.shape == "list":
        if value is None or isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            raise ProbeRefused("E_METRICS_INCOMPLETE", f"metric {name!r} must return a list (shape: list)")
        return [float(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        raise ProbeRefused("E_METRICS_INCOMPLETE", f"metric {name!r} returned None")
    v = float(value)
    if spec.kind == "disclosure" and float(v).is_integer():
        return int(v)
    return v


def evaluate_metrics(
    specs: Mapping[str, Mapping[str, Any]],
    *,
    outputs: Any,
    targets: Any,
    protocol: Mapping[str, Any],
    split_meta: Mapping[str, Any],
    floor_outputs: Any,
    ci: Mapping[str, str | None] | None = None,
    scorer: Any = None,
) -> dict[str, dict[str, Any]]:
    """Call every registered fn named in ``metrics.list/floors/controls``:
    metrics and controls see the head's raw eval outputs; floors see the label-only constant
    ``floor_outputs`` (scored through the same metric function on the eval rows).
    A spec naming ``external`` is computed by the folder's scorer instead (``scorer(outputs) ->
    {name: value}``, the ``probe.py`` ``score`` hook bound to the eval rows and targets): once on
    the head's outputs, once on the floor outputs; a floor takes its own name from the second call, or
    the value of the metric its ``floor_of`` names. Any fn that fails or returns nothing is a
    refusal (``metrics.list`` names not all produced)."""
    protocol = dict(protocol)
    out: dict[str, dict[str, Any]] = {"metrics": {}, "floors": {}, "controls": {}, "ci95": {}}
    failures: list[str] = []
    scored: dict[str, Mapping[str, Any]] = {}

    def external_value(name: str, spec: Any, block: str, inputs: Any) -> Any:
        if scorer is None:
            raise ProbeRefused("E_METRICS_INCOMPLETE", f"metric {name!r} names external {spec.external!r} but the folder probe.py has no score() hook")
        which = "floor" if block == "floors" else "model"
        if which not in scored:
            scored[which] = dict(scorer(inputs) or {})
        vals = scored[which]
        if name in vals:
            return vals[name]
        for target in tuple(getattr(spec, "floor_of", ()) or ()):
            if target in vals:
                return vals[target]
        raise ProbeRefused("E_METRICS_INCOMPLETE", f"the folder scorer returned no value for {name!r} (got {sorted(vals)})")

    for block, inputs in (("metrics", outputs), ("controls", outputs), ("floors", floor_outputs)):
        for name, spec in specs.get(block, {}).items():
            try:
                if getattr(spec, "external", None):
                    value = external_value(name, spec, block, inputs)
                else:
                    value = spec.fn(inputs, targets, protocol, split_meta)
                out[block][name] = _coerce_value(name, spec, value)
            except ProbeRefused as e:
                failures.append(f"{name}: {e.message}")
            except Exception as e:  # noqa: BLE001 - every failure is reported, then refused
                failures.append(f"{name}: {type(e).__name__}: {e}")
    if failures:
        raise ProbeRefused("E_METRICS_INCOMPLETE", "metric functions failed: " + "; ".join(failures))
    for name, kind in (ci or {}).items():
        if not kind:
            continue
        spec = next((specs[b][name] for b in ("metrics", "controls", "floors") if name in specs.get(b, {})), None)
        try:
            lo, hi = interval_fn()(kind, name, outputs, targets, protocol, split_meta, spec=spec)
        except Exception as e:  # noqa: BLE001
            raise ProbeRefused("E_METRICS_INCOMPLETE", f"interval {name!r} ({kind}) failed: {type(e).__name__}: {e}") from e
        out["ci95"][name] = [float(lo), float(hi)]
    return out


# ----------------------------------------------------------------------------- scratch


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_scratch(scratch_root: str | Path, *, host: str | None = None, now: float | None = None) -> list[Path]:
    """Remove sibling ``<unit_id>.<host>.<pid>`` directories whose pid is dead (same host) or that are
    older than 36 h (``os.kill(pid, 0)`` is host-local, so a foreign host's directory is judged by age only)."""
    root = Path(scratch_root)
    if not root.is_dir():
        return []
    host = host or socket.gethostname()
    now = time.time() if now is None else now
    removed: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        parts = d.name.rsplit(".", 2)
        if len(parts) != 3 or not parts[2].isdigit():
            continue
        _, d_host, d_pid = parts
        try:
            age = now - d.stat().st_mtime
        except OSError:
            continue
        stale = age > STALE_SCRATCH_SECONDS
        if d_host == host and int(d_pid) != os.getpid() and not _pid_alive(int(d_pid)):
            stale = True
        if stale:
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
    return removed


class _ScratchCopy:
    """A staged copy of ``features/`` that removes itself on exit, SIGTERM and SIGINT."""

    def __init__(self, path: Path, *, owned: bool) -> None:
        self.path = path
        self.owned = owned
        self._removed = False
        if owned:
            atexit.register(self.remove)

    def remove(self) -> None:
        if self.owned and not self._removed:
            self._removed = True
            shutil.rmtree(self.path, ignore_errors=True)


def stage_features(
    features_root: Path,
    splits: Iterable[str],
    *,
    scratch_root: Path,
    unit_id: str,
    needs_bytes: int,
    stop_flag: StopFlag | None = None,
) -> _ScratchCopy:
    """Copy shards + sidecars + ``_meta.json`` of ``splits`` to ``<scratch>/<unit_id>.<host>.<pid>/features/``
    after pre-flighting free space (exit 3 when short) and sweeping stale siblings. A stop
    request (``stop_flag``) between files, or any failure mid-copy, removes the partial copy before the
    exception propagates (``ProbeInterrupted`` for a signal)."""
    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    sweep_scratch(scratch_root)
    free = shutil.disk_usage(scratch_root).free
    if free < needs_bytes:
        raise ProbeRefused(
            "E_SCRATCH_SPACE",
            f"scratch {scratch_root} has {free / 2**30:.1f} GiB free, probe needs {needs_bytes / 2**30:.1f} GiB",
            exit_code=3,
        )
    # One copy routine for the package (``yavih features stage`` uses the same): shards, sidecars,
    # _meta.json / NOT_APPLICABLE.json, never in-flight tmps, to <scratch>/<unit_id>.<host>.<pid>/features/.
    from yavih.io.stage import StageError, staged_name
    from yavih.io.stage import stage_features as _copy_features

    partial = scratch_root / staged_name(Path(features_root), unit_id)
    should_stop = (lambda: stop_flag.is_set()) if stop_flag is not None else None
    try:
        dest_features = _copy_features(features_root, scratch_root, unit_id=unit_id, splits=list(splits), should_stop=should_stop)
    except StageError as e:
        shutil.rmtree(partial, ignore_errors=True)
        if e.code == "E_PREEMPTED":
            raise ProbeInterrupted("E_PREEMPTED", f"signal {getattr(stop_flag, 'signal', None)} received while staging features; scratch copy removed") from e
        raise ProbeRefused(e.code, str(e), exit_code=3) from e
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return _ScratchCopy(dest_features.parent, owned=True)


def _ram_budget_bytes() -> int:
    """RAM budget: ``yavih.resources.ram_budget_bytes`` (the one implementation)."""
    from yavih.resources import ram_budget_bytes

    return int(ram_budget_bytes())


# ----------------------------------------------------------------------------- plan-time eligibility


def plan_time_missing(spec: Any, profile: Mapping[str, Any]) -> list[str]:
    """What ``plan`` can rule out from a ``profile.json`` alone: ``N == 1`` or ``pos_mode == none``
    for a dense readout (→ the required ``pos.<axis>`` entries), and a ``requires.streams`` name absent
    from ``profile.streams`` (→ ``stream.<name>``). Empty list = nothing rules the readout out."""
    requires = output_requires(spec)
    missing: list[str] = []
    from yavih.readouts import registry

    dense = spec.readout in registry.names() and registry.get(spec.readout).dense
    n = profile.get("N")
    pos_mode = profile.get("pos_mode")
    if dense and (n == 1 or pos_mode in (None, "none")):
        missing.extend(f"pos.{axis}" for axis in requires.pos)
    streams = profile.get("streams") or []
    names = set(streams if isinstance(streams, (list, tuple)) else list(streams))
    for s in requires.streams:
        if s not in names:
            missing.append(f"stream.{s}")
    # dedupe, keep order
    seen: set[str] = set()
    return [m for m in missing if not (m in seen or seen.add(m))]


# ----------------------------------------------------------------------------- context


@dataclass
class ReadoutContext:
    """Everything a ``run_<readout>(ctx)`` needs. Readers are opened lazily and
    shared across seeds; features are consumed only via ``rows()``/``matrix()``."""

    bench: Any
    spec: Any
    readout: str
    cfg: Any
    seed: int
    run_dir: Path
    out_dir: Path
    features_root: Path
    metas: dict[str, dict]
    roles: dict[str, list[str]]
    protocol: dict
    metric_specs: dict[str, dict[str, Any]]
    chance: dict
    ci: dict[str, str | None]
    bootstrap: dict | None
    describe: dict
    pos_mode: str
    N: int
    D: int
    dtype: str
    deadline: float | None = None
    stop_flag: StopFlag = field(default_factory=StopFlag)
    device: str | None = None
    loader_workers: int | None = None
    timing: dict = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    readers: dict[str, Any] = field(default_factory=dict)
    manifests: dict[str, Any] = field(default_factory=dict)
    fit_json: dict = field(default_factory=dict)
    probe_mod: Any = None  # the folder probe.py, or None
    log_samples: bool = False  # the readout returns its scored rows for the samples file
    external: dict = field(default_factory=dict)  # {package: version} checked before training

    # -- helpers used by every readout
    def reader(self, split: str) -> Any:
        if split not in self.readers:
            self.readers[split] = open_reader(self.features_root / split)
        return self.readers[split]

    def present_rows(self, split: str) -> np.ndarray:
        return np.asarray(self.reader(split).present_rows, dtype=np.int64)

    def manifest(self, split: str) -> Any:
        if split not in self.manifests:
            self.manifests[split] = self.bench.manifest(split)
        return self.manifests[split]

    def clip_ids(self, split: str, rows: np.ndarray) -> list[str]:
        m = self.manifest(split)
        idx = _manifest_index(m, rows)
        return [m.clip_ids[i] for i in idx] if hasattr(m, "clip_ids") else [m.clip_id[i] for i in idx]

    @property
    def ckpt_path(self) -> Path:
        return self.out_dir / "probe_ckpt.pt"

    @property
    def features_fingerprint(self) -> str | None:
        """The features fingerprint the result/DONE will carry; part of every checkpoint's resume gate."""
        return _fingerprint_of(self.metas)

    @property
    def label_kind(self) -> str:
        return self.spec.manifest.label.kind

    def hook_targets(self, split: str, rows: np.ndarray) -> Any | None:
        """The folder ``probe.py`` ``targets(bench, split, rows)`` hook: replaces the readout's
        target construction (same shape/dtype); ``None`` when the folder defines none."""
        fn = getattr(self.probe_mod, "targets", None)
        if fn is None:
            return None
        return fn(self.bench, split, np.asarray(rows, dtype=np.int64))

    def scorer(self, splits: Sequence[str], rows: np.ndarray, targets: Any) -> Any:
        """The folder ``score(bench, split, rows, outputs, targets)`` hook bound to the eval rows, for the
        metric specs that name ``external``; ``None`` when the folder defines none.
        Several eval splits are passed joined with ``+``."""
        fn = getattr(self.probe_mod, "score", None)
        if fn is None:
            return None
        split = "+".join(splits)
        rows = np.asarray(rows, dtype=np.int64)
        return lambda outputs: fn(self.bench, split, rows, outputs, targets)

    def stream_names(self) -> list[str]:
        from yavih.readouts.coords import stream_names

        return stream_names(self.describe)

    def durations(self, split: str, rows: np.ndarray) -> np.ndarray:
        """``float64 [n, S]`` ``media.duration_s`` per declared stream."""
        from yavih.readouts.coords import row_durations

        return row_durations(self.reader(split), rows, self.stream_names(), split_dir=self.features_root / split)

    def check_interrupt(self) -> None:
        if self.stop_flag.is_set():
            raise ProbeInterrupted("E_PREEMPTED", f"signal {self.stop_flag.signal} received")
        if self.deadline is not None and time.time() >= self.deadline:
            raise ProbeInterrupted("E_DEADLINE", "deadline reached")


def _manifest_index(manifest: Any, rows: np.ndarray) -> np.ndarray:
    """Positions in the manifest table of the given within-split rows (``row`` is
    the 0-based index within the split in file order, so this is the identity unless the manifest is reordered)."""
    rows = np.asarray(rows, dtype=np.int64)
    mrows = np.asarray(manifest.row, dtype=np.int64)
    if mrows.shape[0] == 0:
        return rows[:0]
    if np.array_equal(mrows, np.arange(mrows.shape[0])):
        return rows
    order = np.argsort(mrows, kind="stable")
    pos = np.searchsorted(mrows[order], rows)
    pos = np.clip(pos, 0, mrows.shape[0] - 1)
    if not np.array_equal(mrows[order][pos], rows):
        raise ProbeRefused("E_ROWS", "feature rows not found in the manifest split")
    return order[pos]


# ----------------------------------------------------------------------------- the entry point


def _read_meta(split_dir: Path) -> dict | None:
    p = split_dir / META_FILE
    return _read_json(p) if p.is_file() else None


def _read_na(split_dir: Path) -> dict | None:
    p = split_dir / NOT_APPLICABLE_FILE
    return _read_json(p) if p.is_file() else None


def check_meta_fresh(split_dir: str | Path, meta: Mapping[str, Any]) -> None:
    """Refuse (``E_META_STALE``) a ``_meta.json`` that no longer describes the shards next to it.

    The probe may READ ``eligibility`` and ``validation.ok`` from ``_meta.json`` instead of
    re-deriving them; that is only sound while the file still belongs to the sidecars on disk. A
    re-extraction into the same run directory moves the old shards to ``_stale/`` and writes new ones
    without touching ``_meta.json``, so the features fingerprint is recomputed from the live sidecars
    (``yavih.io.finalize.meta_fingerprint``, the same formula validate's cheap re-check uses) and every
    sidecar's ``model`` / ``reading`` / ``bridge.describe_sha256`` (the configuration) is compared
    with the meta's. A directory without any
    sidecar has nothing to compare (the reader then yields no rows); resolved at call time so tests can
    substitute a fake world.
    """
    from yavih.io.finalize import meta_fingerprint
    from yavih.io.reader import list_sidecars
    from yavih.io.sidecar import read_sidecar

    split_dir = Path(split_dir)
    try:
        sidecars = [read_sidecar(p) for p in list_sidecars(split_dir)]
    except Exception as e:  # noqa: BLE001 - an unreadable sidecar is validate's job to name
        raise ProbeRefused("E_META_STALE", f"{split_dir}: a shard sidecar is unreadable ({type(e).__name__}: {e}); re-run `yavih features validate`") from e
    if not sidecars:
        return
    from yavih.io.sidecar import identity_value

    def config(rec: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return rec.get("model"), rec.get("reading"), identity_value(dict(rec), "bridge.describe_sha256")

    meta_cfg = config(meta)
    for sc in sidecars:
        sc_cfg = config(sc)
        if sc_cfg != meta_cfg:
            raise ProbeRefused(
                "E_META_STALE",
                f"{split_dir}: shard {sc.get('shard', {}).get('index')} was written by {sc_cfg[0]} {sc_cfg[1]} (describe "
                f"{str(sc_cfg[2])[:12]}) but _meta.json describes {meta_cfg[0]} {meta_cfg[1]} (describe {str(meta_cfg[2])[:12]}): "
                "the split was re-extracted after validation; re-run `yavih features validate`",
            )
    try:
        fp = meta_fingerprint(sidecars, pos_sha256=(meta.get("features") or {}).get("pos_sha256"), n_present=(meta.get("coverage") or {}).get("present"))
    except (KeyError, TypeError, ValueError) as e:
        raise ProbeRefused("E_META_STALE", f"{split_dir}: cannot recompute the features fingerprint from the sidecars ({e}); re-run `yavih features validate`") from e
    if fp != meta.get("fingerprint"):
        raise ProbeRefused(
            "E_META_STALE",
            f"{split_dir}: _meta.json fingerprint {str(meta.get('fingerprint'))[:12]}… ≠ {fp[:12]}… recomputed from the sidecars on disk: "
            "shards changed after validation; re-run `yavih features validate`",
        )


def _roles(spec: Any) -> dict[str, list[str]]:
    s = spec.splits
    return {
        "train": list(s.train),
        "eval": list(s.eval),
        "floor": list(s.floor or ()),
        "folds": list(s.folds or ()),
    }


def _role_splits(roles: Mapping[str, list[str]]) -> list[str]:
    out: list[str] = []
    for role in ("train", "eval", "floor", "folds"):
        for s in roles[role]:
            if s not in out:
                out.append(s)
    return out


def _agree(metas: Mapping[str, dict], path: Sequence[str], code: str) -> Any:
    values: dict[str, Any] = {}
    for split, meta in metas.items():
        if tuple(path) == ("benchmark", "input_hash"):
            # a recorded input hash mapped through the folder changelog's supersedes_input
            from yavih.io.sidecar import identity_value

            values[split] = identity_value(meta, ".".join(path))
            continue
        v: Any = meta
        for k in path:
            v = v.get(k) if isinstance(v, Mapping) else None
        values[split] = v
    distinct = {json.dumps(v, sort_keys=True, default=str) for v in values.values()}
    if len(distinct) > 1:
        raise ProbeRefused(code, f"splits disagree on {'.'.join(path)}: {values}")
    return next(iter(values.values()))


def _benchmark_block(bench: Any) -> dict:
    return {
        "name": bench.name,
        "protocol_version": int(bench.spec.protocol_version),
        "readout": bench.spec.readout,
        "readout_version": READOUT_VERSION[bench.spec.readout],
        "manifest_hash": bench.manifest_hash,
        "protocol_hash": bench.protocol_hash,
    }


def _bridge_block(meta_bridge: Mapping[str, Any] | None, *, describe: Mapping[str, Any] | None = None, name: str | None = None) -> dict:
    describe = dict(describe if describe is not None else (meta_bridge or {}).get("describe") or {})
    name = name or (meta_bridge or {}).get("name") or str(describe.get("name", "?"))
    return {
        "name": name,
        "describe": describe,
        "describe_sha256": (meta_bridge or {}).get("describe_sha256") or (hashing.describe_sha256(describe) if describe else None),
        "source_sha256": (meta_bridge or {}).get("source_sha256"),
    }


def _bridge_warnings(bridge: Mapping[str, Any]) -> list[dict]:
    """``W_WEIGHTS_UNFINGERPRINTED`` when the recorded ``describe()["weights_sha256"]`` is null: a
    recorded fact, never identity, never a flag on a report row."""
    describe = bridge.get("describe") if isinstance(bridge.get("describe"), Mapping) else {}
    if "weights_sha256" in describe and describe["weights_sha256"] is None:
        return [{"code": W_WEIGHTS_UNFINGERPRINTED, "message": "describe()['weights_sha256'] is null: the checkpoint is not fingerprinted "
                 "(recorded only; the model name is the user's)", "where": None}]
    return []


def _features_block(metas: Mapping[str, dict], *, backend: str | None = None) -> dict:
    first = next(iter(metas.values()))
    feats = first.get("features", {})
    return {
        # one string, identical to DONE.features_fingerprint: the split's features fingerprint, or the
        # sha256 over the sorted {split: fingerprint} map when the probe spans several splits
        "fingerprint": _fingerprint_of(metas),
        "dtype": feats.get("dtype"),
        "N": feats.get("N"),
        "D": feats.get("D"),
        "pos_mode": feats.get("pos_mode"),
        "bytes": int(sum(int((m.get("features") or {}).get("bytes") or 0) for m in metas.values())),
        "backend": backend,
        "canary_max_rel_l2": max((float((m.get("canary") or {}).get("max_rel_l2") or 0.0) for m in metas.values()), default=None),
        # the declared (or default 1e-5) batch-invariance tolerance validate applied
        "batch_tolerance": _batch_tolerance(metas),
        "rules_checked": _json_native(first.get("rules_checked")) if first.get("rules_checked") is not None else None,
    }


def _batch_tolerance(metas: Mapping[str, dict]) -> float | None:
    tols = [(m.get("canary") or {}).get("tolerance") for m in metas.values()]
    tols = [float(t) for t in tols if isinstance(t, (int, float)) and not isinstance(t, bool)]
    return max(tols) if tols else None


def _waived_of(metas: Mapping[str, dict]) -> list[dict]:
    """``result.json`` ``waived``: the validation waivers in force over the probe's splits, one entry per
    ``(code, source)`` with the summed ``count`` and the splits it covers. Recorded and marked in the report; never
    greyed, excluded or penalized."""
    by: dict[tuple[str, str], dict] = {}
    for split, m in sorted(metas.items()):
        for w in m.get("waived") or []:
            if not isinstance(w, Mapping) or not w.get("code"):
                continue
            key = (str(w["code"]), str(w.get("source") or "describe"))
            e = by.setdefault(key, {"code": key[0], "reason": str(w.get("reason") or ""), "source": key[1], "count": 0, "splits": []})
            e["count"] += int(w.get("count") or 0)
            e["splits"].append(str(split))
    return [by[k] for k in sorted(by)]


def _fingerprint_of(metas: Mapping[str, dict]) -> str | None:
    """One string for DONE: the single split's fingerprint, else sha256 over the sorted per-split ones."""
    fps = {s: m.get("fingerprint") for s, m in sorted(metas.items())}
    if not fps:
        return None
    if len(fps) == 1:
        return next(iter(fps.values()))
    return hashing.sha256_canon(fps)


def _coverage_from_meta(meta: Mapping[str, Any]) -> dict:
    c = meta.get("coverage") or {}
    return {"present": int(c.get("present", 0)), "nominal": int(c.get("nominal", 0)), "ids_sha256": c.get("ids_sha256")}


def _empty_features_block() -> dict:
    """The result's ``features`` block of a bridge-declared ``not_applicable`` without any ``_meta.json``: every
    key present, the geometry null (no shard was ever written)."""
    return {"fingerprint": None, "dtype": None, "N": None, "D": None, "pos_mode": None, "bytes": 0, "backend": None,
            "canary_max_rel_l2": None, "batch_tolerance": None, "rules_checked": None}


def _nominal_rows(bench: Any, split: str) -> int:
    """``Benchmark.n_rows(split)``, else the manifest length, else 0 (unknown)."""
    try:
        return int(bench.n_rows(split))
    except Exception:  # noqa: BLE001 - a partial Benchmark surface (tests): fall back to the manifest
        pass
    try:
        return int(len(bench.manifest(split).row))
    except Exception:  # noqa: BLE001
        return 0


def _coverage_nominal(bench: Any, metas: Mapping[str, dict], splits: Sequence[str]) -> dict | None:
    """Coverage of ``splits`` when the bridge declared the run not applicable: the splits that do have a
    ``_meta.json`` contribute their counts, the others count as nominal rows with nothing present."""
    if not splits:
        return None
    present = nominal = 0
    for s in splits:
        if s in metas:
            c = metas[s].get("coverage") or {}
            present += int(c.get("present", 0))
            nominal += int(c.get("nominal", 0))
        else:
            nominal += _nominal_rows(bench, s)
    sha = (metas[splits[0]].get("coverage") or {}).get("ids_sha256") if len(splits) == 1 and splits[0] in metas else None
    return {"present": present, "nominal": nominal, "ids_sha256": sha}


def _coverage_union(metas: Mapping[str, dict], splits: Sequence[str], ctx: ReadoutContext | None) -> dict | None:
    if not splits:
        return None
    present = sum(int((metas[s].get("coverage") or {}).get("present", 0)) for s in splits)
    nominal = sum(int((metas[s].get("coverage") or {}).get("nominal", 0)) for s in splits)
    if len(splits) == 1 and (ctx is None or (metas[splits[0]].get("coverage") or {}).get("ids_sha256")):
        sha = (metas[splits[0]].get("coverage") or {}).get("ids_sha256")
    elif ctx is not None:
        ids: list[str] = []
        for s in splits:
            ids.extend(ctx.clip_ids(s, ctx.present_rows(s)))
        sha = ids_sha256(ids)
    else:
        sha = hashing.sha256_canon([(metas[s].get("coverage") or {}).get("ids_sha256") for s in splits])
    return {"present": present, "nominal": nominal, "ids_sha256": sha}


def _probe_block(cfg: Any, qh: str, seed: int, fit: Mapping[str, Any] | None) -> dict:
    fit = fit or {}
    return {
        "hash": qh,
        "config": _cfg_dict(cfg),
        "seed": int(seed),
        "selected_lr": fit.get("selected_lr"),
        "epochs_run": fit.get("epochs_run"),
        "best_epoch": fit.get("best_epoch"),
        "stop": fit.get("stop"),
        "n_train": fit.get("n_train"),
        "n_val": fit.get("n_val"),
    }


def _timing(t: Mapping[str, Any], wall_s: float) -> dict:
    return {
        "validate_s": float(t.get("validate_s", 0.0)),
        "materialize_s": float(t.get("materialize_s", 0.0)),
        "fit_s": float(t.get("fit_s", 0.0)),
        "score_s": float(t.get("score_s", 0.0)),
        "wall_s": float(wall_s),
        "s_per_epoch": (float(t["s_per_epoch"]) if t.get("s_per_epoch") is not None else None),
    }


def _expect_blocks(spec: Any) -> dict:
    return {"metrics": list(spec.metrics.list), "floors": list(spec.metrics.floors), "controls": list(spec.metrics.controls), "ci": dict(spec.metrics.ci)}


def _standard(cfg: Any, readout: str, bench: Any) -> tuple[Any, bool]:
    """``cfg=None`` → pinned (standard); any other cfg requires ``YAVIH_TESTING=1`` and is standard only
    when its ``probe_hash`` equals the pinned one."""
    from yavih.probes.config import testing_enabled

    if cfg is None:
        return pinned_config(readout, bench), True
    if not testing_enabled():
        raise ProbeRefused("E_NONSTANDARD_CONFIG", "a non-pinned ProbeConfig requires YAVIH_TESTING=1", exit_code=1)
    if getattr(cfg, "readout", readout) != readout:
        raise ProbeRefused("E_NONSTANDARD_CONFIG", f"cfg.readout {cfg.readout!r} != benchmark readout {readout!r}", exit_code=1)
    try:
        pinned = pinned_config(readout, bench)
        return cfg, hashing.probe_hash(cfg) == hashing.probe_hash(pinned)
    except NotImplementedError:
        return cfg, False


def _move_stale(seed_dir: Path, run_dir: Path) -> None:
    """Stale outputs go to ``probe/_stale/<utc>/<eval_hash12>/seed<k>/``."""
    if not seed_dir.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = run_dir / "probe" / "_stale" / stamp / seed_dir.parent.name / seed_dir.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(seed_dir), str(dest))


def _done_matches(done: Mapping[str, Any] | None, *, protocol_hash: str, probe_hash: str, model: str, reading: str,
                  describe_sha256: str | None, features_fingerprint: str | None) -> bool:
    """The probe's own DONE gate: DONE's ``protocol_hash``, ``probe_hash``, ``model``, ``reading``,
    ``describe_sha256`` and ``features_fingerprint`` equal the live ones — a reconfigured bridge or features that
    changed under an unchanged configuration are re-probed; anything else is ``skipped`` without training."""
    if not done:
        return False
    return (
        done.get("protocol_hash") == protocol_hash
        and done.get("probe_hash") == probe_hash
        and done.get("model") == model
        and done.get("reading") == reading
        and done.get("describe_sha256") == describe_sha256
        and done.get("features_fingerprint") == features_fingerprint
    )


def _install_signal_handlers(flag: StopFlag) -> list[tuple[int, Any]]:
    """SIGTERM/SIGUSR1 (and SIGINT) set the flag; only possible from the main thread."""
    installed: list[tuple[int, Any]] = []
    if threading.current_thread() is not threading.main_thread():
        return installed

    def _handler(signum: int, _frame: Any) -> None:
        flag.set(signum)

    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        try:
            prev = signal.signal(sig, _handler)
            installed.append((sig, prev))
        except (ValueError, OSError):  # pragma: no cover - not available on this platform/thread
            pass
    return installed


def _restore_signal_handlers(installed: list[tuple[int, Any]]) -> None:
    for sig, prev in installed:
        try:
            signal.signal(sig, prev if prev is not None else signal.SIG_DFL)
        except (ValueError, OSError):  # pragma: no cover
            pass


def _not_applicable_result(
    *,
    bench: Any,
    cfg: Any,
    qh: str,
    seed: int,
    standard: bool,
    model: str,
    reading: str,
    bridge: dict,
    features: dict | None,
    coverage: dict | None,
    missing: list[str],
    reason: str,
    decided_by: str,
    started: float,
    waived: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "not_applicable",
        "reason": reason,
        "standard": standard,
        "eval_hash": hashing.eval_hash(bench.protocol_hash, qh),
        "benchmark": _benchmark_block(bench),
        "model": model,
        "reading": reading,
        "bridge": bridge,
        "features": features,
        "probe": _probe_block(cfg, qh, seed, None),
        "chance": _json_native(dict(bench.spec.metrics.chance)),
        "coverage": coverage,
        "not_applicable": {"missing": list(missing), "reason": reason, "decided_by": decided_by},
        "timing": _timing({}, time.time() - started),
        "env": env_block(),
        "created": utc_now(),
        "warnings": _bridge_warnings(bridge),
        "external": {},
        "waived": list(waived or []),
    }


def _plan_time_probe(
    run_dir: Path,
    *,
    unit: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
    seeds: Sequence[int],
    cfg: Any,
    bench: Any,
    started: float,
) -> list[ProbeResult]:
    """Plan-time ``not_applicable``: no features exist; the result is built from the planned
    profile (``bridge.describe`` from ``profile.describe``, ``model`` and ``reading`` planned, ``features: null``,
    ``coverage: null``) and written under ``runs/<bench>/<model>/<reading>/probe/<eval_hash12>/seed<k>/`` with
    DONE."""
    if profile is None:
        raise ProbeRefused("E_PLAN_PROFILE", "a plan-time not_applicable unit needs the bridge profile (profile.json)", exit_code=1)
    if bench is None:
        bench = load_benchmark(str(unit["benchmark"]))
    readout = bench.spec.readout
    if unit.get("protocol_hash") not in (None, bench.protocol_hash):
        raise ProbeRefused("E_HASH_PROTOCOL", f"unit protocol_hash {unit.get('protocol_hash')} != loaded benchmark {bench.protocol_hash}")
    cfg, standard = _standard(cfg, readout, bench)
    qh = hashing.probe_hash(cfg)
    if unit.get("probe_hash") not in (None, qh):
        raise ProbeRefused("E_HASH_PROBE", f"unit probe_hash {unit.get('probe_hash')} != {qh}")
    if unit.get("eval_hash") not in (None, hashing.eval_hash(bench.protocol_hash, qh)):
        raise ProbeRefused("E_HASH_PROBE", f"unit eval_hash {unit.get('eval_hash')} != {hashing.eval_hash(bench.protocol_hash, qh)}")
    describe = dict(profile.get("describe") or {})
    model = str(unit.get("model") or profile.get("model") or describe.get("model") or "")
    reading = str(unit.get("reading") or profile.get("reading") or describe.get("reading") or "")
    if not model or not reading:
        raise ProbeRefused("E_PLAN_PROFILE", "the planned unit and its profile name no model/reading (re-run `bridge check` "
                           "and `plan build`)", exit_code=1)
    bridge = _bridge_block(
        {"describe_sha256": profile.get("describe_sha256") or unit.get("describe_sha256"), "source_sha256": profile.get("source_sha256")},
        describe=describe,
        name=str(unit.get("bridge") or describe.get("name") or "?"),
    )
    missing = plan_time_missing(bench.spec, profile)
    reason = str(unit.get("reason") or f"planned not_applicable for readout {readout!r}: missing {missing}")
    seed_list = [int(unit["seed"])] if unit.get("seed") is not None else [int(s) for s in seeds]
    results: list[ProbeResult] = []
    for seed in seed_list:
        out = result_dir(run_dir, bench.protocol_hash, qh, seed)
        done = read_done(out)
        if _done_matches(done, protocol_hash=bench.protocol_hash, probe_hash=qh, model=model, reading=reading,
                         describe_sha256=bridge["describe_sha256"], features_fingerprint=None):
            rec = _read_json(out / RESULT_FILE)
            results.append(ProbeResult(rec["status"], seed, out / RESULT_FILE, rec, rec.get("reason"), skipped=True))
            continue
        if done is not None:
            _move_stale(out, run_dir)
        rec = _not_applicable_result(
            bench=bench, cfg=cfg, qh=qh, seed=seed, standard=standard, model=model, reading=reading, bridge=bridge,
            features=None, coverage=None, missing=missing, reason=reason, decided_by="plan", started=started,
        )
        path = write_result(out, rec, features_fingerprint=None)
        results.append(ProbeResult("not_applicable", seed, path, rec, reason))
    return results


def probe(
    run_dir,
    *,
    seeds: Sequence[int] = (0,),
    readout: str | None = None,
    cfg: ProbeConfig | None = None,
    deadline: float | str | None = None,
    scratch: str | None = None,
    allow_low_coverage: bool = False,
    loader_workers: int | None = None,
    log_samples: bool = False,
    results_root: str | os.PathLike | None = None,
    write_results: bool = True,
    bench: Any | None = None,
    unit: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | str | os.PathLike | None = None,
    validate_first: bool = False,
    device: str | None = None,
    out: str | os.PathLike | None = None,
    waive: Mapping[str, str] | None = None,
) -> list[ProbeResult]:
    """The probe entry point. The benchmark is read from ``features/*/_meta.json`` (all splits must
    agree); ``readout`` defaults to ``benchmark.yaml[readout]``; ``cfg=None`` = ``ProbeConfig.pinned``;
    any other ``cfg`` requires ``YAVIH_TESTING=1`` and sets ``result.standard = false``.

    After the DONE markers, unless ``write_results`` is false, one results-store file for the
    invocation is written to ``<results_root>/<model>/results_<timestamp>.json`` (entry
    ``<benchmark>:<reading>``) and, with ``log_samples``, ``samples_<benchmark>_<reading>_<timestamp>.jsonl``
    next to it. The root is decided BEFORE any training (:func:`resolve_results_root`):
    ``results_root`` → ``YAVIH_RESULTS`` / config ``[results] root`` → ``results/`` next to the runs root of a
    standard run directory ``<runs root>/<benchmark>/<model>/<reading>``; any other layout is refused
    (``E_RESULTS_ROOT``, exit 1). Every returned ``ProbeResult`` carries ``results_file`` / ``samples_file``.
    A capped run (``plan build --sample N``: a ``_meta.json`` with ``sample: [lo, hi]``) never writes the store, whatever
    ``write_results`` says, and needs no results root; its ``result.json`` records ``sample: {split: [lo, hi]}``.

    Extra keywords (beyond the documented readout signature): ``bench`` =
    an already-loaded ``Benchmark`` or a folder path/ref (else ``Benchmark.load(<_meta name>)``);
    ``unit``/``profile`` = the plan unit record and its ``profile.json`` for a plan-time
    ``not_applicable``; ``validate_first`` runs ``validate_features`` per split as the prologue
    before reading ``_meta.json``; ``device`` overrides cuda/cpu auto-detection; ``waive`` = an operator's
    validation waivers ``{CODE: reason}`` the prologue passes to ``validate_features`` (``probe run --waive``, the matrix
    entry key ``waive``; recorded with source ``cli``).

    ``out`` (``probe run --out``): probe a feature set that cannot be written (a read-only group store, a
    read-only bind, an archive). The source's shards and sidecars are mirrored into ``<out>/features/<split>/``
    as symlinks (:func:`mirror_features`), and ``<out>`` is the run directory for everything the probe writes:
    the validate prologue's ``_meta.json``, ``probe/<eval_hash12>/seed<k>/``. The source tree is never written;
    the results-root layout check looks at the source run directory.
    """
    state: dict[str, Any] = {"waive": dict(waive or {})}  # an operator's validation waivers for the validate prologue
    src_run = Path(run_dir).absolute()
    target = mirror_features(src_run, Path(out)) if out is not None else src_run
    if write_results:
        state["resolve_results"] = lambda b, model, reading: resolve_results_root(src_run, bench=b, model=model, reading=reading,
                                                                                  explicit=results_root)
    results = _probe(
        target, seeds=seeds, readout=readout, cfg=cfg, deadline=deadline, scratch=scratch,
        allow_low_coverage=allow_low_coverage, loader_workers=loader_workers, log_samples=log_samples,
        bench=bench, unit=unit, profile=profile, validate_first=validate_first, device=device, state=state,
    )
    if write_results and results and not state.get("sample"):  # a capped run's result never reaches the store
        _write_results_store(results, bench=state.get("bench"), results_root=state.get("results_root"), log_samples=log_samples)
    return results


#: Files of a split directory that ``mirror_features`` links (``_meta.json`` is copied, never linked: validate
#: rewrites it in the mirror; tmps, ``_stale/`` and logs are never mirrored).
_MIRRORED = re.compile(r"^(shard-\d+\.(tar|json)|NOT_APPLICABLE\.json)$")


def mirror_features(src_run: str | Path, out: str | Path) -> Path:
    """``probe run --out``: mirror ``<src_run>/features/<split>/`` into ``<out>/features/<split>/``.

    Shards, sidecars and ``NOT_APPLICABLE.json`` become symlinks to the source files (absolute targets); a link
    whose source file is gone, or that points elsewhere, is replaced or dropped, so a re-run after more shards were
    extracted picks them up. A source ``_meta.json`` is copied when the mirror has none (the validate prologue
    rewrites it there; a stale one is refused by ``check_meta_fresh``). Returns ``<out>`` (absolute). Nothing is
    written under ``src_run``."""
    src_run = Path(src_run).absolute()
    out = Path(out).absolute()
    if out == src_run:
        return out
    src_features = src_run / "features"
    dst_features = out / "features"
    dst_features.mkdir(parents=True, exist_ok=True)
    if not src_features.is_dir():
        return out  # the probe refuses E_NO_FEATURES, naming the mirror; nothing to link
    for src_split in sorted(p for p in src_features.iterdir() if p.is_dir() and not p.name.startswith("_")):
        dst_split = dst_features / src_split.name
        dst_split.mkdir(parents=True, exist_ok=True)
        wanted = {p.name: p for p in src_split.iterdir() if _MIRRORED.match(p.name) and p.is_file()}
        for child in list(dst_split.iterdir()):
            if child.is_symlink():
                src = wanted.get(child.name)
                if src is None or os.readlink(child) != os.fspath(src):
                    child.unlink()
        for name, src in sorted(wanted.items()):
            dst = dst_split / name
            if not dst.is_symlink() and not dst.exists():
                os.symlink(os.fspath(src), dst)
        meta_src, meta_dst = src_split / META_FILE, dst_split / META_FILE
        if meta_src.is_file() and not meta_dst.exists():
            shutil.copyfile(meta_src, meta_dst)
    return out


def resolve_results_root(run_dir: str | Path, *, bench: Any, model: str | None, reading: str | None,
                         explicit: str | os.PathLike | None = None) -> Path:
    """The results-store root of one probe invocation, decided before any training: ``explicit``
    (``--results``) → ``YAVIH_RESULTS`` / config ``[results] root`` → ``results/`` next to the runs root of a
    standard run directory ``<runs root>/<benchmark or alias>/<model>/<reading>`` (the absolute path's last three
    components name the feature set being probed). Any other layout is ``E_RESULTS_ROOT`` (exit 1): the root is
    never guessed from directory depth."""
    from yavih.config import configured_results_root
    from yavih.config import results_root as _results_root

    if explicit is not None:
        return Path(explicit).expanduser().absolute()
    configured = configured_results_root()
    if configured is not None:
        return configured.absolute()
    rd = Path(run_dir).absolute()
    names = {getattr(bench, "name", None), *(getattr(getattr(bench, "spec", None), "aliases", None) or ())}
    names.discard(None)
    parts = rd.parts
    standard = (len(parts) >= 5 and model is not None and reading is not None
                and parts[-1] == reading and parts[-2] == model and parts[-3] in names)
    if not standard:
        raise ProbeRefused(
            "E_RESULTS_ROOT",
            f"{rd} is not a standard run directory <runs root>/<benchmark>/<model>/<reading> (here "
            f"{getattr(bench, 'name', '?')}/{model}/{reading}), so the results-store root cannot be derived from it; pass "
            "--results DIR (probe run), set YAVIH_RESULTS or [results] root in the config, or --no-results",
            exit_code=1,
        )
    return _results_root(rd.parents[2])


def _write_results_store(results: list[ProbeResult], *, bench: Any, results_root: str | os.PathLike | None, log_samples: bool) -> None:
    """After every DONE: the invocation's results file (+ samples file) under the root
    :func:`resolve_results_root` decided before training. The store is never read to decide anything, so a
    failure to write it is reported on stderr, never fatal."""
    from yavih.report import store

    if results_root is None:  # the invocation ended before its model/reading were known (nothing ran)
        return
    try:
        paths = store.write_invocation(
            [r.result for r in results],
            result_paths=[str(r.path) if r.path else None for r in results],
            bench=bench,
            results_root=Path(results_root),
            samples=[line for r in results for line in (r.samples or [])] if log_samples else None,
        )
    except Exception as e:  # noqa: BLE001 - the store is never read to decide anything
        print(f"[yavih.probe] could not write the results store under {results_root}: {type(e).__name__}: {e}", file=sys.stderr)
        return
    for r in results:
        r.results_file = paths.get("results_file")
        r.samples_file = paths.get("samples_file")


def _probe(
    run_dir,
    *,
    seeds: Sequence[int],
    readout: str | None,
    cfg: Any,
    deadline: float | str | None,
    scratch: str | None,
    allow_low_coverage: bool,
    loader_workers: int | None,
    log_samples: bool,
    bench: Any | None,
    unit: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | str | os.PathLike | None,
    validate_first: bool,
    device: str | None,
    state: dict[str, Any],
) -> list[ProbeResult]:
    started = time.time()
    run_dir = Path(run_dir)
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ProbeRefused("E_USAGE", "at least one seed is required", exit_code=1)
    if isinstance(profile, (str, os.PathLike)):
        profile = _read_json(Path(profile))
    if bench is not None and not hasattr(bench, "protocol_hash"):
        bench = load_benchmark(str(bench))

    # -- plan-time not_applicable: no features exist by construction
    if unit is not None and unit.get("expected_status") == "not_applicable":
        if bench is None:
            bench = load_benchmark(str(unit["benchmark"]))
        state["bench"] = bench
        if state.get("resolve_results") is not None:
            state["results_root"] = state["resolve_results"](bench, unit.get("model"), unit.get("reading"))
        return _plan_time_probe(run_dir, unit=unit, profile=profile, seeds=seeds, cfg=cfg, bench=bench, started=started)

    features_root = run_dir / "features"
    if not features_root.is_dir():
        raise ProbeRefused("E_NO_FEATURES", f"{features_root} does not exist: extract (and validate) first", exit_code=3)

    split_dirs = sorted(p for p in features_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not split_dirs:
        raise ProbeRefused("E_NO_FEATURES", f"{features_root} has no split directories", exit_code=3)

    # -- benchmark identity from the splits that exist (all splits must agree)
    timing: dict[str, Any] = {}
    bench_name: str | None = None
    if bench is None:
        names: set[str] = set()
        for d in split_dirs:
            meta = _read_meta(d)
            na = _read_na(d)
            if na is not None:  # first: a _meta.json beside it is an earlier configuration's (never read, see below)
                names.add(str(na.get("benchmark")))
            elif meta is not None:
                names.add(str((meta.get("benchmark") or {}).get("name")))
            else:
                for sc in sorted(d.glob("shard-*.json")):
                    try:
                        names.add(str((_read_json(sc).get("benchmark") or {}).get("name")))
                        break
                    except (OSError, ValueError):
                        continue
        names.discard("None")
        if len(names) != 1:
            raise ProbeRefused("E_HASH_MISMATCH", f"split directories do not name exactly one benchmark: {sorted(names)}")
        bench_name = names.pop()
        bench = load_benchmark(bench_name)
    state["bench"] = bench
    spec = bench.spec
    if readout is not None and readout != spec.readout:
        raise ProbeRefused("E_READOUT", f"readout {readout!r} differs from benchmark.yaml[readout] = {spec.readout!r} (one protocol, one readout)", exit_code=1)
    readout = spec.readout
    roles = _roles(spec)
    needed = _role_splits(roles)

    # -- prologue: validate (optional), then read _meta.json / NOT_APPLICABLE.json per role split
    if validate_first:
        from yavih.io.validate import validate_features

        t0 = time.time()
        for s in needed:
            d = features_root / s
            if d.is_dir() and _read_na(d) is None:
                validate_features(d, bench=bench, waive=state.get("waive") or None)
        timing["validate_s"] = time.time() - t0

    metas: dict[str, dict] = {}
    nas: dict[str, dict] = {}
    for s in needed:
        d = features_root / s
        if not d.is_dir():
            role = "eval" if s in roles["eval"] else "train" if s in roles["train"] else "floor" if s in roles["floor"] else "fold"
            raise ProbeRefused("E_SPLIT_MISSING", f"{role} split {s!r} has no features directory {d}")
        na = _read_na(d)
        if na is not None:
            # Validate writes _meta.json only next to shards, and shards never sit beside
            # NOT_APPLICABLE.json, so a _meta.json in this split belongs to an earlier configuration (extract moves
            # it to _stale/ when it writes the verdict). Never read it: its bridge
            # block and fingerprint would make the current verdict look like the old configuration's result.
            nas[s] = na
            continue
        meta = _read_meta(d)
        if meta is None:
            raise ProbeRefused("E_META_MISSING", f"{d / META_FILE} absent: run `yavih features validate` first")
        fv = meta.get("format_version")
        if isinstance(fv, bool) or fv != FEATURE_FORMAT_VERSION:
            raise ProbeRefused("E_FORMAT", f"{s}: unsupported format_version {fv!r} (this yavih reads {FEATURE_FORMAT_VERSION}); re-extract")
        if not (meta.get("validation") or {}).get("ok", False):
            raise ProbeRefused("E_VALIDATION", f"{s}: _meta.json.validation.ok is false; fix the findings first")
        check_meta_fresh(d, meta)
        metas[s] = meta

    if metas:
        _agree(metas, ("benchmark", "name"), "E_HASH_MISMATCH")
        mh = _agree(metas, ("benchmark", "manifest_hash"), "E_HASH_MISMATCH")
        ih = _agree(metas, ("benchmark", "input_hash"), "E_HASH_MISMATCH")
        model = _agree(metas, ("model",), "E_DESCRIBE_MISMATCH")
        reading = _agree(metas, ("reading",), "E_DESCRIBE_MISMATCH")
        for key in ("dtype", "N", "D", "pos_mode"):
            _agree(metas, ("features", key), "E_DESCRIBE_MISMATCH")
        _agree(metas, ("bridge", "describe_sha256"), "E_DESCRIBE_MISMATCH")
        if mh != bench.manifest_hash:
            raise ProbeRefused("E_HASH_MANIFEST", f"features were extracted under manifest_hash {mh}, loaded benchmark has {bench.manifest_hash}")
        if ih != getattr(bench, "input_hash", None):
            # Features are checked against the INPUT side only; a scoring edit never refuses them
            from yavih.hash_history import describe_input_mismatch
            from yavih.versions import writer_stamp

            fm = next(iter(metas.values()))
            ym = fm.get("yavih") or {}
            raise ProbeRefused("E_HASH_INPUT", "features: " + describe_input_mismatch(
                fm.get("benchmark"), want_input_hash=str(getattr(bench, "input_hash", None)),
                want_components=getattr(getattr(bench, "hashes", None), "input_components", None),
                writer={"version": ",".join(ym.get("writer_versions") or []) or None, "commit": ",".join(ym.get("writer_commits") or []) or None},
                reader_version=writer_stamp()["version"], reader_commit=writer_stamp()["commit"]))
        first = next(iter(metas.values()))
        bridge = _bridge_block(first.get("bridge"))
    else:
        first_na = next(iter(nas.values()))
        model, reading = first_na.get("model"), first_na.get("reading")
        bridge = _bridge_block({"describe": first_na.get("bridge"), "describe_sha256": first_na.get("describe_sha256")},
                               name=str((first_na.get("bridge") or {}).get("name", "?")))
    if nas:
        # one configuration across every NOT_APPLICABLE.json and every other split's _meta.json: model, reading AND
        # describe_sha256 (the run directory does not name the configuration, so a split left by an
        # earlier configuration must not lend its verdict, or its features, to this one). A null describe_sha256 (the
        # bridge raised in __init__, before describe()) never agrees with a meta: __init__ sees no split, so a bridge
        # that is not applicable there is not applicable on every split.
        configs = {(n.get("model"), n.get("reading"), n.get("describe_sha256"))
                   for n in nas.values()}
        if metas:
            configs |= {(model, reading, bridge["describe_sha256"])}
        if len(configs) > 1:
            shown = sorted(f"{m}/{r} describe {str(dsha)[:12]}" for m, r, dsha in configs)
            raise ProbeRefused(
                "E_DESCRIBE_MISMATCH",
                f"NOT_APPLICABLE.json / _meta.json name different configurations across splits: {shown}; re-extract every split "
                "with the current bridge (the earlier configuration's files then move to _stale/)",
            )
    if not isinstance(model, str) or not model or not isinstance(reading, str) or not reading:
        raise ProbeRefused("E_FORMAT", f"{features_root}: the feature set names no model/reading: re-extract with this yavih")
    # A capped run (`plan build --sample N`: `_meta.json` `sample: [lo, hi]`) is scored like any other but never
    # reaches the results store; its result records `sample` and every report row from it is `partial`.
    sample = {s: [int(x) for x in m["sample"]] for s, m in metas.items() if isinstance(m.get("sample"), list)}
    state["sample"] = sample
    # The results-store root is decided (or refused, exit 1) before anything is trained
    if state.get("resolve_results") is not None and not sample:
        state["results_root"] = state["resolve_results"](bench, model, reading)

    cfg, standard = _standard(cfg, readout, bench)
    qh = hashing.probe_hash(cfg)
    fingerprint = _fingerprint_of(metas)
    expect = _expect_blocks(spec)
    results: list[ProbeResult] = []

    def _skip_or_stale(seed: int) -> ProbeResult | None:
        out = result_dir(run_dir, bench.protocol_hash, qh, seed)
        done = read_done(out)
        if _done_matches(done, protocol_hash=bench.protocol_hash, probe_hash=qh, model=model, reading=reading,
                         describe_sha256=bridge["describe_sha256"], features_fingerprint=fingerprint):
            rec = _read_json(out / RESULT_FILE)
            return ProbeResult(rec["status"], seed, out / RESULT_FILE, rec, rec.get("reason"), skipped=True)
        if done is not None:
            _move_stale(out, run_dir)
        return None

    # -- bridge-declared not_applicable: the split is "complete", the result is not_applicable.
    # The result schema reserves null features/coverage for the plan-time case, so the record carries what IS known:
    # the manifest's nominal rows with nothing present, and a features block with null geometry.
    if nas:
        na = next(iter(nas.values()))
        reason = str(na.get("reason") or "bridge declared NotApplicable")
        features = _features_block(metas) if metas else _empty_features_block()
        coverage = {role: _coverage_nominal(bench, metas, roles[role]) for role in ("train", "eval", "floor")}
        if roles["folds"]:
            coverage["folds"] = {f: _coverage_nominal(bench, metas, [f]) for f in roles["folds"]}
        for seed in seeds:
            skipped = _skip_or_stale(seed)
            if skipped is not None:
                results.append(skipped)
                continue
            rec = _not_applicable_result(
                bench=bench, cfg=cfg, qh=qh, seed=seed, standard=standard, model=model, reading=reading, bridge=bridge,
                features=features, coverage=coverage, missing=[], reason=reason, decided_by="bridge", started=started,
                waived=_waived_of(metas),
            )
            _mark_sample(rec, sample)
            path = write_result(result_dir(run_dir, bench.protocol_hash, qh, seed), rec, features_fingerprint=fingerprint)
            results.append(ProbeResult("not_applicable", seed, path, rec, reason))
        return results

    # -- eligibility: READ from _meta.json, every split must agree
    elig = _agree(metas, ("eligibility", "eligible"), "E_ELIGIBILITY_MISMATCH")
    if elig is None:
        raise ProbeRefused("E_META_MISSING", "_meta.json has no eligibility block; re-run validate")
    if not bool(elig):
        block = next(iter(metas.values())).get("eligibility") or {}
        missing = list(block.get("missing") or [])
        reason = str(block.get("reason") or f"ineligible for readout {readout!r}: missing {missing}")
        features = _features_block(metas)
        coverage = {
            "train": _coverage_union(metas, roles["train"], None),
            "eval": _coverage_union(metas, roles["eval"], None),
            "floor": _coverage_union(metas, roles["floor"], None),
        }
        if roles["folds"]:
            coverage["folds"] = {f: _coverage_from_meta(metas[f]) for f in roles["folds"]}
        for seed in seeds:
            skipped = _skip_or_stale(seed)
            if skipped is not None:
                results.append(skipped)
                continue
            rec = _not_applicable_result(
                bench=bench, cfg=cfg, qh=qh, seed=seed, standard=standard, model=model, reading=reading, bridge=bridge,
                features=features, coverage=coverage, missing=missing, reason=reason, decided_by="validate", started=started,
                waived=_waived_of(metas),
            )
            _mark_sample(rec, sample)
            path = write_result(result_dir(run_dir, bench.protocol_hash, qh, seed), rec, features_fingerprint=fingerprint)
            results.append(ProbeResult("not_applicable", seed, path, rec, reason))
        return results

    # -- refusals that need only _meta.json; recorded bridge facts first (result warnings)
    warnings: list[dict] = _bridge_warnings(bridge)
    min_cov = float(spec.splits.min_eval_coverage)
    for s in roles["eval"]:
        c = metas[s].get("coverage") or {}
        nominal, present = int(c.get("nominal", 0)), int(c.get("present", 0))
        frac = present / nominal if nominal else 0.0
        if frac < min_cov:
            msg = f"eval split {s!r} coverage {present}/{nominal} = {frac:.4f} < min_eval_coverage {min_cov}"
            if not allow_low_coverage:
                raise ProbeRefused("E_COVERAGE", msg + " (pass --allow-low-coverage to record it as W_COVERAGE)")
            warnings.append({"code": "W_COVERAGE", "message": msg, "where": s})
    if roles["folds"]:
        n_folds = len([f for f in roles["folds"] if f in metas])
        if n_folds != len(spec.splits.folds or ()):
            raise ProbeRefused("E_FOLDS", f"n_folds {n_folds} != len(splits.folds) {len(spec.splits.folds or ())}")

    # -- the folder probe.py hooks and the authors' packages: every external entry
    # is checked BEFORE staging and training, so a missing package costs nothing (exit 3, the pip line).
    probe_mod = folder_probe(bench)
    metric_specs = resolve_metric_specs(spec, probe_mod)
    external = check_externals(spec)

    # -- signals (a stop requested before this unit started belongs to the previous unit). Installed BEFORE
    # the scratch copy: a SIGTERM during a 64 GB staging must reach the handler so the partial copy is
    # removed, not take the default action and leave it for the 36 h sweep.
    _STOP_REQUESTED.clear()
    stop_flag = StopFlag()
    installed = _install_signal_handlers(stop_flag)
    deadline_ts = resolve_deadline(deadline)

    # -- scratch staging: only when bytes_train exceeds the RAM budget
    bytes_train = sum(int((metas[s].get("features") or {}).get("bytes") or 0) for s in (roles["train"] + roles["folds"]))
    staged: _ScratchCopy | None = None
    unit_id = os.environ.get("YAVIH_UNIT_ID") or hashing.unit_id(
        kind="probe", benchmark=bench.name, model=model, reading=reading, split=None, shard=None,
        protocol_hash=bench.protocol_hash, probe_hash=qh, seed=None,
    )
    try:
        if scratch is not None and (Path(scratch) / "features").is_dir():
            # a copy made by `yavih features stage`: use it, never delete it
            staged = _ScratchCopy(Path(scratch), owned=False)
            features_root = Path(scratch) / "features"
        elif bytes_train > _ram_budget_bytes():
            from yavih.config import configured_scratch_root

            root = scratch or configured_scratch_root()  # YAVIH_SCRATCH → [scratch] root → SLURM_TMPDIR, as exec
            if root:
                staged = stage_features(features_root, needed, scratch_root=Path(root), unit_id=unit_id,
                                        needs_bytes=int(SCRATCH_FACTOR * bytes_train), stop_flag=stop_flag)
                features_root = staged.path / "features"
            else:
                warnings.append({"code": "W_SIZE", "message": "bytes_train exceeds the RAM budget and no scratch is configured (YAVIH_SCRATCH / [scratch] root / SLURM_TMPDIR): memmap backend in place", "where": None})
    except BaseException:
        _restore_signal_handlers(installed)
        raise
    protocol = _json_native(dict(spec.protocol))
    feats = next(iter(metas.values())).get("features") or {}
    readers: dict[str, Any] = {}
    manifests: dict[str, Any] = {}

    try:
        from yavih.readouts import registry

        run = registry.get(readout).driver()  # run_<readout> of the module that declares it (may import torch)
        for seed in seeds:
            skipped = _skip_or_stale(seed)
            if skipped is not None:
                results.append(skipped)
                continue
            out = result_dir(run_dir, bench.protocol_hash, qh, seed)
            out.mkdir(parents=True, exist_ok=True)
            ctx = ReadoutContext(
                bench=bench, spec=spec, readout=readout, cfg=cfg, seed=seed, run_dir=run_dir, out_dir=out,
                features_root=features_root, metas=metas, roles=roles, protocol=protocol, metric_specs=metric_specs,
                chance=_json_native(dict(spec.metrics.chance)), ci=dict(spec.metrics.ci), bootstrap=(dict(spec.metrics.bootstrap) if spec.metrics.bootstrap else None),
                describe=bridge["describe"], pos_mode=str(feats.get("pos_mode")), N=int(feats.get("N") or 0), D=int(feats.get("D") or 0),
                dtype=str(feats.get("dtype")), deadline=deadline_ts, stop_flag=stop_flag, device=device, loader_workers=loader_workers,
                timing=dict(timing), warnings=list(warnings), readers=readers, manifests=manifests,
                probe_mod=probe_mod, log_samples=log_samples, external=dict(external),
            )
            seed_started = time.time()
            from yavih.probes.trainer import reset_determinism

            reset_determinism()  # the determinism record is per seed
            # train ∩ eval clip_ids ≠ ∅ is a refusal; checked on the manifest, per seed is cheap
            _check_leakage(ctx)
            try:
                blocks = run(ctx)
            except (ProbeRefused, ProbeInterrupted):
                raise
            except Exception as e:  # noqa: BLE001 - recorded as status failed (no DONE), then re-raised (exit 5)
                failed = _failed_result(ctx, bridge=bridge, model=model, reading=reading, qh=qh, standard=standard, reason=f"{type(e).__name__}: {e}", started=seed_started)
                _mark_sample(failed, sample)
                try:
                    write_result(out, failed, features_fingerprint=fingerprint)
                except ResultSchemaError:  # pragma: no cover - defensive
                    pass
                raise
            ctx.check_interrupt()
            _record_determinism(ctx, blocks)
            rec = _ok_result(ctx, blocks, bridge=bridge, model=model, reading=reading, qh=qh, standard=standard, started=seed_started)
            _mark_sample(rec, sample)
            _write_side_outputs(ctx, blocks)
            path = write_result(out, rec, features_fingerprint=fingerprint, expect=expect)
            # Every checkpoint of this seed (main head, folds, the box shuffled control) is spent once
            # the result is written: none may be resumed by a later fit in this directory.
            for ckpt in out.glob("probe_ckpt*.pt"):
                try:
                    ckpt.unlink()
                except OSError:  # pragma: no cover - best effort
                    pass
            samples = None
            if log_samples:
                from yavih.readouts.samples import sample_lines

                samples = sample_lines(blocks.get("samples") or [], readout=readout, seed=seed, protocol=ctx.protocol)
            results.append(ProbeResult("ok", seed, path, rec, None, samples=samples))
    except ProbeInterrupted:
        if staged is not None:
            staged.remove()
        raise
    finally:
        _restore_signal_handlers(installed)
        if staged is not None:
            staged.remove()
    return results


def probe_not_applicable(
    run_dir,
    *,
    unit: Mapping[str, Any],
    profile: Mapping[str, Any] | str | os.PathLike | None,
    seed: int = 0,
    bench: Any | None = None,
    write_results: bool = True,
) -> list[ProbeResult]:
    """The plan-time ``not_applicable`` probe: ``yavih exec`` calls this for a unit whose
    record carries ``expected_status: not_applicable``. No features exist; the result is built from the
    planned profile and written with DONE. A thin alias of ``probe(run_dir, seeds=(seed,), unit=unit,
    profile=profile)`` so the plan side needs no second code path; ``write_results=False`` (a unit whose ``cmd`` says
    ``--no-results``, e.g. one of ``plan build --sample N``) writes no results-store file."""
    rec = dict(unit)
    rec.setdefault("expected_status", "not_applicable")
    if rec.get("expected_status") != "not_applicable":
        raise ProbeRefused("E_USAGE", "probe_not_applicable needs a unit with expected_status == not_applicable", exit_code=1)
    return probe(run_dir, seeds=(int(seed),), unit=rec, profile=profile, bench=bench, write_results=write_results)


def _record_determinism(ctx: ReadoutContext, blocks: dict) -> None:
    """The determinism the fit actually achieved goes to ``probe_fit.json`` (``fit.determinism``); ops that
    ran without a deterministic kernel (``warn_only``) add ``W_NONDETERMINISTIC`` to the result's warnings."""
    from yavih.probes.trainer import determinism_record

    det = determinism_record(ctx.device)
    fit = blocks.get("fit")
    if isinstance(fit, dict):
        fit["determinism"] = det
    else:
        blocks["fit"] = {"determinism": det}
    if det["nondeterministic_ops"]:
        ctx.warnings.append({"code": "W_NONDETERMINISTIC", "where": det["device"], "message": (
            f"ops without a deterministic kernel ran during training on {det['device']}: {det['nondeterministic_ops']}; "
            "a rerun agrees to float rounding, not bit for bit")})


def _check_leakage(ctx: ReadoutContext) -> None:
    train_ids: set[str] = set()
    for s in ctx.roles["train"]:
        train_ids.update(ctx.clip_ids(s, ctx.present_rows(s)))
    if not train_ids:
        return
    for s in ctx.roles["eval"]:
        leak = train_ids.intersection(ctx.clip_ids(s, ctx.present_rows(s)))
        if leak:
            raise ProbeRefused("E_LEAKAGE", f"{len(leak)} clip_ids are both train and eval (e.g. {sorted(leak)[:3]})")


def _ok_result(ctx: ReadoutContext, blocks: Mapping[str, Any], *, bridge: dict, model: str, reading: str, qh: str, standard: bool, started: float) -> dict:
    backend = blocks.get("backend")
    fit = blocks.get("probe") or {}
    t = dict(ctx.timing)
    t.update(blocks.get("timing") or {})
    coverage = blocks.get("coverage")
    if coverage is None:
        coverage = {
            "train": _coverage_union(ctx.metas, ctx.roles["train"], ctx),
            "eval": _coverage_union(ctx.metas, ctx.roles["eval"], ctx),
            "floor": _coverage_union(ctx.metas, ctx.roles["floor"], ctx),
        }
        if ctx.roles["folds"]:
            coverage["folds"] = {f: _coverage_from_meta(ctx.metas[f]) for f in ctx.roles["folds"]}
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "ok",
        "reason": None,
        "standard": standard,
        "eval_hash": hashing.eval_hash(ctx.bench.protocol_hash, qh),
        "benchmark": _benchmark_block(ctx.bench),
        "model": model,
        "reading": reading,
        "bridge": bridge,
        "features": _features_block(ctx.metas, backend=backend),
        "probe": _probe_block(ctx.cfg, qh, ctx.seed, fit),
        "metrics": _json_native(blocks["metrics"]),
        "ci95": _json_native(blocks.get("ci95") or {}),
        "floors": _json_native(blocks.get("floors") or {}),
        "controls": _json_native(blocks.get("controls") or {}),
        "chance": ctx.chance,
        "coverage": coverage,
        "not_applicable": None,
        "timing": _timing(t, time.time() - started),
        "env": env_block(),
        "created": utc_now(),
        "warnings": list(ctx.warnings),
        # Exactly the packages whose code produced a reported number (checked before training), with the
        # finer identity a folder recorded while scoring (yavih.external.record_version, e.g. Probe3D's commit)
        "external": _external_used(blocks.get("external", ctx.external)),
        # Validation waivers in force (recorded; the report marks the row, nothing is greyed)
        "waived": _waived_of(ctx.metas),
    }


def _mark_sample(rec: dict, sample: Mapping[str, Any]) -> dict:
    """Record a capped run's ``{split: [lo, hi]}`` on the result (``sample``); a full run's result has no such key."""
    if sample:
        rec["sample"] = {str(k): [int(v[0]), int(v[1])] for k, v in sample.items()}
    return rec


def _external_used(base: Mapping[str, Any] | None) -> dict[str, str]:
    """``{package: version}`` checked before training, refined by what the folder recorded while scoring."""
    from yavih.external import versions_used

    out = {str(k): str(v) for k, v in dict(base or {}).items()}
    out.update({k: v for k, v in versions_used().items() if k in out})
    return out


def _failed_result(ctx: ReadoutContext, *, bridge: dict, model: str, reading: str, qh: str, standard: bool, reason: str, started: float) -> dict:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "failed",
        "reason": reason,
        "standard": standard,
        "eval_hash": hashing.eval_hash(ctx.bench.protocol_hash, qh),
        "benchmark": _benchmark_block(ctx.bench),
        "model": model,
        "reading": reading,
        "bridge": bridge,
        "features": _features_block(ctx.metas),
        "probe": _probe_block(ctx.cfg, qh, ctx.seed, None),
        "chance": ctx.chance,
        "coverage": {
            "train": _coverage_union(ctx.metas, ctx.roles["train"], None),
            "eval": _coverage_union(ctx.metas, ctx.roles["eval"], None),
            "floor": _coverage_union(ctx.metas, ctx.roles["floor"], None),
        },
        "not_applicable": None,
        "timing": _timing(ctx.timing, time.time() - started),
        "env": env_block(),
        "created": utc_now(),
        "warnings": list(ctx.warnings),
        "external": {},
        "waived": _waived_of(ctx.metas),
    }


def _write_side_outputs(ctx: ReadoutContext, blocks: Mapping[str, Any]) -> None:
    """``eval_logits.npy`` (float32, the head's raw eval output), ``eval_rows.npy`` (int64) and
    ``probe_fit.json``."""
    out = ctx.out_dir
    logits = blocks.get("eval_outputs")
    if logits is not None:
        np.save(out / LOGITS_FILE, np.ascontiguousarray(np.asarray(logits, dtype=np.float32)))
    rows = blocks.get("eval_rows")
    if rows is not None:
        np.save(out / ROWS_FILE, np.ascontiguousarray(np.asarray(rows, dtype=np.int64)))
    for name, arr in (blocks.get("extra_arrays") or {}).items():
        np.save(out / f"{name}.npy", np.ascontiguousarray(np.asarray(arr)))
    fit = dict(blocks.get("fit") or {})
    fit.setdefault("backend", blocks.get("backend"))
    fit.setdefault("seed", ctx.seed)
    fit.setdefault("readout", ctx.readout)
    _write_json_atomic(out / FIT_FILE, _json_native(fit))
