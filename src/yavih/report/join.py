"""Cross-reading joins through the benchmark folder's ``eval.py`` (vggsounder's eval.py):
``report --join <benchmark> <reading>=RUN ...`` resolves every RUN to one DONE-marked seed directory,
refuses mixed ``eval_hash``es (protocol or probe) or eval populations and readings of more than one
``model`` (``E_JOIN_MODEL``), records differing non-null ``weights_sha256`` as ``W_JOIN_WEIGHTS`` (a bridge
decision: an audio-only reading may fingerprint only the modules it loads), checks the folder's external
packages, and calls the folder's ``join(bench, results: dict[str, ResultRef]) -> dict`` hook — the only
place a join is computed (the authors' scorer lives there, never in yavih). The joined numbers are written
to the model's results directory under ``<benchmark>:a+v+av`` (``<results_root>/<model>/``).
stdlib at module scope; numpy (the logits) and the benchmark loader are imported lazily.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["JoinError", "ResultRef", "locate_result_dir", "load_ref", "run_join"]


class JoinError(Exception):
    """A refused join: exit 2 (missing run / DONE, mixed hashes or eval populations), 1 (usage, no
    ``join`` hook), 3 (the authors' package absent or mis-versioned: not ready)."""

    def __init__(self, code: str, message: str, exit_code: int = 2) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.exit_code = exit_code


def _has_done(seed_dir: Path) -> bool:
    return (seed_dir / "result.json").is_file() and (seed_dir / "DONE").is_file()


def locate_result_dir(run: str | os.PathLike, *, seed: int | None = None) -> Path:
    """Resolve ``RUN`` to the seed directory holding ``result.json`` + ``DONE`` + ``eval_logits.npy``.

    ``RUN`` may be the seed directory itself, a ``probe/<eval_hash12>/`` directory, or a run directory
    ``runs/<bench>/<model>/<reading>/`` (then exactly one probe column must exist, else name it). With
    several seeds the lowest (or ``seed``) is used: a join scores ONE replicate, never an average
    of logits.
    """
    p = Path(run).expanduser()
    if not p.exists():
        raise JoinError("E_JOIN_MISSING", f"run {p} does not exist")
    if _has_done(p):
        return p
    if any(_has_done(s) for s in p.glob("seed*")):
        columns = [p]  # RUN is a probe/<eval_hash12>/ column directory
    else:
        probe = p / "probe" if (p / "probe").is_dir() else p
        columns = sorted(d for d in probe.iterdir() if d.is_dir() and not d.name.startswith("_") and any(_has_done(s) for s in d.glob("seed*")))
    if not columns:
        raise JoinError("E_JOIN_MISSING", f"no probe result with DONE under {p}")
    if len(columns) > 1:
        raise JoinError("E_JOIN_AMBIGUOUS", f"{p} holds {len(columns)} probe columns; name one: " + ", ".join(c.name for c in columns))
    seeds = sorted((s for s in columns[0].glob("seed*") if _has_done(s)), key=lambda s: int(s.name[4:]) if s.name[4:].isdigit() else 0)
    if seed is not None:
        seeds = [s for s in seeds if s.name == f"seed{seed}"]
        if not seeds:
            raise JoinError("E_JOIN_MISSING", f"seed{seed} not found with DONE under {columns[0]}")
    return seeds[0]


@dataclass
class ResultRef:
    """One reading's scored result handed to a folder ``join()`` hook: the DONE-marked ``result.json``
    record and lazy access to the probe outputs ``eval_logits.npy`` / ``eval_rows.npy``."""

    reading: str
    seed_dir: Path
    result: dict[str, Any]
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def model(self) -> str:
        """``result["model"]``: the user's name for the model."""
        return str(self.result.get("model") or "")

    @property
    def eval_hash(self) -> str:
        return str(self.result.get("eval_hash") or "")

    @property
    def describe(self) -> dict[str, Any]:
        d = (self.result.get("bridge") or {}).get("describe")
        return d if isinstance(d, dict) else {}

    @property
    def eval_logits_path(self) -> Path:
        return self.seed_dir / "eval_logits.npy"

    @property
    def eval_rows_path(self) -> Path:
        return self.seed_dir / "eval_rows.npy"

    def eval_logits(self) -> Any:
        """``float32 [n_eval, ...]`` memmapped (the head's raw eval output)."""
        import numpy as np

        if "logits" not in self._cache:
            if not self.eval_logits_path.is_file():
                raise JoinError("E_JOIN_MISSING", f"{self.seed_dir}: eval_logits.npy missing (a probe output)")
            self._cache["logits"] = np.load(self.eval_logits_path, mmap_mode="r")
        return self._cache["logits"]

    def eval_rows(self) -> Any:
        """``int64 [n_eval]`` within-split manifest rows of the eval role, in the logits' order."""
        import numpy as np

        if "rows" not in self._cache:
            if not self.eval_rows_path.is_file():
                raise JoinError("E_JOIN_MISSING", f"{self.seed_dir}: eval_rows.npy missing (a probe output)")
            self._cache["rows"] = np.load(self.eval_rows_path)
        return self._cache["rows"]


def load_ref(reading: str, run: str | os.PathLike, *, seed: int | None = None) -> ResultRef:
    from yavih.report.migrate import migrate

    seed_dir = locate_result_dir(run, seed=seed)
    result = migrate(json.loads((seed_dir / "result.json").read_text(encoding="utf-8")))
    if result.get("status") != "ok":
        raise JoinError("E_JOIN_STATUS", f"{seed_dir}: status {result.get('status')!r}, need 'ok'")
    return ResultRef(reading=str(reading), seed_dir=seed_dir, result=result)


def _one_model(refs: Mapping[str, ResultRef]) -> str:
    """The joined readings' single ``model`` (else ``E_JOIN_MODEL``)."""
    models = {r.model for r in refs.values()}
    if len(models) != 1 or "" in models:
        raise JoinError("E_JOIN_MODEL", f"the readings are not one model: models {sorted(models)} (a join combines the readings of ONE "
                        "model name)")
    return models.pop()


def _weights_warnings(refs: Mapping[str, ResultRef]) -> list[dict[str, Any]]:
    """``W_JOIN_WEIGHTS`` when the readings' non-null ``weights_sha256`` differ: recorded, never a refusal (the
    weights fingerprint is a bridge decision; an audio-only reading may fingerprint only what it loads)."""
    weights = {k: r.describe.get("weights_sha256") for k, r in refs.items()}
    distinct = {w for w in weights.values() if w is not None}
    if len(distinct) <= 1:
        return []
    shown = ", ".join(f"{k}={str(w)[:12] if w else None}" for k, w in weights.items())
    return [{"code": "W_JOIN_WEIGHTS", "message": f"the joined readings fingerprint different weights ({shown}); recorded, not refused "
             "", "where": None}]


def run_join(
    benchmark: str,
    routing: Mapping[str, str | os.PathLike],
    *,
    bench: Any = None,
    seed: int | None = None,
    results_root: str | os.PathLike | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """``report --join <benchmark> <reading>=RUN ...``: refuses results of different ``eval_hash``es or
    ``coverage.eval.ids_sha256``s and readings of different ``model``s (exit 2) and an absent or mis-versioned
    package (exit 3, the pip line), then returns ``{"benchmark", "model", "reading", "eval_hash", "routing",
    "results": <join() output>, "warnings", "external", "path"}`` and writes
    ``<results_root>/<model>/results_<ts>.json`` with the entry ``<benchmark>:a+v+av``."""
    if not routing:
        raise JoinError("E_JOIN_USAGE", f"join {benchmark} needs <reading>=RUN pairs", exit_code=1)
    if bench is None:
        try:
            from yavih.benchmark import Benchmark  # lazy: numpy + pyyaml

            bench = Benchmark.load(str(benchmark))
        except Exception as e:  # noqa: BLE001 - an unknown join name is a usage error
            raise JoinError("E_JOIN_UNKNOWN", f"join {benchmark!r}: no such benchmark ({type(e).__name__}: {e})", exit_code=1) from e
    from yavih.readouts.base import ProbeRefused, check_externals, folder_eval

    try:
        mod = folder_eval(bench)
    except ProbeRefused as e:
        raise JoinError(e.code, e.message, exit_code=e.exit_code) from e
    hook = getattr(mod, "join", None)
    if hook is None:
        raise JoinError("E_JOIN_UNSUPPORTED", f"benchmark {benchmark!r} defines no join(): its folder has no eval.py join hook", exit_code=1)

    refs = {str(k): load_ref(str(k), v, seed=seed) for k, v in routing.items()}
    names = {(r.result.get("benchmark") or {}).get("name") for r in refs.values()}
    if names != {getattr(bench, "name", benchmark)}:
        raise JoinError("E_JOIN_MIXED", f"join {benchmark}: results come from benchmarks {sorted(map(str, names))}")
    hashes = {r.eval_hash for r in refs.values()}
    if len(hashes) != 1 or "" in hashes:
        raise JoinError("E_JOIN_MIXED", f"results carry different eval_hash values (protocol or probe): {sorted(h[:12] for h in hashes)} "
                        "")
    model = _one_model(refs)
    warnings = _weights_warnings(refs)
    ids_sha = {((r.result.get("coverage") or {}).get("eval") or {}).get("ids_sha256") for r in refs.values()}
    if len(ids_sha) != 1 or None in ids_sha:
        raise JoinError("E_JOIN_POPULATION", f"eval ids_sha256 differ between readings: {sorted(map(str, ids_sha))}")
    try:
        external = check_externals(bench.spec)
    except ProbeRefused as e:
        raise JoinError(e.code, e.message, exit_code=e.exit_code) from e

    try:
        joined = dict(hook(bench, refs) or {})
    except JoinError:
        raise
    except Exception as e:  # a folder's own refusal (e.g. vggsounder JoinRefused / ExternalPackageError) keeps its code
        code = getattr(e, "code", None)
        if not code:
            raise
        exit_code = getattr(e, "exit_code", None)
        if exit_code is None:
            exit_code = 3 if str(code).startswith("E_EXTERNAL") else 2
        raise JoinError(str(code), str(getattr(e, "message", None) or e), exit_code=int(exit_code)) from e
    if isinstance(joined.get("external"), Mapping):  # versions the hook reports it actually used
        external = {**external, **{str(k): str(v) for k, v in joined["external"].items()}}
    for w in joined.get("warnings") or []:  # the folder's own recorded findings (e.g. its W_JOIN_WEIGHTS)
        if isinstance(w, Mapping) and not any(x.get("code") == w.get("code") for x in warnings):
            warnings.append(dict(w))
    from yavih.report.store import join_reading

    out: dict[str, Any] = {
        "benchmark": getattr(bench, "name", benchmark),
        "model": model,
        "reading": join_reading(list(refs)),
        "eval_hash": hashes.pop(),
        "routing": {k: str(r.seed_dir) for k, r in refs.items()},
        "results": joined,
        "warnings": warnings,
        "external": external,
        "path": None,
    }
    if write:
        from yavih.report import store

        if results_root is None:
            from yavih.config import results_root as _results_root
            from yavih.report.index import runs_root_of

            # seed_dir = <runs root>/<bench>/<model>/<reading>/probe/<eval_hash12>/seed<k>
            results_root = _results_root(runs_root_of([r.seed_dir / "result.json" for r in refs.values()]))
        out["path"] = str(store.write_join(results_root, model, benchmark=out["benchmark"], joined=joined,
                                           refs={k: r.result for k, r in refs.items()},
                                           unit_results={k: str(r.seed_dir / "result.json") for k, r in refs.items()},
                                           external=external, warnings=warnings))
    return out
