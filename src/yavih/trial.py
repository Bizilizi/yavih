"""``yavih bench trial <name|folder> --synthetic``: the whole pipeline of one benchmark folder on a tiny stand-in, in
process, with no data staged.

The chain is the one a real run takes, stage by stage:

0. ``conformance`` — ``yavih.conformance.check`` on the REAL folder: its scoring path (readout, metrics, ``eval.py``
   hooks, pinned ``external`` package) reproduces ``conformance/expected.json`` on the fixture's generated population;
1. ``stand_in`` — ``yavih.testing.make_synthetic_like``: the folder's name, readout, stream names and kinds, guaranteed
   media facts and label facts over tiny generated media (10 s 16 x 16 mp4, 10 s 8 kHz wav, 48 x 32 png) in the work
   directory, plus its media-info census; every stage is handed the stand-in's folder path, never its name (which
   resolves to the real folder);
2. ``bridge`` — the oracle test bridge (``yavih.testing.write_bridge_file``: features that encode the label), pinned to
   the stand-in's folder;
3. ``bridge_check`` — every hard check, the profile written next to the bridge;
4. ``extract`` — ``run_extract`` for every role split in two shards (``--canaries 2``);
5. ``validate`` — ``validate_features`` per split (``_meta.json``);
6. ``probe`` — ``yavih.readouts.probe`` with ``write_results=False`` (the pinned ``ProbeConfig``, or its test-only
   ``fast()`` variant with ``readout_fast``);
7. ``report`` — ``yavih.report.report`` over the stand-in's run directory;
8. ``headline`` — the headline metric against its floor: the oracle must clear it under ``readout_fast`` (the pinned
   schedule may underfit a few dozen synthetic rows, so without it a miss is a ``warning``).

Stand-ins: a folder that ships ``testing.py`` with ``make_stand_in(work, real, *, n_eval, n_train) -> folder`` builds its
own stand-in (a folder with the real folder's name, its ``eval.py`` and scorer included, over generated media), and may
ship ``write_bridge(path, kind, *, D, bench_folder) -> path`` for its own oracle/random bridge (else the core test
bridge is used). Without one, the core fixtures build the stand-in for the scorers they cover
(``yavih.testing.READOUTS``); for any other scorer the trial runs the ``conformance`` stage and records every later
stage as ``not_applicable`` (exit 0): the folder's own scorer is exercised by its conformance fixture and by the
fresh-agent trial on real data.

Limits: the ``conformance`` stage (first) runs the real folder's scoring path, ``eval.py`` hooks and pinned ``external``
package included, on the fixture's generated population; every later stage runs on the stand-in, which never reads
the real data and is single-label, so a ``multi`` / ``multi_sets`` folder's ``probe`` stage trials as single-label
``acc`` (the stand-in has no ``eval.py`` and no ``external``), with the oracle bridge, never the user's; speed and
numbers measured on the stand-in carry over to nothing. The fresh-agent trial on real data covers the rest
(https://github.com/Bizilizi/yavih/blob/main/docs/trial-run.md).

Isolation: the trial never touches the user's runs root, results store, ``$YAVIH_HOME`` or configuration. Inside
:func:`run_trial` every ``YAVIH_*`` variable is replaced (``YAVIH_HOME``, ``YAVIH_RUNS``, ``YAVIH_RESULTS`` under the
work directory, ``YAVIH_CONFIG`` a missing file, ``YAVIH_TESTING=1`` for the oracle bridge and ``fast()``), the CLI
configuration layer is cleared, ``XDG_CONFIG_HOME`` points into the work directory and ``CUBLAS_WORKSPACE_CONFIG``
gets its deterministic default; all of it is restored on return and on error. The work directory is a temporary
directory removed afterwards, or ``keep`` (kept, its paths returned).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

__all__ = ["run_trial", "TrialError", "TRIAL_DOC", "STAGES", "E_TRIAL", "E_TRIAL_NOT_READY", "format_trial"]

#: The trial-run protocol (the fresh-agent trial on real data).
TRIAL_DOC = "https://github.com/Bizilizi/yavih/blob/main/docs/trial-run.md"
#: The stages, in order.
STAGES: tuple[str, ...] = ("conformance", "stand_in", "bridge", "bridge_check", "extract", "validate", "probe", "report", "headline")
#: A stage failed (exit 2; exit 3 when that stage itself was not ready, e.g. a decoder missing).
E_TRIAL = "E_TRIAL"
#: The stand-in cannot be built: a readout or stream kind the synthetic fixtures do not cover, or no media writer
#: (exit 3).
E_TRIAL_NOT_READY = "E_TRIAL_NOT_READY"

_SHARDS = 2
_CANARIES = 2
_D = 32
_BRIDGE_CHECK_N = 8


class TrialError(Exception):
    """``bench trial`` cannot run (``code`` / ``exit_code`` as the CLI maps them)."""

    def __init__(self, code: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code


# ----------------------------------------------------------------------------- isolation


@contextlib.contextmanager
def _isolated(work: Path) -> Iterator[None]:
    """Replace every ``YAVIH_*`` variable and the CLI configuration layer for the trial; restore both afterwards."""
    from yavih import config

    names = ("XDG_CONFIG_HOME", "CUBLAS_WORKSPACE_CONFIG")
    saved_env = {k: v for k, v in os.environ.items() if k.startswith("YAVIH_") or k in names}
    saved_cli = config.cli_overrides()
    try:
        for k in [k for k in os.environ if k.startswith("YAVIH_")]:
            del os.environ[k]
        os.environ.update({
            "YAVIH_HOME": str(work / "home"),
            "YAVIH_RUNS": str(work / "runs"),
            "YAVIH_RESULTS": str(work / "results"),
            "YAVIH_CONFIG": str(work / "no-config.toml"),  # absent: no project file, no discovery
            "YAVIH_TESTING": "1",  # the oracle bridge and ProbeConfig.fast() are test-only
            "XDG_CONFIG_HOME": str(work / "xdg"),  # no user config file
        })
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS when a GPU is used
        config.set_cli()
        yield
    finally:
        for k in [k for k in os.environ if k.startswith("YAVIH_") or k in names]:
            del os.environ[k]
        os.environ.update(saved_env)
        with config._LOCK:  # the CLI layer exactly as it was (set_cli takes another shape)
            config._CLI.clear()
            config._CLI.update(saved_cli)
            config._CACHE["key"] = None


# ----------------------------------------------------------------------------- stages


class _Stages:
    """Runs the stages in order, timing each and recording ``{stage, status, seconds, detail, error}``."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.failed: dict[str, Any] | None = None

    def not_applicable(self, name: str, reason: str) -> None:
        """Record ``name`` as ``not_applicable`` (no stand-in for this folder: nothing to run, nothing failed)."""
        self.records.append({"stage": name, "status": "not_applicable", "seconds": 0.0, "detail": {"reason": reason}, "error": None})

    def run(self, name: str, fn) -> Any:
        if self.failed is not None:
            self.records.append({"stage": name, "status": "skipped", "seconds": 0.0, "detail": None, "error": None})
            return None
        from yavih.cli import exit_code_for

        t0 = time.perf_counter()
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - a failing stage is the trial's finding, not a crash
            exit_code, code = exit_code_for(exc)
            err = {"code": code, "message": str(exc) or type(exc).__name__, "exit_code": exit_code}
            if exit_code == 5:  # an internal error: keep the traceback for the report
                err["traceback"] = traceback.format_exc(limit=8)
            self.failed = {"stage": name, **err}
            self.records.append({"stage": name, "status": "failed", "seconds": round(time.perf_counter() - t0, 3), "detail": None,
                                 "error": err})
            return None
        status = "ok"
        if isinstance(detail, dict) and detail.get("_status"):
            status = detail.pop("_status")
            if status == "failed":
                err = {"code": detail.pop("_code", E_TRIAL), "message": detail.pop("_message", f"{name} failed"), "exit_code": 2}
                self.failed = {"stage": name, **err}
                self.records.append({"stage": name, "status": status, "seconds": round(time.perf_counter() - t0, 3), "detail": detail,
                                     "error": err})
                return detail
        self.records.append({"stage": name, "status": status, "seconds": round(time.perf_counter() - t0, 3), "detail": detail, "error": None})
        return detail


#: The folder file that supplies a folder's own stand-in (``make_stand_in``) and test bridge (``write_bridge``).
FOLDER_TESTING = "testing.py"


def folder_testing(real: Any) -> Any | None:
    """The folder's ``testing.py`` module (imported by path, like ``eval.py``), or ``None``."""
    from yavih.benchmark import _load_module

    path = Path(real.folder) / FOLDER_TESTING
    if not path.is_file():
        return None
    return _load_module(path, "yavih_benchmark_testing")


def stand_in_source(real: Any) -> tuple[str, str | None]:
    """Who builds the stand-in: ``("folder", None)`` (its ``testing.py`` ``make_stand_in``), ``("core", None)`` (the
    synthetic fixtures cover its scorer and stream kinds) or ``("none", reason)``."""
    from yavih import testing

    mod = folder_testing(real)
    if mod is not None and callable(getattr(mod, "make_stand_in", None)):
        return "folder", None
    readout = real.spec.readout
    if readout not in testing.READOUTS:
        return "none", (f"scorer {readout!r} has no synthetic stand-in (the core fixtures cover {', '.join(testing.READOUTS)}; a folder adds "
                        f"its own with {FOLDER_TESTING} make_stand_in)")
    kinds = {s: st.kind for s, st in real.spec.streams.items()}
    bad = sorted(s for s, k in kinds.items() if k not in ("video", "audio", "image"))
    if bad:
        return "none", (f"stream(s) {bad} of kind(s) {[kinds[s] for s in bad]} have no synthetic stand-in (video, audio and image do; a "
                        f"folder adds its own with {FOLDER_TESTING} make_stand_in)")
    return "core", None


def _headline(bench: Any, result: dict, *, readout_fast: bool) -> dict:
    from yavih.report.compare import floor_name_for

    metrics = dict(result.get("metrics") or {})
    floors = dict(result.get("floors") or {})
    name = getattr(bench.spec.metrics, "headline", None)
    if name not in metrics:
        name = next((m for m in metrics if floor_name_for(m, list(metrics), list(floors))), None)
    if name is None:
        return {"_status": "failed", "_message": f"the result names no headline metric with a floor: {sorted(metrics)}"}
    floor_name = floor_name_for(name, list(metrics), list(floors))
    value, floor = metrics.get(name), floors.get(floor_name) if floor_name else None
    ci = (result.get("ci95") or {}).get(name)
    lower = ci[0] if isinstance(ci, list) and len(ci) == 2 else value
    clears = isinstance(lower, (int, float)) and isinstance(floor, (int, float)) and lower > floor
    out: dict[str, Any] = {"metric": name, "value": value, "floor_name": floor_name, "floor": floor, "ci95": ci, "clears_floor": bool(clears),
                           "standard": bool(result.get("standard"))}
    if not clears:
        msg = (f"the oracle's {name} {'CI lower bound ' if ci else ''}{lower} does not clear {floor_name} {floor}: the features encode the "
               "label, so the folder's readout, labels or floor do not fit together")
        if readout_fast:
            out.update(_status="failed", _message=msg)
        else:
            out.update(_status="warning", note=msg + " (or the pinned schedule underfits the stand-in's few rows: re-run with --readout-fast)")
    return out


def run_trial(
    ref: str,
    *,
    readout_fast: bool = False,
    bridge: str = "oracle",
    n_eval: int = 32,
    n_train: int = 64,
    keep: str | os.PathLike | None = None,
) -> dict:
    """Trial benchmark folder ``ref`` (a name or a folder path) on a synthetic stand-in, in process (module
    docstring). Returns ``{benchmark, folder, readout, synthetic, readout_fast, bridge, ok, exit_code, stages, headline,
    failed, paths, limits, stand_in}``: ``stages`` one record per :data:`STAGES` entry (``status`` ``ok`` | ``warning`` |
    ``failed`` | ``skipped`` | ``not_applicable``, ``seconds``, ``detail``, ``error``), ``stand_in`` ``folder`` | ``core`` | ``none``, ``headline`` ``{metric, value, floor_name, floor, ci95,
    clears_floor, standard}``, ``exit_code`` 0, 2 (a stage failed a contract) or 3 (a stage was not ready), ``paths``
    the kept files (``keep`` only). The real folder is loaded first, in the caller's environment; then the
    trial runs isolated (no user root, store, home or config is read or written).

    Raises ``TrialError``: ``E_TRIAL_NOT_READY`` (exit 3) when the stand-in's media cannot be written here,
    ``E_EXISTS`` (exit 1) when ``keep`` is a non-empty directory. A folder no stand-in covers is not an error: its later
    stages are ``not_applicable``."""
    from yavih.benchmark import Benchmark

    if bridge not in ("oracle", "random"):
        raise TrialError("E_USAGE", f"bridge must be 'oracle' or 'random', got {bridge!r}", 1)
    real = Benchmark.load(os.fspath(ref))
    if keep is not None:
        work = Path(keep).expanduser().absolute()
        if work.exists() and (not work.is_dir() or any(work.iterdir())):
            raise TrialError("E_EXISTS", f"--keep {work}: exists and is not an empty directory", 1)
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="yavih-trial-"))
    try:
        with _isolated(work):
            rec = _run(real, work, readout_fast=readout_fast, bridge=bridge, n_eval=int(n_eval), n_train=int(n_train))
    finally:
        if keep is None:
            shutil.rmtree(work, ignore_errors=True)
    rec["ref"] = os.fspath(ref)
    if keep is None:
        rec["paths"] = {}
    return rec


def _run(real: Any, work: Path, *, readout_fast: bool, bridge: str, n_eval: int, n_train: int) -> dict:
    from yavih import testing
    from yavih.benchmark import Benchmark

    st = _Stages()
    state: dict[str, Any] = {}
    runs = work / "runs"
    source, why_not = stand_in_source(real)
    own = folder_testing(real) if source == "folder" else None

    def stand_in() -> dict:
        try:
            if own is not None:
                folder = Path(own.make_stand_in(work / "bench", real, n_eval=n_eval, n_train=n_train))
            else:
                folder = testing.make_synthetic_like(work / "bench", real, n_eval=n_eval, n_train=n_train)
        except ImportError as e:
            raise TrialError(E_TRIAL_NOT_READY, f"the stand-in's tiny media are written with PyAV, soundfile and pillow: {e}; "
                             "pip install av soundfile pillow (or yavih[media])", 3) from e
        except ValueError as e:
            raise TrialError(E_TRIAL_NOT_READY, f"{real.name}: no synthetic stand-in: {e}", 3) from e
        from yavih.media.info import write_census

        write_census(Benchmark.load(str(folder)))
        state.update(folder=folder, bench=Benchmark.load(str(folder)))
        roles = state["bench"].roles()
        return {"folder": str(folder), "splits": {s: state["bench"].n_rows(s) for s in _role_splits(roles)},
                "label_kind": _label_kind(real), "source": source}

    def write_bridge() -> dict:
        target = work / "bridges" / f"{bridge}_bridge.py"
        if own is not None and callable(getattr(own, "write_bridge", None)):
            target.parent.mkdir(parents=True, exist_ok=True)
            path = Path(own.write_bridge(target, bridge, D=_D, bench_folder=state["folder"]))
        else:
            path = testing.write_bridge_file(target, bridge, D=_D, bench_folder=state["folder"])
        state["bridge"] = path
        return {"path": str(path), "kind": bridge}

    def check() -> dict:
        from yavih.extract import bridge_check

        profile_path = work / "bridges" / f"{bridge}.profile.json"
        prof = bridge_check(bridge=str(state["bridge"]), bench=str(state["folder"]), n=_BRIDGE_CHECK_N, out=profile_path)
        state.update(profile=prof, profile_path=profile_path)
        return {"model": prof.get("model"), "reading": prof.get("reading"), "N": prof.get("N"), "D": prof.get("D"),
                "invariance": prof.get("invariance"), "profile": str(profile_path)}

    def extract() -> dict:
        from yavih.extract import run_extract

        prof = state["profile"]
        run_dir = runs / real.name / str(prof["model"]) / str(prof["reading"])
        state["run_dir"] = run_dir
        rows: dict[str, int] = {}
        for split in _role_splits(state["bench"].roles()):
            for i in range(_SHARDS):
                summary = run_extract(bench=str(state["folder"]), bridge=str(state["bridge"]), split=split, shard=(i, _SHARDS),
                                      out=run_dir, canaries=_CANARIES)
                if summary.get("status") != "ok":
                    return {"_status": "failed", "_message": f"extract {split} {i}/{_SHARDS}: status {summary.get('status')} "
                            f"({summary.get('reason')})", "split": split}
                rows[split] = rows.get(split, 0) + len(summary.get("members") or [])
        return {"run_dir": str(run_dir), "shards": _SHARDS, "rows": rows}

    def validate() -> dict:
        from yavih.io.validate import validate_features

        out: dict[str, Any] = {}
        for split in _role_splits(state["bench"].roles()):
            rep = validate_features(state["run_dir"] / "features" / split, bench=state["bench"])
            out[split] = {"ok": rep.ok, "coverage": rep.coverage, "eligible": (rep.eligibility or {}).get("eligible"),
                          "errors": [f"{f.code}: {f.message}" for f in rep.errors][:5],
                          "warnings": sorted({f.code for f in rep.warnings})}
            if not rep.ok:
                return {"_status": "failed", "_message": f"validate {split}: " + "; ".join(out[split]["errors"]), **out}
        return out

    def run_probe() -> dict:
        from yavih.readouts import probe, registry

        bench = state["bench"]
        scorer = registry.get(bench.spec.readout)
        cfg = testing.fast(scorer.pinned_config(bench)) if readout_fast and scorer.is_probe else None
        res = probe(state["run_dir"], seeds=(0,), cfg=cfg, write_results=False, bench=str(state["folder"]))
        if len(res) != 1 or res[0].status != "ok":
            got = res[0] if res else None
            return {"_status": "failed", "_message": f"probe status {getattr(got, 'status', None)}: {getattr(got, 'reason', None)}"}
        state["result"] = res[0].result
        return {"result": str(res[0].path), "standard": bool(res[0].result.get("standard")),
                "metrics": dict(res[0].result.get("metrics") or {}), "floors": dict(res[0].result.get("floors") or {})}

    def run_report() -> dict:
        from yavih.report import report

        rep = report([str(state["run_dir"])], fmt="md", runs_root=runs)
        state["report"] = rep.paths.get("report")
        return {"rows": len(rep.rows), "columns": len(rep.columns), "warnings": sorted({str(w.get("code")) for w in rep.warnings}),
                "report": rep.paths.get("report")}

    def conformance() -> dict:
        from yavih.conformance import check as check_conformance
        rep = check_conformance(real)
        out = {"ok": rep.ok, "skipped": rep.skipped, "compared": rep.compared, "protocol_version": rep.protocol_version,
               "warnings": [f.code for f in rep.warnings]}
        if not rep.ok:
            err = rep.errors[0]
            return {"_status": "failed", "_code": err.code, "_message": err.message, **out}
        return out

    st.run("conformance", conformance)
    head = None
    if source == "none":
        for name in STAGES[1:]:
            st.not_applicable(name, why_not or "no stand-in")
    else:
        st.run("stand_in", stand_in)
        st.run("bridge", write_bridge)
        st.run("bridge_check", check)
        st.run("extract", extract)
        st.run("validate", validate)
        st.run("probe", run_probe)
        st.run("report", run_report)
        head = st.run("headline", lambda: _headline(state["bench"], state["result"], readout_fast=readout_fast))

    failed = st.failed
    exit_code = 0 if failed is None else (3 if failed.get("exit_code") == 3 else 2)
    paths = {k: str(v) for k, v in (("work", work), ("folder", state.get("folder")), ("bridge", state.get("bridge")),
                                     ("profile", state.get("profile_path")), ("run_dir", state.get("run_dir")),
                                     ("report", state.get("report"))) if v}
    headline = {k: v for k, v in (head or {}).items() if not k.startswith("_")} if isinstance(head, dict) else None
    return {
        "benchmark": real.name,
        "folder": str(real.folder),
        "readout": real.spec.readout,
        "synthetic": True,
        "readout_fast": bool(readout_fast),
        "bridge": bridge,
        "ok": failed is None,
        "exit_code": exit_code,
        "stages": st.records,
        "headline": headline,
        "failed": failed,
        "paths": paths,
        "limits": _limits(real, source=source, why_not=why_not),
        "stand_in": source,
    }


def _role_splits(roles: dict) -> list[str]:
    out: list[str] = []
    for role in ("train", "eval", "floor", "folds"):
        for s in roles.get(role) or []:
            if s not in out:
                out.append(s)
    return out


def _label_kind(real: Any) -> str | None:
    raw = getattr(real.spec, "raw", None) or {}
    return ((raw.get("manifest") or {}).get("label") or {}).get("kind")


def _limits(real: Any, *, source: str = "core", why_not: str | None = None) -> list[str]:
    if source == "none":
        return [f"no synthetic stand-in: {why_not}; only the conformance stage ran, the fresh-agent trial on real data covers the rest "
                f"({TRIAL_DOC})"]
    out = ["synthetic stand-in: speed and numbers carry over to nothing; the fresh-agent trial on real data covers the rest "
           f"({TRIAL_DOC})"]
    if source == "folder":
        out.append(f"the stand-in is the folder's own ({FOLDER_TESTING} make_stand_in): its eval.py and scorer run in every stage")
        return out
    kind = _label_kind(real)
    if kind not in (None, "single", "none"):
        out.append(f"label kind {kind!r}: the stand-in is single-label, so the readout trials as single-label acc")
    folder = Path(real.folder)
    if (folder / "eval.py").is_file():
        out.append("the folder's eval.py hooks run in the conformance stage only (the stand-in has none)")
    if (getattr(real.spec, "raw", None) or {}).get("external"):
        out.append("the folder's external packages run in the conformance stage only (the stand-in lists none)")
    return out


def format_trial(rec: dict) -> str:
    """The human rendering: one line per stage, the headline, the limits and (with ``--keep``) the paths."""
    lines = [f"bench trial {rec['benchmark']} ({rec['readout']}, synthetic stand-in, {rec['bridge']} bridge"
             f"{', readout-fast' if rec.get('readout_fast') else ''})"]
    for s in rec["stages"]:
        note = ""
        if s.get("error"):
            note = f"  {s['error']['code']}: {s['error']['message']}"
        elif isinstance(s.get("detail"), dict) and s["detail"].get("note"):
            note = f"  {s['detail']['note']}"
        elif s.get("status") == "not_applicable" and s["stage"] == "stand_in":
            note = f"  {s['detail']['reason']}"
        lines.append(f"  {s['stage']:<13} {s['status']:<8} {s['seconds']:>7.1f}s{note}")
    h = rec.get("headline") or {}
    if h.get("metric"):
        ci = h.get("ci95")
        lines.append(f"headline {h['metric']} = {h.get('value')}" + (f" (ci95 {ci})" if ci else "") +
                     f", floor {h.get('floor_name')} = {h.get('floor')}: {'clears' if h.get('clears_floor') else 'does NOT clear'}")
    for lim in rec.get("limits") or []:
        lines.append(f"limit: {lim}")
    for k, v in (rec.get("paths") or {}).items():
        lines.append(f"{k}: {v}")
    lines.append("trial ok" if rec["ok"] else f"trial FAILED at {rec['failed']['stage']} (exit {rec['exit_code']})")
    return "\n".join(lines) + "\n"
