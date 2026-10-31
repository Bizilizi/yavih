"""``yavih exec``: dependencies (exit 3), DONE check (skipped), deadline refusal (exit 4),
launcher substitution, tee logs, signal forwarding (killed child → exit 4 ``preempted``), in-process
probe/report, ``logs/last.json``, ``--slots K``, ``--range a:b``, scratch pre-flight. Imports only stdlib for
extract units.

Module scope is stdlib only. ``yavih.plan.units`` (stdlib + ``yavih.hashing``) is imported inside
``exec_unit``; ``yavih.readouts`` / ``yavih.report`` (torch) are imported only when a probe or report
unit actually runs, so an extract controller never loads them.

``--deps plan.jsonl`` for dependency lookup from a ``--pending`` file; a ``running`` record at start; in-process
units write a log with the traceback; ``--deadline-margin`` default 900 s (> the templates' 600 s signal lead).

- Deadlines: one deadline parser (``yavih.deadline``: ``+secs``, durations, epochs, ISO times, ``none``, ``slurm``);
  ``--deadline slurm`` inside a job without ``SLURM_JOB_END_TIME`` is a configuration error (exit 1), never "now".
  A unit whose child exits 1 (usage / configuration) gets its own status ``misconfigured`` (exit 1, never
  resubmitted by a driver loop). Every record carries ``history = {attempts, status, streak}`` so ``plan status`` can
  flag a unit that keeps exiting 3 or 4 the same way. Without a configured scratch root a probe unit runs, as
  ``eval run`` does, on the features in place (memmap) instead of exiting 3.
- Slots: ``--slots K`` gives slot ``g`` the ``g``-th entry of the INHERITED ``CUDA_VISIBLE_DEVICES`` (a non-cgroup
  SLURM node, Kubernetes GPU UUIDs, ``docker --gpus device=4,5``), and every slot — probes included — is a separate
  ``yavih exec --id`` process, so devices, torch globals and signal flags never cross between units.
- Paths and CPUs: ``--root`` re-anchors the plan's relative paths; ``--cpus`` overrides the unit's CPU budget (else it is
  clamped to the CPUs this process may use: affinity mask and cgroup quota, ``yavih.resources``).
- Pending files: ``--plan-sha`` refuses a pending file that changed since the job was submitted.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from yavih.deadline import DEFAULT_DEADLINE_MARGIN_S, DeadlineError, parse_deadline

if TYPE_CHECKING:  # pragma: no cover
    from yavih.plan.units import Unit

__all__ = [
    "exec_unit",
    "resolve_deadline",
    "build_command",
    "device_masks",
    "parse_range",
    "ExecError",
    "DeadlineError",
    "STATUSES",
    "LAST_JSON",
    "EXIT_FOR_STATUS",
    "MIN_HEADROOM_S",
    "HEADROOM_FRACTION",
    "STALE_SCRATCH_S",
    "FORWARDED_SIGNALS",
    "DEFAULT_DEADLINE_MARGIN_S",
    "RUNNING",
    "SLOT_GPU_ENV",
]

STATUSES: frozenset[str] = frozenset(
    {"done", "skipped", "not_applicable", "preempted", "deadline", "not_ready", "misconfigured", "failed"}
)
#: Provisional status of ``logs/last.<id>.json`` while a unit runs (written once its checks passed, replaced by
#: the final record; never written to the shared ``logs/last.json``). ``plan status`` reports it as ``running``
#: while the recorded process can still be alive.
RUNNING = "running"
LAST_JSON = "logs/last.json"
#: Exit code per status (``failed`` keeps the child's own code when it has one). ``misconfigured`` = the unit
#: exited 1: a usage / configuration error (a stale plan's ``E_LAYOUT``, a bad ``--deadline``) that no resubmission
#: heals; a driver loop must stop on it as on ``failed``.
EXIT_FOR_STATUS: dict[str, int] = {
    "done": 0, "skipped": 0, "not_applicable": 0, "misconfigured": 1, "not_ready": 3, "deadline": 4, "preempted": 4,
    "failed": 5,
}
#: Refuse to start below ``max(10 min, 0.25 * needs.minutes)`` of remaining wall.
MIN_HEADROOM_S = 600.0
HEADROOM_FRACTION = 0.25
#: A foreign-host scratch dir older than this is swept (pid liveness is unknowable).
STALE_SCRATCH_S = 36 * 3600
FORWARDED_SIGNALS: tuple[int, ...] = tuple(
    s for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None), getattr(signal, "SIGUSR1", None)) if s
)
#: Set by ``--slots`` for each slot process: its device mask, applied AFTER the unit's own ``env`` so a matrix
#: ``env: {CUDA_VISIBLE_DEVICES: ...}`` cannot put two slots on one GPU.
SLOT_GPU_ENV = "YAVIH_SLOT_GPU"
_SCRATCH_DIR = re.compile(r"^(?P<id>[0-9a-f]{12})\.(?P<host>.+)\.(?P<pid>\d+)$")

_CHILDREN: set[subprocess.Popen] = set()
_CHILDREN_LOCK = threading.Lock()
_PREEMPT = threading.Event()
_FORWARDED: list[int] = []


_UNRESOLVED: Any = object()


class ExecError(ValueError):
    """Bad ``exec`` invocation (unknown id, index out of range, ``--slots`` with ``--id``) — exit 1."""

    code = "E_EXEC"
    exit_code = 1


# ----------------------------------------------------------------------------- deadline / command


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_deadline(spec: str | float | None, *, margin_s: float = DEFAULT_DEADLINE_MARGIN_S,
                     env: dict | None = None) -> float | None:
    """``--deadline`` → epoch or ``None`` through the shared parser (``yavih.deadline.parse_deadline``): ``slurm``
    = ``SLURM_JOB_END_TIME − margin`` (inside a job without it: ``DeadlineError``, exit 1; outside SLURM: no
    deadline), ``+<secs>`` / a duration, an epoch, an ISO time, ``none``. ``exec`` does not inherit
    ``$YAVIH_DEADLINE``: it is the process that sets it."""
    return parse_deadline(spec, env=env, margin_s=margin_s, inherit=False)


def resolve_launcher(spec: str | None, cwd: str | os.PathLike | None = None) -> str | None:
    """A unit's ``launcher`` as the command to run: a value with a path separator or whitespace is the command itself;
    a bare NAME is ``[launchers] NAME`` of the configuration discovered from ``cwd`` (the unit's cwd = the matrix
    directory, so the project's ``yavih-bench/yavih.toml`` and its site tables are found wherever ``exec`` runs), else
    NAME when it is a command on PATH, else ``ExecError``; ``None`` is the configured default ``launcher``, else
    ``None`` (the unit runs with the interpreter running ``exec``)."""
    from yavih import config

    if spec is not None and not str(spec).strip():
        spec = None
    if spec is not None and ("/" in spec or os.sep in spec or any(ch.isspace() for ch in spec)):
        return spec
    try:
        settings = config.load(cwd=cwd) if cwd else config.load()
    except config.ConfigError as e:
        raise ExecError(f"launcher {spec!r}: the configuration does not load: {e}") from e
    if spec is None:
        return settings.launcher()
    named = settings.launcher(spec)
    if named is not None:
        return named
    if shutil.which(spec):
        return spec
    known = sorted(settings.launchers())
    raise ExecError(f"launcher {spec!r} is neither a [launchers] name of the active configuration "
                    f"({', '.join(known) if known else 'none defined'}) nor a command on PATH; define `[launchers] {spec} = ...` "
                    f"in {settings.edit_hint()} (per site under [site.<id>]), or write the command itself in the matrix")


def build_command(unit: Any, launcher: str | None = _UNRESOLVED) -> list[str]:
    """``launcher`` (already resolved; default ``resolve_launcher(unit.launcher, unit.cwd)``) + ``cmd``: a launcher whose
    last token matches ``python*`` replaces ``cmd[0]``; otherwise it is a prefix. With no launcher the planned ``cmd[0]``
    (the literal ``python``) is replaced by ``sys.executable``, the interpreter running ``exec``, never whatever
    ``python`` is on PATH."""
    cmd = list(unit.cmd)
    if launcher is _UNRESOLVED:
        launcher = resolve_launcher(unit.launcher, getattr(unit, "cwd", None))
    if not launcher or not str(launcher).strip():
        return [sys.executable, *cmd[1:]] if cmd else cmd
    toks = shlex.split(str(launcher))
    if toks and fnmatch.fnmatch(os.path.basename(toks[-1]), "python*"):
        return toks + cmd[1:]
    return toks + cmd


def device_masks(slots: int, env: dict | None = None) -> list[str]:
    """``CUDA_VISIBLE_DEVICES`` of each of ``slots`` slots: the ``g``-th entry of the inherited list (indices, or GPU
    UUIDs under Kubernetes), never a bare ``g`` that could name another job's GPU; ``"0" … "K-1"`` only when nothing
    is inherited (the whole node is this job's). An inherited empty value means no GPU: every slot gets ``""``."""
    env = os.environ if env is None else env
    inherited = env.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return [str(g) for g in range(slots)]
    if inherited.strip() == "":
        return [""] * slots
    devs = [d.strip() for d in inherited.split(",") if d.strip()]
    if len(devs) < slots:
        raise ExecError(f"--slots {slots} but CUDA_VISIBLE_DEVICES={inherited!r} lists {len(devs)} device(s); "
                        "request as many GPUs as slots, or lower --slots")
    return devs[:slots]


def parse_range(text: str, n_units: int) -> tuple[int, int]:
    """``--range a:b`` → ``(a, b)`` (half-open; ``a:`` = to the end, ``:b`` = from 0), checked against the plan."""
    m = re.fullmatch(r"\s*(\d*)\s*:\s*(\d*)\s*", str(text))
    if not m:
        raise ExecError(f"--range expects a:b (half-open unit indices), got {text!r}")
    a = int(m.group(1)) if m.group(1) else 0
    b = int(m.group(2)) if m.group(2) else n_units
    if not (0 <= a < b <= n_units):
        raise ExecError(f"--range {text} selects no unit of {n_units} (need 0 <= a < b <= {n_units})")
    return a, b


# ----------------------------------------------------------------------------- helpers


def _write_json_atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # pid AND thread: several writers of the shared logs/last.json must not truncate each other's tmp.
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True, indent=1)
    os.replace(tmp, path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _log(msg: str) -> None:
    # stderr: under --json stdout is exactly one envelope.
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _log_append(log_path: str, text: str) -> None:
    """Append to a unit log; a log that cannot be written never fails the unit."""
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as e:  # pragma: no cover - unwritable out dir
        _log(f"[yavih] cannot write {log_path}: {e}")


def _log_header(log_path: str, unit: Unit, what: str) -> None:
    _log_append(log_path, f"[yavih] exec {unit.id} {unit.kind} {what} started={_now_utc()}\n")


def _log_traceback(log_path: str) -> None:
    """The current exception's full traceback into the unit log (call inside an ``except`` block)."""
    _log_append(log_path, f"[yavih] exception at {_now_utc()}:\n{traceback.format_exc()}")


def _write_running(unit: Unit, *, out_dir: str, host: str, gpu: str | None, slurm_job_id: str | None, needs: dict,
                   started: str, started_t: float, deadline: float | None, log_path: str, history: dict) -> None:
    """Provisional ``logs/last.<id>.json`` (status ``running``) written once the unit's checks passed and it is
    about to run; the final record replaces it. ``pid`` / ``host`` / ``started_epoch`` / ``deadline`` let
    ``plan status`` tell a live unit from one killed without a record (SIGKILL, OOM, node loss)."""
    from yavih.plan.units import INPUTS_SCHEME

    rec = {
        "id": unit.id, "kind": unit.kind, "status": RUNNING, "exit_code": None, "wall_s": None, "host": host,
        "pid": os.getpid(), "gpu": gpu, "slurm_job_id": slurm_job_id, "inputs_hash": unit.inputs_hash,
        "inputs_scheme": INPUTS_SCHEME, "needs": needs, "started": started, "started_epoch": round(started_t, 3),
        "ended": None, "deadline": None if deadline is None else float(deadline), "log": log_path, "message": None,
        "history": history,
    }
    try:
        _write_json_atomic(os.path.join(out_dir, "logs", f"last.{unit.id}.json"), rec)
    except OSError as e:  # pragma: no cover - unwritable out dir
        _log(f"[yavih] cannot write the running record of {unit.id}: {e}")


def _forward_signal(signum: int, _frame: Any) -> None:
    """Forward to every live child and remember that we were told to stop. For an
    in-process probe the flag is what the readout checks between micro-batches."""
    _PREEMPT.set()
    _FORWARDED.append(signum)
    with _CHILDREN_LOCK:
        procs = list(_CHILDREN)
    for p in procs:
        try:
            p.send_signal(signum)
        except (ProcessLookupError, OSError):
            pass
    try:  # the readouts' cooperative stop flag, when exposed
        import yavih.readouts.base as rb  # noqa: PLC0415

        hook = getattr(rb, "request_stop", None)
        if callable(hook):
            hook(signum)
    except Exception:  # pragma: no cover - the readouts package may be absent in a model env
        pass


class _Signals:
    """Install forwarding handlers on the main thread for the duration of a run; restore afterwards."""

    def __init__(self) -> None:
        self._old: dict[int, Any] = {}

    def __enter__(self) -> _Signals:
        if threading.current_thread() is not threading.main_thread():
            return self
        for s in FORWARDED_SIGNALS:
            try:
                self._old[s] = signal.signal(s, _forward_signal)
            except (ValueError, OSError):  # pragma: no cover
                pass
        return self

    def __exit__(self, *exc: Any) -> None:
        for s, old in self._old.items():
            try:
                signal.signal(s, old)
            except (ValueError, OSError):  # pragma: no cover
                pass


def _exit_for(status: str, child_rc: int | None = None) -> int:
    if status == "failed" and child_rc is not None and 1 <= child_rc <= 5:
        return child_rc
    return EXIT_FOR_STATUS[status]


def _exception_exit(exc: BaseException) -> int:
    ec = getattr(exc, "exit_code", None)
    if isinstance(ec, int) and 1 <= ec <= 5:
        return ec
    if isinstance(exc, TimeoutError):
        return 4
    try:
        from yavih.cli import exit_code_for  # noqa: PLC0415

        return exit_code_for(exc)[0]
    except Exception:  # pragma: no cover - the CLI mapping is a convenience
        return 5


def _status_for_code(rc: int) -> str:
    """The status of a unit that ended with exit code ``rc`` without a DONE marker."""
    return {1: "misconfigured", 3: "not_ready", 4: "deadline"}.get(rc, "failed")


# ----------------------------------------------------------------------------- scratch (step 8)


def _scratch_preflight(unit: Unit, *, env: dict[str, str], log_path: str | None = None) -> tuple[str | None, str | None]:
    """Returns ``(scratch_dir, problem)``: ``problem`` set → exit 3 not_ready. Without a
    configured scratch root (``YAVIH_SCRATCH`` → config ``[scratch] root`` → ``SLURM_TMPDIR``) the probe reads the
    features in place (memmap backend, ``W_SIZE`` in its result) exactly as ``eval run`` does — slower, never a
    not-ready loop."""
    need_gb = float(unit.needs.get("scratch_gb") or 0)
    if need_gb <= 0:
        return None, None
    from yavih.config import configured_scratch_root  # stdlib

    root_p = configured_scratch_root(env)
    if root_p is None:
        msg = (f"[yavih] probe needs {need_gb:g} GB scratch but no YAVIH_SCRATCH / [scratch] root / SLURM_TMPDIR is set: "
               "the features are read in place (memmap backend, W_SIZE)\n")
        _log(msg.rstrip())
        if log_path:
            _log_append(log_path, msg)
        return None, None
    root = str(root_p)
    os.makedirs(root, exist_ok=True)
    _sweep_scratch(root)
    free = shutil.disk_usage(root).free
    if free < need_gb * 2**30:
        return root, f"scratch {root} has {free / 2**30:.1f} GiB free, unit needs {need_gb:g} GB"
    return root, None


def _sweep_scratch(root: str, *, now: float | None = None, host: str | None = None) -> list[str]:
    """Remove ``<unit_id>.<host>.<pid>`` siblings whose pid is dead (same host) or older than 36 h."""
    now = time.time() if now is None else now
    host = host or socket.gethostname()
    removed: list[str] = []
    try:
        names = os.listdir(root)
    except OSError:
        return removed
    for name in names:
        m = _SCRATCH_DIR.match(name)
        if not m:
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            age = now - os.stat(path).st_mtime
        except OSError:
            continue
        same_host = m.group("host") == host
        dead = same_host and not _pid_alive(int(m.group("pid"))) and int(m.group("pid")) != os.getpid()
        if dead or age > STALE_SCRATCH_S:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed


# ----------------------------------------------------------------------------- runners (step 5)


def _run_extract(unit: Unit, *, env: dict[str, str], log_path: str, cpus: int | None = None,
                 launcher: str | None = _UNRESOLVED) -> tuple[int, bool]:
    """Spawn ``launcher + cmd``, tee output to the log and to stderr, forward signals; returns
    ``(returncode, killed_by_signal)``. With ``cpus`` (``--cpus``, a slot's share, or the usable CPUs when fewer
    than planned) the child's ``--decode-workers`` is rewritten to ``cpus - 1`` (decode workers =
    cpus − 1). ``launcher`` is the resolved launcher (``_run_unit`` resolves it before the unit is marked running)."""
    from yavih.plan.units import resolve_path

    argv = build_command(unit, launcher)
    if cpus is not None and "--decode-workers" in argv:
        i = argv.index("--decode-workers")
        if i + 1 < len(argv):
            argv[i + 1] = str(max(1, cpus - 1))
    cwd = resolve_path(unit, unit.cwd) if unit.cwd else None
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "ab") as log:
        log.write(f"[yavih] exec {unit.id} {unit.kind} argv={argv!r} cwd={cwd!r}\n".encode())
        log.flush()
        try:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except (FileNotFoundError, PermissionError, NotADirectoryError) as e:
            msg = f"[yavih] cannot start {argv[0]!r}: {e}\n"
            log.write(msg.encode())
            _log(msg.rstrip())
            return 1, False
        with _CHILDREN_LOCK:
            _CHILDREN.add(proc)
        try:
            assert proc.stdout is not None
            err = getattr(sys.stderr, "buffer", None)
            for chunk in iter(proc.stdout.readline, b""):
                log.write(chunk)
                log.flush()
                if err is not None:
                    try:
                        err.write(chunk)
                        err.flush()
                    except (ValueError, OSError):  # pragma: no cover - closed stderr under capture
                        pass
                else:  # pragma: no cover
                    sys.stderr.write(chunk.decode("utf-8", "replace"))
            rc = proc.wait()
        finally:
            with _CHILDREN_LOCK:
                _CHILDREN.discard(proc)
    killed = rc < 0 or rc > 128
    return rc, killed


def _run_probe(unit: Unit, *, deadline: float | None, scratch: str | None) -> None:
    """In-process probe. A plan-time ``not_applicable`` unit goes through
    ``yavih.readouts.probe_not_applicable`` (no features exist; the result is built from the planned
    profile); the ordinary path is ``yavih.readouts.probe(run_dir, seeds=(seed,), ...)`` with the validate
    prologue (``validate_first``: validate every split before reading ``_meta.json``)."""
    import yavih.readouts as readouts  # torch, only here
    from yavih.plan.units import resolve_path

    run_dir = resolve_path(unit, unit.out)
    seed = int(unit.seed if unit.seed is not None else 0)
    if unit.expected_status == "not_applicable":
        profile: dict = {}
        if unit.profile:
            from yavih.plan.matrix import load_profile

            profile = load_profile(resolve_path(unit, unit.profile))
        readouts.probe_not_applicable(run_dir, unit=unit.to_json(), profile=profile, seed=seed, write_results="--no-results" not in unit.cmd)
        return
    kw: dict[str, Any] = {}
    if "--no-results" in unit.cmd:  # `plan build --sample N`: no results-store file, as `eval run --no-results`
        kw["write_results"] = False
    results = _cmd_option(unit.cmd, "--results")
    if results:  # matrix `results` → the probe's results-store root, as `eval run --results` would
        kw["results_root"] = resolve_path(unit, results)
    waive = [cmd_v for i, tok in enumerate(unit.cmd[:-1]) if tok == "--waive" for cmd_v in (unit.cmd[i + 1],)]
    if waive:  # matrix `waive`: the operator's waivers for the validate prologue
        from yavih.waivers import parse_cli_waivers

        kw["waive"] = parse_cli_waivers(waive)
    readouts.probe(run_dir, seeds=(seed,), deadline=deadline, scratch=scratch, validate_first=True, **kw)


def _cmd_option(cmd: list[str], flag: str) -> str | None:
    """The value after ``flag`` in a unit's ``cmd`` (the record the in-process run must agree with)."""
    for i, tok in enumerate(cmd[:-1]):
        if tok == flag:
            return cmd[i + 1]
    return None


def _run_report(unit: Unit) -> None:
    """In-process report: tables to ``<out>/report.{md,json}``, then DONE."""
    from yavih.plan.units import resolve_path
    from yavih.report import report

    out = resolve_path(unit, unit.out)
    runs_root = os.path.dirname(out)
    rep = report([os.path.join(runs_root, "*")], fmt="md")
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "report.md.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(rep.text)
    os.replace(tmp, os.path.join(out, "report.md"))
    _write_json_atomic(os.path.join(out, "report.json"), rep.to_json())
    from yavih.versions import __version__

    _write_json_atomic(resolve_path(unit, unit.done), {
        "created": _now_utc(), "yavih_version": __version__, "n_rows": len(rep.rows), "n_columns": len(rep.columns),
        "probe_units": list(unit.depends_on),
    })


# ----------------------------------------------------------------------------- one unit


def _effective_cpus(unit: Unit, cpus: int | None) -> int | None:
    """The CPU budget a unit runs with: ``--cpus`` / a slot's share when given, else ``needs.cpus`` clamped to the
    CPUs this process may use (``yavih.resources.available_cpus``); ``None`` when that equals ``needs.cpus`` (the
    planned command runs unchanged)."""
    if cpus is not None:
        return max(1, int(cpus))
    from yavih.resources import available_cpus

    planned = int(unit.needs.get("cpus") or 0)
    avail = available_cpus()
    if planned and avail < planned:
        _log(f"[yavih] unit {unit.id}: planned for {planned} CPUs, this process may use {avail} (affinity / cgroup); "
             f"decode workers {max(1, avail - 1)}")
        return avail
    return None


def _history(prev: dict | None) -> dict:
    """``history`` of the previous record of this unit (its own ``history``, or a fresh one)."""
    h = (prev or {}).get("history")
    if not isinstance(h, dict):
        h = {"attempts": 0, "status": None, "streak": 0}
    return {"attempts": int(h.get("attempts") or 0), "status": h.get("status"), "streak": int(h.get("streak") or 0)}


def _run_unit(units_by_id: dict[str, Unit], unit: Unit, *, deadline: float | None,
              env_overrides: dict[str, str] | None = None, cpus: int | None = None) -> dict:
    from yavih.plan.units import INPUTS_SCHEME, check_done, last_record, resolve_path

    started_t = time.time()
    started = _now_utc()
    host = socket.gethostname()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id and os.environ.get("SLURM_ARRAY_TASK_ID"):
        slurm_job_id = f"{os.environ.get('SLURM_ARRAY_JOB_ID', slurm_job_id)}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    env = dict(os.environ)
    env.update(unit.env)
    env.update(env_overrides or {})
    if os.environ.get(SLOT_GPU_ENV) is not None:  # a --slots slot: its mask wins over the unit's env
        env["CUDA_VISIBLE_DEVICES"] = os.environ[SLOT_GPU_ENV]
    env["YAVIH_UNIT_ID"] = unit.id
    if deadline is not None:
        env["YAVIH_DEADLINE"] = repr(float(deadline))
    eff_cpus = _effective_cpus(unit, cpus)
    if eff_cpus is not None:
        env["YAVIH_CPUS"] = str(eff_cpus)
    gpu = env.get("CUDA_VISIBLE_DEVICES")
    out_dir = resolve_path(unit, unit.out)
    tag = slurm_job_id or str(os.getpid())
    log_path = os.path.join(out_dir, "logs", f"{unit.id}.{tag}.{started.replace(':', '')}.log")
    needs = dict(unit.needs)
    if eff_cpus is not None:
        needs["cpus"] = eff_cpus
    prev = _history(last_record(unit))
    history = dict(prev, attempts=prev["attempts"] + 1)

    _log(f"[yavih] unit {unit.id} {unit.kind} start host={host} gpu={gpu or '-'} slurm_job_id={slurm_job_id or '-'}")
    status = "failed"
    exit_code = 5
    message: str | None = None
    child_rc: int | None = None
    refused_at_start = False

    # 2. dependencies
    for dep in unit.depends_on:
        d = units_by_id.get(dep)
        if d is None:
            status, message = "not_ready", (
                f"dependency {dep} is not in the plan; a `plan build --pending --kind {unit.kind}` file holds no "
                f"units of other kinds: pass the full plan with `--deps plan.jsonl`"
            )
            break
        if check_done(d) is None:
            status, message = "not_ready", f"dependency {dep} ({d.kind} {d.benchmark} {d.split or ''} {d.shard or ''}) has no matching DONE marker"
            break
    else:
        # 3. DONE check
        already = check_done(unit)
        launcher: str | None = None
        launcher_error: str | None = None
        if already is None and unit.kind == "extract":
            # 3b. the launcher: a [launchers] name of the configuration discovered from the unit's cwd; a name the
            # active site does not define is `misconfigured` before anything is marked running
            try:
                launcher = resolve_launcher(unit.launcher, unit.cwd)
            except ExecError as e:
                launcher_error = str(e)
        if already is not None:
            status, message = "skipped", f"already {already}: {unit.done}"
        elif launcher_error is not None:
            status, message, child_rc = "misconfigured", f"launcher: {launcher_error}", 1
        else:
            # 4. deadline headroom
            headroom = max(MIN_HEADROOM_S, HEADROOM_FRACTION * float(needs.get("minutes") or 0) * 60.0)
            if deadline is not None and deadline - time.time() < headroom:
                status, message = "deadline", f"{deadline - time.time():.0f}s left before the deadline, need {headroom:.0f}s"
                refused_at_start = True
            else:
                _write_running(unit, out_dir=out_dir, host=host, gpu=gpu, slurm_job_id=slurm_job_id, needs=needs,
                               started=started, started_t=started_t, deadline=deadline, log_path=log_path, history=history)
                status, message, child_rc = _dispatch(unit, env=env, log_path=log_path, deadline=deadline, cpus=eff_cpus,
                                                      launcher=launcher)

    exit_code = _exit_for(status, child_rc)
    if exit_code == 0:
        history.update(status=status, streak=0)
    else:
        same = prev["status"] == status
        history.update(status=status, streak=prev["streak"] + 1 if same else 1)
    ended_t = time.time()
    last = {
        "id": unit.id,
        "kind": unit.kind,
        "status": status,
        "exit_code": exit_code,
        "wall_s": round(ended_t - started_t, 3),
        "host": host,
        "gpu": gpu,
        "slurm_job_id": slurm_job_id,
        "inputs_hash": unit.inputs_hash,
        "inputs_scheme": INPUTS_SCHEME,
        "needs": needs,
        "started": started,
        "ended": _now_utc(),
        "log": log_path if os.path.exists(log_path) else None,
        "message": message,
        "history": history,
        # where and how it ran: logged, never part of inputs_hash
        "launcher": unit.launcher,
        "cwd": unit.cwd,
        "env": dict(unit.env),
    }
    if refused_at_start:
        last["refused_at_start"] = True  # plan calibrate never bumps needs.minutes for a start refusal
    # 6. logs/last.json (shared) + logs/last.<id>.json (per unit: shards of one split share `out`).
    logs = os.path.join(out_dir, "logs")
    try:
        _write_json_atomic(os.path.join(logs, "last.json"), last)
        _write_json_atomic(os.path.join(logs, f"last.{unit.id}.json"), last)
    except OSError as e:  # pragma: no cover - unwritable out dir
        _log(f"[yavih] cannot write {logs}/last.json: {e}")
    _log(f"[yavih] unit {unit.id} {unit.kind} {status} wall_s={last['wall_s']} exit={exit_code}" + (f" ({message})" if message else ""))
    return last


def _dispatch(unit: Unit, *, env: dict[str, str], log_path: str, deadline: float | None,
              cpus: int | None = None, launcher: str | None = _UNRESOLVED) -> tuple[str, str | None, int | None]:
    """Step 5 for one kind. Returns ``(status, message, child_rc)``."""
    from yavih.plan.units import check_done

    _PREEMPT.clear()
    _FORWARDED.clear()
    if unit.kind == "extract":
        rc, killed = _run_extract(unit, env=env, log_path=log_path, cpus=cpus, launcher=launcher)
        if killed:
            return "preempted", f"child killed by signal (rc={rc})", rc
        if rc == 0:
            got = check_done(unit)
            if got is None:
                return "failed", f"child exited 0 but {unit.done} is missing or does not match the plan", 5
            return got, None, 0
        if rc == 1:
            return "misconfigured", f"child exit 1: usage / configuration error, resubmitting will not help (see {log_path})", rc
        if rc == 3:
            return "not_ready", "child exit 3", rc
        if rc == 4:
            return ("preempted" if _FORWARDED else "deadline"), "child exit 4", rc
        return "failed", f"child exit {rc} (see {log_path})", rc

    # probe / report: in-process; the same signals set the flag and the unit exits 4 preempted. The unit's
    # log gets a header and, on any exception, the full traceback, so a failed in-process unit is as
    # inspectable as an extract child.
    _log_header(log_path, unit, f"in-process pid={os.getpid()}")
    try:
        if unit.kind == "probe":
            scratch, problem = _scratch_preflight(unit, env=env, log_path=log_path)
            if problem:
                _log_append(log_path, f"[yavih] not_ready: {problem}\n")
                return "not_ready", problem, None
            _run_probe(unit, deadline=deadline, scratch=scratch)
        else:
            _run_report(unit)
    except KeyboardInterrupt:
        _log_traceback(log_path)
        return "preempted", "interrupted", None
    except TimeoutError as e:
        _log_traceback(log_path)
        return "deadline", f"{type(e).__name__}: {e}", None
    except BaseException as e:  # noqa: BLE001 - every failure must land in last.json
        _log_traceback(log_path)
        if _PREEMPT.is_set():
            return "preempted", f"{type(e).__name__}: {e}", None
        rc = _exception_exit(e)
        st = _status_for_code(rc) if rc in (1, 3) else "failed"
        return st, f"{type(e).__name__}: {e} (traceback: {log_path})", rc
    got = check_done(unit)
    if got is None:
        if _PREEMPT.is_set():
            return "preempted", "stopped by signal before DONE", None
        return "failed", f"{unit.kind} returned without writing {unit.done}", 5
    return got, None, None


# ----------------------------------------------------------------------------- several units


def _worst(lasts: list[dict]) -> dict:
    return max(lasts, key=lambda r: int(r.get("exit_code", 5)))


def _stops_the_run(rec: dict) -> bool:
    """After this result no further unit is started: the deadline is near or a signal arrived."""
    return rec.get("status") in ("deadline", "preempted") or _PREEMPT.is_set()


def _slot_record(unit: Unit, rc: int, out: str, err_tail: str) -> dict:
    """The ``last`` record a slot process reported (its ``--json`` envelope), or a failed one when it reported none."""
    try:
        env = json.loads(out.strip().splitlines()[-1]) if out.strip() else None
    except (ValueError, IndexError):
        env = None
    data = env.get("data") if isinstance(env, dict) else None
    if isinstance(data, dict) and data.get("id") == unit.id and "status" in data:
        return data
    return {"id": unit.id, "kind": unit.kind, "status": _status_for_code(rc) if rc else "failed",
            "exit_code": rc if 1 <= rc <= 5 else (4 if rc < 0 or rc > 128 else 5),
            "message": f"slot process exited {rc} without a record{': ' + err_tail if err_tail else ''}",
            "inputs_hash": unit.inputs_hash}


def _slot_argv(plan_path: str, unit: Unit, *, deadline: float | None, deps: str | None, root: str | None,
               cpus: int | None) -> list[str]:
    argv = [sys.executable, "-m", "yavih.cli", "--json", "exec", os.path.abspath(plan_path), "--id", unit.id,
            "--deadline", "none" if deadline is None else repr(float(deadline))]
    if deps:
        argv += ["--deps", os.path.abspath(deps)]
    if root:
        argv += ["--root", os.path.abspath(root)]
    if cpus is not None:
        argv += ["--cpus", str(int(cpus))]
    return argv


def run_pool(plan_path: str, *, pick: Callable[[], Unit | None], on_done: Callable[[Unit, dict], None], slots: int,
             deadline: float | None, deps: str | None = None, root: str | None = None, cpus: int | None = None,
             masks: list[str] | None = None) -> None:
    """Run units as ``slots`` concurrent ``yavih exec --id`` processes. ``pick()`` returns the
    next unit to start (``None``: nothing startable now); ``on_done(unit, last)`` receives each result. Slot ``g``
    gets device mask ``masks[g]`` (``device_masks``), ``cpus`` CPUs and ``OMP_NUM_THREADS=1``. A deadline or
    preemption result, or a signal to this process (forwarded to every slot), stops new starts; running slots are
    waited for. Separate processes: a probe's device choice, torch globals and stop flag stay its own."""
    masks = masks if masks is not None else device_masks(slots)
    free = list(range(slots))
    running: dict[int, tuple[threading.Thread, Unit, dict]] = {}
    stop = False

    def launch(g: int, unit: Unit, box: dict) -> None:
        env = dict(os.environ)
        env.update({SLOT_GPU_ENV: masks[g], "CUDA_VISIBLE_DEVICES": masks[g], "OMP_NUM_THREADS": "1"})
        if cpus is not None:
            env["YAVIH_CPUS"] = str(int(cpus))
        argv = _slot_argv(plan_path, unit, deadline=deadline, deps=deps, root=root, cpus=cpus)
        try:
            proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=None, text=True)
        except OSError as e:
            box["rec"] = _slot_record(unit, 5, "", f"{type(e).__name__}: {e}")
            return
        with _CHILDREN_LOCK:
            _CHILDREN.add(proc)
        try:
            out, _ = proc.communicate()
        finally:
            with _CHILDREN_LOCK:
                _CHILDREN.discard(proc)
        box["rec"] = _slot_record(unit, proc.returncode, out or "", "")
        box["rec"].setdefault("gpu", masks[g])

    while True:
        while free and not stop and not _PREEMPT.is_set():
            unit = pick()
            if unit is None:
                break
            g = free.pop(0)
            box: dict = {}
            t = threading.Thread(target=launch, args=(g, unit, box), name=f"yavih-slot-{g}", daemon=True)
            running[g] = (t, unit, box)
            t.start()
        if not running:
            return
        for g, (t, unit, box) in list(running.items()):
            t.join(0.1)  # a timed join keeps the main thread receptive to signals
            if t.is_alive():
                continue
            del running[g]
            free.append(g)
            free.sort()
            rec = box.get("rec") or _slot_record(unit, 5, "", "slot thread ended without a result")
            on_done(unit, rec)
            if _stops_the_run(rec):
                stop = True


# ----------------------------------------------------------------------------- entry point


def _plan_sha_check(plan_path: str, plan_sha: str | None) -> None:
    """``--plan-sha HEX``: the plan file's sha256 must start with ``HEX`` (8+ hex digits): an ``--index`` into a
    pending file rebuilt after the array was submitted would run another unit than planned."""
    if not plan_sha:
        return
    want = plan_sha.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{8,64}", want):
        raise ExecError(f"--plan-sha expects 8-64 hex digits of the plan's sha256, got {plan_sha!r}")
    with open(plan_path, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    if not got.startswith(want):
        raise ExecError(f"{plan_path} changed since it was submitted (sha256 {got[:12]}…, expected {want[:12]}…): "
                        "rebuild the pending file and resubmit; --index would select another unit")


def exec_unit(plan_path: str, *, index: int | None = None, unit_id: str | None = None,
              deadline: str | float | None = None, deadline_margin: float = DEFAULT_DEADLINE_MARGIN_S,
              slots: int | None = None, deps: str | None = None, unit_range: str | None = None,
              root: str | None = None, cpus: int | None = None, plan_sha: str | None = None) -> dict:
    """Run one unit, a ``--range``, or ``slots`` units concurrently; returns the ``logs/last.json`` content of a single
    unit, else ``{"units": [last…], "status", "exit_code", "id", "not_started"}`` with the worst exit code.

    ``slots=K`` with ``index``: units ``K*index … K*index+K-1``; with ``unit_range``: the range's units, at most ``K``
    at a time. Each slot is a separate ``yavih exec --id`` process with ``CUDA_VISIBLE_DEVICES`` = the ``g``-th
    inherited device, ``cpus // K`` (or ``cpus`` given) CPUs and ``OMP_NUM_THREADS=1`` (step 7). ``unit_range``
    alone runs the range in order in this process. A deadline or preemption stops further starts.

    ``deps`` (``--deps plan.jsonl``): a second plan file whose units are used **only to look up
    dependencies** (step 2). ``root`` (``--root``): the directory the plan's relative paths mean here
    (``read_plan``). ``cpus`` (``--cpus``): the CPU budget per unit (decode workers ``cpus - 1``).
    """
    from yavih.plan.units import read_plan

    selectors = sum(x is not None for x in (index, unit_id, unit_range))
    if selectors != 1:
        raise ExecError("exactly one of --index / --id / --range is required")
    if cpus is not None and cpus < 1:
        raise ExecError("--cpus must be >= 1")
    units = read_plan(plan_path, root=root)
    _plan_sha_check(plan_path, plan_sha)
    by_id: dict[str, Unit] = {}
    if deps:
        by_id.update((u.id, u) for u in read_plan(deps, root=root))
    by_id.update((u.id, u) for u in units)
    dl = resolve_deadline(deadline, margin_s=deadline_margin)

    if slots is not None:
        if slots < 1:
            raise ExecError("--slots must be >= 1")
        if unit_id is not None:
            raise ExecError("--slots needs --index (units K*index … K*index+K-1) or --range")
        if index is not None:
            lo = slots * index
            chosen = units[lo:lo + slots]
            if not chosen:
                raise ExecError(f"--index {index} with --slots {slots} selects no unit of {len(units)}")
        else:
            a, b = parse_range(unit_range or "", len(units))
            chosen = units[a:b]
        return _exec_slots(plan_path, chosen, deadline=dl, slots=slots, deps=deps, root=root, cpus=cpus)

    if unit_range is not None:
        a, b = parse_range(unit_range, len(units))
        return _exec_sequential(by_id, units[a:b], deadline=dl, cpus=cpus)

    if unit_id is not None:
        unit = next((u for u in units if u.id == unit_id), None)
        if unit is None:
            raise ExecError(f"unit {unit_id!r} is not in {plan_path}")
    else:
        assert index is not None
        if not 0 <= index < len(units):
            raise ExecError(f"--index {index} out of range for {len(units)} units")
        unit = units[index]
    with _Signals():
        return _run_unit(by_id, unit, deadline=dl, cpus=cpus)


def _summary(lasts: list[dict], chosen: list[Unit], *, key: str) -> dict:
    ran = {r["id"] for r in lasts}
    not_started = [u.id for u in chosen if u.id not in ran]
    if not lasts:
        return {key: [], "units": [], "status": "preempted", "exit_code": 4, "id": "", "not_started": not_started}
    worst = _worst(lasts)
    out = {"units": lasts, "status": worst["status"], "exit_code": int(worst["exit_code"]),
           "id": ",".join(r["id"] for r in lasts), "not_started": not_started}
    out[key] = lasts
    return out


def _exec_sequential(by_id: dict[str, Unit], chosen: list[Unit], *, deadline: float | None, cpus: int | None) -> dict:
    """``--range a:b``: the units in plan order, in this process, until a deadline or a signal stops the run."""
    lasts: list[dict] = []
    _PREEMPT.clear()
    _FORWARDED.clear()
    with _Signals():
        for u in chosen:
            if _PREEMPT.is_set():
                break
            rec = _run_unit(by_id, u, deadline=deadline, cpus=cpus)
            lasts.append(rec)
            if _stops_the_run(rec):
                break
    return _summary(lasts, chosen, key="units")


def _exec_slots(plan_path: str, chosen: list[Unit], *, deadline: float | None, slots: int, deps: str | None,
                root: str | None, cpus: int | None) -> dict:
    masks = device_masks(slots)
    queue = list(chosen)
    results: dict[str, dict] = {}
    per_slot = cpus if cpus is not None else max(1, int(max((u.needs.get("cpus") or 1) for u in chosen)) // slots)

    def pick() -> Unit | None:
        return queue.pop(0) if queue else None

    def on_done(u: Unit, rec: dict) -> None:
        results[u.id] = rec

    _PREEMPT.clear()
    _FORWARDED.clear()
    with _Signals():
        run_pool(plan_path, pick=pick, on_done=on_done, slots=slots, deadline=deadline, deps=deps, root=root,
                 cpus=per_slot, masks=masks)
    lasts = [results[u.id] for u in chosen if u.id in results]
    return _summary(lasts, chosen, key="slots")
