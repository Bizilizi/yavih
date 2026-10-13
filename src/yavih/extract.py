"""``python -m yavih.extract`` (the ``yavih extract`` and ``yavih bridge check`` backend).

args → ``Benchmark.load`` → ``Bridge(bench.view())`` → torch-free byte read-ahead → ``Bridge.prepare`` in
a thread pool ahead of ``encode`` (when the bridge defines it) → batching in row order → ``encode`` → ``used``
check → canaries → ``FeatureWriter``, each batch's stored bytes released once written; ``MediaUndecodable``
retry; the audit hook that turns a network connection during ``encode`` or ``prepare`` into ``E_NETWORK`` and
records writes outside the run directory; ``bridge check``'s hard rerun / batch / order invariance checks and
its ``prepare`` determinism check;
``E_TRAIN_MODE`` and precision enforcement when torch is importable in the MODEL env; the
``--expect-profile`` configuration-drift check (``E_CONFIG_DRIFT``); deadline and
SIGTERM/SIGUSR1 handling (exit 4, nothing partial). Core module: torch only if present, imported inside
functions.

Nothing the bridge decides is identity. Records carry the declared ``model`` and
``reading`` (names) and ``describe_sha256`` (consistency): resumption compares them, a drifted configuration
is refused against the ``bridge check`` profile, and a null ``weights_sha256`` is only the warning
``W_WEIGHTS_UNFINGERPRINTED``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import socket
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from yavih import hashing
from yavih import sample as _sample
from yavih.benchmark import Benchmark
from yavih.bridge import ArgError as _ArgError
from yavih.bridge import (
    BenchmarkView,
    Bridge,
    DescribeError,
    MediaUndecodable,
    NotApplicable,
    args_unconsumed,
    batch_tolerance,
    bridge_label,
    has_prepare,
    parse_args,
    preprocessing_source_check,
    validate_describe,
)
from yavih.io.claim import acquire_claim
from yavih.io.layout import LayoutError, freeze_layout, frozen_layout, layout_identity
from yavih.io.sidecar import (
    peek_shard,
    plan_values,
    read_not_applicable,
    read_sidecar,
    resume_shard,
    retire_foreign,
    utc_now,
    write_json_atomic,
    write_not_applicable,
)
from yavih.io.validate import rel_l2_stored
from yavih.io.writer import FeatureWriter, split_dir_for
from yavih.media import DecoderMissing, MediaMissing
from yavih.resources import MAX_DECODE_WORKERS  # per task; the one definition
from yavih.versions import __version__
from yavih.waivers import W_WAIVED, waivers_for

__all__ = [
    "main",
    "run_extract",
    "parse_rows",
    "bridge_check",
    "load_bridge_class",
    "source_sha256",
    "default_decode_workers",
    "parse_deadline",
    "scramble",
    "ExtractError",
    "E_TRAIN_MODE",
    "E_PRECISION_UNDECLARED",
    "E_CONFIG_DRIFT",
    "E_PROFILE",
    "E_USED",
    "E_NETWORK",
    "E_RERUN_DRIFT",
    "E_PREPARE_NONDETERMINISTIC",
    "E_BATCH_INVARIANCE",
    "E_ORACLE",
    "E_SMOKE",
    "E_TEMPLATE_UNFILLED",
    "E_CONSTANT_FEATURES",
    "E_SYNTHETIC",
    "unfilled_describe",
    "constant_features",
    "E_BRIDGE",
    "E_FEATURES",
    "E_PRECISION_DRIFT",
    "E_WRITE",
    "E_DEADLINE",
    "E_PREEMPTED",
    "W_SIDE_EFFECT",
    "W_NO_PREPROCESSING_SOURCE",
    "W_WEIGHTS_UNFINGERPRINTED",
    "W_RERUN_NONDETERMINISTIC",
    "N_RULE_CUSTOM",
    "MAX_DECODE_WORKERS",
    "DEFAULT_CANARIES",
    "DEFAULT_READ_AHEAD",
    "DEFAULT_DEADLINE_MARGIN_S",
]

E_TRAIN_MODE = "E_TRAIN_MODE"  # a torch.nn.Module in training mode after bridge __init__ (exit 2)
E_PRECISION_UNDECLARED = "E_PRECISION_UNDECLARED"  # null precision field other than autocast (exit 2)
E_CONFIG_DRIFT = "E_CONFIG_DRIFT"  # live describe() drifted from the --expect-profile (model, reading or describe_sha256; exit 2)
E_PROFILE = "E_PROFILE"  # --expect-profile unreadable or lacking model / reading / describe_sha256 (usage error, exit 1)
E_USED = "E_USED"  # encode() returned a missing or ill-formed `used` (exit 2)
E_NETWORK = "E_NETWORK"  # a socket.connect to the network inside encode or prepare (exit 2)
E_RERUN_DRIFT = "E_RERUN_DRIFT"  # bridge check: encoding the same batches twice is not bit-identical (exit 2)
E_PREPARE_NONDETERMINISTIC = "E_PREPARE_NONDETERMINISTIC"  # bridge check: prepare() twice on one sample differs (exit 2)
E_BATCH_INVARIANCE = "E_BATCH_INVARIANCE"  # bridge check: alone / shuffled pass differs beyond the tolerance (exit 2)
E_ORACLE = "E_ORACLE"  # bridge check refuses yavih.testing's oracle bridge unless YAVIH_TESTING=1 (exit 2)
E_SMOKE = "E_SMOKE"  # bridge check: no present row to encode in the smoke set (exit 3, data not ready)
E_TEMPLATE_UNFILLED = "E_TEMPLATE_UNFILLED"  # bridge check: TODO left in model / weights_ref / norm.source, or a null norm constant (exit 2)
E_CONSTANT_FEATURES = "E_CONSTANT_FEATURES"  # bridge check: every smoke clip encodes to the same features (exit 2)
E_SYNTHETIC = "E_SYNTHETIC"  # bridge check --synthetic: the tiny stand-in benchmark cannot be built (exit 1 usage / 3 media writer missing)
E_BRIDGE = "E_BRIDGE"  # the bridge raised inside encode or prepare: a bridge bug is not coverage (exit 5)
E_FEATURES = "E_FEATURES"  # encode() returned a wrong shape / pos for the declared pos_mode (exit 2)
E_PRECISION_DRIFT = "E_PRECISION_DRIFT"  # torch precision flags differ after encode from the enforced declaration (exit 2)
E_WRITE = "E_WRITE"  # the writer could not serialize a row's provenance/features (exit 5)
E_DEADLINE = "E_DEADLINE"  # --deadline reached; shard abandoned, nothing partial (exit 4)
E_PREEMPTED = "E_PREEMPTED"  # SIGTERM/SIGUSR1 received; shard abandoned, nothing partial (exit 4)
W_SIDE_EFFECT = "W_SIDE_EFFECT"  # writes outside the run directory during encode or prepare (warning)
W_NO_PREPROCESSING_SOURCE = "W_NO_PREPROCESSING_SOURCE"  # the bridge's docstring slot for the model's preprocessing source is empty
W_WEIGHTS_UNFINGERPRINTED = "W_WEIGHTS_UNFINGERPRINTED"  # describe()["weights_sha256"] is null: recorded, nothing gated
W_RERUN_NONDETERMINISTIC = "W_RERUN_NONDETERMINISTIC"  # bridge check: rerun within tolerance but not bit-identical (deterministic: false)
N_RULE_CUSTOM = "N_RULE_CUSTOM"  # a custom (opted-out) rule: a note in the profile, never a warning

DEFAULT_CANARIES = 8  # CLI/plan default; the library default is 0
DEFAULT_READ_AHEAD = 2  # batches read (and prepared) ahead of the one being encoded (--read-ahead)
DEFAULT_DEADLINE_MARGIN_S = 900.0  # `--deadline slurm` = SLURM_JOB_END_TIME - margin; > the 600 s sbatch signal lead, = exec's
FP16_MAX = 1e4  # FeatureWriter's float16 refusal threshold, applied to bridge check's smoke set
SCRAMBLE_ROW_BASE = 10**9  # bridge check's synthetic row ids


class ExtractError(Exception):
    """Carries the CLI exit code and an ``E_*`` code."""

    def __init__(self, code: str, message: str, exit_code: int = 2) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.exit_code = exit_code


def _log(msg: str) -> None:
    sys.stderr.write(f"[yavih] {msg}\n")
    sys.stderr.flush()


def _reason(e: BaseException) -> str:
    cls = type(e).__name__
    text = str(e)
    return text if text.startswith(cls) else f"{cls}: {text}"


# ----------------------------------------------------------------------------- decode workers


def default_decode_workers(env: Mapping[str, str] | None = None) -> int:
    """``yavih.resources.decode_workers`` (config/env (``YAVIH_CPUS`` set by ``exec --cpus`` / ``--slots``
    included), else CPUs − 1 for an allocation, else CPUs // 2, capped at 16). One definition for the whole package
    (``yavih.sample.default_decode_workers``): ``bridge check`` and
    ``extract`` must measure at the same worker count."""
    return _sample.default_decode_workers(env)


# ----------------------------------------------------------------------------- encode audit hook


class _Audit:
    """Process-wide state of the ``sys.addaudithook`` guard. ``sys.addaudithook`` cannot be removed, so the
    hook is installed once and does nothing unless ``armed`` (only around ``encode``) or the event comes from a
    ``yavih-prepare`` thread (those run bridge code only, so they are always audited)."""

    lock = threading.Lock()
    installed = False
    armed = 0
    run_dir: str | None = None
    network: list[str] = []
    writes: set[str] = set()
    #: ``--allow-loopback`` (opt-in): connects to 127.0.0.0/8, ::1 or ``localhost`` pass and are recorded
    allow_loopback = False
    loopback: list[str] = []
    #: the same opt-in for the prepare threads, set for the lifetime of an engine run (encode resets ``allow_loopback``)
    prepare_loopback = False


_PREPARE_THREADS = "yavih-prepare"  # thread-name prefix of Bridge.prepare's pool: always audited
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_IGNORED_PREFIXES = ("/dev/", "/proc/", "/sys/")


def _is_loopback(addr: Any) -> bool:
    """``True`` for a connect address on the loopback interface (``127.0.0.0/8``, ``::1``, ``localhost``)."""
    import ipaddress

    host = addr[0] if isinstance(addr, tuple) and addr else addr
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    host = host.strip().lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _audit_hook(event: str, args: tuple) -> None:
    if not _Audit.armed and not threading.current_thread().name.startswith(_PREPARE_THREADS):
        return
    if event == "socket.connect":
        sock = args[0] if args else None
        fam = getattr(sock, "family", None)
        if fam in (socket.AF_INET, getattr(socket, "AF_INET6", -1)):
            addr = args[1] if len(args) > 1 else None
            in_prepare = threading.current_thread().name.startswith(_PREPARE_THREADS)
            if (_Audit.allow_loopback or (in_prepare and _Audit.prepare_loopback)) and _is_loopback(addr):
                _Audit.loopback.append(repr(addr))
                return
            _Audit.network.append(repr(addr))
            raise ConnectionRefusedError(f"yavih: network access during encode or prepare is forbidden: connect to {addr!r}")
        return
    if event != "open" or threading.current_thread().name.startswith("yavih-read"):
        return
    path, mode, flags = (list(args) + [None, None, None])[:3]
    if not isinstance(path, str | bytes | os.PathLike):
        return
    writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (mode is None and isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
    if not writing:
        return
    p = os.path.abspath(os.fsdecode(path))
    if p.startswith(_IGNORED_PREFIXES):
        return
    rd = _Audit.run_dir
    if rd is not None and (p == rd or p.startswith(rd + os.sep)):
        return
    _Audit.writes.add(p)


def _install_audit_hook() -> None:
    with _Audit.lock:
        if not _Audit.installed:
            sys.addaudithook(_audit_hook)
            _Audit.installed = True


@contextlib.contextmanager
def _guarded(run_dir: str | os.PathLike | None, *, allow_loopback: bool = False) -> Iterator[None]:
    """Arm the network/side-effect guard for one ``encode`` call (``allow_loopback``: the ``--allow-loopback`` opt-in)."""
    _install_audit_hook()
    _Audit.run_dir = os.path.abspath(os.fspath(run_dir)) if run_dir is not None else None
    _Audit.allow_loopback = bool(allow_loopback)
    _Audit.armed += 1
    try:
        yield
    finally:
        _Audit.armed -= 1
        _Audit.allow_loopback = False


def _reset_audit() -> None:
    _Audit.network = []
    _Audit.writes = set()
    _Audit.loopback = []


# ----------------------------------------------------------------------------- bridge loading


def source_sha256(path: str | os.PathLike) -> str:
    """sha256 of the bridge file's bytes (sidecar ``bridge.source_sha256``, provenance only)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bridge_class(path: str) -> type[Bridge]:
    """Import the bridge file and return its single ``Bridge`` subclass (the matrix ``script``).

    ``path`` is a ``.py`` file; ``module:Class`` or ``module`` (dotted) is also accepted for packaged
    bridges. A file defining several subclasses must name the one to use in a module-level
    ``BRIDGE``; otherwise zero or several candidates is a usage error (exit 1).

    The bridge file's directory is put first on ``sys.path`` before the import, so bridges of one model
    can share a sibling module (``from mymodel_common import MyModelBase``) whatever the cwd, under
    ``bridge check`` and extraction alike. An import failure keeps its traceback in the message.
    """
    p = Path(path)
    if p.is_file():
        modname = f"yavih_bridge_{hashlib.sha256(os.fspath(p.resolve()).encode()).hexdigest()[:12]}"
        spec = importlib.util.spec_from_file_location(modname, p)
        if spec is None or spec.loader is None:
            raise _ArgError(f"cannot import bridge file {path!r}")
        bridge_dir = os.fspath(p.resolve().parent)
        if bridge_dir not in sys.path:
            sys.path.insert(0, bridge_dir)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            sys.modules.pop(modname, None)
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()
            raise _ArgError(f"bridge file {path!r} failed to import: {type(e).__name__}: {e}\n{tb}") from e
        explicit = None
    else:
        modpath, _, clsname = path.partition(":")
        try:
            mod = importlib.import_module(modpath)
        except ImportError as e:
            raise _ArgError(f"bridge {path!r} is neither a file nor an importable module: {e}") from e
        explicit = clsname or None
    if explicit:
        cls = getattr(mod, explicit, None)
        if not (inspect.isclass(cls) and issubclass(cls, Bridge)):
            raise _ArgError(f"{path!r}: {explicit} is not a Bridge subclass")
        return cls
    picked = getattr(mod, "BRIDGE", None)
    if inspect.isclass(picked) and issubclass(picked, Bridge):
        return picked
    candidates = [
        c for c in vars(mod).values()
        if inspect.isclass(c) and issubclass(c, Bridge) and c is not Bridge and c.__module__ == mod.__name__
    ]
    if len(candidates) != 1:
        names = [c.__name__ for c in candidates]
        raise _ArgError(
            f"{path!r} must define exactly one yavih.Bridge subclass (found {names or 'none'}); "
            "set a module-level BRIDGE = <class> to pick one"
        )
    return candidates[0]


def _bridge_source(bridge_cls: type) -> tuple[str, str | None]:
    """``(source_sha256, source_path)`` of a bridge class given directly (tests, ``yavih.testing``):
    the file when there is one, else the source text, else the qualified name."""
    try:
        f = inspect.getsourcefile(bridge_cls)
    except (TypeError, OSError):
        f = None
    if f and Path(f).is_file():
        return source_sha256(f), f
    try:
        return hashlib.sha256(inspect.getsource(bridge_cls).encode()).hexdigest(), None
    except (TypeError, OSError):
        return hashlib.sha256(f"{bridge_cls.__module__}.{bridge_cls.__qualname__}".encode()).hexdigest(), None


# ----------------------------------------------------------------------------- deadline / signals


def parse_deadline(text: str | None, env: Mapping[str, str] | None = None, *, margin: float = DEFAULT_DEADLINE_MARGIN_S) -> float | None:
    """``--deadline`` → epoch seconds or ``None``: ``yavih.deadline.parse_deadline``, the one parser
    ``exec``, ``extract`` and ``probe run`` share. ``None`` → ``$YAVIH_DEADLINE`` (set by ``yavih exec``) if present;
    ``none``; ``slurm`` (``SLURM_JOB_END_TIME − margin``; unset inside a SLURM job → ``DeadlineError``, exit 1);
    ``+<seconds>`` / a duration; an epoch; an ISO time."""
    from yavih.deadline import parse_deadline as _parse

    return _parse(text, env=env, margin_s=margin)


class _Deadline:
    """Shared deadline state: ``--deadline`` plus the SIGTERM/SIGUSR1 handler that sets
    ``deadline = now`` so the current batch finishes and the shard is abandoned."""

    SIGNALS = ("SIGTERM", "SIGUSR1")

    def __init__(self, deadline: float | None) -> None:
        self.deadline = deadline
        self.signalled: str | None = None
        self._previous: dict[int, Any] = {}

    def _handler(self, signum: int, frame: Any) -> None:  # noqa: ARG002 - signal signature
        self.signalled = signal.Signals(signum).name
        self.deadline = time.time()

    def install(self) -> None:
        # signal.signal only works in the main thread; a library caller on a worker thread simply
        # gets no preemption handler (the deadline still works).
        if threading.current_thread() is not threading.main_thread():
            return
        for name in self.SIGNALS:
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                self._previous[signum] = signal.signal(signum, self._handler)
            except (ValueError, OSError):  # pragma: no cover - platform without the signal
                pass

    def restore(self) -> None:
        for signum, prev in self._previous.items():
            try:
                signal.signal(signum, prev if prev is not None else signal.SIG_DFL)
            except (ValueError, OSError):  # pragma: no cover
                pass
        self._previous.clear()

    def check(self, estimated_batch_s: float = 0.0) -> None:
        """Raise exit 4 when the next batch would not fit (``now + estimated_batch_s > deadline``)."""
        if self.signalled is not None:
            raise ExtractError(E_PREEMPTED, f"{self.signalled} received; shard abandoned after the current batch", exit_code=4)
        if self.deadline is not None and time.time() + max(0.0, estimated_batch_s) > self.deadline:
            raise ExtractError(
                E_DEADLINE,
                f"deadline reached (next batch ≈ {estimated_batch_s:.1f}s would overrun); shard abandoned, resubmit",
                exit_code=4,
            )


# ----------------------------------------------------------------------------- torch (optional)


def _torch() -> Any | None:
    """The torch module when importable in this env, else ``None`` (torch only if present)."""
    if "torch" in sys.modules:
        return sys.modules["torch"]
    try:
        if importlib.util.find_spec("torch") is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _find_training_modules(bridge: Bridge, torch: Any, *, max_depth: int = 8) -> list[str]:
    """Walk the bridge instance's attributes (recursively through containers and plain objects), the
    bridge class's own attributes and its module's globals for ``torch.nn.Module`` objects and return
    the dotted names of those in training mode. The class/module walk covers the common
    agent-written shape ``MODEL = load()`` at import time, which ``vars(bridge)`` never sees."""
    found: list[str] = []
    seen: set[int] = set()
    Module, Tensor = torch.nn.Module, torch.Tensor

    def visit(x: Any, path: str, depth: int) -> None:
        if depth > max_depth or id(x) in seen:
            return
        if isinstance(x, Module):
            seen.add(id(x))
            for name, m in x.named_modules():
                if m.training:
                    found.append(f"{path}.{name}" if name else path)
            return
        if x is None or isinstance(x, str | bytes | int | float | bool | np.ndarray | Tensor | type):
            return
        if inspect.ismodule(x) or inspect.isroutine(x):
            return
        if hasattr(x, "manifest_hash") and hasattr(x, "protocol_hash"):
            return  # a Benchmark is the reader's, never the model's
        seen.add(id(x))
        if isinstance(x, Mapping):
            for k, v in x.items():
                visit(v, f"{path}[{k!r}]", depth + 1)
        elif isinstance(x, list | tuple | set | frozenset):
            for i, v in enumerate(x):
                visit(v, f"{path}[{i}]", depth + 1)
        elif hasattr(x, "__dict__"):
            for k, v in vars(x).items():
                visit(v, f"{path}.{k}", depth + 1)

    for k, v in vars(bridge).items():
        if k == "bench":  # the Benchmark is the reader's, never the model's
            continue
        visit(v, f"self.{k}", 1)
    cls = type(bridge)
    for k, v in vars(cls).items():
        if not k.startswith("__"):
            visit(v, f"{cls.__name__}.{k}", 1)
    mod = sys.modules.get(cls.__module__)
    if mod is not None:
        for k, v in vars(mod).items():
            if k.startswith("__") or inspect.isclass(v) or inspect.ismodule(v) or inspect.isroutine(v):
                continue
            visit(v, f"{cls.__module__}.{k}", 1)
    return found


def _precision_live(torch: Any) -> dict[str, Any]:
    return {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _enforce_precision(prec: Mapping[str, Any], torch: Any | None) -> dict[str, Any] | None:
    """Apply ``describe()["precision"]``: matmul precision, TF32, deterministic algorithms.
    ``autocast`` is declared, not applied: the bridge wraps its own forward. Returns the live values."""
    if torch is None:
        return None
    # Order matters: set_float32_matmul_precision rewrites the TF32 matmul flag, so the explicit
    # allow_tf32 declaration is applied afterwards and the live block reports what actually holds.
    torch.set_float32_matmul_precision(prec["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = bool(prec["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(prec["allow_tf32"])
    if prec["deterministic"]:
        # cuBLAS needs this before the first CUDA matmul for deterministic algorithms (PyTorch docs).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(bool(prec["deterministic"]))
    return _precision_live(torch)


def _inference_ctx(torch: Any | None) -> contextlib.AbstractContextManager:
    return torch.inference_mode() if torch is not None else contextlib.nullcontext()


def _env_block(torch: Any | None, decode_workers: int, precision_live: dict | None, env: Mapping[str, str] | None = None, *, read_ahead: int = DEFAULT_READ_AHEAD) -> dict[str, Any]:
    env = os.environ if env is None else env
    job = env.get("SLURM_JOB_ID")
    task = env.get("SLURM_ARRAY_TASK_ID")
    slurm_id = f"{env.get('SLURM_ARRAY_JOB_ID', job)}_{task}" if job and task else job
    cuda = gpu = None
    tversion = None
    if torch is not None:
        tversion = str(torch.__version__)
        try:
            if torch.cuda.is_available():
                cuda = torch.version.cuda
                gpu = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 - a broken driver must not fail a shard's provenance
            pass
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": tversion,
        "cuda": cuda,
        "gpu": gpu,
        "slurm_job_id": slurm_id,
        "decode_workers": int(decode_workers),
        "read_ahead": int(read_ahead),
        "precision_live": precision_live,
    }


def _peak_gpu_gb(torch: Any | None, bridge: Bridge) -> float | None:
    """Peak CUDA memory of the smoke set; ``0.0`` when torch is present but sees no CUDA device (a CPU-only run: plan
    then requests ``gpu: 0``), the bridge's own ``profile()`` figure, else ``None`` (unknown, e.g. no torch)."""
    prof = bridge.profile() if bridge is not None else None
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return float(torch.cuda.max_memory_allocated()) / 1e9
            if not (isinstance(prof, Mapping) and prof.get("peak_gpu_gb") is not None):
                return 0.0
        except Exception:  # noqa: BLE001
            pass
    if isinstance(prof, Mapping) and prof.get("peak_gpu_gb") is not None:
        return float(prof["peak_gpu_gb"])
    return None


# ----------------------------------------------------------------------------- benchmark helpers


def _load_bench(bench: str | Any, data: dict | None) -> Any:
    """``Benchmark.load(name)`` for a string; a ``Benchmark``(-like) object is used as-is so library
    callers and tests can hand one in directly."""
    if isinstance(bench, str | os.PathLike):
        return Benchmark.load(os.fspath(bench), data=data)
    if not hasattr(bench, "reader"):
        raise TypeError(f"bench must be a benchmark name/path or a Benchmark, got {type(bench).__name__}")
    return bench


def _bench_view(bench_obj: Any) -> BenchmarkView:
    """What the bridge receives: ``Benchmark.view()``, or the same built from the ``Benchmark`` surface."""
    return BenchmarkView.from_benchmark(bench_obj)


def _media_keys(bench_obj: Any, split: str) -> Sequence[str | None] | None:
    """Manifest ``media_key`` per split row: written into members by ``run_extract``, never handed
    to the bridge. ``None`` when the benchmark object has no manifest (duck-typed test benchmarks)."""
    fn = getattr(bench_obj, "manifest", None)
    if not callable(fn):
        return None
    try:
        return list(fn(split).media_key)
    except Exception:  # noqa: BLE001 - provenance only (feeds E_DUPLICATE_CLIPS exemptions)
        return None


def _is_oracle(bridge_cls: type) -> bool:
    """``yavih.testing.oracle_bridge`` (features looked up by clip_id: breaks the encode guardrails on purpose)."""
    if getattr(bridge_cls, "ORACLE", False):
        return True
    return any(c.__name__ == "OracleBridgeBase" and c.__module__ == "yavih.testing" for c in getattr(bridge_cls, "__mro__", ()))


def _canary_of(s: Any) -> int | None:
    v = getattr(s, "_canary_of", None)
    return None if v is None else int(v)


def _check_header_backends(bench_obj: Any) -> None:
    """Fail at start, not per row: every row's media facts come from the site census or a live header
    probe (PyAV / soundfile / pillow, else ffprobe); with neither for a stream kind every row would raise
    ``DecoderMissing``. Raises it once (exit 3), naming the fixes. A duck-typed benchmark without ``media_info``
    or ``spec.streams`` is not checked."""
    from yavih.media.info import missing_header_backends

    fn = getattr(bench_obj, "media_info", None)
    streams = getattr(getattr(bench_obj, "spec", None), "streams", None)
    if not callable(fn) or not isinstance(streams, Mapping):
        return
    try:
        if fn() is not None:
            return
    except Exception:  # noqa: BLE001 - an unreadable census is the reader's error to report
        return
    kinds = [str(getattr(s, "kind", s)) for s in streams.values()]
    missing = missing_header_backends(kinds)
    if missing:
        raise DecoderMissing(
            "av" if any(k in ("video", "audio") for k in missing) else "pillow",
            f"no media-info census for {getattr(bench_obj, 'name', '?')} and no header backend for stream kind(s) {missing}: "
            "pip install av soundfile pillow (or 'yavih[media]'), or put ffprobe on PATH, or copy a census made elsewhere with "
            f"`yavih bench verify {getattr(bench_obj, 'name', '<b>')} --census` into $YAVIH_HOME/media_info/",
        )


# ----------------------------------------------------------------------------- preflight


@dataclass
class _Prepared:
    """Everything the bridge preflight produces before any row is read."""

    bench: Any
    view: BenchmarkView
    bridge_cls: type[Bridge]
    bridge: Bridge | None  # None when __init__ raised NotApplicable
    args: dict[str, Any]
    describe: dict | None
    model: str | None  # describe()["model"] (or the profile's when __init__ raised NotApplicable)
    reading: str | None
    describe_sha256: str | None
    source_sha256: str
    source_path: str | None
    precision_live: dict | None
    torch: Any
    not_applicable: str | None = None


def _resolve_bridge(bridge: str | type[Bridge]) -> tuple[type[Bridge], str, str | None]:
    if isinstance(bridge, type) and issubclass(bridge, Bridge):
        sha, path = _bridge_source(bridge)
        return bridge, sha, path
    if isinstance(bridge, str | os.PathLike):
        cls = load_bridge_class(os.fspath(bridge))
        p = Path(bridge)
        if p.is_file():
            return cls, source_sha256(p), os.fspath(p)
        sha, path = _bridge_source(cls)
        return cls, sha, path
    raise TypeError(f"bridge must be a path, 'module:Class' or a Bridge subclass, got {type(bridge).__name__}")


@dataclass(frozen=True)
class _Planned:
    """What ``--expect-profile`` pins: the planned configuration from the ``bridge check`` profile."""

    path: str
    model: str
    reading: str
    describe_sha256: str
    describe: dict
    batch_size: int | None
    dtype: str | None
    pos_mode: str | None


def _load_profile(path: str | os.PathLike) -> _Planned:
    """Read the ``--expect-profile``: ``model``, ``reading`` and ``describe_sha256`` (top level, else
    derived from ``describe``). Unreadable or incomplete → ``ExtractError(E_PROFILE)``, exit 1 (a usage error: plan
    always passes the profile it planned from)."""
    try:
        with open(path, encoding="utf-8") as fh:
            prof = json.load(fh)
    except (OSError, ValueError) as e:
        raise ExtractError(E_PROFILE, f"--expect-profile {os.fspath(path)}: unreadable ({e}); re-run `bridge check` and `plan build`", exit_code=1) from e
    if not isinstance(prof, Mapping):
        raise ExtractError(E_PROFILE, f"--expect-profile {os.fspath(path)}: not a JSON object", exit_code=1)
    d = prof.get("describe") if isinstance(prof.get("describe"), Mapping) else {}
    model = prof.get("model") or d.get("model")
    reading = prof.get("reading") or d.get("reading")
    dsha = prof.get("describe_sha256") or (hashing.describe_sha256(d) if d else None)
    lacking = [k for k, v in (("model", model), ("reading", reading), ("describe_sha256", dsha)) if not isinstance(v, str) or not v]
    if lacking:
        raise ExtractError(E_PROFILE, f"--expect-profile {os.fspath(path)} lacks {lacking}; "
                           "re-run `bridge check` and `plan build`", exit_code=1)

    def _opt(key: str, typ: type) -> Any:
        v = prof.get(key, d.get(key))
        return typ(v) if v is not None else None

    return _Planned(path=os.fspath(path), model=str(model), reading=str(reading), describe_sha256=str(dsha), describe=dict(d),
                    batch_size=_opt("batch_size", int), dtype=_opt("dtype", str), pos_mode=_opt("pos_mode", str))


def config_drift(live: Mapping[str, Any], planned: Mapping[str, Any]) -> list[str]:
    """``describe()`` keys whose values differ between the live and the planned configuration, ``DESCRIBE_SHA_EXCLUDED``
    left out (``E_CONFIG_DRIFT``)."""
    a, b = hashing.describe_subset(live), hashing.describe_subset(planned)
    return [k for k in sorted(set(a) | set(b)) if hashing.canon(a.get(k)) != hashing.canon(b.get(k))]


def _prepare(
    *,
    bench: str | Any,
    bridge: str | type[Bridge],
    args: Mapping[str, Any] | Sequence[str] | None,
    planned: _Planned | None,
    data: dict | None,
    decode_workers: int = 1,
) -> _Prepared:
    """The bridge start-up contract, in order: load benchmark, import bridge, parse ``--arg`` (unknown →
    exit 1), construct with the ``BenchmarkView`` (``NotApplicable`` allowed), ``describe()`` →
    ``validate_describe`` (exit 2), ``E_TRAIN_MODE`` walk, precision enforcement, the configuration-drift
    check against the planned profile (``E_CONFIG_DRIFT``), ``Bridge.decode_workers``."""
    bench_obj = _load_bench(bench, data)
    view = _bench_view(bench_obj)
    bridge_cls, src_sha, src_path = _resolve_bridge(bridge)
    parsed = parse_args(getattr(bridge_cls, "ARGS", {}) or {}, args)
    torch = _torch()

    try:
        inst = bridge_cls(view, parsed)
    except NotApplicable as na:
        return _Prepared(bench_obj, view, bridge_cls, None, parsed, None,
                         planned.model if planned else None, planned.reading if planned else None, None,
                         src_sha, src_path, None, torch, na.reason)

    try:
        d = inst.describe()
    except NotApplicable:
        raise
    except Exception as e:
        raise DescribeError(f"describe() raised {type(e).__name__}: {e}") from e
    validate_describe(d, inst, view)
    d = json.loads(hashing.canon(d))  # the describe() exactly as hashed: tuples → lists, numpy scalars → python
    dsha = hashing.describe_sha256(d)
    if planned is not None and (d["model"] != planned.model or d["reading"] != planned.reading or dsha != planned.describe_sha256):
        keys = config_drift(d, planned.describe) if planned.describe else []
        for k, live_v, plan_v in (("model", d["model"], planned.model), ("reading", d["reading"], planned.reading)):
            if live_v != plan_v and k not in keys:
                keys.insert(0, k)
        raise ExtractError(
            E_CONFIG_DRIFT,
            f"the bridge's configuration drifted since `bridge check`: keys {keys or ['(describe_sha256 only)']} differ "
            f"(live {d['model']}/{d['reading']} describe_sha256 {dsha[:12]}, profile {planned.model}/{planned.reading} "
            f"{planned.describe_sha256[:12]} in {planned.path}); re-run `bridge check` and `plan build`; nothing written",
            exit_code=2,
        )
    if torch is not None:
        training = _find_training_modules(inst, torch)
        if training:
            raise ExtractError(
                E_TRAIN_MODE,
                f"torch module(s) still in training mode after {bridge_cls.__name__}.__init__: {training[:8]}"
                f"{' …' if len(training) > 8 else ''}; call .eval() in the bridge (yavih never fixes it silently)",
                exit_code=2,
            )
    live = _enforce_precision(d["precision"], torch)
    inst.decode_workers = max(1, int(decode_workers))  # the bridge's CPU budget for its own decoding
    return _Prepared(bench_obj, view, bridge_cls, inst, parsed, d, str(d["model"]), str(d["reading"]), dsha, src_sha, src_path, live, torch)


# ----------------------------------------------------------------------------- batch engine


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _read_sample(sample: Any) -> float:
    """Read every stored byte of a Sample and resolve its media facts: ``yavih.sample.fetch``. The
    RawStreams cache both, so the bridge's own read is a memory hit and the member's ``media`` block is free."""
    return _sample.fetch(sample)


def _release_bytes(sample: Any) -> None:
    """Drop the cached stored bytes of every stream of ``sample`` (``yavih.sample._Source.release``; a later read
    reads the store again). Streams without a byte source (duck-typed test samples) are left alone."""
    for st in (getattr(sample, "streams", None) or {}).values():
        rel = getattr(getattr(st, "_src", None), "release", None)
        if rel is not None:
            rel()


def _set_prepared(sample: Any, payload: Any) -> None:
    """``sample.prepared = payload`` on the frozen ``Sample`` (or a duck-typed stand-in)."""
    object.__setattr__(sample, "prepared", payload)


@dataclass
class _Stats:
    read_s: float = 0.0
    prepare_s: float = 0.0
    encode_s: float = 0.0
    n_encoded: int = 0
    batch_times: list[float] = field(default_factory=list)

    def estimated_batch_s(self) -> float:
        # The max of the last three batches: a decode-bound shard has a heavy tail (grouped manifests),
        # and overrunning a deadline costs a whole resubmission.
        return max(self.batch_times[-3:]) if self.batch_times else 0.0


@dataclass
class _Out:
    """One encoded batch: the kept samples and, aligned with them, features, positions and ``used``."""

    samples: list[Any]
    feats: np.ndarray | None = None
    pos: np.ndarray | None = None  # [N, 4] (constant) | [B, N, 4] (per_sample) | None
    used: list[dict] = field(default_factory=list)

    def pos_of(self, k: int) -> np.ndarray | None:
        if self.pos is None:
            return None
        return self.pos if self.pos.ndim == 2 else self.pos[k]


class _Engine:
    """Drives the byte read-ahead, the bridge's ``prepare`` (when it has one) and ``encode`` over batches in row order.

    Missing rows (media absent or unreadable in the read-ahead, ``MediaUndecodable`` / ``MediaMissing`` from
    ``prepare`` or ``encode``) are reported through ``on_missing`` and never reach the bridge again; any other
    exception out of ``prepare`` or ``encode`` is a bridge bug (``E_BRIDGE``, exit 5). A network connection inside
    either is ``E_NETWORK`` (exit 2); writes outside ``run_dir`` are collected for ``W_SIDE_EFFECT``.

    A bridge that overrides ``Bridge.prepare`` gets a ``yavih-prepare`` pool of ``decode_workers`` threads that
    prepares the batches the read-ahead delivered (``read_ahead + 1`` batches in flight) while ``encode`` runs;
    without ``prepare`` no pool is created. ``release_bytes`` (``run_extract``): the stored bytes of a sample are
    dropped after its ``prepare`` (unless ``Bridge.keep_bytes``) and, for every bridge, once its batch has been
    consumed (written); ``bridge check`` keeps them, its passes share the byte sources.
    """

    def __init__(
        self,
        bridge: Bridge,
        describe: Mapping[str, Any],
        *,
        decode_workers: int,
        read_ahead: int,
        torch: Any,
        on_missing: Callable[[Any, str], None],
        kinds: Mapping[str, str],
        run_dir: str | os.PathLike | None,
        allow_not_applicable: bool = False,
        allow_loopback: bool = False,
        release_bytes: bool = False,
    ) -> None:
        self.bridge = bridge
        self.has_prepare = has_prepare(bridge)
        self.keep_bytes = bool(getattr(bridge, "keep_bytes", False))
        self.release_bytes = bool(release_bytes)
        self._net_mark = self._loop_mark = 0
        self.allow_loopback = bool(allow_loopback)
        self.loopback: list[str] = []  # loopback connects the --allow-loopback opt-in let through (recorded)
        self.describe = describe
        self.inputs = list((describe.get("input") or {}).keys())
        self.workers = max(1, int(decode_workers))
        self.read_ahead = max(0, int(read_ahead))
        self.torch = torch
        self.on_missing = on_missing
        self.kinds = dict(kinds)
        self.run_dir = run_dir
        self.allow_not_applicable = allow_not_applicable
        self.stats = _Stats()
        self.first_encode_done = False
        self.side_effects: set[str] = set()

    def _note_loopback(self, since: int) -> None:
        for a in _Audit.loopback[since:]:
            if a not in self.loopback:
                self.loopback.append(a)

    # -- prepare (decode ahead) -------------------------------------------------------
    def _prepare_one(self, s: Any) -> tuple[Any, float]:
        """``bridge.prepare(s)`` in a ``yavih-prepare`` thread under ``torch.inference_mode()``; then the sample's
        stored bytes are released unless the bridge keeps them. Returns ``(payload, seconds)``."""
        t0 = time.perf_counter()
        try:
            with _inference_ctx(self.torch):
                payload = self.bridge.prepare(s)
        finally:
            if self.release_bytes and not self.keep_bytes:
                _release_bytes(s)
        return payload, time.perf_counter() - t0

    def _collect_prepared(self, rb: _sample.ReadBatch, futures: list[Future]) -> _sample.ReadBatch:
        """Wait for one batch's ``prepare`` calls and set ``sample.prepared``. ``MediaUndecodable`` naming the sample or
        ``MediaMissing`` → a missing row (added to the batch's ``missing``); ``DecoderMissing`` propagates; a network
        connect → ``E_NETWORK``; any other exception → ``E_BRIDGE`` (exit 5), first failing row in order."""
        outcomes: list[tuple[Any, tuple[Any, float] | None, Exception | None]] = []
        for s, f in zip(rb.samples, futures, strict=True):
            try:
                outcomes.append((s, f.result(), None))
            except Exception as e:  # noqa: BLE001 - classified below, after the network check
                outcomes.append((s, None, e))
        self.side_effects |= _Audit.writes
        self._note_loopback(self._loop_mark)
        net = _Audit.network[self._net_mark:]
        if net:
            raise ExtractError(E_NETWORK, f"the bridge opened network connection(s) during prepare: {net[:3]}; prepare decodes the sample "
                               "it is handed and nothing else", exit_code=2)
        present: list[Any] = []
        missing = list(rb.missing)
        name = type(self.bridge).__name__
        for s, res, e in outcomes:
            if e is None:
                assert res is not None
                _set_prepared(s, res[0])
                self.stats.prepare_s += res[1]
                present.append(s)
            elif isinstance(e, ExtractError | DecoderMissing):
                raise e
            elif isinstance(e, NotApplicable):
                if self.allow_not_applicable and not self.first_encode_done:
                    raise e
                raise ExtractError(E_BRIDGE, f"NotApplicable raised by prepare after the first encode ({e.reason}); it is only valid from "
                                   "__init__ or the first batch", exit_code=5) from e
            elif isinstance(e, MediaMissing) or (isinstance(e, MediaUndecodable) and s.clip_id in e.clip_ids):
                missing.append((s, e))
            else:
                tb = "".join(traceback.format_exception(type(e), e, e.__traceback__)).rstrip()
                what = (f"MediaUndecodable naming {e.clip_ids[:5]}, not the sample it was handed ({s.clip_id})" if isinstance(e, MediaUndecodable)
                        else f"{type(e).__name__}: {e}")
                raise ExtractError(E_BRIDGE, f"bridge {name}.prepare raised {what} on row {s.row} — a bridge bug is not coverage\n{tb}",
                                   exit_code=5) from e
        return _sample.ReadBatch(present, missing, rb.read_s)

    def _prepare_ahead(self, reads: Iterator[_sample.ReadBatch]) -> Iterator[_sample.ReadBatch]:
        """The read-ahead's batches with ``prepare`` run on every present Sample by a ``yavih-prepare`` pool of
        ``decode_workers`` threads, ``read_ahead + 1`` batches of per-sample futures in flight, in input order."""
        self._net_mark, self._loop_mark = len(_Audit.network), len(_Audit.loopback)
        pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=_PREPARE_THREADS)
        pending: deque[tuple[_sample.ReadBatch, list[Future]]] = deque()
        exhausted = False
        try:
            while True:
                while not exhausted and len(pending) < self.read_ahead + 1:
                    try:
                        rb = next(reads)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append((rb, [pool.submit(self._prepare_one, s) for s in rb.samples]))
                if not pending:
                    return
                rb, futures = pending.popleft()
                yield self._collect_prepared(rb, futures)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _batch_done(self, rb: _sample.ReadBatch) -> None:
        """After the consumer handled a batch: drop the prepared payloads and, with ``release_bytes``, the stored bytes."""
        for s in [*rb.samples, *(m for m, _ in rb.missing)]:
            if self.has_prepare:
                _set_prepared(s, None)
            if self.release_bytes:
                _release_bytes(s)

    # -- encode ---------------------------------------------------------------------
    def _check_used(self, samples: list[Any], used: Any) -> list[dict]:
        from yavih.io.validate import _infer_kind, _well_formed_used

        B = len(samples)
        if not isinstance(used, list | tuple) or len(used) != B:
            raise ExtractError(E_USED, f"encode() must return used as a list of B={B} dicts (one per sample), got "
                               f"{type(used).__name__}{'' if not isinstance(used, list | tuple) else f' of length {len(used)}'}", exit_code=2)
        out: list[dict] = []
        for s, u in zip(samples, used, strict=True):
            if not isinstance(u, Mapping):
                raise ExtractError(E_USED, f"{s.clip_id}: used entry must be a dict per stream, got {type(u).__name__}", exit_code=2)
            missing = [n for n in self.inputs if n not in u]
            if missing:
                raise ExtractError(E_USED, f"{s.clip_id}: used lacks stream(s) {missing} declared in describe()['input']", exit_code=2)
            for name in self.inputs:
                entry = (self.describe.get("input") or {}).get(name) or {}
                info = getattr(s.streams.get(name), "info", None)
                kind = self.kinds.get(name) or _infer_kind(entry, u[name], info) or "video"
                bad = _well_formed_used(kind, u[name])
                if bad is not None:
                    raise ExtractError(E_USED, f"{s.clip_id}: used[{name!r}]: {bad}", exit_code=2)
            try:
                out.append(json.loads(hashing.canon(dict(u))))
            except (TypeError, ValueError) as e:
                raise ExtractError(E_USED, f"{s.clip_id}: used is not JSON primitives: {e}", exit_code=2) from e
        return out

    def _encode_once(self, samples: list[Any]) -> _Out:
        t0 = time.perf_counter()
        n_net = len(_Audit.network)
        n_loop = len(_Audit.loopback)
        try:
            with _guarded(self.run_dir, allow_loopback=self.allow_loopback), _inference_ctx(self.torch):
                out = self.bridge.encode(list(samples))
        finally:
            self.stats.encode_s += time.perf_counter() - t0
            self.side_effects |= _Audit.writes
            self._note_loopback(n_loop)
            net = _Audit.network[n_net:]
            if net:
                raise ExtractError(E_NETWORK, f"the bridge opened network connection(s) during encode: {net[:3]}; weights and everything else "
                                   "must be loaded before the first sample", exit_code=2)
        if not (isinstance(out, tuple | list) and len(out) == 3):
            got = f"a {len(out)}-tuple" if isinstance(out, tuple | list) else type(out).__name__
            raise ExtractError(E_USED if isinstance(out, tuple | list) and len(out) == 2 else E_FEATURES,
                               f"encode() must return (feats, pos, used), got {got}", exit_code=2)
        feats = _to_numpy(out[0])
        pos = _to_numpy(out[1]) if out[1] is not None else None
        B = len(samples)
        if feats.ndim != 3 or feats.shape[0] != B:
            raise ExtractError(E_FEATURES, f"encode() feats must be [B={B}, N, D], got shape {tuple(feats.shape)}", exit_code=2)
        mode = self.describe["pos_mode"]
        if mode == "none":
            if pos is not None:
                raise ExtractError(E_FEATURES, "pos_mode 'none' but encode() returned pos", exit_code=2)
        elif mode == "constant":
            if pos is None or pos.ndim != 2 or pos.shape != (feats.shape[1], 4):
                got = None if pos is None else tuple(pos.shape)
                raise ExtractError(E_FEATURES, f"pos_mode 'constant' needs pos [N={feats.shape[1]}, 4], got {got}", exit_code=2)
        else:
            if pos is None or pos.ndim != 3 or pos.shape != (B, feats.shape[1], 4):
                got = None if pos is None else tuple(pos.shape)
                raise ExtractError(E_FEATURES, f"pos_mode 'per_sample' needs pos [B={B}, N={feats.shape[1]}, 4], got {got}", exit_code=2)
        if pos is not None:
            pos = np.ascontiguousarray(pos, dtype=np.float32)
        used = self._check_used(list(samples), out[2])
        return _Out(list(samples), feats, pos, used)

    def encode(self, samples: list[Any]) -> _Out:
        """Encode a batch. ``MediaUndecodable`` (or a ``MediaMissing`` the bridge's own read raised) drops the
        named rows and re-encodes the rest in one batch — exact, by batch invariance."""
        if not samples:
            return _Out([])
        try:
            res = self._encode_once(samples)
        except NotApplicable as na:
            if self.allow_not_applicable and not self.first_encode_done:
                raise
            raise ExtractError(E_BRIDGE, f"NotApplicable raised after the first encode ({na.reason}); it is only valid from __init__ or the first encode", exit_code=5) from na
        except (ExtractError, DecoderMissing):
            raise
        except (MediaUndecodable, MediaMissing) as e:
            ids = list(e.clip_ids) if isinstance(e, MediaUndecodable) else [str(getattr(e, "clip_id", ""))]
            bad = [s for s in samples if s.clip_id in set(ids)]
            if not bad:
                raise ExtractError(E_BRIDGE, f"{type(e).__name__} names clip(s) {ids[:5]} that are not in the batch {[s.clip_id for s in samples][:8]}", exit_code=5) from e
            for s in bad:
                self.on_missing(s, _reason(e))
            return self.encode([s for s in samples if all(s is not b for b in bad)])
        except Exception as e:  # noqa: BLE001 - a bridge exception aborts the shard
            raise ExtractError(
                E_BRIDGE,
                f"bridge {type(self.bridge).__name__}.encode raised {type(e).__name__}: {e} on rows "
                f"{[s.row for s in samples][:8]} — a bridge bug is not coverage\n{traceback.format_exc()}",
                exit_code=5,
            ) from e
        self.first_encode_done = True
        self.stats.n_encoded += len(res.samples)
        return res

    # -- the batch loop -------------------------------------------------------------
    def run(self, batches: Sequence[list[Any]], deadline: _Deadline | None = None) -> Iterator[_Out]:
        """Yield one ``_Out`` per batch in order while ``read_ahead`` further batches are read by
        ``yavih.sample.ReadAhead`` (and prepared, for a bridge with ``prepare``). The deadline is checked after every
        batch; the batch's payloads (and, with ``release_bytes``, its bytes) are dropped once the consumer is done."""
        queue = [list(b) for b in batches]
        if not queue:
            return
        if self.has_prepare:  # prepare threads are audited from their first call
            _install_audit_hook()
            _Audit.run_dir = os.path.abspath(os.fspath(self.run_dir)) if self.run_dir is not None else None
            _Audit.prepare_loopback = self.allow_loopback
        ra = _sample.ReadAhead(self.workers, depth=self.read_ahead)
        reads = ra.run(queue)
        if self.has_prepare:
            reads = self._prepare_ahead(reads)
        try:
            t0 = time.perf_counter()
            while True:
                try:
                    rb = next(reads)
                except StopIteration:
                    return
                except (DecoderMissing, ExtractError, NotApplicable):
                    raise
                except Exception as e:  # noqa: BLE001 - a reader bug is internal, not coverage
                    raise ExtractError("E_READ_INTERNAL", f"reading the stored bytes raised {_reason(e)}", exit_code=5) from e
                self.stats.read_s += rb.read_s
                for s, exc in rb.missing:
                    self.on_missing(s, _reason(exc))
                res = self.encode(rb.samples)
                now = time.perf_counter()
                self.stats.batch_times.append(now - t0)
                t0 = now
                yield res
                self._batch_done(rb)
                if deadline is not None:
                    deadline.check(self.stats.estimated_batch_s())
        finally:
            reads.close()
            ra.shutdown(wait=False)
            _Audit.prepare_loopback = False


def _batched(samples: Sequence[Any], batch_size: int) -> list[list[Any]]:
    return [list(samples[i : i + batch_size]) for i in range(0, len(samples), batch_size)]


def _media_block(sample: Any) -> dict[str, Any]:
    """``{stream: MediaInfo}`` copied from the RawStreams (the bridge cannot alter it)."""
    out: dict[str, Any] = {}
    for name, st in sample.streams.items():
        info = getattr(st, "info", None)
        out[str(name)] = json.loads(hashing.canon(dict(info))) if isinstance(info, Mapping) else None
    return out


def _row_meta(sample: Any, used: dict, media_keys: Sequence[str | None] | None) -> dict[str, Any]:
    """The row's ``.json`` provenance: ``media_key`` (from the manifest), ``media`` (RawStream
    facts), ``used`` (the bridge's report). ``canary_of`` is added by the writer for canary members."""
    row = _canary_of(sample)
    row = sample.row if row is None else row
    mk = media_keys[row] if media_keys is not None and 0 <= row < len(media_keys) else None
    return {"media_key": mk, "media": _media_block(sample), "used": used}


def _rel_l2(c: np.ndarray, o: np.ndarray) -> float:
    c64, o64 = c.astype(np.float64).ravel(), o.astype(np.float64).ravel()
    return float(np.linalg.norm(c64 - o64) / max(float(np.linalg.norm(o64)), 1e-12))


def _write_rows(writer: FeatureWriter, res: _Out, media_keys: Sequence[str | None] | None, *, canary: bool = False) -> None:
    """Row provenance + features into the shard; a serialization failure is a named internal error
    (``E_WRITE``, exit 5), not a bare TypeError out of the engine's error mapping."""
    rows = [int(_canary_of(s)) if canary else int(s.row) for s in res.samples]
    try:
        writer.write(rows, [s.clip_id for s in res.samples], res.feats, res.pos,
                     meta=[_row_meta(s, u, media_keys) for s, u in zip(res.samples, res.used, strict=True)], canary=canary)
    except (TypeError, ValueError, OSError) as e:
        raise ExtractError(E_WRITE, f"cannot write rows {rows[:5]} ({'canary' if canary else 'regular'}): {_reason(e)}", exit_code=5) from e


# ----------------------------------------------------------------------------- run_extract


def run_extract(
    *,
    bench: str | Any,
    bridge: str | type[Bridge],
    split: str,
    shard: tuple[int, int] | None = None,
    out: str | os.PathLike,
    args: dict | Sequence[str] | None = None,
    canaries: int = DEFAULT_CANARIES,
    canary_seed: int = 0,
    decode_workers: int | None = None,
    read_ahead: int = DEFAULT_READ_AHEAD,
    deadline: float | str | None = None,
    expect_profile: str | os.PathLike | None = None,
    data: dict | None = None,
    allow_loopback: bool = False,
    rows: tuple[int, int] | None = None,
) -> dict:
    """Extract one shard (``yavih extract``); returns the sidecar summary with a ``status`` key
    (``ok`` | ``skipped`` | ``not_applicable``). Idempotent: a matching sidecar → skipped; a
    mismatching one → moved to ``_stale/<utc>/`` and re-run. Raises ``ExtractError`` with the exit code
    (4 on deadline/SIGTERM with nothing partial left behind); ``NotApplicable`` → ``NOT_APPLICABLE.json``
    + ``status: not_applicable`` (exit 0).

    ``bench`` may be a name/path (``Benchmark.load``) or a loaded ``Benchmark``; ``bridge`` a file path,
    ``module:Class`` or a ``Bridge`` subclass. ``out`` is the run directory (shards land in
    ``<out>/features/<split>/``) or that split directory itself; writes under it are never side effects.

    ``expect_profile`` (always passed by ``plan``): the ``bridge check`` profile the unit was planned
    from. Its ``model``, ``reading`` and ``describe_sha256`` drive the skip check before the model is loaded, and a
    live ``describe()`` that differs from it is refused (``E_CONFIG_DRIFT``, exit 2).

    ``allow_loopback`` (``--allow-loopback``): ``encode`` may connect to the loopback interface (a local
    Triton / vLLM server); every other connect is still ``E_NETWORK``. The opt-in and the loopback addresses used are
    recorded in the sidecar's ``side_effects`` block.

    ``rows=(lo, hi)`` (``--rows lo:hi``, instead of ``shard``): a capped run over manifest rows ``[lo, hi)`` of the split,
    written as the one shard ``0/1`` of a layout frozen with ``sample: [lo, hi]`` (``yavih.io.layout``). The sample
    belongs to the layout's identity: a split directory holding a full layout, or a sample of another range, is
    refused with ``E_LAYOUT`` (exit 1), and so is a full-split unit in a sampled directory.
    """
    if (shard is None) == (rows is None):
        raise _ArgError("pass exactly one of --shard i/n and --rows lo:hi")
    sample: tuple[int, int] | None = None
    if rows is not None:
        sample = (int(rows[0]), int(rows[1]))
        if not (0 <= sample[0] < sample[1]):
            raise _ArgError(f"--rows lo:hi needs 0 <= lo < hi, got {sample[0]}:{sample[1]}")
        shard = (0, 1)
    assert shard is not None
    i, n = int(shard[0]), int(shard[1])
    if not (0 <= i < n):
        raise _ArgError(f"--shard i/n needs 0 <= i < n, got {i}/{n}")
    planned = _load_profile(expect_profile) if expect_profile else None
    workers = default_decode_workers() if decode_workers is None else max(1, int(decode_workers))
    if deadline is None or isinstance(deadline, str):
        # The CLI hands the raw --deadline text through; None honours $YAVIH_DEADLINE set by `yavih exec`.
        deadline = parse_deadline(deadline)
    split_dir = split_dir_for(out, split)
    started_wall = time.perf_counter()
    started = utc_now()

    # Fast path: with the planned model, reading and describe_sha256 in hand the DONE check needs no model load.
    # READ-ONLY (peek_shard): the live configuration is not known yet, so nothing may be moved on the profile's word —
    # a shard the live bridge wrote must survive an E_CONFIG_DRIFT refusal. The moving resume runs after _prepare.
    bench_obj = _load_bench(bench, data)
    if sample is not None and sample[1] > _n_rows(bench_obj, split):
        raise _ArgError(f"--rows {sample[0]}:{sample[1]}: split {split!r} of {bench_obj.name} has {_n_rows(bench_obj, split)} rows")
    if planned is not None:
        bridge_cls, _, _ = _resolve_bridge(bridge)
        # the split's frozen layout: a unit planned before another host froze it with another
        # n (or for another sample) is refused here, before the model load (E_LAYOUT, exit 1: re-run `plan build`,
        # which reads the layout)
        frozen = frozen_layout(split_dir, layout_identity(
            manifest_hash=bench_obj.manifest_hash, input_hash=bench_obj.input_hash, model=planned.model,
            reading=planned.reading, describe_sha256=planned.describe_sha256, sample=sample), n_rows=_n_rows(bench_obj, split))
        _check_layout(frozen, split_dir, i, n)
        if sample is not None and frozen is None:
            frozen = {"ranges": [list(sample)]}  # not frozen yet: the sample's one range is the shard's
        state = peek_shard(
            split_dir, i, manifest_hash=bench_obj.manifest_hash, input_hash=bench_obj.input_hash,
            model=planned.model, reading=planned.reading, describe_sha256=planned.describe_sha256, n=n,
            dtype=str(planned.dtype or bridge_cls.dtype), pos_mode=str(planned.pos_mode or bridge_cls.pos_mode),
            rows=frozen["ranges"][i] if frozen is not None and frozen.get("ranges") else None,
        )
        if state == "skip":
            return _skipped(split_dir, i)

    _check_header_backends(bench_obj)  # no census and no header backend → exit 3 now, not per row
    prep = _prepare(bench=bench_obj, bridge=bridge, args=args, planned=planned, data=data, decode_workers=workers)
    bench_obj = prep.bench
    bridge_cls = prep.bridge_cls

    if prep.bridge is None:  # NotApplicable from __init__
        # the live describe() is unknown; the planned configuration's shards are not "another configuration's"
        return _write_not_applicable(split_dir, prep, split, prep.not_applicable or "",
                                     keep_describe=planned.describe_sha256 if planned is not None else None)

    assert prep.describe is not None and prep.model is not None and prep.reading is not None and prep.describe_sha256 is not None
    d = prep.describe
    tol, _tol_reason = batch_tolerance(d)
    if d.get("weights_sha256") is None:
        _log(f"{W_WEIGHTS_UNFINGERPRINTED}: describe()['weights_sha256'] is null; the weights are recorded only by weights_ref "
             f"({d.get('weights_ref')!r})")
    plan_kw = dict(
        manifest_hash=bench_obj.manifest_hash, input_hash=bench_obj.input_hash,
        model=prep.model, reading=prep.reading, describe_sha256=prep.describe_sha256,
        dtype=str(bridge_cls.dtype), pos_mode=str(bridge_cls.pos_mode),
    )
    want = plan_values(**plan_kw)
    # The run directory does not name the configuration, so first move EVERY shard another
    # configuration left here, its _layout.json and a foreign _meta.json aside, in one step; then freeze (or read) this
    # configuration's layout and move its own shards of an earlier shard plan (a split without _layout.json); then the unit's own
    # resume (skip / run) under the shard's claim (one writer per shard).
    swept = retire_foreign(split_dir, want)
    if swept is not None:
        _log(f"extract {bench_obj.name} {split}: shards of another configuration moved to {swept}")
    ident = layout_identity(manifest_hash=bench_obj.manifest_hash, input_hash=bench_obj.input_hash, model=prep.model,
                            reading=prep.reading, describe_sha256=prep.describe_sha256, sample=sample)
    layout = freeze_layout(split_dir, ident, n=n, batch_size=int(bridge_cls.batch_size), n_rows=_n_rows(bench_obj, split))
    _check_layout(layout, split_dir, i, n)
    swept = retire_foreign(split_dir, want, layout=layout)
    if swept is not None:
        _log(f"extract {bench_obj.name} {split}: shards of an earlier shard plan moved to {swept} (the layout is frozen at n={n})")
    lo, hi = (int(x) for x in layout["ranges"][i])
    claim = acquire_claim(split_dir, i)  # ClaimHeld (exit 3) when a live writer has this shard
    if claim.took_over:
        _log(f"extract {bench_obj.name} {split} shard {i}: took over the stale claim of {claim.took_over}")
    writer: FeatureWriter | None = None
    dl = _Deadline(deadline)
    try:
        state = resume_shard(split_dir, i, **plan_kw, n=n, rows=(lo, hi))
        if state == "skip":
            return _skipped(split_dir, i)
        _retire_not_applicable(split_dir)

        dl.install()
        _reset_audit()
        run_dir = Path(out)
        dl.check(0.0)
        batch_size = int(bridge_cls.batch_size)
        rkw: dict[str, Any] = {}
        if int(layout.get("batch_size") or batch_size) != batch_size:
            rkw["align"] = int(layout["batch_size"])  # the frozen layout's batch alignment fixes the range
        if sample is not None:  # a capped run reads exactly the sampled range
            rkw = {"rows": range(lo, hi)}
        else:
            rkw["shard"] = (i, n)
        samples = list(bench_obj.reader(split, batch_size=batch_size, canaries=int(canaries), canary_seed=int(canary_seed), **rkw))
        regular = [s for s in samples if _canary_of(s) is None]
        canary_samples = [s for s in samples if _canary_of(s) is not None]
        rows = [s.row for s in regular]
        if rows != sorted(rows) or len(set(rows)) != len(rows):
            raise ExtractError("E_READER", "reader yielded rows out of ascending order or duplicated", exit_code=5)
        _check_rows(rows, lo, hi)
        media_keys = _media_keys(bench_obj, split)

        writer = FeatureWriter(out, split, shard=(i, n), rows=(lo, hi), bench=bench_obj, bridge=prep.bridge,
                               describe=d, source_sha256=prep.source_sha256, claim=claim,
                               batch_size=int(layout.get("batch_size") or batch_size))
        store_dtype = np.dtype(bridge_cls.dtype)
        canary_targets = {int(_canary_of(s)) for s in canary_samples}
        originals: dict[int, tuple[np.ndarray, np.ndarray | None, dict]] = {}

        def on_missing(s: Any, reason: str) -> None:
            if _canary_of(s) is not None:
                return  # a canary whose media vanished between the two reads: nothing to compare
            writer.record_missing(s.row, s.clip_id, reason)

        # release_bytes: a batch's stored bytes are dropped once it is written (and after prepare); the canaries are
        # Samples of their own (their own byte sources), read again by the canary pass
        engine = _Engine(prep.bridge, d, decode_workers=workers, read_ahead=read_ahead, torch=prep.torch, on_missing=on_missing,
                         kinds=prep.view.streams, run_dir=run_dir, allow_not_applicable=True, allow_loopback=allow_loopback,
                         release_bytes=True)
        canary_stats: dict[str, Any] = {"max_abs_diff": 0.0, "max_rel_l2": 0.0, "n": 0, "tolerance": tol, "mismatch": []}
        try:
            for res in engine.run(_batched(regular, batch_size), dl):
                if not res.samples:
                    continue
                _write_rows(writer, res, media_keys)
                for k, s in enumerate(res.samples):
                    if s.row in canary_targets:
                        originals[s.row] = (np.ascontiguousarray(res.feats[k], dtype=store_dtype), res.pos_of(k), res.used[k])
            live_canaries = [s for s in canary_samples if int(_canary_of(s)) in originals]
            for res in engine.run(_batched(live_canaries, batch_size), dl):
                if not res.samples:
                    continue
                _write_rows(writer, res, media_keys, canary=True)
                for k, s in enumerate(res.samples):
                    o_feat, o_pos, o_used = originals[int(_canary_of(s))]
                    c = np.ascontiguousarray(res.feats[k], dtype=store_dtype)
                    diff = float(np.abs(c.astype(np.float64) - o_feat.astype(np.float64)).max()) if c.size else 0.0
                    canary_stats["max_abs_diff"] = max(canary_stats["max_abs_diff"], diff)
                    canary_stats["max_rel_l2"] = max(canary_stats["max_rel_l2"], rel_l2_stored(c, o_feat))  # net of fp16 rounding
                    canary_stats["n"] += 1
                    p = res.pos_of(k)
                    if hashing.canon(res.used[k]) != hashing.canon(o_used) or (p is not None and o_pos is not None and not np.array_equal(p, o_pos)):
                        canary_stats["mismatch"].append(int(_canary_of(s)))
        except NotApplicable as na:
            writer.abort()
            writer = None
            return _write_not_applicable(split_dir, prep, split, na.reason)
        if not canary_stats["mismatch"]:
            canary_stats.pop("mismatch")

        # The precision the sidecar records must be the precision the features were computed under.
        if prep.torch is not None and prep.precision_live is not None:
            live_now = _precision_live(prep.torch)
            if live_now != prep.precision_live:
                raise ExtractError(
                    E_PRECISION_DRIFT,
                    f"torch precision changed during encode: enforced {prep.precision_live}, now {live_now}; the bridge must not "
                    "alter matmul precision / TF32 / deterministic algorithms after describe()",
                    exit_code=2,
                )

        wall = time.perf_counter() - started_wall
        n_written = len(writer.members)
        timing = {
            "started": started,
            "finished": utc_now(),
            "wall_s": round(wall, 3),
            "clips_per_s": round(n_written / wall, 4) if wall > 0 else None,
            "read_s": round(engine.stats.read_s, 3),
            "prepare_s": round(engine.stats.prepare_s, 3),
            "encode_s": round(engine.stats.encode_s, 3),
        }
        env = _env_block(prep.torch, workers, prep.precision_live, read_ahead=read_ahead)
        side_effects: dict[str, Any] = {"writes_outside_run_dir": sorted(engine.side_effects)}
        if allow_loopback:  # the --allow-loopback opt-in is provenance: who allowed which local connections
            side_effects.update(allow_loopback=True, loopback_connects=list(engine.loopback))
        if engine.side_effects:
            _log(f"{W_SIDE_EFFECT}: the bridge wrote outside the run directory during encode or prepare: {sorted(engine.side_effects)[:5]}")
        sidecar_path = writer.close(timing=timing, env=env, canary_stats=canary_stats, side_effects=side_effects)
        writer = None
        summary = read_sidecar(sidecar_path)
        summary["status"] = "ok"
        summary["sidecar_path"] = os.fspath(sidecar_path)
        _log(f"extract {bench_obj.name} {split} shard {i}/{n} ok rows={n_written}/{hi - lo} missing={len(summary.get('missing', []))} "
             f"canary_max_rel_l2={canary_stats['max_rel_l2']:.2e} wall_s={wall:.1f}")
        return summary
    finally:
        if writer is not None:
            writer.abort()  # exit 4 / any failure: only the .tmp existed and it is gone
        dl.restore()
        claim.release()


def _n_rows(bench_obj: Any, split: str) -> int:
    return int(bench_obj.n_rows(split))


def _check_layout(layout: dict | None, split_dir: Path, i: int, n: int) -> None:
    """``E_LAYOUT`` (exit 1) when the split's frozen layout has another shard count than ``--shard i/n``."""
    if layout is None or int(layout["n"]) == n:
        return
    from yavih.io.layout import LAYOUT_FILE

    where = (f"{split_dir / LAYOUT_FILE} (frozen {layout.get('created')} on {layout.get('host')})" if not layout.get("inferred_from_shards")
             else f"the finished shards in {split_dir}")
    raise LayoutError(f"--shard {i}/{n}: this split's shard layout is frozen at n={layout['n']} by {where}; the unit comes from a "
                      "plan built before that: re-run `yavih plan build` (it plans the frozen layout), or move the split "
                      "directory aside to re-shard it")


def _check_rows(rows: list[int], lo: int, hi: int) -> None:
    """The reader's rows are exactly the frozen layout's range ``[lo, hi)`` of this shard."""
    if rows and (rows[0] < lo or rows[-1] >= hi or len(rows) != hi - lo):
        raise ExtractError("E_READER", f"reader rows [{rows[0]}, {rows[-1]}] ({len(rows)}) disagree with the shard range [{lo}, {hi})", exit_code=5)
    if not rows and hi != lo:
        raise ExtractError("E_READER", f"reader yielded nothing for a non-empty range [{lo}, {hi})", exit_code=5)


def _skipped(split_dir: Path, i: int) -> dict:
    from yavih.io.format import shard_sidecar_name

    path = split_dir / shard_sidecar_name(i)
    summary = read_sidecar(path)
    summary["status"] = "skipped"
    summary["sidecar_path"] = os.fspath(path)
    _log(f"extract shard {i} skipped: {path} matches the plan")
    return summary


def _retire_not_applicable(split_dir: Path) -> None:
    """A stale ``NOT_APPLICABLE.json`` must not sit next to shards (``E_NA_AND_SHARDS``): move it aside."""
    from yavih.io.format import NOT_APPLICABLE_FILE, STALE_DIR

    p = split_dir / NOT_APPLICABLE_FILE
    if p.exists():
        stale = split_dir / STALE_DIR / utc_now().replace(":", "")
        stale.mkdir(parents=True, exist_ok=True)
        os.replace(p, stale / NOT_APPLICABLE_FILE)


def _write_not_applicable(split_dir: Path, prep: _Prepared, split: str, reason: str, *, keep_describe: str | None = None) -> dict:
    # model / reading: from describe(); when __init__ raised before the bridge could describe itself, the plan's profile
    # supplies them (or they are null)
    # An earlier configuration's shards (another describe_sha256, model or reading) and ANY
    # _meta.json go to _stale/ first — validate writes _meta.json only next to shards, so one beside
    # NOT_APPLICABLE.json describes an earlier configuration and the probe must never read it as this one's.
    # ``keep_describe``: when __init__ raised (no live describe), the planned profile's describe_sha256 — that
    # configuration's shards stay (validate names the conflict, E_NA_AND_SHARDS) instead of being moved on a guess.
    swept = retire_foreign(split_dir, {
        "benchmark.manifest_hash": prep.bench.manifest_hash, "benchmark.input_hash": prep.bench.input_hash,
        "model": prep.model, "reading": prep.reading,
        "bridge.describe_sha256": prep.describe_sha256 if prep.describe_sha256 is not None else keep_describe,
    }, meta="always")
    if swept is not None:
        _log(f"extract {prep.bench.name} {split}: an earlier configuration's shards / _meta.json moved to {swept}")
    from yavih.io.layout import LAYOUT_FILE

    try:  # the verdict replaces this configuration's shard plan; a later extract re-freezes (or re-infers) it
        (split_dir / LAYOUT_FILE).unlink()
    except FileNotFoundError:
        pass
    path = write_not_applicable(split_dir, reason=reason, benchmark=prep.bench.name, split=split, model=prep.model, reading=prep.reading,
                                describe=prep.describe or {}, describe_sha256=prep.describe_sha256)
    rec = read_not_applicable(split_dir) or {}
    rec["path"] = os.fspath(path)
    _log(f"extract {prep.bench.name} {split}: not_applicable ({reason})")
    return rec


# ----------------------------------------------------------------------------- bridge check


def _pick_split(bench_obj: Any, split: str | None) -> str:
    if split:
        return split
    # ``Benchmark.roles()`` is the contract; a spec object/dict ``splits`` block is the fallback.
    roles_fn = getattr(bench_obj, "roles", None)
    roles: Mapping[str, Any] = roles_fn() if callable(roles_fn) else {}
    spec = getattr(bench_obj, "spec", None)
    splits = spec.get("splits") if isinstance(spec, Mapping) else getattr(spec, "splits", None)
    for role in ("eval", "train", "folds", "floor"):
        vals = roles.get(role) if roles else None
        if not vals and splits is not None:
            vals = splits.get(role) if isinstance(splits, Mapping) else getattr(splits, role, None)
        if vals:
            return str(list(vals)[0])
    raise _ArgError("bridge check: cannot pick a split (benchmark declares none); pass split=")


def scramble(sample: Any, k: int) -> Any:
    """The Sample with ``row``, ``clip_id`` and ``split`` replaced by synthetic values (the alone pass):
    ``10**9 + k``, ``"bridge-check-<k>"``, ``"bridge-check"``, and its streams handed over under fresh opaque
    paths (new stream objects over the same stored bytes, which stay cached), so a cache keyed on the identity,
    the path or the stream object misses."""
    new = {"row": SCRAMBLE_ROW_BASE + int(k), "clip_id": f"bridge-check-{int(k)}", "split": "bridge-check"}
    return _sample.rehandle(sample, f"bridge-check-alone-{int(k)}", **new)


@dataclass
class _PassResult:
    """Per input index: ``(feat float32 [N, D], pos [N, 4] | None, used dict)``; missing indices are absent."""

    by_index: dict[int, tuple[np.ndarray, np.ndarray | None, dict]] = field(default_factory=dict)
    wall_s: float = 0.0


def _run_pass(engine: _Engine, samples: list[Any], order: Sequence[int], batch_size: int) -> _PassResult:
    """Encode ``samples`` in the given index ``order`` and batch size; results keyed by original index."""
    seq = [samples[j] for j in order]
    index_of = {id(s): j for s, j in zip(seq, order, strict=True)}
    out = _PassResult()
    t0 = time.perf_counter()
    for res in engine.run(_batched(seq, batch_size)):
        for k, s in enumerate(res.samples):
            out.by_index[index_of[id(s)]] = (np.ascontiguousarray(res.feats[k], dtype=np.float32), res.pos_of(k), res.used[k])
    out.wall_s = time.perf_counter() - t0
    return out


def _compare(name: str, a: _PassResult, b: _PassResult, samples: list[Any], tol: float, *, exact: bool) -> tuple[float, list[str]]:
    """``(max rel_l2, problems)`` of pass ``b`` against pass ``a``."""
    worst = 0.0
    problems: list[str] = []
    for j, (fa, pa, ua) in sorted(a.by_index.items()):
        if j not in b.by_index:
            problems.append(f"{samples[j].clip_id}: encoded in pass A but missing in pass {name}")
            continue
        fb, pb, ub = b.by_index[j]
        if fa.shape != fb.shape:
            problems.append(f"{samples[j].clip_id}: feature shape {fb.shape} ≠ {fa.shape}")
            continue
        r = 0.0 if fa.tobytes() == fb.tobytes() else _rel_l2(fb, fa)
        worst = max(worst, r)
        if (exact and r != 0.0) or (not exact and r > tol):
            problems.append(f"{samples[j].clip_id}: rel_l2 {r:.3e}" + ("" if exact else f" > {tol:.1e}"))
        if (pa is None) != (pb is None) or (pa is not None and not np.array_equal(pa, pb)):
            problems.append(f"{samples[j].clip_id}: pos differs")
        if hashing.canon(ua) != hashing.canon(ub):
            problems.append(f"{samples[j].clip_id}: used differs ({ua} vs {ub})")
    return worst, problems


def _payload_diff(a: Any, b: Any, where: str = "prepared") -> str | None:
    """Where two ``prepare`` payloads differ (``None`` when equal): mappings by key, sequences by position, arrays and
    tensors by shape, dtype and bytes (``_to_numpy``), anything else by ``==``."""
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not (isinstance(a, Mapping) and isinstance(b, Mapping)) or set(a) != set(b):
            return f"{where}: keys {sorted(map(str, a)) if isinstance(a, Mapping) else type(a).__name__} vs " \
                   f"{sorted(map(str, b)) if isinstance(b, Mapping) else type(b).__name__}"
        for k in a:
            d = _payload_diff(a[k], b[k], f"{where}[{k!r}]")
            if d:
                return d
        return None
    arrayish = (isinstance(a, np.ndarray | np.generic) or hasattr(a, "detach")) and (isinstance(b, np.ndarray | np.generic) or hasattr(b, "detach"))
    if arrayish:
        x, y = np.asarray(_to_numpy(a)), np.asarray(_to_numpy(b))
        if x.shape != y.shape or x.dtype != y.dtype:
            return f"{where}: {x.dtype}{list(x.shape)} vs {y.dtype}{list(y.shape)}"
        return None if x.tobytes() == y.tobytes() else f"{where}: values differ"
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            return f"{where}: length {len(a)} vs {len(b)}"
        for k, (x, y) in enumerate(zip(a, b, strict=True)):
            d = _payload_diff(x, y, f"{where}[{k}]")
            if d:
                return d
        return None
    try:
        same = bool(a == b)
    except Exception:  # noqa: BLE001 - an object without a usable ==: compare its repr
        same = repr(a) == repr(b)
    return None if same else f"{where}: {a!r} vs {b!r}"[:200]


def _prepare_twice(bridge: Bridge, samples: Sequence[Any], torch: Any = None) -> list[str]:
    """``bridge check``'s ``prepare`` determinism check: ``prepare`` on two fresh copies of each sample (new opaque
    paths over the same cached bytes, in a ``yavih-prepare`` thread under ``torch.inference_mode()``); returns the
    differences (``E_PREPARE_NONDETERMINISTIC``). A call that raises on a sample pass A prepared is a difference too."""
    problems: list[str] = []

    def twice(k: int, s: Any) -> str | None:
        out = []
        for tag in ("1", "2"):
            copy = _sample.rehandle(s, f"bridge-check-prepare-{tag}")
            try:
                with _inference_ctx(torch):
                    out.append(bridge.prepare(copy))
            except Exception as e:  # noqa: BLE001 - reported as nondeterminism of this sample
                return f"{s.clip_id}: call {tag} raised {_reason(e)}"
        d = _payload_diff(out[0], out[1])
        return f"{s.clip_id}: {d}" if d else None

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix=_PREPARE_THREADS) as ex:
        for res in ex.map(lambda ks: twice(*ks), enumerate(samples)):
            if res:
                problems.append(res)
    return problems


def unfilled_describe(d: Mapping[str, Any]) -> list[str]:
    """What ``bridge check`` refuses as an unfilled template (``E_TEMPLATE_UNFILLED``): ``TODO`` left in ``model``,
    ``weights_ref`` or a stream's ``norm.source``, and a ``norm`` constant left null. A model that normalizes nothing
    declares ``{"source": "none: <why>"}`` without constants."""
    problems: list[str] = []
    for key in ("model", "weights_ref"):
        if "TODO" in str(d.get(key) or ""):
            problems.append(f"{key} = {d.get(key)!r}")
    inp = d.get("input")
    for sname, entry in (inp.items() if isinstance(inp, Mapping) else ()):
        norm = entry.get("norm") if isinstance(entry, Mapping) else None
        if not isinstance(norm, Mapping):
            problems.append(f"input[{sname!r}]['norm'] is null")
            continue
        if "TODO" in str(norm.get("source") or ""):
            problems.append(f"input[{sname!r}]['norm']['source'] = {norm.get('source')!r}")
        nulls = sorted(k for k, v in norm.items() if k != "source" and v is None)
        if nulls:
            problems.append(f"input[{sname!r}]['norm'] constant(s) {nulls} null")
    return problems


def constant_features(res: _PassResult) -> bool:
    """True when at least two clips were encoded and every clip's features equal the first's (``E_CONSTANT_FEATURES``)."""
    feats = [f for _, (f, _, _) in sorted(res.by_index.items())]
    if len(feats) < 2:
        return False
    first = feats[0]
    return all(f.shape == first.shape and (f.tobytes() == first.tobytes() or _rel_l2(f, first) <= 1e-7) for f in feats[1:])


def _fresh_bridge(prep: _Prepared, workers: int) -> Bridge:
    """Replace ``prep.bridge`` by a second instance built exactly as ``_prepare`` built the first (the alone
    pass): the first is released first (GPU memory), the new one must describe itself identically and be in eval
    mode."""
    import gc

    prep.bridge = None
    gc.collect()
    torch = prep.torch
    if torch is not None:
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    inst = prep.bridge_cls(prep.view, dict(prep.args))
    try:
        d2 = json.loads(hashing.canon(inst.describe()))
    except NotApplicable:
        raise
    except Exception as e:
        raise DescribeError(f"describe() of a second instance raised {type(e).__name__}: {e}") from e
    if prep.describe is not None and hashing.canon(d2) != hashing.canon(prep.describe):
        drift = sorted(k for k in set(d2) | set(prep.describe) if hashing.canon(d2.get(k)) != hashing.canon(prep.describe.get(k)))
        raise ExtractError(E_RERUN_DRIFT, f"a second instance of {prep.bridge_cls.__name__} describes itself differently (keys {drift}): describe() "
                           "must be a pure function of the bridge's arguments and weights", exit_code=2)
    if torch is not None:
        training = _find_training_modules(inst, torch)
        if training:
            raise ExtractError(E_TRAIN_MODE, f"torch module(s) of a second {prep.bridge_cls.__name__} instance in training mode: {training[:8]}",
                               exit_code=2)
    inst.decode_workers = max(1, int(workers))
    prep.bridge = inst
    return inst


def bridge_check(
    *,
    bridge: str | type[Bridge],
    bench: str | Any,
    n: int = 32,
    args: dict | Sequence[str] | None = None,
    decode_workers: int | None = None,
    out: str | os.PathLike | None = None,
    split: str | None = None,
    read_ahead: int = DEFAULT_READ_AHEAD,
    data: dict | None = None,
    synthetic: bool = False,
    allow_loopback: bool = False,
) -> dict:
    """``yavih bridge check`` backend; see ``_bridge_check`` for the checks.

    ``synthetic=True`` runs every check on a tiny stand-in of ``bench`` (``yavih.testing.make_synthetic_like``: the
    benchmark's name, readout, stream names and kinds, guaranteed media facts and label facts, over tiny generated
    media in a temporary directory, removed afterwards), so a bridge can be smoke-tested before any data is staged.
    It writes no profile (``out`` is refused): ``plan`` needs one measured on the real data. The result carries
    ``synthetic: true``."""
    if not synthetic:
        return _bridge_check(bridge=bridge, bench=bench, n=n, args=args, decode_workers=decode_workers, out=out, split=split,
                             read_ahead=read_ahead, data=data, allow_loopback=allow_loopback)
    import shutil
    import tempfile

    from yavih import testing

    if out:
        raise ExtractError(E_SYNTHETIC, "bridge check --synthetic writes no profile (speed and media are not the benchmark's); drop --out, "
                           "and run `bridge check --out …` on the staged data for the profile `plan` needs", exit_code=1)
    real = _load_bench(bench, data)
    tmp = tempfile.mkdtemp(prefix="yavih-synthetic-")
    try:
        try:
            folder = testing.make_synthetic_like(tmp, real, n_eval=max(2, int(n)))
        except ImportError as e:
            raise ExtractError(E_SYNTHETIC, f"bridge check --synthetic writes tiny media with PyAV, soundfile and pillow: {e}; "
                               "pip install av soundfile pillow (or yavih[media])", exit_code=3) from e
        except ValueError as e:
            raise ExtractError(E_SYNTHETIC, f"bridge check --synthetic: {e}", exit_code=1) from e
        profile = _bridge_check(bridge=bridge, bench=os.fspath(folder), n=n, args=args, decode_workers=decode_workers, out=None,
                                split=None, read_ahead=read_ahead, data=None, allow_loopback=allow_loopback)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    profile["synthetic"] = True
    return profile


def _bridge_check(
    *,
    bridge: str | type[Bridge],
    bench: str | Any,
    n: int = 32,
    args: dict | Sequence[str] | None = None,
    decode_workers: int | None = None,
    out: str | os.PathLike | None = None,
    split: str | None = None,
    read_ahead: int = DEFAULT_READ_AHEAD,
    data: dict | None = None,
    allow_loopback: bool = False,
) -> dict:
    """``yavih bridge check`` backend (writes ``profile.json``).

    The bridge preflight, then on the smoke set (``n`` present rows, eval split first) at the declared
    ``batch_size``: pass A (row order), **rerun** (A again: feats, pos and ``used`` bit-identical, else
    ``E_RERUN_DRIFT``), **shuffled** (permuted with ``default_rng(0)`` and re-batched: per clip ``rel_l2 ≤ tol``,
    pos and ``used`` equal, else ``E_BATCH_INVARIANCE``), **alone** (on a FRESH bridge instance, each Sample in its
    own batch with ``row``/``clip_id``/``split`` scrambled, same test). Every pass after A hands over new Samples
    whose streams carry fresh opaque paths over the same cached bytes, so a cache keyed on the path, the stream
    object or the clip id misses. Also the per-member validation checks (``E_TIMES_OUT_OF_MEDIA``,
    ``E_RULE_MISMATCH``, ``E_POS_SHAPE``/``E_POS_RANGE``, fixed ``N``/``D``, fp16 range), ``E_NETWORK``,
    ``W_SIDE_EFFECT`` and ``W_NO_PREPROCESSING_SOURCE``. An unfilled template is refused before any row is read
    (``E_TEMPLATE_UNFILLED``: ``unfilled_describe``), and features equal on every smoke clip after the passes
    (``E_CONSTANT_FEATURES``: ``constant_features``). A failed hard check raises ``ExtractError`` (exit 2)
    and writes no profile; otherwise the profile is written atomically to ``out`` when given. A check the bridge
    waives in ``describe()["waive"]`` (``E_RERUN_DRIFT``, ``E_BATCH_INVARIANCE``, ``E_CONSTANT_FEATURES``, the waivable
    per-member checks, ``E_ARGS_UNCONSUMED``; ``yavih.waivers``) is recorded as ``W_WAIVED`` instead and listed in the
    profile's ``waived``.

    The profile records the declared ``model``, ``reading`` and ``describe_sha256`` (what ``plan`` keys the run
    directory on and ``run_extract --expect-profile`` checks for drift) and a ``notes`` list: a ``custom`` rule is the
    note ``N_RULE_CUSTOM`` (an opt-out is legitimate, never a warning). A null ``weights_sha256`` is the warning
    ``W_WEIGHTS_UNFINGERPRINTED``; nothing is gated by it.
    """
    from yavih.io.validate import (
        CODES,
        _check_pos,
        check_member,
        custom_rule_message,
        custom_rules,
        rules_checked,
    )

    workers = default_decode_workers() if decode_workers is None else max(1, int(decode_workers))
    warnings: list[dict[str, Any]] = []
    bridge_cls0, _, _ = _resolve_bridge(bridge)
    if _is_oracle(bridge_cls0) and os.environ.get("YAVIH_TESTING") != "1":
        raise ExtractError(E_ORACLE, f"{bridge_cls0.__name__} is yavih.testing's oracle bridge (features looked up by clip_id, breaking the encode guardrails "
                           "on purpose); bridge check refuses it unless YAVIH_TESTING=1", exit_code=2)
    bench = _load_bench(bench, data)
    _check_header_backends(bench)  # before the model load: without facts no smoke row can be read
    prep = _prepare(bench=bench, bridge=bridge, args=args, planned=None, data=data, decode_workers=workers)
    bench_obj = prep.bench
    notes: list[dict[str, Any]] = []
    base: dict[str, Any] = {
        "bench": bench_obj.name,
        "bridge_name": bridge_label(prep.bridge_cls),  # a recorded label, never a key
        "args": prep.args,
        "source_sha256": prep.source_sha256,
        "source_path": prep.source_path,
        "yavih_version": __version__,
        "created": utc_now(),
    }
    if prep.bridge is None:
        base.update({"status": "not_applicable", "reason": prep.not_applicable, "describe": None, "describe_sha256": None, "model": None,
                     "reading": None, "warnings": warnings, "notes": notes})
        if out:
            write_json_atomic(out, base)
        return base
    assert prep.describe is not None and prep.model is not None and prep.reading is not None
    d = prep.describe
    unfilled = unfilled_describe(d)
    if unfilled:
        raise ExtractError(E_TEMPLATE_UNFILLED, f"describe() is still a template: {unfilled}; fill the bridge from the model's own code "
                           "(model name, weights_ref, the pretraining normalization constants and their source)", exit_code=2)
    ident = {"model": prep.model, "reading": prep.reading, "describe_sha256": prep.describe_sha256}
    # The bridge's describe()["waive"] (validated by validate_describe): a waived hard check is recorded as
    # W_WAIVED in the profile's warnings and `waived`, never silently passed
    waivers = waivers_for(d)

    def waive_or_raise(code: str, message: str, where: str) -> None:
        if waivers.covers(code):
            waivers.record(code)
            warnings.append({"code": W_WAIVED, "message": waivers.message(code, message), "where": where})
            return
        raise ExtractError(code, message, exit_code=2)

    if waivers.covers("E_ARGS_UNCONSUMED"):
        for problem in args_unconsumed(d, prep.bridge):
            waive_or_raise("E_ARGS_UNCONSUMED", problem, "describe")
    tol, tol_reason = batch_tolerance(d)
    if d.get("weights_sha256") is None:
        warnings.append({"code": W_WEIGHTS_UNFINGERPRINTED, "message": "describe()['weights_sha256'] is null: the weights are recorded only by "
                         f"weights_ref ({d.get('weights_ref')!r}); nothing is marked, forked or greyed", "where": "describe"})
    for sname, which in custom_rules(d).items():
        notes.append({"code": N_RULE_CUSTOM, "message": custom_rule_message(sname, which), "where": sname})
    pre_text, pre_why = preprocessing_source_check(prep.bridge_cls)
    if pre_text is None:
        # the explanation names every docstring searched and the layout that passes
        warnings.append({"code": W_NO_PREPROCESSING_SOURCE, "message": "the bridge's docstring has no filled 'Preprocessing source:' slot: paste the model's own "
                         f"preprocessing code there and match describe()['input'] to it line by line; {pre_why}",
                         "where": prep.bridge_cls.__name__})
    if d.get("batch_tolerance") is not None:
        warnings.append({"code": "W_LOOSE_TOLERANCE", "message": f"declared batch_tolerance.rel_l2 = {tol:g}: {tol_reason}", "where": "describe"})

    use_split = _pick_split(bench_obj, split)
    n_rows = int(bench_obj.n_rows(use_split))
    want = max(1, int(n))
    batch_size = int(prep.bridge_cls.batch_size)
    kinds = dict(prep.view.streams)
    missing: list[dict[str, Any]] = []

    # ---- the smoke set: the first `n` PRESENT rows of the split (bytes read once, then cached by the RawStreams)
    smoke: list[Any] = []
    for s in bench_obj.reader(use_split, rows=range(0, n_rows), batch_size=batch_size):
        if _canary_of(s) is not None:
            continue
        try:
            _read_sample(s)
        except (MediaMissing, OSError) as e:
            missing.append({"row": s.row, "clip_id": s.clip_id, "reason": _reason(e)})
            continue
        smoke.append(s)
        if len(smoke) >= want:
            break
    if not smoke:
        roots_fn = getattr(bench_obj, "resolved_roots", None)
        try:
            roots = dict(roots_fn()) if callable(roots_fn) else None
        except Exception as e:  # noqa: BLE001 - the message names what it can
            roots = f"unresolvable ({type(e).__name__}: {e})"
        first = f"; first: {missing[0]['clip_id']}: {missing[0]['reason']}" if missing else ""
        raise ExtractError(E_SMOKE, f"bridge check: no present row among the {len(missing)} read rows of split {use_split!r} (data not ready; "
                           f"data roots searched: {roots}{first}); stage the data (`yavih doctor --bench {bench_obj.name}`), or smoke-test the "
                           "bridge without it: `yavih bridge check … --synthetic`", exit_code=3)

    def on_missing(s: Any, reason: str) -> None:
        missing.append({"row": s.row, "clip_id": s.clip_id, "reason": reason})

    _reset_audit()

    def engine() -> _Engine:
        return _Engine(prep.bridge, d, decode_workers=workers, read_ahead=read_ahead, torch=prep.torch, on_missing=on_missing,
                       kinds=kinds, run_dir=None, allow_not_applicable=True, allow_loopback=allow_loopback)

    status, reason = "ok", None
    idx = list(range(len(smoke)))
    try:
        pass_a = _run_pass(engine(), smoke, idx, batch_size)
        if has_prepare(prep.bridge):
            # prepare must be a pure function of the sample: twice on fresh copies of every row pass A prepared
            n_net = len(_Audit.network)
            prepared = [smoke[j] for j in sorted(pass_a.by_index)]
            problems = _prepare_twice(prep.bridge, prepared, prep.torch)
            if _Audit.network[n_net:]:
                raise ExtractError(E_NETWORK, f"the bridge opened network connection(s) during prepare: {_Audit.network[n_net:][:3]}", exit_code=2)
            if problems:
                raise ExtractError(E_PREPARE_NONDETERMINISTIC, f"prepare() on the same sample twice differs: {problems[:4]}; prepare must be a "
                                   "pure, deterministic function of the one sample (no random crop or jitter, no state kept across calls)",
                                   exit_code=2)
        # every pass after A gets NEW Samples over the same cached bytes, under fresh opaque paths, so a
        # cache keyed on the path, the stream object or the clip id cannot replay pass A's features
        rerun = _run_pass(engine(), [_sample.rehandle(s, "bridge-check-rerun") for s in smoke], idx, batch_size)
        perm = [int(j) for j in np.random.default_rng(0).permutation(len(smoke))]
        shuffled = _run_pass(engine(), [_sample.rehandle(s, "bridge-check-order") for s in smoke], perm, batch_size)
        # alone: a FRESH bridge instance (built exactly as the first), so even an instance-level content cache is cold
        _fresh_bridge(prep, workers)
        alone_samples = [scramble(s, k) for k, s in enumerate(smoke)]
        alone = _run_pass(engine(), alone_samples, idx, 1)
    except NotApplicable as na:
        status, reason = "not_applicable", na.reason
        profile = {**base, "status": status, "reason": reason, "split": use_split, "describe": d, **ident, "warnings": warnings, "notes": notes}
        if out:
            write_json_atomic(out, profile)
            profile["path"] = os.fspath(out)
        return profile

    # ---- hard checks
    # Bit identity is demanded only when the bridge declares precision.deterministic; a bridge that
    # declares non-deterministic kernels (deterministic: false) is held to its declared tolerance instead.
    deterministic = bool((d.get("precision") or {}).get("deterministic", True))
    rerun_max, rerun_problems = _compare("rerun", pass_a, rerun, smoke, tol, exact=deterministic)
    if rerun_problems:
        mode = "not bit-identical (precision.deterministic is true)" if deterministic else f"above the declared tolerance {tol:.1e} (precision.deterministic is false)"
        waive_or_raise(E_RERUN_DRIFT, f"encoding the same batches twice is {mode} (max rel_l2 {rerun_max:.3e}): {rerun_problems[:4]}; "
                       "check dropout / train mode, random crops or jitter, non-deterministic kernels", "rerun")
    elif rerun_max > 0.0:
        warnings.append({"code": W_RERUN_NONDETERMINISTIC, "message": f"rerun is not bit-identical (max rel_l2 {rerun_max:.3e} <= tolerance {tol:.1e}); "
                         "accepted because describe()['precision']['deterministic'] is false", "where": "encode"})
    alone_max, alone_problems = _compare("alone", pass_a, alone, smoke, tol, exact=False)
    if alone_problems:
        waive_or_raise(E_BATCH_INVARIANCE, f"pass alone (batch of 1, row/clip_id/split scrambled): {alone_problems[:4]}; the features depend on the "
                       "batch or on the sample's identity (batch statistics, padding to the batch maximum, caches keyed on clip id)", "alone")
    order_max, order_problems = _compare("order", pass_a, shuffled, smoke, tol, exact=False)
    if order_problems:
        waive_or_raise(E_BATCH_INVARIANCE, f"pass order (shuffled, re-batched): {order_problems[:4]}; the features depend on batch composition or order "
                       "(cross-sample state, online adaptation)", "order")
    if constant_features(pass_a):
        waive_or_raise(E_CONSTANT_FEATURES, f"every one of the {len(pass_a.by_index)} smoke clips encodes to the same features: the bridge "
                       "ignores its input (an unfilled template's zeros, a constant fallback, a model that never ran); validate would "
                       "refuse the extracted set (E_DEAD_ROWS)", "encode")

    # ---- per-member validation checks on pass A
    class _Collect:
        def __init__(self) -> None:
            self.items: list[dict[str, Any]] = []

        def add(self, code: str, message: str, where: str | None = None) -> None:
            self.items.append({"code": code, "message": message, "where": where})

    fx = _Collect()
    n_streams = len(d.get("streams") or [])
    unchecked: dict[str, int] = {}
    shapes = {fa.shape for fa, _, _ in pass_a.by_index.values()}
    if len(shapes) > 1:
        fx.add("E_MEMBERS", f"N/D differ across clips: {sorted(shapes)} (fixed N and D for every sample)", "encode")
    fp16_max = 0.0
    for j, (fa, pa, ua) in sorted(pass_a.by_index.items()):
        s = smoke[j]
        where = f"{s.clip_id} (row {s.row})"
        if not bool(np.isfinite(fa).all()):
            fx.add("E_FINITE", f"{int((~np.isfinite(fa)).sum())} non-finite feature values", where)
        if d["dtype"] == "float16" and fa.size:
            fp16_max = max(fp16_max, float(np.abs(fa[np.isfinite(fa)]).max()) if np.isfinite(fa).any() else 0.0)
        if pa is not None:
            _check_pos(pa, int(fa.shape[0]), n_streams, where, fx)  # type: ignore[arg-type]
        check_member({"media": _media_block(s), "used": ua}, pa, describe=d, kinds=kinds, where=where, fx=fx, unchecked=unchecked)
    if fp16_max > FP16_MAX:
        fx.add("E_FEATURES", f"float16 max|x| = {fp16_max:.3g} > {FP16_MAX:g}: the writer would refuse it; declare dtype float32", "encode")
    elif fp16_max > 1e3:
        fx.add("W_FP16_RANGE", f"float16 max|x| = {fp16_max:.3g} > 1e3; consider float32", "encode")
    for f in fx.items:  # a waived per-member error (E_TIMES_OUT_OF_MEDIA, E_RULE_MISMATCH, E_POS_RANGE) → W_WAIVED
        if CODES.get(f["code"], "error") == "error" and waivers.covers(f["code"]):
            waivers.record(f["code"])
            f.update(code=W_WAIVED, message=waivers.message(f["code"], f["message"]))
    errors = [f for f in fx.items if CODES.get(f["code"], "error") == "error"]
    warnings += [f for f in fx.items if CODES.get(f["code"], "error") != "error"]
    if errors:
        raise ExtractError(errors[0]["code"], f"{len(errors)} output-contract violation(s) on the smoke set: "
                           + "; ".join(f"{e['code']} {e['where']}: {e['message']}" for e in errors[:4]), exit_code=2)
    rc = rules_checked(d, kinds)
    for sname, k_rows in sorted(unchecked.items()):
        if sname in rc:
            rc[sname]["temporal_unchecked_rows"] = int(k_rows)
        warnings.append({"code": "W_RULE_UNCHECKED", "message": f"{sname}: the declared {(rc.get(sname) or {}).get('temporal')!r} rule could not be "
                         f"evaluated on {k_rows} smoke row(s) (a media fact it needs is null there)", "where": sname})
    if _Audit.writes:
        warnings.append({"code": W_SIDE_EFFECT, "message": f"writes outside the run directory during encode: {sorted(_Audit.writes)[:5]}", "where": "encode"})
    if missing:
        warnings.append({"code": "W_MISSING_ROWS", "message": f"{len(missing)} sampled row(s) had missing or undecodable media", "where": use_split})

    first_j = min(pass_a.by_index) if pass_a.by_index else None
    N = D = None
    used0: dict[str, Any] = {}
    media0: dict[str, Any] = {}
    if first_j is not None:
        f0, _, u0 = pass_a.by_index[first_j]
        N, D = int(f0.shape[0]), int(f0.shape[1])
        used0 = dict(u0)
        media0 = _media_block(smoke[first_j])
    clips_per_s = round(len(pass_a.by_index) / pass_a.wall_s, 4) if pass_a.wall_s > 0 and pass_a.by_index else None
    profile = {
        **base,
        "status": status,
        "reason": reason,
        "split": use_split,
        "n": len(smoke),
        "describe": d,
        **ident,
        "used": used0,
        "media": media0,
        "N": N,
        "D": D,
        "pos_mode": d["pos_mode"],
        "streams": list(d["streams"]),
        "batch_size": batch_size,
        "dtype": d["dtype"],
        "invariance": {
            "rerun_bit_identical": rerun_max == 0.0,
            "rerun_rel_l2": rerun_max,
            "batch_rel_l2": alone_max,
            "order_rel_l2": order_max,
            "tolerance": tol,
            "tolerance_reason": tol_reason,
        },
        "rules_checked": rc,
        "clips_per_s": clips_per_s,
        "decode_workers": workers,
        "peak_gpu_gb": _peak_gpu_gb(prep.torch, prep.bridge),
        "env": _env_block(prep.torch, workers, prep.precision_live, read_ahead=read_ahead),
        "missing": missing,
        "warnings": warnings,
        "notes": notes,
        # The waivers the bridge declared, with how many smoke-set findings each waived (W_WAIVED in warnings)
        "waived": waivers.to_json(),
    }
    if allow_loopback:  # the --allow-loopback opt-in the profile was measured under (extract must be given it too)
        profile["allow_loopback"] = True
        profile["loopback_connects"] = sorted(set(_Audit.loopback))
    if out:
        write_json_atomic(out, profile)
        profile["path"] = os.fspath(out)
    return profile


# ----------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m yavih.extract", description="Extract one feature shard (`yavih extract`).")
    p.add_argument("--bench", required=True, help="benchmark name, alias or folder")
    p.add_argument("--bridge", required=True, metavar="P", help="bridge .py file (or module:Class)")
    p.add_argument("--split", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--shard", metavar="i/n")
    g.add_argument("--rows", metavar="lo:hi", help="a capped run over manifest rows [lo, hi) (one shard; plan build --sample)")
    p.add_argument("--out", required=True, metavar="DIR", help="run directory (shards go to DIR/features/<split>/)")
    p.add_argument("--arg", action="append", default=[], metavar="k=v", help="bridge argument (Bridge.ARGS)")
    p.add_argument("--data", action="append", default=[], metavar="root=PATH", help="data root override")
    p.add_argument("--canaries", type=int, default=DEFAULT_CANARIES)
    p.add_argument("--canary-seed", type=int, default=0)
    p.add_argument("--decode-workers", type=int, default=None)
    p.add_argument("--read-ahead", type=int, default=DEFAULT_READ_AHEAD, help="batches whose bytes are read ahead")
    p.add_argument("--deadline", default=None, metavar="slurm|+S|EPOCH", help="default: $YAVIH_DEADLINE when set")
    p.add_argument("--expect-profile", default=None, metavar="profile.json",
                   help="the bridge check profile this unit was planned from: skip check + configuration-drift check")
    p.add_argument("--allow-loopback", action="store_true",
                   help="let encode() connect to 127.0.0.0/8 / ::1 (a local inference server); recorded in the sidecar")
    p.add_argument("--json", action="store_true", help="print the sidecar summary as one JSON object on stdout")
    return p


def _parse_shard(text: str) -> tuple[int, int]:
    try:
        a, b = text.split("/", 1)
        i, n = int(a), int(b)
    except ValueError as e:
        raise _ArgError(f"--shard expects i/n, got {text!r}") from e
    if not (0 <= i < n):
        raise _ArgError(f"--shard i/n needs 0 <= i < n, got {text!r}")
    return i, n


def parse_rows(text: str) -> tuple[int, int]:
    """``--rows lo:hi`` → ``(lo, hi)`` with ``0 <= lo < hi`` (``ArgError``, exit 1, otherwise)."""
    try:
        a, b = text.split(":", 1)
        lo, hi = int(a), int(b)
    except ValueError as e:
        raise _ArgError(f"--rows expects lo:hi, got {text!r}") from e
    if not (0 <= lo < hi):
        raise _ArgError(f"--rows lo:hi needs 0 <= lo < hi, got {text!r}")
    return lo, hi


def _kv(items: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise _ArgError(f"expected k=v, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m yavih.extract --bench B --bridge P --split S (--shard i/n | --rows lo:hi) --out DIR [--arg k=v]...
    [--canaries 8] [--decode-workers N] [--read-ahead 2] [--deadline ...] [--expect-profile P]``. Returns the CLI exit code; prints one
    ``[yavih] extract …`` header/footer line on stderr and, with ``--json``, the summary on stdout."""
    from yavih.cli import exit_code_for

    parser = _build_parser()
    try:
        ns = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        return 0 if code == 0 else 1
    as_json = bool(ns.json) or os.environ.get("YAVIH_JSON", "") == "1"
    _log(f"extract start bench={ns.bench} split={ns.split} shard={ns.shard} bridge={ns.bridge} yavih={__version__}")
    try:
        summary = run_extract(
            bench=ns.bench,
            bridge=ns.bridge,
            split=ns.split,
            shard=_parse_shard(ns.shard) if ns.shard is not None else None,
            rows=parse_rows(ns.rows) if ns.rows is not None else None,
            out=ns.out,
            args=ns.arg,
            canaries=ns.canaries,
            canary_seed=ns.canary_seed,
            decode_workers=ns.decode_workers,
            read_ahead=ns.read_ahead,
            deadline=parse_deadline(ns.deadline),
            expect_profile=ns.expect_profile,
            data=_kv(ns.data) or None,
            allow_loopback=bool(ns.allow_loopback),
        )
    except BaseException as exc:  # noqa: BLE001 - the process boundary maps everything to an exit code
        exit_code, code = exit_code_for(exc)
        if exit_code == 5:
            traceback.print_exc(file=sys.stderr)
        _log(f"extract failed {code} exit={exit_code}: {exc}")
        if as_json:
            sys.stdout.write(json.dumps({"status": "error", "data": {}, "errors": [{"code": code, "message": str(exc), "where": None}], "warnings": [], "paths": {}}, sort_keys=True) + "\n")
        return exit_code
    status = summary.get("status", "ok")
    _log(f"extract end status={status}")
    if as_json:
        paths = {k: summary[k] for k in ("sidecar_path", "path") if k in summary}
        sys.stdout.write(json.dumps({"status": status, "data": summary, "errors": [], "warnings": [], "paths": paths}, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
