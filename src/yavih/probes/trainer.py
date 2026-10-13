"""Probe training loop (the readouts' common rules): ``fork_rng`` + ``manual_seed``, seeded val split
from train only, AdamW, linear warmup → cosine 1→0, token-capped micro-batching with exact
per-sample-sum accumulation, selection on val loss (per lr for dense), patience, deadline
between epochs, ``probe_ckpt.pt`` (an in-memory copy every epoch, on disk at most every
``CKPT_EVERY_S`` and always at the end of an arm and on deadline / SIGTERM), SIGTERM flag between
micro-batches.

Features reach the loop through an abstract row gather: any object with ``.shape == (n, N, D)`` and
``__getitem__(np.ndarray[int]) -> [k, N, D]`` (``yavih.io.reader.RowView``, ``np.ndarray``, ``np.memmap``,
``torch.Tensor``). Rows are gathered per mini-batch and upcast to fp32 there (on CUDA after the
host→device copy, so fp16 features cross the link at half the bytes; fp16→fp32 is exact either way); a
background thread gathers the next ``PREFETCH_DEPTH`` batches in epoch order. The loop never materialises the split.
"""

from __future__ import annotations

import collections
import contextlib
import math
import os
import re
import time
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from yavih.hashing import probe_hash
from yavih.probes.attentive import build_attentive_probe
from yavih.probes.ckpt import load_ckpt, rng_state, save_ckpt, set_rng_state
from yavih.probes.config import ProbeConfig
from yavih.probes.dense import build_dense_av_head, build_dense_query_head
from yavih.probes.losses import per_sample_loss

__all__ = [
    "FitResult",
    "train_head",
    "build_head_for",
    "micro_batch_size",
    "accumulate_batch",
    "predict",
    "ProbeDeadline",
    "ProbeInterrupted",
    "val_split",
    "lr_lambda",
    "precision_scope",
    "determinism_record",
    "reset_determinism",
    "default_device",
    "to_tensor_targets",
    "CKPT_EVERY_S",
    "ckpt_every_s",
    "PREFETCH_DEPTH",
]

#: Deadline: ``probe_ckpt.pt`` reaches disk at most every this many seconds; between writes the last
#: completed epoch is held in memory (CPU) and written on the deadline, on SIGTERM and at the end of every arm, so a
#: resume continues from exactly the same epoch as before. Only a hard kill (SIGKILL, OOM) loses up to this much
#: work. A dense head's checkpoint is 0.6–0.8 GB, too large to rewrite every epoch on network storage.
#: ``YAVIH_PROBE_CKPT_EVERY_S`` overrides (0 = every epoch). Not hashed: it decides
#: when the file is written, never what a fit or a resume computes.
CKPT_EVERY_S: float = 600.0

#: Training batches gathered ahead by the loader thread (``loader_workers >= 1``), consumed in epoch order: the
#: batch sequence, and therefore every number, is the inline loop's.
PREFETCH_DEPTH: int = 2


class ProbeDeadline(TimeoutError):
    """The deadline passed between epochs: ``probe_ckpt.pt`` is on disk, exit 4."""

    exit_code = 4
    code = "E_DEADLINE"

    def __init__(self, message: str, partial: FitResult | None = None) -> None:
        super().__init__(message)
        self.partial = partial


class ProbeInterrupted(RuntimeError):
    """``stop_flag`` was set between micro-batches (SIGTERM/SIGUSR1): exit 4."""

    exit_code = 4
    code = "E_INTERRUPTED"

    def __init__(self, message: str, partial: FitResult | None = None) -> None:
        super().__init__(message)
        self.partial = partial


@dataclass
class FitResult:
    """What ``result.json[probe]`` and ``probe_fit.json`` need."""

    head: Any  # the trained torch module (best state restored)
    selected_lr: float
    epochs_run: int
    best_epoch: int
    stop: str  # "patience" | "max_epochs" | "deadline"
    n_train: int
    n_val: int
    s_per_epoch: float
    history: list[dict] = field(default_factory=list)
    best_val: float = math.inf
    arms: list[dict] = field(default_factory=list)  # one {lr, best_val, epochs_run, best_epoch, stop} per lr arm
    device: str = "cpu"

    def to_json(self) -> dict[str, Any]:
        return {
            "selected_lr": self.selected_lr,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "stop": self.stop,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "s_per_epoch": self.s_per_epoch,
            "best_val": self.best_val,
            "arms": list(self.arms),
            "history": list(self.history),
            "device": self.device,
        }


# ----------------------------------------------------------------------------- helpers


def micro_batch_size(batch: int, seq_len: int, max_tokens_per_forward: int) -> int:
    """Largest micro-batch with ``mb * seq_len <= max_tokens_per_forward`` (≥ 1)."""
    if seq_len < 1:
        return max(1, int(batch))
    return max(1, min(int(batch), int(max_tokens_per_forward) // int(seq_len)))


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def val_split(n_train: int, val_fraction: float, seed: int) -> tuple[Tensor, Tensor]:
    """Seeded permutation (``torch.Generator().manual_seed(seed)``) of the train rows; ``n_val = max(1,
    round(val_fraction * n_train))`` capped at ``n_train - 1``. With a single
    row there is nothing to carve: val = train (selection then degenerates to train loss)."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(int(n_train), generator=g)
    if n_train <= 1:
        return perm, perm.clone()
    n_val = min(max(1, int(round(val_fraction * n_train))), n_train - 1)
    return perm[: n_train - n_val], perm[n_train - n_val :]


def lr_lambda(total_steps: int, warmup_fraction: float) -> Callable[[int], float]:
    """Linear warmup over ``ceil(warmup_fraction * total)`` steps (≥ 1 when the fraction is > 0), then cosine
    1→0 over the remaining steps. ``warmup_fraction == 0`` is a plain cosine."""
    total = max(1, int(total_steps))
    warm = max(1, math.ceil(warmup_fraction * total)) if warmup_fraction > 0 else 0

    def f(step: int) -> float:
        if step < warm:
            return float(step + 1) / float(warm)
        progress = min(1.0, max(0.0, (step - warm) / max(1, total - warm)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return f


#: Ops torch reported as having run without a deterministic kernel under ``warn_only=True`` since the last
#: ``reset_determinism()`` (the readout driver resets per seed); ``determinism_record`` reports them.
_NONDETERMINISTIC_OPS: set[str] = set()
#: The SDPA kernels warn only ONCE per process (``TORCH_WARN_ONCE``): once seen, a later CUDA fit in the same process
#: that entered ``precision_scope`` is assumed to have run them again (it cannot be observed a second time).
_ONCE_OPS: set[str] = set()
_SCOPES_SINCE_RESET = 0
#: torch's two forms: ``<op> does not have a deterministic implementation, but you set ...warn_only=True`` and the SDPA
#: kernels' ``Memory Efficient attention defaults to a non-deterministic algorithm`` (also Flash Attention).
_NONDET_RES = (
    re.compile(r"^\s*(?P<op>\S+) does not have a deterministic implementation"),
    re.compile(r"^\s*(?P<op>(?:Memory Efficient|Flash) [Aa]ttention) defaults to a non-deterministic algorithm"),
)


def _nondeterministic_op(message: str) -> str | None:
    for rx in _NONDET_RES:
        m = rx.match(message)
        if m is not None:
            return m.group("op").replace(" ", "_").lower() if " " in m.group("op") else m.group("op")
    return None


def reset_determinism() -> None:
    """Start a new determinism record (one per fitted seed)."""
    global _SCOPES_SINCE_RESET
    _NONDETERMINISTIC_OPS.clear()
    _SCOPES_SINCE_RESET = 0


def determinism_record(device: str | None) -> dict[str, Any]:
    """The determinism this process actually achieved: ``mode`` is ``warn_only`` (what
    ``precision_scope`` requests), ``nondeterministic_ops`` the ops torch reported as having run without a
    deterministic kernel, ``bitwise_reproducible`` whether none did (a rerun then reproduces bit for bit)."""
    ops = set(_NONDETERMINISTIC_OPS)
    if _SCOPES_SINCE_RESET and str(device or "").startswith("cuda"):
        ops |= _ONCE_OPS
    ops = sorted(ops)
    return {"mode": "warn_only", "device": device, "nondeterministic_ops": ops, "bitwise_reproducible": not ops}


@contextlib.contextmanager
def precision_scope() -> Iterator[None]:
    """Determinism and precision: fp32, ``allow_tf32 = False``, ``matmul_precision highest``,
    ``use_deterministic_algorithms(True, warn_only=True)``, ``cudnn.benchmark = False``; every torch global is
    restored on exit. ``CUBLAS_WORKSPACE_CONFIG`` is set (if unset) for the process, as ``run_extract`` does: cuBLAS
    reads it when its handle is created, i.e. at the first CUDA matmul, which in a probe process happens in here.

    Not covered (``warn_only=True``): ops without a deterministic CUDA kernel still run, notably the
    fp32 attention backward (torch's memory-efficient SDPA kernel is deterministic only under ``warn_only=False``,
    which would refuse the grid head's bilinear-upsample backward), so CUDA reruns agree to float rounding. Those ops
    are recorded (torch's warn_only warning is captured, not shown) and reported by ``determinism_record``."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_mm = torch.get_float32_matmul_precision()
    prev_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    prev_bench = torch.backends.cudnn.benchmark
    global _SCOPES_SINCE_RESET
    _SCOPES_SINCE_RESET += 1
    caught = warnings.catch_warnings(record=True)
    records = caught.__enter__()
    warnings.simplefilter("always")
    try:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        caught.__exit__(None, None, None)
        # Record the ops torch ran without a deterministic kernel; every other warning is re-emitted unchanged.
        for w in records:
            op = _nondeterministic_op(str(w.message))
            if op is not None:
                _NONDETERMINISTIC_OPS.add(op)
                if op.endswith("_attention"):
                    _ONCE_OPS.add(op)
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
        torch.set_float32_matmul_precision(prev_mm)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev_tf32
        torch.backends.cudnn.benchmark = prev_bench


def to_tensor_targets(targets: Any) -> Tensor:
    """Targets as a CPU tensor: integer kinds → int64, everything else → float32 (NaN = invalid)."""
    if isinstance(targets, Tensor):
        t = targets.detach().cpu()
    else:
        t = torch.as_tensor(np.asarray(targets))
    if t.is_floating_point():
        return t.to(torch.float32)
    if t.dtype == torch.bool:
        return t.to(torch.float32)
    return t.to(torch.int64)


def _cast_on_device(device: Any) -> bool:
    """Whether gathered rows travel in their stored dtype and are upcast after the host→device copy: on CUDA (half
    the bytes over the link for fp16 features). On CPU the copy is a no-op and the gather upcasts."""
    return device is not None and torch.device(device).type == "cuda"


def _gather(features: Any, idx: Tensor, *, keep_dtype: bool = False) -> Tensor:
    """``features[idx] -> [k, N, D]`` CPU tensor through the abstract row gather: fp32, or with ``keep_dtype``
    the stored dtype (pinned when CUDA is available) for the caller's ``.to(device, non_blocking=True).float()``.
    The fp16→fp32 cast is exact, so both orders give the same fp32 values bit for bit."""
    ix = idx.numpy() if isinstance(idx, Tensor) else np.asarray(idx)
    if isinstance(features, Tensor):
        x = features[torch.as_tensor(ix, dtype=torch.long)]
    else:
        x = features[ix]
        if not isinstance(x, Tensor):
            x = torch.from_numpy(np.ascontiguousarray(x))
    if not keep_dtype:
        return x.float()
    if x.device.type == "cpu" and torch.cuda.is_available():
        x = x.pin_memory()
    return x


def _gather_pos(pos: Any, idx: Tensor, *, keep_dtype: bool = False) -> Tensor | None:
    """Constant ``[N, 4]`` pos is passed through; per-sample ``[n, N, 4]`` is gathered like features."""
    if pos is None:
        return None
    if isinstance(pos, Tensor) and pos.dim() == 2:
        return pos.float()
    if not isinstance(pos, Tensor) and getattr(pos, "ndim", len(getattr(pos, "shape", (0, 0, 0)))) == 2:
        return torch.as_tensor(np.asarray(pos), dtype=torch.float32)
    return _gather(pos, idx, keep_dtype=keep_dtype)


def _to_device(t: Tensor | None, device: Any) -> Tensor | None:
    """Host→device copy, then the fp32 upcast (a no-op for rows already gathered as fp32)."""
    return None if t is None else t.to(device, non_blocking=True).float()


def _forward(head: nn.Module, x: Tensor, p: Tensor | None) -> Tensor:
    return head(x) if p is None else head(x, p)


def _flag_set(stop_flag: Any) -> bool:
    if stop_flag is None:
        return False
    if hasattr(stop_flag, "is_set"):
        return bool(stop_flag.is_set())
    if callable(stop_flag):
        return bool(stop_flag())
    return bool(stop_flag)


def _clone_state(head: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone().cpu() for k, v in head.state_dict().items()}


def _cpu_copy(obj: Any) -> Any:
    """A CPU copy of a (nested) state dict: tensors copied off the device, containers rebuilt, scalars shared."""
    if isinstance(obj, Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_copy(v) for v in obj)
    return obj


def ckpt_every_s(env: Any = None) -> float:
    """``YAVIH_PROBE_CKPT_EVERY_S`` (seconds, ``0`` = every epoch), else :data:`CKPT_EVERY_S`."""
    raw = (os.environ if env is None else env).get("YAVIH_PROBE_CKPT_EVERY_S")
    if raw is None or str(raw).strip() == "":
        return CKPT_EVERY_S
    try:
        val = float(raw)
    except ValueError as e:
        raise ValueError(f"YAVIH_PROBE_CKPT_EVERY_S={raw!r} is not a number of seconds") from e
    if not math.isfinite(val) or val < 0:
        raise ValueError(f"YAVIH_PROBE_CKPT_EVERY_S={raw!r} must be a finite number >= 0")
    return val


def predict(head: nn.Module, features: Any, idx: Any, *, pos: Any = None, batch_size: int = 256,
            max_tokens_per_forward: int = 131072, device: str | None = None) -> Tensor:
    """Chunked, token-capped scoring (the validation/scoring path is capped like training). Returns the head's raw output on CPU, fp32,
    rows in ``idx`` order."""
    device = device or next(head.parameters()).device
    idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    n_tokens = int(features.shape[1])
    chunk = micro_batch_size(batch_size, n_tokens, max_tokens_per_forward)
    outs: list[Tensor] = []
    defer = _cast_on_device(device)
    head.eval()
    with torch.no_grad(), precision_scope():
        for i in range(0, len(idx_t), chunk):
            sel = idx_t[i : i + chunk]
            x = _to_device(_gather(features, sel, keep_dtype=defer), device)
            p = _to_device(_gather_pos(pos, sel, keep_dtype=defer), device)
            outs.append(_forward(head, x, p).float().cpu())
    if not outs:
        return torch.empty(0)
    return torch.cat(outs, dim=0)


def accumulate_batch(head: nn.Module, criterion: Callable[[Tensor, Tensor], Tensor], x: Tensor, y: Tensor,
                     p: Tensor | None, *, micro: int, device: str, stop_flag: Any = None,
                     on_stop: Callable[[], FitResult | None] | None = None) -> float:
    """Backward of the batch loss ``per_sample.sum() / batch`` in micro-batches of ``micro`` rows.

    ``sum() / batch`` in every micro-batch: the sums add up to the whole-batch numerator, so the
    accumulated gradient equals the whole-batch gradient exactly (up to fp32 summation order) for every
    loss, not only for batch-mean CE. ``stop_flag`` is polled between micro-batches.
    ``x`` / ``p`` may be in the stored dtype (fp16): each micro-batch is upcast to fp32 after its copy to ``device``.
    Returns the batch's per-sample loss SUM (for the epoch's train-loss average)."""
    batch = int(x.shape[0])
    total = 0.0
    for j in range(0, batch, micro):
        if _flag_set(stop_flag):
            raise ProbeInterrupted("probe interrupted between micro-batches", on_stop() if on_stop else None)
        xb = _to_device(x[j : j + micro], device)
        yb = y[j : j + micro].to(device, non_blocking=True)
        pb = None if p is None else _to_device(p if p.dim() == 2 else p[j : j + micro], device)
        per = criterion(_forward(head, xb, pb).float(), yb)
        (per.sum() / batch).backward()
        total += float(per.detach().sum().item())
    return total


# ----------------------------------------------------------------------------- one lr arm


@dataclass
class _ArmState:
    epoch: int = 0  # completed epochs
    best_val: float = math.inf
    best_epoch: int = -1
    best_state: dict[str, Tensor] | None = None
    patience_left: int = 0
    history: list[dict] = field(default_factory=list)
    s_total: float = 0.0


def _train_one(
    features: Any,
    targets: Tensor,
    *,
    pos: Any,
    build_head: Callable[[int], nn.Module],
    cfg: ProbeConfig,
    seed: int,
    lr: float,
    arm: int,
    criterion: Callable[[Tensor, Tensor], Tensor],
    train_idx: Tensor,
    val_idx: Tensor,
    device: str,
    deadline: float | None,
    ckpt_path: str | None,
    hashes: dict,
    completed: list[dict],
    stop_flag: Any,
    loader_workers: int,
) -> tuple[nn.Module, _ArmState, str]:
    """Train one head at one lr inside its own ``fork_rng`` + ``manual_seed`` scope so every arm sees the
    same split and the same init. Returns ``(head, state, stop)``."""
    n_tokens, dim = int(features.shape[1]), int(features.shape[-1])
    n_train = int(len(train_idx))
    batch_size = min(cfg.batch_size, n_train)
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    total_steps = steps_per_epoch * cfg.max_epochs

    rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=rng_devices), precision_scope():
        torch.manual_seed(int(seed))
        head = build_head(dim).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=float(lr), betas=(0.9, 0.999),
                                      weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(total_steps, cfg.warmup_fraction))
        loader_gen = torch.Generator().manual_seed(int(seed))
        st = _ArmState(patience_left=cfg.patience)

        resumed = load_ckpt(ckpt_path, hashes=hashes) if ckpt_path else None
        if resumed is not None and resumed.get("arm") == arm:
            head.load_state_dict(resumed["state_dict"])
            optimizer.load_state_dict(resumed["optimizer"])
            scheduler.load_state_dict(resumed["scheduler"])
            set_rng_state(resumed["rng"])
            loader_gen.set_state(resumed["loader_rng"])
            st = _ArmState(
                epoch=int(resumed["epoch"]),
                best_val=float(resumed["best_val"]),
                best_epoch=int(resumed.get("best_epoch", -1)),
                best_state=resumed["best_state"],
                patience_left=int(resumed["patience_left"]),
                history=list(resumed.get("history", [])),
                s_total=float(resumed.get("s_total", 0.0)),
            )

        # The last completed epoch, held in memory until it is written (CKPT_EVERY_S): the snapshot is a COPY, so
        # the partial epoch a SIGTERM interrupts can never leak into it.
        every_s = ckpt_every_s()
        ck: dict[str, Any] = {"pending": None, "written": time.monotonic()}

        def snapshot() -> dict:
            return {
                "arm": arm,
                "lr": float(lr),
                "completed": list(completed),
                "state_dict": _clone_state(head),
                "optimizer": _cpu_copy(optimizer.state_dict()),
                "scheduler": _cpu_copy(scheduler.state_dict()),
                "epoch": st.epoch,
                "rng": rng_state(),
                "loader_rng": loader_gen.get_state(),
                "best_state": st.best_state,
                "best_val": st.best_val,
                "best_epoch": st.best_epoch,
                "patience_left": st.patience_left,
                "history": list(st.history),
                "s_total": st.s_total,
            }

        def flush_ckpt() -> None:
            if ckpt_path is not None and ck["pending"] is not None:
                save_ckpt(ckpt_path, ck["pending"], hashes=hashes)
                ck["written"] = time.monotonic()
            ck["pending"] = None

        def partial(stop: str) -> FitResult:
            return FitResult(head=head, selected_lr=float(lr), epochs_run=st.epoch, best_epoch=st.best_epoch,
                             stop=stop, n_train=n_train, n_val=int(len(val_idx)), s_per_epoch=st.s_total / max(1, st.epoch),
                             history=list(st.history), best_val=st.best_val, arms=list(completed), device=str(device))

        stop = "max_epochs"
        if st.patience_left <= 0:
            stop = "patience"  # resumed from the checkpoint of an arm that had already stopped: train nothing more
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yavih-loader") if loader_workers > 0 else None
        defer = _cast_on_device(device)
        try:
            while st.epoch < cfg.max_epochs and st.patience_left > 0:
                # Deadline between epochs: the checkpoint of the last completed epoch is on disk.
                if deadline is not None and time.monotonic() >= deadline:
                    raise ProbeDeadline(f"probe deadline passed after epoch {st.epoch}", partial("deadline"))
                t0 = time.monotonic()
                head.train()
                perm = train_idx[torch.randperm(n_train, generator=loader_gen)]
                train_loss_sum = 0.0

                def batch_at(i: int, perm: Tensor = perm) -> tuple[Tensor, Tensor | None, Tensor]:
                    sel = perm[i : i + batch_size]
                    return _gather(features, sel, keep_dtype=defer), _gather_pos(pos, sel, keep_dtype=defer), targets[sel]

                starts = list(range(0, n_train, batch_size))
                # One loader thread, up to PREFETCH_DEPTH batches queued ahead, consumed first-in first-out: the
                # batch order is the inline loop's.
                ahead: collections.deque = collections.deque()
                queued = 0
                if pool is not None:
                    while queued < len(starts) and len(ahead) < PREFETCH_DEPTH:
                        ahead.append(pool.submit(batch_at, starts[queued]))
                        queued += 1
                for i in starts:
                    if pool is not None:
                        x, p, y = ahead.popleft().result()
                        if queued < len(starts):
                            ahead.append(pool.submit(batch_at, starts[queued]))
                            queued += 1
                    else:
                        x, p, y = batch_at(i)
                    batch = int(x.shape[0])
                    micro = micro_batch_size(batch, n_tokens, cfg.max_tokens_per_forward)
                    optimizer.zero_grad(set_to_none=True)
                    train_loss_sum += accumulate_batch(head, criterion, x, y, p, micro=micro, device=device,
                                                       stop_flag=stop_flag,
                                                       on_stop=lambda: partial("interrupted"))
                    optimizer.step()
                    scheduler.step()

                head.eval()
                val_loss = _eval_loss(head, features, targets, val_idx, pos=pos, criterion=criterion, cfg=cfg,
                                      n_tokens=n_tokens, device=device)
                st.epoch += 1
                dt = time.monotonic() - t0
                st.s_total += dt
                st.history.append({"epoch": st.epoch, "train_loss": train_loss_sum / max(1, n_train),
                                   "val_loss": val_loss, "lr": float(optimizer.param_groups[0]["lr"]), "s": dt})
                if val_loss < st.best_val:
                    st.best_val, st.best_epoch, st.patience_left = val_loss, st.epoch, cfg.patience
                    st.best_state = _clone_state(head)
                else:
                    st.patience_left -= 1
                if ckpt_path is not None:
                    ck["pending"] = snapshot()
                    if every_s <= 0 or time.monotonic() - ck["written"] >= every_s:
                        flush_ckpt()
                if st.patience_left <= 0:
                    stop = "patience"
                    break
            flush_ckpt()  # the arm's last epoch: a kill before the next arm's first write loses nothing of this arm
        except (ProbeDeadline, ProbeInterrupted):
            flush_ckpt()  # the last COMPLETED epoch is on disk before exit 4
            raise
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)

        if st.best_state is not None:
            head.load_state_dict(st.best_state)
    head.eval()
    return head, st, stop


def _eval_loss(head: nn.Module, features: Any, targets: Tensor, idx: Tensor, *, pos: Any, criterion, cfg: ProbeConfig,
               n_tokens: int, device: str) -> float:
    """Mean per-sample loss over ``idx``, chunked under the same token cap (exact: per-sample sums)."""
    chunk = micro_batch_size(cfg.batch_size, n_tokens, cfg.max_tokens_per_forward)
    total, seen = 0.0, 0
    defer = _cast_on_device(device)
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            sel = idx[i : i + chunk]
            x = _to_device(_gather(features, sel, keep_dtype=defer), device)
            p = _to_device(_gather_pos(pos, sel, keep_dtype=defer), device)
            total += float(criterion(_forward(head, x, p).float(), targets[sel].to(device)).sum().item())
            seen += int(len(sel))
    return total / max(1, seen)


# ----------------------------------------------------------------------------- entry point


def train_head(features: Any, targets: Any, *, build_head, cfg, seed: int, loss: str | None = None,
               deadline: float | None = None, ckpt_path: str | None = None, device: str | None = None,
               loader_workers: int | None = None, stop_flag=None, pos: Any = None,
               ckpt_hashes: dict | None = None) -> FitResult:
    """One probe fit per lr in ``cfg.lr_grid or (cfg.lr,)``; every arm re-forks and re-seeds so each sees the
    same split and init. Selection = lowest val loss across arms
    (val loss per lr).

    ``features``: row gather ``(n, N, D)`` (see module docstring); ``targets``: ``[n, ...]`` (int64 or float
    with NaN = invalid); ``pos``: ``None``, ``[N, 4]`` or a per-sample gather ``(n, N, 4)``; ``build_head(dim)``
    returns the head module (``head(x)`` or ``head(x, pos)``); ``loss`` defaults to ``cfg.loss``; ``deadline``
    is an absolute ``time.monotonic()`` value; ``ckpt_hashes`` (default ``{probe_hash, seed}``) gate the resume;
    ``loader_workers`` ``None`` (default) or ``>= 1`` = one loader thread gathering ``PREFETCH_DEPTH`` batches ahead,
    ``0`` = gather inline (same numbers either way).
    Raises ``ProbeDeadline`` / ``ProbeInterrupted`` (both ``exit_code 4``, with ``.partial`` set)."""
    if not isinstance(cfg, ProbeConfig):
        raise TypeError("train_head: cfg must be a ProbeConfig")
    n = int(features.shape[0])
    if n == 0:
        raise ValueError("train_head: features is empty (n == 0)")
    targets_t = to_tensor_targets(targets)
    if int(targets_t.shape[0]) != n:
        raise ValueError(f"train_head: features/targets length mismatch ({n} vs {int(targets_t.shape[0])})")
    device = device or default_device()
    criterion = per_sample_loss(loss or cfg.loss)
    hashes = dict(ckpt_hashes) if ckpt_hashes is not None else {}
    hashes.setdefault("probe_hash", probe_hash(cfg))
    hashes.setdefault("seed", int(seed))
    workers = int(loader_workers) if loader_workers is not None else 1

    train_idx, val_idx = val_split(n, cfg.val_fraction, seed)
    grid = cfg.lr_grid or (cfg.lr,)

    # Arms finished before a resume live in the checkpoint; skip them (same seed → same numbers).
    completed: list[dict] = []
    resumed = load_ckpt(ckpt_path, hashes=hashes) if ckpt_path else None
    if resumed is not None:
        completed = [dict(a) for a in resumed.get("completed", [])]
    done_lrs = {float(a["lr"]) for a in completed}

    best: tuple[nn.Module, _ArmState, str, float] | None = None
    for arm, lr in enumerate(grid):
        if float(lr) in done_lrs:
            continue
        head, st, stop = _train_one(
            features, targets_t, pos=pos, build_head=build_head, cfg=cfg, seed=seed, lr=float(lr), arm=arm,
            criterion=criterion, train_idx=train_idx, val_idx=val_idx, device=device, deadline=deadline,
            ckpt_path=ckpt_path, hashes=hashes, completed=completed, stop_flag=stop_flag, loader_workers=workers,
        )
        rec = {"lr": float(lr), "best_val": st.best_val, "epochs_run": st.epoch, "best_epoch": st.best_epoch,
               "stop": stop, "s_per_epoch": st.s_total / max(1, st.epoch), "history": list(st.history),
               "best_state": st.best_state}
        completed.append(rec)
        if best is None or st.best_val < best[3]:
            best = (head, st, stop, st.best_val)

    win = min(completed, key=lambda a: a["best_val"])
    if best is not None and best[3] == win["best_val"]:
        head, st, stop, _ = best
    else:
        # The winner is an arm completed before a resume: rebuild it from its saved best state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            head = build_head(int(features.shape[-1])).to(device)
        head.load_state_dict(win["best_state"])
        head.eval()
        st = _ArmState(epoch=win["epochs_run"], best_val=win["best_val"], best_epoch=win["best_epoch"],
                       history=list(win["history"]), s_total=win["s_per_epoch"] * max(1, win["epochs_run"]))
        stop = win["stop"]
    arms = [{k: v for k, v in a.items() if k not in {"best_state", "history"}} for a in completed]
    return FitResult(
        head=head,
        selected_lr=float(win["lr"]),
        epochs_run=st.epoch,
        best_epoch=st.best_epoch,
        stop=stop,
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        s_per_epoch=st.s_total / max(1, st.epoch),
        history=list(st.history),
        best_val=float(st.best_val),
        arms=arms,
        device=str(device),
    )


def build_head_for(cfg: ProbeConfig, *, visual_tag: int | None = None,
                   conditioning_tag: int | None = None) -> Callable[[int], nn.Module]:
    """``build_head`` closure for ``train_head`` from ``cfg.head`` (head families). ``visual_tag`` /
    ``conditioning_tag`` are indices into ``describe()["streams"]`` (dense heads; the readout resolves
    ``readout_params.visual_stream`` / ``conditioning_stream`` to them)."""
    head = dict(cfg.head)
    fam = head.get("family")
    if fam == "attentive_probe_v1":
        return lambda dim: build_attentive_probe(dim, int(head["out"]), head)
    if fam == "dense_query_v1":
        return lambda dim: build_dense_query_head(dim, int(head["out"]), head, pos_stream_tag=visual_tag)
    if fam == "dense_av_v1":
        v = 0 if visual_tag is None else int(visual_tag)
        c = 1 if conditioning_tag is None else int(conditioning_tag)
        return lambda dim: build_dense_av_head(dim, head, visual_tag=v, conditioning_tag=c)
    raise ValueError(f"unknown head.family {fam!r}")
