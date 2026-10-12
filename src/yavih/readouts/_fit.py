"""Torch-side helpers shared by the readouts (common rules): row views over
``FeatureReader`` segments (features are consumed only via ``rows()``/``matrix()``, never gathered
whole in Python lists), micro-batched fp32 scoring, the ``train_head`` call, and the probe-fit
record.

Contract with ``yavih.probes.trainer.train_head`` (its module docstring): ``features`` is any row
gather with ``.shape == (n, N, D)`` and ``__getitem__(ndarray[int]) -> [k, N, D]``; ``targets`` is
``[n, ...]`` (int64, or float with NaN = invalid) or such a gather; ``pos`` is ``None``, ``[N, 4]`` or a
per-sample gather ``(n, N, 4)``; ``build_head(dim)`` returns the module (``head(x)`` or ``head(x, pos)``);
``deadline`` is an absolute ``time.monotonic()`` value. The resulting ``FitResult`` carries the trained
head with its best state restored.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from yavih.readouts.base import ProbeInterrupted, ProbeRefused, ReadoutContext

__all__ = [
    "FeatureRows",
    "TargetRows",
    "prepare",
    "score",
    "fit",
    "ckpt_path_for",
    "fit_record",
    "floor_outputs",
    "stream_index",
    "derangement",
    "micro_batch_size",
    "num_classes",
    "monotonic_deadline",
]


# ----------------------------------------------------------------------------- torch setup


def prepare(ctx: ReadoutContext) -> str:
    """Device + precision: probe training and scoring in fp32, ``allow_tf32 = False``."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if ctx.device is None:
        ctx.device = "cuda" if torch.cuda.is_available() else "cpu"
    return ctx.device


def micro_batch_size(batch: int, seq_len: int, max_tokens: int) -> int:
    try:
        from yavih.probes import trainer

        return int(trainer.micro_batch_size(batch, seq_len, max_tokens))
    except NotImplementedError:
        return max(1, min(int(batch), int(max_tokens) // max(1, int(seq_len))))


def monotonic_deadline(epoch_deadline: float | None) -> float | None:
    """The driver keeps deadlines in Unix epoch seconds (like ``SLURM_JOB_END_TIME``); the trainer
    wants ``time.monotonic()``. Same instant, other clock."""
    if epoch_deadline is None:
        return None
    return time.monotonic() + (float(epoch_deadline) - time.time())


def num_classes(ctx: ReadoutContext) -> int:
    """``len(classes.csv)`` (the index column is the ONLY logit vocabulary)."""
    try:
        return len(ctx.bench.classes())
    except NotImplementedError:
        out = (ctx.cfg.head or {}).get("out")
        if isinstance(out, int) and out > 1:
            return out
        raise


def stream_index(describe: dict, name: str) -> int:
    """Index of the stream NAMED ``name`` in ``describe()["streams"]`` (pos column 0,
    dense-av heads). Entries may be names or ``{"name": ...}`` records."""
    streams = describe.get("streams") or []
    for i, s in enumerate(streams):
        s_name = s.get("name") if isinstance(s, dict) else s
        if s_name == name:
            return i
    raise ProbeRefused("E_STREAM", f"stream {name!r} is not among describe()['streams'] = {streams}; validate should have recorded eligible: false")


def derangement(n: int, seed: int) -> np.ndarray:
    """Seeded permutation without fixed points (box shuffled control). ``n == 1`` has none:
    the identity is returned and the control is meaningless by construction."""
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    for _ in range(1000):
        fixed = np.flatnonzero(perm == np.arange(n))
        if fixed.size == 0:
            return perm.astype(np.int64)
        if fixed.size == 1:
            # swap the single fixed point with a random other position (keeps it a permutation)
            j = int(rng.integers(n - 1))
            j = j + 1 if j >= fixed[0] else j
            perm[fixed[0]], perm[j] = perm[j], perm[fixed[0]]
        else:
            perm[fixed] = perm[np.roll(fixed, 1)]
    return np.roll(np.arange(n), 1).astype(np.int64)  # pragma: no cover - unreachable in practice


def _index(idx: Any, n: int) -> tuple[np.ndarray, bool]:
    """``(ids int64 [k], scalar)`` for an int, a slice or an integer array index (the trainer and the
    floors stream with slices, the gathers use arrays)."""
    if isinstance(idx, slice):
        return np.arange(*idx.indices(n), dtype=np.int64), False
    scalar = np.isscalar(idx) or isinstance(idx, (int, np.integer))
    ids = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    ids = np.where(ids < 0, ids + n, ids)
    return ids, scalar


# ----------------------------------------------------------------------------- row views


class FeatureRows:
    """Indexable ``[n, N, D]`` view over one or more ``(reader, manifest rows)`` segments (train
    rows = union of splits.train present rows). ``__getitem__`` gathers through
    ``reader.matrix()`` (RAM when it fits; positional in ``present_rows`` order) or
    ``reader.rows()`` (memmap gather by manifest row) and returns the stored dtype; the trainer upcasts
    per mini-batch. ``pos_rows`` is the matching per-sample pos gather."""

    def __init__(self, segments: Sequence[tuple[Any, np.ndarray]]) -> None:
        self.segments = [(r, np.asarray(rows, dtype=np.int64)) for r, rows in segments]
        lengths = [rows.shape[0] for _, rows in self.segments]
        self._offsets = np.cumsum([0, *lengths])
        self.n = int(self._offsets[-1])
        first = self.segments[0][0] if self.segments else None
        self.N = int(getattr(first, "N", 0)) if first is not None else 0
        self.D = int(getattr(first, "D", 0)) if first is not None else 0
        self._views: list[Any] = [None] * len(self.segments)
        self._positions: list[np.ndarray | None] = [None] * len(self.segments)
        self.backend: str | None = None
        self._pos_const: np.ndarray | None = None
        self._pos_checked = False

    def __len__(self) -> int:
        return self.n

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n, self.N, self.D)

    def _view(self, k: int) -> Any:
        if self._views[k] is None:
            reader, rows = self.segments[k]
            try:
                view = reader.matrix()
                present = np.asarray(reader.present_rows, dtype=np.int64)
                pos = np.searchsorted(present, rows)
                if rows.size and not np.array_equal(present[np.clip(pos, 0, len(present) - 1)], rows):
                    raise ProbeRefused("E_ROWS", "requested rows are not present in the feature set")
                self._positions[k] = pos.astype(np.int64)
                self._views[k] = view
                self.backend = getattr(view, "backend", None) or self.backend
            except NotImplementedError:
                self._views[k] = reader  # fall back to rows() gather
                self._positions[k] = None
                self.backend = self.backend or "memmap"
        return self._views[k]

    def _gather(self, k: int, local: np.ndarray) -> np.ndarray:
        view = self._view(k)
        reader, rows = self.segments[k]
        pos = self._positions[k]
        if pos is None:
            return np.asarray(reader.rows(rows[local]))
        return np.asarray(view[pos[local]])

    def __getitem__(self, idx: int | slice | np.ndarray | Sequence[int]) -> np.ndarray:
        ids, scalar = _index(idx, self.n)
        if ids.size and (ids.min() < 0 or ids.max() >= self.n):
            raise IndexError(f"row index out of range [0, {self.n})")
        seg = np.searchsorted(self._offsets, ids, side="right") - 1
        out: np.ndarray | None = None
        for k in np.unique(seg):
            m = seg == k
            block = self._gather(int(k), ids[m] - self._offsets[k])
            if out is None:
                out = np.empty((ids.size, *block.shape[1:]), dtype=block.dtype)
            out[m] = block
        if out is None:
            out = np.empty((0, self.N, self.D), dtype=np.float32)
        return out[0] if scalar else out

    # -- pos
    def _pos_modes(self) -> list[str | None]:
        return [getattr(r, "pos_mode", None) for r, _ in self.segments]

    def pos_const(self) -> np.ndarray | None:
        """The constant ``[N, 4]`` pos shared by every segment (all ``pos_mode == constant``), else ``None``.

        Only a reader whose ``pos_mode`` is ``constant`` is asked for ``pos()``: a ``per_sample`` reader's
        ``pos()`` returns ``[n_present, N, 4]`` over ITS present rows, which is neither constant
        nor in this view's row order. A reader without a
        ``pos_mode`` attribute is accepted only when its ``pos()`` is 2-D."""
        if not self._pos_checked and self.segments:
            self._pos_checked = True
            self._pos_const = None
            modes = self._pos_modes()
            if any(m in ("none", "per_sample") for m in modes):
                return None
            consts = []
            for reader, _ in self.segments:
                p = reader.pos()
                if p is None:
                    return None
                p = np.asarray(p, dtype=np.float32)
                if p.ndim != 2:
                    return None
                consts.append(p)
            if all(c.shape == consts[0].shape and np.array_equal(c, consts[0], equal_nan=True) for c in consts[1:]):
                self._pos_const = consts[0]
        return self._pos_const

    def pos_rows(self, ids: np.ndarray) -> np.ndarray:
        """``[k, N, 4]`` pos of view rows ``ids``, each read from its own segment by MANIFEST row
        (``reader.pos(row)``; a constant reader returns its ``[N, 4]`` for any row)."""
        ids = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        seg = np.searchsorted(self._offsets, ids, side="right") - 1
        out = np.empty((ids.size, self.N, 4), dtype=np.float32)
        for j, (k, i) in enumerate(zip(seg, ids, strict=True)):
            reader, rows = self.segments[k]
            p = reader.pos(int(rows[i - self._offsets[k]]))
            if p is None:
                raise ProbeRefused("E_POS_MISSING", f"segment {int(k)} of the feature view has no pos (pos_mode {getattr(reader, 'pos_mode', None)!r})")
            out[j] = np.asarray(p, dtype=np.float32)
        return out

    def pos_gather(self) -> Any:
        """What ``train_head(pos=...)`` wants: ``[N, 4]`` when every segment has the same constant pos, a
        per-sample gather (by manifest row, segment by segment) when any segment is ``per_sample`` or the
        constants differ, ``None`` when a segment has no pos."""
        const = self.pos_const()
        if const is not None:
            return const
        if not self.segments:
            return None
        modes = self._pos_modes()
        if any(m == "none" for m in modes):
            return None
        if any(m == "per_sample" for m in modes):
            return _PosGather(self)
        if all(m == "constant" for m in modes) and all(r.pos() is not None for r, _ in self.segments):
            return _PosGather(self)  # constants that differ between splits: per row
        return None


class _PosGather:
    """``(n, N, 4)`` per-sample pos gather parallel to a ``FeatureRows``."""

    def __init__(self, rows: FeatureRows) -> None:
        self.rows = rows
        self.shape = (rows.n, rows.N, 4)

    def __len__(self) -> int:
        return self.rows.n

    def __getitem__(self, idx: Any) -> np.ndarray:
        ids, scalar = _index(idx, self.rows.n)
        out = self.rows.pos_rows(ids)
        return out[0] if scalar else out


class TargetRows:
    """Indexable view over a ``TargetReader`` for one role; ``nan_invalid`` replaces entries
    outside ``(0, max_depth]`` by NaN so the loss masks them."""

    def __init__(self, target_reader: Any, rows: np.ndarray, *, max_depth: float | None = None, nan_invalid: bool = False) -> None:
        self.tr = target_reader
        self.rows = np.asarray(rows, dtype=np.int64)
        self.max_depth = max_depth
        self.nan_invalid = nan_invalid

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self), *tuple(self.tr.shape[1:]))

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.float32)

    def __getitem__(self, idx: int | slice | np.ndarray | Sequence[int]) -> np.ndarray:
        ids, scalar = _index(idx, len(self))
        t = np.asarray(self.tr.rows(self.rows[ids]), dtype=np.float32)
        if self.nan_invalid:
            invalid = ~(t > 0)
            if self.max_depth is not None:
                invalid |= t > float(self.max_depth)
            t = t.copy()
            t[invalid] = np.nan
        return t[0] if scalar else t

    def all(self) -> np.ndarray:
        return self[np.arange(len(self))] if len(self) else np.empty(self.shape, dtype=np.float32)


# ----------------------------------------------------------------------------- scoring / fitting


def _to_tensor(a: Any) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a
    arr = np.asarray(a)
    return torch.from_numpy(np.ascontiguousarray(arr))


@torch.no_grad()
def score(ctx: ReadoutContext, head: Any, features: Any, *, pos: Any = None, batch_size: int | None = None) -> np.ndarray:
    """Head outputs on every row of ``features`` in micro-batches (token-capped, fp32),
    checking the stop flag between micro-batches. ``pos`` = ``None``, ``[N, 4]`` or a per-sample
    gather. Returns ``float32 [n, ...]``."""
    n = len(features)
    if n == 0:
        return np.empty((0,), dtype=np.float32)
    device = ctx.device or prepare(ctx)
    seq_len = getattr(features, "N", None) or ctx.N or 1
    mb = micro_batch_size(batch_size or int(ctx.cfg.batch_size), int(seq_len), int(ctx.cfg.max_tokens_per_forward))
    if isinstance(head, nn.Module):
        head.eval()
        head.to(device)
    const_pos = None
    if pos is not None and isinstance(pos, np.ndarray | torch.Tensor) and np.asarray(pos).ndim == 2:
        const_pos = _to_tensor(pos).float().to(device)
    chunks: list[np.ndarray] = []
    for i in range(0, n, mb):
        ctx.check_interrupt()
        ids = np.arange(i, min(n, i + mb))
        x = _to_tensor(features[ids]).float().to(device)
        if pos is None:
            out = head(x)
        elif const_pos is not None:
            out = head(x, const_pos)
        else:
            out = head(x, _to_tensor(pos[ids]).float().to(device))
        chunks.append(out.detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def ckpt_path_for(ctx: ReadoutContext, tag: str = "") -> Path:
    """``<seed dir>/probe_ckpt[.<tag>].pt``: one checkpoint file per fit of the seed (deadline checkpoint)."""
    return ctx.out_dir / (f"probe_ckpt{('.' + tag) if tag else ''}.pt")


def fit(ctx: ReadoutContext, features: Any, targets: Any, *, build_head: Callable[[int], Any], loss: str, pos: Any = None, tag: str = "") -> Any:
    """``yavih.probes.trainer.train_head`` under the common readout rules; the trainer's deadline/preemption
    exceptions become ``ProbeInterrupted`` (exit 4) after it wrote ``probe_ckpt.pt``.

    Every fit of a seed directory (main head, each ESC-50 fold, the box shuffled control) gets its own
    checkpoint identity: the file name carries ``tag`` and the resume gate carries ``tag``, the
    features fingerprint and the train-set size besides ``{probe_hash, seed}``, so one fit never resumes another fit's checkpoint (fold k
    never resumes fold 1's head, which has the same probe_hash and seed).
    A checkpoint is deleted once its fit completes normally: it exists only to resume an interrupted
    fit, never to be picked up by the next fit in the same directory.
    """
    from yavih.probes import trainer

    ctx.check_interrupt()
    device = ctx.device or prepare(ctx)
    ckpt = ckpt_path_for(ctx, tag)
    hashes = {"fit": tag or "main", "features_fingerprint": ctx.features_fingerprint, "n_rows": int(len(features))}
    t0 = time.time()
    try:
        res = trainer.train_head(
            features,
            targets,
            build_head=build_head,
            cfg=ctx.cfg,
            seed=ctx.seed,
            loss=loss,
            deadline=monotonic_deadline(ctx.deadline),
            ckpt_path=str(ckpt),
            device=device,
            loader_workers=ctx.loader_workers,
            stop_flag=ctx.stop_flag,
            pos=pos,
            ckpt_hashes=hashes,
        )
    except TimeoutError as e:  # trainer.ProbeDeadline
        raise ProbeInterrupted(getattr(e, "code", "E_DEADLINE"), f"{e}; probe_ckpt.pt written, resubmit") from e
    except RuntimeError as e:
        if getattr(e, "exit_code", None) == 4:  # trainer.ProbeInterrupted
            raise ProbeInterrupted(getattr(e, "code", "E_PREEMPTED"), str(e)) from e
        raise
    finally:
        ctx.timing["fit_s"] = ctx.timing.get("fit_s", 0.0) + (time.time() - t0)
    if ctx.stop_flag.is_set():
        raise ProbeInterrupted("E_PREEMPTED", f"signal {ctx.stop_flag.signal} received during fit")
    if getattr(res, "stop", None) == "deadline":
        raise ProbeInterrupted("E_DEADLINE", "deadline reached during fit; probe_ckpt.pt written, resubmit")
    try:
        ckpt.unlink(missing_ok=True)  # a finished fit leaves nothing a later fit could resume
    except OSError:  # pragma: no cover - best effort; the resume gate still refuses it
        pass
    return res


def fit_record(res: Any) -> dict:
    """The ``result.json`` ``probe`` block fields plus what ``probe_fit.json`` records."""
    return {
        "selected_lr": float(res.selected_lr) if getattr(res, "selected_lr", None) is not None else None,
        "epochs_run": int(res.epochs_run),
        "best_epoch": int(res.best_epoch),
        "stop": str(res.stop),
        "n_train": int(res.n_train),
        "n_val": int(res.n_val),
        "s_per_epoch": float(res.s_per_epoch),
        "best_val": (float(res.best_val) if getattr(res, "best_val", None) is not None and np.isfinite(float(res.best_val)) else None),
        "arms": list(getattr(res, "arms", []) or []),
        "history": list(getattr(res, "history", []) or []),
    }


def floor_outputs(ctx: ReadoutContext, train_targets: Any, n_eval: int, *, num_classes: int | None = None) -> Any:
    """Label-only constant eval outputs (floors are computed from TRAIN targets only)."""
    from yavih.probes import floors

    kwargs: dict[str, Any] = {"protocol": ctx.protocol, "label_kind": ctx.label_kind}
    if num_classes is not None:
        kwargs["num_classes"] = int(num_classes)
    return floors.floor_outputs(ctx.readout, train_targets, int(n_eval), **kwargs)
