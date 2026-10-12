"""Host resources: the CPU and memory a process may use, the RAM budget and the decode-worker default.

ONE implementation for the package (``io.reader``, ``io.validate``, ``readouts.base``, ``plan.units``,
``plan.exec``, ``plan.local``, ``sample`` and ``extract`` call it). Every value is read in this order:

- CPUs: ``YAVIH_CPUS`` / config ``[resources] cpus`` → ``SLURM_CPUS_PER_TASK`` → PBS ``NCPUS`` /
  ``PBS_NUM_PPN`` → the scheduler affinity mask (``os.sched_getaffinity``) and the cgroup CPU quota
  (``cpu.max`` v2, ``cpu.cfs_quota_us`` v1; Kubernetes, Docker ``--cpus``, ``taskset``) → ``os.cpu_count()``.
  ``exec --cpus`` / ``exec --slots`` set ``YAVIH_CPUS`` for their children, so a slot's decode workers follow its
  share. :func:`available_cpus` is the count alone (``exec``'s per-unit and per-slot CPU budget).
- memory limit: ``SLURM_MEM_PER_NODE`` (MB) → ``SLURM_MEM_PER_CPU`` × CPUs → the cgroup memory limit
  (``memory.max`` v2, ``memory.limit_in_bytes`` v1). The limit that OOM-kills the process, never the
  node's physical RAM.
- RAM budget: ``YAVIH_RAM_BUDGET_GB`` / ``[resources] ram_budget_gb`` → ``min(16 GiB, 0.45 ×
  memory limit)`` → 16 GiB. A malformed value is a ``ConfigError`` everywhere (never silently ignored).
- decode workers: ``YAVIH_DECODE_WORKERS`` / ``[resources] decode_workers`` → ``CPUs − 1``
  when the CPUs are an allocation (scheduler, affinity or cgroup), else ``CPUs // 2`` (a shared
  workstation), capped at :data:`MAX_DECODE_WORKERS`, at least 1.

With an explicit ``env`` mapping only that mapping is read (no config file, no affinity or cgroup
probe), so the rules are testable; ``env=None`` reads ``yavih.config.load()`` and the host.

stdlib only.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "MAX_DECODE_WORKERS",
    "DEFAULT_RAM_BUDGET_BYTES",
    "RAM_BUDGET_FRACTION",
    "RAM_BUDGET_ENV",
    "available_cpus",
    "cgroup_cpu_quota",
    "cpu_limit",
    "memory_limit_bytes",
    "ram_budget_bytes",
    "decode_workers",
    "describe",
]

MAX_DECODE_WORKERS = 16  # per task
DEFAULT_RAM_BUDGET_BYTES: int = 16 * 1024**3
RAM_BUDGET_FRACTION: float = 0.45
RAM_BUDGET_ENV = "YAVIH_RAM_BUDGET_GB"
_MIB = 1024**2
_CGROUP = Path("/sys/fs/cgroup")


def _configured(key: str, env: Mapping[str, str] | None, settings: Any = None) -> tuple[Any, str] | None:
    """``[resources] <key>`` with its env override, as ``(value, source)`` (from ``settings`` when given)."""
    from yavih import config

    if env is not None:
        name = {"ram_budget_gb": RAM_BUDGET_ENV, "cpus": "YAVIH_CPUS", "decode_workers": "YAVIH_DECODE_WORKERS"}[key]
        raw = env.get(name)
        if raw in (None, ""):
            return None
        try:
            return config._coerce_resource(key, raw), f"env:{name}"
        except config.ConfigError as e:
            raise config.ConfigError(f"{name}={raw!r}: {e}") from None
    v = (settings if settings is not None else config.load()).resource(key)
    return (v.value, v.source) if v is not None else None


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def cgroup_cpu_quota(root: Path | str = _CGROUP) -> int | None:
    """``ceil(quota / period)`` of the cgroup at ``root`` (v2 ``cpu.max``, else v1 ``cpu/cpu.cfs_quota_us``),
    ``None`` when unlimited or unreadable."""
    root = Path(root)
    raw = _read(root / "cpu.max")
    if raw:
        parts = raw.split()
        if len(parts) >= 2 and parts[0] != "max":
            try:
                quota, period = float(parts[0]), float(parts[1])
            except ValueError:
                return None
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))
        return None
    try:
        quota = float(_read(root / "cpu" / "cpu.cfs_quota_us") or "-1")
        period = float(_read(root / "cpu" / "cpu.cfs_period_us") or "0")
    except ValueError:
        return None
    if quota > 0 and period > 0:
        return max(1, math.ceil(quota / period))
    return None


def _cgroup_memory(root: Path = _CGROUP) -> int | None:
    for path in (root / "memory.max", root / "memory" / "memory.limit_in_bytes"):
        try:
            raw = path.read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            v = int(raw)
        except ValueError:
            continue
        return v if 0 < v < 2**60 else None  # v1 reports "no limit" as a huge number
    return None


def cpu_limit(env: Mapping[str, str] | None = None, *, settings: Any = None) -> tuple[int, str, bool]:
    """``(cpus, source, allocated)``: the CPUs this process may use; ``allocated`` = they come from a
    scheduler, an affinity mask or a cgroup quota rather than the whole machine."""
    e = os.environ if env is None else env
    conf = _configured("cpus", env, settings)
    if conf is not None:
        return int(conf[0]), conf[1], True
    for name in ("SLURM_CPUS_PER_TASK", "NCPUS", "PBS_NUM_PPN"):
        raw = e.get(name)
        if raw:
            try:
                return max(1, int(raw)), f"env:{name}", True
            except ValueError:
                return 1, f"env:{name} (unparsable {raw!r})", True
    total = os.cpu_count() or 2
    if env is not None:
        return total, "os.cpu_count", False
    n, source = total, "os.cpu_count"
    try:
        aff = len(os.sched_getaffinity(0))
        if aff < n:
            n, source = aff, "sched_getaffinity"
    except (AttributeError, OSError):
        pass
    quota = cgroup_cpu_quota()
    if quota is not None and quota < n:
        n, source = quota, "cgroup cpu.max"
    return n, source, n < total


def available_cpus(env: Mapping[str, str] | None = None) -> int:
    """The CPUs this process may use (:func:`cpu_limit` without its provenance); at least 1."""
    return max(1, cpu_limit(env)[0])


def memory_limit_bytes(env: Mapping[str, str] | None = None) -> tuple[int | None, str]:
    """``(bytes | None, source)``: the memory limit of this job (SLURM, else the cgroup)."""
    e = os.environ if env is None else env
    per_node = e.get("SLURM_MEM_PER_NODE")
    if per_node:
        try:
            return int(float(per_node) * _MIB), "env:SLURM_MEM_PER_NODE"
        except ValueError:
            pass
    per_cpu = e.get("SLURM_MEM_PER_CPU")
    cpus = e.get("SLURM_CPUS_ON_NODE") or e.get("SLURM_CPUS_PER_TASK")
    if per_cpu and cpus:
        try:
            return int(float(per_cpu) * float(cpus) * _MIB), "env:SLURM_MEM_PER_CPU"
        except ValueError:
            pass
    if env is None:
        mem = _cgroup_memory()
        if mem is not None:
            return mem, "cgroup memory.max"
    return None, "none"


def _budget(env: Mapping[str, str] | None, settings: Any = None) -> tuple[int, str]:
    conf = _configured("ram_budget_gb", env, settings)
    if conf is not None:
        return max(0, int(float(conf[0]) * 1024**3)), conf[1]
    limit, source = memory_limit_bytes(env)
    if limit is None:
        return DEFAULT_RAM_BUDGET_BYTES, "default (16 GiB)"
    return min(DEFAULT_RAM_BUDGET_BYTES, int(RAM_BUDGET_FRACTION * limit)), f"min(16 GiB, {RAM_BUDGET_FRACTION} x {source})"


def ram_budget_bytes(env: Mapping[str, str] | None = None) -> int:
    """The RAM budget: ``YAVIH_RAM_BUDGET_GB`` / ``[resources] ram_budget_gb`` → ``min(16 GiB, 0.45 ×
    memory limit)`` → 16 GiB. ``ConfigError`` (a ``ValueError``) on a malformed value."""
    return _budget(env)[0]


def _workers(env: Mapping[str, str] | None, settings: Any = None) -> tuple[int, str]:
    conf = _configured("decode_workers", env, settings)
    if conf is not None:
        return max(1, min(MAX_DECODE_WORKERS, int(conf[0]))), conf[1]
    n, source, allocated = cpu_limit(env, settings=settings)
    w = n - 1 if allocated else n // 2
    return max(1, min(MAX_DECODE_WORKERS, w)), f"{'cpus - 1' if allocated else 'cpus // 2'} ({source})"


def decode_workers(env: Mapping[str, str] | None = None) -> int:
    """The default decode workers (``bridge check`` and ``extract`` must measure at the same count)."""
    return _workers(env)[0]


def describe(settings: Any = None) -> dict[str, dict[str, Any]]:
    """``{key: {value, source}}`` of the effective resources (``yavih config show``, ``doctor``)."""
    out: dict[str, dict[str, Any]] = {}
    cpus, csrc, _alloc = cpu_limit(settings=settings)
    out["cpus"] = {"value": cpus, "source": csrc}
    mem, msrc = memory_limit_bytes()
    out["memory_limit_gb"] = {"value": round(mem / 1024**3, 2) if mem is not None else None, "source": msrc}
    b, bsrc = _budget(None, settings)
    out["ram_budget_gb"] = {"value": round(b / 1024**3, 3), "source": bsrc}
    w, wsrc = _workers(None, settings)
    out["decode_workers"] = {"value": w, "source": wsrc}
    return out
