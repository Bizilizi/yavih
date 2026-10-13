"""Resource estimation: shard count from ``clips_per_s``, the 2 GiB cap and the rows cap; extract
``minutes = ceil(1.5 * rows / clips_per_s / 60)``, ``mem_gb = 16 + 4 * cpus``; probe ``mem_gb = max(16,
1.3 * bytes_train / 2**30 + 8)``, ``scratch_gb`` when above the RAM budget, ``minutes = probe_minutes_ref
* clamp(N * D / (16 * 768), 1, 24) * seeds * fits`` (floor 30).

``fits`` is the number of full lr-grid fits one seed trains: 2 when the benchmark's
``readout_params.shuffled_control`` is set (vggss_box trains the shuffled-audio control as a second full
fit, ``readouts/box.py``), else 1; without it a vggss_box probe is planned at half its real wall time.

``multipliers`` is the ``runs/_calibration.json[multipliers]`` block (``yavih plan calibrate``):
``{"extract_minutes": f, "probe_minutes": f}``; absent keys mean 1.0.

``gpu``: 1, or 0 for a CPU-only profile — ``bridge check`` measured
``peak_gpu_gb == 0`` (torch present, no CUDA device). An unknown peak (``null``: no torch, e.g. a JAX bridge) keeps 1.
"""

from __future__ import annotations

import math

from yavih.resources import DEFAULT_RAM_BUDGET_BYTES  # the default RAM budget; the one definition

__all__ = [
    "n_shards_for",
    "shard_ranges",
    "extract_needs",
    "probe_needs",
    "bytes_per_row",
    "SHARD_BYTES_CAP",
    "MAX_ROWS_PER_SHARD",
    "EXTRACT_MARGIN",
    "PROBE_MINUTES_FLOOR",
    "EXTRACT_MINUTES_FLOOR",
    "TOKEN_REF",
    "TOKEN_SCALE_CAP",
    "DEFAULT_RAM_BUDGET_BYTES",
    "minutes_bucket",
    "gpu_for_profile",
]

SHARD_BYTES_CAP = 2 * 1024**3
#: Default rows cap per shard (matrix ``max_rows_per_shard``): bounds a unit's resident sample objects, its writer
#: index and the work a preempted unit loses, whatever the measured speed.
MAX_ROWS_PER_SHARD = 10_000
EXTRACT_MARGIN = 1.5
#: The minimum wall minutes of a probe unit.
PROBE_MINUTES_FLOOR = 30
#: An empty or tiny shard still pays interpreter + model load, so a unit never plans
#: 0 minutes.
EXTRACT_MINUTES_FLOOR = 1
#: ``N * D`` of the reference pooled ViT-B bridge the ``probe_minutes_ref`` budgets were measured at.
TOKEN_REF = 16 * 768
TOKEN_SCALE_CAP = 24.0

_ITEMSIZE = {"float16": 2, "float32": 4}
_BUCKETS = (15, 30, 60, 120, 240, 480, 720, 1440)


def bytes_per_row(N: int, D: int, dtype: str, pos_mode: str) -> int:
    """``N * D * itemsize (+ N*16 if per_sample pos)``."""
    if dtype not in _ITEMSIZE:
        raise ValueError(f"dtype must be float16|float32, got {dtype!r}")
    if N < 1 or D < 1:
        raise ValueError(f"N and D must be >= 1, got N={N} D={D}")
    n = int(N) * int(D) * _ITEMSIZE[dtype]
    if pos_mode == "per_sample":
        n += int(N) * 16  # float32 [N, 4] pos per row
    return n


def n_shards_for(rows: int, *, clips_per_s: float, shard_minutes: int, bytes_per_row: int,
                 max_rows_per_shard: int | None = MAX_ROWS_PER_SHARD) -> int:
    """``max(ceil(rows / rows_per_shard_time), ceil(rows * bytes_per_row / 2 GiB), ceil(rows / max_rows_per_shard))``
    with ``rows_per_shard_time = clips_per_s * 60 * shard_minutes``; ``max_rows_per_shard=None`` drops the rows cap.
    At least 1 (an empty split still gets one empty shard so the split directory carries a sidecar)."""
    if clips_per_s <= 0:
        raise ValueError("clips_per_s must be > 0")
    if shard_minutes < 1:
        raise ValueError("shard_minutes must be >= 1")
    if max_rows_per_shard is not None and max_rows_per_shard < 1:
        raise ValueError("max_rows_per_shard must be >= 1")
    if rows <= 0:
        return 1
    rows_per_shard_time = clips_per_s * 60.0 * shard_minutes
    by_time = math.ceil(rows / rows_per_shard_time)
    by_bytes = math.ceil(rows * bytes_per_row / SHARD_BYTES_CAP)
    by_rows = math.ceil(rows / max_rows_per_shard) if max_rows_per_shard is not None else 1
    return max(1, by_time, by_bytes, by_rows)


def shard_ranges(n_rows: int, batch_size: int, n_shards: int) -> list[tuple[int, int]]:
    """The contiguous shard ranges, via ``yavih.io.shards.plan_shards`` so the plan's ``rows_in_shard`` and the
    reader's rows can never disagree. Lazy import keeps this module's import stdlib-only."""
    from yavih.io.shards import plan_shards

    return plan_shards(n_rows, batch_size, n_shards)


def _mult(multipliers: dict | None, key: str) -> float:
    if not multipliers:
        return 1.0
    v = multipliers.get(key, 1.0)
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
        raise ValueError(f"calibration multiplier {key!r} must be a positive number, got {v!r}")
    return float(v)


def gpu_for_profile(profile: dict) -> int:
    """GPUs a unit planned from ``profile`` requests: 0 when ``bridge check`` measured ``peak_gpu_gb == 0`` (a CPU-only
    run), else 1 (module docstring)."""
    peak = profile.get("peak_gpu_gb")
    return 0 if isinstance(peak, (int, float)) and not isinstance(peak, bool) and peak == 0 else 1


def extract_needs(rows_in_shard: int, *, clips_per_s: float, cpus: int, multipliers: dict | None = None, gpu: int = 1) -> dict:
    """``minutes = ceil(1.5 * rows_in_shard / clips_per_s / 60)``, ``mem_gb = 16 + 4 * cpus``,
    ``gpu`` (1, or 0 for a CPU-only profile), ``scratch_gb = 0``. The 1.5 margin absorbs grouped manifests making some contiguous
    shards heavier than the mean ``clips_per_s`` predicts."""
    if clips_per_s <= 0:
        raise ValueError("clips_per_s must be > 0")
    if cpus < 1:
        raise ValueError("cpus must be >= 1")
    raw = EXTRACT_MARGIN * max(0, rows_in_shard) / clips_per_s / 60.0
    minutes = math.ceil(raw * _mult(multipliers, "extract_minutes"))
    return {
        "gpu": int(gpu),
        "cpus": int(cpus),
        "mem_gb": 16 + 4 * int(cpus),
        "scratch_gb": 0,
        "minutes": max(EXTRACT_MINUTES_FLOOR, int(minutes)),
    }


def probe_needs(*, bytes_train: int, N: int, D: int, probe_minutes_ref: float, seeds: int,
                ram_budget_bytes: int, multipliers: dict | None = None, cpus: int = 8, fits: int = 1, gpu: int = 1) -> dict:
    """``mem_gb = max(16, 1.3 * bytes_train / 2**30 + 8)``; ``scratch_gb = 1.1 * bytes_train /
    2**30`` only when ``bytes_train`` exceeds the RAM budget (the probe then stages to ``YAVIH_SCRATCH``);
    ``minutes = probe_minutes_ref * clamp(N * D / (16 * 768), 1, 24) * seeds * fits``, floor 30.
    ``seeds`` is the number of seeds *this unit* trains (one per unit in the MVP plan); ``fits`` the full
    fits per seed (2 with a shuffled control, module docstring)."""
    if N < 1 or D < 1:
        raise ValueError("N and D must be >= 1")
    if seeds < 1:
        raise ValueError("seeds must be >= 1")
    if fits < 1:
        raise ValueError("fits must be >= 1")
    if probe_minutes_ref <= 0:
        raise ValueError("probe_minutes_ref must be > 0 (benchmark.yaml budget.probe_minutes_ref)")
    gib = bytes_train / 2**30
    mem_gb = max(16.0, 1.3 * gib + 8.0)
    scratch_gb = 1.1 * gib if bytes_train > ram_budget_bytes else 0.0
    scale = min(TOKEN_SCALE_CAP, max(1.0, (N * D) / TOKEN_REF))
    minutes = probe_minutes_ref * scale * seeds * int(fits) * _mult(multipliers, "probe_minutes")
    return {
        "gpu": int(gpu),
        "cpus": int(cpus),
        "mem_gb": int(math.ceil(mem_gb)),
        "scratch_gb": int(math.ceil(scratch_gb)) if scratch_gb else 0,
        "minutes": max(PROBE_MINUTES_FLOOR, int(math.ceil(minutes))),
    }


def minutes_bucket(minutes: int | float) -> int:
    """Coarse wall bucket for the plan's ``(kind, needs.gpu, needs.minutes bucket)`` grouping: the first
    of 15/30/60/120/240/480/720/1440 that holds ``minutes``, else ``minutes`` itself."""
    for b in _BUCKETS:
        if minutes <= b:
            return b
    return int(math.ceil(minutes))
