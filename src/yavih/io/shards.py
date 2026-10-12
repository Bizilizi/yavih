"""Shard planning.

Ranges are contiguous on purpose (one shard = one independently scheduled array task;
contiguity keeps tar-index page-cache locality); the scheduler absorbs a heavy shard.
"""

from __future__ import annotations

import bisect

__all__ = ["plan_shards", "shard_of_row", "n_batches"]


def n_batches(n_rows: int, batch_size: int) -> int:
    """``ceil(n_rows / batch_size)``: the number of extraction batches of a split."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")
    return -(-n_rows // batch_size)


def plan_shards(n_rows: int, batch_size: int, n_shards: int) -> list[tuple[int, int]]:
    """``n_shards`` half-open ``[lo, hi)`` ranges, ascending, disjoint, tiling ``[0, n_rows)``, every ``lo``
    a multiple of ``batch_size``, batches dealt as evenly as possible (``b_lo = (i * n_batches) // n_shards``),
    the short final batch in the last non-empty shard; empty ranges allowed when ``n_shards > n_batches``.

    Batch alignment guarantees bit identity: global batch
    ``j`` is always rows ``[j*B, (j+1)*B)``, so any shard count replays exactly the same batches and the
    encoder (which is NOT batch-invariant) sees identical inputs.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    nb = n_batches(n_rows, batch_size)
    out: list[tuple[int, int]] = []
    for i in range(n_shards):
        b_lo = (i * nb) // n_shards
        b_hi = ((i + 1) * nb) // n_shards
        lo = b_lo * batch_size
        # min() puts the single short final batch into whichever shard holds the last batch; every
        # other boundary is a multiple of batch_size by construction.
        hi = min(b_hi * batch_size, n_rows)
        if b_lo == b_hi:
            hi = lo  # empty shard: a sidecar with no tar members
        out.append((lo, hi))
    return out


def shard_of_row(row: int, ranges: list[tuple[int, int]]) -> int:
    """Index of the range holding ``row`` (helper for validate/reader); ``ranges`` as returned by
    ``plan_shards`` (ascending). ``IndexError`` when no range contains ``row`` (empty ranges never do)."""
    if row < 0:
        raise IndexError(f"row {row} < 0")
    his = [hi for _, hi in ranges]
    # The first range whose hi exceeds row is the only candidate (ranges are ascending and disjoint;
    # empty ranges have hi == lo <= row and are skipped by the strict comparison).
    i = bisect.bisect_right(his, row)
    if i < len(ranges) and ranges[i][0] <= row < ranges[i][1]:
        return i
    raise IndexError(f"row {row} is in no shard range of {len(ranges)} ranges")
