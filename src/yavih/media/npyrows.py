"""Rows of a ``.npy`` array, read in place (``layout: npy_rows``).

A dataset that ships its samples as one array (``[n, ...]``: point clouds of a fixed size, spectrograms, embeddings,
sensor windows) is frozen as manifest cells ``<file relative to the root>#<row>``. The reader memory-maps the file and
hands the row over as the bytes of a ``.npy`` file holding that row alone (``numpy.load(io.BytesIO(raw.bytes()))``);
presence is the file's header (``row < shape[0]``); the media fingerprint reads the row's own bytes in the file.
Object arrays are refused (``allow_pickle=False``). numpy only.
"""

from __future__ import annotations

import io
import os
import threading
from typing import Any

import numpy as np

__all__ = ["open_array", "has_row", "read_row", "row_location"]

_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()


def open_array(path: str) -> np.ndarray:
    """A read-only memory map of ``path`` (one per path per process)."""
    key = os.path.abspath(path)
    with _LOCK:
        a = _CACHE.get(key)
        if a is None:
            a = np.load(key, mmap_mode="r", allow_pickle=False)
            if a.dtype.hasobject or a.ndim < 1:
                raise ValueError(f"{path}: an npy_rows file holds a numeric array of at least one axis, got {a.dtype} {a.shape}")
            _CACHE[key] = a
        return a


def has_row(path: str, row: int) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        return 0 <= row < open_array(path).shape[0]
    except (OSError, ValueError):
        return False


def read_row(path: str, row: int) -> bytes:
    """The ``.npy`` bytes of ``array[row]``."""
    a = open_array(path)
    if not 0 <= row < a.shape[0]:
        raise IndexError(f"{path}: row {row} out of range ({a.shape[0]} rows)")
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a[row]), allow_pickle=False)
    return buf.getvalue()


def row_location(path: str, row: int) -> tuple[str, int, int]:
    """``(path, offset, size)`` of ``array[row]``'s bytes in the file (C-order rows are contiguous)."""
    a = open_array(path)
    if not a.flags.c_contiguous:
        raise ValueError(f"{path}: npy_rows needs a C-order array")
    row_bytes = int(a.itemsize * int(np.prod(a.shape[1:], dtype=np.int64)))
    return path, int(a.offset) + row * row_bytes, row_bytes
