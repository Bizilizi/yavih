"""Parquet cells: one field's value of one row of a Parquet file, read in place (``layout: parquet``).

A Hugging Face dataset stores its media inside Parquet files: an ``Audio`` or ``Image`` feature is a struct column
``{bytes, path}``, a binary column holds raw bytes. A benchmark folder freezes the rows it scores as manifest cells
``<file relative to the root>#<row within the file>`` and names the column (and, for a struct, its ``bytes`` field) in
``layout_params``; the reader decodes only that column of only the row group holding the row. Presence is the file's
metadata (row count); the media fingerprint reads the row group's column chunk (``chunk_location``).

pyarrow is the optional ``hf`` extra, imported on first use.
"""

from __future__ import annotations

import os
import threading
from typing import Any

__all__ = ["ParquetMissing", "parse_cell", "open_parquet", "read_cell", "has_row", "chunk_location"]


class ParquetMissing(ImportError):
    """pyarrow is not installed (``pip install 'yavih[hf]'``)."""


_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()


def parse_cell(cell: str) -> tuple[str, int]:
    """``"data/test-00001.parquet#17"`` → ``("data/test-00001.parquet", 17)``."""
    path, sep, row = str(cell).rpartition("#")
    if not sep or not path or not row.isdigit():
        raise ValueError(f"a parquet cell is '<file>#<row>', got {cell!r}")
    return path, int(row)


class _File:
    """A Parquet file's metadata and row-group boundaries (one per path per process)."""

    def __init__(self, path: str) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover - the [hf] extra
            raise ParquetMissing("layout: parquet needs pyarrow: pip install 'yavih[hf]'") from e
        self.path = path
        self.file = pq.ParquetFile(path)
        md = self.file.metadata
        self.num_rows = int(md.num_rows)
        self.starts: list[int] = []
        n = 0
        for i in range(md.num_row_groups):
            self.starts.append(n)
            n += int(md.row_group(i).num_rows)
        self.columns = list(self.file.schema_arrow.names)
        self.lock = threading.Lock()

    def locate(self, row: int) -> tuple[int, int]:
        """``(row group, row within it)``."""
        if not 0 <= row < self.num_rows:
            raise IndexError(f"{self.path}: row {row} out of range ({self.num_rows} rows)")
        lo, hi = 0, len(self.starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.starts[mid] <= row:
                lo = mid
            else:
                hi = mid - 1
        return lo, row - self.starts[lo]


def open_parquet(path: str) -> _File:
    key = os.path.abspath(path)
    with _LOCK:
        f = _CACHE.get(key)
        if f is None:
            f = _File(key)
            _CACHE[key] = f
        return f


def has_row(path: str, row: int) -> bool:
    """The file exists and holds ``row`` (metadata only)."""
    if not os.path.isfile(path):
        return False
    try:
        return 0 <= row < open_parquet(path).num_rows
    except (OSError, ValueError):
        return False


def read_cell(path: str, row: int, column: str, struct_field: str | None = None) -> tuple[bytes, str | None]:
    """The cell's bytes and, for a ``{bytes, path}`` struct, the stored file name's extension. Reads ``column`` of
    the one row group holding ``row``; a null value raises ``ValueError``."""
    f = open_parquet(path)
    if column not in f.columns:
        raise KeyError(f"{path}: no column {column!r} (columns {f.columns})")
    rg, i = f.locate(row)
    with f.lock:
        table = f.file.read_row_group(rg, columns=[column])
    value = table.column(column)[i].as_py()
    ext = None
    if isinstance(value, dict):
        name = value.get("path")
        if isinstance(name, str) and "." in name.rsplit("/", 1)[-1]:
            ext = name.rsplit(".", 1)[-1].lower()
        value = value.get(struct_field or "bytes")
    if value is None:
        raise ValueError(f"{path}#{row}: {column}{'.' + struct_field if struct_field else ''} is null")
    if isinstance(value, str):
        value = value.encode("utf-8")
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"{path}#{row}: {column} holds {type(value).__name__}, not bytes (name the struct field in layout_params)")
    return bytes(value), ext


def chunk_location(path: str, row: int, column: str) -> tuple[str, int, int]:
    """``(path, offset, size)`` of the column chunk holding ``row`` (the media fingerprint reads it in place)."""
    f = open_parquet(path)
    rg, _ = f.locate(row)
    md = f.file.metadata.row_group(rg)
    names = [md.column(j).path_in_schema for j in range(md.num_columns)]
    hits = [j for j, n in enumerate(names) if n == column or n.startswith(column + ".")]
    if not hits:
        raise KeyError(f"{path}: no column chunk for {column!r}")
    cols = [md.column(j) for j in hits]
    start = min(int(c.dictionary_page_offset or c.data_page_offset) for c in cols)
    end = max(int(c.dictionary_page_offset or c.data_page_offset) + int(c.total_compressed_size) for c in cols)
    return path, start, end - start
