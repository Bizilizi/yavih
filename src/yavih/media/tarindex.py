"""Tar index reader: ``index.json`` ``{"version": 1, "shards": [...],
"members": {key: [shard_idx, offset, size]}}`` read with ``os.pread``.

File descriptors are cached per pid and dropped on pickle; shard paths are joined onto the index's
directory — an index holding ABSOLUTE shard paths is not relocatable and must be rewritten before
shipping.
"""

from __future__ import annotations

import json
import os
import threading

__all__ = ["TarIndex", "TAR_INDEX_VERSION", "open_tar_index", "pread_member"]

TAR_INDEX_VERSION = 1

_INDEX_CACHE: dict[str, TarIndex] = {}
_INDEX_LOCK = threading.Lock()


def open_tar_index(index_path: str, shard_root: str | None = None) -> TarIndex:
    """One ``TarIndex`` per index path per process (shared fds across the Streams of a shard)."""
    key = os.path.abspath(index_path)
    with _INDEX_LOCK:
        idx = _INDEX_CACHE.get(key)
        if idx is None:
            idx = TarIndex(index_path, shard_root)
            _INDEX_CACHE[key] = idx
        return idx


_FD_CACHE: dict[tuple[int, str], int] = {}
_FD_LOCK = threading.Lock()


def pread_member(path: str, offset: int, size: int) -> bytes:
    """``size`` bytes of ``path`` at ``offset`` with ``os.pread`` (one cached fd per (pid, path)); a short
    read raises ``EOFError`` (truncated shard or stale index). The reader's ``RawStream.bytes()`` for
    ``tar_index`` members: no decoder, no temp file."""
    key = (os.getpid(), os.path.abspath(path))
    with _FD_LOCK:
        fd = _FD_CACHE.get(key)
        if fd is None:
            fd = os.open(key[1], os.O_RDONLY)
            _FD_CACHE[key] = fd
    data = os.pread(fd, int(size), int(offset))
    if len(data) != int(size):
        raise EOFError(f"short read from '{path}' at {offset}: got {len(data)} of {size} bytes (truncated shard or stale index)")
    return data


class TarIndex:
    """Map member key → bytes without extracting; any decoder reads the bytes in place."""

    def __init__(self, index_path: str, shard_root: str | None = None) -> None:
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"tar index not found at '{index_path}'")
        with open(index_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        version = payload.get("version")
        if version != TAR_INDEX_VERSION:
            raise ValueError(f"tar index '{index_path}' has version {version!r}; this build understands {TAR_INDEX_VERSION}")
        for field in ("shards", "members"):
            if field not in payload:
                raise ValueError(f"tar index '{index_path}' is malformed: missing {field!r}")
        self.index_path = os.path.abspath(index_path)
        # Relative shard names resolve against the index's directory; an absolute shard path is
        # taken verbatim (os.path.join drops shard_root): such an index is not relocatable.
        self.shard_root = shard_root or os.path.dirname(self.index_path)
        self.shards: list[str] = [str(s) for s in payload["shards"]]
        self.members: dict[str, tuple[int, int, int]] = {
            str(k): (int(v[0]), int(v[1]), int(v[2])) for k, v in payload["members"].items()
        }
        self._fds: dict[tuple[int, int], int] = {}
        self._lock = threading.Lock()

    def shard_path(self, shard_idx: int) -> str:
        return os.path.join(self.shard_root, self.shards[shard_idx])

    def has(self, key: str) -> bool:
        return key in self.members

    def locate(self, key: str) -> tuple[str, int, int] | None:
        """``(shard path, data offset, size)`` of member ``key`` or ``None`` (the ``RawStream.member`` tuple)."""
        m = self.members.get(key)
        if m is None or not 0 <= m[0] < len(self.shards):
            return None
        return (self.shard_path(m[0]), m[1], m[2])

    def keys(self) -> list[str]:
        return list(self.members)

    def _fd(self, shard_idx: int) -> int:
        # Per-pid cache: a forked worker reopens lazily instead of sharing the parent's fd state.
        cache_key = (os.getpid(), shard_idx)
        with self._lock:
            fd = self._fds.get(cache_key)
            if fd is None:
                fd = os.open(self.shard_path(shard_idx), os.O_RDONLY)
                self._fds[cache_key] = fd
            return fd

    def read(self, key: str) -> bytes:
        try:
            shard_idx, offset, size = self.members[key]
        except KeyError:
            raise KeyError(f"{key!r} is not in tar index '{self.index_path}'") from None
        # pread takes an explicit offset: no shared file position to race on across threads/forks.
        data = os.pread(self._fd(shard_idx), size, offset)
        if len(data) != size:
            raise RuntimeError(
                f"short read for {key!r} from '{self.shards[shard_idx]}': got {len(data)} of {size} bytes "
                "(truncated shard or stale index)"
            )
        return data

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds.clear()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_fds"] = {}
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._fds = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:
        return f"TarIndex({self.index_path!r}, shards={len(self.shards)}, members={len(self.members)})"
