"""In-memory upgrade of ``result.json`` records to the current ``RESULT_SCHEMA_VERSION``.

``migrate`` is the identity (a deep copy) for the current schema, refuses a newer one ("a reader MUST refuse a
version greater than it knows") and refuses an older one for which no upgrade step is registered. A schema bump
registers one ``_UPGRADES[v] = fn(v-record) -> (v+1)-record`` step; the chain is applied in memory only, never
written back (the on-disk record stays what the probe wrote).

Every record: ``benchmark.protocol_hash`` is mapped to the newest hash of its benchmark through the folders'
changelog ``supersedes`` lists (:func:`upgrade_protocol_hash`; any other hash is kept and stays its own column) and
``eval_hash`` is recomputed when it moved.

stdlib only."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from yavih import hash_history as _history
from yavih import hashing
from yavih.versions import RESULT_SCHEMA_VERSION

__all__ = [
    "migrate",
    "upgrade_protocol_hash",
    "PROTOCOL_HASH_UPGRADES",
    "RESULT_SCHEMA_VERSION",
    "MigrationError",
]


class _FolderUpgrades(Mapping):
    """``{old protocol hash: newest protocol hash}`` built from the benchmark folders' changelog ``supersedes`` lists
    (``yavih.hash_history``; no central table). Read-only, rebuilt when the set of folders
    changes."""

    def _map(self) -> dict[str, str]:
        return _history.history().mapping()

    def __getitem__(self, key: str) -> str:
        return self._map()[key]

    def __iter__(self):
        return iter(self._map())

    def __len__(self) -> int:
        return len(self._map())

    def __repr__(self) -> str:
        return f"PROTOCOL_HASH_UPGRADES({self._map()!r})"


#: Every earlier protocol hash a folder's changelog ``supersedes``, mapped to that folder's newest
#: one (transitively). A view over the folders, not a table: a protocol change records its old hash in its own
#: ``benchmark.yaml`` (``bench lint`` and ``tests/test_hashing.py`` fail until it does).
PROTOCOL_HASH_UPGRADES: Mapping[str, str] = _FolderUpgrades()


class MigrationError(ValueError):
    """A ``result.json`` whose ``schema_version`` this package cannot read (missing, newer, or older without a step)."""


def upgrade_protocol_hash(protocol_hash: Any) -> Any:
    """The newest protocol hash of the benchmark whose changelog supersedes ``protocol_hash``, else ``protocol_hash``
    unchanged."""
    return _history.upgrade_protocol_hash(protocol_hash)


def _v3_to_v4(record: dict[str, Any]) -> dict[str, Any]:
    """Schema 4 added ``benchmark.folder_source`` (the folder's ``probe.py`` / ``reader.py`` digests at scoring,
    provenance only). A schema-3 record did not record them: ``null``, which the report reads as unknown (never as a
    differing source). Its ``protocol_hash`` (computed under an earlier protocol schema) is moved to the current
    column by :func:`_current_hashes` through the folders' ``supersedes`` lists."""
    bench = record.get("benchmark")
    if isinstance(bench, dict):
        bench.setdefault("folder_source", None)
    return record


#: ``{from_version: upgrade}``; each step produces ``from_version + 1``. Schemas 1 and 2 are not read.
_UPGRADES: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {3: _v3_to_v4}


def migrate(result: dict) -> dict:
    """Return a ``RESULT_SCHEMA_VERSION`` record; ``MigrationError`` (a ``ValueError``) on a missing or
    malformed ``schema_version``, on a schema newer than known and on an older one without an upgrade step."""
    if not isinstance(result, dict):
        raise TypeError(f"result.json record must be a JSON object, got {type(result).__name__}")
    raw = result.get("schema_version")
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise MigrationError(f"result.json schema_version must be an int, got {raw!r}")
    version = raw
    if version > RESULT_SCHEMA_VERSION:
        raise MigrationError(
            f"result.json schema_version {version} is newer than RESULT_SCHEMA_VERSION "
            f"{RESULT_SCHEMA_VERSION}; upgrade yavih to read it"
        )
    if version < 1:
        raise MigrationError(f"result.json schema_version must be >= 1, got {version}")
    out = copy.deepcopy(result)
    while version < RESULT_SCHEMA_VERSION:
        step = _UPGRADES.get(version)
        if step is None:
            raise MigrationError(f"result.json schema_version {version} is older than this yavih reads "
                                 f"({RESULT_SCHEMA_VERSION}); re-run the probe")
        out = step(out)
        version += 1
    out["schema_version"] = RESULT_SCHEMA_VERSION
    return _current_hashes(out)


def _current_hashes(record: dict[str, Any]) -> dict[str, Any]:
    """The newest protocol hash of the benchmark and the ``eval_hash`` that follows from it (see the module
    docstring). In memory only."""
    bench = record.get("benchmark") if isinstance(record.get("benchmark"), dict) else None
    if bench is not None:
        ph = bench.get("protocol_hash")
        new = upgrade_protocol_hash(ph)
        if new != ph:
            bench["protocol_hash"] = new
            qh = (record.get("probe") or {}).get("hash") if isinstance(record.get("probe"), dict) else None
            if isinstance(new, str) and new and isinstance(qh, str) and qh:
                record["eval_hash"] = hashing.eval_hash(new, qh)
    return record
