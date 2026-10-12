"""Protocol-hash history read from the benchmark folders (``benchmark.yaml`` ``changelog``).

Each folder's ``benchmark.yaml`` ``changelog`` records, on the entry of the version it describes, the ``protocol_hash``
and ``input_hash`` that version computes, the earlier protocol hashes it ``supersedes`` (same benchmark: their results
are this version's results, their feature sets are this version's inputs) and the earlier input hashes it
``supersedes_input`` (feature sets that stay valid). The changelog is outside every hashed payload, so recording the
digest there is not circular. There is no central table: this module scans the folders yavih can resolve — packaged,
``$YAVIH_HOME/benchmarks``, ``$YAVIH_BENCH_PATH`` and ``yavih.benchmarks`` entry-point packs — plus every folder a
``Benchmark`` was loaded from in this process (``register``), so an out-of-tree benchmark's history travels with it.

Two consumers:

- ``report.migrate`` maps a ``result.json``'s protocol hash to the newest one (:func:`upgrade_protocol_hash`), so an
  old result of an unchanged benchmark lands in the same column.
- the feature side (sidecar resumption, ``features validate``, the probe) compares a feature set's
  :func:`effective_input_hash` (the record's ``input_hash`` through ``supersedes_input``) with the loaded benchmark's
  ``input_hash``.

stdlib + PyYAML only (both envs have it); the yaml is read without full schema validation (a malformed hash is
skipped, ``bench lint`` reports it).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "HashRecord",
    "History",
    "register",
    "history",
    "reset",
    "records_from_changelog",
    "upgrade_protocol_hash",
    "effective_input_hash",
    "describe_input_mismatch",
]

_HEX = frozenset("0123456789abcdef")


def _is_hex64(x: Any) -> bool:
    return isinstance(x, str) and len(x) == 64 and set(x) <= _HEX


@dataclass(frozen=True)
class HashRecord:
    """The hash keys of one changelog entry of benchmark ``name``."""

    name: str
    version: int
    protocol_hash: str | None
    input_hash: str | None
    supersedes: tuple[str, ...] = ()
    supersedes_input: tuple[str, ...] = ()


def records_from_changelog(name: str, changelog: Iterable[Any]) -> list[HashRecord]:
    """``HashRecord``s from parsed ``ChangelogEntry`` objects or raw yaml mappings (malformed digests skipped)."""
    out: list[HashRecord] = []
    for e in changelog or ():
        get = e.get if isinstance(e, Mapping) else (lambda k, d=None, _e=e: getattr(_e, k, d))
        ph, ih = get("protocol_hash"), get("input_hash")
        sup = tuple(h for h in (get("supersedes") or ()) if _is_hex64(h))
        sup_in = tuple(h for h in (get("supersedes_input") or ()) if _is_hex64(h))
        try:
            version = int(get("version", 0) or 0)
        except (TypeError, ValueError):
            version = 0
        if not (ph or ih or sup or sup_in):
            continue
        out.append(HashRecord(name=str(name), version=version, protocol_hash=ph if _is_hex64(ph) else None,
                              input_hash=ih if _is_hex64(ih) else None, supersedes=sup, supersedes_input=sup_in))
    return out


class History:
    """The merged hash history of every known folder (see the module docstring)."""

    def __init__(self, records: Iterable[HashRecord]) -> None:
        self.records = list(records)
        self._up: dict[str, str] = {}  # old protocol hash -> the protocol hash of the entry that supersedes it
        self._up_input: dict[str, str] = {}  # old input hash -> input hash of the entry that supersedes it
        self._owner: dict[str, str] = {}  # any recorded digest -> benchmark name
        for r in self.records:
            if r.protocol_hash:
                self._owner.setdefault(r.protocol_hash, r.name)
            if r.input_hash:
                self._owner.setdefault(r.input_hash, r.name)
            for old in r.supersedes:
                self._owner.setdefault(old, r.name)
                if r.protocol_hash and old != r.protocol_hash:
                    self._up.setdefault(old, r.protocol_hash)
            for old in r.supersedes_input:
                self._owner.setdefault(old, r.name)
                if r.input_hash and old != r.input_hash:
                    self._up_input.setdefault(old, r.input_hash)

    @staticmethod
    def _follow(table: Mapping[str, str], h: str) -> str:
        seen = {h}
        while h in table:
            h = table[h]
            if h in seen:  # a cycle in hand-written yaml: stop where it closes (bench lint reports it)
                break
            seen.add(h)
        return h

    def upgrade_protocol_hash(self, ph: Any) -> Any:
        """The newest protocol hash ``ph`` is superseded by (transitively), else ``ph``."""
        return self._follow(self._up, ph) if isinstance(ph, str) else ph

    def upgrade_input_hash(self, ih: Any) -> Any:
        return self._follow(self._up_input, ih) if isinstance(ih, str) else ih

    def owner(self, h: Any) -> str | None:
        return self._owner.get(h) if isinstance(h, str) else None

    def mapping(self) -> dict[str, str]:
        """``{old protocol hash: newest protocol hash}``."""
        return {old: self.upgrade_protocol_hash(old) for old in self._up}


_LOCK = threading.RLock()
_REGISTERED: dict[str, list[HashRecord]] = {}
_SCANNED: tuple[tuple[str | None, str | None], dict[str, list[HashRecord]]] | None = None
_MERGED: tuple[Any, History] | None = None


def register(name: str, aliases: Iterable[str], changelog: Iterable[Any]) -> None:
    """Called by ``Benchmark.__init__``: the loaded folder's history wins over a scanned folder of the same name."""
    global _MERGED
    recs = records_from_changelog(name, changelog)
    with _LOCK:
        if _REGISTERED.get(name) != recs:
            _REGISTERED[name] = recs
            _MERGED = None


def reset() -> None:
    """Forget scanned and registered folders (tests)."""
    global _SCANNED, _MERGED
    with _LOCK:
        _REGISTERED.clear()
        _SCANNED = None
        _MERGED = None


def _scan_key() -> tuple[str | None, str | None]:
    return (os.environ.get("YAVIH_BENCH_PATH"), os.environ.get("YAVIH_HOME"))


def _read_folder(folder: Path) -> tuple[str, list[HashRecord]] | None:
    import yaml

    try:
        with open(folder / "benchmark.yaml", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(d, Mapping):
        return None
    name = str(d.get("name") or folder.name)
    return name, records_from_changelog(name, d.get("changelog") or ())


def _scan() -> dict[str, list[HashRecord]]:
    global _SCANNED
    key = _scan_key()
    with _LOCK:
        if _SCANNED is not None and _SCANNED[0] == key:
            return _SCANNED[1]
        found: dict[str, list[HashRecord]] = {}
        try:
            from yavih.benchmark import _candidate_folders

            folders = list(_candidate_folders().values())
        except Exception:  # noqa: BLE001 - history is best effort: an unreadable search path names nothing
            folders = []
        for folder in folders:
            got = _read_folder(Path(folder))
            if got is not None and got[1]:
                found.setdefault(got[0], got[1])
        _SCANNED = (key, found)
        return found


def history() -> History:
    """The merged history (scanned folders, overridden by registered ones); cached until either changes."""
    global _MERGED
    with _LOCK:
        scanned = _scan()
        stamp = (id(scanned), tuple(sorted((k, len(v)) for k, v in _REGISTERED.items())))
        if _MERGED is not None and _MERGED[0] == stamp:
            return _MERGED[1]
        by_name = {**scanned, **_REGISTERED}
        hist = History(r for name in sorted(by_name) for r in by_name[name])
        _MERGED = (stamp, hist)
        return hist


def upgrade_protocol_hash(ph: Any) -> Any:
    """``history().upgrade_protocol_hash(ph)``: the newest protocol hash of the same benchmark, else ``ph``."""
    return history().upgrade_protocol_hash(ph)


def effective_input_hash(block: Any) -> str | None:
    """The input hash a feature set's ``benchmark`` block (sidecar, ``_meta.json``, ``_layout.json``) stands for: its
    ``input_hash`` through ``supersedes_input``; ``None`` when the block records none."""
    if not isinstance(block, Mapping):
        return None
    ih = block.get("input_hash")
    if isinstance(ih, str) and ih:
        return history().upgrade_input_hash(ih)
    return None


def describe_input_mismatch(block: Any, *, want_input_hash: str, want_components: Mapping[str, str] | None = None,
                            writer: Any = None, reader_version: str | None = None, reader_commit: str | None = None) -> str:
    """A human sentence for ``E_HASH_INPUT``: which input components differ (when the record carries them) and both
    yavih versions."""
    b = block if isinstance(block, Mapping) else {}
    parts: list[str] = []
    ih = b.get("input_hash")
    if isinstance(ih, str) and ih:
        parts.append(f"input_hash {ih[:12]}… ≠ benchmark {want_input_hash[:12]}…")
        comps = b.get("input_components")
        if isinstance(comps, Mapping) and want_components:
            diff = sorted(k for k in set(comps) | set(want_components) if comps.get(k) != want_components.get(k))
            if diff:
                parts.append(f"differing input components: {', '.join(diff)}")
    else:
        parts.append(f"the record names no input_hash (benchmark {want_input_hash[:12]}…); re-extract")
    w = writer if isinstance(writer, Mapping) else {}
    wv, wc = w.get("version"), w.get("commit")
    parts.append(f"written by yavih {wv or '?'}{' @' + str(wc)[:12] if wc else ''}, read by yavih {reader_version or '?'}"
                 f"{' @' + str(reader_commit)[:12] if reader_commit else ''}")
    return "; ".join(parts)


def check_recorded(bench: Any) -> list[str]:
    """What a loaded ``Benchmark``'s newest ``changelog`` entry must record (``bench lint`` code
    ``E_CHANGELOG_HASH``, ``tests/test_hashing.py``): its ``protocol_hash`` and ``input_hash`` equal to the computed ones.
    Each message tells the folder author exactly what to write. Empty when the record is right."""
    log = list(getattr(getattr(bench, "spec", None), "changelog", ()) or ())
    if not log:
        return ["benchmark.yaml has no changelog entry"]
    e = log[-1]
    v = getattr(e, "version", "?")
    ph, ih = bench.protocol_hash, bench.input_hash
    rec_ph, rec_ih = getattr(e, "protocol_hash", None), getattr(e, "input_hash", None)
    out: list[str] = []
    if rec_ph is None or rec_ih is None:
        out.append(f"changelog entry version {v} (the newest) must record the hashes this folder computes; add to it:\n"
                   f"    protocol_hash: {ph}\n    input_hash: {ih}")
        return out
    if rec_ph != ph:
        out.append(f"changelog entry version {v} records protocol_hash {rec_ph} but the folder computes {ph} (a protocol edit). "
                   f"If results stay comparable (same rows, inputs and scoring semantics, e.g. a payload-schema bump or a "
                   f"refactor), set `protocol_hash: {ph}` and append `{rec_ph}` to that entry's `supersedes`; if the numbers "
                   f"can change, bump protocol_version and add a new changelog entry (invalidates: true) recording "
                   f"`protocol_hash: {ph}` and `input_hash: {ih}`.")
    if rec_ih != ih:
        out.append(f"changelog entry version {v} records input_hash {rec_ih} but the folder computes {ih} (an input-side edit: "
                   f"manifest, splits, streams, contracts, input-side protocol/readout_params keys or reader.py). Set "
                   f"`input_hash: {ih}`; append `{rec_ih}` to `supersedes_input` only if the bridge is still handed exactly "
                   f"the same thing and must produce the same output (else extracted features must be re-extracted).")
    if ph in getattr(e, "supersedes", ()):
        out.append(f"changelog entry version {v} lists its own protocol_hash {ph} under supersedes")
    if ih in getattr(e, "supersedes_input", ()):
        out.append(f"changelog entry version {v} lists its own input_hash {ih} under supersedes_input")
    return out
