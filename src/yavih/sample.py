"""``RawStream``, ``Sample``; the generic reader ``iter_samples``;
canaries; the byte read-ahead; the site media-info table.

Core module: numpy + stdlib (+ ``yavih.media.tarindex``). The reader never decodes: it resolves each
row's media to its stored bytes and attaches facts. ``RawStream.info`` comes from the site's media-info
table (``bench verify --census``) when present, else from a lazy header probe by
``yavih.media.info.probe_info`` (PyAV / soundfile / pillow, imported only then; ``DecoderMissing`` when
absent). Nothing here decodes a sample, cuts a window, crops, selects frames, pads or computes a statistic.
"""

from __future__ import annotations

import atexit
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

# the reader's error; yavih.media is numpy + stdlib only
from yavih.media import MediaMissing
from yavih.resources import MAX_DECODE_WORKERS  # per task; the one definition

__all__ = [
    "MediaMissing",
    "RawStream",
    "Sample",
    "STREAM_KINDS",
    "LAYOUTS",
    "MEDIA_INFO_KEYS",
    "INFO_VOCAB",
    "TRACK_FACTS",
    "MediaInfoTable",
    "media_info_path",
    "build_sample",
    "iter_samples",
    "canary_rows",
    "batches",
    "fetch",
    "rehandle",
    "ReadAhead",
    "ReadBatch",
    "default_decode_workers",
    "MAX_DECODE_WORKERS",
]

STREAM_KINDS: frozenset[str] = frozenset({"video", "audio", "image"})
LAYOUTS: frozenset[str] = frozenset({"dir", "tar_index"})
#: Every ``RawStream.info`` carries these keys (null where unknown or not applicable).
#: ``has_track`` (video, audio): ``True`` when the stored file holds the track the stream names, ``False`` when it
#: holds none (the row is still handed over; the track's facts are then null), null for image streams.
MEDIA_INFO_KEYS: tuple[str, ...] = ("duration_s", "fps", "n_frames", "sample_rate", "n_samples", "channels", "width", "height", "container", "bytes",
                                    "has_track")
#: Keys ``input_contract.info[stream]`` may guarantee, per stream kind. ``rotation`` is
#: recorded for video when tagged but can never be guaranteed.
INFO_VOCAB: Mapping[str, frozenset[str]] = {
    "video": frozenset({"duration_s", "fps", "n_frames", "width", "height", "container", "bytes"}),
    "audio": frozenset({"duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"}),
    "image": frozenset({"width", "height", "container", "bytes"}),
}
#: Facts of a stream's TRACK (null when ``has_track`` is false: a file without that track); ``container``
#: and ``bytes`` describe the file and stay set.
TRACK_FACTS: Mapping[str, frozenset[str]] = {k: v - {"container", "bytes"} for k, v in INFO_VOCAB.items() if k in ("video", "audio")}


def _missing(clip_id: str, stream: str, reason: str) -> Exception:
    return MediaMissing(clip_id, stream, reason)


# ----------------------------------------------------------------------------- byte sources


class _Source:
    """The stored bytes behind one media cell, read once and shared by every RawStream of the row that names
    the same location (AVE / avsync: ``video`` and ``audio`` are two tracks of one mp4).

    ``loc``: ``("file", path)`` | ``("member", tar_path, offset, size)`` | ``("absent", reason)``.
    """

    __slots__ = ("loc", "_data", "_lock")

    def __init__(self, loc: tuple) -> None:
        self.loc = loc
        self._data: bytes | None = None
        self._lock = threading.Lock()

    @property
    def size(self) -> int | None:
        if self.loc[0] == "member":
            return int(self.loc[3])
        if self.loc[0] == "file":
            try:
                return os.stat(self.loc[1]).st_size
            except OSError:
                return None
        return None

    @property
    def cached(self) -> bool:
        return self._data is not None

    def release(self) -> None:
        """Drop the cached bytes; a later ``read`` reads the store again. For a long-lived holder of many sources
        (``write_census`` keeps one RawStream per media cell until the table is written): without it every
        probed file stays resident (a census over tens of thousands of files would hold them all in memory)."""
        with self._lock:
            self._data = None

    def read(self, clip_id: str, stream: str) -> bytes:
        if self._data is not None:
            return self._data
        with self._lock:
            if self._data is not None:
                return self._data
            kind = self.loc[0]
            if kind == "absent":
                raise _missing(clip_id, stream, str(self.loc[1]))
            if kind == "file":
                path = self.loc[1]
                try:
                    with open(path, "rb") as fh:
                        expected = os.fstat(fh.fileno()).st_size
                        data = fh.read()
                except FileNotFoundError:
                    raise _missing(clip_id, stream, f"absent file {path}") from None
                except OSError as exc:
                    raise _missing(clip_id, stream, f"unreadable file {path}: {exc}") from exc
                if len(data) != expected:  # pragma: no cover - racing writer
                    raise _missing(clip_id, stream, f"short read of {path}: {len(data)} of {expected} bytes")
            else:
                from yavih.media.tarindex import pread_member

                _, tar_path, offset, size = self.loc
                try:
                    data = pread_member(tar_path, offset, size)
                except FileNotFoundError:
                    raise _missing(clip_id, stream, f"absent tar shard {tar_path}") from None
                except (OSError, EOFError) as exc:
                    raise _missing(clip_id, stream, f"short read of member at {tar_path}:{offset}: {exc}") from exc
            self._data = data
            return data

    def __getstate__(self) -> dict:
        return {"loc": self.loc, "data": self._data}

    def __setstate__(self, state: dict) -> None:
        self.loc = state["loc"]
        self._data = state["data"]
        self._lock = threading.Lock()


# ----------------------------------------------------------------------------- opaque paths
#
# ``RawStream.path`` is an opaque handle: a symlink, in a per-process temporary directory (sharded by the first two
# characters of the name, so no directory grows to a whole split's worth of entries), to the stored file, named by a hash. The stored path itself can carry the label (k400 ``val/<class>/…``, in1k ``val/<wnid>/…``,
# ESC-50's ``…-<target>.wav``) and the split, and the bridge sees media only. Each Sample's
# streams carry a salt (``_handle``: manifest hash, split, row; canaries and every bridge-check pass get their own;
# the link name hashes the salt with the stored path, so two streams of one file share one handle), so a cache
# keyed on the path misses across rows, canaries and passes.

_LINK_DIR: str | None = None
_LINK_LOCK = threading.Lock()


def _link_dir() -> str:
    """The per-process directory holding the opaque symlinks (created on first use, removed at exit)."""
    global _LINK_DIR
    d = _LINK_DIR
    if d is not None and os.path.isdir(d):
        return d
    with _LINK_LOCK:
        if _LINK_DIR is None or not os.path.isdir(_LINK_DIR):
            _LINK_DIR = tempfile.mkdtemp(prefix="yavih-media-")
            atexit.register(shutil.rmtree, _LINK_DIR, True)
        return _LINK_DIR


def _opaque_name(handle: str, real: str, ext: str | None) -> str:
    name = hashlib.sha256(f"{handle}\0{real}".encode()).hexdigest()[:24]
    return f"{name}.{ext}" if ext else name


def _opaque_link(handle: str, real: str, ext: str | None) -> str:
    """``<link dir>/<name[:2]>/<name>`` with ``name = <sha256(handle, real)[:24]>.<ext>`` → ``real`` (created if
    absent; idempotent, race-safe)."""
    name = _opaque_name(handle, real, ext)
    link = os.path.join(_link_dir(), name[:2], name)
    if os.path.islink(link):
        return link
    os.makedirs(os.path.dirname(link), exist_ok=True)
    try:
        os.symlink(real, link)
    except FileExistsError:
        pass
    return link


# ----------------------------------------------------------------------------- RawStream

#: Video containers whose header carries no frame count: ``n_frames`` needs a demuxing probe.
_NO_FRAME_COUNT_EXTS: frozenset[str] = frozenset({"webm", "mkv"})


class RawStream:
    """One stream of a Sample as stored. Read-only.

    ``path``: an OPAQUE absolute path to the stored file (layout ``dir``) — a symlink named by a hash, so the
    stored path's directories and name (which can carry the label or the split) never reach the bridge — else
    ``None``. ``member``: ``(tar path, data offset, size)`` for layout ``tar_index``, else ``None``. ``info``: the media facts, a fresh dict on every access; the keys
    ``input_contract.info[name]`` lists are non-null (a row whose header cannot supply one is a missing row),
    except on a stream whose file holds no such track (``has_track: false``): it is handed over all the same,
    its track facts null and ``container`` / ``bytes`` set, and the bridge decides what to feed for it.
    ``bytes()`` / ``open()`` give the stored bytes.
    """

    __slots__ = ("_name", "_kind", "_path", "_member", "_src", "_clip_id", "_info", "_info_entry", "_required", "_ext", "_handle", "_info_lock")

    def __init__(
        self,
        name: str,
        kind: str,
        *,
        src: _Source | None = None,
        clip_id: str = "?",
        info: Mapping[str, Any] | None = None,
        info_entry: Mapping[str, Any] | None = None,
        required_info: Sequence[str] = (),
        ext: str | None = None,
        handle: str = "",
    ) -> None:
        if kind not in STREAM_KINDS:
            raise ValueError(f"RawStream.kind must be one of {sorted(STREAM_KINDS)}, got {kind!r}")
        if src is None:
            raise ValueError("RawStream: a stream needs a byte source")
        self._name = str(name)
        self._kind = kind
        self._src = src
        loc = src.loc
        self._path = str(loc[1]) if loc[0] == "file" else None
        self._member = (str(loc[1]), int(loc[2]), int(loc[3])) if loc[0] == "member" else None
        self._clip_id = str(clip_id)
        self._info = dict(info) if info is not None else None
        self._info_entry = dict(info_entry) if info_entry is not None else None
        self._required = tuple(required_info)
        self._ext = ext
        self._handle = str(handle)
        self._info_lock = threading.Lock()

    # -- read-only fields ----------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def path(self) -> str | None:
        """An opaque absolute path to the stored file (a hash-named symlink), or ``None``."""
        if self._path is None:
            return None
        return _opaque_link(self._handle, self._path, self._ext or _ext(self._path))

    @property
    def _real_path(self) -> str | None:
        """The stored path (yavih-internal: census, verify, tests). A bridge MUST NOT read it."""
        return self._path

    def with_handle(self, handle: str) -> RawStream:
        """The same stream (same byte source, cached bytes and facts) under another opaque ``path``."""
        new = object.__new__(RawStream)
        for k in self.__slots__:
            if k != "_info_lock":
                object.__setattr__(new, k, getattr(self, k))
        object.__setattr__(new, "_handle", str(handle))
        object.__setattr__(new, "_info_lock", threading.Lock())
        return new

    @property
    def member(self) -> tuple[str, int, int] | None:
        return self._member

    @property
    def info(self) -> dict:
        """The media facts as a fresh dict: census entry, else a lazy header probe."""
        if self._info is None:
            with self._info_lock:
                if self._info is None:
                    self._info = self._resolve_info()
        return dict(self._info)

    # -- bytes --------------------------------------------------------------------
    def bytes(self) -> bytes:
        """The stored file, byte for byte (tar member: the member's bytes); cached after the first read and
        shared with any other stream of the row that names the same file."""
        return self._src.read(self._clip_id, self._name)

    def open(self) -> BinaryIO:
        """Read-only, seekable file object over ``bytes()`` (PyAV, soundfile and pillow accept one)."""
        return io.BytesIO(self.bytes())

    @property
    def nbytes(self) -> int | None:
        """Stored size without reading."""
        return self._src.size

    # -- facts --------------------------------------------------------------------
    def _resolve_info(self) -> dict:
        """Census entry, else a header probe. A probe failure is a missing row only when a guaranteed key is left null. A file that
        holds no track of this stream's kind (``has_track: false``) is NOT missing: its track facts are null
        whatever the contract guarantees, and the row is handed over (the bridge decides)."""
        from yavih.media import DecodeError, DecoderMissing

        info = {k: None for k in MEDIA_INFO_KEYS}
        if self._info_entry is not None:
            info.update(self._info_entry)
        else:
            info["bytes"] = self.nbytes
            # a guaranteed frame count of a container whose header never carries one (webm / mkv): one demuxing
            # probe instead of a header probe followed by a demuxing one
            demux = self._kind == "video" and "n_frames" in self._required and (self._ext or "").lower() in _NO_FRAME_COUNT_EXTS
            try:
                info.update(self._probe(demux=demux))
                missing = [k for k in self._required if info.get(k) is None]
                if not demux and missing and self._kind == "video" and "n_frames" in missing and info.get("has_track") is not False:
                    info.update({k: v for k, v in self._probe(demux=True).items() if v is not None})
            except (DecodeError, DecoderMissing):
                if self._required:
                    raise
        trackless = TRACK_FACTS.get(self._kind, frozenset()) if info.get("has_track") is False else frozenset()
        missing = [k for k in self._required if info.get(k) is None and k not in trackless]
        if missing:
            raise _missing(self._clip_id, self._name, f"media fact(s) {missing} are null but the input contract guarantees them (header probe; run `yavih bench verify --census`)")
        return info

    def _probe(self, *, demux: bool) -> dict:
        from yavih.media import DecodeError
        from yavih.media.info import probe_info

        try:
            out = probe_info(self.bytes(), self._kind, demux=demux, ext=self._ext)
            return {k: v for k, v in out.items() if v is not None}
        except DecodeError as exc:
            if not self._required:
                raise
            raise _missing(self._clip_id, self._name, f"unreadable header: {exc}") from exc

    # -- identity / pickling ----------------------------------------------------------
    def _identity(self) -> tuple:
        return (self._name, self._kind, self._src.loc, self._handle)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RawStream):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def __getstate__(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__ if k != "_info_lock"}

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            object.__setattr__(self, k, v)
        self._info_lock = threading.Lock()

    def __setattr__(self, key: str, value: Any) -> None:
        if key != "_info" and hasattr(self, "_info_lock") and key in self.__slots__:
            raise AttributeError(f"RawStream is read-only: cannot set {key!r}")
        object.__setattr__(self, key, value)

    def __repr__(self) -> str:
        if self._path is not None:
            where = _opaque_name(self._handle, self._path, self._ext or _ext(self._path))
        else:
            where = f"{self._member[0]}@{self._member[1]}" if self._member else "absent"
        return f"RawStream(name={self._name!r}, kind={self._kind!r}, {where})"


@dataclass(frozen=True)
class Sample:
    """One manifest row as the bridge sees it: stored media and facts, never a label.

    ``meta`` is ``{"facts": {key: protocol[key] for key in input_contract.facts}}``. ``_canary_of`` is private:
    a bridge MUST NOT read it, and it is excluded from equality, so a canary equals its original.
    ``prepared`` is what ``Bridge.prepare`` returned for this sample, set by the extractor before ``encode``
    (``None`` for a bridge without ``prepare``); not an init argument, excluded from equality and repr, and not
    carried over by ``rehandle``.
    """

    row: int
    clip_id: str
    split: str
    streams: dict[str, RawStream]
    meta: dict = field(default_factory=dict)
    _canary_of: int | None = field(default=None, repr=False, compare=False)
    prepared: Any = field(default=None, init=False, repr=False, compare=False)


# ----------------------------------------------------------------------------- media-info table


def media_info_path(name: str, manifest_hash: str) -> Path:
    """``$YAVIH_HOME/media_info/<name>-<manifest_hash12>.jsonl.gz``."""
    from yavih import config

    return config.yavih_home() / "media_info" / f"{name}-{manifest_hash[:12]}.jsonl.gz"


class MediaInfoTable:
    """The site census: one JSON line per distinct ``(stream, media cell)``,
    ``{"stream", "key", "info"}``, gzipped. ``sha256`` is over the decompressed bytes (like the manifest)."""

    def __init__(self, entries: Mapping[tuple[str, str], Mapping[str, Any]], *, path: str | None = None, sha256: str | None = None) -> None:
        self._entries = {k: dict(v) for k, v in entries.items()}
        self.path = path
        self.sha256 = sha256

    @classmethod
    def load(cls, path: str | os.PathLike) -> MediaInfoTable:
        p = Path(path)
        with gzip.open(p, "rb") as fh:
            raw = fh.read()
        entries: dict[tuple[str, str], dict[str, Any]] = {}
        for n, line in enumerate(raw.decode("utf-8").splitlines()):
            if not line.strip():
                continue
            d = json.loads(line)
            if not isinstance(d, dict) or set(d) != {"stream", "key", "info"} or not isinstance(d["info"], dict):
                raise ValueError(f"{p}: line {n} is not {{stream, key, info}}")
            entries[(str(d["stream"]), str(d["key"]))] = d["info"]
        return cls(entries, path=str(p), sha256=hashlib.sha256(raw).hexdigest())

    @staticmethod
    def encode(entries: Mapping[tuple[str, str], Mapping[str, Any]]) -> bytes:
        """The decompressed table bytes: lines sorted by ``(stream, key)``, canonical JSON."""
        lines = [
            json.dumps({"stream": s, "key": k, "info": dict(entries[(s, k)])}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for s, k in sorted(entries)
        ]
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    @classmethod
    def write(cls, path: str | os.PathLike, entries: Mapping[tuple[str, str], Mapping[str, Any]]) -> MediaInfoTable:
        """Write atomically (gzip ``mtime=0``, so an identical census is byte-identical)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = cls.encode(entries)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        with open(tmp, "wb") as fh_raw, gzip.GzipFile(fileobj=fh_raw, mode="wb", mtime=0) as fh:
            fh.write(raw)
        os.replace(tmp, p)
        return cls(entries, path=str(p), sha256=hashlib.sha256(raw).hexdigest())

    def get(self, stream: str, key: str) -> dict | None:
        e = self._entries.get((stream, key))
        return dict(e) if e is not None else None

    def __contains__(self, item: object) -> bool:
        return item in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"MediaInfoTable({self.path!r}, rows={len(self)})"


# ----------------------------------------------------------------------------- generic reader


def _spec_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _roles(spec: Any) -> dict[str, list[str]]:
    splits = _spec_get(spec, "splits")
    out: dict[str, list[str]] = {}
    for role in ("train", "eval", "floor", "folds", "unused"):
        vals = _spec_get(splits, role)
        if vals:
            out[role] = list(vals)
    return out


def _per_split(value: Any, split: str, roles: Mapping[str, Sequence[str]], *, mapping_values: bool = False) -> Any:
    """Resolve a ``x | dict[split-or-role, x]`` field for ``split`` (``streams`` fields)."""
    if not isinstance(value, Mapping) or not value:
        return value
    keys = set(roles) | {v for vs in roles.values() for v in vs}
    if not set(value) <= keys:
        return value
    if mapping_values and not all(isinstance(v, Mapping) for v in value.values()):
        return value
    if split in value:
        return value[split]
    for role, vals in roles.items():
        if split in vals and role in value:
            return value[role]
    raise KeyError(f"per-split value {sorted(value)} has no entry for split {split!r}")


def _resolve_roots(bench: Any, roots: Mapping[str, str] | None) -> dict[str, str]:
    if roots is not None:
        return dict(roots)
    fn = getattr(bench, "resolved_roots", None)
    if callable(fn):
        return dict(fn())
    from yavih.config import resolve_data

    data = _spec_get(bench.spec, "data")
    raw_roots = _spec_get(data, "roots") or {}
    roots_dict = {k: ({"default": _spec_get(v, "default")} if not isinstance(v, str) else v) for k, v in raw_roots.items()}
    return resolve_data(bench.name, roots_dict, getattr(bench, "data_overrides", None))


def _ext(key: str) -> str | None:
    base = key.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else None


class _StreamCtx:
    """One stream's per-split reading context (layout, root, tar index, pts …), resolved once per split."""

    __slots__ = ("name", "kind", "layout", "root", "params", "required", "info_keys", "tar", "tar_error", "pts", "n_members", "pattern", "nominal")

    def __init__(self, bench: Any, split: str, name: str, sspec: Any, roots: Mapping[str, str], roles: Mapping[str, Sequence[str]], info_keys: Sequence[str]) -> None:
        self.name = name
        self.kind = _spec_get(sspec, "kind")
        self.layout = _per_split(_spec_get(sspec, "layout", "dir"), split, roles)
        root_name = _per_split(_spec_get(sspec, "root"), split, roles)
        if root_name not in roots:
            raise KeyError(f"stream {name!r} names root {root_name!r}, not in resolved roots {sorted(roots)}")
        self.root = str(roots[root_name])
        self.params = dict(_per_split(_spec_get(sspec, "layout_params") or {}, split, roles, mapping_values=True) or {})
        self.required = bool(_spec_get(sspec, "required", True))
        self.info_keys = tuple(info_keys)
        if self.layout not in LAYOUTS:
            raise ValueError(f"unknown layout {self.layout!r} for stream {name!r} (layouts: {sorted(LAYOUTS)})")
        self.tar = None
        self.tar_error: str | None = None
        if self.layout == "tar_index":
            rel = _spec_get(_spec_get(bench.spec, "data"), "tar_index")
            if not rel:
                raise ValueError(f"layout {self.layout!r} needs benchmark.yaml data.tar_index")
            index_path = rel if os.path.isabs(rel) else os.path.join(self.root, rel)
            from yavih.media.tarindex import open_tar_index

            try:
                self.tar = open_tar_index(index_path)
            except FileNotFoundError:
                self.tar_error = f"tar index {index_path} is missing"

    def source(self, cell: str) -> _Source:
        if self.layout == "dir":
            return _Source(("file", os.path.join(self.root, cell)))
        return self.member_source(cell)

    def member_source(self, key: str) -> _Source:
        if self.tar is None:
            return _Source(("absent", self.tar_error or "no tar index"))
        loc = self.tar.locate(key)
        if loc is None:
            return _Source(("absent", f"no index entry {key!r} in {self.tar.index_path}"))
        return _Source(("member", loc[0], int(loc[1]), int(loc[2])))

    def present(self, cell: str) -> bool:
        """Cheap presence (stat / index lookup), for ``required: false`` streams only."""
        if self.layout == "dir":
            return os.path.isfile(os.path.join(self.root, cell))
        return self.tar is not None and self.tar.has(cell)


class _SplitCtx:
    """Everything ``build_sample`` needs for one split, resolved once."""

    def __init__(self, bench: Any, split: str, roots: Mapping[str, str], info_table: MediaInfoTable | None) -> None:
        spec = bench.spec
        self.split = split
        roles = _roles(spec)
        ic = _spec_get(spec, "input_contract")
        info_req = _spec_get(ic, "info") or {}
        facts_keys = list(_spec_get(ic, "facts") or [])
        protocol = _spec_get(spec, "protocol") or {}
        self.facts = json.loads(json.dumps({k: protocol[k] for k in facts_keys}))  # a private deep copy
        self.streams = {
            name: _StreamCtx(bench, split, name, sspec, roots, roles, info_req.get(name, ()))
            for name, sspec in _spec_get(spec, "streams").items()
        }
        self.info_table = info_table


def build_sample(bench: Any, split: str, manifest: Any, i: int, roots: Mapping[str, str] | None = None, *, ctx: _SplitCtx | None = None, canary_of: int | None = None) -> Sample:
    """The ``Sample`` of row ``i`` (0-based within ``split``) of a normalized ``Manifest``.

    Never touches the media bytes: missing media surfaces as ``MediaMissing`` from ``bytes()`` / ``info``
    (the read-ahead does both before a bridge sees the row). A ``required: false`` stream whose cell is empty
    or whose media is absent is omitted from ``Sample.streams``."""
    if ctx is None:
        ctx = _SplitCtx(bench, split, _resolve_roots(bench, roots), _media_info_of(bench))
    clip_id = str(manifest.clip_id[i])
    cells = manifest.media[i]
    sources: dict[tuple, _Source] = {}
    salt = f"{getattr(bench, 'manifest_hash', '')}:{split}:{int(manifest.row[i])}" + (":canary" if canary_of is not None else "")

    def shared(src: _Source) -> _Source:
        return sources.setdefault(src.loc, src)

    streams: dict[str, RawStream] = {}
    for name, sc in ctx.streams.items():
        cell = cells.get(name)
        if cell is None or cell == "":
            if sc.required:
                raise KeyError(f"manifest row {clip_id!r} has no media cell for required stream {name!r}")
            continue
        if not sc.required and not sc.present(cell):
            continue
        entry = ctx.info_table.get(name, cell) if ctx.info_table is not None else None
        # one handle per row: two streams of one stored file (AVE, avsync) share one opaque path
        streams[name] = RawStream(name, sc.kind, src=shared(sc.source(cell)), clip_id=clip_id, info_entry=entry, required_info=sc.info_keys, ext=_ext(cell),
                                  handle=salt)
    meta = {"facts": json.loads(json.dumps(ctx.facts))}
    return Sample(row=int(manifest.row[i]), clip_id=clip_id, split=split, streams=streams, meta=meta, _canary_of=canary_of)


def rehandle(sample: Sample, salt: str, **identity: Any) -> Sample:
    """A copy of ``sample`` whose streams hand over the same stored bytes under fresh opaque paths (salted by
    ``salt``), optionally with ``row`` / ``clip_id`` / ``split`` replaced (bridge check's passes after A
    are new Samples, so a cache keyed on the path, the stream object or the identity misses)."""
    import copy
    import dataclasses

    streams = {n: (st.with_handle(f"{salt}:{st._handle}") if isinstance(st, RawStream) else copy.copy(st)) for n, st in sample.streams.items()}
    if dataclasses.is_dataclass(sample):
        return dataclasses.replace(sample, streams=streams, **identity)
    clone = copy.copy(sample)
    for key, v in {"streams": streams, **identity}.items():
        object.__setattr__(clone, key, v)
    return clone


def _media_info_of(bench: Any) -> MediaInfoTable | None:
    fn = getattr(bench, "media_info", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except (OSError, ValueError):
        return None


def canary_rows(n_rows: int, rows: range, batch_size: int, canaries: int, *, manifest_hash: str, split: str, shard_index: int, canary_seed: int) -> list[int]:
    """Canaries: ``min(K, n_in_shard // 4)`` regular rows drawn without replacement by
    ``default_rng(sha256(f"{manifest_hash}:{split}:{i}:{canary_seed}"))`` from rows at least
    ``2 * batch_size`` positions before the end of the shard. Returns row indices in draw order."""
    from yavih.hashing import rng_seed

    k = min(int(canaries), len(rows) // 4)
    if k <= 0:
        return []
    candidates = list(rows)[: max(0, len(rows) - 2 * int(batch_size))]
    k = min(k, len(candidates))
    if k <= 0:
        return []
    rng = np.random.default_rng(rng_seed(f"{manifest_hash}:{split}:{shard_index}:{canary_seed}"))
    return [int(r) for r in rng.choice(np.asarray(candidates, dtype=np.int64), size=k, replace=False)]


def iter_samples(
    bench: Any,
    split: str,
    rows: range | None = None,
    batch_size: int = 1,
    canaries: int = 0,
    canary_seed: int = 0,
    *,
    shard: tuple[int, int] | None = None,
    roots: Mapping[str, str] | None = None,
) -> Iterator[Sample]:
    """The generic reader: the Samples of ``rows`` in ascending row order, then the canaries
    (fresh copies with ``_canary_of`` set, seeded by ``shard[0]``). ``rows`` defaults to the shard's range of
    ``plan_shards(n, batch_size, shard[1])`` when ``shard`` is given, else the whole split. Every folder
    ``reader.py`` delegates here; the media-info table is taken from ``bench.media_info()`` when present."""
    manifest = bench.manifest(split)
    n = len(manifest)
    if rows is None:
        if shard is not None and int(shard[1]) > 1:
            from yavih.io.shards import plan_shards

            lo, hi = plan_shards(n, int(batch_size), int(shard[1]))[int(shard[0])]
            rows = range(lo, hi)
        else:
            rows = range(0, n)
    if rows.start < 0 or rows.stop > n or rows.step != 1:
        raise ValueError(f"rows {rows} do not lie in [0, {n}) with step 1")
    ctx = _SplitCtx(bench, split, _resolve_roots(bench, roots), _media_info_of(bench))
    for i in rows:
        yield build_sample(bench, split, manifest, i, ctx=ctx)
    shard_index = int(shard[0]) if shard else 0
    for i in canary_rows(n, rows, batch_size, canaries, manifest_hash=bench.manifest_hash, split=split, shard_index=shard_index, canary_seed=canary_seed):
        yield build_sample(bench, split, manifest, i, ctx=ctx, canary_of=int(manifest.row[i]))


def batches(samples: Iterable[Sample], batch_size: int) -> Iterator[list[Sample]]:
    """Group Samples into lists of ``batch_size`` in the order given; canaries start a
    batch of their own, so a canary never shares a batch with the regular rows."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    cur: list[Sample] = []
    in_canaries = False
    for s in samples:
        is_canary = s._canary_of is not None
        if cur and (len(cur) >= batch_size or is_canary != in_canaries):
            yield cur
            cur = []
        in_canaries = is_canary
        cur.append(s)
    if cur:
        yield cur


# ----------------------------------------------------------------------------- byte read-ahead


def default_decode_workers(env: Mapping[str, str] | None = None) -> int:
    """``YAVIH_DECODE_WORKERS`` / ``[resources] decode_workers``, else ``CPUs − 1`` for an allocation
    (SLURM, PBS, ``YAVIH_CPUS``, affinity mask, cgroup quota) or ``CPUs // 2`` on a whole machine, capped at 16,
    at least 1. One implementation: ``yavih.resources.decode_workers``."""
    from yavih.resources import decode_workers

    return decode_workers(env)


def fetch(sample: Sample) -> float:
    """Read every stored byte of ``sample`` (every RawStream) and resolve every stream's
    ``info``; returns the seconds spent. Raises ``MediaMissing`` for an absent file, absent index entry, short
    read or a header that cannot supply a guaranteed fact. ``DecoderMissing`` propagates."""
    t0 = time.perf_counter()
    for st in sample.streams.values():
        st.bytes()
        st.info  # noqa: B018 - resolves (and caches) the facts before the bridge sees the row
    return time.perf_counter() - t0


@dataclass
class ReadBatch:
    """One batch after the read-ahead: the Samples whose bytes are in memory, the rows that turned out
    missing (``(sample, MediaMissing)``, never handed to the bridge) and the read seconds summed over rows."""

    samples: list[Sample]
    missing: list[tuple[Sample, Exception]]
    read_s: float


class ReadAhead:
    """Torch-free byte read-ahead: while the bridge encodes batch ``b``, ``workers`` threads call
    ``fetch`` on every Sample of batches ``b+1 … b+depth``. No decoding happens here."""

    def __init__(self, workers: int | None = None, depth: int = 2) -> None:
        self.workers = int(workers) if workers is not None else default_decode_workers()
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if depth < 0:
            raise ValueError("depth must be >= 0")
        self.depth = int(depth)
        self._pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="yavih-read")

    def _submit(self, batch: list[Sample]) -> tuple[list[Sample], list[Future]]:
        return batch, [self._pool.submit(fetch, s) for s in batch]

    @staticmethod
    def _collect(batch: list[Sample], futures: list[Future]) -> ReadBatch:
        present, missing, read_s = [], [], 0.0
        for s, f in zip(batch, futures, strict=True):
            try:
                read_s += f.result()
                present.append(s)
            except (MediaMissing, OSError) as exc:  # absent / unreadable / short read → a missing row
                missing.append((s, exc))
        return ReadBatch(present, missing, read_s)

    def run(self, batches_in: Iterable[list[Sample]]) -> Iterator[ReadBatch]:
        """Yield ``ReadBatch``es in input order, keeping up to ``depth + 1`` batches in flight."""
        pending: deque[tuple[list[Sample], list[Future]]] = deque()
        it = iter(batches_in)
        exhausted = False
        while True:
            while not exhausted and len(pending) < self.depth + 1:
                try:
                    pending.append(self._submit(next(it)))
                except StopIteration:
                    exhausted = True
            if not pending:
                return
            batch, futures = pending.popleft()
            yield self._collect(batch, futures)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> ReadAhead:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()
