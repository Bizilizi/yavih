"""The input vocabulary: one home for what a benchmark folder may declare it hands a bridge.

A folder's ``fields`` (``benchmark.yaml`` schema 3) name what each manifest row carries for the bridge. Every field
has a **storage**: ``file`` (stored bytes under a data root: a media file the reader probes for facts, or any other
declared kind handed over as bytes) or ``cell`` (a typed manifest column handed over as is: a caption, an option
list, a point query). The media kinds keep their header facts and rule checks; the meaning of a field is the
folder's, the storage and its checks are yavih's. Stdlib only.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "MEDIA_KINDS",
    "STREAM_KINDS",
    "STORAGES",
    "FIELD_ROLES",
    "LAYOUTS",
    "COLUMN_ROLES",
    "COLUMN_TYPES",
    "MEDIA_INFO_KEYS",
    "INFO_VOCAB",
    "TRACK_FACTS",
    "SPATIAL_KINDS",
    "TARGET_KINDS",
    "TARGET_SOURCES",
    "FIELD_DEFAULTS",
    "is_media_kind",
    "file_fields",
    "spec_streams",
    "cell_fields",
]

#: Stream kinds with header facts, decoders and rule checks.
MEDIA_KINDS: frozenset[str] = frozenset({"video", "audio", "image"})
#: The kinds a ``file`` field may declare today (the media kinds).
STREAM_KINDS: frozenset[str] = MEDIA_KINDS
#: ``file``: stored bytes under a data root (``Sample.streams``); ``cell``: a typed manifest column (``Sample.cells``).
STORAGES: frozenset[str] = frozenset({"file", "cell"})
#: What a field is for; ``input`` (handed to the bridge) is the only role today.
FIELD_ROLES: frozenset[str] = frozenset({"input"})
#: The defaults a field entry may leave out; ``hashing.fields_view`` drops them, so a schema-2 ``streams`` block and
#: its schema-3 ``fields`` migration hash identically.
FIELD_DEFAULTS: Mapping[str, str] = {"storage": "file", "role": "input"}
#: Layouts of a ``file`` field.
LAYOUTS: frozenset[str] = frozenset({"dir", "tar_index"})
#: ``manifest.columns[*].role``: ``media`` (a file field's relative path), ``cell`` (a cell field's value), ``extra``
#: (recorded, never handed over), ``label_aux`` (label structure).
COLUMN_ROLES: frozenset[str] = frozenset({"media", "cell", "extra", "label_aux"})
#: ``manifest.columns[*].type``; ``json`` is parsed at load (an option list, a point query).
COLUMN_TYPES: frozenset[str] = frozenset({"str", "int", "float", "json"})
#: Every ``RawStream.info`` carries these keys (null where unknown or not applicable). ``has_track`` (video, audio):
#: ``True`` when the stored file holds the track the stream names, ``False`` when it holds none, null for images.
MEDIA_INFO_KEYS: tuple[str, ...] = ("duration_s", "fps", "n_frames", "sample_rate", "n_samples", "channels", "width", "height", "container",
                                    "bytes", "has_track")
#: Keys ``input_contract.info[field]`` may guarantee, per media kind. ``rotation`` is recorded for video when
#: tagged but can never be guaranteed.
INFO_VOCAB: Mapping[str, frozenset[str]] = {
    "video": frozenset({"duration_s", "fps", "n_frames", "width", "height", "container", "bytes"}),
    "audio": frozenset({"duration_s", "sample_rate", "n_samples", "channels", "container", "bytes"}),
    "image": frozenset({"width", "height", "container", "bytes"}),
}
#: Facts of a stream's TRACK (null when ``has_track`` is false); ``container`` and ``bytes`` describe the file.
TRACK_FACTS: Mapping[str, frozenset[str]] = {k: v - {"container", "bytes"} for k, v in INFO_VOCAB.items() if k in ("video", "audio")}
#: Media kinds with a spatial rule (frames).
SPATIAL_KINDS: frozenset[str] = frozenset({"video", "image"})
#: ``targets[*].kind`` / ``source``: label arrays a CSV cell cannot hold.
TARGET_KINDS: frozenset[str] = frozenset({"dense_depth"})
TARGET_SOURCES: frozenset[str] = frozenset({"derived_file"})


def is_media_kind(kind: str) -> bool:
    return kind in MEDIA_KINDS


def file_fields(fields: Mapping[str, object] | None) -> dict[str, dict]:
    """The ``file`` entries of a raw ``fields`` block, without ``storage`` / ``role``: the schema-2 ``streams`` shape."""
    out: dict[str, dict] = {}
    for name, e in dict(fields or {}).items():
        e = dict(e or {})  # type: ignore[arg-type]
        if e.get("storage", "file") == "file":
            out[str(name)] = {k: v for k, v in e.items() if k not in ("storage", "role")}
    return out


def spec_streams(spec: object) -> dict:
    """The file fields of a loaded ``BenchmarkSpec`` (``.streams``) or of a raw / duck-typed spec (a ``streams``
    mapping, else the ``file`` entries of its ``fields`` block)."""
    if isinstance(spec, Mapping):
        if spec.get("streams") is not None:
            return dict(spec["streams"])  # type: ignore[arg-type]
        return file_fields(spec.get("fields"))  # type: ignore[arg-type]
    streams = getattr(spec, "streams", None)
    if streams is not None:
        return dict(streams)
    return file_fields(getattr(spec, "fields", None))


def cell_fields(fields: Mapping[str, object] | None) -> dict[str, dict]:
    """The ``cell`` entries of a raw ``fields`` block."""
    return {str(n): dict(e) for n, e in dict(fields or {}).items() if dict(e or {}).get("storage") == "cell"}  # type: ignore[arg-type]
