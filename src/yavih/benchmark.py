"""``Benchmark``: ``benchmark.yaml`` schema 2, manifest, classes, splits/roles, targets, input/output
contracts, ``external``, data status, media-info table, alias resolution (the
``bench *`` commands).

Core module: numpy + pyyaml + stdlib only. Metric names are resolved against
``yavih.metrics.registry`` overlaid by the folder's ``eval.py`` ``METRICS`` (names and protocol keys only;
no sklearn). Decoders are never imported here; ``bench verify`` reads bytes through ``yavih.sample`` and
decodes its health sample through the optional ``yavih.media.decode``.

Layering:

- ``load_yaml`` parses and validates everything that can be checked from the yaml alone
  (schema, unknown keys at every level, cross references, metric names and protocol keys) plus the
  folder's ``eval.py`` hooks (``METRICS``, ``PROTOCOL_KEYS``) when one sits next to the yaml.
- ``Benchmark.from_dir`` adds the folder checks (``name == folder``, manifest/classes sha256,
  label parsing, media columns per split) and computes ``manifest_hash``/``protocol_hash``.
- ``bench_verify`` is the only function that touches media or derived files;
  ``data_status`` touches the filesystem for presence only.
"""

from __future__ import annotations

import csv
import datetime as _dt
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from yavih import config as _config
from yavih import hash_history as _hash_history
from yavih import hashing
from yavih.metrics import registry as _registry
from yavih.sample import INFO_VOCAB, MediaInfoTable, Sample, media_info_path
from yavih.shipped import bench_file, bench_file_text, bench_files, prepare_steps
from yavih.suites import CAPABILITIES
from yavih.versions import BENCHMARK_SCHEMA_VERSION

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkError",
    "SchemaError",
    "ManifestError",
    "UnknownBenchmark",
    "NoTargets",
    "DataNotReady",
    "ChangelogEntry",
    "SourceSpec",
    "BudgetSpec",
    "LabelSpec",
    "ColumnSpec",
    "CurationSpec",
    "UpstreamSpec",
    "ManifestSpec",
    "SplitsSpec",
    "StreamSpec",
    "TargetSpec",
    "RootSpec",
    "DataSpec",
    "MetricsSpec",
    "RequiresSpec",
    "InputContractSpec",
    "OutputContractSpec",
    "ExternalSpec",
    "BenchmarkSpec",
    "Manifest",
    "TargetReader",
    "DataStatus",
    "Benchmark",
    "load_yaml",
    "packaged_benchmarks",
    "list_benchmarks",
    "bench_info",
    "bench_fetch",
    "bench_verify",
    "bench_files",
    "bench_file",
    "bench_file_text",
    "prepare_steps",
    "media_fingerprint",
    "data_status_path",
    "home_benchmarks_dir",
    "READOUTS",
    "LABEL_KINDS",
    "SOURCE_KINDS",
    "OBTAIN_TIERS",
    "ObtainSpec",
    "HeaderSpec",
    "CAPABILITIES",
    "HEADER_KEYS",
    "HEADER_ABSTRACT_MAX_WORDS",
    "LAYOUTS",
    "COLUMN_ROLES",
    "COLUMN_TYPES",
    "STREAM_KINDS",
    "CLASS_KEYS",
    "VIEWS",
    "TARGET_KINDS",
    "TARGET_SOURCES",
    "PROTOCOL_FIXED_KEYS",
    "READOUT_PROTOCOL_KEYS",
    "READOUT_PARAM_KEYS",
    "FINGERPRINT_BYTES",
]

class _RegistryView(Mapping):
    """``{readout: getattr(Readout, attr)}`` over the readout registry (``yavih.readouts.registry``), read on use:
    the readouts describe themselves, this module hard-codes none."""

    def __init__(self, attr: str) -> None:
        self._attr = attr

    def _map(self) -> dict[str, Any]:
        from yavih.readouts import registry

        return {n: getattr(r, self._attr) for n, r in registry.all_scorers().items()}

    def __getitem__(self, key: str) -> Any:
        return self._map()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._map())

    def __len__(self) -> int:
        return len(self._map())


def _readout(name: str) -> Any:
    from yavih.readouts import registry

    return registry.get(name)


def _readout_names() -> frozenset[str]:
    from yavih.readouts import registry

    return frozenset(registry.names())


def _scorer_families(name: str) -> frozenset[str]:
    """The names a metric may be declared for to serve scorer ``name`` (itself and its family); an unregistered
    name serves only itself (the schema check reports it)."""
    from yavih.readouts import registry

    try:
        return registry.get(name).families()
    except registry.UnknownScorer:
        return frozenset({name})


#: The registered readout names (a live view; ``yavih.readouts.registry``).
READOUTS: Mapping[str, Any] = _RegistryView("name")
#: How ``manifest.label`` is parsed: the core's manifest vocabulary; each readout names the kinds it scores.
LABEL_KINDS: frozenset[str] = frozenset({"single", "multi", "multi_sets", "box", "segments", "none"})
SOURCE_KINDS: frozenset[str] = frozenset({"public_url", "torrent", "manual_portal", "private_mirror", "derived"})
#: ``obtain.tier``: who must act before the data can be staged. ``auto`` = a script/agent can fetch it unattended;
#: ``manual-download`` = a person downloads the files by hand from an official portal or the authors' links
#: (browser-only or quota-limited links, possibly after a free registration or click-through terms; no data owner
#: involved); ``needs-licence`` = a human enters a licence / terms-of-access agreement that grants access to their
#: own account (an agent never accepts it for them); ``needs-data-owner`` = part of it is only available from the
#: person who holds it (decayed YouTube media, a private file); ``site-only`` = staged at the origin site only.
#: Provenance for people and agents; never hashed.
OBTAIN_TIERS: frozenset[str] = frozenset({"auto", "manual-download", "needs-licence", "needs-data-owner", "site-only"})
from yavih.fields import (  # noqa: E402 - the one vocabulary
    COLUMN_ROLES,
    COLUMN_TYPES,
    FIELD_ROLES,
    LAYOUTS,
    STORAGES,
    STREAM_KINDS,
)

CLASS_KEYS: frozenset[str] = frozenset({"name", "mid", "index"})
VIEWS: frozenset[str] = frozenset({"none", "rows"})
from yavih.fields import TARGET_KINDS, TARGET_SOURCES  # noqa: E402

#: Keys every ``protocol`` block carries; anything else is a per-benchmark label fact that
#: must be consumed (a metric, the readout, ``eval.py`` ``PROTOCOL_KEYS`` or ``input_contract.facts``).
PROTOCOL_FIXED_KEYS: frozenset[str] = frozenset({"clip_seconds", "train_draw", "views"})
#: Per-benchmark protocol keys the readout driver consumes that no registered metric declares.
#: A key only one benchmark consumes belongs to its ``eval.py`` ``PROTOCOL_KEYS``.
READOUT_PROTOCOL_KEYS: Mapping[str, frozenset[str]] = _RegistryView("protocol_keys")
#: Allowed ``readout_params`` keys per readout (the key set must be exactly this). Declared by each readout.
READOUT_PARAM_KEYS: Mapping[str, frozenset[str]] = _RegistryView("param_keys")
#: Bytes hashed per sampled media file by ``media_fingerprint`` (``bench verify``).
FINGERPRINT_BYTES = 64 * 1024

_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_.#:-]+$")
_MEDIA_RE = re.compile(r"^[A-Za-z0-9_.#:/-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEGMENT_COLUMNS = ("event_start", "event_end")
_STATUS_SCHEMA = 1


# ----------------------------------------------------------------------------- errors


class BenchmarkError(Exception):
    """Base class; CLI maps subclasses to exit codes."""


class SchemaError(BenchmarkError, ValueError):
    """``benchmark.yaml`` violates the schema (unknown key, wrong type, bad cross-reference). Exit 1."""


class ManifestError(BenchmarkError, ValueError):
    """Manifest/classes are invalid (sha mismatch, unknown label, empty cell, charset). Exit 2."""


class UnknownBenchmark(BenchmarkError, KeyError):
    """``Benchmark.load(ref)`` resolved nothing in the resolution order. Exit 1."""

    def __str__(self) -> str:  # KeyError would repr() the message
        return str(self.args[0]) if self.args else ""


class NoTargets(BenchmarkError, LookupError):
    """``Benchmark.targets(split)`` on a readout whose labels are manifest cells."""


class DataNotReady(BenchmarkError, FileNotFoundError):
    """Media/derived data missing on this site (exit 3)."""


# ----------------------------------------------------------------------------- schema dataclasses


@dataclass(frozen=True)
class ChangelogEntry:
    """One ``benchmark.yaml`` ``changelog`` entry. The hash keys sit outside every hashed payload, so
    recording them is not circular: ``protocol_hash`` / ``input_hash`` are the digests this version computes (required
    on the newest entry by ``bench lint``); ``supersedes`` lists earlier protocol hashes of the same benchmark whose
    results ARE this version's (``report.migrate`` maps them here); ``supersedes_input``
    lists earlier input hashes whose feature sets stay valid (an input-side edit that hands the bridge the same
    thing)."""

    version: int
    date: str
    reason: str
    invalidates: bool
    protocol_hash: str | None = None
    input_hash: str | None = None
    supersedes: tuple[str, ...] = ()
    supersedes_input: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSpec:
    kind: str  # SOURCE_KINDS
    urls: tuple[str, ...]
    licence: str
    bytes: int | None = None
    sha256: Mapping[str, str] | None = None
    notes: str | None = None
    revision: str | None = None  # pinned release revision (e.g. a Hugging Face commit sha); provenance, unhashed


@dataclass(frozen=True)
class ObtainSpec:
    """``benchmark.yaml`` ``obtain`` (optional): how hard the data is to get here, for ``bench list`` / ``bench info``
    and the skill's "When blocked" rule. Outside the protocol hash."""

    tier: str  # OBTAIN_TIERS
    how: str  # one line: what a user does to get the data


#: ``benchmark.yaml`` ``header`` keys: the README header image (``python -m yavih.headers``). Not hashed.
HEADER_KEYS: tuple[str, ...] = ("name", "title", "venue", "abstract")
#: The longest ``header.abstract`` (words) that fits the header card.
HEADER_ABSTRACT_MAX_WORDS = 60


@dataclass(frozen=True)
class HeaderSpec:
    """``benchmark.yaml`` ``header`` (optional in the schema, required by ``bench lint``): what the README header
    image shows. ``name`` is the benchmark's display name, ``title`` the paper title, ``venue`` "authors · venue
    year" and ``abstract`` a short neutral summary of the dataset and task. Outside every hash."""

    name: str
    title: str
    venue: str
    abstract: str


@dataclass(frozen=True)
class BudgetSpec:
    clips: Mapping[str, int]
    media_gb: float
    extract_min_per_1k_clips_ref: float
    probe_minutes_ref: Mapping[str, float]


@dataclass(frozen=True)
class LabelSpec:
    column: str | None
    kind: str  # LABEL_KINDS
    sep: str = ";"
    class_key: str = "name"  # "name" | "mid" | "index"
    sets: Mapping[str, tuple[str, ...]] | None = None
    train_set: str | None = None
    allow_unknown: bool = False
    empty_ok: bool = False

    def label_columns(self) -> tuple[str, ...]:
        """Every manifest column that holds labels: ``column`` plus the ``multi_sets`` columns."""
        cols: list[str] = []
        if self.column is not None:
            cols.append(self.column)
        for members in (self.sets or {}).values():
            for c in members:
                if c not in cols:
                    cols.append(c)
        return tuple(cols)


@dataclass(frozen=True)
class ColumnSpec:
    type: str  # "str" | "int" | "float"
    role: str  # COLUMN_ROLES
    stream: str | tuple[str, ...] | None = None  # role media: the stream(s) this column feeds (AVE: one mp4 for video+audio)
    field: str | None = None  # role cell: the cell field this column's value is handed over as

    def streams(self) -> tuple[str, ...]:
        if self.stream is None:
            return ()
        return (self.stream,) if isinstance(self.stream, str) else tuple(self.stream)


@dataclass(frozen=True)
class CurationSpec:
    total_upstream: int
    per_split: Mapping[str, Mapping[str, int]]  # split -> {listed, staged, missing}
    excluded: Mapping[str, int]
    short_clips: int = 0
    leakage_audit: Mapping[str, Any] | None = None

    def invariant_holds(self) -> bool:
        """``sum(per_split.*.listed) + sum(excluded.*) == total_upstream``."""
        listed = sum(int(v.get("listed", 0)) for v in self.per_split.values())
        return listed + sum(int(v) for v in self.excluded.values()) == self.total_upstream


@dataclass(frozen=True)
class UpstreamSpec:
    source: str
    sha256: str
    rows: int
    note: str | None = None
    #: How ``protocol.train_draw`` was drawn, in words (provenance; the hashed parameters are ``cap``/``seed``/``balance``).
    draw: str | None = None
    upstream_files: Mapping[str, str] = field(default_factory=dict)
    curation: CurationSpec | None = None


@dataclass(frozen=True)
class ManifestSpec:
    file: str
    sha256: str
    classes: str | None
    classes_sha256: str | None
    key: str
    label: LabelSpec
    columns: Mapping[str, ColumnSpec]
    upstream: UpstreamSpec | None = None


@dataclass(frozen=True)
class SplitsSpec:
    train: tuple[str, ...]
    eval: tuple[str, ...]
    floor: tuple[str, ...] = ()
    folds: tuple[str, ...] | None = None
    unused: tuple[str, ...] = ()
    min_eval_coverage: float = 0.95

    def role_of(self, split: str) -> str | None:
        """``"train" | "eval" | "floor" | "folds" | "unused"`` for a manifest split value."""
        for role in ("train", "eval", "floor", "folds", "unused"):
            values = getattr(self, role)
            if values is not None and split in values:
                return role
        return None

    def all_values(self) -> tuple[str, ...]:
        return tuple(self.train) + tuple(self.eval) + tuple(self.floor) + tuple(self.folds or ()) + tuple(self.unused)

    def read_values(self) -> tuple[str, ...]:
        """Split values the harness reads (everything but ``unused``)."""
        return tuple(self.train) + tuple(self.eval) + tuple(self.floor) + tuple(self.folds or ())


def _per_split(value: Any, split: str, role: str | None, what: str) -> Any:
    """Resolve a ``str | dict[split, …]`` field: by manifest split value first, then by the
    split's role (``{train: …, eval: …}``, the role words)."""
    if not isinstance(value, Mapping):
        return value
    if split in value:
        return value[split]
    if role is not None and role in value:
        return value[role]
    raise SchemaError(f"{what} has no entry for split {split!r} (role {role!r}); keys: {sorted(value)}")


@dataclass(frozen=True)
class CellSpec:
    """A ``cell`` field: the value of one typed manifest column, handed to the bridge as ``Sample.cells[name]`` (a
    caption, an option list, a point query). ``kind`` names what the value means (``[a-z][a-z0-9_]*``, the folder's
    vocabulary); ``column`` is a ``manifest.columns`` entry with ``role: cell`` and ``field: <name>``."""

    kind: str
    column: str


@dataclass(frozen=True)
class StreamSpec:
    """Any of ``media_column/root/layout/layout_params`` may be a per-split dict.
    ``layout_params_per_split`` is decided at load: true iff every key is a split value or role
    and every value a mapping."""

    kind: str  # "video" | "audio" | "image"
    media_column: str | Mapping[str, str]
    root: str | Mapping[str, str]
    layout: str | Mapping[str, str] = "dir"
    layout_params: Mapping[str, Any] = field(default_factory=dict)
    required: bool = True
    layout_params_per_split: bool = False

    def for_split(self, split: str, role: str | None = None) -> dict[str, Any]:
        """Resolve the per-split variants to one flat dict for ``split``."""
        params = self.layout_params
        if self.layout_params_per_split:
            params = _per_split(params, split, role, "streams.*.layout_params")
        return {
            "kind": self.kind,
            "media_column": _per_split(self.media_column, split, role, "streams.*.media_column"),
            "root": _per_split(self.root, split, role, "streams.*.root"),
            "layout": _per_split(self.layout, split, role, "streams.*.layout"),
            "layout_params": dict(params),
            "required": self.required,
        }

    def media_columns(self) -> tuple[str, ...]:
        if isinstance(self.media_column, Mapping):
            return tuple(dict.fromkeys(self.media_column.values()))
        return (self.media_column,)

    def layouts(self) -> tuple[str, ...]:
        if isinstance(self.layout, Mapping):
            return tuple(dict.fromkeys(self.layout.values()))
        return (self.layout,)

    def roots(self) -> tuple[str, ...]:
        if isinstance(self.root, Mapping):
            return tuple(dict.fromkeys(self.root.values()))
        return (self.root,)


@dataclass(frozen=True)
class TargetSpec:
    kind: str  # "dense_depth"
    source: str  # "derived_file"
    root: str
    file: str
    sha256: str
    index_column: str | None = None


@dataclass(frozen=True)
class RootSpec:
    default: str
    description: str | None = None
    prepare_only: bool = False  # read only by prepare/ (doctor reports it missing as a note, not a warning)


@dataclass(frozen=True)
class DataSpec:
    roots: Mapping[str, RootSpec]
    tar_index: str | None = None
    derived: Mapping[str, str] | None = None


@dataclass(frozen=True)
class MetricsSpec:
    headline: str
    list: tuple[str, ...]
    floors: tuple[str, ...]
    controls: tuple[str, ...]
    chance: Mapping[str, float]
    ci: Mapping[str, str | None]
    bootstrap: Mapping[str, int] | None = None
    aggregation: Mapping[str, Any] | None = None
    #: the report's default markdown columns (a subset of ``list`` holding the headline); ``None`` = all of ``list``
    table: tuple[str, ...] | None = None
    #: per-metric absolute conformance tolerance (``yavih.conformance``); absent names use the default. Not hashed.
    tolerance: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RequiresSpec:
    pos: tuple[str, ...] = ()
    streams: tuple[str, ...] = ()
    grid: Mapping[str, Any] | None = None
    t_bins: int | None = None
    # {fact, stream, tol_s}: a token of `stream` must lie within tol_s of the label fact's second (vggss_box: the
    # box is drawn on the frame at annotated_second 5.0), median over validate's sample
    annotated_frame: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class InputContractSpec:
    """``input_contract``: what the reader hands over, which facts are guaranteed, which label facts
    reach ``Sample.meta["facts"]``, and the one label-required prepare-time transform (avsync)."""

    handed_over: Mapping[str, str]
    info: Mapping[str, tuple[str, ...]]
    facts: tuple[str, ...] = ()
    materialize: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class OutputContractSpec:
    """``output_contract``: pinned ``shape: fixed`` (every symbol of the schema fixed per feature set) and
    ``coordinates: media``, the spatial target frame, the eligibility ``requires``, the ``schema`` of what a bridge
    returns per row (``None`` = the default feature matrix, ``yavih.codecs.DEFAULT_SCHEMA``) and ``sampling``
    (``required``: the bridge generates and must declare ``describe()["sampling"]``; ``optional`` otherwise)."""

    shape: str
    coordinates: str
    target_frame: str | None
    requires: RequiresSpec
    schema: Mapping[str, Any] | None = None
    sampling: str = "optional"

    @property
    def resolved_schema(self) -> dict[str, dict[str, Any]]:
        from yavih.codecs import schema_of

        return schema_of({"schema": self.schema})


@dataclass(frozen=True)
class ExternalSpec:
    """``external`` entry: an original-author package the folder's ``eval.py`` calls (never a yavih
    dependency)."""

    name: str
    pip: str
    version_pin: str
    used_for: str


@dataclass(frozen=True)
class BenchmarkSpec:
    """Frozen mirror of ``benchmark.yaml``. ``protocol`` keeps the raw mapping because
    it carries free per-benchmark keys; ``raw`` is the parsed yaml exactly as hashed."""

    schema_version: int
    name: str
    aliases: tuple[str, ...]
    title: str
    readout: str
    protocol_version: int
    changelog: tuple[ChangelogEntry, ...]
    source: SourceSpec
    budget: BudgetSpec
    manifest: ManifestSpec
    splits: SplitsSpec
    streams: Mapping[str, StreamSpec]  # the file fields
    targets: Mapping[str, TargetSpec] | None
    data: DataSpec
    protocol: Mapping[str, Any]
    metrics: MetricsSpec
    input_contract: InputContractSpec
    output_contract: OutputContractSpec
    external: tuple[ExternalSpec, ...]
    readout_params: Mapping[str, Any]
    raw: Mapping[str, Any] = field(repr=False, compare=False, default_factory=dict)
    #: the cell fields (a typed manifest column handed over as is)
    cells: Mapping[str, CellSpec] = field(default_factory=dict)
    #: names resolved through the folder's ``eval.py`` ``METRICS`` (provenance; not hashed)
    folder_metrics: tuple[str, ...] = field(default=(), compare=False)
    #: ``obtain`` (optional; not hashed)
    obtain: ObtainSpec | None = field(default=None, compare=False)
    #: ``header`` (optional; not hashed)
    header: HeaderSpec | None = field(default=None, compare=False)
    #: ``capability`` (optional in the schema, required by ``bench lint``; not hashed): ids of
    #: ``yavih.suites.CAPABILITIES`` the benchmark measures, in the yaml's order; ``()`` when absent
    capability: tuple[str, ...] = field(default=(), compare=False)

    @property
    def requires(self) -> RequiresSpec:
        """``output_contract.requires``; a convenience alias."""
        return self.output_contract.requires

    @property
    def protocol_extra_keys(self) -> frozenset[str]:
        """The per-benchmark protocol keys (everything outside ``PROTOCOL_FIXED_KEYS``)."""
        return frozenset(self.protocol) - PROTOCOL_FIXED_KEYS

    @property
    def target_region(self) -> tuple[float, float, float, float]:
        """``protocol.target_region`` ``(y0, x0, y1, x1)`` in full-frame fractions, else the whole frame."""
        r = self.protocol.get("target_region")
        return (0.0, 0.0, 1.0, 1.0) if r is None else tuple(float(x) for x in r)  # type: ignore[return-value]

    def stream_for(self, name: str, split: str) -> dict[str, Any]:
        return self.streams[name].for_split(split, self.splits.role_of(split))


# ----------------------------------------------------------------------------- schema parsing helpers


def _fmt_path(path: str) -> str:
    return path or "<root>"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return (isinstance(v, (int, float))) and not isinstance(v, bool)


def _expect_mapping(v: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(v, Mapping):
        raise SchemaError(f"{_fmt_path(path)}: expected a mapping, got {type(v).__name__}")
    for k in v:
        if not isinstance(k, str):
            raise SchemaError(f"{_fmt_path(path)}: keys must be strings, got {k!r}")
    return v


def _check_keys(d: Mapping[str, Any], path: str, required: Sequence[str], optional: Sequence[str] = ()) -> None:
    """Unknown keys at any level are an error; required keys must be present."""
    unknown = sorted(set(d) - set(required) - set(optional))
    if unknown:
        raise SchemaError(f"{_fmt_path(path)}: unknown key(s) {unknown}; allowed: {sorted(set(required) | set(optional))}")
    missing = [k for k in required if k not in d]
    if missing:
        raise SchemaError(f"{_fmt_path(path)}: missing required key(s) {missing}")


def _str(v: Any, path: str, *, nullable: bool = False) -> Any:
    if v is None and nullable:
        return None
    if not isinstance(v, str):
        raise SchemaError(f"{_fmt_path(path)}: expected str, got {type(v).__name__}")
    return v


def _int(v: Any, path: str, *, nullable: bool = False, minimum: int | None = None) -> Any:
    if v is None and nullable:
        return None
    if not _is_int(v):
        raise SchemaError(f"{_fmt_path(path)}: expected int, got {type(v).__name__}")
    if minimum is not None and v < minimum:
        raise SchemaError(f"{_fmt_path(path)}: expected >= {minimum}, got {v}")
    return v


def _num(v: Any, path: str, *, nullable: bool = False) -> Any:
    if v is None and nullable:
        return None
    if not _is_num(v):
        raise SchemaError(f"{_fmt_path(path)}: expected a number, got {type(v).__name__}")
    return float(v)


def _bool(v: Any, path: str) -> bool:
    if not isinstance(v, bool):
        raise SchemaError(f"{_fmt_path(path)}: expected bool, got {type(v).__name__}")
    return v


def _str_list(v: Any, path: str, *, nullable: bool = False) -> Any:
    if v is None and nullable:
        return None
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise SchemaError(f"{_fmt_path(path)}: expected list[str]")
    if len(set(v)) != len(v):
        raise SchemaError(f"{_fmt_path(path)}: duplicate entries in {v}")
    return tuple(v)


def _enum(v: Any, path: str, allowed: frozenset[str] | set[str]) -> str:
    v = _str(v, path)
    if v not in allowed:
        raise SchemaError(f"{_fmt_path(path)}: {v!r} is not one of {sorted(allowed)}")
    return v


def _sha_or_none(v: Any, path: str) -> str | None:
    if v is None:
        return None
    v = _str(v, path)
    if not _SHA256_RE.match(v):
        raise SchemaError(f"{_fmt_path(path)}: expected 64 lower-case hex characters")
    return v


def _str_dict(v: Any, path: str, *, nullable: bool = False) -> Any:
    if v is None and nullable:
        return None
    m = _expect_mapping(v, path)
    for k, x in m.items():
        if not isinstance(x, str):
            raise SchemaError(f"{_fmt_path(path)}.{k}: expected str")
    return dict(m)


def _json_primitives(v: Any, path: str) -> None:
    """``protocol`` holds JSON primitives, lists and dicts only: it is hashed by ``canon``."""
    if v is None or isinstance(v, (bool, int, float, str)):
        if isinstance(v, float) and not math.isfinite(v):
            raise SchemaError(f"{_fmt_path(path)}: non-finite float is not hashable")
        return
    if isinstance(v, list):
        for i, x in enumerate(v):
            _json_primitives(x, f"{path}[{i}]")
        return
    if isinstance(v, Mapping):
        for k, x in v.items():
            if not isinstance(k, str):
                raise SchemaError(f"{_fmt_path(path)}: mapping keys must be str, got {k!r}")
            _json_primitives(x, f"{path}.{k}")
        return
    raise SchemaError(f"{_fmt_path(path)}: {type(v).__name__} is not a JSON primitive")


# ----------------------------------------------------------------------------- block parsers


def _parse_changelog(v: Any, path: str) -> tuple[ChangelogEntry, ...]:
    if not isinstance(v, list) or not v:
        raise SchemaError(f"{path}: expected a non-empty list")
    out: list[ChangelogEntry] = []
    for i, e in enumerate(v):
        p = f"{path}[{i}]"
        e = _expect_mapping(e, p)
        _check_keys(e, p, ["version", "date", "reason", "invalidates"], ["protocol_hash", "input_hash", "supersedes", "supersedes_input"])
        lists: dict[str, tuple[str, ...]] = {}
        for key in ("supersedes", "supersedes_input"):
            raw_list = e.get(key)
            if raw_list is None:
                lists[key] = ()
                continue
            if not isinstance(raw_list, list):
                raise SchemaError(f"{p}.{key}: expected a list of 64-hex digests")
            lists[key] = tuple(str(_sha_or_none(h, f"{p}.{key}[{j}]")) for j, h in enumerate(raw_list) if h is not None)
        out.append(
            ChangelogEntry(
                version=_int(e["version"], f"{p}.version", minimum=1),
                date=_str(e["date"], f"{p}.date"),
                reason=_str(e["reason"], f"{p}.reason"),
                invalidates=_bool(e["invalidates"], f"{p}.invalidates"),
                protocol_hash=_sha_or_none(e.get("protocol_hash"), f"{p}.protocol_hash"),
                input_hash=_sha_or_none(e.get("input_hash"), f"{p}.input_hash"),
                supersedes=lists["supersedes"],
                supersedes_input=lists["supersedes_input"],
            )
        )
    versions = [e.version for e in out]
    if versions != sorted(versions) or len(set(versions)) != len(versions):
        raise SchemaError(f"{path}: entries must be newest last with strictly increasing versions, got {versions}")
    return tuple(out)


def _parse_source(v: Any, path: str) -> SourceSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["kind", "urls", "licence"], ["bytes", "sha256", "notes", "revision"])
    revision = _str(d.get("revision"), f"{path}.revision", nullable=True)
    if revision is not None and not revision.strip():
        raise SchemaError(f"{path}.revision: must be a non-empty string when present")
    return SourceSpec(
        kind=_enum(d["kind"], f"{path}.kind", SOURCE_KINDS),
        urls=tuple(_str_list(d["urls"], f"{path}.urls")),
        licence=_str(d["licence"], f"{path}.licence"),
        bytes=_int(d.get("bytes"), f"{path}.bytes", nullable=True, minimum=0),
        sha256=_str_dict(d.get("sha256"), f"{path}.sha256", nullable=True),
        notes=_str(d.get("notes"), f"{path}.notes", nullable=True),
        revision=revision,
    )


def _parse_obtain(v: Any, path: str) -> ObtainSpec | None:
    if v is None:
        return None
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["tier", "how"])
    how = _str(d["how"], f"{path}.how")
    if not how.strip():
        raise SchemaError(f"{path}.how: must be a non-empty string")
    return ObtainSpec(tier=_enum(d["tier"], f"{path}.tier", OBTAIN_TIERS), how=how)


def _parse_header(v: Any, path: str) -> HeaderSpec | None:
    if v is None:
        return None
    d = _expect_mapping(v, path)
    _check_keys(d, path, list(HEADER_KEYS))
    vals = {k: _str(d[k], f"{path}.{k}") for k in HEADER_KEYS}
    for k, x in vals.items():
        if not x.strip():
            raise SchemaError(f"{path}.{k}: must be a non-empty string")
    n = len(vals["abstract"].split())
    if n > HEADER_ABSTRACT_MAX_WORDS:
        raise SchemaError(f"{path}.abstract: {n} words, at most {HEADER_ABSTRACT_MAX_WORDS} fit the header")
    return HeaderSpec(**vals)


def _parse_capability(v: Any, path: str) -> tuple[str, ...]:
    """``capability``: absent or a non-empty list of distinct ids of :data:`yavih.suites.CAPABILITIES` (the one
    vocabulary suites group by); ``()`` when absent (``bench lint`` requires it)."""
    if v is None:
        return ()
    caps = _str_list(v, path)
    if not caps:
        raise SchemaError(f"{path}: must be a non-empty list of capability ids ({', '.join(CAPABILITIES)})")
    for c in caps:
        if c not in CAPABILITIES:
            raise SchemaError(f"{path}: {c!r} is not a capability; the vocabulary is {', '.join(CAPABILITIES)} "
                              "(yavih.suites.CAPABILITIES)")
    return caps


def _parse_budget(v: Any, path: str, readout: str) -> BudgetSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["clips", "media_gb", "extract_min_per_1k_clips_ref", "probe_minutes_ref"])
    clips = _expect_mapping(d["clips"], f"{path}.clips")
    for k, n in clips.items():
        _int(n, f"{path}.clips.{k}", minimum=0)
    pmr = _expect_mapping(d["probe_minutes_ref"], f"{path}.probe_minutes_ref")
    for k, n in pmr.items():
        if k not in _readout_names():
            raise SchemaError(f"{path}.probe_minutes_ref: {k!r} is not a readout")
        _num(n, f"{path}.probe_minutes_ref.{k}")
    if readout not in pmr:
        raise SchemaError(f"{path}.probe_minutes_ref: missing an entry for the benchmark readout {readout!r}")
    return BudgetSpec(
        clips={k: int(n) for k, n in clips.items()},
        media_gb=_num(d["media_gb"], f"{path}.media_gb"),
        extract_min_per_1k_clips_ref=_num(d["extract_min_per_1k_clips_ref"], f"{path}.extract_min_per_1k_clips_ref"),
        probe_minutes_ref={k: float(n) for k, n in pmr.items()},
    )


def _parse_label(v: Any, path: str) -> LabelSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["column", "kind"], ["sep", "class_key", "sets", "train_set", "allow_unknown", "empty_ok"])
    kind = _enum(d["kind"], f"{path}.kind", LABEL_KINDS)
    column = _str(d["column"], f"{path}.column", nullable=True)
    sep = _str(d.get("sep", ";"), f"{path}.sep")
    if not sep:
        raise SchemaError(f"{path}.sep: must be a non-empty string")
    class_key = _enum(d.get("class_key", "name"), f"{path}.class_key", CLASS_KEYS)
    sets_raw = d.get("sets")
    sets: dict[str, tuple[str, ...]] | None = None
    if sets_raw is not None:
        m = _expect_mapping(sets_raw, f"{path}.sets")
        if not m:
            raise SchemaError(f"{path}.sets: must name at least one set")
        sets = {}
        for s, cols in m.items():
            if not re.match(r"^[a-z0-9_]+$", s):
                raise SchemaError(f"{path}.sets: set name {s!r} must match [a-z0-9_]+ (it becomes a metric name)")
            cols_t = _str_list(cols, f"{path}.sets.{s}")
            if not cols_t:
                raise SchemaError(f"{path}.sets.{s}: must list at least one column")
            sets[s] = cols_t
    train_set = _str(d.get("train_set"), f"{path}.train_set", nullable=True)
    allow_unknown = _bool(d.get("allow_unknown", False), f"{path}.allow_unknown")
    empty_ok = _bool(d.get("empty_ok", False), f"{path}.empty_ok")
    if kind == "multi_sets":
        if sets is None:
            raise SchemaError(f"{path}: kind multi_sets requires `sets`")
        if train_set is not None and train_set not in sets:
            raise SchemaError(f"{path}.train_set: {train_set!r} is not one of the declared sets {sorted(sets)}")
        if train_set is None and column is None:
            raise SchemaError(f"{path}: kind multi_sets needs `column` (train labels) or `train_set`")
    else:
        if sets is not None or train_set is not None:
            raise SchemaError(f"{path}: `sets`/`train_set` are only valid for kind multi_sets")
        if kind == "none" and column is not None:
            raise SchemaError(f"{path}.column: must be null for kind none")
        if kind != "none" and column is None:
            raise SchemaError(f"{path}.column: required for kind {kind!r}")
    return LabelSpec(
        column=column,
        kind=kind,
        sep=sep,
        class_key=class_key,
        sets=sets,
        train_set=train_set,
        allow_unknown=allow_unknown,
        empty_ok=empty_ok,
    )


def _parse_columns(v: Any, path: str) -> dict[str, ColumnSpec]:
    d = _expect_mapping(v, path)
    out: dict[str, ColumnSpec] = {}
    for name, c in d.items():
        p = f"{path}.{name}"
        c = _expect_mapping(c, p)
        _check_keys(c, p, ["type", "role"], ["stream", "field"])
        role = _enum(c["role"], f"{p}.role", COLUMN_ROLES)
        fld = _str(c.get("field"), f"{p}.field", nullable=True)
        if role == "cell" and fld is None:
            raise SchemaError(f"{p}: a cell column must name its `field`")
        if role != "cell" and fld is not None:
            raise SchemaError(f"{p}: `field` is only valid with role cell")
        stream_raw = c.get("stream")
        stream: str | tuple[str, ...] | None
        if isinstance(stream_raw, list):
            stream = _str_list(stream_raw, f"{p}.stream")
            if not stream:
                raise SchemaError(f"{p}.stream: must name at least one stream")
        else:
            stream = _str(stream_raw, f"{p}.stream", nullable=True)
        if role == "media" and stream is None:
            raise SchemaError(f"{p}: a media column must name its `stream`")
        if role != "media" and stream is not None:
            raise SchemaError(f"{p}: `stream` is only valid with role media")
        out[name] = ColumnSpec(type=_enum(c["type"], f"{p}.type", COLUMN_TYPES), role=role, stream=stream, field=fld)
    return out


def _parse_curation(v: Any, path: str) -> CurationSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["total_upstream", "per_split", "excluded"], ["short_clips", "leakage_audit"])
    per_split_raw = _expect_mapping(d["per_split"], f"{path}.per_split")
    per_split: dict[str, dict[str, int]] = {}
    for s, counts in per_split_raw.items():
        p = f"{path}.per_split.{s}"
        counts = _expect_mapping(counts, p)
        _check_keys(counts, p, ["listed", "staged", "missing"])
        per_split[s] = {k: _int(counts[k], f"{p}.{k}", minimum=0) for k in ("listed", "staged", "missing")}
    excluded_raw = _expect_mapping(d["excluded"], f"{path}.excluded")
    excluded = {k: _int(n, f"{path}.excluded.{k}", minimum=0) for k, n in excluded_raw.items()}
    la = d.get("leakage_audit")
    if la is not None:
        la = _expect_mapping(la, f"{path}.leakage_audit")
        _check_keys(la, f"{path}.leakage_audit", ["against", "excluded"])
        _str(la["against"], f"{path}.leakage_audit.against")
        _int(la["excluded"], f"{path}.leakage_audit.excluded", minimum=0)
        la = dict(la)
    return CurationSpec(
        total_upstream=_int(d["total_upstream"], f"{path}.total_upstream", minimum=0),
        per_split=per_split,
        excluded=excluded,
        short_clips=_int(d.get("short_clips", 0), f"{path}.short_clips", minimum=0),
        leakage_audit=la,
    )


def _parse_upstream(v: Any, path: str) -> UpstreamSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["source", "sha256", "rows"], ["note", "draw", "upstream_files", "curation"])
    return UpstreamSpec(
        source=_str(d["source"], f"{path}.source"),
        sha256=_str(d["sha256"], f"{path}.sha256"),
        rows=_int(d["rows"], f"{path}.rows", minimum=0),
        note=_str(d.get("note"), f"{path}.note", nullable=True),
        draw=_str(d.get("draw"), f"{path}.draw", nullable=True),
        upstream_files=_str_dict(d.get("upstream_files", {}), f"{path}.upstream_files"),
        curation=_parse_curation(d["curation"], f"{path}.curation") if d.get("curation") is not None else None,
    )


def _parse_manifest(v: Any, path: str) -> ManifestSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["file", "sha256", "classes", "classes_sha256", "key", "label", "columns"], ["upstream"])
    sha = _sha_or_none(d["sha256"], f"{path}.sha256")
    if sha is None:
        raise SchemaError(f"{path}.sha256: required")
    classes = _str(d["classes"], f"{path}.classes", nullable=True)
    classes_sha = _sha_or_none(d["classes_sha256"], f"{path}.classes_sha256")
    if (classes is None) != (classes_sha is None):
        raise SchemaError(f"{path}: classes and classes_sha256 must both be set or both be null")
    label = _parse_label(d["label"], f"{path}.label")
    if label.kind in {"single", "multi", "multi_sets", "segments"} and classes is None:
        raise SchemaError(f"{path}.classes: label kind {label.kind!r} needs classes.csv")
    if label.kind in {"box", "none"} and classes is not None:
        raise SchemaError(f"{path}.classes: label kind {label.kind!r} has no classes (must be null)")
    key = _str(d["key"], f"{path}.key")
    columns = _parse_columns(d["columns"], f"{path}.columns")
    reserved = {key, "split", *label.label_columns()}
    for c in columns:
        if c in reserved:
            raise SchemaError(f"{path}.columns.{c}: the key, `split` and label columns are implicit, not declared")
    if label.kind == "segments":
        for c in _SEGMENT_COLUMNS:
            spec = columns.get(c)
            if spec is None or spec.role != "label_aux" or spec.type != "int":
                raise SchemaError(f"{path}.columns: label kind segments needs `{c}` with {{type: int, role: label_aux}}")
    upstream = _parse_upstream(d["upstream"], f"{path}.upstream") if d.get("upstream") is not None else None
    return ManifestSpec(
        file=_str(d["file"], f"{path}.file"),
        sha256=sha,
        classes=classes,
        classes_sha256=classes_sha,
        key=key,
        label=label,
        columns=columns,
        upstream=upstream,
    )


def _parse_splits(v: Any, path: str) -> SplitsSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["train", "eval"], ["floor", "folds", "unused", "min_eval_coverage"])
    train = _str_list(d["train"], f"{path}.train")
    ev = _str_list(d["eval"], f"{path}.eval")
    floor = _str_list(d.get("floor", []), f"{path}.floor")
    folds = _str_list(d.get("folds"), f"{path}.folds", nullable=True)
    unused = _str_list(d.get("unused", []), f"{path}.unused")
    cov = _num(d.get("min_eval_coverage", 0.95), f"{path}.min_eval_coverage")
    if not (0.0 < cov <= 1.0):
        raise SchemaError(f"{path}.min_eval_coverage: must be in (0, 1], got {cov}")
    # Every split value is assigned exactly once.
    seen: dict[str, str] = {}
    for role, values in (("train", train), ("eval", ev), ("floor", floor), ("folds", folds or ()), ("unused", unused)):
        for s in values:
            if s in seen:
                raise SchemaError(f"{path}: split value {s!r} appears in both {seen[s]!r} and {role!r}")
            seen[s] = role
    if folds is not None:
        if len(folds) < 2:
            raise SchemaError(f"{path}.folds: cross-validation needs at least two folds")
        if train or ev:
            raise SchemaError(f"{path}: with `folds` every fold is held out once; `train` and `eval` must be empty")
    else:
        if not ev:
            raise SchemaError(f"{path}: `eval` must name at least one split value (or use `folds`)")
    return SplitsSpec(train=train, eval=ev, floor=floor, folds=folds, unused=unused, min_eval_coverage=cov)


def _split_keys_ok(keys: Sequence[str], splits: SplitsSpec) -> bool:
    """True iff every key is a manifest split value or a role word (per-split dict detection)."""
    allowed = set(splits.all_values()) | {"train", "eval", "floor", "folds", "unused"}
    return bool(keys) and all(k in allowed for k in keys)


def _check_per_split_cover(value: Any, path: str, splits: SplitsSpec, *, value_type: type | tuple[type, ...]) -> None:
    """A per-split dict must resolve for every read split (by value or role) and have no stray key."""
    if not isinstance(value, Mapping):
        if not isinstance(value, value_type):
            raise SchemaError(f"{path}: expected {value_type} or a per-split mapping")
        return
    m = _expect_mapping(value, path)
    if not _split_keys_ok(list(m), splits):
        raise SchemaError(f"{path}: per-split keys {sorted(m)} must be manifest split values or role words")
    for k, x in m.items():
        if not isinstance(x, value_type):
            raise SchemaError(f"{path}.{k}: expected {value_type}")
    for s in splits.read_values():
        _per_split(m, s, splits.role_of(s), path)


_LAYOUT_PARAM_KEYS: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "dir": ((), ()),
    "tar_index": ((), ()),
}


def _parse_streams(v: Any, path: str, splits: SplitsSpec, columns: Mapping[str, ColumnSpec], roots: Mapping[str, RootSpec]) -> dict[str, StreamSpec]:
    d = _expect_mapping(v, path)  # may be empty: a benchmark whose fields are all cells (``_parse_fields`` needs one field)
    out: dict[str, StreamSpec] = {}
    for name, s in d.items():
        p = f"{path}.{name}"
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            raise SchemaError(f"{p}: stream names must match [a-z][a-z0-9_]*")
        s = _expect_mapping(s, p)
        _check_keys(s, p, ["kind", "media_column", "root"], ["layout", "layout_params", "required"])
        kind = _enum(s["kind"], f"{p}.kind", STREAM_KINDS)
        media_column = s["media_column"]
        _check_per_split_cover(media_column, f"{p}.media_column", splits, value_type=str)
        root = s["root"]
        _check_per_split_cover(root, f"{p}.root", splits, value_type=str)
        layout = s.get("layout", "dir")
        _check_per_split_cover(layout, f"{p}.layout", splits, value_type=str)
        layout_params = _expect_mapping(s.get("layout_params", {}), f"{p}.layout_params")
        per_split_params = (
            bool(layout_params)
            and _split_keys_ok(list(layout_params), splits)
            and all(isinstance(x, Mapping) for x in layout_params.values())
        )
        spec = StreamSpec(
            kind=kind,
            media_column=dict(media_column) if isinstance(media_column, Mapping) else media_column,
            root=dict(root) if isinstance(root, Mapping) else root,
            layout=dict(layout) if isinstance(layout, Mapping) else layout,
            layout_params={k: (dict(x) if isinstance(x, Mapping) else x) for k, x in layout_params.items()},
            required=_bool(s.get("required", True), f"{p}.required"),
            layout_params_per_split=per_split_params,
        )
        # Cross references, per read split: media column declared with role media and this
        # stream, root declared under data.roots, layout known, layout_params per layout.
        for split in splits.read_values():
            role = splits.role_of(split)
            flat = spec.for_split(split, role)
            col = flat["media_column"]
            cspec = columns.get(col)
            if cspec is None:
                raise SchemaError(f"{p}.media_column: {col!r} (split {split!r}) is not declared in manifest.columns")
            if cspec.role != "media" or name not in cspec.streams():
                raise SchemaError(f"{p}.media_column: manifest.columns.{col} must have {{role: media, stream: {name}}} (or list {name!r} under stream)")
            if flat["root"] not in roots:
                raise SchemaError(f"{p}.root: {flat['root']!r} (split {split!r}) is not declared in data.roots")
            layout_s = flat["layout"]
            if layout_s not in LAYOUTS:
                raise SchemaError(f"{p}.layout: {layout_s!r} is not one of {sorted(LAYOUTS)}")
            req, opt = _LAYOUT_PARAM_KEYS[layout_s]
            _check_keys(flat["layout_params"], f"{p}.layout_params", req, opt)
        out[name] = spec
    return out


def _parse_targets(v: Any, path: str, splits: SplitsSpec, roots: Mapping[str, RootSpec], columns: Mapping[str, ColumnSpec]) -> dict[str, TargetSpec] | None:
    if v is None:
        return None
    d = _expect_mapping(v, path)
    if not d:
        raise SchemaError(f"{path}: use null when there are no dense targets")
    out: dict[str, TargetSpec] = {}
    for split, t in d.items():
        p = f"{path}.{split}"
        if splits.role_of(split) is None:
            raise SchemaError(f"{p}: {split!r} is not a declared split value")
        t = _expect_mapping(t, p)
        _check_keys(t, p, ["kind", "source", "root", "file", "sha256"], ["index_column"])
        root = _str(t["root"], f"{p}.root")
        if root not in roots:
            raise SchemaError(f"{p}.root: {root!r} is not declared in data.roots")
        sha = _sha_or_none(t["sha256"], f"{p}.sha256")
        if sha is None:
            raise SchemaError(f"{p}.sha256: required")
        idx = _str(t.get("index_column"), f"{p}.index_column", nullable=True)
        if idx is not None and idx not in columns:
            raise SchemaError(f"{p}.index_column: {idx!r} is not declared in manifest.columns")
        out[split] = TargetSpec(
            kind=_enum(t["kind"], f"{p}.kind", TARGET_KINDS),
            source=_enum(t["source"], f"{p}.source", TARGET_SOURCES),
            root=root,
            file=_str(t["file"], f"{p}.file"),
            sha256=sha,
            index_column=idx,
        )
    for s in splits.read_values():
        if s not in out:
            raise SchemaError(f"{path}: missing targets for read split {s!r}")
    return out


def _parse_data(v: Any, path: str) -> DataSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["roots"], ["tar_index", "derived"])
    roots_raw = _expect_mapping(d["roots"], f"{path}.roots")
    if not roots_raw:
        raise SchemaError(f"{path}.roots: at least one root is required")
    roots: dict[str, RootSpec] = {}
    for name, r in roots_raw.items():
        p = f"{path}.roots.{name}"
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            raise SchemaError(f"{p}: root names must match [a-z][a-z0-9_]* (they become env var parts)")
        r = _expect_mapping(r, p)
        _check_keys(r, p, ["default"], ["description", "prepare_only"])
        roots[name] = RootSpec(default=_str(r["default"], f"{p}.default"), description=_str(r.get("description"), f"{p}.description", nullable=True),
                               prepare_only=_bool(r["prepare_only"], f"{p}.prepare_only") if "prepare_only" in r else False)
    derived = d.get("derived")
    if derived is not None:
        derived = _str_dict(derived, f"{path}.derived")
        for k, sha in derived.items():
            if not _SHA256_RE.match(sha):
                raise SchemaError(f"{path}.derived.{k}: expected a 64-hex sha256")
    return DataSpec(roots=roots, tar_index=_str(d.get("tar_index"), f"{path}.tar_index", nullable=True), derived=derived)


def _check_region(v: Any, path: str) -> None:
    if not (isinstance(v, list) and len(v) == 4 and all(_is_num(x) for x in v)):
        raise SchemaError(f"{path}: expected [y0, x0, y1, x1] in full-frame fractions")
    y0, x0, y1, x1 = (float(x) for x in v)
    if not (0.0 <= y0 < y1 <= 1.0 and 0.0 <= x0 < x1 <= 1.0):
        raise SchemaError(f"{path}: need 0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1, got {v}")


def _check_protocol(proto: Mapping[str, Any], path: str) -> None:
    """``protocol``: label facts only (windows, frame rules, crops and padding are the bridge's declared decisions)."""
    _json_primitives(proto, path)
    missing = sorted(PROTOCOL_FIXED_KEYS - set(proto))
    if missing:
        raise SchemaError(f"{path}: missing required key(s) {missing}")
    cs = _num(proto["clip_seconds"], f"{path}.clip_seconds", nullable=True)
    if cs is not None and cs <= 0:
        raise SchemaError(f"{path}.clip_seconds: must be > 0 or null")
    td = proto["train_draw"]
    if td is not None:
        td = _expect_mapping(td, f"{path}.train_draw")
        if "drawn_from" in td:
            raise SchemaError(f"{path}.train_draw.drawn_from: prose is not part of the hashed protocol; move it to manifest.upstream.draw")
        _check_keys(td, f"{path}.train_draw", ["cap", "seed", "balance"])
        _int(td["cap"], f"{path}.train_draw.cap", minimum=1)
        _int(td["seed"], f"{path}.train_draw.seed")
        if td["balance"] is not None:
            _enum(td["balance"], f"{path}.train_draw.balance", {"class", "first_label", "session"})
    _enum(proto["views"], f"{path}.views", VIEWS)
    if "target_region" in proto:
        _check_region(proto["target_region"], f"{path}.target_region")


_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _parse_fields(v: Any, path: str, columns: Mapping[str, ColumnSpec]) -> tuple[dict[str, Any], dict[str, CellSpec]]:
    """``fields``: name -> ``{storage: file, kind, media_column, root, layout?, layout_params?, required?}`` (a file the
    reader hands over as ``Sample.streams[name]``) or ``{storage: cell, kind, column}`` (``Sample.cells[name]``).
    Returns the file entries without ``storage`` (parsed by ``_parse_streams``) and the cell fields."""
    d = _expect_mapping(v, path)
    if not d:
        raise SchemaError(f"{path}: at least one field is required (a file or a cell)")
    files: dict[str, Any] = {}
    cells: dict[str, CellSpec] = {}
    for name, e in d.items():
        p = f"{path}.{name}"
        if not isinstance(name, str) or not re.match(r"^[a-z][a-z0-9_]*$", name):
            raise SchemaError(f"{p}: a field name matches [a-z][a-z0-9_]*")
        e = _expect_mapping(e, p)
        storage = _enum(e.get("storage"), f"{p}.storage", STORAGES)
        role = e.get("role", "input")
        if role not in FIELD_ROLES:
            raise SchemaError(f"{p}.role: one of {sorted(FIELD_ROLES)}")
        if storage == "file":
            files[name] = {k: x for k, x in e.items() if k not in ("storage", "role")}
            continue
        _check_keys(e, p, ["storage", "kind", "column"], ["role"])
        kind = _str(e["kind"], f"{p}.kind")
        if not _KIND_RE.match(kind):
            raise SchemaError(f"{p}.kind: {kind!r} must match [a-z][a-z0-9_]* (what the value means: text, options, point_query, ...)")
        col = _str(e["column"], f"{p}.column")
        c = columns.get(col)
        if c is None or c.role != "cell" or c.field != name:
            raise SchemaError(f"{p}.column: {col!r} must be a manifest column with role cell and field {name!r}")
        cells[name] = CellSpec(kind=kind, column=col)
    return files, cells


def _parse_input_contract(v: Any, path: str, streams: Mapping[str, StreamSpec], proto: Mapping[str, Any], columns: Mapping[str, ColumnSpec],
                          cells: Mapping[str, CellSpec] | None = None) -> InputContractSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["handed_over", "info", "facts", "materialize"])
    names = set(streams)
    every = names | set(cells or {})
    ho = _str_dict(d["handed_over"], f"{path}.handed_over")
    if set(ho) != every:
        raise SchemaError(f"{path}.handed_over: keys must be exactly the fields {sorted(every)}, got {sorted(ho)}")
    for k, sentence in ho.items():
        if not sentence.strip():
            raise SchemaError(f"{path}.handed_over.{k}: one non-empty sentence per field")
    info_raw = _expect_mapping(d["info"], f"{path}.info")
    if set(info_raw) != names:
        raise SchemaError(f"{path}.info: keys must be exactly the file fields {sorted(names)} (a cell has no media facts), got {sorted(info_raw)}")
    info: dict[str, tuple[str, ...]] = {}
    for k, keys in info_raw.items():
        keys_t = _str_list(keys, f"{path}.info.{k}")
        kind = streams[k].kind
        bad = sorted(set(keys_t) - INFO_VOCAB[kind])
        if bad:
            raise SchemaError(f"{path}.info.{k}: {bad} are not media facts of a {kind!r} stream (known: {sorted(INFO_VOCAB[kind])})")
        info[k] = keys_t
    facts = _str_list(d["facts"], f"{path}.facts")
    for f in facts:
        if f not in proto:
            raise SchemaError(f"{path}.facts: {f!r} is not a top-level protocol key (facts copy label facts from protocol)")
    mat = d["materialize"]
    if mat is not None:
        mat = _parse_materialize(mat, f"{path}.materialize", columns)
    return InputContractSpec(handed_over=ho, info=info, facts=facts, materialize=mat)


def _parse_tool_pins(v: Any, path: str) -> None:
    """``materialize.tool.pins``: further builds the recipe's bytes depend on (e.g. ffmpeg's libx264), each
    ``{value, probe, match}``: ``probe`` is the command that reveals the build on this host (``{tool}`` = the
    tool's path), ``match`` a regex whose first match in its output must equal ``value`` (``yavih doctor --bench``)."""
    pins = _expect_mapping(v, path)
    for name, pin in pins.items():
        p = _expect_mapping(pin, f"{path}.{name}")
        _check_keys(p, f"{path}.{name}", ["value", "probe", "match"])
        _str(p["value"], f"{path}.{name}.value")
        probe = p["probe"]
        if not (isinstance(probe, list) and probe and all(isinstance(a, str) for a in probe)):
            raise SchemaError(f"{path}.{name}.probe: expected a non-empty list of command arguments")
        try:
            re.compile(_str(p["match"], f"{path}.{name}.match"))
        except re.error as e:
            raise SchemaError(f"{path}.{name}.match: not a regular expression ({e})") from e


def _parse_materialize(v: Any, path: str, columns: Mapping[str, ColumnSpec]) -> dict[str, Any]:
    """A label-required prepare-time transform, every argument pinned. The core checks the envelope only:
    ``kind`` names the folder's recipe (its ``prepare/`` script interprets every other key) and ``tool`` pins the
    program (``name``, ``version_pin``, optional ``pins`` of further builds, and its pinned argument lists). The
    recipe is the pin, never the output bytes: a re-encoded file need not be byte-identical across hosts, so no
    output sha256 ships (``outputs`` is refused; a per-row record of a site's own files is per-site, never shipped).
    Recipe keys are JSON values; a key ending in ``_column`` / ``_columns`` names manifest columns, which must be
    declared."""
    m = _expect_mapping(v, path)
    for k in ("kind", "tool"):
        if k not in m:
            raise SchemaError(f"{path}: missing required key {k!r}")
    if "outputs" in m:
        raise SchemaError(f"{path}.outputs: no output sha256 ships (re-encoded files differ between hosts); the recipe is the pin")
    if not _NAME_RE.match(_str(m["kind"], f"{path}.kind")):
        raise SchemaError(f"{path}.kind: {m['kind']!r} must match [a-z0-9_]+ (the folder recipe's name)")
    for k, x in m.items():
        if k in ("kind", "tool"):
            continue
        _json_primitives(x, f"{path}.{k}")
        if k.endswith("_column"):
            if _str(x, f"{path}.{k}") not in columns:
                raise SchemaError(f"{path}.{k}: {x!r} is not declared in manifest.columns")
        elif k.endswith("_columns"):
            cols = list(x.values()) if isinstance(x, Mapping) else x
            if not (isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols)):
                raise SchemaError(f"{path}.{k}: expected a mapping or list of manifest column names")
            for col in cols:
                if col not in columns:
                    raise SchemaError(f"{path}.{k}: {col!r} is not declared in manifest.columns")
    tool = _expect_mapping(m["tool"], f"{path}.tool")
    for k in ("name", "version_pin"):
        if k not in tool:
            raise SchemaError(f"{path}.tool: missing required key {k!r}")
        _str(tool[k], f"{path}.tool.{k}")
    for k, x in tool.items():
        if k in ("name", "version_pin"):
            continue
        if k == "pins":
            _parse_tool_pins(x, f"{path}.tool.pins")
            continue
        if not (isinstance(x, list) and all(isinstance(a, str) for a in x)) and not isinstance(x, str):
            raise SchemaError(f"{path}.tool.{k}: expected a list of pinned string arguments")
    return json.loads(json.dumps(m))


SAMPLING_MODES: frozenset[str] = frozenset({"optional", "required"})


def _parse_output_contract(v: Any, path: str, readout: str, streams: Mapping[str, StreamSpec], codecs: Mapping[str, Any] | None = None) -> OutputContractSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["shape", "coordinates", "target_frame", "requires"], ["schema", "sampling"])
    if d["shape"] != "fixed":
        raise SchemaError(f"{path}.shape: pinned to 'fixed' (one (N, D) per feature set)")
    if d["coordinates"] != "media":
        raise SchemaError(f"{path}.coordinates: pinned to 'media'")
    tf = d["target_frame"]
    if tf is not None and tf != "full_frame":
        raise SchemaError(f"{path}.target_frame: expected null or 'full_frame'")
    if (tf == "full_frame") != _readout(readout).full_frame:
        full = sorted(n for n, r in _RegistryView("full_frame").items() if r)
        raise SchemaError(f"{path}.target_frame: must be 'full_frame' iff the readout is one of {full} (readout {readout!r})")
    requires = _parse_requires(d["requires"], f"{path}.requires", readout, streams)
    from yavih import codecs as _codecs

    schema = d.get("schema")
    if schema is not None:
        reg = _codecs.registry(codecs)
        problems = _codecs.check_schema(schema, reg)
        if problems:
            raise SchemaError(f"{path}." + f"; {path}.".join(problems))
        schema = {str(k): dict(e) for k, e in schema.items()}
    if not _codecs.has_default_feature_keys(schema):
        used = [k for k in ("pos", "streams") if getattr(requires, k)] + [k for k in ("grid", "t_bins", "annotated_frame") if getattr(requires, k) is not None]
        if used:
            raise SchemaError(f"{path}.requires: {used} measure media positions of the default feature keys (feat, pos); "
                              "a schema without them declares requires {pos: [], streams: [], grid: null, t_bins: null}")
        rdesc = _readout(readout)
        if rdesc.dense or rdesc.requires_grid is True or rdesc.requires_t_bins is True:
            raise SchemaError(f"{path}.schema: scorer {readout!r} reads the default feature keys (feat, pos); keep them in the schema")
    sampling = d.get("sampling", "optional")
    if sampling not in SAMPLING_MODES:
        raise SchemaError(f"{path}.sampling: one of {sorted(SAMPLING_MODES)}")
    return OutputContractSpec(shape="fixed", coordinates="media", target_frame=tf, requires=requires, schema=schema, sampling=str(sampling))


def _parse_external(v: Any, path: str) -> tuple[ExternalSpec, ...]:
    if not isinstance(v, list):
        raise SchemaError(f"{path}: expected a list (empty when the folder uses no original-author code)")
    out: list[ExternalSpec] = []
    for i, e in enumerate(v):
        p = f"{path}[{i}]"
        e = _expect_mapping(e, p)
        _check_keys(e, p, ["name", "pip", "version_pin", "used_for"])
        vals = {k: _str(e[k], f"{p}.{k}") for k in ("name", "pip", "version_pin", "used_for")}
        for k, x in vals.items():
            if not x.strip():
                raise SchemaError(f"{p}.{k}: must be non-empty")
        out.append(ExternalSpec(**vals))
    names = [e.name for e in out]
    if len(set(names)) != len(names):
        raise SchemaError(f"{path}: duplicate package names {sorted(n for n in set(names) if names.count(n) > 1)}")
    return tuple(out)


@dataclass(frozen=True)
class FolderHooks:
    """What ``Benchmark.load`` reads from a folder's ``eval.py``: ``METRICS`` (name -> MetricSpec,
    overlaying the registry for this benchmark only), ``PROTOCOL_KEYS`` (consumption check) and ``SCORER`` (the
    folder's own scorer, registered under the folder's name). ``present`` is whether the folder ships an
    ``eval.py`` at all (a non-empty ``external`` requires one)."""

    present: bool = False
    metrics: Mapping[str, Any] = field(default_factory=dict)
    protocol_keys: frozenset[str] = frozenset()
    sha256: str | None = None
    scorer: Any | None = None
    codecs: Mapping[str, Any] = field(default_factory=dict)


EVAL_FILE = "eval.py"
#: The pre-1.1 name of the hooks file; its presence is refused with a rename hint.
_LEGACY_EVAL_FILE = "probe" + ".py"


def eval_path(folder: str | os.PathLike) -> Path | None:
    """The folder's ``eval.py`` (its scoring hooks), or ``None`` when it ships none. A ``eval.py`` (the pre-1.1
    name) is refused: rename it."""
    folder = Path(folder)
    legacy = folder / _LEGACY_EVAL_FILE
    if legacy.is_file():
        raise SchemaError(f"{legacy}: a folder's scoring hooks live in {EVAL_FILE} (rename {_LEGACY_EVAL_FILE} to {EVAL_FILE})")
    p = folder / EVAL_FILE
    return p if p.is_file() else None


def _resolve_metric_names(names: Sequence[str], readout: str, *, sets: Sequence[str] | None, kinds: Sequence[str], folder: Mapping[str, Any]) -> dict[str, Any]:
    """``REGISTRY`` overlaid by the folder's ``METRICS``: a folder metric may replace a registry
    entry of the same name only if it keeps its ``readouts``, ``shape`` and ``kind``. A metric is allowed when it is
    declared for the scorer or for the scorer's family (``Scorer.families()``)."""
    families = _scorer_families(readout)
    out: dict[str, Any] = {}
    for name in names:
        if name in folder:
            spec = folder[name]
            try:
                base = _registry.resolve(name, sets=sets)
            except _registry.UnknownMetric:
                base = None
            if base is not None and (frozenset(spec.readouts) != frozenset(base.readouts) or spec.shape != base.shape or spec.kind != base.kind):
                raise _registry.MetricNotAllowed(f"eval.py METRICS[{name!r}] replaces a registry metric but changes its readouts/shape/kind")
            if not _registry.allows(spec, families):
                raise _registry.MetricNotAllowed(f"metric {name!r} (eval.py) is not allowed for scorer {readout!r} (allowed: {sorted(spec.readouts)})")
            if spec.kind not in kinds:
                raise _registry.MetricNotAllowed(f"metric {name!r} (eval.py) has kind {spec.kind!r}, expected one of {sorted(kinds)}")
            out[name] = spec
        else:
            out.update(_registry.resolve_names([name], families, sets=sets, kinds=kinds))
    return out


def _parse_metrics(v: Any, path: str, readout: str, label: LabelSpec, splits: SplitsSpec, folder_metrics: Mapping[str, Any] | None = None) -> tuple[MetricsSpec, dict[str, Any]]:
    d = _expect_mapping(v, path)
    _check_keys(d, path, ["headline", "list", "floors", "controls", "chance", "ci"], ["bootstrap", "aggregation", "table", "tolerance"])
    names = _str_list(d["list"], f"{path}.list")
    floors = _str_list(d["floors"], f"{path}.floors")
    controls = _str_list(d["controls"], f"{path}.controls")
    if not names:
        raise SchemaError(f"{path}.list: must name at least one metric")
    sets = list(label.sets) if label.sets else None
    try:
        fm = folder_metrics or {}
        resolved = _resolve_metric_names(names, readout, sets=sets, kinds=("metric", "disclosure"), folder=fm)
        resolved.update(_resolve_metric_names(floors, readout, sets=sets, kinds=("floor",), folder=fm))
        resolved.update(_resolve_metric_names(controls, readout, sets=sets, kinds=("control",), folder=fm))
    except (_registry.UnknownMetric, _registry.MetricNotAllowed) as e:
        raise SchemaError(f"{path}: {e}") from e
    overlap = (set(names) & set(floors)) | (set(names) & set(controls)) | (set(floors) & set(controls))
    if overlap:
        raise SchemaError(f"{path}: {sorted(overlap)} listed under more than one of list/floors/controls")
    headline = _str(d["headline"], f"{path}.headline")
    if headline not in names:
        raise SchemaError(f"{path}.headline: {headline!r} is not in metrics.list")
    chance = _expect_mapping(d["chance"], f"{path}.chance")
    for k, x in chance.items():
        if k not in names:
            raise SchemaError(f"{path}.chance: {k!r} is not in metrics.list")
        _num(x, f"{path}.chance.{k}")
    ci = _expect_mapping(d["ci"], f"{path}.ci")
    for k, x in ci.items():
        if k not in names:
            raise SchemaError(f"{path}.ci: {k!r} is not in metrics.list")
        if x is not None and x not in _registry.CI_KINDS:
            raise SchemaError(f"{path}.ci.{k}: {x!r} is not one of {sorted(_registry.CI_KINDS)} or null")
        folder_ci = getattr((folder_metrics or {}).get(k), "ci", None) or frozenset()
        if x is not None and k not in _registry.CI_ADMISSIBLE[x] and x not in folder_ci:
            raise SchemaError(f"{path}.ci.{k}: no {x} interval is defined for {k!r} ("
                              f"{x} admits {sorted(_registry.CI_ADMISSIBLE[x])} and the eval.py metrics that declare it in MetricSpec.ci)")
    bootstrap = d.get("bootstrap")
    if bootstrap is not None:
        bootstrap = _expect_mapping(bootstrap, f"{path}.bootstrap")
        _check_keys(bootstrap, f"{path}.bootstrap", ["samples", "seed"])
        _int(bootstrap["samples"], f"{path}.bootstrap.samples", minimum=1)
        _int(bootstrap["seed"], f"{path}.bootstrap.seed")
        bootstrap = dict(bootstrap)
    if any(x == "bootstrap_clip" for x in ci.values()) and bootstrap is None:
        raise SchemaError(f"{path}.bootstrap: required when a ci entry is bootstrap_clip")
    agg = d.get("aggregation")
    if agg is not None:
        agg = _expect_mapping(agg, f"{path}.aggregation")
        _check_keys(agg, f"{path}.aggregation", ["over", "report"], ["ci"])
        if agg["over"] != "folds":
            raise SchemaError(f"{path}.aggregation.over: only 'folds' exists")
        rep = _str_list(agg["report"], f"{path}.aggregation.report")
        bad = sorted(set(rep) - {"mean", "pstd", "per_fold"})
        if bad:
            raise SchemaError(f"{path}.aggregation.report: unknown entries {bad}")
        if agg.get("ci") is not None and agg["ci"] not in _registry.CI_KINDS:
            raise SchemaError(f"{path}.aggregation.ci: {agg['ci']!r} is not a CI kind")
        if splits.folds is None:
            raise SchemaError(f"{path}.aggregation: only valid with splits.folds")
        agg = dict(agg)
    elif splits.folds is not None:
        raise SchemaError(f"{path}.aggregation: required with splits.folds")
    table = d.get("table")
    if table is not None:
        table = _str_list(table, f"{path}.table")
        if not table:
            raise SchemaError(f"{path}.table: must name at least one metric (omit the key to show all of metrics.list)")
        bad = [k for k in table if k not in names]
        if bad:
            raise SchemaError(f"{path}.table: {bad} not in metrics.list")
        if headline not in table:
            raise SchemaError(f"{path}.table: must hold the headline {headline!r}")
    tolerance: dict[str, float] = {}
    if d.get("tolerance") is not None:
        tol = _expect_mapping(d["tolerance"], f"{path}.tolerance")
        for k, x in tol.items():
            if k not in names and k not in floors and k not in controls:
                raise SchemaError(f"{path}.tolerance: {k!r} is not in metrics.list, floors or controls")
            v = _num(x, f"{path}.tolerance.{k}")
            if not v >= 0:
                raise SchemaError(f"{path}.tolerance.{k}: must be >= 0")
            tolerance[k] = float(v)
    spec = MetricsSpec(
        headline=headline,
        list=names,
        floors=floors,
        controls=controls,
        chance={k: float(x) for k, x in chance.items()},
        ci=dict(ci),
        bootstrap=bootstrap,
        aggregation=agg,
        table=table,
        tolerance=tolerance,
    )
    return spec, resolved


def _parse_requires(v: Any, path: str, readout: str, streams: Mapping[str, StreamSpec]) -> RequiresSpec:
    d = _expect_mapping(v, path)
    _check_keys(d, path, [], ["pos", "streams", "grid", "t_bins", "annotated_frame"])
    pos = _str_list(d.get("pos", []), f"{path}.pos")
    bad = sorted(set(pos) - {"t", "y", "x"})
    if bad:
        raise SchemaError(f"{path}.pos: unknown axes {bad}")
    st = _str_list(d.get("streams", []), f"{path}.streams")
    for s in st:
        if s not in streams:
            raise SchemaError(f"{path}.streams: {s!r} is not a declared stream")
    grid = d.get("grid")
    if grid is not None:
        grid = _expect_mapping(grid, f"{path}.grid")
        _check_keys(grid, f"{path}.grid", ["h", "w", "min_cell_coverage"])
        _int(grid["h"], f"{path}.grid.h", minimum=1)
        _int(grid["w"], f"{path}.grid.w", minimum=1)
        c = _num(grid["min_cell_coverage"], f"{path}.grid.min_cell_coverage")
        if not (0.0 < c <= 1.0):
            raise SchemaError(f"{path}.grid.min_cell_coverage: must be in (0, 1]")
        grid = dict(grid)
    t_bins = _int(d.get("t_bins"), f"{path}.t_bins", nullable=True, minimum=1)
    r = _readout(readout)
    if r.requires_t_bins is True and t_bins is None:
        raise SchemaError(f"{path}.t_bins: required for readout {readout}")
    if r.requires_grid is True and grid is None:
        raise SchemaError(f"{path}.grid: required for readout {readout}")
    if r.requires_grid is False and r.requires_t_bins is False and (grid is not None or t_bins is not None):
        raise SchemaError(f"{path}: {readout} declares neither grid nor t_bins")
    if r.requires_grid is False and grid is not None:
        raise SchemaError(f"{path}.grid: readout {readout} declares no grid")
    if r.requires_t_bins is False and t_bins is not None:
        raise SchemaError(f"{path}.t_bins: readout {readout} declares no t_bins")
    af = d.get("annotated_frame")
    if af is not None:
        af = _expect_mapping(af, f"{path}.annotated_frame")
        _check_keys(af, f"{path}.annotated_frame", ["fact", "stream", "tol_s"])
        if not isinstance(af["fact"], str) or not af["fact"]:
            raise SchemaError(f"{path}.annotated_frame.fact: a protocol key named in input_contract.facts")
        if af["stream"] not in streams or streams[af["stream"]].kind != "video":
            raise SchemaError(f"{path}.annotated_frame.stream: {af['stream']!r} is not a declared video stream")
        if not (_num(af["tol_s"], f"{path}.annotated_frame.tol_s") > 0):
            raise SchemaError(f"{path}.annotated_frame.tol_s: must be > 0")
        af = dict(af)
    return RequiresSpec(pos=pos, streams=st, grid=grid, t_bins=t_bins, annotated_frame=af)


def _check_readout_params(rp: Any, path: str, readout: str, streams: Mapping[str, StreamSpec], requires: RequiresSpec, proto: Mapping[str, Any] | None = None) -> dict[str, Any]:
    d = _expect_mapping(rp, path)
    allowed = READOUT_PARAM_KEYS[readout]
    if set(d) != set(allowed):
        raise SchemaError(f"{path}: keys for readout {readout!r} must be exactly {sorted(allowed)}, got {sorted(d)}")
    for k, x in d.items():
        if k.endswith("_stream"):
            if x not in streams:
                raise SchemaError(f"{path}.{k}: {x!r} is not a declared stream")
        elif k == "shuffled_control":
            _bool(x, f"{path}.{k}")
        else:
            _int(x, f"{path}.{k}", minimum=0 if k == "shuffle_seed" else 1)
    try:  # the readout's own cross-checks (a message starts with the key it names, e.g. ".T: ...")
        _readout(readout).validate_spec(dict(d), requires, proto or {})
    except ValueError as e:
        raise SchemaError(f"{path}{e}") from e
    return dict(d)


def _check_protocol_keys(proto: Mapping[str, Any], readout: str, resolved: Mapping[str, Any], *, probe_keys: frozenset[str] = frozenset(), facts: Sequence[str] = ()) -> None:
    """Every protocol key the metrics declare is present, and no per-benchmark key is left
    unconsumed — consumed means named by a metric, the readout (``READOUT_PROTOCOL_KEYS``), ``eval.py``'s
    ``PROTOCOL_KEYS`` or ``input_contract.facts``."""
    required = _registry.protocol_keys_for(resolved.values())
    present = frozenset(proto) - PROTOCOL_FIXED_KEYS
    missing = sorted(required - present)
    if missing:
        raise SchemaError(f"protocol: missing key(s) {missing} required by metrics {sorted(n for n, s in resolved.items() if s.protocol_keys & set(missing))}")
    missing_probe = sorted(probe_keys - frozenset(proto))
    if missing_probe:
        raise SchemaError(f"protocol: missing key(s) {missing_probe} declared by eval.py PROTOCOL_KEYS")
    allowed = required | READOUT_PROTOCOL_KEYS[readout] | probe_keys | frozenset(facts)
    undeclared = sorted(present - allowed)
    if undeclared:
        raise SchemaError(
            f"protocol: per-benchmark key(s) {undeclared} are consumed by no registered metric, not by the {readout!r} readout, "
            "not by eval.py PROTOCOL_KEYS and not listed in input_contract.facts"
        )


_TOP_KEYS = [
    "schema_version",
    "name",
    "aliases",
    "title",
    "scorer",
    "protocol_version",
    "changelog",
    "source",
    "budget",
    "manifest",
    "splits",
    "fields",
    "targets",
    "data",
    "protocol",
    "metrics",
    "input_contract",
    "output_contract",
    "external",
    "readout_params",
]


def parse_spec(raw: Any, *, origin: str = "benchmark.yaml", hooks: FolderHooks | None = None) -> BenchmarkSpec:
    """Validate a parsed yaml mapping into a ``BenchmarkSpec``. ``hooks`` are the folder's
    ``eval.py`` ``METRICS`` / ``PROTOCOL_KEYS`` / ``SCORER`` (``load_yaml`` reads them next to the yaml)."""
    hooks = hooks or FolderHooks()
    d = _expect_mapping(raw, "")
    if isinstance(d, Mapping) and d.get("schema_version") == BENCHMARK_SCHEMA_VERSION - 1:
        raise SchemaError(f"schema_version: 2 is the 1.0 schema (`streams:`); rewrite the folder with `yavih bench migrate <folder> --write` "
                          f"(schema {BENCHMARK_SCHEMA_VERSION}: `fields:`; the hashes do not move) ({origin})")
    if isinstance(d, Mapping) and "readout" in d and "scorer" not in d:
        raise SchemaError(f"readout: the key is `scorer` (the scorer the folder is scored by; `yavih bench migrate`) ({origin})")
    _check_keys(d, "", _TOP_KEYS, ["obtain", "header", "capability"])
    sv = _int(d["schema_version"], "schema_version")
    if sv == BENCHMARK_SCHEMA_VERSION - 1:
        raise SchemaError(f"schema_version: {sv} is the 1.0 schema (`streams:`); rewrite the folder with `yavih bench migrate <folder> --write` "
                          f"(schema {BENCHMARK_SCHEMA_VERSION}: `fields:`; the hashes do not move) ({origin})")
    if sv != BENCHMARK_SCHEMA_VERSION:
        raise SchemaError(f"schema_version: {sv} != BENCHMARK_SCHEMA_VERSION {BENCHMARK_SCHEMA_VERSION} ({origin})")
    name = _str(d["name"], "name")
    if not _NAME_RE.match(name):
        raise SchemaError(f"name: {name!r} must match [a-z0-9_]+")
    aliases = _str_list(d["aliases"], "aliases")
    for a in aliases:
        if not _NAME_RE.match(a):
            raise SchemaError(f"aliases: {a!r} must match [a-z0-9_]+")
        if a == name:
            raise SchemaError(f"aliases: {a!r} is the benchmark's own name")
    title = _str(d["title"], "title")
    readout = _enum(d["scorer"], "scorer", _readout_names())
    rdesc = _readout(readout)
    if hooks.scorer is not None and readout != hooks.scorer.name:
        raise SchemaError(f"scorer: {readout!r}, but eval.py declares SCORER {hooks.scorer.name!r} (a folder with its own scorer is scored by it)")
    protocol_version = _int(d["protocol_version"], "protocol_version", minimum=1)
    changelog = _parse_changelog(d["changelog"], "changelog")
    if changelog[-1].version != protocol_version:
        raise SchemaError(f"changelog: the newest entry has version {changelog[-1].version}, protocol_version is {protocol_version}")
    source = _parse_source(d["source"], "source")
    obtain = _parse_obtain(d.get("obtain"), "obtain")
    header = _parse_header(d.get("header"), "header")
    capability = _parse_capability(d.get("capability"), "capability")
    budget = _parse_budget(d["budget"], "budget", readout)
    manifest = _parse_manifest(d["manifest"], "manifest")
    splits = _parse_splits(d["splits"], "splits")
    for s in budget.clips:
        if splits.role_of(s) is None:
            raise SchemaError(f"budget.clips: {s!r} is not a declared split value")
    data = _parse_data(d["data"], "data")
    file_raw, cells = _parse_fields(d["fields"], "fields", manifest.columns)
    streams = _parse_streams(file_raw, "fields", splits, manifest.columns, data.roots)
    for cname, c in manifest.columns.items():
        for st in c.streams():
            if st not in streams:
                raise SchemaError(f"manifest.columns.{cname}.stream: {st!r} is not a declared file field")
        if c.role == "cell" and c.field not in cells:
            raise SchemaError(f"manifest.columns.{cname}.field: {c.field!r} is not a declared cell field")
    needs_index = any(lay == "tar_index" for s in streams.values() for lay in s.layouts())
    if needs_index and data.tar_index is None:
        raise SchemaError("data.tar_index: required by a tar_index layout")
    if not needs_index and data.tar_index is not None:
        raise SchemaError("data.tar_index: no stream uses a tar_index layout")
    targets = _parse_targets(d["targets"], "targets", splits, data.roots, manifest.columns)
    if rdesc.targets != (targets is not None):
        with_targets = sorted(n for n, x in _RegistryView("targets").items() if x)
        raise SchemaError(f"targets: present iff the readout takes its labels from targets ({with_targets})")
    if targets is not None and manifest.label.kind != "none":
        raise SchemaError("manifest.label.kind: must be none when labels come from targets")
    if rdesc.is_probe and not splits.train and not splits.folds:
        raise SchemaError(f"splits.train: scorer {readout!r} trains a probe and needs at least one train split value (or use `folds`)")
    if manifest.label.kind not in rdesc.label_kinds:
        raise SchemaError(f"manifest.label.kind: readout {readout} needs kind {' or '.join(sorted(rdesc.label_kinds))}, got {manifest.label.kind!r}")
    proto = _expect_mapping(d["protocol"], "protocol")
    _check_protocol(proto, "protocol")
    input_contract = _parse_input_contract(d["input_contract"], "input_contract", streams, proto, manifest.columns, cells)
    output_contract = _parse_output_contract(d["output_contract"], "output_contract", readout, streams, hooks.codecs)
    af = output_contract.requires.annotated_frame
    if af is not None:
        if af["fact"] not in input_contract.facts:
            raise SchemaError(f"output_contract.requires.annotated_frame.fact: {af['fact']!r} is not in input_contract.facts (the bridge must be told it)")
        if not isinstance(proto.get(af["fact"]), int | float) or isinstance(proto.get(af["fact"]), bool):
            raise SchemaError(f"output_contract.requires.annotated_frame.fact: protocol.{af['fact']} must be a number of media seconds")
    external = _parse_external(d["external"], "external")
    if external and not hooks.present:
        raise SchemaError("external: a non-empty external block requires the folder's eval.py, which calls the package")
    metrics, resolved = _parse_metrics(d["metrics"], "metrics", readout, manifest.label, splits, hooks.metrics)
    if rdesc.floor_outputs is None and metrics.floors:
        raise SchemaError(f"metrics.floors: scorer {readout!r} declares no floor outputs; declare `floors: []` (the report then greys no row)")
    ext_names = {e.name for e in external}
    for mname, mspec in resolved.items():
        ext = getattr(mspec, "external", None)
        if ext is not None and ext not in ext_names:
            raise SchemaError(f"metrics: {mname!r} is computed by external package {ext!r}, which is not declared under `external`")
    _check_protocol_keys(proto, readout, resolved, probe_keys=hooks.protocol_keys, facts=input_contract.facts)
    readout_params = _check_readout_params(d["readout_params"], "readout_params", readout, streams, output_contract.requires, proto)
    return BenchmarkSpec(
        schema_version=sv,
        name=name,
        aliases=aliases,
        title=title,
        readout=readout,
        protocol_version=protocol_version,
        changelog=changelog,
        source=source,
        budget=budget,
        manifest=manifest,
        splits=splits,
        streams=streams,
        cells=cells,
        targets=targets,
        data=data,
        protocol=dict(proto),
        metrics=metrics,
        input_contract=input_contract,
        output_contract=output_contract,
        external=external,
        readout_params=readout_params,
        raw=d,
        folder_metrics=tuple(sorted(n for n in resolved if n in hooks.metrics)),
        obtain=obtain,
        header=header,
        capability=capability,
    )


def _load_module(path: Path, prefix: str) -> Any:
    """Import a folder file by path, keyed on its content so an edited file is re-imported, never stale."""
    data = path.read_bytes()
    mod_name = f"{prefix}_{path.parent.name}_{hashing.hash12(hashing.sha256_hex(data))}"
    mod = sys.modules.get(mod_name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
    return mod


def folder_hooks(folder: str | os.PathLike) -> FolderHooks:
    """The folder's ``eval.py`` hooks: ``METRICS``, ``PROTOCOL_KEYS`` and ``SCORER`` (registered under the folder's
    name with ``yavih.readouts.registry.register_folder`` so the yaml's ``scorer:`` resolves), its sha256."""
    path = eval_path(folder)
    if path is None:
        return FolderHooks()
    try:
        mod = _load_module(path, "yavih_benchmark_eval")
    except Exception as e:  # noqa: BLE001 - surfaced as a schema error naming the file
        raise SchemaError(f"{path}: cannot import ({type(e).__name__}: {e}); eval.py imports only stdlib, numpy and yavih core at module scope") from e
    metrics = getattr(mod, "METRICS", {}) or {}
    if not isinstance(metrics, Mapping):
        raise SchemaError(f"{path}: METRICS must be a dict[str, MetricSpec]")
    for k, m in metrics.items():
        if not isinstance(k, str) or not all(hasattr(m, a) for a in ("readouts", "kind", "shape", "protocol_keys")):
            raise SchemaError(f"{path}: METRICS[{k!r}] must be a yavih.metrics.MetricSpec")
    keys = getattr(mod, "PROTOCOL_KEYS", frozenset()) or frozenset()
    if isinstance(keys, str) or not all(isinstance(k, str) for k in keys):
        raise SchemaError(f"{path}: PROTOCOL_KEYS must be a frozenset[str]")
    scorer = getattr(mod, "SCORER", None)
    if scorer is not None:
        from yavih.readouts import registry

        if not isinstance(scorer, registry.Scorer):
            raise SchemaError(f"{path}: SCORER must be a yavih.readouts.registry.Scorer")
        if scorer.name != Path(folder).resolve().name:
            raise SchemaError(f"{path}: SCORER.name {scorer.name!r} must be the folder's own name {Path(folder).resolve().name!r}")
        try:
            registry.register_folder(scorer)
        except ValueError as e:
            raise SchemaError(f"{path}: {e}") from e
    from yavih.codecs import folder_codecs

    try:
        codecs = folder_codecs(mod)
    except ValueError as e:
        raise SchemaError(f"{path}: {e}") from e
    return FolderHooks(present=True, metrics=dict(metrics), protocol_keys=frozenset(keys), sha256=hashing.sha256_hex(path.read_bytes()),
                       scorer=scorer, codecs=codecs)


def migrate_schema(raw: Mapping[str, Any]) -> dict[str, Any]:
    """A schema-2 ``benchmark.yaml`` mapping as schema 3: ``streams`` becomes ``fields`` with ``storage: file`` first in
    every entry, and a ``readout`` key becomes ``scorer``. Pure; a schema-3 mapping is returned as a copy. The hashes
    do not move (``hashing.fields_view`` drops the defaults)."""
    d = dict(raw)
    if d.get("schema_version") != 2:
        return d
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k == "schema_version":
            out[k] = 3
        elif k == "streams":
            out["fields"] = {n: {"storage": "file", **dict(e or {})} for n, e in dict(v or {}).items()}
        elif k == "readout":
            out["scorer"] = v
        else:
            out[k] = v
    return out


def migrate_text(text: str) -> str:
    """:func:`migrate_schema` on the yaml text, keeping its comments and layout: ``schema_version: 2`` becomes 3, the
    top-level ``streams:`` key ``fields:``, ``readout:`` ``scorer:``, and ``storage: file`` is inserted as the first
    key of every field entry. ``ValueError`` when the result does not parse to :func:`migrate_schema` of the input."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_fields = False
    for line in lines:
        bare = line.rstrip("\n")
        if re.match(r"^schema_version:\s*2\b", bare):
            line = re.sub(r"^schema_version:\s*2", "schema_version: 3", line)
        elif re.match(r"^readout:", bare):
            line = "scorer:" + line[len("readout:"):]
        elif re.match(r"^streams:", bare):
            line = "fields:" + line[len("streams:"):]
            in_fields = True
            out.append(line)
            continue
        elif in_fields and bare and not bare.startswith((" ", "#")):
            in_fields = False
        if in_fields:
            m = re.match(r"^(  )([a-z][a-z0-9_]*):\s*(#.*)?$", bare)
            if m:
                out.append(line)
                out.append("    storage: file\n")
                continue
        out.append(line)
    new = "".join(out)
    if yaml.safe_load(new) != migrate_schema(yaml.safe_load(text)):
        raise ValueError("the text migration does not reproduce migrate_schema (a flow-style `streams:` block?); edit the file by hand")
    return new


def load_yaml(path: str | os.PathLike) -> BenchmarkSpec:
    """Parse and validate ``benchmark.yaml`` into a ``BenchmarkSpec``, with the hooks of an
    ``eval.py`` next to it. Unknown keys at any level are an error (``SchemaError``). Does not touch the
    manifest, media or derived files."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise SchemaError(f"{p}: not found") from e
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SchemaError(f"{p}: yaml parse error: {e}") from e
    return parse_spec(raw, origin=str(p), hooks=folder_hooks(p.parent))


# ----------------------------------------------------------------------------- manifest / targets


@dataclass
class Manifest:
    """Normalized manifest table: numpy structured arrays + lists, rows in manifest
    file order after filtering to the split (``row`` is the 0-based index within the split).

    ``label_ids``: ``int | None`` for ``single``/``segments``, ``list[int]`` for ``multi`` and
    ``multi_sets`` (train targets), ``None`` for ``box``/``none``. ``boxes`` (``float32 [n, 4]``,
    ``x1 y1 x2 y2``) is set for kind ``box``; ``segments`` (``int64 [n, 2]``) for kind ``segments``.
    """

    split: str | None
    row: np.ndarray  # int64 [n]
    clip_id: list[str]
    splits: list[str]  # the split value per row (``split`` column)
    label_ids: list[list[int] | int | None]
    label_sets: list[dict[str, list[int]]] | None
    media: list[dict[str, str]]  # stream -> media cell
    media_key: list[str | None]
    extra: list[dict[str, Any]]
    boxes: np.ndarray | None = None
    segments: np.ndarray | None = None
    file_row: np.ndarray | None = None  # int64 [n]: index into the whole manifest file

    def __len__(self) -> int:
        return int(self.row.shape[0])

    def index_of(self, clip_id: str) -> int:
        try:
            return self.clip_id.index(clip_id)
        except ValueError as e:
            raise KeyError(clip_id) from e


def _parse_typed(cell: str, ctype: str, where: str) -> Any:
    if cell == "":
        return None
    try:
        if ctype == "int":
            return int(cell)
        if ctype == "float":
            return float(cell)
        if ctype == "json":
            return json.loads(cell)
    except ValueError as e:
        raise ManifestError(f"{where}: cannot parse {cell!r} as {ctype}") from e
    return cell


class _Classes:
    """``classes.csv``: index contiguous, names unique, mids unique when present."""

    def __init__(self, data: bytes, origin: str) -> None:
        text = data.decode("utf-8")
        reader = csv.reader(io.StringIO(text, newline=""))
        try:
            header = next(reader)
        except StopIteration as e:
            raise ManifestError(f"{origin}: empty classes.csv (header required)") from e
        if header[:2] != ["index", "name"] or header not in (["index", "name"], ["index", "name", "mid"]):
            raise ManifestError(f"{origin}: header must be index,name[,mid], got {header}")
        self.has_mid = len(header) == 3
        self.rows: list[dict[str, Any]] = []
        for i, r in enumerate(reader):
            if not r:
                continue
            if len(r) != len(header):
                raise ManifestError(f"{origin}: row {i} has {len(r)} fields, header has {len(header)}")
            try:
                idx = int(r[0])
            except ValueError as e:
                raise ManifestError(f"{origin}: row {i}: index {r[0]!r} is not an int") from e
            if idx != i:
                raise ManifestError(f"{origin}: indices must be contiguous 0..n-1; row {i} has index {idx}")
            if r[1] == "":
                raise ManifestError(f"{origin}: row {i}: empty name")
            d: dict[str, Any] = {"index": idx, "name": r[1]}
            if self.has_mid:
                if r[2] == "":
                    raise ManifestError(f"{origin}: row {i}: empty mid")
                d["mid"] = r[2]
            self.rows.append(d)
        if not self.rows:
            raise ManifestError(f"{origin}: classes.csv has no classes")
        names = [r["name"] for r in self.rows]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ManifestError(f"{origin}: duplicate class names {dup[:5]}")
        self.by_name = {r["name"]: r["index"] for r in self.rows}
        self.by_mid: dict[str, int] = {}
        if self.has_mid:
            mids = [r["mid"] for r in self.rows]
            if len(set(mids)) != len(mids):
                raise ManifestError(f"{origin}: duplicate class mids")
            self.by_mid = {r["mid"]: r["index"] for r in self.rows}

    def __len__(self) -> int:
        return len(self.rows)

    def lookup(self, token: str, class_key: str) -> int | None:
        if class_key == "name":
            return self.by_name.get(token)
        if class_key == "mid":
            return self.by_mid.get(token)
        try:
            idx = int(token)
        except ValueError:
            return None
        return idx if 0 <= idx < len(self.rows) else None


class _Table:
    """The whole manifest file, normalized once; ``Manifest`` views filter by split."""

    def __init__(self, spec: BenchmarkSpec, manifest_bytes: bytes, classes: _Classes | None) -> None:
        self.spec = spec
        self.classes = classes
        self.unknown_labels = 0
        text = manifest_bytes.decode("utf-8")
        reader = csv.reader(io.StringIO(text, newline=""))
        try:
            header = next(reader)
        except StopIteration as e:
            raise ManifestError("manifest: empty file (header required)") from e
        m = spec.manifest
        label = m.label
        label_cols = label.label_columns()
        declared = {m.key, "split", *label_cols, *m.columns}
        if len(set(header)) != len(header):
            raise ManifestError(f"manifest: duplicate header columns in {header}")
        missing = sorted(declared - set(header))
        if missing:
            raise ManifestError(f"manifest: declared column(s) {missing} are missing from the header {header}")
        undeclared = sorted(set(header) - declared)
        if undeclared:
            raise ManifestError(f"manifest: column(s) {undeclared} are not declared in manifest.columns (nothing is silently dropped)")
        col = {h: i for i, h in enumerate(header)}
        key_i, split_i = col[m.key], col["split"]
        media_cols = {name: col[c] for name, c in ((n, c) for n, cs in ((n, s.media_columns()) for n, s in spec.streams.items()) for c in cs)}
        _ = media_cols  # media cells are looked up per split below
        # Every declared column lands in `extra`, typed, media ones included.
        extra_specs = [(name, col[name], c) for name, c in m.columns.items()]
        known_splits = set(spec.splits.all_values())

        self.clip_id: list[str] = []
        self.split: list[str] = []
        self.label_ids: list[list[int] | int | None] = []
        self.label_sets: list[dict[str, list[int]]] | None = [] if label.kind == "multi_sets" else None
        self.media: list[dict[str, str]] = []
        self.media_key: list[str | None] = []
        self.extra: list[dict[str, Any]] = []
        boxes: list[list[float]] = []
        segments: list[list[int]] = []
        seen: set[str] = set()
        for n, r in enumerate(reader):
            if not r:
                continue
            where = f"manifest row {n}"
            if len(r) != len(header):
                raise ManifestError(f"{where}: {len(r)} fields, header has {len(header)}")
            cid = r[key_i]
            if not _CLIP_ID_RE.match(cid):
                raise ManifestError(f"{where}: {m.key} {cid!r} violates the charset [A-Za-z0-9_.#:-]")
            if cid in seen:
                raise ManifestError(f"{where}: duplicate {m.key} {cid!r}")
            seen.add(cid)
            split = r[split_i]
            if split not in known_splits:
                raise ManifestError(f"{where}: split {split!r} is assigned to no role (splits.* / unused)")
            role = spec.splits.role_of(split)
            extra: dict[str, Any] = {}
            for name, i, cspec in extra_specs:
                extra[name] = _parse_typed(r[i], cspec.type, f"{where}: column {name}")
            # media per stream (only the column that applies to this row's split)
            media: dict[str, str] = {}
            if role != "unused":
                for sname, sspec in spec.streams.items():
                    flat = sspec.for_split(split, role)
                    cell = r[col[flat["media_column"]]]
                    if cell == "":
                        if sspec.required:
                            raise ManifestError(f"{where}: empty media cell {flat['media_column']!r} for required stream {sname!r}")
                        continue
                    if not _MEDIA_RE.match(cell):
                        raise ManifestError(f"{where}: media cell {cell!r} violates the charset [A-Za-z0-9_.#:/-]")
                    if cell.startswith("/") or ".." in cell.split("/"):
                        raise ManifestError(f"{where}: media cell {cell!r} must be a relative path without '..'")
                    media[sname] = cell
            # Equal for rows whose media cells are identical (AVE's events of one video); feeds only
            # E_DUPLICATE_CLIPS and the member record, never the bridge.
            media_key: str | None
            if not media:
                media_key = None
            elif len(media) == 1:
                media_key = next(iter(media.values()))
            else:
                media_key = "|".join(f"{k}={media[k]}" for k in sorted(media))
            # labels
            ids: list[int] | int | None = None
            sets_row: dict[str, list[int]] | None = None
            if role != "unused" or label.kind != "none":
                ids, sets_row, box, seg = self._parse_labels(r, col, where, extra)
                if box is not None:
                    boxes.append(box)
                if seg is not None:
                    segments.append(seg)
            elif label.kind == "box":
                boxes.append([0.0, 0.0, 0.0, 0.0])
            self.clip_id.append(cid)
            self.split.append(split)
            self.label_ids.append(ids)
            if self.label_sets is not None:
                self.label_sets.append(sets_row or {})
            self.media.append(media)
            self.media_key.append(media_key)
            self.extra.append(extra)
        if not self.clip_id:
            raise ManifestError("manifest: no rows")
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if label.kind == "box" else None
        self.segments = np.asarray(segments, dtype=np.int64).reshape(-1, 2) if label.kind == "segments" else None
        self.split_arr = np.asarray(self.split)
        if self.unknown_labels and not label.allow_unknown:  # pragma: no cover - raised earlier per cell
            raise ManifestError(f"manifest: {self.unknown_labels} unknown labels")
        for s in spec.splits.read_values():
            if not np.any(self.split_arr == s):
                raise ManifestError(f"manifest: declared split {s!r} has no rows")

    def _tokens(self, cell: str, where: str) -> list[int]:
        label = self.spec.manifest.label
        assert self.classes is not None
        out: list[int] = []
        for tok in cell.split(label.sep):
            if tok == "":
                raise ManifestError(f"{where}: empty label token in {cell!r}")
            idx = self.classes.lookup(tok, label.class_key)
            if idx is None:
                if label.allow_unknown:
                    self.unknown_labels += 1
                    continue
                raise ManifestError(f"{where}: label {tok!r} is not in classes.csv ({label.class_key}); set manifest.label.allow_unknown to drop such cells")
            if idx not in out:
                out.append(idx)
        return out

    def _parse_labels(self, r: list[str], col: Mapping[str, int], where: str, extra: Mapping[str, Any]) -> tuple[Any, dict[str, list[int]] | None, list[float] | None, list[int] | None]:
        label = self.spec.manifest.label
        kind = label.kind
        if kind == "none":
            return None, None, None, None
        assert label.column is not None or kind == "multi_sets"
        if kind == "box":
            cell = r[col[label.column]]  # type: ignore[index]
            if cell == "":
                if label.empty_ok:
                    return None, None, [0.0, 0.0, 0.0, 0.0], None
                raise ManifestError(f"{where}: empty box cell")
            parts = cell.split("|")
            if len(parts) != 4:
                raise ManifestError(f"{where}: box must be x1|y1|x2|y2, got {cell!r}")
            try:
                box = [float(x) for x in parts]
            except ValueError as e:
                raise ManifestError(f"{where}: box {cell!r} has a non-numeric coordinate") from e
            return None, None, box, None
        if kind == "multi_sets":
            sets_row: dict[str, list[int]] = {}
            assert label.sets is not None
            for sname, cols in label.sets.items():
                acc: list[int] = []
                for c in cols:
                    cell = r[col[c]]
                    if cell == "":
                        if not label.empty_ok:
                            raise ManifestError(f"{where}: empty label cell in column {c!r} (set manifest.label.empty_ok)")
                        continue
                    for i in self._tokens(cell, f"{where}: column {c}"):
                        if i not in acc:
                            acc.append(i)
                sets_row[sname] = acc
            if label.train_set is not None:
                ids: list[int] = list(sets_row[label.train_set])
            else:
                cell = r[col[label.column]]  # type: ignore[index]
                if cell == "":
                    if not label.empty_ok:
                        raise ManifestError(f"{where}: empty label cell in column {label.column!r}")
                    ids = []
                else:
                    ids = self._tokens(cell, f"{where}: column {label.column}")
            return ids, sets_row, None, None
        cell = r[col[label.column]]  # type: ignore[index]
        if cell == "":
            if not label.empty_ok:
                raise ManifestError(f"{where}: empty label cell in column {label.column!r} (set manifest.label.empty_ok)")
            return ([] if kind == "multi" else None), None, None, ([0, 0] if kind == "segments" else None)
        toks = self._tokens(cell, f"{where}: column {label.column}")
        if kind == "multi":
            return toks, None, None, None
        # single / segments: exactly one class
        if label.sep in cell or len(toks) != 1:
            if not toks and label.allow_unknown:
                return None, None, None, ([0, 0] if kind == "segments" else None)
            raise ManifestError(f"{where}: kind {kind!r} needs exactly one class per cell, got {cell!r}")
        seg = None
        if kind == "segments":
            a, b = extra.get("event_start"), extra.get("event_end")
            if a is None or b is None:
                raise ManifestError(f"{where}: segments need integer event_start/event_end cells")
            if not (0 <= int(a) < int(b)):
                raise ManifestError(f"{where}: need 0 <= event_start < event_end, got {a}, {b}")
            seg = [int(a), int(b)]
        return toks[0], None, None, seg

    def view(self, split: str | None) -> Manifest:
        if split is None:
            idx = np.arange(len(self.clip_id), dtype=np.int64)
        else:
            idx = np.nonzero(self.split_arr == split)[0].astype(np.int64)
        sel = idx.tolist()
        return Manifest(
            split=split,
            row=np.arange(len(sel), dtype=np.int64),
            clip_id=[self.clip_id[i] for i in sel],
            splits=[self.split[i] for i in sel],
            label_ids=[self.label_ids[i] for i in sel],
            label_sets=[self.label_sets[i] for i in sel] if self.label_sets is not None else None,
            media=[dict(self.media[i]) for i in sel],
            media_key=[self.media_key[i] for i in sel],
            extra=[dict(self.extra[i]) for i in sel],
            boxes=self.boxes[idx] if self.boxes is not None else None,
            segments=self.segments[idx] if self.segments is not None else None,
            file_row=idx,
        )


# Once-per-process cache of verified derived-file digests: {abs_path: (size, mtime_ns, sha256)}.
_TARGET_SHA_CACHE: dict[str, tuple[int, int, str]] = {}


def _file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class TargetReader:
    """Dense targets a CSV cell cannot hold: memmaps ``targets[split].file`` under its
    root, verifies the sha256 once per process (cached in ``data_status.json``), exposes rows in
    manifest row order. numpy only; the probe env needs only the derived root for ``grid``."""

    shape: tuple[int, ...]
    dtype: np.dtype
    kind: str

    def __init__(self, bench: Benchmark, split: str) -> None:
        if bench.spec.targets is None:
            raise NoTargets(f"benchmark {bench.name!r} ({bench.spec.readout}) takes its labels from manifest cells")
        if split not in bench.spec.targets:
            raise NoTargets(f"benchmark {bench.name!r} declares no targets for split {split!r}")
        tspec = bench.spec.targets[split]
        self.spec = tspec
        self.kind = tspec.kind
        self.split = split
        roots = bench.resolved_roots()
        self.path = Path(roots[tspec.root]) / tspec.file
        if not self.path.is_file():
            raise DataNotReady(f"targets for {bench.name}/{split}: {self.path} is missing (root {tspec.root!r} = {roots[tspec.root]}); run prepare/")
        self._verify_sha(bench)
        self._arr = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if self._arr.ndim < 1:
            raise ManifestError(f"targets {self.path}: expected an array with a leading row axis")
        self.dtype = np.dtype(self._arr.dtype)
        manifest = bench.manifest(split)
        if tspec.index_column is not None:
            idx = [e.get(tspec.index_column) for e in manifest.extra]
            if any(i is None for i in idx):
                raise ManifestError(f"targets {bench.name}/{split}: index_column {tspec.index_column!r} has empty cells")
            self._index = np.asarray([int(i) for i in idx], dtype=np.int64)
        else:
            self._index = np.arange(len(manifest), dtype=np.int64)
        n = int(self._arr.shape[0])
        if self._index.size and (int(self._index.max()) >= n or int(self._index.min()) < 0):
            raise ManifestError(f"targets {self.path}: manifest indexes rows up to {int(self._index.max())} but the file has {n}")
        self.shape = (int(self._index.size),) + tuple(int(s) for s in self._arr.shape[1:])

    def _verify_sha(self, bench: Benchmark) -> None:
        st = self.path.stat()
        key = str(self.path.resolve())
        cached = _TARGET_SHA_CACHE.get(key)
        if cached is None:
            status = _read_status_file(data_status_path(bench.name, bench.resolved_roots()))
            entry = ((status or {}).get("targets") or {}).get(key)
            if entry and entry.get("size") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns:
                cached = (st.st_size, st.st_mtime_ns, str(entry["sha256"]))
        if cached is None or cached[0] != st.st_size or cached[1] != st.st_mtime_ns:
            cached = (st.st_size, st.st_mtime_ns, _file_sha256(self.path))
            _TARGET_SHA_CACHE[key] = cached
            _update_status_targets(bench.name, key, cached, bench.resolved_roots())
        else:
            _TARGET_SHA_CACHE[key] = cached
        if cached[2] != self.spec.sha256:
            raise ManifestError(f"targets {self.path}: sha256 {cached[2]} != benchmark.yaml targets.{self.split}.sha256 {self.spec.sha256}")
        self.sha256 = cached[2]

    def rows(self, idx: np.ndarray | list[int]) -> np.ndarray:
        """Targets of the given manifest rows, e.g. float32 ``[len(idx), S, S]``."""
        rows = np.asarray(idx, dtype=np.int64).reshape(-1)
        if rows.size and (int(rows.max()) >= self.shape[0] or int(rows.min()) < 0):
            raise IndexError(f"row index out of range for {self.shape[0]} rows")
        file_rows = self._index[rows]
        order = np.argsort(file_rows, kind="stable")  # memmap reads in file order, then un-permute
        gathered = np.asarray(self._arr[file_rows[order]])
        out = np.empty_like(gathered)
        out[order] = gathered
        return out

    def __len__(self) -> int:
        return self.shape[0]


# ----------------------------------------------------------------------------- data status


def home_benchmarks_dir() -> Path:
    """``$YAVIH_HOME/benchmarks`` (second in the resolution order; written by ``bench fetch``)."""
    return _config.yavih_home() / "benchmarks"


def data_status_path(name: str, roots: Mapping[str, str] | None = None) -> Path:
    """``$YAVIH_HOME/data_status/<name>-<roots_digest12>.json``: keyed by the resolved roots, so
    two sites sharing one HOME keep separate presence and ``verified`` stamps. ``roots=None`` = the
    benchmark's roots as resolved now (``<name>.json`` when they cannot be resolved)."""
    if roots is None:
        try:
            folder = resolve_folder(name)
            roots = Benchmark.from_dir(folder).resolved_roots() if folder is not None else None
        except (BenchmarkError, _config.ConfigError, OSError):
            roots = None
    suffix = f"-{_config.roots_digest(roots)}" if roots else ""
    return _config.yavih_home() / "data_status" / f"{name}{suffix}.json"


def _cache_write_failed(path: Path, e: OSError) -> None:
    """A cache under ``$YAVIH_HOME`` could not be written (read-only HOME): warn once, never fail."""
    _config.warn_once("W_HOME_READONLY", f"cannot write the cache {path} ({e.__class__.__name__}: {e}); presence and digest "
                      "caches are not kept, so every run re-scans; set YAVIH_HOME (or [home] root) to a writable directory")


def _read_status_file(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return d if isinstance(d, dict) else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _update_status_targets(name: str, key: str, entry: tuple[int, int, str], roots: Mapping[str, str] | None = None) -> None:
    path = data_status_path(name, roots)
    status = _read_status_file(path) or {"schema": _STATUS_SCHEMA, "name": name}
    targets = dict(status.get("targets") or {})
    targets[key] = {"size": entry[0], "mtime_ns": entry[1], "sha256": entry[2]}
    status["targets"] = targets
    try:
        _write_json_atomic(path, status)
    except OSError as e:
        _cache_write_failed(path, e)  # the cache is a convenience; the in-process cache already holds the digest


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class DataStatus:
    """Presence per split, media fingerprint and resolved roots (``bench verify``);
    cached in ``$YAVIH_HOME/data_status/<name>.json``."""

    name: str
    manifest_hash: str
    roots: dict[str, str]
    per_split: dict[str, dict[str, Any]]  # split -> {present, nominal, missing_first10}
    media_fingerprint: str | None
    targets_sha_ok: bool | None
    verified: str | None  # UTC timestamp of the last `bench verify`
    path: str | None = None
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)  # verified derived-file digests
    scanned: str | None = None  # UTC timestamp of the presence scan
    media_info_sha256: str | None = None  # sha256 of the site census, copied into sidecars

    def present(self) -> int:
        return sum(int(v["present"]) for v in self.per_split.values())

    def nominal(self) -> int:
        return sum(int(v["nominal"]) for v in self.per_split.values())

    def status_word(self) -> str:
        """``"missing" | "partial 61%" | "verified"`` for ``bench list``. ``verified``
        needs a ``bench verify`` run with full presence; otherwise the percentage is honest even at
        100 % (the media were seen, not verified)."""
        nominal, present = self.nominal(), self.present()
        if nominal == 0 or present == 0:
            return "missing"
        if present >= nominal and self.verified is not None:
            return "verified"
        pct = int(math.floor(100.0 * present / nominal))
        return f"partial {pct}%"

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = _STATUS_SCHEMA
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any], path: Path | None = None) -> DataStatus:
        return cls(
            name=str(d["name"]),
            manifest_hash=str(d["manifest_hash"]),
            roots=dict(d.get("roots") or {}),
            per_split={k: dict(v) for k, v in (d.get("per_split") or {}).items()},
            media_fingerprint=d.get("media_fingerprint"),
            targets_sha_ok=d.get("targets_sha_ok"),
            verified=d.get("verified"),
            path=str(path) if path is not None else d.get("path"),
            targets={k: dict(v) for k, v in (d.get("targets") or {}).items()},
            scanned=d.get("scanned"),
            media_info_sha256=d.get("media_info_sha256"),
        )

    def save(self, path: Path | None = None) -> Path:
        p = path or Path(self.path or data_status_path(self.name, self.roots))
        self.path = str(p)
        _write_json_atomic(p, self.to_json())
        return p


class _TarIndex:
    """``index.json`` v1: ``{"version": 1, "shards": [...], "members": {key: [shard_idx, offset, size]}}``."""

    def __init__(self, path: Path, root: Path) -> None:
        if not path.is_file():
            raise DataNotReady(f"tar index {path} is missing")
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("version") != 1 or not isinstance(d.get("members"), dict) or not isinstance(d.get("shards"), list):
            raise ManifestError(f"tar index {path}: expected {{version: 1, shards: [...], members: {{...}}}}")
        self.path = path
        self.members: dict[str, list] = d["members"]
        # An absolute shard path in the index is used as is (never joined to the root).
        self.shards = [Path(s) if os.path.isabs(s) else root / s for s in d["shards"]]

    def has(self, key: str) -> bool:
        return key in self.members

    def locate(self, key: str) -> tuple[Path, int, int] | None:
        m = self.members.get(key)
        if m is None:
            return None
        shard, offset, size = int(m[0]), int(m[1]), int(m[2])
        if shard >= len(self.shards):
            return None
        return self.shards[shard], offset, size


class _Presence:
    """Media presence per (stream, split) with one ``scandir`` per directory instead of a stat per
    file (fast on network filesystems with hundreds of thousands of files)."""

    def __init__(self, bench: Benchmark, roots: Mapping[str, str]) -> None:
        self.bench = bench
        self.roots = roots
        self._dir_cache: dict[Path, set[str]] = {}
        self._tar_cache: dict[Path, _TarIndex] = {}

    def _listing(self, directory: Path) -> set[str]:
        names = self._dir_cache.get(directory)
        if names is None:
            try:
                with os.scandir(directory) as it:
                    names = {e.name for e in it}
            except (FileNotFoundError, NotADirectoryError, PermissionError):
                names = set()
            self._dir_cache[directory] = names
        return names

    def _tar_index(self, root: Path) -> _TarIndex:
        rel = self.bench.spec.data.tar_index
        assert rel is not None
        path = root / rel
        idx = self._tar_cache.get(path)
        if idx is None:
            idx = _TarIndex(path, root)
            self._tar_cache[path] = idx
        return idx

    def locate(self, stream: str, split: str, cell: str, extra: Mapping[str, Any]) -> list[tuple[Path, int, int | None]] | None:
        """``[(path, offset, size), …]`` of the media bytes behind ``cell`` when present, else ``None``."""
        flat = self.bench.spec.stream_for(stream, split)
        root = Path(self.roots[flat["root"]])
        layout = flat["layout"]
        if layout == "dir":
            p = root / cell
            if p.name in self._listing(p.parent):
                return [(p, 0, None)]
            return None
        if layout == "tar_index":
            loc = self._tar_index(root).locate(cell)
            return [loc] if loc is not None else None
        raise SchemaError(f"unknown layout {layout!r}")  # pragma: no cover - schema refuses it

    def scan(self) -> tuple[dict[str, dict[str, Any]], list[tuple[Path, int, int | None]]]:
        """Per-split presence counts and the ordered list of present media locations (for the
        fingerprint). A row is present iff every ``required`` stream's media is present."""
        per_split: dict[str, dict[str, Any]] = {}
        located: list[tuple[Path, int, int | None]] = []
        self.present_rows: dict[str, list[int]] = {}
        for split in self.bench.spec.splits.read_values():
            man = self.bench.manifest(split)
            present = 0
            rows_here: list[int] = []
            self.present_rows[split] = rows_here
            missing_first10: list[str] = []
            for i in range(len(man)):
                ok = True
                row_locs: list[tuple[Path, int, int | None]] = []
                for sname, sspec in self.bench.spec.streams.items():
                    cell = man.media[i].get(sname)
                    if cell is None:
                        continue
                    loc = self.locate(sname, split, cell, man.extra[i])
                    if loc is None:
                        if sspec.required:
                            ok = False
                        continue
                    row_locs.extend(loc)
                if ok:
                    present += 1
                    rows_here.append(i)
                    located.extend(row_locs)
                elif len(missing_first10) < 10:
                    missing_first10.append(man.clip_id[i])
            per_split[split] = {"present": present, "nominal": len(man), "missing_first10": missing_first10}
        return per_split, located


def media_fingerprint(locations: Sequence[tuple[Path, int, int | None]], manifest_hash: str, sample: int = 256) -> str | None:
    """``sha256`` over the first 64 KiB of a seeded sample (``default_rng(sha256(manifest_hash))``)
    of present media (``bench verify``). ``locations`` are ``(path, offset, size)`` in
    manifest order; tar members are read at their offset. ``None`` when nothing is present."""
    if not locations or sample <= 0:
        return None
    # Dedupe by (path, offset) keeping first occurrence so shared media (AVE rows) count once.
    uniq = list(dict.fromkeys((str(p), int(off), size) for p, off, size in locations))
    rng = np.random.default_rng(hashing.rng_seed(manifest_hash))
    k = min(sample, len(uniq))
    chosen = sorted(int(i) for i in rng.choice(len(uniq), size=k, replace=False))
    h = hashlib.sha256()
    for i in chosen:
        path, offset, size = uniq[i]
        n = FINGERPRINT_BYTES if size is None else min(FINGERPRINT_BYTES, size)
        h.update(f"{os.path.basename(path)}:{offset}\n".encode())
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            h.update(b"<unreadable>")
            continue
        try:
            h.update(os.pread(fd, n, offset))
        except OSError:
            h.update(b"<unreadable>")
        finally:
            os.close(fd)
    return h.hexdigest()


# ----------------------------------------------------------------------------- reader interface


class Benchmark:
    """A loaded benchmark folder. ``protocol_hash`` (= ``H(input_hash, scoring_hash)``) is computed by ``load``
    from the yaml, the manifest, ``reader.py``, ``eval.py`` and ``yavih.versions`` only, so it is identical in every env."""

    name: str
    spec: BenchmarkSpec
    folder: Path
    manifest_hash: str
    protocol_hash: str
    input_hash: str

    def __init__(self, spec: BenchmarkSpec, folder: Path, *, data: dict | None = None) -> None:
        self.spec = spec
        self.name = spec.name
        self.folder = Path(folder)
        self._data_overrides = dict(data or {})
        self.data_overrides: dict[str, Any] = self._data_overrides  # read by yavih.sample._resolve_roots
        # the config schema checks [data.<name>] / YAVIH_DATA__<NAME>__* against this folder's roots too
        _config.register_benchmark(spec.name, spec.aliases, spec.raw["data"]["roots"])
        self._roots: dict[str, str] | None = None
        self._status: DataStatus | None = None
        self._views: dict[str | None, Manifest] = {}
        m = spec.manifest
        manifest_path = self.folder / m.file
        if not manifest_path.is_file():
            raise ManifestError(f"{self.name}: manifest {manifest_path} is missing (run `yavih bench fetch`)")
        try:
            with gzip.open(manifest_path, "rb") as fh:
                manifest_bytes = fh.read()
        except (OSError, EOFError, gzip.BadGzipFile) as e:
            raise ManifestError(f"{self.name}: cannot read gzip manifest {manifest_path}: {e}") from e
        sha = hashing.sha256_hex(manifest_bytes)
        if sha != m.sha256:
            raise ManifestError(f"{self.name}: manifest sha256 {sha} != benchmark.yaml manifest.sha256 {m.sha256} (decompressed bytes)")
        classes_bytes: bytes | None = None
        classes: _Classes | None = None
        if m.classes is not None:
            classes_path = self.folder / m.classes
            if not classes_path.is_file():
                raise ManifestError(f"{self.name}: classes file {classes_path} is missing")
            classes_bytes = classes_path.read_bytes()
            csha = hashing.sha256_hex(classes_bytes)
            if csha != m.classes_sha256:
                raise ManifestError(f"{self.name}: classes sha256 {csha} != benchmark.yaml manifest.classes_sha256 {m.classes_sha256}")
            classes = _Classes(classes_bytes, str(classes_path))
        self.manifest_hash = hashing.manifest_hash(manifest_bytes, classes_bytes)
        self._classes = classes
        self._table = _Table(spec, manifest_bytes, classes)
        self.unknown_labels = self._table.unknown_labels
        if spec.protocol.get("background_index") is not None and classes is not None:
            bi = int(spec.protocol["background_index"])
            if not (0 <= bi < len(classes)):
                raise ManifestError(f"{self.name}: protocol.background_index {bi} is outside classes.csv (0..{len(classes) - 1})")
        # reader.py decides what is handed over (input side) and eval.py decides the scorer, targets, metrics and the
        # external call (scoring side). Neither enters a hash (yavih-protocol-5: identity is the declaration, behaviour
        # is pinned by protocol_version and the conformance fixture); the digest of each one's normalized source is
        # PROVENANCE, recorded in sidecars and results so two numbers under one hash can be traced to the code that
        # produced them.
        reader_py = self.folder / "reader.py"
        eval_py = eval_path(self.folder)
        self.reader_sha256: str | None = hashing.source_sha256(reader_py.read_bytes()) if reader_py.is_file() else None
        self.eval_sha256: str | None = hashing.source_sha256(eval_py.read_bytes()) if eval_py is not None else None
        #: ``{"probe": sha256|None, "reader": sha256|None}``: the folder's source provenance (never identity). The key
        #: ``probe`` names the folder's ``eval.py`` digest (the result-schema key is kept).
        self.folder_source: dict[str, str | None] = {"probe": self.eval_sha256, "reader": self.reader_sha256}
        raw = spec.raw
        self.codecs: dict[str, Any] = self._codec_registry()
        schema = spec.output_contract.schema
        codec_versions = {str(e["type"]): int(self.codecs[str(e["type"])].version) for e in (schema or {}).values()}
        hashes = hashing.protocol_hashes(
            name=spec.name,
            protocol_version=spec.protocol_version,
            protocol=raw["protocol"],
            splits=raw["splits"],
            streams=hashing.fields_view(raw["fields"]),
            input_contract=raw["input_contract"],
            output_contract=raw["output_contract"],
            external=raw["external"],
            metrics=raw["metrics"],
            readout_params=raw["readout_params"],
            targets=raw["targets"],
            manifest=raw["manifest"],
            manifest_hash=self.manifest_hash,
            readout=spec.readout,
            codec_versions=codec_versions,
        )
        self.hashes: hashing.ProtocolHashes = hashes
        self.protocol_hash = hashes.protocol_hash
        #: What feature sets are checked against: a scoring edit leaves it, and the features, alone.
        self.input_hash: str = hashes.input_hash
        self.scoring_hash: str = hashes.scoring_hash
        _hash_history.register(self.name, spec.aliases, spec.changelog)
        # The per-set metrics are concrete specs after expansion at load, overlaid by
        # the folder's eval.py METRICS for this benchmark only.
        sets = list(spec.manifest.label.sets) if spec.manifest.label.sets else None
        hooks = folder_hooks(self.folder) if self.eval_sha256 is not None else FolderHooks()
        self.metric_specs: dict[str, Any] = {}
        for names, kinds in ((spec.metrics.list, ("metric", "disclosure")), (spec.metrics.floors, ("floor",)), (spec.metrics.controls, ("control",))):
            self.metric_specs.update(_resolve_metric_names(names, spec.readout, sets=sets, kinds=kinds, folder=hooks.metrics))
        self._media_info: MediaInfoTable | None | bool = False  # False = not looked up yet

    # -- construction ------------------------------------------------------------------
    @classmethod
    def load(cls, ref: str, *, data: dict | None = None, fetch: bool = False) -> Benchmark:
        """Resolution order (``resolve_folder``): existing directory → ``$YAVIH_HOME/benchmarks/<ref>`` → packaged
        ``yavih/benchmarks/<ref>`` → ``$YAVIH_BENCH_PATH`` → ``yavih.benchmarks`` entry-point packs → alias."""
        folder = resolve_folder(ref)
        if folder is None:
            known = sorted(_candidate_folders())
            raise UnknownBenchmark(f"no benchmark {ref!r}: not a directory, not under {home_benchmarks_dir()}, not packaged, not on "
                                   f"$YAVIH_BENCH_PATH or in an installed benchmark pack, not an alias; known: {known}")
        if fetch:
            folder = Path(bench_fetch(ref)["folder"])
        return cls.from_dir(folder, data=data)

    @classmethod
    def from_dir(cls, path: str | os.PathLike, *, data: dict | None = None) -> Benchmark:
        folder = Path(path)
        yaml_path = folder / "benchmark.yaml"
        if not folder.is_dir() or not yaml_path.is_file():
            raise UnknownBenchmark(f"{folder} is not a benchmark folder (no benchmark.yaml)")
        spec = load_yaml(yaml_path)
        if spec.name != folder.resolve().name:
            raise SchemaError(f"name: {spec.name!r} != folder name {folder.resolve().name!r}")
        _check_home_copy(folder.resolve())
        return cls(spec, folder.resolve(), data=data)

    # -- manifest / classes ------------------------------------------------------------
    def manifest(self, split: str | None = None) -> Manifest:
        if split is not None and self.spec.splits.role_of(split) is None:
            raise KeyError(f"{self.name}: unknown split {split!r}; declared: {sorted(self.spec.splits.all_values())}")
        if split not in self._views:
            self._views[split] = self._table.view(split)
        return self._views[split]

    def n_rows(self, split: str) -> int:
        return len(self.manifest(split))

    def classes(self) -> list[dict[str, Any]]:
        """Rows of ``classes.csv`` (``index, name[, mid]``) in index order."""
        if self._classes is None:
            return []
        return [dict(r) for r in self._classes.rows]

    @property
    def num_classes(self) -> int:
        return len(self._classes) if self._classes is not None else 0

    def roles(self) -> dict[str, list[str]]:
        """``{"train": [...], "eval": [...], "floor": [...], "folds": [...]}``."""
        s = self.spec.splits
        return {"train": list(s.train), "eval": list(s.eval), "floor": list(s.floor), "folds": list(s.folds or ())}

    def targets(self, split: str) -> TargetReader:
        return TargetReader(self, split)

    # -- data roots / status -------------------------------------------------------------
    def resolved_roots(self) -> dict[str, str]:
        """``data.roots`` resolved through ``yavih.config.resolve_data``."""
        if self._roots is None:
            self._roots = _config.resolve_data(self.name, self.spec.raw["data"]["roots"], self._data_overrides, aliases=self.spec.aliases)
        return self._roots

    def data_status(self, *, refresh: bool = False) -> DataStatus:
        """Presence per split, fingerprint and roots, cached in ``$YAVIH_HOME/data_status/<name>.json``.
        The cache is reused when its ``manifest_hash`` and roots match; otherwise the
        presence is re-scanned (no fingerprint: that is ``bench verify``)."""
        if self._status is not None and not refresh:
            return self._status
        roots = self.resolved_roots()
        path = data_status_path(self.name, roots)
        cached = _read_status_file(path)
        if cached is not None and not refresh:
            try:
                st = DataStatus.from_json(cached, path)
            except (KeyError, TypeError, ValueError):
                st = None
            if st is not None and st.manifest_hash == self.manifest_hash and st.roots == roots and st.per_split:
                self._status = st
                return st
        per_split, _ = _Presence(self, roots).scan()
        # A re-scan that finds the same presence under the same roots keeps the last
        # verification (fingerprint, verified stamp); any change voids it.
        same = bool(cached) and cached.get("manifest_hash") == self.manifest_hash and cached.get("roots") == roots and cached.get("per_split") == per_split
        st = DataStatus(
            name=self.name,
            manifest_hash=self.manifest_hash,
            roots=dict(roots),
            per_split=per_split,
            media_fingerprint=cached.get("media_fingerprint") if same else None,
            targets_sha_ok=cached.get("targets_sha_ok") if same else None,
            verified=cached.get("verified") if same else None,
            path=str(path),
            targets=dict((cached or {}).get("targets") or {}),
            scanned=_utc_now(),
            media_info_sha256=self.media_info().sha256 if self.media_info() is not None else None,
        )
        try:
            st.save(path)
        except OSError as e:
            _cache_write_failed(path, e)
        self._status = st
        return st

    # -- reader ------------------------------------------------------------------------
    def reader(
        self,
        split: str,
        *,
        shard: tuple[int, int] | None = None,
        batch_size: int = 1,
        rows: range | None = None,
        canaries: int = 0,
        canary_seed: int = 0,
        align: int | None = None,
    ) -> Iterator[Sample]:
        """Yields the Samples of shard range ``i`` of ``plan_shards(n_rows, align or batch_size, n)`` in ascending row
        order, then ``min(canaries, n_in_shard // 4)`` canary copies. Each Sample hands over
        the stored media (``RawStream``) and facts, never a decoded or windowed signal. Delegates to the
        folder's ``reader.py`` (``iter_samples``) when present, else to ``yavih.sample.iter_samples``.
        ``align``: the batch alignment of the split's frozen layout (``features/<split>/_layout.json``) when it
        differs from ``batch_size``."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if canaries < 0:
            raise ValueError("canaries must be >= 0")
        role = self.spec.splits.role_of(split)
        if role is None or role == "unused":
            raise KeyError(f"{self.name}: split {split!r} is not read (role {role!r})")
        n = self.n_rows(split)
        shard_index, n_shards = 0, 1
        if shard is not None:
            if rows is not None:
                raise ValueError("pass either shard or rows, not both")
            i, k = int(shard[0]), int(shard[1])
            if not (0 <= i < k):
                raise ValueError(f"shard index {i} out of range for {k} shards")
            from yavih.io.shards import plan_shards

            lo, hi = plan_shards(n, int(align or batch_size), k)[i]
            rows = range(lo, hi)
            shard_index, n_shards = i, k
        elif rows is None:
            rows = range(0, n)
        else:
            if rows.step != 1 or rows.start < 0 or rows.stop > n:
                raise ValueError(f"rows must be a step-1 range within [0, {n})")
        module = self._folder_reader()
        if module is not None:
            yield from self._call_folder_reader(module, split, rows, batch_size, canaries, canary_seed, (shard_index, n_shards))
            return
        yield from self._generic_reader(split, rows, batch_size, canaries, canary_seed, shard_index)

    def _call_folder_reader(self, module: Any, split: str, rows: range, batch_size: int, canaries: int, canary_seed: int, shard: tuple[int, int]) -> Iterator[Sample]:
        """The folder reader's ``iter_samples`` call with ``shard=(i, n)`` as a keyword (the canary draw is seeded with
        the shard index)."""
        return module.iter_samples(self, split, rows, batch_size, canaries, canary_seed, shard=shard)

    def _folder_reader(self) -> Any | None:
        path = self.folder / "reader.py"
        if not path.is_file():
            return None
        mod = _load_module(path, "yavih_benchmark_reader")
        if not callable(getattr(mod, "iter_samples", None)):
            raise ImportError(f"{path} must define iter_samples(bench, split, rows, batch_size, canaries, canary_seed, *, shard=None, roots=None)")
        return mod

    def _codec_registry(self) -> dict[str, Any]:
        """Every codec this folder's schema may name: the built-ins, entry points and the folder's ``eval.py``
        ``CODECS``."""
        from yavih.codecs import registry

        hooks = folder_hooks(self.folder) if eval_path(self.folder) is not None else FolderHooks()
        return registry(hooks.codecs)

    def output_schema(self) -> dict[str, tuple[dict[str, Any], Any]]:
        """``{key: (entry, codec)}`` of the folder's output schema (the default feature matrix when none is declared)."""
        from yavih.codecs import resolve_schema

        return resolve_schema(self.spec.output_contract.resolved_schema, self.codecs)

    def eval_module(self) -> Any | None:
        """The folder's ``eval.py`` as a module (its hooks), or ``None`` when the folder ships none."""
        path = eval_path(self.folder)
        return _load_module(path, "yavih_benchmark_eval") if path is not None else None

    def media_info(self) -> MediaInfoTable | None:
        """The site census ``$YAVIH_HOME/media_info/<name>-<manifest_hash12>.jsonl.gz``, or ``None``
        when absent. Loaded once per Benchmark; facts are site facts and never hashed."""
        if self._media_info is False:
            path = media_info_path(self.name, self.manifest_hash)
            self._media_info = MediaInfoTable.load(path) if path.is_file() else None
        return self._media_info  # type: ignore[return-value]

    def view(self) -> Any:
        """What a bridge receives (``BenchmarkView``): name, aliases, readout, protocol version and hash,
        ``{stream: kind}``, the input/output contracts and the facts — no manifest, labels, splits or roots."""
        raw = self.spec.raw
        fields_ = {
            "name": self.name,
            "aliases": list(self.spec.aliases),
            "readout": self.spec.readout,
            "protocol_version": self.spec.protocol_version,
            "protocol_hash": self.protocol_hash,
            "input_hash": self.input_hash,
            "streams": {n: s.kind for n, s in self.spec.streams.items()},
            "input_contract": json.loads(json.dumps(raw["input_contract"])),
            "output_contract": json.loads(json.dumps(raw["output_contract"])),
            "facts": json.loads(json.dumps({k: self.spec.protocol[k] for k in self.spec.input_contract.facts})),
            "fields": {**{n: {"storage": "file", "kind": s.kind} for n, s in self.spec.streams.items()},
                       **{n: {"storage": "cell", "kind": c.kind} for n, c in self.spec.cells.items()}},
        }
        from yavih.bridge import BenchmarkView

        return BenchmarkView(**fields_)

    def canary_rows(self, split: str, rows: range, batch_size: int, canaries: int, canary_seed: int, shard_index: int = 0) -> list[int]:
        """The canary rows of a shard: ``min(K, n // 4)`` rows drawn without replacement
        by ``default_rng(sha256(f"{manifest_hash}:{split}:{i}:{canary_seed}"))`` from rows at least
        ``2 * batch_size`` positions before the end of the shard, in draw order
        (``yavih.sample.canary_rows``; the reader yields them in this order)."""
        from yavih.sample import canary_rows

        return canary_rows(self.n_rows(split), rows, batch_size, canaries, manifest_hash=self.manifest_hash, split=split, shard_index=shard_index, canary_seed=canary_seed)

    def _generic_reader(self, split: str, rows: range, batch_size: int, canaries: int, canary_seed: int, shard_index: int) -> Iterator[Sample]:
        """The generic reader is ``yavih.sample.iter_samples`` over this benchmark's normalized
        manifest, streams, input contract and resolved roots."""
        from yavih.sample import iter_samples

        return iter_samples(self, split, rows, batch_size, canaries, canary_seed, shard=(shard_index, 1), roots=self._roots_or_none())

    def sample(self, split: str, row: int, *, manifest: Manifest | None = None) -> Sample:
        """The ``Sample`` of one manifest row via ``yavih.sample.build_sample``: stored media and
        facts; the bridge never sees labels."""
        from yavih.sample import build_sample

        man = manifest or self.manifest(split)
        if not (0 <= row < len(man)):
            raise IndexError(f"row {row} out of range for split {split!r} ({len(man)} rows)")
        return build_sample(self, split, man, int(row), self.resolved_roots())

    def _roots_or_none(self) -> dict[str, str] | None:
        """``resolved_roots()`` or ``None`` when a root does not resolve (``yavih.sample`` then resolves again and
        raises when iteration starts)."""
        try:
            return self.resolved_roots()
        except _config.ConfigError:
            return None

    def __repr__(self) -> str:
        return f"Benchmark({getattr(self, 'name', '?')!r})"


# ----------------------------------------------------------------------------- resolution / listing


def _home_benchmarks() -> list[str]:
    root = home_benchmarks_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "benchmark.yaml").is_file())


def packaged_benchmarks() -> list[str]:
    """Names of the benchmark folders shipped inside the wheel (``yavih/benchmarks/<name>/``),
    found through ``importlib.resources``."""
    from yavih.benchmarks import packaged_names

    return packaged_names()


def _external_folders() -> dict[str, Path]:
    """``$YAVIH_BENCH_PATH`` and entry-point packs, minus names a packaged folder already has (never shadowed)."""
    from yavih.benchmarks import external_folders, packaged_folder

    return {n: f for n, f in external_folders().items() if packaged_folder(n) is None}


def _candidate_folders() -> dict[str, Path]:
    """``name → folder`` over ``$YAVIH_HOME/benchmarks`` (wins), the packaged folders, then ``$YAVIH_BENCH_PATH`` and
    entry-point packs (which never shadow a packaged name)."""
    from yavih.benchmarks import packaged_folder

    out: dict[str, Path] = dict(_external_folders())
    for name in packaged_benchmarks():
        p = packaged_folder(name)
        if p is not None:
            out[name] = p
    for name in _home_benchmarks():
        out[name] = home_benchmarks_dir() / name
    return out


def _aliases_of(folder: Path) -> tuple[str, tuple[str, ...]]:
    """``(name, aliases)`` from a folder's yaml without full validation (fast alias scan)."""
    try:
        with open(folder / "benchmark.yaml", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return folder.name, ()
    if not isinstance(d, Mapping):
        return folder.name, ()
    aliases = d.get("aliases") or ()
    if not isinstance(aliases, list):
        aliases = ()
    return str(d.get("name", folder.name)), tuple(str(a) for a in aliases)


def resolve_folder(ref: str) -> Path | None:
    """Resolution order: existing directory → ``$YAVIH_HOME/benchmarks/<ref>`` → packaged →
    ``$YAVIH_BENCH_PATH`` (colon-separated, first wins) → ``yavih.benchmarks`` entry-point packs → alias over all of
    them. An out-of-tree folder never shadows a packaged name. ``None`` when nothing matches."""
    from yavih.benchmarks import packaged_folder

    p = Path(ref).expanduser()
    if p.is_dir() and (p / "benchmark.yaml").is_file():
        return p
    if "/" not in ref and os.sep not in ref:
        home = home_benchmarks_dir() / ref
        if home.is_dir() and (home / "benchmark.yaml").is_file():
            return home
        packaged = packaged_folder(ref)
        if packaged is not None:
            return packaged
        external = _external_folders().get(ref)
        if external is not None:
            return external
        for _name, folder in _candidate_folders().items():
            name, aliases = _aliases_of(folder)
            if ref in aliases:
                return folder
    return None


def list_benchmarks(*, data: dict | None = None, scan: bool = True) -> list[dict[str, Any]]:
    """Rows for ``yavih bench list``: ``{name, aliases, readout, clips: {split: n},
    media_gb, source_kind, status}``. ``status`` comes from the cached ``data_status.json`` (a
    presence scan when ``scan`` and no cache exists; ``"missing"`` when the roots cannot resolve).
    A folder that fails to load is reported with its error instead of hiding it."""
    rows: list[dict[str, Any]] = []
    for name, folder in sorted(_candidate_folders().items()):
        try:
            bench = Benchmark.from_dir(folder, data=data)
        except BenchmarkError as e:
            rows.append({"name": name, "aliases": [], "scorer": None, "capability": [], "clips": {}, "media_gb": None, "source_kind": None, "obtain": None, "status": "invalid", "error": f"{type(e).__name__}: {e}", "folder": str(folder)})
            continue
        status = _status_word(bench, scan=scan)
        rows.append(
            {
                "name": bench.name,
                "aliases": list(bench.spec.aliases),
                "scorer": bench.spec.readout,
                "capability": list(bench.spec.capability),
                "clips": {s: bench.n_rows(s) for s in bench.spec.splits.read_values()},
                "media_gb": bench.spec.budget.media_gb,
                "source_kind": bench.spec.source.kind,
                "obtain": asdict(bench.spec.obtain) if bench.spec.obtain else None,
                "status": status,
                "folder": str(folder),
            }
        )
    return rows


def _status_word(bench: Benchmark, *, scan: bool) -> str:
    try:
        if not scan:
            cached = _read_status_file(data_status_path(bench.name, bench.resolved_roots()))
            if cached is None:
                return "missing"
            st = DataStatus.from_json(cached)
            return st.status_word() if st.manifest_hash == bench.manifest_hash else "missing"
        return bench.data_status().status_word()
    except (_config.ConfigError, DataNotReady, ManifestError, OSError):
        return "missing"


def bench_info(ref: str, *, readme_only: bool = False, setup: bool = False, data: dict | None = None, scan: bool = True) -> dict[str, Any]:
    """``yavih bench info``: README text (``setup.md`` with ``setup``) + ``{roles: {role: {nominal,
    present}}, floors, metrics, source, budget, external, history}``."""
    bench = Benchmark.load(ref, data=data)
    readme_path = bench.folder / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    setup_path = bench.folder / "setup.md"
    setup_text = setup_path.read_text(encoding="utf-8") if setup_path.is_file() else ""
    if setup:
        return {"name": bench.name, "setup": setup_text}
    if readme_only:
        return {"name": bench.name, "readme": readme}
    status: DataStatus | None = None
    if scan:
        try:
            status = bench.data_status()
        except (_config.ConfigError, DataNotReady, OSError):
            status = None
    roles: dict[str, dict[str, Any]] = {}
    for role, values in bench.roles().items():
        if not values:
            continue
        nominal = sum(bench.n_rows(s) for s in values)
        present = sum(int(status.per_split.get(s, {}).get("present", 0)) for s in values) if status else None
        roles[role] = {"splits": list(values), "nominal": nominal, "present": present}
    return {
        "name": bench.name,
        "aliases": list(bench.spec.aliases),
        "title": bench.spec.title,
        "capability": list(bench.spec.capability),
        "readout": bench.spec.readout,
        "protocol_version": bench.spec.protocol_version,
        "manifest_hash": bench.manifest_hash,
        "protocol_hash": bench.protocol_hash,
        "readme": readme,
        "roles": roles,
        "floors": list(bench.spec.metrics.floors),
        "controls": list(bench.spec.metrics.controls),
        "metrics": {"headline": bench.spec.metrics.headline, "list": list(bench.spec.metrics.list), "chance": dict(bench.spec.metrics.chance), "ci": dict(bench.spec.metrics.ci)},
        "source": asdict(bench.spec.source),
        "obtain": asdict(bench.spec.obtain) if bench.spec.obtain else None,
        "budget": asdict(bench.spec.budget),
        "external": [asdict(e) for e in bench.spec.external],
        "input_contract": json.loads(json.dumps(bench.spec.raw["input_contract"])),
        "output_contract": json.loads(json.dumps(bench.spec.raw["output_contract"])),
        "history": [asdict(e) for e in bench.spec.changelog],
        "status": status.status_word() if status else None,
        "data_roots": dict(status.roots) if status else None,
        "folder": str(bench.folder),
    }


_FOLDER_FILES = ("benchmark.yaml", "README.md", "setup.md", "manifest.csv.gz", "classes.csv", "reader.py", "eval.py", "bridge_template.py", "make_manifest.py")
#: ``$YAVIH_HOME/benchmarks/<name>/.yavih_fetch.json``: the yavih version, protocol hash and per-file sha256 of the
#: source a copy was synced from, so a copy left behind by an older yavih is detected instead of silently winning.
FETCH_STAMP = ".yavih_fetch.json"
_HOME_CHECKED: dict[str, tuple[str, str] | None] = {}


def _folder_files(folder: Path) -> list[str]:
    names = [f for f in _FOLDER_FILES if (folder / f).is_file()]
    prep = folder / "prepare"
    if prep.is_dir():
        names += sorted(p.relative_to(folder).as_posix() for p in prep.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return names


def _home_copy_problem(folder: Path) -> tuple[str, str] | None:
    """``(code, message)`` when the ``$YAVIH_HOME`` copy ``folder`` no longer matches the packaged folder of the
    same name (``E_HOME_BENCH_STALE``) or carries no fetch stamp (``W_HOME_BENCH_UNSTAMPED``); ``None`` when it
    matches or has no packaged counterpart (a user's own folder)."""
    from yavih.benchmarks import packaged_folder
    from yavih.versions import __version__

    pkg = packaged_folder(folder.name)
    if pkg is None or pkg.resolve() == folder.resolve():
        return None
    fix = f"run `yavih bench fetch {folder.name} --refresh`, or delete {folder}"
    stamp = _read_status_file(folder / FETCH_STAMP)
    if stamp is None:
        return ("W_HOME_BENCH_UNSTAMPED", f"{folder} overrides the packaged {folder.name} folder but carries no {FETCH_STAMP} (copied by an "
                f"older yavih); it cannot be checked against yavih {__version__}'s folder; {fix}")
    synced = dict(stamp.get("files") or {})
    changed = sorted(n for n in set(synced) | set(_folder_files(pkg)) if not (pkg / n).is_file() or synced.get(n) != _file_sha256(pkg / n))
    if changed:
        return ("E_HOME_BENCH_STALE", f"{folder} was copied from yavih {stamp.get('yavih_version')} (protocol_hash "
                f"{str(stamp.get('protocol_hash'))[:12]}) and overrides the packaged folder, which has changed since in "
                f"{changed[:6]}{' …' if len(changed) > 6 else ''}; the stale copy would be used silently. {fix} "
                "(YAVIH_LENIENT=1 uses it anyway)")
    return None


def _check_home_copy(folder: Path) -> None:
    """Refuse (strict) or warn (lenient) when a ``$YAVIH_HOME/benchmarks`` copy is stale."""
    try:
        if folder.parent.resolve() != home_benchmarks_dir().resolve():
            return
    except OSError:
        return
    key = str(folder)
    if key not in _HOME_CHECKED:
        _HOME_CHECKED[key] = _home_copy_problem(folder)
    problem = _HOME_CHECKED[key]
    if problem is None:
        return
    code, message = problem
    if code.startswith("E_"):
        try:
            lenient = _config.load().lenient
        except _config.ConfigError:
            lenient = os.environ.get("YAVIH_LENIENT", "") not in ("", "0")
        if not lenient:
            raise _config.ConfigError(message)
        code = "W_HOME_BENCH_STALE"
    _config.warn_once(code, message, where=str(folder))


def bench_fetch(ref: str, *, refresh: bool = False) -> dict[str, Any]:
    """``yavih bench fetch``: makes the frozen manifest available under ``$YAVIH_HOME``
    and verifies its sha256. yavih never downloads data: the manifest ships in the wheel, so fetch
    syncs the packaged (or referenced) folder's files into ``$YAVIH_HOME/benchmarks/<name>``: absent
    files are copied, files still equal to what the last fetch copied are refreshed when the source
    changed, locally edited files are kept (``kept``) unless ``refresh``. It writes :data:`FETCH_STAMP`
    and re-checks ``manifest.sha256``/``classes_sha256`` on the copy."""
    from yavih.benchmarks import packaged_folder
    from yavih.versions import __version__

    folder = resolve_folder(ref)
    if folder is None:
        raise UnknownBenchmark(f"no benchmark {ref!r} to fetch")
    spec = load_yaml(folder / "benchmark.yaml")
    dest = home_benchmarks_dir() / spec.name
    pkg = packaged_folder(spec.name)
    if folder.resolve() == dest.resolve() and pkg is not None:
        folder = pkg  # a name resolving to the home copy syncs from the packaged folder
    copied: list[str] = []
    kept: list[str] = []
    if folder.resolve() != dest.resolve():
        dest.mkdir(parents=True, exist_ok=True)
        stamp = _read_status_file(dest / FETCH_STAMP) or {}
        synced = dict(stamp.get("files") or {})
        files: dict[str, str] = {}
        for rel in _folder_files(folder):
            src, dst = folder / rel, dest / rel
            sha = _file_sha256(src)
            files[rel] = sha
            if dst.is_file():
                have = _file_sha256(dst)
                if have == sha:
                    continue
                if not refresh and have != synced.get(rel):
                    kept.append(rel)  # edited after the last fetch (or unstamped): never overwritten silently
                    continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
        bench = Benchmark(load_yaml(dest / "benchmark.yaml"), dest.resolve())  # raises ManifestError on a sha mismatch
        payload = {"yavih_version": __version__, "protocol_hash": bench.protocol_hash, "source_folder": str(folder),
                   "fetched": _utc_now(), "files": files, "kept": kept}
        _write_json_atomic(dest / FETCH_STAMP, payload)
        _HOME_CHECKED.pop(str(dest.resolve()), None)
        _HOME_CHECKED.pop(str(dest), None)
    bench = Benchmark.from_dir(dest)  # raises ManifestError on a sha mismatch
    return {"name": bench.name, "folder": str(dest), "source_folder": str(folder), "copied": copied, "kept": kept,
            "manifest_sha_ok": True, "manifest_hash": bench.manifest_hash}


def _sample_picks(bench: Benchmark, n: int, status: DataStatus, present_rows: Mapping[str, Sequence[int]] | None = None) -> list[tuple[str, int]]:
    """A seeded sample of up to ``n`` PRESENT rows over the read splits (``present_rows`` from the presence
    scan; on a partially staged site an absent row says nothing about media health). With every row present
    the picks are the same as sampling the whole split."""
    rng = np.random.default_rng(hashing.rng_seed(f"{bench.manifest_hash}:decode_sample"))
    picks: list[tuple[str, int]] = []
    splits = bench.spec.splits.read_values()
    for split in splits:
        if int(status.per_split.get(split, {}).get("present", 0)) == 0:
            continue
        rows = list(present_rows[split]) if present_rows is not None and split in present_rows else list(range(len(bench.manifest(split))))
        if not rows:
            continue
        for i in rng.choice(len(rows), size=min(len(rows), max(1, n // max(1, len(splits)))), replace=False):
            picks.append((split, int(rows[int(i)])))
    return picks[:n]


def _read_sample(bench: Benchmark, picks: Sequence[tuple[str, int]]) -> dict[str, Any]:
    """``bench verify`` ``read``: the stored bytes of every sampled row (every stream, every frame)."""
    from yavih.media import MediaMissing

    ok = failed = 0
    failures: list[dict[str, str]] = []
    for split, r in picks:
        try:
            s = bench.sample(split, r)
            for st in s.streams.values():
                st.bytes()
            ok += 1
        except MediaMissing as e:
            failed += 1
            if len(failures) < 10:
                failures.append({"clip_id": bench.manifest(split).clip_id[r], "error": str(e)})
    return {"ok": ok, "failed": failed, "failures": failures}


def _decode_sample(bench: Benchmark, picks: Sequence[tuple[str, int]]) -> dict[str, Any]:
    """``bench verify`` ``decode_sample``: decode the sampled rows with the optional ``yavih.media.decode``
    helpers as a media-health check (never a protocol step). Skipped without ``[media]``."""
    if not picks:
        return {"ok": 0, "failed": 0, "skipped": "decode sample disabled or no media present"}
    from yavih.media import DecoderMissing

    try:
        from yavih.media import decode as _decode

        lib = {k: _decode.decoder_info(k) for k in sorted({s.kind for s in bench.spec.streams.values()})}
    except DecoderMissing as e:
        return {"ok": 0, "failed": 0, "skipped": f"no decoder installed: {e}"}
    ok = failed = 0
    failures: list[dict[str, str]] = []
    for split, r in picks:
        try:
            s = bench.sample(split, r)
            for st in s.streams.values():
                if st.kind in ("video", "audio") and st.info.get("has_track") is False:
                    continue  # a file without this track: nothing to decode, and not a failure
                if st.kind == "video":
                    _decode.frames_at(st, [0.0])
                elif st.kind == "audio":
                    _decode.audio(st)
                else:
                    _decode.image(st)
            ok += 1
        except DecoderMissing as e:
            return {"ok": ok, "failed": failed, "skipped": f"no decoder installed: {e}", "failures": failures, "lib": lib}
        except Exception as e:  # MediaMissing, DecodeError, …
            failed += 1
            if len(failures) < 10:
                failures.append({"clip_id": bench.manifest(split).clip_id[r], "error": f"{type(e).__name__}: {e}"})
    return {"ok": ok, "failed": failed, "failures": failures, "lib": lib}


def _derived_sha_ok(bench: Benchmark, roots: Mapping[str, str]) -> tuple[bool | None, list[str]]:
    """``data.derived`` (``{relative file: sha256}``) checked under the root that holds each file (the
    ``derived_root`` root first)."""
    derived = bench.spec.data.derived or {}
    if not derived:
        return None, []
    order = sorted(roots, key=lambda k: (k != "derived_root", k))
    errors: list[str] = []
    for rel, sha in sorted(derived.items()):
        path = next((Path(roots[k]) / rel for k in order if (Path(roots[k]) / rel).is_file()), None)
        if path is None:
            errors.append(f"{rel}: missing")
        elif _file_sha256(path) != sha:
            errors.append(f"{rel}: sha256 mismatch")
    return not errors, errors[:10]


def _external_status(bench: Benchmark) -> dict[str, dict[str, Any]]:
    """``{name: {ok, version, version_pin, pip}}`` per ``external`` entry, without importing it."""
    from yavih.external import status

    return {e.name: status(e.name, e.pip, e.version_pin) for e in bench.spec.external}


def bench_verify(ref: str, *, census: bool = False, decode_sample: int = 32, fingerprint_sample: int = 256, data: dict | None = None) -> dict[str, Any]:
    """``yavih bench verify``: ``{manifest_sha_ok, per_split, read, decode_sample, labels_ok,
    targets_sha_ok, derived_sha_ok, curation_invariants_ok, media_info, external,
    media_fingerprint, data_roots, data_status_path}``. ``census`` writes the media-info table (needs
    ``[media]``). Writes ``data_status.json`` with the fingerprint, roots, census sha256 and timestamp."""
    bench = Benchmark.load(ref, data=data)  # manifest sha + labels are verified by load itself
    roots = bench.resolved_roots()
    presence = _Presence(bench, roots)
    per_split, located = presence.scan()
    fingerprint = media_fingerprint(located, bench.manifest_hash, fingerprint_sample)
    targets_sha_ok: bool | None = None
    targets_errors: list[str] = []
    if bench.spec.targets is not None:
        targets_sha_ok = True
        for split in bench.spec.targets:
            try:
                bench.targets(split)
            except (DataNotReady, ManifestError) as e:
                targets_sha_ok = False
                targets_errors.append(str(e))
    curation = bench.spec.manifest.upstream.curation if bench.spec.manifest.upstream else None
    curation_ok: bool | None = None
    if curation is not None:
        counts = {s: bench.n_rows(s) for s in bench.spec.splits.all_values()}
        listed_ok = all(int(v.get("listed", -1)) == counts.get(s, -2) for s, v in curation.per_split.items())
        curation_ok = curation.invariant_holds() and listed_ok
    derived_ok, derived_errors = _derived_sha_ok(bench, roots)
    census_result: dict[str, Any] | None = None
    if census:
        from yavih.media.info import write_census

        census_result = write_census(bench)
        bench._media_info = False  # reload the fresh table
    table = bench.media_info()
    media_info = {"present": table is not None, "rows": len(table) if table is not None else 0, "sha256": table.sha256 if table is not None else None, "path": table.path if table is not None else str(media_info_path(bench.name, bench.manifest_hash))}
    if census_result is not None:
        media_info.update({k: census_result[k] for k in ("missing", "failed", "failures")})
    previous = _read_status_file(data_status_path(bench.name, roots)) or {}
    status = DataStatus(
        name=bench.name,
        manifest_hash=bench.manifest_hash,
        roots=dict(roots),
        per_split=per_split,
        media_fingerprint=fingerprint,
        targets_sha_ok=targets_sha_ok,
        verified=_utc_now(),
        path=str(data_status_path(bench.name, roots)),
        targets=dict(previous.get("targets") or {}),
        scanned=_utc_now(),
        media_info_sha256=media_info["sha256"],
    )
    for key, (size, mtime_ns, sha) in _TARGET_SHA_CACHE.items():
        status.targets[key] = {"size": size, "mtime_ns": mtime_ns, "sha256": sha}
    try:
        status.save()
    except OSError as e:
        _cache_write_failed(Path(status.path or ""), e)
    bench._status = status
    picks = _sample_picks(bench, decode_sample, status, presence.present_rows) if status.present() and decode_sample > 0 else []
    return {
        "name": bench.name,
        "manifest_sha_ok": True,
        "manifest_hash": bench.manifest_hash,
        "protocol_hash": bench.protocol_hash,
        "per_split": per_split,
        "read": _read_sample(bench, picks) if picks else {"ok": 0, "failed": 0, "skipped": "no media present" if not status.present() else "disabled"},
        "decode_sample": _decode_sample(bench, picks),
        "labels_ok": True,
        "unknown_labels": bench.unknown_labels,
        "targets_sha_ok": targets_sha_ok,
        "targets_errors": targets_errors,
        "derived_sha_ok": derived_ok,
        "derived_errors": derived_errors,
        "curation_invariants_ok": curation_ok,
        "media_info": media_info,
        "external": _external_status(bench),
        "media_fingerprint": fingerprint,
        "data_roots": dict(roots),
        "data_status_path": status.path,
        "status": status.status_word(),
    }
