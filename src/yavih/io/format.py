"""Feature format constants, member naming and errors.

Re-exports ``FEATURE_FORMAT_VERSION`` from ``yavih.versions`` (nothing else defines a
version).
"""

from __future__ import annotations

import hashlib
import io
import re

import numpy as np

from yavih.versions import FEATURE_FORMAT_VERSION, READABLE_FEATURE_FORMATS

__all__ = [
    "FEATURE_FORMAT_VERSION",
    "READABLE_FEATURE_FORMATS",
    "output_member_name",
    "FormatError",
    "FeatureShapeError",
    "FeatureValueError",
    "FeatureMetaError",
    "FEATURE_DTYPES",
    "POS_MODES",
    "SHARD_BYTES_CAP",
    "shard_tar_name",
    "shard_sidecar_name",
    "member_basename",
    "canary_basename",
    "parse_member_name",
    "NOT_APPLICABLE_FILE",
    "META_FILE",
    "CONSTANT_POS_MEMBER",
    "MEMBER_RE",
    "STALE_DIR",
    "TMP_SUFFIX",
    "tmp_name",
    "npy_bytes",
    "pos_sha256",
    "parse_shard_name",
    "SHARD_RE",
]

FEATURE_DTYPES: frozenset[str] = frozenset({"float16", "float32"})
POS_MODES: frozenset[str] = frozenset({"none", "constant", "per_sample"})
#: No shard exceeds 2 GiB.
SHARD_BYTES_CAP: int = 2 * 1024**3

NOT_APPLICABLE_FILE = "NOT_APPLICABLE.json"
META_FILE = "_meta.json"
CONSTANT_POS_MEMBER = "_pos.npy"
#: Mismatching shards are moved to ``_stale/<utc>/``.
STALE_DIR = "_stale"
#: Every file of the format is written as ``<name>.tmp`` then ``os.replace``d.
TMP_SUFFIX = ".tmp"
#: ``shard-{i:06d}.tar`` | ``shard-{i:06d}.json``.
SHARD_RE = re.compile(r"^shard-(?P<index>\d{6})\.(?P<ext>tar|json)$")
#: ``{row:09d}`` or ``c{row:09d}`` (canary) followed by ``.json`` | ``.feat.npy`` | ``.pos.npy`` (the default schema) or,
#: for a declared ``output_contract.schema``, ``.<key>.<ext>`` per schema key (format 5).
MEMBER_RE = re.compile(r"^(?P<canary>c?)(?P<row>\d{9})\.(?P<ext>json|feat\.npy|pos\.npy|[a-z][a-z0-9_]{0,31}\.[a-z0-9]{1,8})$")


class FormatError(ValueError):
    """A reader refuses an unknown ``format_version`` or a malformed member/sidecar (``E_FORMAT``)."""


class FeatureShapeError(ValueError):
    """``FeatureWriter.write``: a sample's ``(N, D)`` differs from the first call's."""

    def __init__(self, clip_id: str, got: tuple, expected: tuple) -> None:
        super().__init__(
            f"{clip_id}: feature shape {tuple(got)} != expected {tuple(expected)} (N and D are fixed "
            f"per feature set; pad/crop to a fixed length inside the bridge, never to the batch maximum)"
        )
        self.clip_id, self.got, self.expected = clip_id, tuple(got), tuple(expected)


class FeatureValueError(ValueError):
    """``FeatureWriter.write``: non-finite after cast, or float16 ``max|x| > 1e4``."""

    def __init__(self, clip_id: str, message: str) -> None:
        super().__init__(f"{clip_id}: {message}")
        self.clip_id = clip_id


class FeatureMetaError(ValueError):
    """``FeatureWriter.write``: a row's ``meta`` lacks ``media`` or a ``used`` entry for a stream in
    ``describe()["input"]``."""

    def __init__(self, clip_id: str, message: str) -> None:
        super().__init__(f"{clip_id}: {message}")
        self.clip_id = clip_id


def shard_tar_name(i: int) -> str:
    return f"shard-{i:06d}.tar"


def shard_sidecar_name(i: int) -> str:
    return f"shard-{i:06d}.json"


def member_basename(row: int) -> str:
    """``{row:09d}`` so stock WebDataset grouping by first dot works."""
    return f"{row:09d}"


def canary_basename(row: int) -> str:
    return f"c{row:09d}"


def parse_member_name(name: str) -> tuple[int, str, bool]:
    """``(row, ext, is_canary)`` from a member name; ``FormatError`` on anything else."""
    m = MEMBER_RE.match(name)
    if m is None:
        raise FormatError(f"malformed member name {name!r}")
    return int(m.group("row")), m.group("ext"), bool(m.group("canary"))


def output_member_name(row: int, key: str, ext: str, *, canary: bool = False) -> str:
    """``{row:09d}.{key}.{ext}`` (``c`` prefix for a canary): one schema key of one row (format 5)."""
    return f"{canary_basename(row) if canary else member_basename(row)}.{key}.{ext}"


def parse_shard_name(name: str) -> tuple[int, str]:
    """``(index, ext)`` from ``shard-{i:06d}.tar`` / ``.json``; ``FormatError`` on anything else."""
    m = SHARD_RE.match(name)
    if m is None:
        raise FormatError(f"malformed shard file name {name!r}")
    return int(m.group("index")), m.group("ext")


def tmp_name(name: str) -> str:
    return name + TMP_SUFFIX


def npy_bytes(arr: np.ndarray) -> tuple[bytes, int]:
    """``(payload, header_len)`` of ``arr`` as an uncompressed C-order ``.npy`` v1.0 member
    (``np.lib.format.write_array(allow_pickle=False)``). ``header_len`` lets a reader memmap the
    data at ``offset_data + header_len`` without extracting."""
    arr = np.ascontiguousarray(arr)
    buf = io.BytesIO()
    np.lib.format.write_array(buf, arr, version=(1, 0), allow_pickle=False)
    payload = buf.getvalue()
    return payload, len(payload) - arr.nbytes


def pos_sha256(payload: bytes) -> str:
    """sha256 over the serialized ``_pos.npy`` member bytes (the sidecar's ``features.pos_sha256``;
    ``E_POS_RANGE`` compares it across shards)."""
    return hashlib.sha256(payload).hexdigest()
