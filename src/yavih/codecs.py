"""Output codecs: how one value a bridge returns per row is stored, checked and compared.

A benchmark folder declares in ``output_contract.schema`` what a bridge returns per row, one entry per key::

    schema:
      feat:   {type: array, dtype: [float16, float32], shape: [N, D]}     # symbols bind once per feature set
      pos:    {type: array, dtype: float32, shape: [N, 4], optional: true, constant: allowed}
      loglik: {type: array, dtype: float32, shape: ["*"]}                # "*": free per row
      text:   {type: json, kind: str}
      audio:  {type: wav, channels: 1, duration_s: [9.5, 10.5]}

Every ``type`` names a :class:`Codec`: ``name`` and ``version`` (they enter the hashed output contract when the
schema is declared), the member ``ext``, ``encode`` / ``decode`` between a value and the bytes of its tar member,
``check`` against the entry (binding free symbols on first sight), ``distance`` for the determinism gates (``0`` for
equality; ``exact`` codecs are compared for equality only), optional ``stats`` (the dead-row / duplicate / constant
sanity accumulators), ``synthetic`` (a sample for tests and the lint self-test) and ``nbytes`` (the per-row size when
it follows from the entry). Built-ins: ``array`` (npy), ``json``, ``bytes`` (an opaque file with a declared
container) and ``wav`` (PCM audio, decoded for its distance). A folder registers its own codecs by exporting
``CODECS: dict[str, Codec]`` from ``eval.py``; a package through the entry-point group ``yavih.codecs``. A codec's
code is provenance (held by the folder's conformance fixture and the lint self-test), never identity.

Without a declared schema a folder has :data:`DEFAULT_SCHEMA`, the feature matrix every 1.0 benchmark hands over
(``feat [N, D]`` plus optional media positions ``pos [N, 4]``), and its hashes are unchanged. numpy + stdlib;
``wav`` imports ``soundfile`` (the ``media`` extra) lazily and falls back to the stdlib ``wave`` module for PCM.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import re
import struct
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from yavih.hashing import DEFAULT_OUTPUT_SCHEMA, canon, sha256_canon

__all__ = [
    "ARRAY_DTYPES",
    "POS_MODES",
    "FP16_MAX_ABS",
    "FP16_WARN_ABS",
    "DEFAULT_SCHEMA",
    "ENTRY_POINT_GROUP",
    "RESERVED_KEYS",
    "Codec",
    "CodecFinding",
    "Stats",
    "ArrayStats",
    "BUILTIN",
    "ARRAY",
    "JSON",
    "BYTES",
    "WAV",
    "registry",
    "resolve_schema",
    "check_schema",
    "schema_of",
    "is_default_schema",
    "has_default_feature_keys",
    "schema_sha256",
    "bind_shape",
    "self_test",
    "to_numpy",
    "rel_l2",
    "rel_l2_stored",
    "npy_bytes",
    "read_npy",
    "wav_bytes",
    "read_wav",
]

Spec = Mapping[str, Any]
Bound = MutableMapping[str, int]

#: Stored array dtypes a schema entry may allow.
ARRAY_DTYPES: frozenset[str] = frozenset({"float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"})
#: Modes of the default schema's ``pos`` key (``describe()["pos_mode"]``).
POS_MODES: frozenset[str] = frozenset({"none", "constant", "per_sample"})
#: ``array`` with ``fp16_guard`` (default): a float16 member whose ``max|x|`` exceeds this is refused (``E_FEATURES``)
#: and one above the warning level is ``W_FP16_RANGE``.
FP16_MAX_ABS: float = 1e4
FP16_WARN_ABS: float = 1e3
ENTRY_POINT_GROUP = "yavih.codecs"
#: Member names a schema key may not take (the row's provenance ``.json`` and its fields).
RESERVED_KEYS: frozenset[str] = frozenset({"json", "used", "media", "row", "clip_id", "media_key", "canary_of"})
KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
CODEC_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
EXT_RE = re.compile(r"^[a-z0-9]{1,8}$")
#: Keys every schema entry accepts besides the codec's own constraints.
COMMON_KEYS: frozenset[str] = frozenset({"type", "optional", "constant", "stats", "exact"})
CONSTANT_MODES: frozenset[str] = frozenset({"never", "allowed"})


@dataclass(frozen=True)
class CodecFinding:
    """One check result: a validate code (``E_MEMBERS``, ``E_FINITE``, ``E_FEATURES``, ``W_FP16_RANGE``, ...) and a
    message. Codes starting with ``E_`` are errors."""

    code: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.code.startswith("E_")


class Stats:
    """A codec's sanity accumulator over a scan (one per key and split). ``add`` sees every present row in scan
    order (``sampled`` marks the seeded sample); ``merge`` folds a shard's partial in shard order so a parallel scan
    accumulates what a serial one does; ``findings`` reports once at the end. Subclasses override all three."""

    bytes: int = 0

    def add(self, row: int, clip_id: str, media_key: str | None, value: Any, *, sampled: bool) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def merge(self, other: Stats) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def findings(self, *, key: str, n_tokens: int | None = None) -> list[CodecFinding]:  # pragma: no cover - interface
        raise NotImplementedError

    def empty(self) -> Stats:  # pragma: no cover - interface
        """A fresh, non-deferred accumulator of the same kind (the split's, into which partials merge)."""
        raise NotImplementedError

    def partial(self) -> Stats:  # pragma: no cover - interface
        """A fresh shard partial: order-sensitive accumulations are queued and run by :meth:`merge`."""
        raise NotImplementedError


def _no_problems(spec: Spec) -> list[str]:
    return []


@dataclass(frozen=True)
class Codec:
    """One output type. ``encode(value, spec) -> bytes`` / ``decode(data, spec) -> value``; ``check(value, spec,
    bound) -> findings`` binds free symbols into ``bound`` on first sight and verifies them after; ``distance(a, b,
    spec) -> float`` is ``0`` for equality (the determinism gates compare it with the tolerance, or with ``0`` for an
    ``exact`` codec); ``spec_check(spec) -> problems`` is the entry grammar ``Benchmark.load`` runs; ``stats(spec,
    bound) -> Stats`` builds the sanity accumulator; ``synthetic(spec, bound, rng) -> value`` a sample; ``nbytes(spec,
    bound) -> int | None`` the per-row size when derivable."""

    name: str
    version: int
    ext: str
    encode: Callable[[Any, Spec], bytes]
    decode: Callable[[bytes, Spec], Any]
    check: Callable[[Any, Spec, Bound], list[CodecFinding]]
    distance: Callable[[Any, Any, Spec], float]
    exact: bool = False
    spec_check: Callable[[Spec], list[str]] = _no_problems
    stats: Callable[[Spec, Bound], Stats] | None = None
    synthetic: Callable[[Spec, Bound, np.random.Generator], Any] | None = None
    nbytes: Callable[[Spec, Bound], int | None] | None = None

    def __post_init__(self) -> None:
        if not CODEC_NAME_RE.match(self.name or ""):
            raise ValueError(f"codec name {self.name!r} must match [a-z][a-z0-9_]{{0,31}}")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError(f"codec {self.name}: version must be an int >= 1")
        if not EXT_RE.match(self.ext or ""):
            raise ValueError(f"codec {self.name}: ext {self.ext!r} must match [a-z0-9]{{1,8}}")

    def ext_for(self, spec: Spec) -> str:
        """The member extension of an entry: the ``bytes`` codec's declared ``container``, else ``ext``."""
        c = spec.get("container")
        return str(c) if self.name == "bytes" and c else self.ext

    def is_exact(self, spec: Spec) -> bool:
        return bool(spec.get("exact", self.exact))


# ----------------------------------------------------------------------------- shared helpers


def to_numpy(x: Any) -> np.ndarray:
    """ndarray as-is; a tensor via ``.detach().float().cpu().numpy()``; anything else via ``np.asarray``."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def rel_l2(c: Any, o: Any) -> float:
    """``‖c−o‖₂ / max(‖o‖₂, 1e-12)`` in float64."""
    c64 = np.asarray(c, dtype=np.float64).ravel()
    o64 = np.asarray(o, dtype=np.float64).ravel()
    return float(np.linalg.norm(c64 - o64) / max(float(np.linalg.norm(o64)), 1e-12))


def rel_l2_stored(c: np.ndarray, o: np.ndarray) -> float:
    """``rel_l2`` net of storage quantization: for float16 members a difference of at most one float16 ulp of the
    larger magnitude is not counted (a float32 drift of ~1e-6 already flips some values by one ulp when both copies
    are rounded to float16); real batch dependence moves values by many ulps and is still measured."""
    c = np.asarray(c)
    o = np.asarray(o)
    if c.dtype != np.float16 or o.dtype != np.float16:
        return rel_l2(c, o)
    c64 = c.astype(np.float64).ravel()
    o64 = o.astype(np.float64).ravel()
    ulp = np.spacing(np.maximum(np.abs(c), np.abs(o)).astype(np.float16)).astype(np.float64).ravel()
    diff = np.abs(c64 - o64)
    diff = np.where(diff <= ulp, 0.0, diff)
    return float(np.linalg.norm(diff) / max(float(np.linalg.norm(o64)), 1e-12))


def npy_bytes(arr: np.ndarray) -> tuple[bytes, int]:
    """``(payload, header_len)`` of ``arr`` as an uncompressed C-order ``.npy`` v1.0 member."""
    arr = np.ascontiguousarray(arr)
    buf = io.BytesIO()
    np.lib.format.write_array(buf, arr, version=(1, 0), allow_pickle=False)
    payload = buf.getvalue()
    return payload, len(payload) - arr.nbytes


def read_npy(data: bytes) -> tuple[np.ndarray, int]:
    """Parse one ``.npy`` member (v1.0 header, C-order, no object dtype, exact payload size) into a read-only array
    over ``data``; ``(array, header_len)``. ``ValueError`` names what is malformed."""
    fh = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        else:
            raise ValueError(f"unsupported .npy header version {version}")
    except (ValueError, OSError) as e:
        raise ValueError(f"malformed .npy header: {e}") from e
    header_len = fh.tell()
    if version != (1, 0):
        raise ValueError(f".npy header version {version} != 1.0")
    if fortran:
        raise ValueError(".npy member is Fortran-ordered (C-order)")
    if dtype.hasobject:
        raise ValueError(".npy member has object dtype")
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if len(data) - header_len != count * dtype.itemsize:
        raise ValueError(f".npy payload {len(data) - header_len} B != {count * dtype.itemsize} B for {shape} {dtype}")
    return np.frombuffer(data, dtype=dtype, count=count, offset=header_len).reshape(shape), header_len


def bind_shape(spec_shape: Iterable[Any], actual: tuple[int, ...], bound: Bound, *, where: str) -> str | None:
    """The one shape rule: an int must equal, a symbol binds on first sight and must equal afterwards, ``"*"`` is
    free. Returns the problem, or ``None``."""
    spec_shape = list(spec_shape)
    if len(spec_shape) != len(actual):
        return f"{where}: shape {tuple(actual)} has {len(actual)} dims, the schema declares {len(spec_shape)} ({spec_shape})"
    for i, (want, got) in enumerate(zip(spec_shape, actual, strict=True)):
        if want == "*":
            continue
        if isinstance(want, int) and not isinstance(want, bool):
            if got != want:
                return f"{where}: dim {i} is {got}, the schema declares {want}"
            continue
        sym = str(want)
        if sym in bound:
            if bound[sym] != got:
                return f"{where}: dim {i} ({sym}) is {got}, bound to {bound[sym]} by an earlier row (every symbol is fixed per feature set)"
        else:
            if got < 1:
                return f"{where}: dim {i} ({sym}) must be >= 1, got {got}"
            bound[sym] = int(got)
    return None


# ----------------------------------------------------------------------------- array


def _dtypes_of(spec: Spec) -> tuple[str, ...]:
    d = spec.get("dtype")
    if d is None:
        return tuple(sorted(ARRAY_DTYPES))
    return (str(d),) if isinstance(d, str) else tuple(str(x) for x in d)


def _array_spec_check(spec: Spec) -> list[str]:
    problems: list[str] = []
    for k in spec:
        if k not in COMMON_KEYS | {"dtype", "shape", "finite", "fp16_guard"}:
            problems.append(f"unknown array constraint {k!r}")
    d = spec.get("dtype")
    if d is not None:
        for x in _dtypes_of(spec):
            if x not in ARRAY_DTYPES:
                problems.append(f"dtype {x!r} is not one of {sorted(ARRAY_DTYPES)}")
    shape = spec.get("shape")
    if not isinstance(shape, list | tuple) or not shape:
        problems.append("shape: a non-empty list of ints, symbols ([A-Z][A-Z0-9_]*) or \"*\"")
    else:
        for x in shape:
            if x == "*" or (isinstance(x, int) and not isinstance(x, bool) and x >= 1) or (isinstance(x, str) and SYMBOL_RE.match(x)):
                continue
            problems.append(f"shape entry {x!r}: an int >= 1, a symbol ([A-Z][A-Z0-9_]*) or \"*\"")
    for k in ("finite", "fp16_guard"):
        if k in spec and not isinstance(spec[k], bool):
            problems.append(f"{k}: a bool")
    return problems


def _array_encode(value: Any, spec: Spec) -> bytes:
    arr = to_numpy(value)
    allowed = _dtypes_of(spec)
    if arr.dtype.name not in allowed:
        if len(allowed) == 1:
            arr = np.ascontiguousarray(arr, dtype=allowed[0])
        else:
            raise ValueError(f"array dtype {arr.dtype.name} is not one of the declared {list(allowed)}; cast inside the bridge")
    return npy_bytes(arr)[0]


def _array_decode(data: bytes, spec: Spec) -> np.ndarray:
    return read_npy(data)[0]


def _array_check(value: Any, spec: Spec, bound: Bound) -> list[CodecFinding]:
    out: list[CodecFinding] = []
    if not isinstance(value, np.ndarray):
        try:
            value = to_numpy(value)
        except Exception as e:  # noqa: BLE001 - reported, never raised
            return [CodecFinding("E_MEMBERS", f"not an array: {type(e).__name__}: {e}")]
    problem = bind_shape(spec.get("shape") or (), tuple(int(s) for s in value.shape), bound, where="shape")
    if problem is not None:
        out.append(CodecFinding("E_MEMBERS", problem))
    allowed = _dtypes_of(spec)
    if value.dtype.name not in allowed:
        out.append(CodecFinding("E_MEMBERS", f"dtype {value.dtype.name} is not one of the declared {list(allowed)}"))
    if value.dtype.kind == "f" and spec.get("finite", True) and value.size and not bool(np.isfinite(value).all()):
        out.append(CodecFinding("E_FINITE", "non-finite values (NaN / Inf)"))
    if value.dtype == np.float16 and spec.get("fp16_guard", True) and value.size:
        m = float(np.abs(value[np.isfinite(value)]).max()) if bool(np.isfinite(value).any()) else 0.0
        if m > FP16_MAX_ABS:
            out.append(CodecFinding("E_FEATURES", f"float16 max|x| = {m:.4g} > {FP16_MAX_ABS:g}: saturated; store float32 or normalize inside the bridge"))
        elif m > FP16_WARN_ABS:
            out.append(CodecFinding("W_FP16_RANGE", f"float16 max|x| = {m:.3g} > {FP16_WARN_ABS:g}; consider float32 or a saner layer"))
    return out


def _array_distance(a: Any, b: Any, spec: Spec) -> float:
    a = to_numpy(a)
    b = to_numpy(b)
    if a.shape != b.shape:
        return float("inf")
    if a.tobytes() == b.tobytes():
        return 0.0
    return rel_l2_stored(a, b)


def _array_synthetic(spec: Spec, bound: Bound, rng: np.random.Generator) -> np.ndarray:
    shape = []
    for x in spec.get("shape") or ():
        if x == "*":
            shape.append(int(rng.integers(1, 5)))
        elif isinstance(x, int):
            shape.append(int(x))
        elif str(x) in bound:
            shape.append(int(bound[str(x)]))
        else:
            n = int(rng.integers(2, 6))
            bound[str(x)] = n
            shape.append(n)
    dtype = _dtypes_of(spec)[0]
    if np.dtype(dtype).kind == "f":
        return rng.standard_normal(shape).astype(dtype)
    if dtype == "bool":
        return rng.integers(0, 2, size=shape).astype(bool)
    return rng.integers(0, 10, size=shape).astype(dtype)


def _array_nbytes(spec: Spec, bound: Bound) -> int | None:
    n = 1
    for x in spec.get("shape") or ():
        if x == "*":
            return None
        if isinstance(x, int):
            n *= int(x)
        elif str(x) in bound:
            n *= int(bound[str(x)])
        else:
            return None
    return n * np.dtype(_dtypes_of(spec)[0]).itemsize


#: ``E_DEAD_ROWS``: variance below this fraction of the mean token variance.
DEAD_ROW_REL_VAR: float = 1e-12
#: ``E_DUPLICATE_CLIPS``: more than this fraction of sampled clips byte-identical.
DUPLICATE_MAX_FRACTION: float = 0.01
#: ``W_FP16_RANGE``: zero fraction above this is suspicious.
FP16_WARN_ZERO_FRACTION: float = 0.5


class ArrayStats(Stats):
    """The ``array`` sanity accumulators: per-token variance over the sampled rows (``E_DEAD_ROWS``), byte-identical
    sampled clips with different media keys (``E_DUPLICATE_CLIPS``), the float16 range and zero fraction
    (``W_FP16_RANGE``), clips whose tokens are all identical or all zero (``W_CONSTANT_CLIP``). ``deferred`` queues
    the order-sensitive sums for :meth:`merge` (a parallel scan reproduces a serial one bit for bit)."""

    def __init__(self, spec: Spec, bound: Bound, *, deferred: bool = False) -> None:
        self.spec = spec
        self.deferred = deferred
        self.sum: np.ndarray | None = None
        self.sumsq: np.ndarray | None = None
        self.n_sampled_seen = 0
        self._queue: list[np.ndarray] = []
        self.max_abs = 0.0
        self.n_values = 0
        self.n_zero = 0
        self.bytes = 0
        self.shape: tuple[int, ...] | None = None
        self.constant_clips: list[tuple[int, str, str | None]] = []
        self.dup_hash: dict[int, tuple[str, str | None]] = {}
        self.dtype: str | None = None

    def empty(self) -> ArrayStats:
        return ArrayStats(self.spec, {})

    def partial(self) -> ArrayStats:
        return ArrayStats(self.spec, {}, deferred=True)

    def add(self, row: int, clip_id: str, media_key: str | None, value: Any, *, sampled: bool) -> None:
        arr = np.asarray(value)
        self.bytes += arr.nbytes
        self.n_values += arr.size
        self.n_zero += int(np.count_nonzero(arr == 0))
        self.dtype = arr.dtype.name
        if arr.size and arr.dtype.kind in "fiu":
            self.max_abs = max(self.max_abs, float(np.abs(arr).max()))
            if (arr.ndim >= 1 and arr.shape[0] > 1 and bool((arr == arr[0]).all())) or not bool(arr.any()):
                self.constant_clips.append((row, clip_id, media_key))
        if sampled:
            if self.deferred:
                self._queue.append(arr)
            else:
                self._accumulate(arr)
            self.dup_hash[row] = (hashlib.sha256(arr.tobytes()).hexdigest(), media_key)

    def _accumulate(self, arr: np.ndarray) -> None:
        if arr.dtype.kind not in "fiu":
            return
        if self.shape is None:
            self.shape = tuple(arr.shape)
            self.sum = np.zeros(arr.shape, dtype=np.float64)
            self.sumsq = np.zeros(arr.shape, dtype=np.float64)
        if tuple(arr.shape) != self.shape:
            return  # a ragged key has no per-position variance
        assert self.sum is not None and self.sumsq is not None
        f64 = arr.astype(np.float64)
        self.sum += f64
        self.sumsq += f64 * f64
        self.n_sampled_seen += 1

    def merge(self, other: Stats) -> None:
        assert isinstance(other, ArrayStats)
        self.bytes += other.bytes
        self.n_values += other.n_values
        self.n_zero += other.n_zero
        self.max_abs = max(self.max_abs, other.max_abs)
        self.dtype = other.dtype or self.dtype
        self.constant_clips.extend(other.constant_clips)
        for arr in other._queue:
            self._accumulate(arr)
        self.dup_hash.update(other.dup_hash)

    def dead_rows(self) -> list[int]:
        """Token positions (leading axis) whose variance over the sample is below ``DEAD_ROW_REL_VAR`` × mean."""
        if self.n_sampled_seen < 2 or self.sum is None or self.sumsq is None or not self.shape:
            return []
        n = float(self.n_sampled_seen)
        with np.errstate(invalid="ignore"):
            var = np.maximum(self.sumsq / n - (self.sum / n) ** 2, 0.0)
            per_token = var.reshape(var.shape[0], -1).mean(axis=1) if var.ndim >= 1 else var.reshape(1)
            per_token = np.nan_to_num(per_token, nan=np.inf)
            finite = per_token[np.isfinite(per_token)]
            mean_var = float(finite.mean()) if finite.size else 0.0
            dead = per_token < DEAD_ROW_REL_VAR * mean_var
        return [int(i) for i in np.nonzero(dead)[0]]

    def duplicate_rows(self) -> list[int]:
        """Sampled rows byte-identical to a row with a different media key."""
        by_sha: dict[str, list[tuple[int, str | None]]] = {}
        for row, (sha, mk) in self.dup_hash.items():
            by_sha.setdefault(sha, []).append((row, mk))
        out: list[int] = []
        for group in by_sha.values():
            if len(group) < 2:
                continue
            for row, mk in group:
                if not all(omk is not None and omk == mk for orow, omk in group if orow != row):
                    out.append(row)
        return sorted(out)

    def findings(self, *, key: str, n_tokens: int | None = None) -> list[CodecFinding]:
        out: list[CodecFinding] = []
        dead = self.dead_rows()
        if dead:
            n = self.shape[0] if self.shape else "?"
            out.append(CodecFinding("E_DEAD_ROWS", f"{key}: {len(dead)} of {n} token positions have variance < 1e-12 · mean across "
                                    f"{self.n_sampled_seen} sampled clips: {dead[:10]} (if intended, e.g. padding or register tokens: "
                                    "describe()['waive'] = {'E_DEAD_ROWS': '<reason>'})"))
        if len(self.dup_hash) > 1:
            dups = self.duplicate_rows()
            frac = len(dups) / len(self.dup_hash)
            if frac > DUPLICATE_MAX_FRACTION:
                out.append(CodecFinding("E_DUPLICATE_CLIPS", f"{key}: {len(dups)} of {len(self.dup_hash)} sampled clips ({frac:.1%}) are "
                                        f"byte-identical to a clip with a different media_key: {dups[:10]} (features do not depend on the "
                                        "input; if the model's input is genuinely identical for those media, e.g. digitally silent audio: "
                                        "describe()['waive'] = {'E_DUPLICATE_CLIPS': '<reason>'})"))
        if self.dtype == "float16" and self.n_values:
            zero_frac = self.n_zero / self.n_values
            if self.max_abs > FP16_WARN_ABS or zero_frac > FP16_WARN_ZERO_FRACTION:
                out.append(CodecFinding("W_FP16_RANGE", f"{key}: float16 max|x| = {self.max_abs:.3g}, zero fraction = {zero_frac:.2f} "
                                        f"(thresholds {FP16_WARN_ABS:g} / {FP16_WARN_ZERO_FRACTION}); consider float32 or a saner layer"))
        if self.constant_clips:
            groups: dict[str, tuple[int, str]] = {}
            for row, cid, mk in self.constant_clips:
                groups.setdefault(mk if mk is not None else f"clip:{cid}", (row, cid))
            sample = [cid for _, cid in list(groups.values())[:5]]
            out.append(CodecFinding("W_CONSTANT_CLIP", f"{key}: {len(groups)} clip(s) whose tokens are all identical (or all zero): {sample}"))
        return out


ARRAY = Codec(
    name="array", version=1, ext="npy", encode=_array_encode, decode=_array_decode, check=_array_check, distance=_array_distance,
    spec_check=_array_spec_check, stats=lambda spec, bound: ArrayStats(spec, bound), synthetic=_array_synthetic, nbytes=_array_nbytes,
)


# ----------------------------------------------------------------------------- json

JSON_KINDS: frozenset[str] = frozenset({"any", "str", "number", "object", "list", "bool"})


def _json_spec_check(spec: Spec) -> list[str]:
    problems: list[str] = []
    for k in spec:
        if k not in COMMON_KEYS | {"kind", "max_bytes"}:
            problems.append(f"unknown json constraint {k!r}")
    if spec.get("kind", "any") not in JSON_KINDS:
        problems.append(f"kind: one of {sorted(JSON_KINDS)}")
    mb = spec.get("max_bytes")
    if mb is not None and (not isinstance(mb, int) or isinstance(mb, bool) or mb < 1):
        problems.append("max_bytes: an int >= 1")
    return problems


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _json_encode(value: Any, spec: Spec) -> bytes:
    return canon(_jsonable(value)).encode("utf-8")


def _json_decode(data: bytes, spec: Spec) -> Any:
    return json.loads(data.decode("utf-8"))


def _json_kind_ok(value: Any, kind: str) -> bool:
    return {
        "any": True, "str": isinstance(value, str), "bool": isinstance(value, bool),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "object": isinstance(value, Mapping), "list": isinstance(value, list | tuple),
    }[kind]


def _json_check(value: Any, spec: Spec, bound: Bound) -> list[CodecFinding]:
    try:
        text = canon(_jsonable(value))
    except (TypeError, ValueError) as e:
        return [CodecFinding("E_MEMBERS", f"not JSON-serializable: {e}")]
    out: list[CodecFinding] = []
    kind = str(spec.get("kind", "any"))
    if not _json_kind_ok(_jsonable(value), kind):
        out.append(CodecFinding("E_MEMBERS", f"json value is a {type(value).__name__}, the schema declares kind {kind!r}"))
    mb = spec.get("max_bytes")
    if mb is not None and len(text.encode("utf-8")) > int(mb):
        out.append(CodecFinding("E_MEMBERS", f"json value is {len(text.encode('utf-8'))} B > max_bytes {int(mb)}"))
    return out


def _json_distance(a: Any, b: Any, spec: Spec) -> float:
    return 0.0 if canon(_jsonable(a)) == canon(_jsonable(b)) else 1.0


def _json_synthetic(spec: Spec, bound: Bound, rng: np.random.Generator) -> Any:
    kind = str(spec.get("kind", "any"))
    word = "".join(chr(int(c)) for c in rng.integers(97, 123, size=6))
    if kind == "number":
        return float(rng.standard_normal())
    if kind == "bool":
        return bool(rng.integers(0, 2))
    if kind == "object":
        return {"text": word, "n": int(rng.integers(0, 9))}
    if kind == "list":
        return [word, int(rng.integers(0, 9))]
    return word


JSON = Codec(name="json", version=1, ext="json", encode=_json_encode, decode=_json_decode, check=_json_check, distance=_json_distance,
             exact=True, spec_check=_json_spec_check, synthetic=_json_synthetic)


# ----------------------------------------------------------------------------- bytes


def _bytes_spec_check(spec: Spec) -> list[str]:
    problems: list[str] = []
    for k in spec:
        if k not in COMMON_KEYS | {"container", "max_bytes"}:
            problems.append(f"unknown bytes constraint {k!r}")
    c = spec.get("container")
    if not isinstance(c, str) or not EXT_RE.match(c):
        problems.append("container: the file's extension, [a-z0-9]{1,8} (wav, png, ply, ...)")
    mb = spec.get("max_bytes")
    if mb is not None and (not isinstance(mb, int) or isinstance(mb, bool) or mb < 1):
        problems.append("max_bytes: an int >= 1")
    return problems


def _bytes_encode(value: Any, spec: Spec) -> bytes:
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value)
    raise ValueError(f"a bytes value is expected, got {type(value).__name__}")


def _bytes_decode(data: bytes, spec: Spec) -> bytes:
    return bytes(data)


def _bytes_check(value: Any, spec: Spec, bound: Bound) -> list[CodecFinding]:
    if not isinstance(value, bytes | bytearray | memoryview):
        return [CodecFinding("E_MEMBERS", f"a bytes value is expected, got {type(value).__name__}")]
    out: list[CodecFinding] = []
    if len(value) == 0:
        out.append(CodecFinding("E_MEMBERS", "empty file"))
    mb = spec.get("max_bytes")
    if mb is not None and len(value) > int(mb):
        out.append(CodecFinding("E_MEMBERS", f"{len(value)} B > max_bytes {int(mb)}"))
    return out


def _bytes_distance(a: Any, b: Any, spec: Spec) -> float:
    return 0.0 if hashlib.sha256(bytes(a)).digest() == hashlib.sha256(bytes(b)).digest() else 1.0


def _bytes_synthetic(spec: Spec, bound: Bound, rng: np.random.Generator) -> bytes:
    if str(spec.get("container")) == "wav":
        return wav_bytes(rng.standard_normal(1600).astype(np.float32) * 0.1, 16000)
    return rng.bytes(64)


BYTES = Codec(name="bytes", version=1, ext="bin", encode=_bytes_encode, decode=_bytes_decode, check=_bytes_check, distance=_bytes_distance,
              exact=True, spec_check=_bytes_spec_check, synthetic=_bytes_synthetic)


# ----------------------------------------------------------------------------- wav


def wav_bytes(samples: Any, sample_rate: int, *, subtype: str = "PCM_16") -> bytes:
    """A wav file from float samples in ``[-1, 1]`` (``[n]`` or ``[n, channels]``): ``soundfile`` when installed,
    else the stdlib ``wave`` module (PCM_16 only)."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    try:
        import soundfile as sf
    except ImportError:
        if subtype != "PCM_16":
            raise ImportError("writing a non-PCM_16 wav needs soundfile (pip install soundfile, the [media] extra)") from None
        import wave

        pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).round().astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(arr.shape[1])
            w.setsampwidth(2)
            w.setframerate(int(sample_rate))
            w.writeframes(pcm.tobytes())
        return buf.getvalue()
    buf = io.BytesIO()
    sf.write(buf, arr, int(sample_rate), format="WAV", subtype=subtype)
    return buf.getvalue()


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """``(samples float64 [n, channels], sample_rate)`` of a wav file: ``soundfile`` when installed, else the stdlib
    ``wave`` module (PCM 8 / 16 / 32 only). ``ValueError`` for a file that is not a wav."""
    try:
        import soundfile as sf
    except ImportError:
        import wave

        try:
            with wave.open(io.BytesIO(bytes(data)), "rb") as w:
                n, ch, width, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
                raw = w.readframes(n)
        except (wave.Error, EOFError, struct.error) as e:
            raise ValueError(f"not a PCM wav file: {e}") from e
        if width == 2:
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        elif width == 4:
            arr = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
        elif width == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported PCM width {width}") from None
        return arr.reshape(-1, ch), int(sr)
    try:
        arr, sr = sf.read(io.BytesIO(bytes(data)), dtype="float64", always_2d=True)
    except (RuntimeError, ValueError) as e:
        raise ValueError(f"not a wav file soundfile can read: {e}") from e
    return np.asarray(arr, dtype=np.float64), int(sr)


def _wav_header(data: bytes) -> tuple[int, int, int] | None:
    """``(channels, sample_rate, n_frames)`` from a RIFF/WAVE header without decoding (None if not a wav)."""
    try:
        import wave

        with wave.open(io.BytesIO(bytes(data)), "rb") as w:
            return w.getnchannels(), w.getframerate(), w.getnframes()
    except Exception:  # noqa: BLE001 - a float wav the stdlib refuses: decode instead
        try:
            arr, sr = read_wav(data)
        except ValueError:
            return None
        return int(arr.shape[1]), int(sr), int(arr.shape[0])


def _wav_spec_check(spec: Spec) -> list[str]:
    problems: list[str] = []
    for k in spec:
        if k not in COMMON_KEYS | {"channels", "sample_rate", "duration_s", "max_bytes"}:
            problems.append(f"unknown wav constraint {k!r}")
    ch = spec.get("channels")
    if ch is not None and (not isinstance(ch, int) or isinstance(ch, bool) or ch < 1):
        problems.append("channels: an int >= 1")
    sr = spec.get("sample_rate")
    if sr is not None:
        rates = [sr] if isinstance(sr, int) else list(sr) if isinstance(sr, list | tuple) else [None]
        if any(not isinstance(r, int) or isinstance(r, bool) or r < 1 for r in rates):
            problems.append("sample_rate: an int >= 1 or a list of them")
    d = spec.get("duration_s")
    if d is not None and not (isinstance(d, list | tuple) and len(d) == 2 and all(isinstance(x, int | float) and x >= 0 for x in d) and d[0] <= d[1]):
        problems.append("duration_s: [lo, hi] seconds")
    return problems


def _wav_check(value: Any, spec: Spec, bound: Bound) -> list[CodecFinding]:
    if not isinstance(value, bytes | bytearray | memoryview):
        return [CodecFinding("E_MEMBERS", f"a wav file's bytes are expected, got {type(value).__name__}")]
    head = _wav_header(bytes(value))
    if head is None:
        return [CodecFinding("E_MEMBERS", "not a wav file")]
    ch, sr, n = head
    out: list[CodecFinding] = []
    want_ch = spec.get("channels")
    if want_ch is not None and ch != int(want_ch):
        out.append(CodecFinding("E_MEMBERS", f"wav has {ch} channel(s), the schema declares {int(want_ch)}"))
    want_sr = spec.get("sample_rate")
    if want_sr is not None:
        rates = [want_sr] if isinstance(want_sr, int) else list(want_sr)
        if sr not in rates:
            out.append(CodecFinding("E_MEMBERS", f"wav sample rate {sr} is not one of the declared {rates}"))
    d = spec.get("duration_s")
    if d is not None:
        dur = n / float(sr) if sr else 0.0
        if not (float(d[0]) <= dur <= float(d[1])):
            out.append(CodecFinding("E_MEMBERS", f"wav duration {dur:.3f} s is outside the declared [{float(d[0]):g}, {float(d[1]):g}] s"))
    mb = spec.get("max_bytes")
    if mb is not None and len(value) > int(mb):
        out.append(CodecFinding("E_MEMBERS", f"{len(value)} B > max_bytes {int(mb)}"))
    return out


def _wav_distance(a: Any, b: Any, spec: Spec) -> float:
    """``rel_l2`` over the decoded samples (``inf`` when the rates or lengths differ); ``0`` for identical bytes."""
    if bytes(a) == bytes(b):
        return 0.0
    try:
        sa, ra = read_wav(bytes(a))
        sb, rb = read_wav(bytes(b))
    except ValueError:
        return float("inf")
    if ra != rb or sa.shape != sb.shape:
        return float("inf")
    return rel_l2(sa, sb)


def _wav_synthetic(spec: Spec, bound: Bound, rng: np.random.Generator) -> bytes:
    sr = spec.get("sample_rate")
    sr = int(sr if isinstance(sr, int) else (sr[0] if isinstance(sr, list | tuple) else 16000))
    d = spec.get("duration_s")
    seconds = float(d[0]) if d is not None else 1.0
    seconds = max(seconds, 0.01)
    ch = int(spec.get("channels") or 1)
    return wav_bytes(rng.standard_normal((int(round(sr * seconds)), ch)).astype(np.float32) * 0.1, sr)


WAV = Codec(name="wav", version=1, ext="wav", encode=_bytes_encode, decode=_bytes_decode, check=_wav_check, distance=_wav_distance,
            spec_check=_wav_spec_check, synthetic=_wav_synthetic)

BUILTIN: dict[str, Codec] = {c.name: c for c in (ARRAY, JSON, BYTES, WAV)}


# ----------------------------------------------------------------------------- schemas

#: The 1.0 hand-off: a token matrix per row plus optional media positions (``yavih.hashing.DEFAULT_OUTPUT_SCHEMA``;
#: a folder declaring no schema, or exactly this one, hashes as before).
DEFAULT_SCHEMA: dict[str, dict[str, Any]] = DEFAULT_OUTPUT_SCHEMA


def schema_of(output_contract: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The declared ``output_contract.schema``, else :data:`DEFAULT_SCHEMA` (a copy)."""
    s = (output_contract or {}).get("schema") if output_contract is not None else None
    if s is None:
        return {k: dict(v) for k, v in DEFAULT_SCHEMA.items()}
    return {str(k): dict(v) for k, v in s.items()}


def is_default_schema(schema: Mapping[str, Any] | None) -> bool:
    return schema is None or canon(dict(schema)) == canon(DEFAULT_SCHEMA)


def has_default_feature_keys(schema: Mapping[str, Any] | None) -> bool:
    """Whether the schema carries the 1.0 feature keys (``feat`` and ``pos`` as the default declares them): the
    feature-specific ``describe()`` keys, media positions and the position checks then apply."""
    if schema is None:
        return True
    return all(k in schema and canon(dict(schema[k])) == canon(DEFAULT_SCHEMA[k]) for k in DEFAULT_SCHEMA)


def schema_sha256(schema: Mapping[str, Any] | None) -> str:
    return sha256_canon(dict(schema) if schema is not None else DEFAULT_SCHEMA)


def _entry_point_codecs() -> dict[str, Codec]:
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return {}
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python < 3.10 API
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    out: dict[str, Codec] = {}
    for ep in eps:
        modname = ep.value.split(":", 1)[0]
        try:
            mod = importlib.import_module(modname)
            codecs = getattr(mod, "CODECS", None)
            if not isinstance(codecs, Mapping):
                raise TypeError(f"{modname} does not export CODECS = {{name: Codec}}")
            for name, c in codecs.items():
                if not isinstance(c, Codec) or c.name != name:
                    raise TypeError(f"{modname}: CODECS[{name!r}] must be a Codec named {name!r}")
                if name in BUILTIN:
                    raise ValueError(f"{modname}: CODECS[{name!r}] redefines a built-in codec")
                out[name] = c
        except Exception as e:  # noqa: BLE001 - a broken plugin must name itself, not break the core
            import warnings

            warnings.warn(f"codec entry point {ep.name!r} ({ep.value}) failed to load: {type(e).__name__}: {e}", stacklevel=2)
    return out


def folder_codecs(eval_mod: Any) -> dict[str, Codec]:
    """``CODECS`` of a folder's ``eval.py`` module (``{}`` when it exports none); a built-in name is refused."""
    codecs = getattr(eval_mod, "CODECS", None) if eval_mod is not None else None
    if codecs is None:
        return {}
    if not isinstance(codecs, Mapping):
        raise ValueError("eval.py CODECS must be a dict {name: Codec}")
    out: dict[str, Codec] = {}
    for name, c in codecs.items():
        if not isinstance(c, Codec) or c.name != name:
            raise ValueError(f"eval.py CODECS[{name!r}] must be a yavih.codecs.Codec named {name!r}")
        if name in BUILTIN:
            raise ValueError(f"eval.py CODECS[{name!r}] redefines a built-in codec; pick another name")
        out[name] = c
    return out


def registry(extra: Mapping[str, Codec] | None = None) -> dict[str, Codec]:
    """Every codec a schema may name: the built-ins, the ``yavih.codecs`` entry points and ``extra`` (a folder's
    ``CODECS``). A name defined twice is refused."""
    out = dict(BUILTIN)
    for name, c in _entry_point_codecs().items():
        out[name] = c
    for name, c in (extra or {}).items():
        if name in out and out[name] is not c:
            raise ValueError(f"codec {name!r} is already registered")
        out[name] = c
    return out


def check_schema(schema: Mapping[str, Any], codecs: Mapping[str, Codec]) -> list[str]:
    """The schema grammar: key names, known codec types, each codec's constraints, the common keys, consistent
    symbols. Returns the problems (empty = valid)."""
    problems: list[str] = []
    if not isinstance(schema, Mapping) or not schema:
        return ["schema: a non-empty mapping {key: {type: <codec>, ...}}"]
    fixed: dict[str, int] = {}
    for key, entry in schema.items():
        where = f"schema.{key}"
        if not isinstance(key, str) or not KEY_RE.match(key) or key in RESERVED_KEYS:
            problems.append(f"{where}: key must match [a-z][a-z0-9_]{{0,31}} and not be one of {sorted(RESERVED_KEYS)}")
            continue
        if not isinstance(entry, Mapping) or not isinstance(entry.get("type"), str):
            problems.append(f"{where}: {{type: <codec>, ...}}")
            continue
        codec = codecs.get(entry["type"])
        if codec is None:
            problems.append(f"{where}.type: unknown codec {entry['type']!r}; known: {sorted(codecs)}")
            continue
        for k in ("optional", "stats", "exact"):
            if k in entry and not isinstance(entry[k], bool):
                problems.append(f"{where}.{k}: a bool")
        if "constant" in entry and entry["constant"] not in CONSTANT_MODES:
            problems.append(f"{where}.constant: one of {sorted(CONSTANT_MODES)}")
        if entry.get("stats") and codec.stats is None:
            problems.append(f"{where}.stats: codec {codec.name!r} has no statistics")
        for p in codec.spec_check(entry):
            problems.append(f"{where}: {p}")
        for x in entry.get("shape") or () if codec.name == "array" and isinstance(entry.get("shape"), list | tuple) else ():
            if isinstance(x, str) and SYMBOL_RE.match(x):
                fixed.setdefault(x, 0)
    return problems


def resolve_schema(schema: Mapping[str, Any], codecs: Mapping[str, Codec]) -> dict[str, tuple[dict[str, Any], Codec]]:
    """``{key: (entry, codec)}`` for a valid schema; ``ValueError`` listing the problems otherwise."""
    problems = check_schema(schema, codecs)
    if problems:
        raise ValueError("; ".join(problems))
    return {str(k): (dict(v), codecs[v["type"]]) for k, v in schema.items()}


def self_test(codec: Codec, spec: Spec, bound: Bound | None = None, rng: np.random.Generator | None = None) -> list[str]:
    """What a codec must satisfy for the determinism gates to mean anything: a synthetic value round-trips through
    ``encode`` / ``decode`` losslessly, ``distance(x, x) == 0``, ``distance(x, y) > 0`` for a second sample, and
    ``check`` reports no error on its own sample. Returns the problems (empty = passes)."""
    problems: list[str] = []
    if codec.synthetic is None:
        return [f"codec {codec.name!r} has no synthetic(); the lint self-test needs one"]
    bound = {} if bound is None else bound
    rng = np.random.default_rng(0) if rng is None else rng
    x = codec.synthetic(spec, bound, rng)
    y = codec.synthetic(spec, bound, rng)
    try:
        if codec.distance(x, x, spec) != 0.0:
            problems.append("distance(x, x) != 0")
        back = codec.decode(codec.encode(x, spec), spec)
        if codec.distance(back, x, spec) != 0.0:
            problems.append("encode/decode round trip is not lossless")
        if not (codec.distance(x, y, spec) > 0.0):
            problems.append("distance(x, y) is 0 for two different samples: the gates could not see a change")
        errors = [f for f in codec.check(x, spec, dict(bound)) if f.is_error]
        if errors:
            problems.append("check() refuses the codec's own sample: " + "; ".join(f"{f.code}: {f.message}" for f in errors))
    except Exception as e:  # noqa: BLE001 - a self-test reports, never raises
        problems.append(f"{type(e).__name__}: {e}")
    return problems


def jsonable(value: Any) -> Any:
    """JSON primitives for a json-codec value (numpy scalars and arrays included)."""
    return _jsonable(value)


def key_nbytes(resolved: Mapping[str, tuple[Mapping[str, Any], Codec]], bound: Bound) -> dict[str, int | None]:
    """Per-key bytes per row when derivable (``Codec.nbytes``), else ``None``."""
    out: dict[str, int | None] = {}
    for key, (spec, codec) in resolved.items():
        out[key] = codec.nbytes(spec, bound) if codec.nbytes is not None else None
    return out


def shape_text(spec: Spec, bound: Bound) -> list[int | None]:
    """The array entry's shape with bound symbols filled in and free dims as ``None``."""
    out: list[int | None] = []
    for x in spec.get("shape") or ():
        if isinstance(x, int):
            out.append(int(x))
        elif str(x) in bound:
            out.append(int(bound[str(x)]))
        else:
            out.append(None)
    return out


def itemsize(dtype: str) -> int:
    return int(np.dtype(dtype).itemsize)


def _unused(*_a: Any) -> None:  # keep `math` imported for folder codecs that expect it re-exported
    return None


_unused(math)
