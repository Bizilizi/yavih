"""``FeatureWriter``: one shard tar + sidecar, written tmp → ``os.replace``.

Deterministic tars: members with ``mtime = 0``, ``uid = gid = 0``, mode ``0644``; ``.npy`` v1.0
headers via ``np.lib.format.write_array(allow_pickle=False)``; tar sha256 computed
streaming through a hashing file wrapper (the 2 GiB shard is never re-read).

Invariants the writer enforces (so ``validate_features`` finds a well-formed shard or nothing):
regular members in strictly ascending row order inside ``[lo, hi)``; canaries only after all
regular members and only for rows actually written; every row of the range either written or
recorded missing before ``close``; ``_pos.npy`` the first member when ``pos_mode == constant``.

The sidecar carries ``model`` and ``reading`` copied from ``describe()`` (names the user
chose) and ``bridge.describe_sha256`` (consistency digest).

Single writer: the tar is written to a per-writer tmp
(``<tar>.<host>.<pid>.<thread>.tmp``, created ``O_EXCL``), never a name another writer shares; with a ``claim``
(``yavih.io.claim``) the writer checks it still holds the shard before the tar rename and before the sidecar, and
checks the renamed tar's size against the bytes it wrote before writing the sidecar (the DONE marker).
"""

from __future__ import annotations

import hashlib
import inspect
import io
import os
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from yavih import codecs as _codecs
from yavih import hashing
from yavih.io.format import (
    CONSTANT_POS_MEMBER,
    FEATURE_DTYPES,
    FEATURE_FORMAT_VERSION,
    POS_MODES,
    FeatureMetaError,
    FeatureShapeError,
    FeatureValueError,
    canary_basename,
    member_basename,
    npy_bytes,
    output_member_name,
    pos_sha256,
    shard_sidecar_name,
    shard_tar_name,
)
from yavih.io.sidecar import build_sidecar, jsonable, utc_now, write_json_atomic, writer_tmp_name
from yavih.versions import DESCRIBE_SCHEMA, __version__, writer_stamp

if TYPE_CHECKING:  # pragma: no cover
    from yavih.benchmark import Benchmark
    from yavih.bridge import Bridge

__all__ = ["FeatureWriter", "FP16_MAX_ABS", "split_dir_for"]

#: float16 saturation guard: a value above this is refused with a hint to use float32.
FP16_MAX_ABS: float = 1e4

#: Write buffer of the tmp tar and the tar's member copy buffer (bytes): a large clip's ``.feat.npy`` reaches the
#: file in a few large writes instead of 16 KiB ones. Buffering never changes the bytes written.
WRITE_BUFFER_BYTES: int = 8 << 20


def split_dir_for(out_dir: str | os.PathLike, split: str) -> Path:
    """``<out_dir>/features/<split>`` — unless ``out_dir`` already IS that split directory
    (``…/features/<split>``), which is accepted as-is so callers holding a unit's ``out`` need no
    re-derivation."""
    out = Path(out_dir)
    if out.name == split and out.parent.name == "features":
        return out
    return out / "features" / split


class _HashingFile:
    """Write-only file wrapper: forwards to the real file, hashes every byte and counts them, so the tar
    sha256 is known the moment the tar is closed (the tar sha256 is computed streaming)."""

    def __init__(self, fh: io.BufferedWriter) -> None:
        self._fh = fh
        self.sha = hashlib.sha256()
        self.nbytes = 0

    def write(self, data: bytes) -> int:
        n = self._fh.write(data)
        self.sha.update(data)
        self.nbytes += len(data)
        return n

    def tell(self) -> int:
        return self.nbytes

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _to_numpy(x: Any) -> np.ndarray:
    """ndarray as-is; anything else via ``.detach().float().cpu().numpy()``."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _to_pos(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif hasattr(x, "detach"):
        arr = x.detach().float().cpu().numpy()
    else:
        arr = np.asarray(x)
    return np.ascontiguousarray(arr, dtype=np.float32)


def output_schema_of(bench: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """``(declared schema | None, {key: (entry, codec)} | None)``: ``None, None`` for the default feature matrix."""
    oc = getattr(getattr(bench, "spec", None), "output_contract", None)
    schema = getattr(oc, "schema", None) if oc is not None else None
    if schema is None and isinstance(oc, Mapping):
        schema = oc.get("schema")
    if schema is None or _codecs.is_default_schema(schema):
        return None, None
    codecs = getattr(bench, "codecs", None) or _codecs.registry()
    return {str(k): dict(v) for k, v in schema.items()}, _codecs.resolve_schema(schema, codecs)


def _duration(info: Any) -> float | None:
    d = info.get("duration_s") if isinstance(info, dict) else None
    return None if d is None else float(d)


class FeatureWriter:
    """Writes one shard of a split. Context manager: ``__exit__`` with an exception → ``abort()``."""

    def __init__(
        self,
        out_dir: str | os.PathLike,
        split: str,
        *,
        shard: tuple[int, int],
        rows: tuple[int, int],  # half-open [lo, hi)
        bench: Benchmark,
        bridge: Bridge,
        describe: dict,
        source_sha256: str,
        claim: Any = None,
        batch_size: int | None = None,
    ) -> None:
        """``claim``: the shard's ``yavih.io.claim.Claim`` (verified before publishing; ``None`` = unclaimed, the
        library default for tests and single-process use). ``batch_size``: the batch alignment of the split's frozen
        layout recorded in ``shard.batch_size`` (default: the bridge's)."""
        index, n = int(shard[0]), int(shard[1])
        lo, hi = int(rows[0]), int(rows[1])
        if not (0 <= index < n):
            raise ValueError(f"shard index {index} out of range for n={n}")
        if lo < 0 or hi < lo:
            raise ValueError(f"rows must be a half-open [lo, hi) with 0 <= lo <= hi, got {rows}")
        # The output schema: the default feature matrix (written exactly as format 4: .feat.npy / .pos.npy and the
        # `features` block) or a declared schema (generic: one member per key, the `outputs` block).
        self.schema, self.resolved = output_schema_of(bench)
        self.generic = self.resolved is not None
        if self.generic:
            dtype, pos_mode = "float32", "none"  # unused by the generic path; the `features` block records nulls
        else:
            dtype = str(getattr(bridge, "dtype", describe.get("dtype")))
            pos_mode = str(getattr(bridge, "pos_mode", describe.get("pos_mode")))
            if dtype not in FEATURE_DTYPES:
                raise ValueError(f"dtype {dtype!r} not in {sorted(FEATURE_DTYPES)}")
            if pos_mode not in POS_MODES:
                raise ValueError(f"pos_mode {pos_mode!r} not in {sorted(POS_MODES)}")

        self.split = split
        self.split_dir = split_dir_for(out_dir, split)
        self.shard_index, self.n_shards = index, n
        self.lo, self.hi = lo, hi
        self.dtype = np.dtype(dtype)
        self.pos_mode = pos_mode
        self.batch_size = int(batch_size) if batch_size is not None else int(getattr(bridge, "batch_size", describe.get("batch_size", 1)))
        self.claim = claim
        self.describe = dict(describe)
        model, reading = self.describe.get("model"), self.describe.get("reading")
        if not isinstance(model, str) or not model or not isinstance(reading, str) or not reading:
            raise ValueError(f"describe() must declare 'model' and 'reading', got model={model!r}, reading={reading!r}")
        self.model: str = model
        self.reading: str = reading
        self.source_sha256 = source_sha256
        self.bench = bench
        self.bridge = bridge
        # a recorded label only: Bridge.name when set, else the class name
        self.bridge_name = str(getattr(bridge, "name", "") or type(bridge).__name__)
        self.describe_sha256 = hashing.describe_sha256(self.describe)

        self.N: int | None = None
        self.D: int | None = None
        self.bound: dict[str, int] = {}  # generic: every schema symbol, bound by the first row that carries it
        self.emitted: list[str] | None = None  # generic: the schema keys this feature set carries (fixed by the first write)
        self.key_dtypes: dict[str, str] = {}  # generic: the stored dtype of each array key (one per feature set)
        self.members: list[dict[str, Any]] = []
        self.canary_members: list[dict[str, Any]] = []
        self.missing: list[dict[str, Any]] = []
        self.started = utc_now()
        self._written_rows: set[int] = set()
        self._missing_rows: set[int] = set()
        self._last_row = lo - 1
        self._pos_payload: bytes | None = None  # serialized _pos.npy (constant mode)
        self._pos_array: np.ndarray | None = None
        self._canary_phase = False
        self._closed = False

        self.tar_path = self.split_dir / shard_tar_name(index)
        self.sidecar_path = self.split_dir / shard_sidecar_name(index)
        # one tmp per writer (host, pid, thread) and O_EXCL ("xb"): two writers on one shard never share or truncate a
        # file, and neither can rename the other's partial tar into place
        self.tmp_path = self.split_dir / writer_tmp_name(self.tar_path.name)
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self._raw = open(self.tmp_path, "xb", buffering=WRITE_BUFFER_BYTES)
        self._hf = _HashingFile(self._raw)
        # USTAR: fixed 512-byte headers, names < 100 chars, sizes < 8 GiB — the most portable and the
        # most predictable layout (every member's data starts exactly one block after its header).
        self._tar = tarfile.open(fileobj=self._hf, mode="w", format=tarfile.USTAR_FORMAT, copybufsize=WRITE_BUFFER_BYTES)

    # ------------------------------------------------------------------ members

    @property
    def n_range(self) -> int:
        return self.hi - self.lo

    def _tarinfo(self, name: str, size: int) -> tarfile.TarInfo:
        ti = tarfile.TarInfo(name)
        ti.size = size
        ti.mtime = 0
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mode = 0o644
        ti.type = tarfile.REGTYPE
        return ti

    def _add_member(self, name: str, payload: bytes) -> int:
        """Append a member; returns the tar offset of its first data byte."""
        self._tar.addfile(self._tarinfo(name, len(payload)), io.BytesIO(payload))
        padded = -(-len(payload) // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
        return self._tar.offset - padded

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("FeatureWriter is closed")

    def _check_row(self, row: int, clip_id: str) -> None:
        if not (self.lo <= row < self.hi):
            raise ValueError(f"{clip_id}: row {row} outside the shard range [{self.lo}, {self.hi})")
        if row in self._written_rows or row in self._missing_rows:
            raise ValueError(f"{clip_id}: row {row} already written or recorded missing (E_ROWS)")

    def _fix_shape(self, clip_id: str, shape: tuple[int, ...]) -> None:
        if self.N is None:
            if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
                raise FeatureShapeError(clip_id, shape, ("N>=1", "D>=1"))
            self.N, self.D = int(shape[0]), int(shape[1])
        elif tuple(shape) != (self.N, self.D):
            raise FeatureShapeError(clip_id, shape, (self.N, self.D))

    def _cast(self, clip_id: str, x: np.ndarray) -> np.ndarray:
        """Cast to the declared dtype with the value checks. The fp16 range check runs on the
        float32 values BEFORE the cast so an overflow is reported as saturation (with the float32 hint),
        not as a bare non-finite."""
        x32 = np.ascontiguousarray(x, dtype=np.float32) if x.dtype != np.float32 else x
        if not np.isfinite(x32).all():
            raise FeatureValueError(clip_id, "non-finite feature values (NaN/Inf) before cast; the bridge must return finite features")
        if self.dtype == np.float16:
            m = float(np.abs(x32).max()) if x32.size else 0.0
            if m > FP16_MAX_ABS:
                raise FeatureValueError(
                    clip_id,
                    f"max|x| = {m:.4g} > {FP16_MAX_ABS:g} would saturate float16; set Bridge.dtype = 'float32' "
                    f"(or normalize inside the bridge)",
                )
        out = np.ascontiguousarray(x32, dtype=self.dtype)
        if not np.isfinite(out).all():
            raise FeatureValueError(clip_id, f"non-finite after cast to {self.dtype.name}")
        return out

    def _ensure_constant_pos(self, pos: Any, clip_id: str) -> None:
        if pos is None:
            if self._pos_payload is None:
                raise ValueError(f"{clip_id}: pos_mode is 'constant' but the first write passed pos=None")
            return
        arr = _to_pos(pos)
        if arr.ndim == 3 and arr.shape[0] >= 1 and all(np.array_equal(arr[0], a) for a in arr[1:]):
            arr = arr[0]  # a bridge that broadcast its constant pos per sample
        if arr.shape != (self.N, 4):
            raise ValueError(f"{clip_id}: constant pos shape {arr.shape} != ({self.N}, 4)")
        if self._pos_payload is None:
            if self.members or self.canary_members:
                raise RuntimeError("internal: _pos.npy must be the first member")
            payload, _ = npy_bytes(arr)
            self._add_member(CONSTANT_POS_MEMBER, payload)
            self._pos_payload, self._pos_array = payload, arr
        elif not np.array_equal(arr, self._pos_array):
            raise ValueError(f"{clip_id}: constant pos differs from the first write's (identical across shards)")

    def _check_meta(self, clip_id: str, meta: dict | None) -> dict[str, Any]:
        """``meta[i] = {"media_key", "media", "used", "canary_of"?}``; ``FeatureMetaError`` when
        ``media`` is absent or ``used`` lacks a stream of ``describe()["input"]``."""
        m = dict(meta) if meta is not None else {}
        media = m.get("media")
        if not isinstance(media, dict):
            raise FeatureMetaError(clip_id, "row meta lacks `media` (the RawStream facts run_extract copies)")
        used = m.get("used")
        if not isinstance(used, dict):
            raise FeatureMetaError(clip_id, "row meta lacks `used` (what the bridge consumed per stream)")
        from yavih.bridge import input_streams

        missing = [s for s in input_streams(self.describe.get("input")) if s not in used]
        if missing:
            raise FeatureMetaError(clip_id, f"`used` lacks stream(s) {missing} declared in describe()['input']")
        return m

    def write(
        self,
        rows: list[int],
        clip_ids: list[str],
        feats: Any,
        pos: Any = None,
        meta: list[dict] | None = None,
        canary: bool = False,
    ) -> None:
        """Assert lengths; ``.detach().float().cpu().numpy()`` when not an ndarray; fix ``(N, D)`` from the first
        call (``FeatureShapeError`` after); cast to the declared dtype; ``FeatureValueError`` on non-finite or
        float16 ``max|x| > 1e4``; write ``.json``, ``.feat.npy``, optional ``.pos.npy`` members."""
        self._check_open()
        rows = [int(r) for r in rows]
        clip_ids = [str(c) for c in clip_ids]
        feats_np = _to_numpy(feats)
        if not (len(rows) == len(clip_ids) == feats_np.shape[0]):
            raise AssertionError(
                f"write: len(rows)={len(rows)}, len(clip_ids)={len(clip_ids)}, feats.shape[0]={feats_np.shape[0]} must agree"
            )
        if meta is not None and len(meta) != len(rows):
            raise AssertionError(f"write: len(meta)={len(meta)} != len(rows)={len(rows)}")
        if len(rows) == 0:
            return
        first = clip_ids[0]
        if feats_np.ndim != 3:
            raise FeatureShapeError(first, feats_np.shape, ("B", "N", "D"))
        self._fix_shape(first, feats_np.shape[1:])
        for cid in clip_ids:
            # N and D are fixed per feature set, so a per-sample check is the same check; the loop
            # exists only to name the clip in the error.
            self._fix_shape(cid, feats_np.shape[1:])

        if canary:
            for r, cid in zip(rows, clip_ids, strict=True):
                if r not in self._written_rows:
                    raise ValueError(f"{cid}: canary of row {r}, which was not written in this shard")
            self._canary_phase = True
        else:
            if self._canary_phase:
                raise ValueError("regular members after canary members (canaries come after all regular members)")
            for r, cid in zip(rows, clip_ids, strict=True):
                self._check_row(r, cid)
            if any(b <= a for a, b in zip([self._last_row, *rows], rows, strict=False)):
                raise ValueError(f"rows {rows} not strictly ascending after row {self._last_row}")

        # Positions: validated before any member of this call is written so a bad call writes nothing.
        pos_arrays: list[np.ndarray] | None = None
        if self.pos_mode == "none":
            if pos is not None:
                raise ValueError(f"{first}: pos_mode is 'none' but encode returned pos")
        elif self.pos_mode == "constant":
            if not canary:
                self._ensure_constant_pos(pos, first)
        elif not canary:
            if pos is None:
                raise ValueError(f"{first}: pos_mode is 'per_sample' but encode returned pos=None")
            parr = _to_pos(pos)
            if parr.shape != (len(rows), self.N, 4):
                raise ValueError(f"{first}: per_sample pos shape {parr.shape} != ({len(rows)}, {self.N}, 4)")
            pos_arrays = [parr[i] for i in range(len(rows))]

        metas = [self._check_meta(cid, meta[i] if meta is not None else None) for i, cid in enumerate(clip_ids)]
        casted = self._cast(first, feats_np)
        for i, (r, cid) in enumerate(zip(rows, clip_ids, strict=True)):
            m = metas[i]
            rec: dict[str, Any] = {
                "row": r,
                "clip_id": cid,
                "media_key": m.pop("media_key", None),
                "media": m.pop("media"),
                "used": m.pop("used"),
            }
            if canary:
                rec["canary_of"] = r
            m.pop("canary_of", None)
            rec.update(m)  # any further provenance run_extract adds (JSON primitives)
            base = canary_basename(r) if canary else member_basename(r)
            json_payload = hashing.canon(jsonable(rec)).encode("utf-8")
            feat_payload, header_len = npy_bytes(casted[i])
            self._add_member(base + ".json", json_payload)
            offset_data = self._add_member(base + ".feat.npy", feat_payload)
            # ``nbytes`` is the ARRAY's byte count (e.g. 1568·1024·2 for float16 [1568, 1024]); the serialized
            # member is ``header_len + nbytes`` and a reader maps the array at ``offset_data + header_len``.
            entry: dict[str, Any] = {
                "row": r,
                "clip_id": cid,
                "media_key": rec["media_key"],
                "offset_data": offset_data,
                "header_len": header_len,
                "nbytes": len(feat_payload) - header_len,
                # each stream's media.duration_s (null for images), so seq turns t into seconds
                # without opening members.
                "duration_s": {str(k): _duration(v) for k, v in rec["media"].items()},
            }
            if pos_arrays is not None:
                pos_payload, pos_header_len = npy_bytes(pos_arrays[i])
                entry["pos_offset_data"] = self._add_member(base + ".pos.npy", pos_payload)
                entry["pos_header_len"] = pos_header_len
                entry["pos_nbytes"] = len(pos_payload) - pos_header_len
            if canary:
                self.canary_members.append(entry)
            else:
                self.members.append(entry)
                self._written_rows.add(r)
                self._last_row = r

    def write_outputs(
        self,
        rows: list[int],
        clip_ids: list[str],
        outputs: Mapping[str, Any],
        meta: list[dict] | None = None,
        canary: bool = False,
    ) -> None:
        """The generic path (a declared ``output_contract.schema``): ``outputs = {key: values}`` with one value per
        row for every key the feature set carries (an ``[B, ...]`` array or a list). Every non-optional key is
        required, the carried keys are fixed by the first write, each value passes its codec's ``check`` (symbols
        bound once per feature set; ``FeatureShapeError`` for shape/dtype, ``FeatureValueError`` for values), and
        one ``{row}.{key}.{ext}`` member is written per key after the row's ``.json``."""
        self._check_open()
        if not self.generic:
            raise RuntimeError("write_outputs: this benchmark has the default schema; use write(rows, clip_ids, feats, pos)")
        assert self.resolved is not None
        rows = [int(r) for r in rows]
        clip_ids = [str(c) for c in clip_ids]
        if meta is not None and len(meta) != len(rows):
            raise AssertionError(f"write_outputs: len(meta)={len(meta)} != len(rows)={len(rows)}")
        if len(rows) == 0:
            return
        first = clip_ids[0]
        unknown = sorted(set(outputs) - set(self.resolved))
        if unknown:
            raise FeatureShapeError(first, tuple(unknown), tuple(self.resolved))
        keys = [k for k in self.resolved if k in outputs]
        missing = [k for k, (spec, _) in self.resolved.items() if k not in outputs and not spec.get("optional", False)]
        if missing:
            raise FeatureMetaError(first, f"outputs lack the schema key(s) {missing}")
        if self.emitted is None:
            self.emitted = keys
        elif keys != self.emitted:
            raise FeatureMetaError(first, f"outputs carry keys {keys}, earlier rows carried {self.emitted} (one key set per feature set)")
        per_key: dict[str, list[Any]] = {}
        for k in keys:
            vals = outputs[k]
            seq = list(vals) if not isinstance(vals, np.ndarray) else [vals[i] for i in range(vals.shape[0])]
            if len(seq) != len(rows):
                raise AssertionError(f"write_outputs: outputs[{k!r}] holds {len(seq)} values for {len(rows)} rows")
            per_key[k] = seq

        if canary:
            for r, cid in zip(rows, clip_ids, strict=True):
                if r not in self._written_rows:
                    raise ValueError(f"{cid}: canary of row {r}, which was not written in this shard")
            self._canary_phase = True
        else:
            if self._canary_phase:
                raise ValueError("regular members after canary members (canaries come after all regular members)")
            for r, cid in zip(rows, clip_ids, strict=True):
                self._check_row(r, cid)
            if any(b <= a for a, b in zip([self._last_row, *rows], rows, strict=False)):
                raise ValueError(f"rows {rows} not strictly ascending after row {self._last_row}")

        # every value is checked before any member of this call is written: a bad call writes nothing
        encoded: list[dict[str, bytes]] = [{} for _ in rows]
        header_lens: list[dict[str, int]] = [{} for _ in rows]
        for k in keys:
            spec, codec = self.resolved[k]
            for i, cid in enumerate(clip_ids):
                v = per_key[k][i]
                if codec.name == "array":
                    v = _codecs.to_numpy(v)
                    if v.dtype.name != self.key_dtypes.setdefault(k, v.dtype.name):
                        raise FeatureShapeError(cid, (k, v.dtype.name), (k, self.key_dtypes[k]))
                findings = codec.check(v, spec, self.bound)
                errors = [f for f in findings if f.is_error]
                if errors:
                    msg = f"output {k!r}: " + "; ".join(f"{f.code}: {f.message}" for f in errors)
                    if any(f.code == "E_MEMBERS" for f in errors):
                        raise FeatureShapeError(cid, (k,), (msg,))
                    raise FeatureValueError(cid, msg)
                payload = codec.encode(v, spec)
                encoded[i][k] = payload
                if codec.name == "array":
                    header_lens[i][k] = len(payload) - int(_codecs.to_numpy(v).nbytes)
        metas = [self._check_meta(cid, meta[i] if meta is not None else None) for i, cid in enumerate(clip_ids)]
        for i, (r, cid) in enumerate(zip(rows, clip_ids, strict=True)):
            m = metas[i]
            rec: dict[str, Any] = {"row": r, "clip_id": cid, "media_key": m.pop("media_key", None), "media": m.pop("media"), "used": m.pop("used")}
            if canary:
                rec["canary_of"] = r
            m.pop("canary_of", None)
            rec.update(m)
            base = canary_basename(r) if canary else member_basename(r)
            self._add_member(base + ".json", hashing.canon(jsonable(rec)).encode("utf-8"))
            parts: dict[str, dict[str, int]] = {}
            for k in keys:
                spec, codec = self.resolved[k]
                payload = encoded[i][k]
                off = self._add_member(output_member_name(r, k, codec.ext_for(spec), canary=canary), payload)
                part = {"offset_data": off, "nbytes": len(payload)}
                if k in header_lens[i]:
                    part["header_len"] = header_lens[i][k]
                    part["nbytes"] = len(payload) - header_lens[i][k]
                parts[k] = part
            entry: dict[str, Any] = {
                "row": r, "clip_id": cid, "media_key": rec["media_key"], "parts": parts,
                "duration_s": {str(k): _duration(v) for k, v in rec["media"].items()},
            }
            if canary:
                self.canary_members.append(entry)
            else:
                self.members.append(entry)
                self._written_rows.add(r)
                self._last_row = r

    def _outputs_block(self) -> dict[str, Any]:
        """The sidecar's ``outputs`` block: the declared schema (``null`` = the default feature matrix), its sha256,
        the bound symbols and, per carried key, the codec, its version, the member extension and an array's dtype."""
        if not self.generic:
            return {"schema": None, "schema_sha256": _codecs.schema_sha256(None),
                    "bound": {"N": self.N, "D": self.D} if self.N is not None else {}, "keys": None}
        assert self.resolved is not None
        keys = {}
        for k in self.emitted or []:
            spec, codec = self.resolved[k]
            keys[k] = {"codec": codec.name, "version": int(codec.version), "ext": codec.ext_for(spec), "dtype": self.key_dtypes.get(k)}
        return {"schema": self.schema, "schema_sha256": _codecs.schema_sha256(self.schema), "bound": dict(self.bound), "keys": keys}

    def record_missing(self, row: int, clip_id: str, reason: str) -> None:
        """A missing row: it goes to the sidecar ``missing`` list and is NOT written."""
        self._check_open()
        row = int(row)
        self._check_row(row, clip_id)
        self.missing.append({"row": row, "clip_id": str(clip_id), "reason": str(reason)})
        self._missing_rows.add(row)

    # ------------------------------------------------------------------ close / abort

    def _benchmark_block(self) -> dict[str, Any]:
        b = self.bench
        spec = getattr(b, "spec", None)
        pv = getattr(spec, "protocol_version", None)
        if pv is None and isinstance(spec, dict):
            pv = spec.get("protocol_version")
        media_fp, media_info, roots = None, None, {}
        try:
            # Provenance only (never hashed): a benchmark whose data_status cannot be
            # computed here still gets a complete sidecar.
            ds = b.data_status()
            media_fp = getattr(ds, "media_fingerprint", None)
            media_info = getattr(ds, "media_info_sha256", None)
            roots = dict(getattr(ds, "roots", {}) or {})
        except Exception:  # noqa: BLE001 - provenance must never fail a finished shard
            pass
        hashes = getattr(b, "hashes", None)
        return {
            "name": getattr(b, "name", None),
            "protocol_version": pv,
            "manifest_hash": getattr(b, "manifest_hash", None),
            # identity of the shard: the input side only; its components name what differs on a
            # mismatch (E_HASH_INPUT). protocol_hash / scoring_hash: what the benchmark was when written (provenance).
            "input_hash": getattr(b, "input_hash", None),
            "input_components": dict(getattr(hashes, "input_components", None) or {}) or None,
            "folder_source": dict(getattr(b, "folder_source", None) or {}) or None,  # provenance, never identity
            "protocol_hash": getattr(b, "protocol_hash", None),
            "scoring_hash": getattr(b, "scoring_hash", None),
            "media_fingerprint": media_fp,
            "media_info_sha256": media_info,
            "data_roots": roots,
        }

    def _bridge_block(self) -> dict[str, Any]:
        source_path: str | None = None
        try:
            f = inspect.getsourcefile(type(self.bridge))
            if f:
                p = Path(f)
                try:
                    source_path = os.fspath(p.relative_to(Path.cwd()))
                except ValueError:
                    source_path = os.fspath(p)
        except (TypeError, OSError):
            source_path = None
        args = getattr(self.bridge, "args", None)
        if args is None:
            args = self.describe.get("args", {})
        return {
            "name": self.bridge_name,
            "describe": self.describe,
            "describe_sha256": self.describe_sha256,
            "describe_schema": DESCRIBE_SCHEMA,
            "source_sha256": self.source_sha256,
            "source_path": source_path,
            "args": dict(args),
        }

    def close(self, *, timing: dict, env: dict, canary_stats: dict, side_effects: dict | None = None) -> Path:
        """Rename the tar, then write the sidecar (the DONE marker); returns the sidecar path.
        Refuses to mark a shard done while a row of its range is neither written nor missing (``E_ROWS``)."""
        self._check_open()
        accounted = len(self._written_rows) + len(self._missing_rows)
        if accounted != self.n_range:
            self.abort()
            raise ValueError(
                f"close: {accounted} of {self.n_range} rows in [{self.lo}, {self.hi}) accounted for; a DONE marker must "
                f"cover every row (E_ROWS) — use abort() to leave nothing behind"
            )
        self._tar.close()
        self._hf.flush()
        os.fsync(self._raw.fileno())
        self._hf.close()
        self._closed = True
        if self.claim is not None:
            try:
                self.claim.verify()  # another writer took the shard over: publish nothing
            except BaseException:
                self._unlink_tmp()
                raise
        os.replace(self.tmp_path, self.tar_path)
        size = os.stat(self.tar_path).st_size
        if size != self._hf.nbytes:
            # a tar that is not the one this writer streamed (and hashed): never mark it done
            raise OSError(f"{self.tar_path.name}: {size} bytes on disk after the rename, this writer wrote {self._hf.nbytes} "
                          "(another writer replaced it?); no sidecar written, rerun the shard")

        timing = dict(timing or {})
        timing.setdefault("started", self.started)
        timing.setdefault("finished", utc_now())
        sidecar = build_sidecar(
            format_version=FEATURE_FORMAT_VERSION,
            yavih_version=__version__,
            benchmark=self._benchmark_block(),
            split=self.split,
            model=self.model,
            reading=self.reading,
            shard={"index": self.shard_index, "n": self.n_shards, "rows": [self.lo, self.hi], "batch_size": self.batch_size},
            rows={
                "n_range": self.n_range,
                "n_written": len(self.members),
                "n_missing": len(self.missing),
                "n_canary": len(self.canary_members),
            },
            missing=sorted(self.missing, key=lambda m: m["row"]),
            features={
                "dtype": None if self.generic else self.dtype.name,
                "N": self.N,
                "D": self.D,
                "pos_mode": self.pos_mode,
                "pos_sha256": pos_sha256(self._pos_payload) if self._pos_payload is not None else None,
            },
            outputs=self._outputs_block(),
            bridge=self._bridge_block(),
            tar={"file": self.tar_path.name, "bytes": self._hf.nbytes, "sha256": self._hf.sha.hexdigest()},
            members=self.members,
            canary=dict(canary_stats or {}),
            side_effects=dict(side_effects or {"writes_outside_run_dir": []}),
            env=dict(env or {}),
            timing=timing,
            canary_members=self.canary_members,
            yavih=writer_stamp(),
        )
        if self.claim is not None:
            sidecar["claim"] = self.claim.to_json()
            self.claim.verify()
        return write_json_atomic(self.sidecar_path, sidecar)

    def abort(self) -> None:
        """Remove the tmp; nothing partial is left behind."""
        if self._closed:
            return
        self._closed = True
        try:
            self._tar.close()
        except Exception:  # noqa: BLE001 - the file goes away regardless
            pass
        try:
            self._hf.close()
        except Exception:  # noqa: BLE001
            pass
        self._unlink_tmp()

    def _unlink_tmp(self) -> None:
        try:
            self.tmp_path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> FeatureWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        elif not self._closed:
            # Leaving the block without close() is not a success: nothing half-written may remain.
            self.abort()

    def __del__(self) -> None:  # pragma: no cover - safety net for forgotten handles
        try:
            if not getattr(self, "_closed", True):
                self.abort()
        except Exception:  # noqa: BLE001
            pass
