"""``FeatureReader`` and ``RowView``: memmap-backed access to one split directory.

``rows(idx)`` gathers by manifest row through the sidecar ``members[].offset_data +
header_len`` offsets without extracting: each shard tar is mapped once (read-only
``mmap``) and a member is a zero-copy ``np.frombuffer`` view at its offset. ``iter()`` is the
independent streaming path (stdlib ``tarfile`` + ``np.lib.format.read_array``), so the test
"memmap rows(present_rows) == streamed iter()" cross-checks the two. ``matrix()`` is a contiguous
RAM array when it fits the budget (``min(16 GiB, 0.45 * SLURM_MEM_PER_NODE)``, env
``YAVIH_RAM_BUDGET_GB``), else a memmap view; it is cached per reader (one RAM array per split however many
fits share the reader). Stored fp16 stays fp16 in RAM (the probe upcasts per mini-batch).
"""

from __future__ import annotations

import io
import mmap
import os
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from yavih import hashing
from yavih import resources as _resources
from yavih.io.format import (
    CONSTANT_POS_MEMBER,
    FEATURE_DTYPES,
    FEATURE_FORMAT_VERSION,
    META_FILE,
    FormatError,
    parse_member_name,
    parse_shard_name,
)
from yavih.io.sidecar import read_json, read_not_applicable, read_sidecar

__all__ = [
    "FeatureReader",
    "RowView",
    "ram_budget_bytes",
    "list_sidecars",
    "DEFAULT_RAM_BUDGET_BYTES",
    "RAM_BUDGET_FRACTION",
    "RAM_BUDGET_ENV",
]

DEFAULT_RAM_BUDGET_BYTES: int = _resources.DEFAULT_RAM_BUDGET_BYTES
RAM_BUDGET_FRACTION: float = _resources.RAM_BUDGET_FRACTION
RAM_BUDGET_ENV = _resources.RAM_BUDGET_ENV
_MIB = 1024**2


def ram_budget_bytes(env: dict | None = None) -> int:
    """``YAVIH_RAM_BUDGET_GB`` / ``[resources] ram_budget_gb`` → else ``min(16 GiB, 0.45 × memory limit)`` → else
    16 GiB. The memory limit is SLURM's (``SLURM_MEM_PER_NODE``, else ``SLURM_MEM_PER_CPU`` × CPUs) or
    the cgroup's (Kubernetes, Docker, PBS with cgroups): the limit that OOM-kills a probe, never the node's
    physical RAM. One implementation for the package: ``yavih.resources.ram_budget_bytes``."""
    return _resources.ram_budget_bytes(env)


# The documented signature of ``matrix`` names its parameter ``ram_budget_bytes``, shadowing the function.
_default_budget = ram_budget_bytes


def list_sidecars(split_dir: str | os.PathLike) -> list[Path]:
    """``shard-*.json`` of a split directory sorted by shard index (``_meta.json`` and tmps excluded)."""
    out = []
    for p in Path(split_dir).iterdir():
        if not p.is_file() or not p.name.startswith("shard-") or not p.name.endswith(".json"):
            continue
        try:
            parse_shard_name(p.name)
        except FormatError:
            continue
        out.append(p)
    return sorted(out, key=lambda p: parse_shard_name(p.name)[0])


class RowView:
    """``len``, ``__getitem__(int | array) -> [.., N, D]`` stored dtype, ``.shape``, ``.backend in {"ram", "memmap"}``.

    Indices are POSITIONS into ``present_rows`` (``0 .. n_present-1``), not manifest rows: a probe works on
    the present population and maps back through ``.present_rows`` when it needs manifest rows.
    """

    shape: tuple[int, int, int]
    backend: str

    def __init__(self, reader: FeatureReader, *, array: np.ndarray | None) -> None:
        self._reader = reader
        self._array = array
        self.backend = "ram" if array is not None else "memmap"
        self.present_rows = reader.present_rows
        self.dtype = reader.dtype
        self.shape = (int(len(reader.present_rows)), int(reader.N), int(reader.D))
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int | np.ndarray | Sequence[int]) -> np.ndarray:
        if isinstance(idx, (int, np.integer)):
            i = int(idx)
            if i < 0:
                i += len(self)
            if not (0 <= i < len(self)):
                raise IndexError(f"position {idx} out of range for {len(self)} present rows")
            if self._array is not None:
                return self._array[i]
            return self._reader.rows([int(self.present_rows[i])])[0]
        pos = np.asarray(idx, dtype=np.int64).reshape(-1)
        if pos.size and (pos.min() < -len(self) or pos.max() >= len(self)):
            raise IndexError(f"positions out of range for {len(self)} present rows")
        pos = np.where(pos < 0, pos + len(self), pos)
        if self._array is not None:
            return self._array[pos]
        return self._reader.rows(self.present_rows[pos])

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: D105 - numpy protocol
        arr = self._array if self._array is not None else self._reader.rows(self.present_rows)
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def __repr__(self) -> str:
        return f"RowView(shape={self.shape}, dtype={self.dtype.name}, backend={self.backend!r})"


class _ShardMap:
    """One read-only mmap of a shard tar; members are zero-copy views at their offsets."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = open(path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def view(self, offset: int, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        count = int(np.prod(shape))
        if offset + count * dtype.itemsize > len(self._mm):
            raise FormatError(f"{self.path}: member at offset {offset} runs past the end of the tar")
        return np.frombuffer(self._mm, dtype=dtype, count=count, offset=offset).reshape(shape)

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fh.close()


class FeatureReader:
    """``features_dir`` is a SPLIT directory (``<run_dir>/features/<split>/``).

    Attributes beyond the documented API: ``sidecars`` (all sidecar records by shard index), ``split``, ``pos_mode``,
    ``n_shards``, ``n_present``, ``missing`` (the union of the sidecars' ``missing`` lists),
    ``not_applicable`` (the ``NOT_APPLICABLE.json`` record, else ``None``; then ``N = D = n_rows = 0``).
    """

    meta: dict
    N: int
    D: int
    dtype: np.dtype
    n_rows: int
    present_rows: np.ndarray  # int64, present manifest rows, ascending

    def __init__(self, features_dir: str | os.PathLike, *, require_meta: bool = True) -> None:
        self.features_dir = Path(features_dir)
        if not self.features_dir.is_dir():
            raise FileNotFoundError(f"{self.features_dir}: not a split directory")
        meta_path = self.features_dir / META_FILE
        self.meta = {}
        if meta_path.exists():
            self.meta = read_json(meta_path)
            fv = self.meta.get("format_version")
            if isinstance(fv, bool) or fv != FEATURE_FORMAT_VERSION:
                raise FormatError(f"{meta_path}: unsupported format_version {fv!r} (this yavih reads {FEATURE_FORMAT_VERSION}); re-extract")
        elif require_meta:
            raise FormatError(
                f"{self.features_dir}: no {META_FILE}; the split is not finalized — run `yavih features validate` "
                f"or pass require_meta=False to read raw shards"
            )

        self.not_applicable = read_not_applicable(self.features_dir)
        self.sidecars: list[dict] = [read_sidecar(p) for p in list_sidecars(self.features_dir)]
        self._maps: dict[int, _ShardMap] = {}
        self._const_pos: np.ndarray | None = None
        self._matrix: np.ndarray | None = None  # matrix(): the cached RAM array (one per reader)
        self._closed = False

        if self.not_applicable is not None and self.sidecars:
            raise FormatError(f"{self.features_dir}: NOT_APPLICABLE.json together with shards (E_NA_AND_SHARDS)")
        if self.meta and self.sidecars:
            self._check_meta_matches_sidecars()

        self.split = str(self.meta.get("split") or (self.sidecars[0]["split"] if self.sidecars else self.features_dir.name))
        self._index_sidecars()

    def _check_meta_matches_sidecars(self) -> None:
        """``_meta.json`` is the single source of truth only while it describes the sidecars
        next to it. A re-extraction into the same run dir replaces the shards and leaves the old
        meta behind; its fingerprint is therefore recomputed from the live sidecars and the
        sidecars' ``model``, ``reading`` and ``bridge.describe_sha256`` compared with the meta's (``FormatError``:
        re-run validate)."""
        from yavih.io.finalize import meta_fingerprint
        from yavih.io.sidecar import identity_value

        def who(rec: dict) -> tuple[Any, Any, Any]:
            return rec.get("model"), rec.get("reading"), identity_value(rec, "bridge.describe_sha256")

        meta_who = who(self.meta)
        if any(v is not None for v in meta_who):
            for sc in self.sidecars:
                if who(sc) != meta_who:
                    m, r, d = who(sc)
                    raise FormatError(
                        f"{self.features_dir}: shard {sc.get('shard', {}).get('index')} belongs to {m}/{r} (describe_sha256 "
                        f"{str(d)[:12]}) but {META_FILE} describes {meta_who[0]}/{meta_who[1]} ({str(meta_who[2])[:12]}): stale meta "
                        "after a re-extraction; re-run `yavih features validate`"
                    )
        if self.meta.get("fingerprint") is None:
            return  # not a finalize product (no fingerprint): nothing to cross-check
        try:
            fp = meta_fingerprint(
                self.sidecars,
                pos_sha256=(self.meta.get("features") or {}).get("pos_sha256"),
                n_present=(self.meta.get("coverage") or {}).get("present"),
            )
        except (KeyError, TypeError, ValueError):
            return  # a malformed sidecar is validate's finding (E_FORMAT), not the reader's
        if fp != self.meta.get("fingerprint"):
            raise FormatError(
                f"{self.features_dir}: {META_FILE} fingerprint does not match the sidecars on disk: the shards changed after "
                "validation; re-run `yavih features validate`"
            )

    # ------------------------------------------------------------------ index

    def _index_sidecars(self) -> None:
        feats_meta = self.meta.get("features") if self.meta else None
        dtype_name = N = D = pos_mode = n = None
        rows_parts: list[np.ndarray] = []
        shard_of: list[np.ndarray] = []
        off: list[np.ndarray] = []
        self._members: list[dict] = []
        self._canary_entries: list[tuple[int, dict]] = []  # (shard_index, entry)
        self.missing: list[dict] = []
        self._ranges: list[tuple[int, int]] = []
        max_hi = 0
        seen_idx: set[int] = set()
        for sc in self.sidecars:
            si = int(sc["shard"]["index"])
            if si in seen_idx:
                raise FormatError(f"{self.features_dir}: shard index {si} appears twice (E_SHARD_SET)")
            seen_idx.add(si)
            if n is None:
                n = int(sc["shard"]["n"])
            elif int(sc["shard"]["n"]) != n:
                raise FormatError(f"{self.features_dir}: sidecars disagree on n ({n} vs {sc['shard']['n']})")
            lo, hi = (int(v) for v in sc["shard"]["rows"])
            self._ranges.append((lo, hi))
            max_hi = max(max_hi, hi)
            f = sc["features"]
            if f.get("N") is not None:  # empty shards carry null N/D (no member fixed them)
                if N is None:
                    N, D = int(f["N"]), int(f["D"])
                elif (int(f["N"]), int(f["D"])) != (N, D):
                    raise FormatError(f"{self.features_dir}: sidecars disagree on (N, D) (E_DESCRIBE_MISMATCH)")
            if dtype_name is None:
                dtype_name, pos_mode = str(f["dtype"]), str(f["pos_mode"])
            elif (str(f["dtype"]), str(f["pos_mode"])) != (dtype_name, pos_mode):
                raise FormatError(f"{self.features_dir}: sidecars disagree on dtype/pos_mode (E_DESCRIBE_MISMATCH)")
            members = sc.get("members", [])
            if members:
                r = np.array([int(m["row"]) for m in members], dtype=np.int64)
                rows_parts.append(r)
                shard_of.append(np.full(len(members), si, dtype=np.int64))
                off.append(np.array([int(m["offset_data"]) + int(m["header_len"]) for m in members], dtype=np.int64))
                self._members.extend(members)
            self._canary_entries.extend((si, e) for e in sc.get("canary_members", []))
            self.missing.extend(sc.get("missing", []))

        if feats_meta:
            # _meta.json is the single source of truth once finalized.
            dtype_name = str(feats_meta.get("dtype", dtype_name))
            N = int(feats_meta["N"]) if feats_meta.get("N") is not None else N
            D = int(feats_meta["D"]) if feats_meta.get("D") is not None else D
            pos_mode = str(feats_meta.get("pos_mode", pos_mode))

        if dtype_name is None:
            dtype_name = "float32"
        if dtype_name not in FEATURE_DTYPES:
            raise FormatError(f"{self.features_dir}: dtype {dtype_name!r} not in {sorted(FEATURE_DTYPES)}")
        self.dtype = np.dtype(dtype_name)
        self.N = int(N or 0)
        self.D = int(D or 0)
        self.pos_mode = pos_mode or "none"
        self.n_shards = int(n or 0)

        if rows_parts:
            rows = np.concatenate(rows_parts)
            shard_idx = np.concatenate(shard_of)
            offsets = np.concatenate(off)
            order = np.argsort(rows, kind="stable")
            rows, shard_idx, offsets = rows[order], shard_idx[order], offsets[order]
            self._members = [self._members[i] for i in order]
            if rows.size > 1 and np.any(np.diff(rows) <= 0):
                raise FormatError(f"{self.features_dir}: a row is written twice across shards (E_ROWS)")
        else:
            rows = np.zeros(0, dtype=np.int64)
            shard_idx = np.zeros(0, dtype=np.int64)
            offsets = np.zeros(0, dtype=np.int64)
        self.present_rows = rows
        self._shard_idx = shard_idx
        self._offsets = offsets
        self.n_present = int(rows.size)
        cov = self.meta.get("coverage") if self.meta else None
        self.n_rows = int(cov["nominal"]) if cov and cov.get("nominal") is not None else int(max_hi)

    # ------------------------------------------------------------------ low level

    def _map(self, shard_index: int) -> _ShardMap:
        m = self._maps.get(shard_index)
        if m is None:
            m = _ShardMap(self.features_dir / self.sidecars[self._sidecar_pos(shard_index)]["tar"]["file"])
            self._maps[shard_index] = m
        return m

    def _sidecar_pos(self, shard_index: int) -> int:
        for i, sc in enumerate(self.sidecars):
            if int(sc["shard"]["index"]) == shard_index:
                return i
        raise KeyError(shard_index)

    def _positions(self, idx: Sequence[int] | np.ndarray) -> np.ndarray:
        """Positions into ``present_rows`` for manifest rows ``idx``; ``KeyError`` names a missing row."""
        want = np.asarray(idx, dtype=np.int64).reshape(-1)
        if want.size == 0:
            return np.zeros(0, dtype=np.int64)
        pos = np.searchsorted(self.present_rows, want)
        bad = (pos >= self.present_rows.size) | (self.present_rows[np.minimum(pos, self.present_rows.size - 1)] != want)
        if bad.any():
            raise KeyError(f"rows {want[bad][:10].tolist()} are not present in {self.features_dir} (missing rows)")
        return pos

    def _feat_at(self, pos: int) -> np.ndarray:
        return self._map(int(self._shard_idx[pos])).view(int(self._offsets[pos]), self.dtype, (self.N, self.D))

    # ------------------------------------------------------------------ public API

    def iter(self, *, with_pos: bool = False) -> Iterator[tuple[int, np.ndarray, np.ndarray | None]]:
        """Streaming, row order: ``(row, feat [N, D], pos | None)``. Reads the tars with stdlib ``tarfile``
        (no sidecar offsets involved), so it doubles as an integrity path for ``rows()``."""
        for sc in self.sidecars:
            if not sc.get("members"):
                continue
            path = self.features_dir / sc["tar"]["file"]
            with tarfile.open(path, mode="r:") as tar:
                pending: dict[int, dict[str, Any]] = {}
                const_pos: np.ndarray | None = None
                for ti in tar:
                    if not ti.isfile():
                        continue
                    if ti.name == CONSTANT_POS_MEMBER:
                        const_pos = self._read_npy(tar, ti)
                        continue
                    row, ext, is_canary = parse_member_name(ti.name)
                    if is_canary:
                        continue  # canaries live after all regular members; canaries() reads them
                    entry = pending.setdefault(row, {})
                    if ext == "feat.npy":
                        entry["feat"] = self._read_npy(tar, ti)
                    elif ext == "pos.npy":
                        entry["pos"] = self._read_npy(tar, ti) if with_pos else True
                    else:
                        entry["json"] = True
                    if "feat" in entry and "json" in entry and (self.pos_mode != "per_sample" or "pos" in entry):
                        pending.pop(row)
                        feat = entry["feat"]
                        if feat.shape != (self.N, self.D) or feat.dtype != self.dtype:
                            raise FormatError(f"{path}: member {row} is {feat.dtype} {feat.shape}, expected {self.dtype.name} ({self.N}, {self.D}) (E_MEMBERS)")
                        pos: np.ndarray | None = None
                        if with_pos:
                            pos = const_pos if self.pos_mode == "constant" else (entry.get("pos") if self.pos_mode == "per_sample" else None)
                        yield row, feat, pos
                if pending:
                    raise FormatError(f"{path}: incomplete member groups for rows {sorted(pending)[:10]} (E_MEMBERS)")

    @staticmethod
    def _read_npy(tar: tarfile.TarFile, ti: tarfile.TarInfo) -> np.ndarray:
        fh = tar.extractfile(ti)
        if fh is None:  # pragma: no cover - isfile() filtered already
            raise FormatError(f"cannot extract {ti.name}")
        with fh:
            # numpy's read_array probes fileno() on the stream (numpy >= 2.4), which tarfile's extract
            # object lacks; a member is at most a few MB, so buffering it is free.
            return np.lib.format.read_array(io.BytesIO(fh.read()), allow_pickle=False)

    def rows(self, idx: Sequence[int]) -> np.ndarray:
        """``[len(idx), N, D]`` in the stored dtype; memmap-backed gather by manifest row."""
        pos = self._positions(idx)
        out = np.empty((pos.size, self.N, self.D), dtype=self.dtype)
        for k, p in enumerate(pos):
            out[k] = self._feat_at(int(p))
        return out

    def pos(self, row: int | None = None) -> np.ndarray | None:
        """``None`` for ``pos_mode none``; the constant ``[N, 4]`` for ``constant`` (``row`` ignored); for
        ``per_sample`` the ``[N, 4]`` of ``row``, or ``[n_present, N, 4]`` over all present rows when ``row`` is
        ``None``."""
        if self.pos_mode == "none":
            return None
        if self.pos_mode == "constant":
            if self._const_pos is None:
                for sc in self.sidecars:
                    if sc.get("members"):
                        with tarfile.open(self.features_dir / sc["tar"]["file"], mode="r:") as tar:
                            ti = tar.next()
                            if ti is None or ti.name != CONSTANT_POS_MEMBER:
                                raise FormatError(f"{sc['tar']['file']}: first member is not {CONSTANT_POS_MEMBER}")
                            self._const_pos = np.ascontiguousarray(self._read_npy(tar, ti), dtype=np.float32)
                        break
                else:
                    return None
            return self._const_pos
        if row is None:
            out = np.empty((self.n_present, self.N, 4), dtype=np.float32)
            for k in range(self.n_present):
                out[k] = self._pos_at(k)
            return out
        return np.array(self._pos_at(int(self._positions([row])[0])))

    def _pos_at(self, pos: int) -> np.ndarray:
        m = self._members[pos]
        if "pos_offset_data" not in m:
            raise FormatError(f"member {m['row']}: no .pos.npy offsets in the sidecar (per_sample pos)")
        return self._map(int(self._shard_idx[pos])).view(int(m["pos_offset_data"]) + int(m["pos_header_len"]), np.dtype(np.float32), (self.N, 4))

    @property
    def nbytes(self) -> int:
        """Bytes of all present features in the stored dtype (what ``matrix()`` compares to the budget)."""
        return self.n_present * self.N * self.D * self.dtype.itemsize

    def matrix(self, *, ram_budget_bytes: int | None = None) -> RowView:
        """Contiguous RAM array when ``nbytes <= budget``, else a memmap view. The values and the
        row order are identical in both backends (the backend never enters any hash).

        The RAM array is cached per reader: every lr arm, fold and seed that shares this reader (the probe opens
        one reader per split) gathers from ONE array instead of re-reading the split. A budget that selects the
        memmap backend drops it. The cached array is read-only; an array-index gather (``view[idx]``) returns a fresh copy."""
        budget = _default_budget() if ram_budget_bytes is None else int(ram_budget_bytes)
        if self.nbytes > budget:
            self._matrix = None  # the memmap backend: never keep a RAM array the budget no longer allows
            return RowView(self, array=None)
        if self._matrix is None:
            arr = self.rows(self.present_rows)
            arr.flags.writeable = False
            self._matrix = arr
        return RowView(self, array=self._matrix)

    def clip_ids(self) -> list[str]:
        """clip_ids of the present rows, in row order (parallel to ``present_rows``)."""
        return [str(m["clip_id"]) for m in self._members]

    def media_keys(self) -> list[str | None]:
        return [m.get("media_key") for m in self._members]

    def durations(self, idx: Sequence[int], stream: str) -> np.ndarray:
        """``float64 [len(idx)]``: ``media.duration_s`` of ``stream`` for manifest rows ``idx`` from the sidecars'
        ``members[].duration_s``; NaN when null (an image, or a stream the row lacks)."""
        pos = self._positions(idx)
        out = np.full(pos.size, np.nan, dtype=np.float64)
        for k, p in enumerate(pos.tolist()):
            d = (self._members[p].get("duration_s") or {}).get(stream)
            if d is not None:
                out[k] = float(d)
        return out

    def canaries(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """``(row, canary_feat, original_feat)`` pairs (``c{row}`` members), shard by shard."""
        for si, e in self._canary_entries:
            canary = self._map(si).view(int(e["offset_data"]) + int(e["header_len"]), self.dtype, (self.N, self.D))
            original = self.rows([int(e["row"])])[0]
            yield int(e["row"]), np.array(canary), original

    def fingerprint(self) -> str:
        """``features_fingerprint`` over this split's sidecars (``shards`` in row order)."""
        shards = sorted(
            ([int(sc["shard"]["rows"][0]), int(sc["shard"]["rows"][1]), str(sc["tar"]["sha256"])] for sc in self.sidecars),
            key=lambda s: (s[0], s[1]),
        )
        pos_sha = None
        for sc in self.sidecars:
            if sc["features"].get("pos_sha256") is not None:
                pos_sha = sc["features"]["pos_sha256"]
                break
        return hashing.features_fingerprint(
            format_version=FEATURE_FORMAT_VERSION,
            dtype=self.dtype.name,
            N=self.N,
            D=self.D,
            pos_mode=self.pos_mode,
            pos_sha256=pos_sha,
            shards=shards,
            n_present=self.n_present,
        )

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._matrix = None
        for m in self._maps.values():
            m.close()
        self._maps.clear()

    def __enter__(self) -> FeatureReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def __repr__(self) -> str:
        return f"FeatureReader({os.fspath(self.features_dir)!r}, split={self.split!r}, present={self.n_present}/{self.n_rows}, N={self.N}, D={self.D}, dtype={self.dtype.name})"
