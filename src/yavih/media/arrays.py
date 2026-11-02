"""Array files: the ``array`` field kind (``.npy``, ``.npz``, ``.safetensors``), read with numpy only.

A file field of kind ``array`` hands the bridge stored arrays: one per file (a point cloud, a spectrogram, an
embedding, a sensor window) or several named ones (``.npz`` / ``.safetensors``: points and normals). The facts are read
from the header alone (``array_facts``: ``dtype``, ``shape``, ``n_arrays``); ``load_arrays`` decodes every array of
the file. Pickle never runs: ``.npy`` / ``.npz`` load with ``allow_pickle=False`` and object dtypes are refused, and a
pickle-based container (``.pt``, ``.pth``, ``.pkl``, ``.ckpt``) is refused by name. A ``layout: npy_rows`` row is
handed over as the ``.npy`` bytes of that row, so it is an ``array`` too.
"""

from __future__ import annotations

import io
import json
import struct
import zipfile
from typing import Any

import numpy as np

__all__ = ["CONTAINERS", "PICKLED", "ArrayError", "container_of", "array_facts", "load_arrays"]

CONTAINERS: tuple[str, ...] = ("npy", "npz", "safetensors")
#: Containers that unpickle on load: refused (a stored file must never run code).
PICKLED: frozenset[str] = frozenset({"pt", "pth", "pkl", "pickle", "ckpt", "joblib"})
#: The name ``load_arrays`` gives the one array of a ``.npy`` file (numpy's own default for ``savez``).
NPY_KEY = "arr_0"
_ST_DTYPES: dict[str, str] = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1",
                              "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "u1", "BOOL": "?"}


class ArrayError(ValueError):
    """An array file yavih refuses or cannot read."""


def container_of(ext: str | None) -> str:
    """The container named by a file extension; ``ArrayError`` for a pickle-based or an unknown one."""
    e = (ext or "").lower().lstrip(".")
    if e in PICKLED:
        raise ArrayError(f".{e} files unpickle on load and are refused: store the arrays as .npy, .npz or .safetensors")
    if e not in CONTAINERS:
        raise ArrayError(f"an array field reads {list(CONTAINERS)}, got {'.' + e if e else 'a file without an extension'}")
    return e


def _npy_header(f: Any, where: str) -> tuple[np.dtype, tuple[int, ...]]:
    try:
        major, _ = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        shape, _, dtype = read(f)  # header only, no data
    except ValueError as e:
        raise ArrayError(f"{where}: not a .npy array ({e})") from e
    if dtype.hasobject:
        raise ArrayError(f"{where}: object arrays are refused (they need pickle)")
    return dtype, tuple(int(s) for s in shape)


def _st_header(data: bytes, where: str) -> dict[str, dict[str, Any]]:
    if len(data) < 8:
        raise ArrayError(f"{where}: not a safetensors file (truncated)")
    (n,) = struct.unpack("<Q", data[:8])
    if n > len(data) - 8:
        raise ArrayError(f"{where}: not a safetensors file (header of {n} bytes in a {len(data)}-byte file)")
    try:
        head = json.loads(data[8 : 8 + n])
    except ValueError as e:
        raise ArrayError(f"{where}: unreadable safetensors header ({e})") from e
    head.pop("__metadata__", None)
    for k, t in head.items():
        if t.get("dtype") not in _ST_DTYPES:
            raise ArrayError(f"{where}: tensor {k!r} is {t.get('dtype')}, which numpy cannot hold ({sorted(_ST_DTYPES)})")
    return head


def _headers(data: bytes, container: str, where: str) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    if container == "npy":
        return {NPY_KEY: _npy_header(io.BytesIO(data), where)}
    if container == "npz":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                out = {}
                for name in z.namelist():
                    if not name.endswith(".npy"):
                        raise ArrayError(f"{where}: member {name!r} of an .npz is not an .npy array")
                    with z.open(name) as f:
                        out[name[: -len(".npy")]] = _npy_header(f, f"{where}[{name}]")
                return out
        except zipfile.BadZipFile as e:
            raise ArrayError(f"{where}: not an .npz archive ({e})") from e
    head = _st_header(data, where)
    return {k: (np.dtype(_ST_DTYPES[t["dtype"]]), tuple(int(s) for s in t["shape"])) for k, t in head.items()}


def array_facts(data: bytes, container: str, where: str = "array") -> dict[str, Any]:
    """``{dtype, shape, n_arrays}`` from the header: ``dtype`` (numpy's name, e.g. ``float32``) and ``shape`` (a list)
    when the file holds exactly one array, else null."""
    heads = _headers(data, container, where)
    if len(heads) == 1:
        dtype, shape = next(iter(heads.values()))
        return {"dtype": dtype.name, "shape": list(shape), "n_arrays": 1}
    return {"dtype": None, "shape": None, "n_arrays": len(heads)}


def load_arrays(data: bytes, container: str, where: str = "array") -> dict[str, np.ndarray]:
    """Every array of the file, by name (a ``.npy`` file's one array is ``arr_0``), in file order."""
    if container == "npy":
        _npy_header(io.BytesIO(data), where)
        return {NPY_KEY: np.load(io.BytesIO(data), allow_pickle=False)}
    if container == "npz":
        _headers(data, container, where)
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    head = _st_header(data, where)
    (n,) = struct.unpack("<Q", data[:8])
    body = memoryview(data)[8 + n :]
    out = {}
    for k, t in head.items():
        lo, hi = (int(x) for x in t["data_offsets"])
        dtype = np.dtype(_ST_DTYPES[t["dtype"]])
        shape = tuple(int(s) for s in t["shape"])
        if not 0 <= lo <= hi <= len(body) or hi - lo != dtype.itemsize * int(np.prod(shape, dtype=np.int64)):
            raise ArrayError(f"{where}: tensor {k!r} has offsets {lo}..{hi} that do not match {t['dtype']} {list(shape)}")
        out[k] = np.frombuffer(body[lo:hi], dtype=dtype).reshape(shape).copy()
    return out
