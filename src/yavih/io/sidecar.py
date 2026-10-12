"""Sidecar ``shard-{i:06d}.json`` and ``NOT_APPLICABLE.json`` records:
build, write atomically, read, and the resumption match.

Every JSON file of the format is written tmp → ``os.replace`` so a killed task leaves only a
``.tmp``; the sidecar is written AFTER the tar rename, so its existence is
the DONE marker.

Records carry the bridge's declared ``model`` and ``reading`` (names, never
digests); resumption compares them and ``bridge.describe_sha256`` (consistency,
not identity): a shard written under another bridge configuration is moved to ``_stale/``.

The benchmark side of a shard's identity is ``benchmark.input_hash`` — a scoring edit
(metric pin, ``probe.py``, readout version) leaves finished shards alone. Every comparison goes through
:func:`identity_value`, which maps a recorded input hash through the folder changelog's ``supersedes_input``
(``yavih.hash_history``).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yavih.io.format import (
    FEATURE_FORMAT_VERSION,
    META_FILE,
    NOT_APPLICABLE_FILE,
    SHARD_RE,
    STALE_DIR,
    TMP_SUFFIX,
    FormatError,
    shard_sidecar_name,
    shard_tar_name,
    tmp_name,
)

__all__ = [
    "build_sidecar",
    "write_json_atomic",
    "writer_tmp_name",
    "read_json",
    "read_sidecar",
    "sidecar_matches_plan",
    "write_not_applicable",
    "read_not_applicable",
    "utc_now",
    "jsonable",
    "dotted_get",
    "clean_tmp",
    "move_stale",
    "resume_shard",
    "peek_shard",
    "plan_values",
    "retire_foreign",
    "identity_value",
    "SIDECAR_KEYS",
    "RESUME_MATCH_KEYS",
]

#: Top-level keys of a sidecar, in their documented order.
SIDECAR_KEYS: tuple[str, ...] = (
    "format_version",
    "yavih_version",
    "benchmark",
    "split",
    "model",
    "reading",
    "shard",
    "rows",
    "missing",
    "features",
    "bridge",
    "tar",
    "members",
    "canary",
    "side_effects",
    "env",
    "timing",
)
#: What resumption compares to decide "already done": the manifest and INPUT hashes (never the scoring side), model,
#: reading, the bridge's ``describe_sha256`` (another configuration under the same model name → stale),
#: features.dtype, features.pos_mode — each read through :func:`identity_value`.
#: ``shard.n`` / ``shard.batch_size`` are not identity: the split's layout is
#: frozen in ``_layout.json`` (``yavih.io.layout``) and a shard is checked against that layout instead.
RESUME_MATCH_KEYS: tuple[str, ...] = (
    "benchmark.manifest_hash",
    "benchmark.input_hash",
    "model",
    "reading",
    "bridge.describe_sha256",
    "features.dtype",
    "features.pos_mode",
)


def utc_now() -> str:
    """``2026-10-07T12:00:00Z`` (the format's timestamp form)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonable(x: Any) -> Any:
    """Make ``x`` JSON-serializable: tuples/arrays → lists, numpy scalars → Python scalars, paths →
    str, non-finite floats → ``None`` (JSON has no NaN; a sidecar must stay stock-parseable)."""
    if x is None or isinstance(x, (bool, str)):
        return x
    if hasattr(x, "tolist") and not isinstance(x, (int, float)):
        # numpy arrays and scalars; .tolist() on a 0-d array returns a Python scalar.
        return jsonable(x.tolist())
    if hasattr(x, "item") and not isinstance(x, (int, float)):
        return jsonable(x.item())
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in x]
    if isinstance(x, os.PathLike):
        return os.fspath(x)
    raise TypeError(f"not JSON-serializable: {type(x).__name__}: {x!r}")


def dotted_get(d: dict, key: str, default: Any = None) -> Any:
    """``dotted_get(sidecar, "shard.n")`` → ``sidecar["shard"]["n"]`` (``default`` when any level is absent)."""
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def identity_value(rec: dict, key: str) -> Any:
    """``dotted_get(rec, key)`` for the identity keys, except ``benchmark.input_hash``, which is
    ``hash_history.effective_input_hash(rec["benchmark"])``: the recorded input hash mapped through the folder
    changelog's ``supersedes_input``. Works on sidecars, ``_meta.json`` and ``_layout.json`` records."""
    if not isinstance(rec, dict):
        return None
    if key == "benchmark.input_hash":
        from yavih.hash_history import effective_input_hash

        return effective_input_hash(rec.get("benchmark"))
    return dotted_get(rec, key)


def writer_tmp_name(name: str) -> str:
    """``<name>.<host>.<pid>.<thread>.tmp``: one tmp per writer. ``NOT_APPLICABLE.json`` is written by EVERY
    array task of a split whose bridge raises ``NotApplicable`` at run time, so each writer gets its own tmp."""
    return f"{name}.{socket.gethostname()}.{os.getpid()}.{threading.get_ident()}{TMP_SUFFIX}"


def write_json_atomic(path: str | Path, obj: dict) -> Path:
    """tmp → ``os.replace`` (a killed task leaves only a ``.tmp``). The tmp name is unique per
    writer; should a racing writer still have published ``path`` first, its result is kept (the files
    written this way are idempotent per unit: sidecar, ``NOT_APPLICABLE.json``, profile)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / writer_tmp_name(path.name)
    data = json.dumps(jsonable(obj), indent=1, ensure_ascii=False, allow_nan=False).encode("utf-8")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.replace(tmp, path)
    except FileNotFoundError:
        if not path.exists():
            raise
    return path


def read_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise FormatError(f"{path}: expected a JSON object")
    return obj


def build_sidecar(**parts: Any) -> dict:
    """Assemble the sidecar record from its blocks; every key of ``SIDECAR_KEYS`` must be present.
    Keys come out in the documented order (extra keys, if any, after them)."""
    missing = [k for k in SIDECAR_KEYS if k not in parts]
    if missing:
        raise ValueError(f"build_sidecar: missing blocks {missing}")
    out: dict[str, Any] = {k: parts[k] for k in SIDECAR_KEYS}
    for k, v in parts.items():
        if k not in out:
            out[k] = v
    return out


def _check_format_version(obj: dict, where: str) -> None:
    fv = obj.get("format_version")
    if not isinstance(fv, int) or isinstance(fv, bool):
        raise FormatError(f"{where}: missing or non-integer format_version (E_FORMAT)")
    if fv > FEATURE_FORMAT_VERSION:
        raise FormatError(
            f"{where}: format_version {fv} is newer than this yavih knows ({FEATURE_FORMAT_VERSION}); "
            f"upgrade yavih (E_FORMAT)"
        )


def read_sidecar(path: str | Path) -> dict:
    """Load and refuse a ``format_version`` greater than known (``E_FORMAT``)."""
    obj = read_json(path)
    _check_format_version(obj, str(path))
    return obj


def sidecar_matches_plan(
    sidecar: dict,
    *,
    manifest_hash: str,
    input_hash: str,
    model: str,
    reading: str,
    describe_sha256: str,
    n: int | None = None,
    dtype: str,
    pos_mode: str,
    rows: list[int] | tuple[int, int] | None = None,
) -> bool:
    """Equal on every ``RESUME_MATCH_KEYS`` → the unit is skipped (exit 0); else stale. ``n`` / ``rows``
    (the frozen layout's shard count and this shard's range) are a layout check, not identity: a sidecar of the same
    configuration written under another layout does not count as this shard."""
    want = plan_values(manifest_hash=manifest_hash, input_hash=input_hash, model=model, reading=reading,
                       describe_sha256=describe_sha256, dtype=dtype, pos_mode=pos_mode)
    if not all(identity_value(sidecar, k) == v for k, v in want.items()):
        return False
    if n is not None and dotted_get(sidecar, "shard.n") != int(n):
        return False
    return rows is None or list(dotted_get(sidecar, "shard.rows") or []) == [int(rows[0]), int(rows[1])]


def plan_values(
    *,
    manifest_hash: str,
    input_hash: str,
    model: str | None,
    reading: str | None,
    describe_sha256: str | None,
    dtype: str,
    pos_mode: str,
) -> dict[str, Any]:
    """The ``RESUME_MATCH_KEYS`` values a writer expects (``{dotted key: value}``), as compared by
    :func:`sidecar_matches_plan` and :func:`retire_foreign`. The shard layout is frozen per split
    (``yavih.io.layout``), not part of the identity."""
    return {
        "benchmark.manifest_hash": manifest_hash,
        "benchmark.input_hash": input_hash,
        "model": model,
        "reading": reading,
        "bridge.describe_sha256": describe_sha256,
        "features.dtype": dtype,
        "features.pos_mode": pos_mode,
    }


def write_not_applicable(
    split_dir: str | Path,
    *,
    reason: str,
    benchmark: str,
    split: str,
    model: str | None,
    reading: str | None,
    describe: dict,
    describe_sha256: str | None = None,
) -> Path:
    """Atomically write ``NOT_APPLICABLE.json``. ``describe_sha256`` defaults to the digest of a non-empty
    ``describe`` (``None`` when the bridge raised ``NotApplicable`` before it could describe itself; ``model`` and
    ``reading`` then come from the plan's profile, or are ``None``)."""
    from yavih.hashing import describe_sha256 as _describe_sha256

    split_dir = Path(split_dir)
    if describe_sha256 is None and describe:
        describe_sha256 = _describe_sha256(describe)
    obj = {
        "status": "not_applicable",
        "reason": str(reason),
        "benchmark": benchmark,
        "split": split,
        "model": model,
        "reading": reading,
        "describe_sha256": describe_sha256,
        "bridge": describe,
        "created": utc_now(),
    }
    return write_json_atomic(split_dir / NOT_APPLICABLE_FILE, obj)


def read_not_applicable(split_dir: str | Path) -> dict | None:
    """The ``NOT_APPLICABLE.json`` record of a split directory, or ``None`` when the split has shards (or nothing yet)."""
    p = Path(split_dir) / NOT_APPLICABLE_FILE
    if not p.exists():
        return None
    obj = read_json(p)
    if obj.get("status") != "not_applicable":
        raise FormatError(f"{p}: status must be 'not_applicable'")
    return obj


# ----------------------------------------------------------------------------- resumption


def clean_tmp(split_dir: str | Path, index: int) -> list[Path]:
    """Delete shard ``index``'s ``*.tmp`` files (the first resumption step); returns what was removed. Called only while
    holding the shard's claim (``yavih.io.claim``), so any other writer's tmp of this index is a dead writer's
    leftover. Other shards' tmps are left alone: they may belong to a running sibling task. Both tmp spellings
    are swept: ``<name>.tmp`` and the per-writer ``<name>.<host>.<pid>.<thread>.tmp``."""
    split_dir = Path(split_dir)
    removed = []
    for name in (shard_tar_name(index), shard_sidecar_name(index)):
        candidates = [split_dir / tmp_name(name), *sorted(split_dir.glob(f"{name}.*{TMP_SUFFIX}"))]
        for p in candidates:
            if p.exists() and p not in removed:
                p.unlink()
                removed.append(p)
    return removed


def move_stale(split_dir: str | Path, index: int, *, when: str | None = None) -> Path:
    """Move ``shard-{i}.{tar,json}`` to ``_stale/<utc>/``; returns the stale directory.
    The sidecar is moved FIRST so no window exists where a DONE marker points at a missing tar."""
    split_dir = Path(split_dir)
    stale = split_dir / STALE_DIR / (when or utc_now().replace(":", ""))
    stale.mkdir(parents=True, exist_ok=True)
    for name in (shard_sidecar_name(index), shard_tar_name(index)):
        _move_if_present(split_dir / name, stale / name)
    return stale


def _move_if_present(src: Path, dst: Path) -> bool:
    """Move ``src`` to ``dst``; ``False`` when ``src`` vanished first (a sibling array task of the same split moved it:
    every unit of a reconfigured split sweeps the same foreign shards)."""
    try:
        shutil.move(os.fspath(src), os.fspath(dst))
    except FileNotFoundError:
        return False
    return True


def peek_shard(
    split_dir: str | Path,
    index: int,
    *,
    manifest_hash: str,
    input_hash: str,
    model: str,
    reading: str,
    describe_sha256: str,
    n: int | None = None,
    dtype: str,
    pos_mode: str,
    rows: list[int] | tuple[int, int] | None = None,
) -> str:
    """The READ-ONLY half of :func:`resume_shard`: ``"skip"`` when the unit's sidecar and tar exist and match the plan,
    else ``"run"``. Nothing is moved or deleted: ``extract --expect-profile`` asks it before the model is loaded,
    i.e. before the live configuration has been checked against the profile (``E_CONFIG_DRIFT``), so a shard written by
    the live configuration must not be moved aside on the profile's word."""
    split_dir = Path(split_dir)
    side = split_dir / shard_sidecar_name(index)
    if not side.exists() or not (split_dir / shard_tar_name(index)).exists():
        return "run"
    try:
        sc = read_sidecar(side)
    except (FormatError, ValueError, OSError):
        return "run"
    ok = sidecar_matches_plan(sc, manifest_hash=manifest_hash, input_hash=input_hash, model=model, reading=reading,
                              describe_sha256=describe_sha256, n=n, dtype=dtype, pos_mode=pos_mode, rows=rows)
    return "skip" if ok else "run"


#: The ``_meta.json`` keys that name the configuration it validated: compared by :func:`retire_foreign`.
_META_CONFIG_KEYS: tuple[str, ...] = ("benchmark.manifest_hash", "benchmark.input_hash", "model", "reading", "bridge.describe_sha256")


def retire_foreign(
    split_dir: str | Path,
    want: dict[str, Any],
    *,
    meta: str = "foreign",
    when: str | None = None,
    layout: dict | None = None,
) -> Path | None:
    """Move what another configuration or another shard plan left in a split directory to ``_stale/<utc>/``, in one
    step. The run directory ``runs/<benchmark>/<model>/<reading>/`` does not name the bridge
    configuration, so a reconfigured bridge writes where the previous one wrote; ``resume_shard`` handles only the
    unit's own index, which leaves the old configuration's shards ``k >= n`` (fewer shards, another batch size) and its
    ``_meta.json`` behind.

    ``want`` is ``{dotted key: value}`` over ``RESUME_MATCH_KEYS`` (:func:`plan_values`) or a subset of them (a
    ``NOT_APPLICABLE.json`` writer passes ``model``, ``reading``, ``bridge.describe_sha256`` and the benchmark hashes:
    it has no shard plan). Every finished shard — a readable sidecar with a ``shard-*.tar`` or not — whose sidecar
    differs on any ``want`` key is moved with its tar. An unreadable sidecar, a tar without a sidecar (a sibling may sit
    between its tar rename and its sidecar write) and every ``*.tmp`` are left to their own unit and to validate.

    ``meta``: ``"foreign"`` moves ``_meta.json`` when its configuration (model, reading, ``describe_sha256``, benchmark
    hashes) differs from ``want``'s or when a shard was moved; ``"always"`` moves it unconditionally (a
    ``NOT_APPLICABLE.json`` writer: validate writes ``_meta.json`` only next to shards, so one beside a
    ``NOT_APPLICABLE.json`` belongs to an earlier configuration and must not be read as this one's).

    ``_layout.json`` (``yavih.io.layout``) of another configuration (any ``want`` key differs) is moved with them.
    ``layout``: the split's frozen layout for ``want``'s configuration; a finished shard of that configuration whose
    ``shard.n`` or ``shard.rows`` does not fit it (written under another shard plan) is moved too.

    Returns the ``_stale/<utc>/`` directory when anything was moved, else ``None``. Concurrent sweeps by sibling array
    tasks are safe: a file another task moved first is skipped."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return None
    stamp = when or utc_now().replace(":", "")
    stale = split_dir / STALE_DIR / stamp
    moved = False
    for side in sorted(split_dir.glob("shard-*.json")):
        m = SHARD_RE.match(side.name)
        if m is None:
            continue
        try:
            sc = read_sidecar(side)
        except FileNotFoundError:
            continue
        except (FormatError, ValueError, OSError):
            continue
        idx = int(m.group("index"))
        if all(identity_value(sc, k) == v for k, v in want.items()) and _fits_layout(sc, idx, layout):
            continue
        stale.mkdir(parents=True, exist_ok=True)
        # the sidecar first: no window where a DONE marker points at a missing tar (as move_stale)
        if _move_if_present(side, stale / side.name):
            moved = True
        _move_if_present(split_dir / shard_tar_name(idx), stale / shard_tar_name(idx))
    from yavih.io.layout import LAYOUT_FILE, layout_matches, read_layout

    layout_path = split_dir / LAYOUT_FILE
    if layout_path.exists():
        lrec = read_layout(split_dir)
        ident = {k: v for k, v in want.items() if k in _META_CONFIG_KEYS}
        if lrec is None or not layout_matches(lrec, ident):
            stale.mkdir(parents=True, exist_ok=True)
            moved = _move_if_present(layout_path, stale / LAYOUT_FILE) or moved
    meta_path = split_dir / META_FILE
    if meta_path.exists():
        retire = meta == "always" or moved
        if not retire:
            try:
                mrec = read_json(meta_path)
            except FileNotFoundError:
                mrec = None
            except (FormatError, ValueError, OSError):
                mrec = None
                retire = True  # an unreadable _meta.json is never this configuration's
            if mrec is not None:
                retire = any(identity_value(mrec, k) != want[k] for k in _META_CONFIG_KEYS if k in want)
        if retire:
            stale.mkdir(parents=True, exist_ok=True)
            moved = _move_if_present(meta_path, stale / META_FILE) or moved
    return stale if moved else None


def _fits_layout(sc: dict, idx: int, layout: dict | None) -> bool:
    """A sidecar's ``shard.n`` and ``shard.rows`` agree with the frozen layout (no layout: nothing to check)."""
    if layout is None:
        return True
    if dotted_get(sc, "shard.n") != layout.get("n"):
        return False
    ranges = layout.get("ranges")
    if isinstance(ranges, list) and 0 <= idx < len(ranges):
        return list(dotted_get(sc, "shard.rows") or []) == [int(x) for x in ranges[idx]]
    return False


def resume_shard(
    split_dir: str | Path,
    index: int,
    *,
    manifest_hash: str,
    input_hash: str,
    model: str,
    reading: str,
    describe_sha256: str,
    n: int | None = None,
    dtype: str,
    pos_mode: str,
    rows: list[int] | tuple[int, int] | None = None,
) -> str:
    """The start-of-unit resumption protocol: delete own tmps; a matching sidecar → ``"skip"`` (exit 0);
    a mismatching one → tar + sidecar moved to ``_stale/<utc>/`` and ``"run"``; nothing → ``"run"``.
    A tar without a sidecar (killed between rename and sidecar write) is also moved aside."""
    split_dir = Path(split_dir)
    clean_tmp(split_dir, index)
    side = split_dir / shard_sidecar_name(index)
    tar = split_dir / shard_tar_name(index)
    if side.exists():
        try:
            sc = read_sidecar(side)
            ok = tar.exists() and sidecar_matches_plan(
                sc,
                manifest_hash=manifest_hash,
                input_hash=input_hash,
                model=model,
                reading=reading,
                describe_sha256=describe_sha256,
                n=n,
                dtype=dtype,
                pos_mode=pos_mode,
                rows=rows,
            )
        except (FormatError, ValueError, OSError):
            ok = False
        if ok:
            return "skip"
        move_stale(split_dir, index)
        return "run"
    if tar.exists():
        move_stale(split_dir, index)
    return "run"
