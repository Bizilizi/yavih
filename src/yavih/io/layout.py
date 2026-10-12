"""The frozen shard layout of a split directory: ``features/<split>/_layout.json``.

The first extract unit that writes into a split freezes its
layout — ``n``, the batch alignment and the row range of every shard — and everything after reuses it:
``plan build`` plans exactly those shards, ``run_extract`` reads its range from the file and refuses a
``--shard i/m`` of another ``m`` (``E_LAYOUT``, exit 1: re-plan). The layout is not configuration identity:
re-running ``bridge check`` on another GPU type, another ``--shard-minutes`` or a second cluster sharing the run
directory keeps every finished shard (only another configuration's or manifest's shards are moved aside, together
with its layout).

For a split without ``_layout.json`` the layout is inferred from the finished shards of the current configuration
(the ``n`` most of them carry) and frozen on first use. stdlib only.
"""

from __future__ import annotations

import json
import os
import socket
from collections import Counter
from pathlib import Path
from typing import Any

from yavih.versions import DESCRIBE_SCHEMA

__all__ = [
    "LAYOUT_FILE",
    "LAYOUT_VERSION",
    "LAYOUT_IDENTITY_KEYS",
    "LayoutError",
    "layout_identity",
    "read_layout",
    "layout_matches",
    "infer_layout",
    "frozen_layout",
    "freeze_layout",
    "build_layout",
]

LAYOUT_FILE = "_layout.json"
LAYOUT_VERSION = 1
#: The configuration a layout belongs to (dotted keys, as in a sidecar): another manifest, benchmark input side
#: (``input_hash``), model name, reading or bridge configuration owns other shards and therefore another
#: layout. Compared through ``yavih.io.sidecar.identity_value``.
LAYOUT_IDENTITY_KEYS: tuple[str, ...] = (
    "benchmark.manifest_hash",
    "benchmark.input_hash",
    "model",
    "reading",
    "bridge.describe_sha256",
)


class LayoutError(ValueError):
    """A unit asks for a shard plan other than the split's frozen one: a stale plan (exit 1, re-plan)."""

    code = "E_LAYOUT"
    exit_code = 1


def _get(d: Any, dotted: str) -> Any:
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def layout_identity(*, manifest_hash: str | None, input_hash: str | None, model: str | None, reading: str | None,
                    describe_sha256: str | None) -> dict[str, Any]:
    """``{dotted key: value}`` over ``LAYOUT_IDENTITY_KEYS``."""
    return {
        "benchmark.manifest_hash": manifest_hash,
        "benchmark.input_hash": input_hash,
        "model": model,
        "reading": reading,
        "bridge.describe_sha256": describe_sha256,
    }


def read_layout(split_dir: str | os.PathLike) -> dict | None:
    """The split's ``_layout.json``, or ``None`` when absent or unreadable (an unreadable one is never trusted)."""
    try:
        with open(Path(split_dir) / LAYOUT_FILE, encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or not isinstance(rec.get("n"), int) or not isinstance(rec.get("ranges"), list):
        return None
    return rec


def layout_matches(layout: dict | None, ident: dict[str, Any]) -> bool:
    """``layout`` belongs to the configuration ``ident`` (every given key equal, read through
    ``yavih.io.sidecar.identity_value``)."""
    if layout is None:
        return False
    from yavih.io.sidecar import identity_value

    for k, v in ident.items():
        if identity_value(layout, k) != v:
            return False
    return True


def build_layout(ident: dict[str, Any], *, n: int, batch_size: int, n_rows: int, ranges: list | None = None,
                 inferred: bool = False) -> dict:
    """The ``_layout.json`` record: ``ranges`` = ``plan_shards(n_rows, batch_size, n)`` unless given."""
    if ranges is None:
        from yavih.io.shards import plan_shards

        ranges = [list(r) for r in plan_shards(int(n_rows), int(batch_size), int(n))]
    from yavih.io.sidecar import utc_now

    rec: dict[str, Any] = {
        "layout_version": LAYOUT_VERSION,
        "n": int(n),
        "batch_size": int(batch_size),
        "n_rows": int(n_rows),
        "ranges": [[int(a), int(b)] for a, b in ranges],
        "benchmark": {"manifest_hash": ident.get("benchmark.manifest_hash"), "input_hash": ident.get("benchmark.input_hash")},
        "model": ident.get("model"),
        "reading": ident.get("reading"),
        "bridge": {"describe_sha256": ident.get("bridge.describe_sha256"), "describe_schema": DESCRIBE_SCHEMA},
        "created": utc_now(),
        "host": socket.gethostname(),
        "inferred_from_shards": bool(inferred),
    }
    return rec


def infer_layout(split_dir: str | os.PathLike, ident: dict[str, Any], *, n_rows: int | None = None) -> dict | None:
    """The layout of a split without ``_layout.json``: among the finished shards of configuration ``ident``, the
    ``(n, batch_size)`` most of them carry (ties: the larger ``n``). ``None`` when there is no such shard."""
    from yavih.io.sidecar import identity_value

    votes: Counter[tuple[int, int]] = Counter()
    for side in sorted(Path(split_dir).glob("shard-*.json")):
        try:
            with open(side, encoding="utf-8") as fh:
                sc = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(sc, dict) or not all(identity_value(sc, k) == v for k, v in ident.items()):
            continue
        n, bs = _get(sc, "shard.n"), _get(sc, "shard.batch_size")
        if isinstance(n, int) and n >= 1 and isinstance(bs, int) and bs >= 1:
            votes[(n, bs)] += 1
    if not votes:
        return None
    (n, bs), _ = max(votes.items(), key=lambda kv: (kv[1], kv[0][0]))
    if n_rows is None:
        return {"n": n, "batch_size": bs, "ranges": None, "inferred_from_shards": True}
    return build_layout(ident, n=n, batch_size=bs, n_rows=n_rows, inferred=True)


def frozen_layout(split_dir: str | os.PathLike, ident: dict[str, Any], *, n_rows: int | None = None) -> dict | None:
    """What ``plan`` and ``run_extract`` must follow: ``_layout.json`` when it belongs to ``ident``, else the layout
    inferred from ``ident``'s finished shards (a split without ``_layout.json``), else ``None`` (nothing frozen yet). Read-only."""
    rec = read_layout(split_dir)
    if layout_matches(rec, ident):
        return rec
    return infer_layout(split_dir, ident, n_rows=n_rows)


def freeze_layout(split_dir: str | os.PathLike, ident: dict[str, Any], *, n: int, batch_size: int, n_rows: int) -> dict:
    """Return the split's layout for ``ident``, freezing it first when none is frozen: an existing matching
    ``_layout.json`` wins, then one inferred from ``ident``'s finished shards, then ``(n, batch_size)`` of the caller.
    Written with ``O_EXCL`` (two first writers on two hosts: one wins, the other reads the winner's). A
    ``_layout.json`` of another configuration must have been moved aside first (``retire_foreign``)."""
    split_dir = Path(split_dir)
    existing = read_layout(split_dir)
    if existing is not None:
        if layout_matches(existing, ident):
            return existing
        raise LayoutError(f"{split_dir / LAYOUT_FILE} belongs to another configuration; it must be moved aside first")
    rec = infer_layout(split_dir, ident, n_rows=n_rows) or build_layout(ident, n=n, batch_size=batch_size, n_rows=n_rows)
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / LAYOUT_FILE
    from yavih.io.sidecar import writer_tmp_name

    tmp = split_dir / writer_tmp_name(LAYOUT_FILE)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        os.link(tmp, path)  # atomic create-if-absent of a complete file (O_EXCL semantics, NFS-safe)
    except FileExistsError:
        winner = read_layout(split_dir)
        if winner is None or not layout_matches(winner, ident):
            raise LayoutError(f"{path}: a concurrent writer froze another configuration's layout") from None
        rec = winner
    except OSError:  # a filesystem without hard links: O_EXCL on the final name, then fill it
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            winner = read_layout(split_dir)
            if winner is None or not layout_matches(winner, ident):
                raise LayoutError(f"{path}: a concurrent writer froze another configuration's layout") from None
            rec = winner
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=1, sort_keys=True)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return rec
