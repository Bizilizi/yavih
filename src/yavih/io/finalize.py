"""``finalize``: write ``_meta.json`` atomically from validated sidecars; the single
writer of that file. Rewrites only if changed.

``_meta.json`` is derived data: everything in it comes from the sidecars plus the verdict of
``validate_features``. Keeping one writer (this module) is what lets the probe trust
``eligibility`` and ``validation.ok`` without re-deriving them.

The record carries ``model`` + ``reading`` (the bridge's declared names); ``fingerprint`` is integrity and
staleness only. Decoders are declared in ``describe()["input"]`` and recorded there, not in a block of their own;
``canary`` carries the tolerance it was judged against; ``rules_checked`` names the declared rules
validate re-evaluated. The ``benchmark`` block is copied verbatim from the sidecars, so whatever
provenance the writer records there (``media_info_sha256``, an ``external`` pin list) passes through.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yavih import hashing
from yavih.io.format import FEATURE_FORMAT_VERSION, META_FILE, FormatError
from yavih.versions import __version__, writer_stamp

__all__ = ["finalize", "build_meta", "read_meta", "meta_fingerprint", "META_KEYS", "META_FORMAT_VERSION", "VOLATILE_META_PATHS"]

#: Top-level keys of a ``_meta.json`` record, in their documented order.
META_KEYS: tuple[str, ...] = (
    "format_version",
    "yavih",
    "finalized",
    "benchmark",
    "split",
    "model",
    "reading",
    "bridge",
    "features",
    "shards",
    "coverage",
    "canary",
    "rules_checked",
    "dead_rows",
    "waived",
    "fingerprint",
    "eligibility",
    "validation",
)
#: The ``_meta.json`` format this module writes (``FEATURE_FORMAT_VERSION``); validate reuses the verdict of a
#: ``_meta.json`` of this format only (``_cheap_recheck``).
META_FORMAT_VERSION: int = int(FEATURE_FORMAT_VERSION)
#: Default batch-invariance tolerance, recorded when neither validate nor a sidecar names one.
DEFAULT_BATCH_TOLERANCE: float = 1e-5
#: Timestamps that differ between two otherwise identical finalizations; ignored by the
#: "rewrite only if changed" rule.
VOLATILE_META_PATHS: tuple[tuple[str, ...], ...] = (("finalized",), ("validation", "validated"), ("yavih", "finalizer_commit"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ordered_shards(sidecars: list[dict]) -> list[dict]:
    return sorted(sidecars, key=lambda s: int(s["shard"]["index"]))


def meta_fingerprint(sidecars: list[dict], *, pos_sha256: str | None = None, n_present: int | None = None) -> str:
    """``features_fingerprint`` from the sidecars alone: ``shards = [[lo, hi, tar sha256]
    …]`` in row order, ``n_present = Σ rows.n_written``. The validator's re-run check recomputes
    this and compares with ``_meta.json.fingerprint``."""
    ordered = _ordered_shards(sidecars)
    first = ordered[0]
    feats = next((s["features"] for s in ordered if (s.get("features") or {}).get("N") is not None), first["features"])
    if pos_sha256 is None:
        pos_sha256 = feats.get("pos_sha256")
    if n_present is None:
        n_present = sum(len(s["members"]) for s in ordered)
    return hashing.features_fingerprint(
        format_version=int(first["format_version"]),
        dtype=str(feats["dtype"]),
        N=int(feats["N"]),
        D=int(feats["D"]),
        pos_mode=str(feats["pos_mode"]),
        pos_sha256=pos_sha256,
        shards=[[int(s["shard"]["rows"][0]), int(s["shard"]["rows"][1]), str(s["tar"]["sha256"])] for s in ordered],
        n_present=int(n_present),
    )


def build_meta(
    split_dir: str | os.PathLike,
    sidecars: list[dict],
    *,
    eligibility: dict,
    validation: dict,
    nominal: int | None = None,
    canary: dict | None = None,
    dead_rows: list[int] | None = None,
    pos_sha256: str | None = None,
    present_clip_ids: list[str] | None = None,
    bytes_total: int | None = None,
    rules_checked: dict | None = None,
    finalized: str | None = None,
    waived: list[dict] | None = None,
) -> dict:
    """Assemble the ``_meta.json`` record: ``coverage.ids_sha256 = sha256("\\n".join(sorted present clip_ids))``,
    ``fingerprint`` as in :func:`meta_fingerprint`, ``shards`` in row order.

    Everything optional is derivable from the sidecars (``nominal`` = the tiled range, ``canary`` =
    the worst sidecar canary, Σ ``rows.n_canary`` and the sidecars' tolerance, clip ids from ``members``,
    ``rules_checked`` = ``{}``); ``validate`` passes what its streaming pass measured so the record never
    disagrees with the verdict.
    """
    if not sidecars:
        raise ValueError("build_meta: no sidecars (an empty split has no _meta.json; NOT_APPLICABLE.json covers the other case)")
    ordered = _ordered_shards(sidecars)
    first = ordered[0]
    # An empty shard (hi == lo) records null N/D/pos_sha256; take them from a shard that wrote rows.
    feat_src = next((s for s in ordered if (s.get("features") or {}).get("N") is not None), first)
    feats = dict(feat_src["features"])
    bridge = dict(first["bridge"])
    if feats.get("N") is None or feats.get("D") is None:
        raise ValueError("build_meta: no shard recorded features.N/D (every shard empty?)")
    N, D = int(feats["N"]), int(feats["D"])
    itemsize = 2 if feats["dtype"] == "float16" else 4

    if present_clip_ids is None:
        present_clip_ids = [str(m["clip_id"]) for s in ordered for m in s["members"]]
    n_present = len(present_clip_ids)
    if nominal is None:
        nominal = max((int(s["shard"]["rows"][1]) for s in ordered), default=0)
    if bytes_total is None:
        bytes_total = n_present * N * D * itemsize + (n_present * N * 16 if feats["pos_mode"] == "per_sample" else 0)
    if canary is None:
        canary = {
            "max_rel_l2": max((float((s.get("canary") or {}).get("max_rel_l2") or 0.0) for s in ordered), default=0.0),
            "n": sum(int((s.get("rows") or {}).get("n_canary") or 0) for s in ordered),
        }
    if canary.get("tolerance") is None:
        tols = [float(t) for s in ordered if (t := (s.get("canary") or {}).get("tolerance")) is not None]
        canary = {**canary, "tolerance": max(tols) if tols else DEFAULT_BATCH_TOLERANCE}
    if pos_sha256 is None:
        pos_sha256 = feats.get("pos_sha256")

    shards = [
        {
            "file": str(s["tar"]["file"]),
            "rows": [int(s["shard"]["rows"][0]), int(s["shard"]["rows"][1])],
            "n_written": int((s.get("rows") or {}).get("n_written", len(s["members"]))),
            "bytes": int(s["tar"]["bytes"]),
            "sha256": str(s["tar"]["sha256"]),
        }
        for s in ordered
    ]
    ids_sha256 = hashlib.sha256("\n".join(sorted(present_clip_ids)).encode("utf-8")).hexdigest()
    meta: dict[str, Any] = {
        "format_version": META_FORMAT_VERSION,
        "yavih": {
            "writer_versions": sorted({str(s.get("yavih_version", "?")) for s in ordered}),
            "writer_commits": sorted({str((s.get("yavih") or {}).get("commit")) for s in ordered if (s.get("yavih") or {}).get("commit")}),
            "finalizer_version": __version__,
            "finalizer_commit": writer_stamp()["commit"],
        },
        "finalized": finalized or _utc_now(),
        "benchmark": dict(first["benchmark"]),
        "split": str(first["split"]),
        "model": first.get("model"),
        "reading": first.get("reading"),
        "bridge": {
            "name": bridge.get("name"),
            "describe": bridge.get("describe"),
            "describe_sha256": bridge.get("describe_sha256"),
            "describe_schema": bridge.get("describe_schema"),
            "source_sha256": bridge.get("source_sha256"),
        },
        "features": {
            "dtype": str(feats["dtype"]),
            "N": N,
            "D": D,
            "pos_mode": str(feats["pos_mode"]),
            "pos_sha256": pos_sha256,
            "bytes": int(bytes_total),
        },
        "shards": shards,
        "coverage": {
            "nominal": int(nominal),
            "present": int(n_present),
            "missing": int(max(int(nominal) - n_present, 0)),
            "ids_sha256": ids_sha256,
        },
        "canary": {"max_rel_l2": float(canary.get("max_rel_l2", 0.0)), "n": int(canary.get("n", 0)), "tolerance": float(canary["tolerance"])},
        "rules_checked": dict(rules_checked or {}),
        "dead_rows": [int(i) for i in (dead_rows or [])],
        # the validation waivers in force (yavih.waivers): [{code, reason, source: describe|cli, count}]
        "waived": [dict(w) for w in (waived or [])],
        "fingerprint": meta_fingerprint(ordered, pos_sha256=pos_sha256, n_present=n_present),
        "eligibility": dict(eligibility),
        "validation": dict(validation),
    }
    return meta


def _strip_volatile(meta: dict) -> dict:
    out = json.loads(json.dumps(meta))
    for path in VOLATILE_META_PATHS:
        node = out
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return out


def finalize(split_dir: str | os.PathLike, meta: dict) -> Path:
    """Atomic write of ``_meta.json``; unchanged content (ignoring the two timestamps) is not
    rewritten, so a re-run of validate leaves the file's mtime alone."""
    path = Path(split_dir) / META_FILE
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh)
        except (OSError, ValueError):
            old = None
        if isinstance(old, dict) and _strip_volatile(old) == _strip_volatile(meta):
            return path
    text = json.dumps(meta, indent=1, sort_keys=False, ensure_ascii=False, allow_nan=False)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def read_meta(split_dir: str | os.PathLike) -> dict:
    """Load ``_meta.json``; refuse a ``format_version`` greater than known."""
    p = Path(split_dir)
    path = p if p.name == META_FILE else p / META_FILE
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if not isinstance(meta, dict):
        raise FormatError(f"{path}: not a JSON object")
    fv = meta.get("format_version")
    if not isinstance(fv, int) or isinstance(fv, bool) or fv < 1:
        raise FormatError(f"{path}: format_version {fv!r} is not a positive int")
    if fv > META_FORMAT_VERSION:
        raise FormatError(f"{path}: format_version {fv} > {META_FORMAT_VERSION} known to this yavih; upgrade yavih")
    return meta
