"""Content hashing: canonical JSON, manifest/protocol/probe/eval/describe/feature/unit hashes.

A comparability key covers only what yavih controls — the benchmark (``protocol_hash``) and the probe
(``probe_hash``), folded into ``eval_hash``. The bridge's ``describe_sha256`` is a consistency and provenance
digest, never identity (the model is the user-chosen name ``describe()["model"]``).

``protocol_hash`` (``yavih-protocol-4``) is
``H(input_hash, scoring_hash)``. ``input_hash`` covers what decides the features a bridge sees and must produce and is
the only benchmark digest feature sets are checked against; ``scoring_hash`` covers metrics, pins, targets, the readout
version and ``probe.py``, so a scoring edit re-probes but never re-extracts. ``reader.py`` / ``probe.py`` enter as the
digest of their normalized source (comments and docstrings do not count). ``describe_sha256`` (``yavih-describe-2``)
leaves the batch size and the decoder builds out.

stdlib + ``yavih.versions`` only: these digests must be computable in the model env
(no torch, maybe no numpy) and in the probe env with identical results (a
parity test checks it). ``numpy`` scalars are normalized by duck typing
on an already-imported ``numpy`` so this module never imports it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from yavih.versions import (
    DESCRIBE_SCHEMA,
    PLAN_SCHEMA_VERSION,
    PROBE_SCHEMA,
    PROTOCOL_SCHEMA,
    READOUT_VERSION,
)

__all__ = [
    "canon",
    "normalize",
    "sha256_hex",
    "sha256_canon",
    "hash12",
    "rng_seed",
    "manifest_hash",
    "normalized_source",
    "source_sha256",
    "protocol_targets",
    "PROTOCOL_MANIFEST_KEYS",
    "INPUT_MANIFEST_KEYS",
    "INPUT_READOUT_PARAMS",
    "INPUT_PROTOCOL_KEYS",
    "protocol_manifest",
    "input_readout_view",
    "input_components",
    "scoring_components",
    "component_digests",
    "input_hash_of",
    "scoring_hash_of",
    "combine_protocol_hash",
    "ProtocolHashes",
    "protocol_hashes",
    "protocol_hash",
    "input_hash",
    "DESCRIBE_SHA_EXCLUDED",
    "DESCRIBE_PROVENANCE_INPUT_KEYS",
    "describe_subset",
    "describe_sha256",
    "PROBE_TRAIN_FIELDS",
    "probe_hash",
    "eval_hash",
    "features_fingerprint",
    "unit_id",
    "inputs_hash",
]

# --------------------------------------------------------------------------- canonical JSON


def _is_numpy_scalar(x: Any) -> bool:
    # Duck-typed so hashing stays stdlib-only; if numpy was never imported there cannot
    # be a numpy scalar in the payload.
    np = sys.modules.get("numpy")
    return np is not None and isinstance(x, np.generic)


def normalize(x: Any) -> Any:
    """Return the JSON-ready form of ``x`` for canonical JSON.

    tuples → lists; numpy scalars → Python scalars; floats → ``float`` (serialized with
    ``repr``, the shortest round-trip form); bools stay bools; dict keys must be ``str``.
    Anything else raises ``TypeError`` — never ``default=str``: a repr silently embedded in a digest makes it
    unstable across versions and hides what was hashed.
    """
    if _is_numpy_scalar(x):
        x = x.item()
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x)
    if isinstance(x, Mapping):
        out: dict[str, Any] = {}
        for k, v in x.items():
            if not isinstance(k, str):
                raise TypeError(f"canon: dict keys must be str, got {type(k).__name__}: {k!r}")
            out[k] = normalize(v)
        return out
    if isinstance(x, (list, tuple)):
        return [normalize(v) for v in x]
    if isinstance(x, (bytes, bytearray)):
        raise TypeError("canon: bytes are not JSON; hash them with sha256_hex and pass the hex")
    raise TypeError(f"canon: unsupported type {type(x).__name__}: {x!r}")


def canon(x: Any) -> str:
    """Canonical JSON text of ``x``: sorted keys, no whitespace, non-ASCII kept,
    NaN/Inf refused. The hashed bytes are this text in UTF-8."""
    return json.dumps(normalize(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_hex(data: bytes | bytearray | memoryview) -> str:
    """Lower-case hex sha256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_canon(x: Any) -> str:
    """``sha256(canon(x))`` as hex."""
    return sha256_hex(canon(x).encode("utf-8"))


def hash12(digest: str) -> str:
    """First 12 hex characters, for display and paths."""
    return digest[:12]


def rng_seed(text: str) -> int:
    """Integer seed for ``numpy.random.default_rng`` from ``sha256(text)``: the
    ``default_rng(sha256(f"..."))`` idiom (canaries, validate's duplicate sample, the media fingerprint)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")


# --------------------------------------------------------------------------- manifest hash


def manifest_hash(manifest_bytes: bytes, classes_bytes: bytes | None) -> str:
    """``sha256(b"manifest:" + sha256(manifest).hex + b"\\nclasses:" + (sha256(classes).hex | "none"))``
    over the DECOMPRESSED csv bytes."""
    classes_part = sha256_hex(classes_bytes) if classes_bytes is not None else "none"
    payload = b"manifest:" + sha256_hex(manifest_bytes).encode("ascii") + b"\nclasses:" + classes_part.encode("ascii")
    return sha256_hex(payload)


# --------------------------------------------------------------------------- normalized source

#: AST fields never serialized by :func:`normalized_source`: source positions and per-version bookkeeping that do not
#: change what the code does (``kind`` is the ``u`` prefix of a string literal; ``type_comment`` a ``# type:`` comment).
_AST_SKIP_FIELDS: frozenset[str] = frozenset({"type_comment", "kind"})


def _strip_docstrings(tree: ast.AST) -> None:
    """Drop every bare string statement (module, class and function docstrings, attribute docstrings): they never run."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            node.body = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))]


def _ast_json(node: Any) -> Any:
    if isinstance(node, ast.AST):
        out: list[Any] = [type(node).__name__]
        for name in node._fields:
            if name in _AST_SKIP_FIELDS:
                continue
            value = getattr(node, name, None)
            if value is None or (isinstance(value, list) and not value):
                continue  # Python-version stable: 3.12 added empty ``type_params``, 3.13 dumps omit empties
            out.append([name, _ast_json(value)])
        return out
    if isinstance(node, list):
        return [_ast_json(v) for v in node]
    if isinstance(node, (str, bool, int, float)) or node is None:
        return [type(node).__name__, repr(node)]
    return [type(node).__name__, repr(node)]  # bytes, complex, Ellipsis


def normalized_source(text: str | bytes) -> str:
    """The comparison form of a folder's ``reader.py`` / ``probe.py``: the parsed AST with comments (never
    in an AST) and every docstring / bare string statement removed, serialized field by field without source
    positions and without empty or absent fields, as canonical JSON. A comment, docstring, blank-line or formatting
    edit leaves it unchanged; any change to what the code does changes it. The serialization does not use
    ``ast.dump`` (whose output differs between Python 3.10 and 3.13), so the digest is the same in every supported
    interpreter (``tests/test_hashing.py`` pins one)."""
    tree = ast.parse(text)
    _strip_docstrings(tree)
    return canon(_ast_json(tree))


def source_sha256(text: str | bytes) -> str:
    """``sha256(normalized_source(text))``: what ``protocol_hash`` folds in for ``reader.py`` / ``probe.py``."""
    return sha256_hex(normalized_source(text).encode("utf-8"))


# --------------------------------------------------------------------------- protocol = input + scoring


def protocol_targets(targets: Mapping[str, Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    """The hashed view of ``benchmark.yaml[targets]``: ``[{split, kind, file, sha256, index_column}, …]``
    sorted by split, or ``None``. ``index_column`` (null when absent) decides which
    memmap row a manifest row is scored against, so it changes numbers. ``source`` and ``root`` are site/layout
    facts and stay unhashed. Scoring side: targets never reach the bridge."""
    if targets is None:
        return None
    return [
        {"split": split, "kind": t["kind"], "file": t["file"], "sha256": t["sha256"], "index_column": t.get("index_column")}
        for split, t in sorted(targets.items())
    ]


#: The ``benchmark.yaml[manifest]`` keys hashed: ``key`` and ``columns`` (which rows and which media a
#: Sample carries) on the input side, ``label`` (how targets are read) on the scoring side.
PROTOCOL_MANIFEST_KEYS: tuple[str, ...] = ("key", "label", "columns")
INPUT_MANIFEST_KEYS: tuple[str, ...] = ("key", "columns")


def protocol_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The hashed view of ``benchmark.yaml[manifest]``: ``{"key", "label", "columns"}`` as written in the
    yaml (``spec.raw``; an absent key is ``None``). ``file``, ``sha256``, ``classes``, ``classes_sha256`` (their content
    is ``manifest_hash``) and ``upstream`` (provenance) are not part of it."""
    if not isinstance(manifest, Mapping):
        raise TypeError(f"protocol_manifest: expected the benchmark.yaml manifest block (a mapping), got {type(manifest).__name__}")
    return {k: manifest.get(k) for k in PROTOCOL_MANIFEST_KEYS}


#: ``readout_params`` keys on the INPUT side whatever the readout: what ``features validate`` reads to
#: decide whether a feature set serves the readout (the streams it must tag, the scored segments), i.e. output-shape
#: requirements a bridge must meet. A readout adds its own through ``Readout.input_keys``.
INPUT_READOUT_PARAMS: tuple[str, ...] = ("T", "visual_stream", "conditioning_stream")
#: ``protocol`` keys on the INPUT side whatever the readout: ``views`` (a bridge rule) and the keys
#: ``features validate`` measures eligibility against; the values a bridge receives (``input_contract.facts``) and
#: the fact an ``output_contract.requires.annotated_frame`` names are added per benchmark.
INPUT_PROTOCOL_KEYS: tuple[str, ...] = ("views", "segments_scored", "num_segments", "segment_offset", "segment_seconds", "target_region")


def _readout_input_keys(readout: str) -> frozenset[str]:
    try:
        from yavih.readouts import registry

        return frozenset(getattr(registry.get(readout), "input_keys", frozenset()) or ())
    except Exception:  # noqa: BLE001 - an unknown readout is refused by protocol_hash's READOUT_VERSION lookup
        return frozenset()


def input_readout_view(*, readout: str, protocol: Mapping[str, Any], readout_params: Mapping[str, Any],
                       input_contract: Mapping[str, Any], output_contract: Mapping[str, Any]) -> dict[str, Any]:
    """The ``protocol`` / ``readout_params`` values on the input side: ``{"params": {...}, "protocol": {...}}``
    restricted to :data:`INPUT_READOUT_PARAMS`, :data:`INPUT_PROTOCOL_KEYS`, ``input_contract.facts``, the
    ``annotated_frame`` fact and the readout's ``input_keys`` — the keys that are present."""
    extra = _readout_input_keys(readout)
    facts = list((input_contract or {}).get("facts") or [])
    req = (output_contract or {}).get("requires") or {}
    af = req.get("annotated_frame") if isinstance(req, Mapping) else None
    if isinstance(af, Mapping) and isinstance(af.get("fact"), str):
        facts.append(af["fact"])
    pkeys = set(INPUT_READOUT_PARAMS) | extra
    qkeys = set(INPUT_PROTOCOL_KEYS) | set(facts) | extra
    return {
        "params": {k: v for k, v in sorted((readout_params or {}).items()) if k in pkeys},
        "protocol": {k: v for k, v in sorted((protocol or {}).items()) if k in qkeys},
    }


def input_components(
    *,
    name: str,
    splits: Mapping[str, Any],
    streams: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    output_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    readout_params: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_hash: str,
    readout: str,
    reader_sha256: str | None = None,
) -> dict[str, Any]:
    """The INPUT side of a benchmark, one entry per component: everything that decides what a bridge is
    handed and what its features must satisfy — the rows and media (``manifest_hash``, the manifest's ``key`` and
    ``columns``), the split values and roles, the streams, both contracts, the readout NAME, the input-side protocol
    / readout_params values (:func:`input_readout_view`) and the folder's ``reader.py`` (normalized source digest,
    ``None`` = the generic ``yavih.sample.iter_samples``). The behaviour of ``yavih.sample`` itself is covered by the
    schema tag (``PROTOCOL_SCHEMA``)."""
    if not isinstance(manifest, Mapping):
        raise TypeError(f"input_components: expected the benchmark.yaml manifest block (a mapping), got {type(manifest).__name__}")
    return {
        "name": name,
        "manifest_hash": manifest_hash,
        "manifest": {k: manifest.get(k) for k in INPUT_MANIFEST_KEYS},
        "splits": splits,
        "streams": streams,
        "input_contract": input_contract,
        "output_contract": output_contract,
        "readout": readout,
        "readout_inputs": input_readout_view(readout=readout, protocol=protocol, readout_params=readout_params,
                                             input_contract=input_contract, output_contract=output_contract),
        "reader": reader_sha256,
    }


def scoring_components(
    *,
    protocol_version: int,
    protocol: Mapping[str, Any],
    external: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    readout_params: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
    manifest: Mapping[str, Any],
    readout: str,
    probe_sha256: str | None = None,
) -> dict[str, Any]:
    """The SCORING side of a benchmark: how the features are turned into numbers — ``protocol_version``,
    the whole ``protocol`` and ``readout_params`` blocks, ``metrics``, ``external`` (with its ``version_pin``s),
    ``targets``, the manifest's ``label`` read rule, the readout name and ``READOUT_VERSION``, and the folder's
    ``probe.py`` (normalized source digest). None of it reaches the bridge."""
    if readout not in READOUT_VERSION:
        raise ValueError(f"unknown readout {readout!r}; known: {sorted(READOUT_VERSION)}")
    if not isinstance(manifest, Mapping):
        raise TypeError(f"scoring_components: expected the benchmark.yaml manifest block (a mapping), got {type(manifest).__name__}")
    if targets is None or isinstance(targets, Mapping):
        targets_view = protocol_targets(targets)  # type: ignore[arg-type]
    else:
        targets_view = [dict(t) for t in targets]
    return {
        "protocol_version": protocol_version,
        "protocol": protocol,
        "readout_params": readout_params,
        "metrics": metrics,
        "external": external,
        "targets": targets_view,
        "label": manifest.get("label"),
        "readout": {"name": readout, "version": READOUT_VERSION[readout]},
        "probe": probe_sha256,
    }


def component_digests(components: Mapping[str, Any]) -> dict[str, str]:
    """``{component: sha256(canon(value))}``: recorded next to ``input_hash`` in sidecars so a mismatch names what
    differs (``E_HASH_INPUT``)."""
    return {k: sha256_canon(v) for k, v in sorted(components.items())}


def _part_hash(part: str, components: Mapping[str, Any]) -> str:
    return sha256_canon({"schema": PROTOCOL_SCHEMA, "part": part, "components": component_digests(components)})


def input_hash_of(components: Mapping[str, Any]) -> str:
    """``input_hash`` from :func:`input_components`."""
    return _part_hash("input", components)


def scoring_hash_of(components: Mapping[str, Any]) -> str:
    """``scoring_hash`` from :func:`scoring_components`."""
    return _part_hash("scoring", components)


def combine_protocol_hash(input_hash: str, scoring_hash: str) -> str:
    """``protocol_hash = sha256(canon({"schema", "input_hash", "scoring_hash"}))``."""
    for label, h in (("input_hash", input_hash), ("scoring_hash", scoring_hash)):
        if not isinstance(h, str) or not h:
            raise TypeError(f"combine_protocol_hash: {label} must be a non-empty hex digest str, got {h!r}")
    return sha256_canon({"schema": PROTOCOL_SCHEMA, "input_hash": input_hash, "scoring_hash": scoring_hash})


@dataclass(frozen=True)
class ProtocolHashes:
    """Every digest of one benchmark's protocol: ``protocol_hash = H(input_hash, scoring_hash)`` and the
    per-component digests of each side (what ``E_HASH_INPUT`` diffs)."""

    protocol_hash: str
    input_hash: str
    scoring_hash: str
    input_components: dict[str, str]
    scoring_components: dict[str, str]


def protocol_hashes(
    *,
    name: str,
    protocol_version: int,
    protocol: Mapping[str, Any],
    splits: Mapping[str, Any],
    streams: Mapping[str, Any],
    input_contract: Mapping[str, Any],
    output_contract: Mapping[str, Any],
    external: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    readout_params: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
    manifest: Mapping[str, Any],
    manifest_hash: str,
    readout: str,
    reader_sha256: str | None = None,
    probe_sha256: str | None = None,
) -> ProtocolHashes:
    """``yavih-protocol-4``: the input side, the scoring side and their combination. Everything hashed is
    the benchmark's: nothing bridge- or site-side. ``reader_sha256`` / ``probe_sha256`` are
    :func:`source_sha256` digests of the folder's ``reader.py`` / ``probe.py`` (``None`` when absent)."""
    if manifest is None:
        raise TypeError("protocol_hashes: manifest (the benchmark.yaml manifest block) is required")
    inp = input_components(name=name, splits=splits, streams=streams, input_contract=input_contract, output_contract=output_contract,
                           protocol=protocol, readout_params=readout_params, manifest=manifest, manifest_hash=manifest_hash,
                           readout=readout, reader_sha256=reader_sha256)
    sco = scoring_components(protocol_version=protocol_version, protocol=protocol, external=external, metrics=metrics,
                             readout_params=readout_params, targets=targets, manifest=manifest, readout=readout, probe_sha256=probe_sha256)
    ih, sh = input_hash_of(inp), scoring_hash_of(sco)
    return ProtocolHashes(protocol_hash=combine_protocol_hash(ih, sh), input_hash=ih, scoring_hash=sh,
                          input_components=component_digests(inp), scoring_components=component_digests(sco))


def protocol_hash(**kw: Any) -> str:
    """``protocol_hashes(**kw).protocol_hash``."""
    return protocol_hashes(**kw).protocol_hash


def input_hash(**kw: Any) -> str:
    """``protocol_hashes(**kw).input_hash``: what feature sets are checked against."""
    return protocol_hashes(**kw).input_hash


# --------------------------------------------------------------------------- describe hash

#: Top-level ``describe()`` keys outside ``describe_sha256``: a path edit, a looser declared
#: tolerance or another batch size changes nothing that was extracted (batch invariance is checked on its own: the
#: canaries and ``N_CANARY_DRIFT``). Recorded in every sidecar, never compared.
DESCRIBE_SHA_EXCLUDED: frozenset[str] = frozenset({"weights_ref", "batch_tolerance", "batch_size"})
#: Per-stream ``describe()["input"][stream]`` keys outside ``describe_sha256``:
#: the ``decoder`` block (``{lib, version, libav, …}``) is the host's decoding library build — provenance like the
#: torch or CUDA version, not a configuration decision; a PyAV patch release must not read as another configuration.
DESCRIBE_PROVENANCE_INPUT_KEYS: frozenset[str] = frozenset({"decoder"})


def describe_subset(describe: Mapping[str, Any]) -> dict[str, Any]:
    """``describe()`` minus ``DESCRIBE_SHA_EXCLUDED`` and minus each ``input.<stream>.decoder`` — the subset
    ``describe_sha256`` covers (``DESCRIBE_SCHEMA``)."""
    out = {k: v for k, v in describe.items() if k not in DESCRIBE_SHA_EXCLUDED}
    inp = out.get("input")
    if isinstance(inp, Mapping):
        out["input"] = {s: ({k: v for k, v in e.items() if k not in DESCRIBE_PROVENANCE_INPUT_KEYS} if isinstance(e, Mapping) else e)
                        for s, e in inp.items()}
    return out


def describe_sha256(describe: Mapping[str, Any]) -> str:
    """``sha256(canon(describe_subset(describe)))``: the CONSISTENCY and PROVENANCE digest of
    the bridge's configuration — one per feature set, compared on resume, carried by probe DONE markers and
    ``inputs_hash``, checked against the ``bridge check`` profile (``E_CONFIG_DRIFT``) and used by the report's
    ``W_BRIDGE_CHANGED``. Never identity, never comparability, never part of a path or a key."""
    return sha256_canon(describe_subset(describe))


assert DESCRIBE_SCHEMA == "yavih-describe-2", "describe_subset implements yavih-describe-2; bump both together"


# --------------------------------------------------------------------------- probe hash

#: ProbeConfig fields hashed under ``train``, in documented order. Seed is NOT included.
PROBE_TRAIN_FIELDS: tuple[str, ...] = (
    "lr",
    "lr_grid",
    "weight_decay",
    "max_epochs",
    "batch_size",
    "patience",
    "warmup_fraction",
    "schedule",
    "val_fraction",
    "loss",
    "selection",
    "max_tokens_per_forward",
    "precision",
)


def _field(cfg: Any, name: str) -> Any:
    if isinstance(cfg, Mapping):
        if name not in cfg:
            raise KeyError(f"probe_hash: config is missing field {name!r}")
        return cfg[name]
    try:
        return getattr(cfg, name)
    except AttributeError as e:
        raise KeyError(f"probe_hash: config is missing field {name!r}") from e


def probe_hash(cfg: Any) -> str:
    """``probe_hash`` over a ``ProbeConfig`` (dataclass, any object with the attributes, or a
    dict). Hashing never imports ``yavih.probes`` (torch), hence the duck typing."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        known = {f.name for f in fields(cfg)}
        missing = (set(PROBE_TRAIN_FIELDS) | {"readout", "head"}) - known
        if missing:
            raise KeyError(f"probe_hash: config lacks fields {sorted(missing)}")
    payload = {
        "schema": PROBE_SCHEMA,
        "readout": _field(cfg, "readout"),
        "head": _field(cfg, "head"),
        "train": {name: _field(cfg, name) for name in PROBE_TRAIN_FIELDS},
    }
    return sha256_canon(payload)


# --------------------------------------------------------------------------- eval hash


def eval_hash(protocol_hash: str, probe_hash: str) -> str:
    """``sha256(canon({"protocol_hash": ph, "probe_hash": qh}))``: THE comparability key — the benchmark
    and the probe, i.e. what yavih controls, and nothing else. Two results are comparable iff ``benchmark.name``
    (after alias), ``eval_hash`` and ``standard`` agree. No schema tag of its own: each component carries one."""
    for label, h in (("protocol_hash", protocol_hash), ("probe_hash", probe_hash)):
        if not isinstance(h, str) or not h:
            raise TypeError(f"eval_hash: {label} must be a non-empty hex digest str, got {h!r}")
    return sha256_canon({"protocol_hash": protocol_hash, "probe_hash": probe_hash})


# --------------------------------------------------------------------------- features fingerprint


def features_fingerprint(
    *,
    format_version: int,
    dtype: str,
    N: int,
    D: int,
    pos_mode: str,
    pos_sha256: str | None,
    shards: Sequence[Sequence[Any]],
    n_present: int,
) -> str:
    """The feature set's integrity digest; ``shards`` = ``[[lo, hi, sha256], …]`` in row order (the caller sorts by
    ``lo``; this function does not reorder so a wrong order is a different fingerprint)."""
    for s in shards:
        if len(s) != 3:
            raise ValueError(f"features_fingerprint: each shard entry is [lo, hi, sha256], got {s!r}")
    payload = {
        "format_version": format_version,
        "dtype": dtype,
        "N": N,
        "D": D,
        "pos_mode": pos_mode,
        "pos_sha256": pos_sha256,
        "shards": [[int(lo), int(hi), str(sha)] for lo, hi, sha in shards],
        "n_present": n_present,
    }
    return sha256_canon(payload)


# --------------------------------------------------------------------------- unit ids


def unit_id(
    *,
    kind: str,
    benchmark: str,
    model: str,
    reading: str,
    split: str | None,
    shard: Sequence[int] | None,
    protocol_hash: str | None,
    probe_hash: str | None,
    seed: int | None,
    input_hash: str | None = None,
) -> str:
    """12-hex unit id: a scheduling handle. ``model`` and ``reading`` are the bridge's declared names;
    ``shard`` is ``[i, n]`` or ``None``; ``plan_schema`` is ``PLAN_SCHEMA_VERSION``. An extract unit is keyed by the
    benchmark's ``input_hash`` (``protocol_hash`` ``None``: a scoring edit does not re-extract), probe and report units
    by ``protocol_hash``."""
    payload = {
        "kind": kind,
        "benchmark": benchmark,
        "model": model,
        "reading": reading,
        "split": split,
        "shard": [int(shard[0]), int(shard[1])] if shard is not None else None,
        "protocol_hash": protocol_hash,
        "input_hash": input_hash,
        "probe_hash": probe_hash,
        "seed": seed,
        "plan_schema": PLAN_SCHEMA_VERSION,
    }
    return hash12(sha256_canon(payload))


def inputs_hash(
    *,
    cmd: Sequence[str],
    protocol_hash: str | None,
    probe_hash: str | None,
    model: str,
    reading: str,
    describe_sha256: str,
    input_hash: str | None = None,
) -> str:
    """What the unit computes, the bridge's configuration (``describe_sha256``) included — staleness,
    not comparability; ``plan --pending`` treats a DONE marker whose ``logs/last.json.inputs_hash`` differs as stale.

    ``cmd`` is the unit's *identity* command (``yavih.plan.units.identity_cmd``: host paths and worker counts
    removed). The launcher, ``env``, ``cwd``, worker counts and the resource estimate (``needs``) are where and how a
    unit runs, not what it computes: they are logged in ``logs/last.json``, never hashed, so a plan built on the host
    is not stale inside a container that mounts the project elsewhere, and ``HF_HUB_OFFLINE`` or a calibrated
    ``needs.minutes`` re-plans nothing. Extract units pass ``input_hash`` (and ``protocol_hash``
    ``None``), probe and report units ``protocol_hash``."""
    payload = {
        "cmd": list(cmd),
        "protocol_hash": protocol_hash,
        "input_hash": input_hash,
        "probe_hash": probe_hash,
        "model": model,
        "reading": reading,
        "describe_sha256": describe_sha256,
    }
    return sha256_canon(payload)
