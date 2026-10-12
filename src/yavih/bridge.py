"""``Bridge`` base class, ``BenchmarkView``, ``describe()`` validation incl. the declared-rule vocabulary,
weight fingerprinting, ``grid_pos``.

Core module: numpy only. Torch objects are handled by duck typing (a tensor is recognised by
its methods and torch is taken from ``sys.modules`` only when such a tensor is already in
hand); a ``torch.nn.Module`` is walked by ``yavih.extract`` only when torch is importable.

The bridge receives a ``BenchmarkView`` (no manifest, labels, splits or data roots), reads the
RAW media itself and declares every model-side decision in ``describe()["input"]``; ``encode``
returns ``(feats, pos, used)``. Every bridge decision is RECORDED, never identity;
the model is the user-chosen name ``describe()["model"]`` read in ``describe()["reading"]``, and
``Bridge.name`` is an optional label.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from yavih import hashing as _hashing
from yavih import waivers as _waivers
from yavih.hashing import canon
from yavih.media import rules as _rules
from yavih.waivers import WaiveError, describe_waivers

if TYPE_CHECKING:  # pragma: no cover
    from yavih.sample import Sample

__all__ = [
    "Bridge",
    "BenchmarkView",
    "ArgSpec",
    "ArgError",
    "NotApplicable",
    "MediaUndecodable",
    "DescribeError",
    "validate_describe",
    "validate_input",
    "stream_kinds",
    "fingerprint_state_dict",
    "grid_pos",
    "parse_args",
    "preprocessing_source",
    "preprocessing_source_check",
    "batch_tolerance",
    "bridge_label",
    "check_model_name",
    "check_reading",
    "BRIDGE_NAME_RE",
    "MODEL_RE",
    "READING_RE",
    "DESCRIBE_REQUIRED",
    "DESCRIBE_SHA_EXCLUDED",
    "DTYPES",
    "POS_MODES",
    "STREAM_KINDS",
    "TEMPORAL_RULES",
    "SPATIAL_KINDS",
    "AUTOCAST_MODES",
    "MATMUL_PRECISIONS",
    "PRECISION_KEYS",
    "DEFAULT_BATCH_TOLERANCE",
    "MAX_BATCH_TOLERANCE",
    "E_DESCRIBE",
    "E_INPUT",
    "E_PRECISION_UNDECLARED",
    "E_ARGS_UNCONSUMED",
    "E_ARG",
    "E_WAIVE_INVALID",
    "args_unconsumed",
]

#: ``Bridge.name`` when set: an optional label of the bridge file, recorded as ``bridge.name``.
BRIDGE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
#: ``describe()["model"]``: the user's NAME of the model, shared by all its bridges; names the run and
#: results directories, so it must be filesystem-safe. The FIRST character is a letter or digit: a leading ``_``
#: would collide with yavih's reserved directories (``_stale``, ``_report``, ``_data``; the report index and the
#: join never enter a ``_``-prefixed directory), a leading ``.`` hides the directory from ``*`` globs (and covers
#: ``.`` / ``..``), a leading ``-`` reads as a command-line option.
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
#: ``describe()["reading"]``: ``a`` | ``v`` | ``av`` or another declared tag; enters paths and the results
#: key ``<benchmark>:<reading>``, so ``:`` and ``+`` stay free for keys; the first character is a letter or digit (as
#: ``model``).
READING_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
DTYPES: frozenset[str] = frozenset({"float16", "float32"})
POS_MODES: frozenset[str] = frozenset({"none", "constant", "per_sample"})
STREAM_KINDS: frozenset[str] = frozenset({"video", "audio", "image"})
AUTOCAST_MODES: frozenset[str | None] = frozenset({None, "bf16", "fp16"})
MATMUL_PRECISIONS: frozenset[str] = frozenset({"highest", "high", "medium"})
#: Keys of ``describe()["precision"]``; only ``autocast`` may be null.
PRECISION_KEYS: tuple[str, ...] = ("autocast", "matmul_precision", "allow_tf32", "deterministic")
#: Keys ``describe()`` MUST contain, with their types checked by ``validate_describe``.
#: ``batch_tolerance`` and ``waive`` are optional.
DESCRIBE_REQUIRED: tuple[str, ...] = (
    "model",
    "reading",
    "weights_sha256",
    "weights_ref",
    "dtype",
    "batch_size",
    "streams",
    "pos_mode",
    "layer",
    "pooling",
    "views",
    "input",
    "precision",
    "args",
)
#: ``describe()`` keys outside ``describe_sha256``. The hashing
#: module owns the digest and the set; this is the same object.
DESCRIBE_SHA_EXCLUDED: frozenset[str] = _hashing.DESCRIBE_SHA_EXCLUDED
#: The batch-/order-invariance tolerance (``rel_l2`` per clip) and the loosest a bridge may declare.
DEFAULT_BATCH_TOLERANCE: float = 1e-5
MAX_BATCH_TOLERANCE: float = 1e-3

#: The temporal rule vocabulary: rule → stream kinds it is legal for (one definition: ``yavih.media.rules``).
TEMPORAL_RULES: Mapping[str, frozenset[str]] = _rules.TEMPORAL_RULES
#: Kinds that carry a ``spatial`` rule.
SPATIAL_KINDS: frozenset[str] = frozenset({"video", "image"})

# Error codes carried by DescribeError / ArgError (the CLI prints them).
E_DESCRIBE = "E_DESCRIBE"  # generic describe() schema violation (exit 2)
E_INPUT = "E_INPUT"  # describe()["input"] violates the declared-rule vocabulary (exit 2)
E_PRECISION_UNDECLARED = "E_PRECISION_UNDECLARED"  # null precision field other than autocast (exit 2)
E_ARGS_UNCONSUMED = "E_ARGS_UNCONSUMED"  # describe()["args"] does not echo a consumed --arg (exit 2)
E_ARG = "E_ARG"  # unknown / malformed --arg (exit 1)
E_WAIVE_INVALID = _waivers.E_WAIVE_INVALID  # describe()["waive"] names a non-waivable code or has no reason (exit 2)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class NotApplicable(Exception):
    """Raised by a bridge from ``__init__`` or the first ``encode`` when it cannot serve the
    benchmark; ``run_extract`` writes ``NOT_APPLICABLE.json`` and exits 0."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MediaUndecodable(Exception):
    """Raised by a bridge's ``encode`` when it cannot decode the media of some clips:
    ``run_extract`` records those rows as missing and re-encodes the batch without them (exact, by batch
    invariance). ``clip_ids`` names the offending clips; any other exception aborts the shard (exit 5)."""

    def __init__(self, clip_ids: str | Sequence[str], reason: str) -> None:
        ids = [clip_ids] if isinstance(clip_ids, str) else [str(c) for c in clip_ids]
        super().__init__(f"MediaUndecodable: {ids[:5]}{' …' if len(ids) > 5 else ''}: {reason}")
        self.clip_ids: list[str] = ids
        self.reason = str(reason)


class DescribeError(ValueError):
    """``describe()`` violates the ``describe()`` schema; ``run_extract`` exits 2.

    ``code`` names the rule (``E_DESCRIBE`` by default, ``E_INPUT``,
    ``E_PRECISION_UNDECLARED``, ``E_ARGS_UNCONSUMED``) so the CLI envelope can report it.
    """

    exit_code = 2

    def __init__(self, message: str, code: str = E_DESCRIBE) -> None:
        super().__init__(message)
        self.code = code


class ArgError(ValueError):
    """An unknown or malformed ``--arg k=v`` (usage error, exit 1)."""

    exit_code = 1
    code = E_ARG


@dataclass(frozen=True)
class ArgSpec:
    """One ``--arg k=v`` the bridge consumes (``Bridge.ARGS``): ``{"layer": {"type": int, "default": 12}}``."""

    type: type = str
    default: Any = None
    choices: tuple[Any, ...] | None = None
    help: str | None = None


@dataclass(frozen=True)
class BenchmarkView:
    """What a bridge receives: JSON primitives only — never the manifest, labels, splits or
    data roots. ``Benchmark.view()`` builds it; ``from_benchmark`` does the same for any object exposing the
    ``Benchmark`` surface (duck-typed test benchmarks)."""

    name: str
    aliases: list[str]
    readout: str
    protocol_version: int
    protocol_hash: str
    streams: dict[str, str]  # {name: kind}
    input_contract: dict = field(default_factory=dict)
    output_contract: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)  # {key: protocol[key] for key in input_contract.facts}
    #: The benchmark's input side: what the bridge is handed and must produce. A bridge that records a
    #: benchmark digest records this one — ``protocol_hash`` also moves on scoring edits, which never touch features.
    input_hash: str = ""

    @classmethod
    def from_benchmark(cls, bench: Any) -> BenchmarkView:
        if isinstance(bench, BenchmarkView):
            return bench
        view_fn = getattr(bench, "view", None)
        if callable(view_fn):
            try:
                v = view_fn()
            except TypeError:  # a view(split) of some other object, not Benchmark's
                v = None
            if isinstance(v, BenchmarkView):
                return v
            if v is not None and all(hasattr(v, k) for k in ("name", "streams", "input_contract")):
                return cls(
                    name=str(v.name), aliases=list(getattr(v, "aliases", []) or []), readout=str(getattr(v, "readout", "")),
                    protocol_version=int(getattr(v, "protocol_version", 0) or 0), protocol_hash=str(getattr(v, "protocol_hash", "")),
                    streams=dict(v.streams), input_contract=_plain(getattr(v, "input_contract", {})),
                    output_contract=_plain(getattr(v, "output_contract", {})), facts=_plain(getattr(v, "facts", {})),
                    input_hash=str(getattr(v, "input_hash", "") or ""),
                )
        spec = getattr(bench, "spec", None)
        ic = _plain(_get(spec, "input_contract", {}) or {})
        protocol = _plain(_get(spec, "protocol", {}) or {})
        facts = {str(k): protocol.get(k) for k in (ic.get("facts") or []) if k in protocol}
        return cls(
            name=str(getattr(bench, "name", "") or _get(spec, "name", "")),
            aliases=list(_get(spec, "aliases", []) or []),
            readout=str(_get(spec, "readout", "") or ""),
            protocol_version=int(_get(spec, "protocol_version", 0) or 0),
            protocol_hash=str(getattr(bench, "protocol_hash", "") or ""),
            streams=stream_kinds(bench),
            input_contract=ic,
            output_contract=_plain(_get(spec, "output_contract", {}) or {}),
            facts=facts,
            input_hash=str(getattr(bench, "input_hash", "") or ""),
        )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _plain(obj: Any) -> Any:
    """Dataclasses / mappings / tuples → JSON-shaped dicts and lists (``BenchmarkView`` holds primitives)."""
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    fields = getattr(obj, "__dataclass_fields__", None)
    if fields:
        return {k: _plain(getattr(obj, k)) for k in fields}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def stream_kinds(bench: Any) -> dict[str, str]:
    """``{stream name: kind}`` of a ``BenchmarkView``, a ``Benchmark`` (``spec.streams``) or a mapping."""
    if isinstance(bench, BenchmarkView):
        return dict(bench.streams)
    spec = getattr(bench, "spec", bench)
    streams = _get(spec, "streams", None)
    if streams is None:
        streams = getattr(bench, "streams", None)
    out: dict[str, str] = {}
    for name, s in dict(streams or {}).items():
        kind = s if isinstance(s, str) else _get(s, "kind")
        if kind is not None:
            out[str(name)] = str(kind)
    return out


class Bridge:
    """Model → feature adapter, one per ``(model, benchmark)``.

    Subclasses set the class attributes and override ``describe`` and ``encode``. ``__init__``
    stores the ``BenchmarkView`` and applies ``ARGS`` to ``args`` (types coerced, defaults filled),
    so ``self.args`` is exactly the dict ``describe()["args"]`` must echo; an unknown key raises
    ``ArgError`` (exit 1). ``decode_workers`` is set by ``run_extract`` before the first ``encode``:
    the bridge's CPU budget for its own decoding.
    """

    name: ClassVar[str] = ""  # optional label, [A-Za-z0-9_.-]{1,64} when set (default: the class name); never a key
    batch_size: ClassVar[int] = 1
    dtype: ClassVar[str] = "float16"  # "float16" | "float32"
    pos_mode: ClassVar[str] = "none"  # "none" | "constant" | "per_sample"
    streams: ClassVar[tuple[str, ...]] = ()  # stream tags used in pos; order = tag index
    ARGS: ClassVar[dict[str, ArgSpec | dict]] = {}  # argparse-like spec of consumed --arg k=v
    decode_workers: int = 1

    def __init__(self, bench: BenchmarkView, args: dict | None = None) -> None:
        self.bench = bench
        self.args: dict[str, Any] = parse_args(self.ARGS, args or {})

    def describe(self) -> dict:
        """REQUIRED override. JSON primitives only; MUST contain the ``DESCRIBE_REQUIRED`` keys."""
        raise NotImplementedError("Bridge.describe must be overridden by the bridge")

    def encode(self, samples: list[Sample]) -> tuple[Any, Any | None, list[dict]]:
        """REQUIRED override. Returns ``(feats [B, N, D], pos None | [N, 4] | [B, N, 4], used)`` with ``used`` a
        list of ``B`` dicts, one entry per stream in ``describe()["input"]``."""
        raise NotImplementedError("Bridge.encode must be overridden by the bridge")

    def profile(self) -> dict | None:
        """Optional: ``{"peak_gpu_gb": ...}``."""
        return None


# ----------------------------------------------------------------------------- --arg parsing


def _as_argspec(name: str, spec: ArgSpec | Mapping[str, Any]) -> ArgSpec:
    if isinstance(spec, ArgSpec):
        return spec
    if isinstance(spec, Mapping):
        unknown = set(spec) - {"type", "default", "choices", "help"}
        if unknown:
            raise ArgError(f"ARGS[{name!r}]: unknown ArgSpec keys {sorted(unknown)}")
        choices = spec.get("choices")
        return ArgSpec(
            type=spec.get("type", str),
            default=spec.get("default"),
            choices=tuple(choices) if choices is not None else None,
            help=spec.get("help"),
        )
    raise ArgError(f"ARGS[{name!r}] must be an ArgSpec or a dict, got {type(spec).__name__}")


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce(name: str, value: Any, spec: ArgSpec) -> Any:
    """Coerce one ``--arg`` value. A value already of the declared type passes through so a
    bridge constructed programmatically with typed args round-trips."""
    typ = spec.type
    if value is None:
        return None
    if typ is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise ArgError(f"--arg {name}={value!r}: expected a boolean (true/false)")
    if isinstance(typ, type) and isinstance(value, typ) and not (typ is int and isinstance(value, bool)):
        return value
    try:
        return typ(value)
    except (TypeError, ValueError) as e:
        tname = getattr(typ, "__name__", repr(typ))
        raise ArgError(f"--arg {name}={value!r}: not a valid {tname}: {e}") from e


def parse_args(spec: Mapping[str, ArgSpec | dict], raw: Sequence[str] | Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply ``Bridge.ARGS`` to ``--arg k=v`` pairs: types coerced, defaults filled, unknown
    key → ``ArgError`` (exit 1). ``raw`` is a list of ``"k=v"`` strings or an already split mapping.
    The result has exactly the keys of ``spec`` (defaults applied), so passing a default explicitly
    changes nothing; ``describe()["args"]`` must echo exactly this dict."""
    given: dict[str, Any] = {}
    if raw is None:
        pass
    elif isinstance(raw, Mapping):
        given = dict(raw)
    else:
        for item in raw:
            if not isinstance(item, str) or "=" not in item:
                raise ArgError(f"--arg expects k=v, got {item!r}")
            k, v = item.split("=", 1)
            k = k.strip()
            if not k:
                raise ArgError(f"--arg expects k=v with a non-empty key, got {item!r}")
            if k in given:
                raise ArgError(f"--arg {k} given twice")
            given[k] = v
    specs = {name: _as_argspec(name, s) for name, s in spec.items()}
    unknown = sorted(set(given) - set(specs))
    if unknown:
        raise ArgError(f"unknown --arg key(s) {unknown}; this bridge consumes {sorted(specs)}")
    out: dict[str, Any] = {}
    for name, s in specs.items():
        value = _coerce(name, given[name], s) if name in given else s.default
        if s.choices is not None and value is not None and value not in s.choices:
            raise ArgError(f"--arg {name}={value!r}: not one of {list(s.choices)}")
        out[name] = value
    return out


# ----------------------------------------------------------------------------- describe() schema


def _require(cond: bool, message: str, code: str = E_DESCRIBE) -> None:
    if not cond:
        raise DescribeError(f"describe(): {message}", code)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(float(x))


def _in(where: str, cond: bool, message: str) -> None:
    if not cond:
        raise DescribeError(f"describe()['input']{where}: {message}", E_INPUT)


def _rule_check(where: str, fn: Any, *args: Any) -> None:
    """Run a ``yavih.media.rules`` vocabulary check, turning its ``ValueError`` into ``E_INPUT``."""
    try:
        fn(*args)
    except ValueError as e:
        raise DescribeError(f"describe()['input']{where}: {e}", E_INPUT) from e


def validate_input(inp: Any, kinds: Mapping[str, str], tagged: Sequence[str]) -> None:
    """The declared-rule checks on ``describe()["input"]``: non-empty, keys ⊆ the benchmark's streams and ⊇ the tagged
    ``streams``; per stream ``temporal`` (every kind but image; the closed vocabulary of
    ``yavih.media.rules.check_temporal``), ``spatial`` (video, image; ``check_spatial``), ``front_end`` (audio: at least ``sample_rate``, ``resample``, ``features``), ``norm`` (every kind,
    with a non-empty ``source``) and ``decoder`` (every kind: ``{lib, version, …}``). Raises ``DescribeError``
    (``E_INPUT``)."""
    _in("", isinstance(inp, Mapping) and len(inp) > 0, f"must be a non-empty dict per stream the bridge reads, got {inp!r}")
    unknown = sorted(set(inp) - set(kinds))
    _in("", not unknown, f"names stream(s) {unknown} the benchmark does not have ({sorted(kinds)})")
    untagged = sorted(set(tagged) - set(inp))
    _in("", not untagged, f"lacks the tagged stream(s) {untagged}: every stream in describe()['streams'] is read and declared")
    for sname, entry in inp.items():
        where = f"[{sname!r}]"
        kind = kinds[sname]
        _in(where, isinstance(entry, Mapping), f"must be a dict, got {type(entry).__name__}")
        allowed = {"norm", "decoder"} | ({"temporal"} if kind != "image" else set()) | ({"spatial"} if kind in SPATIAL_KINDS else set()) | (
            {"front_end"} if kind == "audio" else set()
        )
        missing = sorted(allowed - set(entry))
        _in(where, not missing, f"a {kind!r} stream must declare {sorted(allowed)}; missing {missing}")
        extra = sorted(set(entry) - allowed)
        _in(where, not extra, f"unknown key(s) {extra} for a {kind!r} stream (allowed {sorted(allowed)})")
        if kind != "image":
            _rule_check(where, _rules.check_temporal, entry["temporal"], kind, "temporal")
        if kind in SPATIAL_KINDS:
            _rule_check(where, _rules.check_spatial, entry["spatial"], "spatial")
        if kind == "audio":
            fe = entry["front_end"]
            _in(where + "['front_end']", isinstance(fe, Mapping), f"must be a dict, got {fe!r}")
            fmissing = [k for k in ("sample_rate", "resample", "features") if k not in fe]
            _in(where + "['front_end']", not fmissing, f"must name at least sample_rate, resample and features; missing {fmissing}")
            _in(where + "['front_end']", _is_int(fe["sample_rate"]) and fe["sample_rate"] > 0, f"sample_rate must be a positive int, got {fe['sample_rate']!r}")
        norm = entry["norm"]
        _in(where + "['norm']", isinstance(norm, Mapping), f"must be a dict of constants with their source, got {norm!r}")
        _in(where + "['norm']", isinstance(norm.get("source"), str) and norm["source"].strip() != "",
            "must name its 'source' (the model's own pretraining recipe: repo, file, commit) — never batch or dataset statistics")
        dec = entry["decoder"]
        _in(where + "['decoder']", isinstance(dec, Mapping) and isinstance(dec.get("lib"), str) and dec["lib"] != "" and "version" in dec,
            f"must be {{lib, version, …}} as the decoding library reports it, got {dec!r}")


def _guaranteed_info(bench: Any) -> dict[str, list[str]]:
    """``input_contract.info`` of a ``BenchmarkView`` / ``Benchmark``: ``{stream: [guaranteed MediaInfo keys]}``."""
    ic = bench.input_contract if isinstance(bench, BenchmarkView) else _get(_get(bench, "spec", None), "input_contract", None)
    info = _get(ic, "info", None) or {}
    if not isinstance(info, Mapping):
        return {}
    return {str(k): [str(x) for x in v] for k, v in info.items() if isinstance(v, list | tuple)}


def _check_rule_facts(inp: Mapping[str, Any], kinds: Mapping[str, str], bench: Any) -> None:
    """A declared checkable temporal rule must be evaluable on every row, so every media fact it needs
    (``yavih.media.rules.rule_facts``) must be one the benchmark's ``input_contract.info[stream]`` guarantees;
    otherwise validate could not check ``used`` against it on rows where the fact is null. Refused with
    ``E_INPUT``; the bridge then declares a rule the facts support, or opts out with ``custom`` (a legitimate
    choice, recorded as the note ``N_RULE_CUSTOM``). A stream the folder lists no ``info`` for is not checked here."""
    guaranteed = _guaranteed_info(bench)
    for sname, entry in inp.items():
        if not isinstance(entry, Mapping) or sname not in guaranteed:
            continue
        kind = kinds.get(sname)
        t = entry.get("temporal")
        if kind == "image" or not isinstance(t, Mapping):
            continue
        need = _rules.rule_facts(t, str(kind))
        lacking = sorted(need - set(guaranteed[sname]))
        _in(f"[{sname!r}]['temporal']", not lacking,
            f"rule {t.get('rule')!r} needs the media fact(s) {lacking}, which this benchmark's input_contract.info[{sname!r}] does not "
            f"guarantee ({guaranteed[sname]}): validate could not check `used` against it on every row; declare a rule these facts "
            "support (e.g. uniform / fps on duration_s), or opt out with {'rule': 'custom', 'description': '<one line: what the "
            "bridge does>'} — a legitimate choice, recorded as the note N_RULE_CUSTOM")


def bridge_label(bridge: Any) -> str:
    """The recorded ``bridge.name``: ``Bridge.name`` when set, else the class name. A label only — never
    a key, a path or identity (the model is ``describe()["model"]``)."""
    cls = bridge if isinstance(bridge, type) else type(bridge)
    name = getattr(bridge, "name", "") or ""
    return str(name) if name else str(getattr(cls, "__name__", "Bridge"))


def check_model_name(model: Any) -> str | None:
    """``None`` when ``model`` is a legal ``describe()["model"]`` (``[A-Za-z0-9_.-]{1,64}`` starting with a letter or
    digit: never ``.``, ``..``, ``_stale`` or another ``_``-reserved directory name), else the reason."""
    if not isinstance(model, str) or MODEL_RE.match(model) is None:
        return (f"model must be a name matching [A-Za-z0-9_.-]{{1,64}} that starts with a letter or digit (it names "
                f"runs/<benchmark>/<model>/ and <results_root>/<model>/; '_'-prefixed directories such as _stale/ are yavih's), "
                f"got {model!r}")
    return None


def check_reading(reading: Any) -> str | None:
    """``None`` when ``reading`` is a legal ``describe()["reading"]`` (``[A-Za-z0-9_-]{1,32}`` starting with a letter or
    digit), else the reason."""
    if not isinstance(reading, str) or READING_RE.match(reading) is None:
        return (f"reading must match [A-Za-z0-9_-]{{1,32}} and start with a letter or digit ('a', 'v', 'av' or another declared "
                f"tag; it names runs/<benchmark>/<model>/<reading>/ and the results key <benchmark>:<reading>), got {reading!r}")
    return None


def validate_describe(d: dict, bridge: Bridge, bench: Any) -> None:
    """Refuse a missing or mismatching ``describe()`` key (exit 2), a ``model`` or ``reading`` outside its charset,
    ``views != 1``, an ``input`` block that fails the declared-rule checks, a ``batch_tolerance`` above ``1e-3`` or without a reason, a
    null precision field other than ``autocast`` (``E_PRECISION_UNDECLARED``). ``bench`` is a ``BenchmarkView`` or a
    ``Benchmark``. Raises ``DescribeError``.

    ``weights_sha256: null`` is accepted: it is recorded and surfaces as the warning
    ``W_WEIGHTS_UNFINGERPRINTED``; nothing is gated, marked or forked by it.
    """
    _require(isinstance(d, Mapping), f"must return a dict, got {type(d).__name__}")
    missing = [k for k in DESCRIBE_REQUIRED if k not in d]
    _require(not missing, f"missing required key(s) {missing}")
    try:
        canon(d)  # JSON primitives only; a tensor or a Path would land here
    except (TypeError, ValueError) as e:
        raise DescribeError(f"describe(): not JSON primitives: {e}", E_DESCRIBE) from e

    name = getattr(bridge, "name", "") or ""
    _require(isinstance(name, str) and (name == "" or BRIDGE_NAME_RE.match(name) is not None),
             f"Bridge.name {name!r} (an optional label) must match [A-Za-z0-9_.-]{{1,64}} when set")
    why = check_model_name(d["model"])
    _require(why is None, str(why))
    why = check_reading(d["reading"])
    _require(why is None, str(why))

    w = d["weights_sha256"]
    _require(w is None or (isinstance(w, str) and _HEX64.match(w) is not None),
             f"weights_sha256 must be 64 lower-case hex or null, got {w!r}")
    _require(isinstance(d["weights_ref"], str), f"weights_ref must be a str, got {d['weights_ref']!r}")

    _require(d["dtype"] in DTYPES, f"dtype must be one of {sorted(DTYPES)}, got {d['dtype']!r}")
    _require(d["dtype"] == bridge.dtype, f"dtype {d['dtype']!r} != Bridge.dtype {bridge.dtype!r}")
    _require(bridge.dtype in DTYPES, f"Bridge.dtype must be one of {sorted(DTYPES)}, got {bridge.dtype!r}")
    bs = d["batch_size"]
    _require(_is_int(bs) and bs >= 1, f"batch_size must be an int >= 1, got {bs!r}")
    _require(bs == bridge.batch_size, f"batch_size {bs!r} != Bridge.batch_size {bridge.batch_size!r}")
    _require(d["pos_mode"] in POS_MODES, f"pos_mode must be one of {sorted(POS_MODES)}, got {d['pos_mode']!r}")
    _require(d["pos_mode"] == bridge.pos_mode, f"pos_mode {d['pos_mode']!r} != Bridge.pos_mode {bridge.pos_mode!r}")
    _require(_is_int(d["views"]) and d["views"] == 1,
             f"views must be 1 (one deterministic view per clip: no multi-crop, multi-clip or test-time-augmentation averaging), got {d['views']!r}")

    streams = d["streams"]
    _require(isinstance(streams, list | tuple) and all(isinstance(s, str) for s in streams),
             f"streams must be a list of str, got {streams!r}")
    _require(list(streams) == list(bridge.streams), f"streams {list(streams)!r} != Bridge.streams {list(bridge.streams)!r}")
    _require(len(set(streams)) == len(streams), f"streams has duplicates: {list(streams)!r}")
    kinds = stream_kinds(bench)
    unknown = [s for s in streams if s not in kinds]
    _require(not unknown, f"streams {unknown!r} are not streams of the benchmark ({sorted(kinds)})")

    _require(isinstance(d["pooling"], str) and d["pooling"] != "", f"pooling must be a non-empty str, got {d['pooling']!r}")
    _require(isinstance(d["layer"], str | int) and not isinstance(d["layer"], bool), f"layer must be str | int, got {d['layer']!r}")
    validate_input(d["input"], kinds, list(streams))
    _check_rule_facts(d["input"], kinds, bench)

    prec = d["precision"]
    _require(isinstance(prec, Mapping), f"precision must be a dict, got {type(prec).__name__}")
    pmissing = [k for k in PRECISION_KEYS if k not in prec]
    _require(not pmissing, f"precision is missing {pmissing}", E_PRECISION_UNDECLARED)
    nulls = [k for k in PRECISION_KEYS if k != "autocast" and prec[k] is None]
    _require(not nulls, f"precision.{nulls} is null; the bridge must DECLARE matmul_precision, allow_tf32 and deterministic",
             E_PRECISION_UNDECLARED)
    _require(prec["autocast"] in AUTOCAST_MODES, f"precision.autocast must be null | 'bf16' | 'fp16', got {prec['autocast']!r}")
    _require(prec["matmul_precision"] in MATMUL_PRECISIONS,
             f"precision.matmul_precision must be one of {sorted(MATMUL_PRECISIONS)}, got {prec['matmul_precision']!r}")
    _require(isinstance(prec["allow_tf32"], bool), f"precision.allow_tf32 must be a bool, got {prec['allow_tf32']!r}")
    _require(isinstance(prec["deterministic"], bool), f"precision.deterministic must be a bool, got {prec['deterministic']!r}")

    if "batch_tolerance" in d:
        bt = d["batch_tolerance"]
        _require(isinstance(bt, Mapping) and set(bt) == {"rel_l2", "reason"},
                 f"batch_tolerance must be {{rel_l2: float, reason: str}}, got {bt!r}")
        _require(_is_num(bt["rel_l2"]) and 0.0 < float(bt["rel_l2"]) <= MAX_BATCH_TOLERANCE,
                 f"batch_tolerance.rel_l2 must be in (0, {MAX_BATCH_TOLERANCE:g}], got {bt['rel_l2']!r}")
        _require(isinstance(bt["reason"], str) and bt["reason"].strip() != "",
                 "batch_tolerance needs a non-empty reason (e.g. 'bf16 attention kernels differ between batch sizes')")

    # describe()["waive"]: a waivable code with a non-empty reason, or "*"; anything else is E_WAIVE_INVALID
    try:
        waived = describe_waivers(d)
    except WaiveError as e:
        raise DescribeError(f"describe(): {e}", E_WAIVE_INVALID) from e

    args = d["args"]
    _require(isinstance(args, Mapping), f"args must be a dict, got {type(args).__name__}")
    if "E_ARGS_UNCONSUMED" not in waived:  # waived: bridge check records W_WAIVED (args_unconsumed)
        problems = args_unconsumed(d, bridge)
        _require(not problems, "; ".join(problems), E_ARGS_UNCONSUMED)


def args_unconsumed(d: Mapping[str, Any], bridge: Bridge) -> list[str]:
    """``E_ARGS_UNCONSUMED``: consumed ``--arg`` values ``describe()["args"]`` does not echo, and echoed keys that
    are not in ``Bridge.ARGS``. Empty when consistent."""
    args = d.get("args") if isinstance(d.get("args"), Mapping) else {}
    out: list[str] = []
    consumed = getattr(bridge, "args", None)
    if isinstance(consumed, Mapping):
        bad = [k for k, v in consumed.items() if k not in args or args[k] != v]
        if bad:
            out.append(f"args does not echo consumed --arg {bad} (bridge.args={dict(consumed)!r}, describe args={dict(args)!r})")
    spec_keys = set(getattr(bridge, "ARGS", {}) or {})
    extra = sorted(set(args) - spec_keys - set(consumed or {}))
    if extra:
        out.append(f"args echoes {extra} which are not in Bridge.ARGS")
    return out


def batch_tolerance(d: Mapping[str, Any] | None) -> tuple[float, str | None]:
    """``(rel_l2, reason)`` of a ``describe()``: the declared ``batch_tolerance`` (capped at ``1e-3``) or the
    default ``1e-5``."""
    bt = (d or {}).get("batch_tolerance") if isinstance(d, Mapping) else None
    if isinstance(bt, Mapping) and _is_num(bt.get("rel_l2")):
        return min(float(bt["rel_l2"]), MAX_BATCH_TOLERANCE), (str(bt.get("reason")) if bt.get("reason") is not None else None)
    return DEFAULT_BATCH_TOLERANCE, None


# ----------------------------------------------------------------------------- preprocessing source slot

#: The heading of the slot: a docstring line that starts with "Preprocessing source" (any case). Text after its first
#: ``": "`` (a colon followed by a space, or ending the line) on the SAME line is an inline body
#: (``Preprocessing source: repo@commit file::fn``); everything else on the heading's paragraph is heading prose.
_PREPROC_RE = re.compile(r"^(?P<indent>[ \t]*)preprocessing\s+source\b(?P<head>.*)$", re.IGNORECASE)
#: A body line that means the template was never filled: it starts with a TODO / FIXME / TBD marker.
_TODO_LINE_RE = re.compile(r"^(?:TODO|FIXME|TBD)\b", re.IGNORECASE)
#: Template prose that is not pasted source: a line starting "e.g.", "(fill" or "paste".
_PROSE_LINE_RE = re.compile(r"^(?:e\.g\.|\(fill\b|paste\b)", re.IGNORECASE)
#: A ``<…>`` placeholder (may span lines).
_PLACEHOLDER_RE = re.compile(r"<[^<>]*>", re.DOTALL)
_PREPROC_LAYOUT = (
    "a docstring line starting 'Preprocessing source' (the heading; its own prose is ignored), then the pasted source "
    "as an indented block below it, e.g.\n    Preprocessing source:\n        <repo>@<commit> <file>::<function>\n"
    "        <the source lines, verbatim>\n(or inline: 'Preprocessing source: <repo>@<commit> <file>::<function>'); "
    "a body whose lines start with TODO, or that holds only <...> placeholders, counts as unfilled"
)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _slot(lines: list[str], i: int) -> tuple[list[str], list[str]]:
    """``(inline, body)`` of the slot whose heading is ``lines[i]``. The heading paragraph is the heading line plus the
    non-blank lines that follow at the same indentation, up to and including the first one that ends with ``:``
    (a heading line that ends with ``:`` or carries an inline body is a paragraph of its own). The body is the block
    indented deeper than the heading that follows the paragraph (blank lines inside it, or between it and the
    heading, are kept / skipped), or, when the heading ends with ``:`` and the next line is at the heading's
    indentation, that paragraph up to the next blank line."""
    m = _PREPROC_RE.match(lines[i])
    assert m is not None
    ind = _indent(lines[i])
    head = m.group("head")
    # the first ":" that closes the heading: followed by whitespace or the end of the line (not "https://", "a::b")
    inline: list[str] = []
    for j, ch in enumerate(head):
        if ch == ":" and (j + 1 == len(head) or head[j + 1] in " \t") and (j == 0 or head[j - 1] != ":"):
            rest = head[j + 1:].strip()
            if rest:
                inline = [rest]
            break
    k = i + 1
    closed = bool(inline) or head.rstrip().endswith(":")
    if not closed:  # multi-line heading prose: up to the first line that ends with ":" (same indentation)
        while k < len(lines) and lines[k].strip() and _indent(lines[k]) == ind:
            k += 1
            if lines[k - 1].rstrip().endswith(":"):
                closed = True
                break
        if not closed:
            return inline, []
    body: list[str] = []
    if k < len(lines) and lines[k].strip() and _indent(lines[k]) == ind and not inline:
        while k < len(lines) and lines[k].strip() and _indent(lines[k]) >= ind:
            body.append(lines[k].strip())
            k += 1
        return inline, body
    j = k
    while j < len(lines) and not lines[j].strip():
        j += 1
    while j < len(lines) and (not lines[j].strip() or _indent(lines[j]) > ind):
        body.append(lines[j].strip())
        j += 1
    while body and not body[-1]:
        body.pop()
    return inline, body


def _filled(text_lines: list[str]) -> bool:
    """A slot body is filled when no line of it starts with TODO / FIXME / TBD (template text left in) and something
    alphanumeric remains once ``<…>`` placeholders and "e.g." / "(fill" / "paste" prose lines are removed."""
    content = [t for t in text_lines if t]
    if not content or any(_TODO_LINE_RE.match(t) for t in content):
        return False
    text = _PLACEHOLDER_RE.sub("", "\n".join(t for t in content if not _PROSE_LINE_RE.match(t)))
    return re.search(r"[A-Za-z0-9]", text) is not None


def _preproc_docs(bridge_cls: type) -> list[tuple[str, str]]:
    """``[(where, docstring)]``: the bridge class's own docstring, then its module's."""
    out: list[tuple[str, str]] = []
    doc = getattr(bridge_cls, "__doc__", None)  # a class docstring is not inherited
    if isinstance(doc, str):
        out.append((f"{getattr(bridge_cls, '__qualname__', bridge_cls)}.__doc__", inspect.cleandoc(doc)))
    modname = getattr(bridge_cls, "__module__", "")
    mod = sys.modules.get(modname)
    mdoc = getattr(mod, "__doc__", None) if mod is not None else None
    if isinstance(mdoc, str):
        out.append((f"module {modname}.__doc__", inspect.cleandoc(mdoc)))
    return out


def preprocessing_source_check(bridge_cls: type) -> tuple[str | None, str]:
    """``(text, explanation)`` of the "Preprocessing source" docstring slot. ``text`` is the first FILLED
    slot found in the bridge class's docstring, then its module's (``None`` when there is none); ``explanation`` names
    the docstrings searched, what was found in each and the layout that satisfies the check, for the
    ``W_NO_PREPROCESSING_SOURCE`` message."""
    docs = _preproc_docs(bridge_cls)
    seen: list[str] = []
    for where, doc in docs:
        lines = doc.splitlines()
        found = False
        for i, line in enumerate(lines):
            if not _PREPROC_RE.match(line):
                continue
            found = True
            inline, body = _slot(lines, i)
            if _filled(inline + body):
                return "\n".join(inline + body), f"filled slot in {where} (line {i + 1})"
            what = "only template text (TODO / <...> placeholders)" if [t for t in inline + body if t] else "nothing (no indented block below the heading)"
            seen.append(f"{where} line {i + 1}: the slot holds {what}")
        if not found:
            seen.append(f"{where}: no line starts with 'Preprocessing source'")
    if not docs:
        seen.append(f"{getattr(bridge_cls, '__qualname__', bridge_cls)} and its module have no docstring")
    return None, "searched " + "; ".join(seen) + ". Expected layout: " + _PREPROC_LAYOUT


def preprocessing_source(bridge_cls: type) -> str | None:
    """The text of the "Preprocessing source" docstring slot of a bridge, or ``None`` when it is missing or
    unfilled (``bridge check`` then warns ``W_NO_PREPROCESSING_SOURCE``). Searched: the bridge class's own
    docstring, then its module's; the first filled slot wins. A slot is a line starting with ``Preprocessing source``
    (the heading: its paragraph's prose up to the line ending in ``:`` is ignored) followed by the pasted source as
    an indented block (blank lines before or inside it are fine), or an inline body after ``Preprocessing source:``
    on the same line. A body with a line starting ``TODO`` (template text left in), or holding only ``<…>``
    placeholders / "e.g." prose, is unfilled. ``preprocessing_source_check`` explains a ``None``."""
    return preprocessing_source_check(bridge_cls)[0]


# ----------------------------------------------------------------------------- weights fingerprint


def _is_tensor(x: Any) -> bool:
    return all(hasattr(x, a) for a in ("detach", "cpu", "numpy", "dtype", "shape")) and not isinstance(x, np.ndarray)


def _tensor_bytes(t: Any) -> tuple[str, list[int], bytes]:
    """``(dtype, shape, raw bytes)`` of a torch tensor without importing torch: it is already
    imported if a tensor exists. bf16/fp8 have no numpy dtype, hence the uint8 view."""
    torch = sys.modules.get("torch")
    t = t.detach().cpu().contiguous()
    dtype = str(t.dtype).removeprefix("torch.")
    shape = [int(s) for s in t.shape]
    if torch is not None:
        flat = t.reshape(-1)
        raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    else:  # pragma: no cover - a tensor without torch loaded cannot happen
        raw = t.numpy().tobytes()
    return dtype, shape, raw


def fingerprint_state_dict(model_or_sd: Any) -> str:
    """sha256 over sorted ``(name, dtype, shape, raw bytes)`` of a state dict or a module's
    ``state_dict()`` (``describe()["weights_sha256"]``). Duck-typed; never imports torch at module scope.

    Values may be torch tensors, numpy arrays or JSON primitives (some state dicts carry
    step counters). The per-entry header is ``canon([name, dtype, shape, nbytes])`` so that
    a reshape or a dtype change forks the digest even when the bytes do not.
    """
    sd = model_or_sd.state_dict() if callable(getattr(model_or_sd, "state_dict", None)) else model_or_sd
    if not isinstance(sd, Mapping):
        raise TypeError(f"fingerprint_state_dict: expected a module or a mapping, got {type(model_or_sd).__name__}")
    h = hashlib.sha256()
    for name in sorted(sd, key=str):
        v = sd[name]
        if _is_tensor(v):
            dtype, shape, raw = _tensor_bytes(v)
        elif isinstance(v, np.ndarray):
            arr = np.ascontiguousarray(v)
            dtype, shape, raw = arr.dtype.name, list(arr.shape), arr.tobytes()
        else:
            raw = canon(v).encode("utf-8")
            dtype, shape = "json", []
        h.update(canon([str(name), dtype, shape, len(raw)]).encode("utf-8"))
        h.update(b"\0")
        h.update(raw)
        h.update(b"\n")
    return h.hexdigest()


# ----------------------------------------------------------------------------- grid_pos


def grid_pos(
    T: int,
    H: int,
    W: int,
    stream: int,
    t_centres: Sequence[float] | None = None,
    region: Sequence[float] = (0.0, 0.0, 1.0, 1.0),
    cls: bool = False,
) -> np.ndarray:
    """float32 ``[(1 if cls) + T*H*W, 4]`` token centres ``(stream, t, y, x)`` row-major over ``(t, y, x)`` in MEDIA
    coordinates.

    ``t`` = ``t_centres`` (``len == T``, each a fraction of the stream's media duration in ``[0, 1]`` — e.g. the
    frame times the bridge used divided by ``duration_s``, or the mean over a tubelet's frames; ``-1`` marks a
    token centred in padding the bridge added), defaulting to ``(k + 0.5) / T`` (correct only for a ``uniform``
    rule over ``span: full``). ``y, x`` = patch centres mapped into ``region = (y0, x0, y1, x1)``, the part of the
    ORIGINAL frame the model input shows (``yavih.media.rules.crop_region``): ``y = y0 + (i + 0.5)(y1 − y0)/H``.
    The optional CLS row comes FIRST and is ``(stream, -1, -1, -1)``.
    """
    for label, v in (("T", T), ("H", H), ("W", W)):
        if not isinstance(v, int | np.integer) or isinstance(v, bool) or v < 1:
            raise ValueError(f"grid_pos: {label} must be an int >= 1, got {v!r}")
    if not isinstance(stream, int | np.integer) or isinstance(stream, bool) or stream < 0:
        raise ValueError(f"grid_pos: stream must be a non-negative int (index into describe()['streams']), got {stream!r}")
    reg = [float(v) for v in region]
    if len(reg) != 4 or not (0.0 <= reg[0] < reg[2] <= 1.0 and 0.0 <= reg[1] < reg[3] <= 1.0):
        raise ValueError(f"grid_pos: region must be (y0, x0, y1, x1) with 0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1, got {tuple(region)!r}")
    y0, x0, y1, x1 = reg
    if t_centres is None:
        t = (np.arange(T, dtype=np.float64) + 0.5) / T
    else:
        t = np.asarray(list(t_centres), dtype=np.float64)
        if t.shape != (T,):
            raise ValueError(f"grid_pos: t_centres must have length T={T}, got {t.shape[0]}")
        ok = np.isfinite(t) & (((t >= 0.0) & (t <= 1.0)) | (t == -1.0))
        if not bool(ok.all()):
            raise ValueError("grid_pos: t_centres must lie in [0, 1] (fractions of the media duration) or be -1 (padding)")
    y = y0 + (np.arange(H, dtype=np.float64) + 0.5) * (y1 - y0) / H
    x = x0 + (np.arange(W, dtype=np.float64) + 0.5) * (x1 - x0) / W
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")  # row-major (t, y, x): t slowest, x fastest
    grid = np.stack([np.full(tt.size, float(stream)), tt.ravel(), yy.ravel(), xx.ravel()], axis=1)
    if cls:
        grid = np.concatenate([np.array([[float(stream), -1.0, -1.0, -1.0]]), grid], axis=0)
    return np.ascontiguousarray(grid, dtype=np.float32)
