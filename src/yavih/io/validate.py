"""``validate_features``: every ``E_*``/``W_*``/``N_*`` check in the documented order, the
``ValidationReport``, the per-member output-contract checks (``E_TIMES_OUT_OF_MEDIA``,
``E_RULE_MISMATCH``) that ``bridge check`` reuses on its smoke set, and the ONLY place a readout's
eligibility is decided (``N_ELIGIBILITY`` → ``_meta.json.eligibility``).

The validator reads the tars itself with stdlib ``tarfile`` instead of going through
``FeatureReader``: the reader trusts the sidecar's member offsets, and the whole point of
validation is to check that trust (member set, offsets, shapes, dtype, finiteness) before any
probe memmaps the shards. Streaming one pass over every member is what makes ``E_FINITE`` and the
per-member rule checks "over all members" affordable; the sampled checks (dead rows, duplicates,
eligibility) use one seeded draw so verdicts are reproducible on every host. Shards are scanned concurrently
(``YAVIH_VALIDATE_WORKERS``), each on private state, and merged in shard order: the report is the same whatever
the number of workers.

Token positions are in MEDIA coordinates (``t`` = fraction of the stream's duration, ``(y, x)``
= fraction of the ORIGINAL frame). Each ``.json`` member carries the media facts (``media``, copied
by ``run_extract`` from ``RawStream.info``) and the bridge's report of what it consumed (``used``);
validate re-evaluates the rule the bridge declared in ``describe()["input"]`` on those
facts.

``E_FORMAT`` refuses any format other than this yavih's (re-extract); ``E_DESCRIBE_MISMATCH`` covers ``model``,
``reading`` and ``describe_sha256`` (consistency of one feature set, never identity); a ``custom`` rule is a
legitimate opt-out recorded as the note ``N_RULE_CUSTOM``; ``W_RULE_UNCHECKED`` is only for a checkable rule a
null media fact kept from being evaluated; ``W_WEIGHTS_UNFINGERPRINTED`` records a null ``weights_sha256``.
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import math
import os
import re
import socket
import tarfile
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from yavih import hashing
from yavih.hash_history import describe_input_mismatch
from yavih.io.format import (
    CONSTANT_POS_MEMBER,
    FEATURE_DTYPES,
    FEATURE_FORMAT_VERSION,
    META_FILE,
    NOT_APPLICABLE_FILE,
    POS_MODES,
    FormatError,
    parse_member_name,
    shard_sidecar_name,
    shard_tar_name,
)
from yavih.io.sidecar import identity_value
from yavih.media.rules import OutOfMedia, audio_ranges, crop_region, rule_times, select_nearest
from yavih.versions import DESCRIBE_SCHEMA, __version__
from yavih.waivers import W_WAIVED, WaiveError, Waivers, check_waive, describe_waivers

if TYPE_CHECKING:  # pragma: no cover
    from yavih.benchmark import Benchmark

__all__ = [
    "Finding",
    "features_block",
    "iter_members",
    "list_shards",
    "MemberRecord",
    "ValidationReport",
    "ValidationError",
    "validate_features",
    "validate_tree",
    "is_split_dir",
    "split_dirs",
    "eligibility_line",
    "check_member",
    "rules_checked",
    "rel_l2_stored",
    "select_nearest",
    "rule_times",
    "audio_ranges",
    "crop_region",
    "CODES",
    "LEVELS",
    "STALE_TMP_SECONDS",
    "DEAD_ROW_REL_VAR",
    "DUPLICATE_MAX_FRACTION",
    "TIME_TOL_S",
    "SPATIAL_SLACK",
    "DEFAULT_BATCH_TOLERANCE",
    "MAX_BATCH_TOLERANCE",
    "FP16_WARN_ABS",
    "FP16_WARN_ZERO_FRACTION",
    "rules_line",
    "custom_rules",
    "custom_rule_message",
    "validate_workers",
    "VALIDATE_WORKERS_ENV",
]

LEVELS: frozenset[str] = frozenset({"error", "warning", "note"})
#: Every validation code with its level.
CODES: dict[str, str] = {
    "E_FORMAT": "error",
    "E_NA_AND_SHARDS": "error",
    "E_SHARD_SET": "error",
    "W_STALE_TMP": "warning",
    "E_RANGE_TILING": "error",
    "E_HASH_MANIFEST": "error",
    "E_HASH_INPUT": "error",  # the benchmark's input side differs
    "E_DESCRIBE_MISMATCH": "error",
    "E_ARGS_UNCONSUMED": "error",
    "E_TAR_SHA": "error",
    "E_MEMBERS": "error",
    "E_ROWS": "error",
    "E_TIMES_OUT_OF_MEDIA": "error",
    "E_RULE_MISMATCH": "error",
    "E_FINITE": "error",
    "E_POS_SHAPE": "error",
    "E_POS_RANGE": "error",
    "E_BATCH_INVARIANCE": "error",
    "E_DEAD_ROWS": "error",
    "E_DUPLICATE_CLIPS": "error",
    # a waivable error the bridge (describe()["waive"]) or the operator (--waive) waived with a reason (yavih.waivers)
    "W_WAIVED": "warning",
    "E_WAIVE_INVALID": "error",
    "N_ELIGIBILITY": "note",
    # in-tolerance canary drift is a recorded fact, not a problem
    "N_CANARY_DRIFT": "note",
    "N_RULE_CUSTOM": "note",
    "W_LOOSE_TOLERANCE": "warning",
    "W_RULE_UNCHECKED": "warning",
    "W_COVERAGE": "warning",
    "W_MEDIA_FP": "warning",
    "W_FP16_RANGE": "warning",
    "W_CONSTANT_CLIP": "warning",
    "W_SIDE_EFFECT": "warning",
    "W_SIZE": "warning",
    "W_WEIGHTS_UNFINGERPRINTED": "warning",
    "W_SOURCE_SHA": "warning",  # sidecars of one split were written by different bridge FILES (same describe)
}

#: Threads scanning shards concurrently (``validate_workers``); ``1`` = one shard after the other.
VALIDATE_WORKERS_ENV = "YAVIH_VALIDATE_WORKERS"
#: ``.tmp`` older than this is ``W_STALE_TMP``, younger counts as "in progress".
STALE_TMP_SECONDS: float = 36 * 3600.0
#: ``E_DEAD_ROWS``: variance below this fraction of the mean token variance.
DEAD_ROW_REL_VAR: float = 1e-12
#: ``E_DUPLICATE_CLIPS``: more than this fraction of sampled clips byte-identical.
DUPLICATE_MAX_FRACTION: float = 0.01
#: ``E_TIMES_OUT_OF_MEDIA`` slack, added to every ``E_RULE_MISMATCH`` time tolerance (1 ms).
TIME_TOL_S: float = 1e-3
#: ``E_RULE_MISMATCH``: token centres may lie this far outside the declared crop region.
SPATIAL_SLACK: float = 1e-6
#: Default and loosest batch-invariance tolerance (``rel_l2`` per clip).
DEFAULT_BATCH_TOLERANCE: float = 1e-5
MAX_BATCH_TOLERANCE: float = 1e-3
#: ``W_FP16_RANGE`` thresholds.
FP16_WARN_ABS: float = 1e3
FP16_WARN_ZERO_FRACTION: float = 0.5
#: Dense readouts: pooled features (``N == 1``) can never serve them.
def _dense(readout: str) -> bool:
    """Whether the readout needs per-token media positions (``Readout.dense``, ``yavih.readouts.registry``)."""
    from yavih.readouts import registry

    try:
        return bool(registry.get(readout).dense)
    except registry.UnknownReadout:
        return False

_SIDECAR_RE = re.compile(r"^shard-(\d{6})\.json$")
_TAR_RE = re.compile(r"^shard-(\d{6})\.tar$")
_POS_AXES: tuple[str, ...] = ("t", "y", "x")


@dataclass(frozen=True)
class Finding:
    """``{code, message, where}`` (the CLI's ``--json`` envelope)."""

    code: str
    message: str
    where: str | None = None

    @property
    def level(self) -> str:
        return CODES.get(self.code, "error")

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message, "where": self.where}

    @classmethod
    def from_json(cls, d: dict) -> Finding:
        return cls(code=str(d["code"]), message=str(d.get("message", "")), where=d.get("where"))


class ValidationError(Exception):
    """Raised by ``ValidationReport.raise_for_errors`` (exit 2)."""

    def __init__(self, errors: list[Finding]) -> None:
        super().__init__("; ".join(f"{e.code}: {e.message}" for e in errors))
        self.errors = errors


@dataclass
class ValidationReport:
    """The validation report. ``shards`` (not a documented field) carries the shard table the CLI prints
    (``shards 31/32 done, 0 in progress, 1 missing``); ``rules_checked`` the rules line;
    ``meta_path`` is where ``_meta.json`` went."""

    ok: bool
    errors: list[Finding]
    warnings: list[Finding]
    notes: list[Finding]
    split: str
    coverage: dict
    N: int
    D: int
    dtype: str
    pos_mode: str
    bytes: int
    backend_hint: str
    eligibility: dict  # the _meta.json eligibility block
    missing_shards: list[str] = field(default_factory=list)  # ["--shard 7/32", ...]
    shards: dict = field(default_factory=dict)  # {n, done, in_progress, missing}
    rules_checked: dict = field(default_factory=dict)  # _meta.json rules_checked
    meta_path: str | None = None
    describe: dict = field(default_factory=dict, repr=False)  # the feature set's describe() (custom-rule descriptions for the rules line)
    waived: list = field(default_factory=list)  # [{code, reason, source, count}]: the waivers in force (yavih.waivers)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValidationError(self.errors)

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "errors": [f.to_json() for f in self.errors],
            "warnings": [f.to_json() for f in self.warnings],
            "notes": [f.to_json() for f in self.notes],
            "split": self.split,
            "coverage": self.coverage,
            "N": self.N,
            "D": self.D,
            "dtype": self.dtype,
            "pos_mode": self.pos_mode,
            "bytes": self.bytes,
            "backend_hint": self.backend_hint,
            "eligibility": self.eligibility,
            "missing_shards": list(self.missing_shards),
            "shards": dict(self.shards),
            "rules_checked": dict(self.rules_checked),
            "meta_path": self.meta_path,
            "waived": [dict(w) for w in self.waived],
        }

    def summary_lines(self) -> list[str]:
        """The lines ``yavih features validate`` prints: shard table, rules, eligibility."""
        sh = self.shards
        out = []
        if sh:
            line = f"shards {sh.get('done', 0)}/{sh.get('n', '?')} done, {sh.get('in_progress', 0)} in progress, {sh.get('missing', 0)} missing"
            if self.missing_shards:
                line += ": " + " ".join(self.missing_shards)
            out.append(line)
        if self.rules_checked:
            out.append(rules_line(self.rules_checked, self.describe))
        out.append(eligibility_line(self.eligibility))
        return out


def custom_rules(describe: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """``{stream: {"temporal" | "spatial": description}}`` of the rules a bridge declared ``custom`` (the
    opt-out from the rule vocabulary, recorded as ``N_RULE_CUSTOM``)."""
    out: dict[str, dict[str, str]] = {}
    for sname, entry in dict((describe or {}).get("input") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        for which in ("temporal", "spatial"):
            r = entry.get(which)
            if isinstance(r, Mapping) and r.get("rule") == "custom":
                out.setdefault(str(sname), {})[which] = str(r.get("description") or "").strip()
    return out


def custom_rule_message(sname: str, which: Mapping[str, str]) -> str:
    """The ``N_RULE_CUSTOM`` message for one stream: which rule the bridge opted out of, its description
    and what still ran."""
    parts = [f"{w} rule custom — {d or '(no description)'}" for w, d in which.items()]
    checked = ["out-of-media", "range", "positions"]
    if "spatial" in which and "temporal" not in which:
        checked = ["out-of-media", "range (token centres in [0, 1])", "positions in the media span"]
    return f"{sname}: " + "; ".join(parts) + f"; checked: {', '.join(checked)} (the bridge opted out of the rule vocabulary)"


def rules_line(rc: Mapping[str, Any], describe: Mapping[str, Any] | None = None) -> str:
    """``rules checked: video uniform + spatial; audio span`` or ``audio custom (opted out: <description>)``.
    ``describe`` supplies the descriptions of custom rules."""
    customs = custom_rules(describe)
    parts = []
    for sname, r in rc.items():
        t = r.get("temporal")
        txt = f"{sname} {t}" if t else str(sname)
        k = r.get("temporal_unchecked_rows")
        if t and isinstance(k, int) and k > 0:
            txt += f" (unchecked on {k} row{'s' if k != 1 else ''})"
        if r.get("spatial") is True:
            txt += " + spatial"
        c = customs.get(str(sname)) or {}
        if c:
            desc = "; ".join(f"{w}: {d}" if len(c) > 1 or w == "spatial" else d for w, d in c.items() if d) or "no description"
            txt += f" (opted out: {desc})"
        parts.append(txt)
    return "rules checked: " + ("; ".join(parts) if parts else "none")


def eligibility_line(elig: dict) -> str:
    """``eligible for box: no (grid coverage median 0.18 < 0.25 over the full frame) → probe will record not_applicable``."""
    readout = elig.get("readout", "?")
    ok = elig.get("eligible")
    if ok is True:
        return f"eligible for {readout}: yes"
    if ok is None:
        return f"eligible for {readout}: undecided ({elig.get('reason')})"
    detail = elig.get("reason") or ("missing: " + ", ".join(elig.get("missing") or []))
    return f"eligible for {readout}: no ({detail}) → probe will record not_applicable"


# ----------------------------------------------------------------------------- helpers


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Findings:
    """Ordered collector; the documented check order is the order of ``add`` calls. ``cap`` bounds the number of
    findings per code so a systematically wrong bridge does not produce one finding per member."""

    def __init__(self, cap: int = 20) -> None:
        self.errors: list[Finding] = []
        self.warnings: list[Finding] = []
        self.notes: list[Finding] = []
        self.cap = cap
        self.counts: dict[str, int] = {}
        #: waivers in force (yavih.waivers): a waived error is recorded as W_WAIVED and counted, never dropped
        self.waivers: Waivers | None = None

    def add(self, code: str, message: str, where: str | None = None) -> None:
        level = CODES[code]
        n = self.counts.get(code, 0) + 1
        self.counts[code] = n
        if level == "error" and self.waivers is not None and self.waivers.covers(code):
            self.waivers.record(code)
            if n <= self.cap:
                self.warnings.append(Finding(W_WAIVED, self.waivers.message(code, message), where))
            elif n == self.cap + 1:
                self.warnings.append(Finding(W_WAIVED, self.waivers.message(code, f"further {code} findings suppressed (more than {self.cap})"), None))
            return
        bucket = {"error": self.errors, "warning": self.warnings, "note": self.notes}[level]
        if n <= self.cap:
            bucket.append(Finding(code, message, where))
        elif n == self.cap + 1:
            bucket.append(Finding(code, f"further {code} findings suppressed (more than {self.cap}); fix the first ones", None))

    def has(self, code: str) -> bool:
        return self.counts.get(code, 0) > 0

    @property
    def ok(self) -> bool:
        return not self.errors


def is_split_dir(path: str | os.PathLike) -> bool:
    """A split directory holds sidecars, tars, ``NOT_APPLICABLE.json`` or ``_meta.json``."""
    p = Path(path)
    if not p.is_dir():
        return False
    for child in p.iterdir():
        if child.name in (NOT_APPLICABLE_FILE, META_FILE) or _SIDECAR_RE.match(child.name) or _TAR_RE.match(child.name):
            return True
    return False


def split_dirs(features_root: str | os.PathLike) -> list[Path]:
    """The split subdirectories of a ``features/`` parent, sorted by name."""
    root = Path(features_root)
    return sorted(p for p in root.iterdir() if p.is_dir() and is_split_dir(p))


def validate_tree(path: str | os.PathLike, **kwargs: Any) -> list[ValidationReport]:
    """``yavih features validate <dir>``: a split directory → one report; its ``features/``
    parent → one report per split subdirectory (exit 2 if any fails)."""
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"{p}: not a directory")
    if is_split_dir(p):
        return [validate_features(p, **kwargs)]
    subs = split_dirs(p)
    if not subs:
        raise FileNotFoundError(f"{p}: neither a split directory nor a features/ parent with split subdirectories")
    return [validate_features(s, **kwargs) for s in subs]


def _ram_budget_bytes() -> int:
    """The RAM budget: ``yavih.resources.ram_budget_bytes`` (the one implementation)."""
    from yavih.resources import ram_budget_bytes

    return int(ram_budget_bytes())


def _rel_l2(c: np.ndarray, o: np.ndarray) -> float:
    """``‖c−o‖₂ / max(‖o‖₂, 1e-12)`` per clip in float64 (``E_BATCH_INVARIANCE``)."""
    c64 = c.astype(np.float64).ravel()
    o64 = o.astype(np.float64).ravel()
    return float(np.linalg.norm(c64 - o64) / max(float(np.linalg.norm(o64)), 1e-12))


def rel_l2_stored(c: np.ndarray, o: np.ndarray) -> float:
    """The canary ``rel_l2`` net of storage quantization (``E_BATCH_INVARIANCE``): for float16 members a
    difference of at most one float16 ulp of the larger magnitude is not counted, because a float32 drift of
    ~1e-6 (well inside ``bridge check``'s tolerance) already flips some values by one ulp when both copies are
    rounded to float16. Real batch dependence (batch statistics, padding to the batch maximum) moves values by
    many ulps and is still measured. float32 members: plain ``rel_l2``."""
    c64 = c.astype(np.float64).ravel()
    o64 = o.astype(np.float64).ravel()
    d = c64 - o64
    if c.dtype == np.float16 and o.dtype == np.float16:
        big = np.maximum(np.abs(c), np.abs(o)).astype(np.float16)
        ulp = np.spacing(big).astype(np.float64).ravel()
        d[np.abs(d) <= ulp] = 0.0
    return float(np.linalg.norm(d) / max(float(np.linalg.norm(o64)), 1e-12))


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    v = float(x)
    return v if math.isfinite(v) else None


# ----------------------------------------------------------------------------- declared-rule evaluation
#
# One semantics: validate re-evaluates a declared rule with ``yavih.media.rules``, the same
# torch-free helpers a bridge may use to pick its times, so a bridge that follows its declaration is never
# flagged. ``select_nearest``, ``rule_times``, ``audio_ranges`` and ``crop_region`` are re-exported here.


# ----------------------------------------------------------------------------- benchmark view


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    out = {}
    for k in getattr(obj, "__dataclass_fields__", {}) or ():
        out[k] = getattr(obj, k)
    if out:
        return out
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")} if hasattr(obj, "__dict__") else {}


class _BenchView:
    """What validate needs from a ``Benchmark``, tolerant of a partially
    implemented one: every optional facet degrades to ``None`` instead of failing validation."""

    def __init__(self, bench: Any) -> None:
        self.bench = bench
        spec = getattr(bench, "spec", None)
        self.manifest_hash: str | None = getattr(bench, "manifest_hash", None)
        self.protocol_hash: str | None = getattr(bench, "protocol_hash", None)
        self.input_hash: str | None = getattr(bench, "input_hash", None)
        self.input_components: dict | None = getattr(getattr(bench, "hashes", None), "input_components", None)
        self.name: str | None = getattr(bench, "name", None)
        self.readout: str | None = _get(spec, "readout")
        oc = _get(spec, "output_contract")
        self.requires: dict = _as_dict(_get(oc, "requires"))
        self.readout_params: dict = dict(_get(spec, "readout_params") or {})
        self.protocol: dict = dict(_get(spec, "protocol") or {})
        self.splits: dict = _as_dict(_get(spec, "splits"))
        self.streams: dict[str, Any] = dict(_get(spec, "streams") or {})
        self.stream_kinds: dict[str, str] = {}
        for sname, s in self.streams.items():
            kind = _get(s, "kind")
            if kind is not None:
                self.stream_kinds[str(sname)] = str(kind)

    def role_of(self, split: str) -> str | None:
        for role in ("train", "eval", "floor", "folds", "unused"):
            if split in tuple(self.splits.get(role) or ()):
                return role
        return None

    def target_region(self) -> list[float] | None:
        tr = self.protocol.get("target_region")
        if isinstance(tr, list | tuple) and len(tr) == 4:
            return [float(v) for v in tr]
        return None

    def n_rows(self, split: str) -> int | None:
        fn = getattr(self.bench, "n_rows", None)
        if fn is None:
            return None
        try:
            return int(fn(split))
        except Exception:  # noqa: BLE001 - optional facet
            return None

    def data_status(self) -> Any:
        fn = getattr(self.bench, "data_status", None)
        if fn is None:
            return None
        try:
            return fn()
        except Exception:  # noqa: BLE001 - never touches the verdict; W_MEDIA_FP is best effort
            return None


def _load_bench_view(bench: Any, sidecars: list[dict]) -> tuple[_BenchView | None, str | None]:
    """The given ``Benchmark`` or ``Benchmark.load(sidecar.benchmark.name)``; ``(None, why)`` when neither works."""
    if bench is not None:
        return _BenchView(bench), None
    names = {str((s.get("benchmark") or {}).get("name")) for s in sidecars}
    if len(names) != 1:
        return None, f"sidecars name several benchmarks: {sorted(names)}"
    name = names.pop()
    try:
        from yavih.benchmark import Benchmark

        return _BenchView(Benchmark.load(name)), None
    except Exception as e:  # noqa: BLE001 - eligibility is then undecided, recorded as such
        return None, f"benchmark {name!r} not loadable here: {type(e).__name__}: {e}"


# ----------------------------------------------------------------------------- per-member output-contract checks


def _infer_kind(entry: Mapping[str, Any], used: Any, media: Any) -> str | None:
    """A stream's kind when the benchmark is not at hand: from the declared input block, ``used`` and ``media``."""
    if not isinstance(entry, Mapping):
        return None
    if "front_end" in entry or (isinstance(used, Mapping) and "samples" in used):
        return "audio"
    if "temporal" not in entry:
        return "image"
    return "video"


def _well_formed_used(kind: str, u: Any) -> str | None:
    """Why ``used[stream]`` is ill-formed for ``kind``, or ``None``."""
    if not isinstance(u, Mapping):
        return f"used entry must be an object, got {type(u).__name__}"
    if kind == "video":
        t = u.get("times")
        if not isinstance(t, list) or not all(_num(v) is not None for v in t):
            return "a video used entry must be {'times': [media seconds, …]}"
    elif kind == "audio":
        s, p = u.get("samples"), u.get("pad")
        if not (isinstance(s, list) and all(isinstance(r, list) and len(r) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in r) for r in s)):
            return "an audio used entry must have 'samples': [[lo, hi], …] (native-rate ints)"
        if not (isinstance(p, list) and len(p) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in p)):
            return "an audio used entry must have 'pad': [left, right] (sample counts >= 0)"
    return None


def rules_checked(describe: Mapping[str, Any], kinds: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """``_meta.json`` ``rules_checked``: per stream read, the temporal rule validate re-evaluates (``"custom"`` = the bridge
    opted out, ``N_RULE_CUSTOM``; ``None`` for images) and whether the spatial rule is checked (``False`` = custom,
    ``None`` where a kind has no spatial rule)."""
    out: dict[str, dict[str, Any]] = {}
    for sname, entry in dict((describe or {}).get("input") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        kind = kinds.get(sname) or _infer_kind(entry, None, None) or "video"
        t = entry.get("temporal")
        sp = entry.get("spatial")
        out[str(sname)] = {
            "temporal": (t.get("rule") if isinstance(t, Mapping) else None) if kind != "image" else None,
            "spatial": (isinstance(sp, Mapping) and sp.get("rule") != "custom") if kind in ("video", "image") else None,
        }
    return out


def check_member(
    meta: Mapping[str, Any],
    pos: np.ndarray | None,
    *,
    describe: Mapping[str, Any],
    kinds: Mapping[str, str],
    where: str | None,
    fx: Any,
    unchecked: dict[str, int] | None = None,
) -> None:
    """Per-member checks on one ``.json`` member (and its token positions): ``E_MEMBERS`` for a
    missing ``media``/``used`` entry or an ill-formed ``used``, ``E_TIMES_OUT_OF_MEDIA``, ``E_RULE_MISMATCH``
    (temporal rule re-evaluated on the member's ``media``; spatial: token centres inside ``crop_region``;
    temporal positions: the ``t`` of every token of a stream inside the media span the bridge reported in
    ``used``, image tokens ``t = -1``). ``kinds`` = ``{stream: kind}`` (may be empty: kinds are then inferred);
    ``fx`` = anything with ``add(code, message, where)``;
    ``unchecked`` (optional) counts, per stream, the members whose declared checkable temporal rule could not
    be evaluated because a fact it needs is null (``W_RULE_UNCHECKED``). Used by validate and by ``bridge check``."""
    media = meta.get("media")
    used = meta.get("used")
    inp = dict((describe or {}).get("input") or {})
    if not isinstance(media, Mapping):
        fx.add("E_MEMBERS", ".json lacks 'media' (the RawStream facts run_extract copies)", where)
        media = {}
    if not isinstance(used, Mapping):
        fx.add("E_MEMBERS", ".json lacks 'used' (what the bridge consumed per stream)", where)
        return
    tags = list((describe or {}).get("streams") or [])
    for sname, entry in inp.items():
        if not isinstance(entry, Mapping):
            continue
        if sname not in used:
            fx.add("E_MEMBERS", f".json used lacks stream {sname!r} declared in describe()['input']", where)
            continue
        u = used[sname]
        info = media.get(sname)
        if not isinstance(info, Mapping):
            fx.add("E_MEMBERS", f".json media lacks stream {sname!r} that the bridge reads", where)
            continue
        kind = kinds.get(sname) or _infer_kind(entry, u, info) or "video"
        bad = _well_formed_used(kind, u)
        if bad is not None:
            fx.add("E_MEMBERS", f"used[{sname!r}]: {bad}", where)
            continue
        if kind in ("video", "audio") and info.get("has_track") is False:
            _check_trackless(sname, kind, u, where, fx)
            continue  # no track: no duration, rate or frame size, so no rule, crop or position applies
        evaluated = True
        if kind == "video":
            evaluated = _check_times(sname, entry.get("temporal") or {}, u, info, where, fx)
        elif kind == "audio":
            evaluated = _check_audio(sname, entry.get("temporal") or {}, u, info, where, fx)
        if not evaluated and unchecked is not None:
            unchecked[str(sname)] = unchecked.get(str(sname), 0) + 1
        sp = entry.get("spatial")
        if kind in ("video", "image") and isinstance(sp, Mapping) and sp.get("rule") != "custom" and pos is not None and sname in tags:
            _check_spatial(sname, sp, tags.index(sname), info, pos, where, fx)
        if pos is not None and sname in tags:
            _check_pos_t(sname, kind, tags.index(sname), u, info, pos, where, fx)


def _check_trackless(sname: str, kind: str, u: Mapping[str, Any], where: str | None, fx: Any) -> None:
    """The row's file holds no track of this stream (``media[stream].has_track`` false). The row was
    handed over and the bridge chose what to feed for it (e.g. its own silence), so the only valid report is an
    empty consumption: no times (video) / no sample ranges (audio); the audio ``pad`` is the bridge's own and is
    not checked. Not an unchecked rule (``W_RULE_UNCHECKED``): there is nothing a rule could select."""
    if kind == "video" and u.get("times"):
        fx.add("E_TIMES_OUT_OF_MEDIA", f"used[{sname!r}].times {list(u['times'])[:4]} but this clip's file holds no {kind} track (has_track: false): "
               "report times [] for a track-less stream", where)
    elif kind == "audio" and u.get("samples"):
        fx.add("E_TIMES_OUT_OF_MEDIA", f"used[{sname!r}].samples {list(u['samples'])[:3]} but this clip's file holds no {kind} track (has_track: false): "
               "report samples [] for a track-less stream", where)


def _check_pos_t(sname: str, kind: str, tag: int, u: Mapping[str, Any], info: Mapping[str, Any],
                 pos: np.ndarray, where: str | None, fx: Any) -> None:
    """``E_RULE_MISMATCH`` (temporal positions): ``t`` is a fraction of the stream's media
    duration, so ``tau = t · duration_s`` of every timed token of the stream must lie in the media span the
    bridge reported consuming: video ``[min(times) − h, max(times) + h]`` with ``h`` half a frame
    + 1 ms; audio ``[min(lo)/sr, max(hi)/sr]`` ± 1 ms (a token
    centred in padding carries ``t = -1``). Image tokens carry ``t = -1``. Skipped when ``duration_s`` is null
    or ``used`` reports no times / samples (a ``custom`` rule that reads nothing it can name)."""
    if not (isinstance(pos, np.ndarray) and pos.ndim == 2 and pos.shape[1] == 4):
        return
    p = pos.astype(np.float64)
    m = (p[:, 0] == float(tag)) & (p[:, 1] >= 0)
    if not bool(m.any()):
        return
    if kind == "image":
        fx.add("E_RULE_MISMATCH", f"{int(m.sum())} {sname!r} tokens carry a time t ≥ 0, but an image has no time axis: image tokens carry t = -1",
               where)
        return
    dur = _num(info.get("duration_s"))
    if dur is None or dur <= 0:
        return
    tau = p[m, 1] * dur
    if kind == "video":
        times = [float(v) for v in (u.get("times") or [])]
        if not times:
            return
        half = _half_frame(info) or 0.0
        lo, hi = min(times) - half - TIME_TOL_S, max(times) + half + TIME_TOL_S
        span = f"[{min(times):.3f}, {max(times):.3f}] s ± {half * 1e3:.1f} ms"
    elif kind == "audio":
        rng = [r for r in (u.get("samples") or []) if isinstance(r, list | tuple) and len(r) == 2]
        sr = _num(info.get("sample_rate"))
        if not rng or sr is None or sr <= 0:
            return
        lo = min(int(r[0]) for r in rng) / sr - TIME_TOL_S
        hi = max(int(r[1]) for r in rng) / sr + TIME_TOL_S
        span = f"[{lo + TIME_TOL_S:.3f}, {hi - TIME_TOL_S:.3f}] s"
    else:
        return
    out = (tau < lo) | (tau > hi)
    if bool(out.any()):
        k = int(np.argmax(out))
        fx.add("E_RULE_MISMATCH", f"pos t of {int(out.sum())} {sname!r} tokens lies outside the media span the bridge consumed {span} "
               f"(first: t={p[m, 1][k]:.4f} → {tau[k]:.3f} s of duration_s={dur}); positions are fractions of the media duration, "
               "e.g. used times / duration_s, not of the window", where)


def _check_times(sname: str, temporal: Mapping[str, Any], u: Mapping[str, Any], info: Mapping[str, Any],
                 where: str | None, fx: Any) -> bool:
    """False when a declared checkable rule could not be evaluated on this member (a fact it needs is null)."""
    got = np.asarray([float(v) for v in u["times"]], dtype=np.float64)
    dur = _num(info.get("duration_s"))
    # E_TIMES_OUT_OF_MEDIA on what was reported
    if dur is not None and got.size:
        out = got[(got < -TIME_TOL_S) | (got > dur + TIME_TOL_S)]
        if out.size:
            fx.add("E_TIMES_OUT_OF_MEDIA", f"used[{sname!r}].times {out[:4].round(4).tolist()} outside [0, duration_s={dur}] (1 ms slack)", where)
            return True
    rule = temporal.get("rule")
    if rule in (None, "custom"):
        return True
    try:
        expected = rule_times(temporal, info)
    except OutOfMedia as e:
        fx.add("E_TIMES_OUT_OF_MEDIA", f"the declared {rule!r} rule does not fit this clip's media (duration_s={dur}): {e}; declare edge: clamp "
               "or a rule that fits short media", where)
        return True
    except (ValueError, KeyError, TypeError):
        return False  # a fact the rule needs is null for this row: unchecked, counted
    if expected is None:
        return True
    half = _half_frame(info)
    if half is None:
        return False  # neither fps nor (duration_s, n_frames): the nearest-frame tolerance is unknown
    tol = half + TIME_TOL_S
    fps = _num(info.get("fps"))
    # a decoded frame's pts never exceeds the last frame's: compare against the nearest frame that exists
    if fps is not None and dur is not None:
        expected = np.clip(expected, 0.0, max(0.0, dur - 1.0 / fps))
    if got.shape != expected.shape:
        fx.add("E_RULE_MISMATCH", f"used[{sname!r}] reports {got.size} times, the declared {rule!r} rule gives {expected.size}", where)
        return True
    if got.size:
        dev = float(np.abs(got - expected).max())
        if dev > tol:
            fx.add("E_RULE_MISMATCH", f"used[{sname!r}].times deviate {dev * 1e3:.2f} ms (> {tol * 1e3:.2f} ms) from the declared {rule!r} rule on this clip's media "
                   f"(expected {expected[:4].round(4).tolist()}…, got {got[:4].round(4).tolist()}…)", where)
    return True


def _half_frame(info: Mapping[str, Any]) -> float | None:
    fps = _num(info.get("fps"))
    if fps is not None and fps > 0:
        return 0.5 / fps
    dur, nf = _num(info.get("duration_s")), info.get("n_frames")
    if dur is not None and isinstance(nf, int) and nf > 0:
        return 0.5 * dur / nf
    return None


def _check_audio(sname: str, temporal: Mapping[str, Any], u: Mapping[str, Any], info: Mapping[str, Any], where: str | None, fx: Any) -> bool:
    """False when a declared checkable rule could not be evaluated on this member (sample_rate / n_samples null)."""
    got = np.asarray(u["samples"], dtype=np.int64).reshape(-1, 2)
    pad = [int(v) for v in u["pad"]]
    ns = info.get("n_samples")
    if isinstance(ns, int) and got.size:
        bad = got[(got[:, 0] < 0) | (got[:, 1] > ns) | (got[:, 1] < got[:, 0])]
        if bad.size:
            fx.add("E_TIMES_OUT_OF_MEDIA", f"used[{sname!r}].samples {bad[:3].tolist()} outside [0, n_samples={ns}] (or lo > hi)", where)
            return True
    rule = temporal.get("rule")
    if rule in (None, "custom"):
        return True
    try:
        exp = audio_ranges(temporal, info)
    except OutOfMedia as e:
        fx.add("E_TIMES_OUT_OF_MEDIA", f"the declared {rule!r} rule does not fit this clip's audio (n_samples={ns}): {e}; declare pad: zeros "
               "or a shorter span", where)
        return True
    except (ValueError, KeyError, TypeError):
        return False  # sample_rate / n_samples unknown for this row: unchecked, counted
    if exp is None:
        return True
    ranges, (pl, pr) = exp
    if got.shape != ranges.shape:
        fx.add("E_RULE_MISMATCH", f"used[{sname!r}] reports {len(got)} sample ranges, the declared {rule!r} rule gives {len(ranges)}", where)
        return True
    dev = int(np.abs(got - ranges).max()) if got.size else 0
    pdev = max(abs(pad[0] - pl), abs(pad[1] - pr))
    if dev > 1 or pdev > 1:
        fx.add("E_RULE_MISMATCH", f"used[{sname!r}] samples {got[:2].tolist()} pad {pad} differ from the declared {rule!r} rule "
               f"({ranges[:2].tolist()} pad {[pl, pr]}) by more than one sample", where)
    return True


def _check_spatial(sname: str, spatial: Mapping[str, Any], tag: int, info: Mapping[str, Any], pos: np.ndarray, where: str | None, fx: Any) -> None:
    w, h = info.get("width"), info.get("height")
    if _num(w) is None or _num(h) is None or not (float(w) > 0 and float(h) > 0):
        return
    try:
        region = crop_region(spatial, int(w), int(h))
    except (ValueError, KeyError, TypeError):
        return
    if region is None or not (isinstance(pos, np.ndarray) and pos.ndim == 2 and pos.shape[1] == 4):
        return
    y0, x0, y1, x1 = region
    p = pos.astype(np.float64)
    m = (p[:, 0] == float(tag)) & (p[:, 2] >= 0) & (p[:, 3] >= 0)
    if not bool(m.any()):
        return
    ys, xs = p[m, 2], p[m, 3]
    out = (ys < y0 - SPATIAL_SLACK) | (ys > y1 + SPATIAL_SLACK) | (xs < x0 - SPATIAL_SLACK) | (xs > x1 + SPATIAL_SLACK)
    if bool(out.any()):
        k = int(np.argmax(out))
        fx.add("E_RULE_MISMATCH", f"{int(out.sum())} {sname!r} token centres lie outside the declared crop region "
               f"[{y0:.4f}, {x0:.4f}, {y1:.4f}, {x1:.4f}] of the {int(w)}x{int(h)} original frame (first: y={ys[k]:.4f}, x={xs[k]:.4f}); "
               "positions are fractions of the ORIGINAL frame (grid_pos(region=crop_region(...)))", where)


# ----------------------------------------------------------------------------- npy members


def _read_npy(data: bytes, where: str, fx: _Findings) -> tuple[np.ndarray | None, int]:
    """Parse one ``.npy`` member (v1.0 header, C-order); returns ``(array | None, header_len)``."""
    fh = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
        else:
            fx.add("E_MEMBERS", f"unsupported .npy header version {version}", where)
            return None, 0
    except (ValueError, OSError) as e:
        fx.add("E_MEMBERS", f"malformed .npy header: {e}", where)
        return None, 0
    header_len = fh.tell()
    if version != (1, 0):
        fx.add("E_MEMBERS", f".npy header version {version} ≠ 1.0", where)
    if fortran:
        fx.add("E_MEMBERS", ".npy member is Fortran-ordered (C-order)", where)
    if dtype.hasobject:
        fx.add("E_MEMBERS", ".npy member has object dtype", where)
        return None, header_len
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if len(data) - header_len != count * dtype.itemsize:
        fx.add("E_MEMBERS", f".npy payload {len(data) - header_len} B ≠ {count * dtype.itemsize} B for {shape} {dtype}", where)
        return None, header_len
    arr = np.frombuffer(data, dtype=dtype, count=count, offset=header_len).reshape(shape)
    return arr, header_len


# ----------------------------------------------------------------------------- shard records


@dataclass
class _Shard:
    index: int
    sidecar_path: Path
    sidecar: dict
    tar_path: Path | None

    @property
    def where(self) -> str:
        return self.sidecar_path.name

    @property
    def lo(self) -> int:
        return int(self.sidecar["shard"]["rows"][0])

    @property
    def hi(self) -> int:
        return int(self.sidecar["shard"]["rows"][1])


_SIDECAR_REQUIRED = ("format_version", "benchmark", "split", "model", "reading", "shard", "rows", "missing", "features", "bridge", "tar", "members")


def _read_sidecar(path: Path, fx: _Findings) -> dict | None:
    """``E_FORMAT``: unknown ``format_version``, ``format_version 1`` or
    ``format_version 2`` (re-extract), or an unparsable/incomplete sidecar."""
    try:
        with open(path, encoding="utf-8") as fh:
            sc = json.load(fh)
    except (OSError, ValueError) as e:
        fx.add("E_FORMAT", f"sidecar unreadable: {e}", path.name)
        return None
    if not isinstance(sc, dict):
        fx.add("E_FORMAT", "sidecar is not a JSON object", path.name)
        return None
    fv = sc.get("format_version")
    if isinstance(fv, bool) or fv != FEATURE_FORMAT_VERSION:
        fx.add("E_FORMAT", f"unsupported format_version {fv!r} (this yavih reads {FEATURE_FORMAT_VERSION}); re-extract", path.name)
        return None
    missing = [k for k in _SIDECAR_REQUIRED if k not in sc]
    if missing:
        fx.add("E_FORMAT", f"sidecar lacks keys {missing}", path.name)
        return None
    sh = sc["shard"]
    if not (isinstance(sh, dict) and isinstance(sh.get("rows"), list) and len(sh["rows"]) == 2 and isinstance(sh.get("n"), int) and isinstance(sh.get("index"), int)):
        fx.add("E_FORMAT", "sidecar shard block malformed (needs index, n, rows [lo, hi], batch_size)", path.name)
        return None
    feats = sc["features"]
    if not isinstance(feats, dict) or feats.get("dtype") not in FEATURE_DTYPES or feats.get("pos_mode") not in POS_MODES:
        fx.add("E_FORMAT", f"sidecar features block malformed: dtype ∈ {sorted(FEATURE_DTYPES)}, pos_mode ∈ {sorted(POS_MODES)}", path.name)
        return None
    if not isinstance(sc.get("members"), list) or not isinstance(sc.get("missing"), list):
        fx.add("E_FORMAT", "sidecar members/missing must be lists", path.name)
        return None
    return sc


# ----------------------------------------------------------------------------- the scan


@dataclass
class _EligSpec:
    """What the per-row eligibility measurements need (``N_ELIGIBILITY``)."""

    grid_hw: tuple[int, int] | None = None
    region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    visual_tag: int | None = None
    seq_seconds: frozenset[int] | None = None  # scored absolute seconds
    segment_seconds: float = 1.0
    tag_names: list[str] = field(default_factory=list)
    # requires.annotated_frame: the stream's tag, the fact's media second and the tolerance
    annot_tag: int | None = None
    annot_stream: str | None = None
    annot_second: float | None = None
    annot_tol: float | None = None


class _Stats:
    """Accumulators of the single streaming pass (the checks that need every member or the
    seeded sample). All tensors are ``[N, D]`` float64 so fp16 inputs do not lose the variance.

    ``deferred=True`` is a shard's partial (``_scan_shard_isolated``): the order-sensitive accumulations (the
    float64 ``sum`` / ``sumsq`` over sampled rows, the per-row eligibility) are queued and run by :meth:`merge` into
    the split's accumulator in shard order, so a parallel scan accumulates exactly what a serial one does, in the same
    order, bit for bit."""

    def __init__(self, N: int, D: int, sampled: set[int] | frozenset[int], elig: _EligSpec, *, deferred: bool = False) -> None:
        self.N, self.D = N, D
        self.sampled = sampled
        self.elig = elig
        self.deferred = deferred
        self.n_sampled_seen = 0
        self.sum: np.ndarray | None = None if deferred else np.zeros((N, D), dtype=np.float64)
        self.sumsq: np.ndarray | None = None if deferred else np.zeros((N, D), dtype=np.float64)
        self._sampled_queue: list[np.ndarray] = []  # deferred: sampled rows in scan order (stored dtype)
        self._elig_queue: list[tuple[int, Mapping[str, Any], np.ndarray]] = []  # deferred: row_eligibility calls
        self.max_abs = 0.0
        self.n_values = 0
        self.n_zero = 0
        self.constant_clips: list[tuple[int, str, str | None]] = []  # (row, clip_id, media_key)
        self.dup_hash: dict[int, tuple[str, str | None]] = {}  # row -> (sha, media_key)
        self.clip_ids: list[str] = []
        self.present_rows: list[int] = []
        self.bytes = 0
        self.canary_pairs: list[tuple[int, float]] = []  # (row, rel_l2)
        self.canary_mismatch: list[tuple[int, str]] = []  # (row, what differs: pos / used)
        self.pos_streams: set[int] = set()
        self.pos_axes: set[str] = set()
        self.cell_cov: list[float] = []  # per sampled row, over visual-stream tokens in the target region
        self.t_bins: list[int] = []  # per sampled row, scored seconds with a timed token
        self.annot_dt: list[float] = []  # per sampled row, min |t·duration − annotated second| over the stream's timed tokens
        self._elig_rows: set[int] = set()

    def feat(self, row: int, clip_id: str, media_key: str | None, arr: np.ndarray) -> None:
        self.present_rows.append(row)
        self.clip_ids.append(clip_id)
        self.bytes += arr.nbytes
        self.n_values += arr.size
        self.n_zero += int(np.count_nonzero(arr == 0))
        if arr.size:
            # |x| and max are exact in the stored dtype (fp16 included): no float32 copy
            self.max_abs = max(self.max_abs, float(np.abs(arr).max()))
            # "all tokens identical" is only informative with N > 1; an all-zero clip always is.
            if (arr.shape[0] > 1 and bool((arr == arr[0]).all())) or not bool(arr.any()):
                self.constant_clips.append((row, clip_id, media_key))
        if row in self.sampled:
            if self.deferred:
                self._sampled_queue.append(arr)
            else:
                self._accumulate(arr)
            self.dup_hash[row] = (hashlib.sha256(arr.tobytes()).hexdigest(), media_key)

    def _accumulate(self, arr: np.ndarray) -> None:
        assert self.sum is not None and self.sumsq is not None
        f64 = arr.astype(np.float64)
        self.sum += f64
        self.sumsq += f64 * f64
        self.n_sampled_seen += 1

    def merge(self, part: _Stats) -> None:
        """Fold a shard's deferred partial into this (split) accumulator. Called in shard order, it reproduces the
        serial pass: lists extend, sets union, ``max_abs`` max (exact), ``dup_hash`` update (a later shard's
        duplicate row overwrites, as in one pass), counts add, and the queued sampled rows and eligibility calls run
        here in their scan order."""
        self.present_rows.extend(part.present_rows)
        self.clip_ids.extend(part.clip_ids)
        self.bytes += part.bytes
        self.n_values += part.n_values
        self.n_zero += part.n_zero
        self.max_abs = max(self.max_abs, part.max_abs)
        self.constant_clips.extend(part.constant_clips)
        for arr in part._sampled_queue:
            self._accumulate(arr)
        self.dup_hash.update(part.dup_hash)
        self.canary_pairs.extend(part.canary_pairs)
        self.canary_mismatch.extend(part.canary_mismatch)
        self.pos_streams |= part.pos_streams
        self.pos_axes |= part.pos_axes
        for row, meta, p in part._elig_queue:
            self.row_eligibility(row, meta, p)

    def pos_summary(self, p: np.ndarray) -> None:
        s = p[:, 0]
        tagged = s >= 0
        self.pos_streams.update(int(v) for v in np.unique(s[tagged]))
        for col, axis in enumerate(_POS_AXES, start=1):
            if bool((p[:, col] >= 0).any()):
                self.pos_axes.add(axis)

    def row_eligibility(self, row: int, meta: Mapping[str, Any], p: np.ndarray) -> None:
        """Per sampled row: grid/box cell coverage of the target region and seq scored seconds."""
        if row not in self.sampled or not (p.ndim == 2 and p.shape[1] == 4):
            return
        if self.deferred:
            self._elig_queue.append((row, meta, p))  # run by merge(), where a row seen in an earlier shard is skipped
            return
        if row in self._elig_rows:
            return
        self._elig_rows.add(row)
        self.pos_summary(p)
        e = self.elig
        s = p[:, 0]
        if e.grid_hw is not None:
            h, w = e.grid_hw
            y0, x0, y1, x1 = e.region
            m = (p[:, 2] >= 0) & (p[:, 3] >= 0)
            if e.visual_tag is not None:
                m &= s == e.visual_tag
            ys, xs = p[m, 2].astype(np.float64), p[m, 3].astype(np.float64)
            inside = (ys >= y0) & (ys <= y1) & (xs >= x0) & (xs <= x1)
            ys, xs = ys[inside], xs[inside]
            cells: set[tuple[int, int]] = set()
            if h > 0 and w > 0 and ys.size:
                cy = np.minimum(((ys - y0) / (y1 - y0) * h).astype(np.int64), h - 1)
                cx = np.minimum(((xs - x0) / (x1 - x0) * w).astype(np.int64), w - 1)
                cells = set(zip(cy.tolist(), cx.tolist(), strict=True))
            self.cell_cov.append(len(cells) / float(h * w) if h > 0 and w > 0 else 0.0)
        if e.seq_seconds is not None:
            media = meta.get("media") if isinstance(meta.get("media"), Mapping) else {}
            secs: set[int] = set()
            for tag in np.unique(s[(s >= 0) & (p[:, 1] >= 0)]):
                ti = int(tag)
                if not (0 <= ti < len(e.tag_names)):
                    continue
                dur = _num((media.get(e.tag_names[ti]) or {}).get("duration_s")) if isinstance(media.get(e.tag_names[ti]), Mapping) else None
                if dur is None:
                    continue
                t = p[(s == tag) & (p[:, 1] >= 0), 1].astype(np.float64)
                secs.update(int(v) for v in np.floor(t * dur / e.segment_seconds + 1e-9).tolist())
            self.t_bins.append(len(secs & set(e.seq_seconds)))
        if e.annot_second is not None:
            media = meta.get("media") if isinstance(meta.get("media"), Mapping) else {}
            info = media.get(e.annot_stream) if isinstance(media.get(e.annot_stream), Mapping) else {}
            dur = _num(info.get("duration_s"))
            m = (s == float(e.annot_tag if e.annot_tag is not None else -2)) & (p[:, 1] >= 0)
            if dur is None or not bool(m.any()):
                self.annot_dt.append(math.inf)  # no timed token of the stream: nothing near the annotated frame
            else:
                tau = p[m, 1].astype(np.float64) * dur
                self.annot_dt.append(float(np.abs(tau - e.annot_second).min()))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class _ScanCtx:
    """Fixed per-split inputs of ``_scan_shard``."""

    fx: _Findings | _Recorder
    N: int
    D: int
    dtype: np.dtype
    pos_mode: str
    describe: dict
    kinds: dict[str, str]
    stats: _Stats
    verify_sha: bool
    pos_blobs: dict[int, tuple[str, str]]
    unchecked: dict[str, int] = field(default_factory=dict)  # stream -> members whose checkable rule needed a null fact


class _Recorder:
    """A shard scan's findings, recorded in ``add`` order and replayed into the split's ``_Findings`` (caps,
    waivers) in shard order by :func:`_merge_scan`."""

    def __init__(self) -> None:
        self.findings: list[tuple[str, str, str | None]] = []

    def add(self, code: str, message: str, where: str | None = None) -> None:
        self.findings.append((code, message, where))


@dataclass(frozen=True)
class _ScanStatic:
    """The split-level, read-only inputs of one shard scan (shared by every scan of the split; picklable)."""

    N: int
    D: int
    dtype: np.dtype
    pos_mode: str
    describe: dict
    kinds: dict[str, str]
    verify_sha: bool
    sampled: frozenset[int]
    elig: _EligSpec


@dataclass
class _ShardScan:
    """What one shard's scan produced: the recorded findings, the deferred ``_Stats`` partial, the constant-pos blob
    shas and the per-stream unchecked-rule counts."""

    index: int
    findings: list[tuple[str, str, str | None]]
    stats: _Stats
    pos_blob: tuple[str, str] | None
    unchecked: dict[str, int]


def _scan_shard_isolated(sh: _Shard, static: _ScanStatic) -> _ShardScan:
    """``_scan_shard`` on private state: a pure function of the shard's files and ``static``, safe to run for several
    shards at once (threads; inputs and result are picklable). :func:`_merge_scan` folds the results into the split's
    findings and stats in shard order."""
    rec = _Recorder()
    part = _Stats(static.N, static.D, static.sampled, static.elig, deferred=True)
    blobs: dict[int, tuple[str, str]] = {}
    unchecked: dict[str, int] = {}
    ctx = _ScanCtx(fx=rec, N=static.N, D=static.D, dtype=static.dtype, pos_mode=static.pos_mode, describe=static.describe,
                   kinds=static.kinds, stats=part, verify_sha=static.verify_sha, pos_blobs=blobs, unchecked=unchecked)
    _scan_shard(sh, ctx)
    return _ShardScan(index=sh.index, findings=rec.findings, stats=part, pos_blob=blobs.get(sh.index), unchecked=unchecked)


def _merge_scan(scan: _ShardScan, fx: _Findings, stats: _Stats, pos_blobs: dict[int, tuple[str, str]], unchecked: dict[str, int]) -> None:
    """Replay one shard's findings into ``fx`` (in their order: caps and waivers apply as in one pass) and fold its
    stats, pos blob and unchecked counts into the split's."""
    for code, message, where in scan.findings:
        fx.add(code, message, where)
    stats.merge(scan.stats)
    if scan.pos_blob is not None:
        pos_blobs[scan.index] = scan.pos_blob
    for sname, k in scan.unchecked.items():
        unchecked[sname] = unchecked.get(sname, 0) + k


def validate_workers(shards: list[_Shard] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Threads scanning shards concurrently: ``YAVIH_VALIDATE_WORKERS`` (``1`` = one shard after the other), else
    ``min(n_shards, CPUs, RAM budget // largest shard tar)`` (a shard scan holds its tar's features in memory; CPUs as
    ``yavih.resources`` derives them: SLURM, affinity, cgroup). The report never depends on it."""
    from yavih import resources

    e = os.environ if env is None else env
    raw = e.get(VALIDATE_WORKERS_ENV)
    if raw is not None and str(raw).strip() != "":
        try:
            n = int(str(raw).strip())
        except ValueError as err:
            raise ValueError(f"{VALIDATE_WORKERS_ENV}={raw!r} is not an integer") from err
        if n < 1:
            raise ValueError(f"{VALIDATE_WORKERS_ENV}={raw!r} must be >= 1")
        return n
    shards = shards or []
    n = min(max(1, len(shards)), resources.available_cpus(env))
    largest = max((int((s.sidecar.get("tar") or {}).get("bytes") or 0) for s in shards), default=0)
    if largest > 0:
        n = min(n, max(1, resources.ram_budget_bytes(env) // largest))
    return max(1, n)


def _scan_shards(shards: list[_Shard], static: _ScanStatic, workers: int, fx: _Findings, stats: _Stats,
                 pos_blobs: dict[int, tuple[str, str]], unchecked: dict[str, int]) -> None:
    """Scan ``shards`` (``workers`` at a time) and merge each result in shard order; at most ``2 × workers`` results
    wait for their merge. ``workers == 1`` scans inline. Both paths run the same isolate-then-merge steps, so the
    report is identical whatever ``workers`` is."""
    todo = [sh for sh in shards if sh.tar_path is not None]
    if workers <= 1 or len(todo) <= 1:
        for sh in todo:
            _merge_scan(_scan_shard_isolated(sh, static), fx, stats, pos_blobs, unchecked)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="yavih-validate") as pool:
        ahead: collections.deque = collections.deque()
        nxt = 0
        try:
            while nxt < len(todo) or ahead:
                while nxt < len(todo) and len(ahead) < 2 * workers:
                    ahead.append(pool.submit(_scan_shard_isolated, todo[nxt], static))
                    nxt += 1
                _merge_scan(ahead.popleft().result(), fx, stats, pos_blobs, unchecked)
        except BaseException:
            for fut in ahead:
                fut.cancel()
            raise


def _scan_shard(sh: _Shard, ctx: _ScanCtx) -> None:
    """One streaming pass over a shard's tar: ``E_TAR_SHA``, ``E_MEMBERS``, ``E_ROWS``, the per-member
    ``E_TIMES_OUT_OF_MEDIA`` / ``E_RULE_MISMATCH``, ``E_FINITE``, ``E_POS_SHAPE/RANGE``, canary pairs and the
    sampled stats."""
    fx, N, D, stats = ctx.fx, ctx.N, ctx.D, ctx.stats
    n_streams = len(ctx.describe.get("streams") or [])
    sc = sh.sidecar
    where = sh.where
    tar_path = sh.tar_path
    assert tar_path is not None
    tar_block = sc.get("tar") or {}
    try:
        size = tar_path.stat().st_size
    except OSError as e:
        fx.add("E_SHARD_SET", f"tar unreadable: {e}", where)
        return
    if isinstance(tar_block.get("bytes"), int) and tar_block["bytes"] != size:
        # a truncated/grown tar is a malformed member set; E_TAR_SHA is reserved for the opt-in sha check
        fx.add("E_MEMBERS", f"tar is {size} B, sidecar says {tar_block['bytes']} B (truncated or rewritten tar)", where)
    if ctx.verify_sha:
        got = _sha256_file(tar_path)
        if got != tar_block.get("sha256"):
            fx.add("E_TAR_SHA", f"tar sha256 {got[:12]}… ≠ sidecar {str(tar_block.get('sha256'))[:12]}…", where)

    lo, hi = sh.lo, sh.hi
    sidecar_members = {int(m["row"]): m for m in sc["members"] if isinstance(m, dict) and "row" in m}
    missing_rows = {int(m["row"]): m for m in sc["missing"] if isinstance(m, dict) and "row" in m}
    if len(sidecar_members) != len(sc["members"]):
        fx.add("E_MEMBERS", "sidecar members list has duplicate or malformed entries", where)

    try:
        tf = tarfile.open(tar_path, "r:")
    except (tarfile.TarError, OSError) as e:
        fx.add("E_MEMBERS", f"tar unreadable: {e}", where)
        return
    with tf:
        try:
            infos = tf.getmembers()
        except (tarfile.TarError, OSError, EOFError) as e:
            fx.add("E_MEMBERS", f"tar truncated or malformed: {e}", where)
            return

        seen_json: dict[int, dict] = {}
        seen_feat: dict[int, tuple[int, int]] = {}  # row -> (offset_data, header_len)
        seen_pos: dict[int, np.ndarray | None] = {}
        canary_feats: dict[int, np.ndarray] = {}
        canary_json: dict[int, dict] = {}
        canary_pos: dict[int, np.ndarray] = {}
        orig_feats: dict[int, np.ndarray] = {}
        written_twice: set[int] = set()
        regular_rows_in_order: list[int] = []
        const_pos: np.ndarray | None = None
        saw_constant_pos = False
        checked: set[int] = set()

        def member_checks(row: int, pos: np.ndarray | None, mwhere: str) -> None:
            if row in checked:
                return
            checked.add(row)
            j = seen_json.get(row, {})
            check_member(j, pos, describe=ctx.describe, kinds=ctx.kinds, where=mwhere, fx=fx, unchecked=ctx.unchecked)
            if pos is not None:
                stats.row_eligibility(row, j, pos.astype(np.float32, copy=False))

        for idx, ti in enumerate(infos):
            name = ti.name
            if name == CONSTANT_POS_MEMBER:
                saw_constant_pos = True
                if idx != 0:
                    fx.add("E_POS_SHAPE", f"{CONSTANT_POS_MEMBER} is member #{idx}, must be first", where)
                if ctx.pos_mode != "constant":
                    fx.add("E_POS_SHAPE", f"{CONSTANT_POS_MEMBER} present but pos_mode is {ctx.pos_mode!r}", where)
                data = tf.extractfile(ti).read()  # type: ignore[union-attr]
                arr, _hl = _read_npy(data, f"{where}:{name}", fx)
                if arr is not None:
                    if _check_pos(arr, N, n_streams, f"{where}:{name}", fx):
                        const_pos = arr
                        stats.pos_summary(arr.astype(np.float32, copy=False))
                    stats.bytes += arr.nbytes
                    ctx.pos_blobs[sh.index] = (hashlib.sha256(data).hexdigest(), hashlib.sha256(arr.tobytes()).hexdigest())
                continue
            try:
                row, ext, is_canary = parse_member_name(name)
            except FormatError:
                fx.add("E_MEMBERS", f"unexpected member {name!r}", where)
                continue
            if not ti.isfile():
                fx.add("E_MEMBERS", f"member {name!r} is not a regular file", where)
                continue
            data = tf.extractfile(ti).read()  # type: ignore[union-attr]
            mwhere = f"{where}:{name}"

            if ext == "json":
                try:
                    j = json.loads(data.decode("utf-8"))
                except ValueError as e:
                    fx.add("E_MEMBERS", f"json member unparsable: {e}", mwhere)
                    j = {}
                if not isinstance(j, dict):
                    fx.add("E_MEMBERS", "json member is not an object", mwhere)
                    j = {}
                if j.get("row") != row:
                    fx.add("E_MEMBERS", f".json.row {j.get('row')!r} ≠ member name row {row}", mwhere)
                if "clip_id" not in j:
                    fx.add("E_MEMBERS", ".json lacks 'clip_id'", mwhere)
                if is_canary:
                    canary_json[row] = j
                    continue
                if row in seen_json:
                    written_twice.add(row)
                seen_json[row] = j
                regular_rows_in_order.append(row)
                if not (lo <= row < hi):
                    fx.add("E_ROWS", f"row {row} outside shard range [{lo}, {hi})", mwhere)
                sm = sidecar_members.get(row)
                if sm is not None and sm.get("clip_id") != j.get("clip_id"):
                    fx.add("E_MEMBERS", f"clip_id {j.get('clip_id')!r} ≠ sidecar members {sm.get('clip_id')!r}", mwhere)
                if ctx.pos_mode == "none":
                    member_checks(row, None, mwhere)
                elif ctx.pos_mode == "constant" and const_pos is not None:
                    member_checks(row, const_pos, mwhere)
                continue

            arr, header_len = _read_npy(data, mwhere, fx)
            if ext == "feat.npy":
                if is_canary:
                    if arr is not None:
                        canary_feats[row] = arr
                    continue
                if row in seen_feat:
                    written_twice.add(row)
                seen_feat[row] = (int(ti.offset_data), header_len)
                if arr is None:
                    continue
                if tuple(arr.shape) != (N, D):
                    fx.add("E_MEMBERS", f"feature shape {tuple(arr.shape)} ≠ (N, D) = ({N}, {D})", mwhere)
                    continue
                if arr.dtype != ctx.dtype:
                    fx.add("E_MEMBERS", f"feature dtype {arr.dtype} ≠ declared {ctx.dtype}", mwhere)
                    continue
                if not bool(np.isfinite(arr).all()):
                    fx.add("E_FINITE", f"{int((~np.isfinite(arr)).sum())} non-finite values", mwhere)
                sm = sidecar_members.get(row)
                if sm is not None:
                    if int(sm.get("offset_data", -1)) != int(ti.offset_data) or int(sm.get("header_len", -1)) != header_len:
                        fx.add("E_MEMBERS", f"sidecar offset_data/header_len ({sm.get('offset_data')}, {sm.get('header_len')}) ≠ tar ({ti.offset_data}, {header_len})", mwhere)
                    # nbytes = array bytes; a writer that counts the .npy header too is accepted.
                    if int(sm.get("nbytes", -1)) not in (arr.nbytes, arr.nbytes + header_len):
                        fx.add("E_MEMBERS", f"sidecar nbytes {sm.get('nbytes')} ≠ {arr.nbytes} (array bytes)", mwhere)
                j = seen_json.get(row, {})
                stats.feat(row, str(j.get("clip_id", sm.get("clip_id") if sm else "")), j.get("media_key", sm.get("media_key") if sm else None), arr)
                orig_feats[row] = arr
                continue

            # pos.npy
            if is_canary:
                if arr is not None:
                    canary_pos[row] = arr
                continue
            if ctx.pos_mode != "per_sample":
                fx.add("E_POS_SHAPE", f"per-row pos member present but pos_mode is {ctx.pos_mode!r}", mwhere)
            ok = arr is not None and _check_pos(arr, N, n_streams, mwhere, fx)
            if arr is not None:
                stats.bytes += arr.nbytes
            seen_pos[row] = arr if ok else None
            member_checks(row, arr if ok else None, mwhere)

        # rows whose checks never ran (per_sample without pos, constant without _pos.npy): temporal checks still apply
        for row in seen_json:
            if row not in checked:
                member_checks(row, None, f"{where}:{row:09d}.json")

        # member set vs sidecar
        if regular_rows_in_order != sorted(regular_rows_in_order):
            fx.add("E_MEMBERS", "members are not in ascending row order", where)
        feat_rows = set(seen_feat)
        json_rows = set(seen_json)
        if feat_rows != json_rows:
            fx.add("E_MEMBERS", f"rows with .feat.npy but no .json: {sorted(feat_rows - json_rows)[:10]}; .json without .feat.npy: {sorted(json_rows - feat_rows)[:10]}", where)
        if set(sidecar_members) != feat_rows:
            only_tar = sorted(feat_rows - set(sidecar_members))[:10]
            only_sc = sorted(set(sidecar_members) - feat_rows)[:10]
            fx.add("E_MEMBERS", f"member set ≠ sidecar members (only in tar: {only_tar}, only in sidecar: {only_sc})", where)
        if ctx.pos_mode == "per_sample" and set(seen_pos) != feat_rows:
            fx.add("E_POS_SHAPE", f"pos missing for rows {sorted(feat_rows - set(seen_pos))[:10]} (pos_mode per_sample)", where)
        if ctx.pos_mode == "constant" and not saw_constant_pos and infos:
            fx.add("E_POS_SHAPE", f"{CONSTANT_POS_MEMBER} missing (pos_mode constant)", where)
        rows_block = sc.get("rows") or {}
        if isinstance(rows_block.get("n_written"), int) and rows_block["n_written"] != len(feat_rows):
            fx.add("E_MEMBERS", f"sidecar rows.n_written {rows_block['n_written']} ≠ {len(feat_rows)} members", where)
        if isinstance(rows_block.get("n_missing"), int) and rows_block["n_missing"] != len(missing_rows):
            fx.add("E_MEMBERS", f"sidecar rows.n_missing {rows_block['n_missing']} ≠ {len(missing_rows)} missing entries", where)

        # rows: each of [lo, hi) is written xor missing
        if written_twice:
            fx.add("E_ROWS", f"rows written twice: {sorted(written_twice)[:10]}", where)
        both = feat_rows & set(missing_rows)
        if both:
            fx.add("E_ROWS", f"rows both written and listed missing: {sorted(both)[:10]}", where)
        unaccounted = [r for r in range(lo, hi) if r not in feat_rows and r not in missing_rows]
        if unaccounted:
            fx.add("E_ROWS", f"{len(unaccounted)} rows neither written nor in missing: {unaccounted[:10]}", where)
        stray_missing = [r for r in missing_rows if not (lo <= r < hi)]
        if stray_missing:
            fx.add("E_ROWS", f"missing entries outside shard range: {sorted(stray_missing)[:10]}", where)

        # canaries: re-encodes of rows of the same shard in another batch composition
        for crow, cfeat in sorted(canary_feats.items()):
            orig = orig_feats.get(crow)
            if orig is None:
                fx.add("E_MEMBERS", f"canary c{crow:09d} has no original member", where)
                continue
            if orig.shape != cfeat.shape:
                fx.add("E_MEMBERS", f"canary c{crow:09d} shape {cfeat.shape} ≠ original", where)
                continue
            stats.canary_pairs.append((crow, rel_l2_stored(cfeat, orig)))
            cj = canary_json.get(crow)
            if cj is None:
                fx.add("E_MEMBERS", f"canary c{crow:09d}.feat.npy has no c{crow:09d}.json", where)
            elif "used" in cj and hashing.canon(cj.get("used")) != hashing.canon((seen_json.get(crow) or {}).get("used")):
                stats.canary_mismatch.append((crow, "used"))
            if crow in canary_pos and seen_pos.get(crow) is not None and not np.array_equal(canary_pos[crow], seen_pos[crow]):
                stats.canary_mismatch.append((crow, "pos"))
        for crow in sorted(set(canary_json) - set(canary_feats)):
            fx.add("E_MEMBERS", f"canary c{crow:09d}.json has no c{crow:09d}.feat.npy", where)


def _check_pos(arr: np.ndarray, N: int, n_streams: int, where: str, fx: _Findings) -> bool:
    """``E_POS_SHAPE`` / ``E_POS_RANGE`` on one pos array; True when usable for the spatial checks."""
    ok = True
    if arr.dtype != np.float32:
        fx.add("E_POS_SHAPE", f"pos dtype {arr.dtype} ≠ float32", where)
    if arr.ndim != 2 or arr.shape != (N, 4):
        fx.add("E_POS_SHAPE", f"pos shape {tuple(arr.shape)} ≠ (N, 4) = ({N}, 4)", where)
        return False
    if not bool(np.isfinite(arr).all()):
        # structural: a NaN coordinate or stream tag cannot be read as "absent" or placed anywhere
        fx.add("E_POS_SHAPE", "pos has non-finite values", where)
        return False
    a = arr.astype(np.float64)
    # the stream column names a tag of describe()["streams"] (or -1): structural, the readouts index streams by it
    s = a[:, 0]
    tagged = s != -1.0
    if bool((s[tagged] != np.floor(s[tagged])).any()):
        fx.add("E_POS_SHAPE", "pos stream column has non-integer values", where)
        ok = False
    elif bool(((s[tagged] < 0) | (s[tagged] >= n_streams)).any()):
        fx.add("E_POS_SHAPE", f"pos stream index outside [0, len(describe.streams) = {n_streams}) (or -1)", where)
        ok = False
    # the coordinates t, y, x: media fractions in [0, 1] or -1 (waivable: the readouts carry them as they are)
    c = a[:, 1:]
    bad = ~(((c >= 0.0) & (c <= 1.0)) | (c == -1.0))
    if bool(bad.any()):
        fx.add("E_POS_RANGE", f"{int(bad.sum())} pos coordinates (t, y, x) outside [0, 1] ∪ {{-1}} (media coordinates)", where)
        ok = False
    return ok


# ----------------------------------------------------------------------------- eligibility


def _elig_spec(view: _BenchView | None, tag_names: list[str]) -> _EligSpec:
    """The measurement inputs of ``N_ELIGIBILITY`` from the benchmark's output contract."""
    e = _EligSpec(tag_names=list(tag_names))
    if view is None:
        return e
    rp, req, proto = view.readout_params, view.requires, view.protocol
    vs = rp.get("visual_stream")
    if vs in tag_names:
        e.visual_tag = tag_names.index(vs)
    elif vs is not None:
        e.visual_tag = -2  # declared visual stream the bridge does not tag: no token can count
    g = req.get("grid")
    if isinstance(g, Mapping) and isinstance(g.get("h"), int) and isinstance(g.get("w"), int):
        e.grid_hw = (int(g["h"]), int(g["w"]))
    tr = view.target_region()
    if tr is not None:
        e.region = (tr[0], tr[1], tr[2], tr[3])
    af = req.get("annotated_frame")
    if isinstance(af, Mapping) and _num(proto.get(af.get("fact"))) is not None:
        e.annot_stream = str(af.get("stream"))
        e.annot_tag = tag_names.index(e.annot_stream) if e.annot_stream in tag_names else -2
        e.annot_second = float(proto[af["fact"]])
        e.annot_tol = float(af.get("tol_s") or 0.0)
    if req.get("t_bins") is not None:  # every readout that requires t_bins declares it (Readout.requires_t_bins)
        T = proto.get("segments_scored", rp.get("T", proto.get("num_segments")))
        if isinstance(T, int) and T > 0:
            off = int(proto.get("segment_offset", 0) or 0)
            e.seq_seconds = frozenset(range(off, off + T))
            e.segment_seconds = float(proto.get("segment_seconds", 1.0) or 1.0)
    return e


def _median_min(values: list[float] | list[int], as_int: bool = False) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    med, mn = float(np.median(arr)), float(arr.min())
    if as_int:
        return {"median": med if med != int(med) else int(med), "min": int(mn)}
    return {"median": round(med, 6), "min": round(mn, 6)}


def _eligibility(
    *,
    view: _BenchView | None,
    why_no_bench: str | None,
    N: int,
    pos_mode: str,
    tag_names: list[str],
    stats: _Stats,
    espec: _EligSpec,
) -> dict:
    """``N_ELIGIBILITY``: ``output_contract.requires`` evaluated on the features (medians over the
    sample, mins recorded). The ``_meta.json`` eligibility block; decided here and nowhere else."""
    tagged = sorted(tag_names[i] for i in stats.pos_streams if 0 <= i < len(tag_names))
    block: dict[str, Any] = {
        "readout": view.readout if view is not None else None,
        "eligible": None,
        "missing": [],
        "reason": None,
        "target_region": list(espec.region) if (view is not None and view.target_region() is not None) else None,
        "grid_cell_coverage": _median_min(stats.cell_cov) if espec.grid_hw is not None else None,
        "t_bins": _median_min(stats.t_bins, as_int=True) if espec.seq_seconds is not None else None,
        "stream_tags": tagged,
    }
    if espec.annot_second is not None:
        mm = _median_min(stats.annot_dt)
        if mm is not None:
            mm = {k: (round(float(v), 4) if math.isfinite(float(v)) else None) for k, v in mm.items()}
        block["annotated_frame_dt"] = mm
    if view is None:
        block["reason"] = f"undecided: {why_no_bench}"
        return block
    missing: list[str] = []
    reasons: list[str] = []
    readout = view.readout or "?"
    req, rp = view.requires, view.readout_params
    if _dense(readout) and N == 1:
        missing.append("tokens>1")
        reasons.append(f"N == 1 (pooled features) cannot serve the dense readout {readout!r}")
    for axis in tuple(req.get("pos") or ()):
        if pos_mode == "none" or axis not in stats.pos_axes:
            missing.append(f"pos.{axis}")
    required_streams = list(req.get("streams") or ())
    for key in ("visual_stream", "conditioning_stream"):
        if rp.get(key) and rp[key] not in required_streams:
            required_streams.append(rp[key])
    for sname in required_streams:
        if sname not in tagged:
            missing.append(f"stream:{sname}")
    g = req.get("grid")
    if isinstance(g, Mapping) and g.get("min_cell_coverage") is not None:
        cov = block["grid_cell_coverage"]
        need = float(g["min_cell_coverage"])
        if cov is None or cov["median"] < need:
            missing.append("grid_cell_coverage")
            frame = "the target region" if block["target_region"] is not None else "the full frame"
            got = "no positioned tokens" if cov is None else f"median {cov['median']:.2f}"
            reasons.append(f"grid coverage {got} < {need:g} over {frame}")
    tb = req.get("t_bins")
    if tb is not None:
        got_tb = block["t_bins"]
        if got_tb is None or got_tb["median"] < int(tb):
            missing.append("t_bins")
            reasons.append(f"scored seconds with a timed token: median {None if got_tb is None else got_tb['median']} < {int(tb)}")
    af = req.get("annotated_frame")
    if isinstance(af, Mapping) and espec.annot_second is not None:
        dt = block.get("annotated_frame_dt")
        tol = float(espec.annot_tol or 0.0)
        if dt is None or dt.get("median") is None or float(dt["median"]) > tol:
            missing.append("annotated_frame")
            got = "no timed token" if dt is None or dt.get("median") is None else f"median {float(dt['median']):.2f} s"
            reasons.append(f"no {espec.annot_stream!r} token within {tol:g} s of the annotated second {espec.annot_second:g} s ({got} away): "
                           "the target is drawn on that frame")
    seen: set[str] = set()
    missing = [m for m in missing if not (m in seen or seen.add(m))]
    block["missing"] = missing
    block["eligible"] = not missing
    if missing:
        rest = [m for m in missing if m not in ("tokens>1", "grid_cell_coverage", "t_bins", "annotated_frame")]
        if rest:
            reasons.append("missing: " + ", ".join(rest))
        block["reason"] = "; ".join(reasons)
    return block


# ----------------------------------------------------------------------------- not applicable


def _report_not_applicable(split_dir: Path, split: str, fx: _Findings, view: _BenchView | None) -> ValidationReport:
    try:
        with open(split_dir / NOT_APPLICABLE_FILE, encoding="utf-8") as fh:
            na = json.load(fh)
    except (OSError, ValueError) as e:
        fx.add("E_FORMAT", f"{NOT_APPLICABLE_FILE} unreadable: {e}", NOT_APPLICABLE_FILE)
        na = {}
    reason = str(na.get("reason", "not applicable"))
    nominal = view.n_rows(split) if view else None
    coverage = {"nominal": nominal, "present": 0, "missing": nominal, "ids_sha256": None, "not_applicable": True, "reason": reason}
    readout = view.readout if view else None
    elig = {"readout": readout, "eligible": False, "missing": [], "reason": f"not_applicable: {reason}", "target_region": None,
            "grid_cell_coverage": None, "t_bins": None, "stream_tags": []}
    fx.add("N_ELIGIBILITY", eligibility_line(elig), NOT_APPLICABLE_FILE)
    return ValidationReport(
        ok=fx.ok, errors=fx.errors, warnings=fx.warnings, notes=fx.notes, split=split, coverage=coverage, N=0, D=0, dtype="",
        pos_mode="none", bytes=0, backend_hint="none", eligibility=elig, missing_shards=[],
        shards={"n": 0, "done": 0, "in_progress": 0, "missing": 0, "not_applicable": True},
    )


# ----------------------------------------------------------------------------- cheap re-run


def _cheap_recheck(split_dir: Path, meta: dict, shards: list[_Shard], view: _BenchView | None, fx: _Findings) -> ValidationReport | None:
    """Re-run: a finalized directory is re-checked against ``_meta.json`` (sidecars →
    fingerprint, tar sizes) and the manifest/protocol hashes; the previous verdict is reused when
    nothing moved. Returns ``None`` when a full pass is needed."""
    from yavih.io.finalize import meta_fingerprint

    if not isinstance(meta.get("validation"), dict) or not meta["validation"].get("ok"):
        return None
    if meta.get("format_version") != FEATURE_FORMAT_VERSION:
        return None
    if view is not None and view.readout is not None and (meta.get("eligibility") or {}).get("readout") != view.readout:
        return None  # eligibility was decided for another readout; re-derive it
    sidecars = [s.sidecar for s in shards]
    try:
        fp = meta_fingerprint(sidecars, pos_sha256=(meta.get("features") or {}).get("pos_sha256"), n_present=(meta.get("coverage") or {}).get("present"))
    except (KeyError, TypeError, ValueError):
        return None
    if fp != meta.get("fingerprint"):
        return None
    by_file = {s["file"]: s for s in meta.get("shards", []) if isinstance(s, dict)}
    for sh in shards:
        if sh.tar_path is None:
            return None
        ms = by_file.get(sh.tar_path.name)
        if ms is None:
            return None
        try:
            if sh.tar_path.stat().st_size != int(ms.get("bytes", -1)) or ms.get("sha256") != (sh.sidecar.get("tar") or {}).get("sha256"):
                return None
        except OSError:
            return None
    old = meta["validation"]
    present_codes = {f.code for f in fx.warnings + fx.notes}
    for f in [*old.get("warnings", []), *old.get("notes", [])]:
        # a code not in the table is dropped; the level is the table's
        level = CODES.get(f.get("code")) if isinstance(f, dict) else None
        if level in ("warning", "note") and f.get("code") not in present_codes:
            (fx.warnings if level == "warning" else fx.notes).append(Finding.from_json(f))
    feats = meta.get("features") or {}
    n_shards = len(shards)
    coverage = dict(meta.get("coverage") or {})
    if isinstance(meta.get("sample"), list):  # a capped run's range (see validate_features)
        coverage["sample"] = list(meta["sample"])
    return ValidationReport(
        ok=fx.ok, errors=fx.errors, warnings=fx.warnings, notes=fx.notes, split=str(meta.get("split")),
        coverage=coverage, N=int(feats.get("N", 0)), D=int(feats.get("D", 0)), dtype=str(feats.get("dtype", "")),
        pos_mode=str(feats.get("pos_mode", "none")), bytes=int(feats.get("bytes", 0)),
        backend_hint="memmap" if any(w.code == "W_SIZE" for w in fx.warnings) else "ram",
        eligibility=dict(meta.get("eligibility") or {}), missing_shards=[],
        shards={"n": n_shards, "done": n_shards, "in_progress": 0, "missing": 0, "reused_meta": True},
        rules_checked=dict(meta.get("rules_checked") or {}), meta_path=str(split_dir / META_FILE),
        describe=dict((meta.get("bridge") or {}).get("describe") or {}),
        waived=[dict(w) for w in meta.get("waived") or [] if isinstance(w, dict)],
    )


# ----------------------------------------------------------------------------- main entry


def validate_features(
    features_dir: str | os.PathLike,
    *,
    bench: Benchmark | None = None,
    verify_sha: bool = False,
    sample_rows: int = 512,
    canary_rel_tol: float | None = None,
    write_meta: bool = True,
    waive: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Validate a SPLIT directory. When ``ok`` and ``write_meta``, writes ``_meta.json`` via
    ``yavih.io.finalize`` including the eligibility block whatever its verdict.

    ``waive`` = an operator's waivers ``{CODE: reason}`` (``"*"`` = every waivable check; ``yavih.waivers``,
    ``features validate --waive``), added to the bridge's ``describe()["waive"]``: a waived error becomes ``W_WAIVED``
    and every waiver in force is recorded in ``_meta.json`` ``waived`` (``source: describe | cli``). An operator's
    waivers recorded in an existing ``_meta.json`` of the same configuration are carried over. A waiver of a
    non-waivable code or without a reason raises ``WaiveError`` (CLI) or is ``E_WAIVE_INVALID`` (describe).

    ``canary_rel_tol=None`` = the bridge's declared ``describe()["batch_tolerance"]["rel_l2"]``, else ``1e-5``;
    never above ``1e-3``. ``bench=None`` loads ``Benchmark.load(sidecar.benchmark.name)``; when that is
    impossible the hash checks are skipped, stream kinds are inferred from the members, and
    ``eligibility.eligible`` is ``null`` (undecided, reason recorded).
    """
    from yavih.io.finalize import build_meta, finalize, read_meta

    cli_waive = check_waive(waive, where="--waive", exit_code=1) if waive else {}
    split_dir = Path(features_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"{split_dir}: not a directory (a split directory <run_dir>/features/<split>/)")
    fx = _Findings()
    split_name = split_dir.name
    now = time.time()

    # ---- inventory
    sidecar_paths: dict[int, Path] = {}
    tar_paths: dict[int, Path] = {}
    tmp_fresh: list[Path] = []
    for child in sorted(split_dir.iterdir()):
        m = _SIDECAR_RE.match(child.name)
        if m:
            sidecar_paths[int(m.group(1))] = child
            continue
        m = _TAR_RE.match(child.name)
        if m:
            tar_paths[int(m.group(1))] = child
            continue
        if child.name.endswith(".tmp"):
            try:
                age = now - child.stat().st_mtime
            except OSError:
                continue
            if age > STALE_TMP_SECONDS:
                fx.add("W_STALE_TMP", f"{child.name} is {age / 3600:.0f} h old: a killed task's leftover; delete it or rerun its shard", child.name)
            else:
                tmp_fresh.append(child)
    na_path = split_dir / NOT_APPLICABLE_FILE

    # ---- E_FORMAT: sidecars
    shards: list[_Shard] = []
    for idx in sorted(sidecar_paths):
        sc = _read_sidecar(sidecar_paths[idx], fx)
        if sc is None:
            continue
        if sc["shard"]["index"] != idx:
            fx.add("E_FORMAT", f"sidecar shard.index {sc['shard']['index']} ≠ file index {idx}", sidecar_paths[idx].name)
            continue
        shards.append(_Shard(index=idx, sidecar_path=sidecar_paths[idx], sidecar=sc, tar_path=tar_paths.get(idx)))
    sidecars = [s.sidecar for s in shards]
    splits_named = {str(s.get("split")) for s in sidecars}
    if len(splits_named) > 1:
        fx.add("E_DESCRIBE_MISMATCH", f"sidecars name several splits: {sorted(splits_named)}")
    elif splits_named:
        split_name = splits_named.pop()

    # ---- benchmark view (needed by NA too for nominal rows)
    view, why_no_bench = _load_bench_view(bench, sidecars) if (bench is not None or sidecars) else (None, "no sidecars")

    # ---- E_NA_AND_SHARDS
    if na_path.exists():
        if sidecar_paths or tar_paths:
            fx.add("E_NA_AND_SHARDS", f"{NOT_APPLICABLE_FILE} present together with {len(sidecar_paths)} sidecars / {len(tar_paths)} tars", NOT_APPLICABLE_FILE)
        else:
            return _report_not_applicable(split_dir, split_name, fx, view)

    # ---- E_SHARD_SET
    missing_shards: list[str] = []
    ns = sorted({s.sidecar["shard"]["n"] for s in shards})
    n_total: int | None = ns[0] if len(ns) == 1 else None
    if len(ns) > 1:
        fx.add("E_SHARD_SET", f"sidecars disagree on n: {ns}")
    for sh in shards:
        if sh.tar_path is None:
            fx.add("E_SHARD_SET", "sidecar without tar", sh.where)
    for idx, tp in sorted(tar_paths.items()):
        if idx not in sidecar_paths:
            fx.add("E_SHARD_SET", "tar without sidecar (shard not finished; its unit is still pending)", tp.name)
    present_idx = {s.index for s in shards if s.tar_path is not None}
    if n_total is not None:
        out_of_range = sorted(i for i in present_idx if i >= n_total)
        if out_of_range:
            fx.add("E_SHARD_SET", f"shard indices ≥ n={n_total}: {out_of_range}")
        missing_idx = [i for i in range(n_total) if i not in present_idx]
        if missing_idx:
            missing_shards = [f"--shard {i}/{n_total}" for i in missing_idx]
            fx.add("E_SHARD_SET", f"{len(missing_idx)} of {n_total} shards missing: {' '.join(missing_shards[:8])}{' …' if len(missing_shards) > 8 else ''}")
    elif not shards:
        fx.add("E_SHARD_SET", f"no sidecars in {split_dir} (nothing extracted yet, or not a split directory)")
    shard_table = {"n": n_total, "done": len(present_idx), "in_progress": len(tmp_fresh), "missing": len(missing_shards) if n_total is not None else None}
    complete = n_total is not None and not missing_shards and all(s.tar_path is not None for s in shards) and len(shards) == n_total

    # ---- hash checks (cheap, before tiling so a wrong-benchmark set is named first). The benchmark side of a feature
    # set is its INPUT hash: a scoring edit (metric pin, probe.py, readout version) never fails features. A recorded
    # input hash the folder changelog lists under `supersedes_input` stands for the current one (identity_value).
    nominal: int | None = view.n_rows(split_name) if view else None
    # A capped run (`extract --rows lo:hi`, `plan build --sample N`): the split's _layout.json says `sample: [lo, hi]`;
    # its one shard tiles [lo, hi) and the nominal row count is hi - lo, not the split's.
    from yavih.io.layout import layout_sample, read_layout

    sample_range = layout_sample(read_layout(split_dir))
    if sample_range is not None:
        nominal = sample_range[1] - sample_range[0]
    if view is not None and view.manifest_hash is not None:
        from yavih.versions import writer_stamp

        for sh in shards:
            b = sh.sidecar.get("benchmark") or {}
            if b.get("manifest_hash") != view.manifest_hash:
                fx.add("E_HASH_MANIFEST", f"sidecar manifest_hash {str(b.get('manifest_hash'))[:12]}… ≠ benchmark {view.manifest_hash[:12]}…", sh.where)
            if view.input_hash is not None and identity_value(sh.sidecar, "benchmark.input_hash") != view.input_hash:
                writer = sh.sidecar.get("yavih") or {"version": sh.sidecar.get("yavih_version")}
                fx.add("E_HASH_INPUT", "sidecar " + describe_input_mismatch(
                    b, want_input_hash=view.input_hash, want_components=view.input_components, writer=writer,
                    reader_version=__version__, reader_commit=writer_stamp()["commit"]), sh.where)
    elif shards:
        ref = shards[0].sidecar.get("benchmark") or {}
        ref_ih = identity_value(shards[0].sidecar, "benchmark.input_hash")
        for sh in shards[1:]:
            b = sh.sidecar.get("benchmark") or {}
            if b.get("manifest_hash") != ref.get("manifest_hash"):
                fx.add("E_HASH_MANIFEST", "manifest_hash differs across sidecars", sh.where)
            ih = identity_value(sh.sidecar, "benchmark.input_hash")
            if ih != ref_ih:
                fx.add("E_HASH_INPUT", "input_hash differs across sidecars", sh.where)

    # ---- E_RANGE_TILING (only meaningful on a complete, single-n set)
    if complete and sample_range is not None:
        for sh in shards:
            if (sh.lo, sh.hi) != sample_range:
                fx.add("E_RANGE_TILING", f"shard.rows [{sh.lo}, {sh.hi}) is not the sampled range [{sample_range[0]}, {sample_range[1]}) "
                       "of this split's _layout.json", sh.where)
    elif complete:
        expect_lo = 0
        bad = False
        for sh in sorted(shards, key=lambda s: s.index):
            lo, hi = sh.lo, sh.hi
            bs = sh.sidecar["shard"].get("batch_size")
            if lo != expect_lo or hi < lo:
                fx.add("E_RANGE_TILING", f"shard.rows [{lo}, {hi}) does not continue from {expect_lo}", sh.where)
                bad = True
            if isinstance(bs, int) and bs > 0 and lo % bs != 0:
                fx.add("E_RANGE_TILING", f"lo={lo} is not a multiple of batch_size={bs}", sh.where)
                bad = True
            expect_lo = max(hi, expect_lo)
        if not bad and nominal is not None and expect_lo != nominal:
            fx.add("E_RANGE_TILING", f"shard ranges tile [0, {expect_lo}) but the split has {nominal} rows")
        if nominal is None:
            nominal = expect_lo

    # ---- consensus values (E_DESCRIBE_MISMATCH: describe incl. its declared rules and decoders)
    def _consensus(getter, code: str, label: str, *, name_shards: bool = False) -> Any:
        # An empty shard (hi == lo) never saw a feature, so its writer records null for N, D,
        # pos_sha256; null therefore means "unknown here", not a competing value.
        values: dict[str, Any] = {}
        where: dict[str, list[int]] = {}
        for sh in shards:
            try:
                v = getter(sh.sidecar)
            except (KeyError, TypeError):
                v = None
            if v is None:
                continue
            k = json.dumps(v, sort_keys=True, default=str)
            values.setdefault(k, v)
            where.setdefault(k, []).append(sh.index)
        if len(values) > 1:
            msg = f"{label} differs across sidecars ({len(values)} distinct values)"
            if name_shards:
                # the run directory does not name the configuration; name which shards are whose and how
                # they leave (any extract unit of this split moves another configuration's shards to _stale/)
                groups = "; ".join(f"{str(values[k])[:12]}: shards {where[k]}" for k in sorted(where, key=lambda k: where[k]))
                msg += (f" — {groups}. Re-run `yavih extract` for this split with the bridge you mean: its first unit moves the "
                        "other configuration's shards (and its _meta.json) to _stale/")
            fx.add(code, msg)
        return next(iter(values.values())) if values else None

    # The describe subset: the configuration minus provenance (batch size, weights path, tolerance, decoder builds)
    # must agree; shards decoded by two PyAV builds or run at two batch sizes are one feature set (each sidecar keeps its own
    # describe as provenance; batch invariance is the canaries' job).
    _consensus(lambda s: hashing.describe_subset(s["bridge"]["describe"]), "E_DESCRIBE_MISMATCH", "bridge.describe (incl. declared input rules)")
    describe = next((sh.sidecar["bridge"].get("describe") for sh in sorted(shards, key=lambda x: x.index)
                     if isinstance((sh.sidecar.get("bridge") or {}).get("describe"), dict)), None) or {}
    describe_sha = _consensus(lambda s: s["bridge"].get("describe_sha256"), "E_DESCRIBE_MISMATCH", "bridge.describe_sha256", name_shards=True)
    model = _consensus(lambda s: s["model"], "E_DESCRIBE_MISMATCH", "model", name_shards=True)
    reading = _consensus(lambda s: s["reading"], "E_DESCRIBE_MISMATCH", "reading", name_shards=True)
    _consensus(lambda s: s["shard"]["batch_size"], "E_DESCRIBE_MISMATCH", "shard.batch_size")
    dtype_name = _consensus(lambda s: s["features"]["dtype"], "E_DESCRIBE_MISMATCH", "features.dtype") or "float32"
    pos_mode = _consensus(lambda s: s["features"]["pos_mode"], "E_DESCRIBE_MISMATCH", "features.pos_mode") or "none"
    N = _consensus(lambda s: s["features"]["N"], "E_DESCRIBE_MISMATCH", "features.N")
    D = _consensus(lambda s: s["features"]["D"], "E_DESCRIBE_MISMATCH", "features.D")
    pos_sha_declared = _consensus(lambda s: s["features"].get("pos_sha256"), "E_DESCRIBE_MISMATCH", "features.pos_sha256")
    # Same describe(), different bridge FILE: an edit that leaves describe() alone can still change the numbers.
    _consensus(lambda s: s["bridge"].get("source_sha256"), "W_SOURCE_SHA", "bridge.source_sha256 (shards extracted by different bridge files)")
    if not isinstance(describe, dict):
        describe = {}
    if shards and describe:
        try:
            want_sha = hashing.describe_sha256(describe)
            if describe_sha != want_sha:
                fx.add("E_DESCRIBE_MISMATCH", f"bridge.describe_sha256 {str(describe_sha)[:12]}… ≠ sha256(canon(describe_subset(describe))) "
                       f"{want_sha[:12]}… ({DESCRIBE_SCHEMA})")
        except TypeError as e:
            fx.add("E_DESCRIBE_MISMATCH", f"describe is not canonicalizable: {e}")
        for key, val in (("model", model), ("reading", reading)):
            if val is not None and describe.get(key) != val:
                fx.add("E_DESCRIBE_MISMATCH", f"sidecar {key} {val!r} ≠ describe()[{key!r}] {describe.get(key)!r} (copied from describe())")

    # ---- waivers (yavih.waivers): the bridge's describe()["waive"],
    # then the operator's --waive (and those an earlier validation of this configuration recorded); in force from here on
    meta_path = split_dir / META_FILE
    try:
        declared = describe_waivers(describe)
    except WaiveError as e:
        fx.add("E_WAIVE_INVALID", f"{e} (the bridge's describe(): fix the bridge, re-run bridge check)")
        declared = {}
    carried: dict[str, str] = {}
    if meta_path.exists() and describe_sha is not None:
        try:
            old_meta = read_meta(split_dir)
        except (FormatError, OSError, ValueError):
            old_meta = None
        if isinstance(old_meta, dict) and (old_meta.get("bridge") or {}).get("describe_sha256") == describe_sha:
            carried = {str(w["code"]): str(w["reason"]) for w in old_meta.get("waived") or []
                       if isinstance(w, dict) and w.get("source") == "cli" and w.get("code") and w.get("reason")}
    try:
        carried = check_waive(carried, where="_meta.json waived") if carried else {}
    except WaiveError:
        carried = {}  # a code a newer yavih no longer waives: the operator re-states it
    waivers = Waivers()
    for code, reason in declared.items():
        waivers.entries[code] = (reason, "describe")
    for code, reason in {**carried, **cli_waive}.items():
        waivers.entries.setdefault(code, (reason, "cli"))
    fx.waivers = waivers

    # ---- E_ARGS_UNCONSUMED
    for sh in shards:
        args = (sh.sidecar.get("bridge") or {}).get("args") or {}
        echoed = (sh.sidecar["bridge"].get("describe") or {}).get("args") or {}
        unconsumed = sorted(k for k in args if k not in echoed)
        if unconsumed:
            fx.add("E_ARGS_UNCONSUMED", f"--arg keys not echoed in describe()['args']: {unconsumed}", sh.where)

    if shards and (not isinstance(N, int) or not isinstance(D, int) or N < 1 or D < 1):
        fx.add("E_FORMAT", f"features.N/D must be ints ≥ 1, got {N!r}/{D!r}")
    if not isinstance(N, int) or not isinstance(D, int):
        N, D = int(N or 0), int(D or 0)
    if shards and describe and describe.get("weights_sha256", "") is None:
        fx.add("W_WEIGHTS_UNFINGERPRINTED", "describe()['weights_sha256'] is null: the weights are recorded only by weights_ref "
               f"({describe.get('weights_ref')!r}); nothing is marked, forked or greyed")

    # ---- batch tolerance: the declared one, never above 1e-3
    bt = describe.get("batch_tolerance") if isinstance(describe.get("batch_tolerance"), dict) else None
    declared_tol = _num((bt or {}).get("rel_l2"))
    tolerance = DEFAULT_BATCH_TOLERANCE if declared_tol is None else declared_tol
    if canary_rel_tol is not None:
        tolerance = float(canary_rel_tol)
    tolerance = min(tolerance, MAX_BATCH_TOLERANCE)
    if declared_tol is not None and declared_tol > DEFAULT_BATCH_TOLERANCE:
        fx.add("W_LOOSE_TOLERANCE", f"the bridge declares batch_tolerance.rel_l2 = {declared_tol:g} > {DEFAULT_BATCH_TOLERANCE:g}: {bt.get('reason')!s}")

    # ---- rule opt-out: a custom rule is a recorded fact, never a warning
    kinds: dict[str, str] = dict(view.stream_kinds) if view is not None else {}
    rc = rules_checked(describe, kinds)
    for sname, which in custom_rules(describe).items():
        fx.add("N_RULE_CUSTOM", custom_rule_message(sname, which))

    # ---- side effects during encode
    for sh in shards:
        writes = (sh.sidecar.get("side_effects") or {}).get("writes_outside_run_dir") or []
        if writes:
            fx.add("W_SIDE_EFFECT", f"{len(writes)} write(s) outside the run directory during encode: {list(writes)[:5]}", sh.where)

    # ---- cheap re-run path on a finalized directory
    if complete and fx.ok and not verify_sha and meta_path.exists():
        try:
            meta = read_meta(split_dir)
        except (FormatError, OSError, ValueError):
            meta = None
        if meta is not None:
            report = _cheap_recheck(split_dir, meta, shards, view, fx)
            if report is not None:
                return report

    # ---- the streaming pass
    tag_names: list[str] = [str(s) for s in (describe.get("streams") or [])]
    espec = _elig_spec(view, tag_names)
    present_from_sidecars = sorted({int(m["row"]) for sh in shards for m in sh.sidecar["members"] if isinstance(m, dict) and "row" in m})
    manifest_hash_sc = str((shards[0].sidecar.get("benchmark") or {}).get("manifest_hash")) if shards else ""
    rng = np.random.default_rng(hashing.rng_seed(f"{manifest_hash_sc}:{split_name}:dup"))
    k = min(int(sample_rows), len(present_from_sidecars))
    sampled = set(int(r) for r in rng.choice(np.asarray(present_from_sidecars, dtype=np.int64), size=k, replace=False)) if k > 0 else set()
    stats = _Stats(max(N, 0), max(D, 0), sampled, espec)
    pos_blobs: dict[int, tuple[str, str]] = {}
    unchecked_rows: dict[str, int] = {}
    if N >= 1 and D >= 1:
        static = _ScanStatic(N=N, D=D, dtype=np.dtype(dtype_name), pos_mode=pos_mode, describe=describe, kinds=kinds,
                             verify_sha=verify_sha, sampled=frozenset(sampled), elig=espec)
        _scan_shards(shards, static, validate_workers(shards), fx, stats, pos_blobs, unchecked_rows)
    # a declared checkable rule that could not be evaluated on some rows (a fact it needs is null there)
    for sname, k_rows in sorted(unchecked_rows.items()):
        if sname in rc:
            rc[sname]["temporal_unchecked_rows"] = int(k_rows)
        rule = (rc.get(sname) or {}).get("temporal")
        fx.add("W_RULE_UNCHECKED", f"{sname}: the declared {rule!r} rule could not be evaluated on {k_rows} row(s) (a media fact it needs is null "
               "there; the folder's input_contract.info does not guarantee it); only the out-of-media and position checks applied on those rows")

    # ---- rows written twice ACROSS shards
    if stats.present_rows:
        vals, cnts = np.unique(np.asarray(stats.present_rows, dtype=np.int64), return_counts=True)
        dup_rows = vals[cnts > 1]
        if dup_rows.size:
            fx.add("E_ROWS", f"rows written in more than one shard: {dup_rows[:10].tolist()}")

    # ---- constant pos identity across shards
    pos_sha256: str | None = None
    if pos_mode == "constant" and pos_blobs:
        member_shas = {v[0] for v in pos_blobs.values()}
        data_shas = {v[1] for v in pos_blobs.values()}
        if len(member_shas) > 1:
            fx.add("E_POS_SHAPE", f"{CONSTANT_POS_MEMBER} differs across shards ({len(member_shas)} distinct)")
        pos_sha256 = next(iter(member_shas))
        if pos_sha_declared is not None and pos_sha_declared not in member_shas | data_shas:
            fx.add("E_POS_SHAPE", f"features.pos_sha256 {str(pos_sha_declared)[:12]}… matches neither the {CONSTANT_POS_MEMBER} member nor its data bytes")
        elif pos_sha_declared in data_shas:
            pos_sha256 = str(pos_sha_declared)

    # ---- canaries → E_BATCH_INVARIANCE
    canary_max = max((r for _, r in stats.canary_pairs), default=0.0)
    if stats.canary_pairs:
        worst = max(stats.canary_pairs, key=lambda p: p[1])
        if canary_max > tolerance:
            fx.add("E_BATCH_INVARIANCE", f"canary rel_l2 {canary_max:.3e} > {tolerance:.1e} (row {worst[0]}): the bridge's features depend on batch "
                   "composition (batch statistics, padding to the batch maximum, cross-sample state)")
        elif canary_max > 0:
            fx.add("N_CANARY_DRIFT", f"canary rel_l2 {canary_max:.3e} ≤ {tolerance:.1e}: batch-composition drift within tolerance; recorded in _meta.json canary.max_rel_l2, no action needed")
    if stats.canary_mismatch:
        what = sorted({w for _, w in stats.canary_mismatch})
        fx.add("E_BATCH_INVARIANCE", f"{len(stats.canary_mismatch)} canary(ies) report a different {' / '.join(what)} than their original "
               f"(rows {[r for r, _ in stats.canary_mismatch][:5]}): what the bridge consumes must not depend on the batch")

    # ---- dead rows (token positions constant across the sample)
    dead_rows: list[int] = []
    if stats.n_sampled_seen >= 2 and N >= 1:
        n = float(stats.n_sampled_seen)
        with np.errstate(invalid="ignore"):  # non-finite members (already E_FINITE) yield NaN, never "dead"
            var = np.maximum(stats.sumsq / n - (stats.sum / n) ** 2, 0.0)  # [N, D]
            per_token = np.nan_to_num(var.mean(axis=1), nan=np.inf)
            mean_var = float(np.nanmean(np.where(np.isfinite(per_token), per_token, np.nan))) if np.isfinite(per_token).any() else 0.0
            dead_mask = per_token < DEAD_ROW_REL_VAR * mean_var
        dead_rows = [int(i) for i in np.nonzero(dead_mask)[0]]
        if dead_rows:
            msg = f"{len(dead_rows)} of {N} token positions have variance < 1e-12 · mean across {stats.n_sampled_seen} sampled clips: {dead_rows[:10]}"
            fx.add("E_DEAD_ROWS", msg if fx.waivers is not None and fx.waivers.covers("E_DEAD_ROWS") else
                   msg + " (if intended, e.g. padding or register tokens: describe()['waive'] = {'E_DEAD_ROWS': '<reason>'})")

    # ---- duplicate clips (byte-identical, media_key differs; rows sharing a media_key read identical media)
    if len(stats.dup_hash) > 1:
        by_sha: dict[str, list[tuple[int, str | None]]] = {}
        for row, (sha, mk) in stats.dup_hash.items():
            by_sha.setdefault(sha, []).append((row, mk))
        dup_rows_list: list[int] = []
        for group in by_sha.values():
            if len(group) < 2:
                continue
            for row, mk in group:
                exempt = all(other_mk is not None and other_mk == mk for orow, other_mk in group if orow != row)
                if not exempt:
                    dup_rows_list.append(row)
        frac = len(dup_rows_list) / len(stats.dup_hash)
        if frac > DUPLICATE_MAX_FRACTION:
            msg = (f"{len(dup_rows_list)} of {len(stats.dup_hash)} sampled clips ({frac:.1%}) are byte-identical to a clip with a different "
                   f"media_key: {sorted(dup_rows_list)[:10]}")
            fx.add("E_DUPLICATE_CLIPS", msg if fx.waivers is not None and fx.waivers.covers("E_DUPLICATE_CLIPS") else
                   msg + " (features do not depend on the input; if the model's input is genuinely identical for those media, e.g. "
                   "digitally silent audio: describe()['waive'] = {'E_DUPLICATE_CLIPS': '<reason>'})")

    # ---- eligibility
    elig = _eligibility(view=view, why_no_bench=why_no_bench, N=N, pos_mode=pos_mode, tag_names=tag_names, stats=stats, espec=espec)
    fx.add("N_ELIGIBILITY", eligibility_line(elig))

    # ---- coverage
    present = len(set(stats.present_rows))
    if nominal is None:
        nominal = max((s.hi for s in shards), default=present)
    ids_sha = hashlib.sha256("\n".join(sorted(stats.clip_ids)).encode("utf-8")).hexdigest() if stats.clip_ids else None
    coverage = {"nominal": int(nominal), "present": int(present), "missing": int(max(nominal - present, 0)), "ids_sha256": ids_sha}
    if sample_range is not None:
        coverage["sample"] = list(sample_range)
    if nominal and present < nominal:
        fx.add("W_COVERAGE", f"present {present} / nominal {nominal} = {present / nominal:.1%} (< 100 %)")

    # ---- media fingerprint / media-info census
    if view is not None and shards:
        ds = view.data_status()
        site = {"media_fingerprint": getattr(ds, "media_fingerprint", None), "media_info_sha256": getattr(ds, "media_info_sha256", None)}
        for key, site_v in site.items():
            if site_v is None:
                continue
            for sh in shards:
                sc_v = (sh.sidecar.get("benchmark") or {}).get(key)
                if sc_v is not None and sc_v != site_v:
                    fx.add("W_MEDIA_FP", f"sidecar {key} differs from this site's data_status.json: features were extracted from differently staged media", sh.where)
                    break

    # ---- fp16 range
    if dtype_name == "float16" and stats.n_values:
        zero_frac = stats.n_zero / stats.n_values
        if stats.max_abs > FP16_WARN_ABS or zero_frac > FP16_WARN_ZERO_FRACTION:
            fx.add("W_FP16_RANGE", f"float16 max|x| = {stats.max_abs:.3g}, zero fraction = {zero_frac:.2f} (thresholds {FP16_WARN_ABS:g} / {FP16_WARN_ZERO_FRACTION}); consider float32 or a saner layer")

    # ---- constant clips (same media_key compared once)
    if stats.constant_clips:
        groups: dict[str, tuple[int, str]] = {}
        for row, cid, mk in stats.constant_clips:
            groups.setdefault(mk if mk is not None else f"clip:{cid}", (row, cid))
        sample = [cid for _, cid in list(groups.values())[:5]]
        fx.add("W_CONSTANT_CLIP", f"{len(groups)} clip(s) whose tokens are all identical (or all zero): {sample}")

    # ---- size
    total_bytes = int(stats.bytes)
    budget = _ram_budget_bytes()
    backend_hint = "ram"
    if total_bytes > budget:
        backend_hint = "memmap"
        fx.add("W_SIZE", f"features are {total_bytes / 1024**3:.1f} GiB > RAM budget {budget / 1024**3:.1f} GiB: probe will use memmap backend; request scratch")

    report = ValidationReport(
        ok=fx.ok, errors=fx.errors, warnings=fx.warnings, notes=fx.notes, split=split_name, coverage=coverage, N=int(N), D=int(D),
        dtype=str(dtype_name), pos_mode=str(pos_mode), bytes=total_bytes, backend_hint=backend_hint, eligibility=elig,
        missing_shards=missing_shards, shards=shard_table, rules_checked=rc, describe=dict(describe or {}), waived=waivers.to_json(),
    )

    # ---- _meta.json: written when ok; a stale finalized meta is demoted on failure
    if write_meta and shards:
        validation = {
            "ok": report.ok,
            "errors": [f.to_json() for f in report.errors],
            "warnings": [f.to_json() for f in report.warnings],
            "notes": [f.to_json() for f in report.notes],
            "validated": _utc_now(),
            "yavih_version": __version__,
        }
        if report.ok:
            meta = build_meta(
                split_dir, sidecars, eligibility=elig, validation=validation, nominal=int(nominal),
                canary={"max_rel_l2": float(canary_max), "n": len(stats.canary_pairs), "tolerance": float(tolerance)},
                dead_rows=dead_rows, pos_sha256=pos_sha256, present_clip_ids=stats.clip_ids, bytes_total=total_bytes, rules_checked=rc,
                waived=report.waived,
            )
            if sample_range is not None:  # the probe reads it: a sampled result is never stored, always `partial`
                meta["sample"] = list(sample_range)
            report.meta_path = str(finalize(split_dir, meta))
        elif meta_path.exists():
            try:
                old = read_meta(split_dir)
            except (FormatError, OSError, ValueError):
                old = None
            if old is not None and (old.get("validation") or {}).get("ok", True):
                old["validation"] = validation
                old["eligibility"] = elig
                old["waived"] = report.waived
                finalize(split_dir, old)
                report.meta_path = str(meta_path)
    return report


# ----------------------------------------------------------------------------- shared scanning API (inspect / diff)


@dataclass(frozen=True)
class MemberRecord:
    """One present row as the scan yields it: ``(shard, row, json, feat, pos)``."""

    shard: int
    row: int
    meta: dict
    feat: np.ndarray
    pos: np.ndarray | None


def features_block(sidecars: list[dict]) -> dict:
    """The ``features`` block of the first sidecar that wrote rows: an empty shard (``hi == lo``)
    records null ``N/D/pos_sha256`` and must not be read as the set's description."""
    for sc in sidecars:
        f = sc.get("features") or {}
        if f.get("N") is not None:
            return dict(f)
    return dict((sidecars[0].get("features") or {}) if sidecars else {})


def list_shards(split_dir: str | os.PathLike) -> list[tuple[int, Path, Path]]:
    """``[(index, sidecar_path, tar_path)]`` of the finished shards, ascending."""
    p = Path(split_dir)
    out = []
    for child in sorted(p.iterdir()):
        m = _SIDECAR_RE.match(child.name)
        if m:
            idx = int(m.group(1))
            tar = p / shard_tar_name(idx)
            if tar.exists():
                out.append((idx, p / shard_sidecar_name(idx), tar))
    return out


def iter_members(split_dir: str | os.PathLike, *, rows: set[int] | None = None, with_pos: bool = True) -> Iterator[MemberRecord]:
    """Stream every regular (non-canary) row of a split directory in shard/row order, reading the
    tars directly (member layout). ``rows`` restricts to those manifest rows."""
    for idx, sc_path, tar_path in list_shards(split_dir):
        with open(sc_path, encoding="utf-8") as fh:
            pos_mode = str(json.load(fh)["features"]["pos_mode"])
        # A row is complete once json + feat (+ pos under per_sample) have streamed by; members
        # of one row are adjacent in the tar, so `pending` holds at most one row.
        want = {"json", "feat"} | ({"pos"} if pos_mode == "per_sample" else set())
        with tarfile.open(tar_path, "r:") as tf:
            const_pos: np.ndarray | None = None
            pending: dict[int, dict[str, Any]] = {}
            for ti in tf:
                if ti.name == CONSTANT_POS_MEMBER:
                    const_pos = _npy_from_bytes(tf.extractfile(ti).read())  # type: ignore[union-attr]
                    continue
                try:
                    row, ext, is_canary = parse_member_name(ti.name)
                except FormatError:
                    continue
                if is_canary or (rows is not None and row not in rows):
                    continue
                data = tf.extractfile(ti).read()  # type: ignore[union-attr]
                rec = pending.setdefault(row, {})
                if ext == "json":
                    rec["json"] = json.loads(data.decode("utf-8"))
                elif ext == "feat.npy":
                    rec["feat"] = _npy_from_bytes(data)
                else:
                    rec["pos"] = _npy_from_bytes(data)
                if want <= rec.keys():
                    pos = rec.get("pos", const_pos) if with_pos else None
                    yield MemberRecord(idx, row, rec["json"], rec["feat"], pos)
                    del pending[row]


def _npy_from_bytes(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def host_tag() -> str:
    """``<host>.<pid>`` for staged-copy names (``features stage``)."""
    return f"{socket.gethostname().split('.')[0]}.{os.getpid()}"
