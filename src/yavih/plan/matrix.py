"""``matrix.yaml``: schema, defaults merge, profile checks (``profile.bench == benchmark``,
``profile.args == resolved args``, ``validate_describe(profile.describe)``; ``model``/``reading`` are
taken from the profile, never computed).

stdlib at module scope; ``yaml`` (a core dependency) is imported inside
``load_matrix`` only. Every refusal here is exit 1: the matrix or a profile is
*configuration*, so the agent fixes the file and resubmits.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from yavih import hashing
from yavih.waivers import WaiveError, check_waive, describe_waivers

__all__ = [
    "Matrix",
    "MatrixEntry",
    "MatrixError",
    "ProfileError",
    "load_matrix",
    "load_profile",
    "check_profile",
    "check_invariance",
    "check_describe",
    "MATRIX_SCHEMA_VERSION",
    "PROFILE_KEYS",
    "DESCRIBE_REQUIRED",
    "PRECISION_KEYS",
    "MODEL_RE",
    "READING_RE",
    "check_model_name",
    "check_reading",
]

MATRIX_SCHEMA_VERSION = 1
#: ``profile.json`` keys written by ``bridge check``. Extra keys (``used``, ``media``,
#: ``warnings`` ...) are allowed: the profile *is* the ``bridge check`` output.
PROFILE_KEYS: tuple[str, ...] = (
    "bench", "args", "describe", "describe_sha256", "model", "reading", "N", "D", "pos_mode", "streams",
    "invariance", "rules_checked", "clips_per_s", "decode_workers", "peak_gpu_gb", "yavih_version", "created",
)
#: The model NAME and the reading tag, the only model-side parts of a path or a key.
#: Kept here (not imported from ``yavih.bridge``, which needs numpy) so ``plan`` stays stdlib.
#: The first character is a letter or digit (``yavih.bridge.MODEL_RE``): ``_stale``, ``_report`` and ``.``-names are refused.
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
READING_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
#: ``profile.invariance``: the hard checks ``bridge check`` passed.
INVARIANCE_KEYS: tuple[str, ...] = ("rerun_bit_identical", "batch_rel_l2", "order_rel_l2", "tolerance", "tolerance_reason")
#: The default batch-/order-invariance tolerance and the loosest a bridge may declare.
DEFAULT_TOLERANCE: float = 1e-5
MAX_TOLERANCE: float = 1e-3
#: The required ``describe()`` keys → accepted Python types (``None`` listed where allowed).
DESCRIBE_REQUIRED: dict[str, tuple[type, ...]] = {
    "weights_sha256": (str, type(None)),
    "weights_ref": (str,),
    "dtype": (str,),
    "batch_size": (int,),
    "streams": (list,),
    "pos_mode": (str,),
    "model": (str,),
    "reading": (str,),
    "layer": (str, int),
    "pooling": (str,),
    "views": (int,),
    "input": (dict,),
    "precision": (dict,),
    "args": (dict,),
}
PRECISION_KEYS: tuple[str, ...] = ("autocast", "matmul_precision", "allow_tf32", "deterministic")

#: Top-level matrix keys; ``results`` is the results-store root (``null`` = the usual resolution).
_TOP_KEYS = {"schema_version", "out", "results", "seeds", "shard_minutes", "canaries", "defaults", "bridges"}
_DEFAULT_KEYS = {"launcher", "env", "cpus", "args", "allow_loopback"}
#: ``shards`` (an int or ``{split: int}``) pins the shard count of a split that has no frozen layout yet;
#: ``allow_loopback`` passes ``--allow-loopback`` to the entry's extract units; ``waive``
#: (``{CODE: reason}``, yavih.waivers) passes ``--waive CODE=reason`` to its probe units' validate prologue.
_ENTRY_KEYS = {"script", "benchmark", "args", "profile", "launcher", "env", "cpus", "shards", "allow_loopback", "waive"}
_BENCH_NAME = re.compile(r"^[a-z0-9_]+$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class MatrixError(ValueError):
    """``matrix.yaml`` is malformed (unknown key, wrong type, missing entry) — exit 1."""

    code = "E_MATRIX"
    exit_code = 1


class ProfileError(ValueError):
    """A ``profile.json`` is missing, malformed or disagrees with its matrix entry — exit 1."""

    code = "E_PROFILE"
    exit_code = 1


@dataclass(frozen=True)
class MatrixEntry:
    key: str
    script: str
    benchmark: str
    profile: str
    args: dict[str, Any] = field(default_factory=dict)
    launcher: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    cpus: int = 8
    #: pinned shard count: ``None``, an int for every split, or ``{split: int}`` (a frozen layout wins over it)
    shards: int | dict[str, int] | None = None
    allow_loopback: bool = False
    #: an operator's validation waivers for the entry's features ``{code: reason}`` (expanded, validated)
    waive: dict[str, str] = field(default_factory=dict)

    def pinned_shards(self, split: str) -> int | None:
        if isinstance(self.shards, dict):
            return self.shards.get(split)
        return self.shards


@dataclass(frozen=True)
class Matrix:
    out: str
    seeds: tuple[int, ...]
    shard_minutes: int
    canaries: int
    entries: tuple[MatrixEntry, ...]
    schema_version: int = MATRIX_SCHEMA_VERSION
    #: Directory of the matrix file; relative ``script``/``profile``/``out`` paths resolve against it.
    root: str = "."
    #: ``results``: results-store root for the probe units (``probe run --results``); ``None`` = the
    #: usual resolution (``YAVIH_RESULTS`` / config ``[results] root`` / sibling ``results`` of the runs root).
    results: str | None = None


# ----------------------------------------------------------------------------- helpers


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise MatrixError(msg)


def _env_block(raw: Any, where: str) -> dict[str, str]:
    """Env values must be strings for ``subprocess``; yaml scalars (``OMP_NUM_THREADS: 8``) are
    coerced so a quoting slip does not become an exit-5 inside ``exec``."""
    if raw is None:
        return {}
    _expect(isinstance(raw, dict), f"{where}.env must be a mapping")
    out: dict[str, str] = {}
    for k, v in raw.items():
        _expect(isinstance(k, str) and k, f"{where}.env keys must be non-empty strings")
        if isinstance(v, bool):
            v = "1" if v else "0"
        _expect(isinstance(v, (str, int, float)), f"{where}.env.{k} must be a scalar")
        out[k] = str(v)
    return out


def _args_block(raw: Any, where: str) -> dict[str, Any]:
    if raw is None:
        return {}
    _expect(isinstance(raw, dict), f"{where}.args must be a mapping")
    for k, v in raw.items():
        _expect(isinstance(k, str) and k, f"{where}.args keys must be non-empty strings")
        _expect(v is None or isinstance(v, (str, int, float, bool)), f"{where}.args.{k} must be a scalar")
    return dict(raw)


def _cpus(raw: Any, where: str) -> int:
    _expect(isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1, f"{where}.cpus must be an int >= 1")
    return int(raw)


def _shards(raw: Any, where: str) -> int | dict[str, int] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        for k, v in raw.items():
            _expect(isinstance(k, str) and k and isinstance(v, int) and not isinstance(v, bool) and v >= 1,
                    f"{where}.shards must map split names to ints >= 1")
        return {str(k): int(v) for k, v in raw.items()}
    _expect(isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1, f"{where}.shards must be an int >= 1 or {{split: int}}")
    return int(raw)


def _flag(raw: Any, where: str) -> bool:
    _expect(isinstance(raw, bool), f"{where} must be true or false")
    return bool(raw)


# ----------------------------------------------------------------------------- matrix


def load_matrix(path: str) -> Matrix:
    """Parse + validate; unknown keys are an error (exit 1)."""
    import yaml  # core dependency; lazy so the plan package stays stdlib at import

    if not os.path.isfile(path):
        raise MatrixError(f"matrix file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as e:  # pragma: no cover - message passthrough
            raise MatrixError(f"matrix yaml does not parse: {e}") from e
    _expect(isinstance(raw, dict), "matrix.yaml must be a mapping at the top level")
    unknown = set(raw) - _TOP_KEYS
    _expect(not unknown, f"unknown matrix keys: {sorted(unknown)}")
    _expect(raw.get("schema_version") == MATRIX_SCHEMA_VERSION,
            f"matrix schema_version must be {MATRIX_SCHEMA_VERSION}, got {raw.get('schema_version')!r}")

    out = raw.get("out")
    if out is None:
        from yavih.config import runs_root

        out = str(runs_root())
    _expect(isinstance(out, str) and out, "matrix.out must be a non-empty string")

    results = raw.get("results")
    _expect(results is None or (isinstance(results, str) and results.strip() != ""),
            "matrix.results must be null or a non-empty path string")

    seeds = raw.get("seeds", [0])
    _expect(isinstance(seeds, list) and seeds and all(isinstance(s, int) and not isinstance(s, bool) for s in seeds),
            "matrix.seeds must be a non-empty list of ints")
    _expect(len(set(seeds)) == len(seeds), "matrix.seeds must be distinct")

    shard_minutes = raw.get("shard_minutes", 30)
    _expect(isinstance(shard_minutes, int) and not isinstance(shard_minutes, bool) and shard_minutes >= 1,
            "matrix.shard_minutes must be an int >= 1")
    canaries = raw.get("canaries", 8)
    _expect(isinstance(canaries, int) and not isinstance(canaries, bool) and canaries >= 0,
            "matrix.canaries must be an int >= 0")

    defaults = raw.get("defaults") or {}
    _expect(isinstance(defaults, dict), "matrix.defaults must be a mapping")
    unknown = set(defaults) - _DEFAULT_KEYS
    _expect(not unknown, f"unknown matrix.defaults keys: {sorted(unknown)}")
    d_launcher = defaults.get("launcher")
    _expect(d_launcher is None or (isinstance(d_launcher, str) and d_launcher.strip()),
            "matrix.defaults.launcher must be a non-empty string")
    d_env = _env_block(defaults.get("env"), "defaults")
    d_args = _args_block(defaults.get("args"), "defaults")
    d_cpus = _cpus(defaults.get("cpus", 8), "defaults")
    d_loopback = _flag(defaults.get("allow_loopback", False), "defaults.allow_loopback")

    bridges = raw.get("bridges")
    _expect(isinstance(bridges, dict) and bridges, "matrix.bridges must be a non-empty mapping of entries")

    entries: list[MatrixEntry] = []
    for key, ent in bridges.items():
        where = f"bridges.{key}"
        _expect(isinstance(key, str) and re.match(r"^[A-Za-z0-9_.-]+$", key or ""),
                f"{where}: entry keys must match [A-Za-z0-9_.-]+")
        _expect(isinstance(ent, dict), f"{where} must be a mapping")
        unknown = set(ent) - _ENTRY_KEYS
        _expect(not unknown, f"{where}: unknown keys {sorted(unknown)}")
        for req in ("script", "benchmark", "profile"):
            _expect(isinstance(ent.get(req), str) and ent[req], f"{where}.{req} is required")
        _expect(_BENCH_NAME.match(ent["benchmark"]), f"{where}.benchmark must match [a-z0-9_]+")
        launcher = ent.get("launcher", d_launcher)
        _expect(launcher is None or (isinstance(launcher, str) and launcher.strip()),
                f"{where}.launcher must be a non-empty string")
        env = dict(d_env)
        env.update(_env_block(ent.get("env"), where))
        args = dict(d_args)
        args.update(_args_block(ent.get("args"), where))
        cpus = _cpus(ent["cpus"], where) if "cpus" in ent else d_cpus
        loopback = _flag(ent["allow_loopback"], f"{where}.allow_loopback") if "allow_loopback" in ent else d_loopback
        try:
            waive = check_waive(ent.get("waive"), where=f"{where}.waive", exit_code=1)
        except WaiveError as e:
            raise MatrixError(f"{e} (E_WAIVE_INVALID)") from e
        entries.append(MatrixEntry(key=key, script=ent["script"], benchmark=ent["benchmark"],
                                   profile=ent["profile"], args=args, launcher=launcher, env=env, cpus=cpus,
                                   shards=_shards(ent.get("shards"), where), allow_loopback=loopback, waive=waive))

    return Matrix(out=out, seeds=tuple(seeds), shard_minutes=shard_minutes, canaries=canaries,
                  entries=tuple(entries), root=os.path.dirname(os.path.abspath(path)), results=results)


# ----------------------------------------------------------------------------- profile


def load_profile(path: str) -> dict:
    """Load ``profile.json`` and check ``PROFILE_KEYS``."""
    if not os.path.isfile(path):
        raise ProfileError(f"profile not found: {path} (run `yavih bridge check --out {path}` first)")
    with open(path, encoding="utf-8") as fh:
        try:
            prof = json.load(fh)
        except json.JSONDecodeError as e:
            raise ProfileError(f"profile {path} is not valid JSON: {e}") from e
    if not isinstance(prof, dict):
        raise ProfileError(f"profile {path} must be a JSON object")
    missing = [k for k in PROFILE_KEYS if k not in prof]
    if missing:
        raise ProfileError(f"profile {path} lacks keys {missing}; re-run `yavih bridge check`")
    if not isinstance(prof["describe"], dict):
        raise ProfileError(f"profile {path}: describe must be an object")
    if not isinstance(prof["N"], int) or not isinstance(prof["D"], int) or prof["N"] < 1 or prof["D"] < 1:
        raise ProfileError(f"profile {path}: N and D must be positive ints")
    if prof["pos_mode"] not in ("none", "constant", "per_sample"):
        raise ProfileError(f"profile {path}: pos_mode {prof['pos_mode']!r} unknown")
    if not isinstance(prof["streams"], list):
        raise ProfileError(f"profile {path}: streams must be a list")
    cps = prof["clips_per_s"]
    if not isinstance(cps, (int, float)) or isinstance(cps, bool) or cps <= 0:
        raise ProfileError(f"profile {path}: clips_per_s must be > 0")
    check_model_name(prof["model"], where=f"profile {path}: model")
    check_reading(prof["reading"], where=f"profile {path}: reading")
    prec = prof["describe"].get("precision")
    deterministic = not (isinstance(prec, dict) and prec.get("deterministic") is False)
    try:
        waived = set(describe_waivers(prof["describe"]))
    except WaiveError as e:
        raise ProfileError(f"profile {path}: {e} (E_WAIVE_INVALID); fix the bridge and re-run `yavih bridge check`") from e
    check_invariance(prof["invariance"], where=f"profile {path}", deterministic=deterministic, waived=waived)
    return prof


def check_invariance(inv: Any, *, where: str = "profile", deterministic: bool = True, waived: set[str] | frozenset[str] = frozenset()) -> None:
    """The profile must record PASSED hard checks — rerun bit-identical and the batch- and
    order-invariance ``rel_l2`` within the tolerance (default ``1e-5``, a declared one at most ``1e-3``).
    ``bridge check`` writes no profile on a failed hard check; this refuses a hand-edited one.

    ``deterministic`` is the profile's ``describe.precision.deterministic``: when it is false, a rerun that is not
    bit-identical passes if the recorded ``rerun_rel_l2`` is within the tolerance. ``waived`` = the codes the
    profile's ``describe()["waive"]`` waives: a waived ``E_RERUN_DRIFT`` / ``E_BATCH_INVARIANCE`` lets the
    recorded numbers exceed the tolerance (``bridge check`` recorded ``W_WAIVED``); the record must still be complete."""
    if not isinstance(inv, dict):
        raise ProfileError(f"{where}: invariance must be an object {list(INVARIANCE_KEYS)}; re-run `yavih bridge check`")
    missing = [k for k in INVARIANCE_KEYS if k not in inv]
    if missing:
        raise ProfileError(f"{where}: invariance lacks {missing}; re-run `yavih bridge check`")
    tol = inv["tolerance"]
    if not isinstance(tol, (int, float)) or isinstance(tol, bool) or not (0.0 < float(tol) <= MAX_TOLERANCE):
        raise ProfileError(f"{where}: invariance.tolerance {tol!r} must be in (0, {MAX_TOLERANCE:g}]")
    if inv["rerun_bit_identical"] is not True and "E_RERUN_DRIFT" not in waived:
        rr = inv.get("rerun_rel_l2")
        within = isinstance(rr, (int, float)) and not isinstance(rr, bool) and 0.0 <= float(rr) <= float(tol)
        if deterministic or not within:
            why = ("precision.deterministic is true, so the rerun must be bit-identical" if deterministic
                   else f"invariance.rerun_rel_l2 {rr!r} is not within the tolerance {tol:g}")
            raise ProfileError(f"{where}: invariance.rerun_bit_identical is {inv['rerun_bit_identical']!r} and {why}: the bridge failed the "
                               "rerun hard check (E_RERUN_DRIFT)")
    if float(tol) > DEFAULT_TOLERANCE and not (isinstance(inv["tolerance_reason"], str) and inv["tolerance_reason"].strip()):
        raise ProfileError(f"{where}: a tolerance looser than {DEFAULT_TOLERANCE:g} needs a tolerance_reason")
    for k in ("batch_rel_l2", "order_rel_l2"):
        v = inv[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or float(v) < 0.0:
            raise ProfileError(f"{where}: invariance.{k} = {v!r} is not a measured rel_l2: the bridge failed a hard check or the profile was "
                               "edited (E_BATCH_INVARIANCE); re-run `yavih bridge check`")
        if "E_BATCH_INVARIANCE" not in waived and not float(v) <= float(tol):
            raise ProfileError(f"{where}: invariance.{k} = {v!r} is not within the tolerance {tol:g}: the bridge failed a hard check "
                               "(E_BATCH_INVARIANCE)")


def check_model_name(v: Any, *, where: str = "describe.model") -> str:
    """``model`` is ``[A-Za-z0-9_.-]{1,64}`` starting with a letter or digit (it names a directory; ``_stale``
    and the other ``_``-names are yavih's)."""
    if not isinstance(v, str) or not MODEL_RE.match(v):
        raise ProfileError(f"{where} {v!r} must match [A-Za-z0-9_.-]{{1,64}} and start with a letter or digit")
    return v


def check_reading(v: Any, *, where: str = "describe.reading") -> str:
    """``reading`` is ``[A-Za-z0-9_-]{1,32}`` starting with a letter or digit (``a``, ``v``, ``av`` or another
    declared tag)."""
    if not isinstance(v, str) or not READING_RE.match(v):
        raise ProfileError(f"{where} {v!r} must match [A-Za-z0-9_-]{{1,32}} and start with a letter or digit")
    return v


def check_describe(d: dict) -> None:
    """Structural check of a *recorded* ``describe()``.

    ``yavih.bridge.validate_describe(d, bridge, bench)`` needs a live bridge instance, which only
    exists in the model env; the controller running ``plan`` has the profile only. This checks
    everything that can be checked from the record: required keys and types, a 64-hex
    ``weights_sha256`` or ``null``, a fully declared ``precision`` block (``null`` only for
    ``autocast``, else ``E_PRECISION_UNDECLARED``). Raises ``ProfileError``.
    """
    if not isinstance(d, dict):
        raise ProfileError("describe must be an object")
    missing = [k for k in DESCRIBE_REQUIRED if k not in d]
    if missing:
        raise ProfileError(f"describe lacks required keys {missing}")
    for k, types in DESCRIBE_REQUIRED.items():
        v = d[k]
        if isinstance(v, bool) and bool not in types:
            raise ProfileError(f"describe.{k} has the wrong type (bool)")
        if not isinstance(v, types):
            raise ProfileError(f"describe.{k} has the wrong type {type(v).__name__}")
    check_model_name(d["model"])
    check_reading(d["reading"])
    w = d["weights_sha256"]
    if w is not None and not _HEX64.match(w):
        raise ProfileError("describe.weights_sha256 must be 64 lower-case hex or null")
    if d["dtype"] not in ("float16", "float32"):
        raise ProfileError(f"describe.dtype {d['dtype']!r} must be float16|float32")
    if d["pos_mode"] not in ("none", "constant", "per_sample"):
        raise ProfileError(f"describe.pos_mode {d['pos_mode']!r} unknown")
    if d["views"] != 1:
        raise ProfileError(f"describe.views must be 1 (one deterministic view per clip), got {d['views']!r}")
    if not d["input"]:
        raise ProfileError("describe.input must declare the rules of every stream the bridge reads")
    bt = d.get("batch_tolerance")
    if bt is not None:
        if not (isinstance(bt, dict) and set(bt) == {"rel_l2", "reason"} and isinstance(bt["rel_l2"], (int, float))
                and 0.0 < float(bt["rel_l2"]) <= MAX_TOLERANCE and isinstance(bt["reason"], str) and bt["reason"].strip()):
            raise ProfileError(f"describe.batch_tolerance must be {{rel_l2 <= {MAX_TOLERANCE:g}, reason}}, got {bt!r}")
    if d["batch_size"] < 1:
        raise ProfileError("describe.batch_size must be >= 1")
    try:
        describe_waivers(d)  # describe()["waive"]: codes and reasons
    except WaiveError as e:
        raise ProfileError(f"{e} (E_WAIVE_INVALID)") from e
    if not all(isinstance(s, str) for s in d["streams"]):
        raise ProfileError("describe.streams must be a list of stream names")
    prec = d["precision"]
    lacking = [k for k in PRECISION_KEYS if k not in prec]
    if lacking:
        raise ProfileError(f"describe.precision lacks {lacking}")
    for k in PRECISION_KEYS:
        if k != "autocast" and prec[k] is None:
            raise ProfileError(f"describe.precision.{k} is null: E_PRECISION_UNDECLARED")
    if prec["autocast"] not in (None, "bf16", "fp16"):
        raise ProfileError("describe.precision.autocast must be null|bf16|fp16")
    if prec["matmul_precision"] not in ("highest", "high", "medium"):
        raise ProfileError("describe.precision.matmul_precision must be highest|high|medium")
    if not isinstance(prec["allow_tf32"], bool) or not isinstance(prec["deterministic"], bool):
        raise ProfileError("describe.precision.allow_tf32/deterministic must be bools")
    try:
        hashing.canon(d)
    except TypeError as e:
        raise ProfileError(f"describe is not JSON-primitive: {e}") from e


def _arg_equal(a: Any, b: Any) -> bool:
    # Matrix yaml and bridge-check both parse `--arg k=v` through the bridge's ARGS types, but the
    # matrix may spell `layer: "12"`; compare loosely on the string form after an exact miss.
    if a == b and type(a) is type(b):
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return str(a).lower() == str(b).lower()
    return str(a) == str(b)


def check_profile(profile: dict, entry: MatrixEntry, *, bench_names: tuple[str, ...] = ()) -> None:
    """The matrix refusals (exit 1): ``profile.bench != benchmark`` (an alias of the same benchmark is
    accepted via ``bench_names``), ``profile.args`` differing from the entry's resolved ``args``, a
    ``describe`` failing the schema, or a profile that disagrees with its own ``describe``
    (``describe_sha256``, ``model``, ``reading``)."""
    accepted = {entry.benchmark, *bench_names}
    if profile["bench"] not in accepted:
        raise ProfileError(
            f"entry {entry.key!r}: profile.bench {profile['bench']!r} != benchmark {entry.benchmark!r} "
            f"(re-run `yavih bridge check --bench {entry.benchmark}`)"
        )
    pargs = profile["args"] if isinstance(profile["args"], dict) else {}
    diffs = [k for k, v in entry.args.items() if k not in pargs or not _arg_equal(v, pargs[k])]
    if diffs:
        raise ProfileError(
            f"entry {entry.key!r}: profile.args differ from the entry's args on {sorted(diffs)} "
            f"(profile {pargs}, matrix {entry.args}); re-run `yavih bridge check` with the matrix args"
        )
    check_describe(profile["describe"])
    d = profile["describe"]
    # Consistency, not computation (plan takes model and reading from the profile and never
    # computes them): the profile must agree with its own describe, else bridge check, extract's drift
    # check (--expect-profile, E_CONFIG_DRIFT) and the DONE checks would disagree.
    if profile["describe_sha256"] != hashing.describe_sha256(d):
        raise ProfileError(f"entry {entry.key!r}: profile.describe_sha256 does not match profile.describe; "
                           "re-run `yavih bridge check`")
    for key in ("model", "reading"):
        if profile[key] != d[key]:
            raise ProfileError(f"entry {entry.key!r}: profile.{key} {profile[key]!r} disagrees with describe.{key} "
                               f"{d[key]!r}; re-run `yavih bridge check`")
    # The top-level copies bridge check writes must agree with describe (plan reads describe for bytes_per_row).
    for key in ("dtype", "pos_mode"):
        if key in profile and profile[key] != d[key]:
            raise ProfileError(f"entry {entry.key!r}: profile.{key} {profile[key]!r} disagrees with describe.{key} {d[key]!r}")
    if list(profile["streams"]) != list(d["streams"]):
        raise ProfileError(f"entry {entry.key!r}: profile.streams disagrees with describe.streams")
