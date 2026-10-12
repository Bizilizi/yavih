"""Validation waivers: a check that is wrong for a model is waived WITH A REASON, never edited out.

A bridge declares ``describe()["waive"] = {CODE: reason}`` (``{"*": reason}`` = every waivable check); an operator
adds ``yavih features validate <dir> --waive CODE=reason`` (or the matrix entry key ``waive``) for features already
extracted. A waived error becomes the warning ``W_WAIVED`` (``"<CODE>: <message> — waived: <reason>"``), is counted,
and is recorded wherever the features go (``_meta.json`` ``waived``, ``result.json``, the results store) and marked in
``report build`` — never greyed, excluded or penalized: like the ``custom`` rule opt-out it is a legitimate,
recorded decision.

Only checks whose failure leaves the features well-formed and identifiable are waivable (:data:`WAIVABLE`); the
structural and identity checks and the unfilled-template guard are not (:data:`NOT_WAIVABLE` says why). A waiver of
anything else, or without a non-empty reason, is ``E_WAIVE_INVALID``.

``waive`` is a bridge decision: part of ``describe()`` and therefore of ``describe_sha256`` (consistency), never of an
input, protocol or eval hash. CLI / matrix waivers are an operator's decision about existing features and change no
hash.

stdlib only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "W_WAIVED",
    "E_WAIVE_INVALID",
    "ALL",
    "WAIVABLE",
    "NOT_WAIVABLE",
    "WaiveError",
    "Waivers",
    "check_waive",
    "describe_waivers",
    "parse_cli_waivers",
    "waivers_for",
    "waived_codes",
]

W_WAIVED = "W_WAIVED"
E_WAIVE_INVALID = "E_WAIVE_INVALID"
#: ``{"*": reason}`` in ``describe()["waive"]``; ``all=reason`` on the command line.
ALL = "*"
_CLI_ALL = "all"

#: Checks a bridge (or an operator) may waive with a reason, and what a waiver means for each. Every one leaves the
#: feature set well-formed (shards, rows, shapes, finite values, identity) — it relaxes a check of the bridge's
#: behaviour or of the declared rules, which the waiver then records.
WAIVABLE: dict[str, str] = {
    "E_TIMES_OUT_OF_MEDIA": "reported times / sample ranges outside the media (e.g. a model that reads a fixed window past a short clip's end)",
    "E_RULE_MISMATCH": "the reported `used` (or token positions) differ from the declared rule",
    "E_POS_RANGE": "token coordinates t / y / x outside [0, 1] ∪ {-1} (e.g. letterbox padding tokens); the readouts carry them as they "
                   "are and place no such token in a cell (the stream column and cross-shard pos checks are E_POS_SHAPE, not waivable)",
    "E_BATCH_INVARIANCE": "features depend on the batch (validate canaries; bridge check alone and order passes)",
    "E_RERUN_DRIFT": "bridge check: encoding the same batches twice differs (feature drift; a second instance that describes itself "
                     "differently is never waivable)",
    "E_CONSTANT_FEATURES": "bridge check: every smoke clip encodes to the same features",
    "E_DEAD_ROWS": "token positions constant across the sampled clips (e.g. padding or register tokens)",
    "E_DUPLICATE_CLIPS": "byte-identical features for different media (e.g. digitally silent audio the model's input cannot tell apart)",
    "E_ARGS_UNCONSUMED": "a --arg not echoed in describe()['args'] (e.g. a device or cache path that cannot change the features)",
}
#: Checks that are never waivable, with the reason: structural (the feature set itself is broken), identity
#: (the features are not those of this benchmark / configuration), and the guards a waiver would defeat.
NOT_WAIVABLE: dict[str, str] = {
    **{c: "structural: the shards, rows or arrays are malformed and no reader can trust them" for c in (
        "E_FORMAT", "E_SHARD_SET", "E_RANGE_TILING", "E_MEMBERS", "E_ROWS", "E_TAR_SHA", "E_POS_SHAPE", "E_FINITE", "E_NA_AND_SHARDS")},
    **{c: "identity: the features are not this benchmark's input or one consistent configuration" for c in (
        "E_HASH_MANIFEST", "E_HASH_INPUT", "E_DESCRIBE_MISMATCH", "E_CONFIG_DRIFT")},
    "E_TEMPLATE_UNFILLED": "the unfilled-template guard: fill the bridge from the model's own code instead",
    "E_WAIVE_INVALID": "a malformed waiver is fixed, not waived",
}


class WaiveError(ValueError):
    """``E_WAIVE_INVALID``: a waiver of a non-waivable code, of an unknown code, or without a reason. Exit 2 in a
    ``describe()`` (a contract the bridge must fix), 1 on the command line / in a matrix (usage)."""

    code = E_WAIVE_INVALID

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _why_not(code: str) -> str:
    if code in NOT_WAIVABLE:
        return f"{code} is not waivable ({NOT_WAIVABLE[code]})"
    if code[:2] in ("W_", "N_"):
        return f"{code} is a {'warning' if code.startswith('W_') else 'note'}: it never fails anything, so there is nothing to waive"
    return f"{code} is not a waivable check"


def check_waive(m: Any, *, where: str = "describe()['waive']", exit_code: int = 2) -> dict[str, str]:
    """Validate a ``{CODE: reason}`` mapping (``"*"`` = every waivable check) and expand it: ``{code: reason}`` over
    :data:`WAIVABLE`; an explicit code beats ``"*"``. Raises :class:`WaiveError`."""
    if m is None:
        return {}
    if not isinstance(m, Mapping):
        raise WaiveError(f"{where} must be a mapping {{CODE: reason}}, got {type(m).__name__}", exit_code)
    out: dict[str, str] = {}
    star: str | None = None
    for code, reason in m.items():
        if not isinstance(code, str):
            raise WaiveError(f"{where}: key {code!r} must be a code string", exit_code)
        if not isinstance(reason, str) or not reason.strip():
            raise WaiveError(f"{where}[{code!r}] needs a non-empty reason string (why the check is wrong for this model), got {reason!r}",
                             exit_code)
        if code == ALL:
            star = reason.strip()
            continue
        if code not in WAIVABLE:
            raise WaiveError(f"{where}[{code!r}]: {_why_not(code)}; waivable: {', '.join(sorted(WAIVABLE))} or '*'", exit_code)
        out[code] = reason.strip()
    if star is not None:
        out = {**dict.fromkeys(WAIVABLE, star), **out}
    return out


def describe_waivers(d: Mapping[str, Any] | None) -> dict[str, str]:
    """The waivers a ``describe()`` declares: its ``waive`` mapping, validated and expanded by :func:`check_waive`."""
    if not isinstance(d, Mapping) or "waive" not in d:
        return {}
    return check_waive(d.get("waive"))


def parse_cli_waivers(items: Iterable[str] | None, *, where: str = "--waive") -> dict[str, str]:
    """``["CODE=reason", "all=reason"]`` → ``{code: reason}`` (expanded as :func:`check_waive`; exit 1 on error)."""
    raw: dict[str, str] = {}
    for item in items or ():
        code, sep, reason = str(item).partition("=")
        code = code.strip()
        if not sep:
            raise WaiveError(f"{where} {item!r}: give CODE=reason (or all=reason), e.g. --waive E_DUPLICATE_CLIPS='digitally silent audio'", 1)
        raw[ALL if code.lower() == _CLI_ALL else code] = reason
    return check_waive(raw, where=where, exit_code=1)


@dataclass
class Waivers:
    """Waivers in force for one check run: ``{code: (reason, source)}`` (``source`` ∈ ``describe``, ``cli``) and how many
    findings each turned into ``W_WAIVED``."""

    entries: dict[str, tuple[str, str]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def covers(self, code: str) -> bool:
        return code in self.entries

    def reason(self, code: str) -> str:
        return self.entries[code][0]

    def record(self, code: str) -> None:
        self.counts[code] = self.counts.get(code, 0) + 1

    def message(self, code: str, message: str) -> str:
        reason, source = self.entries[code]
        return f"{code}: {message} — waived: {reason}" + (" (--waive)" if source == "cli" else "")

    def to_json(self) -> list[dict[str, Any]]:
        """``[{code, reason, source, count}]`` sorted by code: every waiver in force, ``count`` = findings it waived."""
        return [{"code": c, "reason": r, "source": s, "count": int(self.counts.get(c, 0))} for c, (r, s) in sorted(self.entries.items())]


def waivers_for(describe: Mapping[str, Any] | None = None, cli: Mapping[str, str] | None = None) -> Waivers:
    """The waivers in force: the bridge's (``source: describe``) first; an operator's (``source: cli``) for the codes the
    bridge did not waive."""
    w = Waivers()
    for code, reason in describe_waivers(describe).items():
        w.entries[code] = (reason, "describe")
    for code, reason in (cli or {}).items():
        w.entries.setdefault(code, (reason, "cli"))
    return w


def waived_codes(records: Iterable[Mapping[str, Any]] | None) -> list[str]:
    """The codes of recorded waivers (``_meta.json`` / ``result.json`` ``waived``)."""
    return sorted({str(r.get("code")) for r in (records or []) if isinstance(r, Mapping) and r.get("code")})
