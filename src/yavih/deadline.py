"""The one ``--deadline`` parser shared by ``yavih exec``, ``yavih extract`` and ``yavih eval run``.

Accepted forms (case-insensitive):

- ``none`` / ``off`` → no deadline.
- ``slurm`` → ``SLURM_JOB_END_TIME − margin``. Inside a SLURM job (``SLURM_JOB_ID`` set) whose
  ``SLURM_JOB_END_TIME`` is unset (older Slurm releases) or unparseable this is a configuration error, never
  "now": every unit would be refused as out of time and a driver loop would resubmit them forever. Outside SLURM
  it means no deadline (a workstation run of a SLURM recipe).
- ``+<duration>`` or ``<duration>`` with a unit → now + duration: ``+3600``, ``+90m``, ``2h``, ``1h30m``,
  ``45s``, ``1d`` (``s``/``m``/``h``/``d``; a bare ``+N`` is seconds).
- an epoch in seconds (``1760000000``); a bare number below ``1e9`` (before 2001) is refused as an obvious
  duration typo (``--deadline 3600`` meant ``+3600``).
- an ISO-8601 timestamp (``2026-10-10T18:00:00Z``; no offset = UTC).

``None`` / ``""`` → ``$YAVIH_DEADLINE`` (what ``exec`` hands its child) when ``inherit``, else no deadline.
Anything else raises ``DeadlineError`` (exit 1, ``E_DEADLINE_SPEC``). stdlib only.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import time
from collections.abc import Mapping

__all__ = ["parse_deadline", "parse_duration", "DeadlineError", "DEFAULT_DEADLINE_MARGIN_S", "MIN_EPOCH"]

#: ``slurm`` = ``SLURM_JOB_END_TIME − margin``; larger than the templates' ``--signal=B:TERM@600`` lead so a unit
#: stops on its own clock before SLURM's TERM.
DEFAULT_DEADLINE_MARGIN_S = 900.0
#: A bare number below this is a duration typed without ``+``, not an epoch (2001-09-09).
MIN_EPOCH = 1e9

_DURATION = re.compile(r"^(?:(?P<d>\d+(?:\.\d+)?)d)?(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?(?:(?P<s>\d+(?:\.\d+)?)s?)?$")
_UNIT_S = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}


class DeadlineError(ValueError):
    """A ``--deadline`` that cannot be honoured: a configuration error (exit 1), not a transient state."""

    code = "E_DEADLINE_SPEC"
    exit_code = 1


def parse_duration(text: str) -> float | None:
    """Seconds of ``90``, ``90s``, ``15m``, ``2h``, ``1h30m``, ``1d``; ``None`` when ``text`` is not a duration."""
    t = text.strip().lower()
    if not t:
        return None
    m = _DURATION.match(t)
    if not m or not any(m.group(k) for k in "dhms"):
        return None
    return sum(float(m.group(k)) * _UNIT_S[k] for k in "dhms" if m.group(k))


def parse_deadline(spec: str | float | None, *, env: Mapping[str, str] | None = None,
                   margin_s: float = DEFAULT_DEADLINE_MARGIN_S, now: float | None = None,
                   inherit: bool = True) -> float | None:
    """``--deadline`` → epoch seconds or ``None`` (module docstring). Raises ``DeadlineError``."""
    env = os.environ if env is None else env
    now = time.time() if now is None else now
    if spec is None or (isinstance(spec, str) and spec.strip() == ""):
        if not inherit:
            return None
        raw = env.get("YAVIH_DEADLINE") or None
        if raw is None:
            return None
        try:  # an epoch exec wrote (repr of a float): taken as is
            return float(raw)
        except ValueError:
            spec = raw
    if isinstance(spec, bool):
        raise DeadlineError(f"--deadline must be a string or a number, got {spec!r}")
    if isinstance(spec, (int, float)):
        # a computed epoch (``time.time() + s``, ``$YAVIH_DEADLINE``) is taken as is; the before-2001 typo check is
        # for text a human typed
        v = float(spec)
        if v != v or v in (float("inf"), float("-inf")):
            raise DeadlineError(f"--deadline {spec!r} is not a finite epoch")
        return v
    text = str(spec).strip()
    low = text.lower()
    if low in ("none", "off"):
        return None
    if low == "slurm":
        return _slurm(env, margin_s)
    if low.startswith("+"):
        secs = parse_duration(low[1:])
        if secs is None:
            raise DeadlineError(f"--deadline {text!r}: '+' must be followed by a duration (+3600, +90m, +2h, +1h30m)")
        return now + secs
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None:
        return _epoch(number, text)
    secs = parse_duration(low)
    if secs is not None:
        return now + secs
    iso = _iso(text)
    if iso is not None:
        return iso
    raise DeadlineError(f"--deadline expects none, slurm, +<seconds>, a duration (90m, 2h), an epoch in seconds or an "
                        f"ISO-8601 time, got {text!r}")


def _epoch(value: float, text: str) -> float:
    if value != value or value in (float("inf"), float("-inf")):
        raise DeadlineError(f"--deadline {text!r} is not a finite epoch")
    if value < MIN_EPOCH:
        raise DeadlineError(f"--deadline {text!r} is an epoch before 2001: a duration? write +{text} (seconds from now)")
    return value


def _slurm(env: Mapping[str, str], margin_s: float) -> float | None:
    end = env.get("SLURM_JOB_END_TIME")
    if end:
        try:
            return float(end) - float(margin_s)
        except ValueError:
            raise DeadlineError(f"--deadline slurm: SLURM_JOB_END_TIME={end!r} is not an epoch; pass --deadline "
                                "+<seconds> or an epoch instead") from None
    if env.get("SLURM_JOB_ID"):
        raise DeadlineError(
            "--deadline slurm inside SLURM job "
            f"{env.get('SLURM_JOB_ID')} but SLURM_JOB_END_TIME is not set (older Slurm releases do not export it): "
            "pass --deadline +<seconds> (e.g. the job's --time minus a margin), an epoch, or none"
        )
    return None


def _iso(text: str) -> float | None:
    t = text.strip()
    if t.endswith(("Z", "z")):
        t = t[:-1] + "+00:00"
    try:
        d = _dt.datetime.fromisoformat(t)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    return d.timestamp()
