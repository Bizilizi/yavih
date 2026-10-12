"""``yavih plan status``: ``{done, pending, running, not_ready, preempted, failed, not_applicable,
by_kind}``; ``preempted`` (exit 4) is resubmittable and never counted as ``failed``.

A unit's status is decided from its outputs first (a matching DONE marker) and from its
``logs/last.json`` otherwise: exit 4 → ``preempted`` (deadline or signal, both resubmittable), exit 3 →
``not_ready``, exit 1/2/5 → ``failed``; no record → ``pending``. A provisional ``running`` record (written by
``exec`` when the unit starts) counts as ``running`` while its process can still be alive and as ``pending``
once it cannot (killed without a final record: SIGKILL, OOM, node loss, a batch shell that died on TERM);
``stale_running`` names those units.

``misconfigured`` (exit 1: a usage / configuration error such as a stale plan's
``E_LAYOUT`` or a bad ``--deadline``) is its own count, so a driver loop stops on it like on ``failed``; ``repeated``
names the units whose last ``REPEAT_THRESHOLD`` attempts ended with the same exit 3 or 4 status (``exec``'s
``history.streak``): resubmitting them again will not help, their message says what to fix.
"""

from __future__ import annotations

import os
import socket
import time

from yavih.plan.units import Unit, check_done, last_record, read_plan

__all__ = ["plan_status", "unit_status", "running_alive", "format_status", "STATUS_KEYS", "RUNNING_GRACE_S",
           "REPEAT_THRESHOLD"]

STATUS_KEYS: tuple[str, ...] = ("done", "pending", "running", "not_ready", "preempted", "misconfigured", "failed",
                                "not_applicable")
#: Consecutive identical exit-3 / exit-4 attempts after which ``plan status`` lists a unit under ``repeated``.
REPEAT_THRESHOLD = 3
#: A ``running`` record from another host (pid not checkable) is believed for twice the unit's planned
#: minutes plus this grace, and never past its recorded deadline plus this grace.
RUNNING_GRACE_S = 1800.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def running_alive(rec: dict, *, now: float | None = None, host: str | None = None) -> bool:
    """Is the process that wrote this ``running`` record plausibly still running the unit?"""
    now = time.time() if now is None else now
    host = socket.gethostname() if host is None else host
    pid = rec.get("pid")
    if rec.get("host") == host and isinstance(pid, int) and pid > 0:
        return _pid_alive(pid)
    started = rec.get("started_epoch")
    if not isinstance(started, (int, float)):
        return False
    deadline = rec.get("deadline")
    if isinstance(deadline, (int, float)) and now > deadline + RUNNING_GRACE_S:
        return False
    minutes = (rec.get("needs") or {}).get("minutes") or 0
    minutes = float(minutes) if isinstance(minutes, (int, float)) else 0.0
    return now - float(started) <= 2.0 * minutes * 60.0 + RUNNING_GRACE_S


def unit_status(unit: Unit) -> tuple[str, int | None]:
    """``(status, last_exit_code)`` for one unit."""
    st, rc, _ = _unit_status(unit)
    return st, rc


def _unit_status(unit: Unit) -> tuple[str, int | None, dict | None]:
    got = check_done(unit)
    rec = last_record(unit)
    rc = rec.get("exit_code") if rec else None
    rc = int(rc) if isinstance(rc, int) else None
    if got is not None:
        return got, rc, rec
    if rec is None:
        return "pending", None, None
    if rec.get("status") == "running":
        return ("running" if running_alive(rec) else "pending"), None, rec
    if rc == 4 or rec.get("status") in ("preempted", "deadline"):
        return "preempted", rc, rec
    if rc == 3 or rec.get("status") == "not_ready":
        return "not_ready", rc, rec
    if rc == 1 or rec.get("status") == "misconfigured":
        return "misconfigured", rc, rec
    if rec.get("status") in ("skipped", "done", "not_applicable") or rc == 0:
        # exec said done but the marker is gone (moved to _stale/ by a re-plan): it must run again.
        return "pending", rc, rec
    return "failed", rc, rec


def _streak(rec: dict | None) -> int:
    h = (rec or {}).get("history")
    return int(h.get("streak") or 0) if isinstance(h, dict) else 0


def plan_status(plan_path: str, *, root: str | None = None) -> dict:
    units = read_plan(plan_path, root=root)
    counts = dict.fromkeys(STATUS_KEYS, 0)
    by_kind: dict[str, dict[str, int]] = {}
    rows: list[dict] = []
    stale: list[str] = []
    repeated: list[str] = []
    for u in units:
        st, rc, rec = _unit_status(u)
        counts[st] += 1
        by_kind.setdefault(u.kind, dict.fromkeys(STATUS_KEYS, 0))[st] += 1
        row = {"id": u.id, "kind": u.kind, "benchmark": u.benchmark, "bridge": u.bridge, "split": u.split,
               "shard": u.shard, "seed": u.seed, "status": st, "exit_code": rc,
               "message": (rec or {}).get("message") if st not in ("done", "not_applicable") else None,
               "log": (rec or {}).get("log") if st not in ("done", "not_applicable") else None,
               "host": (rec or {}).get("host") if st == "running" else None,
               "attempts": int(((rec or {}).get("history") or {}).get("attempts") or 0) if rec else 0}
        if st in ("not_ready", "preempted") and _streak(rec) >= REPEAT_THRESHOLD:
            repeated.append(u.id)
            row["streak"] = _streak(rec)
        if st == "pending" and rec is not None and rec.get("status") == "running":
            stale.append(u.id)
            row["message"] = (f"stale running record (host {rec.get('host')} pid {rec.get('pid')} started "
                              f"{rec.get('started')}): the process ended without a final record")
        rows.append(row)
    return {**counts, "total": len(units), "by_kind": by_kind, "stale_running": stale, "repeated": repeated, "units": rows}


def _last_line(msg: str | None, width: int = 200) -> str:
    lines = [ln.strip() for ln in str(msg or "").splitlines() if ln.strip()]
    line = lines[-1] if lines else ""
    return line if len(line) <= width else line[: width - 1] + "…"


def _unit_label(r: dict) -> str:
    parts = [r["id"], r["kind"], r.get("benchmark") or "-", r.get("bridge") or "-"]
    if r.get("split"):
        parts.append(f"{r['split']}")
    if r.get("shard"):
        parts.append(f"shard {r['shard'][0]}/{r['shard'][1]}")
    if r.get("seed") is not None:
        parts.append(f"seed {r['seed']}")
    return " ".join(str(p) for p in parts)


def format_status(data: dict, *, plan_path: str = "", max_listed: int = 20) -> str:
    """Human summary for ``plan status`` without ``--json``: totals, one line per kind, then the units that
    need attention (failed, not_ready, preempted, running, stale running) with the last line of their
    message. ``--json`` keeps the full per-unit list."""
    keys = [k for k in STATUS_KEYS if k in data]
    head = f"{plan_path + ': ' if plan_path else ''}{data.get('total', 0)} units  " + "  ".join(f"{k} {data[k]}" for k in keys)
    lines = [head]
    for kind, c in (data.get("by_kind") or {}).items():
        lines.append(f"  {kind:<8} " + "  ".join(f"{k} {c.get(k, 0)}" for k in keys if c.get(k, 0)))
    rows = data.get("units") or []
    stale = set(data.get("stale_running") or [])
    repeated = set(data.get("repeated") or [])
    sections = (("misconfigured (exit 1: fix the configuration, resubmitting will not help)", lambda r: r["status"] == "misconfigured"),
                ("failed", lambda r: r["status"] == "failed"),
                (f"repeated (the same exit 3/4 {REPEAT_THRESHOLD}+ times in a row: read the message, resubmitting will not help)",
                 lambda r: r["id"] in repeated),
                ("not_ready", lambda r: r["status"] == "not_ready"),
                ("preempted", lambda r: r["status"] == "preempted"),
                ("running", lambda r: r["status"] == "running"),
                ("stale running (killed without a record; resubmit)", lambda r: r["id"] in stale))
    for title, pred in sections:
        sel = [r for r in rows if pred(r)]
        if not sel:
            continue
        lines.append(f"{title}:")
        for r in sel[:max_listed]:
            extra = f" exit={r['exit_code']}" if r.get("exit_code") is not None else ""
            if r["status"] == "running" and r.get("host"):
                extra += f" host={r['host']}"
            msg = _last_line(r.get("message"))
            lines.append(f"  {_unit_label(r)}{extra}" + (f": {msg}" if msg else ""))
            if r["status"] in ("failed", "misconfigured") and r.get("log"):
                lines.append(f"    log: {r['log']}")
        if len(sel) > max_listed:
            lines.append(f"  … {len(sel) - max_listed} more (plan status --json lists every unit)")
    return "\n".join(lines) + "\n"
