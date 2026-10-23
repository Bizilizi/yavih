"""``yavih plan calibrate runs/``: refine the resource multipliers from ``logs/last.json``,
``probe_fit.json`` (``s_per_epoch``) and ``result.json[timing]``; raise ``minutes`` 1.5× for any unit whose
last attempt was stopped by its deadline while running; write ``<runs root>/_calibration.json``.

Only a real deadline stop counts: ``status: deadline`` with ``wall_s > 0`` and
not refused at start (``refused_at_start``). A start refusal says the job had too little wall left, a preemption
(``status: preempted``, SIGTERM) says nothing about the unit's length, so neither is bumped.

Record shape (consumed by ``yavih.plan.units`` at ``plan build``)::

    {"schema": 1, "updated": utc,
     "multipliers": {"extract_minutes": f, "probe_minutes": f},
     "units": {"<id>": {"minutes": f, "bumped_at": utc, "reason": "exit 4"}},
     "observations": {...}}

Multipliers are the 90th percentile of ``observed wall / planned minutes`` over finished units of a
kind (the plan already carries the 1.5× margin, so a ratio ≤ 1 leaves the plan alone down to a floor
of 0.5). A unit's bump compounds only when a *new* exit-4 attempt is seen since the last bump, so
running ``calibrate`` twice in a row does not double it.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any

__all__ = ["calibrate", "CALIBRATION_FILE", "DEADLINE_BUMP", "MULT_FLOOR", "MULT_CEIL", "CALIBRATION_SCHEMA"]

CALIBRATION_FILE = "_calibration.json"
DEADLINE_BUMP = 1.5
MULT_FLOOR = 0.5
MULT_CEIL = 8.0
CALIBRATION_SCHEMA = 1


def _load(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _quantile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    if not ys:
        return 1.0
    pos = q * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def _clamp(x: float) -> float:
    return max(MULT_FLOOR, min(MULT_CEIL, x))


def _scan(runs_root: str) -> tuple[list[dict], list[dict], list[dict]]:
    """``(last records, probe_fit records, result timings)`` found under ``runs_root``."""
    lasts: list[dict] = []
    fits: list[dict] = []
    results: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(runs_root):
        dirnames[:] = [d for d in dirnames if d != "_stale"]
        base = os.path.basename(dirpath)
        if base == "logs":
            for fn in filenames:
                if fn.startswith("last.") and fn.endswith(".json") and fn != "last.json":
                    rec = _load(os.path.join(dirpath, fn))
                    if rec and isinstance(rec.get("id"), str):
                        lasts.append(rec)
            if "last.json" in filenames:
                rec = _load(os.path.join(dirpath, "last.json"))
                if rec and isinstance(rec.get("id"), str) and not any(r.get("id") == rec["id"] for r in lasts):
                    lasts.append(rec)
        if "probe_fit.json" in filenames:
            rec = _load(os.path.join(dirpath, "probe_fit.json"))
            if rec:
                fits.append(rec)
        if "result.json" in filenames and "DONE" in filenames:
            rec = _load(os.path.join(dirpath, "result.json"))
            if rec and isinstance(rec.get("timing"), dict):
                results.append({"timing": rec["timing"], "readout": (rec.get("benchmark") or {}).get("readout"),
                                "status": rec.get("status")})
    return lasts, fits, results


def _deadline_stop(rec: dict) -> bool:
    """The unit ran and its own deadline stopped it (module docstring)."""
    if rec.get("status") != "deadline" or rec.get("refused_at_start"):
        return False
    wall = rec.get("wall_s")
    return isinstance(wall, (int, float)) and wall > 0


def calibrate(runs_root: str) -> dict:
    """``yavih plan calibrate``; returns the written ``_calibration.json`` record."""
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"runs root not found: {runs_root}")
    path = os.path.join(runs_root, CALIBRATION_FILE)
    prev = _load(path) or {}
    prev_mult = prev.get("multipliers") or {}
    prev_units: dict[str, dict[str, Any]] = dict(prev.get("units") or {})

    lasts, fits, results = _scan(runs_root)

    # Observed-over-planned ratios per kind from finished units that recorded the needs they ran with.
    ratios: dict[str, list[float]] = {"extract": [], "probe": [], "report": []}
    for rec in lasts:
        if rec.get("status") not in ("done", "not_applicable"):
            continue
        needs = rec.get("needs") or {}
        planned = float(needs.get("minutes") or 0) * 60.0
        wall = rec.get("wall_s")
        if planned > 0 and isinstance(wall, (int, float)) and wall > 0:
            ratios.setdefault(str(rec.get("kind") or "extract"), []).append(float(wall) / planned)

    multipliers = {
        "extract_minutes": float(prev_mult.get("extract_minutes", 1.0)),
        "probe_minutes": float(prev_mult.get("probe_minutes", 1.0)),
    }
    if ratios["extract"]:
        multipliers["extract_minutes"] = _clamp(multipliers["extract_minutes"] * _quantile(ratios["extract"], 0.9))
    if ratios["probe"]:
        multipliers["probe_minutes"] = _clamp(multipliers["probe_minutes"] * _quantile(ratios["probe"], 0.9))

    # Per-unit bump: 1.5× for a unit whose LAST attempt exited 4, once per new attempt.
    units: dict[str, dict[str, Any]] = {}
    bumped: list[str] = []
    for rec in lasts:
        uid = rec["id"]
        old = prev_units.get(uid) or {}
        factor = float(old.get("minutes", 1.0)) if isinstance(old.get("minutes", 1.0), (int, float)) else 1.0
        if _deadline_stop(rec):
            ended = rec.get("ended") or ""
            if ended != old.get("bumped_at"):
                factor *= DEADLINE_BUMP
                bumped.append(uid)
                units[uid] = {"minutes": factor, "bumped_at": ended, "reason": "deadline"}
            else:
                units[uid] = dict(old)
        elif rec.get("status") in ("done", "not_applicable", "skipped"):
            continue  # finished: its bump history is no longer needed
        elif factor != 1.0:
            units[uid] = dict(old)
    for uid, old in prev_units.items():  # keep bumps for units without any new record
        if uid not in units and not any(r.get("id") == uid for r in lasts):
            units[uid] = old

    s_per_epoch: dict[str, list[float]] = {}
    for f in fits:
        v = f.get("s_per_epoch")
        if isinstance(v, (int, float)) and v > 0:
            s_per_epoch.setdefault(str(f.get("readout") or "all"), []).append(float(v))
    walls: dict[str, list[float]] = {}
    for r in results:
        w = r["timing"].get("wall_s")
        if isinstance(w, (int, float)) and w > 0:
            walls.setdefault(str(r.get("readout") or "all"), []).append(float(w))

    record = {
        "schema": CALIBRATION_SCHEMA,
        "updated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "multipliers": multipliers,
        "units": units,
        "observations": {
            "n_last": len(lasts),
            "n_ratio": {k: len(v) for k, v in ratios.items()},
            "ratio_p90": {k: round(_quantile(v, 0.9), 3) for k, v in ratios.items() if v},
            "bumped": sorted(bumped),
            "probe_s_per_epoch_median": {k: round(_quantile(v, 0.5), 3) for k, v in s_per_epoch.items()},
            "probe_wall_s_median": {k: round(_quantile(v, 0.5), 1) for k, v in walls.items()},
        },
    }
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, sort_keys=True, indent=1)
    os.replace(tmp, path)
    return record
