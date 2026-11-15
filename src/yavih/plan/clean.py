"""The feature cleanup policy: drop a run directory's feature tars once every probe that reads them is done.

Features are the bulk of a run (gigabytes per benchmark and reading) and most are read by one probe and never again,
so by default (``[features] keep = false``) yavih drops them as soon as they have served:

- ``yavih exec``, after a probe unit ends ``done`` / ``skipped`` / ``not_applicable``, calls :func:`after_probe`: when
  every probe unit of the plan with that run directory is done (and not stale) and no other unit of it is running,
  the run directory's feature tars and its ``features/<split>/_stale/`` directories are deleted;
- ``yavih plan clean <plan.jsonl>`` (:func:`clean_plan`) applies the same rule to every run directory of a plan: runs
  probed with ``eval run`` by hand, or finished before the policy applied (``--dry-run`` lists what it would free).

What stays: the sidecars (each shard's DONE marker, with the tar's sha256), ``_layout.json``, ``_meta.json``,
``NOT_APPLICABLE.json``, the logs, and ``probe/`` (results, DONE markers). Each cleaned split gets a record,
``features/<split>/_cleaned.json`` (:data:`yavih.io.format.CLEANED_FILE`), written before anything is deleted. With it,
an extract unit whose sidecar matches but whose tar is gone is ``cleaned`` (``yavih.plan.units.check_done``): done
while every probe that reads it is done, and planned again by ``plan build --pending`` once one of them is not (a new
seed, a bumped protocol): the features come back by re-extraction, never silently.

The policy is the configuration's ``[features] keep`` (``yavih.config.Settings.features_keep``): ``false``, ``true``
(keep everything) or a list of ``<benchmark>/<model>/<reading>`` globs naming the run directories to keep. It is read
from the configuration discovered from the unit's ``cwd`` (the matrix directory), so the project's
``yavih-bench/yavih.toml`` decides wherever ``exec`` runs. ``yavih features keep`` writes it there
(:func:`set_features_keep`), with the reason as a comment: the override a person asks for is recorded in the project.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from yavih.config import Settings
    from yavih.plan.units import Unit

__all__ = [
    "after_probe",
    "clean_plan",
    "drop_features",
    "run_dir_finished",
    "policy_of",
    "set_features_keep",
    "KeepEditError",
]


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


def _du(path: Path) -> tuple[int, int]:
    """``(bytes, files)`` of the regular files under ``path`` (symlinks not followed; a missing path is ``(0, 0)``)."""
    total = files = 0
    stack = [str(path)]
    while stack:
        here = stack.pop()
        try:
            with os.scandir(here) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            total += e.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total, files


# ----------------------------------------------------------------------------- policy


def policy_of(settings: Settings) -> dict[str, Any]:
    """``{keep, source}``: the ``[features] keep`` value in force (``False``, ``True`` or the globs) and where it is set."""
    keep = settings.features_keep()
    return {"keep": keep if isinstance(keep, bool) else list(keep), "source": settings.features_keep_source()}


def _policy_text(policy: dict[str, Any]) -> str:
    keep = policy["keep"]
    value = json.dumps(keep) if isinstance(keep, list) else str(keep).lower()
    return f"[features] keep = {value} ({policy['source']})"


def _settings_for(cwd: str | None, cache: dict[str, Settings]) -> Settings:
    from yavih import config

    key = cwd or ""
    if key not in cache:
        cache[key] = config.load(cwd=cwd) if cwd else config.load()
    return cache[key]


# ----------------------------------------------------------------------------- finished?


def _live_records(run_dir: Path, *, exclude: set[str]) -> list[str]:
    """Ids of the units of ``run_dir`` (probes in ``logs/``, extract shards in ``features/<split>/logs/``) whose provisional
    ``running`` record says they may still be running (``plan status``'s rule), other than ``exclude``: the probes whose
    DONE marker already matches are finishing, not reading (two seeds that end together must not wait on each other)."""
    from yavih.plan.status import running_alive

    logs = [run_dir / "logs"]
    features = run_dir / "features"
    if features.is_dir():
        logs += [d / "logs" for d in sorted(features.iterdir()) if d.is_dir()]
    live: list[str] = []
    for d in logs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("last.*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(rec, dict) or rec.get("status") != "running" or rec.get("id") in exclude:
                continue
            if running_alive(rec):
                live.append(str(rec.get("id") or p.name))
    return live


def run_dir_finished(probes: list[Unit], run_dir: str | os.PathLike) -> tuple[bool, str]:
    """Whether the features of ``run_dir`` have served: every probe unit of it (``probes``) has a matching, non-stale DONE
    marker, and no other unit of the run directory (an extract shard being written again, a probe of another plan) may
    still be running. ``(False, why)`` otherwise."""
    from yavih.plan.units import check_done, is_stale

    if not probes:
        return False, "no probe unit of the plan reads these features"
    waiting = [p for p in probes if check_done(p) is None or is_stale(p)]
    if waiting:
        return False, f"{len(waiting)} of {len(probes)} probe units not done yet"
    live = _live_records(Path(run_dir), exclude={p.id for p in probes})
    if live:
        return False, f"{len(live)} unit(s) of this run directory may still be running ({', '.join(live[:3])})"
    return True, "every probe done"


# ----------------------------------------------------------------------------- dropping


def _split_dirs(features: Path) -> list[Path]:
    if not features.is_dir():
        return []
    return sorted(d for d in features.iterdir() if d.is_dir() and not d.name.startswith("_"))


def drop_features(run_dir: str | os.PathLike, *, dry_run: bool = False, policy: dict[str, Any] | None = None,
                  by: str = "") -> dict[str, Any]:
    """Delete the shard tars (those with a sidecar) and the ``_stale/`` directory of every split of ``run_dir/features``,
    after writing each split's ``_cleaned.json`` record. Sidecars, ``_layout.json``, ``_meta.json``,
    ``NOT_APPLICABLE.json``, logs and ``*.tmp`` files stay. Concurrent calls are safe (a file another call removed
    first is skipped). Returns ``{status: dropped | nothing | would_drop, run_dir, splits: {split: {tars, bytes,
    stale_bytes}}, files, bytes}``."""
    from yavih.io.format import CLEANED_FILE, SHARD_RE, STALE_DIR, shard_tar_name
    from yavih.io.sidecar import write_json_atomic
    from yavih.versions import writer_stamp

    run_dir = Path(run_dir)
    splits: dict[str, dict[str, int]] = {}
    files = total = 0
    for split_dir in _split_dirs(run_dir / "features"):
        tars: list[tuple[Path, int]] = []
        for side in sorted(split_dir.glob("shard-*.json")):
            m = SHARD_RE.match(side.name)
            if m is None:
                continue
            tar = split_dir / shard_tar_name(int(m.group("index")))
            try:
                if tar.is_file():
                    tars.append((tar, tar.stat().st_size))
            except OSError:
                continue
        stale = split_dir / STALE_DIR
        stale_bytes, stale_files = _du(stale)
        if not tars and not stale.exists():
            continue
        tar_bytes = sum(b for _, b in tars)
        splits[split_dir.name] = {"tars": len(tars), "bytes": tar_bytes, "stale_bytes": stale_bytes}
        files += len(tars) + stale_files
        total += tar_bytes + stale_bytes
        if dry_run:
            continue
        # the record first: a cleanup killed half way leaves tars next to a record (done, cleaned next time), never a
        # sidecar without its tar and without a record (which validation reports as a broken shard set)
        rec_path = split_dir / CLEANED_FILE
        try:
            prev = json.loads(rec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
        history = prev.get("cleanups") if isinstance(prev, dict) and isinstance(prev.get("cleanups"), list) else []
        entry = {"cleaned": _now_utc(), "by": by or None, "tars": len(tars), "bytes": tar_bytes, "stale_bytes": stale_bytes,
                 "policy": policy, "writer": writer_stamp()}
        write_json_atomic(rec_path, {"cleanups": [*history, entry]})
        for tar, _ in tars:
            try:
                tar.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(stale, ignore_errors=True)
    status = ("would_drop" if dry_run else "dropped") if splits else "nothing"
    return {"status": status, "run_dir": str(run_dir), "splits": splits, "files": files, "bytes": total}


# ----------------------------------------------------------------------------- exec hook and plan clean


def _probes_of(run_dir: str, units: Iterable[Unit]) -> list[Unit]:
    from yavih.plan.units import resolve_path

    return [u for u in units if u.kind == "probe" and os.path.normpath(resolve_path(u, u.out)) == run_dir]


def after_probe(unit: Unit, units_by_id: dict[str, Unit]) -> dict[str, Any] | None:
    """``yavih exec``'s step after a finished probe unit. ``None`` while the run directory is not finished (another probe
    of it is not done, or a unit of it may be running); else ``{status: dropped | nothing | kept, run_dir, policy, ...}``
    with a ``message`` (logged by ``exec``) when something was dropped."""
    from yavih.plan.units import resolve_path

    run_dir = os.path.normpath(resolve_path(unit, unit.out))
    if not os.path.isdir(os.path.join(run_dir, "features")):
        return None  # a plan-time not_applicable probe: nothing was extracted
    probes = _probes_of(run_dir, units_by_id.values())
    if unit.id not in {p.id for p in probes}:
        probes.append(unit)
    ok, _why = run_dir_finished(probes, run_dir)
    if not ok:
        return None
    settings = _settings_for(unit.cwd, {})
    policy = policy_of(settings)
    if settings.keeps_features(unit.benchmark, unit.model, unit.reading):
        return {"status": "kept", "run_dir": run_dir, "policy": policy}
    rec = drop_features(run_dir, policy=policy, by=f"exec {unit.id}")
    rec["policy"] = policy
    if rec["status"] == "dropped":
        rec["message"] = (f"cleanup: every probe of {run_dir} is done; dropped {rec['files']} feature files "
                          f"({_gb(rec['bytes'])}) under {_policy_text(policy)}; sidecars and results stay. "
                          f"To keep features: yavih features keep <benchmark>/<model>/<reading> --reason '...'")
    return rec


def clean_plan(plan_path: str, *, root: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    """``yavih plan clean``: :func:`drop_features` on every run directory of the plan whose probes are all done, under
    the ``[features] keep`` policy. Returns ``{run_dirs: [{run_dir, benchmark, model, reading, status, why, files,
    bytes}], dropped, kept, not_finished, files, bytes, dry_run}``; ``status`` ∈ ``dropped``, ``would_drop``,
    ``nothing``, ``kept``, ``not_finished``."""
    from yavih.plan.units import read_plan, resolve_path

    units = read_plan(plan_path, root=root)
    by_dir: dict[str, list[Unit]] = {}
    for u in units:
        if u.kind == "probe":
            by_dir.setdefault(os.path.normpath(resolve_path(u, u.out)), []).append(u)
    cache: dict[str, Settings] = {}
    rows: list[dict[str, Any]] = []
    for run_dir, probes in sorted(by_dir.items()):
        first = probes[0]
        row: dict[str, Any] = {"run_dir": run_dir, "benchmark": first.benchmark, "model": first.model, "reading": first.reading,
                               "files": 0, "bytes": 0}
        if not os.path.isdir(os.path.join(run_dir, "features")):
            rows.append({**row, "status": "nothing", "why": "no features directory"})
            continue
        ok, why = run_dir_finished(probes, run_dir)
        if not ok:
            rows.append({**row, "status": "not_finished", "why": why})
            continue
        settings = _settings_for(first.cwd, cache)
        policy = policy_of(settings)
        if settings.keeps_features(first.benchmark, first.model, first.reading):
            rows.append({**row, "status": "kept", "why": _policy_text(policy)})
            continue
        rec = drop_features(run_dir, dry_run=dry_run, policy=policy, by="plan clean")
        rows.append({**row, "status": rec["status"], "why": why, "files": rec["files"], "bytes": rec["bytes"],
                     "splits": rec["splits"]})
    count = {k: sum(r["status"] == k for r in rows) for k in ("dropped", "would_drop", "kept", "not_finished", "nothing")}
    return {"run_dirs": rows, **count, "files": sum(r["files"] for r in rows), "bytes": sum(r["bytes"] for r in rows),
            "dry_run": dry_run}


def format_clean(data: dict[str, Any]) -> str:
    """Text mode of ``plan clean``."""
    verb = "would free" if data["dry_run"] else "freed"
    lines = [f"{len(data['run_dirs'])} run directories: dropped {data['dropped'] + data['would_drop']}  kept {data['kept']}  "
             f"not finished {data['not_finished']}  nothing to drop {data['nothing']}; {verb} {_gb(data['bytes'])} "
             f"({data['files']} files)"]
    for r in data["run_dirs"]:
        if r["status"] in ("dropped", "would_drop"):
            lines.append(f"  {r['status']:<12} {r['run_dir']}  {_gb(r['bytes'])}")
        elif r["status"] in ("kept", "not_finished"):
            lines.append(f"  {r['status']:<12} {r['run_dir']}  ({r['why']})")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- writing the policy


class KeepEditError(ValueError):
    """The project file cannot be edited safely (a multi-line ``keep`` value, a result that does not read back)."""

    code = "E_KEEP_EDIT"
    exit_code = 1


_HEADER = re.compile(r"^\s*\[\s*([^\[\]]+?)\s*\]\s*(#.*)?$")
_KEEP_LINE = re.compile(r"^\s*keep\s*=")


def _toml_value(keep: bool | list[str]) -> str:
    if isinstance(keep, bool):
        return "true" if keep else "false"
    return "[" + ", ".join(json.dumps(p) for p in keep) + "]"


def set_features_keep(path: str | os.PathLike, keep: bool | list[str], *, comment: str | None = None) -> str:
    """Write ``keep`` as ``[features] keep`` into the TOML file ``path`` (the project's ``yavih.toml``), with ``comment``
    as a ``#`` line just above it; every other line is kept as it is. The table is appended when absent, the key added
    or its one-line value replaced; a ``keep`` written over several lines is refused (edit it by hand). The result must
    read back with the new value, else nothing is written. Returns the new text."""
    from yavih.config import tomllib

    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    new_lines = ([f"# {c}" for c in comment.splitlines() if c.strip()] if comment else []) + [f"keep = {_toml_value(keep)}"]
    start = next((i for i, ln in enumerate(lines) if (m := _HEADER.match(ln)) and m.group(1).strip() == "features"), None)
    if start is None:
        out = list(lines)
        while out and not out[-1].strip():
            out.pop()
        out += ["", "[features]", *new_lines] if out else ["[features]", *new_lines]
    else:
        end = next((i for i in range(start + 1, len(lines)) if _HEADER.match(lines[i]) or lines[i].lstrip().startswith("[[")), len(lines))
        at = next((i for i in range(start + 1, end) if _KEEP_LINE.match(lines[i])), None)
        if at is None:
            out = lines[: start + 1] + new_lines + lines[start + 1:]
        else:
            value = lines[at].split("=", 1)[1].split("#", 1)[0].strip()
            if value.startswith("[") and not value.endswith("]"):
                raise KeepEditError(f"{path}:{at + 1}: [features] keep spans several lines; edit it by hand "
                                    f"(keep = {_toml_value(keep)})")
            out = lines[:at] + new_lines + lines[at + 1:]
    new_text = "\n".join(out) + "\n"
    try:
        got = tomllib.loads(new_text).get("features", {}).get("keep")
    except Exception as e:  # noqa: BLE001 - any parse failure means the edit is unsafe
        raise KeepEditError(f"{path}: the edited file does not parse ({e}); nothing written") from e
    if got != keep:
        raise KeepEditError(f"{path}: the edited file reads back keep = {got!r}, not {keep!r}; nothing written")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    os.replace(tmp, path)
    return new_text
