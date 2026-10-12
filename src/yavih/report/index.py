"""``runs/_index.jsonl``: one line per ``result.json`` with a sibling ``DONE``,
mtime-cached.

Invariants:

- A ``result.json`` without a sibling ``DONE`` is ignored: a ``failed`` status never
  writes DONE, and a probe that died mid-write leaves no marker.
- A DONE marker must be a JSON object carrying ``result_sha256`` equal to the sha256 of the
  ``result.json`` next to it (and ``protocol_hash`` / ``probe_hash`` / ``eval_hash`` / ``model`` / ``reading``
  equal to the result's when both carry them); anything else — including an empty or non-JSON DONE — is treated as absent (the
  harness never writes a bare marker).
- The cache key is ``(path, mtime_ns, size)`` of ``result.json``; a cached line is reused verbatim
  when both match, so re-running ``report`` over thousands of runs costs one ``stat`` per result.
- Every indexed record is upgraded in memory via :func:`yavih.report.migrate.migrate`.
- Results are found at ANY depth under the globs; the layout ``runs/<bench>/<model>/<reading>/probe/<eval_hash12>/seed<k>/``
  only locates the index when no runs root is given.
- A result reached through a ``_stale`` directory (an output a re-probe moved aside) is indexed with
  ``"moved_aside": true``: never a row, never a feature set of a live configuration, only an older configuration
  of a live row when its ``describe_sha256`` differs from every live one (``W_BRIDGE_CHANGED``).

stdlib only.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from yavih.report.migrate import MigrationError, migrate

__all__ = ["build_index", "iter_results", "iter_moved_aside", "load_index", "index_path", "runs_root_of", "INDEX_FILE", "DONE_FILE",
           "RESULT_FILE", "MOVED_ASIDE"]

INDEX_FILE = "_index.jsonl"
DONE_FILE = "DONE"
RESULT_FILE = "result.json"
#: Where moved-aside outputs live (``yavih.io.format.STALE_DIR``; not imported: this module stays stdlib-only).
STALE_DIR = "_stale"
#: Index-record flag of a result found under ``_stale/`` (:func:`iter_moved_aside`): never a row, only an older
#: configuration of a live row (``W_BRIDGE_CHANGED``).
MOVED_ASIDE = "moved_aside"

#: Layout ``runs/<bench>/<model>/<reading>/probe/<eval_hash12>/seed<k>/result.json`` → ``runs`` is ``parents[6]``.
#: Used only to locate the index when no ``runs_root`` is given.
_DEPTH_TO_RUNS_ROOT = 6


def _expand(pattern: str) -> list[Path]:
    p = Path(pattern).expanduser()
    if glob.has_magic(str(p)):
        return [Path(m) for m in glob.glob(str(p), recursive=True)]
    return [p] if p.exists() else []


def _has_valid_done(result_path: Path) -> bool:
    """A valid DONE marker next to ``result.json``: a JSON object whose ``result_sha256`` matches the file
    bytes and whose recorded hashes and names (when the result carries them) equal the result's. Every marker the
    harness writes (``readouts.base.write_result``) has that shape, so an empty / non-JSON / sha-less
    DONE can only be hand-made and is treated as absent (with a notice): it would otherwise let an
    edited or hand-written result.json — including a ``failed`` one, which never earns a DONE — into
    the tables."""
    done = result_path.with_name(DONE_FILE)
    if not done.is_file():
        return False
    try:
        text = done.read_text(encoding="utf-8").strip()
        marker = json.loads(text) if text else None
    except (OSError, ValueError):
        marker = None
    if not isinstance(marker, dict) or not marker.get("result_sha256"):
        print(f"[yavih.report] skipping {result_path}: DONE is not a JSON marker with result_sha256", file=sys.stderr)
        return False
    try:
        data = result_path.read_bytes()
    except OSError:
        return False
    if hashlib.sha256(data).hexdigest() != marker["result_sha256"]:
        print(f"[yavih.report] skipping {result_path}: result.json bytes differ from DONE.result_sha256", file=sys.stderr)
        return False
    try:
        result = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False
    if not isinstance(result, dict):
        return False
    recorded = {
        "protocol_hash": (result.get("benchmark") or {}).get("protocol_hash") if isinstance(result.get("benchmark"), dict) else None,
        "probe_hash": (result.get("probe") or {}).get("hash") if isinstance(result.get("probe"), dict) else None,
        "eval_hash": result.get("eval_hash"),
        "model": result.get("model"),
        "reading": result.get("reading"),
    }
    for k, want in recorded.items():
        if want is not None and k in marker and marker[k] != want:
            print(f"[yavih.report] skipping {result_path}: DONE.{k} differs from result.json", file=sys.stderr)
            return False
    return True


def _find(globs: Sequence[str] | str | os.PathLike) -> tuple[list[Path], list[Path]]:
    """``(live, moved_aside)``: every ``result.json`` with a valid sibling ``DONE`` under the globs, split by whether a
    recursive search reached it through a ``_stale`` path component below the matched directory. A result reached
    both ways is live."""
    if isinstance(globs, (str, os.PathLike)):
        globs = [str(globs)]
    live: set[Path] = set()
    aside: set[Path] = set()
    for pattern in globs:
        for match in _expand(str(pattern)):
            if match.is_dir():
                candidates = [(c, STALE_DIR in c.relative_to(match).parts) for c in match.rglob(RESULT_FILE)]
            elif match.name == RESULT_FILE and match.is_file():
                candidates = [(match, False)]
            else:
                continue
            for cand, stale in candidates:
                if cand.is_file() and _has_valid_done(cand):
                    (aside if stale else live).add(cand.resolve())
    return sorted(live), sorted(aside - live)


def iter_results(globs: Sequence[str]) -> list[Path]:
    """Every ``result.json`` under the globs that has a sibling ``DONE``; others are ignored.

    A glob may match run directories (searched recursively), ``probe/`` trees, seed directories or
    ``result.json`` files themselves. Results are returned sorted and de-duplicated.

    A recursive search never returns a result below ``_stale/``: outputs moved aside because
    their configuration or features changed are never a row. :func:`iter_moved_aside` lists them; the report uses
    them only as older configurations of a live row (``W_BRIDGE_CHANGED`` when their ``describe_sha256`` differs
    from every live one), never as a feature set of the same configuration. A glob that names a ``_stale`` tree or
    file explicitly reports it as live.
    """
    return _find(globs)[0]


def iter_moved_aside(globs: Sequence[str]) -> list[Path]:
    """The results with a valid ``DONE`` that a recursive search under the globs reached through ``_stale/`` (a
    reconfigured bridge's re-probe moves the old output to ``probe/_stale/<utc>/<eval_hash12>/seed<k>/``) and
    that no glob reports as live."""
    return _find(globs)[1]


def _runs_root_of_one(result_path: Path) -> Path | None:
    """The runs root implied by one ``result.json`` path (see ``_DEPTH_TO_RUNS_ROOT``)."""
    r = Path(result_path).resolve()
    if len(r.parents) <= _DEPTH_TO_RUNS_ROOT or r.parents[2].name != "probe":
        return None
    return r.parents[_DEPTH_TO_RUNS_ROOT]


def runs_root_of(results: Sequence[Path]) -> Path | None:
    """The common runs root implied by the layout
    ``runs/<bench>/<model>/<reading>/probe/<eval_hash12>/seed<k>/result.json``; ``None`` when there are no
    results or they do not share one root."""
    roots: set[Path] = set()
    for r in results:
        root = _runs_root_of_one(Path(r))
        if root is None:
            return None
        roots.add(root)
    if len(roots) != 1:
        return None
    return roots.pop()


def index_path(runs_root: str | Path) -> Path:
    return Path(runs_root) / INDEX_FILE


def load_index(runs_root: str | Path | None) -> dict[str, dict[str, Any]]:
    """Existing ``_index.jsonl`` lines keyed by result path (empty when absent or unreadable)."""
    if runs_root is None:
        return {}
    path = index_path(runs_root)
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("path"), str):
                    out[rec["path"]] = rec
    except OSError:
        return {}
    return out


def _record(path: Path, st: os.stat_result) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[yavih.report] skipping unreadable {path}: {exc}", file=sys.stderr)
        return None
    try:
        result = migrate(raw)
    except (MigrationError, TypeError) as exc:
        print(f"[yavih.report] skipping {path}: {exc}", file=sys.stderr)
        return None
    return {"path": str(path), "mtime_ns": st.st_mtime_ns, "size": st.st_size, "result": result}


def _write_atomic(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)
            fh.write("\n")
    os.replace(tmp, path)


def build_index(globs: Sequence[str], runs_root: str | Path | None = None) -> list[dict]:
    """Rebuild ``<runs_root>/_index.jsonl`` from the results under ``globs`` and return its records.

    Each record is ``{"path", "mtime_ns", "size", "result"}`` with ``result`` the migrated
    ``result.json`` record; a result found under ``_stale/`` (:func:`iter_moved_aside`) also carries ``"moved_aside": true``.
    Cached lines whose ``(mtime_ns, size)`` still match are reused without re-reading.

    ``runs_root=None`` infers the root from the layout (:func:`runs_root_of`); when it cannot be
    inferred the index is built in memory and not written.
    """
    results, aside = _find(globs)
    root = Path(runs_root) if runs_root is not None else runs_root_of(results)
    cached = load_index(root)
    records: list[dict[str, Any]] = []
    flagged = set(aside)
    for path in [*results, *aside]:
        try:
            st = path.stat()
        except OSError:
            continue
        hit = cached.get(str(path))
        fresh = (
            hit is not None
            and hit.get("mtime_ns") == st.st_mtime_ns
            and hit.get("size") == st.st_size
            and isinstance(hit.get("result"), dict)
        )
        if fresh:
            try:
                hit["result"] = migrate(hit["result"])
            except (MigrationError, TypeError):
                fresh = False
        if not fresh:
            hit = _record(path, st)
        if hit is not None:
            # recomputed every build (it depends on the globs, not only on the file): a moved-aside result is an older
            # configuration at most, never a row
            if path in flagged:
                hit[MOVED_ASIDE] = True
            else:
                hit.pop(MOVED_ASIDE, None)
            records.append(hit)
    if root is not None:
        try:
            _write_atomic(index_path(root), [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records])
        except OSError as exc:
            print(f"[yavih.report] could not write {index_path(root)}: {exc}", file=sys.stderr)
    return records
