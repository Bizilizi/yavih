"""Hugging Face data roots: a pinned dataset revision downloaded into a data root (``yavih bench snapshot``).

A folder declares where a root's files come from with ``data.roots.<root>.hf``::

    media_root:
      default: "${datasets.audioset20k}"
      hf: {repo: 11hu83/audioset20k, revision: <40-hex commit>, repo_type: dataset, allow_patterns: ["*.tar", "index.json"]}

``snapshot`` downloads exactly that revision (``huggingface_hub.snapshot_download``, the ``hf`` extra) into the
resolved root and writes ``.yavih_hf.json``: the repo, the revision, the patterns and every file it holds.
``snapshot_status`` compares a root with its declaration. The revision is provenance, never part of a hash: the
frozen manifest and its sha256 are the protocol, and ``bench verify`` checks the files themselves.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["STAMP", "HubMissing", "parse_hf", "snapshot", "snapshot_status"]

STAMP = ".yavih_hf.json"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
REPO_TYPES = ("dataset", "model")


class HubMissing(ImportError):
    """huggingface_hub is not installed (``pip install 'yavih[hf]'``)."""


def parse_hf(d: Any, path: str) -> dict[str, Any]:
    """The validated ``hf`` block: ``{repo, revision, repo_type, allow_patterns}`` (``ValueError`` names the key)."""
    if not isinstance(d, Mapping):
        raise ValueError(f"{path}: expected a mapping {{repo, revision, repo_type?, allow_patterns?}}")
    unknown = sorted(set(d) - {"repo", "revision", "repo_type", "allow_patterns"})
    if unknown or "repo" not in d or "revision" not in d:
        raise ValueError(f"{path}: keys repo and revision are required, repo_type and allow_patterns optional (unknown: {unknown})")
    repo, rev = d["repo"], d["revision"]
    if not isinstance(repo, str) or not _REPO.match(repo):
        raise ValueError(f"{path}.repo: expected '<owner>/<name>', got {repo!r}")
    if not isinstance(rev, str) or not _REVISION.match(rev):
        raise ValueError(f"{path}.revision: expected a 40-hex commit sha (a branch or tag moves), got {rev!r}")
    rtype = d.get("repo_type", "dataset")
    if rtype not in REPO_TYPES:
        raise ValueError(f"{path}.repo_type: one of {list(REPO_TYPES)}, got {rtype!r}")
    pats = d.get("allow_patterns")
    if pats is not None and (not isinstance(pats, list) or not pats or not all(isinstance(p, str) and p for p in pats)):
        raise ValueError(f"{path}.allow_patterns: a non-empty list of glob patterns")
    return {"repo": repo, "revision": rev, "repo_type": rtype, "allow_patterns": list(pats) if pats else None}


def _files(root: Path) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".cache"]  # huggingface_hub's local_dir bookkeeping
        for f in filenames:
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            if rel != STAMP:
                out.append(rel.replace(os.sep, "/"))
    return sorted(out)


def snapshot(root: str | os.PathLike, hf: Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Download ``hf`` (a parsed block) into ``root`` and stamp it; ``dry_run`` only reports what it would fetch."""
    root = Path(root)
    plan = {"root": str(root), **dict(hf)}
    if dry_run:
        return {**plan, "dry_run": True}
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - the [hf] extra
        raise HubMissing("Hugging Face data roots need huggingface_hub: pip install 'yavih[hf]'") from e
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hf["repo"], repo_type=hf["repo_type"], revision=hf["revision"], local_dir=str(root),
                      allow_patterns=hf.get("allow_patterns"))
    stamp = {**plan, "files": _files(root)}
    tmp = root / (STAMP + ".tmp")
    tmp.write_text(json.dumps(stamp, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, root / STAMP)
    return stamp


def snapshot_status(root: str | os.PathLike, hf: Mapping[str, Any]) -> dict[str, Any]:
    """``{state, message, files?}`` of a root against its ``hf`` declaration. ``state``: ``absent`` (no root),
    ``unstamped`` (files, no stamp: staged some other way), ``ok`` (the declared revision, every stamped file
    present), ``other_revision`` or ``incomplete`` (stamped files missing)."""
    root = Path(root)
    if not root.is_dir():
        return {"state": "absent", "message": f"{root} does not exist: run `yavih bench snapshot`"}
    p = root / STAMP
    if not p.is_file():
        return {"state": "unstamped", "message": f"{root} holds no {STAMP}: staged without `yavih bench snapshot`, revision unknown"}
    try:
        stamp = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"state": "unstamped", "message": f"{p}: unreadable stamp ({e})"}
    if (stamp.get("repo"), stamp.get("revision")) != (hf["repo"], hf["revision"]):
        return {"state": "other_revision", "message": f"{root} holds {stamp.get('repo')}@{str(stamp.get('revision'))[:8]}, "
                                                       f"the folder declares {hf['repo']}@{hf['revision'][:8]}"}
    files: Sequence[str] = stamp.get("files") or []
    lacking = [f for f in files if not (root / f).is_file()]
    if lacking:
        return {"state": "incomplete", "message": f"{len(lacking)} of {len(files)} stamped files are missing (first {lacking[:3]})",
                "files": len(files)}
    return {"state": "ok", "message": f"{hf['repo']}@{hf['revision'][:8]}: {len(files)} files", "files": len(files)}
