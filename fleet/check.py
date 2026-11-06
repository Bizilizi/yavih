"""Fleet checks: a worker branch's footprint, the work queue, one queue entry.

    python fleet/check.py footprint <branch> --base <ref> --name <name>
        exit 0 when every path the branch changes since its merge base with <ref> lies under benchmarks/<name>/,
        exit 1 otherwise (each offending path printed), exit 2 on a usage or git error.
    python fleet/check.py queue [--file fleet/queue.yaml] [--allow-new NAME ...]
        validates the queue (keys, names, statuses, routes, suites) against the repository's benchmarks/ directory:
        every folder has an entry, a shipped entry names an existing folder, any other entry names no existing folder
        (unless listed with --allow-new: the folder a benchmark pull request adds). Exit 0 / 1 / 2 as above.
    python fleet/check.py entry <name> [--file fleet/queue.yaml]
        prints one entry as YAML (the block a worker brief embeds); exit 1 when the name is not queued.
    python fleet/check.py ship [--file fleet/queue.yaml]
        marks `shipped` every entry whose folder exists in benchmarks/ (after a merge; the `catalog` workflow runs it
        on main), editing only those entries' `status:` lines; prints what it changed.

Standard library plus PyYAML (a yavih dependency); runs from any directory of the repository.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "fleet" / "queue.yaml"
NAME = re.compile(r"[a-z][a-z0-9_]*")
STATUSES = ("todo", "claimed", "in_flight", "blocked", "review", "shipped", "dropped")
ROUTES = ("core-clf", "core-seq", "core-grid", "core-box", "core-knn", "core-stream",
          "folder-probe", "folder-stateless", "folder-direct", "suite")
CLASSES = ("A", "B1", "B2", None)
KEYS = ("name", "title", "kind", "survey_class", "survey_reasons", "route", "text_fields", "parents", "survey_note",
        "status", "obtain", "blocked_by", "branch")
TIERS = ("auto", "manual-download", "needs-licence", "needs-data-owner", "site-only")


class CheckError(Exception):
    """A usage or environment error (exit 2)."""


def footprint(branch: str, base: str, name: str, *, repo: Path = ROOT) -> list[str]:
    """The paths ``branch`` changes since its merge base with ``base`` that lie outside ``benchmarks/<name>/``."""
    if not NAME.fullmatch(name):
        raise CheckError(f"{name!r} is not a folder name ([a-z][a-z0-9_]*)")
    proc = subprocess.run(["git", "diff", "--name-only", "--no-renames", f"{base}...{branch}"],
                          cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CheckError(f"git diff {base}...{branch}: {proc.stderr.strip()}")
    changed = [p for p in proc.stdout.splitlines() if p]
    if not changed:
        raise CheckError(f"{branch} changes nothing since {base}")
    prefix = f"benchmarks/{name}/"
    return [p for p in changed if not p.startswith(prefix)]


def load_queue(path: Path = QUEUE) -> list[dict]:
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise CheckError(f"{path}: {e}") from e
    if not isinstance(doc, dict) or doc.get("schema_version") != 1 or not isinstance(doc.get("entries"), list):
        raise CheckError(f"{path}: expected `schema_version: 1` and an `entries` list")
    return doc["entries"]


def check_queue(entries: list[dict], folders: set[str], *, allow_new: frozenset[str] = frozenset()) -> list[str]:
    """Every problem of the queue, one line each (empty: valid). ``folders`` are the repository's folder names."""
    out: list[str] = []
    names: dict[str, int] = {}
    for i, e in enumerate(entries):
        where = f"entries[{i}]"
        if not isinstance(e, dict):
            out.append(f"{where}: not a mapping")
            continue
        n = e.get("name")
        where = f"{where} {n!r}"
        missing = [k for k in KEYS if k not in e]
        if missing:
            out.append(f"{where}: missing keys {missing}")
        if not isinstance(n, str) or not NAME.fullmatch(n):
            out.append(f"{where}: name must match [a-z][a-z0-9_]*")
            continue
        if n in names:
            out.append(f"{where}: duplicate of entries[{names[n]}]")
        names[n] = i
        kind, status, route = e.get("kind"), e.get("status"), e.get("route")
        if kind not in ("benchmark", "suite"):
            out.append(f"{where}: kind must be benchmark or suite")
        if status not in STATUSES:
            out.append(f"{where}: status must be one of {list(STATUSES)}")
        if route not in ROUTES:
            out.append(f"{where}: route must be one of {list(ROUTES)}")
        if (kind == "suite") != (route == "suite"):
            out.append(f"{where}: a suite has route `suite`, and only a suite")
        if e.get("survey_class") not in CLASSES:
            out.append(f"{where}: survey_class must be A, B1, B2 or null")
        reasons = e.get("survey_reasons") or []
        if not set(reasons) <= {"i", "ii", "iii"} or (reasons and e.get("survey_class") != "B2"):
            out.append(f"{where}: survey_reasons is a subset of [i, ii, iii], for B2 only")
        if e.get("obtain") is not None and e.get("obtain") not in TIERS:
            out.append(f"{where}: obtain must be one of {list(TIERS)} or null")
        if status == "blocked" and not e.get("blocked_by"):
            out.append(f"{where}: a blocked entry says what unblocks it (blocked_by)")
        if kind == "suite" and status in ("claimed", "review"):
            out.append(f"{where}: a suite entry is never claimed for a worker (the integrator writes the suite file in "
                       "src/yavih/suites/ once its components are shipped)")
        if status in ("claimed", "review") and e.get("branch") != f"bench/{n}":
            out.append(f"{where}: a {status} entry names its branch, bench/{n}")
        exists = n in folders
        if status == "shipped":
            if kind != "benchmark" or not exists:
                out.append(f"{where}: shipped, but benchmarks/{n}/benchmark.yaml does not exist")
        elif exists and n not in allow_new:
            out.append(f"{where}: status {status}, but benchmarks/{n}/ already exists (another benchmark took the "
                       "name, or the entry should be shipped)")
    for e in entries:
        if not isinstance(e, dict) or not isinstance(e.get("name"), str):
            continue
        for p in e.get("parents") or []:
            if p not in names or (entries[names[p]] or {}).get("kind") != "suite":
                out.append(f"{e['name']!r}: parent {p!r} is not a suite entry")
        if e.get("kind") == "suite":
            comps = e.get("components")
            if not isinstance(comps, list) or not comps:
                out.append(f"{e['name']!r}: a suite lists its components")
                continue
            for c in comps:
                if c not in names:
                    out.append(f"{e['name']!r}: component {c!r} is not an entry")
                elif e["name"] not in (entries[names[c]].get("parents") or []):
                    out.append(f"{e['name']!r}: component {c!r} does not list it under parents")
    for f in sorted(folders - set(names)):
        out.append(f"benchmarks/{f}/ has no queue entry (every folder name is reserved in the queue)")
    return out


def ship(path: Path, folders: set[str]) -> list[str]:
    """Set ``status: shipped`` on every benchmark entry of ``path`` whose folder exists; returns the names changed.
    Each entry block (from its ``- `` line to the next) is read as a whole, whatever its key order; only its
    ``status:`` line is rewritten, so the file's comments and layout stay as they are."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("- ")]
    changed: list[str] = []
    for k, s in enumerate(starts):
        block = range(s, starts[k + 1] if k + 1 < len(starts) else len(lines))
        keys: dict[str, tuple[int, str]] = {}
        for i in block:
            line = lines[i][2:] if i == s else lines[i]
            if line.startswith("  ") and not line.startswith("   ") and ":" in line or i == s:
                key, _, value = line.strip().partition(":")
                keys.setdefault(key.strip(), (i, value.strip()))
        name, kind, status = (keys.get(x, (None, None))[1] for x in ("name", "kind", "status"))
        if kind == "benchmark" and name in folders and "status" in keys and status != "shipped":
            i = keys["status"][0]
            indent = "- " if i == s else "  "
            lines[i] = f"{indent}status: shipped\n"
            changed.append(name)
    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def repo_folders(repo: Path = ROOT) -> set[str]:
    return {p.name for p in (repo / "benchmarks").iterdir() if (p / "benchmark.yaml").is_file()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python fleet/check.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("footprint", help="a worker branch changes benchmarks/<name>/ only")
    s.add_argument("branch")
    s.add_argument("--base", required=True, help="the ref the branch is compared with (its merge base is used)")
    s.add_argument("--name", required=True, help="the folder the branch may change")
    s = sub.add_parser("queue", help="validate fleet/queue.yaml against benchmarks/")
    s.add_argument("--file", type=Path, default=QUEUE)
    s.add_argument("--allow-new", nargs="*", default=[], metavar="NAME",
                   help="folders a pull request adds while their entry is not yet shipped")
    s = sub.add_parser("entry", help="print one queue entry")
    s.add_argument("name")
    s.add_argument("--file", type=Path, default=QUEUE)
    s = sub.add_parser("ship", help="mark the entries of existing folders shipped")
    s.add_argument("--file", type=Path, default=QUEUE)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "footprint":
            bad = footprint(a.branch, a.base, a.name)
            for p in bad:
                print(f"outside benchmarks/{a.name}/: {p}")
            print(f"footprint {'FAILED' if bad else 'ok'}: {a.branch} against {a.base}")
            return 1 if bad else 0
        entries = load_queue(a.file)
        if a.cmd == "ship":
            names = ship(a.file, repo_folders())
            print(f"shipped: {', '.join(names)}" if names else "nothing to ship")
            return 0
        if a.cmd == "queue":
            problems = check_queue(entries, repo_folders(), allow_new=frozenset(a.allow_new))
            for p in problems:
                print(p)
            counts: dict[str, int] = {}
            for e in entries:
                if isinstance(e, dict):
                    counts[str(e.get("status"))] = counts.get(str(e.get("status")), 0) + 1
            print(f"queue {'FAILED' if problems else 'ok'}: {len(entries)} entries "
                  + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
            return 1 if problems else 0
        match = [e for e in entries if isinstance(e, dict) and e.get("name") == a.name]
        if not match:
            print(f"{a.name!r} is not in {a.file}", file=sys.stderr)
            return 1
        print(yaml.safe_dump(match[0], sort_keys=False, allow_unicode=True, width=120), end="")
        return 0
    except CheckError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
