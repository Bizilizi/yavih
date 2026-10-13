"""``yavih plan init``: a ``matrix.yaml`` skeleton and the bridge stubs for one model on one suite.

One matrix entry per suite ``(benchmark, reading)``, keyed ``<model>-<benchmark>-<reading>``, with
``script: <bridges_dir>/<model>_<benchmark>.py`` (ONE bridge file per benchmark: a template that takes the
reading as an argument gets ``args: {reading: <r>}`` per entry), ``profile:
<bridges_dir>/<model>-<benchmark>-<reading>.profile.json`` and a commented ``waive:`` line. Pending suite
entries become a comment block; a ``join`` is a report step, never a matrix entry. The bridge files are
copies of each folder's ``bridge_template.py`` (``yavih.shipped.bridge_new``); an existing bridge file is
kept, never overwritten. ``plan build`` refuses the matrix (``E_PROFILE``) until every profile exists.

stdlib at module scope; the suite, the shipped-file helpers and ``yaml`` are imported lazily.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

__all__ = ["write_matrix_skeleton", "plan_init", "skeleton_entries"]

_PLAIN = re.compile(r"^[A-Za-z0-9_./-]+$")


def _scalar(value: str) -> str:
    """A yaml scalar: plain when it is a simple path or name, else double-quoted (JSON is valid yaml)."""
    return value if _PLAIN.match(value) else json.dumps(value)


def skeleton_entries(suite: Any, *, model: str, bridges_dir: str = "bridges") -> list[dict[str, Any]]:
    """The matrix entries of ``suite`` for ``model``, in suite order: ``{key, capability, benchmark, reading,
    script, profile, args}`` (``args`` is ``{reading: r}`` for a multi-reading template, else ``{}``). Paths are
    relative to the matrix file's directory (``bridges_dir`` may be absolute)."""
    from yavih.suites import load_suite

    suite = load_suite(suite)
    out = []
    for cap, e in suite.entries():
        if e.status != "packaged":
            continue
        script = _script_path(bridges_dir, e.benchmark, model).as_posix()
        for r in e.readings:
            key = f"{model}-{e.benchmark}-{r}"
            out.append({"key": key, "capability": cap, "benchmark": e.benchmark, "reading": r, "script": script,
                        "profile": (Path(bridges_dir) / f"{key}.profile.json").as_posix(),
                        "args": {"reading": r} if e.multi_reading else {}})
    return out


def _check_model(model: str) -> None:
    from yavih.plan.matrix import MODEL_RE
    from yavih.suites import SuiteError

    if not isinstance(model, str) or not MODEL_RE.match(model):
        raise SuiteError(f"--model {model!r} must match [A-Za-z0-9_.-]{{1,64}} and start with a letter or digit "
                         "(the name every bridge of this model uses; results are filed under it)", "E_ARG")


def write_matrix_skeleton(suite: Any, *, model: str, launcher: str, out: str | os.PathLike = "matrix.yaml",
                          bridges_dir: str = "bridges", force: bool = False) -> dict[str, Any]:
    """Write the ``matrix.yaml`` skeleton for ``model`` on ``suite`` in the layout of
    https://github.com/Bizilizi/yavih/blob/main/docs/execution.md; refuses an existing ``out`` unless ``force``
    (``E_EXISTS``). Returns ``{path, suite, model, entries, pending, joins}``."""
    from yavih.suites import SuiteError, load_suite

    _check_model(model)
    if not isinstance(launcher, str) or not launcher.strip():
        raise SuiteError("--launcher must be the model environment's python (e.g. /envs/mymodel/bin/python)", "E_ARG")
    suite = load_suite(suite)
    path = Path(out)
    if path.exists() and not force:
        raise SuiteError(f"{path} exists; pass --force to overwrite it (or --out another path)", "E_EXISTS")
    entries = skeleton_entries(suite, model=model, bridges_dir=bridges_dir)
    if not entries:
        raise SuiteError(f"suite {suite.name!r} has no packaged entry to plan")
    pending = [(cap, e) for cap, e in suite.entries() if e.status != "packaged"]
    joins = [e for _cap, e in suite.entries() if e.status == "packaged" and e.join]
    lines = [
        f"# matrix.yaml for model {model!r} on suite {suite.name!r} (yavih plan init).",
        "# `yavih plan build` refuses this file (E_PROFILE) until every profile below exists: fill every TODO in the",
        "# bridge files, then run `yavih bridge check <script> --bench <benchmark> [--arg reading=<r>] --out <profile>`",
        "# for each entry on the staged data. Schema: https://github.com/Bizilizi/yavih/blob/main/docs/execution.md",
        "schema_version: 1",
        "out: null                           # runs root; null = the configured runs root (yavih config get runs.root)",
        "results: null                       # results-store root for the probe units; null = the usual resolution",
        "seeds: [0]                          # one probe unit per seed",
        "shard_minutes: 30                   # target wall per extract unit",
        "canaries: 8                         # canary rows re-encoded per shard",
        "defaults:                           # merged into every entry",
        f"  launcher: {json.dumps(launcher)}",
        "  env: {}",
        "  cpus: 8                           # decode workers = cpus - 1",
        "  args: {}",
        "  allow_loopback: false",
        "bridges:                            # one entry per (bridge, benchmark, reading)",
    ]
    cap = None
    for ent in entries:
        if ent["capability"] != cap:
            cap = ent["capability"]
            lines.append(f"  # {cap}")
        lines.append(f"  {ent['key']}:")
        lines.append(f"    script: {_scalar(ent['script'])}")
        lines.append(f"    benchmark: {ent['benchmark']}")
        if ent["args"]:
            lines.append(f"    args: {{reading: {json.dumps(ent['reading'])}}}")
        lines.append(f"    profile: {_scalar(ent['profile'])}")
        lines.append("    # waive: {CODE: \"reason\"}       # operator waivers for the probe's validate prologue")
    for e in joins:
        lines.append(f"# join: once the {', '.join(e.readings)} readings of {e.benchmark} are probed, score them together with")
        lines.append(f"#   yavih report build <runs root>/{e.benchmark}/{model} --join {e.benchmark} "
                     + " ".join(f"{r}=<runs root>/{e.benchmark}/{model}/{r}" for r in e.readings))
        lines.append(f"#   (`yavih suite missing {suite.name} --model {model}` prints it with the paths filled in)")
    if pending:
        lines.append("# pending (no packaged benchmark folder yet; add entries once it ships):")
        for cap_, e in pending:
            lines.append(f"#   {cap_}: {e.benchmark} [{', '.join(e.readings)}]")
    text = "\n".join(lines) + "\n"
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path.resolve()), "suite": suite.name, "model": model, "entries": entries,
            "pending": [{"capability": c, "benchmark": e.benchmark, "readings": list(e.readings)} for c, e in pending],
            "joins": [{"benchmark": e.benchmark, "readings": list(e.readings), "reading": e.join_reading} for e in joins]}


def plan_init(suite: Any, *, model: str, launcher: str, out: str | os.PathLike = "matrix.yaml", bridges_dir: str = "bridges",
              force: bool = False) -> dict[str, Any]:
    """``yavih plan init``: one bridge file per packaged suite benchmark (``yavih.shipped.bridge_new`` into
    ``<matrix dir>/<bridges_dir>/<model>_<benchmark>.py``; an existing file is kept) and the matrix skeleton
    (:func:`write_matrix_skeleton`; ``force`` overwrites the matrix only). Returns the skeleton record plus
    ``bridges: [{benchmark, path, action: created|kept}]`` and ``next``: one ``yavih bridge check`` command per
    entry, then the ``plan build`` command, with paths as seen from the current directory."""
    from yavih.shipped import ShippedFileError, bridge_new
    from yavih.suites import SuiteError, load_suite

    _check_model(model)
    suite = load_suite(suite)
    path = Path(out)
    if path.exists() and not force:  # refuse before any bridge file is written
        raise SuiteError(f"{path} exists; pass --force to overwrite it (or --out another path)", "E_EXISTS")
    root = path.parent
    bridges: list[dict[str, Any]] = []
    for e_bench in dict.fromkeys(e.benchmark for _c, e in suite.entries() if e.status == "packaged"):
        dest = root / _script_path(bridges_dir, e_bench, model)
        try:
            rec = bridge_new(e_bench, model=model, out=dest)
            bridges.append({"benchmark": e_bench, "path": rec["path"], "action": "created", "filled": rec["filled"]})
        except ShippedFileError as err:
            if err.code != "E_EXISTS":
                raise
            bridges.append({"benchmark": e_bench, "path": str(dest.resolve()), "action": "kept", "filled": []})
    rec = write_matrix_skeleton(suite, model=model, launcher=launcher, out=path, bridges_dir=bridges_dir, force=force)
    rec["bridges"] = bridges
    commands = []
    for ent in rec["entries"]:
        script, profile = _shown(root / ent["script"]), _shown(root / ent["profile"])
        arg = f" --arg reading={ent['reading']}" if ent["args"] else ""
        commands.append(f"yavih bridge check {script} --bench {ent['benchmark']}{arg} --out {profile}")
    commands.append(f"yavih plan build {_shown(path)} --out {_shown(root / 'plan.jsonl')}")
    rec["next"] = commands
    return rec


def _script_path(bridges_dir: str, bench: str, model: str) -> Path:
    """``<bridges_dir>/<model>_<bench>.py`` (relative to the matrix directory unless ``bridges_dir`` is absolute)."""
    from yavih.shipped import default_bridge_path

    return Path(bridges_dir) / default_bridge_path(bench, model).name


def _shown(p: Path) -> str:
    """A path as typed from the current directory: relative when it lies below it, else absolute."""
    try:
        rel = os.path.relpath(p.resolve(), Path.cwd().resolve())
    except (OSError, ValueError):
        return str(p)
    return str(p.resolve()) if rel.startswith("..") else Path(rel).as_posix()
