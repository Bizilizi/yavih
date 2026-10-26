"""``yavih bench new <name> [--readout R] [--like B] [--dir D]``: scaffold a benchmark folder.

The scaffold is a copy of an existing folder's machinery (``--like``: its frozen manifest, classes and bridge
template, and any ``reader.py`` / ``probe.py`` with a ``# TODO: copied verbatim …`` banner as their first lines; never
its ``curation.json``, notes, tests, prepare steps or header image) with ``benchmark.yaml`` renamed and reset to
``protocol_version`` 1, and a README.md, setup.md, ``make_manifest.py``, ``prepare/`` and ``tests/test_folder.py``
written from the folder contract, plus a ``header:`` block for the README header image
(``python -m yavih.headers <name>``); ``capability:`` keeps the ``--like`` folder's tags under a ``TODO`` comment. Every fact the new benchmark must supply is marked ``TODO`` (``TODO:`` in
``tests/`` and ``prepare/``). The generated ``tests/test_folder.py`` runs the shared checks of
``yavih.testing.folder`` and pins the values the scaffold carries; the tests that need the benchmark's own data
(``make_manifest.py --check``, the reader smoke) skip with a ``TODO:`` reason. The result passes ``yavih bench lint``
(``TODO`` markers are warnings), its tests pass or skip, and it loads with ``Benchmark.load``; it is a starting point,
not a protocol: replace the manifest (``make_manifest.py``), the protocol, the contracts, the copied scripts, the
tests and the docs, then lint again.

Where it goes: ``--dir D`` (the folder is ``D/<name>``), else the project's benchmarks directory
(``yavih-bench/benchmarks/``), else the first ``$YAVIH_BENCH_PATH`` directory, else the current directory. A folder
in the project directory or on ``$YAVIH_BENCH_PATH`` resolves by name; elsewhere use its path.
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

__all__ = ["DEFAULT_LIKE", "ScaffoldError", "new_benchmark"]

#: The folder a readout's scaffold copies when ``--like`` is not given.
DEFAULT_LIKE: dict[str, str] = {"clf": "esc50", "seq": "ave", "grid": "nyuv2_depth", "box": "vggss_box"}
#: Files of the ``--like`` folder that are written fresh instead of copied.
_WRITTEN = {"README.md", "setup.md", "make_manifest.py", "benchmark.yaml"}
#: Files of the ``--like`` folder that are not copied: its own re-freeze record.
_NOT_COPIED = {"curation.json"}
_SKIPPED_DIRS = {"notes", "tests", "prepare", "assets", "conformance", "__pycache__"}
#: Folder scripts copied from ``--like`` (code, not data): a ``TODO`` banner is prepended as their first lines.
_BANNERED = {
    "reader.py": "the reader of this benchmark's own layout (or delete it: without reader.py the generic "
                 "yavih.sample.iter_samples reads what benchmark.yaml declares)",
    "probe.py": "this benchmark's own targets, metrics or authors' scorer (or delete it together with what "
                "benchmark.yaml takes from it)",
}


class ScaffoldError(ValueError):
    """A usage error of ``bench new`` (exit 1)."""


def _yaml_block(obj: Any) -> str:
    return "```yaml\n" + yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).rstrip("\n") + "\n```"


def _readme(y: dict[str, Any], like: str) -> str:
    m = y["metrics"]
    return f"""![{y['header']['name']}](assets/header.png)

TODO: [Paper](https://…) · [Homepage](https://…) · [Dataset](https://…) · [setup.md](setup.md)

## Description
TODO: the task, the data and the official splits, as the benchmark's authors define them.

## Citation
```bibtex
TODO: the BibTeX entry of the paper
```

## In yavih
TODO: what is scored, on which rows, with which readout. Readout `{y['readout']}`; headline `metrics.headline` =
`{m['headline']}`; `protocol_version` {y['protocol_version']}. (Scaffolded from `{like}`: every number here is
`{like}`'s until the manifest and protocol are replaced.) What the bridge gets and returns: see
[setup.md](setup.md).

TODO: variants and aliases, one sentence each; published reference numbers with their citation, if any.

Re-freezing the manifest: TODO: `python make_manifest.py` must re-freeze `manifest.csv.gz` (and `classes.csv`)
byte-identically from the public upstream files (the scaffold carries `{like}`'s frozen manifest).
"""


def _setup(y: dict[str, Any], like: str) -> str:
    name = y["name"]
    ext = y.get("external") or []
    deps = "\n".join(f"- `{e['name']}`: `{e['pip']}` (TODO: what it is used for)" for e in ext) if ext else \
        "None. TODO: list the packages of the benchmark's authors the probe env needs (exact pip lines), or keep None."
    rev = (y.get("source") or {}).get("revision") if isinstance(y.get("source"), dict) else None
    pinned = f" The pinned upstream revision (`source.revision`) is `{rev}` (TODO: still {like}'s)." if rev else ""
    return f"""# {name} — setup

## Obtain the data
TODO: where the media come from, the exact download commands, sizes and sha256s; who must act first
(`obtain.tier` in `benchmark.yaml`: auto, manual-download, needs-licence, needs-data-owner or site-only).{pinned} Commands (`yavih bench prepare` runs the folder's numbered prepare/ scripts):

```bash
yavih bench prepare {name}      # lists the prepare steps (FOLDER and YAVIH_BENCH are set for each)
yavih bench prepare {name} 01   # TODO
```

## Dependencies
{deps}

## Input contract
TODO: what the reader hands over, per stream, in one sentence each; everything else is the bridge's declared decision.

{_yaml_block(y["input_contract"])}

## Output contract
TODO: what the probe needs from the features (positions, streams, grid, time bins).

{_yaml_block(y["output_contract"])}

## Prepare & verify
TODO: the expected output of `yavih bench verify {name}` once the data is staged (presence per split, census).

## Gotchas
TODO: the setup and bridge mistakes that pass validation on this benchmark.
"""


_MAKE_MANIFEST = '''"""Re-freeze ``manifest.csv.gz`` / ``classes.csv`` of {name} byte-identically from the upstream files.

TODO: read the upstream annotation files, apply the documented curation and the seeded train draw
(``yavih.freeze.class_balanced_subsample``), and write with ``yavih.freeze.gzip_bytes`` / ``csv_bytes`` /
``write_if_changed`` so a re-run reproduces the committed bytes. Record the upstream sha256s and the curation counts
in ``benchmark.yaml`` (``manifest.upstream``)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print("TODO: make_manifest.py of {name} is not written yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

_PREPARE = '''#!/usr/bin/env bash
# TODO: obtain and stage the media of {name} (setup.md "## Obtain the data"); resolve roots with
# `yavih config get {name}.<root>`, never a hard-coded site path.
set -euo pipefail
echo "TODO: prepare/01_obtain.sh of {name} is not written yet" >&2
exit 2
'''

def _banner(fname: str, like: str) -> str:
    return (f"# TODO: copied verbatim from {like} by `yavih bench new`; replace it with {_BANNERED[fname]}.\n"
            "# Delete this banner once the file is this benchmark's own.\n")


def _py(obj: Any, indent: int) -> str:
    """``obj`` as a Python literal; continuation lines indented by ``indent`` spaces."""
    import pprint

    return pprint.pformat(obj, width=max(40, 116 - indent), sort_dicts=False).replace("\n", "\n" + " " * indent)


def _test_folder(y: dict[str, Any], like: str, *, readings: list[str], has_probe: bool) -> str:
    """``tests/test_folder.py`` of a scaffold: the shared checks of ``yavih.testing.folder``, pins of the values the
    scaffold carries (still ``like``'s), and ``pytest.skip("TODO: …")`` for the tests that need the benchmark's own
    data. Every test passes or skips on the fresh scaffold; ``bench lint`` reports each ``TODO:`` left."""
    name = y["name"]
    streams = {k: (v or {}).get("kind") for k, v in (y.get("streams") or {}).items()}
    clips = dict((y.get("budget") or {}).get("clips") or {})
    external = [e.get("name") for e in y.get("external") or []]
    head = f'''"""Folder tests of {name} (run: ``python -m pytest "$(yavih bench path {name})/tests" -q -p no:cacheprovider``).

Written by ``yavih bench new`` from {like}: the tests run on the scaffold as it is, so every value pinned below is
still {like}'s. TODO: pin this folder's own facts once its manifest is frozen (columns and charsets, rows and classes
per split, the curation sums, README.md and setup.md facts, the bridge template's ``describe()``) and write the tests
that skip with a ``TODO:`` reason; ``tests/test_folder.py`` of a packaged folder (``yavih bench path {like}``) has each
of them written out. ``yavih bench lint`` reports every ``TODO:`` left in this file.
"""

from __future__ import annotations

import re

import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows = F.spec, F.rows

READOUT = {y.get("readout")!r}
HEADLINE = {(y.get("metrics") or {}).get("headline")!r}
STREAMS = {_py(streams, 10)}
CLIPS = {_py(clips, 8)}
READINGS = {_py(readings, 11)}
OUTPUT_CONTRACT = {_py(y.get("output_contract"), 18)}
EXTERNAL = {_py(external, 11)}
'''
    body = '''

# ----------------------------------------------------------------------------- yaml + files


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_sha256_matches(spec: dict):
    F.assert_manifest_hashes(HERE, spec)


def test_splits_assigned_once(spec: dict, rows: list[dict[str, str]]):
    counts = F.assert_splits_assigned_once(spec, rows)
    assert {s: counts[s] for s in CLIPS} == CLIPS == spec["budget"]["clips"]


def test_metric_names_resolve(spec: dict):
    F.assert_metric_names_resolve(HERE, spec)
    assert spec["readout"] == READOUT and spec["metrics"]["headline"] == HEADLINE


def test_input_contract(spec: dict):
    ic = spec["input_contract"]
    assert {k: v["kind"] for k, v in spec["streams"].items()} == STREAMS
    assert set(ic["handed_over"]) == set(ic["info"]) == set(STREAMS)


def test_output_contract_and_external(spec: dict):
    assert spec["output_contract"] == OUTPUT_CONTRACT
    assert [e["name"] for e in spec["external"] or []] == EXTERNAL
    assert (HERE / "probe.py").is_file() is HAS_PROBE


# ----------------------------------------------------------------------------- README.md + prepare/


def test_readme_facts(spec: dict):
    """Folder facts in README.md (its structure is ``yavih bench lint``'s). TODO: pin the citation key, the counts
    and the published reference numbers once README.md is written."""
    text = (HERE / "README.md").read_text(encoding="utf-8")
    assert spec["metrics"]["headline"] in F.section(text, 2, "In yavih") and "make_manifest.py" in text


def test_prepare_scripts_present():
    names = sorted(p.name for p in (HERE / "prepare").iterdir() if p.is_file())
    assert names == ["01_obtain.sh"]  # TODO: this benchmark's numbered prepare steps
    assert all(re.match(r"^\\d+_[A-Za-z0-9_.-]+\\.(?:sh|py)$", n) for n in names), names


# ----------------------------------------------------------------------------- bridge template


def test_bridge_template_describe_keys():
    from yavih import Benchmark
    from yavih.bridge import DESCRIBE_REQUIRED, Bridge

    mod = F.load_module(HERE / "bridge_template.py")
    (cls,) = [v for v in vars(mod).values()
              if isinstance(v, type) and issubclass(v, Bridge) and v.__module__ == mod.__name__]
    d = cls(Benchmark.from_dir(HERE).view(), {}).describe()
    assert set(DESCRIBE_REQUIRED) <= set(d), set(DESCRIBE_REQUIRED) - set(d)
    assert d["reading"] in READINGS and set(d["input"]) <= set(STREAMS)


# ----------------------------------------------------------------------------- needs this benchmark's data


def test_make_manifest_check():
    pytest.skip("TODO: run make_manifest.py --check on the public upstream files (or a stand-in rebuilt from the "
                "committed manifest) and assert it re-derives manifest.csv.gz byte-identically")


def test_reader_smoke_synthetic_raw():
    pytest.skip("TODO: write synthetic media for a few manifest rows and assert what Benchmark.reader hands over "
                "(streams, info keys, meta, canaries)")
'''
    return head + f"HAS_PROBE = {has_probe!r}\n" + body


def _dest(dir_: str | os.PathLike | None) -> Path:
    from yavih import config

    if dir_:
        return Path(dir_).expanduser()
    project = config.project_benchmarks_dir()
    if project is not None:
        return project
    bp = [p for p in os.environ.get("YAVIH_BENCH_PATH", "").split(os.pathsep) if p]
    return Path(bp[0]).expanduser() if bp else Path.cwd()


def new_benchmark(name: str, *, readout: str | None = None, like: str | None = None, dir: str | os.PathLike | None = None) -> Path:
    """Write the scaffold of benchmark ``name`` and return its folder."""
    import re

    from yavih.benchmark import resolve_folder
    from yavih.lint import template_readings
    from yavih.readouts import registry

    if not re.fullmatch(r"[a-z][a-z0-9_]*", name or ""):
        raise ScaffoldError(f"benchmark name {name!r} must match [a-z][a-z0-9_]*")
    if readout is not None and readout not in registry.names():
        raise ScaffoldError(f"unknown readout {readout!r}; registered: {list(registry.names())}")
    if like is None:
        if readout is None:
            raise ScaffoldError("give --readout (clf, seq, grid, box, …) or --like <benchmark>")
        like = DEFAULT_LIKE.get(readout)
        if like is None:
            raise ScaffoldError(f"readout {readout!r} has no default folder to copy; give --like <benchmark>")
    src = resolve_folder(like)
    if src is None:
        raise ScaffoldError(f"--like {like!r} is not a benchmark folder (name, alias or path)")
    y = yaml.safe_load((src / "benchmark.yaml").read_text(encoding="utf-8"))
    like_name = str(y.get("name", src.name))
    if readout is not None and y.get("readout") != readout:
        raise ScaffoldError(f"--like {like_name} has readout {y.get('readout')!r}, not {readout!r}; pick a folder of that readout")
    if resolve_folder(name) is not None and not Path(name).is_dir():
        raise ScaffoldError(f"a benchmark named {name!r} already resolves ({resolve_folder(name)}); pick another name")
    dest = _dest(dir) / name
    if dest.exists():
        raise ScaffoldError(f"{dest} already exists")

    dest.mkdir(parents=True)
    for p in src.iterdir():
        if p.name in _WRITTEN or p.name in _NOT_COPIED or p.name in _SKIPPED_DIRS or p.name.startswith("."):
            continue
        if p.is_dir():
            shutil.copytree(p, dest / p.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif p.name in _BANNERED:
            (dest / p.name).write_text(_banner(p.name, like_name) + p.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            shutil.copy2(p, dest / p.name)
    today = _dt.date.today().isoformat()
    y["name"] = name
    y["aliases"] = []
    y["title"] = f"TODO title of {name} (scaffolded from {like_name})"
    y["protocol_version"] = 1
    y["changelog"] = [{"version": 1, "date": today, "invalidates": False,
                       "reason": f"TODO: scaffolded from {like_name} by `yavih bench new`; replace the manifest, protocol, contracts and docs"}]
    if isinstance(y.get("source"), dict):
        y["source"]["notes"] = f"TODO: where the data comes from (the scaffold still describes {like_name}'s source)"
    y["obtain"] = {"tier": (y.get("obtain") or {}).get("tier", "auto"), "how": "TODO: who must act before the data can be staged, and how"}
    y["header"] = {"name": name, "title": "TODO: the title of the paper that introduced the benchmark",
                   "venue": "TODO: authors · venue year",
                   "abstract": "TODO: two or three neutral sentences on the data and the task (at most 60 words)."}
    from yavih.suites import CAPABILITIES

    def _write_yaml() -> None:
        body = yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120)
        # `capability` keeps --like's tags (a valid list) with a TODO marker that `bench lint` reports (W_TODO)
        body = re.sub(r"^capability:", f"capability:  # TODO: still {like_name}'s; what {name} measures, one or more of "
                      f"{', '.join(CAPABILITIES)}", body, count=1, flags=re.M)
        (dest / "benchmark.yaml").write_text(
            f"# {name}: scaffolded from {like_name} by `yavih bench new` on {today}. TODO: every value below is {like_name}'s\n"
            "# until this benchmark's manifest, protocol and contracts replace it; run `yavih bench lint` after each edit\n"
            "# (it tells you the changelog's protocol_hash / input_hash to write after a protocol edit).\n"
            + body, encoding="utf-8")

    _write_yaml()
    # the newest changelog entry records the hashes the folder computes (outside the hashed payload)
    try:
        from yavih.benchmark import Benchmark

        b = Benchmark.from_dir(dest)
        y["changelog"][0].update({"protocol_hash": b.protocol_hash, "input_hash": b.input_hash})
        _write_yaml()
    except Exception:  # noqa: BLE001 - a scaffold that does not load yet: `bench lint` names both the cause and the hashes
        pass
    (dest / "README.md").write_text(_readme(y, like_name), encoding="utf-8")
    (dest / "setup.md").write_text(_setup(y, like_name), encoding="utf-8")
    (dest / "make_manifest.py").write_text(_MAKE_MANIFEST.format(name=name), encoding="utf-8")
    (dest / "prepare").mkdir()
    (dest / "prepare" / "01_obtain.sh").write_text(_PREPARE.format(name=name), encoding="utf-8")
    (dest / "prepare" / "01_obtain.sh").chmod(0o755)
    (dest / "tests").mkdir()
    tpl = dest / "bridge_template.py"
    readings = sorted(template_readings(tpl.read_text(encoding="utf-8"))) if tpl.is_file() else []
    (dest / "tests" / "test_folder.py").write_text(
        _test_folder(y, like_name, readings=readings, has_probe=(dest / "probe.py").is_file()), encoding="utf-8")
    # the conformance fixture of the scaffold's (copied) protocol: re-recorded by the author once the protocol is theirs
    try:
        from yavih.benchmark import Benchmark
        from yavih.conformance import record

        record(Benchmark.from_dir(dest), force=True)
    except Exception:  # noqa: BLE001 - `bench lint` reports E_CONFORMANCE_MISSING with the command to run
        pass
    return dest
