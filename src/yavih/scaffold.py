"""``yavih bench new <name> [--readout R] [--like B] [--dir D]``: scaffold a benchmark folder.

The scaffold is a copy of an existing folder's machinery (``--like``: its frozen manifest, classes, any ``reader.py``,
``probe.py`` and bridge template) with ``benchmark.yaml`` renamed and reset to ``protocol_version`` 1, and a
README.md, setup.md, ``make_manifest.py``, ``prepare/`` and ``tests/test_folder.py`` written from the folder
contract, plus a ``header:`` block for the README header image (``python -m yavih.headers <name>``). Every fact the
new benchmark must supply is marked ``TODO``. The result passes ``yavih bench lint``
(``TODO`` markers are warnings) and loads with ``Benchmark.load``; it is a starting point, not a protocol: replace
the manifest (``make_manifest.py``), the protocol, the contracts and the docs, then lint again.

Where it goes: ``--dir D`` (the folder is ``D/<name>``), else the first ``$YAVIH_BENCH_PATH`` directory, else the
current directory. A folder on ``$YAVIH_BENCH_PATH`` resolves by name; elsewhere use its path.
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
_SKIPPED_DIRS = {"notes", "tests", "prepare", "assets", "__pycache__"}


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
    return f"""# {name} — setup

## Obtain the data
TODO: where the media come from, the exact download commands, sizes and sha256s; who must act first
(`obtain.tier` in `benchmark.yaml`: auto, manual-download, needs-licence, needs-data-owner or site-only). Commands (`yavih bench prepare` runs the folder's numbered prepare/ scripts):

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

_TEST = '''"""Folder tests of {name} (run: ``python -m pytest "$(yavih bench path {name})/tests" -q``). TODO: pin this folder's own
facts (row counts, hashes, columns, reader behaviour) once its manifest is frozen."""

from __future__ import annotations

from pathlib import Path

from yavih.lint import lint_folder

FOLDER = Path(__file__).resolve().parents[1]


def test_folder_passes_the_contract():
    rep = lint_folder(FOLDER)
    assert rep.ok, [f"{{f.code}}: {{f.message}}" for f in rep.errors]
'''


def _dest(dir_: str | os.PathLike | None) -> Path:
    if dir_:
        return Path(dir_).expanduser()
    bp = [p for p in os.environ.get("YAVIH_BENCH_PATH", "").split(os.pathsep) if p]
    return Path(bp[0]).expanduser() if bp else Path.cwd()


def new_benchmark(name: str, *, readout: str | None = None, like: str | None = None, dir: str | os.PathLike | None = None) -> Path:
    """Write the scaffold of benchmark ``name`` and return its folder."""
    import re

    from yavih.benchmark import resolve_folder
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
        if p.name in _WRITTEN or p.name in _SKIPPED_DIRS or p.name.startswith("."):
            continue
        if p.is_dir():
            shutil.copytree(p, dest / p.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
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
    def _write_yaml() -> None:
        (dest / "benchmark.yaml").write_text(
            f"# {name}: scaffolded from {like_name} by `yavih bench new` on {today}. TODO: every value below is {like_name}'s\n"
            "# until this benchmark's manifest, protocol and contracts replace it; run `yavih bench lint` after each edit\n"
            "# (it tells you the changelog's protocol_hash / input_hash to write after a protocol edit).\n"
            + yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")

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
    (dest / "tests" / "test_folder.py").write_text(_TEST.format(name=name), encoding="utf-8")
    return dest
