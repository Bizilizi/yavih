"""Shared helpers of the benchmark folders' ``tests/test_folder.py``.

Each folder test keeps only that folder's facts (counts, hashes, columns, gotchas, reader behaviour); what every
folder needs to load its files and what holds for every folder lives here once. The contract every folder must
meet (layout, README/setup headings and pins, contract copies, budget vs rows, …) is :mod:`yavih.lint`, run for all
folders by the repo tests and for one folder by :func:`assert_lint_clean`.

Fixtures resolve the folder from the requesting test module (``<folder>/tests/test_*.py``), so a test module
registers one by binding it to a module-level name (pytest uses that name)::

    from yavih.testing import folder as F

    HERE = F.folder_of(__file__)
    _isolated_yavih_home = F.isolated_yavih_home  # autouse
    spec, rows, bench = F.spec, F.rows, F.bench

- fixtures: :func:`isolated_yavih_home` (autouse), :func:`spec`, :func:`manifest_bytes`, :func:`rows`,
  :func:`classes`, :func:`bench`, :func:`mm` (``make_manifest.py``), :func:`template` (``bridge_template.py``);
- loaders: :func:`folder_of`, :func:`load_module`, :func:`load_yaml`, :func:`read_manifest_bytes`,
  :func:`read_rows`, :func:`read_classes`, :func:`site_root`;
- markdown: :func:`headings`, :func:`section`, :func:`last_yaml_block`, :func:`walk_keys`;
- shared assertions: :func:`assert_manifest_hashes`, :func:`assert_splits_assigned_once`,
  :func:`assert_metric_names_resolve`, :func:`assert_lint_clean`.

Core-only at import time (pytest + pyyaml + stdlib); the benchmark, metrics and lint modules load inside functions.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

__all__ = [
    "folder_of", "load_module", "load_yaml", "read_manifest_bytes", "read_rows", "read_classes", "site_root",
    "headings", "section", "last_yaml_block", "walk_keys", "assert_manifest_hashes",
    "assert_splits_assigned_once", "assert_metric_names_resolve", "assert_lint_clean",
    "isolated_yavih_home", "spec", "manifest_bytes", "rows", "classes", "bench", "mm", "template",
]


# ----------------------------------------------------------------------------- loaders


def folder_of(test_file: str | Path) -> Path:
    """The benchmark folder of a ``<folder>/tests/test_*.py`` file."""
    return Path(test_file).resolve().parent.parent


def load_module(path: str | Path, name: str | None = None) -> ModuleType:
    """Import a folder script (``reader.py``, ``make_manifest.py``, ``bridge_template.py``, …) by path.

    The module is registered in ``sys.modules`` under ``name`` (default ``<folder>_<stem>``), so code that looks a
    module up by name (``preprocessing_source()``, dataclasses) works as for an imported module.
    """
    path = Path(path)
    name = name or f"{path.parent.name}_{path.stem}"
    s = importlib.util.spec_from_file_location(name, path)
    assert s is not None and s.loader is not None, path
    mod = importlib.util.module_from_spec(s)
    sys.modules[name] = mod
    s.loader.exec_module(mod)
    return mod


def load_yaml(folder: Path) -> dict:
    """``benchmark.yaml`` read with pyyaml only (no loader, no validation)."""
    return yaml.safe_load((Path(folder) / "benchmark.yaml").read_text(encoding="utf-8"))


def read_manifest_bytes(folder: Path, spec: dict | None = None) -> bytes:
    """The decompressed ``manifest.csv.gz`` bytes (the ones ``manifest.sha256`` pins)."""
    spec = spec or load_yaml(folder)
    with gzip.open(Path(folder) / spec["manifest"]["file"], "rb") as fh:
        return fh.read()


def _dict_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text, newline="")))


def read_rows(folder: Path, spec: dict | None = None) -> list[dict[str, str]]:
    """The frozen manifest rows as dicts, file order."""
    return _dict_rows(read_manifest_bytes(folder, spec).decode("utf-8"))


def read_classes(folder: Path, spec: dict | None = None) -> list[dict[str, str]]:
    """``classes.csv`` rows as dicts (``[]`` for a folder without classes)."""
    spec = spec or load_yaml(folder)
    rel = spec["manifest"].get("classes")
    return _dict_rows((Path(folder) / rel).read_text(encoding="utf-8")) if rel else []


def site_root(folder: Path, key: str, *, needs: tuple[str, ...] = ()) -> Path | None:
    """The site's configured ``<folder>.<key>`` root (env, config files, sites), or None when it is not configured
    or lacks any of the relative paths in ``needs`` (data not staged on this machine)."""
    from yavih import config

    try:
        root = Path(config.folder_root(Path(folder), key))
    except config.ConfigError:
        return None
    return root if root.exists() and all((root / n).exists() for n in needs) else None


# ----------------------------------------------------------------------------- markdown


def _level(level: int | str) -> int:
    return len(level) if isinstance(level, str) else int(level)


def headings(text: str, level: int | str) -> list[str]:
    """Titles of the ATX headings of ``level`` (``2`` or ``"##"``) outside fenced code blocks, in order."""
    from yavih.lint import headings as _headings

    return [h for _, h in _headings(text, _level(level))]


def section(text: str, level: int | str, title: str) -> str:
    """Body of heading ``title`` at ``level`` up to the next heading of the same or a higher level."""
    from yavih.lint import section as _section

    return _section(text, _level(level), title)


def last_yaml_block(body: str) -> Any:
    """The last fenced ``yaml`` block of ``body``, parsed."""
    from yavih.lint import fenced_blocks

    blocks = fenced_blocks(body, "yaml")
    assert blocks, "section needs a fenced yaml block"
    return yaml.safe_load(blocks[-1])


def walk_keys(obj: Any, out: set[str] | None = None) -> set[str]:
    """Every mapping key anywhere in a parsed yaml/json value."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            walk_keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_keys(v, out)
    return out


# ----------------------------------------------------------------------------- shared assertions


def assert_manifest_hashes(folder: Path, spec: dict | None = None) -> None:
    """``manifest.sha256`` / ``classes_sha256`` match the committed files, and the gzip header is reproducible
    (``mtime`` 0, no file name: a re-freeze is byte-identical)."""
    folder = Path(folder)
    spec = spec or load_yaml(folder)
    m = spec["manifest"]
    assert hashlib.sha256(read_manifest_bytes(folder, spec)).hexdigest() == m["sha256"]
    if m.get("classes"):
        assert hashlib.sha256((folder / m["classes"]).read_bytes()).hexdigest() == m["classes_sha256"]
    raw = (folder / m["file"]).read_bytes()
    assert raw[:3] == b"\x1f\x8b\x08" and raw[4:8] == b"\0\0\0\0" and not raw[3] & 0x08


def assert_splits_assigned_once(spec: dict, rows: list[dict[str, str]]) -> Counter:
    """Every split is assigned to exactly one role, the assigned splits are exactly the manifest's, and ``clip_id``
    is unique. Returns the rows per split, for the folder's own count pins."""
    s = spec["splits"]
    assigned = [*s["train"], *s["eval"], *s["floor"], *s["unused"], *(s.get("folds") or [])]
    assert len(assigned) == len(set(assigned)), assigned
    assert sorted(assigned) == sorted({r["split"] for r in rows})
    assert len({r["clip_id"] for r in rows}) == len(rows), "clip_id is not unique"
    return Counter(r["split"] for r in rows)


def assert_metric_names_resolve(folder: Path, spec: dict | None = None) -> None:
    """Every listed / floor / control metric is a registry metric for the folder's readout or one the folder's
    ``probe.py`` declares in ``METRICS``; ``chance`` and ``ci`` keys are listed metrics."""
    from yavih.metrics.registry import REGISTRY, allows

    folder = Path(folder)
    spec = spec or load_yaml(folder)
    m, readout = spec["metrics"], spec["readout"]
    own = set(getattr(load_module(folder / "probe.py"), "METRICS", {})) if (folder / "probe.py").is_file() else set()
    for name in [*m["list"], *m["floors"], *m["controls"]]:
        assert name in own or (name in REGISTRY and allows(REGISTRY[name], readout)), name
    assert m["headline"] in m["list"]
    assert set(m.get("chance") or {}) <= set(m["list"]) and set(m.get("ci") or {}) <= set(m["list"])


def assert_lint_clean(folder: Path, *, skip: tuple[str, ...] = ()) -> None:
    """``yavih bench lint`` on this folder reports no error (the contract every folder meets)."""
    from yavih.lint import lint_folder

    errors = lint_folder(Path(folder), skip=skip).errors
    assert not errors, "\n".join(f"{f.check} {f.code}: {f.message}" for f in errors)


# ----------------------------------------------------------------------------- fixtures


def _folder(request: pytest.FixtureRequest) -> Path:
    return folder_of(request.module.__file__)


@pytest.fixture(autouse=True)
def isolated_yavih_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never the developer's ~/.yavih: the site's census (bench verify --census) shares the folder's manifest hash
    and would leak real-media facts into synthetic fixtures."""
    home = tmp_path_factory.getbasetemp() / "yavih_home_isolated"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("YAVIH_HOME", str(home))


@pytest.fixture(scope="module")
def spec(request: pytest.FixtureRequest) -> dict:
    """``benchmark.yaml`` (pyyaml only)."""
    return load_yaml(_folder(request))


@pytest.fixture(scope="module")
def manifest_bytes(request: pytest.FixtureRequest) -> bytes:
    """The decompressed manifest bytes."""
    return read_manifest_bytes(_folder(request))


@pytest.fixture(scope="module")
def rows(request: pytest.FixtureRequest) -> list[dict[str, str]]:
    """The frozen manifest rows as dicts, file order."""
    return read_rows(_folder(request))


@pytest.fixture(scope="module")
def classes(request: pytest.FixtureRequest) -> list[dict[str, str]]:
    """``classes.csv`` rows as dicts."""
    return read_classes(_folder(request))


@pytest.fixture(scope="module")
def bench(request: pytest.FixtureRequest):
    """``Benchmark.from_dir(<folder>)`` (the strict loader)."""
    from yavih import Benchmark

    return Benchmark.from_dir(_folder(request))


@pytest.fixture(scope="module")
def mm(request: pytest.FixtureRequest) -> ModuleType:
    """The folder's ``make_manifest.py`` as a module."""
    folder = _folder(request)
    return load_module(folder / "make_manifest.py")


@pytest.fixture(scope="module")
def template(request: pytest.FixtureRequest) -> ModuleType:
    """The folder's ``bridge_template.py`` as a module."""
    folder = _folder(request)
    return load_module(folder / "bridge_template.py")

