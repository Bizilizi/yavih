"""The version constants (docs/hashing.md), their re-exports, and the stdlib-only rule."""

from __future__ import annotations

import subprocess
import sys

import pytest

from yavih import versions


def test_pinned_values():
    """The pinned version constants."""
    assert versions.FEATURE_FORMAT_VERSION == 5 and versions.READABLE_FEATURE_FORMATS == (4, 5)
    assert versions.BENCHMARK_SCHEMA_VERSION == 3
    assert versions.RESULT_SCHEMA_VERSION == 4
    assert versions.RESULTS_FILE_VERSION == 2
    assert versions.PLAN_SCHEMA_VERSION == 3
    # the built-ins (a folder's eval.py SCORER registers when its folder loads, so later entries depend on test order)
    assert {k: versions.READOUT_VERSION[k] for k in ("clf", "seq", "grid", "box")} == {"clf": 1, "seq": 2, "grid": 2, "box": 2}
    assert versions.PROBE_SCHEMA == "yavih-probe-1"
    assert versions.PROTOCOL_SCHEMA == "yavih-protocol-5"
    assert versions.DESCRIBE_SCHEMA == "yavih-describe-2"
    assert isinstance(versions.__version__, str) and versions.__version__
    assert tuple(versions.READOUTS)[:4] == ("clf", "seq", "grid", "box") and set(versions.READOUTS) == set(versions.READOUT_VERSION)
    assert versions.GIT_HASH is None or isinstance(versions.GIT_HASH, str)


def test_readout_version_is_read_only():
    with pytest.raises(TypeError):
        versions.READOUT_VERSION["clf"] = 2  # type: ignore[index]


def test_package_version_comes_from_the_git_tag():
    """pyproject declares the version dynamic (setuptools-scm, tags ``vX.Y.Z``); ``yavih.__version__`` is the
    installed distribution's version, and ``yavih --version`` prints it."""
    import re
    from importlib import metadata
    from pathlib import Path

    from yavih.config import tomllib  # stdlib tomllib, or the 3.10 fallback

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    assert "version" not in data["project"] and "version" in data["project"]["dynamic"]
    assert data["tool"]["setuptools_scm"]["version_file"] == "src/yavih/_version.py"
    import yavih

    assert yavih.__version__ == versions.__version__
    assert re.match(r"^\d+\.\d+\.\d+", versions.__version__), versions.__version__
    try:
        installed = metadata.version("yavih")
    except metadata.PackageNotFoundError:  # a bare source tree on sys.path
        return
    assert versions.__version__ == installed


def test_version_fallbacks(monkeypatch):
    """No version file -> distribution metadata -> ``0.0.0+unknown``; never raises."""
    import importlib.metadata as md
    import sys

    monkeypatch.setitem(sys.modules, "yavih._version", None)  # import yavih._version -> ImportError
    monkeypatch.setattr(md, "version", lambda name: "7.7.7")
    assert versions._package_version() == "7.7.7"

    def missing(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", missing)
    assert versions._package_version() == versions.UNKNOWN_VERSION == "0.0.0+unknown"


def test_reexports_are_the_same_objects():
    """The named modules re-export, nothing else defines a version."""
    from yavih import benchmark
    from yavih.io import format as io_format
    from yavih.plan import units
    from yavih.probes import config as probe_config
    from yavih.readouts import base as readouts_base
    from yavih.report import migrate

    assert io_format.FEATURE_FORMAT_VERSION is versions.FEATURE_FORMAT_VERSION
    assert benchmark.BENCHMARK_SCHEMA_VERSION is versions.BENCHMARK_SCHEMA_VERSION
    assert readouts_base.RESULT_SCHEMA_VERSION is versions.RESULT_SCHEMA_VERSION
    assert migrate.RESULT_SCHEMA_VERSION is versions.RESULT_SCHEMA_VERSION
    assert units.PLAN_SCHEMA_VERSION is versions.PLAN_SCHEMA_VERSION
    assert probe_config.PROBE_SCHEMA is versions.PROBE_SCHEMA
    assert readouts_base.READOUT_VERSION is versions.READOUT_VERSION


@pytest.mark.parametrize("readout", ["clf", "seq", "grid", "box"])
def test_readout_modules_reexport_their_version(readout):
    import importlib

    mod = importlib.import_module(f"yavih.readouts.{readout}")
    assert mod.VERSION == versions.READOUT_VERSION[readout]


def test_versions_is_stdlib_only():
    """The module FILE imports only stdlib: loaded by path with numpy, yaml and torch blocked.
    (``import yavih.versions`` necessarily runs ``yavih/__init__``, whose core re-exports need numpy.)"""
    from pathlib import Path

    path = Path(versions.__file__)
    code = (
        "import sys, importlib.util\n"
        "sys.modules['numpy'] = None; sys.modules['yaml'] = None; sys.modules['torch'] = None\n"
        f"spec = importlib.util.spec_from_file_location('yavih_versions_isolated', {str(path)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(m.PROBE_SCHEMA)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "yavih-probe-1"
