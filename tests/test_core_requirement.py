"""``core:`` in benchmark.yaml: the PEP 440 specifier of the yavih core releases that can load a folder (not hashed;
checked at load, required by ``bench lint``)."""

from __future__ import annotations

import shutil
from unittest import mock

import pytest

from yavih import lint
from yavih.benchmark import (
    Benchmark,
    CoreVersionError,
    SchemaError,
    check_core,
    core_status,
    packaged_benchmarks,
)
from yavih.cli import exit_code_for


@pytest.mark.parametrize(("version", "ok", "dev"), [("1.1.0", True, False), ("1.9.3", True, False), ("1.0.0", False, False),
                                                    ("2.0.0", False, False), ("1.0.1.dev20+g451fa12", False, True)])
def test_core_status(version, ok, dev):
    st = core_status(">=1.1,<2", version)
    assert (st["ok"], st["dev"]) == (ok, dev)


def test_every_folder_declares_core():
    assert all(Benchmark.load(n).spec.core for n in packaged_benchmarks())


def test_a_release_outside_the_specifier_is_refused_with_the_pip_line_and_a_dev_build_is_not():
    spec = Benchmark.load("esc50").spec
    with mock.patch("yavih.versions.__version__", "1.0.0"), pytest.raises(CoreVersionError) as ei:
        check_core(spec)
    assert 'pip install --no-deps "yavih>=1.1,<2"' in str(ei.value)
    assert exit_code_for(ei.value) == (3, "E_CORE_VERSION")
    with mock.patch("yavih.versions.__version__", "1.0.1.dev3+gabc"):
        check_core(spec)  # a development build is reported, never refused


def test_core_is_not_hashed_and_its_syntax_is_checked(tmp_path):
    src = Benchmark.load("esc50").folder
    folder = tmp_path / "esc50"
    shutil.copytree(src, folder, ignore=shutil.ignore_patterns("__pycache__"))
    y = folder / "benchmark.yaml"
    text = y.read_text()
    before = Benchmark.from_dir(folder).protocol_hash
    y.write_text(text.replace('core: ">=1.1,<2"', 'core: ">=1.1,<3"'))
    assert Benchmark.from_dir(folder).protocol_hash == before
    y.write_text(text.replace('core: ">=1.1,<2"', 'core: "one point one"'))
    with pytest.raises(SchemaError, match="PEP 440"):
        Benchmark.from_dir(folder)
    y.write_text("\n".join(line for line in text.splitlines() if not line.startswith("core:")) + "\n")
    codes = [f.code for f in lint.lint_folder(folder, skip=("conformance", "imports")).findings]
    assert "E_CORE" in codes
