"""Import boundary: no module under ``yavih/`` other than ``probes/``, ``readouts/``,
``bridges/examples/`` and ``testing/__init__.py`` (``yavih.testing``) imports torch at module scope.

Each core module is imported in a fresh subprocess with ``sys.modules["torch"] = None`` (so any
``import torch`` raises ``ImportError``), and the set of modules is discovered from the source tree
so new files are covered automatically.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "yavih"

#: Exemptions (module-scope torch allowed).
TORCH_ALLOWED_PREFIXES = ("yavih.probes", "yavih.readouts", "yavih.bridges.examples")
TORCH_ALLOWED_MODULES = {"yavih.testing"}


def _all_modules() -> list[str]:
    mods = []
    for p in sorted(PKG.rglob("*.py")):
        rel = p.relative_to(SRC).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mods.append(".".join(parts))
    return mods


def _core_modules() -> list[str]:
    out = []
    for m in _all_modules():
        if m in TORCH_ALLOWED_MODULES or m.startswith(TORCH_ALLOWED_PREFIXES):
            continue
        out.append(m)
    return out


CORE = _core_modules()
BLOCK = "import sys; sys.modules['torch'] = None\n"


def test_module_discovery_sees_the_package_layout():
    names = set(_all_modules())
    for required in [
        "yavih", "yavih.versions", "yavih.config", "yavih.hashing", "yavih.benchmark", "yavih.sample", "yavih.bridge",
        "yavih.extract", "yavih.cli", "yavih.testing", "yavih.testing.folder", "yavih.freeze", "yavih.external", "yavih.media.rules", "yavih.media.info",
        "yavih.media.decode", "yavih.media.tarindex", "yavih.media.ffmpeg", "yavih.io.format", "yavih.io.writer",
        "yavih.io.reader", "yavih.io.sidecar", "yavih.io.shards", "yavih.io.validate", "yavih.io.finalize",
        "yavih.io.inspect", "yavih.io.diff", "yavih.probes.attentive", "yavih.probes.dense", "yavih.probes.trainer",
        "yavih.probes.config", "yavih.probes.floors", "yavih.probes.ckpt", "yavih.metrics.registry",
        "yavih.metrics.classification", "yavih.metrics.segment", "yavih.metrics.depth", "yavih.metrics.box",
        "yavih.metrics.intervals", "yavih.readouts.base", "yavih.readouts.clf",
        "yavih.readouts.seq", "yavih.readouts.grid", "yavih.readouts.box", "yavih.report.index", "yavih.report.compare",
        "yavih.report.tables", "yavih.report.migrate", "yavih.report.store", "yavih.report.join", "yavih.plan.matrix",
        "yavih.plan.units", "yavih.plan.estimate", "yavih.plan.exec", "yavih.plan.status", "yavih.plan.calibrate",
        "yavih.bridges.examples.hf_reference", "yavih.benchmarks",
    ]:
        assert required in names, required


@pytest.mark.parametrize("module", CORE)
def test_core_module_imports_without_torch(module):
    code = BLOCK + f"import importlib; importlib.import_module({module!r}); print('ok')"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", f"{module} failed with torch blocked:\n{proc.stderr}"


def test_import_yavih_does_not_load_torch_or_extras():
    """``import yavih`` (core re-exports + lazy subpackages) must not pull torch, sklearn, av, PIL or soundfile."""
    code = (
        "import sys, yavih\n"
        "loaded = [m for m in ('torch', 'sklearn', 'av', 'PIL', 'soundfile', 'scipy') if m in sys.modules]\n"
        "print(','.join(loaded))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "", f"import yavih loaded: {proc.stdout.strip()}"


def test_lazy_subpackages_via_pep562():
    code = (
        BLOCK + "import yavih\n"
        "assert 'yavih.plan' not in __import__('sys').modules\n"
        "p = yavih.plan; assert p.__name__ == 'yavih.plan'\n"
        "r = yavih.report; assert r.__name__ == 'yavih.report'\n"
        "try:\n    yavih.nope\nexcept AttributeError:\n    print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


def test_core_modules_import_together_with_torch_blocked():
    """The import-boundary CI line, minus the wheel build."""
    code = BLOCK + ("import yavih, yavih.versions, yavih.hashing, yavih.io, yavih.benchmark, yavih.extract, yavih.metrics.registry, "
                    "yavih.media.rules, yavih.external; print('ok')")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


def test_protocol_and_probe_hash_parity_across_envs(tmp_path):
    """``protocol_hash`` (and the pinned ``probe_hash``) computed in a torch-free
    process equal the ones computed here, so the model env and the controller env agree."""
    from yavih import hashing
    from yavih import testing as T
    from yavih.benchmark import Benchmark
    from yavih.probes.config import ProbeConfig

    folder = T.make_synthetic_benchmark(tmp_path, "clf", n_train=4, n_eval=2)
    bench = Benchmark.from_dir(folder)
    want = [bench.protocol_hash, hashing.probe_hash(ProbeConfig.pinned("clf", bench))]
    assert bench.folder_source["reader"] is not None  # the folder ships a reader.py: provenance, computed in both envs
    code = BLOCK + (
        "from yavih.benchmark import Benchmark\n"
        "from yavih import hashing\n"
        "from yavih.probes.config import ProbeConfig\n"
        f"b = Benchmark.from_dir({str(folder)!r})\n"
        "print(b.protocol_hash)\n"
        "print(hashing.probe_hash(ProbeConfig.pinned('clf', b)))\n"
        "import sys; assert sys.modules['torch'] is None\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == want


def test_core_modules_do_not_mention_torch_import_at_top_level():
    """Static guard: no `import torch` / `from torch` line outside a function body in a core module."""
    offenders = []
    for m in CORE:
        path = SRC / Path(*m.split(".")).with_suffix(".py")
        if not path.exists():
            path = SRC / Path(*m.split(".")) / "__init__.py"
        for line in path.read_text().splitlines():
            if line.startswith(("import torch", "from torch")):
                offenders.append(f"{m}: {line.strip()}")
    assert not offenders, offenders


# --------------------------------------------------------------------------- extras and the wheel

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
AUTHOR_PACKAGES = ("vggsounder", "probe3d", "evals")


def test_no_extra_for_an_original_author_package():
    """Original-author packages are NEVER yavih dependencies or extras."""
    from yavih.config import tomllib  # stdlib tomllib, or the 3.10 fallback

    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    extras = project.get("optional-dependencies") or {}
    assert set(extras) <= {"media", "probe", "dev"}, sorted(extras)
    deps = list(project.get("dependencies") or []) + [d for v in extras.values() for d in v]
    for pkg in AUTHOR_PACKAGES:
        assert not any(pkg in d.lower() for d in deps), f"{pkg} appears in pyproject dependencies {deps}"


def test_installed_metadata_has_no_author_package_extra():
    """The installed dist-info agrees with pyproject (a stale editable install would still advertise a removed
    extra such as yavih[vggsounder])."""
    import importlib.metadata as md

    try:
        reqs = md.requires("yavih") or []
    except md.PackageNotFoundError:
        pytest.skip("yavih is not installed as a distribution (source tree on sys.path)")
    for pkg in AUTHOR_PACKAGES:
        assert not any(pkg in r.lower() for r in reqs), f"installed yavih metadata still requires {pkg}: {reqs}"


@pytest.mark.integration
def test_wheel_imports_without_torch_and_hashes_match(tmp_path):
    """The built wheel imports with torch blocked and every packaged protocol_hash (11 folders + 3
    aliases) equals the source tree's (parity across envs)."""
    import importlib.util
    import json
    import shutil
    import zipfile

    if importlib.util.find_spec("wheel") is None and importlib.util.find_spec("setuptools") is None:
        pytest.skip("no wheel/setuptools to build with")
    # build from a copy: an in-tree build would leave build/ and *.egg-info in the source tree
    tree = tmp_path / "tree"
    repo = PYPROJECT.parent
    shutil.copytree(repo / "src", tree / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "_version.py"))
    for f in ("pyproject.toml", "LICENSE", "README.md"):
        shutil.copy2(repo / f, tree / f)
    out = tmp_path / "dist"
    proc = subprocess.run([sys.executable, "-m", "pip", "wheel", str(tree), "--no-deps", "--no-build-isolation", "-w", str(out)],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        pytest.skip(f"pip wheel unavailable here: {proc.stderr[-500:]}")
    wheels = sorted(out.glob("yavih-*.whl"))
    assert wheels, proc.stdout[-1000:]
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(wheels[-1]) as zf:
        names = zf.namelist()
        assert not any(n.endswith(".pyc") for n in names)
        # a folder's maintainer notes/ never ships; setuptools-scm writes the version file (its fallback version
        # and no commit here: the copied tree is no git checkout)
        assert not [n for n in names if n.startswith("yavih/benchmarks/") and "/notes/" in n]
        assert "yavih/_version.py" in names
        # the shipped suites sit next to yavih/suites.py (the module wins the import; the directory is data)
        assert {"yavih/suites.py", "yavih/suites/minimal.yaml", "yavih/suites/default.yaml"} <= set(names)
        assert any(n.startswith("yavih-0.0.0.dist-info/licenses/LICENSE") for n in names), names[-10:]
        zf.extractall(unpacked)
    from yavih.benchmark import Benchmark
    from yavih.benchmarks import packaged_names

    refs = [*packaged_names(), "imagenet1k", "imagenet"]
    src_hashes = {r: Benchmark.load(r).protocol_hash for r in refs}
    code = (
        "import sys, json; sys.modules['torch'] = None\n"
        f"sys.path.insert(0, {str(unpacked)!r})\n"
        "import yavih; from yavih.benchmark import Benchmark\n"
        f"assert yavih.__file__.startswith({str(unpacked)!r}), yavih.__file__\n"
        "from yavih.suites import load_suite; assert load_suite('default').origin == 'shipped'\n"
        f"print(json.dumps({{r: Benchmark.load(r).protocol_hash for r in {refs!r}}}))\n"
    )
    proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=300,
                          env={"PATH": "/usr/bin:/bin", "YAVIH_CONFIG": str(tmp_path / "none.toml"), "HOME": str(tmp_path)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == src_hashes
