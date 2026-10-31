"""Adding task types and benchmarks without editing core: the readout registry,
``$YAVIH_BENCH_PATH`` and the ``yavih.benchmarks`` entry-point group, and ``yavih bench new``."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from yavih import cli
from yavih.benchmark import Benchmark, SchemaError, resolve_folder
from yavih.benchmarks import external_folders, packaged_folder
from yavih.readouts import registry
from yavih.versions import READOUT_VERSION, READOUTS


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("YAVIH_CONFIG", "/nonexistent/yavih-config.toml")
    monkeypatch.setenv("YAVIH_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("YAVIH_BENCH_PATH", raising=False)
    yield
    registry.reset()


# ----------------------------------------------------------------------------- readout registry


def test_builtin_readouts_describe_themselves():
    # the built-ins come first; a folder's SCORER registers when its folder loads (test order decides what follows)
    builtin = ("clf", "seq", "grid", "box")
    assert registry.names()[:4] == builtin and list(READOUTS)[:4] == list(builtin)
    assert {k: READOUT_VERSION[k] for k in builtin} == {"clf": 1, "seq": 2, "grid": 2, "box": 2}  # hashed: unchanged by the refactor
    assert all(registry.get(n).kind != "probe" for n in registry.names()[4:])
    for name in builtin:
        r = registry.get(name)
        assert r.module == f"yavih.readouts.{name}" and callable(r.driver()) and r.floor_outputs is not None
        assert set(r.train) == {"lr", "lr_grid", "weight_decay", "max_epochs", "batch_size", "patience", "warmup_fraction"}
    assert [n for n in builtin if registry.get(n).dense] == ["seq", "grid", "box"]
    with pytest.raises(registry.UnknownScorer):
        registry.get("nope")


def test_readout_descriptors_import_without_torch():
    import subprocess

    code = ("import sys; sys.modules['torch'] = None\n"
            "from yavih.readouts import registry; from yavih.benchmark import Benchmark\n"
            "print(registry.names(), Benchmark.load('esc50').protocol_hash[:12])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "('clf', 'seq', 'grid', 'box')" in out.stdout, out.stderr


def _toy_readout(module: str = "toy_readout_mod") -> registry.Scorer:
    return registry.Scorer(
        name="toy", version=7, module=module, kind="probe", label_kinds=frozenset({"single"}), requires_t_bins=False, requires_grid=False,
        train=dict(lr=1e-3, lr_grid=(), weight_decay=0.0, max_epochs=1, batch_size=8, patience=1, warmup_fraction=0.0),
        head=lambda **kw: {"family": "toy_head_v1", "out": kw.get("num_classes")}, loss=lambda label_kind: "toy_loss",
        head_families=frozenset({"toy_head_v1"}), losses=frozenset({"toy_loss"}), geometry=lambda cfg: {"num_classes": cfg.head.get("out")},
        floor_outputs=lambda t, n, **kw: np.zeros((n, 1), np.float32),
    )


def _toy_folder(tmp_path: Path) -> Path:
    dst = tmp_path / "toy_bench"
    shutil.copytree(packaged_folder("in1k"), dst, ignore=shutil.ignore_patterns("__pycache__", "notes", "tests"))
    y = yaml.safe_load((dst / "benchmark.yaml").read_text(encoding="utf-8"))
    y.update(name="toy_bench", aliases=[], scorer="toy")
    y["budget"]["probe_minutes_ref"] = {"toy": 1}
    y["metrics"].update(headline="n_eval", list=["n_eval"], floors=[], controls=[], chance={}, ci={})
    (dst / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    return dst


def test_a_registered_readout_is_a_first_class_task_type(tmp_path, monkeypatch):
    """A readout from an entry point validates yamls, hashes, pins its probe config and builds its floor: no core edit."""
    from yavih.probes.config import ProbeConfig
    from yavih.probes.floors import floor_outputs

    folder = _toy_folder(tmp_path)
    with pytest.raises(SchemaError, match="scorer"):
        Benchmark.load(str(folder))  # not registered yet
    mod = types.ModuleType("toy_readout_mod")
    mod.SCORER = _toy_readout()
    monkeypatch.setitem(sys.modules, "toy_readout_mod", mod)
    monkeypatch.setattr(registry, "_entry_point_modules", lambda: [("toy", "toy_readout_mod")])
    registry.reset()
    assert "toy" in registry.names() and READOUT_VERSION["toy"] == 7
    b = Benchmark.load(str(folder))
    assert b.spec.readout == "toy" and len(b.protocol_hash) == 64
    cfg = ProbeConfig.pinned("toy", num_classes=3)
    assert cfg.head == {"family": "toy_head_v1", "out": 3} and cfg.loss == "toy_loss" and cfg.is_pinned()
    assert floor_outputs("toy", None, 4, protocol={}, label_kind="single").shape == (4, 1)


def test_metrics_of_another_readout_are_refused_for_a_registered_one(tmp_path):
    registry.register(_toy_readout())
    folder = _toy_folder(tmp_path)
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y["metrics"].update(headline="acc", list=["acc", "n_eval"])
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    with pytest.raises(SchemaError, match="not allowed for scorer 'toy'"):
        Benchmark.load(str(folder))
    # n_eval serves every readout (MetricSpec.readouts = {"*"})
    from yavih.metrics.registry import REGISTRY, allows

    assert allows(REGISTRY["n_eval"], "toy") and not allows(REGISTRY["acc"], "toy")


# ----------------------------------------------------------------------------- benchmark discovery


def _copy_as(tmp: Path, src: str, name: str) -> Path:
    dst = tmp / name
    shutil.copytree(packaged_folder(src), dst, ignore=shutil.ignore_patterns("__pycache__"))
    y = (dst / "benchmark.yaml").read_text(encoding="utf-8")
    (dst / "benchmark.yaml").write_text(y.replace(f"\nname: {src}\n", f"\nname: {name}\n"), encoding="utf-8")
    return dst


def test_yavih_bench_path_resolves_but_never_shadows_a_packaged_folder(tmp_path, monkeypatch, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    first = _copy_as(a, "esc50", "esc50_local")
    _copy_as(b, "esc50", "esc50_local")  # the first directory wins
    _copy_as(a, "esc50", "esc50")  # a packaged name: ignored
    monkeypatch.setenv("YAVIH_BENCH_PATH", f"{a}:{b}")
    assert external_folders()["esc50_local"] == first
    assert resolve_folder("esc50_local") == first
    assert resolve_folder("esc50") == packaged_folder("esc50")
    assert Benchmark.load("esc50_local").name == "esc50_local"
    assert cli.main(["--json", "bench", "path", "esc50_local"]) == 0  # strict config: YAVIH_BENCH_PATH is a known variable
    assert json.loads(capsys.readouterr().out)["data"]["folder"] == str(first.resolve())
    assert cli.main(["--json", "bench", "list", "--fmt", "md"]) == 0
    assert "esc50_local" in json.loads(capsys.readouterr().out)["data"]["names"]


def test_benchmark_entry_point_pack(tmp_path, monkeypatch):
    import yavih.benchmarks as pkg

    pack = tmp_path / "pack"
    pack.mkdir()
    folder = _copy_as(pack, "esc50", "esc50_pack")
    monkeypatch.setattr(pkg, "_entry_point_roots", lambda: [("entry point demo", pack)])
    assert resolve_folder("esc50_pack") == folder and Benchmark.load("esc50_pack").name == "esc50_pack"


def test_bench_path_is_a_known_env_var():
    from yavih.config import ENV_VARS

    assert "YAVIH_BENCH_PATH" in ENV_VARS


# ----------------------------------------------------------------------------- bench new


def _conformance_warnings(folder: Path) -> set[str]:
    """``{"W_CONFORMANCE_SKIPPED"}`` when the scaffold's conformance check cannot run here (its external package, Probe3D's
    ``evals`` for a ``grid`` scaffold, is absent: the lint reports that warning), else the empty set."""
    from yavih.conformance import check

    return {"W_CONFORMANCE_SKIPPED"} if check(Benchmark.from_dir(folder)).skipped else set()


@pytest.mark.parametrize("readout", ["clf", "seq", "grid", "box"])
def test_bench_new_scaffold_passes_lint_and_loads(tmp_path, readout, capsys):
    from yavih.lint import lint_folder

    assert cli.main(["--json", "bench", "new", f"new_{readout}", "--scorer", readout, "--dir", str(tmp_path)]) == 0
    env = json.loads(capsys.readouterr().out)
    folder = Path(env["data"]["folder"])
    assert folder == tmp_path / f"new_{readout}" and env["data"]["todo"] > 0
    assert env["data"]["conformance"]["state"] in ("recorded", "copied")
    rep = lint_folder(folder)
    assert rep.ok and {f.code for f in rep.warnings} == {"W_TODO"} | _conformance_warnings(folder)
    b = Benchmark.load(str(folder))
    assert b.name == f"new_{readout}" and b.spec.readout == readout and b.spec.protocol_version == 1
    assert not (folder / "notes").exists() and (folder / "tests" / "test_folder.py").is_file()


def test_bench_new_refuses_bad_requests(tmp_path, capsys):
    assert cli.main(["bench", "new", "Bad-Name", "--scorer", "clf", "--dir", str(tmp_path)]) == 1
    assert cli.main(["bench", "new", "x1", "--scorer", "seq", "--like", "esc50", "--dir", str(tmp_path)]) == 1
    assert cli.main(["bench", "new", "esc50", "--scorer", "clf", "--dir", str(tmp_path)]) == 1  # resolves already
    assert cli.main(["bench", "new", "x2", "--dir", str(tmp_path)]) == 1  # neither --readout nor --like
    assert cli.main(["bench", "new", "x3", "--like", "avsync", "--dir", str(tmp_path)]) == 0
    assert cli.main(["bench", "new", "x3", "--like", "avsync", "--dir", str(tmp_path)]) == 1  # exists


def test_bench_new_marks_copied_scripts_and_skips_curation(tmp_path):
    """``--like nyuv2_depth`` copies its eval.py with a ``TODO`` banner as the first line (a ``todo`` finding), never
    its ``curation.json``; the generated tests and the prepare stub carry ``TODO:`` markers the lint reports."""
    from yavih.lint import lint_folder
    from yavih.scaffold import new_benchmark

    folder = new_benchmark("new_depth", like="nyuv2_depth", dir=tmp_path)
    assert (packaged_folder("nyuv2_depth") / "curation.json").is_file() and not (folder / "curation.json").exists()
    first = (folder / "eval.py").read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("# TODO: copied verbatim from nyuv2_depth by `yavih bench new`")
    rep = lint_folder(folder)
    assert rep.ok and {f.code for f in rep.warnings} == {"W_TODO"} | _conformance_warnings(folder)
    assert {"eval.py", "tests/test_folder.py", "prepare/01_obtain.sh"} <= {f.file for f in rep.warnings}
    # the fixture: recorded here, or nyuv2_depth's copied with this folder's identity when Probe3D's evals is absent
    rec = json.loads((folder / "conformance" / "expected.json").read_text(encoding="utf-8"))
    assert rec["benchmark"] == "new_depth" and rec["protocol_version"] == 1
    assert (rec["recorded_with"].get("copied_from") == "nyuv2_depth") is bool(_conformance_warnings(folder))
    # capability: --like's tags, under a TODO comment the lint counts
    assert Benchmark.load(str(folder)).spec.capability == ("visual", "spatial")
    cap_line = next(ln for ln in (folder / "benchmark.yaml").read_text(encoding="utf-8").splitlines() if ln.startswith("capability:"))
    assert cap_line.startswith("capability:  # TODO: still nyuv2_depth's"), cap_line


@pytest.mark.parametrize("like", ["esc50", "nyuv2_depth"])
def test_bench_new_generated_tests_pass_or_skip(tmp_path, like):
    """The scaffold's ``tests/test_folder.py`` collects and runs on the fresh scaffold: the shared folder checks pass,
    the tests that need the benchmark's own data skip with a ``TODO:`` reason, nothing fails."""
    import os
    import subprocess

    import yavih
    from yavih.scaffold import new_benchmark

    folder = new_benchmark(f"new_{like}", like=like, dir=tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(yavih.__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, "-m", "pytest", str(folder / "tests"), "-q", "-rs", "-p", "no:cacheprovider"],
                          capture_output=True, text=True, cwd=tmp_path, env=env, timeout=600)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    summary = proc.stdout.strip().splitlines()[-1]
    assert "failed" not in summary and "error" not in summary and " passed" in summary and "2 skipped" in summary, summary
    assert proc.stdout.count("SKIPPED") == 2 and "TODO: " in proc.stdout


# ----------------------------------------------------------------------------- scorer kinds


def _direct_scorer(name: str = "toy_direct", version: int = 3) -> registry.Scorer:
    return registry.Scorer(
        name=name, version=version, module="toy_direct_mod", kind="direct", label_kinds=frozenset({"none"}),
        requires_t_bins=False, requires_grid=False, score=lambda ctx: {"metrics": {}}, needs_gpu=False,
    )


def test_scorer_kinds_and_configuration_hashes():
    """A probe scorer hashes its ProbeConfig; a non-probe scorer's configuration is the constant
    ``scorer_config_hash(name, version)`` (``ScorerConfig``), so the second slot of ``eval_hash`` keeps its shape."""
    from yavih import hashing
    from yavih.probes.config import ProbeConfig
    from yavih.readouts.registry import ScorerConfig

    clf = registry.get("clf")
    cfg = ProbeConfig.pinned("clf", num_classes=3, label_kind="single")
    assert clf.is_probe and clf.gpu and clf.hash_config(cfg) == hashing.probe_hash(cfg)
    assert clf.pinned_config(None).head["family"] == cfg.head["family"] == "attentive_probe_v1"

    d = _direct_scorer()
    assert not d.is_probe and not d.gpu and d.families() == {"toy_direct"} and d.driver()(None) == {"metrics": {}}
    pinned = d.pinned_config(None)
    assert pinned == ScorerConfig(scorer="toy_direct", version=3, kind="direct") and pinned.readout == "toy_direct"
    assert pinned.to_json() == {"scorer": "toy_direct", "version": 3, "kind": "direct"}
    assert d.hash_config(pinned) == hashing.scorer_config_hash("toy_direct", 3) == d.hash_config(None)
    assert hashing.scorer_config_hash("toy_direct", 3) != hashing.scorer_config_hash("toy_direct", 4)
    assert len(hashing.scorer_config_hash("toy_direct", 3)) == 64
    with pytest.raises(TypeError):
        hashing.scorer_config_hash("toy_direct", 0)
    assert registry.Scorer(name="fam", version=1, module="m", kind="stateless", label_kinds=frozenset(), family="clf").families() == {"fam", "clf"}


def test_scorer_descriptor_refusals():
    with pytest.raises(ValueError, match="kind"):
        registry.Scorer(name="x", version=1, module="m", kind="nope", label_kinds=frozenset())
    with pytest.raises(ValueError, match="trains no probe"):
        registry.Scorer(name="x", version=1, module="m", kind="direct", label_kinds=frozenset(), head=lambda **kw: {})
    with pytest.raises(ValueError, match="needs train, head, loss"):
        registry.Scorer(name="x", version=1, module="m", kind="probe", label_kinds=frozenset())
    with pytest.raises(ValueError, match="family"):
        registry.Scorer(name="x", version=1, module="m", kind="direct", label_kinds=frozenset(), family="Bad-Name")


def test_a_non_probe_scorer_has_no_probe_config():
    from yavih.probes.config import ProbeConfig

    registry.register(_direct_scorer())
    with pytest.raises(ValueError, match="trains no probe"):
        ProbeConfig.pinned("toy_direct")


def test_folder_scorers_register_under_their_own_name_and_never_shadow():
    d = _direct_scorer()
    registry.register_folder(d)
    assert registry.get("toy_direct") is d
    registry.register_folder(_direct_scorer(version=4))  # the folder is loaded again: re-registration is fine
    assert registry.get("toy_direct").version == 4
    with pytest.raises(ValueError, match="already a registered scorer"):
        registry.register_folder(_direct_scorer(name="clf"))
    registry.reset()
    with pytest.raises(registry.UnknownScorer):
        registry.get("toy_direct")
