"""Benchmark folders extend core on every front, with no core edit: a folder scorer that trains its own head with its
own loss (``Scorer.build_head`` / ``Scorer.loss_fn``), its own synthetic stand-in for ``bench trial`` (``testing.py``),
and the trial's ``not_applicable`` path for a scorer no stand-in covers."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from yavih import testing as T
from yavih.benchmark import Benchmark
from yavih.readouts import registry


@pytest.fixture(autouse=True)
def _reset_registry():
    yield
    registry.reset()


TOY_EVAL = '''
"""A folder-owned probe scorer: its own head family (a mean-pooled MLP) and its own loss, trained by the core
trainer through the clf driver (family clf: clf's metrics, floors and conformance population)."""

from yavih.readouts import clf
from yavih.readouts.registry import Scorer

HEAD_FAMILY = "toyfold_mlp_v1"
LOSS = "toyfold_ce"
CALLS = {"build_head": 0, "loss_fn": 0}


def _head(num_classes=None, **_):
    return {"family": HEAD_FAMILY, "out": num_classes, "hidden": 32}


def _build_head(cfg, *, visual_tag=None, conditioning_tag=None):
    CALLS["build_head"] += 1
    out, hidden = int(cfg.head["out"]), int(cfg.head["hidden"])

    def build(dim):
        import torch

        class MeanMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(int(dim), hidden), torch.nn.GELU(), torch.nn.Linear(hidden, out))

            def forward(self, x, pos=None):
                return self.net(x.float().mean(dim=1))

        return MeanMLP()

    return build


def _loss_fn(name):
    CALLS["loss_fn"] += 1
    import torch.nn.functional as F

    def ce(pred, target):
        return F.cross_entropy(pred, target.long(), reduction="none")

    return ce


SCORER = Scorer(
    name="toyfold", version=1, module=__name__, kind="probe", family="clf",
    label_kinds=frozenset({"single"}), requires_t_bins=False, requires_grid=False,
    train=dict(lr=3e-3, lr_grid=(), weight_decay=0.0, max_epochs=30, batch_size=16, patience=30, warmup_fraction=0.0),
    head=_head, loss=lambda label_kind: LOSS, head_families=frozenset({HEAD_FAMILY}), losses=frozenset({LOSS}),
    geometry=lambda cfg: {"num_classes": cfg.head.get("out"), "label_kind": "single"},
    floor_outputs=clf.SCORER.floor_outputs, task_sizes=clf.SCORER.task_sizes,
    build_head=_build_head, loss_fn=_loss_fn, score=lambda ctx: clf.run_clf(ctx),
)
'''

TOY_TESTING = '''
"""The folder's own stand-in for ``yavih bench trial --synthetic``: a copy of the folder itself (its media are generated
in an absolute root), eval.py included."""

import shutil
from pathlib import Path


def make_stand_in(work, real, *, n_eval, n_train):
    dst = Path(work) / real.name
    shutil.copytree(real.folder, dst, ignore=shutil.ignore_patterns("__pycache__", "conformance"))
    return dst
'''


def _toy_folder(tmp_path: Path) -> Path:
    folder = T.make_synthetic_benchmark(tmp_path, "clf", name="toyfold", n_train=48, n_eval=24)
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y["scorer"] = "toyfold"
    y["budget"]["probe_minutes_ref"] = {"toyfold": 1}
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (folder / "eval.py").write_text(textwrap.dedent(TOY_EVAL), encoding="utf-8")
    return folder


def test_a_folder_probe_scorer_owns_its_head_and_loss(tmp_path, yavih_home):
    from yavih.probes.config import ProbeConfig

    b = Benchmark.load(str(_toy_folder(tmp_path)))
    scorer = registry.get("toyfold")
    assert b.spec.readout == "toyfold" and scorer.is_probe and scorer.family == "clf"
    cfg = ProbeConfig.pinned("toyfold", b)
    assert cfg.head == {"family": "toyfold_mlp_v1", "out": b.num_classes, "hidden": 32} and cfg.loss == "toyfold_ce" and cfg.is_pinned()
    torch = pytest.importorskip("torch")
    from yavih.probes.losses import per_sample_loss
    from yavih.probes.trainer import build_head_for

    head = build_head_for(cfg)(8)
    out = head(torch.zeros(3, 5, 8))
    assert tuple(out.shape) == (3, b.num_classes)
    assert tuple(per_sample_loss("toyfold_ce")(out, torch.zeros(3)).shape) == (3,)
    with pytest.raises(ValueError, match="unknown loss"):
        per_sample_loss("nobody_declares_this")


def test_a_folder_probe_scorer_runs_the_whole_chain_on_its_own_stand_in(tmp_path, yavih_home):
    """``bench trial --synthetic`` on a folder whose scorer trains its own head: the folder's ``testing.py`` builds the
    stand-in, the core trainer calls the folder's head builder and loss, and the oracle clears the floor."""
    pytest.importorskip("torch")
    from yavih import conformance
    from yavih.trial import run_trial

    folder = _toy_folder(tmp_path / "src")
    (folder / "testing.py").write_text(textwrap.dedent(TOY_TESTING), encoding="utf-8")
    conformance.record(Benchmark.load(str(folder)))
    rec = run_trial(str(folder), readout_fast=False)
    assert rec["stand_in"] == "folder", rec["limits"]
    assert rec["ok"], [s for s in rec["stages"] if s["status"] != "ok"]
    assert [s["status"] for s in rec["stages"]] == ["ok"] * 9
    assert rec["headline"]["clears_floor"] and rec["headline"]["metric"] == "acc"
    mod = next(m for m in __import__("sys").modules.values() if getattr(m, "HEAD_FAMILY", None) == "toyfold_mlp_v1" and m.CALLS["build_head"])
    assert mod.CALLS["build_head"] >= 1 and mod.CALLS["loss_fn"] >= 1


def test_trial_without_a_stand_in_runs_conformance_and_is_not_applicable(tmp_path, yavih_home, monkeypatch):
    """A scorer neither the core fixtures nor the folder cover: conformance runs, every later stage is
    ``not_applicable``, the trial is ok (exit 0), never ``E_TRIAL_NOT_READY``."""
    from yavih import conformance, trial

    folder = _toy_folder(tmp_path)
    conformance.record(Benchmark.load(str(folder)))
    monkeypatch.setattr(T, "READOUTS", tuple(r for r in T.READOUTS if r != "toyfold"))
    rec = trial.run_trial(str(folder))
    assert rec["stand_in"] == "none" and rec["ok"] and rec["exit_code"] == 0
    statuses = {s["stage"]: s["status"] for s in rec["stages"]}
    assert statuses["conformance"] == "ok" and {statuses[s] for s in trial.STAGES[1:]} == {"not_applicable"}
    assert "toyfold" in trial.format_trial(rec) and "not_applicable" in trial.format_trial(rec)


# ----------------------------------------------------------------------------- A2: open vocabularies


def _codes(folder: Path, check: str) -> dict[str, list[str]]:
    from yavih.lint import lint_folder

    rep = lint_folder(folder)
    found = [f for f in [*rep.errors, *rep.warnings] if f.check == check]
    out: dict[str, list[str]] = {}
    for f in found:
        out.setdefault(f.code, []).append(f.message)
    return out


def _edit_yaml(folder: Path, **changes) -> None:
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    for dotted, value in changes.items():
        node = y
        *path, last = dotted.split("__")
        for k in path:
            node = node[k]
        node[last] = value
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_a_folder_declares_its_own_capability_tag(tmp_path, yavih_home):
    from yavih import doctables
    from yavih.suites import validate_suite

    folder = _toy_folder(tmp_path)
    _edit_yaml(folder, capability=["audio", "foley-sync"])
    assert Benchmark.load(str(folder)).spec.capability == ("audio", "foley-sync")
    codes = _codes(folder, "capability")
    assert set(codes) == {"W_CAPABILITY_UNKNOWN"} and "foley-sync" in codes["W_CAPABILITY_UNKNOWN"][0]
    assert list(doctables._capability_groups({"toyfold": folder})) == ["audio", "foley-sync"]
    _edit_yaml(folder, capability=["Not A Tag"])
    assert set(_codes(folder, "capability")) == {"E_CAPABILITY"}
    suite = validate_suite(yaml.safe_load("schema_version: 1\nname: s\ndescription: d\ngroups:\n  - capability: foley-sync\n"
                                          "    entries:\n      - {benchmark: esc50, readings: [a]}\n"), where="test")
    assert [g.capability for g in suite.groups] == ["foley-sync"]


def test_a_folder_declares_its_own_readings(tmp_path, yavih_home):
    folder = _toy_folder(tmp_path)
    tpl = folder / "bridge_template.py"
    src = tpl.read_text(encoding="utf-8")
    import re

    tpl.write_text(re.sub(r'"reading": "[^"]+",', '"reading": "depth",', src, count=1), encoding="utf-8")
    codes = _codes(folder, "template")
    assert set(codes) == {"W_READING_UNKNOWN"} and "depth" in codes["W_READING_UNKNOWN"][0]
    with (folder / "eval.py").open("a", encoding="utf-8") as fh:
        fh.write('\nREADINGS = ("depth",)\n')
    assert _codes(folder, "template") == {}


TAGGED_EVAL = '''
from yavih.readouts.registry import Scorer

SCORER = Scorer(name="tagfold", version=1, module=__name__, kind="direct", label_kinds=frozenset({"tag_json"}),
                requires_t_bins=False, requires_grid=False, score=lambda ctx: {"metrics": {}}, needs_gpu=False)
'''


def test_a_label_kind_is_valid_when_the_folder_scorer_declares_it(tmp_path, yavih_home):
    folder = T.make_synthetic_benchmark(tmp_path, "clf", name="tagfold", n_train=8, n_eval=8)
    y = yaml.safe_load((folder / "benchmark.yaml").read_text(encoding="utf-8"))
    y["manifest"]["label"]["kind"] = "tag_json"
    y["metrics"].update(headline="n_eval", list=["n_eval"], floors=[], controls=[], chance={}, ci={})
    (folder / "benchmark.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True), encoding="utf-8")
    from yavih.benchmark import SchemaError

    with pytest.raises(SchemaError, match="label kind outside the core vocabulary|needs kind"):
        Benchmark.load(str(folder))  # clf does not score it
    _edit_yaml(folder, scorer="tagfold", budget__probe_minutes_ref={"tagfold": 1})
    (folder / "eval.py").write_text(textwrap.dedent(TAGGED_EVAL), encoding="utf-8")
    b = Benchmark.load(str(folder))
    m = b.manifest()
    col = y["manifest"]["label"]["column"]
    rows = T.read_manifest_rows(folder)
    assert b.spec.manifest.label.kind == "tag_json" and m.label_ids == [r[col] for r in rows]  # the cells as stored
    _edit_yaml(folder, manifest__label__kind="Bad Kind")
    with pytest.raises(SchemaError, match="kind"):
        Benchmark.load(str(folder))


# ----------------------------------------------------------------------------- A3: folder-declared checks


GATE_EVAL = '''
PROTOCOL_KEYS = frozenset({"toy_refuse"})
CALLED = []


def toy_gate(gc):
    """Re-encodes the first smoke clip alone and compares it with pass A."""
    import numpy as np

    again = gc.encode([gc.smoke[0]])[0]
    first = gc.outputs[min(gc.outputs)]
    CALLED.append(len(gc.smoke))
    if not np.array_equal(np.asarray(again[0]), np.asarray(first[0])):
        gc.fail("E_TOY_GATE", "the first clip encodes differently alone")
    return {"clips": len(gc.outputs)}


def refusing_gate(gc):
    gc.fail("E_TOY_GATE", "this gate refuses every bridge")


BRIDGE_CHECKS = (toy_gate,)


def eligible(view):
    if view.protocol.get("toy_refuse"):
        return ["toy_rule"], ["the folder's toy rule refuses this feature set"]
    return []
'''


def _gate_folder(tmp_path: Path, *, refuse: bool = False) -> Path:
    folder = T.make_synthetic_benchmark(tmp_path, "clf", name="gatefold", n_train=8, n_eval=8)
    _edit_yaml(folder, protocol__toy_refuse=refuse)
    (folder / "eval.py").write_text(textwrap.dedent(GATE_EVAL), encoding="utf-8")
    return folder


def test_folder_bridge_checks_run_after_the_hard_checks(tmp_path, yavih_home):
    from yavih.extract import ExtractError, bridge_check

    folder = _gate_folder(tmp_path)
    bridge = T.write_bridge_file(tmp_path / "b" / "oracle.py", "oracle", D=16, bench_folder=folder)
    prof = bridge_check(bridge=str(bridge), bench=str(folder), n=4, out=tmp_path / "b" / "p.json")
    assert prof["gates"] == {"toy_gate": {"clips": 4}} and "stream" not in prof
    src = (folder / "eval.py").read_text(encoding="utf-8")
    (folder / "eval.py").write_text(src.replace("BRIDGE_CHECKS = (toy_gate,)", "BRIDGE_CHECKS = (toy_gate, refusing_gate)"), encoding="utf-8")
    with pytest.raises(ExtractError) as e:
        bridge_check(bridge=str(bridge), bench=str(folder), n=4)
    assert e.value.code == "E_TOY_GATE"
    (folder / "eval.py").write_text(src.replace("BRIDGE_CHECKS = (toy_gate,)", 'BRIDGE_CHECKS = ("stream",)'), encoding="utf-8")
    with pytest.raises(ExtractError) as e:
        bridge_check(bridge=str(bridge), bench=str(folder), n=4)
    assert e.value.code == "E_FOLDER_CHECK" and "output_contract.stream" in str(e.value)


def test_stream_folders_keep_their_gates_by_default():
    from yavih.benchmark import Benchmark
    from yavih.extract import _gate_stream, folder_gates

    assert folder_gates(Benchmark.load("thumos14_oad")) == [("stream", _gate_stream)]
    assert folder_gates(Benchmark.load("esc50")) == []


def test_a_folder_eligibility_rule_applies_after_the_scorers(tmp_path, yavih_home):
    from yavih.extract import run_extract
    from yavih.io.validate import validate_features

    for refuse in (False, True):
        folder = _gate_folder(tmp_path / str(refuse), refuse=refuse)
        bridge = T.write_bridge_file(tmp_path / str(refuse) / "random.py", "random", D=16, bench_folder=folder)
        run = tmp_path / str(refuse) / "run"
        assert run_extract(bench=str(folder), bridge=str(bridge), split="val", shard=(0, 1), out=run, canaries=2)["status"] == "ok"
        rep = validate_features(run / "features" / "val", bench=Benchmark.load(str(folder)))
        assert rep.ok and rep.eligibility["eligible"] is (not refuse)
        if refuse:
            assert rep.eligibility["missing"] == ["toy_rule"] and "toy rule" in rep.eligibility["reason"]


# ----------------------------------------------------------------------------- A5: discovery


def test_bench_extensions_and_the_extends_line(capsys):
    import json

    from yavih import cli

    assert cli.main(["--json", "bench", "extensions"]) == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["benchmarks"]["vggsounder_v2a"]["scorer"]["kind"] == "direct" and "esc50" not in data["benchmarks"]
    assert data["benchmarks"]["thumos14_oad"]["checks"] == ["stream (default)"] and data["doc"].endswith("extending-through-a-folder.md")
    assert cli.main(["--json", "bench", "extensions", "--all"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["benchmarks"]["esc50"] == {}
    assert cli.main(["bench", "info", "esc50"]) == 0
    assert "extends: nothing (core scorer clf)" in capsys.readouterr().out
    assert cli.main(["--json", "bench", "extensions", "nope_not_a_bench"]) == 1


def test_a_scaffold_like_a_folder_scorer_owns_it_under_its_new_name(tmp_path, yavih_home):
    """``bench new --like`` a folder that owns its scorer: the copy's eval.py SCORER, its metrics and the yaml carry the
    new name, so the scaffold loads; ``--scorer stream`` (no table entry) finds a folder that uses it."""
    from yavih.scaffold import _first_folder_of, new_benchmark

    src = _toy_folder(tmp_path / "src")
    folder = new_benchmark("toycopy", like=str(src), dir=tmp_path / "out")
    b = Benchmark.load(str(folder))
    assert b.spec.readout == "toycopy" and registry.get("toycopy").is_probe
    assert 'name="toycopy"' in (folder / "eval.py").read_text(encoding="utf-8")
    assert _first_folder_of("stream") == "ave_causal" and _first_folder_of("no_such_scorer") is None


# ----------------------------------------------------------------------------- A6: the detection folders own their scorer


DETECTION_FOLDERS = ("unav100", "perception_test_sound_loc", "perception_test_action_loc", "epic_sounds_det")


def test_the_detection_folders_own_one_scorer_implementation():
    """The four detection folders carry byte-identical copies of their scorer (``detection.py``), each registered under the
    folder's name as a probe with its own loss; the core ships no detection readout."""
    import importlib.util

    from yavih.benchmarks import packaged_folder
    from yavih.extensions import folder_extensions

    copies = {n: (packaged_folder(n) / "detection.py").read_bytes() for n in DETECTION_FOLDERS}
    assert len(set(copies.values())) == 1, "detection.py differs between folders: keep the copies identical (notes/UNIFY.md)"
    for n in DETECTION_FOLDERS:
        ext = folder_extensions(Benchmark.load(n))
        assert ext["scorer"]["name"] == n and ext["scorer"]["kind"] == "probe" and ext["scorer"]["losses"] == [f"{n}_focal_giou"]
    assert all(importlib.util.find_spec(m) is None for m in ("yavih.readouts.tad", "yavih.readouts.tdet", "yavih.metrics.detection"))
