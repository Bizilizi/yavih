"""Tests for report joins (the folder ``eval.py`` ``join``, the results store, vggsounder): one ``model``,
three readings, refusals on ``eval_hash`` / population / model, ``W_JOIN_WEIGHTS`` recorded, the entry
``<benchmark>:a+v+av`` in the model's results directory).

``report --join <benchmark> ...`` goes through the benchmark folder's ``eval.py`` ``join()`` hook
(:mod:`yavih.report.join`): the generic tests use a fake folder, the vggsounder tests the packaged folder's
``probe.join``.

The official scorer needs the full 12 008-clip population (a sparse subset empties a GT category and
raises), so the folder join's pkl payload and its ``compute_all_results`` call are exercised against a FAKE
``vggsounder`` module installed into ``sys.modules`` (reported at the pinned version); the real package, when
importable, is only asked to confirm the call signature yavih relies on.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_report import IDS2, PH, PH2, QH, make_result, write_result
from yavih import hashing
from yavih.report.join import JoinError, locate_result_dir, run_join

ROUTINGS = ("a", "v", "av")
VGGSOUNDER_FOLDER = Path(__file__).resolve().parents[1] / "src" / "yavih" / "benchmarks" / "vggsounder"
_READS = {"a": ["audio"], "v": ["video"], "av": ["audio", "video"]}

C = 4
N = 6
CLIP_IDS = [f"clip{i:02d}" for i in range(N)]


def _vgg_result(reading: str, model: str = "avm", **kw):
    r = make_result(bench="vggsounder", model=model, reading=reading, eval_present=N, eval_nominal=N,
                    metrics={"auc_pr": 0.3, "hit_rate@1": 0.5}, floors={}, **kw)
    # the reading's declared streams: a = audio only, v = video only, av = both (the folder join checks them)
    r["bridge"]["describe"]["input"] = {s: {} for s in _READS.get(reading, ["audio", "video"])}
    return r


@pytest.fixture
def three_runs(tmp_path: Path) -> dict[str, str]:
    runs = tmp_path / "runs"
    out = {}
    for key in ROUTINGS:
        p = write_result(runs, _vgg_result(key), with_npy=True)
        # distinct logits per routing so the pkl contents are checkable
        seed_dir = p.parent
        np.save(seed_dir / "eval_logits.npy", np.full((N, C), float(ROUTINGS.index(key) + 1), dtype=np.float32))
        out[key] = str(runs / "vggsounder" / "avm" / key)  # the run directory runs/<bench>/<model>/<reading>/
    return out


class _FakeTable:
    """Mimics the ``(metric, modality)`` MultiIndex DataFrame ``vggsounder.benchmark.benchmark`` returns."""

    def __init__(self, model: str, values: dict[tuple[str, str], float]):
        self.columns = list(values)
        self._row = values
        self._model = model

    @property
    def loc(self):
        table = self

        class _Loc:
            def __getitem__(self, model):
                assert model == table._model
                return table._row

        return _Loc()


@pytest.fixture
def fake_vggsounder(monkeypatch: pytest.MonkeyPatch):
    """Installs a fake ``vggsounder`` package recording the ``benchmark()`` call and the pkl payload."""
    calls: dict = {}

    def benchmark(models_path, display_names, metrics, subset_ids=None, dataset_path=None, vggsounder=None, verbose=True):
        pkls = sorted(Path(models_path).glob("*.pkl"))
        assert len(pkls) == 1
        with open(pkls[0], "rb") as fh:
            payload = pickle.load(fh)
        calls.update(models_path=models_path, display_names=display_names, metrics=metrics, subset_ids=subset_ids,
                     vggsounder=vggsounder, verbose=verbose, payload=payload, model=pkls[0].stem)
        if calls.get("raise_key_error"):
            raise KeyError("a only")
        values = {}
        for metric, mods in metrics:
            for m in mods:
                values[(metric, m)] = {"auc_pr": 30.0, "auc_roc": 70.0, "mu": 5.0}[metric] + {"a": 0.1, "v": 0.2, "av": 0.3}[m]
        values[("auc_pr", "a")] = float("nan")  # the scorer leaves NaN for an undefined cell
        return _FakeTable(pkls[0].stem, values)

    class VGGSounder:
        def __init__(self, csv_path=None, modality=None, *, background_music=False, voice_over=None, static_image=None,
                     download_samples=False, cache_dir=None):
            calls["labels_kwargs"] = {"background_music": background_music, "download_samples": download_samples}

    def compute_all_results(models_path, subset_ids, modalities, vggsounder=None, verbose=True):
        pkls = sorted(Path(models_path).glob("*.pkl"))
        assert len(pkls) == 1
        with open(pkls[0], "rb") as fh:
            payload = pickle.load(fh)
        calls.update(models_path=models_path, subset_ids=subset_ids, modalities=list(modalities), vggsounder=vggsounder,
                     verbose=verbose, payload=payload, model=pkls[0].stem)
        if calls.get("raise_key_error"):
            raise KeyError("a only")
        names = [f"{m}@{k}" for k in (1, 3, 5, 10) for m in ("f1", "f1_macro", "accuracy", "jaccard", "precision", "recall", "hit_rate")]
        res = {m: {k: base + {"a": 0.001, "v": 0.002, "av": 0.003}.get(m, 0.0)
                   for k, base in [*((n, 0.5) for n in names), ("auc_roc", 0.70), ("auc_pr", 0.30)]}
               for m in modalities}
        for m in ("a", "v", "av"):  # the package's modality confusion, on a / v / av only
            res[m]["mu"] = 0.05 + {"a": 0.001, "v": 0.002, "av": 0.003}[m]
        return {pkls[0].stem: res}

    pkg = types.ModuleType("vggsounder")
    pkg.__version__ = "1.0"
    bench_mod = types.ModuleType("vggsounder.benchmark")
    bench_mod.benchmark = benchmark
    bench_mod.compute_all_results = compute_all_results
    labels_mod = types.ModuleType("vggsounder.labels")
    labels_mod.VGGSounder = VGGSounder
    pkg.benchmark = bench_mod
    pkg.labels = labels_mod
    monkeypatch.setitem(sys.modules, "vggsounder", pkg)
    monkeypatch.setitem(sys.modules, "vggsounder.benchmark", bench_mod)
    monkeypatch.setitem(sys.modules, "vggsounder.labels", labels_mod)
    # the fake is reported at the pin (the site's editable checkout reports 1.0.0 and is correctly refused)
    import importlib.metadata as md

    real_version = md.version
    monkeypatch.setattr(md, "version", lambda name: "1.0" if name == "vggsounder" else real_version(name))
    from yavih import external

    external.reset()
    return calls


def _vgg_bench(tmp_path: Path | None = None):
    """The packaged vggsounder folder (its real eval.py join) over the N synthetic eval clips."""
    import yaml

    spec = yaml.safe_load((VGGSOUNDER_FOLDER / "benchmark.yaml").read_text(encoding="utf-8"))
    return SimpleNamespace(
        name="vggsounder", folder=VGGSOUNDER_FOLDER, protocol_hash=PH,
        spec=SimpleNamespace(external=spec["external"]),
        roles=lambda: {"eval": ["test"]},
        manifest=lambda split: SimpleNamespace(clip_id=[c + ".mp4" for c in CLIP_IDS]),
    )


def test_locate_result_dir(three_runs, tmp_path: Path):
    run = Path(three_runs["a"])
    seed_dir = locate_result_dir(run)
    assert seed_dir.name == "seed0" and (seed_dir / "DONE").is_file()
    assert locate_result_dir(seed_dir) == seed_dir
    assert locate_result_dir(seed_dir.parent) == seed_dir
    assert locate_result_dir(run / "probe") == seed_dir
    with pytest.raises(JoinError, match="E_JOIN_MISSING"):
        locate_result_dir(tmp_path / "nope")
    with pytest.raises(JoinError, match="E_JOIN_MISSING"):
        locate_result_dir(run, seed=7)
    # a second probe column (another eval_hash) makes the run dir ambiguous
    write_result(tmp_path / "runs", _vgg_result("a", probe_hash="9" * 64), with_npy=True)
    with pytest.raises(JoinError, match="E_JOIN_AMBIGUOUS"):
        locate_result_dir(run)
    # a seed dir without DONE is not a result
    (seed_dir / "DONE").unlink()
    with pytest.raises(JoinError, match="E_JOIN_MISSING"):
        locate_result_dir(seed_dir.parent)


def test_vggsounder_folder_join_calls_official_scorer(three_runs, fake_vggsounder, tmp_path: Path):
    """``report --join vggsounder a=RUN v=RUN av=RUN`` through run_join and the folder's probe.join: one pkl
    with the three readings under their routings, ``compute_all_results`` on it, the version recorded."""
    out = run_join("vggsounder", three_runs, bench=_vgg_bench(), results_root=tmp_path / "results")
    calls = fake_vggsounder
    assert calls["modalities"] == ["a", "v", "av", "a only", "v only"] and set(calls["subset_ids"]) == set(CLIP_IDS)
    assert calls["verbose"] is False and calls["labels_kwargs"] == {"background_music": False, "download_samples": False}
    payload = calls["payload"]
    assert sorted(payload) == CLIP_IDS and payload["clip03"]["logits"]["av"] == [3.0] * C and payload["clip03"]["logits"]["a"] == [1.0] * C
    assert not Path(calls["models_path"]).exists()  # tempdir cleaned up
    joined = out["results"]
    scores = joined["scores"]  # compute_all_results' own output: 5 sets × 30 metrics + mu for a / v / av
    assert sum(len(d) for d in scores.values()) == 5 * 30 + 3
    assert {k: scores[k]["mu"] for k in ROUTINGS} == pytest.approx({"a": 0.051, "v": 0.052, "av": 0.053})
    assert scores["av"]["auc_pr"] == pytest.approx(0.303) and scores["a only"]["hit_rate@1"] == pytest.approx(0.5)
    assert joined["n_eval"] == N and joined["table"]["rows"][2][0] == "AV"
    doc = json.loads(Path(out["path"]).read_text())  # the results store keeps the whole output
    assert doc["results"]["vggsounder:a+v+av"]["scores"]["v only"]["f1@10"] == pytest.approx(0.5)
    assert set(joined["runs"]) == set(ROUTINGS)
    assert out["external"] == {"vggsounder": "1.0"} and joined["external"] == {"vggsounder": "1.0"}
    doc = json.loads(Path(out["path"]).read_text())
    assert doc["config"]["external"] == {"vggsounder": "1.0"}


# --------------------------------------------------------------------------- the folder eval.py path

FOLDER_PROBE = """
import numpy as np

CALLS = []

def join(bench, results):
    CALLS.append(sorted(results))
    out = {}
    for reading, ref in sorted(results.items()):
        logits = np.asarray(ref.eval_logits())
        out[f"mean_{reading}"] = float(logits.mean())
        out[f"n_{reading}"] = int(ref.eval_rows().shape[0])
    out["models"] = sorted(r.model for r in results.values())
    return out
"""


def _fake_bench(tmp_path: Path, *, with_join: bool = True, external=()):
    folder = tmp_path / "vggsounder_folder"
    folder.mkdir(parents=True, exist_ok=True)
    if with_join:
        (folder / "eval.py").write_text(textwrap.dedent(FOLDER_PROBE))
    return SimpleNamespace(name="vggsounder", folder=folder, spec=SimpleNamespace(external=list(external)))


def test_run_join_calls_the_folder_eval_join_and_writes_the_store(three_runs, tmp_path: Path):
    from yavih.report.join import ResultRef, run_join

    bench = _fake_bench(tmp_path)
    out = run_join("vggsounder", three_runs, bench=bench, results_root=tmp_path / "results")
    assert out["results"]["mean_av"] == pytest.approx(3.0) and out["results"]["mean_a"] == pytest.approx(1.0) and out["results"]["n_v"] == N
    assert out["results"]["models"] == ["avm", "avm", "avm"]
    assert (out["model"], out["reading"], out["eval_hash"]) == ("avm", "a+v+av", hashing.eval_hash(PH, QH)) and out["warnings"] == []
    assert out["external"] == {} and set(out["routing"]) == {"a", "v", "av"}
    doc = json.loads(Path(out["path"]).read_text())
    # the model's directory (the same one its readings' results files go to), key <benchmark>:a+v+av
    assert Path(out["path"]).parent == tmp_path / "results" / "avm"
    key = "vggsounder:a+v+av"
    assert set(doc["results"]) == {key} and doc["results"][key]["mean_v"] == pytest.approx(2.0) and doc["results"][key]["status"] == "ok"
    cfg = doc["configs"][key]
    assert (cfg["benchmark"], cfg["reading"], cfg["eval_hash"]) == ("vggsounder", "a+v+av", hashing.eval_hash(PH, QH))
    assert set(cfg["joined"]) == set(ROUTINGS) and "bridge" not in cfg and cfg["warnings"] == []
    assert cfg["joined"]["v"]["unit_results"] == [str(Path(out["routing"]["v"]) / "result.json")] and cfg["joined"]["v"]["describe_sha256"]
    assert doc["config"]["model"] == "avm" and doc["versions"] == {key: 2}
    ref = ResultRef("a", Path(out["routing"]["a"]), {"model": "m"})
    assert ref.model == "m" and ref.eval_rows().tolist() == list(range(N))


def test_join_records_differing_weights_and_refuses_another_model(three_runs, tmp_path: Path):
    """Differing non-null ``weights_sha256`` across the readings is ``W_JOIN_WEIGHTS`` — recorded in the
    join output and its results entry, never a refusal (an audio-only reading may fingerprint only what it loads); a
    null fingerprint is not a difference; readings of two model names are ``E_JOIN_MODEL``."""
    from yavih.report.join import JoinError, run_join

    bench = _fake_bench(tmp_path)
    runs = tmp_path / "runs"
    dirs = {k: str(locate_result_dir(v)) for k, v in three_runs.items()}
    other_w = write_result(runs, _vgg_result("a", weights_sha256="e" * 64), with_npy=True, seed_dir=tmp_path / "alt_a" / "seed0")
    out = run_join("vggsounder", dict(dirs, a=str(other_w.parent)), bench=bench, results_root=tmp_path / "results")
    (w,) = out["warnings"]
    assert w["code"] == "W_JOIN_WEIGHTS" and "a=eeeeeeeeeeee" in w["message"] and out["results"]["mean_av"] == pytest.approx(3.0)
    assert json.loads(Path(out["path"]).read_text())["configs"]["vggsounder:a+v+av"]["warnings"] == [w]
    null_w = write_result(runs, _vgg_result("a", weights_sha256=None), with_npy=True, seed_dir=tmp_path / "null_a" / "seed0")
    assert run_join("vggsounder", dict(dirs, a=str(null_w.parent)), bench=bench, write=False)["warnings"] == []
    other_model = write_result(runs, _vgg_result("av", model="another"), with_npy=True)
    with pytest.raises(JoinError) as e:
        run_join("vggsounder", dict(dirs, av=str(other_model.parent)), bench=bench, write=False)
    assert e.value.code == "E_JOIN_MODEL" and e.value.exit_code == 2


def test_run_join_refusals(three_runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from yavih.report.join import JoinError, run_join

    with pytest.raises(JoinError) as e:
        run_join("vggsounder", three_runs, bench=_fake_bench(tmp_path / "nojoin", with_join=False), write=False)
    assert e.value.code == "E_JOIN_UNSUPPORTED" and e.value.exit_code == 1
    bench = _fake_bench(tmp_path)
    runs = tmp_path / "runs"
    dirs = {k: str(locate_result_dir(v)) for k, v in three_runs.items()}
    # the authors' package absent: exit 3 with the pip line, before join() runs
    ext = [{"name": "no_such_authors_pkg", "pip": "pip install no_such_authors_pkg==2.0", "version_pin": "2.0", "used_for": "mu"}]
    with pytest.raises(JoinError) as e:
        run_join("vggsounder", dirs, bench=_fake_bench(tmp_path / "ext", external=ext), write=False)
    assert e.value.exit_code == 3 and "pip install no_such_authors_pkg==2.0" in str(e.value)
    # different eval populations (same eval_hash: another copy of the reading)
    p = write_result(runs, _vgg_result("v", ids_sha256=IDS2), with_npy=True, seed_dir=tmp_path / "pop_v" / "seed0")
    with pytest.raises(JoinError, match="E_JOIN_POPULATION"):
        run_join("vggsounder", dict(dirs, v=str(p.parent)), bench=bench, write=False)
    # different eval_hashes: another protocol, or another probe
    p = write_result(runs, _vgg_result("av", protocol_hash=PH2), with_npy=True)
    with pytest.raises(JoinError, match="E_JOIN_MIXED"):
        run_join("vggsounder", dict(dirs, av=str(p.parent)), bench=bench, write=False)
    p = write_result(runs, _vgg_result("av", probe_hash="8" * 64), with_npy=True)
    with pytest.raises(JoinError, match="E_JOIN_MIXED"):
        run_join("vggsounder", dict(dirs, av=str(p.parent)), bench=bench, write=False)
    # a result of another benchmark
    other = write_result(runs, make_result(bench="k400", model="avm", reading="av"), with_npy=True)
    with pytest.raises(JoinError, match="E_JOIN_MIXED"):
        run_join("vggsounder", dict(dirs, av=str(other.parent)), bench=bench, write=False)
    # no pairs at all
    with pytest.raises(JoinError, match="E_JOIN_USAGE"):
        run_join("vggsounder", {}, bench=bench, write=False)


def test_join_through_report(three_runs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``report(join=...)`` loads the benchmark by the join's name and calls its folder's join()."""
    import yavih.benchmark as B
    from yavih.report import ReportError, report

    bench = _fake_bench(tmp_path)
    monkeypatch.setattr(B.Benchmark, "load", staticmethod(lambda ref, **kw: bench))
    rep = report([str(tmp_path / "runs" / "*")], join={"vggsounder": dict(three_runs)}, results_root=tmp_path / "res")
    joined = rep.joins["vggsounder"]
    assert joined["results"]["mean_av"] == pytest.approx(3.0) and Path(joined["path"]).parent == tmp_path / "res" / "avm"
    assert joined["model"] == "avm" and joined["reading"] == "a+v+av"
    assert "## Joins" in rep.text and "vggsounder" in rep.text and "mean_av" in rep.text
    # the default root: `results` next to the runs root
    rep = report([str(tmp_path / "runs" / "*")], join={"vggsounder": dict(three_runs)})
    assert Path(rep.joins["vggsounder"]["path"]).parent == tmp_path / "results" / "avm"
    # which readings a join needs is the folder's business (its join() refuses); a refused join surfaces
    # as a ReportError with the join's code and exit code
    monkeypatch.setattr(B.Benchmark, "load", staticmethod(lambda ref, **kw: _fake_bench(tmp_path / "nj", with_join=False)))
    with pytest.raises(ReportError) as e:
        report([str(tmp_path / "runs" / "*")], join={"vggsounder": dict(three_runs)}, write=False)
    assert e.value.code == "E_JOIN_UNSUPPORTED" and e.value.exit_code == 1


def test_vggsounder_folder_join_refusals(three_runs, fake_vggsounder, tmp_path: Path):
    bench = _vgg_bench()
    with pytest.raises(JoinError, match="E_JOIN_MISSING") as ei:
        run_join("vggsounder", {"a": three_runs["a"]}, bench=bench, write=False)
    assert ei.value.exit_code == 2
    runs = tmp_path / "runs"
    dirs = {k: str(locate_result_dir(v)) for k, v in three_runs.items()}  # pinned before extra columns appear
    # swapped routing (the video-only reading as a=): refused before anything is scored
    with pytest.raises(JoinError, match="E_JOIN_ROUTING"):
        run_join("vggsounder", dict(dirs, a=dirs["v"], v=dirs["a"]), bench=bench, write=False)
    # different eval population
    p = write_result(runs, _vgg_result("v", ids_sha256=IDS2), with_npy=True, seed_dir=tmp_path / "pop_v" / "seed0")
    with pytest.raises(JoinError, match="E_JOIN_POPULATION"):
        run_join("vggsounder", dict(dirs, v=str(p.parent)), bench=bench, write=False)
    # different protocol hash
    p = write_result(runs, _vgg_result("av", protocol_hash=PH2), with_npy=True)
    with pytest.raises(JoinError, match="E_JOIN_MIXED"):
        run_join("vggsounder", dict(dirs, av=str(p.parent)), bench=bench, write=False)
    # not ok
    bad = write_result(runs, make_result(bench="vggsounder", model="na", reading="av", status="not_applicable"), with_npy=True)
    with pytest.raises(JoinError, match="E_JOIN_STATUS"):
        run_join("vggsounder", dict(dirs, av=str(bad.parent)), bench=bench, write=False)
    # the run dir of "av" is now ambiguous (two probe columns)
    with pytest.raises(JoinError, match="E_JOIN_AMBIGUOUS"):
        run_join("vggsounder", three_runs, bench=bench, write=False)
    # missing npy
    (Path(dirs["v"]) / "eval_logits.npy").unlink()
    with pytest.raises(Exception, match="eval_logits.npy"):
        run_join("vggsounder", dirs, bench=bench, write=False)


@pytest.mark.skipif(importlib.util.find_spec("vggsounder") is None,
                    reason="vggsounder package not installed (pip install --no-deps vggsounder==1.0, see benchmarks/vggsounder/setup.md)")
def test_real_vggsounder_signature_matches_the_call():
    """The real package must accept the keywords yavih passes."""
    import inspect

    from vggsounder.benchmark import benchmark
    from vggsounder.labels import VGGSounder

    params = inspect.signature(benchmark).parameters
    for name in ("models_path", "display_names", "metrics", "subset_ids", "vggsounder", "verbose"):
        assert name in params, name
    from vggsounder.benchmark import compute_all_results

    cparams = inspect.signature(compute_all_results).parameters
    for name in ("models_path", "subset_ids", "modalities", "vggsounder", "verbose"):  # what probe.official_scores passes
        assert name in cparams, name
    lp = inspect.signature(VGGSounder.__init__).parameters
    assert "background_music" in lp and "download_samples" in lp


def test_run_join_keeps_the_folders_refusal_code(three_runs, tmp_path: Path):
    from yavih.report.join import JoinError, run_join

    folder = tmp_path / "refusing"
    folder.mkdir()
    (folder / "eval.py").write_text(textwrap.dedent('''
        class JoinRefused(Exception):
            def __init__(self, code, message):
                super().__init__(f"{code}: {message}")
                self.code = code

        def join(bench, results):
            raise JoinRefused("E_JOIN_MISSING", "join needs a=RUN v=RUN av=RUN")
    '''))
    bench = SimpleNamespace(name="vggsounder", folder=folder, spec=SimpleNamespace(external=[]))
    with pytest.raises(JoinError) as e:
        run_join("vggsounder", {"a": three_runs["a"]}, bench=bench, write=False)
    assert e.value.code == "E_JOIN_MISSING" and e.value.exit_code == 2


# --------------------------------------------------------------------------- what the report shows for vggsounder


def _vgg_metrics(reading: str) -> dict:
    import yaml

    names = yaml.safe_load((VGGSOUNDER_FOLDER / "benchmark.yaml").read_text(encoding="utf-8"))["metrics"]["list"]
    return {n: (None if reading == "av" and n.endswith("_only") else 0.25) for n in names}


def test_vggsounder_report_shows_metrics_table_by_default_and_everything_in_json():
    """``metrics.table`` (auc_pr, hit_rate@1, f1@1) are the default markdown columns of view 2 and the default rows of
    view 1; json and csv carry all 60 metrics, ``--detail`` prints them; the av reading's ``_only`` values are null."""
    from test_report import _records
    from yavih.report.compare import build_report

    recs = _records(*(make_result(bench="vggsounder", model="avm", reading=k, metrics=_vgg_metrics(k), floors={}) for k in ROUTINGS))
    rep = build_report(recs)
    (col,) = rep.columns
    assert len(col["metrics"]) == 60 and col["table"] == ["auc_pr", "hit_rate@1", "f1@1"] and col["headline"] == "auc_pr"
    header = next(line for line in rep.text.splitlines() if line.startswith("| model | reading"))
    assert header == "| model | reading | seeds | cov eval | **auc_pr** | hit_rate@1 | f1@1 | flags |"
    assert "accuracy@5" not in rep.text and "auc_pr_only" not in rep.text
    av_row = next(r for r in rep.rows if r["reading"] == "av")
    assert av_row["metrics"]["auc_pr_only"]["value"] is None and av_row["metrics"]["auc_pr"]["value"] == 0.25
    js = json.loads(build_report(recs, fmt="json").text)
    assert all(len(r["metrics"]) == 60 for r in js["rows"]) and len(js["lm_eval"]["avm"]) == 3 * 60 - 30
    csv_text = build_report(recs, fmt="csv").text
    assert csv_text.count(",accuracy@5,") == 3 and csv_text.count(",hit_rate@10_only,") == 3
    detail = build_report(recs, detail=True).text
    assert "accuracy@5" in detail and "auc_pr_only" in detail


def test_join_markdown_shows_the_compact_table_and_everything_under_detail(three_runs, fake_vggsounder):
    from yavih.report.compare import Report
    from yavih.report.tables import render_md

    out = run_join("vggsounder", three_runs, bench=_vgg_bench(), write=False)
    text = render_md(Report(joins={"vggsounder": out}))
    assert "|  | mAP | HIT | F1 | mu (A, V, A∩V) |" in text and "| AV | 0.303 | 0.503 | 0.503 | 0.053 |" in text
    assert "mu A∩V (mu_av): mu['av']: Combined confusion rate (both A and V top-1 hit when AV top-1 fails)" in text
    assert "f1_macro@10" not in text and "a only" not in text
    full = render_md(Report(joins={"vggsounder": out}, show_all=True))
    assert "f1_macro@10" in full and "| | a | v | av | a only | v only |" in full
