"""Folder tests of the packaged ``unav100_causal`` benchmark (protocol_version 1).

Run only this folder: ``python -m pytest "$(yavih bench path unav100_causal)/tests" -q -p no:cacheprovider``.

- file-level checks (yaml, manifest, classes, README.md / setup.md facts);
- through the core: the folder loads, the bin targets (coverage rule, clipping, concurrency), the metrics and the onset
  diagnostic on hand-checked inputs, the conformance fixture, the bridge template's describe();
- site: make_manifest.py --check against the staged data (skipped elsewhere).
"""

from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows, bench = F.spec, F.rows, F.bench


@pytest.fixture(scope="module")
def eval_mod():
    return F.load_module(HERE / "eval.py", "unav100_causal_folder_eval")


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_yaml_identity(spec):
    assert spec["name"] == "unav100_causal" and spec["scorer"] == "stream" and spec["protocol_version"] == 1
    assert spec["output_contract"]["stream"] == {"update_interval_s": 0.5}
    assert spec["readout_params"]["groups"] == {"event": {"loss": "bce", "classes": 100}}
    assert spec["data"]["roots"]["media_root"]["hf"]["revision"] == spec["source"]["revision"]
    F.assert_manifest_hashes(HERE, spec)
    F.assert_metric_names_resolve(HERE, spec)


def test_manifest_rows(rows):
    assert Counter(r["split"] for r in rows) == {"train": 6487, "validation": 2133, "test": 2166}
    assert len({r["clip_id"] for r in rows}) == len(rows)
    n_events = 0
    for r in rows:
        d = float(r["duration_s"])
        ev = json.loads(r["events"])
        assert 9.0 < d < 61.0 and ev and r["rel_media"] == f"raw_videos/{r['clip_id']}.mp4"
        for s, e, c in ev:
            assert 0 <= s < e <= d + 1.0 and 0 <= c < 100  # an event may end past the file by < 1 s (the bins clip it)
        n_events += len(ev)
    assert n_events == 30050


def test_classes():
    with open(HERE / "event_classes.csv", newline="", encoding="utf-8") as f:
        cls = list(csv.DictReader(f))
    assert [int(r["index"]) for r in cls] == list(range(100)) and len({r["name"] for r in cls}) == 100


def test_bins(eval_mod):
    times, lab, s, e = eval_mod.bins_of(3.4, [[0.2, 1.7, 5], [1.4, 3.4, 5], [0.0, 0.4, 7], [2.0, 3.4, 9]])
    assert times.tolist() == [1.0, 2.0, 3.0, 3.4]
    assert lab[:, 5].tolist() == [1, 1, 1, 1]          # merged [0.2, 3.4]: every bin >= half covered (the clipped last one too)
    assert lab[:, 7].tolist() == [0, 0, 0, 0]          # 0.4 s of a 1 s bin
    assert lab[:, 9].tolist() == [0, 0, 1, 1]          # concurrent with class 5
    assert s[0, 5] == pytest.approx(0.2) and e[0, 5] == pytest.approx(3.4) and np.isnan(s[0, 9])


def test_targets_hook(bench, eval_mod):
    tg = eval_mod.targets(bench, "test", np.arange(2))
    man = bench.manifest("test")
    for k, t in enumerate(tg):
        d = float(man.extra[k]["duration_s"])
        assert t["times"][-1] == pytest.approx(d) and t["labels"]["event"].shape == (len(t["times"]), 100)
        assert t["meta"]["ev_start"].shape == t["labels"]["event"].shape


def test_metrics_and_onset(eval_mod):
    times, lab, s, e = eval_mod.bins_of(20.0, [[3.0, 9.0, 1], [12.0, 14.0, 2]])
    rec = np.zeros(len(times), dtype=np.int64)
    meta = {"recording": rec, "time": times, "meta": {"ev_start": s, "ev_end": e}}
    perfect = lab.copy()
    assert eval_mod._map({"event": perfect}, {"event": lab}, {}, meta) == 1.0
    rec_, delay, fa, n = eval_mod.onset(perfect, rec, times, meta["meta"], 0.5, 1.0)
    assert (rec_, n, fa) == (1.0, 2, 0.0) and delay == pytest.approx(1.0)  # alerts at the first bin ends: 4 - 3, 13 - 12
    noisy = perfect.copy()
    noisy[17, 3] = 0.8  # a class-3 alert where no class-3 event is: one false alert in 20 s at threshold 0.5
    assert eval_mod.onset(noisy, rec, times, meta["meta"], 0.5, 1.0)[2] == pytest.approx(180.0)
    val = {"outputs": {"event": noisy}, "recording": rec, "time": times, "meta": meta["meta"]}
    theta = eval_mod.operating_point(val, {"onset": {"false_alerts_per_hour": 60, "tolerance_s": 1.0}})
    assert eval_mod.onset(noisy, rec, times, meta["meta"], theta, 1.0)[2] <= 60


def test_conformance_fixture_reproduces(bench):
    from yavih import conformance as C

    rep = C.check(bench)
    assert rep["ok"] if isinstance(rep, dict) else rep.ok


def test_bridge_template_declares_the_stream(bench):
    from yavih.bridge import BenchmarkView, validate_describe

    tmpl = F.load_module(HERE / "bridge_template.py", "unav100_causal_bridge_template")
    for reading in ("av", "av-rt"):
        b = tmpl.Unav100CausalBridge(BenchmarkView.from_benchmark(bench), {"reading": reading})
        validate_describe(b.describe(), b, bench)


def test_docs_name_the_facts():
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    for fact in ("6 487", "2 166", "map_gru", "unav100_causal_recognition_v1", "60 per hour"):
        assert fact in readme
    assert "yavih bench verify unav100_causal" in (HERE / "setup.md").read_text(encoding="utf-8")


def test_manifest_rederives_on_site():
    root = F.site_root(HERE, "media_root", needs=("data/test.parquet", "raw_videos"))
    if root is None:
        pytest.skip("UnAV-100 is not staged here")
    mm = F.load_module(HERE / "make_manifest.py", "unav100_causal_make_manifest")
    assert mm.main(["--root", str(root), "--check"]) == 0
