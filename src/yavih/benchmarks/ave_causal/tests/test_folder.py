"""Folder tests of the packaged ``ave_causal`` benchmark (protocol_version 1).

Run only this folder: ``python -m pytest "$(yavih bench path ave_causal)/tests" -q -p no:cacheprovider``.

- file-level checks (yaml, manifest, its re-derivation from the packaged ave manifest, README.md / setup.md facts);
- through the core: the folder loads, its targets hook, its metrics on hand-checked predictions, the conformance
  fixture, the bridge template's describe() against the stream contract.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, rows, bench = F.spec, F.rows, F.bench


@pytest.fixture(scope="module")
def eval_mod():
    return F.load_module(HERE / "eval.py", "ave_causal_folder_eval")


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_yaml_identity(spec):
    assert spec["name"] == "ave_causal" and spec["scorer"] == "stream" and spec["protocol_version"] == 1
    oc = spec["output_contract"]
    assert oc["shape"] == "sequence" and oc["stream"] == {"update_interval_s": 0.5}
    assert spec["readout_params"]["groups"] == {"event": {"loss": "ce", "classes": 29}}
    F.assert_manifest_hashes(HERE, spec)
    F.assert_metric_names_resolve(HERE, spec)


def test_manifest_rows(spec, rows):
    assert Counter(r["split"] for r in rows) == {"train": 2592, "test": 316}
    assert len({r["clip_id"] for r in rows}) == len(rows)
    for r in rows:
        lab = [int(x) for x in r["seconds"].split(";")]
        assert len(lab) == 10 and all(0 <= x <= 28 for x in lab)
        assert len(set(lab) - {28}) <= 1  # one event per video
        assert r["rel_media"] == f"video/{r['clip_id']}.mp4"


def test_manifest_rederives_from_ave():
    mm = F.load_module(HERE / "make_manifest.py", "ave_causal_make_manifest")
    assert mm.main(["--check"]) == 0


def test_targets_hook(bench, eval_mod):
    tg = eval_mod.targets(bench, "test", np.arange(3))
    man = bench.manifest("test")
    for k, t in enumerate(tg):
        assert t["times"].tolist() == [float(s) for s in range(1, 11)]
        assert t["labels"]["event"].tolist() == [int(x) for x in man.extra[k]["seconds"].split(";")]


def test_metrics_on_hand_checked_predictions(eval_mod):
    labels = np.array([28, 3, 3, 28, 5, 5])
    good = np.eye(29)[labels]
    proto = {"background_index": 28}
    meta = {"recording": np.array([0, 0, 0, 1, 1, 1]), "bootstrap": {"samples": 50, "seed": 0}}
    m = eval_mod.METRICS
    assert m["sec_acc_gru"].fn({"event": good}, {"event": labels}, proto, meta) == 1.0
    assert m["event_map_linear"].fn({"event": good}, {"event": labels}, proto, meta) == 1.0
    half = good.copy()
    half[1] = np.eye(29)[28]
    assert m["sec_acc_linear"].fn({"event": half}, {"event": labels}, proto, meta) == pytest.approx(5 / 6)
    lo, hi = m["sec_acc_linear"].interval("bootstrap_clip", {"event": half}, {"event": labels}, proto, meta)
    assert lo <= 5 / 6 <= hi


def test_conformance_fixture_reproduces(bench):
    from yavih import conformance as C

    rep = C.check(bench)
    assert rep["ok"] if isinstance(rep, dict) else rep.ok


def test_bridge_template_declares_the_stream(bench):
    from yavih.bridge import BenchmarkView, validate_describe

    tmpl = F.load_module(HERE / "bridge_template.py", "ave_causal_bridge_template")
    for reading in ("av", "v-rt"):
        b = tmpl.AveCausalBridge(BenchmarkView.from_benchmark(bench), {"reading": reading})
        d = b.describe()
        assert d["stream"]["mode"] == ("realtime" if reading.endswith("-rt") else "causal")
        validate_describe(d, b, bench)


def test_docs_name_the_facts(spec):
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    setup = (HERE / "setup.md").read_text(encoding="utf-8")
    for fact in ("2 592", "316", "sec_acc_gru", "ave_causal_v1", "16 train videos"):
        assert fact in readme
    assert "yavih bench verify ave_causal" in setup
