"""Folder tests of the packaged ``epic100_anticipation_stream`` benchmark (protocol_version 1).

Run only this folder: ``python -m pytest "$(yavih bench path epic100_anticipation_stream)/tests" -q -p no:cacheprovider``.

- file-level checks (yaml, manifest, the shipped tail / unseen lists, README.md / setup.md facts);
- through the core: the folder loads, the anticipation anchors, the metrics on hand-checked inputs, the conformance
  fixture, the bridge template's describe() (the transcription of the official evaluator is checked against it in
  yavih's tests/test_metrics_stream.py);
- site: make_manifest.py --check against the staged annotations (skipped elsewhere).
"""

from __future__ import annotations

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
    return F.load_module(HERE / "eval.py", "epic100_anticipation_stream_folder_eval")


def test_yaml_identity(spec):
    assert spec["name"] == "epic100_anticipation_stream" and spec["scorer"] == "stream" and spec["protocol_version"] == 1
    assert spec["protocol"]["anticipation_s"] == 1.0 and spec["output_contract"]["stream"] == {"update_interval_s": 0.5}
    assert spec["readout_params"]["groups"] == {"verb": {"loss": "ce", "classes": 97}, "noun": {"loss": "ce", "classes": 300}}
    F.assert_manifest_hashes(HERE, spec)
    F.assert_metric_names_resolve(HERE, spec)


def test_manifest_rows(rows):
    assert Counter(r["split"] for r in rows) == {"train": 495, "validation": 138}
    segs = Counter()
    for r in rows:
        s = json.loads(r["segments"])
        assert r["rel_media"] == f"EPIC-KITCHENS/{r['participant_id']}/videos/{r['clip_id']}.MP4" and r["clip_id"].startswith(r["participant_id"])
        assert s == sorted(s) and all(0 <= v < 97 and 0 <= n < 300 for _, v, n in s)
        segs[r["split"]] += len(s)
    assert segs == {"train": 67217, "validation": 9668}


def test_shipped_lists(eval_mod):
    assert eval_mod.unseen_participants() == {"P18", "P32"}
    assert len(eval_mod.tail_verbs()) > 0 and len(eval_mod.tail_nouns()) > 0


def test_anchors(bench, eval_mod):
    tg = eval_mod.targets(bench, "validation", np.arange(2))
    man = bench.manifest("validation")
    for k, t in enumerate(tg):
        seg = np.asarray(man.extra[k]["segments"], dtype=float)
        assert np.allclose(t["times"], seg[:, 0] - 1.0) and t["labels"]["verb"].tolist() == seg[:, 1].astype(int).tolist()
        assert t["meta"]["unseen"].dtype == bool and len(t["meta"]["tail_noun"]) == len(seg)


def test_metrics_on_hand_checked_inputs(eval_mod):
    yv, yn = np.array([0, 0, 1, 2]), np.array([5, 5, 6, 7])
    pv, pn = np.full((4, 97), 1e-3), np.full((4, 300), 1e-3)
    pv[np.arange(4), yv] = 1.0
    pn[np.arange(4), yn] = 1.0
    meta = {"meta": {"unseen": np.array([0, 0, 1, 1], bool), "tail_verb": np.zeros(4, bool), "tail_noun": np.array([0, 0, 0, 1], bool)}}
    out, tg = {"verb": pv / pv.sum(1, keepdims=True), "noun": pn / pn.sum(1, keepdims=True)}, {"verb": yv, "noun": yn}
    m = eval_mod.METRICS
    assert m["mt5r_action_gru"].fn(out, tg, {}, meta) == 1.0 and m["mt5r_verb_unseen_linear"].fn(out, tg, {}, meta) == 1.0
    assert m["mt5r_noun_tail_gru"].fn(out, tg, {}, meta) == 1.0
    wrong = {"verb": out["verb"][:, ::-1].copy(), "noun": out["noun"]}
    assert m["mt5r_verb_gru"].fn(wrong, tg, {}, meta) == 0.0 and m["mt5r_action_gru"].fn(wrong, tg, {}, meta) == 0.0


def test_bridge_template_declares_the_stream(bench):
    from yavih.bridge import BenchmarkView, validate_describe

    tmpl = F.load_module(HERE / "bridge_template.py", "epic100_anticipation_stream_bridge_template")
    for reading in ("av", "av-rt"):
        b = tmpl.Epic100StreamBridge(BenchmarkView.from_benchmark(bench), {"reading": reading})
        validate_describe(b.describe(), b, bench)


def test_manifest_rederives_on_site():
    root = F.site_root(HERE, "media_root", needs=("annotations/EPIC_100_train.csv",))
    if root is None:
        pytest.skip("the EPIC-KITCHENS-100 annotations are not staged here")
    mm = F.load_module(HERE / "make_manifest.py", "epic100_make_manifest")
    assert mm.main(["--annotations", str(root / "annotations"), "--check"]) == 0
