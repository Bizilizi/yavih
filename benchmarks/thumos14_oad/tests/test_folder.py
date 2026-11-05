"""Folder tests of the packaged ``thumos14_oad`` benchmark (protocol_version 1).

Run only this folder: ``python -m pytest "$(yavih bench path thumos14_oad)/tests" -q -p no:cacheprovider``.

- file-level checks (yaml, manifest, video_sha256.csv, README.md / setup.md facts);
- through the core: the folder loads, the 4 fps chunk targets, ``frame_map`` against LSTR's reference evaluator with
  its THUMOS post-processing (needs sklearn), the conformance fixture, the bridge template's describe();
- site: make_manifest.py --check against the staged data (skipped elsewhere).
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
    return F.load_module(HERE / "eval.py", "thumos14_oad_folder_eval")


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_yaml_identity(spec):
    assert spec["name"] == "thumos14_oad" and spec["scorer"] == "stream" and spec["protocol_version"] == 1
    assert spec["output_contract"]["stream"] == {"update_interval_s": 0.25} and spec["protocol"]["chunk_fps"] == 4.0
    assert spec["readout_params"]["groups"] == {"action": {"loss": "ce", "classes": 21}}
    F.assert_manifest_hashes(HERE, spec)
    F.assert_metric_names_resolve(HERE, spec)


def test_manifest_rows(rows):
    assert Counter(r["split"] for r in rows) == {"validation": 200, "test": 213}
    n = Counter()
    for r in rows:
        assert r["rel_media"] == f"videos/{r['split']}/{r['clip_id']}.mp4"
        for s, e, c in json.loads(r["events"]):
            assert 0 <= s < e and 1 <= c <= 21
            n["ambiguous" if c == 21 else "action"] += 1
    assert n == {"action": 6365, "ambiguous": 188}
    shas = (HERE / "video_sha256.csv").read_text().split()[1:]
    assert sorted(x.split(",")[0] for x in shas) == sorted(f"{r['split']}/{r['clip_id']}" for r in rows)


def test_chunks(eval_mod):
    times, action, gt = eval_mod.chunks_of(2.3, [[0.3, 0.9, 3], [0.6, 1.4, 21], [1.5, 2.3, 7], [1.6, 2.0, 9]])
    assert times.tolist() == [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    assert action.tolist() == [0, 3, -1, -1, -1, -1, 7, 7, 7]  # centres .125 .375 .625 ...; inside Ambiguous [.6, 1.4) -1
    assert gt[6, 7] == 1 and gt[6, 9] == 1 and gt[0, 0] == 1 and gt[2, 21] == 1 and gt[2, 0] == 0


def test_frame_map_is_the_lstr_evaluator(eval_mod):
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    m = 2000
    lab = rng.integers(0, 22, m)
    gt = np.eye(22, dtype=np.uint8)[lab]
    scores = rng.random((m, 21)) + 0.7 * gt[:, :21]
    valid = np.where(gt[:, 21] != 1)[0]  # LSTR thumos_postprocessing: remove ambiguous
    g, p = gt[valid], np.concatenate([scores, np.zeros((m, 1))], axis=1)[valid]
    ref = np.mean([sk.average_precision_score(g[:, c], p[:, c]) for c in range(1, 21) if g[:, c].any()])
    got = eval_mod._frame_map({"action": scores}, {}, {"ambiguous_index": 21}, {"meta": {"gt": gt}})
    assert got == pytest.approx(ref, abs=1e-12)


def test_conformance_fixture_reproduces(bench):
    from yavih import conformance as C

    rep = C.check(bench)
    assert rep["ok"] if isinstance(rep, dict) else rep.ok


def test_bridge_template_declares_the_stream(bench):
    from yavih.bridge import BenchmarkView, validate_describe

    tmpl = F.load_module(HERE / "bridge_template.py", "thumos14_oad_bridge_template")
    for reading in ("v", "av-rt"):
        b = tmpl.Thumos14OadBridge(BenchmarkView.from_benchmark(bench), {"reading": reading})
        validate_describe(b.describe(), b, bench)


def test_docs_name_the_facts():
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    for fact in ("200", "213", "frame_map_gru", "perframe_average_precision", "optical flow"):
        assert fact in readme
    assert "yavih bench verify thumos14_oad" in (HERE / "setup.md").read_text(encoding="utf-8")


def test_manifest_rederives_on_site():
    root = F.site_root(HERE, "media_root", needs=("annotations/Class_Index_Detection.txt", "videos/test"))
    if root is None:
        pytest.skip("THUMOS'14 is not staged here")
    mm = F.load_module(HERE / "make_manifest.py", "thumos14_oad_make_manifest")
    assert mm.main(["--root", str(root), "--check"]) == 0
