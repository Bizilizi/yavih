"""Folder tests for vggsound_retrieval.

Run only this file: ``python -m pytest "$(yavih bench path vggsound_retrieval)/tests" -q -p no:cacheprovider``.

Pure checks of the frozen artefacts and the docs (always run); checks through the core (always run); checks that
need the staged VGGSound listing and media (they skip otherwise).
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_rows, mm, bench, template = F.spec, F.rows, F.mm, F.bench, F.template

LEAKED = {"9wW-losmGws_000097", "JixXL5s6fnE_000070", "K_EsxukdNXM_000030", "o4HqWhVTW2o_000043", "rajXrf_Dzfw_000029", "sKtU5y4czNc_000045"}
DIRS = ("a2v", "v2a")


def _retrieval():
    """The folder's scorer module (``retrieval.py``, loaded by ``eval.py``)."""
    return F.load_module(HERE / "eval.py").retrieval


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


def test_ranks_match_a_naive_loop():
    R = _retrieval()
    rng = np.random.default_rng(1)
    q, g = rng.standard_normal((40, 8)), rng.standard_normal((40, 8))
    g[:20] = q[:20] + 0.01 * rng.standard_normal((20, 8))  # half the pairs are easy
    ours = R.ranks(q, g)
    qn, gn = q / np.linalg.norm(q, axis=1, keepdims=True), g / np.linalg.norm(g, axis=1, keepdims=True)
    for i in range(40):
        sims = [float(qn[i] @ gn[j]) for j in range(40)]
        assert ours[i] == 1 + sum(1 for s in sims if s > sims[i]) + (sum(1 for s in sims if s == sims[i]) - 1) / 2
    out = np.stack([ours, R.ranks(g, q)], axis=1)
    m = R.make_metrics(HERE.name, subsets=("cavmae",))
    assert m["r@1_a2v"].fn(out, None, {}, {}) == float((ours <= 1).mean()) and m["medr_a2v"].fn(out, None, {}, {}) == float(np.median(ours))
    assert m["r@1"].fn(out, None, {}, {}) == pytest.approx(0.5 * (m["r@1_a2v"].fn(out, None, {}, {}) + m["r@1_v2a"].fn(out, None, {}, {})))
    mask = np.arange(40) % 2 == 0
    sub = {"subsets": {"cavmae": {"mask": mask, "ranks": np.where(mask[:, None], out, 0)}}}
    assert m["r@1_a2v_cavmae"].fn(out, None, {}, sub) == float((ours[mask] <= 1).mean())


def test_rank_ties_average_and_chunking_is_invisible(monkeypatch):
    R = _retrieval()
    q = np.ones((5, 3))
    assert R.ranks(q, q).tolist() == [3.0] * 5  # a fully tied query ranks in the middle of the gallery
    assert R.ranks(np.zeros((4, 3)), np.eye(4, 3)).tolist() == [2.5] * 4  # a zero query (a silent clip) is not a hit
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((300, 6)), rng.standard_normal((300, 6))
    full = R.ranks(a, b)
    monkeypatch.setattr(R, "CHUNK", 7)
    assert np.array_equal(R.ranks(a, b), full)


def test_yaml_and_changelog(spec):
    assert spec["schema_version"] == 3 and spec["name"] == HERE.name == "vggsound_retrieval"
    assert spec["scorer"] == HERE.name and spec["aliases"] == [] and spec["protocol_version"] == 1
    assert spec["splits"] == {"train": [], "eval": ["test"], "floor": [], "folds": None, "unused": [], "min_eval_coverage": 0.95}
    assert spec["manifest"]["label"] == {"column": None, "kind": "none"} and spec["manifest"]["classes"] is None
    assert spec["protocol"] == {"clip_seconds": 10.0, "train_draw": None, "views": "none"}
    assert spec["readout_params"] == {"subsets": {"cavmae": "cavmae_sync_5pc"}} and spec["input_contract"]["facts"] == []


def test_output_contract_schema_and_isolation(spec):
    oc = spec["output_contract"]
    assert set(oc["schema"]) == {"audio_emb", "video_emb"} and oc["isolation"] == {"audio_emb": ["audio"], "video_emb": ["video"]}
    assert oc["requires"] == {"pos": [], "streams": [], "grid": None, "t_bins": None}


def test_metrics_block(spec):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    full = ["r@1"] + [f"r@{k}_{d}" for d in DIRS for k in (1, 5, 10)] + [f"medr_{d}" for d in DIRS]
    sub = [f"r@{k}_{d}_cavmae" for d in DIRS for k in (1, 5, 10)] + [f"medr_{d}_cavmae" for d in DIRS]
    assert set(m["list"]) == set(full) | set(sub) | {"n_eval"} and m["headline"] == "r@1"
    assert m["floors"] == [] and m["controls"] == []
    n, s = 15440, 1545
    for k in (1, 5, 10):
        for d in DIRS:
            assert m["chance"][f"r@{k}_{d}"] == pytest.approx(k / n, rel=0.02) and m["chance"][f"r@{k}_{d}_cavmae"] == pytest.approx(k / s, rel=0.02)
    assert m["ci"]["r@1"] == "bootstrap_clip" and m["ci"]["r@1_a2v_cavmae"] == "binomial"


def test_sha256_rows_and_columns(spec, manifest_rows):
    F.assert_manifest_hashes(HERE, spec)
    assert len(manifest_rows) == 15440 and list(manifest_rows[0]) == ["clip_id", "split", "class_name", "rel_video", "rel_audio", "cavmae_sync_5pc"]
    ids = [r["clip_id"] for r in manifest_rows]
    assert len(set(ids)) == len(ids) and ids == sorted(ids) and not (LEAKED & set(ids))
    for r in manifest_rows:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,32}_\d{6}", r["clip_id"]) and r["split"] == "test" and r["cavmae_sync_5pc"] in ("0", "1")
        assert r["rel_video"] == f"video/{r['clip_id']}.mp4" and r["rel_audio"] == f"audio/{r['clip_id']}.wav"
    assert F.assert_splits_assigned_once(spec, manifest_rows) == {"test": 15440}
    classes = Counter(r["class_name"] for r in manifest_rows)
    assert len(classes) == 309 and min(classes.values()) == 48 and max(classes.values()) == 50


def test_subset_list_is_pinned_and_flagged(spec, manifest_rows, mm):
    raw = (HERE / "cavmae_sync_subset.csv").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == mm.SUBSET_SHA == spec["manifest"]["upstream"]["upstream_files"]["cavmae_sync_subset.csv"]
    ids = [r["clip_id"] for r in csv.DictReader(raw.decode().splitlines())]
    flagged = {r["clip_id"] for r in manifest_rows if r["cavmae_sync_5pc"] == "1"}
    assert len(ids) == 1545 and set(ids) == flagged
    per_class = Counter(r["class_name"] for r in manifest_rows if r["cavmae_sync_5pc"] == "1")
    assert len(per_class) == 309 and set(per_class.values()) == {5}


def test_curation_sums(spec):
    cur = spec["manifest"]["upstream"]["curation"]
    assert cur["total_upstream"] == 15446 and cur["excluded"] == {"leakage": 6}
    assert cur["per_split"]["test"]["listed"] == 15440 and spec["budget"]["clips"] == {"test": 15440}


def test_benchmark_loads_and_subset_masks(bench):
    assert bench.spec.readout == HERE.name and bench.spec.output_contract.isolation == {"audio_emb": ("audio",), "video_emb": ("video",)}
    man = bench.manifest("test")
    assert sum(int(man.extra[i]["cavmae_sync_5pc"]) for i in range(bench.n_rows("test"))) == 1545
    ev = bench.eval_module()
    assert ev.SCORER.output_keys == frozenset({"audio_emb", "video_emb"}) and {f"r@1_{d}_cavmae" for d in DIRS} <= set(ev.METRICS)


def test_bridge_template_declares_the_outputs(template):
    from yavih.bridge import Bridge

    cls = next(v for v in vars(template).values() if isinstance(v, type) and issubclass(v, Bridge) and v is not Bridge)
    assert tuple(cls.outputs) == ("audio_emb", "video_emb")


def test_make_manifest_check_against_the_listing(mm):
    root = F.site_root(HERE, "media_root", needs=("metadata/vggsound.csv", "video", "audio"))
    if root is None:
        pytest.skip("VGGSound not staged")
    assert mm.main(["--check", "--vggsound-root", str(root)]) == 0
