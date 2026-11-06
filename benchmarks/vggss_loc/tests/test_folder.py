"""Folder tests for vggss_loc.

Run only this file: ``python -m pytest "$(yavih bench path vggss_loc)/tests" -q -p no:cacheprovider``.

Four groups: pure checks of the frozen artefacts (always run); the folder's scorer (``loc.py``) against a verbatim
transcription of the published VGG-SS evaluator (always run); checks through the core (``Benchmark``, the scorer's
targets, the rasteriser on pinned rows; always run); checks that need local data (VGGSound media, ``vggss.json`` and
``vggsound.csv``; they skip otherwise).
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pytest

from yavih.testing import folder as F

HERE = F.folder_of(__file__)
_isolated_yavih_home = F.isolated_yavih_home
spec, manifest_rows, mm, bench, template = F.spec, F.rows, F.mm, F.bench, F.template

LEAKED = {"9wW-losmGws_000097", "JixXL5s6fnE_000070", "K_EsxukdNXM_000030", "rajXrf_Dzfw_000029"}
# (clip_id, boxes cell, class)
PINNED = [
    ("-0BIyqJj9ZU_000030", "0.343750|0.000000|0.678125|0.734375", "people belly laughing"),
    ("zpWuikVorYg_000032", "0.403125|0.253125|0.600000|0.750000", "elk bugling"),
]


def _loc():
    """The folder's scorer module (``loc.py``, loaded by ``eval.py``)."""
    return F.load_module(HERE / "eval.py").loc


def test_folder_lint_clean():
    F.assert_lint_clean(HERE)


# ----------------------------------------------------------------------------- benchmark.yaml


def test_yaml_and_changelog(spec):
    assert spec["schema_version"] == 3 and spec["name"] == HERE.name == "vggss_loc"
    assert spec["scorer"] == HERE.name and spec["aliases"] == [] and spec["protocol_version"] == 1
    assert [c["version"] for c in spec["changelog"]] == [1] and spec["changelog"][0]["invalidates"] is False
    assert spec["splits"] == {"train": [], "eval": ["test"], "floor": [], "folds": None, "unused": [], "min_eval_coverage": 0.95}
    assert spec["manifest"]["label"] == {"column": None, "kind": "none"} and spec["manifest"]["classes"] is None


def test_protocol_and_readout_params(spec):
    assert spec["protocol"] == {"clip_seconds": 10.0, "train_draw": None, "views": "none", "gt_size": 224, "annotated_second": 5.0}
    assert spec["readout_params"] == {"grid": 14, "min_cell_coverage": 0.25, "boxes_column": "boxes", "shuffle_seed": 20260913}
    assert spec["input_contract"]["facts"] == ["annotated_second"]


def test_output_contract_schema_and_isolation(spec):
    oc = spec["output_contract"]
    assert oc["shape"] == "fixed" and oc["target_frame"] is None and oc["requires"] == {"pos": [], "streams": [], "grid": None, "t_bins": None}
    assert set(oc["schema"]) == {"audio_emb", "video_tokens", "video_pos"}
    assert oc["schema"]["video_pos"] == {"type": "array", "dtype": "float64", "shape": ["*", 2], "exact": True}
    assert oc["isolation"] == {"audio_emb": ["audio"], "video_tokens": ["video"], "video_pos": ["video"]}


def test_metrics_block(spec):
    F.assert_metric_names_resolve(HERE, spec)
    m = spec["metrics"]
    assert m["headline"] == "ciou@0.5" and m["list"] == ["ciou@0.5", "auc", "ciou_mean", "n_empty_gt", "n_eval"]
    assert m["floors"] == ["ciou_floor"] and m["controls"] == ["ciou_shuffled", "audio_blind"] and m["chance"] == {}
    assert m["ci"] == {"ciou@0.5": "binomial", "ciou_mean": "bootstrap_clip"}
    assert spec["external"] == [] and (HERE / "eval.py").is_file() and (HERE / "loc.py").is_file()


# ----------------------------------------------------------------------------- manifest


def test_sha256_rows_and_columns(spec, manifest_rows):
    F.assert_manifest_hashes(HERE, spec)
    assert len(manifest_rows) == 5154 and list(manifest_rows[0]) == ["clip_id", "split", "boxes", "class_name", "rel_video", "rel_audio"]
    ids = [r["clip_id"] for r in manifest_rows]
    assert len(set(ids)) == len(ids) and ids == sorted(ids) and not (LEAKED & set(ids))
    for r in manifest_rows:
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,32}_\d{6}", r["clip_id"]) and r["split"] == "test"
        assert r["rel_video"] == f"video/{r['clip_id']}.mp4" and r["rel_audio"] == f"audio/{r['clip_id']}.wav"
    assert F.assert_splits_assigned_once(spec, manifest_rows) == {"test": 5154}
    assert len({r["class_name"] for r in manifest_rows}) == 221


def test_boxes_cells(manifest_rows):
    per_entry = Counter()
    empty = 0
    for r in manifest_rows:
        quads = r["boxes"].split(";")
        per_entry[len(quads)] += 1
        for q in quads:
            parts = q.split("|")
            assert len(parts) == 4 and all(re.fullmatch(r"\d\.\d{6}", p) for p in parts)
            assert all(0.0 <= float(p) <= 1.0 for p in parts)  # sanitised: never negative
        x1, y1, x2, y2 = (float(p) for p in quads[0].split("|"))
        empty += len(quads) == 1 and not (x2 > x1 and y2 > y1)
    assert per_entry == {1: 4616, 2: 384, 3: 94, 4: 37, 5: 18, 6: 3, 7: 2} and empty == 2
    by = {r["clip_id"]: r for r in manifest_rows}
    for cid, boxes, cls in PINNED:
        assert by[cid]["boxes"] == boxes and by[cid]["class_name"] == cls


def test_curation_sums(spec):
    cur = spec["manifest"]["upstream"]["curation"]
    assert cur["total_upstream"] == 5158 and cur["excluded"] == {"leakage": 4}
    assert cur["per_split"]["test"]["listed"] == 5154 and spec["budget"]["clips"] == {"test": 5154}


# ----------------------------------------------------------------------------- the scorer vs the published evaluator


def _lvs_testset_gt(bboxes, size=224):
    """``testset_gt`` of the LVS ``utils.py`` for VGG-SS (the union of ``int(224 * x)`` boxes), verbatim."""
    gt_map = np.zeros([size, size])
    for bbox in bboxes:
        temp = np.zeros([size, size])
        (xmin, ymin, xmax, ymax) = [int(size * max(x, 0)) for x in bbox]  # the per-coordinate sanitisation of test.py
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp
    gt_map[gt_map > 1] = 1
    return gt_map


def _lvs_cal_ciou(infer_map_upsampled, gtmap):
    """``test.py``'s thresholding + ``Evaluator.cal_CIOU`` (threshold 0.5 on a {0, 1} map), verbatim."""
    pred = infer_map_upsampled.copy()
    threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
    pred[pred > threshold] = 1
    pred[pred < 1] = 0
    infer_map = np.zeros_like(gtmap)
    infer_map[pred >= 0.5] = 1
    return np.sum(infer_map * gtmap) / (np.sum(gtmap) + np.sum(infer_map * (gtmap == 0)))


def _lvs_finalize(cious):
    cious = np.asarray(cious)
    results = [np.sum(cious >= 0.05 * i) / len(cious) for i in range(21)]
    x = [0.05 * i for i in range(21)]
    auc = np.trapezoid(results, x) if hasattr(np, "trapezoid") else np.trapz(results, x)
    return float(np.sum(cious >= 0.5) / len(cious)), float(auc)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n, g = 24, 14
    maps = rng.standard_normal((n, g, g)).astype(np.float32)
    boxes = []
    for i in range(n):
        bb = []
        for _ in range(int(rng.integers(1, 4))):
            x1, y1 = rng.uniform(0, 0.6, 2)
            bb.append([x1, y1, x1 + rng.uniform(0.2, 0.4), y1 + rng.uniform(0.2, 0.4)])
        if i == 0:
            bb[0] = [-2.8823e16, 0.1, 0.5, 0.6]  # a corrupt upstream coordinate
        if i == 1:
            bb = [[0.3, 0.3, 0.3, 0.3]]  # degenerate: empty ground truth
        boxes.append(np.asarray(bb))
    return maps, boxes


def test_rasterizer_matches_the_published_one(toy):
    _, boxes = toy
    ours = _loc().rasterize(boxes, 224)
    for i, bb in enumerate(boxes):
        assert np.array_equal(ours[i], _lvs_testset_gt(bb))
    assert ours[1].sum() == 0


def test_bilinear_upsample_matches_torch(toy):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as tf

    maps, _ = toy
    ref = tf.interpolate(torch.from_numpy(maps)[:, None].double(), size=(224, 224), mode="bilinear", align_corners=False)[:, 0].numpy()
    assert np.allclose(_loc().upsample_bilinear(maps, 224), ref, atol=1e-9)


def test_metrics_match_the_published_scorer(toy):
    L = _loc()
    maps, boxes = toy
    protocol = {"gt_size": 224}
    up = L.upsample_bilinear(maps, 224)
    ref = np.asarray([_lvs_cal_ciou(up[i], _lvs_testset_gt(b)) if _lvs_testset_gt(b).sum() > 0 else 0.0 for i, b in enumerate(boxes)])
    assert np.allclose(L.cious(maps, boxes, protocol), ref)
    at05, auc = _lvs_finalize(ref)
    m = L.make_metrics(HERE.name)
    assert m["ciou@0.5"].fn(maps, boxes, protocol, {}) == pytest.approx(at05)
    assert m["auc"].fn(maps, boxes, protocol, {}) == pytest.approx(auc)
    assert m["ciou_mean"].fn(maps, boxes, protocol, {}) == pytest.approx(ref.mean())
    assert m["n_empty_gt"].fn(maps, boxes, protocol, {}) == 1
    same = {"shuffled": {"outputs": maps.copy()}}
    assert m["audio_blind"].fn(maps, boxes, protocol, same) is True  # indistinguishable arms are flagged
    assert m["ciou_shuffled"].fn(maps, boxes, protocol, same) == pytest.approx(at05)


def test_median_mask_floor_maps_and_pooling(toy):
    L = _loc()
    assert L.median_mask(np.arange(16, dtype=np.float64).reshape(1, 4, 4)).sum() == 7  # sorted[8] = 8 is the threshold
    assert L.median_mask(np.full((1, 4, 4), 3.0)).sum() == 0  # a flat map has no pixel strictly above its median
    _, boxes = toy
    floor = L.mean_mask_maps(boxes, len(boxes), grid=14, size=224, leave_one_out=True)
    full = L.rasterize(boxes, 224).astype(np.float64)
    for i in (0, 5):
        assert np.allclose(floor[i], L.pool_to_grid(np.delete(full, i, axis=0).mean(axis=0)[None], 14)[0], atol=1e-6)
    train = L.mean_mask_maps(boxes[:10], 3, grid=14, size=224, leave_one_out=False)
    assert train.shape == (3, 14, 14) and np.allclose(train[0], train[2])


# ----------------------------------------------------------------------------- through the core


def test_benchmark_loads_targets_and_ceiling(bench):
    L = _loc()
    rasterize = L.rasterize

    assert bench.spec.readout == HERE.name and bench.spec.output_contract.isolation == {"audio_emb": ("audio",), "video_tokens": ("video",), "video_pos": ("video",)}
    man = bench.manifest("test")
    rows = np.arange(bench.n_rows("test"))
    cells = [man.extra[i]["boxes"] for i in range(len(rows))]
    targets = [np.asarray([[float(v) for v in q.split("|")] for q in c.split(";")]) for c in cells]
    area = rasterize(targets, 224).reshape(len(targets), -1).mean(axis=1)
    assert int((area >= 0.25).sum()) == 2603 and int((area == 0).sum()) == 2  # the ceiling of ciou@0.5 and n_empty_gt
    assert bench.eval_module().SCORER.output_keys == frozenset({"audio_emb", "video_tokens", "video_pos"})


def test_bridge_template_declares_the_outputs(template):
    from yavih.bridge import Bridge

    cls = next(v for v in vars(template).values() if isinstance(v, type) and issubclass(v, Bridge) and v is not Bridge)
    assert tuple(cls.outputs) == ("audio_emb", "video_tokens", "video_pos")


# ----------------------------------------------------------------------------- with data


def test_make_manifest_check_against_the_annotations(mm):
    staging = F.site_root(HERE, "annotations_root", needs=("vggss.json",))
    root = F.site_root(HERE, "media_root", needs=("metadata/vggsound.csv", "video", "audio"))
    if staging is None or root is None:
        pytest.skip("VGG-SS annotations or VGGSound media not staged")
    assert mm.main(["--check", "--staging-dir", str(staging), "--vggsound-root", str(root)]) == 0
