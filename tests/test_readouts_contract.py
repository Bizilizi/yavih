"""Readout contract (docs/readouts.md, docs/results.md, folder ``eval.py``): media-coordinate
binning (seq by absolute second, grid/box by full-frame fractions over the target region), per-sample
pos for box incl. the shuffled control, the folder ``eval.py`` hooks (targets, external scorer), the
assert-import of the authors' packages (exit 3 with the pip line, version recorded), and the
lm-evaluation-harness results store + samples file. Reuses the fakes of ``test_readouts.py``."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from test_readouts import (
    D_FEAT,
    DESCRIBE,
    MODEL,
    READING,
    FakeBench,
    avsync_probe,
    classes_list,
    clf_cfg,
    clf_features,
    clf_rows,
    clf_yaml,
    dense_yaml,
    write_fake_features,
)
from test_readouts import (
    fakes as fakes,  # noqa: F401 - the fixture, re-exported for this module
)
from yavih import hashing
from yavih.probes.config import ProbeConfig
from yavih.probes.dense import build_dense_query_head, coord_present, pool_cells
from yavih.readouts import ProbeRefused, check_result, probe, result_dir
from yavih.readouts import coords as C
from yavih.readouts.samples import sample_lines


@pytest.fixture(autouse=True)
def _no_site_results(monkeypatch):
    monkeypatch.delenv("YAVIH_RESULTS", raising=False)


# ============================================================================ coordinates (numpy)


def test_seq_task_pos_reads_tokens_at_absolute_seconds():
    """A token at media second τ gets u = (τ − off·seg) / (T·seg) whatever the clip's duration,
    the frame count or the window the bridge chose."""
    T, off, seg = 10, 0, 1.0
    # bridge A: 20 frames over a 10 s clip; bridge B: 8 tokens over a 12.5 s clip
    ta = (np.arange(20) + 0.5) / 20
    tb = (np.arange(8) + 0.5) / 8
    pa = np.stack([np.zeros(20), ta, -np.ones(20), -np.ones(20)], 1)[None].astype(np.float32)
    pb = np.stack([np.zeros(8), tb, -np.ones(8), -np.ones(8)], 1)[None].astype(np.float32)
    ua = C.seq_task_pos(pa, np.array([[10.0]]), T=T, segment_offset=off, segment_seconds=seg)[0, :, 1]
    ub = C.seq_task_pos(pb, np.array([[12.5]]), T=T, segment_offset=off, segment_seconds=seg)[0, :, 1]
    np.testing.assert_allclose(ua, ta * 10.0 / 10.0, rtol=1e-6)
    np.testing.assert_allclose(ub, tb * 12.5 / 10.0, rtol=1e-6)  # past the scored span: u > 1 is KEPT
    assert ub.max() > 1.0
    # the same absolute second gives the same u
    same = C.seq_task_pos(np.array([[[0, 0.4, -1, -1]], [[0, 0.32, -1, -1]]], np.float32), np.array([[10.0], [12.5]]),
                          T=T, segment_offset=off, segment_seconds=seg)[:, 0, 1]
    assert same[0] == pytest.approx(same[1]) == pytest.approx(0.4)
    # an offset window: second 1 is the scored origin
    u = C.seq_task_pos(np.array([[[0, 0.05, -1, -1]]], np.float32), np.array([[10.0]]), T=8, segment_offset=1, segment_seconds=1.0)
    assert u[0, 0, 1] == pytest.approx((0.5 - 1.0) / 8)  # negative and kept
    # untimed tokens, untagged tokens and streams without a duration are null (NaN), never -1
    p = np.array([[[0, -1, -1, -1], [-1, 0.5, -1, -1], [1, 0.5, -1, -1]]], np.float32)
    out = C.seq_task_pos(p, np.array([[10.0, np.nan]]), T=T, segment_offset=0, segment_seconds=1.0)
    assert np.isnan(out[0, :, 1]).all() and np.isnan(out[0, :, 2:]).all()


def test_region_task_pos_and_target_region():
    assert C.target_region({}) == C.FULL_FRAME
    region = C.target_region({"target_region": [0.0, 0.125, 1.0, 0.875]})
    p = np.array([[0, -1, 0.5, 0.5], [0, -1, 0.25, 0.0], [0, -1, -1, 0.3], [1, 0.5, -1, -1]], np.float32)
    out = C.region_task_pos(p, region)
    assert out[0, 2:].tolist() == pytest.approx([0.5, 0.5])
    assert out[1, 3] == pytest.approx(-0.125 / 0.75)  # outside the region: kept, out of range
    assert np.isnan(out[2, 2]) and out[2, 3] == pytest.approx((0.3 - 0.125) / 0.75)
    assert np.isnan(out[0, 1]) and out[3, 1] == pytest.approx(0.5)
    # the full frame is the identity on present coordinates
    np.testing.assert_allclose(C.region_task_pos(p, C.FULL_FRAME)[:2, 2:], p[:2, 2:])
    with pytest.raises(ValueError):
        C.target_region({"target_region": [0.5, 0.0, 0.4, 1.0]})


def test_task_pos_gather_constant_and_per_row():
    base = np.array([[0, 0.5, -1, -1]], np.float32)
    const = C.task_pos(base, lambda p, d: C.region_task_pos(p), n=3, N=1)
    assert isinstance(const, np.ndarray) and const.shape == (1, 4)
    per_row = C.task_pos(base, lambda p, d: C.seq_task_pos(p, d, T=10, segment_offset=0, segment_seconds=1.0), n=3, N=1,
                         per_row=np.array([[10.0], [5.0], [20.0]]))
    assert per_row.shape == (3, 1, 4) and per_row[np.array([0, 2])][:, 0, 1].tolist() == pytest.approx([0.5, 1.0])
    assert per_row[1][0, 1] == pytest.approx(0.25)


def test_heads_treat_nan_as_null_and_bin_cells_inside_the_region():
    assert coord_present(torch.tensor([0.5, -1.0, float("nan"), 1.7, -0.2])).tolist() == [True, False, False, True, True]
    g = 2
    pos = torch.tensor([[[0, -1, 0.25, 0.25], [0, -1, 1.0, 1.0], [0, -1, -0.1, 0.5], [0, -1, float("nan"), 0.5], [1, 0.5, 0.25, 0.25]]])
    mem = torch.ones(1, 5, 4)
    _, counts = pool_cells(mem, pos, grid=g, stream_tag=0)
    assert counts.tolist() == [[1.0, 0.0, 0.0, 1.0]]  # outside the region / null / other stream: no cell
    # a dense query head embeds out-of-range task coordinates (no clamping): u = 1.5 differs from u = 1.0
    head = build_dense_query_head(16, 3, ProbeConfig.pinned("seq", num_classes=3, T=4).head).eval()
    x = torch.randn(1, 2, 16)
    p1 = torch.tensor([[0, 1.0, float("nan"), float("nan")], [0, 0.2, float("nan"), float("nan")]])
    p2 = p1.clone()
    p2[0, 1] = 1.5
    assert not torch.allclose(head(x, p1), head(x, p2))


# ============================================================================ seq / grid / box through the driver


def _seq_world(tmp_path: Path, *, n_tok: int, duration: float, name: str = "ave_like"):
    T, Cn = 10, 5
    protocol = {"num_segments": 10, "segment_seconds": 1.0, "segments_scored": T, "segment_offset": 0, "background_index": 4}
    d = dense_yaml(name, "seq", protocol=protocol, requires={"pos": ["t"], "streams": ["video"], "grid": None, "t_bins": 8}, readout_params={"T": T},
                   metrics=["seg_acc", "clip_acc", "seg_acc_boundary", "n_boundary", "n_eval"], floors=["seg_acc_floor", "seg_acc_boundary_floor"],
                   controls=[], ci={}, chance={"seg_acc": 0.2},
                   label={"column": "label", "kind": "segments", "sep": ";", "class_key": "name", "sets": None, "train_set": None,
                          "allow_unknown": False, "empty_ok": False},
                   classes="classes.csv", streams=("video",))
    rows = [{"clip_id": f"{s}_{k}", "split": s, "label_ids": k % 4, "extra": {"event_start": k % 3, "event_end": 6}, "segment": [k % 3, 6]}
            for s, n in (("train", 8), ("val", 4)) for k in range(n)]
    bench = FakeBench(d, rows, classes_list(Cn))
    run = tmp_path / name / "runs" / name / MODEL / READING
    pos = np.stack([np.zeros(n_tok), (np.arange(n_tok) + 0.5) / n_tok, -np.ones(n_tok), -np.ones(n_tok)], 1).astype(np.float32)
    for split, n in (("train", 8), ("val", 4)):
        write_fake_features(run, split, bench=bench, feats=np.random.default_rng(0).standard_normal((n, n_tok, D_FEAT)).astype(np.float32),
                            pos=pos, durations={"video": np.full(n, duration)}, describe={**DESCRIBE, "streams": ["video"]})
    return bench, run, ProbeConfig.pinned("seq", num_classes=Cn, T=T).fast()


def test_seq_driver_feeds_absolute_second_coordinates(tmp_path, fakes):
    """A bridge covering [0, 10] with 20 frames and one covering it with 8 tokens see the same targets;
    the head reads each token at its absolute second."""
    bench_a, run_a, cfg = _seq_world(tmp_path, n_tok=20, duration=10.0, name="a")
    bench_b, run_b, _ = _seq_world(tmp_path, n_tok=8, duration=10.0, name="b")
    assert probe(run_a, bench=bench_a, cfg=cfg)[0].status == "ok"
    assert probe(run_b, bench=bench_b, cfg=cfg)[0].status == "ok"
    ta, tb = fakes[0]["targets"], fakes[1]["targets"]
    assert np.array_equal(ta, tb)
    pa, pb = fakes[0]["pos_obj"], fakes[1]["pos_obj"]
    np.testing.assert_allclose(np.asarray(pa[np.array([0])])[0, :, 1], (np.arange(20) + 0.5) / 20, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(pb[np.array([0])])[0, :, 1], (np.arange(8) + 0.5) / 8, rtol=1e-6)


def test_seq_longer_media_moves_tokens_in_u(tmp_path, fakes):
    bench, run, cfg = _seq_world(tmp_path, n_tok=8, duration=12.5)
    probe(run, bench=bench, cfg=cfg)
    u = np.asarray(fakes[0]["pos_obj"][np.array([0])])[0, :, 1]
    np.testing.assert_allclose(u, (np.arange(8) + 0.5) / 8 * 1.25, rtol=1e-6)  # τ = t · 12.5 s; tokens past 10 s keep u > 1


def test_seq_refuses_timed_tokens_without_durations(tmp_path, fakes):
    bench, run, cfg = _seq_world(tmp_path, n_tok=8, duration=float("nan"))
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=cfg)
    assert e.value.code == "E_DURATIONS" and "video" in str(e.value)


def _grid_world(tmp_path: Path, *, region=None):
    S, H = 8, 2
    protocol = {"max_depth": 10.0, "target_size": S, "depth_param": "log", "source_size": [480, 640]}
    if region is not None:
        protocol["target_region"] = region
    targets_yaml = {s: {"kind": "dense_depth", "source": "derived_file", "root": "derived_root", "file": f"{s}.npy", "sha256": "1" * 64, "index_column": None}
                    for s in ("train", "val")}
    d = dense_yaml("nyu_like", "grid", protocol=protocol, requires={"pos": ["y", "x"], "streams": ["image"], "grid": {"h": H, "w": H, "min_cell_coverage": 0.5},
                                                                      "t_bins": None},
                   readout_params={"h": H, "w": H, "out": 1, "visual_stream": "image"}, metrics=["d1", "d2", "d3", "absrel", "rmse", "n_eval"],
                   floors=["d1_floor"], controls=[], ci={}, chance={"d1": 0.0},
                   label={"column": None, "kind": "none", "sep": ";", "class_key": "name", "sets": None, "train_set": None, "allow_unknown": False,
                          "empty_ok": True},
                   classes=None, targets=targets_yaml, streams=("image",))
    from test_readouts import FakeTargetReader

    rng = np.random.default_rng(0)
    depth = {"train": rng.uniform(1.0, 5.0, (6, S, S)).astype(np.float32), "val": rng.uniform(1.0, 5.0, (3, S, S)).astype(np.float32)}
    rows = [{"clip_id": f"{s}_{k}", "split": s, "label_ids": None} for s, n in (("train", 6), ("val", 3)) for k in range(n)]
    bench = FakeBench(d, rows, None, targets={s: FakeTargetReader(a) for s, a in depth.items()})
    run = tmp_path / "runs" / "nyu_like" / MODEL / READING
    # a 4:3 frame read by a centre-square bridge: tokens over x ∈ [0.125, 0.875] of the full frame
    yy, xx = np.meshgrid((np.arange(H) + 0.5) / H, 0.125 + 0.75 * (np.arange(H) + 0.5) / H, indexing="ij")
    pos = np.stack([np.zeros(H * H), -np.ones(H * H), yy.ravel(), xx.ravel()], 1).astype(np.float32)
    for split, n in (("train", 6), ("val", 3)):
        write_fake_features(run, split, bench=bench, feats=rng.standard_normal((n, H * H, D_FEAT)).astype(np.float32), pos=pos,
                            describe={**DESCRIBE, "streams": ["image"]})
    return bench, run, ProbeConfig.pinned("grid", grid=(H, H), target_size=S).fast(), pos


def test_grid_pos_is_region_relative(tmp_path, fakes):
    bench, run, cfg, media = _grid_world(tmp_path, region=[0.0, 0.125, 1.0, 0.875])
    assert probe(run, bench=bench, cfg=cfg)[0].status == "ok"
    task = np.asarray(fakes[0]["pos_obj"])
    assert task.shape == media.shape  # constant pos stays constant
    np.testing.assert_allclose(task[:, 2], media[:, 2], rtol=1e-6)
    np.testing.assert_allclose(task[:, 3], (media[:, 3] - 0.125) / 0.75, rtol=1e-6)
    assert np.isnan(task[:, 1]).all()  # images carry no t: null, never -1


def test_grid_full_frame_without_target_region(tmp_path, fakes):
    bench, run, cfg, media = _grid_world(tmp_path)
    probe(run, bench=bench, cfg=cfg)
    np.testing.assert_allclose(np.asarray(fakes[0]["pos_obj"])[:, 2:], media[:, 2:], rtol=1e-6)


def _box_world(tmp_path: Path, *, per_sample: bool):
    G = 2
    d = dense_yaml("vggss_like", "box", protocol={"box_population": "single", "gt_geometry": "full_frame", "heatmap_grid": G, "annotated_second": 5.0},
                   requires={"pos": ["y", "x"], "streams": ["video", "audio"], "grid": {"h": G, "w": G, "min_cell_coverage": 0.25}, "t_bins": None},
                   readout_params={"grid": G, "visual_stream": "video", "conditioning_stream": "audio", "shuffled_control": True, "shuffle_seed": 7},
                   metrics=["iou_mean", "acc@0.5", "acc@0.3", "center_in_gt", "n_eval", "n_excluded_degenerate"], floors=["iou_floor"],
                   controls=["iou_shuffled", "audio_blind"], ci={}, chance={"iou_mean": 0.0},
                   label={"column": "box", "kind": "box", "sep": ";", "class_key": "name", "sets": None, "train_set": None, "allow_unknown": False,
                          "empty_ok": False},
                   classes=None)
    rows = [{"clip_id": f"{s}_{k}", "split": s, "label_ids": None, "box": [0.1 + 0.05 * (k % 5), 0.2, 0.5 + 0.05 * (k % 5), 0.5]}
            for s, n in (("train", 8), ("val", 4)) for k in range(n)]
    bench = FakeBench(d, rows, None)
    run = tmp_path / "runs" / "vggss_like" / MODEL / READING
    nv, na = G * G, 3
    rng = np.random.default_rng(3)
    for split, n in (("train", 8), ("val", 4)):
        per = []
        for k in range(n):
            # 16:9 frames read by a resize-short + centre-crop bridge: x spans the centre 9/16 of the frame,
            # and that span differs per clip when frame sizes differ → per-sample pos
            half = 0.28125 if not per_sample else 0.25 + 0.01 * (k % 3)
            yy, xx = np.meshgrid((np.arange(G) + 0.5) / G, 0.5 - half + 2 * half * (np.arange(G) + 0.5) / G, indexing="ij")
            per.append(np.concatenate([np.stack([np.zeros(nv), -np.ones(nv), yy.ravel(), xx.ravel()], 1),
                                       np.stack([np.ones(na), (np.arange(na) + 0.5) / na, -np.ones(na), -np.ones(na)], 1)]))
        pos = np.asarray(per, np.float32) if per_sample else np.asarray(per[0], np.float32)
        write_fake_features(run, split, bench=bench, feats=rng.standard_normal((n, nv + na, D_FEAT)).astype(np.float32), pos=pos)
    return bench, run, ProbeConfig.pinned("box", grid=G).fast(), nv


def test_box_per_sample_pos_and_shuffled_control(tmp_path, fakes):
    bench, run, cfg, nv = _box_world(tmp_path, per_sample=True)
    rec = probe(run, bench=bench, cfg=cfg)[0].result
    check_result(rec, expect={"metrics": list(bench.spec.metrics.list), "floors": ["iou_floor"], "controls": ["iou_shuffled", "audio_blind"], "ci": {}})
    assert len(fakes) == 2 and fakes[0]["pos"] == (8, nv + 3, 4)  # per-sample pos reaches the head per row
    main, shuf = fakes[0]["features"], fakes[1]["features"]
    a, b = np.asarray(main[np.arange(8)]), np.asarray(shuf[np.arange(8)])
    assert np.array_equal(a[:, :nv], b[:, :nv]) and np.array_equal(b[:, nv:], a[shuf.perm][:, nv:])


def test_box_constant_pos_still_constant(tmp_path, fakes):
    bench, run, cfg, nv = _box_world(tmp_path, per_sample=False)
    assert probe(run, bench=bench, cfg=cfg)[0].status == "ok"
    assert fakes[0]["pos"] == (nv + 3, 4)


# ============================================================================ folder eval.py + external packages


def _fake_package(root: Path, name: str, version: str) -> Path:
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"__version__ = {version!r}\n\ndef official_acc(pred, target):\n    return float((pred == target).mean())\n")
    dist = root / f"{name}-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return root


PROBE_PY = '''
"""Folder eval.py fixture: an external metric computed by the authors' package."""
from types import SimpleNamespace as MetricSpec  # the registry's MetricSpec shape; the driver reads attributes only

import numpy as np

def _never(*a):
    raise AssertionError("an external metric must go through score()")

METRICS = {
    "official_acc": MetricSpec(name="official_acc", fn=_never, readouts=frozenset({"clf"}), protocol_keys=frozenset(), shape="scalar",
                               kind="metric", external="fakeauthors", floor_of=(), higher_is_better=True),
    "official_acc_floor": MetricSpec(name="official_acc_floor", fn=_never, readouts=frozenset({"clf"}), protocol_keys=frozenset(),
                                     shape="scalar", kind="floor", external="fakeauthors", floor_of=("official_acc",), higher_is_better=True),
}
PROTOCOL_KEYS = frozenset()
CALLS = []

def score(bench, split, rows, outputs, targets):
    from yavih.external import require  # noqa: F401 - the assert-import path when the core module exists
    import fakeauthors
    CALLS.append((split, len(rows)))
    return {"official_acc": fakeauthors.official_acc(np.asarray(outputs).argmax(1), np.asarray(targets))}
'''


def _external_world(tmp_path: Path, *, with_targets_hook: bool = False):
    n_classes = 3
    d = clf_yaml("ext_clf", n_classes=n_classes, metrics=["acc", "official_acc", "n_eval"], floors=["acc_floor", "official_acc_floor"],
                 chance={"acc": 0.33})
    d["external"] = [{"name": "fakeauthors", "pip": "pip install fakeauthors==1.0", "version_pin": "1.0", "used_for": "official_acc"}]
    rows = clf_rows("train", 12, n_classes) + clf_rows("val", 6, n_classes)
    bench = FakeBench(d, rows, classes_list(n_classes))
    folder = tmp_path / "folder"
    folder.mkdir()
    src = PROBE_PY.replace("from yavih.external import require  # noqa: F401 - the assert-import path when the core module exists\n    ", "")
    if with_targets_hook:
        src += "\ndef targets(bench, split, rows):\n    return np.zeros(len(rows), dtype=np.int64)\n"
    (folder / "eval.py").write_text(textwrap.dedent(src))
    bench.folder = folder
    run = tmp_path / "runs" / bench.name / MODEL / READING
    for split in ("train", "val"):
        labels = np.asarray([r["label_ids"] for r in rows if r["split"] == split])
        write_fake_features(run, split, bench=bench, feats=clf_features(labels, n_classes))
    return bench, run


def test_external_package_absent_exits_3_with_the_pip_line_before_training(tmp_path, fakes):
    bench, run = _external_world(tmp_path)
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=clf_cfg(3))
    assert e.value.exit_code == 3 and e.value.code == "E_EXTERNAL_MISSING" and "pip install fakeauthors==1.0" in str(e.value)
    assert fakes == []  # nothing was trained


def test_external_package_wrong_version_exits_3(tmp_path, fakes, monkeypatch):
    bench, run = _external_world(tmp_path)
    monkeypatch.syspath_prepend(str(_fake_package(tmp_path / "site", "fakeauthors", "0.9")))
    with pytest.raises(ProbeRefused) as e:
        probe(run, bench=bench, cfg=clf_cfg(3))
    assert e.value.exit_code == 3 and e.value.code == "E_EXTERNAL_VERSION" and "1.0" in str(e.value)
    assert fakes == []


def test_external_scorer_is_used_and_its_version_recorded(tmp_path, fakes, monkeypatch):
    bench, run = _external_world(tmp_path)
    monkeypatch.syspath_prepend(str(_fake_package(tmp_path / "site", "fakeauthors", "1.0")))
    res = probe(run, bench=bench, cfg=clf_cfg(3), log_samples=True)
    rec = res[0].result
    check_result(rec, expect={"metrics": ["acc", "official_acc", "n_eval"], "floors": ["acc_floor", "official_acc_floor"], "controls": [],
                              "ci": {"acc": "binomial"}})
    assert rec["metrics"]["official_acc"] == rec["metrics"]["acc"] == 1.0
    # the floor went through the same scorer on the floor outputs (floor_of → official_acc)
    assert rec["floors"]["official_acc_floor"] == pytest.approx(rec["floors"]["acc_floor"])
    assert rec["external"] == {"fakeauthors": "1.0"}
    doc = json.loads(res[0].results_file.read_text())
    assert doc["config"]["external"] == {"fakeauthors": "1.0"}
    assert doc["configs"][f"ext_clf:{READING}"]["external"][0]["version_pin"] == "1.0"


def test_external_finer_identity_recorded_by_the_folder_reaches_the_result(tmp_path, fakes, monkeypatch):
    """A folder whose pin is finer than the package version (Probe3D reports 1.0 at every
    commit) records the code it checked (yavih.external.record_version); the result names THAT code."""
    from yavih import external

    bench, run = _external_world(tmp_path)
    src = (bench.folder / "eval.py").read_text().replace(
        "    import fakeauthors\n",
        "    import fakeauthors\n    from yavih.external import record_version\n    record_version('fakeauthors', '1.0+abc1234 (scorer.py sha256 0123456789ab)')\n",
    )
    (bench.folder / "eval.py").write_text(src)
    monkeypatch.syspath_prepend(str(_fake_package(tmp_path / "site", "fakeauthors", "1.0")))
    external.reset()
    res = probe(run, bench=bench, cfg=clf_cfg(3))
    assert res[0].result["external"] == {"fakeauthors": "1.0+abc1234 (scorer.py sha256 0123456789ab)"}
    assert json.loads(res[0].results_file.read_text())["config"]["external"] == res[0].result["external"]
    with pytest.raises(KeyError):
        external.record_version("never_required", "x")
    external.reset()


def test_external_install_hint_is_runnable():
    from yavih.external import status

    assert status("no_such_pkg_xyz", "pip install loguru -e git+https://x/y.git@abc#egg=evals", "1.0")["pip"] == "pip install loguru -e git+https://x/y.git@abc#egg=evals"
    assert status("no_such_pkg_xyz", "pip install --no-deps vggsounder==1.0", "1.0")["pip"] == "pip install --no-deps vggsounder==1.0"


def test_folder_targets_hook_replaces_target_construction(tmp_path, fakes, monkeypatch):
    bench, run = _external_world(tmp_path, with_targets_hook=True)
    monkeypatch.syspath_prepend(str(_fake_package(tmp_path / "site", "fakeauthors", "1.0")))
    rec = probe(run, bench=bench, cfg=clf_cfg(3))[0].result
    assert (np.asarray(fakes[0]["targets"]) == 0).all()
    assert rec["floors"]["acc_floor"] == 1.0  # every hooked target is class 0


# ============================================================================ results store + samples


def _clf_world(tmp_path: Path, n_train=12, n_eval=6, n_classes=3):
    d = clf_yaml("synth_clf", n_classes=n_classes)
    rows = clf_rows("train", n_train, n_classes) + clf_rows("val", n_eval, n_classes)
    bench = FakeBench(d, rows, classes_list(n_classes))
    run = tmp_path / "runs" / bench.name / MODEL / READING
    for split in ("train", "val"):
        labels = np.asarray([r["label_ids"] for r in rows if r["split"] == split])
        write_fake_features(run, split, bench=bench, feats=clf_features(labels, n_classes))
    return bench, run


def test_results_file_and_samples_file(tmp_path, fakes):
    bench, run = _clf_world(tmp_path)
    res = probe(run, bench=bench, cfg=clf_cfg(3), seeds=(0, 1), log_samples=True)
    rf = res[0].results_file
    # default root: the directory `results` next to the runs root
    assert rf.parent == tmp_path / "results" / MODEL and rf.name.startswith("results_") and rf.suffix == ".json"
    assert all(r.results_file == rf for r in res)
    doc = json.loads(rf.read_text())
    assert set(doc) == {"results_version", "results", "configs", "versions", "higher_is_better", "n_samples", "config", "git_hash",
                        "yavih_version", "date", "env"}
    key = f"synth_clf:{READING}"  # <benchmark>:<reading>
    assert set(doc["results"]) == {key}
    r = doc["results"][key]
    assert r["acc"] == 1.0 and r["acc_floor"] == pytest.approx(1 / 3) and r["acc_ci95"] == res[0].result["ci95"]["acc"]
    assert r["n_eval"] == 6 and r["status"] == "ok" and r["headline"] == "acc" and r["n_seeds"] == 2 and r["coverage_eval"] == 1.0
    assert "acc_std" in r and "n_eval_std" not in r
    cfg = doc["configs"][key]
    assert cfg["protocol_hash"] == bench.protocol_hash and cfg["probe_hash"] == hashing.probe_hash(clf_cfg(3))
    assert (cfg["benchmark"], cfg["reading"]) == ("synth_clf", READING) and cfg["eval_hash"] == res[0].result["eval_hash"]
    assert cfg["bridge"]["describe"] == DESCRIBE and cfg["describe_sha256"] == hashing.describe_sha256(DESCRIBE)
    assert cfg["protocol_version"] == 2 and cfg["readout"] == "clf" and cfg["roles"]["eval"] == {"splits": ["val"], "n": 6}
    assert cfg["unit_results"] == [str(x.path) for x in res]
    assert doc["versions"] == {key: 2} and doc["higher_is_better"][key] == {"acc": True}
    assert doc["n_samples"][key] == {"train": 12, "eval": 6}
    assert doc["config"]["model"] == MODEL and doc["config"]["bridge"]["describe"] == DESCRIBE
    assert doc["config"]["probe"]["seeds"] == [0, 1] and doc["config"]["external"] == {}
    sf = res[0].samples_file
    assert sf.name == rf.name.replace("results_", f"samples_synth_clf_{READING}_").replace(".json", ".jsonl")
    lines = [json.loads(x) for x in sf.read_text().splitlines()]
    assert len(lines) == 12 and {x["seed"] for x in lines} == {0, 1}
    first = lines[0]
    assert set(first) == {"clip_id", "row", "seed", "role", "prediction", "target", "correct"}
    assert first["clip_id"] == "val_0000" and first["role"] == "eval" and first["correct"] is True


def test_results_root_override_and_no_results(tmp_path, fakes, monkeypatch):
    bench, run = _clf_world(tmp_path)
    res = probe(run, bench=bench, cfg=clf_cfg(3), results_root=tmp_path / "elsewhere")
    assert res[0].results_file.parent == tmp_path / "elsewhere" / MODEL and res[0].samples_file is None
    bench2, run2 = _clf_world(tmp_path / "second")
    monkeypatch.setenv("YAVIH_RESULTS", str(tmp_path / "env_root"))
    res2 = probe(run2, bench=bench2, cfg=clf_cfg(3), write_results=False)
    assert res2[0].results_file is None and not (tmp_path / "env_root").exists()
    res3 = probe(run2, bench=bench2, cfg=clf_cfg(3), seeds=(1,))
    assert res3[0].results_file.parent == tmp_path / "env_root" / MODEL


def test_not_applicable_goes_to_the_store_as_status_only(tmp_path, fakes):
    d = clf_yaml("synth_clf", n_classes=3)
    rows = clf_rows("train", 6, 3) + clf_rows("val", 3, 3)
    bench = FakeBench(d, rows, classes_list(3))
    run = tmp_path / "runs" / bench.name / MODEL / READING
    for split in ("train", "val"):
        labels = np.asarray([r["label_ids"] for r in rows if r["split"] == split])
        write_fake_features(run, split, bench=bench, feats=clf_features(labels, 3), eligible=False, missing=["pos.t"], reason="no time")
    res = probe(run, bench=bench, cfg=clf_cfg(3), log_samples=True)
    assert res[0].status == "not_applicable"
    doc = json.loads(res[0].results_file.read_text())
    assert doc["results"][f"synth_clf:{READING}"] == {"status": "not_applicable", "reason": "no time"}
    assert res[0].samples_file is not None and res[0].samples_file.read_text() == ""


def test_sample_lines_per_readout():
    rows, ids = np.array([3, 5]), ["a", "b"]
    multi = sample_lines([{"role": "eval", "outputs": np.array([[0.1, 0.9, 0.5], [0.8, 0.1, 0.2]]), "targets": np.array([[0, 1, 1], [1, 0, 0]], np.float32),
                           "rows": rows, "clip_ids": ids}], readout="clf", seed=0)
    assert multi[0]["target"] == [1, 2] and multi[0]["prediction"][0] == [1, 0.9] and len(multi[0]["prediction"]) == 3
    sets = sample_lines([{"role": "eval", "outputs": np.zeros((2, 3)), "targets": {"audible": np.eye(3)[:2], "visible": np.zeros((2, 3))},
                          "rows": rows, "clip_ids": ids, "folds": ["1", "2"]}], readout="clf", seed=1)
    assert sets[1]["target"] == {"audible": [1], "visible": []} and sets[1]["fold"] == "2"
    seq = sample_lines([{"role": "eval", "outputs": np.eye(3)[None].repeat(2, 0), "targets": np.array([[0, 1, 2], [2, 2, 2]]), "rows": rows,
                         "clip_ids": ids}], readout="seq", seed=0)
    assert seq[0]["prediction"] == [0, 1, 2] and seq[1]["target"] == [2, 2, 2]
    depth = np.full((2, 4, 4), 2.0, np.float32)
    grid = sample_lines([{"role": "eval", "outputs": depth, "targets": depth, "rows": rows, "clip_ids": ids}], readout="grid", seed=0,
                        protocol={"max_depth": 10.0, "target_size": 4})
    assert grid[0]["target"] is None and grid[0]["prediction"]["d1"] == 1.0 and grid[0]["prediction"]["rmse"] == 0.0
    box = sample_lines([{"role": "eval", "outputs": np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]),
                         "targets": np.array([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]), "rows": rows, "clip_ids": ids}], readout="box", seed=0)
    assert box[0]["iou"] == pytest.approx(1.0) and box[1]["iou"] == 0.0 and box[0]["target"] == [0.5, 0.5, 0.2, 0.2]


def test_samples_cover_the_floor_role(tmp_path, fakes):
    d = clf_yaml("avsync_like", n_classes=3, floor=("stationary",), controls=["acc_floor_arm", "floor_violated"],
                 protocol_extra={"floor_sigmas": 3.0, "offsets_seconds": [-1.0, 0.0, 1.0]})
    rows = clf_rows("train", 12, 3) + clf_rows("val", 6, 3) + clf_rows("stationary", 4, 3)
    bench = FakeBench(d, rows, classes_list(3), eval_module=avsync_probe())
    run = tmp_path / "runs" / bench.name / MODEL / READING
    for split in ("train", "val", "stationary"):
        labels = np.asarray([r["label_ids"] for r in rows if r["split"] == split])
        write_fake_features(run, split, bench=bench, feats=clf_features(labels, 3))
    res = probe(run, bench=bench, cfg=clf_cfg(3), log_samples=True)
    roles = [json.loads(x)["role"] for x in res[0].samples_file.read_text().splitlines()]
    assert roles.count("eval") == 6 and roles.count("floor") == 4
    doc = json.loads(res[0].results_file.read_text())
    assert set(doc["results"][f"avsync_like:{READING}"]) >= {"acc_floor_arm", "floor_violated"}
    assert doc["configs"][f"avsync_like:{READING}"]["roles"]["floor"] == {"splits": ["stationary"], "n": 4}


def test_result_dir_unchanged_by_the_store(tmp_path, fakes):
    bench, run = _clf_world(tmp_path)
    res = probe(run, bench=bench, cfg=clf_cfg(3))
    assert res[0].path.parent == result_dir(run, bench.protocol_hash, hashing.probe_hash(clf_cfg(3)), 0)
    assert (res[0].path.parent / "DONE").exists()


def test_sample_lines_of_a_folder_scorer_carry_per_clip_values():
    from yavih.readouts.samples import sample_lines

    part = {"role": "eval", "rows": [3, 7], "clip_ids": ["a", "b"], "values": {"clap": [0.25, float("nan")], "kl": [1.0, 2.0]}}
    lines = sample_lines([part], readout="audiocaps_t2a", seed=0)
    assert lines == [
        {"clip_id": "a", "row": 3, "seed": 0, "role": "eval", "prediction": {"clap": 0.25, "kl": 1.0}, "target": None},
        {"clip_id": "b", "row": 7, "seed": 0, "role": "eval", "prediction": {"clap": None, "kl": 2.0}, "target": None},
    ]
    with pytest.raises(ValueError, match="disagree"):
        sample_lines([{**part, "values": {"kl": [1.0]}}], readout="x", seed=0)


def test_a_folder_metric_may_bring_its_own_interval():
    from yavih.metrics import intervals
    from yavih.metrics.registry import MetricSpec

    seen = []

    def own(kind, outputs, targets, protocol, split_meta):
        seen.append((kind, outputs))
        return (0.25, 0.75)

    spec = MetricSpec(name="kad_x", fn=lambda *a: 0.5, readouts=frozenset({"some_folder"}), ci=frozenset({"bootstrap_clip"}), interval=own)
    assert intervals.interval("bootstrap_clip", "kad_x", "OUT", None, {}, {}, spec=spec) == [0.25, 0.75]
    assert seen == [("bootstrap_clip", "OUT")]
    with pytest.raises(ValueError, match="per_clip or an interval"):
        MetricSpec(name="kad_y", fn=lambda *a: 0.5, readouts=frozenset({"some_folder"}), ci=frozenset({"bootstrap_clip"}))


def test_a_failing_folder_scorer_runs_once_per_population():
    from yavih.readouts.base import ProbeRefused, evaluate_metrics
    from yavih.metrics.registry import MetricSpec

    calls = []

    def scorer(inputs):
        calls.append(inputs)
        raise RuntimeError("CUDA out of memory")

    def spec(name, kind="metric", floor_of=()):
        return MetricSpec(name=name, fn=lambda *a: 0.0, readouts=frozenset({"f"}), kind=kind, external="pkg", floor_of=floor_of)

    specs = {"metrics": {"a": spec("a"), "b": spec("b"), "c": spec("c")}, "floors": {"a_floor": spec("a_floor", "floor", ("a",))}}
    with pytest.raises(ProbeRefused, match="CUDA out of memory") as e:
        evaluate_metrics(specs, outputs="OUT", targets=None, protocol={}, split_meta={}, floor_outputs="FLOOR", scorer=scorer)
    assert calls == ["OUT", "FLOOR"] and e.value.message.count("CUDA out of memory") == 4
