"""``pos_mode: per_sample`` features reach every dense readout as a per-row gather keyed by MANIFEST row,
segment by segment: the real ``FeatureReader.pos()`` returns ``[n_present, N, 4]`` for ``per_sample``, which is
neither a constant pos nor in a multi-split view's row order.

These tests use the REAL ``FeatureWriter``/``FeatureReader`` (two train splits, missing rows) and the
driver with the fakes of ``test_readouts.py`` (whose ``FakeReader.pos()`` behaves like the real one).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from test_io_core import FakeBench as IoBench
from test_io_core import FakeBridge as IoBridge
from test_io_core import feats_for, make_describe, vmeta
from test_readouts import (
    D_FEAT,
    DESCRIBE,
    MODEL,
    READING,
    FakeBench,
    FakeReader,
    classes_list,
    dense_yaml,
    write_fake_features,
)
from test_readouts import (
    fakes as fakes,  # noqa: F401 - the fixture, re-exported for this module
)
from yavih.io.reader import FeatureReader
from yavih.io.writer import FeatureWriter, split_dir_for
from yavih.probes.config import ProbeConfig
from yavih.readouts import ProbeRefused, _fit, probe
from yavih.readouts import box as box_mod
from yavih.readouts import grid as grid_mod
from yavih.readouts import seq as seq_mod

N_TOK = 6


@pytest.fixture(autouse=True)
def _no_site_results(monkeypatch):
    monkeypatch.delenv("YAVIH_RESULTS", raising=False)


# ============================================================================ real writer / reader


def _row_pos(split_k: int, r: int) -> np.ndarray:
    """A pos that identifies (split, row): stream 0, t scaled per row, (y, x) per row; all in [0, 1]."""
    k = np.arange(N_TOK)
    t = (k + 0.5) / N_TOK * (0.5 + 0.03 * r + 0.2 * split_k)
    y = np.full(N_TOK, 0.01 * r + 0.3 * split_k)
    x = (k + 0.5) / N_TOK * 0.5 + 0.02 * r
    return np.stack([np.zeros(N_TOK), t, y, x], 1).astype(np.float32)


def _write_per_sample(out: Path, split: str, split_k: int, *, n_rows: int, missing: set[int]) -> Path:
    br = IoBridge(pos_mode="per_sample")
    with FeatureWriter(out, split, shard=(0, 1), rows=(0, n_rows), bench=IoBench(), bridge=br, describe=make_describe(br),
                       source_sha256="s" * 64) as w:
        for b in range(0, n_rows, br.batch_size):
            batch = list(range(b, min(b + br.batch_size, n_rows)))
            keep = [r for r in batch if r not in missing]
            for r in batch:
                if r in missing:
                    w.record_missing(r, f"clip{r:04d}", "MediaMissing: video")
            if keep:
                w.write(keep, [f"clip{r:04d}" for r in keep], feats_for(keep, N_TOK, 16), np.stack([_row_pos(split_k, r) for r in keep]),
                        meta=vmeta(keep))
        w.close(timing={"wall_s": 1.0}, env={"host": "test"}, canary_stats={"max_abs_diff": 0.0, "max_rel_l2": 0.0},
                side_effects={"writes_outside_run_dir": []})
    return split_dir_for(out, split)


@pytest.fixture
def two_split_view(tmp_path):
    """A train role of two splits read by the real FeatureReader; split ``a`` has missing rows, and the
    view takes a SUBSET of b's present rows in non-trivial order, so view position != reader position."""
    da = _write_per_sample(tmp_path, "a", 0, n_rows=9, missing={1, 4})
    db = _write_per_sample(tmp_path, "b", 1, n_rows=7, missing={0})
    ra, rb = FeatureReader(da, require_meta=False), FeatureReader(db, require_meta=False)
    assert ra.pos_mode == rb.pos_mode == "per_sample"
    assert ra.pos().shape == (7, N_TOK, 4)  # the real reader's per_sample pos() is the whole present array
    rows_a = np.asarray(ra.present_rows, dtype=np.int64)
    rows_b = np.array([2, 5, 6], dtype=np.int64)
    view = _fit.FeatureRows([(ra, rows_a), (rb, rows_b)])
    expected = np.stack([_row_pos(0, int(r)) for r in rows_a] + [_row_pos(1, int(r)) for r in rows_b])
    return view, expected


def test_feature_rows_per_sample_pos_is_a_per_row_gather_by_manifest_row(two_split_view):
    view, expected = two_split_view
    assert view.pos_const() is None  # never the [n_present, N, 4] array taken as "constant"
    g = view.pos_gather()
    assert not isinstance(g, np.ndarray) and g.shape == (len(view), N_TOK, 4)
    np.testing.assert_array_equal(np.asarray(g[np.arange(len(view))]), expected)
    np.testing.assert_array_equal(np.asarray(g[np.array([8, 0, 7])]), expected[[8, 0, 7]])


def test_seq_media_to_task_with_per_sample_pos_real_reader(two_split_view):
    """seq's ``media_to_task`` gathers per-sample pos per row (real reader)."""
    view, expected = two_split_view
    T, seg = 10, 1.0
    ctx = NS(protocol={"segments_scored": T, "segment_offset": 0, "segment_seconds": seg}, stream_names=lambda: ["video"])
    durs = (8.0 + np.arange(len(view), dtype=np.float64))[:, None]  # a different media duration per row
    out = seq_mod.media_to_task(ctx, view, durs)
    assert not isinstance(out, np.ndarray) and out.shape == (len(view), N_TOK, 4)
    got = np.asarray(out[np.arange(len(view))])
    np.testing.assert_allclose(got[..., 1], expected[..., 1] * durs / (T * seg), rtol=1e-6)


def test_grid_and_box_task_pos_with_per_sample_pos_real_reader(two_split_view):
    view, expected = two_split_view
    ctx = NS(protocol={})  # full frame: region-relative (y, x) == media (y, x)
    for out in (grid_mod.media_to_task(ctx, view), box_mod.media_to_task(ctx, view, view.pos_gather())):
        got = np.asarray(out[np.arange(len(view))])
        np.testing.assert_allclose(got[..., 2:], expected[..., 2:], rtol=1e-6)


# ============================================================================ box: the annotated frame's time slot


def _frames_pos(times_s: list[float], duration_s: float, g: int = 2, n_audio: int = 2) -> np.ndarray:
    """Media pos ``[len(times)·g² + n_audio, 4]``: a ``g × g`` grid of video tokens (stream 0) per frame time, then
    untimed-in-space audio tokens (stream 1)."""
    c = (np.arange(g) + 0.5) / g
    yy, xx = np.meshgrid(c, c, indexing="ij")
    vis = [np.stack([np.zeros(g * g), np.full(g * g, t / duration_s), yy.reshape(-1), xx.reshape(-1)], 1) for t in times_s]
    aud = np.stack([np.ones(n_audio), (np.arange(n_audio) + 0.5) / n_audio, np.full(n_audio, -1.0), np.full(n_audio, -1.0)], 1)
    return np.concatenate([*vis, aud], 0).astype(np.float32)


def test_annotated_slot_keeps_the_nearest_time_slot_of_the_visual_stream_only():
    from yavih.readouts.coords import annotated_slot, annotated_task_pos

    # row 0: 10 s clip, frames at 1, 5, 9 s; row 1: 6 s clip with the SAME constant pos, i.e. frames at 0.6, 3, 5.4 s
    pos = np.broadcast_to(_frames_pos([1.0, 5.0, 9.0], 10.0), (2, 14, 4))
    durs = np.array([[10.0, 10.0], [6.0, 6.0]])
    keep = annotated_slot(pos, durs, stream_tag=0, second=5.0)
    assert keep[0].tolist() == [False] * 4 + [True] * 4 + [False] * 6  # the 5.0 s frame; never an audio token
    assert keep[1].tolist() == [False] * 8 + [True] * 4 + [False] * 2  # the 5.4 s frame of the shorter clip
    task = annotated_task_pos(pos, durs, stream_tag=0, second=5.0)
    assert np.isfinite(task[0, 4:8, 2:]).all() and np.isnan(task[0, :4, 2:]).all() and np.isnan(task[0, 8:12, 2:]).all()
    np.testing.assert_array_equal(task[..., :2], pos[..., :2])  # stream tag and t untouched
    assert np.isnan(task[:, 12:, 2:]).all()  # audio: no (y, x) before or after

    # two slots equally far (4 s and 6 s) are both kept; a visual token without t is not on the frame
    tie = _frames_pos([4.0, 6.0, 9.0], 10.0)[None].copy()
    tie[0, 8, 1] = -1.0  # one token of the 9 s frame loses its time
    k = annotated_slot(tie, np.array([[10.0, 10.0]]), stream_tag=0, second=5.0)
    assert k[0].tolist() == [True] * 8 + [False] * 6

    # nothing to select by (no duration, or no timed visual token): every visual token is kept
    for d in (None, np.array([[np.nan, 10.0]])):
        assert annotated_slot(pos[:1], d, stream_tag=0, second=5.0)[0].tolist() == [True] * 12 + [False] * 2
    untimed = pos[:1].copy()
    untimed[0, :12, 1] = -1.0
    assert annotated_slot(untimed, durs[:1], stream_tag=0, second=5.0)[0].tolist() == [True] * 12 + [False] * 2


def test_box_cells_hold_the_annotated_frame_not_the_average_of_all_frames():
    """A bridge that feeds three frames is read on the annotated one: the cells equal that frame's tokens."""
    import torch

    from yavih.probes.dense import pool_cells

    g, n = 2, 3
    media = _frames_pos([1.0, 5.0, 9.0], 10.0, g=g)  # constant pos, 14 tokens
    class View:
        N = media.shape[0]

        def __len__(self):
            return n

    ctx = NS(protocol={}, stream_names=lambda: ["video", "audio"])
    durs = np.full((n, 2), 10.0)
    out = box_mod.media_to_task(ctx, View(), media, annotated=(0, 5.0), durations=durs)
    task = torch.from_numpy(np.asarray(out[np.arange(n)]))
    mem = torch.zeros(n, media.shape[0], 4)
    mem[:, 0:4], mem[:, 4:8], mem[:, 8:12] = 1.0, 2.0, 30.0  # frame at 1 s, at 5 s (annotated), at 9 s
    cells, counts = pool_cells(mem, task, grid=g, stream_tag=0)
    assert bool((counts == 1).all()) and bool((cells == 2.0).all())
    # without the requirement (a folder that declares no annotated frame) every frame is averaged, as before
    plain = torch.from_numpy(np.broadcast_to(box_mod.media_to_task(ctx, View(), media), (n, *media.shape)).copy())
    cells_all, counts_all = pool_cells(mem, plain, grid=g, stream_tag=0)
    assert bool((counts_all == 3).all()) and bool((cells_all == 11.0).all())


def test_box_refuses_timed_visual_tokens_without_a_duration():
    media = _frames_pos([1.0, 5.0, 9.0], 10.0)

    class View:
        N = media.shape[0]

        def __len__(self):
            return 2

    ctx = NS(protocol={}, stream_names=lambda: ["video", "audio"])
    with pytest.raises(ProbeRefused) as e:
        box_mod.media_to_task(ctx, View(), media, annotated=(0, 5.0), durations=np.array([[10.0, 10.0], [np.nan, 10.0]]))
    assert e.value.code == "E_DURATIONS" and "'video'" in str(e.value)


def test_box_spec_refuses_an_annotated_frame_on_another_stream():
    params = {"grid": 4, "visual_stream": "video", "conditioning_stream": "audio", "shuffled_control": True, "shuffle_seed": 0}
    req = NS(grid={"h": 4, "w": 4}, annotated_frame={"fact": "annotated_second", "stream": "video", "tol_s": 0.5})
    box_mod.SCORER.validate_spec(dict(params), req, {})
    req.annotated_frame = {"fact": "annotated_second", "stream": "audio", "tol_s": 0.5}
    with pytest.raises(ValueError, match="visual_stream"):
        box_mod.SCORER.validate_spec(dict(params), req, {})


def test_feature_rows_constant_pos_still_constant_and_mixed_splits_go_per_row():
    const = np.random.default_rng(0).random((N_TOK, 4)).astype(np.float32)

    class R:
        def __init__(self, mode, p):
            self.pos_mode, self._p, self.N, self.D = mode, p, N_TOK, 3

        def pos(self, row=None):
            if self.pos_mode == "per_sample":
                return self._p if row is None else self._p[row]
            return self._p

    rows = np.arange(2, dtype=np.int64)
    same = _fit.FeatureRows([(R("constant", const), rows), (R("constant", const.copy()), rows)])
    np.testing.assert_array_equal(same.pos_gather(), const)
    other = const + 1
    differ = _fit.FeatureRows([(R("constant", const), rows), (R("constant", other), rows)])
    g = differ.pos_gather()
    assert not isinstance(g, np.ndarray)
    np.testing.assert_array_equal(g[np.array([0, 3])], np.stack([const, other]))
    per = np.random.default_rng(1).random((2, N_TOK, 4)).astype(np.float32)
    mixed = _fit.FeatureRows([(R("constant", const), rows), (R("per_sample", per), rows)])
    np.testing.assert_array_equal(mixed.pos_gather()[np.array([1, 2, 3])], np.stack([const, per[0], per[1]]))
    assert _fit.FeatureRows([(R("none", None), rows)]).pos_gather() is None


# ============================================================================ driver (fakes)


def test_fake_reader_pos_matches_the_real_reader_contract(tmp_path):
    bench, run, _ = _seq_world(tmp_path)
    r = FakeReader(run / "features" / "train")
    assert r.pos().shape == (len(r.present_rows), 4, 4)


def _seq_world(tmp_path: Path):
    """AVE-like: per_sample pos, a train role of TWO splits with missing rows, per-clip durations."""
    T, Cn, n_tok = 10, 5, 4
    protocol = {"num_segments": 10, "segment_seconds": 1.0, "segments_scored": T, "segment_offset": 0, "background_index": 4}
    d = dense_yaml("ave_like", "seq", protocol=protocol, requires={"pos": ["t"], "streams": ["video"], "grid": None, "t_bins": 8},
                   readout_params={"T": T}, metrics=["seg_acc", "clip_acc", "seg_acc_boundary", "n_boundary", "n_eval"],
                   floors=["seg_acc_floor", "seg_acc_boundary_floor"], controls=[], ci={}, chance={"seg_acc": 0.2},
                   label={"column": "label", "kind": "segments", "sep": ";", "class_key": "name", "sets": None, "train_set": None,
                          "allow_unknown": False, "empty_ok": False},
                   classes="classes.csv", streams=("video",))
    d["splits"]["train"] = ["train", "train2"]
    sizes = {"train": 8, "train2": 6, "val": 4}
    rows = [{"clip_id": f"{s}_{k}", "split": s, "label_ids": k % 4, "extra": {"event_start": k % 3, "event_end": 6}, "segment": [k % 3, 6]}
            for s, n in sizes.items() for k in range(n)]
    bench = FakeBench(d, rows, classes_list(Cn))
    run = tmp_path / "runs" / "ave_like" / MODEL / READING
    truth = {}
    for si, (split, n) in enumerate(sizes.items()):
        t = (np.arange(n_tok) + 0.5) / n_tok
        pos = np.stack([np.stack([np.zeros(n_tok), t * (0.6 + 0.04 * k + 0.1 * si), -np.ones(n_tok), -np.ones(n_tok)], 1) for k in range(n)])
        dur = 9.0 + np.arange(n) + si
        present = np.array([k for k in range(n) if not (split == "train" and k in (2, 5))], dtype=np.int64)
        write_fake_features(run, split, bench=bench, feats=np.random.default_rng(si).standard_normal((n, n_tok, D_FEAT)).astype(np.float32),
                            present=present, pos=pos.astype(np.float32), durations={"video": dur}, describe={**DESCRIBE, "streams": ["video"]})
        truth[split] = (present, pos.astype(np.float32), dur)
    return bench, run, (ProbeConfig.pinned("seq", num_classes=Cn, T=T).fast(), truth, T)


def test_seq_driver_runs_per_sample_pos_over_two_train_splits(tmp_path, fakes):
    bench, run, (cfg, truth, T) = _seq_world(tmp_path)
    res = probe(run, bench=bench, cfg=cfg)
    assert res[0].status == "ok"
    pos_obj = fakes[0]["pos_obj"]
    want = []
    for split in ("train", "train2"):
        present, pos, dur = truth[split]
        want.append(pos[present][..., 1] * dur[present][:, None] / T)
    want = np.concatenate(want)
    assert pos_obj.shape[0] == want.shape[0] == 12
    np.testing.assert_allclose(np.asarray(pos_obj[np.arange(want.shape[0])])[..., 1], want, rtol=1e-6)


# ============================================================================ real chain (integration)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("readout", ["seq", "grid", "box"])
def test_dense_readouts_run_on_per_sample_oracle_features(tmp_path, readout):
    """The documented user path: per_sample pos, real extract → validate → probe."""
    from yavih.benchmark import Benchmark
    from yavih.extract import run_extract
    from yavih.io.validate import validate_features
    from yavih.testing import make_synthetic_benchmark, oracle_bridge

    folder = make_synthetic_benchmark(tmp_path, readout, n_train=16, n_eval=8, media="tiny")
    bench = Benchmark.from_dir(folder)
    bridge = oracle_bridge(bench, N=4, D=16, pos_mode="per_sample")
    run = tmp_path / "runs" / bench.name / bridge.model_name / bridge.reading  # runs/<bench>/<model>/<reading>
    s = bench.spec.splits
    for split in [*s.train, *s.eval, *s.floor, *(s.folds or ())]:
        out = run_extract(bench=bench, bridge=bridge, split=split, shard=(0, 1), out=run, canaries=2)
        assert out["status"] == "ok", out
        rep = validate_features(run / "features" / split, bench=bench)
        assert rep.ok, [(e.code, e.message) for e in rep.errors]
        assert FeatureReader(run / "features" / split).pos_mode == "per_sample"
    res = probe(run, bench=bench, cfg=ProbeConfig.pinned(readout, bench).fast(), seeds=(0,), device="cpu")
    assert res[0].status == "ok", res[0]
    assert res[0].result["metrics"]["n_eval"] == res[0].result["coverage"]["eval"]["present"]
