"""Head families — shapes, pos usage, cell binning, null embeddings, losses."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from yavih.probes.attentive import build_attentive_probe
from yavih.probes.config import ProbeConfig
from yavih.probes.dense import (
    DenseAVHead,
    axis_centres,
    build_dense_av_head,
    build_dense_query_head,
    pool_cells,
    sincos_embed,
)
from yavih.probes.losses import per_sample_loss

D = 32  # divisible by 16 heads and by 4 (2-D sincos)


def _pos_grid(h: int, w: int, stream: int, t: float = 0.5) -> np.ndarray:
    yy, xx = np.meshgrid(axis_centres(h).numpy(), axis_centres(w).numpy(), indexing="ij")
    return np.stack([np.full(h * w, stream), np.full(h * w, t), yy.reshape(-1), xx.reshape(-1)], -1).astype(np.float32)


def _pos_time(n: int, stream: int) -> np.ndarray:
    return np.stack([np.full(n, stream), axis_centres(n).numpy(), np.full(n, -1.0), np.full(n, -1.0)], -1).astype(np.float32)


# ----------------------------------------------------------------------------- attentive


def test_attentive_shape_and_permutation_invariance() -> None:
    torch.manual_seed(0)
    head = build_attentive_probe(D, 5, ProbeConfig.pinned("clf", num_classes=5).head).eval()
    x = torch.randn(3, 7, D)
    y = head(x)
    assert y.shape == (3, 5)
    perm = torch.randperm(7)
    torch.testing.assert_close(head(x[:, perm]), y, atol=1e-5, rtol=1e-5)
    assert head(x, None).shape == (3, 5)  # pos accepted and ignored


def test_attentive_head_geometry() -> None:
    head = build_attentive_probe(D, 5, ProbeConfig.pinned("clf", num_classes=5).head)
    assert len(head.self_attn) == 3 and len(head.cross_attn) == 1
    assert head.self_attn[0].attn.num_heads == 16 and head.self_attn[0].mlp.fc1.out_features == 4 * D
    assert head.query.shape == (1, 1, D)
    with pytest.raises(ValueError):
        build_attentive_probe(30, 5, ProbeConfig.pinned("clf", num_classes=5).head)  # 30 % 16 != 0
    with pytest.raises(ValueError):
        build_attentive_probe(D, 5, {"family": "dense_query_v1"})


# ----------------------------------------------------------------------------- sincos


def test_sincos_embed_shapes_and_scale() -> None:
    e1 = sincos_embed(torch.tensor([[0.0], [0.5], [1.0]]), 16)
    assert e1.shape == (3, 16)
    e2 = sincos_embed(torch.tensor([[0.0, 0.0], [0.5, 1.0]]), 16)
    assert e2.shape == (2, 16)
    # axis scale 64: channel 0 is sin(64 * coord)
    assert math.isclose(float(e1[1, 0]), math.sin(32.0), abs_tol=1e-5)
    with pytest.raises(ValueError):
        sincos_embed(torch.zeros(2, 3), 16)  # 16 % 6 != 0


# ----------------------------------------------------------------------------- dense_query_v1


def test_dense_time_head_shape_and_uses_pos() -> None:
    torch.manual_seed(0)
    cfg = ProbeConfig.pinned("seq", num_classes=4, T=8)
    head = build_dense_query_head(D, 4, cfg.head).eval()
    x = torch.randn(2, 10, D)
    pos = torch.from_numpy(_pos_time(10, 0))
    y = head(x, pos)
    assert y.shape == (2, 8, 4)
    assert head.n_queries == 8 and len(head.self_attn) == 3 and len(head.cross_attn) == 1
    # shuffling the pos column (not the tokens) changes the output: the head reads pos
    shuffled = pos.clone()
    shuffled[:, 1] = pos[torch.randperm(10), 1]
    assert not torch.allclose(head(x, shuffled), y, atol=1e-4)
    # tokens and pos permuted together → same output (geometry lives in pos, not in token order)
    perm = torch.randperm(10)
    torch.testing.assert_close(head(x[:, perm], pos[perm]), y, atol=1e-4, rtol=1e-4)
    # per-sample pos [B, N, 4] accepted
    assert head(x, pos.unsqueeze(0).expand(2, -1, -1)).shape == (2, 8, 4)


def test_dense_null_embedding_for_minus_one_axis() -> None:
    torch.manual_seed(0)
    head = build_dense_query_head(D, 3, ProbeConfig.pinned("seq", num_classes=3, T=4).head).eval()
    x = torch.randn(1, 6, D)
    pos = torch.from_numpy(_pos_time(6, 0))
    pos_null = pos.clone()
    pos_null[:, 1] = -1.0  # no t for any token → null embedding everywhere
    y0, y1 = head(x, pos), head(x, pos_null)
    assert not torch.allclose(y0, y1, atol=1e-4)
    # with the null parameter zeroed and no t, the pos path contributes nothing
    with torch.no_grad():
        head.mem_pos.null.zero_()
    y_zero = head(x, pos_null)
    with torch.no_grad():
        ref = head.in_proj(x)
        for blk in head.self_attn:
            ref = blk(ref)
    assert torch.isfinite(y_zero).all() and ref.shape == (1, 6, D)


def test_dense_grid_head_upsample_exp_and_stream_filter() -> None:
    torch.manual_seed(0)
    cfg = ProbeConfig.pinned("grid", grid=(4, 4), target_size=12)
    head = build_dense_query_head(D, 1, cfg.head, pos_stream_tag=0).eval()
    x = torch.randn(2, 16, D)
    pos = torch.from_numpy(_pos_grid(4, 4, stream=0))
    y = head(x, pos)
    assert y.shape == (2, 12, 12) and bool((y > 0).all())  # exp → depth
    # without upsample: [B, H, W] (out == 1 is squeezed: a depth map); out > 1 keeps the channel axis
    raw = dict(cfg.head, upsample=None, activation=None)
    y_raw = build_dense_query_head(D, 1, raw).eval()(x, pos)
    assert y_raw.shape == (2, 4, 4)
    assert build_dense_query_head(D, 3, raw).eval()(x, pos).shape == (2, 4, 4, 3)
    # tokens of another stream carry no coordinates for this head
    pos_other = pos.clone()
    pos_other[:, 0] = 1.0
    pos_null = pos.clone()
    pos_null[:, 2:] = -1.0
    torch.testing.assert_close(head(x, pos_other), head(x, pos_null), atol=1e-5, rtol=1e-5)
    # the query init follows the task coordinate: cell queries differ from each other
    assert head.query_pos.shape == (1, 16, D) and float(head.query_pos.std()) > 0


def test_dense_query_refuses_unresolved_layout() -> None:
    with pytest.raises(ValueError):
        build_dense_query_head(D, 3, ProbeConfig.pinned("seq", num_classes=3).head)
    with pytest.raises(ValueError):
        build_dense_query_head(D, 3, {"family": "dense_query_v1", "layout": "point"})
    with pytest.raises(ValueError):
        build_dense_query_head(D, 0, {"family": "dense_query_v1", "layout": "time:4"})


# ----------------------------------------------------------------------------- dense_av_v1


def _av_pos(g: int, n_audio: int) -> np.ndarray:
    vis = _pos_grid(g, g, stream=0)
    aud = _pos_time(n_audio, stream=1)
    return np.concatenate([vis, aud], 0)


def test_pool_cells_binning_and_null_cells() -> None:
    g = 4
    pos = torch.from_numpy(_av_pos(g, 3)).unsqueeze(0)
    mem = torch.arange(1, 20, dtype=torch.float32).reshape(1, 19, 1).expand(1, 19, D).clone()
    cells, counts = pool_cells(mem, pos, grid=g, stream_tag=0)
    assert cells.shape == (1, g * g, D) and counts.shape == (1, g * g)
    assert bool((counts == 1).all())  # one visual token per cell, audio tokens excluded
    torch.testing.assert_close(cells[0, :, 0], torch.arange(1, 17, dtype=torch.float32))
    # two tokens in cell 0, none in cell 1
    pos2 = pos.clone()
    pos2[0, 1, 2:] = pos[0, 0, 2:]
    cells2, counts2 = pool_cells(mem, pos2, grid=g, stream_tag=0)
    assert counts2[0, 0] == 2 and counts2[0, 1] == 0
    assert float(cells2[0, 0, 0]) == 1.5 and float(cells2[0, 1, 0]) == 0.0
    # y == 1.0 lands in the last row, not out of range
    pos3 = pos.clone()
    pos3[0, 0, 2] = 1.0
    _, counts3 = pool_cells(mem, pos3, grid=g, stream_tag=0)
    assert counts3[0, (g - 1) * g + 0] == 2


def test_dense_av_shape_sigmoid_and_pos_dependence() -> None:
    torch.manual_seed(0)
    g = 4
    cfg = ProbeConfig.pinned("box", grid=g)
    head = build_dense_av_head(D, cfg.head, visual_tag=0, conditioning_tag=1).eval()
    assert isinstance(head, DenseAVHead) and len(head.self_attn) == 2 and len(head.cross_attn) == 2
    n_a = 5
    x = torch.randn(2, g * g + n_a, D)
    pos = torch.from_numpy(_av_pos(g, n_a))
    y = head(x, pos)
    assert y.shape == (2, 4) and bool(((y > 0) & (y < 1)).all())
    # permuting the VISUAL tokens' cells changes the box (the head is equivariant, not invariant)
    shuffled = pos.clone()
    vis_perm = torch.randperm(g * g)
    shuffled[: g * g, 2:] = pos[vis_perm, 2:]
    assert not torch.allclose(head(x, shuffled), y, atol=1e-4)
    # permuting tokens together with their pos leaves the output unchanged
    perm = torch.randperm(g * g + n_a)
    torch.testing.assert_close(head(x[:, perm], pos[perm]), y, atol=1e-4, rtol=1e-4)
    # the conditioning stream matters
    x2 = x.clone()
    x2[:, g * g :] = torch.randn(2, n_a, D)
    assert not torch.allclose(head(x2, pos), y, atol=1e-4)


def test_dense_av_handles_missing_cells_and_missing_conditioning() -> None:
    torch.manual_seed(0)
    g = 4
    head = build_dense_av_head(D, ProbeConfig.pinned("box", grid=g).head).eval()
    # only 3 visual tokens, zero conditioning tokens: null cells + null conditioning token → finite
    pos = torch.from_numpy(_av_pos(g, 0)[:3])
    x = torch.randn(1, 3, D)
    y = head(x, pos)
    assert y.shape == (1, 4) and torch.isfinite(y).all()
    with pytest.raises(ValueError):
        build_dense_av_head(D, {"family": "dense_av_v1", "layout": "grid:4x4"})


# ----------------------------------------------------------------------------- losses


def test_losses_per_sample_shapes_and_masking() -> None:
    torch.manual_seed(0)
    ce = per_sample_loss("ce")
    logits = torch.randn(4, 3)
    tgt = torch.tensor([0, 1, 2, -1])
    v = ce(logits, tgt)
    assert v.shape == (4,) and float(v[3]) == 0.0 and bool((v[:3] > 0).all())
    tgt_f = torch.tensor([0.0, 1.0, 2.0, float("nan")])
    torch.testing.assert_close(ce(logits, tgt_f), v)

    bce = per_sample_loss("bce")
    t = torch.tensor([[1.0, 0.0, float("nan")], [0.0, 1.0, 1.0]])
    vb = bce(logits[:2], t)
    assert vb.shape == (2,) and torch.isfinite(vb).all()
    manual = torch.nn.functional.binary_cross_entropy_with_logits(logits[0, :2], t[0, :2])
    torch.testing.assert_close(vb[0], manual)

    ce_seq = per_sample_loss("ce_seq")
    seq_logits = torch.randn(2, 3, 4)
    seq_t = torch.tensor([[0, 1, -1], [3, 3, 3]])
    vs = ce_seq(seq_logits, seq_t)
    assert vs.shape == (2,)
    manual = torch.nn.functional.cross_entropy(seq_logits[0, :2], seq_t[0, :2])
    torch.testing.assert_close(vs[0], manual)
    seq_nan = seq_t.to(torch.float32)
    seq_nan[0, 2] = float("nan")
    torch.testing.assert_close(ce_seq(seq_logits, seq_nan), vs)
    with pytest.raises(ValueError):
        ce_seq(seq_logits, torch.zeros(2, 2, dtype=torch.long))

    l1_log = per_sample_loss("l1_log")
    pred = torch.full((2, 3, 3), 2.0)
    target = torch.full((2, 3, 3), 4.0)
    target[0, 0, 0] = 0.0  # invalid (<= 0)
    target[1, 1, 1] = float("nan")
    vl = l1_log(pred, target)
    torch.testing.assert_close(vl, torch.full((2,), math.log(2.0)))
    assert l1_log(pred.unsqueeze(-1), target).shape == (2,)  # trailing singleton aligned

    box = per_sample_loss("box_l1_giou")
    b = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]])
    tb = torch.tensor([[0.5, 0.5, 0.2, 0.2], [float("nan"), 0.5, 0.2, 0.2]])
    vbx = box(b, tb)
    assert abs(float(vbx[0])) < 1e-5 and float(vbx[1]) == 0.0
    far = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    assert float(box(far, b[:1])) > 1.0  # L1 + (1 − GIoU) with GIoU < 0

    with pytest.raises(ValueError):
        per_sample_loss("mse")


def test_losses_require_grad_flow_through_mask() -> None:
    l1_log = per_sample_loss("l1_log")
    pred = torch.full((1, 2, 2), 2.0, requires_grad=True)
    target = torch.tensor([[[4.0, float("nan")], [0.0, 4.0]]])
    l1_log(pred, target).sum().backward()
    g = pred.grad[0]
    assert g[0, 1] == 0 and g[1, 0] == 0 and g[0, 0] != 0 and g[1, 1] != 0
