"""Dense query heads ``dense_query_v1`` (seq/grid) and ``dense_av_v1`` (box).

Memory tokens: ``Linear(D, D)`` + fixed sincos of ``pos`` (axis scale 64), learned null embedding per ``-1`` axis; queries initialized from sincos
of the task coordinate; 3+1 geometry; shared ``Linear``. Dense-av: cell pooling over
``floor(G·y), floor(G·x)`` of the region-relative coordinates, null-cell embedding, 2 self + 2 cross blocks, one box query,
``MLP(D→D→4) + sigmoid``.

``pos`` enters every forward as ``[N, 4]`` (constant) or ``[B, N, 4]`` (per sample) with columns
``(stream, t, y, x)``. The readout driver hands the heads TASK-FRAME coordinates: seq
``u = (t·duration_s − segment_offset·segment_seconds) / (T·segment_seconds)``, grid/box ``(y, x)``
re-expressed in the target region. A token outside the scored span or region keeps its out-of-range
coordinate (no clamping); "no coordinate on that axis" is ``NaN`` (what the readouts emit) or ``-1``
(the media-coordinate marker, also accepted from direct callers).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from yavih.probes._blocks import (
    CrossAttnBlock,
    QueryDecoderBlock,
    SelfAttnBlock,
    check_heads,
    trunc_normal_query,
)
from yavih.probes.config import parse_layout

__all__ = [
    "build_dense_query_head",
    "build_dense_av_head",
    "sincos_embed",
    "SINCOS_AXIS_SCALE",
    "QUERY_POS_INIT_STD",
    "DenseQueryHead",
    "DenseAVHead",
    "axis_centres",
    "pool_cells",
    "coord_present",
]

SINCOS_AXIS_SCALE = 64.0
#: Query positional init is rescaled to the std of the shared ``trunc_normal(0.02)`` base query so
#: neither term of ``base + pos`` dominates at initialisation.
QUERY_POS_INIT_STD = 0.02


# ----------------------------------------------------------------------------- sincos


def _sincos_1d(coords: Tensor, dim: int) -> Tensor:
    """``[..] in [0, 1]`` → ``[.., dim]``; "Attention Is All You Need" ladder over the coordinate × 64."""
    if dim % 2 != 0:
        raise ValueError(f"sincos dim must be even, got {dim}")
    position = coords.to(torch.float32).unsqueeze(-1) * SINCOS_AXIS_SCALE
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=coords.device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(*coords.shape, dim, dtype=torch.float32, device=coords.device)
    pe[..., 0::2] = torch.sin(position * div_term)
    pe[..., 1::2] = torch.cos(position * div_term)
    return pe


def sincos_embed(coords: Any, dim: int) -> Tensor:
    """Fixed factorised sincos of normalized coordinates ``[.., k]`` → ``[.., dim]``: each of the ``k`` axes
    gets ``dim // k`` channels (fixed 2-D/1-D sincos embedding, axis scale 64)."""
    c = torch.as_tensor(coords, dtype=torch.float32)
    k = c.shape[-1]
    if k < 1 or dim % (2 * k) != 0:
        raise ValueError(f"sincos_embed: dim={dim} must be a multiple of 2*k for k={k} axes")
    per = dim // k
    return torch.cat([_sincos_1d(c[..., i], per) for i in range(k)], dim=-1)


def axis_centres(n: int) -> Tensor:
    """``(k + 0.5) / n`` — segment / cell centres (query init)."""
    return (torch.arange(n, dtype=torch.float32) + 0.5) / float(n)


def _query_init(kind: str, shape: tuple[int, ...], dim: int) -> Tensor:
    """``[n_queries, dim]`` sincos of the task coordinate, rescaled to ``QUERY_POS_INIT_STD``."""
    if kind == "time":
        pos = sincos_embed(axis_centres(shape[0]).unsqueeze(-1), dim)
    elif kind == "grid":
        h, w = shape
        yy = axis_centres(h).reshape(h, 1).expand(h, w)
        xx = axis_centres(w).reshape(1, w).expand(h, w)
        pos = sincos_embed(torch.stack([yy, xx], dim=-1).reshape(h * w, 2), dim)
    else:
        raise ValueError(f"dense_query_v1 supports time/grid layouts, got {kind!r}")
    std = float(pos.std())
    return pos * (QUERY_POS_INIT_STD / std) if std > 0 else pos


# ----------------------------------------------------------------------------- pos helpers


def coord_present(coord: Tensor) -> Tensor:
    """``True`` where a pos coordinate exists: finite and not the media marker ``-1``."""
    return torch.isfinite(coord) & (coord != -1.0)


def _pos_batch(pos: Tensor, batch: int) -> Tensor:
    """``[N, 4] | [B, N, 4]`` → ``[B, N, 4]`` float32."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    if pos.dim() == 2:
        pos = pos.unsqueeze(0).expand(batch, -1, -1)
    if pos.dim() != 3 or pos.shape[0] != batch or pos.shape[-1] != 4:
        raise ValueError(f"pos must be [N, 4] or [B, N, 4] with B={batch}, got {tuple(pos.shape)}")
    return pos


# ----------------------------------------------------------------------------- dense_query_v1


class _MemoryPos(nn.Module):
    """Fixed sincos of the token's ``pos`` axes, with a learned null vector per ``-1`` axis."""

    def __init__(self, axes: tuple[int, ...], dim: int) -> None:
        super().__init__()
        self.axes = axes  # pos columns used: (1,) for t, (2, 3) for (y, x)
        k = len(axes)
        if dim % (2 * k) != 0:
            raise ValueError(f"dense_query_v1 needs D divisible by {2 * k} for a {k}-axis sincos, got D={dim}")
        self.per = dim // k
        self.null = nn.Parameter(torch.zeros(k, self.per))
        nn.init.trunc_normal_(self.null, std=0.02)

    def forward(self, pos: Tensor, use: Tensor) -> Tensor:
        """``pos [B, N, 4]``, ``use [B, N]`` bool (token's stream carries coordinates) → ``[B, N, D]``."""
        parts = []
        for i, col in enumerate(self.axes):
            coord = pos[..., col]
            valid = use & coord_present(coord)
            # Out-of-range task coordinates are kept (a token before the scored span or
            # outside the target region is read where it lies); only a missing coordinate is null.
            emb = _sincos_1d(torch.where(valid, coord, torch.zeros_like(coord)), self.per)
            parts.append(torch.where(valid.unsqueeze(-1), emb, self.null[i].to(emb.dtype)))
        return torch.cat(parts, dim=-1)


class DenseQueryHead(nn.Module):
    """``[B, N, D] + pos`` → ``[B, T, out]`` (``time:T``) or ``[B, H, W, out]`` / ``[B, S, S]`` (``grid:HxW``,
    with ``upsample: S`` + ``exp`` for the depth readout, squeezed when ``out == 1``)."""

    def __init__(self, dim: int, out: int, *, kind: str, shape: tuple[int, ...], num_heads: int = 16,
                 self_attn_depth: int = 3, cross_attn_depth: int = 1, mlp_ratio: int = 4,
                 upsample: int | None = None, activation: str | None = None, pos_stream_tag: int | None = None) -> None:
        super().__init__()
        if out < 1:
            raise ValueError("dense_query_v1 needs out >= 1 (head.out unresolved? pass bench= to pinned())")
        check_heads(dim, num_heads)
        self.kind, self.shape, self.out = kind, tuple(shape), int(out)
        self.upsample = int(upsample) if upsample else None
        self.activation = activation
        self.pos_stream_tag = pos_stream_tag
        axes = (1,) if kind == "time" else (2, 3)
        mlp_dim = dim * int(mlp_ratio)
        self.in_proj = nn.Linear(dim, dim)
        self.mem_pos = _MemoryPos(axes, dim)
        self.self_attn = nn.ModuleList(SelfAttnBlock(dim, num_heads, mlp_dim) for _ in range(int(self_attn_depth)))
        self.self_norm = nn.LayerNorm(dim)
        self.query_base = trunc_normal_query(dim)
        self.query_pos = nn.Parameter(_query_init(kind, self.shape, dim).unsqueeze(0))
        self.cross_attn = nn.ModuleList(QueryDecoderBlock(dim, num_heads, mlp_dim) for _ in range(int(cross_attn_depth)))
        self.cross_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.out)

    @property
    def n_queries(self) -> int:
        return int(math.prod(self.shape))

    def forward(self, memory: Tensor, pos: Tensor) -> Tensor:
        b = memory.shape[0]
        pos = _pos_batch(pos, b).to(memory.device)
        use = torch.ones(pos.shape[:2], dtype=torch.bool, device=pos.device)
        if self.pos_stream_tag is not None:
            # Tokens of another stream carry no coordinate for this readout: they get the null embedding
            # (and stay in the memory as content): only pos.(y, x) of the stream named visual_stream counts.
            use = pos[..., 0] == float(self.pos_stream_tag)
        x = self.in_proj(memory) + self.mem_pos(pos, use)
        for blk in self.self_attn:
            x = blk(x)
        x = self.self_norm(x)
        q = (self.query_base + self.query_pos).expand(b, -1, -1)
        for blk in self.cross_attn:
            q = blk(q, x)
        y = self.head(self.cross_norm(q))  # [B, n_queries, out]
        y = y.reshape(b, *self.shape, self.out)
        if self.kind == "grid" and self.upsample is not None:
            s = self.upsample
            y = F.interpolate(y.permute(0, 3, 1, 2), size=(s, s), mode="bilinear", align_corners=False)
            y = y.permute(0, 2, 3, 1)  # [B, S, S, out]
        if self.activation == "exp":
            y = torch.exp(y)
        elif self.activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.activation not in (None, "none"):
            raise ValueError(f"unknown head.activation {self.activation!r}")
        if self.kind == "grid" and self.out == 1:
            y = y.squeeze(-1)  # depth map [B, S, S] — the grid target shape
        return y


def build_dense_query_head(dim: int, out: int, head: dict, *, pos_stream_tag: int | None = None) -> nn.Module:
    """``[B, N, D] + pos`` → ``[B, T, out]`` (``layout time:T``) or ``[B, H, W, out]`` (``layout grid:HxW``);
    ``pos_stream_tag`` (index into ``describe()["streams"]``) restricts coordinates to that stream."""
    if head.get("family") != "dense_query_v1":
        raise ValueError(f"build_dense_query_head: head.family must be dense_query_v1, got {head.get('family')!r}")
    kind, shape = parse_layout(str(head["layout"]))
    if kind not in {"time", "grid"}:
        raise ValueError(f"dense_query_v1 layout must be time:T or grid:HxW, got {head['layout']!r}")
    return DenseQueryHead(
        int(dim),
        int(out),
        kind=kind,
        shape=shape,
        num_heads=int(head.get("num_heads", 16)),
        self_attn_depth=int(head.get("self_attn_depth", 3)),
        cross_attn_depth=int(head.get("cross_attn_depth", 1)),
        mlp_ratio=int(head.get("mlp_ratio", 4)),
        upsample=head.get("upsample"),
        activation=head.get("activation"),
        pos_stream_tag=pos_stream_tag,
    )


# ----------------------------------------------------------------------------- dense_av_v1


def pool_cells(memory: Tensor, pos: Tensor, *, grid: int, stream_tag: int) -> tuple[Tensor, Tensor]:
    """Mean-pool the visual tokens into ``G²`` cells: ``(cells [B, G², D], counts [B, G²])``.

    ``(y, x)`` are coordinates in the target region (the readout re-expresses the
    full-frame fractions in the region; the full frame for vggss_box, where they are unchanged).
    Visual tokens = ``stream == stream_tag`` with both coordinates present and inside ``[0, 1]``; a token
    outside the region falls in no cell. Cell = ``floor(G·y), floor(G·x)`` (``y == 1.0`` lands in the last
    cell). Implemented with ``index_add_`` so a batch is one scatter."""
    b, n, d = memory.shape
    g2 = grid * grid
    y, x = pos[..., 2], pos[..., 3]
    inside = coord_present(y) & coord_present(x) & (y >= 0) & (y <= 1) & (x >= 0) & (x <= 1)
    is_vis = (pos[..., 0] == float(stream_tag)) & inside
    y = torch.where(inside, y, torch.zeros_like(y))
    x = torch.where(inside, x, torch.zeros_like(x))
    cy = torch.floor(y * grid).clamp(0, grid - 1).to(torch.long)
    cx = torch.floor(x * grid).clamp(0, grid - 1).to(torch.long)
    cell = cy * grid + cx  # [B, N]
    # Non-visual tokens are routed to a scratch cell G² that is dropped afterwards.
    cell = torch.where(is_vis, cell, torch.full_like(cell, g2))
    flat_idx = (cell + torch.arange(b, device=cell.device).unsqueeze(1) * (g2 + 1)).reshape(-1)
    sums = torch.zeros(b * (g2 + 1), d, dtype=memory.dtype, device=memory.device)
    sums.index_add_(0, flat_idx, memory.reshape(b * n, d))
    counts = torch.zeros(b * (g2 + 1), dtype=memory.dtype, device=memory.device)
    counts.index_add_(0, flat_idx, torch.ones(b * n, dtype=memory.dtype, device=memory.device))
    sums = sums.reshape(b, g2 + 1, d)[:, :g2]
    counts = counts.reshape(b, g2 + 1)[:, :g2]
    cells = sums / counts.clamp(min=1.0).unsqueeze(-1)
    return cells, counts


class DenseAVHead(nn.Module):
    """``[B, N, D] + pos`` → ``[B, 4]`` ``(cx, cy, w, h)`` after sigmoid (box).

    Cells of the visual stream (null-cell embedding where empty, fixed 2-D sincos added) → 2 self-attn
    blocks → 2 cross-attn blocks into the conditioning stream's tokens (a learned null-conditioning
    token is always present so the softmax never sees an empty key set) → one learned box query over
    the cells → ``MLP(D→D→4)`` + sigmoid."""

    def __init__(self, dim: int, *, grid: int, visual_tag: int, conditioning_tag: int, num_heads: int = 16,
                 self_attn_depth: int = 2, cross_attn_depth: int = 2, mlp_ratio: int = 4, out: int = 4,
                 activation: str | None = "sigmoid") -> None:
        super().__init__()
        check_heads(dim, num_heads)
        if dim % 4 != 0:
            raise ValueError(f"dense_av_v1 needs D divisible by 4 for the 2-D sincos, got D={dim}")
        self.grid, self.visual_tag, self.conditioning_tag = int(grid), int(visual_tag), int(conditioning_tag)
        self.out, self.activation = int(out), activation
        mlp_dim = dim * int(mlp_ratio)
        self.v_proj = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)
        self.null_cell = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.null_cell, std=0.02)
        self.null_cond = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.null_cond, std=0.02)
        g = self.grid
        yy = axis_centres(g).reshape(g, 1).expand(g, g)
        xx = axis_centres(g).reshape(1, g).expand(g, g)
        # A buffer, not a parameter: cell position is a fixed fact about the grid.
        self.register_buffer("cell_pos", sincos_embed(torch.stack([yy, xx], -1).reshape(g * g, 2), dim).unsqueeze(0),
                             persistent=False)
        self.self_attn = nn.ModuleList(SelfAttnBlock(dim, num_heads, mlp_dim) for _ in range(int(self_attn_depth)))
        self.cross_attn = nn.ModuleList(CrossAttnBlock(dim, num_heads, mlp_dim) for _ in range(int(cross_attn_depth)))
        self.cell_norm = nn.LayerNorm(dim)
        self.query = trunc_normal_query(dim)
        self.q_norm = nn.LayerNorm(dim)
        self.q_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.out))

    def forward(self, memory: Tensor, pos: Tensor) -> Tensor:
        b = memory.shape[0]
        pos = _pos_batch(pos, b).to(memory.device)
        cells, counts = pool_cells(memory, pos, grid=self.grid, stream_tag=self.visual_tag)
        x = self.v_proj(cells)
        x = torch.where((counts > 0).unsqueeze(-1), x, self.null_cell.to(x.dtype))
        x = x + self.cell_pos.to(x.dtype)
        cond_mask = pos[..., 0] == float(self.conditioning_tag)  # [B, N]
        cond = torch.cat([self.c_proj(memory), self.null_cond.expand(b, -1, -1).to(memory.dtype)], dim=1)
        pad = torch.cat([~cond_mask, torch.zeros(b, 1, dtype=torch.bool, device=memory.device)], dim=1)
        for blk in self.self_attn:
            x = blk(x)
        for blk in self.cross_attn:
            x = blk(x, cond, key_padding_mask=pad)
        x = self.cell_norm(x)
        kv = x + self.cell_pos.to(x.dtype)
        q = self.q_norm(self.query.expand(b, -1, -1))
        pooled = self.q_attn(q, kv, kv, need_weights=False)[0].squeeze(1)
        y = self.mlp(pooled)
        if self.activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.activation not in (None, "none"):
            raise ValueError(f"unknown head.activation {self.activation!r}")
        return y


def build_dense_av_head(dim: int, head: dict, *, visual_tag: int = 0, conditioning_tag: int = 1) -> nn.Module:
    """``[B, N, D] + pos + stream tags`` → ``[B, 4]`` ``(cx, cy, w, h)`` after sigmoid (box).
    ``visual_tag`` / ``conditioning_tag`` are the indices of ``readout_params.visual_stream`` /
    ``conditioning_stream`` in ``describe()["streams"]`` (resolved by the readout driver)."""
    if head.get("family") != "dense_av_v1":
        raise ValueError(f"build_dense_av_head: head.family must be dense_av_v1, got {head.get('family')!r}")
    if parse_layout(str(head.get("layout", "box")))[0] != "box":
        raise ValueError(f"dense_av_v1 layout must be 'box', got {head.get('layout')!r}")
    return DenseAVHead(
        int(dim),
        grid=int(head.get("grid", 14)),
        visual_tag=int(visual_tag),
        conditioning_tag=int(conditioning_tag),
        num_heads=int(head.get("num_heads", 16)),
        self_attn_depth=int(head.get("self_attn_depth", 2)),
        cross_attn_depth=int(head.get("cross_attn_depth", 2)),
        mlp_ratio=int(head.get("mlp_ratio", 4)),
        out=int(head.get("out", 4)),
        activation=head.get("activation", "sigmoid"),
    )
