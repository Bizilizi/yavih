"""Pre-norm transformer building blocks shared by every head family.

In the pinned shapes: ``nn.MultiheadAttention`` with ``batch_first``, GELU MLP of ``mlp_ratio * dim``, final LayerNorm.
No dropout / stochastic depth (both 0 in every pinned config).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["Mlp", "SelfAttnBlock", "CrossAttnBlock", "QueryDecoderBlock", "check_heads", "trunc_normal_query"]


def check_heads(dim: int, num_heads: int) -> None:
    """``nn.MultiheadAttention`` needs ``dim % num_heads == 0``; fail with the head count in the message so an
    agent sees that the bridge's ``D`` and the pinned 16 heads disagree."""
    if dim % num_heads != 0:
        raise ValueError(f"feature dim D={dim} is not divisible by num_heads={num_heads} (head geometry)")


def trunc_normal_query(dim: int) -> nn.Parameter:
    """A single learned query token ``[1, 1, D]``, ``trunc_normal(std=0.02)``."""
    q = nn.Parameter(torch.zeros(1, 1, dim))
    nn.init.trunc_normal_(q, std=0.02)
    return q


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SelfAttnBlock(nn.Module):
    """``x += SA(LN x); x += MLP(LN x)``."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        check_heads(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_dim)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)[0]
        return x + self.mlp(self.norm2(x))


class CrossAttnBlock(nn.Module):
    """``x += CA(LN x, LN mem); x += MLP(LN x)`` — cells → conditioning tokens (dense_av_v1)."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        check_heads(dim, num_heads)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_dim)

    def forward(self, x: Tensor, memory: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        kv = self.norm_kv(memory)
        x = x + self.attn(self.norm_q(x), kv, kv, need_weights=False, key_padding_mask=key_padding_mask)[0]
        return x + self.mlp(self.norm_mlp(x))


class QueryDecoderBlock(nn.Module):
    """Perceiver-IO / DETR query block: self-attn over the queries, cross-attn into the memory, MLP.
    With one query the self-attention is the identity map up to the
    residual, which keeps the clf and dense heads structurally one family."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int) -> None:
        super().__init__()
        check_heads(dim, num_heads)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_dim)

    def forward(self, q: Tensor, memory: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        h = self.norm_self(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        kv = self.norm_kv(memory)
        q = q + self.cross_attn(self.norm_q(q), kv, kv, need_weights=False, key_padding_mask=key_padding_mask)[0]
        return q + self.mlp(self.norm_mlp(q))
