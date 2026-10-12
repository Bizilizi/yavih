"""Attentive probe head ``attentive_probe_v1``: 3 pre-norm self-attention blocks over the
token set, one cross-attention block with a single learned query (``trunc_normal(std=0.02)``),
``Linear(D, C)``; 16 heads, mlp_ratio 4.

``memory_pos: none`` is load-bearing: the head is permutation
invariant in the token axis, so clf credits token *contents* only."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from yavih.probes._blocks import QueryDecoderBlock, SelfAttnBlock, trunc_normal_query

__all__ = ["build_attentive_probe", "ATTENTIVE_DESCRIPTOR", "AttentiveProbe"]

ATTENTIVE_DESCRIPTOR: dict[str, Any] = {
    "family": "attentive_probe_v1",
    "self_attn_depth": 3,
    "cross_attn_depth": 1,
    "num_heads": 16,
    "mlp_ratio": 4,
    "memory_pos": "none",
    "layout": "point",
}


class AttentiveProbe(nn.Module):
    """``[B, N, D]`` → ``[B, C]`` logits. ``pos`` is accepted and ignored so the trainer can call every head
    family with the same ``(feats, pos)`` signature."""

    def __init__(self, dim: int, num_classes: int, *, num_heads: int = 16, self_attn_depth: int = 3,
                 cross_attn_depth: int = 1, mlp_ratio: int = 4) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("attentive probe needs num_classes >= 1 (head.out unresolved? pass bench= to pinned())")
        mlp_dim = dim * int(mlp_ratio)
        self.self_attn = nn.ModuleList(SelfAttnBlock(dim, num_heads, mlp_dim) for _ in range(int(self_attn_depth)))
        self.self_norm = nn.LayerNorm(dim)
        self.query = trunc_normal_query(dim)
        self.cross_attn = nn.ModuleList(QueryDecoderBlock(dim, num_heads, mlp_dim) for _ in range(int(cross_attn_depth)))
        self.cross_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, int(num_classes))

    def forward(self, memory: Tensor, pos: Tensor | None = None) -> Tensor:
        x = memory
        for blk in self.self_attn:
            x = blk(x)
        x = self.self_norm(x)
        q = self.query.expand(memory.shape[0], -1, -1)
        for blk in self.cross_attn:
            q = blk(q, x)
        pooled = self.cross_norm(q).squeeze(1)
        return self.classifier(pooled)


def build_attentive_probe(dim: int, num_classes: int, head: dict) -> nn.Module:
    """Returns a ``torch.nn.Module`` mapping ``[B, N, D]`` → ``[B, C]`` (clf)."""
    if head.get("family") != "attentive_probe_v1":
        raise ValueError(f"build_attentive_probe: head.family must be attentive_probe_v1, got {head.get('family')!r}")
    if head.get("memory_pos", "none") != "none":
        raise ValueError("attentive_probe_v1 has memory_pos 'none'")
    return AttentiveProbe(
        int(dim),
        int(num_classes),
        num_heads=int(head.get("num_heads", 16)),
        self_attn_depth=int(head.get("self_attn_depth", 3)),
        cross_attn_depth=int(head.get("cross_attn_depth", 1)),
        mlp_ratio=int(head.get("mlp_ratio", 4)),
    )
