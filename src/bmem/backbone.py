"""Gated linear attention backbone (pre-norm GLA + MLP blocks) via flash-linear-attention.

The recurrent state of each GLA layer is a (heads, d_head, d_head) matrix per sequence;
``state_bytes`` reports it so the memory budget comparison counts the backbone too.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

try:
    from fla.layers import GatedLinearAttention
    from fla.modules import RMSNorm
except ImportError as e:  # pragma: no cover
    raise ImportError("flash-linear-attention is required: pip install flash-linear-attention") from e


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, layer_idx: int = 0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GatedLinearAttention(hidden_size=d_model, num_heads=n_heads, layer_idx=layer_idx)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(), nn.Linear(mlp_ratio * d_model, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class GLABackbone(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.d_model, self.n_layers, self.n_heads = d_model, n_layers, n_heads
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, layer_idx=i) for i in range(n_layers))
        self.norm = RMSNorm(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def state_bytes(self, dtype_bytes: int = 4) -> int:
        """Recurrent state per sequence: one (d_head x d_head) matrix per head per layer,
        using fla's default expand ratios (expand_k=0.5, expand_v=1.0)."""
        d_k = int(self.d_model * 0.5) // self.n_heads
        d_v = self.d_model // self.n_heads
        return self.n_layers * self.n_heads * d_k * d_v * dtype_bytes
