"""Gated linear attention backbone (pre-norm GLA + MLP blocks) via flash-linear-attention.

The recurrent state of each GLA layer is a (heads, d_head, d_head) matrix per sequence;
``state_bytes`` reports it so the memory budget comparison counts the backbone too.

``mixer="attention"`` swaps the GLA layer for causal softmax attention with rotary positions
(the full-attention reference of the paper: an unbounded KV cache, no slot memory). Its
"state" is the KV cache, which grows with the sequence, so ``state_bytes`` needs ``seq_len``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

import torch.nn.functional as F

try:
    from fla.layers import GatedLinearAttention
    from fla.modules import RMSNorm, RotaryEmbedding
except ImportError as e:  # pragma: no cover
    raise ImportError("flash-linear-attention is required: pip install flash-linear-attention") from e


class CausalAttention(nn.Module):
    """Multi-head causal softmax attention with rotary position embeddings (reference only)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryEmbedding(self.d_head)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).unbind(2)  # (B, T, H, D) each
        q, k = self.rotary(q, k)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, layer_idx: int = 0, mixer: str = "gla"):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        if mixer == "gla":
            self.attn = GatedLinearAttention(hidden_size=d_model, num_heads=n_heads, layer_idx=layer_idx)
        elif mixer == "attention":
            self.attn = CausalAttention(d_model, n_heads)
        else:
            raise ValueError(f"unknown mixer {mixer!r}; expected 'gla' or 'attention'")
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model), nn.GELU(), nn.Linear(mlp_ratio * d_model, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        a = self.attn(self.norm1(x))
        x = x + (a[0] if isinstance(a, tuple) else a)
        x = x + self.mlp(self.norm2(x))
        return x


class GLABackbone(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, mixer: str = "gla"):
        super().__init__()
        self.d_model, self.n_layers, self.n_heads, self.mixer = d_model, n_layers, n_heads, mixer
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, layer_idx=i, mixer=mixer) for i in range(n_layers))
        self.norm = RMSNorm(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def state_bytes(self, dtype_bytes: int = 4, seq_len: int | None = None) -> int:
        """Recurrent state per sequence: one (d_head x d_head) matrix per head per layer,
        using fla's default expand ratios (expand_k=0.5, expand_v=1.0). For the attention
        mixer the state is the KV cache at ``seq_len`` tokens (keys and values, every layer)."""
        if self.mixer == "attention":
            assert seq_len is not None, "attention state is the KV cache: pass seq_len"
            return self.n_layers * seq_len * 2 * self.d_model * dtype_bytes
        d_k = int(self.d_model * 0.5) // self.n_heads
        d_v = self.d_model // self.n_heads
        return self.n_layers * self.n_heads * d_k * d_v * dtype_bytes
