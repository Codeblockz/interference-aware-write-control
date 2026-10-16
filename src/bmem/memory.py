"""Bounded slot memory.

S slots, each holding a key k_j and a value v_j in R^d. Reads are soft attention over slot
keys. Writes are gated convex updates:

    m_{t,j} = (1 - g_{t,j}) m_{t-1,j} + g_{t,j} u_t

where m_j = (k_j, v_j), u_t = (k_t, v_t) is the incoming association, and g_{t,j} in [0, 1]
is produced by a writer (learned controller or a fixed baseline rule).

Slot *age* (steps since last substantial write) and *usage* (cumulative write mass) are
tracked as writer features and for the recency baseline. They are not part of the model's
learnable state but they are part of the memory's byte budget, and are counted as such.
"""

from __future__ import annotations

import torch
from torch import Tensor


class SlotMemory:
    """Per-batch memory state. Plain tensors, no parameters; writers own the parameters."""

    def __init__(self, batch: int, slots: int, dim: int, device, dtype=torch.float32):
        self.S, self.d = slots, dim
        self.keys = torch.zeros(batch, slots, dim, device=device, dtype=dtype)
        self.values = torch.zeros(batch, slots, dim, device=device, dtype=dtype)
        self.age = torch.zeros(batch, slots, device=device, dtype=dtype)
        self.usage = torch.zeros(batch, slots, device=device, dtype=dtype)
        self.filled = torch.zeros(batch, slots, device=device, dtype=dtype)  # soft occupancy

    def read(self, query: Tensor, temperature: float = 1.0) -> Tensor:
        """query: (B, d) -> (B, d) value readout, attention over occupied slots."""
        scores = torch.einsum("bd,bsd->bs", query, self.keys) / (self.d**0.5 * temperature)
        scores = scores + torch.log(self.filled.clamp_min(1e-6))  # unfilled slots ~ masked
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("bs,bsd->bd", attn, self.values)

    def write(self, gate: Tensor, key: Tensor, value: Tensor) -> None:
        """gate: (B, S) in [0,1]; key, value: (B, d). Convex update of every slot."""
        g = gate.unsqueeze(-1)
        self.keys = (1 - g) * self.keys + g * key.unsqueeze(1)
        self.values = (1 - g) * self.values + g * value.unsqueeze(1)
        self.filled = (1 - gate) * self.filled + gate
        self.age = (self.age + 1) * (1 - gate)
        self.usage = self.usage + gate

    def similarity(self, key: Tensor) -> Tensor:
        """Cosine similarity of an incoming key to every slot key: (B, S)."""
        k = torch.nn.functional.normalize(key, dim=-1).unsqueeze(1)
        mk = torch.nn.functional.normalize(self.keys, dim=-1)
        return (k * mk).sum(-1) * self.filled  # empty slots contribute 0

    def value_disagreement(self, value: Tensor) -> Tensor:
        """L2 distance between an incoming value and every slot value, scaled: (B, S)."""
        diff = self.values - value.unsqueeze(1)
        return diff.norm(dim=-1) / (self.d**0.5)

    @staticmethod
    def state_bytes(slots: int, dim: int, dtype_bytes: int = 4) -> int:
        """Bytes of memory state per sequence: keys + values + age + usage + filled."""
        return slots * (2 * dim + 3) * dtype_bytes
