"""Write controllers: map (incoming association, memory state) -> per-slot gates.

All writers share one interface::

    gates = writer(mem, key, value, pos, seq_len)   # (B, S), entries in [0, 1]

``InterferenceController`` is the proposed method. Per-slot features

    phi_{t,j} = [ sim(k_t, k_j),            key similarity        (interference)
                  ||v_t - v_j||,            value disagreement    (interference)
                  novelty_t = 1 - max_j sim, broadcast to every slot
                  age_j / T,
                  usage_j / T,
                  filled_j ]

go through a 2-layer MLP to a logit per slot. The gate is

    g_t = sigma(w_t) * softmax_j(logit_{t,j})

where w_t is a scalar "write at all" logit from the same MLP's pooled input, so one slot
dominates each write while the whole thing stays differentiable. Setting
``interference=False`` zeros the two interference features; that is the H2 ablation and
doubles as the input-only routing baseline (Raven-style, routing on input novelty alone).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from bmem.memory import SlotMemory


class Writer(nn.Module):
    """Base class. Subclasses implement ``forward``; ``n_features`` documents the stats cost."""

    n_stat_floats: int = 0  # extra per-sequence floats the writer keeps beyond the memory

    def forward(self, mem: SlotMemory, key: Tensor, value: Tensor, pos: int, seq_len: int) -> Tensor:
        raise NotImplementedError


class InterferenceController(Writer):
    def __init__(self, hidden: int = 64, interference: bool = True):
        super().__init__()
        self.interference = interference
        self.n_feat = 6
        self.slot_mlp = nn.Sequential(nn.Linear(self.n_feat, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.write_mlp = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def features(self, mem: SlotMemory, key: Tensor, value: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        sim = mem.similarity(key)  # (B, S)
        dis = mem.value_disagreement(value)  # (B, S)
        novelty = 1 - sim.max(dim=-1, keepdim=True).values  # (B, 1)
        if not self.interference:
            sim = torch.zeros_like(sim)
            dis = torch.zeros_like(dis)
        T = float(seq_len)
        phi = torch.stack(
            [sim, dis, novelty.expand_as(sim), mem.age / T, mem.usage / T, mem.filled], dim=-1
        )  # (B, S, 6)
        pooled = torch.cat([novelty, mem.filled.mean(-1, keepdim=True), dis.mean(-1, keepdim=True)], -1)
        return phi, pooled

    def forward(self, mem, key, value, pos, seq_len):
        phi, pooled = self.features(mem, key, value, seq_len)
        slot_logits = self.slot_mlp(phi).squeeze(-1)  # (B, S)
        write = torch.sigmoid(self.write_mlp(pooled))  # (B, 1)
        return write * torch.softmax(slot_logits, dim=-1)


class RecencyWriter(Writer):
    """Round-robin: always overwrite the slot written longest ago (oldest age; empty first)."""

    def forward(self, mem, key, value, pos, seq_len):
        score = mem.age + (1 - mem.filled) * 1e6  # empty slots first, then oldest
        idx = score.argmax(dim=-1)
        return torch.nn.functional.one_hot(idx, mem.S).to(key.dtype)


class ResidualCacheWriter(Writer):
    """HOLA-style: write only when the memory's current readout for this key disagrees
    with the incoming value (large residual), into the oldest slot. The threshold is
    learned (one scalar) so the write rate adapts; the slot choice is not learned."""

    n_stat_floats = 1

    def __init__(self, init_threshold: float = 0.5):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        self.sharpness = nn.Parameter(torch.tensor(8.0))

    def forward(self, mem, key, value, pos, seq_len):
        readout = mem.read(key)
        residual = (readout - value).norm(dim=-1) / (mem.d**0.5)  # (B,)
        write = torch.sigmoid(self.sharpness * (residual - self.threshold)).unsqueeze(-1)
        score = mem.age + (1 - mem.filled) * 1e6
        idx = score.argmax(dim=-1)
        return write * torch.nn.functional.one_hot(idx, mem.S).to(key.dtype)


class PositionWriter(Writer):
    """Proteus-style incremental activation: slot j is unlocked once pos >= j * T / S,
    writes go round-robin among unlocked slots. Capacity grows with position."""

    def forward(self, mem, key, value, pos, seq_len):
        unlocked = int(min(mem.S, max(1, (pos + 1) * mem.S // seq_len + 1)))
        score = mem.age + (1 - mem.filled) * 1e6
        score[:, unlocked:] = -1e9
        idx = score.argmax(dim=-1)
        return torch.nn.functional.one_hot(idx, mem.S).to(key.dtype)


WRITERS = {
    "controller": lambda: InterferenceController(interference=True),
    "no_interference": lambda: InterferenceController(interference=False),
    "recency": RecencyWriter,
    "residual": ResidualCacheWriter,
    "position": PositionWriter,
}
