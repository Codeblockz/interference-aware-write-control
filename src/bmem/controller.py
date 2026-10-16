"""Write controllers: map (incoming association, memory state) -> per-slot gates.

All writers share one interface::

    gates = writer(mem, key, value, pos, seq_len, ctx)   # (B, S), entries in [0, 1]

``ctx`` is a small projection of the backbone output at the current position (what the
token *is*); only learned writers use it, for the scalar "write at all" decision.

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

    ctx_dim: int = 0  # >0 for writers that take a content projection

    def forward(self, mem: SlotMemory, key: Tensor, value: Tensor, pos: int, seq_len: int, ctx: Tensor | None = None) -> Tensor:
        raise NotImplementedError


class InterferenceController(Writer):
    ctx_dim = 16

    def __init__(self, hidden: int = 64, interference: bool = True, hard: bool = True):
        super().__init__()
        self.interference = interference
        self.hard = hard  # straight-through one-hot slot choice; soft gradient
        self.n_feat = 6
        self.slot_mlp = nn.Sequential(nn.Linear(self.n_feat, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.write_mlp = nn.Sequential(nn.Linear(3 + self.ctx_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

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

    def forward(self, mem, key, value, pos, seq_len, ctx=None):
        phi, pooled = self.features(mem, key, value, seq_len)
        if ctx is None:
            ctx = pooled.new_zeros(pooled.shape[0], self.ctx_dim)
        slot_logits = self.slot_mlp(phi).squeeze(-1)  # (B, S): where to write (interference, age)
        write = torch.sigmoid(self.write_mlp(torch.cat([pooled, ctx], -1)))  # (B, 1): whether to write (content + memory)
        soft = torch.softmax(slot_logits, dim=-1)
        if self.hard:
            one_hot = torch.nn.functional.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
            soft = one_hot + soft - soft.detach()
        return write * soft


class RecencyWriter(Writer):
    """Round-robin: always overwrite the slot written longest ago (oldest age; empty first)."""

    def forward(self, mem, key, value, pos, seq_len, ctx=None):
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

    def forward(self, mem, key, value, pos, seq_len, ctx=None):
        readout = mem.read(key)
        residual = (readout - value).norm(dim=-1) / (mem.d**0.5)  # (B,)
        write = torch.sigmoid(self.sharpness * (residual - self.threshold)).unsqueeze(-1)
        score = mem.age + (1 - mem.filled) * 1e6
        idx = score.argmax(dim=-1)
        return write * torch.nn.functional.one_hot(idx, mem.S).to(key.dtype)


class PositionWriter(Writer):
    """Proteus-style incremental activation: slot j is unlocked once pos >= j * T / S,
    writes go round-robin among unlocked slots. Capacity grows with position."""

    def forward(self, mem, key, value, pos, seq_len, ctx=None):
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
