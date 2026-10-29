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

    uses_pos: bool = False  # True if the gate depends on the position; the compiled path then passes step_inputs(...)[t] as ``pos``

    def step_inputs(self, seq_len: int, slots: int, device):
        """For ``uses_pos`` writers: a (T, ...) tensor whose row t is passed as ``pos`` on the compiled path."""
        return None

    def forward(self, mem: SlotMemory, key: Tensor, value: Tensor, pos: int, seq_len, ctx: Tensor | None = None) -> Tensor:
        """``seq_len`` is the int T on the eager path, or a 0-d tensor holding 1/T on the
        compiled path (``RecallModel.memory_step``); use :func:`_inv_len` to normalise."""
        raise NotImplementedError


def _inv_len(seq_len) -> Tensor | float:
    return seq_len if isinstance(seq_len, Tensor) else 1.0 / float(seq_len)


class InterferenceController(Writer):
    ctx_dim = 16

    def __init__(self, hidden: int = 64, interference: bool = True, hard: bool = True, prior: float = 5.0,
                 novelty: bool = True):
        super().__init__()
        self.interference = interference
        self.novelty = novelty  # False: also zero the key-novelty feature (true input-only ablation, LOG "S8")
        self.hard = hard  # straight-through one-hot slot choice; soft gradient
        self.n_feat = 6
        self.slot_mlp = nn.Sequential(nn.Linear(self.n_feat, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.write_mlp = nn.Sequential(nn.Linear(3 + self.ctx_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Recency prior on the slot choice: at init the controller behaves like the recency
        # writer (empty slot first, then oldest), so retrieval works from step 0 and the
        # learned logits only have to express deviations from it.
        self.prior = nn.Parameter(torch.tensor(prior))

    def features(self, mem: SlotMemory, key: Tensor, value: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        sim = mem.similarity(key)  # (B, S)
        dis = mem.value_disagreement(value)  # (B, S)
        novelty = 1 - sim.max(dim=-1, keepdim=True).values  # (B, 1)
        if not self.interference:
            sim = torch.zeros_like(sim)
            dis = torch.zeros_like(dis)
        if not self.novelty:
            novelty = torch.zeros_like(novelty)
        inv_T = _inv_len(seq_len)
        phi = torch.stack(
            [sim, dis, novelty.expand_as(sim), mem.age * inv_T, mem.usage * inv_T, mem.filled], dim=-1
        )  # (B, S, 6)
        pooled = torch.cat([novelty, mem.filled.mean(-1, keepdim=True), dis.mean(-1, keepdim=True)], -1)
        return phi, pooled

    def forward(self, mem, key, value, pos, seq_len, ctx=None):
        phi, pooled = self.features(mem, key, value, seq_len)
        if ctx is None:
            ctx = pooled.new_zeros(pooled.shape[0], self.ctx_dim)
        slot_logits = self.slot_mlp(phi).squeeze(-1)  # (B, S): where to write (interference, age)
        slot_logits = slot_logits + self.prior * (mem.age * _inv_len(seq_len) + (1 - mem.filled))
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


class PairRecencyWriter(Writer):
    """Structure-aware recency: writes only at the value position of a pair (odd positions
    inside the context + update block of length ``pair_block``), into the oldest slot. The
    non-learned upper reference: it is the FIFO pair-only policy whose recall ceiling
    `experiments/analysis/ceiling.py` computes. Uses the generator's layout, so it is an
    oracle-structure baseline, not a model that could be deployed on unknown data."""

    uses_pos = True

    def __init__(self, pair_block: int = 0):
        super().__init__()
        self.pair_block = pair_block

    def step_inputs(self, seq_len, slots, device):
        t = torch.arange(seq_len, device=device)
        return ((t % 2 == 1) & (t < self.pair_block)).float()  # (T,)

    def forward(self, mem, key, value, pos, seq_len, ctx=None):
        w = pos if isinstance(pos, Tensor) else float(pos % 2 == 1 and pos < self.pair_block)
        score = mem.age + (1 - mem.filled) * 1e6
        idx = score.argmax(dim=-1)
        return torch.nn.functional.one_hot(idx, mem.S).to(key.dtype) * w


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

    uses_pos = True

    @staticmethod
    def unlocked_mask(seq_len: int, slots: int, device) -> Tensor:
        """(T, S) bool: slot j is unlocked at position t iff j < (t + 1) * S // T + 1 (capped at S)."""
        t = torch.arange(seq_len, device=device)
        n = ((t + 1) * slots // seq_len + 1).clamp(1, slots)  # (T,)
        return torch.arange(slots, device=device)[None, :] < n[:, None]

    def step_inputs(self, seq_len, slots, device):
        return self.unlocked_mask(seq_len, slots, device)

    def forward(self, mem, key, value, pos, seq_len, ctx=None):
        score = mem.age + (1 - mem.filled) * 1e6
        if isinstance(pos, Tensor):  # compiled path: pos is the (S,) unlocked mask for this step
            score = score.masked_fill(~pos, -1e9)
        else:
            unlocked = int(min(mem.S, max(1, (pos + 1) * mem.S // seq_len + 1)))
            score = score.clone()
            score[:, unlocked:] = -1e9
        idx = score.argmax(dim=-1)
        return torch.nn.functional.one_hot(idx, mem.S).to(key.dtype)


class NoneWriter(Writer):
    """No slot memory at all (the full-attention reference): ``RecallModel`` skips the memory
    loop and feeds a zero readout to the head, so the head's shape is unchanged."""

    def forward(self, mem, key, value, pos, seq_len, ctx=None):
        return key.new_zeros(key.shape[0], mem.S)


WRITERS = {
    "none": NoneWriter,
    "controller": lambda: InterferenceController(interference=True),
    "no_interference": lambda: InterferenceController(interference=False),
    "input_only": lambda: InterferenceController(interference=False, novelty=False),  # sees nothing from stored keys/values
    "recency": RecencyWriter,
    "pair_recency": PairRecencyWriter,  # needs ModelConfig.pair_block; RecallModel sets it
    "residual": ResidualCacheWriter,
    "position": PositionWriter,
}
