"""RecallModel: embed -> GLA backbone -> (read, then write) bounded slot memory -> answer head.

At every position t the backbone output h_t is projected to a query q_t, a key k_t and a
value v_t. The memory is *read* with q_t first (so a query token sees what was stored
before it) and then *written* with (k_t, v_t) through the chosen writer's gates. The answer
head sees [h_t ; r_t]. Loss is cross-entropy at labelled (answer) positions only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from bmem.backbone import GLABackbone
from bmem.controller import WRITERS, Writer
from bmem.memory import SlotMemory


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    slots: int = 32
    mem_dim: int = 128
    writer: str = "controller"
    controller_hidden: int = 64
    assoc_shift: int = 1  # key projected from h_{t-shift}: an association pairs each value with the preceding token


class RecallModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = GLABackbone(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads)
        self.to_q = nn.Linear(cfg.d_model, cfg.mem_dim)
        self.to_k = nn.Linear(cfg.d_model, cfg.mem_dim)
        self.to_v = nn.Linear(cfg.d_model, cfg.mem_dim)
        self.writer: Writer = WRITERS[cfg.writer]()
        if cfg.writer in ("controller", "no_interference"):
            self.writer = WRITERS[cfg.writer]()
            self.writer.slot_mlp[0] = nn.Linear(self.writer.n_feat, cfg.controller_hidden)
            self.writer.slot_mlp[2] = nn.Linear(cfg.controller_hidden, 1)
        self.to_ctx = nn.Linear(cfg.d_model, self.writer.ctx_dim) if self.writer.ctx_dim else None
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.mem_dim, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.vocab_size)
        )

    def forward(self, tokens: Tensor, return_gates: bool = False) -> tuple[Tensor, dict]:
        B, T = tokens.shape
        h = self.backbone(tokens)  # (B, T, d_model), parallel over T
        h_key = h if self.cfg.assoc_shift == 0 else torch.cat(
            [h.new_zeros(B, self.cfg.assoc_shift, h.shape[-1]), h[:, : -self.cfg.assoc_shift]], dim=1
        )
        q, k, v = self.to_q(h).float(), self.to_k(h_key).float(), self.to_v(h).float()
        ctx = self.to_ctx(h).float() if self.to_ctx is not None else None
        mem = SlotMemory(B, self.cfg.slots, self.cfg.mem_dim, tokens.device)
        reads, gates = [], []
        for t in range(T):
            reads.append(mem.read(q[:, t]))
            g = self.writer(mem, k[:, t], v[:, t], t, T, None if ctx is None else ctx[:, t])
            mem.write(g, k[:, t], v[:, t])
            if return_gates:
                gates.append(g)
        r = torch.stack(reads, dim=1).to(h.dtype)  # (B, T, mem_dim)
        logits = self.head(torch.cat([h, r], dim=-1))
        info = {"gates": torch.stack(gates, 1) if return_gates else None, "mem": mem}
        return logits, info

    def state_bytes(self) -> dict:
        backbone = self.backbone.state_bytes()
        memory = SlotMemory.state_bytes(self.cfg.slots, self.cfg.mem_dim)
        writer = 4 * self.writer.n_stat_floats
        return {"backbone": backbone, "memory": memory, "writer": writer, "total": backbone + memory + writer}

    def n_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        writer = sum(p.numel() for p in self.writer.parameters())
        return {"total": total, "writer": writer}
