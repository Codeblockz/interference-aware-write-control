"""Exact-recall metrics on labelled positions, with per-example bookkeeping for CIs."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from bmem.data import IGNORE


@torch.no_grad()
def recall_per_example(logits: Tensor, labels: Tensor) -> Tensor:
    """Fraction of answer positions predicted exactly, per example: (B,)."""
    mask = labels != IGNORE
    correct = (logits.argmax(-1) == labels) & mask
    return correct.sum(1).float() / mask.sum(1).clamp_min(1).float()


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """CI on mean(a - b) over matched examples (same test sequences, two models)."""
    assert a.shape == b.shape
    return bootstrap_ci(a - b, n_boot, alpha, seed)


@torch.no_grad()
def gate_stats(gates: Tensor, inputs: Tensor, labels: Tensor, filled: Tensor, cfg) -> dict:
    """Diagnostics of where a writer spends its budget, from one batch.

    gates: (B, T, S) write gates; inputs/labels: (B, T); filled: (B, S) end-of-sequence
    occupancy; cfg: the MQARConfig that generated the batch (gives the block layout).

    Returns means over the batch:
      write_frac     mean per-step gate mass (sum over slots), all positions
      slot_entropy   entropy (bits) of the time-averaged slot distribution; max log2(S)
      filled_frac    mean occupancy at the end of the sequence
      write_by_class write_frac at value positions of queried pairs, distractor pairs,
                     update pairs, and at every other position (query keys, filler)
      update_key_present  fraction of update positions at which the updated key is still
                     held by some slot (its earlier pair has not been evicted), judging a slot
                     "written" at a step when its gate mass exceeds 0.5 (hard gates give 0/1
                     times the write scalar)
      update_in_place     fraction of those positions at which the writer puts the new value
                     into the slot that holds the key (overwrite in place, the behaviour the
                     interference features were designed for), rather than a fresh slot
    """
    B, T, S = gates.shape
    mass = gates.float().sum(-1)  # (B, T)
    p = gates.float().sum((0, 1))
    p = p / p.sum().clamp_min(1e-12)
    entropy = float(-(p * torch.log2(p.clamp_min(1e-12))).sum())

    n_pairs = cfg.num_kv_pairs + cfg.num_distractors
    ctx_end, upd_end = 2 * n_pairs, cfg.context_tokens()
    pos = torch.arange(T, device=gates.device)
    value_pos = (pos % 2 == 1)
    in_ctx, in_upd = value_pos & (pos < ctx_end), value_pos & (pos >= ctx_end) & (pos < upd_end)
    # Queried keys are the inputs at labelled positions; a context pair is queried if its key is one.
    qmask = labels != IGNORE
    nq = int(qmask[0].sum())
    classes = {}
    if nq > 0 and bool((qmask.sum(1) == nq).all()):
        qkeys = inputs[qmask].view(B, nq)
        prev_key = torch.roll(inputs, 1, dims=1)  # key token precedes its value token
        is_queried = (prev_key.unsqueeze(-1) == qkeys.unsqueeze(1)).any(-1)  # (B, T)
        ctx_q, ctx_d = in_ctx & is_queried, in_ctx & ~is_queried
        classes = {
            "queried": _masked_mean(mass, ctx_q),
            "distractor": _masked_mean(mass, ctx_d),
            "update": _masked_mean(mass, in_upd),
            "other": _masked_mean(mass, ~(in_ctx | in_upd)),
        }
    present, in_place = _update_handling(gates, inputs, in_upd)
    return {
        "write_frac": float(mass.mean()),
        "slot_entropy": entropy,
        "filled_frac": float(filled.float().mean()),
        "write_by_class": classes,
        "update_key_present": present,
        "update_in_place": in_place,
    }


@torch.no_grad()
def _update_handling(gates: Tensor, inputs: Tensor, in_upd: Tensor, threshold: float = 0.5) -> tuple[float, float]:
    """Replay the hard writes to track which key token each slot holds; at every update
    value position, check whether the updated key is still held and whether the write goes
    to that slot. Returns (update_key_present, update_in_place) as batch fractions."""
    B, T, S = gates.shape
    g = gates.float()
    slot_key = torch.full((B, S), -1, dtype=torch.long, device=gates.device)
    n_upd = n_present = n_in_place = 0
    for t in range(1, T):
        key_tok = inputs[:, t - 1]  # the key token precedes its value token
        gt = g[:, t]
        written = gt.sum(-1) > threshold
        slot = gt.argmax(-1)
        if bool(in_upd[t]):
            held = (slot_key == key_tok.unsqueeze(1)).any(-1)
            chosen_holds = slot_key.gather(1, slot.unsqueeze(1)).squeeze(1) == key_tok
            n_upd += B
            n_present += int(held.sum())
            n_in_place += int((held & written & chosen_holds).sum())
        if bool(written.any()):
            idx = written.nonzero(as_tuple=True)[0]
            slot_key[idx, slot[idx]] = key_tok[idx]
    if n_upd == 0:
        return float("nan"), float("nan")
    return n_present / n_upd, (n_in_place / n_present if n_present else float("nan"))


def _masked_mean(x: Tensor, m: Tensor) -> float:
    m = m.expand_as(x) if m.dim() == 1 else m
    n = int(m.sum())
    return float((x * m).sum() / n) if n else float("nan")
