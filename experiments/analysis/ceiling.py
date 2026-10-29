"""Recall of idealised write policies on the generated task, simulated and in closed form.

Policies (N queried pairs, U updates, D distractors; P = N + D context pairs; Q = N - U
queried keys whose context value is final; S slots):
  pair-only FIFO   admit every pair at its value position, updates to a fresh slot, evict
                   the oldest: U + min(S - U, P) * Q / P correct of N  (= the pair_recency writer)
  in-place FIFO    the same but an update overwrites its key's slot: U + min(S - U, Q + D) * Q / (Q + D)
  foreknowledge    reject distractors and overwrite in place: min(S, N) / N  (the bound for any policy)
The simulation (``ceiling``) replays the generator; ``closed_form`` gives the formulas, which
agree with it within 0.5 pp. python experiments/analysis/ceiling.py"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bmem.data import IGNORE, MQARConfig, generate_mqar  # noqa: E402


def ceiling(S, kv, upd, dist, L, n=1500, reject=False, vocab=8192, inplace=True):
    cfg = MQARConfig(vocab_size=vocab, seq_len=L, num_kv_pairs=kv, num_updates=upd, num_distractors=dist)
    b = generate_mqar(cfg, n, seed=0)
    x, y = b.inputs.numpy(), b.labels.numpy()
    upd_end, hits, tot = cfg.context_tokens(), 0, 0
    for i in range(n):
        qkeys = set(x[i][y[i] != IGNORE].tolist())
        if inplace:
            mem, order = {}, []
            for t in range(1, upd_end, 2):
                k, v = int(x[i][t - 1]), int(x[i][t])
                if reject and k not in qkeys:
                    continue
                if k in mem:
                    mem[k] = v  # overwrite in place; the slot keeps its FIFO position
                else:
                    mem[k] = v
                    order.append(k)
                    if len(order) > S:
                        mem.pop(order.pop(0))
            lookup = mem
        else:
            # pair_recency: every pair write takes a slot (an update's stale copy stays until
            # evicted); a key reads back its most recent write that is still held.
            entries = []  # FIFO of (key, value), newest last
            for t in range(1, upd_end, 2):
                k, v = int(x[i][t - 1]), int(x[i][t])
                if reject and k not in qkeys:
                    continue
                entries.append((k, v))
                if len(entries) > S:
                    entries.pop(0)
            lookup = {}
            for k, v in entries:
                lookup[k] = v  # later entries overwrite earlier ones in the dict
        for t in np.where(y[i] != IGNORE)[0]:
            tot += 1
            hits += lookup.get(int(x[i][t])) == int(y[i][t])
    return hits / tot


def closed_form(S: int, N: int, U: int, D: int) -> dict:
    """Idealised recall (fractions) of the three policies in the module docstring."""
    P, Q = N + D, N - U
    pair_only = (U + min(S - U, P) * Q / P) / N if S > U else min(S, U) / N
    in_place = (U + min(S - U, Q + D) * Q / (Q + D)) / N if S > U else min(S, U) / N
    foreknowledge = min(S, N) / N
    return {"pair_only": pair_only, "in_place": in_place, "foreknowledge": foreknowledge,
            "h1_margin": foreknowledge - pair_only, "in_place_margin": in_place - pair_only}


if __name__ == "__main__":
    torch.manual_seed(0)
    print("N  S   pair-only(sim)  in-place(sim)  foreknowledge(sim) | closed: pair-only  in-place  foreknowledge  H1 margin  of which in-place")
    for N, L in ((64, 512), (128, 1024)):
        for S in (32, 64, 128):
            U = D = N // 4
            cf = closed_form(S, N, U, D)
            print(f"{N:3d} {S:3d}   {ceiling(S, N, U, D, L, inplace=False):.3f}          {ceiling(S, N, U, D, L):.3f}          "
                  f"{ceiling(S, N, U, D, L, reject=True):.3f}            | {cf['pair_only']:.4f}   {cf['in_place']:.4f}   "
                  f"{cf['foreknowledge']:.4f}        {100*cf['h1_margin']:5.2f}      {100*cf['in_place_margin']:5.2f}")
