"""Recall ceiling of a non-clairvoyant writer: admit every pair at its value position, never
write filler, overwrite a key in place on update, FIFO-evict when full. Optionally with
perfect distractor rejection. python experiments/analysis/ceiling.py"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bmem.data import IGNORE, MQARConfig, generate_mqar  # noqa: E402


def ceiling(S, kv, upd, dist, L, n=1500, reject=False, vocab=8192):
    cfg = MQARConfig(vocab_size=vocab, seq_len=L, num_kv_pairs=kv, num_updates=upd, num_distractors=dist)
    b = generate_mqar(cfg, n, seed=0)
    x, y = b.inputs.numpy(), b.labels.numpy()
    upd_end, hits, tot = cfg.context_tokens(), 0, 0
    for i in range(n):
        qkeys = set(x[i][y[i] != IGNORE].tolist())
        mem, order = {}, []
        for t in range(1, upd_end, 2):
            k, v = int(x[i][t - 1]), int(x[i][t])
            if reject and k not in qkeys:
                continue
            if k in mem:
                mem[k] = v
            else:
                mem[k] = v
                order.append(k)
                if len(order) > S:
                    mem.pop(order.pop(0))
        for t in np.where(y[i] != IGNORE)[0]:
            tot += 1
            hits += mem.get(int(x[i][t])) == int(y[i][t])
    return hits / tot


if __name__ == "__main__":
    torch.manual_seed(0)
    for N, L in ((64, 512), (128, 1024)):
        for S in (32, 64, 128):
            print(f"N={N} S={S} L={L}: pair-only FIFO {ceiling(S, N, N // 4, N // 4, L):.3f} | "
                  f"+distractor rejection {ceiling(S, N, N // 4, N // 4, L, reject=True):.3f}")
