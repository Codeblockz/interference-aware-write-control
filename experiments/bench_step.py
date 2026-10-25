"""Time one training step for a config: python experiments/bench_step.py experiments/configs/target.yaml [--writer W] [--steps 20]

Prints seconds per step (forward + backward + optimizer) after warmup, and peak memory.
Used to measure the write-loop speedup (PLAN step 2) on identical shapes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmem.data import MQARConfig, generate_mqar  # noqa: E402
from bmem.model import ModelConfig, RecallModel  # noqa: E402
from bmem.train import loss_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--writer", default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    torch.manual_seed(0)
    dcfg = MQARConfig(**cfg["data"])
    B = cfg["train"]["batch_size"]
    batch = generate_mqar(dcfg, B, seed=0)
    mcfg = ModelConfig(**{**cfg["model"], "vocab_size": dcfg.vocab_size})
    if a.writer:
        mcfg.writer = a.writer
    model = RecallModel(mcfg).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x, y = batch.inputs.cuda(), batch.labels.cuda()
    torch.cuda.reset_peak_memory_stats()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / a.steps
    print(f"{Path(a.config).name} writer={mcfg.writer} B={B} T={dcfg.seq_len} S={mcfg.slots} layers={mcfg.n_layers}: "
          f"{dt:.3f} s/step, peak {torch.cuda.max_memory_allocated()/2**20:.0f} MB")


if __name__ == "__main__":
    main()
