"""Training and evaluation loop."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bmem.data import IGNORE, MQARBatch
from bmem.metrics import bootstrap_ci, recall_per_example
from bmem.model import RecallModel


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup: int = 100
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    bf16: bool = True
    seed: int = 0


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * p))


def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), ignore_index=IGNORE)


@torch.no_grad()
def evaluate(model: RecallModel, split: MQARBatch, device, batch_size: int = 128, max_batches: int | None = None, bf16=True) -> dict:
    model.eval()
    per_ex, t0, n_tok = [], time.perf_counter(), 0
    for i in range(0, len(split), batch_size):
        if max_batches is not None and i // batch_size >= max_batches:
            break
        x = split.inputs[i : i + batch_size].to(device)
        y = split.labels[i : i + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits, _ = model(x)
        per_ex.append(recall_per_example(logits, y).cpu())
        n_tok += x.numel()
    torch.cuda.synchronize()
    per_ex = torch.cat(per_ex).numpy()
    lo, hi = bootstrap_ci(per_ex)
    model.train()
    return {
        "recall": float(per_ex.mean()),
        "recall_ci95": [lo, hi],
        "n_examples": int(len(per_ex)),
        "eval_tokens_per_s": n_tok / (time.perf_counter() - t0),
        "per_example": per_ex,
    }


def train(model: RecallModel, train_split: MQARBatch, val_split: MQARBatch, cfg: TrainConfig, device, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )
    N = len(train_split)
    rng = np.random.default_rng(cfg.seed)
    history, t0, tokens = [], time.perf_counter(), 0
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(cfg.steps):
        idx = torch.from_numpy(rng.integers(0, N, cfg.batch_size))
        x, y = train_split.inputs[idx].to(device), train_split.labels[idx].to(device)
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
            logits, _ = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        tokens += x.numel()
        if step % 50 == 0 or step == cfg.steps - 1:
            log(f"step {step:5d} loss {loss.item():.4f} gnorm {gn.item():.2f} lr {_lr_at(step, cfg):.2e}")
        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            ev = evaluate(model, val_split, device, max_batches=cfg.eval_batches, bf16=cfg.bf16)
            history.append({"step": step + 1, "loss": loss.item(), "val_recall": ev["recall"]})
            log(f"  eval step {step+1}: val recall {ev['recall']:.4f} ci {ev['recall_ci95']}")
    elapsed = time.perf_counter() - t0
    return {
        "history": history,
        "train_seconds": elapsed,
        "train_tokens_per_s": tokens / elapsed,
        "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20,
    }
