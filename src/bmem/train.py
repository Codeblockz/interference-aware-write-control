"""Training and evaluation loop.

Logging: every 50 steps a timestamped line with the 50-step running loss, the pre-clip
gradient norm, the learning rate and seconds per step; at every eval a line with validation
recall, its CI, gate statistics and elapsed time. Checkpoints (`ckpt.pt`, overwritten) go to
`out_dir` every `ckpt_every` steps and `best.pt` (weights only) whenever validation recall
improves. A run can be resumed from `ckpt.pt`, or initialised from another run's weights
with `init_from` (curriculum).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bmem.data import IGNORE, MQARBatch
from bmem.metrics import bootstrap_ci, gate_stats, recall_per_example
from bmem.model import RecallModel


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 64
    lr: float = 3e-4
    min_lr: float = 0.0  # cosine floor; 0 reproduces the original schedule
    weight_decay: float = 0.1
    warmup: int = 100
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    ckpt_every: int | None = None  # default: eval_every
    init_from: str | None = None  # path to a state_dict / checkpoint; weights only
    bf16: bool = True
    seed: int = 0


LOG_EVERY = 50


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * p))


def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), ignore_index=IGNORE)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@torch.no_grad()
def evaluate(model: RecallModel, split: MQARBatch, device, batch_size: int = 128, max_batches: int | None = None,
             bf16=True, gates: bool = False) -> dict:
    """Exact recall over (a prefix of) a split. With ``gates=True`` the first batch is also run
    with gate collection and summarised by :func:`bmem.metrics.gate_stats`."""
    model.eval()
    per_ex, t0, n_tok, stats = [], time.perf_counter(), 0, None
    for i in range(0, len(split), batch_size):
        if max_batches is not None and i // batch_size >= max_batches:
            break
        x = split.inputs[i : i + batch_size].to(device)
        y = split.labels[i : i + batch_size].to(device)
        want_gates = gates and stats is None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits, info = model(x, return_gates=want_gates)
        if want_gates:
            stats = gate_stats(info["gates"], x, y, info["mem"].filled, split.config)
        per_ex.append(recall_per_example(logits, y).cpu())
        n_tok += x.numel()
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_ex = torch.cat(per_ex).numpy()
    lo, hi = bootstrap_ci(per_ex)
    model.train()
    out = {
        "recall": float(per_ex.mean()),
        "recall_ci95": [lo, hi],
        "n_examples": int(len(per_ex)),
        "eval_tokens_per_s": n_tok / (time.perf_counter() - t0),
        "per_example": per_ex,
    }
    if stats is not None:
        out["gates"] = stats
    return out


def load_weights(model: nn.Module, path: str | Path, log=print) -> dict:
    """Copy every tensor whose name and shape match; report the rest. Accepts a bare
    state_dict (``model.pt``/``best.pt``) or a full checkpoint (``ckpt.pt``)."""
    obj = torch.load(path, map_location="cpu")
    src = obj["model"] if isinstance(obj, dict) and "model" in obj and "step" in obj else obj
    own = model.state_dict()
    loaded, skipped, missing = [], [], [k for k in own if k not in src]
    for k, v in src.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            loaded.append(k)
        else:
            skipped.append((k, tuple(v.shape), tuple(own[k].shape) if k in own else None))
    report = {"loaded": len(loaded), "skipped": skipped, "missing": missing}
    log(f"init_from {path}: loaded {len(loaded)} tensors; skipped {skipped or 'none'}; missing {missing or 'none'}")
    return report


def _fmt_gates(g: dict | None) -> str:
    if not g:
        return ""
    s = f" write_frac {g['write_frac']:.3f} slot_entropy {g['slot_entropy']:.2f} filled {g['filled_frac']:.2f}"
    if g.get("write_by_class"):
        c = g["write_by_class"]
        s += " by_class q/d/u/o " + "/".join(f"{c[k]:.2f}" for k in ("queried", "distractor", "update", "other"))
    return s


def train(model: RecallModel, train_split: MQARBatch, val_split: MQARBatch, cfg: TrainConfig, device, log=print,
          out_dir: str | Path | None = None, resume: bool = False,
          on_eval: Callable[[dict], None] | None = None) -> dict:
    """Train ``model``; returns history and cost accounting. ``on_eval(partial_stats)`` is
    called after every eval with the same dict shape as the return value."""
    torch.manual_seed(cfg.seed)
    model.to(device).train()
    if cfg.init_from:
        load_weights(model, cfg.init_from, log)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )
    N = len(train_split)
    rng = np.random.default_rng(cfg.seed)
    out = Path(out_dir) if out_dir is not None else None
    ckpt_every = cfg.ckpt_every or cfg.eval_every
    history, tokens, start_step, prior_elapsed, best = [], 0, 0, 0.0, -1.0

    if resume:
        ck_path = out / "ckpt.pt" if out is not None else None
        if ck_path is None or not ck_path.exists():
            raise FileNotFoundError(f"resume requested but no checkpoint at {ck_path}")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"], device)
        history, tokens, start_step = ck["history"], ck["tokens"], ck["step"]
        prior_elapsed, best = ck["elapsed"], ck["best"]
        log(f"{_now()} resumed from {ck_path} at step {start_step}")

    def save_ckpt(step: int, elapsed: float) -> None:
        if out is None:
            return
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(), "step": step, "history": history,
            "tokens": tokens, "elapsed": elapsed, "best": best,
            "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, out / "ckpt.pt.tmp")
        (out / "ckpt.pt.tmp").replace(out / "ckpt.pt")  # atomic: a kill mid-save leaves the old file

    def stats(elapsed: float) -> dict:
        return {
            "history": history,
            "train_seconds": elapsed,
            "train_tokens_per_s": tokens / max(elapsed, 1e-9),
            "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter() - prior_elapsed
    t_last, recent = time.perf_counter(), deque(maxlen=LOG_EVERY)
    for step in range(start_step, cfg.steps):
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
        recent.append(loss.item())
        last = step == cfg.steps - 1
        if step % LOG_EVERY == 0 or last:
            now = time.perf_counter()
            n_since = (step - start_step) % LOG_EVERY or LOG_EVERY
            s_per_step = (now - t_last) / n_since
            t_last = now
            log(f"{_now()} step {step:5d} loss {np.mean(recent):.4f} gnorm {gn.item():.2f} "
                f"lr {_lr_at(step, cfg):.2e} s/step {s_per_step:.3f}")
        if (step + 1) % cfg.eval_every == 0 or last:
            ev = evaluate(model, val_split, device, max_batches=cfg.eval_batches, bf16=cfg.bf16, gates=True)
            elapsed = time.perf_counter() - t0
            g = ev.get("gates", {})
            history.append({
                "step": step + 1, "loss": float(np.mean(recent)), "gnorm": gn.item(), "lr": _lr_at(step, cfg),
                "val_recall": ev["recall"], "val_ci95": ev["recall_ci95"], "elapsed_s": elapsed,
                "write_frac": g.get("write_frac"), "slot_entropy": g.get("slot_entropy"),
                "filled_frac": g.get("filled_frac"), "write_by_class": g.get("write_by_class"),
            })
            log(f"{_now()}   eval step {step+1}: val recall {ev['recall']:.4f} ci "
                f"[{ev['recall_ci95'][0]:.4f}, {ev['recall_ci95'][1]:.4f}]{_fmt_gates(g)} elapsed {elapsed/60:.1f} min")
            if ev["recall"] > best and out is not None:
                best = ev["recall"]
                torch.save(model.state_dict(), out / "best.pt")
            if on_eval is not None:
                on_eval(stats(elapsed))
        if (step + 1) % ckpt_every == 0 and not last:
            save_ckpt(step + 1, time.perf_counter() - t0)
    elapsed = time.perf_counter() - t0
    save_ckpt(cfg.steps, elapsed)
    return stats(elapsed)
