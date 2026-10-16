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
