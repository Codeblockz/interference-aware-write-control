"""GPU smoke tests: forward/backward shapes, gradient reach, and an overfit check."""

import pytest
import torch

from bmem.data import MQARConfig, generate_mqar
from bmem.model import ModelConfig, RecallModel
from bmem.train import loss_fn

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _tiny(writer="controller"):
    return RecallModel(ModelConfig(vocab_size=512, d_model=64, n_layers=1, n_heads=2, slots=4, mem_dim=32, writer=writer))


@cuda
@pytest.mark.parametrize("writer", ["controller", "no_interference", "recency", "residual", "position"])
def test_forward_backward_all_writers(writer):
    m = _tiny(writer).cuda()
    x = torch.randint(1, 512, (2, 32), device="cuda")
    logits, info = m(x, return_gates=True)
    assert logits.shape == (2, 32, 512)
    assert info["gates"].shape == (2, 32, 4)
    logits.float().sum().backward()
    assert m.to_k.weight.grad is not None  # write path receives gradient
    assert m.to_q.weight.grad is not None  # read path receives gradient


@cuda
def test_overfit_small_batch():
    torch.manual_seed(0)
    cfg = MQARConfig(vocab_size=256, seq_len=32, num_kv_pairs=4, random_non_queries=False)
    b = generate_mqar(cfg, 64, seed=0)
    m = RecallModel(ModelConfig(vocab_size=256, d_model=128, n_layers=2, n_heads=2, slots=8, mem_dim=64)).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    x, y = b.inputs.cuda(), b.labels.cuda()
    for step in range(300):
        logits, _ = m(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"did not overfit: loss {loss.item():.3f}"
