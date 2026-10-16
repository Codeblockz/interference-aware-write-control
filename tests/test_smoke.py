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


@cuda
def test_shift_aligns_key_and_query():
    """With assoc_shift=1 the stored key comes from the key token's position."""
    m = _tiny("recency").cuda()
    m.cfg.assoc_shift = 1
    x = torch.randint(1, 512, (1, 8), device="cuda")
    h = m.backbone(x)
    k_shift = m.to_k(torch.cat([h.new_zeros(1, 1, h.shape[-1]), h[:, :-1]], 1))
    assert torch.allclose(k_shift[0, 3], m.to_k(h)[0, 2], atol=1e-5)


@cuda
@pytest.mark.slow
def test_controller_learns_small_task():
    """Regression for the two learnability bugs (key alignment, diffuse gates)."""
    import numpy as np
    from bmem.data import make_splits
    from bmem.metrics import recall_per_example

    torch.manual_seed(0)
    cfg = MQARConfig(vocab_size=512, seq_len=64, num_kv_pairs=8)
    s = make_splits(cfg, 20000, 512, 1, seed=0)
    m = RecallModel(ModelConfig(vocab_size=512, d_model=128, n_layers=2, n_heads=2, slots=16, mem_dim=64)).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(0)
    for step in range(800):
        idx = torch.from_numpy(rng.integers(0, 20000, 64))
        x, y = s["train"].inputs[idx].cuda(), s["train"].labels[idx].cuda()
        for g in opt.param_groups:
            g["lr"] = 1e-3 * min(1, (step + 1) / 50)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(m(x)[0], y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        r = recall_per_example(m(s["val"].inputs.cuda())[0], s["val"].labels.cuda()).mean().item()
    assert r > 0.5, f"controller recall {r:.3f} on the small task; expected > 0.5"
