import torch

from bmem.controller import (
    InterferenceController,
    PositionWriter,
    RecencyWriter,
    ResidualCacheWriter,
)
from bmem.memory import SlotMemory


def _mem(B=3, S=4, d=8):
    return SlotMemory(B, S, d, device="cpu")


def test_zero_gate_is_identity():
    m = _mem()
    m.write(torch.ones(3, 4), torch.randn(3, 8), torch.randn(3, 8))
    k0, v0 = m.keys.clone(), m.values.clone()
    m.write(torch.zeros(3, 4), torch.randn(3, 8), torch.randn(3, 8))
    assert torch.equal(m.keys, k0) and torch.equal(m.values, v0)


def test_full_gate_overwrites_and_resets_age():
    m = _mem()
    m.write(torch.ones(3, 4), torch.randn(3, 8), torch.randn(3, 8))
    m.write(torch.zeros(3, 4), torch.randn(3, 8), torch.randn(3, 8))
    assert (m.age == 1).all()
    k, v = torch.randn(3, 8), torch.randn(3, 8)
    g = torch.zeros(3, 4)
    g[:, 2] = 1
    m.write(g, k, v)
    assert torch.allclose(m.keys[:, 2], k) and torch.allclose(m.values[:, 2], v)
    assert (m.age[:, 2] == 0).all() and (m.age[:, 0] == 2).all()


def test_read_returns_matching_value():
    m = _mem(B=1, S=4, d=16)
    keys = torch.eye(16)[:4].unsqueeze(0) * 10  # orthogonal, well separated
    vals = torch.randn(1, 4, 16)
    for j in range(4):
        g = torch.zeros(1, 4)
        g[:, j] = 1
        m.write(g, keys[:, j], vals[:, j])
    r = m.read(keys[:, 1])
    assert torch.allclose(r, vals[:, 1], atol=1e-3)


def test_recency_writer_round_robin():
    m = _mem(B=2, S=3, d=8)
    w = RecencyWriter()
    order = []
    for t in range(6):
        g = w(m, torch.randn(2, 8), torch.randn(2, 8), t, 6)
        order.append(g[0].argmax().item())
        m.write(g, torch.randn(2, 8), torch.randn(2, 8))
    assert len(set(order[:3])) == 3  # fills every slot once
    assert order[3:] == order[:3]  # then repeats in the same order


def test_position_writer_unlocks_progressively():
    m = _mem(B=1, S=4, d=8)
    w = PositionWriter()
    g0 = w(m, torch.randn(1, 8), torch.randn(1, 8), 0, 16)
    assert g0[0, 1:].sum() == 0  # only slot 0 at the start


def test_controller_gates_valid_and_differentiable():
    m = _mem(B=2, S=5, d=8)
    m.write(torch.ones(2, 5) * 0.5, torch.randn(2, 8), torch.randn(2, 8))
    c = InterferenceController(hidden=16)
    k, v = torch.randn(2, 8, requires_grad=True), torch.randn(2, 8)
    g = c(m, k, v, 3, 10)
    assert g.shape == (2, 5) and (g >= 0).all() and (g <= 1).all()
    assert (g.sum(-1) <= 1 + 1e-5).all()
    g.sum().backward()
    assert all(p.grad is not None for p in c.parameters())
    assert k.grad is not None


def test_ablation_zeros_interference_features():
    m = _mem(B=2, S=5, d=8)
    m.write(torch.ones(2, 5), torch.randn(2, 8), torch.randn(2, 8))
    c = InterferenceController(hidden=16, interference=False)
    phi, _ = c.features(m, torch.randn(2, 8), torch.randn(2, 8), 10)
    assert (phi[..., 0] == 0).all() and (phi[..., 1] == 0).all()
    assert (phi[..., 2] != 0).any()  # novelty still present


def test_residual_writer_writes_more_when_readout_wrong():
    torch.manual_seed(0)
    m = _mem(B=1, S=4, d=16)
    w = ResidualCacheWriter(init_threshold=0.5)
    k, v = torch.randn(1, 16), torch.randn(1, 16)
    g_empty = w(m, k, v, 0, 8).sum()
    m.write(torch.tensor([[1.0, 0, 0, 0]]), k, v)
    g_known = w(m, k, v, 1, 8).sum()
    assert g_empty > g_known


def test_controller_hard_gate_is_one_hot_times_scalar():
    m = _mem(B=2, S=5, d=8)
    m.write(torch.ones(2, 5) * 0.5, torch.randn(2, 8), torch.randn(2, 8))
    c = InterferenceController(hidden=16, hard=True)
    g = c(m, torch.randn(2, 8), torch.randn(2, 8), 3, 10)
    assert ((g > 0).sum(-1) == 1).all()  # exactly one slot written
    assert torch.allclose(g.max(-1).values, g.sum(-1))
