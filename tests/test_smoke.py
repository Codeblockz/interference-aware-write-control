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
@pytest.mark.parametrize("writer", ["controller", "no_interference", "input_only", "recency", "residual", "position"])
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


@cuda
@pytest.mark.parametrize("writer", ["controller", "no_interference", "input_only", "recency", "pair_recency", "residual", "position"])
def test_fast_path_matches_eager(writer):
    """model.fast (compiled functional memory step) must give the same logits, gates and gradients."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=512, d_model=64, n_layers=1, n_heads=2, slots=4, mem_dim=32, writer=writer, pair_block=12)
    m = RecallModel(cfg).cuda()
    x = torch.randint(1, 512, (3, 24), device="cuda")
    logits_e, info_e = m(x, return_gates=True)
    logits_e.float().sum().backward()
    grads_e = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    m.cfg.fast = True
    logits_f, info_f = m(x, return_gates=True)
    logits_f.float().sum().backward()
    assert torch.allclose(logits_e, logits_f, atol=1e-4, rtol=1e-4)
    assert torch.allclose(info_e["gates"], info_f["gates"], atol=1e-5)
    assert torch.allclose(info_e["mem"].keys, info_f["mem"].keys, atol=1e-5)
    for n, g in grads_e.items():
        assert torch.allclose(g, dict(m.named_parameters())[n].grad, atol=1e-3, rtol=1e-3), n


@cuda
def test_position_unlocked_mask_matches_int_rule():
    from bmem.controller import PositionWriter
    T, S = 37, 8
    m = PositionWriter.unlocked_mask(T, S, "cuda").cpu()
    for t in range(T):
        unlocked = int(min(S, max(1, (t + 1) * S // T + 1)))
        assert m[t].sum().item() == unlocked and bool(m[t, :unlocked].all()) and not bool(m[t, unlocked:].any())


@cuda
def test_write_bias_init_opens_the_gate():
    """write_bias_init sets the controller's initial write probability; default is unchanged."""
    torch.manual_seed(0)
    x = torch.randint(1, 512, (4, 32), device="cuda")
    base = ModelConfig(vocab_size=512, d_model=64, n_layers=1, n_heads=2, slots=4, mem_dim=32)
    m0 = RecallModel(base).cuda()
    m1 = RecallModel(ModelConfig(**{**base.__dict__, "write_bias_init": 3.0})).cuda()
    m1.load_state_dict({k: v for k, v in m0.state_dict().items() if "write_mlp.2.bias" not in k}, strict=False)
    frac0 = m0(x, return_gates=True)[1]["gates"].sum(-1).mean().item()
    frac1 = m1(x, return_gates=True)[1]["gates"].sum(-1).mean().item()
    assert frac1 > frac0 + 0.2 and frac1 > 0.8


@cuda
def test_hard_gates_flag_switches_to_soft():
    torch.manual_seed(0)
    x = torch.randint(1, 512, (2, 16), device="cuda")
    base = dict(vocab_size=512, d_model=64, n_layers=1, n_heads=2, slots=4, mem_dim=32)
    hard = RecallModel(ModelConfig(**base)).cuda()
    soft = RecallModel(ModelConfig(**base, hard_gates=False)).cuda()
    soft.load_state_dict(hard.state_dict())
    gh = hard(x, return_gates=True)[1]["gates"]
    gs = soft(x, return_gates=True)[1]["gates"]
    assert hard.writer.hard and not soft.writer.hard
    assert ((gh > 0).sum(-1) <= 1).all()  # one slot per step (times the scalar write gate)
    assert ((gs > 1e-6).sum(-1) > 1).any()  # soft spreads mass over slots


@cuda
def test_pair_recency_writes_only_at_pair_values():
    cfg = MQARConfig(vocab_size=256, seq_len=32, num_kv_pairs=4, num_updates=1, num_distractors=1)
    b = generate_mqar(cfg, 3, seed=0)
    m = RecallModel(ModelConfig(vocab_size=256, d_model=32, n_layers=1, n_heads=2, slots=4, mem_dim=16,
                                writer="pair_recency", pair_block=cfg.context_tokens())).cuda()
    g = m(b.inputs.cuda(), return_gates=True)[1]["gates"]  # (B, T, S)
    mass = g.sum(-1)
    t = torch.arange(32, device="cuda")
    expect = ((t % 2 == 1) & (t < cfg.context_tokens())).float()
    assert torch.equal(mass[0], expect) and torch.equal(mass[2], expect)


def test_input_only_sees_nothing_from_memory():
    """The S8 ablation: with novelty=False the per-slot features are independent of the stored
    keys and values; the default (novelty=True) keeps the existing controller / ablation
    behaviour, so the sweep's writers are unchanged by the flag."""
    import torch
    from bmem.controller import InterferenceController, WRITERS
    from bmem.memory import SlotMemory
    torch.manual_seed(0)
    B, S, d = 2, 4, 8
    key, value = torch.randn(B, d), torch.randn(B, d)
    def mem_with(seed):
        torch.manual_seed(seed)
        m = SlotMemory(B, S, d, "cpu")
        m.keys, m.values, m.filled = torch.randn(B, S, d), torch.randn(B, S, d), torch.ones(B, S)
        return m
    w = WRITERS["input_only"]()
    assert w.interference is False and w.novelty is False
    phi1, pooled1 = w.features(mem_with(1), key, value, 16)
    phi2, pooled2 = w.features(mem_with(2), key, value, 16)
    assert torch.equal(phi1, phi2) and torch.equal(pooled1, pooled2)  # different stored content, same features
    # default flag: the ablation still sees novelty, so its features DO depend on stored keys
    a = WRITERS["no_interference"]()
    assert a.novelty is True
    phiA, _ = a.features(mem_with(1), key, value, 16)
    phiB, _ = a.features(mem_with(2), key, value, 16)
    assert not torch.equal(phiA, phiB)
    # and the flag default leaves the controller's feature values untouched except column 2 when disabled
    c = InterferenceController(novelty=False)
    phiC, _ = c.features(mem_with(1), key, value, 16)
    phiD, _ = InterferenceController().features(mem_with(1), key, value, 16)
    assert torch.equal(phiC[..., [0, 1, 3, 4, 5]], phiD[..., [0, 1, 3, 4, 5]]) and torch.all(phiC[..., 2] == 0)
