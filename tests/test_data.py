import numpy as np
import torch

from bmem.data import IGNORE, MQARConfig, generate_mqar, make_splits


def _small(**kw):
    base = dict(vocab_size=200, seq_len=64, num_kv_pairs=8, random_non_queries=False)
    base.update(kw)
    return MQARConfig(**base)


def _answers(batch):
    """Yield (example idx, query key token, label value) for every labelled position."""
    pos = (batch.labels != IGNORE).nonzero(as_tuple=False)
    for i, t in pos.tolist():
        yield i, batch.inputs[i, t].item(), batch.labels[i, t].item()


def test_shapes_and_label_count():
    b = generate_mqar(_small(), 16, seed=0)
    assert b.inputs.shape == (16, 64) and b.labels.shape == (16, 64)
    assert (b.labels != IGNORE).sum(dim=1).eq(8).all()


def test_labels_match_context_without_updates():
    b = generate_mqar(_small(), 32, seed=1)
    for i, key, val in _answers(b):
        ctx = b.inputs[i, : b.config.context_tokens()].view(-1, 2).tolist()
        assert [k for k, _ in ctx].count(key) == 1
        assert dict(ctx)[key] == val


def test_updates_return_latest_value():
    cfg = _small(num_updates=4, seq_len=80)
    b = generate_mqar(cfg, 32, seed=2)
    n_updated_answers = 0
    for i, key, val in _answers(b):
        ctx = b.inputs[i, : cfg.context_tokens()].view(-1, 2).tolist()
        occurrences = [v for k, v in ctx if k == key]
        assert val == occurrences[-1]
        if len(occurrences) == 2:
            assert occurrences[0] != occurrences[1]
            n_updated_answers += 1
    assert n_updated_answers == 32 * 4


def test_distractor_keys_never_queried():
    cfg = _small(num_distractors=6, seq_len=96)
    b = generate_mqar(cfg, 16, seed=3)
    for i in range(16):
        ctx_keys = set(b.inputs[i, : cfg.context_tokens() : 2].tolist())
        assert len(ctx_keys) == 14
        queried = {k for j, k, _ in _answers(b) if j == i}
        assert len(queried) == 8 and queried <= ctx_keys


def test_min_delay_respected():
    cfg = _small(min_delay=20, seq_len=128)
    b = generate_mqar(cfg, 32, seed=4)
    first_answer = (b.labels != IGNORE).float().argmax(dim=1)
    assert (first_answer >= cfg.context_tokens() + cfg.min_delay).all()


def test_seed_determinism_and_difference():
    cfg = _small()
    a, b, c = (generate_mqar(cfg, 8, seed=s) for s in (5, 5, 6))
    assert torch.equal(a.inputs, b.inputs) and torch.equal(a.labels, b.labels)
    assert not torch.equal(a.inputs, c.inputs)


def test_random_filler_keeps_labels():
    cfg = _small(random_non_queries=True)
    b = generate_mqar(cfg, 8, seed=7)
    assert (b.inputs == 0).sum() == 0
    assert (b.labels != IGNORE).sum(dim=1).eq(8).all()


def test_splits_disjoint():
    cfg = MQARConfig(vocab_size=64, seq_len=16, num_kv_pairs=2, random_non_queries=False)
    s = make_splits(cfg, n_train=2000, n_val=300, n_test=300, seed=0)
    train = {bytes(r.numpy()) for r in s["train"].inputs}
    for name in ("val", "test"):
        assert all(bytes(r.numpy()) not in train for r in s[name].inputs)
    assert s["val"].meta["duplicates_dropped"] >= 0


def test_generation_speed():
    import time

    cfg = MQARConfig(seq_len=512, num_kv_pairs=64, num_updates=16, num_distractors=16)
    t0 = time.perf_counter()
    generate_mqar(cfg, 10_000, seed=0)
    assert time.perf_counter() - t0 < 10.0


def test_key_skew_concentrates_queried_keys_and_keeps_distractors_cold():
    import numpy as np
    from bmem.data import MQARConfig, generate_mqar
    base = dict(vocab_size=2048, seq_len=128, num_kv_pairs=16, num_updates=4, num_distractors=4)

    def ranks(batch):
        x = batch.inputs.numpy()
        keys = x[:, 0:40:2]                                   # 20 context keys per row
        answer = batch.labels.numpy() != -100
        q = np.where(answer, x, 0)                            # label sits at the query key position
        hot = np.zeros_like(keys, dtype=bool)
        for i in range(keys.shape[0]):                        # per-row membership
            hot[i] = np.isin(keys[i], q[i][q[i] > 0])
        assert hot.sum(1).tolist() == [16] * keys.shape[0]
        return keys[hot].mean(), keys[~hot].mean()

    flat = generate_mqar(MQARConfig(**base), 200, 0)
    skew = generate_mqar(MQARConfig(**base, key_skew=2.0), 200, 0)
    q_flat, d_flat = ranks(flat)
    q_skew, d_skew = ranks(skew)
    assert abs(q_flat - d_flat) < 60                          # uniform: queried and distractor keys alike
    assert q_skew < q_flat / 3 and d_skew > 400               # skewed: queried keys hot, distractors cold
    same = generate_mqar(MQARConfig(**base, key_skew=0.0), 200, 0)
    # key_skew=0 is the old path bit for bit, except filler tokens, which come from the global
    # torch RNG (pre-existing; see LOG 2026-10-27): compare the context block and the labels.
    assert np.array_equal(flat.inputs.numpy()[:, :40], same.inputs.numpy()[:, :40])
    assert np.array_equal(flat.labels.numpy(), same.labels.numpy())
