"""Multi-query associative recall (MQAR) with key updates, distractors, and delay control.

Adapted from HazyResearch/zoology (Apache-2.0):
  zoology/data/multiquery_ar.py and zoology/data/forgetting_mqar.py
  Arora, Eyuboglu, et al. "Zoology: Measuring and Improving Recall in Efficient
  Language Models." arXiv:2312.04927.
See NOTICE for the license text.

Sequence layout (tokens):

    [ context block ] [ update block ] [ query region ]

* context block: ``num_kv_pairs`` queried pairs plus ``num_distractors`` pairs whose keys
  are never queried, interleaved in random order. Each pair is two tokens, key then value.
* update block: ``num_updates`` of the queried keys reappear with a new value. The answer
  for an updated key is its latest value.
* query region: each queried key appears once more; the label at that position is its
  final value. Query positions are drawn with a power-law gap distribution (Zoology's
  ``power_a``) shifted by ``min_delay`` tokens, so delay is controllable independently.

Vocabulary: token 0 is reserved for padding/filler, keys are 1..vocab_size//2 - 1, values
are vocab_size//2..vocab_size - 1. Labels are -100 everywhere except answer positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import torch

IGNORE = -100


@dataclass
class MQARConfig:
    vocab_size: int = 8192
    seq_len: int = 256
    num_kv_pairs: int = 64
    num_updates: int = 0
    num_distractors: int = 0
    power_a: float = 0.01
    key_skew: float = 0.0  # > 0: queried keys ~ rank^-key_skew over the key pool (Zipf); distractor
    #                         keys uniform over the remaining keys, i.e. mostly cold. 0 = uniform (default).
    min_delay: int = 0
    random_non_queries: bool = True

    def context_tokens(self) -> int:
        return 2 * (self.num_kv_pairs + self.num_distractors + self.num_updates)

    def validate(self) -> None:
        assert self.seq_len % 2 == 0, "seq_len must be even"
        assert self.vocab_size > self.seq_len, "vocab_size must exceed seq_len"
        assert 0 <= self.num_updates <= self.num_kv_pairs
        key_vocab = self.vocab_size // 2 - 1
        assert self.num_kv_pairs + self.num_distractors <= key_vocab
        query_space = (self.seq_len - self.context_tokens()) // 2 - self.min_delay // 2
        assert query_space >= self.num_kv_pairs, (
            f"seq_len={self.seq_len} too short for {self.num_kv_pairs} queries after a "
            f"{self.context_tokens()}-token context and min_delay={self.min_delay}"
        )


@dataclass
class MQARBatch:
    inputs: torch.Tensor  # (N, L) int64
    labels: torch.Tensor  # (N, L) int64, IGNORE except at answer positions
    config: MQARConfig
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.inputs.shape[0]


def _choice_rows(rng: np.random.Generator, pool: np.ndarray, n_rows: int, k: int) -> np.ndarray:
    """k distinct draws from ``pool`` for each of n_rows rows, vectorized."""
    idx = np.argsort(rng.random((n_rows, pool.shape[0])), axis=1)[:, :k]
    return pool[idx]


def _skewed_keys(rng: np.random.Generator, pool: np.ndarray, n_rows: int, k: int, d: int, skew: float) -> np.ndarray:
    """k queried keys per row drawn without replacement with p ~ rank^-skew (Gumbel-top-k, exact
    Plackett-Luce sampling), then d distractor keys uniform over the keys not drawn. Returns
    (n_rows, k + d) with the queried keys first, as _choice_rows does."""
    logp = -skew * np.log(np.arange(1, pool.shape[0] + 1, dtype=np.float64))
    g = rng.gumbel(size=(n_rows, pool.shape[0]))
    order = np.argsort(-(logp[None, :] + g), axis=1)
    queried = order[:, :k]
    u = rng.random((n_rows, pool.shape[0]))
    np.put_along_axis(u, queried, np.inf, axis=1)  # never pick a queried key as a distractor
    distract = np.argsort(u, axis=1)[:, :d]
    return pool[np.concatenate([queried, distract], axis=1)]


def generate_mqar(cfg: MQARConfig, num_examples: int, seed: int) -> MQARBatch:
    cfg.validate()
    rng = np.random.default_rng(seed)
    N, L = num_examples, cfg.seq_len
    key_vocab_size = cfg.vocab_size // 2
    key_pool = np.arange(1, key_vocab_size)
    value_pool = np.arange(key_vocab_size, cfg.vocab_size)

    n_ctx_pairs = cfg.num_kv_pairs + cfg.num_distractors
    if cfg.key_skew > 0:
        all_keys = _skewed_keys(rng, key_pool, N, cfg.num_kv_pairs, cfg.num_distractors, cfg.key_skew)
    else:
        all_keys = _choice_rows(rng, key_pool, N, n_ctx_pairs)
    all_values = _choice_rows(rng, value_pool, N, n_ctx_pairs)
    keys, values = all_keys[:, : cfg.num_kv_pairs], all_values[:, : cfg.num_kv_pairs]

    # Context block: queried + distractor pairs in random order.
    order = np.argsort(rng.random((N, n_ctx_pairs)), axis=1)
    ctx_keys = np.take_along_axis(all_keys, order, axis=1)
    ctx_values = np.take_along_axis(all_values, order, axis=1)
    context = np.empty((N, 2 * n_ctx_pairs), dtype=np.int64)
    context[:, 0::2] = ctx_keys
    context[:, 1::2] = ctx_values

    # Update block: some queried keys get a fresh value, different from the original.
    final_values = values.copy()
    update_block = np.empty((N, 2 * cfg.num_updates), dtype=np.int64)
    if cfg.num_updates > 0:
        upd_idx = np.argsort(rng.random((N, cfg.num_kv_pairs)), axis=1)[:, : cfg.num_updates]
        upd_keys = np.take_along_axis(keys, upd_idx, axis=1)
        old = np.take_along_axis(values, upd_idx, axis=1)
        new = rng.choice(value_pool, size=old.shape)
        clash = new == old
        while clash.any():  # rejection-resample the few collisions
            new[clash] = rng.choice(value_pool, size=int(clash.sum()))
            clash = new == old
        np.put_along_axis(final_values, upd_idx, new, axis=1)
        update_block[:, 0::2] = upd_keys
        update_block[:, 1::2] = new

    context_size = cfg.context_tokens()

    # Query region: power-law gaps as in Zoology, shifted right by min_delay.
    region = L - context_size
    space = region // 2 - cfg.min_delay // 2
    p = cfg.power_a * np.arange(1, space + 1) ** (cfg.power_a - 1)
    p = p / p.sum()
    gaps = np.stack(
        [rng.choice(space, size=cfg.num_kv_pairs, replace=False, p=p) for _ in range(N)]
    )
    query_pos = 2 * gaps + (cfg.min_delay // 2) * 2  # even offsets inside the region

    queries = np.zeros((N, region + 1), dtype=np.int64)
    np.put_along_axis(queries, query_pos, keys, axis=1)
    examples = np.concatenate([context, update_block, queries], axis=1)

    labels = np.full((N, L + 1), IGNORE, dtype=np.int64)
    np.put_along_axis(labels, query_pos + context_size + 1, final_values, axis=1)

    inputs = torch.from_numpy(examples[:, :-1].copy())
    labels_t = torch.from_numpy(labels[:, 1:].copy())

    if cfg.random_non_queries:
        filler = inputs == 0
        inputs[filler] = torch.randint(1, cfg.vocab_size, size=inputs.shape)[filler]

    meta = {
        "context_tokens": context_size,
        "num_queries_per_example": cfg.num_kv_pairs,
        "seed": seed,
    }
    return MQARBatch(inputs, labels_t, cfg, meta)


def make_splits(
    cfg: MQARConfig,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
) -> dict[str, MQARBatch]:
    """Three splits from distinct seeds with exact-duplicate sequences removed from
    val/test. With vocab 8192 and 64 random pairs, duplicates essentially never occur,
    but the check makes the disjointness claim in the paper verifiable."""
    train = generate_mqar(cfg, n_train, seed)
    val = generate_mqar(cfg, n_val, seed + 1_000_003)
    test = generate_mqar(cfg, n_test, seed + 2_000_003)
    seen = {bytes(row.numpy()) for row in train.inputs}
    for name, split in (("val", val), ("test", test)):
        keep = torch.tensor([bytes(r.numpy()) not in seen for r in split.inputs])
        dropped = int((~keep).sum())
        if dropped:
            split.inputs, split.labels = split.inputs[keep], split.labels[keep]
        split.meta["duplicates_dropped"] = dropped
        seen.update(bytes(r.numpy()) for r in split.inputs)
    return {"train": train, "val": val, "test": test}


def config_to_dict(cfg: MQARConfig) -> dict:
    return asdict(cfg)
