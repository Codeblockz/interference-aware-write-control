"""Run one experiment: python experiments/run.py --config experiments/configs/pilot.yaml \
       --writer controller --seed 0 --out_dir runs/pilot/controller_s0

Writes out_dir/results.json (metrics, cost accounting, config) and out_dir/log.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmem.data import MQARConfig, make_splits  # noqa: E402
from bmem.model import ModelConfig, RecallModel  # noqa: E402
from bmem.train import TrainConfig, evaluate, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--writer", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "log.txt").open("w")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    data_cfg = MQARConfig(**cfg["data"])
    splits = make_splits(data_cfg, cfg["n_train"], cfg["n_val"], cfg["n_test"], seed=args.seed)
    log(f"data: {data_cfg} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    mcfg = ModelConfig(**{**cfg["model"], "vocab_size": data_cfg.vocab_size})
    if args.writer:
        mcfg.writer = args.writer
    model = RecallModel(mcfg)
    log(f"model: {mcfg}\nparams: {model.n_params()}\nstate_bytes: {model.state_bytes()}")

    tcfg = TrainConfig(**{**cfg["train"], "seed": args.seed})
    train_stats = train(model, splits["train"], splits["val"], tcfg, device, log=log)

    results = {
        "writer": mcfg.writer, "seed": args.seed, "config": cfg,
        "params": model.n_params(), "state_bytes": model.state_bytes(),
        **{k: v for k, v in train_stats.items()},
        "eval": {},
    }
    # Test at the training length and at each extra evaluation length.
    eval_lengths = [data_cfg.seq_len] + list(cfg.get("eval_lengths", []))
    for L in eval_lengths:
        ecfg = MQARConfig(**{**cfg["data"], "seq_len": L})
        test = splits["test"] if L == data_cfg.seq_len else make_splits(ecfg, 1, 1, cfg["n_test"], seed=args.seed)["test"]
        t0 = time.perf_counter()
        ev = evaluate(model, test, device, bf16=tcfg.bf16)
        per_ex = ev.pop("per_example")
        np.save(out / f"per_example_L{L}.npy", per_ex)
        ev["wall_s"] = time.perf_counter() - t0
        results["eval"][f"L{L}"] = ev
        log(f"test L={L}: recall {ev['recall']:.4f} ci95 {ev['recall_ci95']} ({ev['n_examples']} ex)")

    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    torch.save(model.state_dict(), out / "model.pt")
    log(f"wrote {out/'results.json'}")


if __name__ == "__main__":
    main()
