"""Run one experiment.

    python experiments/run.py --config experiments/configs/target.yaml \
        --writer controller --seed 0 --out_dir runs/step1/p1_vocab2048_controller_s0 \
        --set data.vocab_size=2048 --tag "P1 vocab confound"

Writes to out_dir:
  log.txt          timestamped training log (appended on --resume)
  results.json     effective config, git hash, status (running/done), history, cost, test eval
  ckpt.pt          latest training state (overwritten); resume with --resume
  best.pt          weights at the best validation recall; the test evals use it (--select best,
                   the default; --select last uses the final weights). results.json records
                   selected_step / selected_val_recall.
  model.pt         final weights; per_example_L{L}.npy per test length

--set key=value overrides a config entry by dotted path (data.*, model.*, train.*, or a
top-level key such as n_train). Values are coerced to the field's type; unknown keys error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmem.data import MQARConfig, make_splits  # noqa: E402
from bmem.model import ModelConfig, RecallModel  # noqa: E402
from bmem.train import TrainConfig, evaluate, train  # noqa: E402

SECTIONS = {"data": MQARConfig, "model": ModelConfig, "train": TrainConfig}


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    """Apply ``section.key=value`` / ``key=value`` overrides in place; returns cfg."""
    for item in sets:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        if len(parts) == 2 and parts[0] in SECTIONS:
            section, name = parts
            fields = {f.name: f for f in dataclasses.fields(SECTIONS[section])}
            if name not in fields:
                raise SystemExit(f"--set {key}: {SECTIONS[section].__name__} has no field {name!r}")
            cfg.setdefault(section, {})[name] = _coerce(raw, cfg[section].get(name, fields[name].default), fields[name].type)
        elif len(parts) == 1:
            if key not in cfg:
                raise SystemExit(f"--set {key}: not a top-level config key ({sorted(cfg)})")
            cfg[key] = _coerce(raw, cfg[key], None)
        else:
            raise SystemExit(f"--set {key}: use section.key with section in {sorted(SECTIONS)}, or a top-level key")
    return cfg


def _coerce(raw: str, current, annotation):
    val = yaml.safe_load(raw)
    ann = str(annotation or "")
    if current is None or current is dataclasses.MISSING:
        return val
    if isinstance(current, bool) or "bool" in ann:
        return bool(val) if not isinstance(val, str) else val.lower() in ("1", "true", "yes")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(val)
    if isinstance(current, float):
        return float(val)
    if isinstance(current, list):
        return list(val) if isinstance(val, (list, tuple)) else [val]
    return val


def git_info(root: Path) -> dict:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:  # not a git checkout
        return {"commit": None, "dirty": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--writer", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", default="", help="free text recorded in results.json (what this run probes)")
    ap.add_argument("--resume", action="store_true", help="continue from out_dir/ckpt.pt")
    ap.add_argument("--init_from", default=None, help="state_dict or ckpt to initialise weights from (curriculum)")
    ap.add_argument("--eval_only", action="store_true", help="skip training: load the selected checkpoint and run the test evals")
    ap.add_argument("--select", choices=["best", "last"], default="best",
                    help="which weights the test evals use: best validation recall (best.pt, default) or the last step")
    args = ap.parse_args()

    cfg = apply_overrides(yaml.safe_load(Path(args.config).read_text()), args.set)
    if args.writer:
        cfg["model"]["writer"] = args.writer
    cfg["train"]["seed"] = args.seed
    if args.init_from:
        cfg["train"]["init_from"] = args.init_from
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "log.txt").open("a" if (args.resume or args.eval_only) else "w")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    started = datetime.now().isoformat(timespec="seconds")
    prior = json.loads((out / "results.json").read_text()) if (args.resume or args.eval_only) and (out / "results.json").exists() else {}
    started = prior.get("started_at", started)  # keep the original start time across resumes
    meta = {
        "writer": cfg["model"]["writer"], "seed": args.seed, "tag": args.tag, "config": cfg,
        "config_file": args.config, "overrides": args.set, "git": git_info(Path(__file__).resolve().parents[1]),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "started_at": started, "ended_at": None, "status": "running",
    }
    log(f"run: {json.dumps({k: meta[k] for k in ('tag', 'config_file', 'overrides', 'git', 'gpu', 'started_at')})}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    data_cfg = MQARConfig(**cfg["data"])
    splits = make_splits(data_cfg, cfg["n_train"], cfg["n_val"], cfg["n_test"], seed=args.seed)
    log(f"data: {data_cfg} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    mcfg = ModelConfig(**{**cfg["model"], "vocab_size": data_cfg.vocab_size, "pair_block": data_cfg.context_tokens()})
    cfg["model"]["pair_block"] = data_cfg.context_tokens()
    model = RecallModel(mcfg)
    log(f"model: {mcfg}\nparams: {model.n_params()}\nstate_bytes: {model.state_bytes()}")
    meta.update(params=model.n_params(), state_bytes=model.state_bytes())

    tcfg = TrainConfig(**cfg["train"])

    def write_results(train_stats: dict, extra: dict | None = None) -> dict:
        results = {**meta, **train_stats, "steps_done": train_stats["history"][-1]["step"] if train_stats["history"] else 0,
                   "eval": {}, **(extra or {})}
        tmp = out / "results.json.tmp"
        tmp.write_text(json.dumps(results, indent=2, default=float))
        tmp.replace(out / "results.json")
        return results

    meta["select"] = args.select
    if args.eval_only:
        ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
        train_stats = {k: prior.get(k) for k in ("history", "train_seconds", "train_tokens_per_s", "peak_mem_mb")}
        train_stats["history"] = train_stats["history"] or ck["history"]
        model.load_state_dict(ck["model"])
        log(f"eval_only: loaded {out/'ckpt.pt'} at step {ck['step']}")
    else:
        if not args.resume:  # so progress.py sees the tag and schedule before the first eval
            write_results({"history": [], "train_seconds": 0.0, "train_tokens_per_s": 0.0, "peak_mem_mb": 0.0})
        train_stats = train(model, splits["train"], splits["val"], tcfg, device, log=log,
                            out_dir=out, resume=args.resume, on_eval=write_results)
    hist = train_stats["history"]
    if args.select == "best" and (out / "best.pt").exists() and hist:
        best = max(hist, key=lambda e: e["val_recall"])
        model.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
        meta["selected_step"], meta["selected_val_recall"] = best["step"], best["val_recall"]
        log(f"select=best: test evals use best.pt (step {best['step']}, val recall {best['val_recall']:.4f})")
    else:
        meta["selected_step"] = hist[-1]["step"] if hist else None
        meta["selected_val_recall"] = hist[-1]["val_recall"] if hist else None
    model.to(device)

    # Test at the training length and at each extra evaluation length.
    evals = {}
    eval_lengths = [data_cfg.seq_len] + list(cfg.get("eval_lengths", []))
    for L in eval_lengths:
        ecfg = MQARConfig(**{**cfg["data"], "seq_len": L})
        test = splits["test"] if L == data_cfg.seq_len else make_splits(ecfg, 1, 1, cfg["n_test"], seed=args.seed)["test"]
        t0 = time.perf_counter()
        # Logits are (B, L, vocab); fix the token budget per eval batch (64k tokens = 128 x 512,
        # the budget every run so far used) so long-length evals do not blow the memory.
        ev = evaluate(model, test, device, batch_size=max(8, 65536 // L), bf16=tcfg.bf16)
        per_ex = ev.pop("per_example")
        np.save(out / f"per_example_L{L}.npy", per_ex)
        ev["wall_s"] = time.perf_counter() - t0
        evals[f"L{L}"] = ev
        log(f"test L={L}: recall {ev['recall']:.4f} ci95 {ev['recall_ci95']} ({ev['n_examples']} ex)")

    meta["ended_at"] = datetime.now().isoformat(timespec="seconds")
    meta["status"] = "done"
    write_results(train_stats, {"eval": evals})
    torch.save(model.state_dict(), out / "model.pt")
    log(f"wrote {out/'results.json'}")


if __name__ == "__main__":
    main()
