"""Generate the PLAN step 6 sweep job list for jobq.py (6 writers x S{32,64,128} x N{64,128} x 3 seeds).

    python experiments/sweep.py --n 64 > runs/sweep/jobs_n64.txt
    python experiments/jobq.py runs/sweep/jobs_n64.txt --max 3 --free-gb 8 > runs/sweep/queue_n64.log

Each writer gets its frozen hyperparameters from experiments/configs/selected.yaml (LOG "S5").
N=64 uses target.yaml (length 512); N=128 uses target_n128.yaml (length 1024). Rerunning
the generator after a crash is safe: runs whose results.json says "done" are skipped, runs
with a ckpt.pt get --resume, everything else starts fresh. Order: S=64 first (the column that
decides H1/H2), then 128, then 32; within a column, writers interleaved by seed.
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WRITERS = ["controller", "no_interference", "pair_recency", "residual", "recency", "position"]
SLOTS = [64, 128, 32]
CONFIG = {64: "experiments/configs/target.yaml", 128: "experiments/configs/target_n128.yaml"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[64, 128], help="loads to include")
    ap.add_argument("--slots", type=int, nargs="+", default=SLOTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--writers", nargs="+", default=WRITERS)
    ap.add_argument("--out", default="runs/sweep")
    a = ap.parse_args()
    selected = yaml.safe_load((ROOT / "experiments/configs/selected.yaml").read_text())
    n_jobs = n_skip = 0
    for n in a.n:
        for s in a.slots:
            for seed in a.seeds:
                for w in a.writers:
                    out = Path(a.out) / f"{w}_S{s}_N{n}_s{seed}"
                    res = ROOT / out / "results.json"
                    if res.exists() and json.loads(res.read_text()).get("status") == "done":
                        n_skip += 1
                        continue
                    sets = " ".join(f"--set {k}={v}" for k, v in selected[w].items())
                    resume = " --resume" if (ROOT / out / "ckpt.pt").exists() else ""
                    print(f"python experiments/run.py --config {CONFIG[n]} --writer {w} --seed {seed} "
                          f"--out_dir {out} --set model.fast=true --set model.slots={s} {sets}{resume} "
                          f"--tag \"sweep S={s} N={n}\"")
                    n_jobs += 1
    import sys
    print(f"# {n_jobs} jobs, {n_skip} already done", file=sys.stderr)


if __name__ == "__main__":
    main()
