"""Compute cost from results.json files: what a replicator should expect.

    python experiments/analysis/cost.py runs            # all runs
    python experiments/analysis/cost.py runs/step2 --md  # markdown table

Per (writer, code path, config shape): runs, mean s/step, mean train minutes, mean test-eval
seconds per length, peak GPU MB, GPU name. s/step is train_seconds / steps_done, which
includes the periodic validation evals and checkpoint writes (what a wall clock sees).
Runs that shared the GPU with others are slower than a run alone; the log notes how many
ran concurrently. Also prints a projection for the sweep at the measured rates.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main(root: Path, md: bool, sweep_runs: int, sweep_steps: int, concurrency: int):
    groups = defaultdict(list)
    for f in sorted(root.rglob("results.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "done" or not r.get("history") or not r.get("train_seconds"):
            continue  # old-format or eval-only re-runs without timing
        m, d, t = r["config"]["model"], r["config"]["data"], r["config"]["train"]
        key = (r["writer"], "fast" if m.get("fast") else "eager", f"B{t['batch_size']} T{d['seq_len']} S{m['slots']} L{m['n_layers']}")
        steps = r["steps_done"] or r["history"][-1]["step"]
        groups[key].append({
            "s_step": r["train_seconds"] / steps, "train_min": r["train_seconds"] / 60,
            "evals": {k: v.get("wall_s") for k, v in r.get("eval", {}).items()},
            "peak_mb": r.get("peak_mem_mb", 0), "gpu": r.get("gpu", "?"), "steps": steps,
        })
    rows = []
    for (w, path, shape), rs in sorted(groups.items()):
        n = len(rs)
        mean = lambda k: sum(x[k] for x in rs) / n
        evals = defaultdict(list)
        for x in rs:
            for L, s in x["evals"].items():
                if s is not None:
                    evals[L].append(s)
        ev = " ".join(f"{L}:{sum(v)/len(v):.0f}s" for L, v in sorted(evals.items(), key=lambda kv: int(kv[0][1:])))
        rows.append((w, path, shape, n, mean("s_step"), mean("train_min"), int(rs[0]["steps"]), ev, mean("peak_mb"), rs[0]["gpu"]))
    if md:
        print("| writer | path | shape | runs | s/step | train min | steps | test eval wall (10k seq) | peak MB |")
        print("|---|---|---|---|---|---|---|---|---|")
        for w, path, shape, n, ss, tm, st, ev, pk, gpu in rows:
            print(f"| {w} | {path} | {shape} | {n} | {ss:.3f} | {tm:.0f} | {st} | {ev} | {pk:.0f} |")
    else:
        print(f"{'writer':16} {'path':5} {'shape':20} {'n':>2} {'s/step':>7} {'min':>5} {'steps':>5}  test-eval wall       peak MB  gpu")
        for w, path, shape, n, ss, tm, st, ev, pk, gpu in rows:
            print(f"{w:16} {path:5} {shape:20} {n:>2} {ss:7.3f} {tm:5.0f} {st:>5}  {ev:20} {pk:7.0f}  {gpu}")
    fast = [x for k, rs in groups.items() if k[1] == "fast" for x in rs]
    if fast:
        s = sum(x["s_step"] for x in fast) / len(fast)
        total_h = sweep_runs * sweep_steps * s / 3600
        print(f"\nsweep projection: {sweep_runs} runs x {sweep_steps} steps at {s:.3f} s/step (fast-path mean over {len(fast)} runs, "
              f"measured with 2-3 runs sharing the GPU) = {total_h:.1f} GPU-h; at concurrency {concurrency} about "
              f"{total_h/concurrency:.1f} h wall, plus test evals (~2-4 min per run).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--sweep-runs", type=int, default=108, help="6 writers x 3 S x 2 N x 3 seeds")
    ap.add_argument("--sweep-steps", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=3)
    a = ap.parse_args()
    main(Path(a.root), a.md, a.sweep_runs, a.sweep_steps, a.concurrency)
