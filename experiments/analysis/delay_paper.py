"""Delay sweep (LOG S10): table and figure of recall against the minimum query delay at
N = 64, S = 64, for the controller, the input-only ablation and pair-only recency.

    python experiments/analysis/delay_paper.py runs/delay --fig paper/figures/delay.png

The delay-0 points are the sweep / S8 runs (passed with --zero, default runs/sweep and
runs/step4). Prints a Markdown table (per writer x delay: mean test recall at the training
length and at 2048, per-seed values, admission pairs/filler at the best checkpoint) and the
S10 decision rule (delay-sensitive if the delay-192 minus delay-0 three-seed mean differs by
more than 2 pp with all seeds in the same direction).
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot import SERIES, style  # noqa: E402

NAME = re.compile(r"^(?P<w>[a-z_]+)_d(?P<d>\d+)_S64_N64_s(?P<seed>\d+)$")
ZERO = {"controller": "runs/sweep/controller_S64_N64_s{}", "pair_recency": "runs/sweep/pair_recency_S64_N64_s{}",
        "input_only": "runs/step4/s8_input_only_lr3e-4_S64_N64_s{}"}
WRITERS = ["pair_recency", "input_only", "controller"]


def rec(d: Path) -> dict | None:
    f = d / "results.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return r if r.get("status") == "done" else None


def load(root: Path) -> dict:
    runs = defaultdict(dict)  # (w, d) -> seed -> record
    for f in sorted(root.glob("*/results.json")):
        m = NAME.match(f.parent.name)
        r = rec(f.parent)
        if m and r:
            runs[(m["w"], int(m["d"]))][int(m["seed"])] = r
    for w, pat in ZERO.items():
        for seed in range(3):
            r = rec(Path(pat.format(seed)))
            if r:
                runs[(w, 0)][seed] = r
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    runs = load(Path(a.root))
    delays = sorted({d for (_, d) in runs})
    print("| writer | delay | test@512 mean (seeds) | test@2048 | pairs / filler written |")
    print("|---|---|---|---|---|")
    table = {}
    for w in WRITERS:
        for d in delays:
            recs = runs.get((w, d), {})
            if not recs:
                continue
            r512 = [100 * r["eval"]["L512"]["recall"] for r in recs.values()]
            r2048 = [100 * r["eval"]["L2048"]["recall"] for r in recs.values()]
            best = [max(r["history"], key=lambda e: e["val_recall"]) for r in recs.values()]
            pairs = np.mean([b.get("write_by_class", {}).get("queried", np.nan) for b in best])
            fill = np.mean([b.get("write_by_class", {}).get("other", np.nan) for b in best])
            table[(w, d)] = (np.mean(r512), r512, np.mean(r2048))
            print(f"| {w} | {d} | {np.mean(r512):.1f} ({', '.join(f'{x:.1f}' for x in r512)}) | {np.mean(r2048):.1f} | {pairs:.2f} / {fill:.3f} |")
    print()
    for w in WRITERS:
        if (w, 0) in table and (w, 192) in table:
            a0, a192 = table[(w, 0)][1], table[(w, 192)][1]
            diff = np.mean(a192) - np.mean(a0)
            same_dir = all(np.sign(x - np.mean(a0)) == np.sign(diff) for x in a192) if diff != 0 else False
            print(f"{w}: delay 192 - delay 0 = {diff:+.1f} pp (seeds at 192: {[round(x,1) for x in a192]}) -> {'delay-sensitive' if abs(diff) > 2 and same_dir else 'not delay-sensitive'}")
    if a.fig:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
        for ax, L, title in ((axes[0], "L512", "(a) test recall at the training length"), (axes[1], "L2048", "(b) at four times the training length")):
            for w in WRITERS:
                color, label = SERIES[w]
                xs = [d for d in delays if (w, d) in runs]
                means = [np.mean([100 * r["eval"][L]["recall"] for r in runs[(w, d)].values()]) for d in xs]
                ax.plot(xs, means, "-o", color=color, markersize=3, linewidth=1.6, label=label)
                for d in xs:
                    ax.scatter([d] * len(runs[(w, d)]), [100 * r["eval"][L]["recall"] for r in runs[(w, d)].values()], s=10, color=color, alpha=0.5)
            ax.set_xticks(delays); ax.set_xlabel("minimum query delay (tokens)"); ax.set_ylim(-3, 103)
            style(ax, title)
        axes[0].set_ylabel("test recall (%)")
        axes[1].legend(fontsize=7, frameon=False, loc="lower left")
        fig.tight_layout()
        fig.savefig(a.fig, dpi=200)


if __name__ == "__main__":
    main()
