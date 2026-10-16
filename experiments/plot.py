"""Figures from runs/**/results.json.

    python experiments/plot.py runs/pilot --out paper/figures/pilot.png

Left: validation recall during training, one line per writer (mean over seeds).
Right: test recall vs evaluation length, one line per writer, with 95% CI bars.
Palette: validated two/three-slot categorical set (blue, orange, aqua) on a light surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = {  # fixed order, never cycled
    "controller": ("#2a78d6", "Interference controller"),
    "no_interference": ("#eb6834", "No-interference ablation"),
    "recency": ("#1baf7a", "Recency"),
    "residual": ("#eda100", "Residual cache"),
    "position": ("#e87ba4", "Position activation"),
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(root: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(root.rglob("results.json"))]


def style(ax, title):
    ax.set_title(title, loc="left", color=TEXT, fontsize=10, pad=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def main(root: Path, out: Path):
    rows = load(root)
    writers = [w for w in SERIES if any(r["writer"] == w for r in rows)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4), dpi=200, facecolor="#fcfcfb")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")

    # Left: training curves.
    for w in writers:
        color, label = SERIES[w]
        hs = [r["history"] for r in rows if r["writer"] == w]
        steps = [h["step"] for h in hs[0]]
        vals = np.array([[h["val_recall"] for h in hist] for hist in hs]).mean(0)
        ax1.plot(steps, 100 * vals, color=color, linewidth=2, label=label)
        ax1.plot(steps[-1], 100 * vals[-1], "o", color=color, markersize=5)
        ax1.annotate(label, (steps[-1], 100 * vals[-1]), xytext=(4, 0), textcoords="offset points",
                     fontsize=7.5, color=TEXT, va="center")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("validation exact recall (%)")
    ax1.set_ylim(0, 100)
    style(ax1, "Training: validation recall")

    # Right: recall vs eval length.
    for w in writers:
        color, label = SERIES[w]
        rs = [r for r in rows if r["writer"] == w]
        Ls = sorted({int(k[1:]) for r in rs for k in r["eval"]})
        mean = [100 * np.mean([r["eval"][f"L{L}"]["recall"] for r in rs]) for L in Ls]
        lo = [100 * np.mean([r["eval"][f"L{L}"]["recall_ci95"][0] for r in rs]) for L in Ls]
        hi = [100 * np.mean([r["eval"][f"L{L}"]["recall_ci95"][1] for r in rs]) for L in Ls]
        ax2.errorbar(Ls, mean, yerr=[np.subtract(mean, lo), np.subtract(hi, mean)], color=color,
                     linewidth=2, marker="o", markersize=5, capsize=3, label=label)
        ax2.annotate(f"{mean[-1]:.0f}%", (Ls[-1], mean[-1]), xytext=(5, 0), textcoords="offset points",
                     fontsize=7.5, color=TEXT, va="center")
    train_len = rows[0]["config"]["data"]["seq_len"]
    ax2.axvline(train_len, color=GRID, linewidth=1, linestyle="--")
    ax2.annotate("train length", (train_len, 94), xytext=(4, 0), textcoords="offset points", fontsize=7, color=MUTED, ha="left")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(Ls)
    ax2.set_xticklabels([str(L) for L in Ls])
    ax2.set_xlabel("evaluation sequence length")
    ax2.set_ylabel("test exact recall (%)")
    ax2.set_ylim(0, 100)
    ax2.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=TEXT)
    style(ax2, "Test: recall vs. length (95% bootstrap CI)")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="paper/figures/pilot.png")
    a = ap.parse_args()
    main(Path(a.root), Path(a.out))
