"""Paper artefacts from the sweep: Table II rows (LaTeX) and the sweep figure.

    python experiments/analysis/sweep_paper.py runs/sweep --fig paper/figures/sweep.png --tex paper/build/sweep_rows.tex

Figure: (a) test recall at the training length against slot budget, mean over seeds with the
seeds as dots, for N = 64; (b) recall against evaluation length at S = 64; (c) admission rates
(fraction of pairs written, fraction of filler written) at the best checkpoint for the three
writers with a learned or thresholded gate.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from plot import SERIES, TEXT, MUTED, GRID, style  # noqa: E402
from sweep_table import load, paired, NONLEARNED, mean_recall  # noqa: E402

ROWS = [("controller", "Interference controller"), ("no_interference", "No-interference ablation"),
        ("pair_recency", "Pair-only recency"), ("recency", "Recency"),
        ("residual", "Residual cache (HOLA-style)"), ("position", "Position activation (Proteus-style)")]
CELLS = [(64, 32), (64, 64), (64, 128), (128, 32), (128, 64), (128, 128)]


def L0(n):
    return f"L{512 if n == 64 else 1024}"


def tex_rows(runs) -> str:
    out = []
    for w, label in ROWS:
        cells = []
        for n, s in CELLS:
            recs = runs.get((w, s, n), {})
            if not recs:
                cells.append("--")
                continue
            vals = [100 * r["eval"][L0(n)]["recall"] for r in recs.values()]
            cells.append(f"{np.mean(vals):.1f} {{\\scriptsize$\\pm${np.std(vals):.1f}}}" if len(vals) > 1 else f"{vals[0]:.1f}")
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    for name, comp in (("H1: controller $-$ best non-learned", None), ("H2: controller $-$ ablation", "no_interference")):
        cells = []
        for n, s in CELLS:
            b = comp or max((x for x in NONLEARNED if (x, s, n) in runs), key=lambda x: mean_recall(runs[(x, s, n)], L0(n)), default=None)
            r = paired(runs, "controller", b, s, n, L0(n)) if b and ("controller", s, n) in runs else None
            if r is None:
                cells.append("--")
            else:
                seeds = ", ".join(f"{d:+.0f}" for d in r[3])
                cells.append(f"{r[0]:+.1f} {{\\scriptsize({seeds})}}")
        out.append(f"{name} (pp) & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def figure(runs, out: Path, n: int = 64):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.2, 1, 1.1]})
    slots = sorted({s for (w, s, nn) in runs if nn == n})
    ax = axes[0]
    for w, (color, label) in SERIES.items():
        means, pts = [], []
        for s in slots:
            recs = runs.get((w, s, n), {})
            vals = [100 * r["eval"][L0(n)]["recall"] for r in recs.values()]
            means.append(np.mean(vals) if vals else np.nan)
            pts += [(s, v) for v in vals]
        ax.plot(slots, means, "-", color=color, linewidth=1.8, label=label, zorder=3)
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=12, color=color, alpha=0.6, zorder=4)
    ax.set_xscale("log", base=2); ax.set_xticks(slots); ax.set_xticklabels([str(s) for s in slots])
    ax.set_xlabel("slots S"); ax.set_ylabel("test recall (%)"); ax.set_ylim(-3, 103)
    style(ax, f"(a) recall at the training length, N = {n} (dots: seeds)")
    handles = ax.get_legend_handles_labels()
    ax = axes[1]
    s_mid = 64 if 64 in slots else slots[len(slots) // 2]
    for w, (color, label) in SERIES.items():
        recs = runs.get((w, s_mid, n), {})
        if not recs:
            continue
        Ls = sorted({int(k[1:]) for r in recs.values() for k in r["eval"]})
        ys = [np.mean([100 * r["eval"][f"L{L}"]["recall"] for r in recs.values()]) for L in Ls]
        ax.plot(Ls, ys, "-o", color=color, markersize=3, linewidth=1.6)
    ax.set_xscale("log", base=2); ax.set_xticks(Ls); ax.set_xticklabels([str(L) for L in Ls])
    ax.set_xlabel("evaluation length"); ax.set_ylim(-3, 103)
    style(ax, f"(b) length generalization, S = {s_mid}")
    ax = axes[2]
    gated = [("controller", "controller"), ("no_interference", "ablation"), ("residual", "residual")]
    x = np.arange(len(slots)); width = 0.13
    for i, (w, short) in enumerate(gated):
        color = SERIES[w][0]
        pairs, filler = [], []
        for s in slots:
            recs = runs.get((w, s, n), {})
            best = [max(r["history"], key=lambda e: e["val_recall"]) for r in recs.values()]
            pairs.append(100 * np.mean([e["write_by_class"]["queried"] for e in best]) if best else np.nan)
            filler.append(100 * np.mean([e["write_by_class"]["other"] for e in best]) if best else np.nan)
        off = (i - 1) * 2.3 * width
        ax.bar(x + off - width / 2, pairs, width, color=color, label=f"{short}: pairs")
        ax.bar(x + off + width / 2, filler, width, color="none", edgecolor=color, hatch="////", linewidth=0.8, label=f"{short}: filler")
    ax.set_xticks(x); ax.set_xticklabels([f"S={s}" for s in slots]); ax.set_ylim(0, 105)
    ax.set_ylabel("% of positions written")
    style(ax, "(c) admission at the best checkpoint")
    ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100])
    fig.legend(*handles, fontsize=7.5, frameon=False, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()
    runs = load(Path(a.root))
    rows = tex_rows(runs)
    if a.tex:
        Path(a.tex).write_text(rows + "\n")
    print(rows)
    if a.fig:
        figure(runs, Path(a.fig))


if __name__ == "__main__":
    main()
