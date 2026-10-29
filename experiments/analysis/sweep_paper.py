"""Paper artefacts from the sweep: Table II rows (LaTeX) and the sweep figure.

    python experiments/analysis/sweep_paper.py runs/sweep --fig paper/figures/sweep.png --tex paper/build/sweep_rows.tex

Figure: (a) test recall at the training length against slot budget, mean over seeds with the
seeds as dots, for N = 64; (b) recall against evaluation length at S = 64; (c) admission rates
(fraction of pairs written, fraction of filler written) at the best checkpoint for the three
writers with a learned or thresholded gate.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from plot import SERIES, TEXT, MUTED, GRID, style  # noqa: E402
from sweep_table import load, paired, NONLEARNED, mean_recall, seed_interval  # noqa: E402
from bmem.metrics import bootstrap_ci  # noqa: E402

ROWS = [("controller", "Controller"), ("no_interference", "Novelty ablation"),
        ("pair_recency", "Pair-only recency"), ("recency", "Recency"),
        ("residual", "Residual cache"), ("position", "Position activation")]
CELLS = [(64, 32), (64, 64), (64, 128), (128, 32), (128, 64), (128, 128)]


def L0(n):
    return f"L{512 if n == 64 else 1024}"


H2NAME = re.compile(r"^(?:s8_)?(?P<w>[a-z_]+)_lr(?P<lr>[0-9e.-]+)_S(?P<s>\d+)_N(?P<n>\d+)_s(?P<seed>\d+)$")


def load_h2(roots) -> dict:
    """runs/sweep_h2 and runs/step4: (w, lr, s, n) -> seed -> record (equal-lr H2 check, S8 input-only)."""
    runs = defaultdict(dict)
    for f in sorted(f for root in roots for f in Path(root).glob("*/results.json")):
        m = H2NAME.match(f.parent.name)
        r = json.loads(f.read_text())
        if not m or r.get("status") != "done":
            continue
        r["_dir"] = f.parent
        runs[(m["w"], m["lr"], int(m["s"]), int(m["n"]))][int(m["seed"])] = r
    return runs


def paired_recs(ra: dict, rb: dict, L: str):
    seeds = sorted(set(ra) & set(rb))
    if not seeds:
        return None
    per = [np.load(ra[k]["_dir"] / f"per_example_{L}.npy") - np.load(rb[k]["_dir"] / f"per_example_{L}.npy") for k in seeds]
    pooled = np.concatenate(per)
    lo, hi = bootstrap_ci(pooled)
    return 100 * pooled.mean(), 100 * lo, 100 * hi, [100 * float(d.mean()) for d in per]


def tex_rows(runs, h2=None) -> str:
    out = []
    for w, label in ROWS:
        cells = []
        for n, s in CELLS:
            recs = runs.get((w, s, n), {})
            if not recs:
                cells.append("--")
                continue
            vals = [100 * r["eval"][L0(n)]["recall"] for r in recs.values()]
            cells.append(f"{np.mean(vals):.1f} {{\\scriptsize$\\pm${np.std(vals, ddof=1):.1f}}}" if len(vals) > 1 else f"{vals[0]:.1f}")
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    out.append(r"\midrule")
    for name, comp in (("H1: ctrl $-$ best rule-based", None), ("H2: ctrl $-$ novelty ablation", "no_interference")):
        cells = []
        for n, s in CELLS:
            b = comp or max((x for x in NONLEARNED if (x, s, n) in runs), key=lambda x: mean_recall(runs[(x, s, n)], L0(n)), default=None)
            r = paired(runs, "controller", b, s, n, L0(n)) if b and ("controller", s, n) in runs else None
            if r is None:
                cells.append("--")
            else:
                lo, hi = seed_interval(r[3])
                cells.append(f"{r[0]:+.1f} {{\\scriptsize[{lo:+.0f}, {hi:+.0f}]}}")
        out.append(f"{name} (pp) & " + " & ".join(cells) + r" \\")
    if h2:
        out.append(r"\midrule")
        out.append(r"Novelty ablation at $3\times10^{-4}$ & " + " & ".join(
            (lambda recs: f"{np.mean([100 * r['eval'][L0(n)]['recall'] for r in recs.values()]):.1f} {{\\scriptsize$\\pm${np.std([100 * r['eval'][L0(n)]['recall'] for r in recs.values()], ddof=1):.1f}}}" if recs else "--")(h2.get(("no_interference", "3e-4", s, n), {})) for n, s in CELLS) + r" \\")
        for lr, cget, aget, label in (("3e-4", lambda s, n: runs.get(("controller", s, n), {}), lambda s, n: h2.get(("no_interference", "3e-4", s, n), {}), "H2, both at $3\\times10^{-4}$"),
                                      ("1e-3", lambda s, n: h2.get(("controller", "1e-3", s, n), {}), lambda s, n: runs.get(("no_interference", s, n), {}), "H2, both at $10^{-3}$")):
            cells = []
            for n, s in CELLS:
                r = paired_recs(cget(s, n), aget(s, n), L0(n)) if cget(s, n) and aget(s, n) else None
                cells.append("--" if r is None else f"{r[0]:+.1f} {{\\scriptsize[{seed_interval(r[3])[0]:+.0f}, {seed_interval(r[3])[1]:+.0f}]}}")
            out.append(f"{label} (pp) & " + " & ".join(cells) + r" \\")
        if any(k[0] == "input_only" for k in h2):
            out.append(r"Input-only ablation at $3\times10^{-4}$ & " + " & ".join(
                (lambda recs: f"{np.mean([100 * r['eval'][L0(n)]['recall'] for r in recs.values()]):.1f} {{\\scriptsize$\\pm${np.std([100 * r['eval'][L0(n)]['recall'] for r in recs.values()], ddof=1):.1f}}}" if recs else "--")(h2.get(("input_only", "3e-4", s, n), {})) for n, s in CELLS) + r" \\")
            cells = []
            for n, s in CELLS:
                a, b = h2.get(("input_only", "3e-4", s, n), {}), runs.get(("controller", s, n), {})
                r = paired_recs(a, b, L0(n)) if a and b else None
                cells.append("--" if r is None else f"{r[0]:+.1f} {{\\scriptsize[{seed_interval(r[3])[0]:+.0f}, {seed_interval(r[3])[1]:+.0f}]}}")
            out.append(r"Input-only $-$ controller at $3\times10^{-4}$ (pp) & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def figure(runs, out: Path, n: int = 64, h2=None):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.2, 1, 1.1]})
    slots = sorted({s for (w, s, nn) in runs if nn == n})
    ax = axes[0]
    present = {w for (w, _, nn) in runs if nn == n}
    for w, (color, label) in SERIES.items():
        if w not in present:
            continue
        means, pts = [], []
        for s in slots:
            recs = runs.get((w, s, n), {})
            vals = [100 * r["eval"][L0(n)]["recall"] for r in recs.values()]
            means.append(np.mean(vals) if vals else np.nan)
            pts += [(s, v) for v in vals]
        ax.plot(slots, means, "-", color=color, linewidth=1.8, label=label, zorder=3)
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=12, color=color, alpha=0.6, zorder=4)
    if h2:
        means = [np.mean([100 * r["eval"][L0(n)]["recall"] for r in h2.get(("no_interference", "3e-4", s, n), {}).values()] or [np.nan]) for s in slots]
        ax.plot(slots, means, "--", color=SERIES["no_interference"][0], linewidth=1.6, label="Novelty ablation at the controller's lr", zorder=3)
        if any(k[0] == "input_only" for k in h2):
            means = [np.mean([100 * r["eval"][L0(n)]["recall"] for r in h2.get(("input_only", "3e-4", s, n), {}).values()] or [np.nan]) for s in slots]
            ax.plot(slots, means, ":", color=SERIES["input_only"][0], linewidth=1.8, label="Input-only ablation (3e-4)", zorder=3)
    ax.set_xscale("log", base=2); ax.set_xticks(slots); ax.set_xticklabels([str(s) for s in slots])
    ax.set_xlabel("slots S"); ax.set_ylabel("test recall (%)"); ax.set_ylim(-3, 103)
    style(ax, f"(a) recall at training length, N = {n} (dots: seeds)")
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
    gated = [("controller", "controller"), ("no_interference", "novelty abl."), ("residual", "residual")]
    if h2 and any(k[0] == "input_only" for k in h2):
        gated.insert(2, ("input_only", "input-only abl."))
    x = np.arange(len(slots)); width = 0.10
    for i, (w, short) in enumerate(gated):
        color = SERIES[w][0]
        pairs, filler = [], []
        for s in slots:
            recs = h2.get((w, "3e-4", s, n), {}) if (w == "input_only" and h2) else runs.get((w, s, n), {})
            best = [max(r["history"], key=lambda e: e["val_recall"]) for r in recs.values()]
            pairs.append(100 * np.mean([e["write_by_class"]["queried"] for e in best]) if best else np.nan)
            filler.append(100 * np.mean([e["write_by_class"]["other"] for e in best]) if best else np.nan)
        off = (i - (len(gated) - 1) / 2) * 2.3 * width
        ax.bar(x + off - width / 2, pairs, width, color=color, label=f"{short}: pairs")
        ax.bar(x + off + width / 2, filler, width, color="none", edgecolor=color, hatch="////", linewidth=0.8, label=f"{short}: filler")
    ax.set_xticks(x); ax.set_xticklabels([f"S={s}" for s in slots]); ax.set_ylim(0, 105)
    ax.set_ylabel("% of positions written")
    style(ax, "(c) admission at the best checkpoint")
    ax.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper right")
    ax.set_ylim(0, 160); ax.set_yticks([0, 25, 50, 75, 100])
    fig.legend(*handles, fontsize=7, frameon=False, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(out, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--fig-n", type=int, default=64, help="load N shown in the figure")
    ap.add_argument("--tex", default=None)
    ap.add_argument("--h2", nargs="*", default=None, help="roots with <writer>_lr<lr>_S<S>_N<N>_s<seed> runs (runs/sweep_h2, runs/step4)")
    a = ap.parse_args()
    runs = load(Path(a.root))
    h2 = load_h2(a.h2) if a.h2 else None
    rows = tex_rows(runs, h2)
    if a.tex:
        Path(a.tex).write_text(rows + "\n")
    print(rows)
    if a.fig:
        figure(runs, Path(a.fig), n=a.fig_n, h2=h2 if a.fig_n == 64 else None)


if __name__ == "__main__":
    main()
