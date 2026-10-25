"""Training curves for every run under a directory, from results.json history.

    python experiments/curves.py runs/step1 --out runs/step1/curves.png

Panels: running loss, validation recall (%), pre-clip gradient norm (log), and write
fraction, all against training step; one line per run, labelled by tag or directory name.
Works on running runs because results.json is rewritten at every eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot import GRID, MUTED, SERIES, TEXT, style  # noqa: E402

FALLBACK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7b61ff", "#5c5c5c", "#9c2f2f"]


def main(root: Path, out: Path):
    runs = []
    for f in sorted(root.rglob("results.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("history"):
            runs.append((f.parent.name, r))
    if not runs:
        raise SystemExit(f"no results.json with history under {root}")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), dpi=160, facecolor="#fcfcfb")
    axes = axes.ravel()
    for i, (name, r) in enumerate(runs):
        h = r["history"]
        steps = [e["step"] for e in h]
        label = (r.get("tag") or name)[:40] + ("" if r.get("status") == "done" else " (running)")
        color = SERIES.get(r.get("writer"), (FALLBACK[i % len(FALLBACK)],))[0] if len(runs) <= 2 else FALLBACK[i % len(FALLBACK)]
        kw = dict(color=color, linewidth=1.6, label=label)
        axes[0].plot(steps, [e["loss"] for e in h], **kw)
        axes[1].plot(steps, [100 * e["val_recall"] for e in h], **kw)
        if all(e.get("gnorm") is not None for e in h):
            axes[2].plot(steps, [e["gnorm"] for e in h], **kw)
        if all(e.get("write_frac") is not None for e in h):
            axes[3].plot(steps, [e["write_frac"] for e in h], **kw)
    for ax, title, ylab in zip(axes, ["Running loss", "Validation recall", "Gradient norm (pre-clip)", "Write fraction"],
                               ["loss", "exact recall (%)", "||g||", "mean gate mass / step"]):
        ax.set_facecolor("#fcfcfb")
        ax.set_xlabel("training step")
        ax.set_ylabel(ylab)
        style(ax, title)
    axes[1].set_ylim(0, 100)
    axes[2].set_yscale("log")
    axes[3].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, fontsize=7, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out} ({len(runs)} runs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None, help="default: <root>/curves.png")
    a = ap.parse_args()
    main(Path(a.root), Path(a.out) if a.out else Path(a.root) / "curves.png")
