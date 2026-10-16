"""Aggregate runs/**/results.json into a Markdown table and a paired comparison.

    python experiments/aggregate.py runs/pilot            # table of every run under the dir
    python experiments/aggregate.py runs/pilot --pair controller recency --length 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bmem.metrics import paired_bootstrap_ci  # noqa: E402


def load(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.rglob("results.json")):
        r = json.loads(f.read_text())
        r["_dir"] = f.parent
        rows.append(r)
    return rows


def table(rows: list[dict]) -> str:
    lengths = sorted({int(k[1:]) for r in rows for k in r["eval"]})
    head = ["writer", "seed", "params", "state bytes"] + [f"recall L={L}" for L in lengths] + ["train min", "tok/s", "peak MB"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["writer"], str(r["seed"]), f"{r['params']['total']/1e6:.2f}M", f"{r['state_bytes']['total']/1024:.0f} KB"]
        for L in lengths:
            e = r["eval"].get(f"L{L}")
            cells.append("" if e is None else f"{100*e['recall']:.1f} [{100*e['recall_ci95'][0]:.1f}, {100*e['recall_ci95'][1]:.1f}]")
        cells += [f"{r['train_seconds']/60:.0f}", f"{r['train_tokens_per_s']/1e3:.1f}k", f"{r['peak_mem_mb']:.0f}"]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def pair(rows: list[dict], a: str, b: str, L: int) -> str:
    ra = [r for r in rows if r["writer"] == a]
    rb = [r for r in rows if r["writer"] == b]
    lines = []
    for x in ra:
        for y in rb:
            if x["seed"] != y["seed"]:
                continue
            pa = np.load(x["_dir"] / f"per_example_L{L}.npy")
            pb = np.load(y["_dir"] / f"per_example_L{L}.npy")
            lo, hi = paired_bootstrap_ci(pa, pb)
            lines.append(f"seed {x['seed']}, L={L}: {a} - {b} = {100*(pa.mean()-pb.mean()):+.1f} pp, paired 95% CI [{100*lo:+.1f}, {100*hi:+.1f}]")
    return "\n".join(lines) or "no matching seeds"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--length", type=int, default=512)
    args = ap.parse_args()
    rows = load(Path(args.root))
    print(table(rows))
    if args.pair:
        print()
        print(pair(rows, *args.pair, args.length))
