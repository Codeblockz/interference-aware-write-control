"""Step 6 sweep: per-cell table (mean over seeds, test recall at each length) and the H1 / H2
paired bootstraps, from runs/sweep/<writer>_S<S>_N<N>_s<seed>/.

    python experiments/analysis/sweep_table.py runs/sweep            # table + tests, Markdown
    python experiments/analysis/sweep_table.py runs/sweep --n 64 --latex

H1: controller minus the best non-learned writer (pair_recency, recency, residual, position:
the one with the highest mean test recall at the training length in that cell), paired over
the same test sequences per seed; the seed-level differences are averaged and the bootstrap
CI pools the three seeds' per-example differences. H2: controller minus no_interference.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bmem.metrics import bootstrap_ci  # noqa: E402

WRITERS = ["controller", "no_interference", "pair_recency", "recency", "residual", "position"]
NONLEARNED = ["pair_recency", "recency", "residual", "position"]
NAME = re.compile(r"^(?P<w>[a-z_]+)_S(?P<s>\d+)_N(?P<n>\d+)_s(?P<seed>\d+)$")


def load(root: Path) -> dict:
    runs = defaultdict(dict)  # (w, s, n) -> seed -> record
    for f in sorted(root.glob("*/results.json")):
        m = NAME.match(f.parent.name)
        if not m:
            continue
        r = json.loads(f.read_text())
        if r.get("status") != "done":
            continue
        r["_dir"] = f.parent
        runs[(m["w"], int(m["s"]), int(m["n"]))][int(m["seed"])] = r
    return runs


def mean_recall(recs: dict, L: str) -> float:
    return float(np.mean([100 * r["eval"][L]["recall"] for r in recs.values()]))


def gate(recs: dict) -> str:
    best = [max(r["history"], key=lambda e: e["val_recall"]) for r in recs.values()]
    wf = np.mean([e.get("write_frac", np.nan) for e in best])
    q = np.mean([e.get("write_by_class", {}).get("queried", np.nan) for e in best])
    o = np.mean([e.get("write_by_class", {}).get("other", np.nan) for e in best])
    return f"{wf:.2f} / {q:.2f} / {o:.2f}"


T_975_DF2 = 4.302653  # scipy.stats.t.ppf(0.975, 2); hard-coded to avoid the dependency


def seed_interval(per_seed) -> tuple[float, float]:
    """95 % t-interval for the mean of the per-seed differences (df = n_seeds - 1)."""
    x = np.asarray(per_seed, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    from math import sqrt
    half = T_975_DF2 * x.std(ddof=1) / sqrt(len(x)) if len(x) == 3 else float("nan")
    return float(x.mean() - half), float(x.mean() + half)


def verdict(per_seed, threshold: float = 0.0) -> str:
    """Pre-registered rule (paper IV-A): 'supported' if the seed interval lies above the
    threshold, 'reversed' if it lies below 0, else 'not supported'."""
    lo, hi = seed_interval(per_seed)
    if lo > threshold:
        return "supported"
    if hi < 0:
        return "reversed"
    return "not supported"


def paired(runs, a, b, s, n, L):
    """Mean difference (pp) and pooled paired bootstrap CI over the seeds both writers have."""
    ra, rb = runs.get((a, s, n), {}), runs.get((b, s, n), {})
    seeds = sorted(set(ra) & set(rb))
    if not seeds:
        return None
    diffs = np.concatenate([np.load(ra[k]["_dir"] / f"per_example_{L}.npy") - np.load(rb[k]["_dir"] / f"per_example_{L}.npy") for k in seeds])
    lo, hi = bootstrap_ci(diffs)
    per_seed = [100 * float((np.load(ra[k]["_dir"] / f"per_example_{L}.npy") - np.load(rb[k]["_dir"] / f"per_example_{L}.npy")).mean()) for k in seeds]
    return 100 * diffs.mean(), 100 * lo, 100 * hi, per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--n", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    runs = load(Path(a.root))
    for n in a.n:
        cells = sorted({s for (w, s, nn) in runs if nn == n})
        if not cells:
            continue
        L0 = f"L{512 if n == 64 else 1024}"
        lengths = sorted({k for (w, s, nn), recs in runs.items() if nn == n for r in recs.values() for k in r["eval"]}, key=lambda k: int(k[1:]))
        print(f"\n### N = {n}: mean test recall (%) over seeds at each length [n seeds]; gates = write_frac / pairs written / filler written at the best checkpoint\n")
        head = ["writer"] + [f"S={s} {L}" for s in cells for L in lengths] + [f"S={s} gates" for s in cells]
        print("| " + " | ".join(head) + " |"); print("|" + "---|" * len(head))
        for w in WRITERS:
            row = [w]
            for s in cells:
                recs = runs.get((w, s, n), {})
                for L in lengths:
                    row.append(f"{mean_recall(recs, L):.1f} [{len(recs)}]" if recs else "-")
            for s in cells:
                recs = runs.get((w, s, n), {})
                row.append(gate(recs) if recs and w in ("controller", "no_interference", "residual") else "")
            print("| " + " | ".join(row) + " |")
        print(f"\n### N = {n}: hypothesis tests at the training length ({L0}); paired over test sequences, pooled across seeds, 95% bootstrap CI; per-seed differences in brackets\n")
        for s in cells:
            best_nl = max((w for w in NONLEARNED if (w, s, n) in runs), key=lambda w: mean_recall(runs[(w, s, n)], L0), default=None)
            if best_nl:
                r = paired(runs, "controller", best_nl, s, n, L0)
                if r:
                    lo, hi = seed_interval(r[3])
                    print(f"- H1 S={s}: controller - {best_nl} = {r[0]:+.1f} pp, pooled CI [{r[1]:+.1f}, {r[2]:+.1f}], seed interval [{lo:+.1f}, {hi:+.1f}]  {[f'{d:+.1f}' for d in r[3]]}  -> {verdict(r[3], 5.0)}")
            r = paired(runs, "controller", "no_interference", s, n, L0)
            if r:
                lo, hi = seed_interval(r[3])
                print(f"- H2 S={s}: controller - no_interference = {r[0]:+.1f} pp, pooled CI [{r[1]:+.1f}, {r[2]:+.1f}], seed interval [{lo:+.1f}, {hi:+.1f}]  {[f'{d:+.1f}' for d in r[3]]}  -> {verdict(r[3], 0.0)}")


if __name__ == "__main__":
    main()
