"""One-line status per run under a directory.

    python experiments/progress.py runs/step1            # once
    python experiments/progress.py runs/step1 --watch 60  # refresh every 60 s

Columns: run (dir name), tag, step/total, s/step, ETA, loss (50-step mean), gnorm,
val recall with CI, write_frac, slot_entropy, status. Status is done / running / dead
(no log line for --dead-after minutes) / killed (results.json says running but no process
touched the log for a long time and a resume is possible). Reads results.json when present
and the last timestamped lines of log.txt for the live numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

STEP_RE = re.compile(r"^(\S+ \S+) step\s+(\d+) loss ([\d.]+) gnorm ([\d.]+) lr (\S+) s/step ([\d.]+)")
EVAL_RE = re.compile(r"eval step (\d+): val recall ([\d.]+) ci \[([\d.]+), ([\d.]+)\](?: write_frac ([\d.]+) slot_entropy ([\d.]+))?")
OLD_STEP_RE = re.compile(r"^step\s+(\d+) loss ([\d.]+) gnorm ([\d.]+)")


def read_run(d: Path, dead_after_min: float, root: Path) -> dict | None:
    log = d / "log.txt"
    if not log.exists():
        return None
    row = {"run": str(d.relative_to(root)) if root in d.parents else d.name, "tag": "", "step": 0, "total": None, "s_step": None, "loss": None, "gnorm": None,
           "val": None, "ci": None, "write_frac": None, "entropy": None, "status": "running", "eta": None}
    res = d / "results.json"
    if res.exists():
        try:
            r = json.loads(res.read_text())
            row["tag"] = r.get("tag", "")
            row["total"] = r["config"]["train"]["steps"]
            if r.get("status") == "done":
                row["status"] = "done"
            if r.get("selected_step") and r.get("status") == "done":
                row["tag"] = (row["tag"] + f" [best@{r['selected_step']}]")[:40]
            h = r.get("history") or []
            if h:
                last = h[-1]
                row.update(val=last.get("val_recall"), ci=last.get("val_ci95"),
                           write_frac=last.get("write_frac"), entropy=last.get("slot_entropy"))
        except (json.JSONDecodeError, KeyError):
            pass
    for line in log.read_text().splitlines():
        m = STEP_RE.match(line)
        if m:
            row.update(step=int(m[2]), loss=float(m[3]), gnorm=float(m[4]), s_step=float(m[6]))
            continue
        m = OLD_STEP_RE.match(line)
        if m:
            row.update(step=int(m[1]), loss=float(m[2]), gnorm=float(m[3]))
            continue
        m = EVAL_RE.search(line)
        if m:
            row.update(step=max(row["step"], int(m[1])), val=float(m[2]), ci=[float(m[3]), float(m[4])])
            if m[5]:
                row.update(write_frac=float(m[5]), entropy=float(m[6]))
        if line.startswith("wrote "):
            row["status"] = "done"
    if row["status"] == "done" and row["total"]:
        row["step"] = row["total"]
    age_min = (time.time() - log.stat().st_mtime) / 60
    if row["status"] != "done":
        if age_min > dead_after_min:
            row["status"] = f"dead {age_min/60:.1f}h" if age_min > 120 else f"dead {age_min:.0f}m"
        if row["s_step"] and row["total"]:
            row["eta"] = (row["total"] - row["step"]) * row["s_step"]
    return row


def fmt(row: dict) -> str:
    def f(x, spec, none="-"):
        return format(x, spec) if x is not None else none
    eta = "-" if row["eta"] is None else f"{int(row['eta']//3600)}:{int(row['eta']%3600//60):02d}"
    ci = "" if not row["ci"] else f" [{100*row['ci'][0]:.1f},{100*row['ci'][1]:.1f}]"
    val = "-" if row["val"] is None else f"{100*row['val']:.1f}%{ci}"
    total = f"/{row['total']}" if row["total"] else ""
    return (f"{row['run'][:34]:34} {row['tag'][:22]:22} {row['step']:>6}{total:<7} {f(row['s_step'], '.3f'):>6} {eta:>6} "
            f"{f(row['loss'], '.3f'):>7} {f(row['gnorm'], '.2f'):>6} {val:>20} {f(row['write_frac'], '.2f'):>5} "
            f"{f(row['entropy'], '.2f'):>5}  {row['status']}")


HEADER = (f"{'run':34} {'tag':22} {'step':>13} {'s/step':>6} {'ETA':>6} {'loss':>7} {'gnorm':>6} {'val recall [95% CI]':>20} "
          f"{'wfrac':>5} {'H(S)':>5}  status")


def main(root: Path, watch: float | None, dead_after: float):
    while True:
        dirs = sorted({p.parent for p in root.rglob("log.txt")})
        rows = [r for r in (read_run(d, dead_after, root) for d in dirs) if r]
        if watch:
            print("\033[2J\033[H", end="")
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {root}  ({len(rows)} runs)")
        print(HEADER)
        for r in rows:
            print(fmt(r))
        if not watch:
            break
        time.sleep(watch)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--watch", type=float, default=None, help="seconds between refreshes")
    ap.add_argument("--dead-after", type=float, default=10, help="minutes without a log line before a run is 'dead'")
    a = ap.parse_args()
    main(Path(a.root), a.watch, a.dead_after)
