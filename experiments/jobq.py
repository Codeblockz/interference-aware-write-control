"""Run a list of run.py commands with bounded concurrency and a GPU-memory guard.

    python experiments/jobq.py jobs.txt --max 3 --free-gb 8

jobs.txt: one shell command per line (blank lines and # comments ignored). A job starts only
when fewer than --max jobs are running and nvidia-smi reports at least --free-gb free; this
avoids the launch-time OOMs seen when a run was starting while another ran its test evals.
Each job's stdout/stderr go to <out_dir>.stderr next to its --out_dir if that flag is present,
else to queue-<n>.log. Exit codes are printed as jobs finish.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import time
from pathlib import Path


def free_gb() -> float:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return min(float(x) for x in out.split()) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--max", type=int, default=3)
    ap.add_argument("--free-gb", type=float, default=8.0)
    ap.add_argument("--settle", type=float, default=60, help="seconds to wait after a launch before considering the next")
    a = ap.parse_args()
    cmds = [l.strip() for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith("#")]
    running: list[tuple[subprocess.Popen, str]] = []
    i = 0
    while i < len(cmds) or running:
        for p, name in list(running):
            if p.poll() is not None:
                print(f"{time.strftime('%H:%M:%S')} done [{p.returncode}] {name}", flush=True)
                running.remove((p, name))
        if i < len(cmds) and len(running) < a.max and free_gb() >= a.free_gb:
            cmd = cmds[i]
            m = re.search(r"--out_dir\s+(\S+)", cmd)
            logp = Path(m.group(1) + ".stderr") if m else Path(f"queue-{i}.log")
            logp.parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.Popen(shlex.split(cmd), stdout=open(logp, "w"), stderr=subprocess.STDOUT)
            running.append((p, cmd))
            print(f"{time.strftime('%H:%M:%S')} start {cmd}", flush=True)
            i += 1
            time.sleep(a.settle)
            continue
        time.sleep(20)
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
