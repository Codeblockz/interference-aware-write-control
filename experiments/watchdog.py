"""Watch a jobq sweep and exit (loudly) on the first anomaly, or quietly when the launcher is done.

    python experiments/watchdog.py runs/sweep --stall-min 15

Anomalies: a job finished with a non-zero exit code (queue log "done [N]" with N != 0); a run
directory whose process is alive but whose log.txt has not been written for --stall-min
minutes; no jobq.py process alive while the launcher has not written its finished line.
Prints the anomaly and exits 1 so a waiting shell notices. Exits 0 when the launcher is done.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def running_cmds() -> str:
    return subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--stall-min", type=float, default=15)
    ap.add_argument("--every", type=int, default=300)
    a = ap.parse_args()
    root = Path(a.root)
    seen_fail: set[str] = set()
    while True:
        ps = running_cmds()
        for qlog in sorted(root.glob("queue_*.log")):
            for m in re.finditer(r"^(\S+) done \[(\d+)\] (.*)$", qlog.read_text(), re.M):
                if m.group(2) != "0" and m.group(0) not in seen_fail:
                    seen_fail.add(m.group(0))
                    print(f"FAIL {qlog.name}: {m.group(0)[:160]}", flush=True)
                    sys.exit(1)
        now = time.time()
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if f"--out_dir {d}" not in ps and f"--out_dir {d.as_posix()}" not in ps:
                continue
            log = d / "log.txt"
            if log.exists() and now - log.stat().st_mtime > 60 * a.stall_min:
                print(f"STALL {d.name}: log.txt silent for {(now - log.stat().st_mtime)/60:.0f} min", flush=True)
                sys.exit(1)
        finished = any("launcher finished" in q.read_text() for q in root.glob("queue_*.log"))
        if finished:
            print("launcher finished", flush=True)
            sys.exit(0)
        if "jobq.py" not in ps:
            print("QUEUE DEAD: no jobq.py process and launcher not finished", flush=True)
            sys.exit(1)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
