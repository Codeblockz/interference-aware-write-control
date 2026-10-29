"""Gate audit of finished runs: rebuild the model from results.json, load best.pt, run one
validation batch with gate collection, and write gate_audit.json (gate_stats incl. the
update-handling statistics update_key_present / update_in_place) into each run directory.

    python experiments/analysis/gate_audit.py runs/sweep/controller_S64_N64_s0 [more dirs...]
    python experiments/analysis/gate_audit.py --table runs/sweep/controller_S64_N64_s*   # print only

Uses the eager path (fast=false) so no compile/graph is needed; one batch of 128 sequences.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bmem.data import MQARConfig, make_splits  # noqa: E402
from bmem.model import ModelConfig, RecallModel  # noqa: E402
from bmem.metrics import gate_stats  # noqa: E402


def audit(run_dir: Path, batch: int = 128, device: str = "cuda") -> dict:
    res = json.loads((run_dir / "results.json").read_text())
    cfg = res["config"]
    data_cfg = MQARConfig(**cfg["data"])
    mcfg = ModelConfig(**{**cfg["model"], "fast": False, "vocab_size": data_cfg.vocab_size, "pair_block": data_cfg.context_tokens()})
    model = RecallModel(mcfg)
    ck = run_dir / "best.pt"
    model.load_state_dict(torch.load(ck, map_location="cpu"))
    model.to(device).eval()
    seed = res["seed"]
    torch.manual_seed(seed)
    val = make_splits(data_cfg, 1, batch, 1, seed=seed)["val"]  # same val seed stream as training (seed + 1_000_003)
    x, y = val.inputs[:batch].to(device), val.labels[:batch].to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, info = model(x, return_gates=True)
    st = gate_stats(info["gates"], x, y, info["mem"].filled, data_cfg)
    st["checkpoint"] = "best.pt"
    st["selected_step"] = res.get("selected_step")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--table", action="store_true", help="print a row per run instead of (re)writing gate_audit.json")
    a = ap.parse_args()
    for r in a.runs:
        d = Path(r)
        out = d / "gate_audit.json"
        st = json.loads(out.read_text()) if (a.table and out.exists()) else audit(d)
        if not a.table:
            out.write_text(json.dumps(st, indent=2))
        c = st.get("write_by_class", {})
        print(f"{d.name:42s} wf {st['write_frac']:.2f} pairs {c.get('queried', float('nan')):.2f} filler {c.get('other', float('nan')):.3f} "
              f"upd_present {st['update_key_present']:.2f} upd_in_place {st['update_in_place']:.2f}")


if __name__ == "__main__":
    main()
