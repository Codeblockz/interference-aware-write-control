"""Run-script I/O: --set overrides, checkpoint/resume, init_from, mid-run results.json."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from run import apply_overrides  # noqa: E402

from bmem.model import ModelConfig, RecallModel  # noqa: E402
from bmem.train import TrainConfig, _lr_at, load_weights  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cfg():
    return {"n_train": 100, "data": {"vocab_size": 1024, "seq_len": 128}, "model": {"slots": 8, "n_layers": 2},
            "train": {"steps": 300, "lr": 1.0e-3, "bf16": True}}


def test_set_overrides_coerce_to_field_types():
    cfg = apply_overrides(_cfg(), ["train.steps=24000", "train.lr=3e-4", "data.vocab_size=2048", "model.slots=128",
                                   "train.min_lr=1e-5", "train.bf16=false", "n_train=5"])
    assert cfg["train"]["steps"] == 24000 and isinstance(cfg["train"]["steps"], int)
    assert cfg["train"]["lr"] == pytest.approx(3e-4)
    assert cfg["data"]["vocab_size"] == 2048 and cfg["model"]["slots"] == 128
    assert cfg["train"]["min_lr"] == pytest.approx(1e-5)  # field absent from YAML, present on TrainConfig
    assert cfg["train"]["bf16"] is False
    assert cfg["n_train"] == 5


@pytest.mark.parametrize("bad", ["train.nope=1", "nope=1", "a.b.c=1", "train.steps"])
def test_set_rejects_unknown_keys(bad):
    with pytest.raises(SystemExit):
        apply_overrides(_cfg(), [bad])


def test_min_lr_is_cosine_floor():
    cfg = TrainConfig(steps=100, warmup=10, lr=1e-3, min_lr=1e-4)
    assert _lr_at(99, cfg) == pytest.approx(1e-4, rel=1e-2)
    assert _lr_at(10, cfg) == pytest.approx(1e-3)
    assert _lr_at(99, TrainConfig(steps=100, warmup=10, lr=1e-3)) == pytest.approx(0.0, abs=1e-6)


def test_load_weights_reports_shape_mismatch(tmp_path):
    a = RecallModel(ModelConfig(vocab_size=256, d_model=32, n_layers=1, n_heads=2, slots=4, mem_dim=16))
    b = RecallModel(ModelConfig(vocab_size=512, d_model=32, n_layers=1, n_heads=2, slots=8, mem_dim=16))
    torch.save(a.state_dict(), tmp_path / "a.pt")
    msgs = []
    rep = load_weights(b, tmp_path / "a.pt", log=msgs.append)
    assert rep["loaded"] > 0 and any("embed" in k or "head" in k for k, *_ in rep["skipped"])
    assert "skipped" in msgs[0]
    assert torch.equal(b.to_q.weight, a.to_q.weight)


@cuda
def test_run_resume_matches_uninterrupted(tmp_path):
    """Kill after the first checkpoint, resume, and reach the same final validation recall."""
    common = [sys.executable, "experiments/run.py", "--config", "experiments/configs/smoke.yaml", "--writer", "recency",
              "--set", "train.steps=60", "--set", "train.eval_every=20", "--set", "train.eval_batches=1",
              "--set", "n_train=500", "--set", "n_val=256", "--set", "n_test=128"]
    full = tmp_path / "full"
    subprocess.run(common + ["--out_dir", str(full)], cwd=ROOT, check=True, capture_output=True)
    part = tmp_path / "part"
    subprocess.run(common + ["--out_dir", str(part), "--set", "train.steps=20"], cwd=ROOT, check=True, capture_output=True)
    r = json.loads((part / "results.json").read_text())
    assert (part / "ckpt.pt").exists() and r["status"] == "done" and r["steps_done"] == 20
    # Pretend it was killed at step 20 of 60: resume with the full schedule.
    subprocess.run(common + ["--out_dir", str(part), "--resume"], cwd=ROOT, check=True, capture_output=True)
    a = json.loads((full / "results.json").read_text())
    b = json.loads((part / "results.json").read_text())
    assert b["status"] == "done" and b["steps_done"] == 60
    assert [h["step"] for h in b["history"]] == [h["step"] for h in a["history"]] == [20, 40, 60]
    assert "resumed from" in (part / "log.txt").read_text()
    assert b["history"][-1]["val_recall"] == pytest.approx(a["history"][-1]["val_recall"], abs=0.02)
    assert a["git"]["commit"] and a["gpu"]
