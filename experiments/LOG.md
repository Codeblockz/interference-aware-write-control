# Experiment log

Lab notebook for the RP3 experiments. One entry per launched run or pair, newest first.
Write the hypothesis and the numeric prediction **before** launching; fill Result and
Decision after. Never edit a prediction. Read `PLAN.md` for the step list and the plan file
for the probe tree (P0-P7) behind step 1.

Tools: `python experiments/progress.py runs/step1 --watch 300` (status table with ETA),
`python experiments/curves.py runs/step1` (loss / recall / gnorm / write-fraction curves),
`--set key=value` on `run.py` for one-variable probes, `--resume` after a kill,
`--init_from <best.pt>` for curriculum.

## Definitions (step 1)

- **chance**: val recall < 0.5 % (loss ≈ ln(vocab/2); 8.32 at vocab 8192, 6.93 at 2048).
- **learning**: val recall > 5 % and rising over two consecutive evals.
- **solved**: controller beats recency by > 10 pp val recall at the target, 3 seeds, none collapsed.
- **kill rule**: still at chance at 50 % of the schedule with running loss flat (< 0.02 change
  over the last 1000 steps) → kill, log "dead at step X". Exception: P5 runs to completion once.
- Probes read **validation** only. Test-length numbers stay unread until the sweep.

## Status

- 2026-10-25: Part A instrumentation landed (timestamps, ckpt/resume, --set, gate stats,
  progress/curves/bench scripts). P0 finished for recency (chance); controller P0 still running.
  P1 launched. Best known result at target: none (chance). Days to fallback date (2026-11-15): 21.

---

## 2026-10-25 P1 vocab confound — runs/step1/p1_vocab2048_controller_s0

Hypothesis: the target fails because vocab 8192 (vs 2048 in pilot16) spreads the key/value
embeddings over 4x more tokens for the same 100k examples, so the association signal per
token is too weak for the memory path to engage.
Config: target.yaml --set data.vocab_size=2048, controller, seed 0, 8000 steps.
Prediction if true: val recall > 5 % by step 2500 (pilot16 broke through at 1400 of 4000 with
loss 6.9 → 6.2). If false: flat at chance through step 4000 → kill.
Result:
Decision:

## 2026-10-25 P0 target baseline, aligned settings — runs/target/{controller,recency}_s0

Hypothesis: the earlier `pilot` failure at the target came from the 4-layer / lr 1e-3 settings;
the settings that learned in pilot16 (2 layers, lr 3e-4) will learn at the target too.
Config: target.yaml (64 + 16 + 16 associations, 32 slots, len 512, vocab 8192), seed 0,
8000 steps. Launched before the instrumentation, so the logs have no timestamps or gate stats.
Prediction if true: controller > 5 % val by step 3000.
Result: recency: chance for all 8000 steps (val 0.03 %, loss 8.23 → test unread). controller:
chance through step 2150 (loss 8.05, gnorm 0.4) at the time of writing; see below for the end.
Decision: settings alone do not fix it. P1 (vocab) and P2 (slots=128) are the next pair.
