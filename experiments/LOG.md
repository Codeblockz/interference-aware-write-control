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
  P0 controller broke through at the target (16.9 % at step 2500). P1 and the C1 confirm seeds
  running. Fast path (compile + CUDA graph, 3.4x) landed. Days to fallback date (2026-11-15): 21.

---

## 2026-10-25 P1 vocab confound — runs/step1/p1_vocab2048_controller_s0

Hypothesis: the target fails because vocab 8192 (vs 2048 in pilot16) spreads the key/value
embeddings over 4x more tokens for the same 100k examples, so the association signal per
token is too weak for the memory path to engage.
Config: target.yaml --set data.vocab_size=2048, controller, seed 0, 8000 steps.
Prediction if true: val recall > 5 % by step 2500 (pilot16 broke through at 1400 of 4000 with
loss 6.9 → 6.2). If false: flat at chance through step 4000 → kill.
Result (interim): write_frac 0.000, filled 0.00 at steps 500 and 1000: the controller writes
**nothing** during the plateau (the scalar write gate closes), loss sits at ln(1024) = 6.93.
Decision (interim): the plateau is a write-gate cold start, not a capacity problem. Candidate
probe P8: initialise the write-gate bias high (sigmoid ≈ 0.9) so the memory is populated from
step 0; predicted to remove most of the 2000-step plateau. Decide after P1 ends.

## 2026-10-25 C1 confirm seeds — runs/step1/c1_target_controller_s{1,2}

Hypothesis: P0's breakthrough is not a seed accident.
Config: target.yaml, controller, seeds 1 and 2, 8000 steps, `model.fast=true` (same numerics,
0.23 s/step). Recency seeds 1, 2 follow once the GPU frees up.
Prediction if true: both seeds > 5 % val by step 3500 and > 15 % by 8000; recency stays at chance.
Result:
Decision:

## 2026-10-25 P0 target baseline, aligned settings — runs/target/{controller,recency}_s0

Hypothesis: the earlier `pilot` failure at the target came from the 4-layer / lr 1e-3 settings;
the settings that learned in pilot16 (2 layers, lr 3e-4) will learn at the target too.
Config: target.yaml (64 + 16 + 16 associations, 32 slots, len 512, vocab 8192), seed 0,
8000 steps. Launched before the instrumentation, so the logs have no timestamps or gate stats.
Prediction if true: controller > 5 % val by step 3000.
Result: recency: chance for all 8000 steps (val 0.03 %, loss 8.23; test unread). controller:
chance through step 2000 (0.3 %), then **16.9 % at step 2500, 18.5 % at step 3000** → "learning"
by definition. Final numbers added when the run ends.
Decision: prediction held (one eval late). The proposal's target learns with 2 layers / lr 3e-4;
the `pilot` failure was the 4-layer / lr 1e-3 setting. Step 1 moves to the **confirm stage**
(seeds 1, 2 at target, fast path) instead of P2; P2 (slots=128) is dropped unless the confirm
seeds disagree. P1 keeps running because it also measures whether vocab shortens the plateau.
