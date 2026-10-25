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

- 2026-10-25 17:10: **Step 1 answer so far: the target learns.** Controller at 64+16+16 / 32 slots
  / len 512 breaks off chance at step ~2200 on every run tried (P0 s0, C1 s1, P1), recency stays
  at chance. Two open issues: (1) a ~2000-step plateau in which the controller writes nothing
  (write_frac 0.000); (2) at vocab 8192 recall peaks at 17-18 % near step 3000 and drifts down
  to ~11.5 % by 8000, with loss rising slowly and write_frac flat at 0.10. P1 (vocab 2048) is
  at 29 % at step 3500 and still rising with no drift so far. Running: P0 s0 (eager, ends
  ~17:30), P1 (eager, ends ~18:05), C1 s2 queued behind P0. Instrumentation + fast path (3.4x)
  landed. Days to fallback date (2026-11-15): 21.

---

## 2026-10-25 P1 vocab confound — runs/step1/p1_vocab2048_controller_s0

Hypothesis: the target fails because vocab 8192 (vs 2048 in pilot16) spreads the key/value
embeddings over 4x more tokens for the same 100k examples, so the association signal per
token is too weak for the memory path to engage.
Config: target.yaml --set data.vocab_size=2048, controller, seed 0, 8000 steps.
Prediction if true: val recall > 5 % by step 2500 (pilot16 broke through at 1400 of 4000 with
loss 6.9 → 6.2). If false: flat at chance through step 4000 → kill.
Result (interim, step 3700 of 8000): write_frac 0.000 and filled 0.00 through step 1500
(the controller writes **nothing** during the plateau; loss sits at ln(1024) = 6.93); gate opens
at step 2000 (write_frac 0.007), breakthrough at 2500 (**17.6 %**, same step as the vocab-8192
runs), then **22.3 % at 3000, 29.4 % at 3500** and rising; loss 4.84, gnorm 5.4. Writes
concentrate on pair positions: queried 0.32 / distractor 0.33 / update 0.29 / other 0.01.
Decision (interim): prediction half right. Vocab does not shorten the plateau (that is the
write-gate cold start, see P8), but it changes the level reached: 29 % and climbing vs a
17-18 % peak followed by drift at vocab 8192. Final numbers when the run ends. If the gap
holds, the target definition for the sweep becomes vocab 2048 (stated in the paper's
protocol; not part of H1/H2) and the drift at 8192 is a step-3 stability finding.

Candidate probe P8: `--set model.write_bias_init=2.2` (sigmoid 0.9 at init) so the memory
is populated from step 0; predicted to remove most of the 2000-step plateau. Code landed
(commit a1de51c), launch after C1 s2.

## 2026-10-25 C1 confirm seeds — runs/step1/c1_target_controller_s{1,2}

Hypothesis: P0's breakthrough is not a seed accident.
Config: target.yaml, controller, seeds 1 and 2, 8000 steps, `model.fast=true` (same numerics,
0.23 s/step). Recency seeds 1, 2 follow once the GPU frees up.
Prediction if true: both seeds > 5 % val by step 3500 and > 15 % by 8000; recency stays at chance.
Result (interim): s1 (fast path, 0.24 s/step): chance to step 1500, write gate opens at 2000
(write_frac 0.033, filled 0.44), **16.2 % at 2500, 18.1 % at 3000**, then 16.9 / 12.9 / 12.0 /
11.4 / 11.7 / 11.6 / 11.5 % at 3500-7500; write_frac flat at 0.10, filled 0.90-1.00, slot
entropy 5.0 (uniform slot use). Same trajectory as P0 s0 to within 1 pp at every eval.
s2 not started yet (GPU full: two eager runs + a 5 GB Jupyter kernel from the 743 venv).
Decision (interim): the breakthrough is reproducible across seeds and across the eager/fast
paths; the > 15 % at 8000 half of the prediction fails at vocab 8192 because of the drift.
"Learning" is met; "solved" (> 10 pp over recency at end of schedule) is met numerically
(11.5 vs 0.03) but the peak-then-drift shape must be understood before freezing the config.

## 2026-10-25 P0 target baseline, aligned settings — runs/target/{controller,recency}_s0

Hypothesis: the earlier `pilot` failure at the target came from the 4-layer / lr 1e-3 settings;
the settings that learned in pilot16 (2 layers, lr 3e-4) will learn at the target too.
Config: target.yaml (64 + 16 + 16 associations, 32 slots, len 512, vocab 8192), seed 0,
8000 steps. Launched before the instrumentation, so the logs have no timestamps or gate stats.
Prediction if true: controller > 5 % val by step 3000.
Result: recency: chance for all 8000 steps (val 0.03 %, loss 8.23; test unread). controller
(interim, step 5900): chance through 2000 (0.3 %), **16.9 % at 2500, 18.5 % at 3000, 18.1 % at
3500**, then 12.2 / 11.4 / 11.2 / 11.0 % at 4000-5500; running loss rose from 6.72 (step 3250)
to 7.41 (step 5900) with gnorm 0.3-0.5 (no spike), i.e. a slow drift, not a blow-up. No gate
stats (launched before the instrumentation).
Decision: prediction held (one eval late). The proposal's target learns with 2 layers / lr 3e-4;
the `pilot` failure was the 4-layer / lr 1e-3 setting. Step 1 moves to the **confirm stage**
(seeds 1, 2 at target, fast path) instead of P2; P2 (slots=128) is dropped unless the confirm
seeds disagree. P1 keeps running because it also measures whether vocab shortens the plateau.
