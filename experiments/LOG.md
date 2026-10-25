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

- 2026-10-25 18:20: **Step 1 result: the target learns; 3 seeds agree.** Controller at 64+16+16 /
  32 slots / len 512 / vocab 8192, 2 layers, lr 3e-4: breakthrough at step 2000-2500 on all
  seeds, peak 18-19 % val at step 3000, then a drift down to 11-14 % at 8000. Recency: chance.
  Test recall (L=512) 11.4 / 11.7 / 13.6 % for seeds 0 / 1 / 2. P8 (write gate open at init)
  changes nothing: the optimizer closes the gate by step 1000 and reopens it at 2000 like every
  other run. P1 (vocab 2048) reaches 32 % with no drift and holds 32.1 % at L=1024, but the
  generator cannot build L=2048 sequences at vocab 2048, so the protocol's longest test is
  impossible there. Open: (a) the peak-then-drift at vocab 8192 (step 3); (b) vocab choice for
  the sweep (P1b: vocab 4096). Fast path (3.4x) landed; eval OOM at L=2048 fixed (eval batch
  scaled by length; `--eval_only` to redo test evals from `ckpt.pt`). GPU idle.
  Days to fallback date (2026-11-15): 21.

---

## 2026-10-25 P9 weight decay — runs/step1/p9_wd0_controller_s0

Hypothesis (from the best-vs-final checkpoint comparison of s1, s2, P8): the post-peak drift is
weight decay (0.1) shrinking the token embeddings. Between peak and end the embedding norm
falls 1338 → 1284 on all three drifting runs while backbone (58), head (~300) and the memory
projections (which grow) do not shrink; at vocab 2048 the embedding norm is unchanged
(641 → 641) and there is no drift. A row of the embedding only gets gradient when its token
appears, and at vocab 8192 each token appears 4x less often than at 2048, so decay wins for
rare rows and the keys lose contrast. This would also explain the vocab cliff (P1 vs P1b).
Config: target.yaml --set train.weight_decay=0 --set model.fast=true, controller, seed 0,
vocab 8192.
Prediction if true: same breakthrough (step 2000-2500), peak ≥ 18 % and **no decline**:
val recall at 8000 within 2 pp of the peak, embedding norm flat after step 3000. If the
drift persists with wd = 0, the mechanism is elsewhere (schedule, hard gates).
Result:
Decision:

## 2026-10-25 P1b vocab 4096 — runs/step1/p1b_vocab4096_controller_s0

Hypothesis: the vocab effect seen in P1 is graded (32 % at 2048, 18 % at 8192), so 4096, the
smallest vocab that allows the protocol's L=2048 test, lands in between and without drift.
Config: target.yaml --set data.vocab_size=4096 --set model.fast=true, controller, seed 0.
Prediction if true: val recall 22-30 % at 8000 and within 3 pp of its peak (no drift). If it
lands at 17-19 % with drift, the effect is a cliff between 2048 and 4096.
Result:
Decision:

## 2026-10-25 C1b recency seeds — runs/step1/c1_target_recency_s{1,2}

Hypothesis: recency at the target is at chance for every seed (completes the confirm stage).
Config: target.yaml --set model.fast=true, recency, seeds 1 and 2.
Prediction: val recall < 0.5 % at 8000 for both.
Result:
Decision:

## 2026-10-25 P8 write-gate init — runs/step1/p8_wbias2.2_controller_s0

Hypothesis: the ~2000-step plateau is a cold start of the scalar write gate: at init
sigmoid(0) = 0.5 and the fastest early loss reduction is to stop writing (reads become clean
zeros), so the memory stays empty until the gate reopens. Starting the gate open avoids it.
Config: target.yaml --set model.write_bias_init=2.2 (sigmoid 0.9) --set model.fast=true,
controller, seed 0, 8000 steps, vocab 8192 (so it is comparable to P0 / C1).
Prediction if true: write_frac > 0.05 at step 500 and val recall > 5 % by step 1500 (vs 2500
without it). Secondary: whether the peak-then-drift shape changes.
Result: write_frac **0.685 at step 500** (gate open, memory full) but val recall at chance;
by step 1000 write_frac **0.000** (the optimizer closed it); reopens at 2000 (write_frac 0.096,
3.5 %); 17.7 / 19.3 / 17.8 % at 2500 / 3000 / 3500; drift to 13.1 % at 8000. Test 13.1 / 10.3 /
8.7 % at L = 512 / 1024 / 2048. 22 min on the fast path.
Decision: **prediction failed.** The plateau is not a cold start of the gate: with the gate
open and the memory full the loss gradient still shuts writing off, and it comes back on the
same schedule. So the ~2000 steps are spent elsewhere (the read/key/value projections, or the
backbone learning to route the key token), not on the gate. Drop write_bias_init from the
method (keep the option, default 0). The plateau is a "why" for the paper's discussion, not
something to engineer away before the sweep.

## 2026-10-25 P1 vocab confound — runs/step1/p1_vocab2048_controller_s0

Hypothesis: the target fails because vocab 8192 (vs 2048 in pilot16) spreads the key/value
embeddings over 4x more tokens for the same 100k examples, so the association signal per
token is too weak for the memory path to engage.
Config: target.yaml --set data.vocab_size=2048, controller, seed 0, 8000 steps.
Prediction if true: val recall > 5 % by step 2500 (pilot16 broke through at 1400 of 4000 with
loss 6.9 → 6.2). If false: flat at chance through step 4000 → kill.
Result: write_frac 0.000 through step 1500 (writes nothing during the plateau; loss at
ln(1024) = 6.93); gate opens at 2000; 17.6 % at 2500 (same step as vocab 8192), then 22.3 /
29.4 / 31.5 / 31.7 / 31.6 / 31.9 / 32.0 % at 3000-6000, **32.0 % at 8000, no drift**. Loss 4.65,
gnorm 0.5 at the end. Writes land on pair positions: queried 0.32 / distractor 0.33 / update
0.29 / other 0.01. Test **32.1 % at L=512 and 32.1 % at L=1024** (no loss at 2x length).
L=2048 is **not possible**: `MQARConfig.validate` requires vocab_size > seq_len, so the
protocol's longest test length needs vocab ≥ 4096.
Decision: prediction half right. Vocab does not shorten the plateau, but it sets the level:
32 % and stable vs 18 % peak and drift at 8192. Two readings to separate with one probe,
P1b `--set data.vocab_size=4096` (the smallest vocab that allows L=2048): if 4096 lands near
32 %, the 8192 result is a representation-learning limit of the 2-layer backbone at this data
budget and the sweep should use 4096 (protocol change to state in the paper; not part of
H1/H2); if 4096 lands near 18 %, the vocab effect is a cliff and needs its own explanation.
Either way the drift at 8192 remains a step-3 question.

Candidate probe P8: `--set model.write_bias_init=2.2` (sigmoid 0.9 at init) so the memory
is populated from step 0; predicted to remove most of the 2000-step plateau. Code landed
(commit a1de51c), launch after C1 s2.

## 2026-10-25 C1 confirm seeds — runs/step1/c1_target_controller_s{1,2}

Hypothesis: P0's breakthrough is not a seed accident.
Config: target.yaml, controller, seeds 1 and 2, 8000 steps, `model.fast=true` (same numerics,
0.23 s/step). Recency seeds 1, 2 follow once the GPU frees up.
Prediction if true: both seeds > 5 % val by step 3500 and > 15 % by 8000; recency stays at chance.
Result: s1 (fast path, 0.24 s/step, 33 min): chance to 1500, gate opens at 2000 (write_frac
0.033), 16.2 / 18.1 / 16.9 % at 2500 / 3000 / 3500, then 12.9 → 11.5 % by 8000; write_frac flat
at 0.10, filled 0.90-1.00, slot entropy 5.0. Test 11.7 / 9.4 / 8.0 % at L = 512 / 1024 / 2048.
s2 (30 min): gate opens at 2000, 16.8 / 18.8 / 17.1 % at 2500 / 3000 / 3500, then 14.9 → 13.7 %
at 8000. Test 13.6 / 10.7 / 9.0 %. Both processes died in the L=2048 test eval (OOM, 4 GB
logits at eval batch 128); evals redone with `--eval_only`; the eval batch is now scaled by
length. The s1 checkpoint evaluated on the eager path gives the same three numbers to 4
decimals (`runs/checks/s1_eager_eval`), so the fast path is not the reason P0's length curve
is flatter than s1/s2's.
Decision: prediction held for "> 5 % by 3500" on both seeds and for recency at chance; the
"> 15 % at 8000" half failed on all three seeds because of the drift. **"Learning" is met on
3/3 seeds; "solved" (> 10 pp over recency at end of schedule) is met numerically on 3/3
(11.4 / 11.5 / 13.7 vs 0.03).** Step 1 is answered: the proposal's target is learnable with the
2-layer / lr 3e-4 setting. Before freezing `target.yaml` for the sweep, run P1b (vocab) and
address the drift in step 3, because a sweep whose controller loses a third of its peak by
the end of the schedule would make H1 depend on where the schedule stops.

## 2026-10-25 P0 target baseline, aligned settings — runs/target/{controller,recency}_s0

Hypothesis: the earlier `pilot` failure at the target came from the 4-layer / lr 1e-3 settings;
the settings that learned in pilot16 (2 layers, lr 3e-4) will learn at the target too.
Config: target.yaml (64 + 16 + 16 associations, 32 slots, len 512, vocab 8192), seed 0,
8000 steps. Launched before the instrumentation, so the logs have no timestamps or gate stats.
Prediction if true: controller > 5 % val by step 3000.
Result: recency: chance for all 8000 steps (val 0.03 %, loss 8.23; test 0.03 % at every
length). controller (105 min on the eager path): chance through 2000 (0.3 %), **16.9 % at 2500,
18.5 % at 3000, 18.1 % at 3500**, then 12.2 → 11.4 % at 8000; running loss rose from 6.72 (step
3250) to 7.41 (step 5900) with gnorm 0.3-0.5 (no spike): a slow drift, not a blow-up. Test
11.4 / 11.2 / 11.1 % at L = 512 / 1024 / 2048 (flat in length). No gate stats (pre-instrumentation).
Decision: prediction held (one eval late). The proposal's target learns with 2 layers / lr 3e-4;
the `pilot` failure was the 4-layer / lr 1e-3 setting. Step 1 moves to the **confirm stage**
(seeds 1, 2 at target, fast path) instead of P2; P2 (slots=128) is dropped unless the confirm
seeds disagree. P1 keeps running because it also measures whether vocab shortens the plateau.
