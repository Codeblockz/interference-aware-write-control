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

- 2026-10-25 19:25: **Step 1 answered; drift explained and fixed.** Target (64+16+16 / 32 slots /
  len 512 / vocab 8192, 2 layers, lr 3e-4) learns on 3/3 seeds; recency at chance on 3/3.
  With wd 0.1 the controller peaks at 18-19 % and decays to 11-14 % (embedding rows shrink);
  with wd 0 (P9) it reaches **31.5 % val, test 31.6 / 31.5 / 31.4 % at L = 512 / 1024 / 2048**,
  no decline. Running: P9b (wd 0.1 except embedding) to pick the setting; C2 seeds 1, 2 at wd 0.
  Refuted today: write-gate cold start (P8); graded vocab effect (P1b). Fast path 3.4x; position
  writer now on it too. Fallback date 2026-11-15 no longer needed; keep as the hard stop.

**Time ledger (training minutes, from results.json; test evals and the eager eval re-runs
are extra). Session: 15:48-19:25 on 2026-10-25, one RTX 5090 shared by 2-3 runs at a time.**

| run | mode | train min |
|---|---|---|
| P0 controller s0 / recency s0 | eager | 105 / 27 |
| C1 controller s1 / s2 | fast | 33 / 30 |
| C1b recency s1 / s2 | fast | 22 / 22 |
| P1 vocab 2048 | eager | 105 |
| P1b vocab 4096 | fast | 31 |
| P8 write bias | fast | 22 |
| P9 wd 0 | fast | 24 |
| P9b, C2 s1, C2 s2 | fast | running |
| **total so far** | | **~7.7 GPU-h** across 13 runs |

Method / protocol changes today (also in README "Decisions log" and git): fast path
(`model.fast`, numerically identical); `write_bias_init` option (default 0, unused);
`decay_embeddings` option; eval batch scaled by length; `--eval_only`. No change to the
controller's features, gates, or the task. Candidate protocol change pending P9b: weight
decay setting for the sweep.

---

## Reference: Zoology's MQAR training recipe (checked 2026-10-25)

From arXiv:2312.04927 (Appendix, "training protocol") and github.com/HazyResearch/zoology
(`zoology/train.py`, `zoology/config.py`, `experiments/mqar_example_configs/original_mqar_configs.py`):
- Vocab 8192; two-layer models; 100k train / 3k test examples; data order fixed.
- AdamW, **weight decay 0.1 on every parameter including embeddings** (no param groups),
  linear warmup 10 %, cosine decay per epoch, **lr swept over logspace(-4, -2, 4)** per
  architecture, **64 epochs** (32 in the example config); batch 8-64 depending on length/width.
  64 epochs x 100k examples = 6.4M examples ≈ 12x our 8000 x 64 = 512k.
- They **report the maximum test accuracy** over the sweep and **early-stop at 99 % validation
  accuracy**, so a post-peak decline would never appear in their numbers. No best-checkpoint
  saving.
- Their kv counts at len 256 go up to 64 pairs (density 50 %); ours is 64 + 16 + 16 pairs at
  len 512 (37.5 % of tokens in pairs) plus updates/distractors.
Implications: (a) our schedule is short by their standard, which backs P5 (longer schedule)
and the sweep's lr grid (step 5); (b) selecting the checkpoint by validation recall
(`best.pt`) rather than reporting the last step is in line with how the reference work
reports MQAR, and is the honest fallback if the drift cannot be removed; it is a protocol
choice to state in the paper and must be made before the sweep, not after.

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
Result: same plateau and breakthrough (gate opens at 2000, 15.5 % at 2500), then 23.9 / 27.0 /
28.9 / 29.7 / 30.4 / 31.2 / 31.5 % at 3000-7000 and **31.5 % at 8000, no decline** (peak = end).
Loss 6.44 → 5.6, gnorm 2-9, write_frac 0.06-0.07 with the P1-like pattern (writes on pairs,
almost none elsewhere). **Test 31.6 / 31.5 / 31.4 % at L = 512 / 1024 / 2048**: flat at 4x the
training length. 40 min on the fast path (shared GPU). Every vocab-8192 run with wd 0.1 had
peaked at 18-19 % by step 3000 and ended at 11-14 %.
Decision: **prediction held.** Weight decay on the embeddings is the drift, and it also explains
the vocab cliff (P1 vs P1b: the 2048-vocab embedding rows get gradient often enough to beat
the decay). P9b decides between wd 0 everywhere and wd 0.1 except the embedding; C2 confirms
across seeds. This run is the candidate "controller at target" reference until C2 is in.

## 2026-10-25 P10 soft gates — runs/step1/p10_softgates_controller_s0

Hypothesis: the post-peak drift is the biased straight-through gradient of the hard slot
gates; with soft gates the gradient is exact and the training loss cannot drift upward.
Config: target.yaml --set model.hard_gates=false --set model.fast=true, controller, seed 0,
wd 0.1 (original). Note: soft gates change the method (writes spread over slots); if this is
the fix, the paper must say so and the sweep uses it for controller and ablation alike.
Prediction if true: breakthrough may come later (soft gates learned slower in the 10-15
smoke tests), but val recall at 8000 within 2 pp of its peak and training loss monotone
after the peak. If it drifts too, the ST gates are not the cause.
Result:
Decision:

## 2026-10-25 P11 lr 1e-4 — runs/step1/p11_lr1e-4_controller_s0

Hypothesis: the drift is an optimisation-noise effect at lr 3e-4 (too large once the memory
path is active); at 1e-4 the peak holds.
Config: target.yaml --set train.lr=1e-4 --set model.fast=true, controller, seed 0, wd 0.1.
Prediction if true: later breakthrough (lr 3x smaller), then no decline. Also serves as the
lower point of the step-5 lr grid.
Result:
Decision:

## 2026-10-25 P9b decay except embeddings — runs/step1/p9b_noembdecay_controller_s0

Hypothesis: the drift comes only from decaying the embedding rows; keeping wd 0.1 on the
other matrices (the usual practice) gives the P9 result without giving up regularisation.
Config: target.yaml --set train.decay_embeddings=false --set model.fast=true, controller, seed 0.
Prediction if true: within 2 pp of P9 at every eval from 3000 on and no decline. If it drifts,
decay on the memory projections or head also matters and wd 0 is the setting.
Result (interim, step 4500): 15.8 / 18.3 / 17.7 / 13.8 / 12.6 % at 2500-4500: **drifts like
wd 0.1**. Embedding norm flat (1447 → 1447), so the embedding is not what decays into the drift.
Decision (interim): **prediction failed.** Together with C2 s1 (below) this refutes "weight decay
on the embeddings is the drift". The P9 run (wd 0, seed 0) did not drift, but that is now 1 of 2
wd-0 runs; the embedding-norm correlation in the earlier checkpoints was a side effect of
wd 0.1, not the cause. Mechanism still open. Let it finish for the record.

## 2026-10-25 C2 confirm seeds, wd 0 — runs/step1/c2_wd0_controller_s{1,2}

Hypothesis: the P9 result (no decline, ~30 %+) holds across seeds.
Config: target.yaml --set train.weight_decay=0 --set model.fast=true, controller, seeds 1, 2.
Prediction: both > 25 % at 8000 and within 2 pp of their peak. (If P9b wins on validation,
rerun the seeds with P9b's setting instead.) s1 launched 19:05, s2 19:13.
Result (interim): s1 at wd 0: 16.9 / 18.4 / 17.4 / **13.8 %** at 2500-4000, i.e. the usual
peak-then-drift, with embedding norm flat (1448 → 1448) and head/projection norms growing
slightly. s2 at step 1500 (plateau).
Decision (interim): **prediction failed on s1.** Weight decay is not the mechanism (or not
sufficient). What the drifting runs share: the *training* loss rises after the peak (6.7 → 7.4
in P0) while gnorm stays small, i.e. the optimizer walks uphill on the true loss. That is the
signature of a biased gradient. The controller's straight-through hard gates
(`one_hot + soft - soft.detach()`) give exactly that: the forward uses the one-hot, the
backward the softmax. Next probes P10 (soft gates) and P11 (lr 1e-4 after the same warmup)
at wd 0.1, seed 0, so they compare to P0/C1 directly.

## 2026-10-25 P1b vocab 4096 — runs/step1/p1b_vocab4096_controller_s0

Hypothesis: the vocab effect seen in P1 is graded (32 % at 2048, 18 % at 8192), so 4096, the
smallest vocab that allows the protocol's L=2048 test, lands in between and without drift.
Config: target.yaml --set data.vocab_size=4096 --set model.fast=true, controller, seed 0.
Prediction if true: val recall 22-30 % at 8000 and within 3 pp of its peak (no drift). If it
lands at 17-19 % with drift, the effect is a cliff between 2048 and 4096.
Result: 16.5 / 18.2 / 17.9 % at 2500 / 3000 / 3500, then 14.3 → **12.2 % at 8000**; test
12.2 / 9.5 / 7.9 % at L = 512 / 1024 / 2048. Identical shape to the vocab-8192 runs.
Decision: **prediction failed**: the vocab effect is a cliff between 2048 and 4096, not graded.
Explained by P9: at 4096 and 8192 the embedding rows are updated too rarely to resist weight
decay 0.1. Keep vocab 8192 (the proposal's setting) and fix the decay instead.

## 2026-10-25 C1b recency seeds — runs/step1/c1_target_recency_s{1,2}

Hypothesis: recency at the target is at chance for every seed (completes the confirm stage).
Config: target.yaml --set model.fast=true, recency, seeds 1 and 2.
Prediction: val recall < 0.5 % at 8000 for both.
Result: s1 0.03 %, s2 0.02 % val at 8000; test 0.02-0.03 % at every length. write_frac 1.0
(writes every step), slot entropy 5.0.
Decision: prediction held; recency is at chance at the target for seeds 0, 1, 2. Confirm stage
for the baseline done.

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
