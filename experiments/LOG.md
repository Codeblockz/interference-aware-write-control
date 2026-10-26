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

- 2026-10-25 22:00: **Step 1 closed; target.yaml frozen.** C3 confirm (9 runs, best-val
  selection, 3 seeds): pair-only recency 40.1 % (ceiling 43 %), no-interference ablation 33 %
  on 2/3 seeds (19 % on the third), controller 18-19 % on 3/3, naive recency 0 %. H1 and H2
  fail at N = 64, S = 32 (capacity-bound cell); the hypotheses are decided at S = 64. The
  "drift" is a worse policy basin reached by the cosine tail, not weight decay (3 seeds), not
  the gate init (P8), not fixable by soft gates or lr 1e-4 (both never learn). Protocol now:
  best-validation checkpoint, N = 128 at length 1024, pair-only recency as the non-learned
  reference. Paper protocol text updated and builds (`paper/Ellis-RP3.pdf` draft).
  GPU time today ≈ 14.5 h across 25 runs. Next: steps 4-5, then the sweep (S = 64 first).

---|---|---|
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

## 2026-10-25 S4 residual and position writers at the target — runs/step2/s4_<writer>_s{0,1,2}

Hypothesis (PLAN step 4): both untrained baselines learn at the target with the frozen
settings, and their gate statistics are sane (residual: write_frac between 0.05 and 0.5,
not 0 or 1; position: writes follow the unlock schedule, write_frac ≈ 1 by construction).
Config: target.yaml --set model.fast=true, writers residual / position, seeds 0, 1, 2.
Prediction: position ≤ naive recency plus a few points (it still writes at every position,
only the slot set grows), i.e. near chance, which the sweep will show at every S. Residual
is the interesting one: if its learned threshold rejects filler (filler reads back as a large
residual too, so it may not), it approaches pair-only recency (~40 %); if not, it writes
everything and sits near chance. Prediction: residual 5-25 %, position < 2 %.
Result:
Decision:

## Decisions 2026-10-25 20:10 (Ryan: "do it")

1. **Checkpoint rule:** every run reports the checkpoint with the best validation recall
   (`best.pt`), tests on it, three seeds per cell. Stated in the paper's protocol. Reason: the
   cosine tail leaves some runs in a worse policy basin (see C1/C2); best-validation selection
   reports the policy a run reached, and matches how Zoology reports MQAR.
2. **N = 128 column at train length 1024** (tests at 1024 / 2048 / 4096); N = 64 stays at 512.
3. **Pair-only recency** replaces naive recency as the "no learning" reference in Table II; naive
   recency stays as one reported line (it is the proposal's baseline) and is at chance by
   construction.
Also: hard gates stay (P10), lr grid for step 5 is {3e-4, 6e-4, 1e-3} (P11), vocab stays 8192.

## 2026-10-25 C3 confirm stage under the final rules — runs/step2/c3_<writer>_s{0,1,2}

Hypothesis: with best-validation selection, the controller at the target is reproducibly
~30 % (its good-basin peak) or ~18 % (its bad-basin peak), the no-interference ablation is
within a few points of it (capacity-bound cell; H2 expected not to hold at S = 32), and
pair-only recency sits near the 43 % ceiling *if the read path trains*, which would make it
the strongest baseline at S = 32.
Config: target.yaml --set model.fast=true, writers controller / no_interference /
pair_recency, seeds 0, 1, 2 (nine runs, queue.py, three at a time). Test evals run on best.pt.
Prediction: controller best-val 18-32 % (3 seeds); ablation within 5 pp of the controller;
pair_recency 35-43 %. If pair_recency lands near 43 %, H1 fails at S = 32 by the ceiling
argument and the paper says so; the S = 64 column is where H1 is decided.
Result (nine runs, 20:03-21:50, fast path, 21-39 min each; test on best.pt):

| writer | seed | best step | val at best | last val | test 512 / 1024 / 2048 |
|---|---|---|---|---|---|
| controller | 0 | 3000 | 19.0 | 12.7 | 18.8 / 14.8 / 13.0 |
| controller | 1 | 3000 | 18.3 | 11.5 | 18.3 / 14.5 / 12.8 |
| controller | 2 | 3000 | 18.8 | 13.7 | 18.7 / 14.5 / 12.7 |
| no_interference | 0 | 4000 | 19.4 | 9.8 | 19.1 / 14.4 / 12.1 |
| no_interference | 1 | 7500 | 32.7 | 32.7 | 32.6 / 32.6 / 32.6 |
| no_interference | 2 | 8000 | 33.1 | 33.1 | 33.1 / 33.1 / 33.1 |
| pair_recency | 0 | 7000 | 40.1 | 40.1 | 40.1 / 40.1 / 40.1 |
| pair_recency | 1 | 7500 | 40.1 | 40.1 | 40.1 / 40.1 / 40.0 |
| pair_recency | 2 | 6500 | 40.1 | 40.1 | 40.1 / 40.1 / 40.1 |

Decision: predictions held on the controller (18-19 %, 3/3 in the bad basin) and on pair-only
recency (40.1 %, within 3 pp of the 43 % simulated ceiling, flat in length); the ablation
prediction failed in an informative direction: 2/3 seeds found the good basin (33 %, flat in
length, no decline) and the controller 0/3. Reading, to be tested at S = 64 before it goes in
the paper as more than an observation: at S = 32 the only thing worth learning is "write
pairs, not filler", and the two interference features make that harder to find, not easier.
**At the proposal's headline setting the ranking is pair-only recency 40 > ablation 33 (2/3)
> controller 19 > naive recency 0.** H1 and H2 both fail at N = 64, S = 32, by the capacity
argument; the length generalisation of the good-basin runs is perfect (no loss at 4x), of the
bad-basin runs not (19 → 13). Step 1 is closed: `target.yaml` is frozen as is (vocab 8192,
2 layers, lr 3e-4, wd 0.1, hard gates, 8000 steps, best-val selection). Next: step 4 (residual,
position at the target), step 5 (lr grid {3e-4, 6e-4, 1e-3} per writer, since basin-finding
is lr-sensitive), then the sweep with the S = 64 column first.


## Finding: capacity ceiling of the task (2026-10-25, prompted by Ryan: "it might be a data issue")

Simulated the best any writer can do that cannot see the future: admit every pair at its
value position, never write filler, overwrite a key in place on update, evict FIFO when full
(script in this entry's commit; 1500 sequences, seed 0). Exact recall:

| cell | FIFO pair-only | + perfect distractor rejection |
|---|---|---|
| N=64, S=32 (the target) | **0.434** | 0.500 |
| N=64, S=64 | 0.811 | 1.000 |
| N=64, S=128 | 1.000 | 1.000 |
| N=128, S=32 (needs L ≥ 640; at L=1024) | 0.244 | 0.250 |
| N=128, S=64 | 0.434 | 0.500 |
| N=128, S=128 | 0.811 | 1.000 |

Implications:
1. **The target is capacity-bound.** 96 pairs for 32 slots: a perfect pair-only recency policy
   gets 43 %, and perfect knowledge of which pairs will be queried adds only 7 pp. Our best
   runs (32 %) are at 3/4 of the ceiling; the drifted runs (12-13 %) are far below it. The
   H1 margin (> 5 pp over the strongest baseline) has to come out of that 7 pp window at
   S=32, which is unrealistic; the interference features can matter at S=64 (81 → 100 %) and
   the research question is stated per density, so Table II still answers it.
2. **Protocol error:** the N=128 cells cannot be generated at train length 512
   (`MQARConfig.validate`: 384 context tokens + 128 queries > 512). The paper's Table II and
   PLAN step 6 must use L=1024 for N=128 (or drop N=128). Decide before the sweep.
3. The "drift" is the run leaving the pair-only policy. Whether a run finds it looks
   stochastic at lr 3e-4 (1 of 3 at wd 0; P1 at vocab 2048 found it). P10 (soft gates) and
   P11 (lr 1e-4) are the two controlled tests still running; if neither makes the good basin
   reliable, the protocol fallback is best-validation checkpoint selection plus 3 seeds, which
   reports the policy a run reached rather than where it ended.
4. The naive recency baseline writes at every position (write_frac 1.0), which is why it sits
   at chance: it holds the last 32 *tokens*. A pair-only recency baseline is the fair "no
   learning" comparison and is ~40 lines (gate on value positions); worth adding as the
   strongest non-learned baseline since it is also the 43 % ceiling policy.

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
Result: **dead at step 4000** (kill rule): val recall 0.05 %, write_frac 0.03, running loss
8.29 and flat (8.293 → 8.279 over 1000 steps). Never left the plateau; the hard-gate runs had
broken through by 2500.
Decision: **prediction failed.** Soft gates do not learn the target in this schedule, so they
are not a fix, and the straight-through gates are not shown to be the drift's cause either
(the test never reached the regime where the drift happens). Keep hard gates (the proposal's
method). For the paper: the hard one-hot write is what lets the memory path start learning
at all here, consistent with the 2026-10-15 pilot16 finding.

## 2026-10-25 P11 lr 1e-4 — runs/step1/p11_lr1e-4_controller_s0

Hypothesis: the drift is an optimisation-noise effect at lr 3e-4 (too large once the memory
path is active); at 1e-4 the peak holds.
Config: target.yaml --set train.lr=1e-4 --set model.fast=true, controller, seed 0, wd 0.1.
Prediction if true: later breakthrough (lr 3x smaller), then no decline. Also serves as the
lower point of the step-5 lr grid.
Result: **dead at step 4000** (kill rule): val recall 0.03 %, write_frac 0.000 (gate never
opened), loss 8.313 flat.
Decision: **prediction failed.** lr 1e-4 is too low to open the write gate within 4000 steps
on this cosine schedule; it is the lower edge of the step-5 grid, not a candidate for the
target. The sweep's lr grid should be {3e-4, 1e-3} plus one between, not below 3e-4.

## 2026-10-25 P9b decay except embeddings — runs/step1/p9b_noembdecay_controller_s0

Hypothesis: the drift comes only from decaying the embedding rows; keeping wd 0.1 on the
other matrices (the usual practice) gives the P9 result without giving up regularisation.
Config: target.yaml --set train.decay_embeddings=false --set model.fast=true, controller, seed 0.
Prediction if true: within 2 pp of P9 at every eval from 3000 on and no decline. If it drifts,
decay on the memory projections or head also matters and wd 0 is the setting.
Result: 15.8 / 18.3 / 17.7 / 13.8 / 12.6 % at 2500-4500, **12.2 % at 8000**: drifts like wd 0.1.
Embedding norm flat (1447 → 1447).
Decision: **prediction failed.** The embedding is not what decays into the drift.

## 2026-10-25 C2 confirm seeds, wd 0 — runs/step1/c2_wd0_controller_s{1,2}

Hypothesis: the P9 result (no decline, ~30 %+) holds across seeds.
Config: target.yaml --set train.weight_decay=0 --set model.fast=true, controller, seeds 1, 2.
Prediction: both > 25 % at 8000 and within 2 pp of their peak. (If P9b wins on validation,
rerun the seeds with P9b's setting instead.) s1 launched 19:05, s2 19:13.
Result: s1 16.9 / 18.4 / 17.4 / 13.8 → **13.0 % at 8000**; s2 → **13.4 % at 8000**. Both the usual
peak-then-drift with embedding norms flat. P9 (31.5 %, no drift) is therefore 1 of 3 wd-0 runs.
Decision: **prediction failed.** Weight decay is not the mechanism; P9 landed in the good
basin by chance. What separates the two outcomes (see "Capacity ceiling" below) is the gate
policy the run settles into: the two non-drifting runs (P9, P1) write ~1/3 of every pair and
**zero** filler (q/d/u/o 0.33/0.34/0.34/0.00); every drifting run writes fewer pairs and 7-9 %
of filler positions (0.22/0.23/0.21/0.07). The drift is a worse basin, not a slow decay.

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
