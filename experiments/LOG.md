# Experiment log

Lab notebook for the RP3 experiments. One entry per launched run or pair, newest first.
Write the hypothesis and the numeric prediction **before** launching; fill Result and
Decision after. Never edit a prediction. Read `PLAN.md` for the step list and the plan file
for the probe tree (P0-P7) behind step 1.

Tools: `python experiments/progress.py runs/step1 --watch 300` (status table with ETA),
`python experiments/curves.py runs/step1` (loss / recall / gnorm / write-fraction curves),
`--set key=value` on `run.py` for one-variable probes, `--resume` after a kill,
`--init_from <best.pt>` for curriculum.

## Time ledger for replication

Regenerate the per-run cost table with `python experiments/analysis/cost.py runs --md`; it reads
every results.json (runs from before 2026-10-25 are marked `eager*`: timing, throughput, eval
wall and peak memory only, no timestamps / GPU name / commit; all were on the same RTX 5090).

Session spans from git commit times (work time, not GPU time):

| date | span | what |
|---|---|---|
| 2026-10-15 | 22:17-23:55 | scaffold, data, model, first learnability fixes |
| 2026-10-16 | 00:18-00:54 | pilot16 runs, controller cold-start fix |
| 2026-10-24 | 12:47-21:21 | RP2 paper to LaTeX, placeholders, block diagram |
| 2026-10-25 | 15:55-23:05+ | step 1 search, instrumentation, fast path, confirm stages, steps 4-5 |

Pre-instrumentation GPU runs (from results.json): pilot16 controller 16 min, ablation 15 min,
recency 7 min (4000 steps, T=128); `pilot` controller (4 layers) did not finish, recency 32
min; P0 controller 105 min, recency 27 min (eager, T=512, 8000 steps). 2026-10-25 GPU total
≈ 20 h across 37 finished runs by 23:00, plus the twelve S5 runs in flight.

## Definitions (step 1)

- **chance**: val recall < 0.5 % (loss ≈ ln(vocab/2); 8.32 at vocab 8192, 6.93 at 2048).
- **learning**: val recall > 5 % and rising over two consecutive evals.
- **solved**: controller beats recency by > 10 pp val recall at the target, 3 seeds, none collapsed.
- **kill rule**: still at chance at 50 % of the schedule with running loss flat (< 0.02 change
  over the last 1000 steps) → kill, log "dead at step X". Exception: P5 runs to completion once.
- Probes read **validation** only. Test-length numbers stay unread until the sweep.

## Status

- 2026-10-26 23:00: **Sweep N = 64 column done (54 runs) and the S6b equal-lr check (12
  runs).** H1 fails at every S (pair-only recency 70 / 100 vs controller 63.5 / 66.8 at
  S = 64 / 128). H2 not supported: at equal lr the ablation beats the controller at both S
  (65.8 vs 63.5; 96.0 vs 66.8); the sweep's +17.9 pp at S = 64 was the ablation's one-seed lr
  pick. N = 128 column running one at a time since 22:48 (~38 h). Paper Table II / Fig. 2 /
  results text filled for N = 64. Next: fill the N = 128 column, delay sweep (PLAN 6), then
  the discussion.
- 2026-10-26 01:20: **Steps 4 and 5 closed; per-writer settings frozen in
  `experiments/configs/selected.yaml`.** S5 grid (12 runs, 23:01-01:18, one seed): controller
  lr 3e-4, hidden 32 (32.5 % val, monotone to step 8000, vs 19 % at hidden 64); ablation lr
  1e-3 (27.7 %), hidden 32 to match the controller; residual lr 6e-4 (37.2 %, writes 94 % of
  pairs and 6 % of filler); pair-only recency flat at 40 % at every lr; position 0 % at every
  lr (writes every position). All single-seed; the sweep's three seeds are the confirmation.
  Next: step 6, the sweep, S = 64 column first, through `jobq.py`; N = 128 at length 1024.
- 2026-10-25 22:00: **Step 1 closed; target.yaml frozen.** C3 confirm (9 runs, best-val
  selection, 3 seeds): pair-only recency 40.1 % (ceiling 43 %), no-interference ablation 33 %
  on 2/3 seeds (19 % on the third), controller 18-19 % on 3/3, naive recency 0 %. H1 and H2
  fail at N = 64, S = 32 (capacity-bound cell); the hypotheses are decided at S = 64. The
  "drift" is a worse policy basin reached by the cosine tail, not weight decay (3 seeds), not
  the gate init (P8), not fixable by soft gates or lr 1e-4 (both never learn). Protocol now:
  best-validation checkpoint, N = 128 at length 1024, pair-only recency as the non-learned
  reference. Paper protocol text updated and builds (`paper/Ellis-RP3.pdf` draft).

Method / protocol changes 2026-10-25 (also in README "Decisions log" and git): fast path
(`model.fast`, numerically identical); `write_bias_init` option (default 0, unused);
`decay_embeddings` option; eval batch scaled by length; `--eval_only`; best-validation
checkpoint selection; pair-only recency writer added. No change to the controller's
features, gates, or the task. Per-run GPU time: `python experiments/analysis/cost.py runs --md`.

---

## 2026-10-27 S7p key-skew pilot (worktree key-skew-test) — runs/skew/<writer>_k<skew>_s0

Why (Ryan, 23:30: "can we test that theory, with a small test"): the caching argument says a
learned admission policy can only beat pair-only FIFO when something at write time predicts
future reuse. In MQAR as generated so far nothing does: keys are uniform over the key pool and
distractor keys are drawn from the same pool, so a distractor is indistinguishable from a
queried pair when it arrives (note: `power_a` shapes query *delay*, not key frequency; the
"skew" I described to Ryan at 23:20 as already present was wrong). New generator option
`data.key_skew` (default 0 = unchanged): queried keys drawn without replacement with
p ∝ rank^-skew over the key pool (Gumbel-top-k), distractor keys uniform over the remaining
keys, i.e. cold. At skew 2 the queried keys have mean rank 120 of 1023 and 43 % come from the
top 10; distractors have mean rank ~510. A writer can therefore learn "hot key → admit, cold
key → skip" from the key embedding alone.
Config: pilot16.yaml (16 + 4 + 4 associations, 16 slots, length 128, vocab 2048, 4000 steps,
eager path, ~16 min, ~2 GB), controller hidden 32, lr 3e-4 for every writer (no per-writer
selection at this scale), seed 0; skew ∈ {0, 1, 2}; writers pair_recency, controller,
no_interference. Nine runs, one at a time with ≥ 8 GB free, beside the N = 128 column.
Capacity: 24 writes into 16 slots; pair-only FIFO keeps the last 16 (the 4 updates plus 12 of
the 20 context pairs, ~2.4 of them distractors), so perfect distractor rejection is worth
about 2.4 slots ≈ 15 pp and in-place update handling a few more.
Prediction: at skew 0 the ordering of the sweep (pair_recency > learned writers, learned
≈ 50 % as in pilot16). At skew 2 both learned writers beat pair_recency by ≥ 10 pp, with
filler/distractor admission near 0 and pair admission near 1; at skew 1 in between. If the
learned writers do not overtake FIFO even at skew 2, the admission failure is not about
information and the caching explanation is wrong for this model.
Found on the way: filler tokens are drawn with the *global* torch RNG (`torch.randint` with
no generator in `generate_mqar`), so a split is reproducible only through `run.py`'s
`torch.manual_seed(seed)` and the fixed call order. Train-length test sets are generated
before training and are identical across writers (the paired tests at the training length
are valid); the 2x / 4x test sets are generated after training, so their filler differs
across writers while keys, values and query positions are identical. Not changed here; to be
fixed with a seeded `torch.Generator` in the main checkout and noted in the paper's
protocol (pairing at the long lengths is on associations, not on filler).
Result (9 runs, 23:29-00:51, 0 failures; test recall at 128; admission at the best checkpoint,
which was the last step for every learned run):

| writer | skew 0 | skew 1 | skew 2 | admission at skew 2: pairs / distractors / filler |
|---|---|---|---|---|
| pair_recency | 72.4 | 72.5 | 72.7 | 1.00 / 1.00 / 0 (by construction) |
| controller | 51.6 | 58.9 | 63.7 | 0.69 / 0.65 / 0.007 |
| no_interference | 52.2 | 54.7 | 63.7 | 0.69 / 0.69 / 0.006 |

Prediction failed. Skew helps the learned writers (+12 pp from skew 0 to 2, both writers
alike) but they stay 9 pp under pair-only recency, and the gate statistics say the gain is
not from admission: at skew 2 the controller admits 69 % of queried pairs and 65 % of
distractors, i.e. it does not tell hot keys from cold ones even though 43 % of queried keys
come from the top 10 of 1023 and distractors sit around rank 500. The improvement with skew
must come from the read / backbone side (fewer distinct keys to represent). So the admission
policy is not using the one signal that would let it beat FIFO.
Caveats: one seed; 4000 steps with validation still rising at the last eval for all six
learned runs (best checkpoint = step 4000), so "does not learn it" and "has not learned it
yet" are not separated. The write gate sees a 16-dim content projection of the token, which
may be too narrow to carry key identity; this is a model-side limit, not an information one.
Decision: one follow-up before concluding, three runs at 12 000 steps (controller and
ablation at skew 2, controller at skew 0 as the control), same config otherwise. If the
distractor admission at skew 2 stays within 5 pp of pair admission after 12k steps, the
caching-theory section is reframed: the information is there and this gate cannot use it,
which is a stronger and more specific negative result; the S = 64 scale-up (27 runs) is
dropped. If distractor admission falls clearly below pair admission and recall passes 72 %,
the scale-up goes ahead.

## 2026-10-26 S6b H2 same-learning-rate check — runs/sweep_h2/{no_interference_lr3e-4,controller_lr1e-3}_S{64,128}_N64_s{0,1,2}

Why (Ryan, 17:40: "we changed too many variables at a time"): the sweep's H2 comparison is
best-of-each, controller at lr 3e-4 vs ablation at 1e-3 (S5 selection), so the feature
effect is confounded with the learning rate. Twelve runs make it symmetric: the ablation at
the controller's lr and the controller at the ablation's lr, S = 64 and 128, three seeds,
hidden 32, everything else as the sweep. Queued behind the N = 64 column (~3 h at two
concurrent), ahead of the N = 128 column.
Sweep so far (val, 3 seeds): S = 64 controller 62.5-64.3 vs ablation 39.7-57.0; S = 128
controller 47.6-77.5 vs ablation 69.2-92.8.
Prediction: the lr is not the explanation. Ablation at 3e-4 stays below the controller at
S = 64 (all three seeds under 60) and above it at S = 128 (mean > 67); controller at 1e-3
does not exceed its 3e-4 runs at either S (the S5 grid had it less stable at 1e-3). If
instead the ablation at 3e-4 matches the controller at S = 64, the H2 gap there was tuning
and the paper says so. Either way the paper reports both the best-of-each and the same-lr
comparison.
Result (12 runs, 19:22-22:47, 0 failures; test recall at 512, three seeds; gates at best ckpt):

| run | S = 64 | S = 128 | admission (pairs / filler) |
|---|---|---|---|
| ablation at 3e-4 (controller's lr) | 66.4, 65.3, 65.8 (**65.8**) | 96.2, 95.9, 95.8 (**96.0**) | S64 0.70 / 0.002; S128 0.97 / 0.09 |
| controller at 3e-4 (sweep) | 63.5 | 66.8 | S64 0.69 / 0.01; S128 0.82 / 0.13 |
| controller at 1e-3 (ablation's lr) | 49.6, 34.2, 40.8 (41.5) | 70.8, 66.1, 76.6 (71.2) | S64 0.49-0.86 / 0.06; S128 0.95 / 0.17 |
| ablation at 1e-3 (sweep) | 45.6 | 77.3 | S64 1.00 / 0.18 |

Paired controller - ablation at equal lr (pooled CI, per-seed):
- S = 64, both 3e-4: -2.3 pp [-2.4, -2.2] (-2.8, -1.0, -3.1). Both 1e-3: -4.1 (-7.4, -5.4, +0.6).
- S = 128, both 3e-4: -29.2 pp (-48.6, -20.8, -18.2). Both 1e-3: -6.2 (-22.1, -3.1, +6.7).

The prediction was wrong: the learning rate *was* the explanation. The sweep's "H2 holds at
S = 64" came from the ablation sitting at its one-seed-selected lr 1e-3, where it writes every
pair plus 18 % filler and is unstable; at 3e-4 it learns the same admission policy as the
controller (70 % of pairs, 0.2 % filler), with a tighter seed spread (65.3-66.4 vs 62.5-64.3),
and beats the controller by 2.3 pp on every seed. At S = 128 the ablation at 3e-4 reaches
96 % on all three seeds, 4 pp under pair-only recency and 29 pp above the controller. The
interference features never help at equal lr; the controller is the less stable of the two.
Decision: H2 is **not supported** at any budget; the paper's H2 paragraph is rewritten to lead
with the equal-lr comparison and to state that the best-of-each gap at S = 64 was a tuning
artefact of one-seed lr selection (the S5 pick of 1e-3 for the ablation came from a single
27.7 % run at S = 32). Table II gets same-lr H2 rows; Fig. 2 gets the ablation at 3e-4.
The ablation at 3e-4 is now the best learned writer (65.8 / 96.0), 4 pp under the
parameter-free reference at both budgets, so H1 fails for it too but narrowly at S = 128.
Lesson recorded in Decisions: hyperparameter selection on one seed is not safe for a
writer whose basin is stochastic; the N = 128 column keeps selected.yaml as pre-registered
(changing it now would be post hoc), and the ablation-at-3e-4 runs are reported as the
additional condition they are. Not changed: selected.yaml.

## 2026-10-26 S6 the sweep — runs/sweep/<writer>_S<S>_N<N>_s<seed>

Purpose (PLAN step 6, paper Table II, H1 and H2): 6 writers x S {32, 64, 128} x N {64, 128} x
3 seeds = 108 runs. N = 64 on `target.yaml` (length 512, tests 512 / 1024 / 2048); N = 128 on
`target_n128.yaml` (128 + 32 + 32, length 1024, tests 1024 / 2048 / 4096). Per-writer
hyperparameters from `selected.yaml`, applied by `experiments/sweep.py`, which writes the job
list for `jobq.py` (three concurrent; a regenerated list skips finished runs and resumes
crashed ones from ckpt.pt). Order: N = 64 column first with S = 64 before 128 before 32, then
N = 128. Best-validation checkpoint; paired bootstrap over the 10k test sequences
(per_example_L*.npy) decides H1 and H2.
Pre-flight (runs/preflight, 2026-10-26 07:00-07:25): N = 128 at S = 128 trains on the fast path
for controller, pair_recency (pair block 384 confirmed: writes 100 % of pairs, 0 % of filler)
and position; test evals at 1024 / 2048 / 4096 on 10k sequences take 16 / 61 / 228 s; train
peak 8.3 GB at length 1024 (4.2 GB at 512); solo 0.34 s/step at length 1024 and S = 128
(0.12 at 512). Eval batch changed to a fixed 64k-token budget (identical to every run so far
at length 512; halves the batch for the length-1024 column, which otherwise spiked to ~20 GB
at the train-length test eval).
Launch note (2026-10-26 07:35, relaunched 09:07): the first launch at three concurrent OOMed
its third job after two minutes. `bench_step.py` reports allocated peak (4.2 GB at S = 64,
8.3 GB at S = 128 / length 1024) but a running process holds far more on the card because
the CUDA-graph pool keeps every timestep's activations: measured with nvidia-smi over a short
real run (train + test evals), controller: N = 64 S = 64 12.2 GB, S = 128 15.8 GB; N = 128
(length 1024) S = 32 18.6 GB, S = 128 29.6 GB (of 32.6). `expandable_segments` saves ~1.5 GB
only. N = 64 column relaunched at two concurrent (`jobq.py --max 2 --free-gb 17`, S = 64 and
S = 128 jobs interleaved so two S = 128 never share the card), with `experiments/watchdog.py`
(non-zero exit, 15-min log silence, dead queue). The two runs that finished before the pause
(controller 63.6 %, ablation 30.2 % val at S = 64 seed 0) are kept. N = 128 concurrency is
decided after the column from probes of S = 64 / 128 at batch 32 (same tokens per step as the
length-512 column) that run automatically when the queue drains.
Projection (revised): N = 64 column 52 runs x ~30 min / 2 ≈ 13-14 h wall; N = 128 column 54 runs x
~80-90 min / 3 ≈ 25-27 h wall; ≈ 70 GPU-h total.
Hypotheses and predictions (ceilings from `analysis/ceiling.py`: pair-only FIFO 43 / 81 / 100 %
at N = 64, S = 32 / 64 / 128; 24 / 43 / 81 % at N = 128):
- pair_recency lands within 5 pp of its ceiling at every cell, flat in length (it did at S = 32).
- naive recency and position stay below 5 % at every cell (hold only the last S tokens).
- controller and ablation: bimodal across seeds as at S = 32 (good basin found on 1-3 seeds).
  H1 (controller beats the best non-learned writer at the same S) is predicted to **fail at
  S = 32 and S = 64** on the three-seed mean, because the controller at S = 32 admitted only
  35 % of pairs; at S = 128, where every pair fits, it is a coin flip whether the controller's
  update handling beats FIFO. H2 (interference features help): predicted **not supported** at
  any S; the ablation matched or beat the controller at S = 32 (33 vs 19 %).
- residual: bimodal (chance or near pair_recency), 1-2 of 3 seeds in the good basin per cell.
- N = 128: every writer at or below its (lower) ceiling; the same ordering as N = 64.
A result that would change the plan: the controller beating pair_recency at S = 64 on all
three seeds (H1 holds where the task is not capacity-bound), or the controller failing to
beat naive recency anywhere (the learned gate is worse than FIFO even with headroom).
Result, N = 64 column (54 runs, 2026-10-26 07:35-19:21, 0 failures; 09:07 relaunch at two
concurrent; `python experiments/analysis/sweep_table.py runs/sweep --n 64`):

### N = 64: mean test recall (%) over seeds at each length [n seeds]; gates = write_frac / pairs written / filler written at the best checkpoint

| writer | S=32 L512 | S=32 L1024 | S=32 L2048 | S=64 L512 | S=64 L1024 | S=64 L2048 | S=128 L512 | S=128 L1024 | S=128 L2048 | S=32 gates | S=64 gates | S=128 gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| controller | 22.9 [3] | 20.1 [3] | 18.8 [3] | 63.5 [3] | 62.3 [3] | 61.1 [3] | 66.8 [3] | 61.2 [3] | 57.7 [3] | 0.10 / 0.29 / 0.06 | 0.13 / 0.69 / 0.01 | 0.26 / 0.82 / 0.13 |
| no_interference | 27.7 [3] | 27.7 [3] | 27.8 [3] | 45.6 [3] | 45.6 [3] | 45.6 [3] | 77.3 [3] | 77.8 [3] | 78.3 [3] | 0.17 / 0.56 / 0.08 | 0.33 / 1.00 / 0.18 | 0.35 / 1.00 / 0.20 |
| pair_recency | 40.1 [3] | 40.1 [3] | 40.1 [3] | 70.1 [3] | 70.1 [3] | 70.1 [3] | 100.0 [3] | 100.0 [3] | 100.0 [3] |  |  |  |
| recency | 0.0 [3] | 0.0 [3] | 0.0 [3] | 0.0 [3] | 0.0 [3] | 0.0 [3] | 29.0 [3] | 24.0 [3] | 21.3 [3] |  |  |  |
| residual | 30.2 [3] | 29.1 [3] | 28.5 [3] | 0.2 [3] | 0.2 [3] | 0.2 [3] | 0.0 [3] | 0.0 [3] | 0.0 [3] | 0.19 / 0.69 / 0.07 | 0.31 / 0.32 / 0.31 | 0.24 / 0.24 / 0.24 |
| position | 0.0 [3] | 0.0 [3] | 0.0 [3] | 0.0 [3] | 0.0 [3] | 0.0 [3] | 0.0 [3] | 0.0 [3] | 0.0 [3] |  |  |  |

### N = 64: hypothesis tests at the training length (L512); paired over test sequences, pooled across seeds, 95% bootstrap CI; per-seed differences in brackets

- H1 S=32: controller - pair_recency = -17.2 pp, CI [-17.3, -17.1]  ['-20.8', '-21.2', '-9.5']  -> fails
- H2 S=32: controller - no_interference = -4.8 pp, CI [-4.9, -4.7]  ['-7.0', '-5.7', '-1.7']  -> reversed
- H1 S=64: controller - pair_recency = -6.6 pp, CI [-6.6, -6.5]  ['-6.5', '-5.8', '-7.4']  -> fails
- H2 S=64: controller - no_interference = +17.9 pp, CI [+17.8, +18.0]  ['+6.5', '+24.7', '+22.5']  -> holds
- H1 S=128: controller - pair_recency = -33.2 pp, CI [-33.4, -33.1]  ['-52.3', '-24.9', '-22.4']  -> fails
- H2 S=128: controller - no_interference = -10.5 pp, CI [-10.8, -10.3]  ['-45.2', '+5.9', '+7.7']  -> reversed

Reading (three seeds per cell):
- H1 fails at every budget, every seed. Pair-only recency, which has no parameters in the
  writer, beats the controller by 6.6 pp at S = 64 (seeds -5.8 to -7.4), by 17 pp at S = 32
  and by 33 pp at S = 128, where it is perfect and the controller averages 67 with one seed
  at 47.6. Gate statistics say why: the controller admits 69 % of pairs at S = 64 and 82 %
  at S = 128 and still writes filler (13 % at S = 128); FIFO admits 100 % and no filler.
- H2 holds at S = 64 (+17.9 pp, every seed positive: +6.5, +24.7, +22.5) and reverses at
  S = 128 (-10.5 pp pooled, but seeds -45.2, +5.9, +7.7: the reversal is one controller seed
  in the bad basin; 2/3 seeds favour the controller) and at S = 32 (-4.8 pp, all seeds
  negative, inside the capacity-bound cell). The ablation at S = 64 writes every pair plus
  18 % of filler and lands at 45.6; the interference features cut filler to 1 % there.
  Caveat: best-of-each lrs (controller 3e-4, ablation 1e-3); S6b is the same-lr check.
- The pooled test-sequence bootstrap CIs are ±0.1-0.3 pp and understate the uncertainty
  wherever seeds disagree (S = 128); the paper reports the per-seed differences next to the
  pooled CI, and calls S = 128 H2 undecided rather than reversed.
- Residual: 30 % at S = 32 (3/3 seeds learned at lr 6e-4, confirming the S5 pick there) and
  chance at S = 64 and 128 (6/6 runs; writes ~1/4-1/3 of every position class, no
  discrimination). Its lr was chosen at S = 32 and does not transfer. Position 0 % and naive
  recency 0 % everywhere except recency 29 % at S = 128 (holds the last 128 tokens = 64 pairs).
- Length generalization: flat (< 0.3 pp drop 512 -> 2048) for every writer except the
  controller (-2.4 pp at S = 64, -9 pp at S = 128) and naive recency.
- Prediction check: pair_recency within 5 pp of its ceiling (70 vs 81 at S = 64: no, 11 pp
  under; 100 at S = 128: yes; 40 vs 43 at S = 32: yes); naive recency / position < 5 %: yes
  except recency at S = 128; controller bimodal: yes at S = 128 only; H1 fail at S = 32 and
  64: yes; H2 not supported: wrong at S = 64, where it holds on every seed. Residual bimodal:
  no, it is uniform per cell.
Decision: N = 128 column proceeds unchanged (one run at a time at batch 64, ~38 h, after
S6b). The paper's H1 paragraph: not supported, with the admission-rate mechanism; H2:
supported at S = 64, undecided at S = 128 pending S6b, reversed inside the capacity-bound
cell. Residual's lr transfer failure goes in the limitations with the one-seed selection.
Result (N = 128 column): pending.

## 2026-10-25 S5 hyperparameter selection at the target — runs/step3/s5_<writer>_lr<lr>_s0, s5_controller_h{32,128}_s0

Purpose (PLAN step 5, paper "Selection" paragraph): per writer, choose the learning rate from
{3e-4, 6e-4, 1e-3} on validation recall of the best checkpoint at N = 64, S = 32, seed 0, and
the controller hidden size from {32, 64, 128} at its chosen lr; then freeze for the sweep.
The 3e-4 points are the seed-0 runs of C3 / S4 (same config, same seed, reused, not rerun).
Hypothesis: basin-finding is lr-sensitive (P11 at 1e-4 never opened the gate; Zoology sweeps
lr per architecture), so a higher lr may find the pair-only policy more often; hidden size
matters little (the controller's writer has 1.9k parameters).
Prediction: at least one writer changes its chosen lr away from 3e-4; controller hidden 32
within 2 pp of 64. Caveat recorded up front: one seed per point, and the basin outcome is
stochastic, so a ±15 pp swing between neighbouring lrs can be noise; the chosen lr is
confirmed by the sweep's three seeds, not by this grid alone.
Config: target.yaml --set model.fast=true --set train.lr={6e-4,1e-3} for controller,
no_interference, residual, pair_recency, position; --set model.controller_hidden={32,128}
for controller at lr 3e-4. Twelve runs, queue, three at a time.
Result (12 runs, 2026-10-25 23:01 - 10-26 01:18, 20-35 min each at three concurrent; the
3e-4 column is C3/S4 seed 0; val = best-checkpoint validation recall, %; test 512 in
parentheses; gates at the best checkpoint):

| writer | lr 3e-4 | lr 6e-4 | lr 1e-3 | best step / write_frac / pairs written / filler written |
|---|---|---|---|---|
| controller (hidden 64) | 19.0 (18.8) | 20.9 (21.0) | 17.6 (17.6) | 6e-4: 2500 / 0.09 / 0.32 / 0.03; collapses 20 → 2.5 % at step 3000, ends 17.2 |
| no_interference (hidden 64) | 19.4 (19.1) | 19.0 (19.1) | **27.7** (27.5) | 1e-3: 5500 / 0.07 / 0.33 / 0.01 |
| residual | 0.1 (0.1) | **37.2** (37.1) | 0.2 (0.2) | 6e-4: 8000 / 0.23 / **0.94** / 0.06, monotone from step 4000 |
| pair_recency | 40.1 | 40.1 | 40.0 | ceiling policy; the backbone is all that trains |
| position | 0.0 | 0.0 | 0.0 | write_frac 1.00 at every lr: writes every position by construction |

| controller hidden (lr 3e-4) | 32 | 64 | 128 |
|---|---|---|---|
| val (test 512) | **32.5** (32.3) | 19.0 (18.8) | 24.6 (24.9) |
| best step | 8000 (still rising: 18 / 29 / 31.5 / 32.3 / 32.5 at 3k-7k) | 3000 | 6500 |
| pairs / filler written | 0.35 / 0.002 | 0.33 / 0.01 | 0.35 / 0.03 |

Predictions: "at least one writer moves off 3e-4" held (ablation → 1e-3, residual → 6e-4);
"hidden 32 within 2 pp of 64" failed by +13.5 pp, and in the direction of the smaller
network, which is also the only controller run with no post-peak decline. One seed each, so
a ±15 pp swing between neighbours is within the basin noise recorded up front; the hidden-32
curve shape (monotone, filler rate 0.2 %) is the part that looks like more than noise.
Protocol note: the hidden grid ran at 3e-4 concurrently with the lr grid instead of after
the lr choice; since the controller's chosen lr is 3e-4 the pre-registered order is satisfied
in effect, not in time.
Decision (frozen in `experiments/configs/selected.yaml`, read by the sweep launcher):
controller lr 3e-4, hidden 32 (3e-4 vs 6e-4 is 1.9 pp, inside the C3 seed spread, and the
6e-4 and 1e-3 curves are unstable); no_interference lr 1e-3, hidden 32 (the ablation differs
from the controller only in the removed features, so it shares the architecture; its lr is
its own grid winner); residual lr 6e-4 (one seed reaches 37 % = 0.94 of pairs written, close
to the pair-only ceiling; at 3e-4 only 1/3 seeds found a 24 % policy); pair_recency and
recency lr 3e-4 (no gate to learn; flat); position lr 3e-4 (nothing to choose: 0 % at
every lr, the lr cannot change a writer that writes every position). Residual's
`init_threshold` / `sharpness` stay unexposed: the lr grid did help. Residual and the
controller remain basin-stochastic, which the sweep's three seeds will report as spread,
not hide. Step 5 done.

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
Result (six runs, 21:56-22:51, 21-29 min each, best-val selection):

| writer | seed | best step | val | test 512 / 1024 / 2048 | write_frac | by class q/d/u/other |
|---|---|---|---|---|---|---|
| residual | 0 | 7000 | 0.1 | 0.1 / 0.1 / 0.1 | 0.65 | 0.67 / 0.68 / 0.67 / 0.64 |
| residual | 1 | 8000 | 0.1 | 0.1 / 0.1 / 0.1 | 0.68 | 0.70 / 0.70 / 0.72 / 0.67 |
| residual | 2 | 7500 | **24.2** | 24.2 / 24.2 / 24.1 | 0.19 | 0.61 / 0.62 / 0.54 / **0.09** |
| position | 0-2 | - | 0.0 | 0.0 | 1.00 | 1 / 1 / 1 / 1 |

Decision: predictions held for position (writes every position by construction, holds the
last S tokens, 0 % at every seed: it is a naive-recency variant at this density and will stay
near chance at every S; keep it as the proposal's baseline, expect nothing from it) and
half-held for residual: 2/3 seeds never learn to discriminate (threshold admits ~2/3 of every
position class, filler included → chance), 1/3 finds a policy that writes 61 % of pairs and
9 % of filler and reaches 24 %, flat in length. Same basin structure as the learned
controller (good basin rarely found). Gate statistics are sane in the plan's sense (not
all-write, not never-write) only on the seed that learned. Step 4 done: both have results at
the target. Step 5 (lr grid) is where residual's threshold gets a fair shot; its `init_threshold`
and `sharpness` are not config-reachable yet, add them if the lr grid does not help.


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
