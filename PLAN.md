# Plan to finish the paper (RP3, due 2026-12-13)

Read this first in a fresh session. The paper master is `paper/paper.tex`; the final
paper is the midpoint draft with every `\pending{...}` note and the empty Table II
(`tab:final`) filled. Grep `pending{` to list what is still open.

**State as of 2026-10-25.** RP2 submitted to Canvas 2026-10-25 (`paper/Ellis-RP2.pdf`, 8 pp, plus the prototype notebook). Pilot ran one
seed at 24 associations / 16 slots: controller 51.9%, ablation 50.7%, recency 11.0%. The
proposal's target (64 associations / 32 slots) does not leave chance within 8000 steps.
Residual-cache and position-activation writers are implemented and tested, never trained.

## The final experiment (paper Section IV-A)

90 training runs: writers {controller, no_interference, recency, residual, position} x
slots S in {32, 64, 128} x associations N in {64, 128} x seeds {0, 1, 2}. Train at length
512, test at 512 / 1024 / 2048 on fresh sequences, updates = distractors = N/4. Delay sweep
separately at N=64, S=32. Hyperparameters chosen on validation at N=64, S=32, then frozen.
H1: paired bootstrap, controller minus the validation-best baseline, holds where the lower
95% bound > +5 pp. H2: same against the ablation.

## Steps, in dependency order

- [ ] **1. Make N=64 / S=32 learn.** Blocking. Try, in order: longer schedule (Zoology uses
      tens of thousands of steps); curriculum from 16 pairs growing N while keeping S;
      larger memory dim. Done when the controller beats recency on validation at the target.
      Config: `experiments/configs/target.yaml` is the target (64 queried + 16 updates + 16
      distractors, 32 slots, length 512; `pilot.yaml` with 2 layers, lr 3e-4). `pilot32.yaml` is the intermediate rung for the
      curriculum (32 + 8 + 8 = 48 associations, 32 slots, length 256), not the target.
      Both still carry the pre-pilot settings (4 layers, lr 1e-3); the pilot that learned
      used 2 layers and lr 3e-4 (`pilot16.yaml`), so align those before judging a run.
      The search is a tree of one-variable probes (P1 vocab, P2 slots=128, P3 density ladder,
      P4 lr, P5 long schedule, P6 curriculum via `--init_from`, P7 mem_dim), each with a
      pre-registered prediction and result in `experiments/LOG.md`; definitions of
      chance / learning / solved / kill rule are at the top of that file.
      If this is not solved by **2026-11-15**, take the fallback below.
- [ ] **2. Vectorize or fuse the memory write loop** (`src/bmem/memory.py`, `controller.py`).
      It is 98% of step time (0.65 s/step at B=64, T=512; torch.compile gives 2x). Measure with
      `experiments/bench_step.py` before and after each attempt (compile the step body and
      preallocate memory tensors; CUDA graphs; chunked recurrence only as a documented last
      resort, behind a config flag). Done when a target step is under 0.1 s, `pytest -m slow`
      passes, and a 300-step smoke run matches the old loop's loss curve.
- [ ] **3. Gradient stability at scale.** Backbone grad norms of 20-50 rely on clipping. Test
      per-group clipping and truncated backprop through the memory recurrence. Done when no
      run in a 3-seed pilot32 collapses or spikes above 100.
- [ ] **4. Train and tune the residual and position writers.** Validation-tune the residual
      threshold/beta and the position schedule at N=64, S=32. Done when each has a results.json
      at the target and its gate statistics look sane (not all-write, not never-write).
- [ ] **5. Hyperparameter selection** for every writer: lr grid {1e-3, 3e-4, 1e-4}, controller
      hidden {32, 64, 128}, at N=64, S=32 on validation. Record the table for the
      "Hyperparameter selection" placeholder.
- [ ] **6. Run the sweep** (90 runs) plus the delay sweep. One run per
      `python experiments/run.py --config <cfg> --writer <w> --seed <s> --out_dir runs/final/<w>_<cfg>_s<s>`.
      Aggregate with `python experiments/aggregate.py runs/final --pair controller <best>`.
      Before this, make `plot.py` (right panel) and `aggregate.py` form CIs over seeds properly: they
      currently average per-seed CI bounds, which is only right for one seed.
- [ ] **7. Admission logging.** Add per-writer counts of what it wrote: key-value pairs, updates,
      distractors, filler; mean gate mass; slot entropy. Needed to explain the 50% plateau and
      to say where the interference features act. Log in `metrics.py`, surface in aggregate.py.
- [ ] **8. Fill the paper.** Table II, every `\pending{}`, Fig. 1 unchanged, a length-vs-recall
      figure for all writers (`experiments/plot.py`), rewrite Discussion and Conclusion around
      Table II, demote or drop the pilot section. Re-execute `Ellis-RP2-prototype.ipynb` as the
      RP3 notebook. Build with `bash paper/build.sh RP3`.
- [ ] **9. Optional: RAM-Net-style address baseline.** RAM-Net (2602.11958, NeurIPS 2026) names
      interference as its target and avoids it by mapping each key to a sparse slot address. A
      writer that picks the slot by a learned projection of the key (top-1, straight-through) is
      the cleanest "avoid conflict by address" comparison for H2 and is ~40 lines in
      `controller.py`. Add only if steps 1-6 are on schedule.
- [ ] **10. Decide on backbone size.** The proposal said 20M parameters; the pilot uses 2.84M
      (2 layers). Keep 2 layers unless step 1 needs depth; state the choice in the paper.

## Fallback if step 1 fails

Run the full sweep at loads the model learns (N in {24, 48}, S in {16, 32}) and report H1/H2 at
those densities. The research question is stated per density ("at which densities"), so this
is still an answer. Say in the paper that the proposal's target was not reached and why.

## Rubric reminders

Block diagram present (Fig. 1). Results need seeds, CIs, and a hyperparameter search, or the
Application criterion lands in the partial column. Conclusions must answer *why* the work was
done, not restate the table. Rubric: `instructions/Research Project Rubric.md`.
