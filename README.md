# 742 Research Project — interference-aware write control for bounded-memory recurrent models

**Next steps live in `PLAN.md`.** Working folder for the EN.705.742 research paper. Lives outside the Obsidian vault so code,
data, checkpoints, and a venv can go here without being backed up as notes.

Source of truth for course materials is the vault:
`~/JHU/Fall 2026/705.742_Advanced_Applied_ML/` (00-admin/ and 03-assignments/RP*/).
Files here were copied on 2026-10-15; re-copy if the vault versions change.

## Milestones

| Date       | Deliverable                 | Status            |
|------------|-----------------------------|-------------------|
| 2026-09-13 | RP1 — Pick topic (proposal) | submitted, 100/100 |
| 2026-10-25 | RP2 — Midpoint check / draft | submitted 2026-10-25 (`paper/Ellis-RP2.pdf`, 8 pp, plus `Ellis-RP2-prototype.ipynb`); Table II and `\pending{}` notes remain for RP3 |
| 2026-12-13 | RP3 — Final paper           | next: PLAN.md step 1 |

## Contents

- `instructions/` — rubric (5 criteria, 100 pts), the three RP assignment briefs, and the
  general assignment/submission guidelines.
- `proposal/` — the submitted proposal (.docx original, .md transcription, architecture figure).
- `src/bmem/` — the model: `data.py` (MQAR + updates/distractors/delay, adapted from Zoology),
  `backbone.py` (GLA via flash-linear-attention), `memory.py` (slot memory), `controller.py`
  (interference controller + 4 baseline writers), `model.py`, `train.py`, `metrics.py`.
- `experiments/` — `run.py` (one run -> `results.json`), `progress.py` (status table), `curves.py`
  (training curves), `bench_step.py` (step timing), `aggregate.py` (tables, paired bootstrap),
  `plot.py` (paper figures), `block_diagram.py` (paper Fig. 1), `LOG.md` (lab notebook), `configs/`.
- `tests/` — `pytest` (fast) and `pytest -m slow` (GPU learnability regression).
- `Ellis-RP2-prototype.ipynb` — RP2 companion notebook: sample data, model accounting, gate
  heatmaps, an 800-step training run on the small task, the bootstrap statistics, and the pilot
  table. Re-execute with `jupyter nbconvert --to notebook --execute --inplace Ellis-RP2-prototype.ipynb`
  (needs the GPU; ~2 min).
- `paper/` — `paper.tex` master on the course's IEEE conference template (`IEEEtran.cls`
  copied from `conference-latex-template_10-17-19.zip`), `refs.bib`, `prisma/search-log.md`,
  `figures/`, `build.sh` (latexmk -> `Ellis-RP2.pdf` plus an Overleaf-ready zip).
  `paper.md` and the pandoc/.docx files are the superseded first draft; the guide asks for
  LaTeX on Overleaf, so edit `paper.tex` only.
- `runs/` — gitignored outputs.

## How to run

```bash
source ~/venvs/jhu-742/bin/activate
pytest                                   # fast tests
pytest -m slow                           # ~1 min GPU learnability check
python experiments/run.py --config experiments/configs/target.yaml --writer controller --seed 0 \
    --out_dir runs/step1/p1_vocab2048_controller_s0 --set data.vocab_size=2048 --tag "P1 vocab 2048"
python experiments/run.py ... --resume          # continue a killed run from its ckpt.pt
python experiments/progress.py runs/step1 --watch 300   # status table: step, s/step, ETA, loss, val recall, gate stats
python experiments/curves.py runs/step1        # loss / recall / gnorm / write-fraction curves -> runs/step1/curves.png
python experiments/bench_step.py experiments/configs/target.yaml   # s/step for the write-loop speedup work
python experiments/aggregate.py runs/final --pair controller recency
python experiments/plot.py runs/final --out paper/figures/final.png
bash paper/build.sh RP3
```

Every run writes a timestamped `log.txt`, a `results.json` that is rewritten at each eval
(`status: running` → `done`, effective config, git commit, gate statistics in `history`),
`ckpt.pt` for `--resume`, and `best.pt` (best validation recall) for `--init_from` curriculum
runs. `--set section.key=value` overrides any config field. The lab notebook is
`experiments/LOG.md`: one entry per launched run, hypothesis and prediction written before launch.

## Decisions log

- 2026-10-15: key projected from the preceding token (`assoc_shift=1`) and straight-through
  hard slot gates in the controller. Both were needed for the memory path to learn at all;
  see `tests/test_smoke.py::test_controller_learns_small_task`.
- 2026-10-15: fla GLA works on sm_120 (torch 2.11 cu128). The sequential memory loop is
  98% of step time (0.65 s/step at B=64, T=512); torch.compile of the step body gives ~2x.
  Proper fix (fused/vectorized write loop) deferred to after RP2.

## Rubric reminders

- A **block diagram** is required; its absence is penalized under two criteria.
- Results need **metrics, proper runs, and statistics** (seeds, CIs), not just a working model.
- PRISMA 2009 is a Module 5 assigned reading. The rubric never mentions it; using it to structure the related-work section is an inference, not a requirement.

## Environment

Course venv: `~/venvs/jhu-742` (Python 3.13, torch cu128 for the RTX 5090 / sm_120).
Jupyter kernel: `jhu-742`. Keep datasets under `~/datasets/`, not in the vault.
- 2026-10-16: controller cold start fixed with a learnable recency prior on slot logits;
  gradient blow-up in the memory loop traced to 1/||x|| gradients of cosine/L2 features
  (eps=1e-2 added; spikes 65k -> 50); content projection of the token feeds the write
  decision; pilot lr 3e-4, 2 layers. Proposal-scale task (64 assoc) still unlearned.
