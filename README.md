# 742 Research Project — Interference-aware write control for bounded-memory recurrent models

Working folder for the EN.705.742 research paper. Lives outside the Obsidian vault so code,
data, checkpoints, and a venv can go here without being backed up as notes.

Source of truth for course materials is the vault:
`~/JHU/Fall 2026/705.742_Advanced_Applied_ML/` (00-admin/ and 03-assignments/RP*/).
Files here were copied on 2026-10-15; re-copy if the vault versions change.

## Milestones

| Date       | Deliverable                 | Status            |
|------------|-----------------------------|-------------------|
| 2026-09-13 | RP1 — Pick topic (proposal) | submitted, 100/100 |
| 2026-10-25 | RP2 — Midpoint check / draft | not started       |
| 2026-12-13 | RP3 — Final paper           | not started       |

## Contents

- `instructions/` — rubric (5 criteria, 100 pts), the three RP assignment briefs, and the
  general assignment/submission guidelines.
- `proposal/` — the submitted proposal (.docx original, .md transcription, architecture figure).
- `src/bmem/` — the model: `data.py` (MQAR + updates/distractors/delay, adapted from Zoology),
  `backbone.py` (GLA via flash-linear-attention), `memory.py` (slot memory), `controller.py`
  (interference controller + 4 baseline writers), `model.py`, `train.py`, `metrics.py`.
- `experiments/` — `run.py` (one run -> `results.json`), `aggregate.py` (tables, paired
  bootstrap), `plot.py` (figures), `configs/` (smoke, sanity, pilot).
- `tests/` — `pytest` (fast) and `pytest -m slow` (GPU learnability regression).
- `paper/` — `paper.md` master, `refs.bib`, `prisma/search-log.md`, `figures/`,
  `build.sh` (pandoc -> `Ellis-RP2.docx`).
- `runs/` — gitignored outputs.

## How to run

```bash
source ~/venvs/jhu-742/bin/activate
pytest                                   # fast tests
pytest -m slow                           # ~1 min GPU learnability check
python experiments/run.py --config experiments/configs/pilot.yaml --writer controller --seed 0 --out_dir runs/pilot/controller_s0
python experiments/aggregate.py runs/pilot --pair controller recency
python experiments/plot.py runs/pilot --out paper/figures/pilot.png
bash paper/build.sh RP2
```

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
