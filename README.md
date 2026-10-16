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

## Rubric reminders

- A **block diagram** is required; its absence is penalized under two criteria.
- Results need **metrics, proper runs, and statistics** (seeds, CIs), not just a working model.
- PRISMA 2009 is a Module 5 assigned reading. The rubric never mentions it; using it to structure the related-work section is an inference, not a requirement.

## Environment

Course venv: `~/venvs/jhu-742` (Python 3.13, torch cu128 for the RTX 5090 / sm_120).
Jupyter kernel: `jhu-742`. Keep datasets under `~/datasets/`, not in the vault.
