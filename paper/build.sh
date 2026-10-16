#!/usr/bin/env bash
# Build the submission .docx from the Markdown master.
# Usage: bash paper/build.sh [RP2|RP3]   (default RP2)
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-RP2}"
pandoc paper.md \
  --from markdown+tex_math_dollars+pipe_tables+implicit_figures \
  --citeproc --bibliography refs.bib --csl ieee.csl \
  --resource-path=. \
  --reference-doc=reference.docx \
  -o "Ellis-${TAG}.docx"
echo "wrote paper/Ellis-${TAG}.docx"
