#!/usr/bin/env bash
# Build the submission PDF from the LaTeX master (IEEE conference template,
# per the course's Research and Publishing Guide) and a zip ready to upload to Overleaf.
# Usage: bash paper/build.sh [RP2|RP3]   (default RP2)
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-RP2}"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper.tex >/dev/null
cp build/paper.pdf "Ellis-${TAG}.pdf"
rm -f "Ellis-${TAG}-overleaf.zip"
zip -q "Ellis-${TAG}-overleaf.zip" paper.tex refs.bib IEEEtran.cls figures/*.png
echo "wrote paper/Ellis-${TAG}.pdf and paper/Ellis-${TAG}-overleaf.zip"
