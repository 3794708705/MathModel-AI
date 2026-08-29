#!/bin/sh
set -eu

export BIBINPUTS=/workspace:
export HOME=/tmp
export XDG_CACHE_HOME=/tmp/fontconfig
cd /workspace
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=/output /workspace/paper.tex
if grep -q '\\citation' /output/paper.aux; then
    (cd /output && bibtex paper)
    xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=/output /workspace/paper.tex
    xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=/output /workspace/paper.tex
fi
