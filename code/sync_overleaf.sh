#!/bin/bash
# Push the current paper sources to the Overleaf project (git.overleaf.com/6af2c30e2a35462073e64506).
# 1. pull, and stop if Overleaf has commits we have not seen (the author edited online: merge those first);
# 2. copy only the sources the paper needs (as package_overleaf.sh) into overleaf_git/, commit, push;
# 3. clone the project afresh into the scratch folder, compile it and compare its text with paper/build/main.pdf.
# Usage: code/sync_overleaf.sh "commit message"
set -e
cd "$(dirname "$0")/.."
MSG=${1:-"Update from local build"}
G=overleaf_git
STAMP=$G/../.overleaf_last_sync
git -C $G pull -q --ff-only
HEAD=$(git -C $G rev-parse HEAD)
if [ -f "$STAMP" ] && [ "$(cat $STAMP)" != "$HEAD" ]; then
  echo "Overleaf changed since the last sync ($(cat $STAMP) -> $HEAD): merge the online edits into paper/ first"; exit 1
fi
cp paper/main.tex paper/sec_theory.tex paper/refs.bib paper/theory.bib $G/
cp paper/sections/*.tex $G/sections/
cp paper/tables/*.tex $G/tables/
cp paper/figs/*.pdf $G/figs/
if [ -z "$(git -C $G status --porcelain)" ]; then
  echo "Overleaf already up to date ($HEAD)"; echo "$HEAD" > $STAMP; exit 0
fi
git -C $G add -A
git -C $G -c user.name="Hongsheng Wang" -c user.email="wanghongsheng@zju.edu.cn" commit -q -m "$MSG"
git -C $G push -q
git -C $G rev-parse HEAD > $STAMP
echo "pushed $(cat $STAMP)"
# verification from a fresh clone
V=${TMPDIR:-/tmp}/overleaf_verify_$(date +%s)
git clone -q https://git@git.overleaf.com/6af2c30e2a35462073e64506 "$V"
export PATH=$HOME/Library/TinyTeX/bin/universal-darwin:$PATH
(cd "$V" && pdflatex -interaction=nonstopmode main.tex >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null) || true
.venv/bin/python -c "
import fitz, sys
a=''.join(p.get_text() for p in fitz.open('$V/main.pdf')); b=''.join(p.get_text() for p in fitz.open('paper/build/main.pdf'))
print('fresh Overleaf clone compiles to the same text as paper/build/main.pdf' if a.split()==b.split() else 'TEXT DIFFERS'); sys.exit(a.split()!=b.split())"
