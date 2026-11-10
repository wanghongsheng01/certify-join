#!/bin/bash
# Build the Overleaf upload: copy only the sources the paper needs, compile the copy from scratch, then zip it.
# Usage: code/package_overleaf.sh <tag>   ->  submission/CertifyJoin_overleaf_<tag>.zip and .pdf
set -e
cd "$(dirname "$0")/.."
TAG=${1:-$(date +%Y%m%d)}
OUT=submission/overleaf_$TAG
[ -e "$OUT" ] && { echo "$OUT exists: use a new tag (nothing is deleted)"; exit 1; }
mkdir -p "$OUT/sections" "$OUT/tables" "$OUT/figs"
cp paper/main.tex paper/sec_theory.tex paper/refs.bib paper/theory.bib "$OUT/"
cp paper/sections/*.tex "$OUT/sections/"
cp paper/tables/*.tex "$OUT/tables/"
cp paper/figs/*.pdf "$OUT/figs/"
export PATH=$HOME/Library/TinyTeX/bin/universal-darwin:$PATH
(cd "$OUT" && pdflatex -interaction=nonstopmode main.tex >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null)
grep -E "^!|undefined|Overfull" "$OUT/main.log" | sort | uniq -c || true
grep "Output written" "$OUT/main.log"
cp "$OUT/main.pdf" "submission/CertifyJoin_$TAG.pdf"
(cd "$OUT" && zip -qr "../CertifyJoin_overleaf_$TAG.zip" . -x "main.aux" "main.log" "main.out" "main.blg" "main.bbl" "main.pdf")
echo "submission/CertifyJoin_overleaf_$TAG.zip"
