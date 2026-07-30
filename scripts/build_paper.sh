#!/usr/bin/env bash
# Build the paper, and refuse to report success while anything is a placeholder.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="$ROOT/paper"

for d in "$HOME/Library/TinyTeX/bin/universal-darwin" /Library/TeX/texbin /usr/local/texlive/*/bin/*; do
  if [ -x "$d/pdflatex" ]; then export PATH="$d:$PATH"; break; fi
done
command -v pdflatex >/dev/null || { echo "pdflatex not found; see README for install options"; exit 127; }

mkdir -p "$PAPER/figures"
cp -f "$ROOT"/figures/*.pdf "$PAPER/figures/" 2>/dev/null || true

cd "$PAPER"
echo "== pdflatex (1/4) ==" ; pdflatex -interaction=nonstopmode main.tex >  build.log 2>&1
echo "== bibtex =="         ; bibtex   main                              >> build.log 2>&1
echo "== pdflatex (2/4) ==" ; pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1
echo "== pdflatex (3/4) ==" ; pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1
echo "== pdflatex (4/4) ==" ; pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1

[ -f main.pdf ] || { echo "FAIL: no PDF produced"; grep -E "^! " build.log | head; exit 1; }

status=0
errs=$(grep -cE "^! " build.log || true)
if [ "${errs:-0}" -gt 0 ]; then
  echo "FAIL: ${errs} LaTeX error(s):"
  grep -E "^! " build.log | sort -u | head
  status=1
fi
# Placeholders are only visible in the typeset output, not reliably in the log,
# so check for the inputs themselves instead.
missing=""
for f in $(grep -oE '\\gndinput\{[^}]+\}' main.tex appendix.tex | sed -E 's/.*\{(.*)\}/\1.tex/' | sort -u); do
  [ -f "$f" ] || missing="$missing $f"
done
for f in $(grep -oE '\\gndfig\{[^}]+\}' main.tex appendix.tex | sed -E 's/.*\{(.*)\}/\1/' | sort -u); do
  [ -f "figures/$f" ] || missing="$missing figures/$f"
done
if [ -n "$missing" ]; then
  echo "FAIL: missing generated inputs (placeholders will render):"
  for f in $missing; do echo "    $f"; done
  status=1
fi
und=$(grep -oE "Citation \`[^']+' undefined" build.log | sort -u || true)
if [ -n "$und" ]; then echo "FAIL: undefined citations:"; echo "$und" | head; status=1; fi
ref=$(grep -oE "Reference \`[^']+' undefined" build.log | sort -u || true)
if [ -n "$ref" ]; then echo "FAIL: undefined references:"; echo "$ref" | head; status=1; fi

pages=$(grep -oE "Output written on main.pdf \([0-9]+ page" build.log | grep -oE "[0-9]+" | tail -1)

# The venue limit is on the body only, so measure it from the \label{endofbody}
# marker placed just before the bibliography rather than from the total.
body=$(grep -oE '\\newlabel\{endofbody\}\{\{[^}]*\}\{[0-9]+\}' main.aux \
        | grep -oE '[0-9]+\}$' | tr -d '}')
limit=${GND_PAGE_LIMIT:-9}
echo "== body ${body:-?} pages (limit ${limit}); ${pages:-?} pages total incl. references and appendix =="
if [ -n "$body" ] && [ "$body" -gt "$limit" ]; then
  echo "FAIL: body exceeds the ${limit}-page limit"
  status=1
fi
if [ "$status" -eq 0 ]; then echo "OK: paper/main.pdf"; else echo "build completed with problems"; fi
exit $status
