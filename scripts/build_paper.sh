#!/usr/bin/env bash
# Build the paper, and refuse to report success while anything is a placeholder.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="$ROOT/paper"

for d in "$HOME/Library/TinyTeX/bin/universal-darwin" "$HOME"/.TinyTeX/bin/* "$HOME/.local/bin" \
         /Library/TeX/texbin /usr/local/texlive/*/bin/*; do
  if [ -x "$d/pdflatex" ]; then export PATH="$d:$PATH"; break; fi
done
command -v pdflatex >/dev/null || { echo "pdflatex not found; see README for install options"; exit 127; }

# Rebuild the staging directory from scratch.  Copying into a directory left
# over from the previous build would let a figure that has since been deleted or
# renamed keep resolving, and silently embed the old plot.
rm -rf "$PAPER/figures"
mkdir -p "$PAPER/figures"
cp -f "$ROOT"/figures/*.pdf "$PAPER/figures/" 2>/dev/null || true

cd "$ROOT"
PY_BIN="$ROOT/.venv/bin/python"; [ -x "$PY_BIN" ] || PY_BIN=python3
echo "== generated macros =="
"$PY_BIN" scripts/check_macros.py || macro_fail=1

cd "$PAPER"
echo "== pdflatex (1/4) ==" ; pdflatex -interaction=nonstopmode main.tex >  build.log 2>&1
echo "== bibtex =="         ; bibtex   main                              >> build.log 2>&1
echo "== pdflatex (2/4) ==" ; pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1
echo "== pdflatex (3/4) ==" ; pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1
echo "== pdflatex (4/4) ==" ; pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1

[ -f main.pdf ] || { echo "FAIL: no PDF produced"; grep -E "^! " build.log | head; exit 1; }

status=${macro_fail:-0}
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
# pdflatex writes "Citation `key' on page 1 undefined on input line 12.", so the
# key and the word "undefined" are not adjacent; the summary lines are matched
# too, in case a long key wraps across the log's 79-column limit.
und=$(grep -oE "Citation \`[^']+'.*undefined|There were undefined citations" build.log | sort -u || true)
if [ -n "$und" ]; then echo "FAIL: undefined citations:"; echo "$und" | head; status=1; fi
ref=$(grep -oE "Reference \`[^']+'.*undefined|There were undefined references" build.log | sort -u || true)
if [ -n "$ref" ]; then echo "FAIL: undefined references:"; echo "$ref" | head; status=1; fi
# bibtex reports its own failures without a leading "! ", so they survive the
# LaTeX error check above.
bib=$(grep -cE "^I couldn't open|^I found no|Warning--I didn't find a database entry" build.log || true)
if [ "${bib:-0}" -gt 0 ]; then
  echo "FAIL: ${bib} bibtex problem(s):"
  grep -E "^I couldn't open|^I found no|Warning--I didn't find a database entry" build.log | sort -u | head
  status=1
fi

pages=$(grep -oE "Output written on main.pdf \([0-9]+ page" build.log | grep -oE "[0-9]+" | tail -1)

# The venue limit is on the body only, so measure it from the \label{endofbody}
# marker placed just before the bibliography rather than from the total.
body=$(grep -oE '\\newlabel\{endofbody\}\{\{[^}]*\}\{[0-9]+\}' main.aux \
        | grep -oE '[0-9]+\}$' | tr -d '}')
limit=${GND_PAGE_LIMIT:-9}
echo "== body ${body:-?} pages (limit ${limit}); ${pages:-?} pages total incl. references and appendix =="
if [ -z "$body" ]; then
  # Silently skipping the limit check because its marker vanished is how a
  # page-limit violation reaches a submission.
  echo "FAIL: cannot measure the body: no \\label{endofbody} in main.aux"
  status=1
elif [ "$body" -gt "$limit" ]; then
  echo "FAIL: body exceeds the ${limit}-page limit"
  status=1
fi
if [ "$status" -eq 0 ]; then echo "OK: paper/main.pdf"; else echo "build completed with problems"; fi
exit $status
