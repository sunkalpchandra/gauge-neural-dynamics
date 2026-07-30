#!/usr/bin/env python
"""Report, by name, any generated macro the paper cites but does not define.

pdflatex reports these only as a pile of identical "Undefined control sequence"
lines, which says nothing about which experiment is missing. This names them and
maps each back to the experiment that produces it, so a failed build points
straight at the run to repeat.

Exits non-zero if anything is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

# macro prefix -> the experiment that generates it
OWNER = {
    "hip": "exp1_hippocampus",
    "grid": "exp2_grid_cells",
    "mot": "exp3_motor_cortex",
    "abl": "exp4_ablations",
    "rob": "exp5_robustness",
    "cont": "exp6_continuous_context",
}


def main() -> int:
    numbers = PAPER / "generated" / "numbers.tex"
    defined: set[str] = set()
    if numbers.exists():
        defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", numbers.read_text()))

    used: set[str] = set()
    for f in ("main.tex", "appendix.tex"):
        p = PAPER / f
        if p.exists():
            used |= set(re.findall(r"\\([a-z]+[A-Z]\w*)\b", p.read_text()))

    # only consider names that look like ours, i.e. carry a known prefix
    ours = {m for m in used if any(re.match(rf"^{k}[A-Z]", m) for k in OWNER)}
    missing = sorted(ours - defined)

    if not missing:
        print(f"macros: {len(ours)} cited, all defined")
        return 0

    by_exp: dict[str, list[str]] = {}
    for m in missing:
        key = next(k for k in OWNER if re.match(rf"^{k}[A-Z]", m))
        by_exp.setdefault(OWNER[key], []).append(m)

    print(f"FAIL: {len(missing)} macro(s) cited but not defined:")
    for exp, names in sorted(by_exp.items()):
        print(f"    {exp}: {', '.join(names)}")
    print("  -> re-run those experiments, then scripts/make_tables.py")

    unused = sorted(d for d in defined - ours if any(re.match(rf"^{k}[A-Z]", d) for k in OWNER))
    if unused:
        print(f"  ({len(unused)} generated macro(s) defined but never cited, which is fine)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
