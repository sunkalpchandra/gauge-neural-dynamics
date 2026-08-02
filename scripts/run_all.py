#!/usr/bin/env python
"""Reproduce every result, figure and table in the paper.

    python scripts/run_all.py                 # full suite (hours, 8 cores)
    python scripts/run_all.py --quick         # smoke test (minutes)
    python scripts/run_all.py --stage figures # regenerate figures only

Stages run in order and each is skipped if ``--stage`` names a different one.
Experiments run as separate processes so that a failure in one does not lose the
others; each writes ``results/<name>/results.json`` plus the artefacts its figure
needs.

``--quick`` redirects every stage to ``results/quick`` and ``figures/quick`` so
that a smoke test can never overwrite the results and figures the paper is built
from.  The exit code is non-zero if any stage failed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable

EXPERIMENTS = [
    ("exp1_hippocampus", []),
    ("exp2_grid_cells", []),
    ("exp3_motor_cortex", []),
    ("exp4_ablations", []),
    ("exp5_robustness", []),
    ("exp6_continuous_context", []),
]


def run(cmd: list[str], log: Path | None = None, env: dict | None = None) -> int:
    e = {**os.environ, **(env or {})}
    if log is None:
        return subprocess.call(cmd, cwd=ROOT, env=e)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        return subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=e)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny run, for checking the pipeline")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=3, help="experiments to run concurrently")
    ap.add_argument("--threads", type=int, default=2, help="BLAS threads per experiment")
    ap.add_argument("--only", nargs="*", default=None, help="run only these experiments")
    ap.add_argument("--stage", choices=["experiments", "figures", "tables", "paper", "all"],
                    default="all")
    args = ap.parse_args(argv)

    t0 = time.time()
    stages = ["experiments", "figures", "tables", "paper"] if args.stage == "all" else [args.stage]
    failures: list[str] = []

    # A smoke run writes toy numbers.  Send them somewhere the paper never reads,
    # so that following the README's quick start cannot silently replace the
    # multi-seed results and figures under version control.
    scratch = {}
    if args.quick:
        scratch = {"GND_RESULTS_DIR": str(ROOT / "results" / "quick"),
                   "GND_FIGURE_DIR": str(ROOT / "figures" / "quick")}
        print(f"== quick mode: writing to {scratch['GND_RESULTS_DIR']} "
              f"and {scratch['GND_FIGURE_DIR']} ==")

    if "experiments" in stages:
        todo = [(n, a) for n, a in EXPERIMENTS if not args.only or n in args.only]
        print(f"== running {len(todo)} experiments ({args.jobs} at a time) ==")
        env = {"OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
               "PYTHONUNBUFFERED": "1", **scratch}

        def one(item):
            name, extra = item
            cmd = [PY, "-u", "-m", f"gnd.experiments.{name}"]
            cmd += ["--quick"] if args.quick else ["--seeds", *map(str, args.seeds)]
            cmd += extra
            log = ROOT / "results" / "logs" / f"{name}.log"
            s = time.time()
            code = run(cmd, log=log, env=env)
            print(f"  {'ok  ' if code == 0 else 'FAIL'} {name}  "
                  f"({time.time() - s:.0f}s, log: results/logs/{name}.log)")
            return code

        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            codes = list(ex.map(one, todo))
        if any(codes):
            print("one or more experiments failed; see results/logs/")
            failures += [n for (n, _), c in zip(todo, codes) if c]

    if "figures" in stages:
        print("== figures ==")
        if run([PY, "-m", "gnd.figures.make_all"], env=scratch):
            failures.append("figures")

    if "tables" in stages:
        print("== tables and in-text numbers ==")
        if run([PY, str(ROOT / "scripts" / "make_tables.py")], env=scratch):
            failures.append("tables")

    if "paper" in stages and not args.quick:
        print("== paper ==")
        if run(["bash", str(ROOT / "scripts" / "build_paper.sh")]):
            failures.append("paper")
    elif "paper" in stages:
        print("== paper == (skipped: quick mode builds no paper from toy numbers)")

    print(f"\ntotal {time.time() - t0:.0f}s")
    if failures:
        # Returning 0 here would let a broken sweep flow straight through the
        # figure, table and paper stages and still report success to CI.
        print("FAILED stages: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
