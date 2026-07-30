"""Regenerate every figure.  Missing results are reported, not fatal."""

from __future__ import annotations

import argparse
import importlib
import traceback

MODULES = [
    ("fig1_concept", "Figure 1: conceptual overview"),
    ("fig2_architecture", "Figure 2: architecture and objective"),
    ("fig3_hippocampus", "Figure 3: hippocampal remapping"),
    ("fig4_grid_cells", "Figure 4: grid-cell torus and translation group"),
    ("fig5_motor_cortex", "Figure 5: motor-cortex trajectories"),
    ("fig6_ablations", "Figure 6: ablations"),
    ("fig7_robustness", "Figure 7: robustness and scaling"),
    ("fig8_continuous_context", "Figure 8: gauge field over context space"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    failed = []
    for name, desc in MODULES:
        if args.only and name not in args.only:
            continue
        print(f"{desc}")
        try:
            importlib.import_module(f"gnd.figures.{name}").main()
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}")
            failed.append((name, "missing results"))
        except Exception:
            traceback.print_exc()
            failed.append((name, "error"))
    if failed:
        print("\nincomplete:")
        for n, why in failed:
            print(f"  {n}: {why}")
        return 1
    print("\nall figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
