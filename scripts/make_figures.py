"""Standalone figure regeneration entry point.

Stage 0 ships the skeleton; real plotting lives in `src/viz/plots.py` and is
exercised from Stage 4 onwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.viz.plots import regenerate_figures_for_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate figures for a completed stage.")
    parser.add_argument("--stage", required=True, help="Stage name, e.g. 'pilot'.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override the results directory (defaults to results/<stage>/).",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or Path("results") / args.stage
    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        return 1

    regenerate_figures_for_stage(args.stage, results_dir)
    print(f"==> Figures regenerated under {results_dir}/plots/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
