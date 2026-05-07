"""Figure generation (spec §12.5).

Stage 0 ships the entry-point and skeleton so `scripts/make_figures.py` and the
orchestrator have a stable target to call. Real figures land in Stages 4-6 when
there are actual leakage curves to plot.
"""

from __future__ import annotations

from pathlib import Path


def regenerate_figures_for_stage(stage_name: str, results_dir: Path) -> list[Path]:
    """Return the list of figure paths regenerated. Stage 0 returns []."""
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Real plots wired up from Stage 4 onwards.
    return []
