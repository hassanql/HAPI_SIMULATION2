"""Single entry point for every stage (spec §3.1, kickoff §"How this delivery works").

Usage:
    python run.py --stage bootstrap          # run a stage
    python run.py --list                     # show all 9 stages and their status
    python run.py --stage pilot --resume     # resume after interruption (Stage 5+)
    python run.py --stage pilot --dry-run    # estimate runtime, exit
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.orchestrator.stages import (
    REPO_ROOT,
    STAGES,
    list_stages,
    run_stage,
)


def _print_stages_table(results_root: Path) -> None:
    console = Console()
    table = Table(title="A2 Simulation — Stages", show_lines=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Model config", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Description")
    for i, row in enumerate(list_stages(results_root)):
        status = row["status"]
        style = {
            "PASS": "green",
            "FAIL": "red",
            "PARTIAL": "yellow",
            "not-run": "dim",
        }.get(status, "white")
        table.add_row(str(i), row["name"], row["model_config"], f"[{style}]{status}[/{style}]", row["description"])
    console.print(table)
    console.print(
        "\nRun a stage with: [bold]python run.py --stage <name>[/bold] (e.g. "
        "[bold]python run.py --stage bootstrap[/bold])"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="A2 Simulation single-command entry point.",
    )
    parser.add_argument("--stage", default=None, help="Stage to run (e.g. bootstrap).")
    parser.add_argument("--list", action="store_true", help="List all stages and their status.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run (Stage 5+).")
    parser.add_argument("--dry-run", action="store_true", help="Estimate runtime, run nothing.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results",
        help="Override the results directory (default: ./results/).",
    )
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=REPO_ROOT / "configs",
        help="Override the configs directory (default: ./configs/).",
    )
    args = parser.parse_args(argv)

    if args.list and args.stage:
        parser.error("--list and --stage are mutually exclusive.")
    if not args.list and not args.stage:
        parser.error("specify either --list or --stage <name>.")

    if args.list:
        _print_stages_table(args.results_root)
        return 0

    if args.stage not in STAGES:
        print(
            f"Unknown stage: {args.stage!r}. Run `python run.py --list` to see available stages.",
            file=sys.stderr,
        )
        return 2

    console = Console()
    console.print(f"[bold cyan]==> Running stage:[/bold cyan] {args.stage}")
    try:
        report = run_stage(
            args.stage,
            results_root=args.results_root,
            configs_root=args.configs_root,
            dry_run=args.dry_run,
            resume=args.resume,
        )
    except NotImplementedError as e:
        console.print(f"[yellow]Stage not implemented:[/yellow] {e}")
        return 3
    except Exception as e:  # pragma: no cover
        console.print(f"[red]Stage crashed:[/red] {e}")
        traceback.print_exc()
        return 4

    status_style = {
        "PASS": "green",
        "FAIL": "red",
        "PARTIAL": "yellow",
    }.get(report.status, "white")
    console.print(
        f"\n[{status_style}]==> Stage {args.stage}: {report.status}[/{status_style}]"
    )
    console.print(f"    Wall-clock: {report.wall_clock_seconds:.1f}s")
    console.print(f"    LLM calls: {report.llm_calls_total}  (cache hits: {report.cache_hits})")
    console.print(f"    Report: {Path(report.output_dir) / 'STAGE_REPORT.md'}")

    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
