"""CLI entry point for tau2-reliability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tau2-reliability",
        description="Comprehensive agent reliability analysis for tau2-bench",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- analyze ---
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze existing multi-trial results for reliability",
    )
    analyze_parser.add_argument(
        "--results-path", type=str, required=True,
        help="Path to tau2-bench results JSON file or directory",
    )
    analyze_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for report, CSV, and plots",
    )
    analyze_parser.add_argument(
        "--n-bootstrap", type=int, default=200,
        help="Number of bootstrap resamples for SE (default: 200)",
    )
    analyze_parser.add_argument(
        "--bootstrap-seed", type=int, default=42,
        help="Random seed for bootstrap (default: 42)",
    )

    # --- visualize ---
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Generate visualizations from a reliability report",
    )
    viz_parser.add_argument(
        "--report-path", type=str, required=True,
        help="Path to reliability_report.json",
    )
    viz_parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for plots",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "analyze":
        _run_analyze(args)
    elif args.command == "visualize":
        _run_visualize(args)


def _run_analyze(args: argparse.Namespace) -> None:
    from tau2_reliability.runners.analyze import analyze_results

    logger.info(f"Analyzing results from: {args.results_path}")
    report = analyze_results(
        results_path=args.results_path,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    logger.info(f"Analysis complete: {report.num_tasks} tasks, {report.num_trials} trials")
    if report.r_con is not None:
        logger.info(f"  R_Con = {report.r_con:.3f}")
    if report.r_pred is not None:
        logger.info(f"  R_Pred = {report.r_pred:.3f}")
    if report.r_overall is not None:
        logger.info(f"  R_Overall = {report.r_overall:.3f}")


def _run_visualize(args: argparse.Namespace) -> None:
    from tau2_reliability.models import ReliabilityReport

    report_path = Path(args.report_path)
    report = ReliabilityReport.model_validate_json(report_path.read_text())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tau2_reliability.visualization.dashboard import (
        plot_consistency_detail,
        plot_per_task_heatmap,
        plot_reliability_dashboard,
    )

    fig = plot_reliability_dashboard(report)
    fig.savefig(output_dir / "reliability_dashboard.png")
    plt.close(fig)

    fig = plot_consistency_detail(report)
    fig.savefig(output_dir / "consistency_detail.png")
    plt.close(fig)

    fig = plot_per_task_heatmap(report)
    fig.savefig(output_dir / "per_task_heatmap.png")
    plt.close(fig)

    logger.info(f"Plots written to {output_dir}")


if __name__ == "__main__":
    main()
