"""
Command-line interface for calculating transcription benchmark metrics.

This module provides the CLI for calculating various metrics on transcription
benchmarks including WER, Quality Score, Significant WER, and Input Coverage.

Usage:
    python -m tau2.voice.transcription.benchmark.calculate_metrics [options]
"""

import argparse
from pathlib import Path
from typing import List

from loguru import logger

from tau2.voice.transcription.benchmark.metrics import (
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_WORKERS,
    MetricConfig,
    process_benchmark_directory,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate metrics for transcription benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("data/benchmark"),
        help="Directory containing benchmark data",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["wer", "quality", "significant_wer", "input_coverage"],
        default=["wer"],
        help="Metrics to calculate (can specify multiple)",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize text before calculation (default: True)",
    )

    parser.add_argument(
        "--no-normalize", action="store_true", help="Disable text normalization"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model to use for calculations",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of concurrent workers for parallel processing",
    )

    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of all metrics even if they already exist",
    )

    return parser.parse_args()


def get_benchmark_directories(benchmark_dir: Path) -> List[Path]:
    """Get all benchmark directories."""
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory {benchmark_dir} does not exist")

    benchmark_dirs = [d for d in benchmark_dir.iterdir() if d.is_dir()]

    if not benchmark_dirs:
        logger.warning(f"No benchmark directories found in {benchmark_dir}")

    return benchmark_dirs


def main():
    """Main entry point for the script."""
    args = parse_arguments()

    # Handle normalize/no-normalize flags
    normalize = not args.no_normalize

    # Create configuration
    config = MetricConfig(
        normalize=normalize,
        model=args.model,
        max_workers=args.max_workers,
        recompute=args.recompute,
    )

    # Get benchmark directories
    try:
        benchmark_dirs = get_benchmark_directories(args.benchmark_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    if not benchmark_dirs:
        return

    # Log configuration
    logger.info(f"Found {len(benchmark_dirs)} benchmark directories")
    logger.info(f"Metrics to calculate: {', '.join(args.metrics)}")
    logger.info(f"Normalization: {'enabled' if config.normalize else 'disabled'}")
    logger.info(f"Model: {config.model}")
    logger.info(f"Max workers: {config.max_workers}")
    logger.info(
        f"Mode: {'Recompute all' if config.recompute else 'Use cached results'}"
    )

    # Process each benchmark
    for benchmark_dir in benchmark_dirs:
        logger.info(f"\nProcessing {benchmark_dir.name}...")

        try:
            process_benchmark_directory(
                benchmark_dir=benchmark_dir, metrics=args.metrics, config=config
            )
        except Exception as e:
            logger.error(f"Error processing {benchmark_dir.name}: {str(e)}")
            continue

    logger.info("\nMetric calculation complete")


if __name__ == "__main__":
    main()
