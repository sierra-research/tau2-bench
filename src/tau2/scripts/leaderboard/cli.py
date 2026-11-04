"""Package-owned leaderboard submission commands."""

import argparse
from pathlib import Path


def add_submit_args(parser: argparse.ArgumentParser) -> None:
    """Register submission preparation, independent validation, and trajectory tools."""
    submit_subparsers = parser.add_subparsers(
        dest="submit_command", help="Submit subcommands", required=True
    )

    # Prepare ordinary or frozen-matrix submissions.
    submit_prepare_parser = submit_subparsers.add_parser(
        "prepare", help="Prepare a submission for the leaderboard"
    )
    submit_prepare_parser.add_argument(
        "input_paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    submit_prepare_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory to save the submission and trajectories",
    )
    submit_prepare_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip trajectory verification step",
    )
    submit_prepare_parser.add_argument(
        "--voice",
        action="store_true",
        default=None,
        help="Force voice submission mode (auto-detected from input data if not specified)",
    )
    submit_prepare_parser.add_argument(
        "--benchmark",
        help="Verify this published preset; otherwise infer from matrix metadata",
    )
    submit_prepare_parser.add_argument(
        "--metadata-file",
        type=Path,
        help="2.0 packages: JSON with model_name, model_organization, "
        "submitting_organization, submission_date and contact_info.email",
    )
    _add_historical_args(submit_prepare_parser)
    submit_prepare_parser.set_defaults(func=lambda args: run_prepare_submission(args))

    # Submit validate subcommand
    submit_validate_parser = submit_subparsers.add_parser(
        "validate", help="Validate an existing submission directory"
    )
    submit_validate_parser.add_argument(
        "submission_dir",
        help="Path to the submission directory to validate",
    )
    submit_validate_parser.add_argument(
        "--benchmark",
        help="Verify this published preset; otherwise infer from submission metadata",
    )
    submit_validate_parser.add_argument(
        "--results",
        type=Path,
        help="Original saved benchmark matrix (or, with --historical-voice-v2, "
        "the frozen release export) for independent validation",
    )
    _add_historical_args(submit_validate_parser)
    submit_validate_parser.set_defaults(func=lambda args: run_validate_submission(args))

    # Submit verify-trajs subcommand
    submit_verify_parser = submit_subparsers.add_parser(
        "verify-trajs", help="Verify trajectory files"
    )
    submit_verify_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    submit_verify_parser.set_defaults(func=lambda args: run_verify_trajectories(args))

    # Submit interaction-metrics subcommand
    submit_im_parser = submit_subparsers.add_parser(
        "interaction-metrics",
        help="Compute voice interaction metrics (latency, responsiveness, "
        "interrupts, selectivity) from full-duplex trajectories",
    )
    submit_im_parser.add_argument(
        "input_paths",
        nargs="+",
        help="Voice experiment directories (results.json + simulations/) or a "
        "parent directory such as a submission's trajectories/ dir",
    )
    submit_im_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path to write the interaction_metrics JSON block",
    )
    submit_im_parser.set_defaults(func=lambda args: run_interaction_metrics(args))


def _add_historical_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--historical-voice-v2",
        action="store_true",
        help="Maintainers only: read the frozen 2026-10-02 Voice V2 release export; "
        "--benchmark voice (default), banking or entity selects the frame",
    )
    parser.add_argument(
        "--judges-root",
        type=Path,
        action="append",
        help="With --historical-voice-v2 (voice frame only): a provider's saved "
        "verdict directory (<language>/<simulation-id>.json); repeat for each "
        "judge suite",
    )


def _check_historical_args(args, *, needs: tuple[str, ...]) -> bool:
    """True for a historical invocation; reject historical inputs elsewhere.

    The voice frame is judged and needs ``--judges-root``; banking and entity
    carry no judge diagnostics and take none.
    """
    if not args.historical_voice_v2:
        if args.judges_root:
            raise ValueError("--judges-root requires --historical-voice-v2")
        return False
    from tau2.scripts.leaderboard.voice_v2_publication import HISTORICAL_BENCHMARKS

    if args.benchmark not in (None, *HISTORICAL_BENCHMARKS):
        raise ValueError(
            "--historical-voice-v2 packages --benchmark "
            + ", ".join(HISTORICAL_BENCHMARKS)
        )
    missing = [name for name in needs if not getattr(args, name)]
    if missing:
        raise ValueError(
            "--historical-voice-v2 requires "
            + ", ".join("--" + name.replace("_", "-") for name in needs)
        )
    return True


def run_verify_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.leaderboard.verify_trajectories import (
        VerificationMode,
        verify_trajectories,
    )

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    verify_trajectories(args.paths, VerificationMode.PUBLIC)


def _quiet_package_logs() -> None:
    """2.0 packages re-read every call; per-call metric debug logs are noise."""
    import sys

    from loguru import logger

    logger.configure(handlers=[{"sink": sys.stderr, "level": "WARNING"}])


def run_prepare_submission(args):
    """Run the prepare submission command."""
    from tau2.metrics.benchmark import IncompleteSubmissionError

    if _check_historical_args(args, needs=("metadata_file",)):
        if args.no_verify or len(args.input_paths) != 1:
            raise ValueError(
                "Historical Voice V2 requires one saved export and full verification"
            )
        from tau2.scripts.leaderboard.voice_v2_publication import (
            prepare_historical_package,
        )

        _quiet_package_logs()

        prepare_historical_package(
            Path(args.input_paths[0]),
            args.judges_root,
            args.metadata_file,
            Path(args.output),
            args.benchmark or "voice",
        )
        return
    from tau2.scripts.leaderboard.prepare_submission import prepare_submission

    has_matrix = any(
        (Path(path) / "benchmark-frame.json").is_file() for path in args.input_paths
    )
    if args.benchmark or has_matrix:
        if len(args.input_paths) != 1 or args.no_verify:
            raise ValueError(
                "Benchmark submissions require one matrix root and full verification"
            )
        if args.metadata_file is None:
            raise SystemExit(
                "Error: 2.0 packages require --metadata-file (see "
                "docs/leaderboard-submission.md#20-benchmarks-text-v1-and-voice-v2)"
            )
        from tau2.scripts.leaderboard.prepare_submission import (
            prepare_benchmark_submission,
        )

        _quiet_package_logs()

        try:
            prepare_benchmark_submission(
                Path(args.input_paths[0]),
                Path(args.output),
                args.metadata_file,
                args.benchmark,
            )
        except IncompleteSubmissionError as error:
            # An expected outcome of an unfinished run, not a crash.
            raise SystemExit(f"Error: {error}") from None
        return
    if args.metadata_file is not None:
        raise ValueError("--metadata-file is only used for 2.0 packages")
    prepare_submission(
        input_paths=args.input_paths,
        output_dir=args.output,
        run_verification=not args.no_verify,
        voice=args.voice if args.voice else None,
    )


def run_validate_submission(args):
    """Run the validate submission command."""
    from tau2.metrics.benchmark import IncompleteSubmissionError
    from tau2.scripts.leaderboard.benchmark_package import is_package
    from tau2.scripts.leaderboard.prepare_submission import validate_submission

    directory = Path(args.submission_dir)
    if _check_historical_args(args, needs=("results",)):
        from tau2.scripts.leaderboard.voice_v2_publication import (
            validate_historical_package,
        )

        _quiet_package_logs()

        package = validate_historical_package(
            directory, args.results, args.judges_root, args.benchmark
        )
        print(package.submission.summary.model_dump_json(indent=2))
        return
    if args.benchmark or is_package(directory):
        if args.results is None:
            raise ValueError(
                "Benchmark validation requires --results with the original saved matrix"
            )
        from tau2.scripts.leaderboard.prepare_submission import (
            validate_benchmark_submission,
        )

        _quiet_package_logs()

        try:
            package = validate_benchmark_submission(
                directory, args.results, args.benchmark
            )
        except IncompleteSubmissionError as error:
            raise SystemExit(f"Error: {error}") from None
        print(package.submission.summary.model_dump_json(indent=2))
        return
    if args.results is not None:
        raise ValueError("--results is only used for benchmark matrix submissions")
    validate_submission(submission_dir=args.submission_dir)


def run_interaction_metrics(args):
    """Run the interaction metrics computation command."""
    from tau2.scripts.leaderboard.compute_interaction_metrics import (
        compute_interaction_metrics,
    )

    compute_interaction_metrics(
        input_paths=args.input_paths,
        output_path=args.output,
    )
