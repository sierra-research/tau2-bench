"""Package-owned ``tau2 paper`` command registration."""

import argparse
from pathlib import Path

from loguru import logger


def add_paper_args(parser: argparse.ArgumentParser) -> None:
    """Attach paper reproduction and evidence-export commands."""
    commands = parser.add_subparsers(
        dest="paper_command", help="Paper reproduction commands", required=True
    )

    export = commands.add_parser(
        "elicitation-release",
        help="Build the compact tau-Elicitation reviewer evidence archive",
    )
    export.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="Frozen tau-elicit directory containing main_runs/ and ablations/",
    )
    export.add_argument(
        "--validation-root",
        type=Path,
        required=True,
        help=(
            "Directory containing human_failure_validation_90/ and "
            "fidelity_validation_60/"
        ),
    )
    export.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Directory containing the frozen crossed/rollup analysis inputs",
    )
    export.add_argument(
        "--out",
        type=Path,
        default=Path("papers/tau-intake/v1/reproduction"),
    )
    export.set_defaults(func=run_elicitation_release)

    verify = commands.add_parser(
        "elicitation-verify",
        help="Verify a tau-Elicitation reviewer archive and optional detached evidence",
    )
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--evidence-root", type=Path)
    verify.set_defaults(func=run_elicitation_verify)


def run_elicitation_release(args: argparse.Namespace) -> None:
    """Build the compact reviewer archive from the frozen evidence."""
    from tau2.paper.elicitation import build_release

    manifest = build_release(
        evidence_root=args.evidence_root,
        validation_root=args.validation_root,
        analysis_root=args.analysis_root,
        out=args.out,
    )
    logger.info(
        "Exported {} result cells, {} transcripts, and {} speech judgments to {}",
        len(manifest.result_cells),
        manifest.transcript_count,
        manifest.speech_judgment_count,
        args.out,
    )


def run_elicitation_verify(args: argparse.Namespace) -> None:
    """Verify a compact reviewer archive."""
    from tau2.paper.elicitation import verify_release

    report = verify_release(args.root, evidence_root=args.evidence_root)
    logger.info(report.summary)
    if not report.ok:
        raise SystemExit(1)
