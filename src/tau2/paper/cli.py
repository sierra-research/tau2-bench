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

    rescore = commands.add_parser(
        "elicitation-rescore",
        help="Rebuild or verify the tau-Elicitation reward-correction ledger",
    )
    rescore.add_argument(
        "--root",
        type=Path,
        default=Path("papers/tau-intake/v1/reproduction"),
    )
    rescore.add_argument(
        "--check",
        action="store_true",
        help="Verify that the checked correction ledger reproduces exactly",
    )
    rescore.set_defaults(func=run_elicitation_rescore)


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


def run_elicitation_rescore(args: argparse.Namespace) -> None:
    """Rebuild or verify the immutable reward-correction overlay."""
    from tau2.paper.elicitation_scoring import (
        artifact_path,
        build_scoring_correction,
        write_scoring_correction,
    )

    artifact = build_scoring_correction(args.root)
    target = artifact_path(args.root)
    if args.check:
        expected = target.read_text() if target.exists() else ""
        actual = artifact.model_dump_json(indent=2) + "\n"
        if expected != actual:
            raise SystemExit(f"Scoring correction differs from {target}")
        logger.info(
            "Verified {} corrected calls across {} manifest cells",
            artifact.corrected_calls,
            artifact.checks.manifest_cells,
        )
        return
    write_scoring_correction(args.root)
    logger.info("Wrote {} corrections to {}", artifact.corrected_calls, target)
