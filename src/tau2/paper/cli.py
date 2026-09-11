# Copyright Sierra
"""Package-owned ``tau2 paper`` command registration."""

import argparse
from pathlib import Path

from loguru import logger


def add_paper_args(parser: argparse.ArgumentParser) -> None:
    """Attach paper reproduction and evidence-export commands."""
    commands = parser.add_subparsers(
        dest="paper_command", help="Paper reproduction commands", required=True
    )

    audit = commands.add_parser(
        "multilingual-audit",
        help="Audit the frozen τ-Multilingual cohort and reproduce core tables",
    )
    audit.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="Frozen tau-multi results root containing main_runs/ and text_channel/",
    )
    audit.add_argument(
        "--out",
        type=Path,
        default=Path("papers/tau-multilingual/reproduction"),
        help="Output directory for JSON, Markdown, and CSV artifacts",
    )
    audit.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when a required cell or paper claim fails validation",
    )
    audit.set_defaults(func=run_multilingual_audit)

    listening = commands.add_parser(
        "multilingual-listening-sample",
        help="Export the deterministic three-call-per-language listening sample",
    )
    listening.add_argument("--evidence-root", type=Path, required=True)
    listening.add_argument("--out", type=Path, required=True)
    listening.set_defaults(func=run_multilingual_listening_sample)

    prompts = commands.add_parser(
        "multilingual-prompt-snapshots",
        help="Export exact rendered prompts and inputs for every frozen result",
    )
    prompts.add_argument("--evidence-root", type=Path, required=True)
    prompts.add_argument("--out", type=Path, required=True)
    prompts.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository containing the git revisions recorded by the runs",
    )
    prompts.set_defaults(func=run_multilingual_prompt_snapshots)

    verify_prompts = commands.add_parser(
        "multilingual-verify-prompts",
        help="Verify the hashes and cell mappings in a prompt snapshot archive",
    )
    verify_prompts.add_argument("--root", type=Path, required=True)
    verify_prompts.add_argument(
        "--evidence-root",
        type=Path,
        help="Also compare every object with the frozen result that produced it",
    )
    verify_prompts.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository containing historical pack revisions for source comparison",
    )
    verify_prompts.set_defaults(func=run_multilingual_verify_prompts)

    annotations = commands.add_parser(
        "multilingual-annotations",
        help="Export normalized final precision and recall annotation results",
    )
    annotations.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Final annotation labels JSON",
    )
    annotations.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="Frozen annotation-evidence directory with one subdirectory per language",
    )
    annotations.add_argument("--out", type=Path, required=True)
    annotations.set_defaults(func=run_multilingual_annotations)

    experience = commands.add_parser(
        "multilingual-experience",
        help="Recompute τ-Multilingual Interaction and utterance Experience",
    )
    experience.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository containing the frozen cohort and final analysis inputs",
    )
    experience.add_argument(
        "--naturalness-sidecar",
        type=Path,
        required=True,
        help="Completed frozen trial-0 combined-naturalness replay",
    )
    experience.add_argument("--out", type=Path, required=True)
    experience.set_defaults(func=run_multilingual_experience)

    ablation_transcripts = commands.add_parser(
        "multilingual-ablation-transcripts",
        help="Export compact transcripts for the retail localization ablation",
    )
    ablation_transcripts.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="Frozen tau-multi results root containing main_runs/ and ablations/",
    )
    ablation_transcripts.add_argument("--out", type=Path, required=True)
    ablation_transcripts.set_defaults(func=run_multilingual_ablation_transcripts)

    verify_ablation_transcripts = commands.add_parser(
        "multilingual-verify-ablation-transcripts",
        help="Verify a compact retail-ablation transcript export",
    )
    verify_ablation_transcripts.add_argument("--root", type=Path, required=True)
    verify_ablation_transcripts.set_defaults(
        func=run_multilingual_verify_ablation_transcripts
    )


def run_multilingual_audit(args) -> None:
    """Run and write the τ-Multilingual frozen-evidence audit."""
    from tau2.paper.multilingual import audit_multilingual, write_audit_artifacts

    report = audit_multilingual(args.evidence_root)
    write_audit_artifacts(report, args.out)
    logger.info(report.summary_line)
    logger.info("Artifacts: {}", args.out)
    if args.strict and not report.ok:
        raise SystemExit(1)


def run_multilingual_listening_sample(args) -> None:
    """Copy the deterministic listening sample and its provenance manifest."""
    from tau2.paper.multilingual import export_listening_sample

    manifest = export_listening_sample(args.evidence_root, args.out)
    logger.info(
        "Exported {} calls ({} per language) to {}",
        len(manifest.calls),
        manifest.calls_per_language,
        args.out,
    )


def run_multilingual_annotations(args) -> None:
    """Export reviewer-safe final annotation results."""
    from tau2.paper.annotations import export_annotations

    manifest = export_annotations(args.labels, args.evidence_root, args.out)
    logger.info(
        "Exported {} final annotation files to {}", len(manifest.files), args.out
    )


def run_multilingual_prompt_snapshots(args) -> None:
    """Export immutable rendered prompts and inputs for every frozen result."""
    from tau2.paper.prompt_snapshots import export_prompt_snapshots

    manifest = export_prompt_snapshots(
        args.evidence_root,
        args.out,
        repo=args.repo_root,
    )
    logger.info(
        "Exported {} prompt cells and {} unique objects to {}",
        len(manifest.cells),
        len(manifest.objects),
        args.out,
    )


def run_multilingual_verify_prompts(args) -> None:
    """Verify an existing immutable prompt archive."""
    from tau2.paper.prompt_snapshots import (
        verify_prompt_snapshot_export,
        verify_prompt_snapshot_sources,
    )

    report = (
        verify_prompt_snapshot_sources(
            args.evidence_root,
            args.root,
            repo=args.repo_root,
        )
        if args.evidence_root
        else verify_prompt_snapshot_export(args.root)
    )
    if not report.ok:
        for problem in report.problems:
            logger.error(problem)
        raise SystemExit(1)
    checked = getattr(report, "checked_cells", None)
    if checked is None:
        logger.info("Verified {} prompt objects", report.checked_objects)
    else:
        logger.info("Verified prompt sources for {} result cells", checked)


def run_multilingual_experience(args) -> None:
    """Recompute the Interaction and utterance Experience result artifact."""
    from experiments.tau_multilingual.experience_without_fluency import write_analysis

    write_analysis(
        args.repo_root,
        args.out,
        naturalness_sidecar=args.naturalness_sidecar,
    )
    logger.info("Wrote Interaction and utterance Experience analysis to {}", args.out)


def run_multilingual_ablation_transcripts(args) -> None:
    """Export the compact retail-ablation call transcripts."""
    from tau2.paper.ablation_transcripts import export_retail_ablation_transcripts

    manifest = export_retail_ablation_transcripts(args.evidence_root, args.out)
    logger.info(
        "Exported {} retail-ablation transcripts from {} result files to {}",
        manifest.rows,
        len(manifest.source_runs),
        args.out,
    )


def run_multilingual_verify_ablation_transcripts(args) -> None:
    """Verify the compact retail-ablation transcript artifact."""
    from tau2.paper.ablation_transcripts import verify_retail_ablation_transcripts

    manifest = verify_retail_ablation_transcripts(args.root)
    logger.info("Verified {} retail-ablation transcripts", manifest.rows)
