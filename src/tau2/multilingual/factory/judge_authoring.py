"""Factory commands for independent agent rubric authoring and compilation."""

from pathlib import Path

from tau2.utils.artifacts import content_digest


def add_judge_authoring_args(commands) -> None:
    """Register independent agent criteria drafting and reviewed candidate compilation."""
    draft = commands.add_parser("draft-judge", help="Author independent agent criteria")
    draft.add_argument("--request", type=Path, required=True)
    draft.add_argument("--output", type=Path, required=True)
    draft.add_argument(
        "--catalog-entry",
        type=Path,
        action="append",
        help="Additional reviewed candidate sources; contributed entries are discovered automatically",
    )
    draft.set_defaults(func=run_draft_judge)
    compile_ = commands.add_parser(
        "compile-judge", help="Compile reviewed criteria into an UNVALIDATED candidate"
    )
    compile_.add_argument("--request", type=Path, required=True)
    compile_.add_argument("--output", type=Path, required=True)
    compile_.set_defaults(func=run_compile_judge)
    register = commands.add_parser(
        "register-judge",
        help="Publish a reviewed reusable authoring entry; never enable production",
    )
    register.add_argument("--candidate", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    register.set_defaults(func=run_register_judge)
    listing = commands.add_parser(
        "list-judges",
        help="List authoring ideas, validated sources and reusable reviewed entries",
    )
    listing.add_argument(
        "--catalog-entry",
        type=Path,
        action="append",
        help="Additional reviewed candidate source files",
    )
    listing.set_defaults(func=run_list_judges)


def _report_unvalidated(artifact, output: Path) -> None:
    print(
        f"Unvalidated authoring artifact: {output}; review digest {content_digest(artifact)}"
    )


def run_draft_judge(args) -> None:
    """Validate the draft request before invoking the package-owned authoring API."""
    from tau2.judges.authoring import draft_judge
    from tau2.judges.authoring_models import DraftJudgeRequest
    from tau2.judges.authoring_registry import load_authoring_entries

    artifact = draft_judge(
        DraftJudgeRequest.model_validate_json(args.request.read_text()),
        args.output,
        catalog_entries=load_authoring_entries(args.catalog_entry),
    )
    _report_unvalidated(artifact, args.output)


def run_compile_judge(args) -> None:
    """Compile reviewed criteria into an unvalidated candidate."""
    from tau2.judges.authoring import compile_judge
    from tau2.judges.authoring_models import CompileJudgeRequest

    artifact = compile_judge(
        CompileJudgeRequest.model_validate_json(args.request.read_text()), args.output
    )
    _report_unvalidated(artifact, args.output)


def run_register_judge(args) -> None:
    """Publish a reviewed authoring definition, without model calls."""
    from tau2.judges.authoring_models import CandidateJudge
    from tau2.judges.authoring_registry import register_judge

    artifact = register_judge(
        CandidateJudge.model_validate_json(args.candidate.read_text()), args.output
    )
    print(artifact.model_dump_json(indent=2))


def run_list_judges(args) -> None:
    """Read reviewed authoring definitions, without model calls."""
    from tau2.judges.authoring_registry import list_authoring_catalog

    print(list_authoring_catalog(args.catalog_entry).model_dump_json(indent=2))
