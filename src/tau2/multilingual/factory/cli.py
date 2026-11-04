"""Package-owned CLI for reviewed language authoring and publication.

Each verb owns one handler. Handlers import their stage lazily so that selecting
one stage never loads unrelated generation or voice-provider dependencies.
Stage requests may reference earlier artifacts as
``{"$artifact": "<path>", "sha256": "<digest>"}`` (see ``request_files``), and
every writing stage prints its output path and review digest as one JSON line.
"""

import argparse
import json
from pathlib import Path

from loguru import logger

from tau2.utils.artifacts import content_digest, write_artifact


def _relative_to_request(path: Path, request: Path) -> Path:
    """Resolve a request-relative directory the same way for every stage."""
    return path if path.is_absolute() else request.parent / path


def _report(stage: str, output: Path, artifact, **extra) -> None:
    """Print where a stage wrote and the digest the next gate or reference binds."""
    print(
        json.dumps(
            {
                "stage": stage,
                "output": str(output),
                "sha256": content_digest(artifact),
                **{key: str(value) for key, value in extra.items()},
            }
        )
    )


def _run_autoform(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.author_form import run_autoform
    from tau2.multilingual.factory.request_files import load_request
    from tau2.multilingual.factory.schema import AutoformRequest

    artifact = run_autoform(load_request(AutoformRequest, args.request), args.output)
    _report("autoform", args.output, artifact)


def _run_draft(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.pipeline import draft
    from tau2.multilingual.factory.request_files import load_request
    from tau2.multilingual.factory.schema import DraftRequest

    artifact = draft(load_request(DraftRequest, args.request), args.output)
    _report(
        "draft",
        args.output,
        artifact,
        review=args.output.with_suffix(".review") / "index.html",
    )


def _run_stage_pack(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.pipeline import stage_pack
    from tau2.multilingual.factory.request_files import load_request
    from tau2.multilingual.factory.schema import StagePackRequest

    artifact = stage_pack(load_request(StagePackRequest, args.request), args.output)
    _report(
        "stage-pack",
        args.output,
        artifact,
        review=args.output.with_suffix(".review") / "index.html",
    )


def _run_add_domain(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.pipeline import add_domain
    from tau2.multilingual.factory.request_files import load_request
    from tau2.multilingual.factory.schema import AddDomainRequest

    artifact = add_domain(load_request(AddDomainRequest, args.request), args.output)
    _report(
        "add-domain",
        args.output,
        artifact,
        review=args.output.with_suffix(".review") / "index.html",
    )


def _run_validate(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.pipeline import validate_draft
    from tau2.multilingual.factory.schema import DraftArtifact

    value = validate_draft(DraftArtifact.model_validate_json(args.request.read_text()))
    _report("validate", args.request, value)


def _run_review_content(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.pipeline import validate_draft
    from tau2.multilingual.factory.regional_build import write_content_review
    from tau2.multilingual.factory.schema import DraftArtifact

    value = validate_draft(DraftArtifact.model_validate_json(args.request.read_text()))
    page = write_content_review(value, args.output)
    _report("review-content", page, value, draft=args.request)


def _run_approve(args: argparse.Namespace) -> None:
    """Record a human approval for the exact digest the reviewer inspected."""
    from tau2.utils.artifacts import ReviewApproval

    if args.gate == "content":
        from tau2.multilingual.factory.pipeline import validate_draft
        from tau2.multilingual.factory.schema import DraftArtifact

        digest = content_digest(
            validate_draft(DraftArtifact.model_validate_json(args.artifact.read_text()))
        )
    else:
        from tau2.multilingual.factory.voice_assets import (
            VoiceAssetsArtifact,
            listening_digest,
            validate_voices,
        )

        digest = listening_digest(
            validate_voices(
                VoiceAssetsArtifact.model_validate_json(args.artifact.read_text()),
                args.artifact.parent,
            )
        )
    if digest != args.digest:
        raise ValueError(
            f"{args.artifact} has {args.gate} digest {digest}, not the reviewed "
            f"{args.digest}; re-review the current artifact"
        )
    approval = ReviewApproval(content_sha256=digest, reviewed_by=args.reviewed_by)
    write_artifact(approval, args.output)
    _report("approve", args.output, approval, gate=args.gate)


def _run_init_request(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.generation_config import FactoryModelConfig
    from tau2.multilingual.factory.init_request import (
        InitRequestOptions,
        init_draft_request,
    )
    from tau2.utils import DATA_DIR

    options = InitRequestOptions(
        language=args.language,
        domains=args.domains,
        reviewed_by=args.reviewed_by,
        generation=FactoryModelConfig(
            model=args.model, reasoning_effort=args.reasoning_effort
        ),
        country_code=args.country,
        currency_code=args.currency,
        currency_scale=args.currency_scale,
        time_zone=args.time_zone,
    )
    path = init_draft_request(
        options, args.form, args.output, args.data_root or DATA_DIR
    )
    print(json.dumps({"stage": "init-request", "output": str(path)}))


def _run_inventory_database(args: argparse.Namespace) -> None:
    """Inventory complete domain databases and the frozen released task set."""
    from tau2.multilingual.factory.localization_inventory import inventory_localization
    from tau2.utils import DATA_DIR

    result = inventory_localization(args.domain, args.data_root or DATA_DIR)
    write_artifact(result, args.output)
    logger.info(
        "Wrote {} source entries for {} to {}",
        len(result.entries),
        args.domain,
        args.output,
    )


def _run_refresh_regional_bundle(args: argparse.Namespace) -> None:
    """Rebind an installed map to this checkout and republish its runtime bundle."""
    from tau2.multilingual.factory.localization_publish import refresh_regional_bundle
    from tau2.utils import DATA_DIR

    bundle = refresh_regional_bundle(
        args.language,
        args.domain,
        args.revision,
        args.output,
        args.data_root or DATA_DIR,
    )
    logger.info(
        "Refreshed {}/{} into {}; map digest {}",
        args.language,
        args.domain,
        args.output,
        bundle.localization_map_sha256,
    )


def _run_prepare_voices(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.request_files import load_request
    from tau2.multilingual.factory.voice_assets import (
        ElevenLabsVoiceProvider,
        VoiceAssetsRequest,
        listening_digest,
        prepare_voices,
    )

    assets = prepare_voices(
        load_request(VoiceAssetsRequest, args.request),
        args.output,
        ElevenLabsVoiceProvider(),
    )
    _report(
        "prepare-voices",
        args.output / "assets.json",
        assets,
        listening_sha256=listening_digest(assets),
    )


def _run_reuse_voices(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.request_files import load_request
    from tau2.multilingual.factory.voice_assets import (
        ReuseAssetsRequest,
        listening_digest,
        reuse_voices,
    )

    request = load_request(ReuseAssetsRequest, args.request)
    request.previous_assets_root = _relative_to_request(
        request.previous_assets_root, args.request
    )
    assets = reuse_voices(request, args.output)
    _report(
        "reuse-voices",
        args.output / "assets.json",
        assets,
        listening_sha256=listening_digest(assets),
    )


def _run_audition(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.voice_assets import (
        VoiceAssetsArtifact,
        audition_samples,
        listening_digest,
    )

    assets = VoiceAssetsArtifact.model_validate_json(args.request.read_text())
    paths = audition_samples(assets, args.output, assets_root=args.request.parent)
    for path in paths:
        logger.info("Audition sample: {}", path)
    print(
        json.dumps(
            {
                "stage": "audition",
                "output": str(args.output / "index.json"),
                "listening_sha256": listening_digest(assets),
            }
        )
    )


def _run_validate_voices(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.voice_assets import (
        VoiceAssetsArtifact,
        validate_voices,
    )

    value = validate_voices(
        VoiceAssetsArtifact.model_validate_json(args.request.read_text()),
        args.request.parent,
    )
    if args.check_provider:
        from tau2.multilingual.factory.voice_assets import ElevenLabsVoiceProvider

        provider = ElevenLabsVoiceProvider()
        for voice in value.voices.values():
            if provider.validate(voice.voice.voice_id) != voice.voice:
                raise ValueError("provider voice identity mismatch")
    from tau2.multilingual.factory.voice_assets import listening_digest

    _report(
        "validate-voices",
        args.request,
        value,
        listening_sha256=listening_digest(value),
    )


def _run_record_created_voice(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.voice_assets import (
        ElevenLabsVoiceProvider,
        recover_voice,
    )

    recover_voice(
        args.directory, args.persona_id, args.voice_id, ElevenLabsVoiceProvider()
    )
    logger.info("Recorded reconciled voice; rerun prepare-voices to resume")


def _run_finalize(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.pipeline import finalize
    from tau2.multilingual.factory.request_files import load_request
    from tau2.multilingual.factory.schema import PublicationProvenance
    from tau2.multilingual.factory.voice_assets import FinalizeRequest

    request = load_request(FinalizeRequest, args.request)
    finalize(
        request.draft,
        request.assets,
        request.review,
        args.output,
        assets_root=_relative_to_request(request.assets_root, args.request),
    )
    provenance = PublicationProvenance.model_validate_json(
        (args.output / "provenance.json").read_text()
    )
    _report(
        "finalize",
        args.output,
        provenance,
        pack_sha256=provenance.files["pack.yaml"],
    )


def _add_stage(commands, name: str, help_text: str, handler, *, output: bool = True):
    """Register one typed-request stage with its own handler."""
    command = commands.add_parser(name, help=help_text)
    command.add_argument(
        "--request", type=Path, required=True, help="Typed stage request JSON"
    )
    if output:
        command.add_argument(
            "--output",
            type=Path,
            required=True,
            help="New immutable artifact path or stage directory",
        )
    command.set_defaults(func=handler)
    return command


def add_factory_args(parser: argparse.ArgumentParser) -> None:
    """Register the public language-authoring stages on the factory subparser."""
    commands = parser.add_subparsers(dest="factory_command", required=True)
    _add_stage(
        commands,
        "autoform",
        "Draft a validated author form for local review",
        _run_autoform,
    )
    inventory = commands.add_parser(
        "inventory-database",
        help="Inventory canonical databases and frozen tasks as a localization source",
    )
    inventory.add_argument(
        "--domain",
        choices=("airline", "retail", "telecom", "banking_knowledge", "intake"),
        required=True,
    )
    inventory.add_argument(
        "--data-root", type=Path, help="Data directory (default: configured DATA_DIR)"
    )
    inventory.add_argument("--output", type=Path, required=True)
    inventory.set_defaults(func=_run_inventory_database)
    refresh = commands.add_parser(
        "refresh-regional-bundle",
        help="Rebind an installed regional map to this checkout's source content "
        "and republish a byte-identical runtime bundle",
    )
    refresh.add_argument("--language", required=True)
    refresh.add_argument(
        "--domain", choices=("airline", "retail", "telecom"), required=True
    )
    refresh.add_argument(
        "--revision", required=True, help="New regional map revision label"
    )
    refresh.add_argument(
        "--data-root", type=Path, help="Data directory (default: configured DATA_DIR)"
    )
    refresh.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; receives localization/ and regional/ files",
    )
    refresh.set_defaults(func=_run_refresh_regional_bundle)
    init = commands.add_parser(
        "init-request",
        help="Write a valid draft-request.json skeleton (frozen-50 tasks, caller "
        "genders, inventories) for a new language; offline",
    )
    init.add_argument("--language", required=True)
    init.add_argument(
        "--domains",
        nargs="+",
        choices=("airline", "retail", "telecom"),
        required=True,
    )
    init.add_argument(
        "--form", type=Path, required=True, help="Reviewed autoform author-form.json"
    )
    init.add_argument(
        "--reviewed-by", required=True, help="Content reviewer attribution"
    )
    init.add_argument("--model", required=True, help="Pinned generation model")
    init.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        required=True,
        help="Pinned reasoning effort",
    )
    init.add_argument("--country", required=True, help="ISO 3166 alpha-2 code")
    init.add_argument("--currency", required=True, help="ISO 4217 code")
    init.add_argument(
        "--currency-scale", required=True, help="Fixed decimal scale from USD"
    )
    init.add_argument("--time-zone", required=True, help="IANA time zone")
    init.add_argument(
        "--data-root", type=Path, help="Data directory (default: configured DATA_DIR)"
    )
    init.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for draft-request.json and referenced inventories",
    )
    init.set_defaults(func=_run_init_request)
    _add_stage(
        commands,
        "draft",
        "Build localized entities and draft language content for one content review",
        _run_draft,
    )
    _add_stage(
        commands,
        "validate",
        "Validate staged content and print its review digest",
        _run_validate,
        output=False,
    )
    _add_stage(
        commands,
        "review-content",
        "Render staged language, entity and task content for human review",
        _run_review_content,
    )
    approve = commands.add_parser(
        "approve",
        help="Record a reviewer's approval of the exact digest they inspected",
    )
    approve.add_argument(
        "--gate",
        choices=("content", "listening"),
        required=True,
        help="content: a draft.json; listening: a voice assets.json",
    )
    approve.add_argument("--artifact", type=Path, required=True)
    approve.add_argument(
        "--digest",
        required=True,
        help="Digest shown to the reviewer (review page or audition index)",
    )
    approve.add_argument("--reviewed-by", required=True)
    approve.add_argument("--output", type=Path, required=True)
    approve.set_defaults(func=_run_approve)
    _add_stage(
        commands,
        "stage-pack",
        "Stage an existing runtime pack for a new domain revision",
        _run_stage_pack,
    )
    _add_stage(
        commands,
        "add-domain",
        "Bind another reviewed supported domain to a staged language",
        _run_add_domain,
    )
    _add_stage(
        commands,
        "prepare-voices",
        "Provision voices and store resumable audition evidence",
        _run_prepare_voices,
    )
    _add_stage(
        commands,
        "reuse-voices",
        "Reuse reviewed voices after a domain-only revision",
        _run_reuse_voices,
    )
    _add_stage(
        commands,
        "audition",
        "Export exact native-text voice previews for listening review",
        _run_audition,
    )
    validate_voices = _add_stage(
        commands,
        "validate-voices",
        "Validate audio and provenance; optionally check voice access",
        _run_validate_voices,
        output=False,
    )
    validate_voices.add_argument(
        "--check-provider",
        action="store_true",
        help="Check voice access using ElevenLabs (network)",
    )
    recover = commands.add_parser(
        "record-created-voice",
        help="Record an account voice after an uncertain creation",
    )
    recover.add_argument("--directory", type=Path, required=True)
    recover.add_argument("--persona-id", required=True)
    recover.add_argument("--voice-id", required=True)
    recover.set_defaults(func=_run_record_created_voice)
    _add_stage(
        commands,
        "finalize",
        "Publish an immutable reviewed runtime language bundle",
        _run_finalize,
    )
    from tau2.multilingual.factory.judge_authoring import add_judge_authoring_args

    add_judge_authoring_args(commands)
    _add_regional_authoring_args(commands)


def _add_regional_authoring_args(commands) -> None:
    """Selective map drafting/revision, pack extensions and review exports."""
    from tau2.multilingual.factory.localization_cli import (
        add_database_localization_args,
    )
    from tau2.multilingual.factory.localization_entity_csv import (
        add_localization_entity_csv_args,
    )
    from tau2.multilingual.factory.localization_spreadsheet import (
        add_localization_spreadsheet_args,
    )
    from tau2.multilingual.factory.pack_extension import add_pack_extension_args

    add_database_localization_args(commands)
    add_pack_extension_args(commands)
    add_localization_spreadsheet_args(commands)
    add_localization_entity_csv_args(commands)
