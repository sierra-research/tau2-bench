"""Package-owned authoring commands for regional database maps."""

import argparse
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import Field, model_validator

from tau2.config import (
    DEFAULT_FACTORY_LOCALIZATION_BATCH_SIZE,
    DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY,
)
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.localization_reuse import ReviewedReuseSource
from tau2.multilingual.localization import (
    LocalizationCategory,
    LocalizationModel,
    RegionalSettings,
)


class DatabaseDraftCommand(LocalizationModel):
    """File-backed inputs to the complete typed database drafting request."""

    source_file: Path = Field(
        description="Inventory JSON; relative to this request file."
    )
    settings: RegionalSettings = Field(description="Explicit regional defaults.")
    revision: str = Field(min_length=1, description="New regional data revision.")
    generation: FactoryModelConfig = Field(
        description="Pinned authoring model and reasoning effort."
    )
    reuse_source: ReviewedReuseSource | None = Field(
        default=None,
        description="Approved retail identity map to reuse for banking customers or incidental intake callees.",
    )
    reference_files: list[Path] = Field(
        default_factory=list,
        description="Evidence files; relative to this request file.",
    )
    batch_size: int = Field(
        default=DEFAULT_FACTORY_LOCALIZATION_BATCH_SIZE,
        ge=1,
        le=100,
        description="Entries per model request.",
    )
    max_concurrency: int = Field(
        default=DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY,
        ge=1,
        le=10,
        description="Independent concurrent model requests.",
    )


class DatabaseRevisionCommand(LocalizationModel):
    """Explicit previous artifact and selected fields for a new draft revision."""

    previous_file: Path = Field(
        description="Compact map, expanded snapshot or typed failed draft; relative to this request."
    )
    previous_kind: Literal["map", "failed", "snapshot"] = Field(
        default="map",
        description="Explicit artifact kind; failed evidence is never an ordinary map.",
    )
    refresh_inventory: bool = Field(
        default=False,
        description="Explicitly rebind a snapshot/failed draft onto refined ownership over identical canonical documents.",
    )
    revision: str = Field(min_length=1, description="New immutable regional revision.")
    generation: FactoryModelConfig = Field(
        description="Pinned authoring model and reasoning effort."
    )
    redraft_ids: list[str] = Field(
        default_factory=list,
        description="Exact entry IDs to regenerate, including explicitly selected dependent text.",
    )
    redraft_categories: list[LocalizationCategory] = Field(
        default_factory=list,
        description="Categories whose entries are explicitly selected for regeneration.",
    )
    reference_files: list[Path] = Field(
        default_factory=list, description="Evidence files relative to this request."
    )
    feedback_files: list[Path] = Field(
        default_factory=list,
        description="Review facts relative to this request, supplied to the fixed revision prompt.",
    )
    batch_size: int = Field(
        default=DEFAULT_FACTORY_LOCALIZATION_BATCH_SIZE,
        ge=1,
        le=100,
        description="Entries per batch.",
    )
    max_concurrency: int = Field(
        default=DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY,
        ge=1,
        le=10,
        description="Concurrent generation batches.",
    )

    @model_validator(mode="after")
    def selected_entries(self):
        if self.refresh_inventory and self.previous_kind == "map":
            raise ValueError(
                "Inventory refresh requires an expanded snapshot or failed draft"
            )
        if not self.redraft_ids and not self.redraft_categories:
            raise ValueError("Select explicit redraft IDs or categories")
        if len(set(self.redraft_ids)) != len(self.redraft_ids) or len(
            set(self.redraft_categories)
        ) != len(self.redraft_categories):
            raise ValueError("Duplicate regional revision selection")
        return self


def _relative(path: Path, request: Path) -> Path:
    return path if path.is_absolute() else request.parent / path


def run_database_draft(args: argparse.Namespace) -> None:
    """Resolve files into the recorded generation request, then resume drafting."""
    from tau2.multilingual.factory.database_localization import (
        LocalizationReference,
        RegionalDraftRequest,
        draft_database_localization,
    )
    from tau2.multilingual.localization import LocalizationSource

    command = DatabaseDraftCommand.model_validate_json(args.request.read_text())
    source = LocalizationSource.model_validate_json(
        _relative(command.source_file, args.request).read_text()
    )
    reused = []
    if command.reuse_source is not None:
        from tau2.multilingual.factory.localization_reuse import reuse_banking_entities

        reference = command.reuse_source.model_copy(
            update={
                "map_file": _relative(command.reuse_source.map_file, args.request),
                "approval_file": _relative(
                    command.reuse_source.approval_file, args.request
                ),
            }
        )
        if source.domain == "intake":
            from tau2.multilingual.factory.intake_authoring import reuse_intake_callees

            reused = reuse_intake_callees(source, reference, command.settings.language)
        else:
            reused = reuse_banking_entities(
                source, reference, command.settings.language
            )
    request = RegionalDraftRequest(
        source=source,
        reused_entries=reused,
        settings=command.settings,
        revision=command.revision,
        generation=command.generation,
        batch_size=command.batch_size,
        max_concurrency=command.max_concurrency,
        references=[
            LocalizationReference(
                name=str(path), content=_relative(path, args.request).read_text()
            )
            for path in command.reference_files
        ],
    )
    if getattr(args, "pilot_per_category", None) is not None:
        from tau2.multilingual.factory.banking_localization import (
            draft_banking_entity_pilot,
        )

        draft_banking_entity_pilot(
            request, args.output, args.checkpoints, per_category=args.pilot_per_category
        )
    else:
        draft_database_localization(request, args.output, args.checkpoints)
    logger.info("Regional draft awaiting content review: {}", args.output)


def run_database_revise(args: argparse.Namespace) -> None:
    """Repair explicit fields of a source-verified map or failed draft."""
    from tau2.multilingual.factory.database_localization import (
        FailedRegionalDraft,
        LocalizationReference,
        RegionalDraftRequest,
        RegionalInventoryRevision,
        RegionalRevision,
        revise_database_localization,
    )
    from tau2.multilingual.factory.localization_artifact import read_localization_map
    from tau2.multilingual.factory.localization_compile import (
        rebind_localization_inventory,
    )
    from tau2.multilingual.factory.localization_inventory import inventory_localization
    from tau2.multilingual.localization import LocalizationMap
    from tau2.utils import DATA_DIR

    command = DatabaseRevisionCommand.model_validate_json(args.request.read_text())
    previous_path = _relative(command.previous_file, args.request)
    parent = None
    if command.previous_kind in {"failed", "snapshot"}:
        previous = (
            FailedRegionalDraft.model_validate_json(previous_path.read_text()).proposal
            if command.previous_kind == "failed"
            else LocalizationMap.model_validate_json(previous_path.read_text())
        )
        current = inventory_localization(previous.source.domain, DATA_DIR)
        if command.refresh_inventory:
            parent = previous
            previous = rebind_localization_inventory(previous, current)
        elif current != previous.source:
            raise ValueError(
                "Regional snapshot source differs from current inventory; explicit refresh_inventory is required"
            )
    else:
        previous = read_localization_map(previous_path)
    available = {entry.id for entry in previous.source.entries}
    unknown = set(command.redraft_ids) - available
    if unknown:
        raise ValueError(f"Unknown regional revision entry IDs: {sorted(unknown)}")
    selected = set(command.redraft_ids)
    for category in command.redraft_categories:
        matches = {
            entry.id for entry in previous.source.entries if entry.category == category
        }
        if not matches:
            raise ValueError(f"Selected regional category has no entries: {category}")
        selected.update(matches)

    def references(paths: list[Path]) -> list[LocalizationReference]:
        return [
            LocalizationReference(
                name=str(path), content=_relative(path, args.request).read_text()
            )
            for path in paths
        ]

    request = RegionalDraftRequest(
        source=previous.source,
        settings=previous.settings,
        revision=command.revision,
        generation=command.generation,
        references=references(command.reference_files),
        batch_size=command.batch_size,
        max_concurrency=command.max_concurrency,
    )
    revision = RegionalRevision(
        previous=previous,
        request=request,
        redraft_ids=[
            entry.id for entry in previous.source.entries if entry.id in selected
        ],
        feedback=references(command.feedback_files),
    )
    if parent is not None:
        revision = RegionalInventoryRevision(parent=parent, revision=revision)
    revise_database_localization(revision, args.output, args.checkpoints)
    logger.info("Regional revision awaiting content review: {}", args.output)


def add_database_localization_args(commands) -> None:
    """Register selective regional-map drafting and revision with tau2 factory.

    These verbs author banking and intake maps and repair released maps; the
    integrated ``draft``/``stage-pack``/``add-domain`` stages remain the path
    for a new language's airline, retail and telecom maps.
    """
    draft = commands.add_parser(
        "draft-database-localization",
        help="Draft native/romanized regional maps through fixed prompts",
    )
    draft.add_argument("--request", type=Path, required=True)
    draft.add_argument("--output", type=Path, required=True)
    draft.add_argument("--checkpoints", type=Path, required=True)
    draft.add_argument(
        "--pilot-per-category",
        type=int,
        help="Banking only: draft this many uncovered entities per family; output is an incomplete pilot, never publishable.",
    )
    draft.set_defaults(func=run_database_draft)
    revise = commands.add_parser(
        "revise-database-localization",
        help="Redraft selected fields while retaining verified regional mappings",
    )
    revise.add_argument("--request", type=Path, required=True)
    revise.add_argument("--output", type=Path, required=True)
    revise.add_argument("--checkpoints", type=Path, required=True)
    revise.set_defaults(func=run_database_revise)
