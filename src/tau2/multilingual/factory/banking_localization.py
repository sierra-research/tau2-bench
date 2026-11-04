"""Bounded, resumable banking entity pilots before full locale authoring."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from tau2.multilingual.factory.database_localization import (
    DATABASE_LOCALIZATION_CALL_NAME,
    DATABASE_LOCALIZATION_PROMPT_VERSION,
    BatchInput,
    RegionalDraftRequest,
    _generate_batch,
)
from tau2.multilingual.factory.localization_artifact import (
    SourceManifest,
    _manifest,
    automatic_localization,
)
from tau2.multilingual.factory.localization_reuse import validate_reused_entries
from tau2.multilingual.localization import (
    GenerationProvenance,
    LocalizationEntry,
    LocalizationModel,
    RegionalSettings,
)
from tau2.utils.artifacts import content_digest, write_artifact


class BankingEntityPilot(LocalizationModel):
    """Incomplete candidate review artifact; never a publishable localization map."""

    schema_version: Literal["banking-entity-pilot-v1"] = "banking-entity-pilot-v1"
    revision: str = Field(min_length=1, description="Pilot artifact revision.")
    source: SourceManifest = Field(
        description="Complete pinned source manifest; only a subset is authored here."
    )
    settings: RegionalSettings = Field(
        description="Banking identity locale with USD scale 1."
    )
    entries: list[LocalizationEntry] = Field(
        description="Reused candidates and bounded newly generated entity candidates."
    )
    provenance: GenerationProvenance = Field(
        description="Actual fixed-prompt model call and inputs for the new candidates."
    )
    pending_entry_ids: list[str] = Field(
        description="Unwritten nonautomatic entries; this pilot is incomplete."
    )
    status: Literal["draft"] = "draft"


def draft_banking_entity_pilot(
    request: RegionalDraftRequest, output: Path, checkpoints: Path, *, per_category: int
) -> BankingEntityPilot:
    """Draft a bounded sample after reuse, preserving all source and call provenance."""
    if request.source.domain != "banking_knowledge" or per_category < 1:
        raise ValueError(
            "Banking pilot requires banking source and positive sample size"
        )
    if (
        request.settings.currency_code != "USD"
        or request.settings.currency_scale != "1"
    ):
        raise ValueError("Banking pilots preserve USD and scale 1")
    digest = content_digest(request)
    reused = validate_reused_entries(request.source, request.reused_entries)
    pending = [
        e
        for e in request.source.entries
        if e.id not in reused and automatic_localization(e, request.settings) is None
    ]
    selected = []
    for category in ("identity_name", "address", "contact", "other"):
        # IDs encode semantic source identity and remain stable across file ordering.
        selected.extend(
            sorted((e for e in pending if e.category == category), key=lambda e: e.id)[
                :per_category
            ]
        )
    batch = BatchInput(request_sha256=digest, entries=selected, completed=[])
    if output.exists():
        existing = BankingEntityPilot.model_validate_json(output.read_text())
        expected_ids = set(reused) | {e.id for e in selected}
        if (
            existing.provenance.input_sha256 != content_digest(batch)
            or {e.id for e in existing.entries} != expected_ids
        ):
            raise ValueError("Pilot output already exists for different inputs")
        return existing
    generated = _generate_batch(request, batch, checkpoints)
    entries = [*reused.values(), *generated]
    covered = {e.id for e in entries}
    pilot = BankingEntityPilot(
        revision=request.revision,
        settings=request.settings,
        source=_manifest(request.source),
        entries=entries,
        pending_entry_ids=[e.id for e in pending if e.id not in covered],
        provenance=GenerationProvenance(
            model=request.model,
            model_args=request.model_args,
            prompt_version=DATABASE_LOCALIZATION_PROMPT_VERSION,
            call_name=DATABASE_LOCALIZATION_CALL_NAME,
            input_sha256=content_digest(batch),
        ),
    )
    write_artifact(pilot, output)
    return pilot
