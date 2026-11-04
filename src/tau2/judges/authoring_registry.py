"""Explicit reviewed diagnostic definitions, separate from production validation."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from tau2.data_model.judges import ContractModel, DiagnosticId
from tau2.judges.authoring_catalog import SOURCE_REVISION, authoring_catalog
from tau2.judges.authoring_models import CandidateJudge
from tau2.judges.catalog import load_catalog
from tau2.utils.artifacts import content_digest, write_artifact


class AuthoringCatalogItem(ContractModel):
    """Discoverable source identity; validation never transfers to adaptations."""

    source: Annotated[
        Literal["authoring_catalog", "validated_contract", "reviewed_candidate"],
        Field(description="Type of reusable authoring reference."),
    ]
    diagnostic_id: Annotated[
        DiagnosticId, Field(description="Diagnostic selected for adaptation.")
    ]
    source_contract_id: Annotated[
        str, Field(description="Exact source selector for DraftJudgeRequest.")
    ]
    language: Annotated[
        str | None, Field(description="Source language; none for shared factor ideas.")
    ]
    revision: Annotated[str, Field(description="Immutable source revision.")]
    source_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Digest of the exact source artifact.",
        ),
    ]
    status: Annotated[
        Literal["unvalidated", "validated"],
        Field(description="Source status only; any adaptation remains unvalidated."),
    ]
    evaluation_level: Annotated[
        Literal["call", "utterance"] | None,
        Field(description="Source unit, if executable."),
    ]
    agent_gender: Annotated[
        Literal["male", "female"] | None,
        Field(
            description="Source participant restriction; never inherited as validation."
        ),
    ]
    domains: Annotated[
        list[str] | None,
        Field(description="Source domain restriction, where specified."),
    ]


class AuthoringCatalogListing(ContractModel):
    """Typed view of packaged sources and explicitly reviewed portable entries."""

    entries: Annotated[
        list[AuthoringCatalogItem],
        Field(
            description="Selectable source records; no production catalog mutations."
        ),
    ]


def default_catalog_directory() -> Path:
    """Use the same relocatable data root as language packs and domain artifacts."""
    from tau2.utils import DATA_DIR

    return DATA_DIR / "tau2" / "judge_catalog"


def validate_authoring_entries(entries: list[CandidateJudge]) -> list[CandidateJudge]:
    """Reject unreviewed definitions and ambiguous source IDs before generation."""
    checked = [CandidateJudge.model_validate(entry.model_dump()) for entry in entries]
    packaged_ids = {contract.contract_id for contract in load_catalog().contracts}
    seen = set()
    for entry in checked:
        key = entry.contract.contract_id
        if key in seen or key in packaged_ids or key == "authoring-catalog":
            raise ValueError(f"Duplicate authoring source contract ID: {key}")
        seen.add(key)
    return checked


def load_authoring_entries(paths: list[Path] | None = None) -> list[CandidateJudge]:
    """Discover contributed entries and read optional explicit portable sources."""
    discovered = sorted(default_catalog_directory().glob("*.json"))
    files = [*discovered, *(paths or [])]
    return validate_authoring_entries(
        [CandidateJudge.model_validate_json(path.read_text()) for path in files]
    )


def register_judge(candidate: CandidateJudge, output: Path) -> CandidateJudge:
    """Publish immutable reviewed authoring evidence without enabling production."""
    candidate = validate_authoring_entries([candidate])[0]
    directory = default_catalog_directory()
    if output.resolve().parent == directory.resolve():
        others = [
            CandidateJudge.model_validate_json(path.read_text())
            for path in sorted(directory.glob("*.json"))
            if path.resolve() != output.resolve()
        ]
        validate_authoring_entries([*others, candidate])
    write_artifact(candidate, output)
    return candidate


def list_authoring_catalog(paths: list[Path] | None = None) -> AuthoringCatalogListing:
    """List shared ideas, exact validated sources and reviewed reusable candidates."""
    entries = [
        AuthoringCatalogItem(
            source="authoring_catalog",
            diagnostic_id=factor.factor_id,
            source_contract_id="authoring-catalog",
            language=None,
            revision=SOURCE_REVISION,
            source_sha256=content_digest(factor),
            status="unvalidated",
            evaluation_level=None,
            agent_gender=None,
            domains=None,
        )
        for factor in authoring_catalog()
    ]
    catalog = load_catalog()
    contracts = {contract.contract_id: contract for contract in catalog.contracts}
    for pair in catalog.pairs:
        if pair.contract_id in catalog.unvalidated_contract_ids:
            continue
        contract = contracts[pair.contract_id]
        entries.append(
            AuthoringCatalogItem(
                source="validated_contract",
                diagnostic_id=pair.measure,
                source_contract_id=pair.contract_id,
                language=pair.language,
                revision=catalog.revision,
                source_sha256=content_digest(contract),
                status="validated",
                evaluation_level=pair.evaluation_level,
                agent_gender=pair.agent_gender,
                domains=pair.domains,
            )
        )
    for candidate in load_authoring_entries(paths):
        entries.append(
            AuthoringCatalogItem(
                source="reviewed_candidate",
                diagnostic_id=candidate.scope.measure,
                source_contract_id=candidate.contract.contract_id,
                language=candidate.scope.language,
                revision=candidate.revision,
                source_sha256=content_digest(candidate),
                status="unvalidated",
                evaluation_level=candidate.scope.evaluation_level,
                agent_gender=candidate.scope.agent_gender,
                domains=candidate.scope.domains,
            )
        )
    return AuthoringCatalogListing(entries=entries)
