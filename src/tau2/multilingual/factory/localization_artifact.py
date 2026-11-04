"""Compact regional drafts bound to locally verified canonical source inventories."""

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_serializer, model_validator

from tau2.multilingual.address_format import AddressRevisionProvenance
from tau2.multilingual.factory.localization_compile import validate_localization
from tau2.multilingual.factory.localization_inventory import inventory_localization
from tau2.multilingual.localization import (
    CallerNameRevisionProvenance,
    Domain,
    EmailRevisionProvenance,
    EntityReuseProvenance,
    GenerationProvenance,
    IntakeCallerSurface,
    LocalizationCategory,
    LocalizationEntry,
    LocalizationMap,
    LocalizationModel,
    LocalizationSource,
    LocalizedForms,
    MapStatus,
    NameGenderEvidence,
    NameGenderItemEvidence,
    ProseEntityBinding,
    RegionalSettings,
    SourceEntry,
    SourceRefreshProvenance,
    UserIdRevisionProvenance,
    json_content_digest,
)
from tau2.utils.artifacts import content_digest, write_artifact

AUTOMATIC_LOCALIZATION_VERSION = "regional-automatic-v1"


class SourceDescriptor(LocalizationModel):
    """Canonical document identity without its duplicated content snapshot."""

    id: str = Field(min_length=1, description="Stable source document identifier.")
    kind: Literal[
        "database", "user_database", "tasks", "policy", "tool_text", "dependency"
    ] = Field(description="Kind of canonical source document.")
    source_path: str = Field(min_length=1, description="Recorded source location.")
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Canonical JSON content digest."
    )


class SourceManifest(LocalizationModel):
    """Every input document required to reconstruct the exact source inventory."""

    domain: Domain = Field(description="Domain owning the canonical source.")
    documents: list[SourceDescriptor] = Field(
        min_length=1,
        description="Complete set of source documents, pinned by content digest.",
    )


class AuthoredLocalization(LocalizationModel):
    """Reviewable before/after values; source paths are rebuilt from the inventory."""

    id: str = Field(
        min_length=1, description="Stable inventoried source entry identifier."
    )
    category: LocalizationCategory = Field(
        description="Exact inventoried value category."
    )
    source: JsonValue = Field(
        description="Original scalar or object for before/after review."
    )
    reuse: EntityReuseProvenance | None = Field(
        default=None,
        description="Prior reviewed candidate lineage; new context requires review.",
    )

    name_gender: NameGenderEvidence | None = Field(
        default=None,
        description="Source-specific personal-name gender compatibility, bound to forms.",
    )

    name_gender_items: list[NameGenderItemEvidence] = Field(
        default_factory=list,
        description="Gender evidence for each independent mapped name-list slot.",
    )

    @model_validator(mode="after")
    def validate_name_gender(self):
        """Reject stale or incompatible personal-name assessments."""
        if self.name_gender is not None:
            self.name_gender.require(self.forms)
        if len({item.index for item in self.name_gender_items}) != len(
            self.name_gender_items
        ):
            raise ValueError("Duplicate name gender evidence indices")
        for item in self.name_gender_items:
            item.require(self.forms)
        return self

    @model_serializer(mode="wrap")
    def serialize_entry(self, handler):
        """Leave existing generated entry digests unchanged."""
        value = handler(self)
        if not self.name_gender_items:
            value.pop("name_gender_items", None)
        if self.name_gender is None:
            value.pop("name_gender", None)
        if self.reuse is None:
            value.pop("reuse", None)
        return value

    forms: LocalizedForms = Field(
        description="Authored native, romanized and English forms."
    )
    operation: Literal["adapt", "translate", "preserve"] = Field(
        description="Authored regional treatment."
    )
    notes: str = Field(description="English rationale and review limitations.")


class RegionalMapArtifact(LocalizationModel):
    """Compact draft with authored values and a verified canonical source manifest."""

    schema_version: Literal["regional-localization-map-v1"] = Field(
        default="regional-localization-map-v1", description="Compact map schema."
    )
    automatic_rules_version: Literal["regional-automatic-v1"] = Field(
        default=AUTOMATIC_LOCALIZATION_VERSION,
        description="Deterministic preservation and currency rules.",
    )
    revision: str = Field(min_length=1, description="Immutable regional data revision.")
    settings: RegionalSettings = Field(
        description="Recorded target country and language settings."
    )
    source: SourceManifest = Field(description="Exact canonical input documents.")
    entries: list[AuthoredLocalization] = Field(
        description="Exactly the nonautomatic source entries, including preserved authored values."
    )
    provenance: GenerationProvenance = Field(description="Original generation lineage.")
    source_refresh: SourceRefreshProvenance | None = Field(
        default=None, description="Explicit offline source-line refresh lineage."
    )
    email_revision: EmailRevisionProvenance | None = Field(
        default=None, description="Deterministic email rewrite lineage."
    )
    user_id_revision: UserIdRevisionProvenance | None = Field(
        default=None, description="Canonical-to-regional customer-ID correspondence."
    )
    caller_name_revision: CallerNameRevisionProvenance | None = Field(
        default=None, description="Grounded caller-name synchronization lineage."
    )
    address_revision: AddressRevisionProvenance | None = Field(
        default=None, description="Exact deterministic address repair lineage."
    )
    prose_bindings: list[ProseEntityBinding] = Field(
        default_factory=list,
        description="Explicit source-bound incidental entities and protected prose surfaces.",
    )
    intake_caller_surfaces: list[IntakeCallerSurface] = Field(
        default_factory=list, description="Source-bound initial hard time expressions."
    )
    status: MapStatus = Field(
        default="draft",
        description=(
            "A map is a draft until it is released as the source of a published"
            " regional bundle; human review is recorded in the language's"
            " localization README, not in this field."
        ),
    )

    @model_serializer(mode="wrap")
    def serialize_artifact(self, handler):
        """Preserve existing artifact digests when no source refresh occurred."""
        value = handler(self)
        if self.address_revision is None:
            value.pop("address_revision", None)
        if not self.prose_bindings:
            value.pop("prose_bindings", None)
        if not self.intake_caller_surfaces:
            value.pop("intake_caller_surfaces", None)
        if self.source_refresh is None:
            value.pop("source_refresh", None)
        if self.email_revision is None:
            value.pop("email_revision", None)
        if self.user_id_revision is None:
            value.pop("user_id_revision", None)
        if self.caller_name_revision is None:
            value.pop("caller_name_revision", None)
        return value


def automatic_localization(
    entry: SourceEntry, settings: RegionalSettings
) -> LocalizationEntry | None:
    """Apply fixed, versioned preservation and currency rules; never draft text."""
    if entry.category in {"protocol", "date_time", "tool_text"}:
        return LocalizationEntry(
            **entry.model_dump(),
            forms=LocalizedForms(
                native=entry.source,
                romanized=entry.source,
                english="Preserved scenario state, machine identifier or protocol value.",
            ),
            operation="preserve",
            notes="Source-code evidence: regional runtime wiring is required separately; this map does not rewrite executable code."
            if entry.category == "tool_text"
            else "Scenario clock values retain their ordering; the regional timezone must be applied consistently by the runtime."
            if entry.category == "date_time"
            else "Preserves the domain's tool and evaluation contract.",
        )
    if entry.category == "money" and type(entry.source) in {int, float}:
        scaled = Decimal(str(entry.source)) * Decimal(settings.currency_scale)
        if isinstance(entry.source, int):
            if scaled != scaled.to_integral_value():
                raise ValueError(
                    "currency scale would change an integer-valued monetary field's type"
                )
            value = int(scaled)
        else:
            value = float(scaled)
        return LocalizationEntry(
            **entry.model_dump(),
            forms=LocalizedForms(
                native=value,
                romanized=value,
                english=f"{value} {settings.currency_code}; source amount multiplied by fixed benchmark scale {settings.currency_scale}.",
            ),
            operation="adapt" if value != entry.source else "preserve",
            notes="Synthetic benchmark currency scaling, not a quoted market exchange rate. Related task/policy amounts must use the same scale.",
        )
    return None


def _manifest(source: LocalizationSource) -> SourceManifest:
    return SourceManifest(
        domain=source.domain,
        documents=[
            SourceDescriptor.model_validate(document.model_dump(exclude={"content"}))
            for document in source.documents
        ],
    )


def write_localization_map(proposal: LocalizationMap, path: Path) -> bool:
    """Write an immutable compact draft, refusing loss of altered automatic values."""
    proposal = validate_localization(proposal)
    entries = []
    for entry in proposal.entries:
        automatic = automatic_localization(
            SourceEntry.model_validate(
                entry.model_dump(include=set(SourceEntry.model_fields))
            ),
            proposal.settings,
        )
        automatic = _automatic_user_ids(automatic, proposal.user_id_revision)
        if automatic is not None:
            if automatic != entry:
                raise ValueError(
                    f"Automatic entry differs from recorded rules: {entry.id}"
                )
            continue
        entries.append(
            AuthoredLocalization.model_validate(
                entry.model_dump(include=set(AuthoredLocalization.model_fields))
            )
        )
    return write_artifact(
        RegionalMapArtifact(
            revision=proposal.revision,
            settings=proposal.settings,
            source=_manifest(proposal.source),
            entries=entries,
            provenance=proposal.provenance,
            source_refresh=proposal.source_refresh,
            email_revision=proposal.email_revision,
            user_id_revision=proposal.user_id_revision,
            caller_name_revision=proposal.caller_name_revision,
            address_revision=proposal.address_revision,
            prose_bindings=proposal.prose_bindings,
            intake_caller_surfaces=proposal.intake_caller_surfaces,
            status=proposal.status,
        ),
        path,
    )


def read_localization_map(path: Path, data_root: Path | None = None) -> LocalizationMap:
    """Rebuild a complete draft only against the exact configured source documents."""
    if data_root is None:
        from tau2.utils import DATA_DIR

        data_root = DATA_DIR
    artifact = RegionalMapArtifact.model_validate_json(path.read_text())
    source = inventory_localization(artifact.source.domain, data_root)
    return restore_localization_map(artifact, source)


def restore_localization_map(
    artifact: RegionalMapArtifact,
    source: LocalizationSource,
    *,
    validate_candidate: bool = True,
) -> LocalizationMap:
    """Restore exact source coverage; offline repairs may defer candidate validation."""
    if _manifest(source) != artifact.source:
        raise ValueError(
            "Localization source manifest differs from canonical source documents"
        )
    if (
        artifact.source_refresh is not None
        and artifact.source_refresh.source_sha256 != content_digest(source)
    ):
        raise ValueError(
            "Source refresh provenance differs from its recorded inventory"
        )
    authored = {entry.id: entry for entry in artifact.entries}
    if len(authored) != len(artifact.entries):
        raise ValueError("Duplicate authored localization entry IDs")
    entries = []
    expected_authored = set()
    for entry in source.entries:
        automatic = automatic_localization(entry, artifact.settings)
        automatic = _automatic_user_ids(automatic, artifact.user_id_revision)
        if automatic is not None:
            entries.append(automatic)
            continue
        expected_authored.add(entry.id)
        candidate = authored.get(entry.id)
        if candidate is None:
            raise ValueError(f"Missing authored localization entry: {entry.id}")
        if candidate.category != entry.category or json_content_digest(
            candidate.source
        ) != json_content_digest(entry.source):
            raise ValueError(f"Stale authored localization source entry: {entry.id}")
        entries.append(
            LocalizationEntry(
                **entry.model_dump(),
                forms=candidate.forms,
                reuse=candidate.reuse,
                name_gender=candidate.name_gender,
                name_gender_items=candidate.name_gender_items,
                operation=candidate.operation,
                notes=candidate.notes,
            )
        )
    if authored.keys() != expected_authored:
        raise ValueError(
            "Authored entries must cover exactly all nonautomatic source entries"
        )
    candidate = LocalizationMap(
        revision=artifact.revision,
        settings=artifact.settings,
        source=source,
        entries=entries,
        provenance=artifact.provenance,
        source_refresh=artifact.source_refresh,
        email_revision=artifact.email_revision,
        user_id_revision=artifact.user_id_revision,
        caller_name_revision=artifact.caller_name_revision,
        address_revision=artifact.address_revision,
        prose_bindings=artifact.prose_bindings,
        intake_caller_surfaces=artifact.intake_caller_surfaces,
        status=artifact.status,
    )
    return validate_localization(candidate) if validate_candidate else candidate


def _automatic_user_ids(
    entry: LocalizationEntry | None, revision: UserIdRevisionProvenance | None
) -> LocalizationEntry | None:
    if entry is None or revision is None:
        return entry
    from tau2.multilingual.factory.localization_user_ids import localize_user_id_entry

    return localize_user_id_entry(entry, revision.aliases)


def refresh_localization_source(
    artifact: RegionalMapArtifact, source: LocalizationSource, revision: str
) -> LocalizationMap:
    """Rebind a compact map to a newer checkout in which only code evidence moved.

    Tool-text documents are preserve-only evidence; the regional runtime rules are
    fingerprinted separately. Database, task and policy documents, and therefore
    every authored entry, must be identical. No model is called and no authored
    value changes; the result is revalidated against the complete new inventory.
    """
    artifact = RegionalMapArtifact.model_validate(artifact.model_dump())
    source = LocalizationSource.model_validate(source.model_dump())
    if not revision or revision == artifact.revision:
        raise ValueError("Source refresh requires a new data revision")
    if source.domain != artifact.source.domain:
        raise ValueError("Source refresh requires the same domain")
    manifest = _manifest(source)
    if content_digest(manifest) == content_digest(artifact.source):
        raise ValueError("Source refresh requires a changed source")
    before = {d.id: d for d in artifact.source.documents}
    after = {d.id: d for d in manifest.documents}
    if before.keys() != after.keys():
        raise ValueError("Source refresh must retain every source document")
    for key, old in before.items():
        new = after[key]
        if (old.kind, old.source_path) != (new.kind, new.source_path):
            raise ValueError("Source refresh changed a document's identity")
        if old.kind != "tool_text" and old != new:
            raise ValueError(
                "Source refresh cannot change database, tasks or policy documents"
            )
    refreshed = artifact.model_copy(
        update={
            "revision": revision,
            "source": manifest,
            "source_refresh": SourceRefreshProvenance(
                version="regional-source-evidence-v1",
                parent_artifact_sha256=content_digest(artifact),
                previous_source_sha256=content_digest(artifact.source),
                source_sha256=content_digest(source),
            ),
        }
    )
    return restore_localization_map(refreshed, source)
