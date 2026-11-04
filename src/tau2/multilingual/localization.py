"""Source-checked regional database localization and review contracts."""

import hashlib
import json
from decimal import Decimal
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_serializer,
    model_validator,
)

from tau2.data_model.language import NameOrder
from tau2.multilingual.address_format import AddressForms, AddressRevisionProvenance
from tau2.utils.artifacts import ReviewApproval

Domain = Literal["airline", "retail", "telecom", "banking_knowledge", "intake"]
MapStatus = Literal["draft", "published"]
LocalizationCategory = Literal[
    "identity_name",
    "address",
    "contact",
    "airport",
    "route",
    "product_name",
    "product_option",
    "plan",
    "device",
    "billing",
    "money",
    "prose",
    "policy",
    "date_time",
    "protocol",
    "tool_text",
    "other",
]


def json_content_digest(value: JsonValue) -> str:
    """Hash JSON content independently of whitespace and mapping insertion order."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class LocalizationModel(BaseModel):
    """Strict boundary for generated and checked regional data."""

    model_config = ConfigDict(extra="forbid")


class SourceDocument(LocalizationModel):
    """An exact source snapshot; source_path is relative to the configured data root."""

    id: str = Field(min_length=1, description="Stable document identifier.")
    kind: Literal[
        "database", "user_database", "tasks", "policy", "tool_text", "dependency"
    ] = Field(description="Domain-owned source kind.")
    source_path: str = Field(
        min_length=1, description="Source location for provenance; never executed."
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Canonical JSON content digest."
    )
    content: JsonValue = Field(description="Exact domain-owned data or text snapshot.")


class SourceOccurrence(LocalizationModel):
    """One exact occurrence of the source value in a document."""

    document_id: str = Field(min_length=1, description="SourceDocument identifier.")
    path: list[str | int] = Field(
        description="Explicit object keys/array positions; empty selects document root."
    )

    @field_validator("path")
    @classmethod
    def nonnegative_indexes(cls, value: list[str | int]) -> list[str | int]:
        if any(isinstance(part, int) and part < 0 for part in value):
            raise ValueError("localization path indexes must be nonnegative")
        return value


class SourceEntry(LocalizationModel):
    """A semantic value and all of its checked source occurrences."""

    id: str = Field(min_length=1, description="Stable entry identifier.")
    category: LocalizationCategory = Field(
        description="Domain-defined localization treatment."
    )
    source: JsonValue = Field(description="Complete source value; no substring edits.")
    occurrences: list[SourceOccurrence] = Field(
        min_length=1, description="Every exact occurrence represented by this entry."
    )
    context: str = Field(
        description="English domain and identity context for authoring/review."
    )


class LocalizationSource(LocalizationModel):
    """Source inventory spanning the full database and released task selection."""

    domain: Domain = Field(description="Domain owning the source models and semantics.")
    documents: list[SourceDocument] = Field(
        min_length=1,
        description="Exact input snapshots; their SHA-256 digests identify the source.",
    )
    entries: list[SourceEntry] = Field(
        min_length=1, description="Explicit localizable and preserved source coverage."
    )


class LocalizedForms(LocalizationModel):
    """Native form, corresponding romanization and an English review description."""

    native: JsonValue = Field(
        description="Value used by the localized runtime; same JSON shape as source."
    )
    romanized: JsonValue = Field(
        description="Latin-script form of the same value; not an English translation."
    )
    english: str = Field(
        min_length=1, description="English meaning or explanation for review."
    )

    address: AddressForms | None = Field(
        default=None,
        description="Explicit semantic components for a complete postal address, never inferred at runtime.",
    )

    @model_serializer(mode="wrap")
    def serialize_address(self, handler):
        """Non-address forms keep their existing serialized contract."""
        value = handler(self)
        if self.address is None:
            value.pop("address", None)
        return value


NameGender = Literal["male", "female", "unisex", "unknown"]
NameGenderBasis = Literal[
    "reviewed_metadata", "explicit_context", "name_convention", "unknown"
]


class NameGenderEvidence(LocalizationModel):
    """Source-specific name compatibility, never a persona assignment or human approval."""

    source_gender: NameGender = Field(
        description="Source metadata or explicitly qualified conventional name assessment."
    )
    target_gender: NameGender = Field(
        description="Localized name compatibility assessment, pending native review."
    )
    source_basis: NameGenderBasis = Field(
        description="Authority of source evidence; name conventions are model inference."
    )
    source_evidence: str = Field(
        min_length=1,
        description="Exact source evidence or qualified inference rationale.",
    )
    reason: str = Field(
        min_length=1, description="Compatibility or unresolved native-review concern."
    )
    forms_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Exact localized forms covered by this assessment.",
    )
    model: str = Field(
        min_length=1,
        description="Assessment model, or human-review for explicitly approved corrections.",
    )
    prompt_version: str = Field(
        min_length=1, description="Versioned assessment and repair prompt."
    )
    status: Literal["pending_native_review"] = "pending_native_review"

    def require(self, forms: LocalizedForms) -> None:
        """Reject changed names and known incompatible gender, without guessing unknowns."""
        from tau2.utils.artifacts import content_digest

        if not isinstance(forms.native, str) or not isinstance(forms.romanized, str):
            raise ValueError("Name gender evidence requires scalar name strings")
        if self.forms_sha256 != content_digest(forms):
            raise ValueError("Name gender evidence no longer matches localized forms")
        if self.source_gender in {"male", "female"} and self.target_gender not in {
            self.source_gender,
            "unisex",
            "unknown",
        }:
            raise ValueError("Localized name changes known source gender")


class NameGenderItemEvidence(LocalizationModel):
    """Independent gender evidence for a name stored within a mapped list."""

    index: int = Field(
        ge=0, description="Exact index of the independent name or decoy."
    )
    evidence: NameGenderEvidence = Field(
        description="Assessment of this source name, independently from other names in the list."
    )

    def require(self, forms: LocalizedForms) -> None:
        """Bind indexed evidence to the actual native/romanized mapped name."""
        if not isinstance(forms.native, list) or not isinstance(forms.romanized, list):
            raise ValueError("Indexed name gender evidence requires list forms")
        if self.index >= min(len(forms.native), len(forms.romanized)):
            raise ValueError("Indexed name gender evidence is out of range")
        self.evidence.require(
            LocalizedForms(
                native=forms.native[self.index],
                romanized=forms.romanized[self.index],
                english=forms.english,
            )
        )


class EntityReuseProvenance(LocalizationModel):
    """Exact previously approved source values; new domain context stays unapproved."""

    source_domain: Domain = Field(description="Domain whose reviewed entity is reused.")
    source_entry_ids: list[str] = Field(
        min_length=1,
        description="Stable IDs of the source values, including assembled components.",
    )
    source_artifact_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Digest of the prior compact localization map.",
    )
    approval: ReviewApproval = Field(
        description="Prior linguistic approval, not approval of this new mapping."
    )


class LocalizationEntry(SourceEntry):
    """A proposed, reviewable mapping for one source entry."""

    reuse: EntityReuseProvenance | None = Field(
        default=None,
        description="Prior approved candidate lineage; absent for newly authored candidates.",
    )

    name_gender: NameGenderEvidence | None = Field(
        default=None,
        description="Source-specific gender preservation evidence for a personal name.",
    )

    name_gender_items: list[NameGenderItemEvidence] = Field(
        default_factory=list,
        description="Independent source-gender evidence for exact name-list slots.",
    )

    @model_validator(mode="after")
    def validate_name_gender(self):
        """Keep gender assessment bound to the exact candidate name."""
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
        """Unreused entries retain their exact serialized contract."""
        value = handler(self)
        if not self.name_gender_items:
            value.pop("name_gender_items", None)
        if self.name_gender is None:
            value.pop("name_gender", None)
        if self.reuse is None:
            value.pop("reuse", None)
        return value

    forms: LocalizedForms = Field(
        description="Native/romanized pair and English explanation."
    )
    operation: Literal["adapt", "translate", "preserve"] = Field(
        description="Whether the regional value changes meaning, language, or neither."
    )
    notes: str = Field(
        description="English rationale, constraints, or review concerns."
    )


class RegionalSettings(LocalizationModel):
    """Explicit target-market settings used by every domain in a language."""

    language: str = Field(
        pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]+)*$",
        description="Published language identifier.",
    )
    country_code: str = Field(
        pattern=r"^[A-Z]{2}$", description="ISO country for the regional benchmark."
    )
    currency_code: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="Currency of localized prices and policy amounts.",
    )
    currency_scale: str = Field(
        description="Fixed positive decimal benchmark scale; not a current exchange-rate claim."
    )
    time_zone: str = Field(description="IANA timezone for the scenario clock.")
    name_order: NameOrder = Field(
        description="Explicit reviewed display-name ordering."
    )
    notes: list[str] = Field(
        default_factory=list,
        description="English regional constraints for generation and review.",
    )

    @field_validator("currency_scale")
    @classmethod
    def positive_scale(cls, value: str) -> str:
        number = Decimal(value)
        if not number.is_finite() or number <= 0:
            raise ValueError("currency scale must be finite and positive")
        return value

    @field_validator("time_zone")
    @classmethod
    def known_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value


class GenerationProvenance(LocalizationModel):
    """Generation settings and the digest of the full generation request."""

    model: str = Field(min_length=1, description="Authoring model.")
    model_args: dict[str, JsonValue] = Field(
        description="Explicit generation parameters."
    )
    prompt_version: str = Field(
        min_length=1, description="Version of the fixed regional localization prompt."
    )
    call_name: str = Field(
        min_length=1, description="Stable observable generation call name."
    )
    input_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Digest of source, settings, references and generation configuration.",
    )
    parent_map_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Exact prior complete map retained and selectively revised, when applicable.",
    )
    email_rules_version: Literal["regional-email-v1"] | None = Field(
        default=None, description="Deterministic email rules applied during generation."
    )
    user_id_rules_version: Literal["regional-user-ids-v1"] | None = Field(
        default=None,
        description="Deterministic customer-ID rules applied during generation.",
    )

    @model_serializer(mode="wrap")
    def serialize_generation(self, handler):
        """Keep existing generation digests stable when no email rules were recorded."""
        value = handler(self)
        if self.email_rules_version is None:
            value.pop("email_rules_version", None)
        if self.user_id_rules_version is None:
            value.pop("user_id_rules_version", None)
        return value


class SourceRefreshProvenance(LocalizationModel):
    """Offline source refresh; original generation provenance stays unchanged.

    ``regional-source-lines-v1`` records a line-number-only refresh against the
    previous full inventory. ``regional-source-evidence-v1`` records a rebinding
    in which only code-evidence documents changed; its previous digest is that of
    the previous compact source manifest.
    """

    version: Literal["regional-source-lines-v1", "regional-source-evidence-v1"] = Field(
        default="regional-source-evidence-v1", description="Refresh rule."
    )
    parent_artifact_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Digest of the refreshed map artifact."
    )
    previous_source_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Digest of the previous source inventory or manifest (see version).",
    )
    source_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Digest of the new full source inventory.",
    )


class EmailRevisionProvenance(LocalizationModel):
    """Deterministic email rewrite recorded separately from model generation."""

    version: Literal["regional-email-v1"] = "regional-email-v1"
    parent_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_email_count: int = Field(ge=0)
    changed_entry_count: int = Field(ge=0)


class UserIdAlias(LocalizationModel):
    """One canonical customer identifier and its name-derived regional form."""

    canonical: str = Field(min_length=1)
    localized: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*_[0-9]+$")


class UserIdRevisionProvenance(LocalizationModel):
    """Deterministic customer-ID rewrite preserving its canonical correspondence."""

    version: Literal["regional-user-ids-v1"] = "regional-user-ids-v1"
    parent_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    aliases: list[UserIdAlias]
    changed_entry_count: int = Field(ge=0)


class CallerNameRevisionProvenance(LocalizationModel):
    """Grounded caller-name declarations synchronized with the regional database."""

    version: Literal["regional-caller-names-v1"] = "regional-caller-names-v1"
    parent_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_entry_count: int = Field(ge=0)


class ProseEntityBinding(LocalizationModel):
    """An explicit source surface inside one prose entry, separate from whole text."""

    entry_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    forms: LocalizedForms
    kind: Literal["reuse", "generated", "protected", "reviewed"]
    reason: str = Field(min_length=1)
    source_entry_id: str | None = None
    provenance: GenerationProvenance | None = None
    review: ReviewApproval | None = None
    source_part: (
        Literal["address_city", "address_region", "address_city_region"] | None
    ) = Field(
        default=None,
        description="Exact component derived locally from a referenced address; absent for full-value reuse.",
    )

    @model_serializer(mode="wrap")
    def serialize_binding(self, handler):
        """Preserve prior binding digests when component reuse is absent."""
        value = handler(self)
        if self.review is None:
            value.pop("review", None)
        if self.source_part is None:
            value.pop("source_part", None)
        return value


class IntakeCallerSurface(LocalizationModel):
    """Authored initial time expression bound to one immutable hard task answer."""

    canonical_task_id: str = Field(
        pattern=r"^intake_times_hard_[0-9]{2}$",
        description="Exact frozen hard time task receiving this initial expression.",
    )
    canonical_value: str = Field(
        pattern=r"^(1[0-2]|[1-9]):[0-5][0-9] (AM|PM)$",
        description="Unchanged complete canonical time, including its day period.",
    )
    native_spoken: str = Field(
        min_length=1,
        description="Natural initial expression in the target script, omitting the day period.",
    )
    romanized_spoken: str = Field(
        min_length=1,
        description="Romanized view of the same initial expression for review.",
    )
    rationale: str = Field(
        min_length=1,
        description="Specific local interpretation challenge; not a claim of native approval.",
    )
    provenance: GenerationProvenance = Field(
        description="Original fixed-prompt generation lineage, retained after reviewed edits."
    )
    review: ReviewApproval | None = Field(
        default=None,
        description="Approval of the complete corrected surface, including its canonical anchor and generation lineage.",
    )

    @model_serializer(mode="wrap")
    def serialize_surface(self, handler):
        """Keep generation-only artifacts unchanged when no correction was reviewed."""
        value = handler(self)
        if self.review is None:
            value.pop("review", None)
        return value

    @model_validator(mode="after")
    def validate_review(self):
        """Reject edits to previously reviewed wording, anchors or lineage."""
        if self.review is not None:
            self.review.require(self.model_copy(update={"review": None}))
        return self

    @field_validator("native_spoken", "romanized_spoken", "rationale")
    @classmethod
    def nonblank(cls, value: str) -> str:
        """Reject empty speech and embedded private-annotation syntax."""
        if not value.strip():
            raise ValueError("Caller surface must not be blank")
        if "\n" in value or "the real time this refers to" in value:
            raise ValueError("Caller surface must not embed a private annotation")
        return value


class LocalizationMap(LocalizationModel):
    """Complete regional proposal awaiting content review; never fabricates approval."""

    schema_version: Literal["regional-localization-v1"] = "regional-localization-v1"
    revision: str = Field(
        min_length=1, description="New immutable regional data revision."
    )
    settings: RegionalSettings = Field(description="Target-market settings.")
    source: LocalizationSource = Field(
        description="Pinned full database and task inventory."
    )
    entries: list[LocalizationEntry] = Field(
        min_length=1, description="One proposed mapping per source entry."
    )
    provenance: GenerationProvenance = Field(description="Generation lineage.")
    source_refresh: SourceRefreshProvenance | None = Field(
        default=None,
        description="Explicit source-line refresh, separate from generation.",
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
        default_factory=list,
        description="Source-bound initial hard time expressions; complete tool truth remains unchanged.",
    )
    status: MapStatus = Field(
        default="draft",
        description=(
            "A map is a draft until it is released as the source of a published"
            " regional bundle; human review is recorded in the language's"
            " localization README, not in this field."
        ),
    )

    @model_validator(mode="after")
    def validate_intake_caller_surfaces(self):
        """Bind every override to the frozen task and its exact selected answer."""
        if not self.intake_caller_surfaces:
            return self
        if self.source.domain != "intake":
            raise ValueError("Caller surfaces are only supported for intake")
        if self.settings.language not in {"hi", "es", "pt", "zh", "ko"}:
            raise ValueError(
                "Caller surfaces require a supported localized time language; English overrides are unsupported"
            )
        ids = [surface.canonical_task_id for surface in self.intake_caller_surfaces]
        if len(set(ids)) != len(ids):
            raise ValueError("Caller surfaces require unique canonical task IDs")
        documents = {document.id: document for document in self.source.documents}
        if not {"intake_localization", "tasks"} <= documents.keys():
            raise ValueError(
                "Caller surfaces require the selected intake source documents"
            )
        selected = {
            row["canonical_task_id"]: row
            for row in documents["intake_localization"].content["stimuli"]
        }
        tasks = {task["id"]: task for task in documents["tasks"].content}
        import regex

        script = {"hi": "Devanagari", "ko": "Hangul", "zh": "Han"}.get(
            self.settings.language, "Latin"
        )
        for surface in self.intake_caller_surfaces:
            row = selected.get(surface.canonical_task_id)
            if row is None or surface.canonical_task_id not in tasks:
                raise ValueError(
                    "Caller surface is not bound to a selected source task"
                )
            if row["family"] != "times" or row["tier"] != "hard":
                raise ValueError("Caller surface requires a hard time task")
            gold = tasks[surface.canonical_task_id]["evaluation_criteria"]["actions"][
                0
            ]["arguments"]["fields"].get(row["field_name"])
            if (
                surface.canonical_value != row["written"]
                or surface.canonical_value != gold
            ):
                raise ValueError(
                    "Caller surface canonical value differs from frozen source truth"
                )
            if not regex.search(
                rf"\p{{Script={script}}}", surface.native_spoken
            ) or regex.search(
                rf"[\p{{L}}--\p{{Script={script}}}]", surface.native_spoken, regex.V1
            ):
                raise ValueError(
                    "Caller surface native script differs from target language"
                )
            if regex.search(
                r"[\p{L}--\p{Script=Latin}]", surface.romanized_spoken, regex.V1
            ):
                raise ValueError(
                    "Caller surface romanized script must use Latin letters"
                )
        return self

    @model_serializer(mode="wrap")
    def serialize_map(self, handler):
        """Keep the digest of an unrefreshed map unchanged."""
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
