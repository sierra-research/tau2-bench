"""Shared typed authoring contracts and deterministic content validation."""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.backchannel import BackchannelLevel
from tau2.data_model.language import (
    DomainBinding,
    LanguagePack,
    NameOrder,
    PackPersona,
    PackVoice,
    PersonaVoice,
    TaskMetadata,
)
from tau2.data_model.tasks import Task
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.prompt_rules import validate_authoring_instruction
from tau2.multilingual.factory.regional_build import (
    LocalizationEvidence,
    LocalizationInput,
)
from tau2.multilingual.scenario_dispatch import RegionalBundle

AUTOFORM_CALL_NAME = "factory_autoform"

SUPPORTED_DOMAINS = Literal[
    "airline", "retail", "telecom", "banking_knowledge", "intake"
]

# Runtime speech inserts are synthesized audio tags; the period keeps the TTS
# provider from treating a tag-only insert as empty text.
_AUDIO_TAG_INSERT = re.compile(r"\.?(?:\[[a-z][a-z _-]*\])+")


class FormModel(BaseModel):
    """Strict authoring boundary; unexpected model fields are errors."""

    model_config = ConfigDict(extra="forbid")


class LanguageIdentity(FormModel):
    """Authoritative locale being onboarded."""

    language: Annotated[
        str,
        Field(pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]+)*$", description="Language code"),
    ]
    display_name: Annotated[
        str, Field(min_length=1, description="Reviewed language name")
    ]
    script: Annotated[
        str, Field(pattern=r"^[a-z]{4}$", description="Lowercase ISO15924 script code")
    ]
    name_order: Annotated[
        NameOrder,
        Field(
            default=NameOrder.GIVEN_FIRST,
            description="Reviewed name-component ordering; never inferred from language.",
        ),
    ]


class PersonaSketch(FormModel):
    """Language/register seed; task behavior and acoustic voice remain separate."""

    name: Annotated[
        str,
        Field(min_length=1, description="Persona label, not a task-identity override"),
    ]
    gender: Annotated[
        Literal["male", "female"], Field(description="Intended voice-routing gender")
    ]
    sketch: Annotated[
        str,
        Field(
            min_length=1,
            description="Text vocabulary, register and locale sketch; no acoustics, authentication procedure or task attitude",
        ),
    ]


class AuthorForm(FormModel):
    """Human-reviewable language-authoring input, before pack generation."""

    identity: Annotated[
        LanguageIdentity, Field(description="Authoritative target locale")
    ]
    locale_notes: Annotated[
        str,
        Field(
            description="Written locale and ordinary lexical value conventions; no acoustic coaching or authentication procedure"
        ),
    ]
    preferences: Annotated[
        str,
        Field(
            description="Text authoring preferences and register guidance; no acoustic coaching or authentication procedure"
        ),
    ]
    personas: Annotated[
        list[PersonaSketch],
        Field(min_length=2, max_length=2, description="One male and one female sketch"),
    ]

    @model_validator(mode="after")
    def validate_pair(self) -> "AuthorForm":
        """Require a usable pair before any downstream pack drafting."""
        if {p.gender for p in self.personas} != {"male", "female"}:
            raise ValueError("Author form requires one male and one female persona")
        if len({p.name for p in self.personas}) != 2:
            raise ValueError("Author form requires distinct persona labels")
        for field in ("locale_notes", "preferences"):
            validate_authoring_instruction(getattr(self, field), field)
        for persona in self.personas:
            validate_authoring_instruction(persona.sketch, f"{persona.name}.sketch")
        return self


class AutoformRequest(FormModel):
    """Typed request consumed by the public autoform stage."""

    identity: Annotated[
        LanguageIdentity, Field(description="Target locale to preserve exactly")
    ]
    generation: Annotated[
        FactoryModelConfig, Field(description="Pinned generation settings")
    ]
    hints: Annotated[
        str, Field(description="Author-provided locale and persona constraints")
    ]
    persona_names: Annotated[
        dict[Literal["male", "female"], str] | None,
        Field(
            description="Optional reviewed labels by gender that generation must preserve"
        ),
    ] = None

    @model_validator(mode="after")
    def validate_names(self) -> "AutoformRequest":
        """A reviewed name override covers both personas or neither."""
        if self.persona_names is not None and (
            set(self.persona_names) != {"male", "female"}
            or len(set(self.persona_names.values())) != 2
            or not all(name.strip() for name in self.persona_names.values())
        ):
            raise ValueError(
                "Reviewed persona names must cover both genders distinctly"
            )
        return self


class AuthorFormArtifact(FormModel):
    """Draft content and generation provenance persisted together for review."""

    stage: Annotated[
        Literal["author_form"], Field(description="Staged authoring output")
    ] = "author_form"
    form: Annotated[
        AuthorForm, Field(description="Validated draft awaiting ordinary file review")
    ]
    request: Annotated[
        AutoformRequest,
        Field(
            description="Complete original identity, hints, labels and generation settings"
        ),
    ]
    prompt_version: Annotated[
        str, Field(description="Version of the fixed authoring prompt")
    ]
    call_name: Annotated[
        Literal["factory_autoform"], Field(description="Observable LLM call identity")
    ] = AUTOFORM_CALL_NAME
    input_sha256: Annotated[
        str, Field(description="Digest of all request inputs for resume invalidation")
    ]


class DomainBundle(FormModel):
    """Reviewed prebuilt tasks with explicit runtime assignment and source lineage."""

    domain: Annotated[
        SUPPORTED_DOMAINS,
        Field(description="Explicit supported domain or reviewed domain bundle."),
    ]
    revision: Annotated[
        str,
        Field(
            min_length=1,
            description="New immutable artifact revision chosen by its author.",
        ),
    ]
    tasks: Annotated[
        list[Task],
        Field(
            min_length=1,
            description="Ordinary prebuilt domain tasks in canonical selection order.",
        ),
    ]
    metadata: Annotated[
        dict[str, TaskMetadata],
        Field(
            description="Task ID to canonical identity and persona routing metadata."
        ),
    ]
    reviewed_by: Annotated[
        str,
        Field(
            min_length=1,
            description="Attribution for the human who reviewed this exact content.",
        ),
    ]
    user_guidelines: Annotated[
        str, Field(description="Reviewed shared domain or language instructions.")
    ] = ""
    voice_guidelines: Annotated[
        str, Field(description="Reviewed text content for voice-mode localization.")
    ] = ""
    text_guidelines: Annotated[
        str, Field(description="Reviewed typed interaction localization.")
    ] = ""
    regional_bundle: Annotated[
        RegionalBundle | None,
        Field(
            description="Previously published regional scenario, retaining its exact source and content hashes."
        ),
    ] = None

    localization: LocalizationInput | None = Field(
        default=None,
        description="Factory input: reuse or generate localized entities before content review.",
    )
    localization_evidence: LocalizationEvidence | None = Field(
        default=None,
        description="Authoring-only compiled input and review findings; never published.",
    )

    @model_validator(mode="after")
    def consistent_tasks(self):
        """Fail before authoring if task IDs, canonical IDs or domains disagree."""
        if self.localization is not None and (
            self.regional_bundle is not None or self.localization_evidence is not None
        ):
            raise ValueError(
                "choose localization inputs or an already built regional bundle"
            )
        if self.localization_evidence is not None and self.regional_bundle is None:
            raise ValueError(
                "localization evidence requires its compiled regional bundle"
            )
        ids = [task.id for task in self.tasks]
        if len(set(ids)) != len(ids) or set(ids) != set(self.metadata):
            raise ValueError("metadata must cover exactly unique task IDs")
        canonical = [entry.canonical_task_id for entry in self.metadata.values()]
        if len(set(canonical)) != len(canonical):
            raise ValueError("canonical task IDs must be unique")
        if self.regional_bundle is not None:
            if self.regional_bundle.domain != self.domain:
                raise ValueError("regional scenario domain differs from task bundle")
            if {
                key: value.canonical_task_id
                for key, value in self.regional_bundle.metadata.items()
            } != {key: value.canonical_task_id for key, value in self.metadata.items()}:
                raise ValueError(
                    "regional scenario must retain the canonical task cohort"
                )
        for task in self.tasks:
            domain = getattr(task.user_scenario.instructions, "domain", None)
            if domain is not None and domain != self.domain:
                raise ValueError("task scenario domain disagrees with bundle")
        return self

    def binding(self) -> DomainBinding:
        """Derive safe, stable runtime file paths from the supported domain."""
        regional = self.regional_bundle is not None or self.localization is not None
        return DomainBinding(
            revision=self.revision,
            tasks_file=None if regional else f"tasks/{self.domain}.json",
            task_metadata_file=f"tasks/{self.domain}.metadata.json",
            user_guidelines=self.user_guidelines,
            voice_guidelines=self.voice_guidelines,
            text_guidelines=self.text_guidelines,
            regional_bundle_file=f"regional/{self.domain}.json" if regional else None,
        )


class DraftPersonaVoice(PersonaVoice):
    """Reviewed speech content with an explicitly unprovisioned voice in staging."""

    voice_id: Annotated[
        None,
        Field(
            description="Provider voice identity; null only in unprovisioned drafts."
        ),
    ] = None


class DraftPersona(PackPersona):
    """Runtime persona content; factory packs always carry speech content."""

    voice: Annotated[
        DraftPersonaVoice,
        Field(description="Speech behavior with voice provisioning deferred."),
    ]


class DraftPack(LanguagePack):
    """Same runtime schema, with only voice provisioning deferred."""

    personas: Annotated[
        dict[str, DraftPersona],
        Field(
            min_length=2,
            max_length=2,
            description="Exactly two reviewed runtime-compatible personas keyed by label.",
        ),
    ]
    voice: Annotated[
        PackVoice, Field(description="Pack speech settings; factory packs need them.")
    ]


class PackContent(FormModel):
    """LLM-owned content only; generation cannot choose task paths or voice IDs."""

    agent_greeting: Annotated[
        str,
        Field(
            min_length=1,
            pattern=r"\S",
            description="Required reviewed localized mode-neutral initial agent message.",
        ),
    ]
    agent_language_clause: Annotated[
        str, Field(description="Reviewed agent language and register instruction.")
    ]
    user_guidelines: Annotated[
        str, Field(description="Reviewed shared domain or language instructions.")
    ] = ""
    spellout_section: Annotated[
        str,
        Field(
            description="Fixed requested-spelling gates followed by reviewed native examples."
        ),
    ]
    personas: Annotated[
        dict[str, DraftPersona],
        Field(
            min_length=2,
            max_length=2,
            description="Exactly two reviewed runtime-compatible personas keyed by label.",
        ),
    ]


class DraftRequest(FormModel):
    """Reviewed form and explicit domain inputs used by a fixed drafting prompt."""

    form: Annotated[
        AuthorForm,
        Field(description="Reviewed language identity and persona authoring brief."),
    ]
    revision: Annotated[
        str,
        Field(
            min_length=1,
            description="New immutable artifact revision chosen by its author.",
        ),
    ]
    generation: Annotated[
        FactoryModelConfig,
        Field(description="Explicit generation model and reasoning settings."),
    ]
    speech_insert_events_per_minute: Annotated[
        float,
        Field(
            ge=0, description="Explicit shared insert frequency for this language pack."
        ),
    ]
    backchannel_level: Annotated[
        BackchannelLevel,
        Field(description="Explicit backchannel density for this language pack."),
    ]
    domains: Annotated[
        list[DomainBundle],
        Field(
            min_length=1,
            description="Explicit reviewed supported-domain bindings or input bundles.",
        ),
    ]


class DraftArtifact(FormModel):
    """A complete review candidate; never loaded by runtime."""

    stage: Annotated[
        Literal["draft"],
        Field(description="Authoring stage discriminator; never a runtime pack."),
    ] = "draft"
    pack: Annotated[
        DraftPack,
        Field(
            description="Draft content using the runtime-owned language pack contract."
        ),
    ]
    domains: Annotated[
        dict[SUPPORTED_DOMAINS, DomainBundle],
        Field(
            description="Explicit reviewed supported-domain bindings or input bundles."
        ),
    ]
    generation: Annotated[
        FactoryModelConfig | None,
        Field(description="Explicit generation model and reasoning settings."),
    ]
    prompt_version: Annotated[
        str, Field(description="Version of the fixed content-generation prompt.")
    ]
    input_sha256: Annotated[
        str,
        Field(
            pattern="^[0-9a-f]{64}$",
            description="Digest of all validated stage inputs for resume invalidation.",
        ),
    ]
    parent_sha256: Annotated[
        str | None,
        Field(description="Digest of the prior staged artifact for derived revisions."),
    ] = None

    @model_validator(mode="after")
    def validate_bundle(self):
        """Keep staging and published bindings identical and check persona routing."""
        if set(self.pack.domains) != set(self.domains):
            raise ValueError("pack and task domains disagree")
        if {p.gender for p in self.pack.personas.values()} != {"male", "female"}:
            raise ValueError("pack requires one male and one female persona")
        for domain, bundle in self.domains.items():
            if bundle.localization is not None:
                raise ValueError(
                    "draft requires compiled localization before content review"
                )
            if bundle.domain != domain or bundle.binding() != self.pack.domains[domain]:
                raise ValueError("domain binding differs from reviewed task bundle")
            if bundle.regional_bundle is not None and (
                bundle.regional_bundle.settings.language,
                bundle.regional_bundle.settings.name_order,
            ) != (self.pack.language, self.pack.name_order):
                raise ValueError(
                    "regional scenario language/name order differs from staged pack"
                )
            metadata_sets = [bundle.metadata]
            if bundle.regional_bundle is not None:
                metadata_sets.append(bundle.regional_bundle.metadata)
            for metadata in (
                row for values in metadata_sets for row in values.values()
            ):
                if metadata.persona_id is not None:
                    p = self.pack.personas.get(metadata.persona_id)
                    if p is None or (
                        metadata.caller_gender and p.gender != metadata.caller_gender
                    ):
                        raise ValueError(
                            "explicit persona conflicts with caller metadata"
                        )
        validate_content(self.pack)
        return self


def validate_content(pack: DraftPack) -> None:
    """Enforce shared requested-spelling framing and reject known retired coaching.

    This is a structural guard, not a replacement for native-speaker review.
    Lexical style/register and ordinary conversational hesitations remain supported.
    """
    from tau2.multilingual.runtime import resolve_persona
    from tau2.user.guidelines import render_simulation_guidelines

    section = " ".join(pack.voice.spellout_section.split())
    if not (
        section.startswith("## Speaking Special Characters and Numbers")
        and re.search(r"[Ss]pell.*only (?:if|when).*explicitly asks.*spell", section)
        and re.search(r"[Rr]epeat.*(?:not.*spell|normally)", section)
        and re.search(
            r"anchors.*only.*(?:explicitly|specifically) asks for anchors", section
        )
    ):
        raise ValueError("spelling section must include separate requested-only gates")
    clause = pack.agent_language_clause.casefold()
    for name in pack.personas:
        if name.casefold() in clause:
            raise ValueError(
                "agent_language_clause is shared by every call; remove persona names"
            )
    for name, persona in pack.personas.items():
        for tic in persona.voice.vocal_tics:
            if not _AUDIO_TAG_INSERT.fullmatch(tic):
                raise ValueError(
                    f"{name}.voice.vocal_tics must be non-verbal audio tags such as "
                    f"'.[cough]', not words: {tic!r}"
                )
    # Select authored instruction fields explicitly. Literal greetings/inserts,
    # TTS normalization tables, task/evaluator data and voice asset briefs do not
    # contain new simulator instructions.
    for field in ("agent_language_clause", "user_guidelines"):
        validate_authoring_instruction(getattr(pack, field), f"pack.{field}")
    validate_authoring_instruction(
        pack.voice.spellout_section, "pack.voice.spellout_section"
    )
    for name, persona in pack.personas.items():
        for field in (
            "locale",
            "variety",
            "gender_guidance",
            "agent_background_clause",
        ):
            validate_authoring_instruction(getattr(persona, field), f"{name}.{field}")
        for clause in persona.pragmatics_clauses:
            validate_authoring_instruction(clause, f"{name}.pragmatics_clauses")
        for clause in persona.voice.pragmatics_clauses:
            validate_authoring_instruction(clause, f"{name}.voice.pragmatics_clauses")
        for field, example in persona.voice.voice_examples.model_dump().items():
            validate_authoring_instruction(
                example, f"{name}.voice.voice_examples.{field}"
            )
    for domain, binding in pack.domains.items():
        for field in ("user_guidelines", "voice_guidelines", "text_guidelines"):
            validate_authoring_instruction(getattr(binding, field), f"{domain}.{field}")
    for binding in pack.domains.values():
        for persona_id in pack.personas:
            for mode in ("voice", "text"):
                resolved = resolve_persona(pack, persona_id, mode=mode, binding=binding)
                for tools in (False, True):
                    render_simulation_guidelines(resolved, use_tools=tools)


class AddDomainRequest(FormModel):
    """Extend a staged language with separately reviewed domain data and guidance."""

    draft: Annotated[
        DraftArtifact,
        Field(description="Exact validated language draft to review or provision."),
    ]
    domain: Annotated[
        DomainBundle,
        Field(description="Explicit supported domain or reviewed domain bundle."),
    ]
    revision: Annotated[
        str,
        Field(
            min_length=1,
            description="New immutable artifact revision chosen by its author.",
        ),
    ]


class PublicationProvenance(FormModel):
    """Source and content hashes accompanying an immutable runtime bundle."""

    schema_version: Literal[3] = Field(default=3, description="Provenance schema.")
    draft_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Digest of the validated draft kept in the authoring workspace.",
        ),
    ]
    assets_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Digest of the auditioned asset artifact kept in the authoring workspace.",
        ),
    ]
    files: Annotated[
        dict[str, str],
        Field(description="Published relative filename to SHA256 digest."),
    ]


class StagePackRequest(FormModel):
    """Bring an existing published pack into a new authoring revision."""

    pack: Annotated[
        LanguagePack,
        Field(
            description="Existing published runtime pack, including provisioned voice IDs."
        ),
    ]
    domains: Annotated[
        list[DomainBundle],
        Field(description="Exact published domain tasks and metadata."),
    ]
    revision: Annotated[
        str,
        Field(
            min_length=1,
            description="New staging revision; published source stays unchanged.",
        ),
    ]
