"""Published language-pack contracts and the resolved per-call language context.

Every run resolves exactly one language pack (canonical English by default). The
pack is mode-neutral: text-only packs omit the ``voice`` sections, and voice runs
require them. Everything here is data; prompt scaffolds live in
``tau2.user.guidelines`` and pack loading in ``tau2.multilingual.runtime``.
"""

from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from tau2.backchannel import BackchannelLevel
from tau2.config import SCENARIO_LANGUAGE
from tau2.data_model.persona import InterruptTendency, PersonaConfig, Verbosity
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.voice import SpelledEntityTables
from tau2.utils.brevity import (
    BACKCHANNEL_INVENTORY,
    NON_DIRECTED_INVENTORY,
    phrase_inventory_problems,
)

LANGUAGE_PACK_SCHEMA_VERSION = 3
# Bump on any change to runtime framing, persona resolution or scaffold rendering.
LANGUAGE_PROMPT_VERSION = "language-runtime-v8"
# Deterministic persona choice for tasks with or without reviewed metadata.
PERSONA_ASSIGNMENT_VERSION = "canonical-id-sha256-v1"

Gender = Literal["male", "female"]
CallMode = Literal["text", "voice"]


def cross_lingual(language: str) -> bool:
    """Whether a call language differs from the authored scenario language.

    Scenarios, policies and simulator guidelines are authored in
    ``SCENARIO_LANGUAGE``; only a different call language needs translation
    guidance, provider language hints or a judge language addendum.
    """
    return language.split("-")[0] != SCENARIO_LANGUAGE


class NameOrder(str, Enum):
    """Reviewed display-name ordering; never inferred from language or spelling."""

    GIVEN_FIRST = "given_first"
    FAMILY_FIRST = "family_first"


class PackModel(BaseModel):
    """Reject unknown fields rather than silently dropping authored content."""

    model_config = ConfigDict(extra="forbid")


class TaskSource(str, Enum):
    """Where a domain binding's tasks come from."""

    FILE = "file"  # the published tasks file is the complete task population
    REGISTRY = "registry"  # ordinary registered task sets and splits


class CallerSource(str, Enum):
    """Who owns the simulated caller's persona, voice and guidelines."""

    PACK = "pack"
    DOMAIN = "domain"  # e.g. outbound elicitation owns its caller machinery


class DomainBinding(PackModel):
    """Published task data and localization for one supported domain."""

    revision: str = Field(
        min_length=1, description="Published domain artifact revision."
    )
    source: TaskSource = Field(
        default=TaskSource.FILE,
        description="File: the published tasks are the whole population. Registry: "
        "ordinary task sets and splits, with published tasks replacing same-ID tasks.",
    )
    tasks_file: Optional[str] = Field(
        default=None,
        description="Pack-relative JSON list of Tasks: the published tasks of a file "
        "binding without a regional bundle, or optional reviewed replacements (keyed "
        "by canonical ID) for a registry binding.",
    )
    task_metadata_file: Optional[str] = Field(
        default=None,
        description="Pack-relative task ID to metadata JSON for the published tasks.",
    )
    regional_bundle_file: Optional[str] = Field(
        default=None,
        description="Published native regional scenario bundle, relative to this pack.",
    )
    canonical_scenario: bool = Field(
        default=False,
        description="Pin and supply the canonical database/policy snapshot "
        "(airline, retail and telecom bindings without a regional scenario).",
    )
    caller: CallerSource = Field(
        default=CallerSource.PACK,
        description="Pack personas voice the caller unless the domain owns it.",
    )
    user_guidelines: str = Field(
        default="", description="Reviewed domain glossary and user guidance."
    )
    voice_guidelines: str = Field(
        default="", description="Compiled reviewed spoken localization for this domain."
    )
    text_guidelines: str = Field(
        default="", description="Compiled reviewed typed localization for this domain."
    )

    @field_validator("tasks_file", "task_metadata_file", "regional_bundle_file")
    @classmethod
    def relative_file(cls, value: Optional[str]) -> Optional[str]:
        """Published bindings cannot read outside their language bundle."""
        if value is None:
            return None
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("binding file must be a safe relative path")
        return value

    @model_validator(mode="after")
    def consistent_source(self) -> "DomainBinding":
        """Each binding reads its tasks from exactly one published source.

        File bindings publish either a task list or a regional bundle (whose
        tasks are the published source), always with task metadata. Registry
        bindings read the registry and never localize the database. They may add
        reviewed metadata alone (registry tasks unchanged) or reviewed same-ID
        replacement tasks with their metadata; replacements need metadata.
        """
        if self.source == TaskSource.FILE:
            if (self.tasks_file is None) == (self.regional_bundle_file is None):
                raise ValueError(
                    "binding requires exactly one of tasks_file or regional_bundle_file"
                )
            if self.task_metadata_file is None:
                raise ValueError("file bindings require published task metadata")
        else:
            if self.regional_bundle_file is not None:
                raise ValueError("registry bindings cannot carry a regional scenario")
            if self.tasks_file is not None and self.task_metadata_file is None:
                raise ValueError(
                    "registry replacement tasks_file requires task_metadata_file"
                )
        if self.canonical_scenario and self.regional_bundle_file is not None:
            raise ValueError("a binding pins either a regional or canonical scenario")
        return self


class TaskMetadata(PackModel):
    """Explicit correspondence to the canonical domain task."""

    canonical_task_id: str = Field(
        min_length=1,
        description="Canonical task ID; never inferred from filename suffixes.",
    )
    caller_gender: Optional[Gender] = Field(
        default=None, description="Reviewed caller gender, or unknown."
    )
    persona_id: Optional[str] = Field(
        default=None,
        description="Optional explicit persona assignment within this pack.",
    )


class VoiceExamples(PackModel):
    """Required reviewed utterances for the union of voice template slots."""

    disfluency_fillers: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed disfluency fillers text, including punctuation.",
    )
    restart_example: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed restart example text, including punctuation.",
    )
    filler_pause_example: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed filler pause example text, including punctuation.",
    )
    pause_marker_examples: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed pause marker examples text, including punctuation.",
    )
    self_interruption_example: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed self interruption example text, including punctuation.",
    )
    clarification_request: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed clarification request text, including punctuation.",
    )
    emotion_examples: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed emotion examples text, including punctuation.",
    )
    conversational_confirmations: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed conversational confirmations text, including punctuation.",
    )
    silence_checkins: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed silence checkins text, including punctuation.",
    )
    frustrated_endings: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed frustrated endings text, including punctuation.",
    )
    dont_know_responses: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed dont know responses text, including punctuation.",
    )
    make_agent_work_chain: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed make agent work chain text, including punctuation.",
    )
    one_at_a_time_example: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed one at a time example text, including punctuation.",
    )
    forget_details_example: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed forget details example text, including punctuation.",
    )
    vague_openers: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed vague openers text, including punctuation.",
    )
    unknown_info_phrases: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed unknown info phrases text, including punctuation.",
    )
    tool_acknowledgment: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed tool acknowledgment text, including punctuation.",
    )
    tool_result_report: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed tool result report text, including punctuation.",
    )
    tool_failure_report: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed tool failure report text, including punctuation.",
    )
    multiple_actions_pushback: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed multiple actions pushback text, including punctuation.",
    )
    gratitude_examples: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed gratitude examples text, including punctuation.",
    )
    unknown_info_tools_phrases: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed unknown info tools phrases text, including punctuation.",
    )


class LanguageDirective(PackModel):
    """Facts for the shared cross-lingual directive; the scaffold owns the wording."""

    origin: str = Field(
        min_length=1,
        pattern=r"\S",
        description="English description of where the caller is from, e.g. 'Madrid, Spain'.",
    )


class PersonaVoice(PackModel):
    """Speech-only persona behavior and a provisioned voice."""

    voice_id: str = Field(min_length=1, description="Provisioned ElevenLabs voice ID.")
    interrupt_tendency: Optional[InterruptTendency] = Field(
        default=None, description="Whether the caller barges in while the agent speaks."
    )
    pragmatics_clauses: list[str] = Field(
        default_factory=list,
        description="Reviewed behavior that only applies to spoken calls (phone "
        "openings and closings, listening idioms, overlap, interruption and spoken "
        "continuers); rendered after the mode-neutral clauses in voice runs only.",
    )
    backchannel_phrases: list[str] = Field(
        min_length=1,
        description="One or two PURE continuers ('mm-hmm' class), drawn at random.",
    )
    non_directed_phrases: list[str] = Field(
        default_factory=list, description="Short speech addressed away from the phone."
    )
    vocal_tics: list[str] = Field(
        default_factory=list, description="Reviewed in-turn fillers and vocal tics."
    )
    voice_examples: VoiceExamples = Field(
        description="Reviewed utterances for every voice scaffold slot."
    )
    spelled_entity_tables: Optional[SpelledEntityTables] = Field(
        default=None,
        description="Opt-in reviewed normalization for TTS-bound text only.",
    )


class PackPersona(PackModel):
    """Reviewed caller persona; the task owns identity, goals and affect."""

    gender: Gender = Field(description="Caller gender for reviewed task matching.")
    locale: str = Field(description="Human-readable caller background.")
    variety: str = Field(description="Human-readable language variety.")
    verbosity: Verbosity = Field(
        default=Verbosity.STANDARD, description="Caller response verbosity."
    )
    gender_guidance: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed caller gender and grammatical agreement instruction.",
    )
    pragmatics_clauses: list[str] = Field(
        default_factory=list,
        description="Reviewed mode-neutral language/register realization "
        "instructions, rendered in text and voice runs; spoken-call-only behavior "
        "belongs in voice.pragmatics_clauses.",
    )
    agent_background_clause: str = Field(
        default="",
        description="Reviewed regional context for the agent; empty text omits it.",
    )
    language_directive: Optional[LanguageDirective] = Field(
        default=None,
        description="Facts for the cross-lingual call directive; required exactly "
        "when the pack language differs from the scenario language.",
    )
    voice: Optional[PersonaVoice] = Field(
        default=None, description="Speech behavior; required by voice runs."
    )


class PackVoice(PackModel):
    """Pack-level speech settings; required by voice runs."""

    agent_greeting: Optional[str] = Field(
        default=None,
        min_length=1,
        pattern=r"\S",
        description="Spoken initial agent message; defaults to the neutral greeting.",
    )
    spellout_section: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed requested-spelling and requested-anchor guidance, including heading.",
    )
    speech_insert_events_per_minute: float = Field(
        ge=0,
        description="Combined silence-time insert rate; also drives in-turn tic draws.",
    )
    backchannel_level: BackchannelLevel = Field(
        description="Backchannel density knob rendered into the shared decision prompt."
    )


class LanguagePack(PackModel):
    """One published language and explicit bindings, without benchmark membership."""

    schema_version: Literal[3] = Field(
        default=LANGUAGE_PACK_SCHEMA_VERSION, description="Runtime schema version."
    )
    language: str = Field(
        pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]+)*$", description="Language identifier."
    )
    revision: str = Field(min_length=1, description="Published pack revision.")
    display_name: str = Field(min_length=1, description="Human-readable language name.")
    name_order: NameOrder = Field(
        default=NameOrder.GIVEN_FIRST,
        description="Reviewed name-component ordering used to clarify retail caller facts.",
    )
    agent_greeting: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Reviewed mode-neutral initial agent message (text and voice).",
    )
    agent_language_clause: str = Field(
        min_length=1, pattern=r"\S", description="Reviewed agent language instruction."
    )
    user_guidelines: str = Field(
        default="", description="Reviewed shared localization instructions."
    )
    personas: dict[str, PackPersona] = Field(
        min_length=1, description="Named reviewed caller personas."
    )
    domains: dict[str, DomainBinding] = Field(
        default_factory=dict, description="Explicitly supported domain bindings."
    )
    default_binding: Optional[DomainBinding] = Field(
        default=None,
        description="Registry binding for any registered domain without an explicit "
        "entry; absent means only the listed domains are supported.",
    )
    voice: Optional[PackVoice] = Field(
        default=None, description="Speech settings; absent for text-only packs."
    )

    @model_validator(mode="after")
    def consistent_modes(self) -> "LanguagePack":
        """Voice sections are all-or-nothing; phrase inventories stay bounded."""
        if not self.domains and self.default_binding is None:
            raise ValueError("a pack must bind at least one domain")
        if (
            self.default_binding is not None
            and self.default_binding.source != TaskSource.REGISTRY
        ):
            raise ValueError("the default binding must source registry tasks")
        problems = []
        for name, persona in self.personas.items():
            if (persona.voice is None) != (self.voice is None):
                problems.append(
                    f"persona {name!r} voice section must match the pack voice section"
                )
            if (persona.language_directive is not None) != cross_lingual(self.language):
                problems.append(
                    f"persona {name!r} needs a language directive exactly when the "
                    "call language differs from the scenario language"
                )
            if persona.voice is not None:
                for inventory in (BACKCHANNEL_INVENTORY, NON_DIRECTED_INVENTORY):
                    problems.extend(
                        phrase_inventory_problems(
                            f"persona {name!r}",
                            getattr(persona.voice, inventory.field),
                            inventory=inventory,
                            language=self.language,
                        )
                    )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def binding(self, domain: str) -> DomainBinding:
        """The explicit or default binding for one domain."""
        binding = self.domains.get(domain, self.default_binding)
        if binding is None:
            raise ValueError(
                f"language {self.language} does not support domain {domain}"
            )
        return binding


class PackExtension(PackModel):
    """Opt-in custom-run bindings pinned to an unchanged released pack."""

    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    language: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]+)*$")
    base_pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: str = Field(min_length=1)
    domains: dict[str, DomainBinding] = Field(min_length=1)


class RuntimePersona(PersonaConfig):
    """A pack persona resolved for one call mode, language and domain binding."""

    model_config = ConfigDict(extra="forbid")
    persona_id: str = Field(description="Pack persona key.")
    mode: CallMode = Field(description="Call mode this persona was resolved for.")
    gender: Gender = Field(description="Caller gender.")
    locale: str = Field(description="Human-readable caller background.")
    variety: str = Field(description="Human-readable language variety.")
    language: str = Field(description="Resolved call language.")
    display_language: str = Field(description="Resolved human-readable language.")
    gender_guidance: str = Field(description="Caller gender instruction.")
    pragmatics_clauses: list[str] = Field(
        default_factory=list,
        description="Mode-neutral language/register realization clauses; "
        "spoken-call-only clauses live on voice.pragmatics_clauses.",
    )
    user_guidelines: str = Field(
        default="", description="Shared, domain and scenario localization guidance."
    )
    mode_guidelines: str = Field(
        default="", description="Domain localization for this call mode."
    )
    agent_background_clause: str = Field(
        default="", description="Regional context supplied to the agent."
    )
    language_directive: Optional[LanguageDirective] = Field(
        default=None, description="Cross-lingual directive facts, if any."
    )
    spellout_section: str = Field(
        default="", description="Requested-spelling guidance (voice only)."
    )
    voice: Optional[PersonaVoice] = Field(
        default=None, description="Speech behavior (voice only)."
    )

    @model_validator(mode="after")
    def voice_matches_mode(self) -> "RuntimePersona":
        """Voice calls carry speech behavior; text calls carry none."""
        if (self.voice is not None) != (self.mode == "voice"):
            raise ValueError("resolved persona voice section must match its mode")
        return self

    def to_guidelines_text(self, mode: CallMode = "text") -> Optional[str]:
        """Persona behavior appended to the shared scaffold.

        Mode-neutral clauses render in every mode; the voice section's
        spoken-call-only clauses follow them in voice runs and never in text.
        """
        if mode != self.mode:
            raise ValueError(f"persona resolved for {self.mode}, rendered for {mode}")
        sections = [super().to_guidelines_text(mode) or "", self.gender_guidance]
        sections.extend(self.pragmatics_clauses)
        if self.voice is not None:
            sections.extend(self.voice.pragmatics_clauses)
        sections.extend([self.user_guidelines, self.mode_guidelines])
        return "\n\n".join(s for s in sections if s) or None


class LanguageArtifactPins(PackModel):
    """Content expected by every worker in an in-progress run."""

    pack_sha256: str = Field(description="Published pack digest.")
    tasks_sha256: str = Field(description="Resolved task population digest.")
    metadata_sha256: str = Field(description="Resolved task metadata digest.")
    scenario_mode: Literal["pack", "canonical", "regional"] = Field(
        default="pack", description="Selected database and scenario representation."
    )
    scenario_sha256: Optional[str] = Field(
        default=None, description="Digest of the complete DB, policy and task bundle."
    )
    localization_map_sha256: Optional[str] = Field(
        default=None, description="Digest of the exact approved authoring map."
    )
    localization_revision: Optional[str] = Field(
        default=None,
        description="Approved regional map revision; null for canonical scenarios.",
    )
    runtime_rules_version: Optional[str] = Field(
        default=None, description="Declared regional domain rules version."
    )
    runtime_rules_sha256: Optional[str] = Field(
        default=None,
        description="Version and implementation digest of domain regional rules.",
    )


class TaskSelection(PackModel):
    """The registry task selection a binding's population was drawn from."""

    task_set_name: Optional[str] = Field(
        default=None, description="Registered task set; null selects the domain's."
    )
    task_split_name: Optional[str] = Field(
        default=None, description="Registered split; null selects every task."
    )
    task_kwargs: dict[str, str] = Field(
        default_factory=dict, description="Domain-owned task loader options."
    )


class DomainAgentGuidance(BaseModel):
    """Recorded rendered agent instruction with an explicit revision."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, description="Fully rendered domain instruction.")
    version: str = Field(
        min_length=1, description="Version of the fixed domain prompt."
    )


class LanguageTaskInfo(PackModel):
    """Recorded resolution for one task: the call's complete language context."""

    language: str = Field(description="Resolved call language.")
    pack_extension: Optional[str] = Field(
        default=None, description="Explicit separate-run extension, if selected."
    )
    mode: CallMode = Field(description="Call mode the context was resolved for.")
    cross_lingual: bool = Field(
        description="Call language differs from the authored scenario language."
    )
    pack_revision: str = Field(description="Declared pack version.")
    pack_sha256: str = Field(description="Exact published pack bytes digest.")
    domain_revision: str = Field(description="Declared domain binding version.")
    tasks_sha256: str = Field(description="Digest of the resolved task population.")
    metadata_sha256: str = Field(description="Digest of the resolved task metadata.")
    scenario_mode: Literal["pack", "canonical", "regional"] = Field(
        default="pack", description="Effective database/scenario selection."
    )
    scenario_sha256: Optional[str] = Field(
        default=None, description="Exact selected scenario bundle digest."
    )
    localization_map_sha256: Optional[str] = Field(
        default=None, description="Exact approved authoring map digest."
    )
    localization_revision: Optional[str] = Field(
        default=None,
        description="Approved regional map revision; null for canonical scenarios.",
    )
    runtime_rules_version: Optional[str] = Field(
        default=None, description="Declared regional domain rules version."
    )
    runtime_rules_sha256: Optional[str] = Field(
        default=None, description="Regional domain rules implementation digest."
    )
    task_selection: TaskSelection = Field(
        default_factory=TaskSelection,
        description="Task selection the resolved population was drawn from.",
    )
    task_sha256: str = Field(description="Digest of the validated task payload.")
    canonical_task_id: str = Field(description="Canonical domain task ID.")
    agent_greeting: str = Field(
        min_length=1,
        pattern=r"\S",
        description="Initial agent message for this mode.",
    )
    agent_language_clause: str = Field(
        min_length=1, pattern=r"\S", description="Reviewed agent language instruction."
    )
    domain_agent_guidance: Optional[DomainAgentGuidance] = Field(
        default=None,
        description="Rendered domain-specific agent guidance and prompt revision.",
    )
    speech_insert_events_per_minute: Optional[float] = Field(
        default=None, ge=0, description="Resolved pack insert rate (voice only)."
    )
    backchannel_level: Optional[BackchannelLevel] = Field(
        default=None, description="Resolved backchannel density (voice only)."
    )
    backchannel_prompt_version: Optional[str] = Field(
        default=None, description="Backchannel decision template revision (voice only)."
    )
    caller_gender: Optional[Gender] = Field(description="Reviewed task caller gender.")
    persona: Optional[RuntimePersona] = Field(
        description="Resolved persona; null when the domain owns its caller."
    )
    spelling_normalization_version: Optional[str] = Field(
        default=None,
        description="TTS spelling transformation revision; null when not enabled.",
    )
    prompt_version: str = Field(
        default=LANGUAGE_PROMPT_VERSION, description="Runtime framing version."
    )
    guideline_template_version: str = Field(
        min_length=1,
        description="User-simulator scaffold revision (tau2.user.guidelines).",
    )
    assignment_version: str = Field(
        default=PERSONA_ASSIGNMENT_VERSION,
        description="Deterministic persona assignment algorithm.",
    )

    @model_validator(mode="after")
    def mode_consistent(self) -> "LanguageTaskInfo":
        """Speech settings exist exactly for voice calls."""
        voice = self.mode == "voice"
        if any(
            voice != (value is not None)
            for value in (
                self.speech_insert_events_per_minute,
                self.backchannel_level,
                self.backchannel_prompt_version,
            )
        ):
            raise ValueError("speech settings must be present exactly for voice calls")
        if self.persona is not None and self.persona.mode != self.mode:
            raise ValueError("persona mode differs from the call mode")
        return self

    @model_serializer(mode="wrap")
    def serialize_context(self, handler):
        """Do not change existing context snapshots when no domain prompt applies."""
        value = handler(self)
        if self.domain_agent_guidance is None:
            value.pop("domain_agent_guidance", None)
        if self.pack_extension is None:
            value.pop("pack_extension", None)
        return value

    @property
    def persona_id(self) -> Optional[str]:
        """Resolved pack persona key, if the pack voices the caller."""
        return self.persona.persona_id if self.persona is not None else None

    @property
    def provider_language(self) -> Optional[str]:
        """Language hint for providers and judges; None for scenario-language calls."""
        return self.language if self.cross_lingual else None

    @property
    def artifact_pins(self) -> LanguageArtifactPins:
        """The complete published content used for this call."""
        return LanguageArtifactPins.model_validate(
            self.model_dump(include=set(LanguageArtifactPins.model_fields))
        )


class CallContext(BaseModel):
    """Typed per-call context handed to an orchestrator at construction."""

    model_config = ConfigDict(extra="forbid")
    language: Annotated[
        LanguageTaskInfo, Field(description="Resolved language/task/persona context.")
    ]
    communicate_scoring: Annotated[
        CommunicateScoring,
        Field(description="Communicate-info scorer applied when evaluating the call."),
    ]
    agent_voice: Annotated[
        Optional[str], Field(default=None, description="Effective agent voice.")
    ]
    agent_voice_gender: Annotated[
        Optional[str],
        Field(default=None, description="Agent voice gender from the pinned catalog."),
    ]


class LegacyLanguageContext(BaseModel):
    """A per-call language record saved before pack schema v3 (read-only)."""

    model_config = ConfigDict(extra="allow")
    language: str = Field(description="Recorded call language.")
    canonical_task_id: str = Field(description="Recorded canonical task ID.")


RecordedLanguageContext = Annotated[
    Union[LanguageTaskInfo, LegacyLanguageContext],
    Field(union_mode="left_to_right"),
]
