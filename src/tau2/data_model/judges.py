"""Versioned judge contracts; no annotation or replay dependencies."""

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    model_validator,
)


class Measure(str, Enum):
    """Only measures with retained passing human-validation evidence."""

    NATURALNESS = "naturalness"
    TOOL_USE = "tool_use"
    SPEECH_FIDELITY = "speech_fidelity"
    COUNTING_UNITS = "counting_units"
    REGISTER_FORMALITY = "register_formality"
    GENDER_AGREEMENT = "gender_agreement"
    NAME_ADDRESS = "name_address_conventions"
    HONORIFIC_AGREEMENT = "honorific_agreement"
    MODAL_PARTICLES = "modal_particles"


DiagnosticId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,79}$")]


class ImplementationFamily(str, Enum):
    NATIVENESS = "nativeness-v15"
    NATURALNESS = "naturalness-v17"
    HINDI_GENDER = "hi-gender-v16"
    TOOL_USE = "tool-use-v7"
    FIDELITY = "speech-fidelity-v6"
    AGENT_TEXT = "agent-text-v1"


class InputExclusions(str, Enum):
    """Versioned benchmark-authored agent speech removed before judging.

    See ``tau2.judges.exclusions`` for the exact rules. The rule is part of the
    verdict identity, so changing it invalidates cached verdicts.
    """

    NONE = "none"
    OPENER = "injected-opener-v1"
    OPENER_AND_TRANSFER = "injected-opener-and-transfer-line-v1"
    OPENER_AND_CONTROL_TURNS = "injected-opener-and-control-turns-v1"
    OPENER_TRANSFER_AND_CONTROL_TURNS = (
        "injected-opener-transfer-line-and-control-turns-v1"
    )


_COMMIT_SHA = re.compile(r"^[0-9a-f]{7,64}$")


def _neutral_label(value: str) -> str:
    """Provenance labels must not point at unpublished source commits."""
    if _COMMIT_SHA.fullmatch(value):
        raise ValueError(
            "Use a neutral package label, not a source commit, for provenance"
        )
    return value


NeutralLabel = Annotated[str, AfterValidator(_neutral_label)]


class ContractModel(BaseModel):
    """Portable persisted boundaries reject undocumented fields."""

    model_config = ConfigDict(extra="forbid")


class Prompt(ContractModel):
    """Exact archived prompt text; its version is the cache identity."""

    prompt_version: Annotated[str, Field(description="Frozen prompt version.")]
    rubric_version: Annotated[str, Field(description="Frozen rubric version.")]
    system_prompt: Annotated[
        str, Field(description="Unchanged calibrated system prompt.")
    ]
    user_prompt_template: Annotated[
        str, Field(description="Unchanged calibrated user template.")
    ]
    factor_instructions: Annotated[
        dict[str, str], Field(description="Original per-factor context instructions.")
    ]


class NativenessCriterion(ContractModel):
    """Exact reviewed language-specific criterion text."""

    factor_id: Annotated[str, Field(description="Original output factor identifier.")]
    question: Annotated[str, Field(description="Calibrated question.")]
    opportunity: Annotated[str, Field(description="Applicable opportunity definition.")]
    positive_examples: Annotated[
        str, Field(description="Natural target-language examples.")
    ]
    negative_examples: Annotated[
        str, Field(description="Violation examples and exclusions.")
    ]


class ToolCriterion(ContractModel):
    """Original tool-use subtype; validation applies to their combined measure."""

    criterion_id: Annotated[str, Field(description="Original tool criterion ID.")]
    question: Annotated[str, Field(description="Calibrated question.")]
    decision_rule: Annotated[str, Field(description="Original pass/fail rule.")]


class SpeechMaterial(ContractModel):
    """Language guidance and requested reply shape for speech fidelity."""

    output_shape: Annotated[str, Field(description="Exact requested reply shape.")]
    language_guidance: Annotated[
        dict[str, str],
        Field(description="Fidelity guidance for each supported language."),
    ]


class JudgeContract(ContractModel):
    """One exact model/prompt/rubric contract, not an entire donor judge suite."""

    contract_id: Annotated[str, Field(description="Stable frozen contract identity.")]
    implementation: Annotated[
        ImplementationFamily,
        Field(
            description="Explicit versioned input preparation and judging implementation."
        ),
    ]
    evaluation_level: Annotated[
        Literal["call", "utterance"],
        Field(
            default="utterance",
            description="Unit used by the reusable agent-text implementation.",
        ),
    ]
    input_exclusions: Annotated[
        InputExclusions,
        Field(description="Benchmark-authored agent speech removed before judging."),
    ]
    source_revision: Annotated[
        NeutralLabel,
        Field(
            description="Neutral label of the frozen source package, never a commit."
        ),
    ]
    model: Annotated[
        str, Field(description="Judge model; independent of customer model.")
    ]
    model_args: Annotated[
        dict[str, JsonValue], Field(description="Exact model arguments.")
    ]
    prompt: Annotated[
        Prompt, Field(description="Exact prompt text from the evidence authority.")
    ]
    nativeness_criteria: Annotated[
        dict[str, list[NativenessCriterion]],
        Field(
            description="Original batched criteria by language; only allowed measures are exposed."
        ),
    ]
    quality_criteria: Annotated[
        list[ToolCriterion],
        Field(description="Combined tool-use measure's original criterion batch."),
    ]
    scope_note: Annotated[
        str | None, Field(description="Frozen validation applicability caveat.")
    ]
    speech_material: Annotated[
        SpeechMaterial | None,
        Field(default=None, description="Frozen speech-fidelity assembly content."),
    ]


class ValidatedPair(ContractModel):
    """Selected language/measure and participant frame; status lives on the catalog."""

    language: Annotated[str, Field(description="Selected language identifier.")]
    measure: Annotated[
        DiagnosticId,
        Field(
            description="Explicit diagnostic identifier, including reviewed new diagnostics."
        ),
    ]
    contract_id: Annotated[str, Field(description="Frozen judge contract to execute.")]
    evaluation_level: Annotated[
        Literal["call", "utterance"], Field(description="Selected statistical unit.")
    ]
    agent_gender: Annotated[
        Literal["male", "female"] | None,
        Field(
            default=None,
            description="Restricted participant frame when selection specifies one.",
        ),
    ]
    domains: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Selected domain restrictions; None means the historical unrestricted domain frame.",
        ),
    ]
    needs_audio: Annotated[
        bool,
        Field(
            default=False,
            description="Requires delivered audio and reference alignment.",
        ),
    ]


class JudgeCatalog(ContractModel):
    """Compact production selection with explicit validation exceptions."""

    revision: Annotated[
        str,
        Field(description="Selection version, bumped when supported contracts change."),
    ]
    donor_revision: Annotated[
        NeutralLabel,
        Field(
            description="Neutral label of the frozen validation-evidence package, never a commit."
        ),
    ]
    source_prompt_sha256: Annotated[
        str, Field(description="SHA256 of the immutable prompt inventory input.")
    ]
    contracts: Annotated[
        list[JudgeContract],
        Field(description="Exact retained model/prompt/rubric contracts."),
    ]
    pairs: Annotated[
        list[ValidatedPair], Field(description="Selected language/measure pairs.")
    ]
    unvalidated_contract_ids: list[str] = Field(
        default_factory=list,
        description="Selected reviewed contracts without held-out validation evidence.",
    )

    @model_validator(mode="after")
    def cross_references(self):
        """Reject ambiguous contracts or duplicate applicability entries."""
        ids = [contract.contract_id for contract in self.contracts]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate judge contract IDs")
        keys = [(pair.language, pair.measure) for pair in self.pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate selected language/measure pairs")
        if any(pair.contract_id not in ids for pair in self.pairs):
            raise ValueError("Validated pair references missing contract")
        if len(self.unvalidated_contract_ids) != len(
            set(self.unvalidated_contract_ids)
        ):
            raise ValueError("Duplicate unvalidated contract IDs")
        if any(contract_id not in ids for contract_id in self.unvalidated_contract_ids):
            raise ValueError("Unvalidated contract ID references missing contract")
        return self


class JudgeInput(ContractModel):
    """Stable simulation identity plus fingerprint of authoritative judge inputs."""

    domain: Annotated[
        str | None,
        Field(
            default=None, description="Recorded domain for applicability restrictions."
        ),
    ]
    run_id: Annotated[str, Field(description="Ordinary saved run identifier.")]
    simulation_id: Annotated[
        str, Field(description="Stored attempt selected for evaluation.")
    ]
    trial: Annotated[
        int, Field(ge=0, description="Recorded trial; never inferred from file order.")
    ]
    language: Annotated[str, Field(description="Recorded effective language.")]
    agent_gender: Annotated[
        Literal["male", "female", "unknown"],
        Field(
            description="Recorded voice-catalog gender, separate from caller gender."
        ),
    ]
    input_fingerprint: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Digest of exact transcript/tool/task/participant/audio inputs consumed by the adapter.",
        ),
    ]
    has_audio: Annotated[
        bool,
        Field(
            description="Required delivered audio and aligned reference are available."
        ),
    ]
    caller_gender: Annotated[
        Literal["male", "female", "unknown"],
        Field(
            default="unknown",
            description="Explicit recorded caller gender, never inferred.",
        ),
    ]
    audio_fingerprint: Annotated[
        str | None,
        Field(
            default=None,
            description="Delivered audio bytes digest, independent of transcript.",
        ),
    ]
    audio_error: Annotated[
        str | None,
        Field(
            default=None,
            description="Local audio/timeline validation error, if present.",
        ),
    ]


class VerdictIdentity(ContractModel):
    """Everything needed to decide whether an existing verdict is reusable."""

    input: Annotated[
        JudgeInput, Field(description="Frozen input identity and availability.")
    ]
    measure: Annotated[DiagnosticId, Field(description="Reported selected measure.")]
    contract_id: Annotated[str, Field(description="Selected judge contract.")]
    # Recorded strings rather than enums: a verdict from a retired version of
    # this schema still loads and plans as STALE.
    implementation: Annotated[
        str,
        Field(description="Recorded versioned executable method (may be retired)."),
    ]
    input_exclusions: Annotated[
        str, Field(description="Recorded benchmark-speech exclusion rule.")
    ]
    model: Annotated[str, Field(description="Exact judge model.")]
    model_args: Annotated[
        dict[str, JsonValue], Field(description="Exact judge arguments.")
    ]
    prompt_version: Annotated[str, Field(description="Calibrated prompt version.")]
    rubric_version: Annotated[str, Field(description="Calibrated rubric version.")]
    transport_revision: str = Field(
        default="legacy", description="LLM request serialization contract."
    )
    catalog_revision: Annotated[
        str, Field(description="Applicability/selection contract revision.")
    ]


class UnitVerdict(ContractModel):
    """Normalized diagnostic unit; adapters preserve the calibrated unit definition."""

    unit_id: Annotated[
        str,
        Field(
            min_length=1, description="Stable utterance/audio-window ID or whole call."
        ),
    ]
    opportunity: Annotated[
        StrictBool, Field(description="Measure has a judged opportunity in this unit.")
    ]
    violation: Annotated[
        StrictBool, Field(description="True means an observed diagnostic violation.")
    ]
    evidence: Annotated[
        str, Field(description="Recorded explanation from the adapter/judge.")
    ]

    @model_validator(mode="after")
    def valid_opportunity(self):
        """A violation needs a corresponding opportunity."""
        if self.violation and not self.opportunity:
            raise ValueError("A violation without an opportunity is malformed")
        return self


class UnitError(ContractModel):
    """An unjudged unit, never included in pass or opportunity denominators."""

    unit_id: str = Field(min_length=1)
    kind: Literal["input", "response"]
    error: str
    response_text: str | None = None
    finish_reason: str | None = None
    usage: dict[str, JsonValue] | None = None


def is_fidelity_output_limit(measure: str, error: UnitError) -> bool:
    """A speech-fidelity clip whose judge reply still hit its output limit.

    The fidelity judge runs at temperature 0, so retrying such a clip repeats
    the same runaway reply. The clip is left out of scoring instead of blocking
    the call; every other response error still blocks.
    """
    return (
        measure == Measure.SPEECH_FIDELITY
        and error.kind == "response"
        and error.finish_reason == "length"
    )


class MeasureVerdict(ContractModel):
    """Validated adapter output; an empty response is an error, never a pass."""

    units: Annotated[
        list[UnitVerdict],
        Field(description="Explicit measured/no-opportunity units."),
    ]

    unit_errors: list[UnitError] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_units(self):
        """Prevent duplicated units from changing a diagnostic denominator."""
        all_units = [*self.units, *self.unit_errors]
        if not all_units:
            raise ValueError("Empty diagnostic result")
        if len({unit.unit_id for unit in all_units}) != len(all_units):
            raise ValueError("Duplicate diagnostic unit IDs")
        return self


class StoredVerdict(ContractModel):
    """Diagnostic progress sidecar; contains no task reward field."""

    identity: Annotated[VerdictIdentity, Field(description="Complete cache identity.")]
    verdict: Annotated[
        MeasureVerdict | None,
        Field(default=None, description="Validated result or null on error."),
    ]
    error: Annotated[
        str | None, Field(default=None, description="Explicit backend/parsing failure.")
    ]

    @model_validator(mode="after")
    def result_or_error(self):
        """A stored unit must carry either a result or an error."""
        if (self.verdict is None) == (self.error is None):
            raise ValueError("Store exactly one of verdict or error")
        return self


class PlanAction(str, Enum):
    """Distinct work/coverage states, including unsupported and missing inputs."""

    REUSE = "reuse"
    MISSING = "missing"
    STALE = "stale"
    FORCE = "force"
    RETRY_ERROR = "retry_error"
    NOT_APPLICABLE = "not_applicable"
    MISSING_INPUT = "missing_input"


class PlannedVerdict(ContractModel):
    """One saved simulation/measure's proposed action."""

    input: Annotated[JudgeInput, Field(description="Selected recorded simulation.")]
    measure: Annotated[DiagnosticId, Field(description="Requested measure.")]
    action: Annotated[
        PlanAction,
        Field(description="Reuse, work required, or explicit applicability gap."),
    ]
    identity: Annotated[
        VerdictIdentity | None,
        Field(description="Desired cache identity when applicable."),
    ]
    reason: Annotated[
        str, Field(description="Human-readable reason for the planned state.")
    ]


class SuitePlan(ContractModel):
    """Read-only plan; all stored trials included unless explicitly filtered."""

    items: Annotated[
        list[PlannedVerdict],
        Field(description="Selected simulation/measure work items."),
    ]


class SuiteResult(ContractModel):
    """Selected coverage and stored results after a read-only or executing suite."""

    plan: Annotated[SuitePlan, Field(description="Current selected work and coverage.")]
    verdicts: Annotated[
        list[StoredVerdict],
        Field(description="Matching selection, including stale/error records."),
    ]
