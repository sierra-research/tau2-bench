# Copyright Sierra
"""Typed reader and offline verifier for the tau-multilingual review archive."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau2.judges.nativeness.validation import ValidationPrompt

ARCHIVE_LANGUAGES = ("es", "pt", "hi", "ko", "zh")
FCE_LANGUAGES = (*ARCHIVE_LANGUAGES, "en")


class MeasureId(str, Enum):
    """Public validation constructs exposed to paper readers."""

    NATURALNESS = "naturalness"
    SPEECH_FIDELITY = "speech_fidelity"
    TOOL_USE = "tool_use"
    REGISTER_FORMALITY = "register_formality"
    REGIONAL_CONSISTENCY = "regional_consistency"
    MODAL_PARTICLES = "modal_particles"
    GENDER_AGREEMENT = "gender_agreement"
    HONORIFIC_AGREEMENT = "honorific_agreement"
    COUNTING_UNITS = "counting_units"
    NAME_ADDRESS_CONVENTIONS = "name_address_conventions"


EXPECTED_MEASURES: dict[str, tuple[MeasureId, ...]] = {
    "en": (
        MeasureId.SPEECH_FIDELITY,
        MeasureId.TOOL_USE,
    ),
    "es": (
        MeasureId.NATURALNESS,
        MeasureId.SPEECH_FIDELITY,
        MeasureId.TOOL_USE,
        MeasureId.COUNTING_UNITS,
    ),
    "pt": (
        MeasureId.NATURALNESS,
        MeasureId.SPEECH_FIDELITY,
        MeasureId.TOOL_USE,
        MeasureId.REGISTER_FORMALITY,
        MeasureId.GENDER_AGREEMENT,
        MeasureId.COUNTING_UNITS,
        MeasureId.REGIONAL_CONSISTENCY,
    ),
    "hi": (
        MeasureId.NATURALNESS,
        MeasureId.SPEECH_FIDELITY,
        MeasureId.TOOL_USE,
        MeasureId.GENDER_AGREEMENT,
        MeasureId.NAME_ADDRESS_CONVENTIONS,
    ),
    "ko": (
        MeasureId.NATURALNESS,
        MeasureId.SPEECH_FIDELITY,
        MeasureId.TOOL_USE,
        MeasureId.HONORIFIC_AGREEMENT,
        MeasureId.NAME_ADDRESS_CONVENTIONS,
    ),
    "zh": (
        MeasureId.NATURALNESS,
        MeasureId.SPEECH_FIDELITY,
        MeasureId.TOOL_USE,
        MeasureId.COUNTING_UNITS,
        MeasureId.MODAL_PARTICLES,
        MeasureId.NAME_ADDRESS_CONVENTIONS,
    ),
}

VALIDATION_LANGUAGES = tuple(EXPECTED_MEASURES)

DETERMINISTIC_MEASURES: dict[str, tuple[str, ...]] = {
    "en": (),
    "es": ("email_symbol_verbalization",),
    "pt": ("email_symbol_verbalization",),
    "hi": (),
    "ko": ("email_symbol_verbalization",),
    "zh": ("email_symbol_verbalization",),
}

# Exact runtime contracts archived in prompts.json. These are implementation
# factors, not public validation measures: the three tool factors roll up to
# ``tool_use`` and ``tone_meaning_flip`` rolls up to ``speech_fidelity``.
PROMPT_FACTOR_MATRIX: dict[str, tuple[str, ...]] = {
    "en": (
        "incorrect_tool_parameters",
        "auth_arg_mismatch",
        "agent_caused_tool_error",
        "fidelity",
    ),
    "es": (
        "natural_word_choice",
        "counting_units",
        "email_symbol_verbalization",
        "incorrect_tool_parameters",
        "auth_arg_mismatch",
        "agent_caused_tool_error",
        "fidelity",
    ),
    "pt": (
        "natural_word_choice",
        "register_formality",
        "gender_agreement",
        "counting_units",
        "regional_consistency",
        "email_symbol_verbalization",
        "incorrect_tool_parameters",
        "auth_arg_mismatch",
        "agent_caused_tool_error",
        "fidelity",
    ),
    "hi": (
        "natural_word_choice",
        "gender_agreement",
        "name_address_conventions",
        "incorrect_tool_parameters",
        "auth_arg_mismatch",
        "agent_caused_tool_error",
        "fidelity",
    ),
    "ko": (
        "natural_word_choice",
        "honorific_agreement",
        "name_address_conventions",
        "email_symbol_verbalization",
        "incorrect_tool_parameters",
        "auth_arg_mismatch",
        "agent_caused_tool_error",
        "fidelity",
    ),
    "zh": (
        "natural_word_choice",
        "counting_units",
        "modal_particles",
        "name_address_conventions",
        "email_symbol_verbalization",
        "incorrect_tool_parameters",
        "auth_arg_mismatch",
        "agent_caused_tool_error",
        "fidelity",
        "tone_meaning_flip",
    ),
}

MEASURE_LEVELS: dict[MeasureId, "EvaluationLevel"] = {}

MEASURE_SOURCE_FACTORS: dict[MeasureId, frozenset[str]] = {
    MeasureId.NATURALNESS: frozenset(
        {
            "natural_word_choice",
            "translationese",
            "verb_morphology",
            "grammar",
            "regional_consistency",
            "written_diacritics",
        }
    ),
    MeasureId.SPEECH_FIDELITY: frozenset({"fidelity", "tone_meaning_flip"}),
    MeasureId.TOOL_USE: frozenset(
        {
            "incorrect_tool_parameters",
            "auth_arg_mismatch",
            "agent_caused_tool_error",
        }
    ),
    MeasureId.REGISTER_FORMALITY: frozenset({"register_formality"}),
    MeasureId.REGIONAL_CONSISTENCY: frozenset({"regional_consistency"}),
    MeasureId.MODAL_PARTICLES: frozenset({"modal_particles"}),
    MeasureId.GENDER_AGREEMENT: frozenset({"gender_agreement"}),
    MeasureId.HONORIFIC_AGREEMENT: frozenset({"honorific_agreement"}),
    MeasureId.COUNTING_UNITS: frozenset({"counting_units"}),
    MeasureId.NAME_ADDRESS_CONVENTIONS: frozenset({"name_address_conventions"}),
}


class SourceSet(str, Enum):
    """Closed sampling strata represented by the review rows."""

    DEVELOPMENT = "development"
    HELD_OUT = "held_out"
    FIXED_CURATED = "fixed_curated"
    RECALL = "recall"
    PRECISION = "precision"
    UNIFIED = "unified"


class EvaluationLevel(str, Enum):
    """Unit at which a human label and judge verdict are compared."""

    UTTERANCE = "utterance"
    CALL = "call"


class UtteranceAlignmentKind(str, Enum):
    """Stable identity available for one utterance-level comparison."""

    AGENT_TURN = "agent_turn"
    AUDIO_SEGMENT = "audio_segment"


class StoredReferenceAlignment(str, Enum):
    """Relation between a human reference and stored judge metadata preview."""

    EXACT = "exact"
    STORED_PREFIX = "stored_prefix"


MEASURE_LEVELS.update(
    {
        MeasureId.NATURALNESS: EvaluationLevel.UTTERANCE,
        MeasureId.SPEECH_FIDELITY: EvaluationLevel.UTTERANCE,
        MeasureId.TOOL_USE: EvaluationLevel.CALL,
        MeasureId.REGISTER_FORMALITY: EvaluationLevel.CALL,
        MeasureId.REGIONAL_CONSISTENCY: EvaluationLevel.CALL,
        MeasureId.MODAL_PARTICLES: EvaluationLevel.CALL,
        MeasureId.GENDER_AGREEMENT: EvaluationLevel.UTTERANCE,
        MeasureId.HONORIFIC_AGREEMENT: EvaluationLevel.UTTERANCE,
        MeasureId.COUNTING_UNITS: EvaluationLevel.UTTERANCE,
        MeasureId.NAME_ADDRESS_CONVENTIONS: EvaluationLevel.UTTERANCE,
    }
)


class BinaryLabel(str, Enum):
    """Human and judge labels used by the archive."""

    VIOLATION = "violation"
    PASS = "pass"
    NO_OPPORTUNITY = "no_opportunity"
    ERROR = "error"


class LabelOrigin(str, Enum):
    """How the final human label was attached to this comparison unit."""

    DIRECT_UTTERANCE = "direct_utterance"
    PROJECTED_REVIEWED_CALL = "projected_reviewed_call"
    DIRECT_CALL = "direct_call"
    COMBINED_CALL = "combined_call"


class ValidationSourceArtifact(BaseModel):
    """Hash-bound source artifact used to create a canonical validation row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Annotated[
        str, Field(min_length=1, description="Typed role of this source artifact.")
    ]
    path: Annotated[str, Field(min_length=1, description="Neutral source alias.")]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    revision: Annotated[
        str, Field(min_length=1, description="Immutable source revision or export id.")
    ]

    @field_validator("path")
    @classmethod
    def _path_is_archive_relative(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source artifact path must be archive-relative")
        return value


def _blank_to_none(value):
    if isinstance(value, str) and value.strip().upper() in {"", "NA", "N/A"}:
        return None
    return value


def _json_list(value) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, tuple):
        value = list(value)
    return value


class ValidationRow(BaseModel):
    """One human label aligned with one stored judge verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_id: Annotated[str, Field(min_length=12, description="Stable row identifier.")]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    measure_id: Annotated[
        MeasureId, Field(description="Public construct evaluated by this row.")
    ]
    source_factor_ids: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Runtime factors or subtypes contributing to the construct.",
        ),
    ]
    source_sets: Annotated[
        list[SourceSet],
        Field(
            min_length=1,
            description="Validation sampling strata represented by this comparison.",
        ),
    ]
    evaluation_level: Annotated[EvaluationLevel, Field(description="Comparison unit.")]
    label_origin: Annotated[
        LabelOrigin, Field(description="How the human label was localized to the unit.")
    ]
    simulation_id: Annotated[str, Field(description="Source simulation identifier.")]
    task_id: Annotated[str, Field(description="Source benchmark task identifier.")]
    utterance_id: Annotated[
        Optional[str],
        Field(description="Stable identity of the compared utterance or segment."),
    ] = None
    utterance_alignment: Annotated[
        Optional[UtteranceAlignmentKind],
        Field(description="Identity authority for an utterance-level row."),
    ] = None
    agent_turn_index: Annotated[
        Optional[int], Field(ge=0, description="Delivered agent-turn index, if used.")
    ] = None
    agent_turn_id: Annotated[
        Optional[str], Field(description="Stable agent-turn identifier, if used.")
    ] = None
    agent_utterance: Annotated[
        Optional[str], Field(description="Exact delivered agent text, if used.")
    ] = None
    stored_reference_preview: Annotated[
        Optional[str],
        Field(
            min_length=1,
            max_length=200,
            description="Exact reference preview retained with a speech verdict.",
        ),
    ] = None
    stored_reference_alignment: Annotated[
        Optional[StoredReferenceAlignment],
        Field(description="Relation between human reference and stored preview."),
    ] = None
    preceding_customer_utterance: Annotated[
        Optional[str], Field(description="Immediately preceding customer text.")
    ] = None
    audio_start_seconds: Annotated[
        Optional[float], Field(ge=0, description="Segment start, when audio-aligned.")
    ] = None
    audio_end_seconds: Annotated[
        Optional[float], Field(gt=0, description="Segment end, when audio-aligned.")
    ] = None
    judge_quote: Annotated[
        Optional[str], Field(description="Exact span selected by the judge, if any.")
    ] = None
    human_label: Annotated[BinaryLabel, Field(description="Final human label.")]
    judge_label: Annotated[BinaryLabel, Field(description="Stored judge label.")]
    judge_reason: Annotated[
        Optional[str], Field(description="Stored judge explanation, if supplied.")
    ] = None
    agreement: Annotated[
        Optional[bool], Field(description="Binary agreement, when both labels score.")
    ] = None
    judge_model: Annotated[
        Optional[str],
        Field(description="Judge model, or deterministic when applicable."),
    ] = None
    prompt_version: Annotated[
        Optional[str], Field(description="Version of the judge prompt or checker.")
    ] = None
    rubric_version: Annotated[
        Optional[str], Field(description="Version of the factor rubric.")
    ] = None

    _optional_blanks = field_validator(
        "agent_turn_index",
        "utterance_id",
        "utterance_alignment",
        "agent_turn_id",
        "agent_utterance",
        "stored_reference_preview",
        "stored_reference_alignment",
        "preceding_customer_utterance",
        "audio_start_seconds",
        "audio_end_seconds",
        "judge_quote",
        "judge_reason",
        "agreement",
        "judge_model",
        "prompt_version",
        "rubric_version",
        mode="before",
    )(_blank_to_none)

    _parse_json_lists = field_validator(
        "source_factor_ids",
        "source_sets",
        mode="before",
    )(_json_list)

    @field_validator("agreement", mode="before")
    @classmethod
    def _parse_agreement(cls, value):
        value = _blank_to_none(value)
        if value is None or isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        return value

    @model_validator(mode="after")
    def _alignment_is_coherent(self) -> "ValidationRow":
        if self.language not in VALIDATION_LANGUAGES:
            raise ValueError(f"unsupported archive language: {self.language}")
        if self.measure_id not in EXPECTED_MEASURES[self.language]:
            raise ValueError(
                f"unexpected public measure for {self.language}: {self.measure_id}"
            )
        expected_level = MEASURE_LEVELS[self.measure_id]
        if self.evaluation_level is not expected_level:
            raise ValueError(
                f"{self.measure_id.value} must be {expected_level.value}-level"
            )
        if SourceSet.UNIFIED in self.source_sets:
            raise ValueError("unified is derived metric provenance, not a row source")
        if len(self.source_sets) != len(set(self.source_sets)):
            raise ValueError("source_sets must be unique")
        source_order = {source: index for index, source in enumerate(SourceSet)}
        if self.source_sets != sorted(self.source_sets, key=source_order.__getitem__):
            raise ValueError("source_sets must be canonically ordered")
        source_factors = set(self.source_factor_ids)
        if len(source_factors) != len(self.source_factor_ids):
            raise ValueError("source_factor_ids must be unique")
        unknown = source_factors - MEASURE_SOURCE_FACTORS[self.measure_id]
        if unknown:
            raise ValueError(
                f"invalid source factors for {self.measure_id.value}: {sorted(unknown)}"
            )
        utterance_fields = (
            self.utterance_id,
            self.utterance_alignment,
            self.agent_turn_index,
            self.agent_turn_id,
            self.agent_utterance,
            self.stored_reference_preview,
            self.stored_reference_alignment,
            self.audio_start_seconds,
            self.audio_end_seconds,
        )
        if self.evaluation_level is EvaluationLevel.CALL and any(
            value is not None for value in utterance_fields
        ):
            raise ValueError("call rows may not carry utterance fields")
        if self.evaluation_level is EvaluationLevel.UTTERANCE:
            if self.utterance_id is None or self.agent_utterance is None:
                raise ValueError("utterance rows require a stable id and exact text")
            if self.utterance_alignment is UtteranceAlignmentKind.AGENT_TURN:
                if self.agent_turn_index is None or self.agent_turn_id is None:
                    raise ValueError("agent-turn rows require its index and id")
                if any(
                    value is not None
                    for value in (self.audio_start_seconds, self.audio_end_seconds)
                ):
                    raise ValueError("agent-turn rows may not carry an audio interval")
            elif self.utterance_alignment is UtteranceAlignmentKind.AUDIO_SEGMENT:
                if self.audio_start_seconds is None or self.audio_end_seconds is None:
                    raise ValueError("audio-segment rows require a time interval")
                if self.audio_end_seconds <= self.audio_start_seconds:
                    raise ValueError(
                        "audio-segment interval must have positive duration"
                    )
                if self.agent_turn_index is not None or self.agent_turn_id is not None:
                    raise ValueError("audio-segment rows may not claim one agent turn")
            else:
                raise ValueError("utterance rows require an alignment kind")
        if self.measure_id is MeasureId.SPEECH_FIDELITY:
            if self.stored_reference_preview is None:
                raise ValueError(
                    "speech-fidelity rows require the stored reference preview"
                )
            if self.stored_reference_alignment is None:
                raise ValueError(
                    "speech-fidelity rows require stored-reference alignment"
                )
            human = " ".join((self.agent_utterance or "").split())
            preview = " ".join(self.stored_reference_preview.split())
            if self.stored_reference_alignment is StoredReferenceAlignment.EXACT:
                aligned = human == preview
            else:
                aligned = human.startswith(preview) and len(preview) < len(human)
            if not aligned:
                raise ValueError(
                    "speech-fidelity reference relation differs from its alignment"
                )
        elif (
            self.stored_reference_preview is not None
            or self.stored_reference_alignment is not None
        ):
            raise ValueError(
                "only speech-fidelity rows may carry a stored reference preview"
            )
        utterance_origins = {
            LabelOrigin.DIRECT_UTTERANCE,
            LabelOrigin.PROJECTED_REVIEWED_CALL,
        }
        if (
            self.evaluation_level is EvaluationLevel.UTTERANCE
            and self.label_origin not in utterance_origins
        ):
            raise ValueError("utterance rows require an utterance label origin")
        if (
            self.evaluation_level is EvaluationLevel.CALL
            and self.label_origin in utterance_origins
        ):
            raise ValueError("call rows require a call label origin")
        scoring = {BinaryLabel.PASS, BinaryLabel.VIOLATION}
        expected = (
            self.human_label == self.judge_label
            if self.human_label in scoring and self.judge_label in scoring
            else None
        )
        if self.agreement != expected:
            raise ValueError(
                f"agreement mismatch for {self.row_id}: {self.agreement=} {expected=}"
            )
        return self


class MetricRow(BaseModel):
    """Confusion matrix and derived statistics for one comparable row group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    measure_id: MeasureId
    source_set: SourceSet
    evaluation_level: EvaluationLevel
    tp: int = Field(ge=0)
    fn: int = Field(ge=0)
    fp: int = Field(ge=0)
    tn: int = Field(ge=0)
    no_opportunity: int = Field(ge=0)
    errors: int = Field(ge=0)
    n: int = Field(ge=0)
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    kappa: Optional[float]

    _optional_blanks = field_validator(
        "precision", "recall", "f1", "kappa", mode="before"
    )(_blank_to_none)


class FceReviewRow(BaseModel):
    """One completed first-critical-error and voice-experience review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    task_id: str
    simulation_id: str
    trial: int = Field(ge=0)
    error_source: Optional[str] = None
    error_type: Optional[str] = None
    notes: Optional[str] = None
    voice_prosody_quality: Optional[int] = Field(default=None, ge=1, le=4)
    audio_environment_realism: Optional[int] = Field(default=None, ge=1, le=4)
    turn_taking_naturalness: Optional[int] = Field(default=None, ge=1, le=4)
    backchannel_naturalness: Optional[int] = Field(default=None, ge=1, le=4)
    interruption_behavior: Optional[int] = Field(default=None, ge=1, le=4)
    behavioral_plausibility: Optional[int] = Field(default=None, ge=1, le=4)
    speech_accuracy: Optional[int] = Field(default=None, ge=1, le=4)
    phrasing_naturalness: Optional[int] = Field(default=None, ge=1, le=4)
    voice_prosody_quality_notes: Optional[str] = None
    audio_environment_realism_notes: Optional[str] = None
    turn_taking_naturalness_notes: Optional[str] = None
    backchannel_naturalness_notes: Optional[str] = None
    interruption_behavior_notes: Optional[str] = None
    behavioral_plausibility_notes: Optional[str] = None
    speech_accuracy_notes: Optional[str] = None
    phrasing_naturalness_notes: Optional[str] = None
    free_text_comments: Optional[str] = None
    caller_experience: Optional[int] = Field(default=None, ge=1, le=4)
    experience_breaking_point: Optional[str] = None
    experience_breaking_tick: Optional[int] = None
    experience_factors: Optional[str] = None
    primary_factor: Optional[str] = None
    experience_notes: Optional[str] = None
    completed: bool

    _optional_blanks = field_validator(
        *[
            name
            for name in (
                "error_source",
                "error_type",
                "notes",
                "voice_prosody_quality",
                "audio_environment_realism",
                "turn_taking_naturalness",
                "backchannel_naturalness",
                "interruption_behavior",
                "behavioral_plausibility",
                "speech_accuracy",
                "phrasing_naturalness",
                "voice_prosody_quality_notes",
                "audio_environment_realism_notes",
                "turn_taking_naturalness_notes",
                "backchannel_naturalness_notes",
                "interruption_behavior_notes",
                "behavioral_plausibility_notes",
                "speech_accuracy_notes",
                "phrasing_naturalness_notes",
                "free_text_comments",
                "caller_experience",
                "experience_breaking_point",
                "experience_breaking_tick",
                "experience_factors",
                "primary_factor",
                "experience_notes",
            )
        ],
        mode="before",
    )(_blank_to_none)


class LanguageReviewScope(str, Enum):
    """Kinds of native-speaker language-pack evidence in the supporting archive."""

    SPEAKER_OBSERVATION = "speaker_observation"
    RETAINED_FACTOR_CLAUSE = "retained_factor_clause"


class LanguageReviewRow(BaseModel):
    """One final native-speaker observation or retained language-factor clause."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_id: str = Field(pattern=r"^[0-9a-f]{20}$")
    language: str
    review_scope: LanguageReviewScope
    category: str
    title: str
    observed_issue: str
    native_expectation: str
    example: Optional[str] = None
    conversation_ref: Optional[str] = None
    severity: str
    human_note: Optional[str] = None

    _optional_blanks = field_validator(
        "example", "conversation_ref", "human_note", mode="before"
    )(_blank_to_none)

    @field_validator("language")
    @classmethod
    def _supported_language(cls, value: str) -> str:
        if value not in ARCHIVE_LANGUAGES:
            raise ValueError(f"unsupported archive language: {value}")
        return value


class CallExperienceReviewRow(BaseModel):
    """One supplemental native-speaker call-experience finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_id: str = Field(pattern=r"^[0-9a-f]{20}$")
    language: str
    call_id: str
    finding_types: str
    human_note: Optional[str] = None

    _optional_blanks = field_validator("human_note", mode="before")(_blank_to_none)

    @field_validator("language")
    @classmethod
    def _supported_language(cls, value: str) -> str:
        if value not in ARCHIVE_LANGUAGES:
            raise ValueError(f"unsupported archive language: {value}")
        return value


class ArchiveFile(BaseModel):
    """One archive file covered by the root digest manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    rows: Optional[int] = Field(default=None, ge=0)


class RuntimePrompt(BaseModel):
    """Versioned wrapper text shared by one judge family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_version: str
    rubric_version: Optional[str] = None
    system_prompt: str
    user_prompt_template: Optional[str] = None
    factor_instructions: dict[str, str] = Field(default_factory=dict)


class FrozenNativenessCriterion(BaseModel):
    """Exact criterion text supplied to a historical nativeness invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor_id: str
    question: str
    opportunity: str
    positive_examples: str
    negative_examples: str


class FrozenQualityCriterion(BaseModel):
    """Exact process-quality criterion supplied to a stored judge call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str
    question: str
    decision_rule: str


class PromptSourceFile(BaseModel):
    """Content digest for code or data that defines a stored prompt contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StoredPredictionPromptContract(BaseModel):
    """Exact prompt contract represented by one or more canonical rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str
    prompt_family: Literal["nativeness", "quality", "delivery"]
    prompt_version: str
    rubric_version: Optional[str] = None
    judge_models: list[str]
    model_args: dict
    languages: list[str]
    measure_ids: list[MeasureId]
    source_revision: str
    source_files: list[PromptSourceFile]
    prompt: RuntimePrompt
    nativeness_criteria: dict[str, list[FrozenNativenessCriterion]] = Field(
        default_factory=dict
    )
    quality_criteria: list[FrozenQualityCriterion] = Field(default_factory=list)
    package_prompt_files: list[ValidationSourceArtifact] = Field(default_factory=list)
    scope_note: Optional[str] = None

    @model_validator(mode="after")
    def _contract_is_coherent(self) -> "StoredPredictionPromptContract":
        if self.prompt.prompt_version != self.prompt_version:
            raise ValueError("stored prompt version differs from its contract")
        if self.prompt.rubric_version != self.rubric_version:
            raise ValueError("stored rubric version differs from its contract")
        if self.languages != sorted(
            set(self.languages), key=VALIDATION_LANGUAGES.index
        ):
            raise ValueError("stored prompt languages must be unique and ordered")
        if len(self.measure_ids) != len(set(self.measure_ids)):
            raise ValueError("stored prompt measures must be unique")
        if len(self.source_files) != len({item.path for item in self.source_files}):
            raise ValueError("stored prompt source files must be unique")
        if self.prompt_family == "nativeness" and not (
            self.nativeness_criteria or self.package_prompt_files
        ):
            raise ValueError("nativeness prompt needs criteria or exact package files")
        if self.prompt_family == "quality" and not self.quality_criteria:
            raise ValueError("quality prompt needs exact criteria")
        if self.prompt_family == "delivery" and not self.package_prompt_files:
            raise ValueError("delivery prompt needs its exact package prompt")
        return self


class PromptArchive(BaseModel):
    """Only the prompt contracts that produced the frozen paper verdicts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-prompt-archive-v4"]
    languages: list[str]
    retained_factors: dict[str, list[str]]
    combined_naturalness: dict[str, ValidationPrompt]
    email_symbols: dict[str, dict[str, list[str]]]
    stored_prediction_prompts: list[StoredPredictionPromptContract]


FROZEN_PROMPT_SOURCE_CONTRACTS: dict[str, tuple[str, dict[str, str]]] = {
    "nativeness-v15-rubric-v19": (
        "6813fceab54fe6c7c44733be9935e6d38aaca848",
        {
            "src/tau2/judges/nativeness/judge.py": (
                "4389326c8eabf7eeb92b88890c3bb201bcd6ef25f7f69c588e410895b7f80b58"
            ),
            "src/tau2/judges/nativeness/factors.py": (
                "8415a7e0bd07864511b026a73c5ab9d88b80dbcc9edb2096f32e1009d9a0c247"
            ),
            "src/tau2/judges/nativeness/checkers.py": (
                "32bb91f4df66e7cf3637103485ba68152cc0d0879fbbdd61d3b5caa2f8dee9bd"
            ),
            "src/tau2/judges/nativeness/harness.py": (
                "8656697acdd05314121e39b36e60d259bdf3104fa360687fa8a8a27224db97e4"
            ),
            "src/tau2/judges/nativeness/agent_voice.py": (
                "113a82e3fb8cf62228a6fce11e9068ddfb5b8432ea0fffdebc1a4f2cf90a0581"
            ),
            "src/tau2/judges/nativeness/caller_identity.py": (
                "024a6b568f8e0b6f1cc41c3c99393003a53f7a34872087c71bc0ab33b93398aa"
            ),
            "src/tau2/multilingual/nativeness_catalog.py": (
                "45067eb5380c576dc06e10d3964c21ca221484be10113d7b15b3d48bf71e59d1"
            ),
            "src/tau2/multilingual/factor_catalog.py": (
                "2af6937dd8cda8d6250f7a34e9b4a92928b7b9082f2bdcf46affd336c724edc9"
            ),
            "data/tau2/multilingual/es/pack.yaml": (
                "1fb90607902d7c8905d44894d92d7d6ab1ac2549326e1c86d9a65e90fbf6ebbf"
            ),
            "data/tau2/multilingual/pt/pack.yaml": (
                "d878b3c3ff00414379167dbabfe2eed9b733150070ff3e95032585fb430af77c"
            ),
            "data/tau2/multilingual/hi/pack.yaml": (
                "d643bff1ad1e536d1b8e5720927ad5327a7e99b46e953f6008d9983f05e2c5fc"
            ),
            "data/tau2/multilingual/ko/pack.yaml": (
                "7342617f36271af349cfe3364173795fae11b88472f590872135929fce2bdd75"
            ),
            "data/tau2/multilingual/zh/pack.yaml": (
                "f25575e9b71ac6b6078fb0995752a7a9aa5f6c5fa2768df5aea729c55608790a"
            ),
        },
    ),
    "quality-v7-rubric-v4": (
        "6813fceab54fe6c7c44733be9935e6d38aaca848",
        {
            "src/tau2/judges/quality/judge.py": (
                "cc8cdbb1791bdc134fcf16b83a1cf73425ab8fcf166463abe985bd82d576f600"
            ),
            "src/tau2/judges/quality/factors.py": (
                "b6802e7ea62919fa6fcd8b5c3616bd2f979fdfe1468b83eee60e73e28683e3e4"
            ),
            "src/tau2/judges/quality/checkers.py": (
                "60b774c8e8b2e7edf8ea8d5e53221cce90609801f62100e2839bb0bc6f60cf29"
            ),
            "src/tau2/judges/quality/harness.py": (
                "b22edd3d3ea42e0a568f181fb67c754caf1e12f26dfbec26184b76d32aae7a12"
            ),
        },
    ),
    "speech-fidelity-v5": (
        "e1ca72ea1e5b399ffcf3e88f62a746e2ad19591d",
        {
            "src/tau2/judges/delivery/judge.py": (
                "7e1a736bf0e00c5bb0a4b9a38e7d1c38f3ca8b79603b3a7acd9192325f3f881e"
            ),
            "src/tau2/judges/delivery/factors.py": (
                "74e3032b37fdf75989c26869a1bcd99b647741f9ba9ce207ae730742243c7511"
            ),
            "src/tau2/judges/delivery/harness.py": (
                "3482a7257620153c6c264cfee734deaecf3f672b245a03717f381f0f19cfe19f"
            ),
        },
    ),
}


class MeasureCoverage(BaseModel):
    """Number of human-to-judge rows available for one public measure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    measure_id: MeasureId
    human_rows: int = Field(ge=0)
    source_sets: list[SourceSet]
    evaluation_level: EvaluationLevel


class ValidationDesign(str, Enum):
    """Sampling design used for the metric exposed in the archive index."""

    FIXED_HELD_OUT = "fixed_held_out"
    FIXED_CURATED = "fixed_curated"
    UNIFIED_PRECISION_RECALL = "unified_precision_recall"


class ValidationGateStatus(str, Enum):
    """Outcome under the frozen reviewer-facing evidence gate."""

    PASSES = "passes"
    INSUFFICIENT_POSITIVE_SUPPORT = "insufficient_positive_support"
    BELOW_METRIC_THRESHOLD = "below_metric_threshold"


VALIDATION_MINIMUM_PRECISION = 0.75
VALIDATION_MINIMUM_RECALL = 0.75
VALIDATION_MINIMUM_F1 = 0.80
VALIDATION_MINIMUM_HUMAN_POSITIVES = 10
VALIDATION_MINIMUM_HUMAN_NEGATIVES_FOR_KAPPA = 5
VALIDATION_MINIMUM_KAPPA = 0.60


def validation_gate_status(metric: MetricRow) -> ValidationGateStatus:
    """Classify one metric using the frozen validation-evidence gate."""
    human_positives = metric.tp + metric.fn
    human_negatives = metric.tn + metric.fp
    if human_positives < VALIDATION_MINIMUM_HUMAN_POSITIVES:
        return ValidationGateStatus.INSUFFICIENT_POSITIVE_SUPPORT
    if (
        metric.precision is None
        or metric.precision <= VALIDATION_MINIMUM_PRECISION
        or metric.recall is None
        or metric.recall <= VALIDATION_MINIMUM_RECALL
        or metric.f1 is None
        or metric.f1 <= VALIDATION_MINIMUM_F1
    ):
        return ValidationGateStatus.BELOW_METRIC_THRESHOLD
    if human_negatives >= VALIDATION_MINIMUM_HUMAN_NEGATIVES_FOR_KAPPA and (
        metric.kappa is None or metric.kappa <= VALIDATION_MINIMUM_KAPPA
    ):
        return ValidationGateStatus.BELOW_METRIC_THRESHOLD
    return ValidationGateStatus.PASSES


class ValidationIndexRow(BaseModel):
    """One reviewer-facing summary row per language and public measure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    measure_id: Annotated[MeasureId, Field(description="Public measure identity.")]
    evaluation_level: Annotated[
        EvaluationLevel, Field(description="Unit used for every compared row.")
    ]
    validation_design: Annotated[
        ValidationDesign, Field(description="Sampling design used for this summary.")
    ]
    reported_source_set: Annotated[
        SourceSet, Field(description="Metric group selected for reporting.")
    ]
    source_sets: Annotated[
        list[SourceSet], Field(description="Sampling strata present in the corpus.")
    ]
    validation_path: Annotated[
        str, Field(description="Canonical row CSV relative to validations/.")
    ]
    metrics_path: Annotated[
        str, Field(description="Derived metric CSV relative to validations/.")
    ]
    human_positives: int = Field(ge=0)
    human_negatives: int = Field(ge=0)
    gate_status: ValidationGateStatus
    tp: int = Field(ge=0)
    fn: int = Field(ge=0)
    fp: int = Field(ge=0)
    tn: int = Field(ge=0)
    no_opportunity: int = Field(ge=0)
    errors: int = Field(ge=0)
    n: int = Field(ge=0)
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    kappa: Optional[float]

    _parse_source_sets = field_validator("source_sets", mode="before")(_json_list)
    _optional_blanks = field_validator(
        "precision", "recall", "f1", "kappa", mode="before"
    )(_blank_to_none)

    @model_validator(mode="after")
    def _summary_is_coherent(self) -> "ValidationIndexRow":
        if self.language not in VALIDATION_LANGUAGES:
            raise ValueError(f"unsupported archive language: {self.language}")
        if self.measure_id not in EXPECTED_MEASURES[self.language]:
            raise ValueError(
                f"unexpected public measure for {self.language}: {self.measure_id}"
            )
        if self.evaluation_level is not MEASURE_LEVELS[self.measure_id]:
            raise ValueError("index evaluation level differs from measure contract")
        leaf = (
            f"{self.evaluation_level.value}_level/"
            f"{self.measure_id.value}/{self.language}"
        )
        if self.validation_path != f"{leaf}/validation.csv":
            raise ValueError("index validation path differs from its leaf")
        if self.metrics_path != f"{leaf}/metrics.csv":
            raise ValueError("index metrics path differs from its leaf")
        if self.validation_design is ValidationDesign.FIXED_HELD_OUT:
            if self.reported_source_set is not SourceSet.HELD_OUT:
                raise ValueError("fixed validation must report held_out")
        elif self.validation_design is ValidationDesign.FIXED_CURATED:
            if self.reported_source_set is not SourceSet.FIXED_CURATED:
                raise ValueError("fixed curated validation must report fixed_curated")
        elif self.reported_source_set is not SourceSet.UNIFIED:
            raise ValueError("precision/recall validation must report unified")
        if self.human_positives != self.tp + self.fn:
            raise ValueError("index human-positive count differs from confusion matrix")
        if self.human_negatives != self.tn + self.fp:
            raise ValueError("index human-negative count differs from confusion matrix")
        if self.n != self.human_positives + self.human_negatives:
            raise ValueError("index sample size differs from scorable confusion matrix")
        expected_gate_status = validation_gate_status(
            MetricRow(
                language=self.language,
                measure_id=self.measure_id,
                source_set=self.reported_source_set,
                evaluation_level=self.evaluation_level,
                tp=self.tp,
                fn=self.fn,
                fp=self.fp,
                tn=self.tn,
                no_opportunity=self.no_opportunity,
                errors=self.errors,
                n=self.n,
                precision=self.precision,
                recall=self.recall,
                f1=self.f1,
                kappa=self.kappa,
            )
        )
        if self.gate_status is not expected_gate_status:
            raise ValueError("index gate status differs from frozen evidence gate")
        return self


class ValidationLeafManifest(BaseModel):
    """Hash-bound inventory for one language-and-measure validation leaf."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-validation-leaf-v2"]
    evaluation_level: EvaluationLevel
    measure_id: MeasureId
    language: str
    validation: ArchiveFile
    metrics: ArchiveFile
    rows: int = Field(ge=0)
    metric_rows: int = Field(ge=0)


class ValidationPartitionRecord(BaseModel):
    """One leaf manifest covered by the validation-root manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_level: EvaluationLevel
    measure_id: MeasureId
    language: str
    manifest: ArchiveFile


class ValidationRootManifest(BaseModel):
    """Hash-bound inventory for the canonical nested validation archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-validations-v2"]
    index: ArchiveFile
    readme: ArchiveFile
    partitions: list[ValidationPartitionRecord]
    rows: int = Field(ge=0)
    metric_rows: int = Field(ge=0)


class ArchiveManifest(BaseModel):
    """Root inventory and coverage contract for the review archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-human-annotations-v4"]
    validation_languages: list[str]
    fce_languages: list[str]
    supporting_review_languages: list[str]
    expected_measures: dict[str, list[MeasureId]]
    deterministic_measures: dict[str, list[str]]
    validation_rows: dict[EvaluationLevel, int]
    fce_rows: int = Field(ge=0)
    files: list[ArchiveFile]
    measure_coverage: list[MeasureCoverage]


class ArchiveValidationReport(BaseModel):
    """Successful offline verification summary."""

    root: Path
    call_validation_rows: int
    utterance_validation_rows: int
    fce_rows: int
    language_review_rows: int
    call_experience_rows: int
    metric_rows: int
    files_verified: int
    deterministic_measures: list[str]
    fce_recorded_errors: int
    fce_user_simulator_errors: int
    fce_infrastructure_errors: int
    fce_quality_ratings: int
    fce_quality_mean: float
    fce_backchannel_ratings: int
    fce_backchannel_mean: float


def csv_fieldnames(model: type[BaseModel]) -> list[str]:
    """Return a serialized CSV contract directly from a pydantic model."""
    return list(model.model_fields)


def _read_csv(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = csv_fieldnames(model)
        if reader.fieldnames != expected:
            raise ValueError(
                f"{path}: CSV columns differ: {reader.fieldnames=} {expected=}"
            )
        return [model.model_validate(row) for row in reader]


def _canonical_csv_text(value: Optional[str]) -> Optional[str]:
    """Remove transport-only line endings and end-of-line horizontal space."""
    if value is None:
        return None
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip(" \t") for line in normalized.split("\n"))


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str):
        return _canonical_csv_text(value)
    return value


def write_csv(path: Path, model: type[BaseModel], rows: list[BaseModel]) -> None:
    """Write a model-derived CSV contract atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=csv_fieldnames(model), lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                payload = row.model_dump(mode="json", exclude_none=False)
                writer.writerow(
                    {key: _csv_cell(value) for key, value in payload.items()}
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, model: BaseModel) -> None:
    """Write one typed JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(model.model_dump_json(indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text(path: Path, content: str) -> None:
    """Write text atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _binary_metrics(rows: list[ValidationRow]) -> dict:
    tp = sum(
        row.human_label is BinaryLabel.VIOLATION
        and row.judge_label is BinaryLabel.VIOLATION
        for row in rows
    )
    fn = sum(
        row.human_label is BinaryLabel.VIOLATION and row.judge_label is BinaryLabel.PASS
        for row in rows
    )
    fp = sum(
        row.human_label is BinaryLabel.PASS and row.judge_label is BinaryLabel.VIOLATION
        for row in rows
    )
    tn = sum(
        row.human_label is BinaryLabel.PASS and row.judge_label is BinaryLabel.PASS
        for row in rows
    )
    no_opportunity = sum(
        BinaryLabel.NO_OPPORTUNITY in {row.human_label, row.judge_label} for row in rows
    )
    errors = sum(row.judge_label is BinaryLabel.ERROR for row in rows)
    n = tp + fn + fp + tn
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    kappa = None
    if n:
        observed = (tp + tn) / n
        human_positive = (tp + fn) / n
        judge_positive = (tp + fp) / n
        expected = human_positive * judge_positive + (1 - human_positive) * (
            1 - judge_positive
        )
        if expected != 1:
            kappa = (observed - expected) / (1 - expected)
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "no_opportunity": no_opportunity,
        "errors": errors,
        "n": n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
    }


def compute_metric_rows(rows: list[ValidationRow]) -> list[MetricRow]:
    """Recompute source-specific and valid unified metrics from archive rows."""
    groups: dict[
        tuple[str, MeasureId, SourceSet, EvaluationLevel], list[ValidationRow]
    ] = defaultdict(list)
    for row in rows:
        for source_set in row.source_sets:
            groups[
                (row.language, row.measure_id, source_set, row.evaluation_level)
            ].append(row)

    measure_levels: dict[tuple[str, MeasureId], set[EvaluationLevel]] = defaultdict(set)
    measure_rows: dict[tuple[str, MeasureId], list[ValidationRow]] = defaultdict(list)
    for row in rows:
        measure_levels[(row.language, row.measure_id)].add(row.evaluation_level)
        measure_rows[(row.language, row.measure_id)].append(row)
    for key, levels in measure_levels.items():
        source_sets = {
            source_set for row in measure_rows[key] for source_set in row.source_sets
        }
        if (
            len(levels) == 1
            and {
                SourceSet.PRECISION,
                SourceSet.RECALL,
            }
            <= source_sets
        ):
            level = next(iter(levels))
            groups[(key[0], key[1], SourceSet.UNIFIED, level)] = measure_rows[key]

    result = [
        MetricRow(
            language=language,
            measure_id=measure_id,
            source_set=source_set,
            evaluation_level=level,
            **_binary_metrics(group_rows),
        )
        for (language, measure_id, source_set, level), group_rows in groups.items()
    ]
    source_order = {source: index for index, source in enumerate(SourceSet)}
    return sorted(
        result,
        key=lambda row: (
            VALIDATION_LANGUAGES.index(row.language),
            EXPECTED_MEASURES[row.language].index(row.measure_id),
            source_order[row.source_set],
            row.evaluation_level.value,
        ),
    )


def _index_sort_key(row: ValidationIndexRow) -> tuple[int, int]:
    return (
        VALIDATION_LANGUAGES.index(row.language),
        EXPECTED_MEASURES[row.language].index(row.measure_id),
    )


def compute_index_rows(
    rows: list[ValidationRow], metrics: list[MetricRow]
) -> list[ValidationIndexRow]:
    """Select one documented reporting metric per language and measure."""
    by_measure: dict[tuple[str, MeasureId], list[ValidationRow]] = defaultdict(list)
    for row in rows:
        by_measure[(row.language, row.measure_id)].append(row)
    metric_lookup = {
        (row.language, row.measure_id, row.source_set): row for row in metrics
    }

    result = []
    for (language, measure_id), measure_rows in by_measure.items():
        source_sets = sorted(
            {source for row in measure_rows for source in row.source_sets},
            key=lambda source: source.value,
        )
        if SourceSet.HELD_OUT in source_sets:
            design = ValidationDesign.FIXED_HELD_OUT
            reported_source_set = SourceSet.HELD_OUT
        elif SourceSet.FIXED_CURATED in source_sets:
            design = ValidationDesign.FIXED_CURATED
            reported_source_set = SourceSet.FIXED_CURATED
        elif {SourceSet.PRECISION, SourceSet.RECALL} <= set(source_sets):
            design = ValidationDesign.UNIFIED_PRECISION_RECALL
            reported_source_set = SourceSet.UNIFIED
        else:
            raise ValueError(
                f"{language}:{measure_id.value} has neither held-out nor complete "
                "precision/recall evidence"
            )
        metric = metric_lookup[(language, measure_id, reported_source_set)]
        level = MEASURE_LEVELS[measure_id]
        leaf_root = f"{level.value}_level/{measure_id.value}/{language}"
        metric_values = metric.model_dump(
            exclude={"language", "measure_id", "source_set", "evaluation_level"}
        )
        result.append(
            ValidationIndexRow(
                language=language,
                measure_id=measure_id,
                evaluation_level=level,
                validation_design=design,
                reported_source_set=reported_source_set,
                source_sets=source_sets,
                validation_path=f"{leaf_root}/validation.csv",
                metrics_path=f"{leaf_root}/metrics.csv",
                human_positives=metric.tp + metric.fn,
                human_negatives=metric.tn + metric.fp,
                gate_status=validation_gate_status(metric),
                **metric_values,
            )
        )
    return sorted(result, key=_index_sort_key)


def _sorted_validation_rows(rows: list[ValidationRow]) -> list[ValidationRow]:
    source_order = {source: index for index, source in enumerate(SourceSet)}
    return sorted(
        rows,
        key=lambda row: (
            VALIDATION_LANGUAGES.index(row.language),
            EXPECTED_MEASURES[row.language].index(row.measure_id),
            tuple(source_order[source] for source in row.source_sets),
            row.simulation_id,
            row.agent_turn_index if row.agent_turn_index is not None else -1,
            row.row_id,
        ),
    )


def _reject_development_rows(rows: list[ValidationRow]) -> None:
    """Keep prompt-development evidence out of the active paper archive."""
    development_ids = [
        row.row_id for row in rows if SourceSet.DEVELOPMENT in row.source_sets
    ]
    if development_ids:
        raise ValueError(
            "active validation archive may not contain development rows: "
            f"{development_ids[:5]}"
        )


def _archive_file(path: Path, *, relative: str, rows: Optional[int]) -> ArchiveFile:
    return ArchiveFile(
        path=relative,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        rows=rows,
    )


def render_validation_readme(index_rows: list[ValidationIndexRow]) -> str:
    """Render the nested validation archive guide and metric table."""
    lines = [
        "# Human validation",
        "",
        "Final human labels and row-aligned stored judge outputs for retained paper measures.",
        "",
        "- Rows are partitioned once as "
        "`<evaluation-level>_level/<measure>/<language>/validation.csv`.",
        "- Call-level leaves contain one whole-call comparison per row; "
        "utterance-level leaves contain one stable utterance comparison per row, "
        "aligned either to an agent turn or an exact audio segment.",
        "- `utterance_alignment` identifies that authority; audio segments carry "
        "their exact start/end window and never claim a single agent-turn id.",
        "- Speech-fidelity rows preserve the cleaned delivered reference in "
        "`agent_utterance` and v5's exact, at-most-200-character metadata preview "
        "in `stored_reference_preview`; `stored_reference_alignment` is "
        "verifier-enforced as `exact` or `stored_prefix` after whitespace "
        "normalization. The full reference, not only this preview, was prompted.",
        "- Speech-fidelity validation tests the underlying detector at any retained "
        "severity after the final-second filter; the paper's downstream Experience "
        "and standalone summaries separately require severity 2 or higher.",
        "- `precision` and `recall` identify sampling strata in `source_sets`; they are not directory levels.",
        "- `index.csv` selects one paper-facing metric per language and measure. "
        "Fixed packages report their recorded `held_out` or `fixed_curated` set; "
        "precision/recall corpora report `unified`.",
        "- The archive contains only final rows for the paper-facing measures.",
        "- `stored_prediction_prompts` is the sole learned-judge prompt inventory. "
        "It preserves only the exact contracts used for the frozen verdicts. A "
        "multi-factor contract can mention criteria outside the paper-facing "
        "measure set when those criteria were present in the original batched "
        "judge call; their unused outputs are not shipped as measures.",
        "- `gate_status` applies the frozen evidence rule: precision and recall "
        "must each exceed 0.75, F1 must exceed 0.80, at least ten human "
        "violations must be present, and Cohen's kappa must exceed 0.60 when "
        "there are at least five human negatives.",
        "- Runtime factors and tool-use subtypes remain in `source_factor_ids`; acceptance gates apply only to `measure_id`.",
        "- Every `metrics.csv` is derived from its sibling `validation.csv`; leaf "
        "manifests bind both by SHA-256, and the root manifest binds every leaf.",
        "- Portuguese `gender_agreement` validates the frozen v15/v19 female-agent frame recorded in `prompts.json`; it does not validate a later runtime contract.",
        "",
        "| Language | Measure | Unit | Design | Pos/neg | Gate status | N | Precision | Recall | F1 | κ |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in index_rows:
        lines.append(
            f"| {row.language} | {row.measure_id.value} | "
            f"{row.evaluation_level.value} | {row.validation_design.value} | "
            f"{row.human_positives}/{row.human_negatives} | "
            f"{row.gate_status.value} | {row.n} | "
            f"{_format_metric(row.precision)} | "
            f"{_format_metric(row.recall)} | {_format_metric(row.f1)} | "
            f"{_format_metric(row.kappa)} |"
        )
    return "\n".join(lines) + "\n"


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_validation_archive(root: Path, rows: list[ValidationRow]) -> None:
    """Write the canonical nested validation rows, metrics, index, and manifests."""
    validation_root = root.resolve() / "validations"
    _reject_development_rows(rows)
    sorted_rows = _sorted_validation_rows(rows)
    _validate_validation_row_identity(sorted_rows)

    validation_root.mkdir(parents=True, exist_ok=True)
    for level in EvaluationLevel:
        level_root = validation_root / f"{level.value}_level"
        if level_root.exists():
            shutil.rmtree(level_root)

    all_metrics = compute_metric_rows(sorted_rows)
    rows_by_partition: dict[
        tuple[str, MeasureId, EvaluationLevel], list[ValidationRow]
    ] = defaultdict(list)
    for row in sorted_rows:
        rows_by_partition[(row.language, row.measure_id, row.evaluation_level)].append(
            row
        )

    partitions = []
    for language, measure_id, level in sorted(
        rows_by_partition,
        key=lambda item: (
            VALIDATION_LANGUAGES.index(item[0]),
            EXPECTED_MEASURES[item[0]].index(item[1]),
            item[2].value,
        ),
    ):
        leaf_rows = rows_by_partition[(language, measure_id, level)]
        leaf_metrics = [
            row
            for row in all_metrics
            if row.language == language and row.measure_id is measure_id
        ]
        relative_leaf = f"{level.value}_level/{measure_id.value}/{language}"
        leaf_root = validation_root / relative_leaf
        validation_path = leaf_root / "validation.csv"
        metrics_path = leaf_root / "metrics.csv"
        write_csv(validation_path, ValidationRow, leaf_rows)
        write_csv(metrics_path, MetricRow, leaf_metrics)
        manifest = ValidationLeafManifest(
            schema_version="tau-multi-validation-leaf-v2",
            evaluation_level=level,
            measure_id=measure_id,
            language=language,
            validation=_archive_file(
                validation_path,
                relative="validation.csv",
                rows=len(leaf_rows),
            ),
            metrics=_archive_file(
                metrics_path,
                relative="metrics.csv",
                rows=len(leaf_metrics),
            ),
            rows=len(leaf_rows),
            metric_rows=len(leaf_metrics),
        )
        manifest_path = leaf_root / "manifest.json"
        write_json(manifest_path, manifest)
        partitions.append(
            ValidationPartitionRecord(
                evaluation_level=level,
                measure_id=measure_id,
                language=language,
                manifest=_archive_file(
                    manifest_path,
                    relative=f"{relative_leaf}/manifest.json",
                    rows=None,
                ),
            )
        )

    index_rows = compute_index_rows(sorted_rows, all_metrics)
    index_path = validation_root / "index.csv"
    readme_path = validation_root / "README.md"
    write_csv(index_path, ValidationIndexRow, index_rows)
    write_text(readme_path, render_validation_readme(index_rows))
    root_manifest = ValidationRootManifest(
        schema_version="tau-multi-validations-v2",
        index=_archive_file(index_path, relative="index.csv", rows=len(index_rows)),
        readme=_archive_file(readme_path, relative="README.md", rows=None),
        partitions=partitions,
        rows=len(sorted_rows),
        metric_rows=len(all_metrics),
    )
    write_json(validation_root / "manifest.json", root_manifest)


def sha256_file(path: Path) -> str:
    """Return a lower-case SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_archive_path(root: Path, relative: str) -> Path:
    """Resolve one archive-owned relative path without following symlinks."""
    root = root.resolve()
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ValueError(f"archive path must stay relative to its root: {relative}")
    candidate = root
    for part in candidate_relative.parts:
        if part in {"", "."}:
            continue
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"archive path may not traverse a symlink: {relative}")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"archive path escapes its root: {relative}")
    return resolved


def _metric_signature(row: MetricRow) -> dict:
    return row.model_dump(mode="json", exclude_none=False)


FCE_QUALITY_FIELDS = (
    "voice_prosody_quality",
    "audio_environment_realism",
    "turn_taking_naturalness",
    "backchannel_naturalness",
    "interruption_behavior",
    "behavioral_plausibility",
    "speech_accuracy",
    "phrasing_naturalness",
)


def _mean(values: list[int]) -> float:
    if not values:
        raise ValueError("cannot compute an FCE mean without ratings")
    return sum(values) / len(values)


def _expected_measure_coverage(rows: list[ValidationRow]) -> list[MeasureCoverage]:
    result = []
    for language in VALIDATION_LANGUAGES:
        for measure_id in EXPECTED_MEASURES[language]:
            measure_rows = [
                row
                for row in rows
                if row.language == language and row.measure_id is measure_id
            ]
            result.append(
                MeasureCoverage(
                    language=language,
                    measure_id=measure_id,
                    human_rows=len(measure_rows),
                    source_sets=sorted(
                        {source for row in measure_rows for source in row.source_sets},
                        key=lambda item: item.value,
                    ),
                    evaluation_level=MEASURE_LEVELS[measure_id],
                )
            )
    return result


def refresh_archive_manifest(root: Path, rows: list[ValidationRow]) -> ArchiveManifest:
    """Refresh the root human-annotation inventory after canonical generation."""
    root = root.resolve()
    fce_rows = list(
        _read_csv(root / "user_sim_review" / "fce" / "fce.csv", FceReviewRow)
    )
    manifest_path = root / "manifest.json"
    files = []
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        if file_path == manifest_path:
            continue
        relative = file_path.relative_to(root).as_posix()
        row_count = None
        if file_path.suffix == ".csv":
            with file_path.open(encoding="utf-8-sig", newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
        files.append(_archive_file(file_path, relative=relative, rows=row_count))
    manifest = ArchiveManifest(
        schema_version="tau-multi-human-annotations-v4",
        validation_languages=list(VALIDATION_LANGUAGES),
        fce_languages=list(FCE_LANGUAGES),
        supporting_review_languages=list(ARCHIVE_LANGUAGES),
        expected_measures={
            language: list(EXPECTED_MEASURES[language])
            for language in VALIDATION_LANGUAGES
        },
        deterministic_measures={
            language: list(measures)
            for language, measures in DETERMINISTIC_MEASURES.items()
        },
        validation_rows={
            level: sum(row.evaluation_level is level for row in rows)
            for level in EvaluationLevel
        },
        fce_rows=len(fce_rows),
        files=files,
        measure_coverage=_expected_measure_coverage(rows),
    )
    write_json(manifest_path, manifest)
    return manifest


def _verify_archive_file(root: Path, record: ArchiveFile, *, context: str) -> Path:
    """Resolve and verify one hash-bound archive file."""
    path = _resolve_archive_path(root, record.path)
    if not path.is_file():
        raise ValueError(f"{context}: missing file: {record.path}")
    if path.stat().st_size != record.bytes:
        raise ValueError(f"{context}: byte-size mismatch: {record.path}")
    if sha256_file(path) != record.sha256:
        raise ValueError(f"{context}: SHA-256 mismatch: {record.path}")
    return path


def _partition_sort_key(
    partition: ValidationPartitionRecord,
) -> tuple[int, int, str]:
    return (
        VALIDATION_LANGUAGES.index(partition.language),
        EXPECTED_MEASURES[partition.language].index(partition.measure_id),
        partition.evaluation_level.value,
    )


def _validate_validation_row_identity(rows: list[ValidationRow]) -> None:
    """Reject duplicate row ids and comparison units."""
    row_ids = [row.row_id for row in rows]
    duplicate_ids = [key for key, count in Counter(row_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"duplicate validation row ids: {duplicate_ids[:5]}")

    unit_keys = [
        (row.language, row.measure_id, row.simulation_id, row.utterance_id)
        for row in rows
    ]
    duplicate_units = [key for key, count in Counter(unit_keys).items() if count > 1]
    if duplicate_units:
        raise ValueError(f"duplicate validation source units: {duplicate_units[:5]}")


def _read_validation_leaf(
    validation_root: Path,
    partition: ValidationPartitionRecord,
) -> tuple[list[ValidationRow], list[MetricRow]]:
    """Verify and load one language-and-measure validation leaf."""
    context = (
        f"{partition.language}:{partition.measure_id.value}:"
        f"{partition.evaluation_level.value}"
    )
    expected_leaf = (
        f"{partition.evaluation_level.value}_level/"
        f"{partition.measure_id.value}/{partition.language}"
    )
    if partition.manifest.path != f"{expected_leaf}/manifest.json":
        raise ValueError(f"{context}: partition manifest path is not canonical")

    manifest_path = _verify_archive_file(
        validation_root, partition.manifest, context=context
    )
    leaf_root = manifest_path.parent
    manifest = ValidationLeafManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if (
        manifest.evaluation_level is not partition.evaluation_level
        or manifest.measure_id is not partition.measure_id
        or manifest.language != partition.language
    ):
        raise ValueError(f"{context}: leaf manifest identity drifted")
    if manifest.validation.path != "validation.csv":
        raise ValueError(f"{context}: validation path is not canonical")
    if manifest.metrics.path != "metrics.csv":
        raise ValueError(f"{context}: metrics path is not canonical")

    validation_path = _verify_archive_file(
        leaf_root, manifest.validation, context=context
    )
    metrics_path = _verify_archive_file(leaf_root, manifest.metrics, context=context)
    validation_rows = list(_read_csv(validation_path, ValidationRow))
    metric_rows = list(_read_csv(metrics_path, MetricRow))
    if any(
        row.evaluation_level is not partition.evaluation_level
        or row.measure_id is not partition.measure_id
        or row.language != partition.language
        for row in validation_rows
    ):
        raise ValueError(f"{context}: validation CSV contains a cross-leaf row")
    if any(
        row.evaluation_level is not partition.evaluation_level
        or row.measure_id is not partition.measure_id
        or row.language != partition.language
        for row in metric_rows
    ):
        raise ValueError(f"{context}: metrics CSV contains a cross-leaf row")
    if manifest.rows != len(validation_rows) or manifest.validation.rows != len(
        validation_rows
    ):
        raise ValueError(f"{context}: validation row count drifted")
    if manifest.metric_rows != len(metric_rows) or manifest.metrics.rows != len(
        metric_rows
    ):
        raise ValueError(f"{context}: metric row count drifted")
    if validation_rows != _sorted_validation_rows(validation_rows):
        raise ValueError(f"{context}: validation row order is not canonical")

    computed_metrics = compute_metric_rows(validation_rows)
    if [_metric_signature(row) for row in metric_rows] != [
        _metric_signature(row) for row in computed_metrics
    ]:
        raise ValueError(f"{context}: metrics do not match validation rows")
    return validation_rows, metric_rows


def read_validation_archive(
    root: Path,
) -> tuple[list[ValidationRow], list[MetricRow], list[ValidationIndexRow]]:
    """Verify and load every canonical validation leaf under root/validations."""
    validation_root = root.resolve() / "validations"
    manifest_path = _resolve_archive_path(validation_root, "manifest.json")
    manifest = ValidationRootManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.index.path != "index.csv" or manifest.readme.path != "README.md":
        raise ValueError("validation-root metadata paths are not canonical")
    index_path = _verify_archive_file(
        validation_root, manifest.index, context="validation root"
    )
    readme_path = _verify_archive_file(
        validation_root, manifest.readme, context="validation root"
    )

    partition_keys = [
        (row.language, row.measure_id, row.evaluation_level)
        for row in manifest.partitions
    ]
    if len(partition_keys) != len(set(partition_keys)):
        raise ValueError("validation-root manifest contains duplicate partitions")
    manifest_paths = [row.manifest.path for row in manifest.partitions]
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("validation-root manifest contains duplicate leaf paths")
    if manifest.partitions != sorted(manifest.partitions, key=_partition_sort_key):
        raise ValueError("validation-root partitions are not canonically ordered")

    expected_partition_files = {
        path
        for partition in manifest.partitions
        for path in (
            partition.manifest.path,
            str(Path(partition.manifest.path).with_name("validation.csv")),
            str(Path(partition.manifest.path).with_name("metrics.csv")),
        )
    }
    actual_partition_files = {
        path.relative_to(validation_root).as_posix()
        for level in EvaluationLevel
        for level_root in [validation_root / f"{level.value}_level"]
        if level_root.exists()
        for path in level_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_partition_files != expected_partition_files:
        raise ValueError(
            "validation leaf inventory differs from the root manifest: "
            f"missing={sorted(expected_partition_files - actual_partition_files)} "
            f"untracked={sorted(actual_partition_files - expected_partition_files)}"
        )

    validation_rows = []
    metric_rows = []
    for partition in manifest.partitions:
        leaf_rows, leaf_metrics = _read_validation_leaf(validation_root, partition)
        validation_rows.extend(leaf_rows)
        metric_rows.extend(leaf_metrics)

    _reject_development_rows(validation_rows)
    _validate_validation_row_identity(validation_rows)
    if validation_rows != _sorted_validation_rows(validation_rows):
        raise ValueError("validation archive row order is not canonical")
    computed_metrics = compute_metric_rows(validation_rows)
    if [_metric_signature(row) for row in metric_rows] != [
        _metric_signature(row) for row in computed_metrics
    ]:
        raise ValueError("validation archive metric order is not canonical")
    if manifest.rows != len(validation_rows):
        raise ValueError("validation-root row count drifted")
    if manifest.metric_rows != len(metric_rows):
        raise ValueError("validation-root metric-row count drifted")

    index_rows = list(_read_csv(index_path, ValidationIndexRow))
    if manifest.index.rows != len(index_rows):
        raise ValueError("validation-root index row count drifted")
    expected_index = compute_index_rows(validation_rows, computed_metrics)
    if [row.model_dump(mode="json") for row in index_rows] != [
        row.model_dump(mode="json") for row in expected_index
    ]:
        raise ValueError("index.csv does not match the canonical validation rows")
    if readme_path.read_text(encoding="utf-8") != render_validation_readme(index_rows):
        raise ValueError("validation README is not generated from index.csv")
    return validation_rows, metric_rows, index_rows


def _stored_prompt_matches_row(
    contract: StoredPredictionPromptContract, row: ValidationRow
) -> bool:
    """Return whether a stored contract fully covers one row's judge identity."""
    model_matches = any(
        model in {part.strip() for part in (row.judge_model or "").split("|")}
        for model in contract.judge_models
    )
    return (
        contract.prompt_version == row.prompt_version
        and contract.rubric_version == row.rubric_version
        and row.language in contract.languages
        and row.measure_id in contract.measure_ids
        and model_matches
    )


def _validate_stored_prompt_contracts(
    root: Path, prompts: PromptArchive, rows: list[ValidationRow]
) -> None:
    """Require every stored judge version to have one exact archived contract."""
    contracts = {
        contract.contract_id: contract for contract in prompts.stored_prediction_prompts
    }
    if len(contracts) != len(prompts.stored_prediction_prompts):
        raise ValueError("stored prompt contract ids must be unique")
    expected_ids = {
        "nativeness-v15-rubric-v19",
        "naturalness-v16-rubric-v20",
        "hi-gender-v16-rubric-v26",
        "quality-v7-rubric-v4",
        "speech-fidelity-v5",
    }
    if set(contracts) != expected_ids:
        raise ValueError("stored prompt contract inventory differs from the archive")

    for contract_id, (
        revision,
        source_files,
    ) in FROZEN_PROMPT_SOURCE_CONTRACTS.items():
        contract = contracts[contract_id]
        observed = {item.path: item.sha256 for item in contract.source_files}
        if contract.source_revision != revision or observed != source_files:
            raise ValueError(f"{contract_id}: frozen source contract drifted")

    validation_root = root / "validations"
    for contract in contracts.values():
        for artifact in contract.package_prompt_files:
            package_path = _resolve_archive_path(validation_root, artifact.path)
            if sha256_file(package_path) != artifact.sha256:
                raise ValueError(
                    f"{contract.contract_id}: exact package prompt hash drifted"
                )

    for row in rows:
        if not row.prompt_version:
            if row.judge_model != "deterministic":
                raise ValueError(f"{row.row_id}: judge prompt version is missing")
            continue
        matches = [
            contract
            for contract in contracts.values()
            if _stored_prompt_matches_row(contract, row)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{row.row_id}: expected one stored prompt contract, found {len(matches)}"
            )
        contract = matches[0]
        if contract.prompt_family == "nativeness":
            criteria = {
                criterion.factor_id
                for criterion in contract.nativeness_criteria.get(row.language, [])
            }
            required = (
                {"natural_word_choice"}
                if row.measure_id is MeasureId.NATURALNESS
                else set(row.source_factor_ids)
            )
            missing = required - criteria
            if missing:
                raise ValueError(
                    f"{row.row_id}: stored nativeness criteria missing {sorted(missing)}"
                )
        elif contract.prompt_family == "quality":
            criteria = {item.criterion_id for item in contract.quality_criteria}
            missing = set(row.source_factor_ids) - criteria
            if missing:
                raise ValueError(
                    f"{row.row_id}: stored quality criteria missing {sorted(missing)}"
                )

    pt_contract = contracts["nativeness-v15-rubric-v19"]
    pt_gender = {
        criterion.factor_id
        for criterion in pt_contract.nativeness_criteria.get("pt", [])
    }
    if "gender_agreement" not in pt_gender or not pt_contract.scope_note:
        raise ValueError("stored PT gender prompt scope is incomplete")


def validate_archive(root: Path) -> ArchiveValidationReport:
    """Validate hashes, row schemas, coverage, FCE completeness, and metrics."""
    root = root.resolve()
    manifest_path = _resolve_archive_path(root, "manifest.json")
    manifest = ArchiveManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.validation_languages != list(VALIDATION_LANGUAGES):
        raise ValueError(
            "manifest validation-language order does not match the archive contract"
        )
    if manifest.fce_languages != list(FCE_LANGUAGES):
        raise ValueError(
            "manifest FCE-language order does not match the archive contract"
        )
    if manifest.supporting_review_languages != list(ARCHIVE_LANGUAGES):
        raise ValueError(
            "manifest supporting-review languages do not match the archive contract"
        )
    expected_measures = {
        language: list(EXPECTED_MEASURES[language]) for language in VALIDATION_LANGUAGES
    }
    if manifest.expected_measures != expected_measures:
        raise ValueError("manifest measure matrix does not match the archive contract")
    expected_deterministic = {
        language: list(measures)
        for language, measures in DETERMINISTIC_MEASURES.items()
    }
    if manifest.deterministic_measures != expected_deterministic:
        raise ValueError("manifest deterministic-measure matrix drifted")

    disk_entries = list(root.rglob("*"))
    symlink_paths = [
        path.relative_to(root).as_posix() for path in disk_entries if path.is_symlink()
    ]
    if symlink_paths:
        raise ValueError(f"archive may not contain symlinks: {sorted(symlink_paths)}")
    manifest_paths = {record.path for record in manifest.files}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in disk_entries
        if path.is_file() and path != manifest_path
    }
    if manifest_paths != actual_paths:
        raise ValueError(
            "manifest file inventory differs from disk: "
            f"missing={sorted(manifest_paths - actual_paths)} "
            f"untracked={sorted(actual_paths - manifest_paths)}"
        )

    for record in manifest.files:
        path = _resolve_archive_path(root, record.path)
        if not path.is_file():
            raise ValueError(f"missing archive file: {record.path}")
        if path.stat().st_size != record.bytes:
            raise ValueError(f"byte-size mismatch: {record.path}")
        if sha256_file(path) != record.sha256:
            raise ValueError(f"SHA-256 mismatch: {record.path}")

    validation_root = root / "validations"
    legacy_paths = [validation_root / "validation.csv", validation_root / "metrics.csv"]
    if any(path.exists() for path in legacy_paths):
        raise ValueError("flat validation CSVs are not part of the nested archive")
    validation_rows, metric_rows, index_rows = read_validation_archive(root)
    call_rows = [
        row for row in validation_rows if row.evaluation_level is EvaluationLevel.CALL
    ]
    utterance_rows = [
        row
        for row in validation_rows
        if row.evaluation_level is EvaluationLevel.UTTERANCE
    ]
    expected_row_counts = {
        EvaluationLevel.CALL: len(call_rows),
        EvaluationLevel.UTTERANCE: len(utterance_rows),
    }
    if manifest.validation_rows != expected_row_counts:
        raise ValueError("validation row counts differ from manifest")
    fce_rows = list(
        _read_csv(root / "user_sim_review" / "fce" / "fce.csv", FceReviewRow)
    )
    if len(fce_rows) != manifest.fce_rows:
        raise ValueError("FCE row count differs from manifest")
    if not all(row.completed for row in fce_rows):
        raise ValueError("FCE archive contains an incomplete review row")
    language_counts = Counter(row.language for row in fce_rows)
    if language_counts != Counter({language: 30 for language in FCE_LANGUAGES}):
        raise ValueError(
            f"FCE rows must contain 30 calls per language: {language_counts}"
        )
    fce_keys = [(row.language, row.simulation_id) for row in fce_rows]
    if len(fce_keys) != len(set(fce_keys)):
        raise ValueError("FCE archive contains duplicate simulation rows")
    quality_ratings = [
        value
        for row in fce_rows
        for field in FCE_QUALITY_FIELDS
        if (value := getattr(row, field)) is not None
    ]
    backchannel_ratings = [
        row.backchannel_naturalness
        for row in fce_rows
        if row.backchannel_naturalness is not None
    ]

    expected_coverage = _expected_measure_coverage(validation_rows)
    if manifest.measure_coverage != expected_coverage:
        raise ValueError("manifest measure coverage does not match validation rows")
    missing_measures = [
        f"{row.language}:{row.measure_id.value}"
        for row in expected_coverage
        if row.human_rows == 0
    ]
    if missing_measures:
        raise ValueError(f"archive is missing retained measures: {missing_measures}")

    prompts = PromptArchive.model_validate_json(
        (root / "validations" / "prompts.json").read_text(encoding="utf-8")
    )
    if prompts.languages != list(VALIDATION_LANGUAGES):
        raise ValueError("prompts.json language order differs from the archive")
    expected_prompt_factors = {
        language: list(factors) for language, factors in PROMPT_FACTOR_MATRIX.items()
    }
    if prompts.retained_factors != expected_prompt_factors:
        raise ValueError("prompts.json factor matrix differs from the archive")
    naturalness_languages = {
        language
        for language, measures in EXPECTED_MEASURES.items()
        if MeasureId.NATURALNESS in measures
    }
    if set(prompts.combined_naturalness) != naturalness_languages:
        raise ValueError("combined-naturalness prompt coverage differs")
    expected_email_languages = {
        language
        for language, factors in DETERMINISTIC_MEASURES.items()
        if "email_symbol_verbalization" in factors
    }
    if set(prompts.email_symbols) != expected_email_languages:
        raise ValueError(
            "email-symbol contracts differ from the retained factor matrix"
        )
    _validate_stored_prompt_contracts(root, prompts, validation_rows)

    language_review_rows = list(
        _read_csv(root / "misc" / "language_review.csv", LanguageReviewRow)
    )
    call_experience_rows = list(
        _read_csv(root / "misc" / "call_experience.csv", CallExperienceReviewRow)
    )
    review_ids = [row.row_id for row in [*language_review_rows, *call_experience_rows]]
    duplicates = [key for key, count in Counter(review_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate supporting-review row ids: {duplicates[:5]}")
    if {row.language for row in language_review_rows} != set(ARCHIVE_LANGUAGES):
        raise ValueError("language-pack reviews must cover all five languages")

    deterministic_measure_ids = [
        f"{language}:{measure}"
        for language in VALIDATION_LANGUAGES
        for measure in DETERMINISTIC_MEASURES[language]
    ]
    return ArchiveValidationReport(
        root=root,
        call_validation_rows=len(call_rows),
        utterance_validation_rows=len(utterance_rows),
        fce_rows=len(fce_rows),
        language_review_rows=len(language_review_rows),
        call_experience_rows=len(call_experience_rows),
        metric_rows=len(metric_rows),
        files_verified=len(manifest.files),
        deterministic_measures=deterministic_measure_ids,
        fce_recorded_errors=sum(row.error_source is not None for row in fce_rows),
        fce_user_simulator_errors=sum(row.error_source == "user" for row in fce_rows),
        fce_infrastructure_errors=sum(
            row.error_source == "infrastructure" for row in fce_rows
        ),
        fce_quality_ratings=len(quality_ratings),
        fce_quality_mean=_mean(quality_ratings),
        fce_backchannel_ratings=len(backchannel_ratings),
        fce_backchannel_mean=_mean(backchannel_ratings),
    )
