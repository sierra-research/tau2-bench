# Copyright Sierra
"""Frozen Hindi gender-agreement utterance factor test.

The package is intentionally self-contained: it freezes the human label, exact
text/context, participant frame, interruption state, task literals, and source
provenance for every row. Validation and evaluation never reopen mutable source
runs. Evaluation exercises the production hybrid factor path with only
``gender_agreement`` in the batch.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.config import (
    DEFAULT_TAU_MULTI_HI_GENDER_JUDGE,
    DEFAULT_TAU_MULTI_HI_GENDER_JUDGE_ARGS,
)
from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessJudgeSettings,
)
from tau2.judges.base import INTERRUPTED_TURN_MARKER
from tau2.judges.nativeness import checkers as checker_module
from tau2.judges.nativeness.checkers import (
    GENDER_AGREEMENT_CHECKER_VERSION,
    CheckerContext,
    _without_allowed_literals,
    check_hi_gender_agreement,
)
from tau2.judges.nativeness.factors import judge_factors_for
from tau2.judges.nativeness.harness import (
    NATIVENESS_RUBRIC_VERSION,
    AgentTurn,
    evaluate_pack_judge_factors,
)
from tau2.judges.nativeness.judge import (
    NATIVENESS_JUDGE_PROMPT_VERSION,
    NATIVENESS_JUDGE_SYSTEM_PROMPT,
    NativenessJudgeCriterion,
    NativenessJudgeInput,
    build_user_prompt,
)

DATASET_SCHEMA_VERSION = "tau-multi-hi-gender-agreement-dataset-v3"
PROMPT_SCHEMA_VERSION = "tau-multi-factor-validation-prompt-v1"
MANIFEST_SCHEMA_VERSION = "tau-multi-factor-validation-manifest-v3"
RESULTS_SCHEMA_VERSION = "tau-multi-factor-validation-results-v1"
CACHE_SCHEMA_VERSION = "tau-multi-factor-validation-cache-row-v1"
VALIDATION_NAME = "hi_gender_agreement"
FACTOR_ID = "gender_agreement"
LANGUAGE = "hi"
EXPECTED_PROVIDERS = ("gemini", "openai", "xai")
USER_PROMPT_AGENT_CONTEXT_TOKEN = "<ROW_AGENT_CONTEXT>"
USER_PROMPT_CUSTOMER_CONTEXT_TOKEN = "<ROW_CUSTOMER_CONTEXT>"
USER_PROMPT_AGENT_SPEECH_TOKEN = "<ROW_AGENT_SPEECH_WITH_INTERRUPTION_MARKER>"


class ValidationLabel(str, Enum):
    """Human binary label; violation is the positive class."""

    VIOLATION = "violation"
    PASS = "pass"


class AlignmentKind(str, Enum):
    """Identity authority for one exact utterance unit."""

    AGENT_TURN = "agent_turn"
    AUDIO_SEGMENT = "audio_segment"


class InterruptionSource(str, Enum):
    """Authority used to freeze the interruption flag."""

    SIMULATION_TICKS = "simulation_ticks"
    AUDIO_LABEL_TIMELINE = "audio_label_timeline"


class FileReference(BaseModel):
    """Artifact path plus exact byte digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Annotated[str, Field(min_length=1, description="Artifact path.")]
    sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Artifact SHA-256.")
    ]


class HiGenderDatasetRow(BaseModel):
    """One exact reviewed utterance and all context needed for cold replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: Annotated[int, Field(ge=0, description="Zero-based row order.")]
    row_id: Annotated[
        str,
        Field(
            pattern=r"^hi-gender-test-[0-9a-f]{16}$",
            description="Stable provenance-derived row id.",
        ),
    ]
    simulation_id: Annotated[str, Field(min_length=1, description="Source call id.")]
    task_id: Annotated[str, Field(min_length=1, description="Localized task id.")]
    canonical_task_id: Annotated[
        str, Field(min_length=1, description="Language-independent task key.")
    ]
    domain: Literal["airline", "retail", "telecom"]
    experiment: Annotated[
        str, Field(min_length=1, description="Source main-run experiment/cell.")
    ]
    provider: Literal["gemini", "openai", "xai"]
    agent_gender: Literal["male"]
    caller_gender: Literal["male", "female"]
    preceding_customer_text: Annotated[
        Optional[str], Field(description="Immediately preceding reviewed caller text.")
    ] = None
    agent_text: Annotated[
        str, Field(min_length=1, description="Exact reviewed agent text unit.")
    ]
    source_interrupted: Annotated[
        bool, Field(description="Whether the caller cut this source unit off.")
    ]
    interruption_source: InterruptionSource
    allowed_literals: Annotated[
        list[str],
        Field(description="Exact sorted task literals removed before hybrid precheck."),
    ]
    precheck_text: Annotated[
        str,
        Field(description="Exact agent text after removing frozen allowed_literals."),
    ]
    label: ValidationLabel
    alignment_kind: AlignmentKind
    source_path: Annotated[
        str,
        Field(
            min_length=1,
            description="External source identifier; not a package-local file path.",
        ),
    ]
    source_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="External source digest.")
    ]
    source_line_number: Annotated[
        Optional[int], Field(default=None, ge=1, description="One-based source line.")
    ]
    source_agent_turn_index: Annotated[
        Optional[int],
        Field(default=None, ge=0, description="Zero-based delivered AgentTurn index."),
    ]
    source_start_seconds: Annotated[
        Optional[float],
        Field(default=None, ge=0, description="Audio-label interval start."),
    ]
    source_end_seconds: Annotated[
        Optional[float],
        Field(default=None, gt=0, description="Audio-label interval end."),
    ]

    @model_validator(mode="after")
    def _provenance_is_complete(self) -> "HiGenderDatasetRow":
        if self.allowed_literals != sorted(set(self.allowed_literals)):
            raise ValueError("allowed_literals must be sorted and unique")
        expected_precheck = _without_allowed_literals(
            self.agent_text, self.allowed_literals
        )
        if self.precheck_text != expected_precheck:
            raise ValueError("precheck_text does not match frozen allowed_literals")
        times = (self.source_start_seconds, self.source_end_seconds)
        if self.alignment_kind is AlignmentKind.AUDIO_SEGMENT:
            if self.source_line_number is None or None in times:
                raise ValueError("audio segments need a hashed line and time interval")
            if self.source_agent_turn_index is not None:
                raise ValueError("audio segments do not claim one AgentTurn index")
            assert self.source_start_seconds is not None
            assert self.source_end_seconds is not None
            if self.source_end_seconds <= self.source_start_seconds:
                raise ValueError("audio-segment interval must have positive duration")
        elif any(value is not None for value in times):
            raise ValueError("only audio segments carry time intervals")
        if self.alignment_kind is AlignmentKind.AGENT_TURN:
            if self.source_agent_turn_index is None:
                raise ValueError("agent-turn rows need a delivered AgentTurn index")
        return self


class HiGenderDataset(BaseModel):
    """Canonical 60-row Hindi gender-agreement factor-test dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-hi-gender-agreement-dataset-v3"]
    benchmark: Literal["tau-multi"]
    language: Literal["hi"]
    split: Literal["test"]
    role: Literal["fixed_curated_validation"]
    evaluation_level: Literal["utterance"]
    source_reference_policy: Literal["external_hash_only"]
    factor_id: Literal["gender_agreement"]
    agent_gender: Literal["male"]
    agent_gender_policy: Literal["fixed_counterfactual_human_frame"]
    n_rows: Annotated[int, Field(ge=0, description="Total rows.")]
    n_violation: Annotated[int, Field(ge=0, description="Positive rows.")]
    n_pass: Annotated[int, Field(ge=0, description="Negative rows.")]
    n_calls: Annotated[int, Field(ge=0, description="Unique source simulations.")]
    n_task_groups: Annotated[int, Field(ge=0, description="Unique canonical tasks.")]
    provider_counts: Annotated[dict[str, int], Field(description="Rows/provider.")]
    domain_counts: Annotated[dict[str, int], Field(description="Rows/domain.")]
    label_provider_counts: Annotated[
        dict[str, dict[str, int]], Field(description="Rows by label/provider.")
    ]
    interruption_counts: Annotated[
        dict[str, int], Field(description="Interrupted/not-interrupted counts.")
    ]
    rows: Annotated[list[HiGenderDatasetRow], Field(description="Frozen rows.")]

    @model_validator(mode="after")
    def _summaries_match_rows(self) -> "HiGenderDataset":
        if [row.row_index for row in self.rows] != list(range(len(self.rows))):
            raise ValueError("row_index values must be contiguous and ordered")
        if len({row.row_id for row in self.rows}) != len(self.rows):
            raise ValueError("row_id values must be unique")
        expected = {
            "n_rows": len(self.rows),
            "n_violation": sum(
                row.label is ValidationLabel.VIOLATION for row in self.rows
            ),
            "n_pass": sum(row.label is ValidationLabel.PASS for row in self.rows),
            "n_calls": len({row.simulation_id for row in self.rows}),
            "n_task_groups": len({row.canonical_task_id for row in self.rows}),
        }
        if {name: getattr(self, name) for name in expected} != expected:
            raise ValueError("dataset scalar summaries do not match rows")
        if self.provider_counts != _counts(row.provider for row in self.rows):
            raise ValueError("provider_counts do not match rows")
        if self.domain_counts != _counts(row.domain for row in self.rows):
            raise ValueError("domain_counts do not match rows")
        label_provider = {
            label.value: _counts(
                row.provider for row in self.rows if row.label is label
            )
            for label in ValidationLabel
        }
        if self.label_provider_counts != label_provider:
            raise ValueError("label_provider_counts do not match rows")
        interruptions = {
            "interrupted": sum(row.source_interrupted for row in self.rows),
            "not_interrupted": sum(not row.source_interrupted for row in self.rows),
        }
        if self.interruption_counts != interruptions:
            raise ValueError("interruption_counts do not match rows")
        return self


class FactorContract(BaseModel):
    """Frozen production factor wiring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Literal["gender_agreement"]
    category: Literal["gender"]
    severity: Literal[3]
    evaluation_level: Literal["utterance"]
    aggregation: Literal["any"]
    precheck_type: Literal["hi_gender_agreement"]


class ValidationExecution(BaseModel):
    """Factor-isolated test execution contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit: Literal["one_row_one_utterance"]
    factor_batch: list[Literal["gender_agreement"]]
    precheck_scope: Literal["row"]
    positive_class: Literal["violation"]
    production_unresolved_batch_peers: list[Literal["name_address_conventions"]]


class ParticipantFrame(BaseModel):
    """Frozen participant and interruption inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_gender_mode: Literal["fixed"]
    agent_gender: Literal["male"]
    caller_gender_mode: Literal["row_field"]
    caller_gender_field: Literal["caller_gender"]
    interruption_mode: Literal["row_field"]
    interruption_field: Literal["source_interrupted"]
    interruption_marker: Annotated[str, Field(min_length=1)]


class VersionContract(BaseModel):
    """Current test-runtime versions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gender_checker: Literal["v3"]
    judge_prompt: Literal["v16"]
    runtime_rubric: Literal["nativeness-rubric-v26"]


class JudgeContract(BaseModel):
    """Exact LLM configuration and system prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: Annotated[str, Field(min_length=1)]
    model_args: Annotated[dict, Field(description="Effective model arguments.")]
    call_name: Literal["nativeness_judge_batch"]
    system_prompt: Annotated[str, Field(min_length=1)]


class PrecheckPattern(BaseModel):
    """One ordered deterministic v3 pattern."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Annotated[str, Field(min_length=1)]
    regex: Annotated[str, Field(min_length=1)]


class PrecheckContract(BaseModel):
    """Frozen hybrid precheck behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    behavior: Literal["fail_short_circuits_no_opportunity_falls_through"]
    input_text_field: Literal["precheck_text"]
    allowed_literals_field: Literal["allowed_literals"]
    ordered_patterns: list[PrecheckPattern]


class LlmFallbackContract(BaseModel):
    """Exact factor-isolated criterion, contexts, and rendered wrapper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion: NativenessJudgeCriterion
    agent_context_templates: dict[Literal["male", "female"], str]
    user_prompt_template: Annotated[str, Field(min_length=1)]


class BinaryMetrics(BaseModel):
    """Binary confusion matrix and derived scores."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tp: Annotated[int, Field(ge=0)]
    fn: Annotated[int, Field(ge=0)]
    fp: Annotated[int, Field(ge=0)]
    tn: Annotated[int, Field(ge=0)]
    n_errors: Annotated[int, Field(ge=0)]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    kappa: Optional[float]


class HiGenderPrompt(BaseModel):
    """Complete frozen prompt and hybrid-factor execution contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-factor-validation-prompt-v1"]
    language: Literal["hi"]
    factor: FactorContract
    validation_execution: ValidationExecution
    participant_frame: ParticipantFrame
    versions: VersionContract
    judge: JudgeContract
    precheck: PrecheckContract
    llm_fallback: LlmFallbackContract


class DatasetSummary(BaseModel):
    """Manifest projection of the test dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Literal["test/dataset.json"]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    n_rows: Literal[60]
    n_violation: Literal[30]
    n_pass: Literal[30]
    n_calls: Annotated[int, Field(ge=1)]
    n_task_groups: Annotated[int, Field(ge=1)]
    provider_counts: dict[str, int]
    domain_counts: dict[str, int]
    interruption_counts: dict[str, int]


class ResultsSummary(BaseModel):
    """Manifest projection of the sealed test predictions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Literal["test/results.json"]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    n_rows: Literal[60]
    n_llm_rows: Annotated[int, Field(ge=0)]
    n_precheck_short_circuits: Annotated[int, Field(ge=0)]
    max_concurrency: Literal[60]
    metrics: BinaryMetrics


class HiGenderManifest(BaseModel):
    """Root manifest for the canonical Hindi factor-test package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-factor-validation-manifest-v3"]
    benchmark: Literal["tau-multi"]
    validation: Literal["hi_gender_agreement"]
    language: Literal["hi"]
    factor_id: Literal["gender_agreement"]
    evaluation_level: Literal["utterance"]
    split_roles: dict[Literal["test"], Literal["fixed_curated_validation"]]
    agent_gender: Literal["male"]
    agent_gender_policy: Literal["fixed_counterfactual_human_frame"]
    validation_design: Literal["fixed_curated"]
    prompt: FileReference
    test: DatasetSummary
    results: ResultsSummary
    files: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]


class HiGenderResultRow(BaseModel):
    """One cached/evaluated row prediction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: Annotated[int, Field(ge=0)]
    row_id: Annotated[str, Field(min_length=1)]
    label: ValidationLabel
    outcome: JudgeOutcome
    judge_positive: bool
    judge_ran: bool
    factor_check: Optional[NativenessFactorCheck] = None
    error: Optional[str] = None

    @model_validator(mode="after")
    def _state_is_consistent(self) -> "HiGenderResultRow":
        if self.outcome is JudgeOutcome.ERROR:
            if not self.error or self.factor_check is not None:
                raise ValueError("ERROR rows need only an error message")
        else:
            if self.error is not None or self.factor_check is None:
                raise ValueError("non-ERROR rows need exactly one factor check")
            if self.factor_check.outcome is not self.outcome:
                raise ValueError("factor check and row outcome differ")
        if self.judge_positive != (self.outcome is JudgeOutcome.FAIL):
            raise ValueError("judge_positive must mean FAIL")
        return self


class HiGenderCacheRow(BaseModel):
    """Incremental cache record pinned to both immutable inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-factor-validation-cache-row-v1"]
    dataset_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    prompt_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    result: HiGenderResultRow


class HiGenderResults(BaseModel):
    """Row-aligned factor-test results and recomputed aggregate metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-factor-validation-results-v1"]
    dataset_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    prompt_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    factor_id: Literal["gender_agreement"]
    agent_gender: Literal["male"]
    model: str
    model_args: dict
    max_concurrency: Literal[60]
    n_rows: Literal[60]
    n_llm_rows: Annotated[int, Field(ge=0)]
    n_precheck_short_circuits: Annotated[int, Field(ge=0)]
    metrics: BinaryMetrics
    rows: list[HiGenderResultRow]

    @model_validator(mode="after")
    def _summaries_match_rows(self) -> "HiGenderResults":
        if len(self.rows) != 60:
            raise ValueError("results must contain exactly 60 rows")
        if [row.row_index for row in self.rows] != list(range(60)):
            raise ValueError("result rows must be contiguous and ordered")
        if len({row.row_id for row in self.rows}) != 60:
            raise ValueError("result row ids must be unique")
        for row in self.rows:
            if row.factor_check is not None and (
                row.factor_check.id != FACTOR_ID
                or row.factor_check.evaluation_level != "utterance"
            ):
                raise ValueError("result contains a non-gender/non-utterance check")
        if self.n_llm_rows != sum(row.judge_ran for row in self.rows):
            raise ValueError("n_llm_rows does not match rows")
        expected_prechecks = sum(
            not row.judge_ran and row.outcome is not JudgeOutcome.ERROR
            for row in self.rows
        )
        if self.n_precheck_short_circuits != expected_prechecks:
            raise ValueError("n_precheck_short_circuits does not match rows")
        _assert_metrics_equal(self.metrics, _metrics(self.rows))
        return self


class HiGenderValidationReport(BaseModel):
    """Offline package verification report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: str
    dataset_sha256: str
    prompt_sha256: str
    rows_verified: Literal[60]
    pass_rows_clear_of_deterministic_fail: Literal[30]
    results_verified: bool


class HiGenderEvaluationReport(BaseModel):
    """One incremental evaluator invocation summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output: str
    cache_root: str
    rows: Literal[60]
    cache_hits: Annotated[int, Field(ge=0)]
    evaluated_now: Annotated[int, Field(ge=0)]
    llm_calls_now: Annotated[int, Field(ge=0)]
    errors: Annotated[int, Field(ge=0)]
    metrics: BinaryMetrics


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_male_agent_context(caller_gender: Literal["male", "female"]) -> str:
    """Render the exact counterfactual male-agent context for one row."""
    return (
        "The speaker is an AI customer-service voice agent. "
        "Agent gender: male. Use this for the agent's first-person gender "
        "agreement. "
        f"Caller gender: {caller_gender}. Use this only when the agent directly "
        "addresses or refers to the caller with gender-marked language. "
        "Do not infer any missing gender from a name or stereotype."
    )


def runtime_gender_factor():
    """Resolve the sole enabled production Hindi gender factor."""
    matches = [
        factor
        for factor in judge_factors_for(LANGUAGE)
        if factor.id == FACTOR_ID and factor.enabled
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one enabled Hindi gender_agreement factor")
    factor = matches[0]
    if (
        factor.type != "judge"
        or factor.category != "gender"
        or factor.severity != 3
        or factor.evaluation_level != "utterance"
        or factor.aggregation != "any"
        or factor.precheck_type != "hi_gender_agreement"
        or factor.params is None
    ):
        raise ValueError("Hindi gender_agreement runtime wiring drifted")
    return factor


def runtime_prompt() -> HiGenderPrompt:
    """Project the production hybrid factor into the frozen prompt contract."""
    factor = runtime_gender_factor()
    assert factor.params is not None
    criterion = NativenessJudgeCriterion.from_rubric(factor.id, factor.params)
    user_prompt_template = build_user_prompt(
        NativenessJudgeInput(
            language=LANGUAGE,
            evaluation_level="utterance",
            criteria=[criterion],
            agent_text=USER_PROMPT_AGENT_SPEECH_TOKEN,
            customer_context=USER_PROMPT_CUSTOMER_CONTEXT_TOKEN,
            agent_context=USER_PROMPT_AGENT_CONTEXT_TOKEN,
        )
    )
    return HiGenderPrompt(
        schema_version=PROMPT_SCHEMA_VERSION,
        language=LANGUAGE,
        factor=FactorContract(
            id=FACTOR_ID,
            category="gender",
            severity=3,
            evaluation_level="utterance",
            aggregation="any",
            precheck_type="hi_gender_agreement",
        ),
        validation_execution=ValidationExecution(
            unit="one_row_one_utterance",
            factor_batch=[FACTOR_ID],
            precheck_scope="row",
            positive_class="violation",
            production_unresolved_batch_peers=["name_address_conventions"],
        ),
        participant_frame=ParticipantFrame(
            agent_gender_mode="fixed",
            agent_gender="male",
            caller_gender_mode="row_field",
            caller_gender_field="caller_gender",
            interruption_mode="row_field",
            interruption_field="source_interrupted",
            interruption_marker=INTERRUPTED_TURN_MARKER,
        ),
        versions=VersionContract(
            gender_checker=GENDER_AGREEMENT_CHECKER_VERSION,
            judge_prompt=NATIVENESS_JUDGE_PROMPT_VERSION,
            runtime_rubric=NATIVENESS_RUBRIC_VERSION,
        ),
        judge=JudgeContract(
            model=DEFAULT_TAU_MULTI_HI_GENDER_JUDGE,
            model_args=DEFAULT_TAU_MULTI_HI_GENDER_JUDGE_ARGS,
            call_name="nativeness_judge_batch",
            system_prompt=NATIVENESS_JUDGE_SYSTEM_PROMPT,
        ),
        precheck=PrecheckContract(
            behavior="fail_short_circuits_no_opportunity_falls_through",
            input_text_field="precheck_text",
            allowed_literals_field="allowed_literals",
            ordered_patterns=[
                PrecheckPattern(
                    id="female_agent_masculine_self_reference",
                    regex=checker_module._HI_AGENT_MASCULINE_RE.pattern,
                ),
                PrecheckPattern(
                    id="male_agent_feminine_self_reference",
                    regex=checker_module._HI_AGENT_FEMININE_RE.pattern,
                ),
                PrecheckPattern(
                    id="male_caller_feminine_address",
                    regex=checker_module._HI_CALLER_FEMININE_RE.pattern,
                ),
                PrecheckPattern(
                    id="spoken_slash_alternative",
                    regex=checker_module._HI_SLASH_ALTERNATIVE_RE.pattern,
                ),
            ],
        ),
        llm_fallback=LlmFallbackContract(
            criterion=criterion,
            agent_context_templates={
                gender: fixed_male_agent_context(gender)
                for gender in ("male", "female")
            },
            user_prompt_template=user_prompt_template,
        ),
    )


def _checker_context(row: HiGenderDatasetRow) -> CheckerContext:
    return CheckerContext(
        agent_text=row.agent_text,
        agent_turns=[row.agent_text],
        agent_turn_interruptions=[row.source_interrupted],
        user_turns=[row.preceding_customer_text] if row.preceding_customer_text else [],
        has_email=False,
        language=LANGUAGE,
        allowed_literals=row.allowed_literals,
        agent_gender="male",
        caller_gender=row.caller_gender,
    )


def _assert_canonical_dataset(dataset: HiGenderDataset) -> None:
    if (dataset.n_rows, dataset.n_violation, dataset.n_pass) != (60, 30, 30):
        raise ValueError("Hindi gender test must contain 30 violation + 30 pass")
    expected_balance = {provider: 10 for provider in EXPECTED_PROVIDERS}
    if dataset.label_provider_counts != {
        "violation": expected_balance,
        "pass": expected_balance,
    }:
        raise ValueError("each label must contain 10 Gemini/10 OpenAI/10 xAI")
    alignments = _counts(row.alignment_kind.value for row in dataset.rows)
    if alignments != {"agent_turn": 24, "audio_segment": 36}:
        raise ValueError("utterance alignment composition drifted")


def _assert_deterministic_alignment(dataset: HiGenderDataset) -> None:
    pass_fails = []
    for row in dataset.rows:
        outcome, _evidence = check_hi_gender_agreement(_checker_context(row))
        if row.label is ValidationLabel.PASS and outcome is JudgeOutcome.FAIL:
            pass_fails.append(row.row_id)
    if pass_fails:
        raise ValueError(f"human PASS rows fail deterministic v3: {pass_fails}")


def _resolve_inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes package: {relative!r}")
    return path


def validate_package(
    root: Path, *, validate_results_if_present: bool = True
) -> HiGenderValidationReport:
    """Validate hashes, typed contracts, prompt parity, and row invariants offline."""
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = HiGenderManifest.model_validate_json(manifest_path.read_text())
    expected_files = {
        "README.md",
        "prompt.json",
        "test/dataset.json",
        "test/results.json",
    }
    if set(manifest.files) != expected_files:
        raise ValueError(
            "manifest file inventory must include README, prompt, dataset, and results"
        )
    for relative, digest in manifest.files.items():
        if relative == "test/results.json" and not validate_results_if_present:
            continue
        if sha256_file(_resolve_inside(root, relative)) != digest:
            raise ValueError(f"artifact hash mismatch: {relative}")

    prompt_path = _resolve_inside(root, manifest.prompt.path)
    dataset_path = _resolve_inside(root, manifest.test.path)
    if prompt_path != root / "prompt.json":
        raise ValueError("prompt path is not canonical")
    if dataset_path != root / "test/dataset.json":
        raise ValueError("dataset path is not canonical")
    prompt_sha = sha256_file(prompt_path)
    dataset_sha = sha256_file(dataset_path)
    if (
        manifest.prompt.sha256 != prompt_sha
        or manifest.files["prompt.json"] != prompt_sha
    ):
        raise ValueError("prompt hashes disagree")
    if (
        manifest.test.sha256 != dataset_sha
        or manifest.files["test/dataset.json"] != dataset_sha
    ):
        raise ValueError("dataset hashes disagree")

    prompt = HiGenderPrompt.model_validate_json(prompt_path.read_text())
    if prompt != runtime_prompt():
        raise ValueError("packaged prompt differs from current production contract")
    dataset = HiGenderDataset.model_validate_json(dataset_path.read_text())
    _assert_canonical_dataset(dataset)
    _assert_deterministic_alignment(dataset)
    for field in (
        "n_rows",
        "n_violation",
        "n_pass",
        "n_calls",
        "n_task_groups",
        "provider_counts",
        "domain_counts",
        "interruption_counts",
    ):
        if getattr(manifest.test, field) != getattr(dataset, field):
            raise ValueError(f"manifest dataset summary mismatch: {field}")
    results_path = _resolve_inside(root, manifest.results.path)
    if results_path != root / "test/results.json":
        raise ValueError("results path is not canonical")
    results_verified = False
    if validate_results_if_present:
        results = _validate_results_file(
            results_path,
            dataset=dataset,
            prompt=prompt,
            dataset_sha256=dataset_sha,
            prompt_sha256=prompt_sha,
        )
        if manifest.results.sha256 != sha256_file(results_path):
            raise ValueError("manifest results hash disagrees")
        for field in (
            "n_rows",
            "n_llm_rows",
            "n_precheck_short_circuits",
            "max_concurrency",
            "metrics",
        ):
            if getattr(manifest.results, field) != getattr(results, field):
                raise ValueError(f"manifest results summary mismatch: {field}")
        results_verified = True
    return HiGenderValidationReport(
        root=str(root),
        dataset_sha256=dataset_sha,
        prompt_sha256=prompt_sha,
        rows_verified=60,
        pass_rows_clear_of_deterministic_fail=30,
        results_verified=results_verified,
    )


def _metrics(rows: list[HiGenderResultRow]) -> BinaryMetrics:
    tp = fn = fp = tn = errors = 0
    for row in rows:
        if row.outcome is JudgeOutcome.ERROR:
            errors += 1
        elif row.label is ValidationLabel.VIOLATION and row.judge_positive:
            tp += 1
        elif row.label is ValidationLabel.VIOLATION:
            fn += 1
        elif row.judge_positive:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    n = tp + fn + fp + tn
    if not n:
        kappa = None
    else:
        observed = (tp + tn) / n
        expected = (((tp + fn) * (tp + fp)) + ((fp + tn) * (fn + tn))) / (n * n)
        kappa = 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)
    return BinaryMetrics(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        n_errors=errors,
        precision=precision,
        recall=recall,
        f1=f1,
        kappa=kappa,
    )


def _assert_metrics_equal(observed: BinaryMetrics, expected: BinaryMetrics) -> None:
    for field in ("tp", "fn", "fp", "tn", "n_errors"):
        if getattr(observed, field) != getattr(expected, field):
            raise ValueError(f"result metric {field} does not reproduce")
    for field in ("precision", "recall", "f1", "kappa"):
        left = getattr(observed, field)
        right = getattr(expected, field)
        if left is None or right is None:
            if left is not right:
                raise ValueError(f"result metric {field} does not reproduce")
        elif not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"result metric {field} does not reproduce")


def _validate_results_file(
    path: Path,
    *,
    dataset: HiGenderDataset,
    prompt: HiGenderPrompt,
    dataset_sha256: str,
    prompt_sha256: str,
) -> HiGenderResults:
    results = HiGenderResults.model_validate_json(path.read_text())
    if results.dataset_sha256 != dataset_sha256:
        raise ValueError("results dataset hash does not match package")
    if results.prompt_sha256 != prompt_sha256:
        raise ValueError("results prompt hash does not match package")
    if (
        results.model != prompt.judge.model
        or results.model_args != prompt.judge.model_args
    ):
        raise ValueError("results judge settings do not match frozen prompt")
    if results.max_concurrency != 60 or results.n_rows != 60:
        raise ValueError(
            "results did not use the frozen 60-row/concurrency-60 contract"
        )
    if results.n_llm_rows != sum(row.judge_ran for row in results.rows):
        raise ValueError("results n_llm_rows does not reproduce")
    if results.n_precheck_short_circuits != sum(
        not row.judge_ran and row.outcome is not JudgeOutcome.ERROR
        for row in results.rows
    ):
        raise ValueError("results precheck count does not reproduce")
    _assert_metrics_equal(results.metrics, _metrics(results.rows))
    for source, result in zip(dataset.rows, results.rows, strict=True):
        if (
            result.row_index != source.row_index
            or result.row_id != source.row_id
            or result.label is not source.label
        ):
            raise ValueError("result rows do not align with dataset")
        if not result.judge_ran and result.outcome is not JudgeOutcome.ERROR:
            deterministic_outcome, _ = check_hi_gender_agreement(
                _checker_context(source)
            )
            if (
                deterministic_outcome is not JudgeOutcome.FAIL
                or result.outcome is not JudgeOutcome.FAIL
            ):
                raise ValueError(
                    f"{result.row_id}: non-LLM result is not a deterministic FAIL"
                )
    return results


def _atomic_json(path: Path, model: BaseModel) -> None:
    import os
    import tempfile

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


def _evaluate_row(
    row: HiGenderDatasetRow,
    *,
    factor,
    prompt: HiGenderPrompt,
) -> HiGenderResultRow:
    try:
        evaluation = evaluate_pack_judge_factors(
            [
                AgentTurn(
                    index=0,
                    text=row.agent_text,
                    preceding_user_text=row.preceding_customer_text,
                    interrupted=row.source_interrupted,
                )
            ],
            LANGUAGE,
            settings=NativenessJudgeSettings(
                llm_judge=True,
                model=prompt.judge.model,
                model_args=prompt.judge.model_args,
            ),
            agent_context=fixed_male_agent_context(row.caller_gender),
            checker_context=_checker_context(row),
            factors=[factor],
        )
        if len(evaluation.checks) != 1:
            raise ValueError("factor-isolated evaluation did not return one check")
        check = evaluation.checks[0]
        if check.outcome is JudgeOutcome.ERROR:
            return HiGenderResultRow(
                row_index=row.row_index,
                row_id=row.row_id,
                label=row.label,
                outcome=JudgeOutcome.ERROR,
                judge_positive=False,
                judge_ran=evaluation.judge_ran,
                error=check.evidence or "gender judge returned ERROR without evidence",
            )
        return HiGenderResultRow(
            row_index=row.row_index,
            row_id=row.row_id,
            label=row.label,
            outcome=check.outcome,
            judge_positive=check.outcome is JudgeOutcome.FAIL,
            judge_ran=evaluation.judge_ran,
            factor_check=check,
        )
    except Exception as exc:
        return HiGenderResultRow(
            row_index=row.row_index,
            row_id=row.row_id,
            label=row.label,
            outcome=JudgeOutcome.ERROR,
            judge_positive=False,
            judge_ran=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def evaluate_package(
    root: Path,
    *,
    max_concurrency: int,
    output: Path,
    cache_root: Optional[Path] = None,
) -> HiGenderEvaluationReport:
    """Replay the frozen rows without mutating their canonical package."""
    if max_concurrency != 60:
        raise ValueError("the frozen Hindi gender test requires max_concurrency=60")
    root = root.resolve()
    validation = validate_package(root, validate_results_if_present=False)
    dataset = HiGenderDataset.model_validate_json(
        (root / "test/dataset.json").read_text()
    )
    prompt = HiGenderPrompt.model_validate_json((root / "prompt.json").read_text())
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("replay output must be outside the frozen validation package")
    cache_root = (
        cache_root.resolve()
        if cache_root is not None
        else (output.parent / ".work" / validation.dataset_sha256).resolve()
    )
    if cache_root.is_relative_to(root):
        raise ValueError("replay cache must be outside the frozen validation package")
    factor = runtime_gender_factor()
    by_id: dict[str, HiGenderResultRow] = {}
    cache_hits = 0
    pending: list[HiGenderDatasetRow] = []
    for row in dataset.rows:
        cache_path = cache_root / f"{row.row_id}.json"
        if cache_path.is_file():
            try:
                cached = HiGenderCacheRow.model_validate_json(cache_path.read_text())
                if (
                    cached.dataset_sha256 == validation.dataset_sha256
                    and cached.prompt_sha256 == validation.prompt_sha256
                    and cached.result.row_id == row.row_id
                    and cached.result.row_index == row.row_index
                    and cached.result.label is row.label
                    and cached.result.outcome is not JudgeOutcome.ERROR
                ):
                    by_id[row.row_id] = cached.result
                    cache_hits += 1
                    continue
            except (OSError, ValueError):
                pass
        pending.append(row)

    llm_calls_now = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(_evaluate_row, row, factor=factor, prompt=prompt): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            result = future.result()
            by_id[row.row_id] = result
            llm_calls_now += int(result.judge_ran)
            _atomic_json(
                cache_root / f"{row.row_id}.json",
                HiGenderCacheRow(
                    schema_version=CACHE_SCHEMA_VERSION,
                    dataset_sha256=validation.dataset_sha256,
                    prompt_sha256=validation.prompt_sha256,
                    result=result,
                ),
            )

    ordered = [by_id[row.row_id] for row in dataset.rows]
    metrics = _metrics(ordered)
    results = HiGenderResults(
        schema_version=RESULTS_SCHEMA_VERSION,
        dataset_sha256=validation.dataset_sha256,
        prompt_sha256=validation.prompt_sha256,
        factor_id=FACTOR_ID,
        agent_gender="male",
        model=prompt.judge.model,
        model_args=prompt.judge.model_args,
        max_concurrency=max_concurrency,
        n_rows=60,
        n_llm_rows=sum(row.judge_ran for row in ordered),
        n_precheck_short_circuits=sum(
            not row.judge_ran and row.outcome is not JudgeOutcome.ERROR
            for row in ordered
        ),
        metrics=metrics,
        rows=ordered,
    )
    _atomic_json(output, results)
    _validate_results_file(
        output,
        dataset=dataset,
        prompt=prompt,
        dataset_sha256=validation.dataset_sha256,
        prompt_sha256=validation.prompt_sha256,
    )
    report = HiGenderEvaluationReport(
        output=str(output),
        cache_root=str(cache_root),
        rows=60,
        cache_hits=cache_hits,
        evaluated_now=len(pending),
        llm_calls_now=llm_calls_now,
        errors=metrics.n_errors,
        metrics=metrics,
    )
    if report.errors:
        raise RuntimeError(
            f"Hindi gender evaluation completed with {report.errors} ERROR rows; "
            f"partial results and retryable cache were preserved at {output}"
        )
    return report
