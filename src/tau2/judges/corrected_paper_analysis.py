# Copyright Sierra
"""Deterministic post-judge analyses for the corrected tau-Multilingual cohort.

The two artifacts in this module answer narrow paper questions that cannot be
recovered from aggregate scores alone:

* whether a retail call made an exact, task-grounded identity lookup before it
  ultimately succeeded; and
* which material fidelity categories account for failed delivery utterances.

Both builders are API-free.  They first reproduce the split-judge composition
and then bind every output to source hashes and exact denominators.  Existing
outputs are immutable: rerunning with identical inputs verifies and returns the
artifact, while input drift fails instead of overwriting it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import mmap
import re
import tempfile
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from tau2.data_model.simulation import (
    DeliveryInfo,
    JudgeOutcome,
    Results,
    TaskSubsetInfo,
)
from tau2.data_model.tasks import Task
from tau2.judges.corrected_paper_suite import (
    COMPOSE_REPORT_FILENAME,
    CrossWorkspaceComposeReport,
    FinalWindowExclusionManifest,
    FinalWindowExclusionRow,
    GenericJudgeManifest,
    compose_cross_workspace_judgments,
    load_generic_judge_manifest,
    sha256_file,
)
from tau2.judges.delivery.error_exclusions import (
    DELIVERY_ERROR_EXCLUSIONS_SCHEMA_VERSION,
    DeliveryErrorCall,
    DeliveryErrorExclusionsArtifact,
    DeliveryErrorRetryEvidence,
    build_delivery_error_exclusions,
    validate_delivery_error_exclusions,
)
from tau2.metrics.call_timeline import ToolEvent
from tau2.utils.text_match import fold_for_match

IDENTITY_SCHEMA_VERSION = "tau-multi-corrected-retail-identity-v1"
FIDELITY_SCHEMA_VERSION = "tau-multi-corrected-fidelity-census-v1"
ARCHIVED_IDENTITY_SEMANTICS = (
    "tau2.atlas.authentication/auth-v1 result pairing, outcome classification, "
    "entity-id extraction, and conjunctive argument/result matching"
)

LANGUAGES = ("en", "es", "pt", "hi", "ko", "zh")
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "hi": "Hindi",
    "ko": "Korean",
    "zh": "Mandarin",
}
DOMAINS = ("airline", "retail", "telecom")
SYSTEMS = (
    "OpenAI minimal",
    "OpenAI xhigh",
    "Gemini minimal",
    "Gemini high",
    "xAI",
)
SYSTEM_SLUGS = {
    "OpenAI minimal": "openai_minimal",
    "OpenAI xhigh": "openai_xhigh",
    "Gemini minimal": "gemini_minimal",
    "Gemini high": "gemini_high",
    "xAI": "xai_provider_default",
}
SYSTEM_LABELS = {slug: label for label, slug in SYSTEM_SLUGS.items()}
SYSTEM_RUNTIME_IDENTITY = {
    "openai_minimal": ("openai", "gpt-realtime-2", "minimal"),
    "openai_xhigh": ("openai", "gpt-realtime-2", "xhigh"),
    "gemini_minimal": (
        "gemini",
        "gemini-3.1-flash-live-preview",
        "minimal",
    ),
    "gemini_high": ("gemini", "gemini-3.1-flash-live-preview", "high"),
    "xai_provider_default": (
        "xai",
        "grok-voice-think-fast-1.0",
        "provider_default",
    ),
}
REPEATED_SYSTEM_SLUGS = frozenset(
    {"openai_minimal", "openai_xhigh", "gemini_minimal", "gemini_high"}
)
RETAIL_LOOKUP_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


class CorrectedAnalysisConfig(BaseModel):
    """Explicit split-composition inputs shared by both analyses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quality_workspace: Path
    delivery_workspace: Path
    modal_workspace: Path
    composed_workspace: Path
    output: Path


class RetailIdentityAnalysisConfig(CorrectedAnalysisConfig):
    """Inputs for the corrected Korean/Mandarin retail lookup analysis."""

    repo_root: Path


class FidelityCensusConfig(CorrectedAnalysisConfig):
    """Inputs for the complete corrected delivery-category census."""

    canonical_repo_root: Path
    final_window_sidecar: Path
    delivery_error_ledger: Path


class DeliveryErrorLedgerConfig(CorrectedAnalysisConfig):
    """Inputs for the complete 4,500-call delivery ERROR ledger."""

    canonical_repo_root: Path
    retry_evidence: Path | None = None
    retry_logs: tuple[Path, ...] = ()

    @model_validator(mode="after")
    def _one_retry_evidence_source(self) -> "DeliveryErrorLedgerConfig":
        if self.retry_evidence is not None and self.retry_logs:
            raise ValueError("supply retry evidence JSON or retry logs, not both")
        return self


class SourceFileIdentity(BaseModel):
    """One exact results source in an analysis cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    domain: str
    system_slug: str
    source_kind: Literal["canonical", "corrected_composition"]
    results_path: str
    results_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    analysis_inputs_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    calls: Annotated[int, Field(ge=0)]
    task_subset_provenance: Literal[
        "pinned", "legacy_null_verified_against_pinned_xai"
    ] = "pinned"


class CompositionProvenance(BaseModel):
    """Reproduced split-composition identity embedded in downstream artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compose_report_path: str
    compose_report_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    authoritative_manifest_path: str
    authoritative_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    cohort_fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    cohort_lock_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    merged_results_fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    merged_simulations_fingerprint_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]


class RetailExpectedIdentity(BaseModel):
    """The unique synthetic retail user embedded in a localized task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    first_name: str
    last_name: str
    zip_code: str
    email: str


class LookupOutcome(StrEnum):
    """Observed deterministic outcome of one paired retail lookup."""

    SUCCESS = "success"
    NO_MATCH = "no_match"
    TOOL_ERROR = "tool_error"
    UNPAIRED_RESULT = "unpaired_result"


class RetailLookupAttempt(BaseModel):
    """One lookup compared with the unique task-grounded identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_index: Annotated[int, Field(ge=0)]
    order_key: Annotated[int, Field(ge=0)]
    result_order_key: int | None
    tool_call_id: str
    tool_name: Literal["find_user_id_by_email", "find_user_id_by_name_zip"]
    arguments: dict[str, Any]
    argument_match: bool
    result_content: str | None
    result_error: bool | None
    outcome: LookupOutcome
    result_entity_ids: list[str]
    result_match: bool
    exact_expected_success: bool


class RetailIdentityCall(BaseModel):
    """One corrected trial-0 call and its exact-lookup state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    simulation_id: str
    localized_task_id: str
    canonical_task_id: str
    language: Literal["ko", "zh"]
    system_slug: str
    source_results_path: str
    analysis_input_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    expected_identity: RetailExpectedIdentity
    attempts: list[RetailLookupAttempt]
    exact_lookup_success: bool
    task_success: bool

    @model_validator(mode="after")
    def _exact_rollup_matches_attempts(self) -> "RetailIdentityCall":
        if self.exact_lookup_success != any(
            attempt.exact_expected_success for attempt in self.attempts
        ):
            raise ValueError("exact lookup rollup disagrees with attempts")
        return self


class RetailIdentitySummary(BaseModel):
    """Explicit broad and conditional denominators for one named group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: str
    calls: Annotated[int, Field(ge=0)]
    calls_with_lookup_attempt: Annotated[int, Field(ge=0)]
    exact_lookup_successes: Annotated[int, Field(ge=0)]
    exact_lookup_success_rate: float | None
    task_successes: Annotated[int, Field(ge=0)]
    task_success_rate: float | None
    task_successes_after_exact_lookup: Annotated[int, Field(ge=0)]
    conditional_task_success_rate: float | None

    @model_validator(mode="after")
    def _valid_denominators(self) -> "RetailIdentitySummary":
        if not (
            self.calls_with_lookup_attempt <= self.calls
            and self.exact_lookup_successes <= self.calls
            and self.task_successes <= self.calls
            and self.task_successes_after_exact_lookup <= self.exact_lookup_successes
        ):
            raise ValueError("identity summary count exceeds its denominator")
        return self


class TaskSourceIdentity(BaseModel):
    """Localized task file grounding one language's identity facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: Literal["ko", "zh"]
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tasks_in_file: Annotated[int, Field(ge=0)]
    selected_tasks: Annotated[int, Field(ge=0)]


class RetailIdentityArtifact(BaseModel):
    """Task-grounded identity lookup and conditional-success artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-corrected-retail-identity-v1"]
    semantics: Literal[
        "tau2.atlas.authentication/auth-v1 result pairing, outcome classification, "
        "entity-id extraction, and conjunctive argument/result matching"
    ]
    composition: CompositionProvenance
    task_sources: tuple[TaskSourceIdentity, TaskSourceIdentity]
    source_files: list[SourceFileIdentity]
    cohort_fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    calls: list[RetailIdentityCall]
    by_cell: list[RetailIdentitySummary]
    by_language: list[RetailIdentitySummary]
    repeated_four_systems_by_language: list[RetailIdentitySummary]

    @model_validator(mode="after")
    def _closed_cohort(self) -> "RetailIdentityArtifact":
        keys = [(row.simulation_id, row.system_slug) for row in self.calls]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate corrected identity call")
        if sum(row.calls for row in self.by_cell) != len(self.calls):
            raise ValueError("identity cell denominators do not cover calls exactly")
        return self


class FidelityCallCensus(BaseModel):
    """Deduplicated delivery sufficient counts for one voice call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    simulation_id: str
    language: str
    domain: str
    system_slug: str
    source_results_path: str
    analysis_input_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    stored_delivery_utterances: Annotated[int, Field(ge=0)]
    pinned_greetings_excluded: Literal[1]
    eligible_utterances: Annotated[int, Field(ge=0)]
    unscored_utterances: Annotated[int, Field(ge=0)]
    excluded_findings_encountered: Annotated[int, Field(ge=0)]
    excluded_findings_applied_to_eligible_utterances: Annotated[int, Field(ge=0)]
    raw_material_findings: Annotated[int, Field(ge=0)]
    deduplicated_category_occurrences: Annotated[int, Field(ge=0)]
    material_failure_utterances: Annotated[int, Field(ge=0)]
    tone_meaning_flip_utterances: Annotated[int, Field(ge=0)]
    tone_only_failure_utterances: Annotated[int, Field(ge=0)]
    combined_standalone_failure_utterances: Annotated[int, Field(ge=0)]
    category_utterances: dict[str, int]

    @model_validator(mode="after")
    def _consistent_counts(self) -> "FidelityCallCensus":
        if self.eligible_utterances + self.unscored_utterances != (
            self.stored_delivery_utterances - self.pinned_greetings_excluded
        ):
            raise ValueError("fidelity utterance accounting does not close")
        if sum(self.category_utterances.values()) != (
            self.deduplicated_category_occurrences
        ):
            raise ValueError("category counts do not equal deduplicated occurrences")
        for value in (
            self.material_failure_utterances,
            self.tone_meaning_flip_utterances,
            self.tone_only_failure_utterances,
            self.combined_standalone_failure_utterances,
        ):
            if value > self.eligible_utterances:
                raise ValueError("fidelity failure count exceeds eligible utterances")
        if self.combined_standalone_failure_utterances < max(
            self.material_failure_utterances,
            self.tone_meaning_flip_utterances,
        ):
            raise ValueError("combined fidelity union is smaller than a component")
        return self


class FidelityCensusSummary(BaseModel):
    """Summed census counts for one language/system grouping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: str
    calls: Annotated[int, Field(ge=0)]
    stored_delivery_utterances: Annotated[int, Field(ge=0)]
    pinned_greetings_excluded: Annotated[int, Field(ge=0)]
    eligible_utterances: Annotated[int, Field(ge=0)]
    unscored_utterances: Annotated[int, Field(ge=0)]
    raw_material_findings: Annotated[int, Field(ge=0)]
    deduplicated_category_occurrences: Annotated[int, Field(ge=0)]
    material_failure_utterances: Annotated[int, Field(ge=0)]
    tone_meaning_flip_utterances: Annotated[int, Field(ge=0)]
    tone_only_failure_utterances: Annotated[int, Field(ge=0)]
    combined_standalone_failure_utterances: Annotated[int, Field(ge=0)]
    category_utterances: dict[str, int]


class FinalWindowProvenance(BaseModel):
    """Validated hybrid final-window sidecar identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    csv_path: str
    csv_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    manifest_path: str
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    rows: Annotated[int, Field(ge=0)]
    severity_2plus_rows: Annotated[int, Field(ge=0)]
    rows_encountered_in_cohort: Annotated[int, Field(ge=0)]
    rows_applied_to_eligible_utterances: Annotated[int, Field(ge=0)]


class DeliveryErrorProvenance(BaseModel):
    """Validated whole-utterance ERROR ledger used by the fidelity census."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_path: str
    artifact_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    schema_version: Literal["delivery-error-exclusions-v1"]
    rows: Annotated[int, Field(ge=0)]
    greeting_errors: Annotated[int, Field(ge=0)]
    by_reason: dict[str, int]
    by_language: dict[str, int]
    by_system: dict[str, int]


class FidelityCensusArtifact(BaseModel):
    """Complete corrected 4,500-call fidelity finding category census."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tau-multi-corrected-fidelity-census-v1"]
    deduplication_rule: Literal[
        "exclude injected greeting; retain completed utterances; remove keyed "
        "final-window findings; retain fidelity severity>=2; count each "
        "utterance/category once; union material failures with tone_meaning_flip"
    ]
    composition: CompositionProvenance
    final_window: FinalWindowProvenance
    delivery_errors: DeliveryErrorProvenance
    source_files: list[SourceFileIdentity]
    cohort_fingerprint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    calls: list[FidelityCallCensus]
    overall: FidelityCensusSummary
    by_language: list[FidelityCensusSummary]
    by_system: list[FidelityCensusSummary]
    by_language_system: list[FidelityCensusSummary]

    @model_validator(mode="after")
    def _complete_cohort(self) -> "FidelityCensusArtifact":
        if self.overall.calls != len(self.calls):
            raise ValueError("overall fidelity denominator does not cover calls")
        if sum(row.calls for row in self.by_language) != len(self.calls):
            raise ValueError("language fidelity denominators do not cover calls")
        if sum(row.calls for row in self.by_system) != len(self.calls):
            raise ValueError("system fidelity denominators do not cover calls")
        if sum(row.calls for row in self.by_language_system) != len(self.calls):
            raise ValueError("cell fidelity denominators do not cover calls")
        if self.overall.unscored_utterances != self.delivery_errors.rows:
            raise ValueError("unscored fidelity rows do not equal ERROR ledger rows")
        return self


def _verified_composition(
    config: CorrectedAnalysisConfig,
) -> tuple[CrossWorkspaceComposeReport, GenericJudgeManifest, CompositionProvenance]:
    composed = config.composed_workspace.expanduser().resolve()
    report_path = composed / COMPOSE_REPORT_FILENAME
    if not report_path.is_file():
        raise ValueError("composed workspace must already contain compose_report.json")
    report = compose_cross_workspace_judgments(
        quality_workspace=config.quality_workspace,
        delivery_workspace=config.delivery_workspace,
        modal_workspace=config.modal_workspace,
        output_workspace=composed,
    )
    manifest = load_generic_judge_manifest(config.quality_workspace)
    if report.calls != manifest.counts.calls or report.cells != len(manifest.cells):
        raise ValueError("composition report denominators disagree with its manifest")
    provenance = CompositionProvenance(
        compose_report_path=str(report_path),
        compose_report_sha256=sha256_file(report_path),
        authoritative_manifest_path=report.authoritative_manifest_path,
        authoritative_manifest_sha256=report.authoritative_manifest_sha256,
        cohort_fingerprint_sha256=report.cohort_fingerprint_sha256,
        cohort_lock_sha256=report.cohort_lock_sha256,
        merged_results_fingerprint_sha256=(report.merged_results_fingerprint_sha256),
        merged_simulations_fingerprint_sha256=(
            report.merged_simulations_fingerprint_sha256
        ),
    )
    return report, manifest, provenance


def _canonical_task_id(task_id: str, language: str) -> str:
    for suffix in (f"_{language}_identity", f"_{language}"):
        if task_id.endswith(suffix):
            return task_id[: -len(suffix)]
    return task_id


def _task_identity(task: Task) -> RetailExpectedIdentity:
    state = task.initial_state
    initialization = state.initialization_data if state else None
    agent_data = initialization.agent_data if initialization else None
    users = agent_data.get("users") if isinstance(agent_data, dict) else None
    if not isinstance(users, dict) or len(users) != 1:
        raise ValueError(f"task {task.id} must contain exactly one target retail user")
    key, raw = next(iter(users.items()))
    if not isinstance(raw, dict):
        raise ValueError(f"task {task.id} target retail user is not an object")
    name = raw.get("name")
    address = raw.get("address")
    if not isinstance(name, dict) or not isinstance(address, dict):
        raise ValueError(f"task {task.id} target user lacks name/address")
    identity = RetailExpectedIdentity(
        user_id=str(raw.get("user_id") or ""),
        first_name=str(name.get("first_name") or ""),
        last_name=str(name.get("last_name") or ""),
        zip_code=str(address.get("zip") or ""),
        email=str(raw.get("email") or ""),
    )
    if not all(identity.model_dump().values()):
        raise ValueError(f"task {task.id} target user has an incomplete identity")
    if str(key) != identity.user_id:
        raise ValueError(f"task {task.id} user map key disagrees with user_id")
    return identity


def _compact_zip(value: Any) -> str:
    return re.sub(r"[\s-]+", "", str(value).strip()).casefold()


def _lookup_arguments_match(
    tool_name: str, arguments: dict[str, Any], expected: RetailExpectedIdentity
) -> bool:
    if tool_name == "find_user_id_by_email":
        return set(arguments) == {"email"} and fold_for_match(
            str(arguments["email"])
        ) == fold_for_match(expected.email)
    if tool_name == "find_user_id_by_name_zip":
        if set(arguments) != {"first_name", "last_name", "zip"}:
            return False
        return (
            fold_for_match(str(arguments["first_name"]))
            == fold_for_match(expected.first_name)
            and fold_for_match(str(arguments["last_name"]))
            == fold_for_match(expected.last_name)
            and _compact_zip(arguments["zip"]) == _compact_zip(expected.zip_code)
        )
    raise ValueError(f"unsupported retail lookup tool: {tool_name}")


def _parse_result_content(content: str | None) -> Any:
    stripped = (content or "").strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        if stripped.casefold().startswith("error:"):
            return None
        return stripped.strip('"')


def _result_entity_ids(content: str | None) -> list[str]:
    parsed = _parse_result_content(content)
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, dict) and parsed.get("user_id"):
        return [str(parsed["user_id"])]
    return []


def _lookup_outcome(event: ToolEvent) -> LookupOutcome:
    if event.result_order_key is None:
        return LookupOutcome.UNPAIRED_RESULT
    parsed = _parse_result_content(event.result_content)
    if event.result_error is True:
        if "not found" in (event.result_content or "").casefold():
            return LookupOutcome.NO_MATCH
        return LookupOutcome.TOOL_ERROR
    if parsed in (None, [], {}):
        return LookupOutcome.NO_MATCH
    return LookupOutcome.SUCCESS


def analyze_retail_identity_events(
    events: list[ToolEvent], expected: RetailExpectedIdentity
) -> list[RetailLookupAttempt]:
    """Apply the archived exact-lookup state machine to paired current events."""
    attempts: list[RetailLookupAttempt] = []
    for event in events:
        if event.call.name not in RETAIL_LOOKUP_TOOLS:
            continue
        arguments = dict(event.call.arguments or {})
        outcome = _lookup_outcome(event)
        result_ids = _result_entity_ids(event.result_content)
        argument_match = _lookup_arguments_match(event.call.name, arguments, expected)
        result_match = len(result_ids) == 1 and fold_for_match(
            result_ids[0]
        ) == fold_for_match(expected.user_id)
        attempts.append(
            RetailLookupAttempt(
                attempt_index=len(attempts),
                order_key=event.order_key,
                result_order_key=event.result_order_key,
                tool_call_id=event.call.id,
                tool_name=event.call.name,
                arguments=arguments,
                argument_match=argument_match,
                result_content=event.result_content,
                result_error=event.result_error,
                outcome=outcome,
                result_entity_ids=result_ids,
                result_match=result_match,
                exact_expected_success=(
                    outcome is LookupOutcome.SUCCESS and argument_match and result_match
                ),
            )
        )
    return attempts


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _identity_summary(
    group: str, calls: list[RetailIdentityCall]
) -> RetailIdentitySummary:
    attempts = sum(bool(call.attempts) for call in calls)
    exact = sum(call.exact_lookup_success for call in calls)
    successes = sum(call.task_success for call in calls)
    conditional = sum(call.task_success and call.exact_lookup_success for call in calls)
    return RetailIdentitySummary(
        group=group,
        calls=len(calls),
        calls_with_lookup_attempt=attempts,
        exact_lookup_successes=exact,
        exact_lookup_success_rate=_rate(exact, len(calls)),
        task_successes=successes,
        task_success_rate=_rate(successes, len(calls)),
        task_successes_after_exact_lookup=conditional,
        conditional_task_success_rate=_rate(conditional, exact),
    )


def _load_identity_tasks(
    repo_root: Path, manifest: GenericJudgeManifest
) -> tuple[
    dict[str, dict[str, RetailExpectedIdentity]], tuple[TaskSourceIdentity, ...]
]:
    by_language: dict[str, dict[str, RetailExpectedIdentity]] = {}
    sources: list[TaskSourceIdentity] = []
    calls_by_language: dict[str, set[str]] = defaultdict(set)
    for cell in manifest.cells:
        calls_by_language[cell.language].update(call.task_id for call in cell.calls)
    if set(calls_by_language) != {"ko", "zh"}:
        raise ValueError("identity analysis requires exactly Korean and Mandarin")
    for language in ("ko", "zh"):
        path = (
            repo_root
            / "data"
            / "tau2"
            / "multilingual"
            / language
            / f"retail_tasks_{language}_identity.json"
        ).resolve()
        tasks = TypeAdapter(list[Task]).validate_json(path.read_text())
        task_map = {task.id: task for task in tasks}
        if len(task_map) != len(tasks):
            raise ValueError(f"duplicate task ids in {path}")
        selected = calls_by_language[language]
        if len(selected) != manifest.cohort.calls_per_cell:
            raise ValueError(
                f"{language} does not have the required unique task denominator"
            )
        missing = selected - set(task_map)
        if missing:
            raise ValueError(f"missing localized identity tasks: {sorted(missing)[:5]}")
        identities = {
            task_id: _task_identity(task_map[task_id]) for task_id in selected
        }
        by_language[language] = identities
        sources.append(
            TaskSourceIdentity(
                language=language,
                path=str(path),
                sha256=sha256_file(path),
                tasks_in_file=len(tasks),
                selected_tasks=len(selected),
            )
        )
    return by_language, tuple(sources)


def _index_by_id(results: Results) -> dict[str, Any]:
    entries = {entry.id: entry for entry in results.simulation_index}
    if len(entries) != len(results.simulation_index):
        raise ValueError("duplicate simulation ids in results index")
    return entries


def _read_simulation_prefix(path: Path) -> dict[str, Any]:
    """Read top-level evidence without materializing the enormous tick array."""
    marker = b'\n  "ticks": '
    payload = bytearray()
    with path.open("rb") as handle:
        while marker not in payload:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"simulation has no top-level ticks field: {path}")
            payload.extend(chunk)
    prefix = bytes(payload).split(marker, 1)[0].rstrip()
    if prefix.endswith(b","):
        prefix = prefix[:-1]
    parsed = json.loads(prefix + b"\n}")
    if not isinstance(parsed, dict):
        raise ValueError(f"simulation prefix is not an object: {path}")
    return parsed


def _json_array_end(data: mmap.mmap, start: int) -> int:
    """Return the inclusive end of a JSON array without decoding its strings."""
    if data[start] != ord("["):
        raise ValueError("target JSON value is not an array")
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(data)):
        byte = data[position]
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue
        if byte == ord('"'):
            in_string = True
        elif byte == ord("["):
            depth += 1
        elif byte == ord("]"):
            depth -= 1
            if depth == 0:
                return position
    raise ValueError("unterminated JSON array")


def _array_after_key(data: mmap.mmap, key_at: int, key: bytes) -> tuple[list, int]:
    position = key_at + len(key)
    while position < len(data) and chr(data[position]).isspace():
        position += 1
    end = _json_array_end(data, position)
    parsed = json.loads(data[position : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("target JSON value did not decode to an array")
    return parsed, end + 1


def _tool_events_from_path(path: Path) -> list[ToolEvent]:
    """Stream only paired agent tool evidence from a large voice simulation."""
    call_key = b'"agent_tool_calls":'
    result_key = b'"agent_tool_results":'
    events: list[ToolEvent] = []
    by_call_id: dict[str, ToolEvent] = {}
    position = 0
    tick_index = 0
    expect_calls = True
    arrays = 0
    with (
        path.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        while True:
            next_call = data.find(call_key, position)
            next_result = data.find(result_key, position)
            if next_call < 0 and next_result < 0:
                break
            if expect_calls:
                if next_call < 0 or (next_result >= 0 and next_result < next_call):
                    raise ValueError(f"tool arrays are out of tick order in {path}")
                raw_calls, position = _array_after_key(data, next_call, call_key)
                for raw_call in raw_calls:
                    name = str(raw_call.get("name") or "")
                    if name not in RETAIL_LOOKUP_TOOLS:
                        continue
                    event = ToolEvent(
                        order_key=tick_index,
                        call=raw_call,
                    )
                    if not event.call.id or event.call.id in by_call_id:
                        raise ValueError(f"duplicate or empty lookup call id in {path}")
                    events.append(event)
                    by_call_id[event.call.id] = event
                expect_calls = False
                arrays += 1
                continue
            if next_result < 0 or (next_call >= 0 and next_call < next_result):
                raise ValueError(f"tool arrays are out of tick order in {path}")
            raw_results, position = _array_after_key(data, next_result, result_key)
            for result in raw_results:
                event = by_call_id.get(str(result.get("id") or ""))
                if event is None:
                    continue
                event.result_order_key = tick_index
                content = result.get("content")
                event.result_content = (
                    content
                    if isinstance(content, str) or content is None
                    else json.dumps(content, ensure_ascii=False, sort_keys=True)
                )
                error = result.get("error")
                event.result_error = error if isinstance(error, bool) else None
            expect_calls = True
            tick_index += 1
    if not expect_calls or arrays == 0:
        raise ValueError(f"incomplete or empty tick tool inventory in {path}")
    return events


def _analysis_fingerprint(hashes: dict[str, str]) -> str:
    return _sha256_value(sorted(hashes.items()))


def _write_immutable_artifact(path: Path, artifact: BaseModel) -> None:
    path = path.expanduser().resolve()
    payload = artifact.model_dump(mode="json")
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise ValueError(f"analysis output exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_retail_identity_analysis(
    config: RetailIdentityAnalysisConfig,
) -> RetailIdentityArtifact:
    """Build the exact-lookup artifact for composed KO/ZH retail trial 0."""
    report, manifest, composition = _verified_composition(config)
    repo_root = config.repo_root.expanduser().resolve()
    if (repo_root / "data" / "simulations").resolve() != Path(
        report.source_root
    ).resolve():
        raise ValueError(
            "identity task repository does not own the composed source results"
        )
    if manifest.counts.calls != manifest.cohort.expected_calls:
        raise ValueError("generic judge manifest call denominator is incomplete")
    if len(manifest.cells) != 10 or manifest.cohort.calls_per_cell != 50:
        raise ValueError("identity analysis requires the exact 10 x 50 cohort")
    task_identities, task_sources = _load_identity_tasks(repo_root, manifest)
    calls: list[RetailIdentityCall] = []
    sources: list[SourceFileIdentity] = []
    merged_root = Path(report.merged_root)
    for cell in manifest.cells:
        results_path = merged_root / cell.source_relative_path / "results.json"
        metadata = Results.load_metadata(results_path)
        index = _index_by_id(metadata)
        expected_by_id = {call.simulation_id: call for call in cell.calls}
        if set(index) != set(expected_by_id):
            raise ValueError(f"composed results index drifted: {results_path}")
        input_hashes: dict[str, str] = {}
        seen_tasks: set[str] = set()
        for simulation_id in sorted(expected_by_id):
            simulation_path = (
                results_path.parent / "simulations" / f"{simulation_id}.json"
            )
            prefix = _read_simulation_prefix(simulation_path)
            events = _tool_events_from_path(simulation_path)
            locked = expected_by_id[simulation_id]
            if (
                prefix.get("id") != simulation_id
                or prefix.get("task_id") != locked.task_id
                or locked.trial != 0
            ):
                raise ValueError(f"composed call identity drifted: {simulation_id}")
            entry = index[simulation_id]
            if entry.task_id != prefix["task_id"] or entry.trial != locked.trial:
                raise ValueError(
                    f"results index disagrees with simulation: {simulation_id}"
                )
            reward_info = prefix.get("reward_info")
            reward = (
                reward_info.get("reward") if isinstance(reward_info, dict) else None
            )
            if reward not in {0.0, 1.0} or entry.reward != reward:
                raise ValueError(f"non-binary or inconsistent reward: {simulation_id}")
            expected = task_identities[cell.language][prefix["task_id"]]
            attempts = analyze_retail_identity_events(events, expected)
            input_hashes[simulation_id] = _sha256_value(
                {
                    "simulation_id": simulation_id,
                    "task_id": prefix["task_id"],
                    "trial": locked.trial,
                    "reward": reward,
                    "lookup_events": [
                        event.model_dump(mode="json") for event in events
                    ],
                }
            )
            seen_tasks.add(prefix["task_id"])
            calls.append(
                RetailIdentityCall(
                    simulation_id=simulation_id,
                    localized_task_id=prefix["task_id"],
                    canonical_task_id=_canonical_task_id(
                        prefix["task_id"], cell.language
                    ),
                    language=cell.language,
                    system_slug=cell.system_slug,
                    source_results_path=str(results_path),
                    analysis_input_sha256=input_hashes[simulation_id],
                    expected_identity=expected,
                    attempts=attempts,
                    exact_lookup_success=any(
                        attempt.exact_expected_success for attempt in attempts
                    ),
                    task_success=reward == 1.0,
                )
            )
        if seen_tasks != set(task_identities[cell.language]):
            raise ValueError(
                f"cell task frame drifted for {cell.language}/{cell.system_slug}"
            )
        sources.append(
            SourceFileIdentity(
                language=cell.language,
                domain="retail",
                system_slug=cell.system_slug,
                source_kind="corrected_composition",
                results_path=str(results_path),
                results_sha256=sha256_file(results_path),
                analysis_inputs_sha256=_analysis_fingerprint(input_hashes),
                calls=len(expected_by_id),
            )
        )
    calls.sort(key=lambda row: (row.language, row.system_slug, row.canonical_task_id))
    by_cell = [
        _identity_summary(
            f"{language}/{system}",
            [
                call
                for call in calls
                if call.language == language and call.system_slug == system
            ],
        )
        for language in ("ko", "zh")
        for system in SYSTEM_LABELS
    ]
    by_language = [
        _identity_summary(
            language, [call for call in calls if call.language == language]
        )
        for language in ("ko", "zh")
    ]
    repeated = [
        _identity_summary(
            f"{language}/repeated_four_systems",
            [
                call
                for call in calls
                if call.language == language
                and call.system_slug in REPEATED_SYSTEM_SLUGS
            ],
        )
        for language in ("ko", "zh")
    ]
    source_payload = [source.model_dump(mode="json") for source in sources]
    artifact = RetailIdentityArtifact(
        schema_version=IDENTITY_SCHEMA_VERSION,
        semantics=ARCHIVED_IDENTITY_SEMANTICS,
        composition=composition,
        task_sources=task_sources,
        source_files=sources,
        cohort_fingerprint_sha256=_sha256_value(
            {
                "composition": composition.model_dump(mode="json"),
                "tasks": [row.model_dump(mode="json") for row in task_sources],
                "sources": source_payload,
            }
        ),
        calls=calls,
        by_cell=by_cell,
        by_language=by_language,
        repeated_four_systems_by_language=repeated,
    )
    _write_immutable_artifact(config.output, artifact)
    return artifact


def _canonical_cell_directory(root: Path, language: str, domain: str) -> Path:
    if domain == "airline":
        stem = (
            "airline_en_v1"
            if language == "en"
            else f"airline_v1_{LANGUAGE_NAMES[language].lower()}_airline"
        )
    elif domain == "retail":
        stem = f"retail_v1_{LANGUAGE_NAMES[language].lower()}_retail"
    elif language in {"en", "es", "pt"}:
        stem = f"preference_strat50_{LANGUAGE_NAMES[language].lower()}_telecom"
    else:
        stem = f"preference_runs_v1_{LANGUAGE_NAMES[language].lower()}_telecom"
    return root / stem


def _canonical_results_path(
    root: Path, language: str, domain: str, system: str
) -> Path:
    parent = (
        root / f"{domain}_xai_v2_{LANGUAGE_NAMES[language].lower()}_{domain}"
        if system == "xAI"
        else _canonical_cell_directory(root, language, domain)
    )
    slug = SYSTEM_SLUGS[system]
    matches = sorted(parent.glob(f"*/{language}_{domain}_{slug}/results.json"))
    if not matches:
        matches = sorted(parent.glob(f"{language}_{domain}_{slug}/results.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one canonical results file for "
            f"{language}/{domain}/{system}, got {matches}"
        )
    return matches[0]


def _load_final_window(
    path: Path,
) -> tuple[
    list[FinalWindowExclusionRow],
    FinalWindowExclusionManifest,
    dict[tuple[str, int, int], FinalWindowExclusionRow],
]:
    path = path.expanduser().resolve()
    with path.open(newline="") as handle:
        rows = [
            FinalWindowExclusionRow.model_validate(row)
            for row in csv.DictReader(handle)
        ]
    by_key = {(row.sim_id, row.utterance_idx, row.finding_index): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("duplicate final-window exclusion key")
    manifest_path = path.with_suffix(".json")
    manifest = FinalWindowExclusionManifest.model_validate_json(
        manifest_path.read_text()
    )
    if manifest.artifact != path.name:
        raise ValueError("final-window manifest names a different artifact")
    if manifest.final_window_csv_sha256 != sha256_file(path):
        raise ValueError("final-window CSV hash drifted")
    if manifest.excluded_findings != len(rows):
        raise ValueError("final-window manifest row count drifted")
    if manifest.excluded_severity_2plus_findings != sum(
        row.severity >= 2 for row in rows
    ):
        raise ValueError("final-window severity denominator drifted")
    if manifest.by_language != dict(
        sorted(Counter(row.language for row in rows).items())
    ):
        raise ValueError("final-window language counts drifted")
    if manifest.by_system != dict(sorted(Counter(row.system for row in rows).items())):
        raise ValueError("final-window system counts drifted")
    return rows, manifest, by_key


def count_fidelity_call(
    *,
    simulation_id: str,
    language: str,
    domain: str,
    system_slug: str,
    source_results_path: str,
    analysis_input_sha256: str,
    delivery: DeliveryInfo,
    exclusions: dict[tuple[str, int, int], FinalWindowExclusionRow],
    encountered_exclusions: set[tuple[str, int, int]],
    applied_exclusions: set[tuple[str, int, int]],
) -> FidelityCallCensus:
    """Count one call using the paper's exact utterance/category dedupe rule."""
    results = delivery.utterance_results
    if not results or results[0].utterance_idx != 0:
        raise ValueError(f"{simulation_id} delivery inventory lacks greeting index 0")
    indices = [result.utterance_idx for result in results]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{simulation_id} has duplicate delivery utterance indices")
    category_counts: Counter[str] = Counter()
    raw_findings = 0
    eligible = 0
    unscored = 0
    excluded_encountered = 0
    excluded_applied = 0
    material_failures = 0
    tone_failures = 0
    tone_only = 0
    combined_failures = 0
    system = SYSTEM_LABELS[system_slug]
    for position, result in enumerate(results):
        eligible_result = position > 0 and result.outcome in {
            JudgeOutcome.PASS,
            JudgeOutcome.FAIL,
        }
        categories: set[str] = set()
        for finding_index, finding in enumerate(result.findings):
            key = (simulation_id, result.utterance_idx, finding_index)
            exclusion = exclusions.get(key)
            if exclusion is not None:
                if (
                    exclusion.language != language
                    or exclusion.domain != domain
                    or exclusion.system != system
                    or exclusion.severity != finding.severity
                    or exclusion.time_range != (finding.time_range or "")
                ):
                    raise ValueError(f"final-window row metadata drifted for {key}")
                encountered_exclusions.add(key)
                excluded_encountered += 1
                if eligible_result:
                    applied_exclusions.add(key)
                    excluded_applied += 1
                continue
            if eligible_result and finding.axis == "fidelity" and finding.severity >= 2:
                raw_findings += 1
                categories.add(finding.category)
        if position == 0:
            continue
        if not eligible_result:
            unscored += 1
            continue
        eligible += 1
        category_counts.update(categories)
        material = bool(categories)
        tone = any(
            check.id == "tone_meaning_flip" and check.outcome is JudgeOutcome.FAIL
            for check in result.factor_checks
        )
        material_failures += material
        tone_failures += tone
        tone_only += tone and not material
        combined_failures += material or tone
    return FidelityCallCensus(
        simulation_id=simulation_id,
        language=language,
        domain=domain,
        system_slug=system_slug,
        source_results_path=source_results_path,
        analysis_input_sha256=analysis_input_sha256,
        stored_delivery_utterances=len(results),
        pinned_greetings_excluded=1,
        eligible_utterances=eligible,
        unscored_utterances=unscored,
        excluded_findings_encountered=excluded_encountered,
        excluded_findings_applied_to_eligible_utterances=excluded_applied,
        raw_material_findings=raw_findings,
        deduplicated_category_occurrences=sum(category_counts.values()),
        material_failure_utterances=material_failures,
        tone_meaning_flip_utterances=tone_failures,
        tone_only_failure_utterances=tone_only,
        combined_standalone_failure_utterances=combined_failures,
        category_utterances=dict(sorted(category_counts.items())),
    )


def _fidelity_summary(
    group: str, calls: list[FidelityCallCensus]
) -> FidelityCensusSummary:
    categories: Counter[str] = Counter()
    for call in calls:
        categories.update(call.category_utterances)
    return FidelityCensusSummary(
        group=group,
        calls=len(calls),
        stored_delivery_utterances=sum(
            call.stored_delivery_utterances for call in calls
        ),
        pinned_greetings_excluded=sum(call.pinned_greetings_excluded for call in calls),
        eligible_utterances=sum(call.eligible_utterances for call in calls),
        unscored_utterances=sum(call.unscored_utterances for call in calls),
        raw_material_findings=sum(call.raw_material_findings for call in calls),
        deduplicated_category_occurrences=sum(
            call.deduplicated_category_occurrences for call in calls
        ),
        material_failure_utterances=sum(
            call.material_failure_utterances for call in calls
        ),
        tone_meaning_flip_utterances=sum(
            call.tone_meaning_flip_utterances for call in calls
        ),
        tone_only_failure_utterances=sum(
            call.tone_only_failure_utterances for call in calls
        ),
        combined_standalone_failure_utterances=sum(
            call.combined_standalone_failure_utterances for call in calls
        ),
        category_utterances=dict(sorted(categories.items())),
    )


class _DeliverySimulationEvidence(BaseModel):
    """Small, hashable projection consumed by the fidelity census."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    simulation_id: str
    task_id: str
    trial: Literal[0]
    delivery: DeliveryInfo
    analysis_input_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _DeliverySourceEvidence(BaseModel):
    """One verified 50-call cell in the hybrid 4,500-call delivery cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: str
    domain: str
    system_slug: str
    source_kind: Literal["canonical", "corrected_composition"]
    results_path: Path
    results_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    task_subset_provenance: Literal["pinned", "legacy_null_verified_against_pinned_xai"]
    simulations: list[_DeliverySimulationEvidence]
    metadata: Results


def _selected_delivery_evidence(
    results_path: Path,
) -> tuple[list[_DeliverySimulationEvidence], Results]:
    metadata = Results.load_metadata(results_path)
    index = _index_by_id(metadata)
    selected_index = {key: value for key, value in index.items() if value.trial == 0}
    if len(selected_index) != 50:
        raise ValueError(f"expected 50 trial-0 calls in {results_path}")
    simulations: list[_DeliverySimulationEvidence] = []
    for simulation_id, entry in sorted(selected_index.items()):
        simulation_path = results_path.parent / "simulations" / f"{simulation_id}.json"
        prefix = _read_simulation_prefix(simulation_path)
        if (
            prefix.get("id") != simulation_id
            or prefix.get("task_id") != entry.task_id
            or entry.trial != 0
        ):
            raise ValueError(f"index identity drifted for {simulation_id}")
        raw_delivery = prefix.get("delivery_info")
        if raw_delivery is None:
            raise ValueError(f"missing delivery result for {simulation_id}")
        delivery = DeliveryInfo.model_validate(raw_delivery)
        digest = _sha256_value(
            {
                "simulation_id": simulation_id,
                "task_id": entry.task_id,
                "trial": 0,
                "delivery_info": delivery.model_dump(mode="json"),
            }
        )
        simulations.append(
            _DeliverySimulationEvidence(
                simulation_id=simulation_id,
                task_id=entry.task_id,
                trial=0,
                delivery=delivery,
                analysis_input_sha256=digest,
            )
        )
    return simulations, metadata


def _task_subset_provenance(
    subset: TaskSubsetInfo | None,
    *,
    language: str,
    domain: str,
    system_slug: str,
    source_kind: Literal["canonical", "corrected_composition"],
    canonical_root: Path,
    results_path: Path,
) -> Literal["pinned", "legacy_null_verified_against_pinned_xai"]:
    if subset is not None:
        if (
            subset.size != 50
            or subset.tasks_scored != 50
            or subset.name != f"{domain}_50"
        ):
            raise ValueError(f"task subset metadata drifted: {results_path}")
        return "pinned"
    expected_legacy_key = (
        source_kind == "canonical"
        and language == "en"
        and domain == "telecom"
        and system_slug in REPEATED_SYSTEM_SLUGS
        and results_path.is_relative_to(canonical_root)
    )
    if not expected_legacy_key:
        raise ValueError(f"task subset metadata drifted: {results_path}")
    return "legacy_null_verified_against_pinned_xai"


def _load_delivery_cohort(
    config: CorrectedAnalysisConfig,
    canonical_repo_root: Path,
) -> tuple[
    CrossWorkspaceComposeReport,
    GenericJudgeManifest,
    CompositionProvenance,
    list[_DeliverySourceEvidence],
]:
    """Load and close the exact canonical-80 plus corrected-10 cohort."""
    report, manifest, composition = _verified_composition(config)
    canonical_root = (
        canonical_repo_root.expanduser().resolve()
        / "data"
        / "simulations"
        / "paper_runs"
        / "tau-multi"
        / "main_runs"
    )
    replacements = {
        (cell.language, "retail", cell.system_slug): (
            Path(report.merged_root) / cell.source_relative_path / "results.json"
        )
        for cell in manifest.cells
    }
    if len(replacements) != 10:
        raise ValueError("delivery cohort requires exactly ten replacement cells")
    sources: list[_DeliverySourceEvidence] = []
    seen_simulations: set[str] = set()
    task_frames: dict[tuple[str, str], frozenset[str]] = {}
    for language in LANGUAGES:
        for domain in DOMAINS:
            for system in SYSTEMS:
                system_slug = SYSTEM_SLUGS[system]
                key = (language, domain, system_slug)
                source_kind: Literal["canonical", "corrected_composition"]
                if key in replacements:
                    results_path = replacements[key]
                    source_kind = "corrected_composition"
                else:
                    results_path = _canonical_results_path(
                        canonical_root, language, domain, system
                    )
                    source_kind = "canonical"
                simulations, metadata = _selected_delivery_evidence(results_path)
                _validate_system_metadata(metadata, system_slug, results_path)
                if metadata.info.environment_info.domain_name != domain:
                    raise ValueError(f"domain metadata drifted: {results_path}")
                subset_provenance = _task_subset_provenance(
                    metadata.info.task_subset,
                    language=language,
                    domain=domain,
                    system_slug=system_slug,
                    source_kind=source_kind,
                    canonical_root=canonical_root,
                    results_path=results_path,
                )
                task_ids = frozenset(sim.task_id for sim in simulations)
                if len(task_ids) != 50 or any(
                    not (
                        task_id.endswith(f"_{language}")
                        or task_id.endswith(f"_{language}_identity")
                    )
                    for task_id in task_ids
                ):
                    raise ValueError(f"localized task frame drifted: {results_path}")
                frame_key = (language, domain)
                expected_frame = task_frames.setdefault(frame_key, task_ids)
                if task_ids != expected_frame:
                    raise ValueError(
                        f"systems do not share one task frame: {language}/{domain}"
                    )
                for simulation in simulations:
                    if simulation.simulation_id in seen_simulations:
                        raise ValueError(
                            "duplicate simulation id across cells: "
                            f"{simulation.simulation_id}"
                        )
                    seen_simulations.add(simulation.simulation_id)
                sources.append(
                    _DeliverySourceEvidence(
                        language=language,
                        domain=domain,
                        system_slug=system_slug,
                        source_kind=source_kind,
                        results_path=results_path,
                        results_sha256=sha256_file(results_path),
                        task_subset_provenance=subset_provenance,
                        simulations=simulations,
                        metadata=metadata,
                    )
                )
    if (
        len(sources) != 90
        or sum(len(source.simulations) for source in sources) != 4_500
    ):
        raise ValueError("delivery cohort does not contain the exact 90 x 50 calls")
    legacy_sources = {
        (source.language, source.domain, source.system_slug): source
        for source in sources
        if source.task_subset_provenance == "legacy_null_verified_against_pinned_xai"
    }
    expected_legacy_keys = {
        ("en", "telecom", system_slug) for system_slug in REPEATED_SYSTEM_SLUGS
    }
    if set(legacy_sources) != expected_legacy_keys:
        raise ValueError(
            "legacy null task-subset inventory is not the exact four cells"
        )
    [pinned_xai] = [
        source
        for source in sources
        if (source.language, source.domain, source.system_slug)
        == ("en", "telecom", "xai_provider_default")
    ]
    if pinned_xai.task_subset_provenance != "pinned":
        raise ValueError("English telecom xAI task frame is not pinned")
    pinned_xai_tasks = {sim.task_id for sim in pinned_xai.simulations}
    for source in legacy_sources.values():
        if {sim.task_id for sim in source.simulations} != pinned_xai_tasks:
            raise ValueError("legacy English telecom frame differs from pinned xAI")
    return report, manifest, composition, sources


def _delivery_error_calls(
    sources: list[_DeliverySourceEvidence],
) -> list[DeliveryErrorCall]:
    """Project the verified hybrid cohort into typed ledger inputs."""
    return [
        DeliveryErrorCall(
            source_results_sha256=source.results_sha256,
            source_kind=source.source_kind,
            simulation_id=simulation.simulation_id,
            language=source.language,
            domain=source.domain,
            system=source.system_slug,
            task_subset_provenance=source.task_subset_provenance,
            task_id=simulation.task_id,
            trial=simulation.trial,
            delivery=simulation.delivery,
        )
        for source in sources
        for simulation in source.simulations
    ]


_DELIVERY_FAILURE_RE = re.compile(
    r"delivery judge failed for utterance (?P<utterance>\d+) "
    r"\(sim (?P<simulation>[0-9a-f-]+)\):"
)
_TOKEN_LIMIT_WARNING = "Output might be incomplete due to token limit!"


def extract_delivery_retry_evidence(
    calls: list[DeliveryErrorCall], log_paths: tuple[Path, ...]
) -> tuple[DeliveryErrorRetryEvidence, ...]:
    """Bind explicit token-limit log events to the cohort's current ERROR slots."""
    targets = {
        (call.simulation_id, result.utterance_idx): call.source_results_sha256
        for call in calls
        for position, result in enumerate(call.delivery.utterance_results)
        if position > 0 and result.outcome is JudgeOutcome.ERROR
    }
    required = {
        (call.simulation_id, result.utterance_idx)
        for call in calls
        if call.source_kind == "corrected_composition"
        for position, result in enumerate(call.delivery.utterance_results)
        if position > 0 and result.outcome is JudgeOutcome.ERROR
    }
    observations: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for raw_path in log_paths:
        path = raw_path.expanduser().resolve()
        lines = path.read_text().splitlines()
        log_identity = f"{path.name}:{sha256_file(path)}"
        for line_number, line in enumerate(lines):
            match = _DELIVERY_FAILURE_RE.search(line)
            if match is None:
                continue
            key = (match.group("simulation"), int(match.group("utterance")))
            if key not in targets:
                continue
            if line_number == 0 or _TOKEN_LIMIT_WARNING not in lines[line_number - 1]:
                raise ValueError(
                    "delivery retry failure lacks adjacent token-limit evidence: "
                    f"{path}:{line_number + 1}"
                )
            observations[key][log_identity] += 1
    if set(observations) != required:
        missing = sorted(required - set(observations))
        extra = sorted(set(observations) - required)
        raise ValueError(
            "retry logs do not exactly prove corrected ERROR slots: "
            f"missing={missing} extra={extra}"
        )
    return tuple(
        DeliveryErrorRetryEvidence(
            source_results_sha256=targets[key],
            simulation_id=key[0],
            utterance_idx=key[1],
            normalized_reason="max_tokens_truncation",
            retry_count=sum(by_log.values()),
            evidence=tuple(
                f"finish_reason=length; log={log_identity}; occurrences={occurrences}"
                for log_identity, occurrences in sorted(by_log.items())
            ),
        )
        for key, by_log in sorted(observations.items())
    )


def build_delivery_error_ledger(
    config: DeliveryErrorLedgerConfig,
) -> DeliveryErrorExclusionsArtifact:
    """Build the immutable whole-utterance ERROR ledger; no API calls."""
    _report, _manifest, _composition, sources = _load_delivery_cohort(
        config, config.canonical_repo_root
    )
    evidence: tuple[DeliveryErrorRetryEvidence, ...] = ()
    if config.retry_evidence is not None:
        evidence = TypeAdapter(tuple[DeliveryErrorRetryEvidence, ...]).validate_json(
            config.retry_evidence.expanduser().resolve().read_text()
        )
    elif config.retry_logs:
        evidence = extract_delivery_retry_evidence(
            _delivery_error_calls(sources), config.retry_logs
        )
    artifact = build_delivery_error_exclusions(
        _delivery_error_calls(sources), retry_evidence=evidence
    )
    _write_immutable_artifact(config.output, artifact)
    return artifact


def _validate_system_metadata(results: Results, system_slug: str, path: Path) -> None:
    audio = results.info.audio_native_config
    observed = (
        audio.provider if audio else None,
        audio.model if audio else None,
        audio.reasoning_effort if audio else None,
    )
    if observed != SYSTEM_RUNTIME_IDENTITY[system_slug]:
        raise ValueError(f"voice system metadata drifted: {path}")


def build_fidelity_category_census(
    config: FidelityCensusConfig,
) -> FidelityCensusArtifact:
    """Build the full corrected 90-cell delivery category census."""
    report, _manifest, composition, delivery_sources = _load_delivery_cohort(
        config, config.canonical_repo_root
    )
    rows, final_manifest, exclusions = _load_final_window(config.final_window_sidecar)
    if (
        final_manifest.replacement_cohort_fingerprint_sha256
        != report.cohort_fingerprint_sha256
        or final_manifest.replacement_calls != report.calls
        or final_manifest.canonical_sidecar_sha256 is None
    ):
        raise ValueError(
            "final-window sidecar is not the hybrid for this composed cohort"
        )
    sources: list[SourceFileIdentity] = []
    calls: list[FidelityCallCensus] = []
    encountered: set[tuple[str, int, int]] = set()
    applied: set[tuple[str, int, int]] = set()
    for source in delivery_sources:
        for sim in source.simulations:
            calls.append(
                count_fidelity_call(
                    simulation_id=sim.simulation_id,
                    language=source.language,
                    domain=source.domain,
                    system_slug=source.system_slug,
                    source_results_path=str(source.results_path),
                    analysis_input_sha256=sim.analysis_input_sha256,
                    delivery=sim.delivery,
                    exclusions=exclusions,
                    encountered_exclusions=encountered,
                    applied_exclusions=applied,
                )
            )
        sources.append(
            SourceFileIdentity(
                language=source.language,
                domain=source.domain,
                system_slug=source.system_slug,
                source_kind=source.source_kind,
                results_path=str(source.results_path),
                results_sha256=source.results_sha256,
                task_subset_provenance=source.task_subset_provenance,
                analysis_inputs_sha256=_analysis_fingerprint(
                    {
                        sim.simulation_id: sim.analysis_input_sha256
                        for sim in source.simulations
                    }
                ),
                calls=len(source.simulations),
            )
        )
    if len(sources) != 90 or len(calls) != 4_500:
        raise ValueError("fidelity census does not contain the exact 90 x 50 cohort")
    if encountered != set(exclusions):
        missing = sorted(set(exclusions) - encountered)
        raise ValueError(f"final-window rows do not match the cohort: {missing[:5]}")
    calls.sort(
        key=lambda row: (row.language, row.domain, row.system_slug, row.simulation_id)
    )
    sources.sort(key=lambda row: (row.language, row.domain, row.system_slug))
    final_path = config.final_window_sidecar.expanduser().resolve()
    final_manifest_path = final_path.with_suffix(".json")
    final_provenance = FinalWindowProvenance(
        csv_path=str(final_path),
        csv_sha256=sha256_file(final_path),
        manifest_path=str(final_manifest_path),
        manifest_sha256=sha256_file(final_manifest_path),
        rows=len(rows),
        severity_2plus_rows=final_manifest.excluded_severity_2plus_findings,
        rows_encountered_in_cohort=len(encountered),
        rows_applied_to_eligible_utterances=len(applied),
    )
    ledger_path = config.delivery_error_ledger.expanduser().resolve()
    ledger = DeliveryErrorExclusionsArtifact.model_validate_json(
        ledger_path.read_text()
    )
    validate_delivery_error_exclusions(_delivery_error_calls(delivery_sources), ledger)
    delivery_error_provenance = DeliveryErrorProvenance(
        artifact_path=str(ledger_path),
        artifact_sha256=sha256_file(ledger_path),
        schema_version=DELIVERY_ERROR_EXCLUSIONS_SCHEMA_VERSION,
        rows=ledger.excluded_error_utterances,
        greeting_errors=ledger.greeting_errors,
        by_reason=ledger.by_reason,
        by_language=ledger.by_language,
        by_system=ledger.by_system,
    )
    artifact = FidelityCensusArtifact(
        schema_version=FIDELITY_SCHEMA_VERSION,
        deduplication_rule=(
            "exclude injected greeting; retain completed utterances; remove keyed "
            "final-window findings; retain fidelity severity>=2; count each "
            "utterance/category once; union material failures with tone_meaning_flip"
        ),
        composition=composition,
        final_window=final_provenance,
        delivery_errors=delivery_error_provenance,
        source_files=sources,
        cohort_fingerprint_sha256=_sha256_value(
            {
                "composition": composition.model_dump(mode="json"),
                "final_window": final_provenance.model_dump(mode="json"),
                "delivery_errors": delivery_error_provenance.model_dump(mode="json"),
                "sources": [source.model_dump(mode="json") for source in sources],
            }
        ),
        calls=calls,
        overall=_fidelity_summary("all", calls),
        by_language=[
            _fidelity_summary(
                language, [call for call in calls if call.language == language]
            )
            for language in LANGUAGES
        ],
        by_system=[
            _fidelity_summary(
                system_slug,
                [call for call in calls if call.system_slug == system_slug],
            )
            for system_slug in SYSTEM_LABELS
        ],
        by_language_system=[
            _fidelity_summary(
                f"{language}/{system_slug}",
                [
                    call
                    for call in calls
                    if call.language == language and call.system_slug == system_slug
                ],
            )
            for language in LANGUAGES
            for system_slug in SYSTEM_LABELS
        ],
    )
    _write_immutable_artifact(config.output, artifact)
    return artifact


__all__ = [
    "DeliveryErrorLedgerConfig",
    "FIDELITY_SCHEMA_VERSION",
    "IDENTITY_SCHEMA_VERSION",
    "FidelityCensusArtifact",
    "FidelityCensusConfig",
    "RetailExpectedIdentity",
    "RetailIdentityAnalysisConfig",
    "RetailIdentityArtifact",
    "analyze_retail_identity_events",
    "build_fidelity_category_census",
    "build_delivery_error_ledger",
    "build_retail_identity_analysis",
    "count_fidelity_call",
    "extract_delivery_retry_evidence",
]
