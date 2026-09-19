# Copyright Sierra
"""Typed exclusions for delivery-judge ERROR utterances.

Delivery judge failures are missing measurements, not fidelity failures.  This
module builds a closed ledger for those unscored utterances and validates the
complete accounting identity used by paper analyses:

``stored = one greeting per call + scored PASS/FAIL utterances + ledger ERRORs``.

The ledger is keyed by the source ``results.json`` hash in addition to the
simulation and utterance ids, so it cannot be silently reused after a source
run changes.  Max-token truncation is never inferred from malformed JSON alone;
it requires explicit retry evidence containing a provider ``finish_reason`` of
``length``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from typing import Annotated, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.data_model.simulation import DeliveryInfo, JudgeOutcome

DELIVERY_ERROR_EXCLUSIONS_SCHEMA_VERSION = "delivery-error-exclusions-v1"
EXPECTED_DELIVERY_MODEL = "gemini/gemini-3.1-pro-preview"
EXPECTED_DELIVERY_ARGS = {"temperature": 0.0, "max_tokens": 8192, "timeout": 120}
EXPECTED_DELIVERY_PROMPT_VERSION = "v5"
EXPECTED_DELIVERY_SAMPLE_RATE = 1.0
EXPECTED_DELIVERY_MAX_SEGMENTS = 100


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


EXPECTED_DELIVERY_ARGS_SHA256 = _sha256_json(EXPECTED_DELIVERY_ARGS)
EXPECTED_DELIVERY_CONTRACT_SHA256 = _sha256_json(
    {
        "model": EXPECTED_DELIVERY_MODEL,
        "model_args": EXPECTED_DELIVERY_ARGS,
        "prompt_version": EXPECTED_DELIVERY_PROMPT_VERSION,
        "sample_rate": EXPECTED_DELIVERY_SAMPLE_RATE,
        "max_segments": EXPECTED_DELIVERY_MAX_SEGMENTS,
    }
)


class DeliveryErrorReason(StrEnum):
    """Closed normalized reason for an unscored delivery utterance."""

    MAX_TOKENS_TRUNCATION = "max_tokens_truncation"
    RESPONSE_VALIDATION = "response_validation"
    TIMEOUT = "timeout"


class DeliveryErrorCall(BaseModel):
    """One source call and the metadata needed to build ledger rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_results_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="SHA-256 of the source results.json owning the call.",
        ),
    ]
    source_kind: Literal["canonical", "corrected_composition"] = "canonical"
    simulation_id: Annotated[str, Field(description="Stored simulation UUID.")]
    language: Annotated[str, Field(description="Benchmark language code.")]
    domain: Annotated[str, Field(description="Benchmark domain id.")]
    system: Annotated[str, Field(description="Stable paper system slug.")]
    task_subset_provenance: Literal[
        "pinned", "legacy_null_verified_against_pinned_xai"
    ] = "pinned"
    task_id: Annotated[str, Field(description="Localized task id.")]
    trial: Annotated[int, Field(ge=0, description="Stored trial index.")]
    delivery: Annotated[
        DeliveryInfo,
        Field(description="Typed v5 delivery result for this call."),
    ]


class DeliveryErrorRetryEvidence(BaseModel):
    """Optional externally observed retry evidence for one exact ERROR slot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_results_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Owning results hash."),
    ]
    simulation_id: Annotated[str, Field(description="Stored simulation UUID.")]
    utterance_idx: Annotated[
        int, Field(ge=0, description="Stored delivery utterance index.")
    ]
    normalized_reason: Annotated[
        DeliveryErrorReason,
        Field(description="Reason established by the retry evidence."),
    ]
    retry_count: Annotated[
        int, Field(ge=1, description="Observed failed retry attempts.")
    ]
    evidence: Annotated[
        tuple[str, ...],
        Field(
            min_length=1,
            description="Normalized evidence facts, without raw credentials/audio.",
        ),
    ]

    @model_validator(mode="after")
    def _max_tokens_requires_length_finish(self) -> "DeliveryErrorRetryEvidence":
        if (
            self.normalized_reason is DeliveryErrorReason.MAX_TOKENS_TRUNCATION
            and not any("finish_reason=length" in fact for fact in self.evidence)
        ):
            raise ValueError(
                "max_tokens_truncation requires finish_reason=length evidence"
            )
        return self


class DeliveryErrorExclusionRow(BaseModel):
    """One non-greeting delivery ERROR excluded from scored denominators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_results_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Owning results hash."),
    ]
    simulation_id: Annotated[str, Field(description="Stored simulation UUID.")]
    utterance_idx: Annotated[
        int, Field(gt=0, description="Non-greeting delivery utterance index.")
    ]
    language: Annotated[str, Field(description="Benchmark language code.")]
    domain: Annotated[str, Field(description="Benchmark domain id.")]
    system: Annotated[str, Field(description="Stable paper system slug.")]
    task_id: Annotated[str, Field(description="Localized task id.")]
    trial: Annotated[int, Field(ge=0, description="Stored trial index.")]
    judge_model: Annotated[str, Field(description="Exact delivery judge model id.")]
    judge_prompt_version: Annotated[
        str, Field(description="Exact delivery prompt version.")
    ]
    judge_args_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Canonical judge args hash."),
    ]
    judge_contract_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Canonical model/args/prompt/sampling contract hash.",
        ),
    ]
    raw_summary: Annotated[
        str, Field(description="Exact stored ERROR summary from the judge harness.")
    ]
    normalized_reason: Annotated[
        DeliveryErrorReason,
        Field(description="Closed reason for the missing measurement."),
    ]
    retry_count: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Observed failed retries, when external evidence exists.",
        ),
    ]
    retry_evidence: Annotated[
        tuple[str, ...],
        Field(
            default_factory=tuple,
            description="Normalized retry facts, empty when unavailable.",
        ),
    ]

    @property
    def key(self) -> tuple[str, str, int]:
        """Stable source-bound row key."""
        return self.source_results_sha256, self.simulation_id, self.utterance_idx

    @model_validator(mode="after")
    def _evidence_is_consistent(self) -> "DeliveryErrorExclusionRow":
        if bool(self.retry_evidence) != (self.retry_count is not None):
            raise ValueError("retry_count and retry_evidence must appear together")
        if (
            self.normalized_reason is DeliveryErrorReason.MAX_TOKENS_TRUNCATION
            and not any("finish_reason=length" in fact for fact in self.retry_evidence)
        ):
            raise ValueError(
                "max_tokens_truncation requires finish_reason=length evidence"
            )
        return self


class DeliveryErrorExclusionsArtifact(BaseModel):
    """Closed accounting artifact for one complete delivery-judged cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Annotated[str, Field(description="Artifact schema version.")]
    judge_model: Annotated[str, Field(description="Uniform delivery judge model.")]
    judge_prompt_version: Annotated[
        str, Field(description="Uniform delivery prompt version.")
    ]
    judge_args_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Uniform args hash.")
    ]
    judge_contract_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Uniform contract hash."),
    ]
    retry_evidence_sha256: Annotated[
        str | None,
        Field(
            default=None,
            pattern=r"^[0-9a-f]{64}$",
            description="Canonical hash of typed retry evidence, when supplied.",
        ),
    ]
    source_results_files: Annotated[
        int, Field(ge=0, description="Distinct source results files.")
    ]
    source_task_subset_provenance: Annotated[
        dict[str, int],
        Field(description="Distinct source files grouped by task-subset provenance."),
    ]
    legacy_null_task_subset_cells: Annotated[
        tuple[str, ...],
        Field(description="Exact cells whose legacy results omit subset metadata."),
    ]
    calls: Annotated[int, Field(ge=0, description="Calls covered by the ledger.")]
    stored_delivery_utterances: Annotated[
        int, Field(ge=0, description="All stored delivery utterance rows.")
    ]
    greetings: Annotated[int, Field(ge=0, description="One greeting per call.")]
    eligible_pass_fail_utterances: Annotated[
        int, Field(ge=0, description="Scored non-greeting PASS/FAIL rows.")
    ]
    excluded_error_utterances: Annotated[
        int, Field(ge=0, description="Unscored non-greeting ERROR rows.")
    ]
    greeting_errors: Annotated[
        int, Field(ge=0, description="ERROR greetings excluded before scoring.")
    ]
    by_reason: Annotated[
        dict[str, int], Field(description="Ledger rows grouped by reason.")
    ]
    by_language: Annotated[
        dict[str, int], Field(description="Ledger rows grouped by language.")
    ]
    by_system: Annotated[
        dict[str, int], Field(description="Ledger rows grouped by system.")
    ]
    rows: Annotated[
        tuple[DeliveryErrorExclusionRow, ...],
        Field(description="Source-bound ERROR exclusions."),
    ]

    @model_validator(mode="after")
    def _artifact_counts_close(self) -> "DeliveryErrorExclusionsArtifact":
        if self.schema_version != DELIVERY_ERROR_EXCLUSIONS_SCHEMA_VERSION:
            raise ValueError("unsupported delivery error exclusions schema")
        if len({row.key for row in self.rows}) != len(self.rows):
            raise ValueError("duplicate delivery error exclusion key")
        if self.excluded_error_utterances != len(self.rows):
            raise ValueError("delivery error exclusion row count drifted")
        if (
            sum(self.source_task_subset_provenance.values())
            != self.source_results_files
        ):
            raise ValueError("source task-subset provenance counts do not close")
        if sum(self.by_reason.values()) != len(self.rows):
            raise ValueError("delivery error reason counts do not close")
        if sum(self.by_language.values()) != len(self.rows):
            raise ValueError("delivery error language counts do not close")
        if sum(self.by_system.values()) != len(self.rows):
            raise ValueError("delivery error system counts do not close")
        if self.stored_delivery_utterances != (
            self.greetings
            + self.eligible_pass_fail_utterances
            + self.excluded_error_utterances
        ):
            raise ValueError("delivery utterance accounting does not close")
        return self


def _call_key(call: DeliveryErrorCall, utterance_idx: int) -> tuple[str, str, int]:
    return call.source_results_sha256, call.simulation_id, utterance_idx


def _reason_from_summary(summary: str) -> DeliveryErrorReason:
    lowered = summary.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return DeliveryErrorReason.TIMEOUT
    if any(
        token in lowered
        for token in ("validation", "json", "expecting value", "expecting ','")
    ):
        return DeliveryErrorReason.RESPONSE_VALIDATION
    raise ValueError(f"unrecognized delivery ERROR summary: {summary[:160]}")


def _validate_contract(call: DeliveryErrorCall) -> None:
    info = call.delivery
    if (
        info.judge_model != EXPECTED_DELIVERY_MODEL
        or info.judge_args != EXPECTED_DELIVERY_ARGS
        or info.judge_prompt_version != EXPECTED_DELIVERY_PROMPT_VERSION
        or info.sample_rate != EXPECTED_DELIVERY_SAMPLE_RATE
        or info.max_segments != EXPECTED_DELIVERY_MAX_SEGMENTS
    ):
        raise ValueError(f"wrong v5 delivery contract: {call.simulation_id}")
    if info.num_judged != len(info.utterance_results):
        raise ValueError(f"delivery num_judged drifted: {call.simulation_id}")
    if info.num_errors != sum(
        row.outcome is JudgeOutcome.ERROR for row in info.utterance_results
    ):
        raise ValueError(f"delivery num_errors drifted: {call.simulation_id}")


def build_delivery_error_exclusions(
    calls: Iterable[DeliveryErrorCall],
    *,
    retry_evidence: Iterable[DeliveryErrorRetryEvidence] = (),
) -> DeliveryErrorExclusionsArtifact:
    """Build and validate a complete non-greeting ERROR exclusion ledger."""
    call_rows = tuple(calls)
    retry_rows = tuple(retry_evidence)
    evidence_by_key = {
        (row.source_results_sha256, row.simulation_id, row.utterance_idx): row
        for row in retry_rows
    }
    if len(evidence_by_key) != len(retry_rows):
        raise ValueError("duplicate delivery retry evidence key")

    rows: list[DeliveryErrorExclusionRow] = []
    stored = 0
    greetings = 0
    greeting_errors = 0
    eligible = 0
    observed_call_keys: set[tuple[str, str]] = set()
    source_provenance: dict[str, tuple[str, str, str, str]] = {}
    for call in call_rows:
        call_key = (call.source_results_sha256, call.simulation_id)
        if call_key in observed_call_keys:
            raise ValueError(f"duplicate delivery call key: {call.simulation_id}")
        observed_call_keys.add(call_key)
        source_identity = (
            call.task_subset_provenance,
            call.language,
            call.domain,
            call.system,
        )
        prior_source_identity = source_provenance.setdefault(
            call.source_results_sha256, source_identity
        )
        if prior_source_identity != source_identity:
            raise ValueError("source results hash has inconsistent cell provenance")
        _validate_contract(call)
        results = call.delivery.utterance_results
        if not results or results[0].utterance_idx != 0:
            raise ValueError(f"delivery inventory lacks greeting: {call.simulation_id}")
        indices = [row.utterance_idx for row in results]
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"duplicate delivery utterance index: {call.simulation_id}"
            )
        stored += len(results)
        greetings += 1
        greeting_errors += results[0].outcome is JudgeOutcome.ERROR
        for position, result in enumerate(results):
            if position == 0:
                continue
            if result.outcome in {JudgeOutcome.PASS, JudgeOutcome.FAIL}:
                eligible += 1
                continue
            if result.outcome is not JudgeOutcome.ERROR:
                raise ValueError(
                    f"non-greeting delivery row is neither scored nor ERROR: "
                    f"{call.simulation_id}/{result.utterance_idx}"
                )
            key = _call_key(call, result.utterance_idx)
            evidence = evidence_by_key.get(key)
            reason = (
                evidence.normalized_reason
                if evidence is not None
                else _reason_from_summary(result.summary or "")
            )
            rows.append(
                DeliveryErrorExclusionRow(
                    source_results_sha256=call.source_results_sha256,
                    simulation_id=call.simulation_id,
                    utterance_idx=result.utterance_idx,
                    language=call.language,
                    domain=call.domain,
                    system=call.system,
                    task_id=call.task_id,
                    trial=call.trial,
                    judge_model=EXPECTED_DELIVERY_MODEL,
                    judge_prompt_version=EXPECTED_DELIVERY_PROMPT_VERSION,
                    judge_args_sha256=EXPECTED_DELIVERY_ARGS_SHA256,
                    judge_contract_sha256=EXPECTED_DELIVERY_CONTRACT_SHA256,
                    raw_summary=result.summary or "",
                    normalized_reason=reason,
                    retry_count=evidence.retry_count if evidence else None,
                    retry_evidence=evidence.evidence if evidence else (),
                )
            )

    unused_evidence = set(evidence_by_key) - {row.key for row in rows}
    if unused_evidence:
        raise ValueError(
            f"retry evidence does not name an ERROR slot: {sorted(unused_evidence)[:3]}"
        )
    rows.sort(key=lambda row: row.key)
    artifact = DeliveryErrorExclusionsArtifact(
        schema_version=DELIVERY_ERROR_EXCLUSIONS_SCHEMA_VERSION,
        judge_model=EXPECTED_DELIVERY_MODEL,
        judge_prompt_version=EXPECTED_DELIVERY_PROMPT_VERSION,
        judge_args_sha256=EXPECTED_DELIVERY_ARGS_SHA256,
        judge_contract_sha256=EXPECTED_DELIVERY_CONTRACT_SHA256,
        retry_evidence_sha256=(
            _sha256_json([row.model_dump(mode="json") for row in retry_rows])
            if retry_rows
            else None
        ),
        source_results_files=len({call.source_results_sha256 for call in call_rows}),
        source_task_subset_provenance=dict(
            sorted(Counter(row[0] for row in source_provenance.values()).items())
        ),
        legacy_null_task_subset_cells=tuple(
            sorted(
                f"{language}/{domain}/{system}"
                for provenance, language, domain, system in source_provenance.values()
                if provenance == "legacy_null_verified_against_pinned_xai"
            )
        ),
        calls=len(call_rows),
        stored_delivery_utterances=stored,
        greetings=greetings,
        eligible_pass_fail_utterances=eligible,
        excluded_error_utterances=len(rows),
        greeting_errors=greeting_errors,
        by_reason=dict(
            sorted(Counter(row.normalized_reason.value for row in rows).items())
        ),
        by_language=dict(sorted(Counter(row.language for row in rows).items())),
        by_system=dict(sorted(Counter(row.system for row in rows).items())),
        rows=tuple(rows),
    )
    validate_delivery_error_exclusions(call_rows, artifact)
    return artifact


def validate_delivery_error_exclusions(
    calls: Iterable[DeliveryErrorCall],
    artifact: DeliveryErrorExclusionsArtifact,
) -> None:
    """Fail closed unless ledger keys equal every non-greeting ERROR exactly."""
    call_rows = tuple(calls)
    by_call = {
        (call.source_results_sha256, call.simulation_id): call for call in call_rows
    }
    if len(by_call) != len(call_rows):
        raise ValueError("duplicate delivery call key")
    expected: dict[tuple[str, str, int], tuple[DeliveryErrorCall, object]] = {}
    stored = 0
    greetings = 0
    greeting_errors = 0
    eligible = 0
    for call in call_rows:
        _validate_contract(call)
        results = call.delivery.utterance_results
        if not results or results[0].utterance_idx != 0:
            raise ValueError(f"delivery inventory lacks greeting: {call.simulation_id}")
        indices = [row.utterance_idx for row in results]
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"duplicate delivery utterance index: {call.simulation_id}"
            )
        stored += len(results)
        greetings += 1
        greeting_errors += results[0].outcome is JudgeOutcome.ERROR
        for position, result in enumerate(results):
            if position == 0:
                continue
            if result.outcome in {JudgeOutcome.PASS, JudgeOutcome.FAIL}:
                eligible += 1
            elif result.outcome is JudgeOutcome.ERROR:
                expected[_call_key(call, result.utterance_idx)] = (call, result)
            else:
                raise ValueError("non-greeting delivery outcome is not scored or ERROR")

    observed = {row.key: row for row in artifact.rows}
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(
            f"delivery ERROR ledger key mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    for key, (call, result) in expected.items():
        row = observed[key]
        if (
            row.utterance_idx == 0
            or result.outcome is not JudgeOutcome.ERROR
            or row.language != call.language
            or row.domain != call.domain
            or row.system != call.system
            or row.task_id != call.task_id
            or row.trial != call.trial
            or row.raw_summary != (result.summary or "")
            or row.judge_model != EXPECTED_DELIVERY_MODEL
            or row.judge_prompt_version != EXPECTED_DELIVERY_PROMPT_VERSION
            or row.judge_args_sha256 != EXPECTED_DELIVERY_ARGS_SHA256
            or row.judge_contract_sha256 != EXPECTED_DELIVERY_CONTRACT_SHA256
        ):
            raise ValueError(f"delivery ERROR ledger row drifted: {key}")
        if row.normalized_reason is DeliveryErrorReason.MAX_TOKENS_TRUNCATION:
            if row.retry_count is None or not any(
                "finish_reason=length" in fact for fact in row.retry_evidence
            ):
                raise ValueError(f"unproved max-token truncation row: {key}")
        elif row.normalized_reason != _reason_from_summary(row.raw_summary):
            raise ValueError(f"delivery ERROR reason drifted: {key}")

    expected_counts = {
        "source_results_files": len({call.source_results_sha256 for call in call_rows}),
        "calls": len(call_rows),
        "stored_delivery_utterances": stored,
        "greetings": greetings,
        "eligible_pass_fail_utterances": eligible,
        "excluded_error_utterances": len(expected),
        "greeting_errors": greeting_errors,
    }
    for name, expected_value in expected_counts.items():
        if getattr(artifact, name) != expected_value:
            raise ValueError(f"delivery ERROR artifact {name} drifted")
    source_provenance: dict[str, tuple[str, str, str, str]] = {}
    for call in call_rows:
        identity = (
            call.task_subset_provenance,
            call.language,
            call.domain,
            call.system,
        )
        prior = source_provenance.setdefault(call.source_results_sha256, identity)
        if prior != identity:
            raise ValueError("source results hash has inconsistent cell provenance")
    expected_subset_counts = dict(
        sorted(Counter(row[0] for row in source_provenance.values()).items())
    )
    if artifact.source_task_subset_provenance != expected_subset_counts:
        raise ValueError("delivery ERROR artifact subset provenance drifted")
    expected_legacy_cells = tuple(
        sorted(
            f"{language}/{domain}/{system}"
            for provenance, language, domain, system in source_provenance.values()
            if provenance == "legacy_null_verified_against_pinned_xai"
        )
    )
    if artifact.legacy_null_task_subset_cells != expected_legacy_cells:
        raise ValueError("delivery ERROR artifact legacy cell inventory drifted")


__all__ = [
    "DELIVERY_ERROR_EXCLUSIONS_SCHEMA_VERSION",
    "DeliveryErrorCall",
    "DeliveryErrorExclusionRow",
    "DeliveryErrorExclusionsArtifact",
    "DeliveryErrorReason",
    "DeliveryErrorRetryEvidence",
    "build_delivery_error_exclusions",
    "validate_delivery_error_exclusions",
]
