# Copyright Sierra
"""Closed accounting tests for delivery ERROR exclusions."""

import pytest

from tau2.data_model.simulation import (
    DeliveryInfo,
    DeliveryUtteranceResult,
    JudgeOutcome,
)
from tau2.judges.delivery.error_exclusions import (
    EXPECTED_DELIVERY_ARGS,
    EXPECTED_DELIVERY_MAX_SEGMENTS,
    EXPECTED_DELIVERY_MODEL,
    EXPECTED_DELIVERY_PROMPT_VERSION,
    EXPECTED_DELIVERY_SAMPLE_RATE,
    DeliveryErrorCall,
    DeliveryErrorReason,
    DeliveryErrorRetryEvidence,
    build_delivery_error_exclusions,
    validate_delivery_error_exclusions,
)

SOURCE_A = "a" * 64
SOURCE_B = "b" * 64


def _delivery(*rows: DeliveryUtteranceResult) -> DeliveryInfo:
    return DeliveryInfo(
        num_judged=len(rows),
        num_errors=sum(row.outcome is JudgeOutcome.ERROR for row in rows),
        utterance_results=list(rows),
        judge_model=EXPECTED_DELIVERY_MODEL,
        judge_args=dict(EXPECTED_DELIVERY_ARGS),
        judge_prompt_version=EXPECTED_DELIVERY_PROMPT_VERSION,
        sample_rate=EXPECTED_DELIVERY_SAMPLE_RATE,
        max_segments=EXPECTED_DELIVERY_MAX_SEGMENTS,
        seed=42,
    )


def _row(idx: int, outcome: JudgeOutcome, summary: str = ""):
    return DeliveryUtteranceResult(
        utterance_idx=idx,
        expected_text=f"utterance {idx}",
        outcome=outcome,
        summary=summary,
    )


def _calls() -> tuple[DeliveryErrorCall, ...]:
    return (
        DeliveryErrorCall(
            source_results_sha256=SOURCE_A,
            simulation_id="sim-a",
            language="ko",
            domain="retail",
            system="openai_xhigh",
            task_id="63_ko_identity",
            trial=0,
            delivery=_delivery(
                _row(0, JudgeOutcome.PASS),
                _row(7, JudgeOutcome.PASS),
                _row(
                    9,
                    JudgeOutcome.ERROR,
                    "DeliveryJudgeResponse validation: Expecting value",
                ),
            ),
        ),
        DeliveryErrorCall(
            source_results_sha256=SOURCE_B,
            simulation_id="sim-b",
            language="zh",
            domain="airline",
            system="gemini_high",
            task_id="12_zh_identity",
            trial=0,
            delivery=_delivery(
                _row(0, JudgeOutcome.ERROR, "greeting timeout"),
                _row(3, JudgeOutcome.FAIL),
                _row(5, JudgeOutcome.ERROR, "request timed out after 120 seconds"),
            ),
        ),
    )


def test_build_ledger_closes_stored_scored_and_error_counts():
    artifact = build_delivery_error_exclusions(_calls())

    assert artifact.calls == 2
    assert artifact.source_results_files == 2
    assert artifact.source_task_subset_provenance == {"pinned": 2}
    assert artifact.legacy_null_task_subset_cells == ()
    assert artifact.stored_delivery_utterances == 6
    assert artifact.greetings == 2
    assert artifact.greeting_errors == 1
    assert artifact.eligible_pass_fail_utterances == 2
    assert artifact.excluded_error_utterances == 2
    assert [row.utterance_idx for row in artifact.rows] == [9, 5]
    assert artifact.by_reason == {"response_validation": 1, "timeout": 1}
    validate_delivery_error_exclusions(_calls(), artifact)


def test_max_tokens_reason_requires_and_preserves_retry_evidence():
    evidence = DeliveryErrorRetryEvidence(
        source_results_sha256=SOURCE_A,
        simulation_id="sim-a",
        utterance_idx=9,
        normalized_reason=DeliveryErrorReason.MAX_TOKENS_TRUNCATION,
        retry_count=8,
        evidence=("provider finish_reason=length on all observed retries",),
    )
    artifact = build_delivery_error_exclusions(_calls(), retry_evidence=[evidence])
    row = next(row for row in artifact.rows if row.simulation_id == "sim-a")

    assert row.normalized_reason is DeliveryErrorReason.MAX_TOKENS_TRUNCATION
    assert row.retry_count == 8
    assert row.retry_evidence == evidence.evidence
    assert artifact.retry_evidence_sha256 is not None
    validate_delivery_error_exclusions(_calls(), artifact)


def test_retry_evidence_cannot_name_pass_or_greeting():
    evidence = DeliveryErrorRetryEvidence(
        source_results_sha256=SOURCE_A,
        simulation_id="sim-a",
        utterance_idx=7,
        normalized_reason=DeliveryErrorReason.RESPONSE_VALIDATION,
        retry_count=1,
        evidence=("one validation failure",),
    )
    with pytest.raises(ValueError, match="does not name an ERROR slot"):
        build_delivery_error_exclusions(_calls(), retry_evidence=[evidence])


def test_validator_rejects_missing_error_key():
    artifact = build_delivery_error_exclusions(_calls())
    drifted = artifact.model_copy(update={"rows": artifact.rows[:-1]})

    with pytest.raises(ValueError, match="ledger key mismatch"):
        validate_delivery_error_exclusions(_calls(), drifted)


def test_builder_rejects_non_v5_contract():
    calls = list(_calls())
    bad_delivery = calls[0].delivery.model_copy(update={"judge_prompt_version": "v4"})
    calls[0] = calls[0].model_copy(update={"delivery": bad_delivery})

    with pytest.raises(ValueError, match="wrong v5 delivery contract"):
        build_delivery_error_exclusions(calls)


def test_builder_rejects_unaccounted_non_greeting_outcome():
    calls = list(_calls())
    deferred = _row(7, JudgeOutcome.DEFERRED)
    bad_delivery = _delivery(_row(0, JudgeOutcome.PASS), deferred)
    calls[0] = calls[0].model_copy(update={"delivery": bad_delivery})

    with pytest.raises(ValueError, match="neither scored nor ERROR"):
        build_delivery_error_exclusions(calls)
