# Copyright Sierra
"""Deterministic analyses over the corrected tau-Multilingual judge outputs."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import pytest

import tau2.judges.corrected_paper_analysis as corrected_analysis
from tau2.data_model.message import ToolCall
from tau2.data_model.simulation import (
    DeliveryFactorCheck,
    DeliveryFinding,
    DeliveryInfo,
    DeliveryUtteranceResult,
    JudgeOutcome,
)
from tau2.judges.cli import add_judges_args
from tau2.judges.corrected_paper_analysis import (
    RetailExpectedIdentity,
    analyze_retail_identity_events,
    count_fidelity_call,
    extract_delivery_retry_evidence,
)
from tau2.judges.corrected_paper_suite import FinalWindowExclusionRow
from tau2.judges.delivery.error_exclusions import (
    EXPECTED_DELIVERY_ARGS,
    EXPECTED_DELIVERY_MODEL,
    DeliveryErrorCall,
)
from tau2.metrics.call_timeline import ToolEvent


def _event(
    *,
    tool: str,
    arguments: dict,
    content: str | None,
    error: bool | None,
    paired: bool = True,
) -> ToolEvent:
    return ToolEvent(
        order_key=10,
        call=ToolCall(
            id=f"call-{tool}",
            name=tool,
            arguments=arguments,
            requestor="assistant",
        ),
        result_order_key=11 if paired else None,
        result_content=content,
        result_error=error,
    )


def test_identity_lookup_requires_exact_arguments_and_exact_returned_user():
    expected = RetailExpectedIdentity(
        user_id="jose_nunez_1234",
        first_name="José",
        last_name="Núñez",
        zip_code="12-345",
        email="Jose.Nunez@example.com",
    )
    attempts = analyze_retail_identity_events(
        [
            _event(
                tool="find_user_id_by_name_zip",
                arguments={
                    "first_name": "Jose",
                    "last_name": "Nunez",
                    "zip": "12345",
                },
                content="jose_nunez_1234",
                error=False,
            ),
            _event(
                tool="find_user_id_by_email",
                arguments={"email": "wrong@example.com"},
                content="jose_nunez_1234",
                error=False,
            ),
        ],
        expected,
    )

    assert attempts[0].argument_match is True
    assert attempts[0].result_match is True
    assert attempts[0].exact_expected_success is True
    assert attempts[1].argument_match is False
    assert attempts[1].result_match is True
    assert attempts[1].exact_expected_success is False


def test_identity_lookup_distinguishes_no_match_tool_error_and_unpaired():
    expected = RetailExpectedIdentity(
        user_id="a_b_1234",
        first_name="A",
        last_name="B",
        zip_code="12345",
        email="a@example.com",
    )
    attempts = analyze_retail_identity_events(
        [
            _event(
                tool="find_user_id_by_email",
                arguments={"email": expected.email},
                content="Error: User not found",
                error=True,
            ),
            _event(
                tool="find_user_id_by_email",
                arguments={"email": expected.email},
                content="Error: backend timeout",
                error=True,
            ),
            _event(
                tool="find_user_id_by_email",
                arguments={"email": expected.email},
                content=None,
                error=None,
                paired=False,
            ),
        ],
        expected,
    )

    assert [attempt.outcome.value for attempt in attempts] == [
        "no_match",
        "tool_error",
        "unpaired_result",
    ]
    assert not any(attempt.exact_expected_success for attempt in attempts)


def test_large_voice_tool_event_reader_extracts_only_paired_lookup_evidence(
    tmp_path: Path,
):
    path = tmp_path / "simulation.json"
    path.write_text(
        json.dumps(
            {
                "id": "sim",
                "ticks": [
                    {
                        "agent_tool_calls": [
                            {
                                "id": "lookup",
                                "name": "find_user_id_by_email",
                                "arguments": {"email": "a@example.com"},
                                "requestor": "assistant",
                            }
                        ],
                        "agent_tool_results": [],
                    },
                    {
                        "agent_tool_calls": [],
                        "agent_tool_results": [
                            {
                                "id": "lookup",
                                "content": "a_b_1234",
                                "error": False,
                            }
                        ],
                    },
                ],
            },
            indent=2,
        )
    )

    [event] = corrected_analysis._tool_events_from_path(path)

    assert event.order_key == 0
    assert event.result_order_key == 1
    assert event.call.name == "find_user_id_by_email"
    assert event.result_content == "a_b_1234"


def test_fidelity_census_excludes_greeting_and_deduplicates_per_category():
    tone_fail = DeliveryFactorCheck(
        id="tone_meaning_flip",
        category="tone",
        severity=3,
        outcome=JudgeOutcome.FAIL,
    )
    delivery = DeliveryInfo(
        utterance_results=[
            DeliveryUtteranceResult(
                utterance_idx=0,
                outcome=JudgeOutcome.FAIL,
                findings=[
                    DeliveryFinding(
                        axis="fidelity",
                        category="greeting_issue",
                        time_range="0.0-0.2",
                        severity=3,
                    )
                ],
            ),
            DeliveryUtteranceResult(
                utterance_idx=1,
                outcome=JudgeOutcome.FAIL,
                findings=[
                    DeliveryFinding(
                        axis="fidelity",
                        category="mispronunciation",
                        time_range="0.1-0.2",
                        severity=2,
                    ),
                    DeliveryFinding(
                        axis="fidelity",
                        category="mispronunciation",
                        time_range="0.3-0.4",
                        severity=2,
                    ),
                    DeliveryFinding(
                        axis="fidelity",
                        category="mispronunciation",
                        time_range="0.5-0.6",
                        severity=3,
                    ),
                ],
                factor_checks=[tone_fail],
            ),
            DeliveryUtteranceResult(
                utterance_idx=2,
                outcome=JudgeOutcome.PASS,
                factor_checks=[tone_fail],
            ),
            DeliveryUtteranceResult(
                utterance_idx=3,
                outcome=JudgeOutcome.ERROR,
                findings=[
                    DeliveryFinding(
                        axis="fidelity",
                        category="not_scored",
                        time_range="0.1-0.2",
                        severity=3,
                    )
                ],
            ),
        ]
    )
    exclusion = FinalWindowExclusionRow(
        sim_id="sim-1",
        language="ko",
        domain="retail",
        system="OpenAI minimal",
        utterance_idx=1,
        finding_index=0,
        severity=2,
        time_range="0.1-0.2",
        clip_duration_seconds=1.0,
        span_start_seconds=0.1,
        span_end_seconds=0.2,
    )
    encountered: set[tuple[str, int, int]] = set()
    applied: set[tuple[str, int, int]] = set()

    result = count_fidelity_call(
        simulation_id="sim-1",
        language="ko",
        domain="retail",
        system_slug="openai_minimal",
        source_results_path="/source/results.json",
        analysis_input_sha256="0" * 64,
        delivery=delivery,
        exclusions={("sim-1", 1, 0): exclusion},
        encountered_exclusions=encountered,
        applied_exclusions=applied,
    )

    assert result.stored_delivery_utterances == 4
    assert result.pinned_greetings_excluded == 1
    assert result.eligible_utterances == 2
    assert result.unscored_utterances == 1
    assert result.excluded_findings_encountered == 1
    assert result.excluded_findings_applied_to_eligible_utterances == 1
    assert result.raw_material_findings == 2
    assert result.category_utterances == {"mispronunciation": 1}
    assert result.material_failure_utterances == 1
    assert result.tone_meaning_flip_utterances == 2
    assert result.tone_only_failure_utterances == 1
    assert result.combined_standalone_failure_utterances == 2
    assert encountered == {("sim-1", 1, 0)}
    assert applied == encountered

    drifted = exclusion.model_copy(update={"language": "zh"})
    with pytest.raises(ValueError, match="metadata drifted"):
        count_fidelity_call(
            simulation_id="sim-1",
            language="ko",
            domain="retail",
            system_slug="openai_minimal",
            source_results_path="/source/results.json",
            analysis_input_sha256="0" * 64,
            delivery=delivery,
            exclusions={("sim-1", 1, 0): drifted},
            encountered_exclusions=set(),
            applied_exclusions=set(),
        )


def test_corrected_analysis_cli_verbs_have_explicit_inputs():
    parser = ArgumentParser()
    add_judges_args(parser)
    shared = [
        "--quality-workspace",
        "/tmp/quality",
        "--delivery-workspace",
        "/tmp/delivery",
        "--modal-workspace",
        "/tmp/modal",
        "--composed-workspace",
        "/tmp/composed",
    ]
    identity = parser.parse_args(
        [
            "corrected-paper-suite",
            "identity-lookup",
            "--repo-root",
            "/tmp/repo",
            *shared,
            "--output",
            "/tmp/identity.json",
        ]
    )
    assert identity.func.__name__ == "build_corrected_retail_identity_artifact"

    delivery_errors = parser.parse_args(
        [
            "corrected-paper-suite",
            "delivery-error-exclusions",
            "--canonical-repo-root",
            "/tmp/repo",
            *shared,
            "--retry-evidence",
            "/tmp/retries.json",
            "--output",
            "/tmp/delivery-errors.json",
        ]
    )
    assert delivery_errors.func.__name__ == "build_corrected_delivery_error_ledger"

    fidelity = parser.parse_args(
        [
            "corrected-paper-suite",
            "fidelity-census",
            "--canonical-repo-root",
            "/tmp/repo",
            *shared,
            "--final-window-sidecar",
            "/tmp/exclusions.csv",
            "--delivery-error-ledger",
            "/tmp/delivery-errors.json",
            "--output",
            "/tmp/fidelity.json",
        ]
    )
    assert fidelity.func.__name__ == "build_corrected_fidelity_census"


def test_retry_log_evidence_is_adjacent_hashed_and_counted(tmp_path):
    delivery = DeliveryInfo(
        num_judged=2,
        num_errors=1,
        utterance_results=[
            DeliveryUtteranceResult(
                utterance_idx=0,
                expected_text="hello",
                outcome=JudgeOutcome.PASS,
            ),
            DeliveryUtteranceResult(
                utterance_idx=7,
                expected_text="annyeong",
                outcome=JudgeOutcome.ERROR,
                summary="DeliveryJudgeResponse validation: Expecting value",
            ),
        ],
        judge_model=EXPECTED_DELIVERY_MODEL,
        judge_args=EXPECTED_DELIVERY_ARGS,
        judge_prompt_version="v5",
        sample_rate=1.0,
        max_segments=100,
    )
    call = DeliveryErrorCall(
        source_results_sha256="a" * 64,
        source_kind="corrected_composition",
        simulation_id="f3468fe4-81a5-40b9-8b1d-5e6375689561",
        language="ko",
        domain="retail",
        system="gemini_high",
        task_id="63_ko_identity",
        trial=0,
        delivery=delivery,
    )
    log = tmp_path / "retry.log"
    log.write_text(
        "Output might be incomplete due to token limit!\n"
        "delivery judge failed for utterance 7 "
        "(sim f3468fe4-81a5-40b9-8b1d-5e6375689561): invalid JSON\n"
        "Output might be incomplete due to token limit!\n"
        "delivery judge failed for utterance 7 "
        "(sim f3468fe4-81a5-40b9-8b1d-5e6375689561): invalid JSON\n"
    )

    [evidence] = extract_delivery_retry_evidence([call], (log,))

    assert evidence.retry_count == 2
    assert evidence.normalized_reason == "max_tokens_truncation"
    assert "finish_reason=length" in evidence.evidence[0]
    assert log.name in evidence.evidence[0]


def test_legacy_null_subset_exception_is_exactly_en_telecom_repeated_provider(
    tmp_path,
):
    canonical_root = tmp_path / "canonical"
    results_path = canonical_root / "english-telecom" / "results.json"

    assert (
        corrected_analysis._task_subset_provenance(
            None,
            language="en",
            domain="telecom",
            system_slug="openai_minimal",
            source_kind="canonical",
            canonical_root=canonical_root,
            results_path=results_path,
        )
        == "legacy_null_verified_against_pinned_xai"
    )

    with pytest.raises(ValueError, match="task subset metadata drifted"):
        corrected_analysis._task_subset_provenance(
            None,
            language="en",
            domain="telecom",
            system_slug="xai_provider_default",
            source_kind="canonical",
            canonical_root=canonical_root,
            results_path=results_path,
        )
    with pytest.raises(ValueError, match="task subset metadata drifted"):
        corrected_analysis._task_subset_provenance(
            None,
            language="en",
            domain="telecom",
            system_slug="openai_minimal",
            source_kind="corrected_composition",
            canonical_root=canonical_root,
            results_path=results_path,
        )
