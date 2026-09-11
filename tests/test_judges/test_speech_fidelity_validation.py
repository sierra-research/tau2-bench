# Copyright Sierra
"""Pure contract and helper tests for Speech-fidelity validation normalization."""

import pytest
from pydantic import ValidationError

from tau2.judges.speech_fidelity_validation import (
    FROZEN_DELIVERY_JUDGE_ARGS,
    FROZEN_DELIVERY_JUDGE_MODEL,
    FROZEN_DELIVERY_PROMPT_REVISION,
    SPEECH_FIDELITY_ACCEPTANCE_POLICY,
    SPEECH_FIDELITY_LANGUAGES,
    BinaryMetrics,
    FrozenLanguageRubric,
    SpeechFidelityDataset,
    SpeechFidelityDatasetRow,
    SpeechFidelityFinding,
    SpeechFidelityPrompt,
    SpeechFidelityResultRow,
    StoredReferenceAlignment,
    ValidationLabel,
    ValidationLabelOrigin,
    ValidationSourceSet,
    _is_in_final_utterance_window_v1,
    _time_range_bounds_seconds_v1,
    classify_stored_reference_alignment,
    compute_binary_metrics,
    passes_acceptance_policy,
)


def _dataset_row(
    *,
    row_index: int = 0,
    simulation_id: str = "sim-1",
    label_origin: ValidationLabelOrigin = ValidationLabelOrigin.DIRECT_UTTERANCE,
    human_label: ValidationLabel = ValidationLabel.VIOLATION,
    reference_transcript: str = "the delivered words",
) -> SpeechFidelityDatasetRow:
    return SpeechFidelityDatasetRow(
        row_index=row_index,
        row_id=f"{row_index + 1:020x}",
        simulation_id=simulation_id,
        task_id="task-1",
        label_origin=label_origin,
        clip_id="clip-1",
        source_utterance_idx=row_index,
        agent_turn_index=row_index,
        agent_turn_id=f"agent-turn-{row_index:03d}",
        domain="retail",
        provider="openai",
        locale="es-ES",
        preceding_customer_text="customer words",
        reference_transcript=reference_transcript,
        source_interrupted=False,
        human_label=human_label,
    )


def _result_row(
    *,
    label_positive: bool,
    findings: list[SpeechFidelityFinding],
) -> SpeechFidelityResultRow:
    return SpeechFidelityResultRow(
        row_index=0,
        row_id="00000000000000000001",
        simulation_id="sim-1",
        source_utterance_idx=0,
        agent_turn_index=0,
        label_positive=label_positive,
        judge_raw_positive=bool(findings),
        judge_positive=any(
            not finding.excluded_by_final_second_filter for finding in findings
        ),
        clip_duration_seconds=5.0,
        stored_reference_preview="the delivered words",
        stored_reference_alignment=StoredReferenceAlignment.EXACT,
        findings=findings,
    )


def _finding(*, time_range: str, excluded: bool = False) -> SpeechFidelityFinding:
    return SpeechFidelityFinding(
        finding_index=0,
        axis="fidelity",
        category="pronunciation",
        time_range=time_range,
        issue="changed meaning",
        severity=2,
        confidence=0.9,
        excluded_by_final_second_filter=excluded,
    )


def test_stored_reference_alignment_accepts_exact_and_prefix_relations():
    assert (
        classify_stored_reference_alignment(
            "the   delivered\nwords", "the delivered words"
        )
        is StoredReferenceAlignment.EXACT
    )
    assert (
        classify_stored_reference_alignment(
            "the delivered words continue", "the delivered words"
        )
        is StoredReferenceAlignment.STORED_PREFIX
    )


def test_stored_reference_alignment_rejects_substantive_drift():
    with pytest.raises(ValueError, match="must equal or prefix"):
        classify_stored_reference_alignment("the delivered words", "different words")


@pytest.mark.parametrize(
    ("time_range", "duration", "expected"),
    [
        (None, 5.0, False),
        ("not a timestamp", 5.0, False),
        ("4.2", 5.0, True),
        ("00:01-00:03", 5.0, False),
        ("00:04-00:05", 5.0, True),
        ("00:07-00:08", 5.0, False),
    ],
)
def test_frozen_final_window_v1_cases(time_range, duration, expected):
    assert _is_in_final_utterance_window_v1(time_range, duration) is expected


def test_frozen_final_window_v1_point_and_range_parser():
    assert _time_range_bounds_seconds_v1("1:02.5") == (62.5, 62.5)
    assert _time_range_bounds_seconds_v1("00:03-00:05") == (3.0, 5.0)


def test_dataset_row_rejects_unsynthesized_stop_token():
    with pytest.raises(ValidationError, match="unsynthesized control token"):
        _dataset_row(reference_transcript="hello ###STOP###")


def test_projected_call_pass_must_be_a_negative_label():
    with pytest.raises(ValidationError, match="projected call-pass label"):
        _dataset_row(label_origin=ValidationLabelOrigin.PROJECTED_CALL_PASS)


def test_dataset_contract_recomputes_counts_and_rejects_duplicates():
    rows = [
        _dataset_row(),
        _dataset_row(
            row_index=1,
            simulation_id="sim-2",
            label_origin=ValidationLabelOrigin.PROJECTED_CALL_PASS,
            human_label=ValidationLabel.PASS,
        ),
    ]
    dataset = SpeechFidelityDataset(
        schema_version="tau-multi-speech-fidelity-dataset-v3",
        language="es",
        split="test",
        role="fixed_curated_validation",
        source_set=ValidationSourceSet.FIXED_CURATED,
        evaluation_level="utterance",
        grouping_unit="language+simulation_id+agent_turn_index",
        measure_id="speech_fidelity",
        runtime_factor_ids=["fidelity"],
        n_rows=2,
        n_positive=1,
        n_negative=1,
        n_calls=2,
        domain_counts={"retail": 2},
        provider_counts={"openai": 2},
        label_origin_counts={"direct_utterance": 1, "projected_call_pass": 1},
        rows=rows,
    )
    assert dataset.n_positive == dataset.n_negative == 1

    payload = dataset.model_dump(mode="json")
    payload["rows"][1]["simulation_id"] = "sim-1"
    payload["rows"][1]["agent_turn_index"] = 0
    payload["n_calls"] = 1
    with pytest.raises(ValidationError, match="duplicate delivered-utterance"):
        SpeechFidelityDataset.model_validate(payload)


def test_result_contract_recomputes_the_final_second_filter():
    retained = _result_row(
        label_positive=True,
        findings=[_finding(time_range="00:01-00:02")],
    )
    excluded = _result_row(
        label_positive=False,
        findings=[_finding(time_range="00:04-00:05", excluded=True)],
    )
    assert retained.judge_positive
    assert excluded.judge_raw_positive and not excluded.judge_positive

    payload = excluded.model_dump(mode="json")
    payload["findings"][0]["excluded_by_final_second_filter"] = False
    payload["judge_positive"] = True
    with pytest.raises(ValidationError, match="deterministic filter v1"):
        SpeechFidelityResultRow.model_validate(payload)


def test_binary_metrics_and_acceptance_policy_are_deterministic():
    rows = [
        _result_row(
            label_positive=True,
            findings=[_finding(time_range="00:01-00:02")],
        ),
        _result_row(label_positive=False, findings=[]),
        _result_row(
            label_positive=False,
            findings=[_finding(time_range="00:01-00:02")],
        ),
        _result_row(label_positive=True, findings=[]),
    ]
    metrics = compute_binary_metrics(rows)
    assert (metrics.tp, metrics.tn, metrics.fp, metrics.fn) == (1, 1, 1, 1)
    assert metrics.precision == metrics.recall == metrics.f1 == 0.5
    assert metrics.kappa == 0.0

    passing = BinaryMetrics(
        tp=8,
        tn=8,
        fp=2,
        fn=2,
        precision=0.8,
        recall=0.8,
        f1=0.8,
        kappa=0.6,
    )
    assert passes_acceptance_policy(passing, SPEECH_FIDELITY_ACCEPTANCE_POLICY)
    assert not passes_acceptance_policy(
        passing.model_copy(update={"kappa": 0.59}),
        SPEECH_FIDELITY_ACCEPTANCE_POLICY,
    )


def test_prompt_contract_requires_the_frozen_model_and_all_languages():
    rubrics = {
        language: FrozenLanguageRubric(
            display_name=language,
            source="pack",
            factors=[],
        )
        for language in SPEECH_FIDELITY_LANGUAGES
    }
    prompt = SpeechFidelityPrompt(
        schema_version="tau-multi-speech-fidelity-prompt-v1",
        prompt_version="v5",
        source_revision=FROZEN_DELIVERY_PROMPT_REVISION,
        model=FROZEN_DELIVERY_JUDGE_MODEL,
        model_args=FROZEN_DELIVERY_JUDGE_ARGS,
        system_prompt="system",
        user_prompt_template="user",
        output_shape="json",
        language_rubrics=rubrics,
    )
    assert prompt.model == FROZEN_DELIVERY_JUDGE_MODEL

    payload = prompt.model_dump(mode="json")
    payload["model"] = "different-model"
    with pytest.raises(ValidationError, match="judge model drifted"):
        SpeechFidelityPrompt.model_validate(payload)
