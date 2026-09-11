# Copyright Sierra
"""Offline contract tests for the tau-multilingual human-annotation archive."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import tau2.judges.validation_archive as validation_archive_module
from tau2.judges.nativeness.factors import (
    enabled_deterministic_factor_ids_for,
    judge_factors_for,
)
from tau2.judges.validation_archive import (
    ARCHIVE_LANGUAGES,
    EXPECTED_MEASURES,
    PROMPT_FACTOR_MATRIX,
    BinaryLabel,
    EvaluationLevel,
    LabelOrigin,
    MeasureId,
    MetricRow,
    SourceSet,
    StoredReferenceAlignment,
    UtteranceAlignmentKind,
    ValidationGateStatus,
    ValidationRow,
    _read_csv,
    _resolve_archive_path,
    compute_metric_rows,
    read_validation_archive,
    validate_archive,
    validation_gate_status,
    write_validation_archive,
)

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "data/simulations/paper_runs/tau-multi/human_annotations"


def _row(
    row_id: str,
    measure_id: MeasureId,
    source_set: SourceSet,
    human: BinaryLabel,
    judge: BinaryLabel,
    *,
    level: EvaluationLevel | None = None,
    simulation_id: str | None = None,
) -> ValidationRow:
    evaluation_level = (
        level
        or {
            MeasureId.TOOL_USE: EvaluationLevel.CALL,
            MeasureId.NATURALNESS: EvaluationLevel.UTTERANCE,
        }[measure_id]
    )
    source_factors = {
        MeasureId.TOOL_USE: ["incorrect_tool_parameters"],
        MeasureId.NATURALNESS: ["natural_word_choice"],
    }[measure_id]
    utterance = evaluation_level is EvaluationLevel.UTTERANCE
    return ValidationRow(
        row_id=row_id,
        language="es",
        measure_id=measure_id,
        source_factor_ids=source_factors,
        source_sets=[source_set],
        evaluation_level=evaluation_level,
        label_origin=(
            LabelOrigin.DIRECT_UTTERANCE if utterance else LabelOrigin.COMBINED_CALL
        ),
        simulation_id=simulation_id or f"sim-{row_id}",
        task_id="task",
        utterance_id="agent-turn-000" if utterance else None,
        utterance_alignment=(UtteranceAlignmentKind.AGENT_TURN if utterance else None),
        agent_turn_index=0 if utterance else None,
        agent_turn_id="agent-turn-000" if utterance else None,
        agent_utterance="Hola" if utterance else None,
        human_label=human,
        judge_label=judge,
        agreement=human == judge,
    )


def test_checked_in_human_annotation_archive_reproduces_offline():
    report = validate_archive(ARCHIVE)
    assert report.call_validation_rows == 343
    assert report.utterance_validation_rows == 3258
    assert report.fce_rows == 180
    assert report.language_review_rows == 136
    assert report.call_experience_rows == 58
    assert report.metric_rows == 80
    assert report.files_verified == 104
    assert report.deterministic_measures == [
        "es:email_symbol_verbalization",
        "pt:email_symbol_verbalization",
        "ko:email_symbol_verbalization",
        "zh:email_symbol_verbalization",
    ]
    assert report.fce_recorded_errors == 104
    assert report.fce_user_simulator_errors == 5
    assert report.fce_infrastructure_errors == 0
    assert report.fce_quality_ratings == 1439
    assert report.fce_quality_mean == pytest.approx(3.0479499652536486)
    assert report.fce_backchannel_ratings == 179
    assert report.fce_backchannel_mean == pytest.approx(2.8603351955307263)

    validation_rows, _, index_rows = read_validation_archive(ARCHIVE)
    insufficient = {
        (row.language, row.measure_id)
        for row in index_rows
        if row.gate_status is ValidationGateStatus.INSUFFICIENT_POSITIVE_SUPPORT
    }
    assert insufficient == {
        ("pt", MeasureId.REGIONAL_CONSISTENCY),
        ("ko", MeasureId.NAME_ADDRESS_CONVENTIONS),
        ("zh", MeasureId.NAME_ADDRESS_CONVENTIONS),
    }
    assert all(
        row.gate_status
        in {
            ValidationGateStatus.PASSES,
            ValidationGateStatus.INSUFFICIENT_POSITIVE_SUPPORT,
        }
        for row in index_rows
    )
    assert not any(
        row.language == "ko" and row.measure_id is MeasureId.COUNTING_UNITS
        for row in index_rows
    )

    utterance_rows = [
        row
        for row in validation_rows
        if row.evaluation_level is EvaluationLevel.UTTERANCE
    ]
    assert not any(SourceSet.DEVELOPMENT in row.source_sets for row in utterance_rows)
    assert not {"source_row_ids", "source_artifacts", "human_note"} & set(
        ValidationRow.model_fields
    )
    assert not (ARCHIVE / "validations/provenance").exists()
    assert not (ARCHIVE / "user_sim_review/fce/source_manifest.json").exists()
    assert not (ARCHIVE / "misc/source_manifest.json").exists()


def test_prompt_archive_contains_only_frozen_contracts():
    prompt_archive = json.loads(
        (ARCHIVE / "validations/prompts.json").read_text(encoding="utf-8")
    )

    assert prompt_archive["schema_version"] == "tau-multi-prompt-archive-v4"
    assert set(prompt_archive) == {
        "schema_version",
        "languages",
        "retained_factors",
        "combined_naturalness",
        "email_symbols",
        "stored_prediction_prompts",
    }
    assert "counting_units" not in prompt_archive["retained_factors"]["ko"]

    frozen = next(
        contract
        for contract in prompt_archive["stored_prediction_prompts"]
        if contract["contract_id"] == "nativeness-v15-rubric-v19"
    )
    # Counting units was present in the original batched prompt, but its verdicts
    # are not retained as a paper measure for Korean.
    assert "counting_units" in {
        criterion["factor_id"] for criterion in frozen["nativeness_criteria"]["ko"]
    }


def test_archive_path_resolution_rejects_escape_and_symlink(tmp_path):
    root = tmp_path / "archive"
    nested = root / "nested"
    nested.mkdir(parents=True)
    regular = nested / "evidence.csv"
    regular.write_text("safe", encoding="utf-8")
    assert _resolve_archive_path(root, "nested/evidence.csv") == regular

    with pytest.raises(ValueError, match="stay relative"):
        _resolve_archive_path(root, "../outside.csv")
    with pytest.raises(ValueError, match="stay relative"):
        _resolve_archive_path(root, regular.as_posix())

    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="may not traverse a symlink"):
        _resolve_archive_path(root, "linked/evidence.csv")


@pytest.mark.parametrize(
    (
        "tp",
        "fn",
        "fp",
        "tn",
        "precision",
        "recall",
        "f1",
        "kappa",
        "expected",
    ),
    [
        (
            9,
            0,
            0,
            5,
            1.0,
            1.0,
            1.0,
            1.0,
            ValidationGateStatus.INSUFFICIENT_POSITIVE_SUPPORT,
        ),
        (10, 0, 0, 4, 1.0, 1.0, 1.0, None, ValidationGateStatus.PASSES),
        (10, 0, 0, 5, 1.0, 1.0, 1.0, 0.60, ValidationGateStatus.BELOW_METRIC_THRESHOLD),
        (
            10,
            0,
            0,
            5,
            0.75,
            1.0,
            0.86,
            0.7,
            ValidationGateStatus.BELOW_METRIC_THRESHOLD,
        ),
        (9, 3, 0, 5, 1.0, 0.75, 0.86, 0.7, ValidationGateStatus.BELOW_METRIC_THRESHOLD),
        (10, 0, 0, 5, 1.0, 1.0, 0.80, 0.7, ValidationGateStatus.BELOW_METRIC_THRESHOLD),
        (10, 0, 1, 5, 10 / 11, 1.0, 20 / 21, 0.7, ValidationGateStatus.PASSES),
    ],
)
def test_validation_gate_status_uses_frozen_evidence_rule(
    tp, fn, fp, tn, precision, recall, f1, kappa, expected
):
    metric = MetricRow(
        language="ko",
        measure_id=MeasureId.COUNTING_UNITS,
        source_set=SourceSet.UNIFIED,
        evaluation_level=EvaluationLevel.UTTERANCE,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        no_opportunity=0,
        errors=0,
        n=tp + fn + fp + tn,
        precision=precision,
        recall=recall,
        f1=f1,
        kappa=kappa,
    )
    assert validation_gate_status(metric) is expected


def test_korean_honorific_validation_is_utterance_level():
    rows, _, _ = read_validation_archive(ARCHIVE)
    factor_rows = [
        row
        for row in rows
        if row.language == "ko" and row.measure_id is MeasureId.HONORIFIC_AGREEMENT
    ]
    assert factor_rows
    assert all(row.evaluation_level is EvaluationLevel.UTTERANCE for row in factor_rows)
    metric = next(
        row
        for row in compute_metric_rows(factor_rows)
        if row.source_set is SourceSet.FIXED_CURATED
    )
    assert metric.evaluation_level is EvaluationLevel.UTTERANCE


def test_archive_nativeness_matrix_matches_runtime_language_packs():
    non_nativeness = {
        "incorrect_tool_parameters",
        "auth_arg_mismatch",
        "agent_caused_tool_error",
        "fidelity",
        "tone_meaning_flip",
    }
    for language in ARCHIVE_LANGUAGES:
        observed = {
            factor.id
            for factor in judge_factors_for(language)
            if factor.enabled and not factor.shadow
        } | enabled_deterministic_factor_ids_for(language)
        expected = set(PROMPT_FACTOR_MATRIX[language]) - non_nativeness
        assert observed == expected


def test_public_measure_matrix_contains_no_runtime_subtypes():
    public = {
        measure.value for measures in EXPECTED_MEASURES.values() for measure in measures
    }
    assert not public & {
        "incorrect_tool_parameters",
        "auth_arg_mismatch",
        "agent_caused_tool_error",
        "fidelity",
        "tone_meaning_flip",
    }


def test_metric_rows_include_only_level_compatible_unified_groups():
    rows = [
        _row(
            "row-000000000001",
            MeasureId.TOOL_USE,
            SourceSet.RECALL,
            BinaryLabel.VIOLATION,
            BinaryLabel.PASS,
        ),
        _row(
            "row-000000000002",
            MeasureId.TOOL_USE,
            SourceSet.PRECISION,
            BinaryLabel.VIOLATION,
            BinaryLabel.VIOLATION,
        ),
        _row(
            "row-000000000003",
            MeasureId.TOOL_USE,
            SourceSet.PRECISION,
            BinaryLabel.PASS,
            BinaryLabel.VIOLATION,
        ),
    ]
    metrics = compute_metric_rows(rows)
    unified = next(row for row in metrics if row.source_set is SourceSet.UNIFIED)
    assert (unified.tp, unified.fn, unified.fp, unified.tn) == (1, 1, 1, 0)
    assert unified.precision == 0.5
    assert unified.recall == 0.5
    assert unified.f1 == 0.5
    assert unified.kappa == pytest.approx(-0.5)


@pytest.mark.parametrize("source_set", [SourceSet.HELD_OUT, SourceSet.FIXED_CURATED])
def test_metric_rows_do_not_duplicate_fixed_data_as_unified(source_set):
    rows = [
        _row(
            "row-000000000021",
            MeasureId.NATURALNESS,
            source_set,
            BinaryLabel.VIOLATION,
            BinaryLabel.VIOLATION,
        ),
        _row(
            "row-000000000022",
            MeasureId.NATURALNESS,
            source_set,
            BinaryLabel.PASS,
            BinaryLabel.PASS,
        ),
    ]

    metrics = compute_metric_rows(rows)

    assert [row.source_set for row in metrics] == [source_set]


def test_validation_row_rejects_cross_level_measure():
    with pytest.raises(ValidationError, match="naturalness must be utterance-level"):
        _row(
            "row-000000000004",
            MeasureId.NATURALNESS,
            SourceSet.HELD_OUT,
            BinaryLabel.PASS,
            BinaryLabel.PASS,
            level=EvaluationLevel.CALL,
        )


def test_validation_row_rejects_retired_public_measure():
    payload = _row(
        "row-000000000005",
        MeasureId.NATURALNESS,
        SourceSet.HELD_OUT,
        BinaryLabel.PASS,
        BinaryLabel.PASS,
    ).model_dump(mode="json")
    payload["measure_id"] = "tone_meaning_flip"
    with pytest.raises(ValidationError, match="measure_id"):
        ValidationRow.model_validate(payload)


def test_writer_physically_separates_units_and_round_trips_lists(tmp_path):
    rows = [
        _row(
            "row-000000000006",
            MeasureId.TOOL_USE,
            SourceSet.RECALL,
            BinaryLabel.PASS,
            BinaryLabel.PASS,
        ),
        _row(
            "row-000000000007",
            MeasureId.TOOL_USE,
            SourceSet.PRECISION,
            BinaryLabel.VIOLATION,
            BinaryLabel.VIOLATION,
        ),
        _row(
            "row-000000000008",
            MeasureId.NATURALNESS,
            SourceSet.HELD_OUT,
            BinaryLabel.PASS,
            BinaryLabel.PASS,
        ),
    ]
    write_validation_archive(tmp_path, rows)
    validation_root = tmp_path / "validations"
    assert not (validation_root / "validation.csv").exists()
    assert not (validation_root / "metrics.csv").exists()
    call_leaf = validation_root / "call_level/tool_use/es"
    utterance_leaf = validation_root / "utterance_level/naturalness/es"
    call_rows = _read_csv(call_leaf / "validation.csv", ValidationRow)
    utterance_rows = _read_csv(utterance_leaf / "validation.csv", ValidationRow)
    assert [row.measure_id for row in call_rows] == [
        MeasureId.TOOL_USE,
        MeasureId.TOOL_USE,
    ]
    assert [row.measure_id for row in utterance_rows] == [MeasureId.NATURALNESS]
    assert call_rows[0].source_factor_ids == ["incorrect_tool_parameters"]
    loaded_rows, _, index_rows = read_validation_archive(tmp_path)
    assert loaded_rows == [*utterance_rows, *call_rows]
    assert {(row.measure_id, row.evaluation_level) for row in index_rows} == {
        (MeasureId.TOOL_USE, EvaluationLevel.CALL),
        (MeasureId.NATURALNESS, EvaluationLevel.UTTERANCE),
    }
    assert {row.validation_path for row in index_rows} == {
        "call_level/tool_use/es/validation.csv",
        "utterance_level/naturalness/es/validation.csv",
    }
    assert (call_leaf / "manifest.json").is_file()
    assert (utterance_leaf / "manifest.json").is_file()
    assert (validation_root / "manifest.json").is_file()


def test_writer_rejects_duplicate_source_units(tmp_path):
    rows = [
        _row(
            f"row-0000000000{index}",
            MeasureId.TOOL_USE,
            SourceSet.RECALL,
            BinaryLabel.PASS,
            BinaryLabel.PASS,
            simulation_id="same-sim",
        )
        for index in (9, 10)
    ]
    with pytest.raises(ValueError, match="duplicate validation source units"):
        write_validation_archive(tmp_path, rows)


def test_writer_rejects_duplicate_row_ids(tmp_path):
    rows = [
        _row(
            "row-000000000018",
            MeasureId.TOOL_USE,
            SourceSet.RECALL,
            BinaryLabel.PASS,
            BinaryLabel.PASS,
            simulation_id=f"sim-{index}",
        )
        for index in (1, 2)
    ]

    with pytest.raises(ValueError, match="duplicate validation row ids"):
        write_validation_archive(tmp_path, rows)


def test_writer_rejects_duplicate_unit_across_source_sets(tmp_path):
    first = _row(
        "row-000000000011",
        MeasureId.NATURALNESS,
        SourceSet.RECALL,
        BinaryLabel.PASS,
        BinaryLabel.PASS,
        simulation_id="same-sim",
    )
    second = _row(
        "row-000000000012",
        MeasureId.NATURALNESS,
        SourceSet.PRECISION,
        BinaryLabel.VIOLATION,
        BinaryLabel.VIOLATION,
        simulation_id="same-sim",
    )
    with pytest.raises(ValueError, match="duplicate validation source units"):
        write_validation_archive(tmp_path, [first, second])


def test_writer_rejects_development_rows(tmp_path):
    row = _row(
        "row-000000000015",
        MeasureId.NATURALNESS,
        SourceSet.DEVELOPMENT,
        BinaryLabel.PASS,
        BinaryLabel.PASS,
    )
    with pytest.raises(ValueError, match="may not contain development rows"):
        write_validation_archive(tmp_path, [row])


def test_reader_rejects_development_rows(tmp_path, monkeypatch):
    valid_row = _row(
        "row-000000000016",
        MeasureId.NATURALNESS,
        SourceSet.HELD_OUT,
        BinaryLabel.PASS,
        BinaryLabel.PASS,
    )
    development_row = valid_row.model_copy(
        update={"source_sets": [SourceSet.DEVELOPMENT]}
    )
    write_validation_archive(tmp_path, [valid_row])

    def fake_leaf(_validation_root, _partition):
        return [development_row], compute_metric_rows([development_row])

    monkeypatch.setattr(validation_archive_module, "_read_validation_leaf", fake_leaf)
    with pytest.raises(ValueError, match="may not contain development rows"):
        read_validation_archive(tmp_path)


def test_reader_rejects_leaf_hash_drift(tmp_path):
    row = _row(
        "row-000000000017",
        MeasureId.NATURALNESS,
        SourceSet.HELD_OUT,
        BinaryLabel.PASS,
        BinaryLabel.PASS,
    )
    write_validation_archive(tmp_path, [row])
    path = tmp_path / "validations/utterance_level/naturalness/es/validation.csv"
    path.write_text(path.read_text(encoding="utf-8") + "drift", encoding="utf-8")

    with pytest.raises(ValueError, match="byte-size mismatch"):
        read_validation_archive(tmp_path)


def test_writer_removes_stale_generated_leaves(tmp_path):
    naturalness = _row(
        "row-000000000019",
        MeasureId.NATURALNESS,
        SourceSet.HELD_OUT,
        BinaryLabel.PASS,
        BinaryLabel.PASS,
    )
    tool_use = _row(
        "row-000000000020",
        MeasureId.TOOL_USE,
        SourceSet.FIXED_CURATED,
        BinaryLabel.PASS,
        BinaryLabel.PASS,
    )
    write_validation_archive(tmp_path, [naturalness, tool_use])
    write_validation_archive(tmp_path, [naturalness])

    assert not (tmp_path / "validations/call_level").exists()
    rows, _, _ = read_validation_archive(tmp_path)
    assert rows == [naturalness]


def test_validation_row_accepts_exact_audio_segment():
    payload = _row(
        "row-000000000013",
        MeasureId.NATURALNESS,
        SourceSet.FIXED_CURATED,
        BinaryLabel.PASS,
        BinaryLabel.PASS,
    ).model_dump(mode="json")
    payload.update(
        {
            "utterance_id": "audio-segment-013",
            "utterance_alignment": "audio_segment",
            "agent_turn_index": None,
            "agent_turn_id": None,
            "audio_start_seconds": 1.25,
            "audio_end_seconds": 2.5,
        }
    )
    row = ValidationRow.model_validate(payload)
    assert row.utterance_alignment is UtteranceAlignmentKind.AUDIO_SEGMENT


def test_speech_row_requires_matching_stored_reference_preview():
    payload = _row(
        "row-000000000014",
        MeasureId.NATURALNESS,
        SourceSet.HELD_OUT,
        BinaryLabel.PASS,
        BinaryLabel.PASS,
    ).model_dump(mode="json")
    payload.update(
        {
            "measure_id": "speech_fidelity",
            "source_factor_ids": ["fidelity"],
            "source_sets": ["fixed_curated"],
            "agent_utterance": "the full delivered reference",
            "stored_reference_preview": "the full",
            "stored_reference_alignment": "stored_prefix",
        }
    )
    row = ValidationRow.model_validate(payload)
    assert row.stored_reference_alignment is StoredReferenceAlignment.STORED_PREFIX

    payload["stored_reference_preview"] = "different"
    with pytest.raises(ValidationError, match="reference relation"):
        ValidationRow.model_validate(payload)
