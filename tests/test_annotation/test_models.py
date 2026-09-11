# Copyright Sierra
"""Row models: header generation, cell round-trips, lenient legacy parsing."""

import pytest
from pydantic import ValidationError

from tau2.annotation.models import (
    COLD_NOTES_LABEL,
    COLD_TRANSCRIPT_LABEL,
    AuditStatus,
    ColdLabel,
    ColdSheet,
    ColdSheetRow,
    ColdSidecarRow,
    FactorKeyRow,
    JudgePrecisionRow,
    PipelineVerdict,
    PrecisionVerdict,
    TranslationReviewRow,
    YesNo,
    norm_audit_status,
    norm_cold,
    norm_precision,
    norm_yes_no,
)
from tau2.data_model.simulation import JudgeOutcome


def test_precision_headers_reading_order_then_trace():
    headers = JudgePrecisionRow.headers()
    assert headers == [
        "What we're checking",
        "Exact phrase flagged",
        "Why the judge flagged it",
        "Full transcript (context)",
        "Real violation?",
        "Notes",
        "factor_id",
        "sim_id",
        "task_id",
        "language",
    ]
    assert JudgePrecisionRow.hidden_labels() == {
        "factor_id",
        "sim_id",
        "task_id",
        "language",
    }
    assert JudgePrecisionRow.dropdowns() == {
        "Real violation?": ["✅ yes", "❌ no", "🤔 unsure"]
    }
    # Widths come from the ColumnSpecs (the workbook builder reads these).
    assert JudgePrecisionRow.widths()["Full transcript (context)"] == 80


def test_cells_round_trip_with_enums():
    row = TranslationReviewRow(
        row_id=3,
        task_id="1_tl",
        field="task_instructions",
        english_original="Hello",
        translation="Hallo",
        meaning_ok=YesNo.YES,
        natural_ok=YesNo.NO,
        issue_type="naturalness",
        pipeline_verdict=PipelineVerdict.FLAGGED,
        pipeline_detail="meaning drift",
    )
    cells = row.to_cells()
    assert cells["meaning_ok (Y/N)"] == "Y"
    assert cells["natural_ok (Y/N)"] == "N"
    assert cells["pipeline_verdict"] == "flagged"
    assert cells["row_id"] == "3"
    back = TranslationReviewRow.from_cells(cells)
    assert back == row


def test_native_ok_and_pipeline_ok_semantics():
    unlabeled = TranslationReviewRow()
    assert unlabeled.native_ok is None and unlabeled.pipeline_ok is None
    # meaning yes + natural unlabeled counts as ok (port of legacy semantics).
    partial = TranslationReviewRow(meaning_ok=YesNo.YES)
    assert partial.native_ok is True
    bad = TranslationReviewRow(meaning_ok=YesNo.YES, natural_ok=YesNo.NO)
    assert bad.native_ok is False
    verified = TranslationReviewRow(pipeline_verdict=PipelineVerdict.VERIFIED)
    assert verified.pipeline_ok is True
    flagged = TranslationReviewRow(pipeline_verdict=PipelineVerdict.FLAGGED)
    assert flagged.pipeline_ok is False


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("yes", ColdLabel.YES),
        ("✅ yes", ColdLabel.YES),
        ("no", ColdLabel.NO),
        ("❌ no", ColdLabel.NO),
        ("no-opportunity", ColdLabel.NO_OPPORTUNITY),
        ("no opportunity", ColdLabel.NO_OPPORTUNITY),
        ("n/a", ColdLabel.NO_OPPORTUNITY),
        ("no-opp", ColdLabel.NO_OPPORTUNITY),
        ("no_opportunity", ColdLabel.NO_OPPORTUNITY),
        ("🤔 unsure", ColdLabel.UNSURE),
        ("", None),
        ("  ", None),
        (None, None),
    ],
)
def test_norm_cold_absorbs_legacy_forms(raw, expected):
    assert norm_cold(raw) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("✅ yes", PrecisionVerdict.REAL),
        ("❌ no", PrecisionVerdict.FALSE_ALARM),
        ("🤔 unsure", PrecisionVerdict.UNSURE),
        ("yes", PrecisionVerdict.REAL),
        ("real", PrecisionVerdict.REAL),
        ("false_alarm", PrecisionVerdict.FALSE_ALARM),
        ("", None),
    ],
)
def test_norm_precision_absorbs_legacy_forms(raw, expected):
    assert norm_precision(raw) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [("Y", YesNo.YES), ("y", YesNo.YES), ("N", YesNo.NO), ("no", YesNo.NO), ("", None)],
)
def test_norm_yes_no(raw, expected):
    assert norm_yes_no(raw) is expected


def test_unrecognized_labels_are_loud_not_dropped():
    with pytest.raises(ValueError):
        norm_cold("banana")
    with pytest.raises(ValueError):
        norm_precision("maybe?")
    with pytest.raises(ValidationError):
        JudgePrecisionRow.from_cells({"Real violation?": "banana"})


def test_judge_outcome_cell_absorbs_missing_sentinel():
    row = ColdSidecarRow.from_cells(
        {"sim_id": "s1", "factor_id": "f", "judge_outcome": "missing"}
    )
    assert row.judge_outcome is None
    assert (
        ColdSidecarRow.from_cells({"judge_outcome": "fail"}).judge_outcome
        is JudgeOutcome.FAIL
    )
    assert ColdSidecarRow.from_cells({"judge_outcome": ""}).judge_outcome is None


def test_audit_status_parsing_and_display():
    assert norm_audit_status("🔵 Auto — needs review") is AuditStatus.AUTO
    assert norm_audit_status("✅ Confirmed") is AuditStatus.CONFIRMED
    assert norm_audit_status("not_true") is AuditStatus.NOT_TRUE
    assert AuditStatus.options() == [
        "🔵 Auto — needs review",
        "✅ Confirmed",
        "❌ Not true",
        "🤔 Unsure",
    ]


def _cold_sheet() -> ColdSheet:
    header = "Formal address (aap vs tum)"
    key = FactorKeyRow(
        language="hi",
        factor_id="register_formality",
        category="register",
        nuance=header,
        native_does="consistent aap",
        non_native_ref="drifts to tum",
    )
    rows = [
        ColdSheetRow(
            transcript="tx-1",
            sim_id="s1",
            task_id="t1",
            language="hi",
            labels={header: ColdLabel.YES},
        ),
        ColdSheetRow(
            transcript="tx-2",
            sim_id="s2",
            task_id="t2",
            language="hi",
            labels={header: None},
        ),
    ]
    return ColdSheet(rows=rows, factors=[key])


def test_cold_sheet_headers_and_cells_round_trip():
    sheet = _cold_sheet()
    header = "Formal address (aap vs tum)"
    assert sheet.headers() == [
        COLD_TRANSCRIPT_LABEL,
        header,
        COLD_NOTES_LABEL,
        "sim_id",
        "task_id",
        "language",
    ]
    cells = sheet.to_cells()
    assert cells[0][header] == "yes"
    assert cells[1][header] == ""  # unlabeled renders blank
    back = ColdSheet.from_cells(cells, sheet.factors)
    assert back.rows[0].labels[header] is ColdLabel.YES
    assert back.rows[1].labels[header] is None
    assert back.rows[0].transcript == "tx-1"


def test_cold_sheet_melt_maps_headers_to_factor_ids():
    sheet = _cold_sheet()
    melted = list(sheet.melt())
    assert melted[0] == ("hi", "s1", "register_formality", ColdLabel.YES)
    # A header with no key mapping falls back to itself.
    orphan = ColdSheet(
        rows=[
            ColdSheetRow(
                transcript="tx", sim_id="s9", language="hi", labels={"mystery": None}
            )
        ],
        factors=[],
    )
    assert list(orphan.melt()) == [("hi", "s9", "mystery", None)]


# ---------------------------------------------------------------------------
# N/A on the 1-4 rubric cells (unable to evaluate; excluded from means)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("NA", "NA"),
        ("na", "NA"),
        ("N/A", "NA"),
        (" n/a ", "NA"),
        ("3", 3),
        (3, 3),
        ("", None),
        ("  ", None),
        (None, None),
    ],
)
def test_norm_likert_na(raw, expected):
    from tau2.annotation.models import norm_likert_na

    assert norm_likert_na(raw) == expected


def test_realism_row_na_cells_round_trip():
    from tau2.annotation.models import NA_SCORE, RealismRow

    row = RealismRow(speech_accuracy="n/a", voice_prosody_quality=3)
    assert row.speech_accuracy == NA_SCORE
    cells = row.to_cells()
    assert cells["speech_accuracy"] == "NA"
    assert cells["voice_prosody_quality"] == "3"
    assert cells["turn_taking_naturalness"] == ""  # blank = not yet rated
    assert RealismRow.from_cells(cells) == row


def test_likert_cell_junk_and_out_of_range_are_loud():
    from tau2.annotation.models import RealismRow

    with pytest.raises(ValidationError):
        RealismRow(speech_accuracy=5)
    with pytest.raises(ValidationError):
        RealismRow(speech_accuracy="great")


def test_rubric_dropdowns_offer_na():
    from tau2.annotation.models import (
        LIKERT_OPTIONS,
        REALISM_DIMENSION_IDS,
        RealismRow,
        VoiceReviewRow,
    )

    assert LIKERT_OPTIONS == ["1", "2", "3", "4", "NA"]
    for model in (RealismRow, VoiceReviewRow):
        dropdowns = model.dropdowns()
        for dim in REALISM_DIMENSION_IDS:
            assert dropdowns[dim] == LIKERT_OPTIONS


def test_caller_experience_cells_round_trip():
    from tau2.annotation.models import BreakingPoint, ExperienceFactor, VoiceReviewRow

    row = VoiceReviewRow(
        caller_experience="2",
        experience_breaking_point="tick",
        experience_breaking_tick="1450",
        experience_factors="latency, sim_multilingual_broken",
        primary_factor="latency",
        experience_notes="dead air after every request",
    )
    assert row.caller_experience == 2
    assert row.experience_breaking_point is BreakingPoint.TICK
    assert row.experience_breaking_tick == 1450
    assert row.experience_factors == [
        ExperienceFactor.LATENCY,
        ExperienceFactor.SIM_MULTILINGUAL_BROKEN,
    ]
    assert row.primary_factor is ExperienceFactor.LATENCY

    cells = row.to_cells()
    # The multi-select cell renders raw ids (displays contain commas).
    assert cells["experience_factors"] == "latency, sim_multilingual_broken"
    assert cells["caller_experience"] == "2"
    assert cells["experience_breaking_tick"] == "1450"
    assert VoiceReviewRow.from_cells(cells) == row


def test_caller_experience_blank_row_is_clean():
    from tau2.annotation.models import VoiceReviewRow

    row = VoiceReviewRow()
    assert row.caller_experience is None
    assert row.experience_breaking_tick is None  # blank ≠ tick 0
    assert row.experience_factors == []
    cells = row.to_cells()
    assert cells["experience_factors"] == ""
    assert VoiceReviewRow.from_cells(cells) == row


def test_caller_experience_validation_is_loud():
    from tau2.annotation.models import VoiceReviewRow

    # 1-4 scale, no NA sentinel.
    with pytest.raises(ValidationError):
        VoiceReviewRow(caller_experience=5)
    with pytest.raises(ValidationError):
        VoiceReviewRow(caller_experience="NA")
    # Unknown factor ids never pass silently.
    with pytest.raises(ValidationError):
        VoiceReviewRow(experience_factors="latency, vibes")
    # The primary factor must be among the selected factors.
    with pytest.raises(ValidationError, match="not among"):
        VoiceReviewRow(
            experience_factors="latency",
            primary_factor="transcription",
        )


def test_experience_factor_taxonomy_is_the_closed_catalog():
    """The taxonomy is the single source every other layer enumerates, so it
    is pinned here — ids and order both, since the ids are CSV values and the
    order is the rendered checkbox order."""
    from tau2.annotation.models import ExperienceFactor

    assert [m.value for m in ExperienceFactor] == [
        "transcription",
        "latency",
        "comprehension_logic",
        "repetition",
        "turn_taking",
        "unnatural_language",
        "social_offense",
        "intonation_delivery",
        "sim_multilingual_broken",
        "other",
    ]
    # Repetition names the caller-side symptom, not a diagnosis of the agent.
    assert ExperienceFactor.REPETITION.display == (
        "Had to repeat or re-explain (said the same thing twice, spelled a name again)"
    )
    # It is a plain factor: no notes elaboration is forced by selecting it.
    from tau2.annotation.models import EXPERIENCE_ELABORATE_FACTORS

    assert ExperienceFactor.REPETITION not in EXPERIENCE_ELABORATE_FACTORS


def test_repetition_coexists_with_its_causes_and_can_be_primary():
    """Repetition overlaps TRANSCRIPTION / COMPREHENSION_LOGIC by design: the
    symptom and its cause are selected together and either may be primary."""
    from tau2.annotation.models import ExperienceFactor, VoiceReviewRow

    symptom_primary = VoiceReviewRow(
        caller_experience="1",
        experience_breaking_point="overall",
        experience_factors="repetition, transcription, comprehension_logic",
        primary_factor="repetition",
        experience_notes="spelled the booking reference four times",
    )
    assert symptom_primary.experience_factors == [
        ExperienceFactor.REPETITION,
        ExperienceFactor.TRANSCRIPTION,
        ExperienceFactor.COMPREHENSION_LOGIC,
    ]
    assert symptom_primary.primary_factor is ExperienceFactor.REPETITION

    cells = symptom_primary.to_cells()
    assert cells["experience_factors"] == (
        "repetition, transcription, comprehension_logic"
    )
    # The single-select cell renders the display label (the multi-select cell
    # cannot — the displays contain commas).
    assert cells["primary_factor"] == ExperienceFactor.REPETITION.display
    assert VoiceReviewRow.from_cells(cells) == symptom_primary

    # Same pair, cause primary — the other correct answer.
    cause_primary = VoiceReviewRow(
        caller_experience="2",
        experience_breaking_point="overall",
        experience_factors="repetition, transcription",
        primary_factor="transcription",
        experience_notes="one mishearing of the name forced the whole redo",
    )
    assert cause_primary.primary_factor is ExperienceFactor.TRANSCRIPTION
    assert VoiceReviewRow.from_cells(cause_primary.to_cells()) == cause_primary

    # And repetition alone is a legitimate selection.
    alone = VoiceReviewRow(
        experience_factors=["repetition"], primary_factor="repetition"
    )
    assert alone.primary_factor is ExperienceFactor.REPETITION

    # The browser's CSV export writes RAW IDS in both cells (python's
    # to_cells writes the display label for the single-select); ingest of the
    # file annotators actually return must accept that shape too.
    browser_cells = dict(cells)
    browser_cells["primary_factor"] = "repetition"
    assert VoiceReviewRow.from_cells(browser_cells) == symptom_primary
