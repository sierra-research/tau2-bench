# Copyright Sierra
"""Judge-calibration packets: draw, blinding, adjudication, and agreement.

Everything runs offline against a synthetic dir-format Hindi run whose sims
carry COMPLETE stored verdicts for all four judge families (nativeness,
quality, delivery, semantic-via-artifact), real parseable stereo audio, and
speech ticks — so packet builds, CSV round-trips, and agreement math exercise
the real code paths with zero LLM calls.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from fixtures_runs import make_hi_results

from tau2.annotation.artifacts import (
    ArtifactManifest,
    CalibrationDecisionEntry,
    CalibrationPacketEntry,
)
from tau2.annotation.buckets import RETIRED_FACTORS, FactorDisposition
from tau2.annotation.calibration_draw import (
    CALIBRATION_SAMPLER_VERSION,
    CLEAN_CONTROL_STRATUM,
    COLD_UNVALIDATED_FACTORS,
    JUDGE_VISIBLE_BATCH_SUFFIX,
    RANDOM_DRAW_STRATUM,
    AdjudicationDrawConfig,
    CalibrationDrawConfig,
    CalibrationJudge,
    FrameCall,
    ProgressionSeverity,
    SemanticCallRecord,
    SemanticShadowScores,
    build_calibration_frame,
    calibration_instrument_cells,
    draw_adjudication_cohort,
    draw_calibration_cohort,
    judge_verdict_cells,
    load_calibration_frame,
    load_calibration_sidecar,
    load_semantic_artifact,
    stratum_of,
)
from tau2.annotation.calibration_report import (
    CALIBRATION_AGREEMENT_VERSION,
    build_calibration_agreement,
    collect_human_labels,
    collect_owner_decisions,
    render_calibration_agreement,
    tri_state_disagreement,
)
from tau2.annotation.loading import LoadedSim
from tau2.annotation.models import (
    CalibrationDecisionRow,
    ColdLabel,
    NativenessHumanLabelRow,
    RubricAnnotationRow,
    RubricAnswer,
)
from tau2.annotation.packets.calibration import (
    CALIBRATION_PACKET_VERSION,
    JUDGE_CALIBRATION_ADJUDICATION_KIND,
    JUDGE_CALIBRATION_KIND,
    JUDGE_ONLY_BANNER,
    JUDGE_VISIBLE_BANNER,
    CalibrationAdjudicationDrawOptions,
    CalibrationAdjudicationOptions,
    CalibrationPacketOptions,
    _sample_decision_rows,
    build_calibration_adjudication,
    build_calibration_adjudication_draw,
    build_calibration_packet,
)
from tau2.annotation.packets.forms import (
    CALIBRATION_DECISIONS_KIND,
    NATIVENESS_LABEL_KIND,
    RUBRIC_PACKET_KIND,
    ingest_browser_csv,
)
from tau2.annotation.packets.rubric import (
    calibration_sections_for,
    rubric_instrument_sha256,
    rubric_sections_for,
)
from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.message import AssistantMessage, Tick, UserMessage
from tau2.data_model.simulation import (
    DeliveryFactorCheck,
    DeliveryFinding,
    DeliveryInfo,
    DeliveryUtteranceResult,
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessJudgeUnitResult,
    QualityFactorCheck,
    QualityInfo,
)
from tau2.voice.utils.audio_io import save_wav_file
from tau2.voice.utils.audio_preprocessing import convert_to_stereo
from test_annotation.conftest import hi_sim, packet_config, write_family_csv

RATE = 8000
TICK_MS = 200
SAMPLES_PER_TICK = RATE * TICK_MS // 1000

WATERMARK = "SAMPLE — synthetic rater labels"

NATIVENESS_EVIDENCE_SENTINEL = "sentinel nativeness FAIL evidence zq81"
NATIVENESS_QUOTE_SENTINEL = "sentinel offending span zq82"
#: The pinned unit quote: a verbatim substring of the fixture's single agent
#: turn, so the adjudication page can highlight it inside the flagged turn.
NATIVENESS_UNIT_QUOTE = "मदद"
CALLER_CONTEXT_TEXT = "नमस्कार, मुझे सहायता चाहिए"

QUALITY_LLM_IDS = (
    "unnecessary_repetition",
    "agent_caused_tool_error",
    "auth_arg_mismatch",
    "incorrect_tool_parameters",
    "unnecessary_tool_call",
)
#: The deterministic checkers the fixture stores verdicts for (a subset of
#: the catalog is enough — cells only exist for STORED checks).
#: redundant_successful_tool_call is RETIRED from the catalog; an old run's
#: stored verdict for it must be ignored, exactly like an unknown id.
QUALITY_DETERMINISTIC_IDS = ("responsiveness", "redundant_successful_tool_call")
DETERMINISTIC_QUALITY_EVIDENCE = "instrumented dead-air violation zq83"

#: A retired Hindi register-checker verdict stored on every fixture call, to
#: prove that closed-catalog readers ignore checks preserved in older results.
REGISTER_CHECKER_EVIDENCE = "register-test: familiar customer address used"


# ---------------------------------------------------------------------------
# Synthetic frame calls (draw-only tests)
# ---------------------------------------------------------------------------


def _frame_call(index: int, strata: list[str]) -> FrameCall:
    return FrameCall(
        results_path="/runs/hi_run/results.json",
        sim_id=f"s{index:03d}",
        task_id=f"t{index:03d}",
        domain="airline",
        experiment="hi_run",
        has_audio=True,
        judges_present=[CalibrationJudge.QUALITY],
        strata=strata,
    )


def _drawn_keys(calls: list[FrameCall]) -> list[tuple[str, str]]:
    return sorted(call.key for call in calls if call.drawn)


def test_draw_is_deterministic_and_stratum_streams_are_independent():
    def stratum_a_calls() -> list[FrameCall]:
        return [_frame_call(i, ["quality:unnecessary_repetition"]) for i in range(10)]

    config = CalibrationDrawConfig(
        language="hi", n_calls=4, control_fraction=0.0, seed=7
    )

    first = stratum_a_calls()
    second = stratum_a_calls()
    draw_calibration_cohort(first, config)
    draw_calibration_cohort(second, config)
    assert _drawn_keys(first) == _drawn_keys(second)

    # Adding a NEW stratum's calls must not reshuffle stratum A's picks: each
    # stratum draws from its own seeded stream. With budget 5 over two strata
    # the round-robin gives A two draws; those must be exactly the two draws
    # A-alone yields under a budget of two.
    with_b = stratum_a_calls() + [
        _frame_call(100 + i, ["delivery:fidelity"]) for i in range(5)
    ]
    config_bigger = CalibrationDrawConfig(
        language="hi", n_calls=5, control_fraction=0.0, seed=7
    )
    draw_calibration_cohort(with_b, config_bigger)
    a_picks = {
        call.sim_id
        for call in with_b
        if call.drawn and call.strata == ["quality:unnecessary_repetition"]
    }
    assert len(a_picks) == 2
    baseline = stratum_a_calls()
    draw_calibration_cohort(
        baseline,
        CalibrationDrawConfig(language="hi", n_calls=2, control_fraction=0.0, seed=7),
    )
    assert {c.sim_id for c in baseline if c.drawn} == a_picks

    different_seed = stratum_a_calls()
    draw_calibration_cohort(
        different_seed,
        CalibrationDrawConfig(language="hi", n_calls=4, control_fraction=0.0, seed=8),
    )
    # Not asserting inequality of picks (4 of 10 can collide); the drawn_for
    # bookkeeping must be present either way.
    assert all(
        call.drawn_for == ["quality:unnecessary_repetition"]
        for call in different_seed
        if call.drawn
    )


def test_draw_covers_every_defect_stratum_and_fills_controls():
    calls = [
        # Two singleton defect strata plus one bigger stratum.
        _frame_call(1, ["nativeness:honorific_agreement"]),
        _frame_call(2, ["quality:unnecessary_tool_call"]),
        *[_frame_call(10 + i, ["quality:unnecessary_repetition"]) for i in range(6)],
        *[_frame_call(50 + i, [CLEAN_CONTROL_STRATUM]) for i in range(5)],
    ]
    config = CalibrationDrawConfig(
        language="hi", n_calls=8, control_fraction=0.25, seed=7
    )
    accounting = draw_calibration_cohort(calls, config)
    by_id = {row.stratum_id: row for row in accounting}

    # Every non-empty defect stratum is covered at least once.
    assert by_id["nativeness:honorific_agreement"].covered == 1
    assert by_id["quality:unnecessary_tool_call"].covered == 1
    assert by_id["quality:unnecessary_repetition"].covered >= 1
    # Defect budget 6, controls 2.
    assert sum(1 for c in calls if c.drawn and c.strata != [CLEAN_CONTROL_STRATUM]) == 6
    assert by_id[CLEAN_CONTROL_STRATUM].covered == 2
    # Achieved per-stratum inclusion probabilities are recorded.
    assert by_id["nativeness:honorific_agreement"].inclusion_probability == 1.0
    assert by_id[CLEAN_CONTROL_STRATUM].inclusion_probability == pytest.approx(2 / 5)
    assert by_id[CLEAN_CONTROL_STRATUM].judge is None


def test_severity_keyed_semantic_strata_keep_minors_from_swamping_majors():
    # question_quality flags most calls, ~87% of them minor: under one
    # flag-keyed stratum the two majors would almost never be drawn. The
    # severity split makes each a stratum, so coverage guarantees a major
    # even when minors dominate the frame 20:1.
    majors = [_frame_call(i, ["semantic:question_quality:major"]) for i in range(2)]
    minors = [
        _frame_call(100 + i, ["semantic:question_quality:minor"]) for i in range(40)
    ]
    calls = majors + minors
    config = CalibrationDrawConfig(
        language="es", n_calls=4, control_fraction=0.0, seed=7
    )
    accounting = draw_calibration_cohort(calls, config)
    by_id = {row.stratum_id: row for row in accounting}
    assert by_id["semantic:question_quality:major"].covered >= 1
    assert by_id["semantic:question_quality:minor"].covered >= 1
    assert by_id["semantic:question_quality:major"].judge is (CalibrationJudge.SEMANTIC)
    # The severity-keyed stratum id round-trips through the helper.
    assert (
        stratum_of(
            CalibrationJudge.SEMANTIC, "question_quality", ProgressionSeverity.MAJOR
        )
        == "semantic:question_quality:major"
    )


def test_short_pools_are_taken_whole_and_never_back_filled():
    calls = [
        _frame_call(1, ["quality:unnecessary_repetition"]),
        _frame_call(2, [CLEAN_CONTROL_STRATUM]),
    ]
    config = CalibrationDrawConfig(
        language="hi", n_calls=10, control_fraction=0.5, seed=7
    )
    draw_calibration_cohort(calls, config)
    # 1 defect + 1 control is everything the frame holds; nothing invented.
    assert _drawn_keys(calls) == sorted(c.key for c in calls)


def test_annotate_draw_default_is_the_owner_sized_30_calls():
    """Owner sizing 2026-08-20: the blind wave targets 30 calls per language.
    The count is a first-class draw parameter; this pins only its default."""
    config = CalibrationDrawConfig(language="hi")
    assert config.n_calls == 30
    assert config.control_fraction == 0.25
    assert config.n_controls + config.n_defects == 30
    # Explicit sizing still wins.
    assert CalibrationDrawConfig(language="hi", n_calls=150).n_calls == 150


def test_short_control_pool_degrades_gracefully_at_the_30_call_size():
    """The 240-corpus held ~0-1 judge-clean calls: at the 30-call size the
    control quota must still be taken-whole-and-logged, never back-filled
    from defect calls and never an error."""
    calls = [
        *[_frame_call(i, ["quality:unnecessary_repetition"]) for i in range(40)],
        _frame_call(100, [CLEAN_CONTROL_STRATUM]),  # the lone clean call
    ]
    config = CalibrationDrawConfig(language="pt", seed=7)  # defaults: 30 calls
    accounting = draw_calibration_cohort(calls, config)
    by_id = {row.stratum_id: row for row in accounting}
    assert by_id[CLEAN_CONTROL_STRATUM].covered == 1  # all that exists
    assert by_id["quality:unnecessary_repetition"].covered == config.n_defects
    drawn_controls = [
        c for c in calls if c.drawn and c.strata == [CLEAN_CONTROL_STRATUM]
    ]
    assert len(drawn_controls) == 1


# ---------------------------------------------------------------------------
# Adjudicate-mode per-factor draw (independent of the blind wave)
# ---------------------------------------------------------------------------


def test_random_rule_is_judge_blind_uniform_and_deterministic():
    """The recall arm's draw: strata never influence selection (judge-clean
    calls are reachable outside any control budget), the accounting leads
    with the one selective stratum (the whole frame), and short frames are
    taken whole."""

    def frame() -> list[FrameCall]:
        calls = [_frame_call(i, [CLEAN_CONTROL_STRATUM]) for i in range(8)]
        calls += [
            _frame_call(100 + i, ["quality:unnecessary_repetition"]) for i in range(2)
        ]
        return calls

    config = CalibrationDrawConfig(language="hi", rule="random", n_calls=6, seed=7)
    first, second = frame(), frame()
    accounting = draw_calibration_cohort(first, config)
    draw_calibration_cohort(second, config)
    assert _drawn_keys(first) == _drawn_keys(second)

    drawn = [call for call in first if call.drawn]
    assert len(drawn) == 6
    assert all(call.drawn_for == [RANDOM_DRAW_STRATUM] for call in drawn)
    # With 8 of 10 frame calls judge-clean, a uniform 6-call cohort must reach
    # clean calls — the enriched rule with control_fraction 0 never could.
    assert any(CLEAN_CONTROL_STRATUM in call.strata for call in drawn)

    head = accounting[0]
    assert head.stratum_id == RANDOM_DRAW_STRATUM
    assert head.frame_size == 10 and head.drawn == 6 and head.covered == 6
    assert head.inclusion_probability == pytest.approx(0.6)
    # The per-stratum rows are descriptive coverage, never draws of their own.
    assert all(row.drawn == 0 for row in accounting[1:])
    clean_row = next(
        row for row in accounting[1:] if row.stratum_id == CLEAN_CONTROL_STRATUM
    )
    assert clean_row.covered == sum(
        1 for call in drawn if CLEAN_CONTROL_STRATUM in call.strata
    )

    short = frame()
    draw_calibration_cohort(
        short, CalibrationDrawConfig(language="hi", rule="random", n_calls=99, seed=7)
    )
    assert sum(1 for call in short if call.drawn) == 10


def test_random_rule_frame_claims_no_must_cover():
    """A judge-blind draw cannot promise stratum coverage, so its frame must
    not claim (or loudly miss) must-cover cells — es has cold-unvalidated
    factors that the enriched frame would demand."""
    assert COLD_UNVALIDATED_FACTORS.get("es")
    config = CalibrationDrawConfig(language="es", rule="random", n_calls=4, seed=7)
    frame = build_calibration_frame([], config, results=[Path("/nowhere")])
    assert frame.must_cover == [] and frame.must_cover_empty == []
    assert frame.config.rule == "random"


def test_cold_coverage_omits_factors_absorbed_by_combined_naturalness():
    cold = {
        factor for factors in COLD_UNVALIDATED_FACTORS.values() for factor in factors
    }
    assert "translationese" not in cold
    assert "verb_morphology" not in cold
    assert COLD_UNVALIDATED_FACTORS["es"] == ("request_politeness",)
    assert "ko" not in COLD_UNVALIDATED_FACTORS


def test_per_factor_draw_caps_and_takes_short_strata_whole():
    calls = [
        *[_frame_call(i, ["quality:unnecessary_repetition"]) for i in range(25)],
        *[_frame_call(100 + i, ["delivery:fidelity"]) for i in range(2)],
        _frame_call(200, ["nativeness:honorific_agreement"]),
        *[_frame_call(300 + i, [CLEAN_CONTROL_STRATUM]) for i in range(5)],
    ]
    config = AdjudicationDrawConfig(language="hi", per_factor_cap=3, seed=7)
    accounting = draw_adjudication_cohort(calls, config)
    by_id = {row.stratum_id: row for row in accounting}

    # min(cap, frame_size) per factor: capped, never padded.
    assert by_id["quality:unnecessary_repetition"].covered == 3
    assert by_id["delivery:fidelity"].covered == 2
    assert by_id["nativeness:honorific_agreement"].covered == 1
    # Clean controls are accounted but never drawn.
    assert by_id[CLEAN_CONTROL_STRATUM].frame_size == 5
    assert by_id[CLEAN_CONTROL_STRATUM].covered == 0
    assert not any(c.drawn for c in calls if c.strata == [CLEAN_CONTROL_STRATUM])
    # Default cap is the owner-ruled 20.
    assert AdjudicationDrawConfig(language="hi").per_factor_cap == 20


def test_per_factor_draw_is_deterministic_and_counts_overlap_once():
    def frame() -> list[FrameCall]:
        overlap = [
            _frame_call(i, ["quality:unnecessary_repetition", "delivery:fidelity"])
            for i in range(4)
        ]
        rest = [_frame_call(100 + i, ["delivery:fidelity"]) for i in range(10)]
        return overlap + rest

    config = AdjudicationDrawConfig(language="hi", per_factor_cap=4, seed=7)
    first, second = frame(), frame()
    accounting = draw_adjudication_cohort(first, config)
    draw_adjudication_cohort(second, config)
    assert _drawn_keys(first) == _drawn_keys(second)

    by_id = {row.stratum_id: row for row in accounting}
    # The repetition stratum (alphabetically later) is served after fidelity;
    # both reach at least their min(cap, size) target. Repetition's members
    # ALL carry fidelity too, so its draws push fidelity's coverage above
    # fidelity's own target — over-coverage via overlap, never a double draw.
    assert by_id["delivery:fidelity"].covered >= 4
    assert by_id["quality:unnecessary_repetition"].covered == 4
    # Calls covering both strata are drawn ONCE, never twice: the cohort is
    # exactly the per-stratum draws, and each call was drawn for one stratum.
    drawn = [c for c in first if c.drawn]
    assert len(drawn) == sum(row.drawn for row in accounting)
    assert all(len(c.drawn_for) == 1 for c in drawn)


def test_per_factor_draw_streams_are_independent_of_the_blind_wave():
    """The per-factor draw must not consume (or perturb) the blind wave's
    per-stratum streams: the same frame drawn both ways yields each rule's
    own deterministic picks."""
    stratum = "quality:unnecessary_repetition"

    blind = [_frame_call(i, [stratum]) for i in range(10)]
    draw_calibration_cohort(
        blind,
        CalibrationDrawConfig(language="hi", n_calls=3, control_fraction=0.0, seed=7),
    )
    adjudication = [_frame_call(i, [stratum]) for i in range(10)]
    draw_adjudication_cohort(
        adjudication, AdjudicationDrawConfig(language="hi", per_factor_cap=3, seed=7)
    )
    # Re-running either rule reproduces its own picks exactly.
    blind_again = [_frame_call(i, [stratum]) for i in range(10)]
    draw_calibration_cohort(
        blind_again,
        CalibrationDrawConfig(language="hi", n_calls=3, control_fraction=0.0, seed=7),
    )
    assert _drawn_keys(blind) == _drawn_keys(blind_again)
    adjudication_again = [_frame_call(i, [stratum]) for i in range(10)]
    draw_adjudication_cohort(
        adjudication_again,
        AdjudicationDrawConfig(language="hi", per_factor_cap=3, seed=7),
    )
    assert _drawn_keys(adjudication) == _drawn_keys(adjudication_again)


def test_empty_must_cover_strata_are_recorded_on_the_frame():
    config = CalibrationDrawConfig(language="es", n_calls=4, seed=7)
    frame = build_calibration_frame([], config, results=[Path("/nowhere")])
    expected = [
        stratum_of(CalibrationJudge.NATIVENESS, factor)
        for factor in COLD_UNVALIDATED_FACTORS["es"]
    ]
    assert frame.must_cover == expected
    assert frame.must_cover_empty == expected
    assert frame.calls == []


# ---------------------------------------------------------------------------
# The calibration instrument (sections)
# ---------------------------------------------------------------------------


def test_calibration_sections_cover_the_llm_judged_surfaces_for_hi():
    # Instrument cut (2026-08-25): the EVA turn-taking checkers and the
    # semantic progression/conciseness suite are OUT; the generic audio
    # dimensions collapse into ONE fidelity question. What remains is every
    # LLM-judged surface plus the pack's deterministic nativeness checkers.
    sections = calibration_sections_for("hi")
    assert [section.id for section in sections] == [
        "interaction",
        "nativeness",
        "audio",
    ]
    by_id = {section.id: section for section in sections}
    interaction_ids = [q.id for q in by_id["interaction"].questions]
    assert "unnecessary_repetition" in interaction_ids
    for checker in (
        "responsiveness",
        "yielding",
        "monologue",
        "backchannel_selectivity",
    ):
        assert checker not in interaction_ids, checker
    # ONE generic fidelity question, guidance naming the folded dimensions;
    # hi's pack pins its regional accent via the regional_accent_consistency
    # delivery factor, so the pack question rides along verbatim and the
    # generic accent question folds away.
    audio_ids = [q.id for q in by_id["audio"].questions]
    assert audio_ids[:2] == ["fidelity", "intonation"]  # cells-v3 unfold
    assert "number_date_currency" not in audio_ids
    fidelity = by_id["audio"].questions[0]
    assert "mispronunciation" in fidelity.guidance
    assert "intonation" in fidelity.guidance
    assert audio_ids[-1] == "regional_accent_consistency"
    assert "accent" not in audio_ids
    accent = by_id["audio"].questions[-1]
    assert accent.answer_kind == "audio_severity"
    assert "hindustani" in accent.guidance.lower()


def test_native_control_en_has_no_nativeness_and_no_accent_question():
    """en declares no nativeness factors — it is the native control: the
    instrument drops the whole nativeness section (including the overall
    likert) and the generic accent question, keeping only the language-
    independent surfaces."""
    sections = calibration_sections_for("en")
    assert [section.id for section in sections] == [
        "interaction",
        "audio",
    ]
    by_id = {section.id: section for section in sections}
    audio_ids = [q.id for q in by_id["audio"].questions]
    assert "accent" not in audio_ids
    assert audio_ids == ["fidelity", "intonation"]
    all_ids = [q.id for s in sections for q in s.questions]
    assert "overall_nativeness" not in all_ids
    # The regular rubric packet drops the section the same way.
    assert [s.id for s in rubric_sections_for("en")] == ["interaction", "audio"]
    # A language WITH factors keeps its nativeness section and, when its pack
    # declares no regional_accent_consistency delivery factor, the generic
    # accent question.
    assert "nativeness" in [s.id for s in calibration_sections_for("ko")]


def test_calibration_instrument_declares_levels_and_severity_scales():
    """Owner rulings 2026-08-20: utterance-level turn selection on every
    question whose violation lives in agent-utterance content, and the
    two-level minor/major severity on exactly the semantic-source questions
    still on the instrument (the semantic section itself was cut 2026-08-25;
    its interaction twins keep their severity contract)."""
    sections = calibration_sections_for("hi")
    by_key = {(s.id, q.id): q for s in sections for q in s.questions}

    repetition = by_key[("interaction", "unnecessary_repetition")]
    assert repetition.evaluation_level == "utterance"
    assert repetition.severity_scale == "minor_major"
    tool_call = by_key[("interaction", "unnecessary_tool_call")]
    assert tool_call.evaluation_level == "call"
    assert tool_call.severity_scale == "minor_major"
    # Tool-mechanics factors stay call-level, no severity.
    for factor_id in (
        "agent_caused_tool_error",
        "auth_arg_mismatch",
        "incorrect_tool_parameters",
    ):
        question = by_key[("interaction", factor_id)]
        assert question.evaluation_level == "call", factor_id
        assert question.severity_scale == "none", factor_id
    # Overall likert questions stay call-level.
    assert by_key[("interaction", "overall_interaction_quality")].evaluation_level == (
        "call"
    )
    assert by_key[("nativeness", "overall_nativeness")].evaluation_level == "call"


# ---------------------------------------------------------------------------
# A four-sim voice run with verdicts from all judge families
# ---------------------------------------------------------------------------


def _speech_tick(tick_id: int, utterance_id: str, text: str) -> Tick:
    return Tick(
        tick_id=tick_id,
        timestamp="2026-01-01T00:00:00",
        tick_duration_seconds=TICK_MS / 1000,
        agent_chunk=AssistantMessage(
            role="assistant",
            content=text,
            contains_speech=True,
            utterance_ids=[utterance_id],
        ),
    )


def _mono(values: list[int]) -> AudioData:
    samples = np.concatenate(
        [np.full(SAMPLES_PER_TICK, value, dtype=np.int16) for value in values]
    )
    return AudioData(
        data=samples.tobytes(),
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=RATE, channels=1
        ),
    )


def _quality_info(fails: tuple[str, ...]) -> QualityInfo:
    checks = [
        QualityFactorCheck(
            id=factor_id,
            category="workflow",
            severity=1,
            evaluator="llm",
            outcome=(JudgeOutcome.FAIL if factor_id in fails else JudgeOutcome.PASS),
            evidence="repeated the refund terms verbatim"
            if factor_id in fails
            else None,
        )
        for factor_id in QUALITY_LLM_IDS
    ]
    # Deterministic checkers are calibrated too (every interaction judge has
    # a presence in the packets); FAIL only when planted so clean controls
    # stay clean.
    checks.extend(
        QualityFactorCheck(
            id=factor_id,
            category="responsiveness",
            severity=1,
            evaluator="deterministic",
            outcome=(JudgeOutcome.FAIL if factor_id in fails else JudgeOutcome.PASS),
            evidence=DETERMINISTIC_QUALITY_EVIDENCE if factor_id in fails else None,
        )
        for factor_id in QUALITY_DETERMINISTIC_IDS
    )
    return QualityInfo(
        rubric_version="quality-rubric-test",
        metrics_version="metrics-test",
        factor_checks=checks,
    )


def _delivery_info(*, fidelity_finding: bool, digit_fail: bool) -> DeliveryInfo:
    findings = (
        [
            DeliveryFinding(
                axis="fidelity",
                category="mispronunciation",
                severity=2,
                issue="digits read as one number",
            )
        ]
        if fidelity_finding
        else []
    )
    return DeliveryInfo(
        num_judged=1,
        utterance_results=[
            DeliveryUtteranceResult(
                utterance_idx=0,
                outcome=JudgeOutcome.FAIL if findings else JudgeOutcome.PASS,
                findings=findings,
                severity=2 if findings else 0,
                flag_for_review=bool(findings),
            )
        ],
        factor_checks=[
            # A verdict stored by an OLD run for a factor the pack no longer
            # declares (digit_readout was retired to the generic audio
            # taxonomy) must be ignored, exactly like an unknown id.
            DeliveryFactorCheck(
                id="digit_readout",
                category="delivery",
                severity=2,
                outcome=JudgeOutcome.FAIL if digit_fail else JudgeOutcome.PASS,
                evidence="read 1-2-3 as one hundred twenty-three"
                if digit_fail
                else None,
            ),
            # Unknown factor ids must be ignored (closed pack catalog).
            DeliveryFactorCheck(
                id="not_a_pack_factor",
                category="delivery",
                severity=1,
                outcome=JudgeOutcome.FAIL,
            ),
        ],
        rubric_source="pack",
        language="hi",
        judge_model="fake-delivery-judge",
        judge_prompt_version="delivery-prompt-test",
    )


def _voice_sim(
    sim_id: str,
    task_id: str,
    *,
    fail_factor=None,
    quality_fails: tuple[str, ...] = (),
    fidelity_finding: bool = False,
    digit_fail: bool = False,
    register_fail: bool = False,
):
    sim = hi_sim(sim_id, task_id, fail_factor=fail_factor)
    # Preserve one stored verdict for a retired factor. Closed-catalog readers
    # must ignore it even when an older result still carries the check.
    sim.nativeness_info.factor_checks.append(
        NativenessFactorCheck(
            id="register_formality",
            category="register",
            severity=3,
            outcome=(JudgeOutcome.FAIL if register_fail else JudgeOutcome.PASS),
            evidence=REGISTER_CHECKER_EVIDENCE if register_fail else None,
        )
    )
    # The shared fixture's FAIL quote is a real hi-pack rubric example, which
    # the (legitimately rendered) pack guidance would collide with in the
    # blinding assertions — replace it with unique sentinel strings. All hi
    # judge factors are utterance-level, so the FAIL also carries the exact
    # violated unit (turn 0) the adjudication page must pin.
    for check in sim.nativeness_info.factor_checks:
        if check.outcome is JudgeOutcome.FAIL:
            check.evidence = NATIVENESS_EVIDENCE_SENTINEL
            check.quote = NATIVENESS_QUOTE_SENTINEL
            check.unit_results = [
                NativenessJudgeUnitResult(
                    unit_index=0,
                    opportunity=True,
                    violated=True,
                    severity=3,
                    reasoning=NATIVENESS_EVIDENCE_SENTINEL,
                    quote=NATIVENESS_UNIT_QUOTE,
                )
            ]
    sim.ticks = [
        Tick(
            tick_id=0,
            timestamp="2026-01-01T00:00:00",
            tick_duration_seconds=TICK_MS / 1000,
            user_chunk=UserMessage(
                role="user", content=CALLER_CONTEXT_TEXT, contains_speech=True
            ),
        ),
        _speech_tick(1, "u1", "नमस्ते, "),
        _speech_tick(2, "u1", "मैं मदद करूँ?"),
    ]
    sim.quality_info = _quality_info(quality_fails)
    sim.delivery_info = _delivery_info(
        fidelity_finding=fidelity_finding, digit_fail=digit_fail
    )
    return sim


RUN_NAME = "hi_calib_run"


def make_calibration_run(tmp_path: Path) -> Path:
    """A four-sim dir-format run: one defect-rich call (s1), one extra defect
    call (s4), and two judge-clean controls (s2, s3), all with real audio."""
    sims = [
        _voice_sim(
            "s1",
            "t1",
            fail_factor="natural_word_choice",
            quality_fails=("unnecessary_repetition", "responsiveness"),
            fidelity_finding=True,
            digit_fail=True,
            register_fail=True,
        ),
        _voice_sim("s2", "t2"),
        _voice_sim("s3", "t3"),
        _voice_sim("s4", "t4", quality_fails=("unnecessary_tool_call",)),
    ]
    run_dir = make_hi_results(tmp_path, sims, name=RUN_NAME)
    user = _mono([30000, 30000, 30000])
    agent = _mono([900, 1000, 1100])
    for sim in sims:
        audio_dir = (
            run_dir / "tasks" / f"task_{sim.task_id}" / f"sim_{sim.id}" / "audio"
        )
        audio_dir.mkdir(parents=True, exist_ok=True)
        save_wav_file(convert_to_stereo(user, agent), audio_dir / "both.wav")
    return run_dir


def test_recall_corpus_materializes_the_drawn_subset(tmp_path):
    """The recall-arm verb: a seeded judge-blind cohort becomes a standalone
    dir-format corpus (subset results + working audio links) that judges and
    packet builds consume unchanged, with the frame inside as provenance and
    re-materialization refused."""
    from tau2.annotation.recall_corpus import (
        RECALL_FRAME_NAME,
        RecallCorpusOptions,
        build_recall_corpus,
    )
    from tau2.data_model.simulation import Results
    from tau2.judges.delivery.disk_audio import find_both_wav

    run_dir = make_calibration_run(tmp_path)
    out = tmp_path / "recall_corpus"
    options = RecallCorpusOptions(
        results=[run_dir], language="hi", n_calls=2, seed=7, out_dir=out
    )
    build = build_recall_corpus(options)
    assert build.n_calls == 2 and build.cells == {RUN_NAME: 2}
    assert build.frame_path == out / RECALL_FRAME_NAME

    frame = load_calibration_frame(build.frame_path)
    assert frame.config.rule == "random"
    drawn = {call.sim_id for call in frame.drawn_calls()}

    cell_dir = out / RUN_NAME
    results = Results.load(cell_dir)
    assert {sim.id for sim in results.simulations} == drawn
    for sim in results.simulations:
        assert find_both_wav(cell_dir, sim) is not None

    with pytest.raises(FileExistsError):
        build_recall_corpus(options)


def write_semantic_artifact(path: Path, run_dir: Path) -> Path:
    """A conversation-judge artifact (PR #747 shape: HEADLINE
    progression_dimensions + conciseness_turns per call) for s1 (defects) and
    s2 (clean). s1's source is recorded as the results DIR and s2's as the
    results.json file — the reader must normalize both onto the same join key.
    s1's turn ratings [1,1,1,2,2] average to a normalized 0.2 (flagged) and
    s2's [3,3,3,3,2] to 0.9 (clean). s1 carries one MAJOR progression flag
    (information_loss, rating 1) and one MINOR (question_quality, rating 2),
    so the draw's severity-keyed semantic strata are both exercised."""
    results_json = str((run_dir / "results.json").resolve())

    def turn(turn_id: int, rating: int, evidence: str = "") -> dict:
        return {
            "turn_id": turn_id,
            "rating": rating,
            "failure_modes": ["restating"] if rating < 3 else [],
            "evidence": evidence,
        }

    payload = {
        "calls": [
            {
                "source": str(run_dir.resolve()),
                "sim_id": "s1",
                "task_id": "t1",
                "domain": "airline",
                "language": "hi",
                "input_sha256": "0" * 64,
                "progression_dimensions": [
                    {
                        "name": "information_loss",
                        "flagged": True,
                        "rating": 1,
                        "evidence": "asked for the booking code twice",
                    },
                    {
                        "name": "question_quality",
                        "flagged": True,
                        "rating": 2,
                        "evidence": "asked a broad question already answered",
                    },
                    {
                        "name": "redundant_statements",
                        "flagged": False,
                        "rating": 3,
                        "evidence": "",
                    },
                ],
                "conciseness_turns": [
                    turn(1, 1, "restated the full itinerary"),
                    turn(3, 1, "repeated the fare rules verbatim"),
                    turn(5, 1),
                    turn(7, 2),
                    turn(9, 2),
                ],
                "shadow_scores": {"label": "uncalibrated-shadow"},
            },
            {
                "source": results_json,
                "sim_id": "s2",
                "task_id": "t2",
                "domain": "airline",
                "language": "hi",
                "input_sha256": "0" * 64,
                "progression_dimensions": [
                    {
                        "name": "information_loss",
                        "flagged": False,
                        "rating": 3,
                        "evidence": "",
                    }
                ],
                "conciseness_turns": [
                    turn(1, 3),
                    turn(3, 3),
                    turn(5, 3),
                    turn(7, 3),
                    turn(9, 2),
                ],
                "shadow_scores": {"label": "uncalibrated-shadow"},
            },
        ]
    }
    path.write_text(json.dumps(payload))
    return path


def test_judge_verdict_cells_normalize_all_families_and_filter_closed_catalogs(
    tmp_path,
):
    run_dir = make_calibration_run(tmp_path)
    artifact = write_semantic_artifact(tmp_path / "experience.json", run_dir)
    semantic = load_semantic_artifact(artifact).by_key()
    key = (str((run_dir / "results.json").resolve()), "s1")
    assert key in semantic  # dir-form source normalized onto results.json

    sim = _voice_sim(
        "s1",
        "t1",
        fail_factor="natural_word_choice",
        quality_fails=("unnecessary_repetition", "responsiveness"),
        fidelity_finding=True,
        digit_fail=True,
        register_fail=True,
    )
    by_key = {
        (cell.judge, cell.factor_id): cell
        for cell in judge_verdict_cells(sim, "hi", semantic[key])
    }
    cells = {key: cell.outcome for key, cell in by_key.items()}
    assert cells[(CalibrationJudge.NATIVENESS, "natural_word_choice")] is (
        JudgeOutcome.FAIL
    )
    # Retired stored factors are absent; deterministic quality checks still
    # have cells like any other.
    assert (CalibrationJudge.NATIVENESS, "register_formality") not in cells
    assert cells[(CalibrationJudge.QUALITY, "responsiveness")] is JudgeOutcome.FAIL
    assert cells[(CalibrationJudge.QUALITY, "unnecessary_repetition")] is (
        JudgeOutcome.FAIL
    )
    assert cells[(CalibrationJudge.QUALITY, "unnecessary_tool_call")] is (
        JudgeOutcome.PASS
    )
    assert cells[(CalibrationJudge.DELIVERY, "fidelity")] is JudgeOutcome.FAIL
    assert cells[(CalibrationJudge.DELIVERY, "intonation")] is JudgeOutcome.PASS
    assert cells[(CalibrationJudge.SEMANTIC, "information_loss")] is JudgeOutcome.FAIL
    assert cells[(CalibrationJudge.SEMANTIC, "question_quality")] is JudgeOutcome.FAIL
    assert cells[(CalibrationJudge.SEMANTIC, "conciseness")] is JudgeOutcome.FAIL
    # Semantic PROGRESSION cells carry the severity split that keys their
    # strata; every other cell (conciseness included) has none.
    major = by_key[(CalibrationJudge.SEMANTIC, "information_loss")]
    assert major.severity is ProgressionSeverity.MAJOR
    assert major.stratum_id == "semantic:information_loss:major"
    minor = by_key[(CalibrationJudge.SEMANTIC, "question_quality")]
    assert minor.severity is ProgressionSeverity.MINOR
    assert minor.stratum_id == "semantic:question_quality:minor"
    assert by_key[(CalibrationJudge.SEMANTIC, "conciseness")].severity is None
    assert by_key[(CalibrationJudge.QUALITY, "responsiveness")].severity is None
    # Closed catalogs: unknown ids, factors the pack no longer declares, and
    # RETIRED quality factors (redundant_successful_tool_call) never become
    # cells — stored verdicts from retired ids are ignored downstream.
    assert (CalibrationJudge.QUALITY, "not_a_quality_factor") not in cells
    assert (CalibrationJudge.QUALITY, "redundant_successful_tool_call") not in cells
    assert (CalibrationJudge.DELIVERY, "not_a_pack_factor") not in cells
    assert (CalibrationJudge.DELIVERY, "digit_readout") not in cells


def test_instrument_cells_take_quality_from_the_artifact_when_sim_has_none():
    """The real 100-call corpora store the LLM-quality verdicts in the
    conversation artifact's shadow ``ours`` block, not on the sim — the
    folded instrument view must still read them from there while dropping
    the artifact's semantic cells."""
    sim = _voice_sim(
        "s1",
        "t1",
        quality_fails=("unnecessary_repetition", "responsiveness"),
        fidelity_finding=True,
    )
    sim.quality_info = None
    record = SemanticCallRecord(
        source="unused",
        sim_id="s1",
        progression_dimensions=[
            {
                "name": "information_loss",
                "flagged": True,
                "rating": 1,
                "evidence": "asked twice",
            }
        ],
        shadow_scores=SemanticShadowScores(
            ours=_quality_info(("unnecessary_repetition", "responsiveness"))
        ),
    )
    by_key = {
        (cell.judge, cell.factor_id): cell
        for cell in calibration_instrument_cells(sim, "hi", record)
    }
    repetition = by_key[(CalibrationJudge.QUALITY, "unnecessary_repetition")]
    assert repetition.outcome is JudgeOutcome.FAIL
    # The deterministic EVA checkers and the semantic family stay out even
    # though the artifact carries both.
    assert (CalibrationJudge.QUALITY, "responsiveness") not in by_key
    assert not any(j is CalibrationJudge.SEMANTIC for j, _ in by_key)
    # Without the record there is NO quality source at all — the artifact is
    # load-bearing for the surviving quality factors on these corpora.
    without = {
        (cell.judge, cell.factor_id)
        for cell in calibration_instrument_cells(sim, "hi", None)
    }
    assert not any(j is CalibrationJudge.QUALITY for j, _ in without)


# ---------------------------------------------------------------------------
# Annotate-mode packet build (blind by construction)
# ---------------------------------------------------------------------------


def _packet_options(
    tmp_path: Path, run_dir: Path, artifact: Path, *, seed: int = 7, tag: str = ""
) -> CalibrationPacketOptions:
    name = f"hi_calibration_test{tag}"
    return CalibrationPacketOptions(
        results=[run_dir],
        language="hi",
        conversation_artifact=artifact,
        n_calls=4,
        control_fraction=0.25,
        seed=seed,
        batch_name=name,
        out_dir=tmp_path / "packets" / name,
        emit_zip=False,
    )


def _build(tmp_path, monkeypatch, *, seed: int = 7, tag: str = ""):
    monkeypatch.setattr(
        "tau2.annotation.packets.calibration.caller_gender_for_task",
        lambda task_id, language, domain: "male",
    )
    run_dir = make_calibration_run(tmp_path)
    artifact = write_semantic_artifact(tmp_path / "experience.json", run_dir)
    options = _packet_options(tmp_path, run_dir, artifact, seed=seed, tag=tag)
    build = build_calibration_packet(options)
    return run_dir, artifact, options, build


def test_build_calibration_packet_draws_defects_and_writes_sidecars_outside(
    tmp_path, monkeypatch
):
    run_dir, artifact, options, build = _build(tmp_path, monkeypatch)
    packet_dir = options.packet_dir

    # Frame + coverage sidecar live OUTSIDE the packet directory.
    assert build.frame_path.is_file() and build.sidecar_path.is_file()
    assert packet_dir not in build.frame_path.parents
    assert packet_dir not in build.sidecar_path.parents

    frame = load_calibration_frame(build.frame_path)
    assert (
        frame.sampler_version
        == CALIBRATION_SAMPLER_VERSION
        == ("judge-calibration-draw-v6")
    )
    assert frame.n_calls_scanned == 4
    assert frame.language == "hi"
    assert len(frame.calls) == 4  # the WHOLE frame persists, drawn or not
    drawn = {call.sim_id for call in frame.drawn_calls()}
    # s1 covers every one of its strata with one draw; s4 is the only member
    # of quality:unnecessary_tool_call; exactly one clean control joins them.
    assert "s1" in drawn and "s4" in drawn
    assert len(drawn & {"s2", "s3"}) == 1 and len(drawn) == 3
    by_stratum = {row.stratum_id: row for row in frame.strata}
    assert by_stratum[CLEAN_CONTROL_STRATUM].inclusion_probability == pytest.approx(
        1 / 2
    )
    assert by_stratum["quality:unnecessary_tool_call"].inclusion_probability == 1.0
    assert frame.must_cover == [] and frame.must_cover_empty == []  # hi: none

    sidecar = load_calibration_sidecar(build.sidecar_path)
    assert {row.sim_id for row in sidecar.rows} == drawn
    assert all(row.results_path.endswith("results.json") for row in sidecar.rows)
    s1_row = next(row for row in sidecar.rows if row.sim_id == "s1")
    assert "nativeness:natural_word_choice" in s1_row.strata
    # Instrument cut: no semantic strata, no deterministic-quality strata.
    # Intonation is its own axis (cells-v3 unfold); the fixture's intonation
    # axis is judge-clean, so no intonation stratum on this call.
    assert "delivery:fidelity" in s1_row.strata
    assert not any(stratum.startswith("semantic:") for stratum in s1_row.strata)
    assert not any(
        stratum.startswith("delivery:intonation") for stratum in s1_row.strata
    )
    assert "quality:responsiveness" not in s1_row.strata
    assert sidecar.cells_version == "calibration-cells-v3"
    assert s1_row.drawn_for  # sampled for at least one stratum
    assert "nativeness_model" in sidecar.judge_versions
    assert sidecar.judge_versions["delivery_prompt"] == ["delivery-prompt-test"]

    # Packet contents: one page, per-call full + agent-only audio.
    manifest = ArtifactManifest.model_validate_json((build.manifest_path).read_text())
    assert manifest.kind == JUDGE_CALIBRATION_KIND
    assert manifest.language == "hi"
    assert len(manifest.calibration_entries) == 3
    for entry in manifest.calibration_entries:
        assert (packet_dir / entry.full_audio_file).is_file()
        assert (packet_dir / entry.agent_audio_file).is_file()
        assert entry.caller_gender == "male"
    # Current Hindi instrument: 6 interaction + 4 nativeness (overall plus
    # the three selected factors) + 3 audio questions.
    assert manifest.provenance["question_count"] == 13
    assert manifest.provenance["cells_version"] == "calibration-cells-v3"
    assert manifest.provenance["seed"] == 7


def test_rater_packet_and_manifest_leak_no_judge_arm_or_sampling_signal(
    tmp_path, monkeypatch
):
    _, _, options, build = _build(tmp_path, monkeypatch)
    packet_dir = options.packet_dir
    html = (packet_dir / "index.html").read_text()
    manifest_text = build.manifest_path.read_text()

    forbidden = [
        # Judge verdicts and their evidence/quotes.
        NATIVENESS_EVIDENCE_SENTINEL,
        NATIVENESS_QUOTE_SENTINEL,
        "repeated the refund terms",  # quality FAIL evidence
        "digits read as one number",  # delivery finding
        "read 1-2-3 as",  # delivery factor evidence
        "asked for the booking code",  # semantic evidence
        # ("verdict" itself is a JS identifier for the RATER's own label in
        # the shared rubric template, so only concrete verdict content is
        # asserted here.)
        "FAIL",
        # Arms and sources.
        RUN_NAME,  # the run dir name encodes the arm
        "results.json",
        "provider",
        # The manifest's blinding-policy sentence mentions "rewards" by name;
        # the leak vector is a serialized reward key/value.
        '"reward"',
        "reward_info",
        # Sampling rationale.
        "stratum",
        "strata",
        "clean_control",
        "drawn_for",
    ]
    for needle in forbidden:
        assert needle not in html, f"rater page leaks {needle!r}"
        assert needle not in manifest_text, f"packet manifest leaks {needle!r}"

    # The entry model itself has no fields that could carry the join.
    assert {"results_path", "provider", "experiment"}.isdisjoint(
        CalibrationPacketEntry.model_fields
    )
    # The rater page still renders the real instrument and evidence.
    assert "Agent only" in html and "Full conversation" in html
    assert (
        f'"packet_version": "{CALIBRATION_PACKET_VERSION}"'
        in html.replace('","', '", "')
        or CALIBRATION_PACKET_VERSION in html
    )


def test_rebuild_is_idempotent_and_a_different_draw_refuses_to_overwrite(
    tmp_path, monkeypatch
):
    run_dir, artifact, options, build = _build(tmp_path, monkeypatch)
    again = build_calibration_packet(options)
    assert again.manifest_path == build.manifest_path

    changed = options.model_copy(update={"seed": 8})
    with pytest.raises(FileExistsError):
        build_calibration_packet(changed)

    # Same draw but an edited instrument (template/CSS/JS) must be LOUD:
    # neither silently keeping the stale page nor silently clobbering it.
    manifest = ArtifactManifest.model_validate_json(build.manifest_path.read_text())
    assert manifest.provenance["instrument_sha256"] == rubric_instrument_sha256()
    monkeypatch.setattr(
        "tau2.annotation.packets.calibration.rubric_instrument_sha256",
        lambda: "0" * 64,
    )
    with pytest.raises(FileExistsError, match="DIFFERENT instrument"):
        build_calibration_packet(options)

    # Same seed into a fresh directory reproduces the identical cohort.
    other = _packet_options(tmp_path, run_dir, artifact, seed=7, tag="_again")
    rebuilt = build_calibration_packet(other)
    first = ArtifactManifest.model_validate_json(build.manifest_path.read_text())
    second = ArtifactManifest.model_validate_json(rebuilt.manifest_path.read_text())
    assert [e.sim_id for e in first.calibration_entries] == [
        e.sim_id for e in second.calibration_entries
    ]
    assert (
        first.provenance["selection_sha256"] == (second.provenance["selection_sha256"])
    )


# ---------------------------------------------------------------------------
# Rater CSVs (built FROM the row models), adjudication, and agreement
# ---------------------------------------------------------------------------


def _rubric_csv(path: Path, rater: str, rows: list[RubricAnnotationRow]) -> Path:
    write_family_csv(
        path, RubricAnnotationRow.headers(), [row.to_cells() for row in rows]
    )
    return path


def _nativeness_csv(path: Path, rows: list[NativenessHumanLabelRow]) -> Path:
    write_family_csv(
        path, NativenessHumanLabelRow.headers(), [row.to_cells() for row in rows]
    )
    return path


def _interaction_row(rater, clip_id, factor_id, answer) -> RubricAnnotationRow:
    return RubricAnnotationRow(
        batch="calib",
        rater=rater,
        clip_id=clip_id,
        language="hi",
        section="interaction",
        factor_id=factor_id,
        question="q",
        answer=answer,
        completed=True,
    )


def _audio_row(rater, clip_id, factor_id, answer) -> RubricAnnotationRow:
    return RubricAnnotationRow(
        batch="calib",
        rater=rater,
        clip_id=clip_id,
        language="hi",
        section="audio",
        factor_id=factor_id,
        question="q",
        answer=answer,
        audio_consulted=True,
        evidence_mode="agent_only",
        completed=True,
    )


def _nativeness_row(rater, clip_id, label, severity=None) -> NativenessHumanLabelRow:
    return NativenessHumanLabelRow(
        batch="calib",
        rater=rater,
        clip_id=clip_id,
        language="hi",
        factor_id="natural_word_choice",
        evaluation_level="call",
        annotation_label=label,
        severity=severity,
        provenance_json=json.dumps({"packet_batch_id": "calib"}),
        completed=True,
    )


def _synthetic_returns(tmp_path: Path, sidecar_path: Path) -> list[Path]:
    """Two raters' CSVs over the drawn cohort, with hand-designed confusions:

    quality:unnecessary_repetition (judge: s1 FAIL, s4/control PASS):
      r1 yes/no/no, r2 yes/yes/no -> tp=2 fn=1 tn=3, judge kappa 2/3;
      rater-rater [1,0,1,1] -> human kappa 0.4, agreement 2/3.
    delivery: raters answer the generic number_date_currency question, which
      joins the fidelity aggregate: r1 NA -> fa_noopp, r2 "2" -> tp.
    nativeness:natural_word_choice: violations on s1, passes elsewhere.
    """
    sidecar = load_calibration_sidecar(sidecar_path)
    clip_of = {row.sim_id: row.clip_id for row in sidecar.rows}
    s1, s4 = clip_of["s1"], clip_of["s4"]
    control = next(clip for sim, clip in clip_of.items() if sim in {"s2", "s3"})

    yes, no = RubricAnswer.YES, RubricAnswer.NO
    r1_rubric = [
        _interaction_row("r1", s1, "unnecessary_repetition", yes),
        _interaction_row("r1", s4, "unnecessary_repetition", no),
        _interaction_row("r1", control, "unnecessary_repetition", no),
        _audio_row("r1", s1, "number_date_currency", RubricAnswer.AUDIO_NO_OPPORTUNITY),
    ]
    r2_rubric = [
        _interaction_row("r2", s1, "unnecessary_repetition", yes),
        _interaction_row("r2", s4, "unnecessary_repetition", yes),
        _interaction_row("r2", control, "unnecessary_repetition", no),
        _audio_row("r2", s1, "number_date_currency", RubricAnswer.AUDIO_CLEAR),
    ]
    r1_native = [
        _nativeness_row("r1", s1, "violation", severity=3),
        _nativeness_row("r1", s4, "pass"),
        _nativeness_row("r1", control, "not_applicable"),
    ]
    r2_native = [
        _nativeness_row("r2", s1, "violation", severity=2),
        _nativeness_row("r2", s4, "pass"),
        _nativeness_row("r2", control, "pass"),
    ]
    return [
        _rubric_csv(tmp_path / "r1_rubric.csv", "r1", r1_rubric),
        _rubric_csv(tmp_path / "r2_rubric.csv", "r2", r2_rubric),
        _nativeness_csv(tmp_path / "r1_native.csv", r1_native),
        _nativeness_csv(tmp_path / "r2_native.csv", r2_native),
    ]


def test_rubric_csv_round_trips_through_the_sanctioned_ingest(tmp_path):
    row = _interaction_row("r1", "clip_001", "unnecessary_repetition", "Yes")
    path = _rubric_csv(tmp_path / "filled.csv", "r1", [row])
    parsed = ingest_browser_csv(path)
    assert parsed.kind == RUBRIC_PACKET_KIND
    assert parsed.rows == [row]


def test_collect_human_labels_maps_sections_to_judges_and_doubles_semantic(
    tmp_path,
):
    rows = [
        _interaction_row("r1", "clip_001", "unnecessary_repetition", "Yes"),
        _interaction_row("r1", "clip_001", "unnecessary_tool_call", "No"),
        RubricAnnotationRow(
            batch="calib",
            rater="r1",
            clip_id="clip_001",
            language="hi",
            section="semantic",
            factor_id="information_loss",
            question="q",
            answer="Yes",
            completed=True,
        ),
        _audio_row("r1", "clip_001", "intonation", RubricAnswer.AUDIO_MINOR),
        _audio_row("r1", "clip_001", "mispronunciation", RubricAnswer.AUDIO_NO_ISSUE),
        _audio_row("r1", "clip_001", "missing_word", RubricAnswer.AUDIO_SEVERE),
    ]
    path = _rubric_csv(tmp_path / "filled.csv", "r1", rows)
    collected = collect_human_labels([path])
    labels, raters = collected.labels, collected.raters
    assert raters == ["r1"]
    assert collected.verifications == {} and collected.verifiers == []
    q = CalibrationJudge.QUALITY
    s = CalibrationJudge.SEMANTIC
    d = CalibrationJudge.DELIVERY
    assert labels[(q, "unnecessary_repetition", "clip_001", "r1")] is ColdLabel.YES
    # Interaction answers double as progression-dimension labels.
    assert labels[(s, "redundant_statements", "clip_001", "r1")] is ColdLabel.YES
    assert labels[(s, "unnecessary_tool_calls", "clip_001", "r1")] is ColdLabel.NO
    assert labels[(s, "information_loss", "clip_001", "r1")] is ColdLabel.YES
    # Generic fidelity dimensions aggregate (any yes -> yes) onto the axis.
    assert labels[(d, "fidelity", "clip_001", "r1")] is ColdLabel.YES
    assert labels[(d, "intonation", "clip_001", "r1")] is ColdLabel.YES


def test_collect_human_labels_rejects_foreign_packet_csvs(tmp_path):
    from tau2.annotation.models import AudioQualityRow

    path = tmp_path / "foreign.csv"
    write_family_csv(path, AudioQualityRow.headers(), [])
    with pytest.raises(ValueError, match="rubric-answer and nativeness-label"):
        collect_human_labels([path])


def test_tri_state_disagreement_ignores_unscored_states():
    fail, pas = JudgeOutcome.FAIL, JudgeOutcome.PASS
    assert tri_state_disagreement(fail, [ColdLabel.YES]) is False
    assert tri_state_disagreement(fail, [ColdLabel.NO]) is True
    assert tri_state_disagreement(pas, [ColdLabel.NO_OPPORTUNITY]) is True
    assert tri_state_disagreement(pas, [ColdLabel.YES, ColdLabel.NO]) is True
    assert tri_state_disagreement(fail, [None, ColdLabel.UNSURE]) is False
    assert tri_state_disagreement(JudgeOutcome.DEFERRED, [ColdLabel.YES]) is False


def test_adjudication_page_shows_verdicts_highlights_splits_and_watermarks(
    tmp_path, monkeypatch
):
    _, _, options, build = _build(tmp_path, monkeypatch)
    filled = _synthetic_returns(tmp_path, build.sidecar_path)
    out = tmp_path / "adjudication"
    manifest_path = build_calibration_adjudication(
        CalibrationAdjudicationOptions(
            sidecar=build.sidecar_path,
            filled=filled,
            batch_name="hi_calibration_adjudication",
            out_dir=out,
            emit_zip=False,
            watermark=WATERMARK,
        )
    )
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    assert manifest.kind == JUDGE_CALIBRATION_ADJUDICATION_KIND
    assert manifest.provenance["raters"] == ["r1", "r2"]
    assert manifest.provenance["watermark"] == WATERMARK

    html = (out / "index.html").read_text()
    # The owner page is the UN-blind view: watermark, judge evidence, run
    # label, and sampling rationale are all visible.
    assert WATERMARK in html
    assert NATIVENESS_EVIDENCE_SENTINEL in html
    assert RUN_NAME in html
    assert "sampled for:" in html
    assert "ca-row-disagree" in html
    # r2 marked s4 as a repetition issue while the judge passed it: that call
    # must carry at least one highlighted disagreement row.
    assert "disagreement" in html


def test_judge_only_adjudication_renders_verdicts_before_any_rater_returns(
    tmp_path, monkeypatch
):
    _, _, options, build = _build(tmp_path, monkeypatch)
    out = tmp_path / "judge_only"
    judge_only_options = CalibrationAdjudicationOptions(
        sidecar=build.sidecar_path,
        batch_name="hi_calibration_judge_only",
        out_dir=out,
        emit_zip=False,
    )
    manifest_path = build_calibration_adjudication(judge_only_options)
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    assert manifest.kind == JUDGE_CALIBRATION_ADJUDICATION_KIND
    assert manifest.provenance["filled"] == []
    assert manifest.provenance["raters"] == []
    assert manifest.provenance["judge_only"] is True
    assert manifest.provenance["watermark"] == JUDGE_ONLY_BANNER

    html = (out / "index.html").read_text()
    # Fixed banner + judge verdicts, evidence, and sampling rationale visible.
    assert JUDGE_ONLY_BANNER in html
    assert "no rater returns yet" in html
    # The precision view is positives-only: judge-clean controls have no
    # decision candidates, so they are dropped rather than paged empty.
    sidecar = load_calibration_sidecar(build.sidecar_path)
    dropped = set(manifest.provenance["dropped_empty_clips"])
    assert dropped == {
        row.clip_id for row in sidecar.rows if row.strata == [CLEAN_CONTROL_STRATUM]
    }
    assert dropped
    # Every RENDERED call ships the audio pair the audio-judge decisions
    # need, and the page renders a per-call player wired to both tracks;
    # dropped calls keep their audio out of the bundle entirely.
    for clip_row in sidecar.rows:
        full = out / "calls" / f"{clip_row.clip_id}_full.wav"
        agent = out / "calls" / f"{clip_row.clip_id}_agent.wav"
        if clip_row.clip_id in dropped:
            assert not full.exists() and not agent.exists()
            assert clip_row.clip_id not in html
            continue
        assert full.is_file() and full.stat().st_size > 44
        assert agent.is_file() and agent.stat().st_size > 44
        assert f'data-full-src="calls/{clip_row.clip_id}_full.wav"' in html
        assert f'data-agent-src="calls/{clip_row.clip_id}_agent.wav"' in html
    assert "rp-audio-panel" in html
    assert "audio" in manifest.provenance
    assert NATIVENESS_EVIDENCE_SENTINEL in html
    assert "sampled for:" in html
    # No rater columns, no synthetic/unlabeled rater states, no disagreement
    # highlighting or badges (nothing to disagree with). (".ca-state-unlabeled"
    # exists as an unused CSS rule; the rendered state text must not.)
    assert ">unlabeled<" not in html
    assert 'class="ca-row-disagree"' not in html
    assert '<span class="ca-badge-disagree">' not in html
    # Precision view: judge POSITIVES only — no pass/N-A verdict rows.
    assert html.count(">FAIL<") > 0
    assert ">pass<" not in html and ">n/a<" not in html

    # Idempotent rebuild: hashing the empty filled list is deterministic.
    again = ArtifactManifest.model_validate_json(
        build_calibration_adjudication(judge_only_options).read_text()
    )
    assert again.batch_id == manifest.batch_id
    assert (
        again.provenance["selection_sha256"]
        == (manifest.provenance["selection_sha256"])
    )

    # An explicit watermark takes precedence over the fixed banner.
    watermarked = build_calibration_adjudication(
        CalibrationAdjudicationOptions(
            sidecar=build.sidecar_path,
            batch_name="hi_calibration_judge_only_wm",
            out_dir=tmp_path / "judge_only_wm",
            emit_zip=False,
            watermark=WATERMARK,
        )
    )
    wm_html = (tmp_path / "judge_only_wm" / "index.html").read_text()
    assert WATERMARK in wm_html
    assert JUDGE_ONLY_BANNER not in wm_html
    wm_manifest = ArtifactManifest.model_validate_json(watermarked.read_text())
    assert wm_manifest.provenance["watermark"] == WATERMARK
    assert wm_manifest.provenance["judge_only"] is True


# ---------------------------------------------------------------------------
# Decision capture on the adjudicate page (both forms) + decisions CSV
# ---------------------------------------------------------------------------

#: Every judge-positive (judge, factor, level) the fixture run must mint a
#: decision candidate for. s1 carries all defects but unnecessary_tool_call
#: (s4's); the drawn control is judge-clean and mints none. The stored
#: digit_readout verdict mints nothing — the pack no longer declares it. The
#: deterministic quality checkers are call-level by nature. The retired
#: register verdict deliberately stored on s1 must not create a candidate.
# Instrument cut (calibration-cells-v2): no semantic candidates, no
# deterministic-quality candidates; the delivery axes fold into fidelity.
EXPECTED_CANDIDATES = {
    ("s1", "delivery", "fidelity", "utterance"),
    ("s1", "nativeness", "natural_word_choice", "utterance"),
    ("s1", "quality", "unnecessary_repetition", "call"),
    ("s4", "quality", "unnecessary_tool_call", "call"),
}


def _judge_only_adjudication(tmp_path, build, name="hi_decisions"):
    out = tmp_path / name
    manifest_path = build_calibration_adjudication(
        CalibrationAdjudicationOptions(
            sidecar=build.sidecar_path,
            batch_name=name,
            out_dir=out,
            emit_zip=False,
        )
    )
    return out, ArtifactManifest.model_validate_json(manifest_path.read_text())


def test_decision_controls_cover_exactly_the_judge_positives(tmp_path, monkeypatch):
    _, _, options, build = _build(tmp_path, monkeypatch)
    out, manifest = _judge_only_adjudication(tmp_path, build)

    entries = manifest.calibration_decision_entries
    assert entries is not None
    assert {
        (e.sim_id, e.judge, e.factor_id, e.evaluation_level) for e in entries
    } == EXPECTED_CANDIDATES
    assert all(e.judge_verdict == "fail" for e in entries)
    assert len({e.candidate_id for e in entries}) == len(entries)

    html = (out / "index.html").read_text()
    # One Confirmed and one Rejected button per candidate — and none anywhere
    # else (non-positive cells stay read-only table rows).
    confirm_button = '<button type="button" data-decision="confirmed"'
    reject_button = '<button type="button" data-decision="rejected"'
    assert html.count(confirm_button) == len(entries)
    assert html.count(reject_button) == len(entries)
    assert "Select a decision." in html
    assert f"/ {len(entries)} decided" in html  # page-level counter

    config = packet_config(out / "index.html")
    assert config["csv_headers"] == CalibrationDecisionRow.headers()
    assert {c["candidate_id"] for c in config["candidates"]} == {
        e.candidate_id for e in entries
    }
    assert config["sidecar_batch_id"] == manifest.provenance["sidecar_batch_id"]


def test_utterance_positives_pin_the_exact_flagged_turn(tmp_path, monkeypatch):
    _, _, options, build = _build(tmp_path, monkeypatch)
    out, manifest = _judge_only_adjudication(tmp_path, build, name="hi_pinning")
    entries = {(e.judge, e.factor_id): e for e in manifest.calibration_decision_entries}

    native = entries[("nativeness", "natural_word_choice")]
    assert native.evaluation_level == "utterance"
    assert native.agent_turn is not None and native.agent_turn.index == 0
    assert NATIVENESS_UNIT_QUOTE in native.agent_turn.text
    assert native.judge_quote == NATIVENESS_UNIT_QUOTE
    assert native.agent_turn.preceding_customer_text == CALLER_CONTEXT_TEXT

    fidelity = entries[("delivery", "fidelity")]
    assert fidelity.evaluation_level == "utterance"
    assert fidelity.agent_turn is not None and fidelity.agent_turn.index == 0
    assert "digits read as one number" in fidelity.judge_evidence

    html = (out / "index.html").read_text()
    # The judge's pinned quote is highlighted INSIDE the flagged agent turn,
    # with the immediately preceding caller turn as context.
    assert f"<mark>{NATIVENESS_UNIT_QUOTE}</mark>" in html
    assert CALLER_CONTEXT_TEXT in html
    assert "Flagged agent utterance · turn 0" in html
    # Call-level positives carry no pinned turn.
    assert entries[("quality", "unnecessary_repetition")].agent_turn is None


def _decisions_csv(tmp_path, manifest, decide) -> Path:
    """Simulate the page's export: one CalibrationDecisionRow per candidate,
    decided per ``decide(entry) -> Optional[str]`` (None = left undecided)."""
    rows = []
    for entry in manifest.calibration_decision_entries:
        decision = decide(entry)
        turn = entry.agent_turn
        rows.append(
            CalibrationDecisionRow(
                batch=manifest.batch_name,
                rater="owner",
                clip_id=entry.clip_id,
                simulation_id=entry.sim_id,
                task_id=entry.task_id,
                language=entry.language,
                judge=entry.judge,
                factor_id=entry.factor_id,
                evaluation_level=entry.evaluation_level,
                judge_verdict=entry.judge_verdict,
                candidate_id=entry.candidate_id,
                adjudication_decision=decision,
                agent_turn_index=turn.index if turn else None,
                agent_turn_id=turn.turn_id if turn else "",
                agent_text=turn.text if turn else "",
                preceding_customer_text=(
                    (turn.preceding_customer_text or "") if turn else ""
                ),
                provenance_json=json.dumps({"packet_batch_id": manifest.batch_id}),
                completed=decision is not None,
                created_at="2026-08-19T00:00:00+00:00",
            )
        )
    path = tmp_path / "owner_decisions.csv"
    write_family_csv(
        path, CalibrationDecisionRow.headers(), [row.to_cells() for row in rows]
    )
    return path


def test_decisions_csv_round_trips_into_agreement_as_owner_adjudications(
    tmp_path, monkeypatch
):
    _, _, options, build = _build(tmp_path, monkeypatch)
    _, manifest = _judge_only_adjudication(tmp_path, build)

    def decide(entry):
        if entry.factor_id == "unnecessary_repetition":
            return "confirmed"
        if entry.factor_id == "fidelity":
            return "rejected"
        if entry.factor_id == "natural_word_choice":
            return "confirmed"
        return None  # everything else left undecided

    path = _decisions_csv(tmp_path, manifest, decide)
    parsed = ingest_browser_csv(path)
    assert parsed.kind == CALIBRATION_DECISIONS_KIND

    tallies, adjudicators = collect_owner_decisions([path])
    assert adjudicators == ["owner"]

    # Decisions alone are enough (no rater CSVs yet).
    report = build_calibration_agreement(build.sidecar_path, [], decisions=[path])
    assert report.adjudicators == ["owner"]
    assert report.raters == []
    rows = {(row.judge.value, row.factor_id): row for row in report.rows}
    repetition = rows[("quality", "unnecessary_repetition")]
    assert (repetition.n_confirmed, repetition.n_rejected) == (1, 0)
    assert repetition.adjudicated_precision == 1.0
    fidelity = rows[("delivery", "fidelity")]
    assert (fidelity.n_confirmed, fidelity.n_rejected) == (0, 1)
    assert fidelity.adjudicated_precision == 0.0
    assert rows[("nativeness", "natural_word_choice")].adjudicated_precision == 1.0
    assert ("semantic", "information_loss") not in rows
    assert ("delivery", "digit_readout") not in rows

    # Alongside rater CSVs the label-side metrics are untouched.
    filled = _synthetic_returns(tmp_path, build.sidecar_path)
    combined = build_calibration_agreement(build.sidecar_path, filled, decisions=[path])
    combined_rows = {(r.judge.value, r.factor_id): r for r in combined.rows}
    repetition = combined_rows[("quality", "unnecessary_repetition")]
    assert repetition.metrics.precision == 1.0
    assert repetition.adjudicated_precision == 1.0
    rendered = render_calibration_agreement(combined)
    assert "aprec" in rendered and "adjudicators: owner" in rendered

    # A rater/nativeness CSV in --decisions is a loud error.
    with pytest.raises(ValueError, match="decisions CSV"):
        collect_owner_decisions([filled[0]])

    with pytest.raises(ValueError, match="needs rater CSVs"):
        build_calibration_agreement(build.sidecar_path, [], decisions=[])


# ---------------------------------------------------------------------------
# Adjudicate mode from a per-factor draw (no annotate build, no sidecar input)
# ---------------------------------------------------------------------------


def test_per_factor_adjudication_draw_builds_from_results_directly(tmp_path):
    run_dir = make_calibration_run(tmp_path)
    artifact = write_semantic_artifact(tmp_path / "experience.json", run_dir)
    options = CalibrationAdjudicationDrawOptions(
        results=[run_dir],
        language="hi",
        conversation_artifact=artifact,
        seed=7,
        batch_name="hi_per_factor_adjudication",
        out_dir=tmp_path / "per_factor",
        emit_zip=False,
    )
    build = build_calibration_adjudication_draw(options)

    # The persisted frame records the per-factor rule, cap, and accounting.
    frame = load_calibration_frame(build.frame_path)
    assert frame.sampler_version == CALIBRATION_SAMPLER_VERSION
    assert frame.config.draw == "adjudicate_per_factor"
    assert frame.config.per_factor_cap == 20
    by_stratum = {row.stratum_id: row for row in frame.strata}
    # Every defect stratum is taken whole (all pools are under the cap);
    # the clean controls are accounted but never drawn. Instrument cut: no
    # semantic strata, no deterministic-quality strata, delivery folds into
    # one fidelity stratum.
    assert by_stratum["quality:unnecessary_tool_call"].covered == 1
    assert by_stratum["nativeness:natural_word_choice"].covered == 1
    assert by_stratum["delivery:fidelity"].covered == 1
    assert not any(s.startswith("semantic:") for s in by_stratum)
    assert "quality:responsiveness" not in by_stratum
    assert by_stratum[CLEAN_CONTROL_STRATUM].frame_size == 2
    assert by_stratum[CLEAN_CONTROL_STRATUM].covered == 0

    # The sidecar carries the per-factor counts; only flagged calls join.
    sidecar = load_calibration_sidecar(build.sidecar_path)
    assert sidecar.sampler_version == CALIBRATION_SAMPLER_VERSION
    assert {row.sim_id for row in sidecar.rows} == {"s1", "s4"}
    assert sidecar.per_factor_counts is not None
    assert sidecar.per_factor_counts["quality:unnecessary_tool_call"] == 1
    assert CLEAN_CONTROL_STRATUM not in sidecar.per_factor_counts
    assert set(sidecar.per_factor_counts) == {
        row.stratum_id
        for row in frame.strata
        if row.stratum_id != CLEAN_CONTROL_STRATUM
    }

    # The rendered page is the judge-only review view over exactly the
    # judge positives the fixture plants.
    manifest = ArtifactManifest.model_validate_json(build.manifest_path.read_text())
    assert manifest.kind == JUDGE_CALIBRATION_ADJUDICATION_KIND
    assert manifest.provenance["judge_only"] is True
    assert {
        (e.sim_id, e.judge, e.factor_id, e.evaluation_level)
        for e in manifest.calibration_decision_entries
    } == EXPECTED_CANDIDATES
    html = (build.manifest_path.parent / "index.html").read_text()
    assert JUDGE_ONLY_BANNER in html
    assert NATIVENESS_EVIDENCE_SENTINEL in html

    # Deterministic: a second draw into a fresh directory reproduces the
    # cohort and the candidate ids exactly.
    again = build_calibration_adjudication_draw(
        options.model_copy(update={"out_dir": tmp_path / "per_factor_again"})
    )
    sidecar_again = load_calibration_sidecar(again.sidecar_path)
    assert [(r.sim_id, r.clip_id) for r in sidecar_again.rows] == [
        (r.sim_id, r.clip_id) for r in sidecar.rows
    ]
    manifest_again = ArtifactManifest.model_validate_json(
        again.manifest_path.read_text()
    )
    assert {e.candidate_id for e in manifest_again.calibration_decision_entries} == {
        e.candidate_id for e in manifest.calibration_decision_entries
    }


def _decision_entry(
    candidate_id: str, judge: str, factor_id: str, evidence: str = ""
) -> CalibrationDecisionEntry:
    return CalibrationDecisionEntry(
        candidate_id=candidate_id,
        clip_id="clip_001",
        sim_id="s1",
        task_id="t1",
        language="hi",
        judge=judge,
        factor_id=factor_id,
        evaluation_level="call",
        judge_verdict="fail",
        judge_evidence=evidence,
    )


def test_sample_decision_rows_caps_and_stratifies():
    # Over-cap factors get a seeded draw; the folded fidelity cell draws
    # round-robin across evidence-category prefixes so every category
    # survives into the presented set. Under-cap factors keep every row.
    fidelity = [
        _decision_entry(f"fid_pron_{i}", "delivery", "fidelity", f"pronunciation: {i}")
        for i in range(6)
    ] + [
        _decision_entry("fid_omit_0", "delivery", "fidelity", "omission: dropped"),
        _decision_entry("fid_sub_0", "delivery", "fidelity", "substitution: swapped"),
    ]
    uniform = [
        _decision_entry(f"rep_{i}", "quality", "unnecessary_repetition")
        for i in range(5)
    ]
    small = [_decision_entry("hon_0", "nativeness", "honorific_agreement")]
    kept, accounting = _sample_decision_rows(fidelity + uniform + small, 4, "salt")

    assert accounting["delivery:fidelity"] == {"total": 8, "presented": 4}
    assert accounting["quality:unnecessary_repetition"] == {"total": 5, "presented": 4}
    assert accounting["nativeness:honorific_agreement"] == {"total": 1, "presented": 1}
    kept_fidelity = {c for c in kept if c.startswith("fid_")}
    assert len(kept_fidelity) == 4
    # Round-robin: both rare categories survive alongside the dominant one.
    assert "fid_omit_0" in kept_fidelity
    assert "fid_sub_0" in kept_fidelity
    assert len({c for c in kept if c.startswith("rep_")}) == 4
    assert "hon_0" in kept
    # Deterministic under the same salt; the salt changes the draw.
    kept_again, _ = _sample_decision_rows(fidelity + uniform + small, 4, "salt")
    assert kept_again == kept


def test_decision_row_cap_limits_presented_rows_per_factor(tmp_path):
    # The cap bounds PRESENTED decision rows per judge factor (a seeded
    # sample), independent of the call-level draw cap. Provenance records
    # total vs presented so the sampling probability is recoverable.
    run_dir = make_calibration_run(tmp_path)
    artifact = write_semantic_artifact(tmp_path / "experience.json", run_dir)
    build = build_calibration_adjudication_draw(
        CalibrationAdjudicationDrawOptions(
            results=[run_dir],
            language="hi",
            conversation_artifact=artifact,
            seed=7,
            batch_name="hi_capped_adjudication",
            out_dir=tmp_path / "capped",
            emit_zip=False,
            decision_row_cap=1,
        )
    )
    manifest = ArtifactManifest.model_validate_json(build.manifest_path.read_text())
    per_factor: dict[tuple[str, str], int] = {}
    for entry in manifest.calibration_decision_entries:
        key = (entry.judge, entry.factor_id)
        per_factor[key] = per_factor.get(key, 0) + 1
    assert per_factor, "capped draw still presents at least one row per factor"
    assert all(count == 1 for count in per_factor.values())

    accounting = manifest.provenance["decision_rows_per_factor"]
    assert manifest.provenance["decision_row_cap"] == 1
    assert set(accounting) == {f"{judge}:{factor}" for judge, factor in per_factor}
    for stats in accounting.values():
        assert stats["presented"] == 1
        assert stats["total"] >= stats["presented"]

    # The reclassify control ships with the page: every card carries the
    # dropdown, the CSV schema carries the column, and the option list never
    # offers a card its own factor.
    assert "reclassified_factor" in CalibrationDecisionRow.headers()
    html = (build.manifest_path.parent / "index.html").read_text()
    assert "data-reclassify" in html
    assert "Real issue, wrong category?" in html
    # A capped rerun with the same seed reproduces the identical row set.
    again = build_calibration_adjudication_draw(
        CalibrationAdjudicationDrawOptions(
            results=[run_dir],
            language="hi",
            conversation_artifact=artifact,
            seed=7,
            batch_name="hi_capped_adjudication",
            out_dir=tmp_path / "capped_again",
            emit_zip=False,
            decision_row_cap=1,
        )
    )
    manifest_again = ArtifactManifest.model_validate_json(
        again.manifest_path.read_text()
    )
    assert {e.candidate_id for e in manifest_again.calibration_decision_entries} == {
        e.candidate_id for e in manifest.calibration_decision_entries
    }


def test_calls_whose_candidates_all_cap_out_are_dropped_from_the_packet(tmp_path):
    # Regression (hi cal-100 adjudicate, clip_019/clip_028): two calls flagged
    # ONLY for the folded delivery fidelity cell saturate decision_row_cap=1 —
    # the call that loses the seeded draw would render an empty page with
    # nothing to rate. The builder must drop it from the packet, keep its
    # audio out of the bundle, and re-persist the coverage sidecar without it.
    sims = [
        _voice_sim("s1", "t1", fidelity_finding=True),
        _voice_sim("s2", "t2", fidelity_finding=True),
    ]
    run_dir = make_hi_results(tmp_path, sims, name="hi_fidelity_pair")
    user = _mono([30000, 30000, 30000])
    agent = _mono([900, 1000, 1100])
    for sim in sims:
        audio_dir = (
            run_dir / "tasks" / f"task_{sim.task_id}" / f"sim_{sim.id}" / "audio"
        )
        audio_dir.mkdir(parents=True, exist_ok=True)
        save_wav_file(convert_to_stereo(user, agent), audio_dir / "both.wav")

    build = build_calibration_adjudication_draw(
        CalibrationAdjudicationDrawOptions(
            results=[run_dir],
            language="hi",
            seed=7,
            batch_name="hi_saturated_cap",
            out_dir=tmp_path / "saturated",
            emit_zip=False,
            decision_row_cap=1,
        )
    )
    manifest = ArtifactManifest.model_validate_json(build.manifest_path.read_text())

    # Both calls were drawn for delivery:fidelity; the cap keeps one
    # candidate, so exactly one call survives and the other is recorded.
    assert len(manifest.calibration_decision_entries) == 1
    kept_clip = manifest.calibration_decision_entries[0].clip_id
    (dropped_clip,) = manifest.provenance["dropped_empty_clips"]
    assert {kept_clip, dropped_clip} == {"clip_001", "clip_002"}

    packet_dir = build.manifest_path.parent
    html = (packet_dir / "index.html").read_text()
    assert kept_clip in html
    assert dropped_clip not in html
    assert (packet_dir / "calls" / f"{kept_clip}_full.wav").exists()
    assert not (packet_dir / "calls" / f"{dropped_clip}_full.wav").exists()
    assert not (packet_dir / "calls" / f"{dropped_clip}_agent.wav").exists()

    # The coverage sidecar shipped alongside matches the packet: the dropped
    # row is gone and the per-factor coverage counts the rendered calls only.
    sidecar = load_calibration_sidecar(build.sidecar_path)
    assert [row.clip_id for row in sidecar.rows] == [kept_clip]
    assert sidecar.per_factor_counts == {"delivery:fidelity": 1}


def test_per_factor_adjudication_draw_with_no_flagged_calls_fails_loudly(tmp_path):
    sims = [_voice_sim("s2", "t2"), _voice_sim("s3", "t3")]  # judge-clean only
    run_dir = make_hi_results(tmp_path, sims, name="hi_clean_run")
    with pytest.raises(ValueError, match="no.*judge FAIL"):
        build_calibration_adjudication_draw(
            CalibrationAdjudicationDrawOptions(
                results=[run_dir],
                language="hi",
                batch_name="hi_clean_adjudication",
                out_dir=tmp_path / "clean_adjudication",
                emit_zip=False,
            )
        )


# ---------------------------------------------------------------------------
# --show-judge (opt-in judge-visible annotate build)
# ---------------------------------------------------------------------------


def test_frame_drops_language_unattributable_calls(tmp_path):
    # Quality/delivery verdicts alone say nothing about a call's language: a
    # sim with no nativeness block (or no recorded language) must never enter
    # a language-specific frame.
    orphan = _voice_sim("s9", "t9", quality_fails=("unnecessary_repetition",))
    orphan.nativeness_info = None
    unlabeled = _voice_sim("s8", "t8", quality_fails=("unnecessary_repetition",))
    unlabeled.nativeness_info.language = None
    loaded = [
        LoadedSim(
            sim=sim,
            task=None,
            domain="airline",
            results_dir=tmp_path,
            experiment_label=RUN_NAME,
        )
        for sim in (orphan, unlabeled)
    ]
    frame = build_calibration_frame(
        loaded,
        CalibrationDrawConfig(language="hi", n_calls=4, control_fraction=0.25, seed=7),
        results=[tmp_path],
    )
    assert frame.n_dropped_language_unknown == 2
    assert frame.calls == []


def test_frame_drops_fully_mute_calls(tmp_path, monkeypatch):
    # A fully-mute call (stored ticks, none carrying agent speech) cannot
    # render an agent-only clip, so it must never enter an audio-requiring
    # frame — annotate and adjudicate draws alike.
    import tau2.annotation.packets.builder as packets_builder
    from tau2.data_model.audio import (
        AudioEncoding,
        AudioFormat,
        audio_bytes_to_string,
    )
    from tau2.data_model.message import AssistantMessage, Tick

    monkeypatch.setattr(packets_builder, "find_audio", lambda *_: tmp_path / "both.wav")
    mute = _voice_sim("s10", "t10", quality_fails=("unnecessary_repetition",))
    mute.ticks = [Tick(tick_id=0, timestamp="t0")]
    speaking = _voice_sim("s11", "t11", quality_fails=("unnecessary_repetition",))
    speaking.ticks = [
        Tick(
            tick_id=0,
            timestamp="t0",
            agent_chunk=AssistantMessage.voice(
                audio_content=audio_bytes_to_string(b"\x7f" * 400),
                audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
                audio_script_gold="hello",
                utterance_ids=["u0"],
            ),
        )
    ]
    loaded = [
        LoadedSim(
            sim=sim,
            task=None,
            domain="airline",
            results_dir=tmp_path,
            experiment_label=RUN_NAME,
        )
        for sim in (mute, speaking)
    ]
    frame = build_calibration_frame(
        loaded,
        CalibrationDrawConfig(language="hi", n_calls=4, control_fraction=0.25, seed=7),
        results=[tmp_path],
    )
    assert frame.n_dropped_mute == 1
    assert [call.sim_id for call in frame.calls] == ["s11"]
    assert frame.calls[0].has_agent_speech


def test_rubric_page_numbers_sections_and_questions():
    # Display numbering ("3.6" = section 3, question 6) rides on the page and
    # on PACKET_CONFIG.questions so blocker messages can cite the exact
    # on-page question — but never enters the question models themselves.
    from tau2.annotation.artifacts import NativenessAgentTurn
    from tau2.annotation.packets.rubric import RubricCallVM, render_rubric_page

    sections = calibration_sections_for("hi")
    call = RubricCallVM(
        clip_id="clip_001",
        ordinal=1,
        full_audio_file="calls/clip_001_full.wav",
        agent_audio_file="calls/clip_001_agent.wav",
        full_duration="0:10",
        agent_duration="0:05",
        transcript_rows="<tr><td>row</td></tr>",
        agent_transcript=["hello"],
        sim_id="s1",
        task_id="t1",
        agent_turns=[
            NativenessAgentTurn(
                index=0,
                turn_id="agent-turn-000",
                text="hello",
                preceding_customer_text="hi",
            )
        ],
        agent_gender="female",
        caller_gender="male",
    )
    html = render_rubric_page(
        batch_id="batch",
        batch_name="hi_cal",
        language="hi",
        calls=[call],
        sections=sections,
    )
    config = json.loads(
        html.split("window.PACKET_CONFIG = ", 1)[1].split(";</script>", 1)[0]
    )
    expected = [
        f"{section_index}.{question_index}"
        for section_index, section in enumerate(sections, 1)
        for question_index, _ in enumerate(section.questions, 1)
    ]
    assert [question["number"] for question in config["questions"]] == expected
    # The rendered page shows the same numbers on section headings and
    # question labels.
    assert f"1. {sections[0].title}" in html
    assert "1.1 " in html
    last_number = expected[-1]
    assert f"{last_number} " in html


def test_adjudication_refuses_precision_pass_only_returns(tmp_path, monkeypatch):
    # Judge-visible (precision-pass) CSVs are calibration-agreement's input;
    # an adjudication built from ONLY those must fail loudly instead of
    # silently rendering a rater-less judge-vs-raters view.
    _, _, options, build = _build(tmp_path, monkeypatch)
    marked = f"hi_calibration{JUDGE_VISIBLE_BATCH_SUFFIX}"
    row = RubricAnnotationRow(
        batch=marked,
        rater="v1",
        clip_id="clip_001",
        language="hi",
        section="interaction",
        factor_id="unnecessary_repetition",
        question="q",
        answer="Yes",
        completed=True,
    )
    path = _rubric_csv(tmp_path / "precision_only.csv", "v1", [row])
    with pytest.raises(ValueError, match="judge-visible"):
        build_calibration_adjudication(
            CalibrationAdjudicationOptions(
                sidecar=build.sidecar_path,
                filled=[path],
                batch_name="hi_adjudication_precision_only",
                out_dir=tmp_path / "adjudication_precision_only",
                emit_zip=False,
            )
        )


def test_show_judge_build_renders_annotations_and_is_branded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tau2.annotation.packets.calibration.caller_gender_for_task",
        lambda task_id, language, domain: "male",
    )
    run_dir = make_calibration_run(tmp_path)
    artifact = write_semantic_artifact(tmp_path / "experience.json", run_dir)
    options = _packet_options(tmp_path, run_dir, artifact).model_copy(
        update={
            "show_judge": True,
            "batch_name": "hi_calibration_enriched",
            "out_dir": tmp_path / "packets" / "hi_calibration_enriched",
        }
    )
    build = build_calibration_packet(options)
    manifest = ArtifactManifest.model_validate_json(build.manifest_path.read_text())

    # Branded batch name, manifest flag, sidecar flag, fixed banner.
    assert manifest.batch_name.endswith(JUDGE_VISIBLE_BATCH_SUFFIX)
    assert manifest.provenance["judge_visible"] is True
    sidecar = load_calibration_sidecar(build.sidecar_path)
    assert sidecar.judge_visible is True

    html = (build.manifest_path.parent / "index.html").read_text()
    assert JUDGE_VISIBLE_BANNER in html
    # Judge verdicts, evidence, and pinned quotes render inline.
    assert NATIVENESS_EVIDENCE_SENTINEL in html
    assert "repeated the refund terms" in html  # quality evidence
    # Instrument cut: the semantic suite and the EVA turn-taking checkers
    # render nowhere on the page.
    assert "asked for the booking code" not in html
    assert 'data-factor-id="responsiveness"' not in html
    # A fidelity finding renders once, under the SINGLE fidelity question,
    # labeled with its dimension category.
    assert "digits read as one number" in html  # delivery axis evidence
    assert html.count("digits read as one number") == 1
    delivery_idx = html.index("digits read as one number")
    owning_block = html.rindex('data-factor-id="', 0, delivery_idx)
    assert html[owning_block:].startswith('data-factor-id="fidelity"')
    assert "mispronunciation: digits read as one number" in html
    # Utterance-level positives pin the exact flagged agent turn inside the
    # expansion: the unit quote highlighted in the turn text, with the
    # immediately preceding caller turn as context.
    assert "Flagged agent utterance" in html
    assert f"<mark>{NATIVENESS_UNIT_QUOTE}</mark>" in html
    assert CALLER_CONTEXT_TEXT in html
    # Findings render as a plain uniform block on EVERY question — including
    # a fixed no-findings row — with NO per-question disclosure to click:
    # phase 1 hides all of it via the CSS gate, the reveal paints everything
    # open inline, and the uniform shape leaks nothing about which questions
    # the judge flagged.
    assert html.count('<div class="rp-judge-details">') == html.count(
        'class="rp-question"'
    )
    assert "No judge annotations for this question." in html
    assert "Show judge annotations" not in html
    assert "rp-judge-toggle" not in html
    assert '<details class="rp-judge-details"' not in html
    # Two-phase machinery: every call starts phase-gated (judge-phased,
    # findings CSS-hidden until the blind answers are locked), with the lock
    # button rendered disabled and the phase-2 confirm label hidden.
    assert html.count(' judge-phased"') == html.count('data-clip-id="')
    assert "data-lock-blind disabled>" in html
    assert "Lock blind answers &amp; reveal judge findings" in html
    assert "data-phase-badge>Phase 1 of 2" in html
    assert "<label data-confirm-label hidden>" in html
    assert "Confirm final answers (phase 2)" in html
    assert "Two phases per call." in html
    assert ".rp-call.judge-phased:not(.phase-revealed) .rp-judge-details" in html
    # The lock-time snapshot freezes EVERY call-level mutable field that rides
    # on exported rows — a phase-2 edit to the notes, evidence mode, or audio
    # flag must never contaminate pre_reveal (blind-pool) rows — and the
    # export reads those fields from the layer, never the live state.
    assert "evidence_mode: row.evidence_mode," in html
    assert "audio_consulted: row.audio_consulted," in html
    assert "evidence_mode: layer.evidence_mode || 'full_conversation'" in html
    assert "audio_consulted: Boolean(layer.audio_consulted)" in html
    assert "notes: layer.notes || ''" in html
    assert "notes: state.notes || ''" not in html
    config = packet_config(build.manifest_path.parent / "index.html")
    assert config["judge_visible"] is True


def test_blind_build_records_judge_visible_false(tmp_path, monkeypatch):
    _, _, options, build = _build(tmp_path, monkeypatch)
    manifest = ArtifactManifest.model_validate_json(build.manifest_path.read_text())
    assert manifest.provenance["judge_visible"] is False
    assert not manifest.batch_name.endswith(JUDGE_VISIBLE_BATCH_SUFFIX)
    assert load_calibration_sidecar(build.sidecar_path).judge_visible is False
    config = packet_config(build.manifest_path.parent / "index.html")
    assert config["judge_visible"] is False


def test_agreement_routes_judge_visible_returns_to_precision(tmp_path, monkeypatch):
    _, _, options, build = _build(tmp_path, monkeypatch)

    # (a) A rubric CSV whose batch column carries the judge-visible brand is
    # routed to the precision pool, never the blind labels.
    marked = f"hi_calibration{JUDGE_VISIBLE_BATCH_SUFFIX}"
    row = RubricAnnotationRow(
        batch=marked,
        rater="v1",
        clip_id="clip_001",
        language="hi",
        section="interaction",
        factor_id="unnecessary_repetition",
        question="q",
        answer="Yes",
        completed=True,
    )
    path = _rubric_csv(tmp_path / "marked_rubric.csv", "v1", [row])
    collected = collect_human_labels([path])
    assert collected.labels == {} and collected.raters == []
    assert collected.verifiers == ["v1"]
    assert (
        collected.verifications[
            (CalibrationJudge.QUALITY, "unnecessary_repetition", "clip_001", "v1")
        ]
        is ColdLabel.YES
    )

    # (b) A nativeness CSV whose provenance says judge_visible routes too.
    native = NativenessHumanLabelRow(
        batch="innocent_name",
        rater="v1",
        clip_id="clip_001",
        language="hi",
        factor_id="natural_word_choice",
        evaluation_level="call",
        annotation_label="pass",
        provenance_json=json.dumps({"packet_batch_id": "x", "judge_visible": True}),
        completed=True,
    )
    native_path = _nativeness_csv(tmp_path / "marked_native.csv", [native])
    collected = collect_human_labels([native_path])
    assert collected.labels == {} and collected.verifiers == ["v1"]
    assert (
        collected.verifications[
            (CalibrationJudge.NATIVENESS, "natural_word_choice", "clip_001", "v1")
        ]
        is ColdLabel.NO
    )

    # (c) Verified precision: the verifier confirms one judge positive and
    # rejects another; blind stats stay untouched by precision-pass rows.
    report = build_calibration_agreement(build.sidecar_path, [path, native_path])
    assert report.raters == [] and report.verifiers == ["v1"]
    by_key = {(r.judge, r.factor_id): r for r in report.rows}
    confirmed = by_key[(CalibrationJudge.QUALITY, "unnecessary_repetition")]
    assert confirmed.n_verifier_confirmed == 1
    assert confirmed.verified_precision == 1.0
    assert confirmed.metrics.precision is None  # blind pool saw nothing
    rejected = by_key[(CalibrationJudge.NATIVENESS, "natural_word_choice")]
    assert rejected.n_verifier_rejected == 1
    assert rejected.verified_precision == 0.0

    # (c) A judge-visible build's own sidecar is refused outright.
    sidecar = load_calibration_sidecar(build.sidecar_path)
    visible_sidecar_path = tmp_path / "visible_coverage.json"
    visible_sidecar_path.write_text(
        sidecar.model_copy(update={"judge_visible": True}).model_dump_json()
    )
    clean = _rubric_csv(
        tmp_path / "clean_rubric.csv",
        "r1",
        [
            RubricAnnotationRow(
                batch="hi_calibration_test",
                rater="r1",
                clip_id="clip_001",
                language="hi",
                section="interaction",
                factor_id="unnecessary_repetition",
                question="q",
                answer="Yes",
                completed=True,
            )
        ],
    )
    with pytest.raises(ValueError, match="judge-visible"):
        build_calibration_agreement(visible_sidecar_path, [clean])


def test_two_phase_returns_route_pre_reveal_blind_and_post_reveal_precision(
    tmp_path, monkeypatch
):
    marked = f"hi_calibration{JUDGE_VISIBLE_BATCH_SUFFIX}"

    def phased_row(phase, answer, severity=None):
        return RubricAnnotationRow(
            batch=marked,
            rater="v1",
            clip_id="clip_001",
            language="hi",
            section="interaction",
            factor_id="unnecessary_repetition",
            question="q",
            answer=answer,
            severity=severity,
            phase=phase,
            completed=True,
        )

    path = _rubric_csv(
        tmp_path / "two_phase.csv",
        "v1",
        [phased_row("pre_reveal", "Yes", "major"), phased_row("post_reveal", "No")],
    )
    collected = collect_human_labels([path])
    key = (CalibrationJudge.QUALITY, "unnecessary_repetition", "clip_001", "v1")
    twin = (CalibrationJudge.SEMANTIC, "redundant_statements", "clip_001", "v1")
    # The locked blind layer scores recall; the post-reveal revision scores
    # precision — one packet, both pools, the same rater in both rosters.
    assert collected.labels[key] is ColdLabel.YES
    assert collected.verifications[key] is ColdLabel.NO
    assert collected.raters == ["v1"] and collected.verifiers == ["v1"]
    # The human minor/major rides the SEMANTIC twin of the blind YES only.
    assert collected.severities[twin] is ProgressionSeverity.MAJOR
    assert twin not in collected.verifier_severities

    # Nativeness rows route by phase too.
    def phased_native(phase, label, severity=None):
        return NativenessHumanLabelRow(
            batch=marked,
            rater="v1",
            clip_id="clip_001",
            language="hi",
            factor_id="honorific_agreement",
            evaluation_level="call",
            annotation_label=label,
            severity=severity,
            provenance_json=json.dumps(
                {"packet_batch_id": "x", "judge_visible": True, "phase": phase}
            ),
            phase=phase,
            completed=True,
        )

    native_path = _nativeness_csv(
        tmp_path / "two_phase_native.csv",
        [
            phased_native("pre_reveal", "violation", 3),
            phased_native("post_reveal", "pass"),
        ],
    )
    collected = collect_human_labels([native_path])
    native_key = (
        CalibrationJudge.NATIVENESS,
        "honorific_agreement",
        "clip_001",
        "v1",
    )
    assert collected.labels[native_key] is ColdLabel.YES
    assert collected.verifications[native_key] is ColdLabel.NO

    # A two-phase return is a legitimate blind-label source: adjudication
    # accepts it (the pre-reveal layer is the rater column), unlike the
    # legacy single-layer judge-visible returns it still refuses.
    _, _, options, build = _build(tmp_path, monkeypatch)
    manifest_path = build_calibration_adjudication(
        CalibrationAdjudicationOptions(
            sidecar=build.sidecar_path,
            filled=[path],
            batch_name="hi_adjudication_two_phase",
            out_dir=tmp_path / "adjudication_two_phase",
            emit_zip=False,
        )
    )
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    assert manifest.provenance["raters"] == ["v1"]


def test_post_reveal_labels_score_the_phase_two_2x2(tmp_path, monkeypatch):
    _, _, options, build = _build(tmp_path, monkeypatch)
    marked = f"hi_calibration{JUDGE_VISIBLE_BATCH_SUFFIX}"

    def phased_row(phase, answer):
        return RubricAnnotationRow(
            batch=marked,
            rater="v1",
            clip_id="clip_001",
            language="hi",
            section="interaction",
            factor_id="unnecessary_repetition",
            question="q",
            answer=answer,
            phase=phase,
            completed=True,
        )

    # On a judge-FAIL cell the rater passes blind, then flips to a violation
    # on reveal: the blind 2x2 keeps the false alarm, the phase-2 2x2 scores
    # the corrected label as a true positive.
    path = _rubric_csv(
        tmp_path / "two_phase_revised.csv",
        "v1",
        [phased_row("pre_reveal", "No"), phased_row("post_reveal", "Yes")],
    )
    report = build_calibration_agreement(build.sidecar_path, [path])
    assert report.report_version == CALIBRATION_AGREEMENT_VERSION
    by_key = {(r.judge, r.factor_id): r for r in report.rows}
    row = by_key[(CalibrationJudge.QUALITY, "unnecessary_repetition")]
    # Blind 2x2 unchanged: the locked pre-reveal No stays a false alarm.
    assert row.metrics.counts.fp_opp == 1 and row.metrics.counts.tp == 0
    assert row.metrics.precision == 0.0
    # Phase-2 2x2: the post-reveal Yes moves the cell fp -> tp.
    post = row.post_reveal_metrics
    assert post is not None
    assert post.counts.tp == 1 and post.counts.fp_opp == 0
    assert post.precision == 1.0 and post.recall == 1.0 and post.f1 == 1.0
    # Verified precision keeps its judge-positives-only semantics.
    assert row.n_verifier_confirmed == 1 and row.verified_precision == 1.0
    # Factors the precision pool never labeled carry no phase-2 table.
    untouched = [
        r
        for r in report.rows
        if (r.judge, r.factor_id)
        != (CalibrationJudge.QUALITY, "unnecessary_repetition")
    ]
    assert untouched
    assert all(r.post_reveal_metrics is None for r in untouched)
    rendered = render_calibration_agreement(report)
    assert "p2prec" in rendered and "p2rec" in rendered


def test_phase_on_an_unbranded_row_is_loud(tmp_path):
    row = RubricAnnotationRow(
        batch="innocent_blind_batch",
        rater="r1",
        clip_id="clip_001",
        language="hi",
        section="interaction",
        factor_id="unnecessary_repetition",
        question="q",
        answer="Yes",
        phase="pre_reveal",
        completed=True,
    )
    path = _rubric_csv(tmp_path / "contradictory.csv", "r1", [row])
    with pytest.raises(ValueError, match="judge-visible brand"):
        collect_human_labels([path])


def test_legacy_nativeness_headers_without_phase_still_ingest(tmp_path):
    legacy_headers = [
        header for header in NativenessHumanLabelRow.headers() if header != "phase"
    ]
    row = _nativeness_row("r1", "clip_001", "pass")
    cells = {
        header: value for header, value in row.to_cells().items() if header != "phase"
    }
    path = tmp_path / "legacy_native.csv"
    write_family_csv(path, legacy_headers, [cells])
    parsed = ingest_browser_csv(path)
    assert parsed.kind == NATIVENESS_LABEL_KIND
    assert parsed.rows[0].phase is None
    assert parsed.rows[0].annotation_label is not None


def test_severity_join_compares_human_minor_major_to_judge_ratings(
    tmp_path, monkeypatch
):
    # Severity lives only on semantic cells, which exist only in the pre-cut
    # (legacy) cell view. Downgrade the sidecar to a pre-cut one — the shape
    # the live es/pt waves carry — to exercise that agreement path.
    _, artifact, options, build = _build(tmp_path, monkeypatch)
    sidecar = load_calibration_sidecar(build.sidecar_path)
    legacy = sidecar.model_copy(
        update={"cells_version": None, "conversation_artifact": str(artifact)}
    )
    build.sidecar_path.write_text(legacy.model_dump_json(indent=2))
    sidecar = load_calibration_sidecar(build.sidecar_path)
    clip_of = {row.sim_id: row.clip_id for row in sidecar.rows}
    s1 = clip_of["s1"]

    def semantic_row(factor_id, severity):
        return RubricAnnotationRow(
            batch="calib",
            rater="r1",
            clip_id=s1,
            language="hi",
            section="semantic",
            factor_id=factor_id,
            question="q",
            answer="Yes",
            severity=severity,
            completed=True,
        )

    # The fixture stores information_loss as a MAJOR (rating 1) and
    # question_quality as a MINOR (rating 2) on s1.
    path = _rubric_csv(
        tmp_path / "severity.csv",
        "r1",
        [
            semantic_row("information_loss", "major"),
            semantic_row("question_quality", "major"),
        ],
    )
    report = build_calibration_agreement(build.sidecar_path, [path])
    rows = {(row.judge, row.factor_id): row for row in report.rows}
    matched = rows[(CalibrationJudge.SEMANTIC, "information_loss")]
    assert matched.n_severity_compared == 1
    assert matched.n_severity_matched == 1
    assert matched.severity_agreement == 1.0
    mismatched = rows[(CalibrationJudge.SEMANTIC, "question_quality")]
    assert mismatched.n_severity_compared == 1
    assert mismatched.n_severity_matched == 0
    assert mismatched.severity_agreement == 0.0
    rendered = render_calibration_agreement(report)
    assert "sev" in rendered


def test_agreement_report_matches_hand_computed_confusions(tmp_path, monkeypatch):
    _, _, options, build = _build(tmp_path, monkeypatch)
    filled = _synthetic_returns(tmp_path, build.sidecar_path)
    report = build_calibration_agreement(build.sidecar_path, filled)

    assert report.language == "hi"
    assert report.raters == ["r1", "r2"]
    assert report.n_calls == 3
    rows = {(row.judge, row.factor_id): row for row in report.rows}

    # quality:unnecessary_repetition — judge FAIL on s1 only; labels pooled
    # over both raters: tp=2 (s1), fn=1 (r2 yes on s4), tn=3.
    quality = rows[(CalibrationJudge.QUALITY, "unnecessary_repetition")]
    counts = quality.metrics.counts
    assert (counts.tp, counts.fn, counts.fp_opp, counts.tn) == (2, 1, 0, 3)
    assert quality.metrics.precision == 1.0
    assert quality.metrics.recall == pytest.approx(2 / 3)
    assert quality.metrics.kappa == pytest.approx(2 / 3)
    assert quality.metrics.n_opportunity == 6
    assert quality.gate == "LIVE"
    # Rater-rater ceiling on the same cells: [tp,fn,fp,tn] = [1,0,1,1].
    assert quality.human_kappa == pytest.approx(0.4)
    assert quality.human_agreement == pytest.approx(2 / 3)

    # delivery:digit_readout — no longer a pack factor: the stored verdict
    # mints no judge cell and the report carries no row for it.
    assert (CalibrationJudge.DELIVERY, "digit_readout") not in rows

    # delivery:fidelity — the number_date_currency answers join the
    # fidelity aggregate against s1's fidelity-axis FAIL: r1 said
    # no-opportunity (false alarm charged to precision, off the 2x2), r2
    # confirmed.
    fidelity = rows[(CalibrationJudge.DELIVERY, "fidelity")]
    assert fidelity.metrics.counts.fa_noopp == 1
    assert fidelity.metrics.counts.tp == 1
    assert fidelity.metrics.precision == pytest.approx(1 / 2)

    # nativeness:natural_word_choice — both raters confirmed the s1 FAIL.
    native = rows[(CalibrationJudge.NATIVENESS, "natural_word_choice")]
    assert native.metrics.counts.tp == 2
    assert native.metrics.precision == 1.0
    assert native.metrics.recall == 1.0

    # semantic is cut from the instrument (2026-08-25): the folded cell view
    # mints no semantic cells, so the report carries no semantic rows even
    # though the interaction answers used to double as their twins.
    assert not any(judge is CalibrationJudge.SEMANTIC for judge, _ in rows)

    rendered = render_calibration_agreement(report)
    assert "unnecessary_repetition" in rendered
    assert "delivery-prompt-test" in rendered
    assert "hkappa" in rendered


def test_agreement_report_rolls_factors_into_calibration_buckets(tmp_path, monkeypatch):
    """Factor rows carry their bucket/disposition and the report carries the
    construct-level rollup (clip merged per rater, any-yes / any-FAIL)."""
    _, _, options, build = _build(tmp_path, monkeypatch)
    filled = _synthetic_returns(tmp_path, build.sidecar_path)
    report = build_calibration_agreement(build.sidecar_path, filled)

    rows = {(row.judge, row.factor_id): row for row in report.rows}
    rep = rows[(CalibrationJudge.QUALITY, "unnecessary_repetition")]
    assert rep.bucket == "redundancy"
    assert rep.disposition is FactorDisposition.SCORED
    native = rows[(CalibrationJudge.NATIVENESS, "natural_word_choice")]
    assert native.bucket == "phrasing"
    assert native.disposition is FactorDisposition.SCORED

    bucket_rows = {row.bucket_id: row for row in report.bucket_rows}
    assert "redundancy" in bucket_rows
    redundancy = bucket_rows["redundancy"]
    assert redundancy.judge is CalibrationJudge.QUALITY
    assert set(redundancy.factor_ids) == {
        "unnecessary_repetition",
        "unnecessary_tool_call",
    }
    counts = redundancy.metrics.counts
    # The bucket 2x2 is clip-level: it can never hold more scoreable cells
    # than the constituent factor rows combined, and the s1 judge FAIL that
    # both raters confirmed must survive the merge as bucket TPs.
    assert counts.tp >= 2
    assert redundancy.metrics.n_opportunity > 0
    # Retired factors never mint a bucket row of their own.
    assert all(
        factor_id not in RETIRED_FACTORS
        for row in report.bucket_rows
        for factor_id in row.factor_ids
    )

    rendered = render_calibration_agreement(report)
    assert "Calibration buckets" in rendered
    assert "redundancy" in rendered
    assert "-> redundancy" in rendered
    assert "[retired]" in rendered
