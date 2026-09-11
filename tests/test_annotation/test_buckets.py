# Copyright Sierra
"""Bucket catalog integrity and the clip-level merge semantics."""

from tau2.annotation.buckets import (
    AUDITION_FACTORS,
    CALIBRATION_BUCKETS,
    RETIRED_FACTORS,
    SHADOW_FACTORS,
    FactorDisposition,
    bucket_for_factor,
    bucket_label,
    bucket_outcome,
    factor_disposition,
)
from tau2.annotation.calibration_draw import CalibrationJudge
from tau2.annotation.metrics import ColdCounts, accumulate_cold_cell
from tau2.annotation.models import ColdLabel
from tau2.data_model.simulation import JudgeOutcome


def test_catalog_has_no_factor_in_two_buckets():
    seen: set[tuple[CalibrationJudge, str]] = set()
    for bucket in CALIBRATION_BUCKETS:
        for factor_id in bucket.factor_ids:
            key = (bucket.judge, factor_id)
            assert key not in seen, f"{factor_id} appears in two buckets"
            seen.add(key)


def test_combined_naturalness_replaces_only_its_standalone_factors():
    """The calibrated combined judge must not erase independent pack factors."""
    assert (
        bucket_for_factor(CalibrationJudge.NATIVENESS, "natural_word_choice")
        == "phrasing"
    )
    assert (
        bucket_for_factor(CalibrationJudge.NATIVENESS, "regional_consistency")
        == "phrasing"
    )
    assert bucket_for_factor(CalibrationJudge.NATIVENESS, "translationese") is None
    assert bucket_for_factor(CalibrationJudge.NATIVENESS, "verb_morphology") is None


def test_retired_audition_shadow_factors_are_never_bucketed():
    for factor_id in RETIRED_FACTORS | AUDITION_FACTORS | SHADOW_FACTORS:
        for judge in CalibrationJudge:
            assert bucket_for_factor(judge, factor_id) is None


def test_factor_disposition_classifies_all_four_ways():
    assert (
        factor_disposition(CalibrationJudge.NATIVENESS, "request_politeness")
        is FactorDisposition.RETIRED
    )
    assert (
        factor_disposition(CalibrationJudge.DELIVERY, "regional_accent_consistency")
        is FactorDisposition.AUDITION
    )
    # v2 rulings: intonation, ko honorific_levels, and backchannel_frequency
    # are retired outright.
    assert (
        factor_disposition(CalibrationJudge.DELIVERY, "intonation")
        is FactorDisposition.RETIRED
    )
    assert (
        factor_disposition(CalibrationJudge.NATIVENESS, "honorific_levels")
        is FactorDisposition.RETIRED
    )
    assert (
        factor_disposition(CalibrationJudge.NATIVENESS, "backchannel_frequency")
        is FactorDisposition.RETIRED
    )
    assert (
        factor_disposition(CalibrationJudge.QUALITY, "unnecessary_repetition")
        is FactorDisposition.SCORED
    )
    assert (
        factor_disposition(CalibrationJudge.NATIVENESS, "honorific_agreement")
        is FactorDisposition.SCORED
    )
    assert (
        bucket_for_factor(CalibrationJudge.NATIVENESS, "honorific_agreement")
        == "grammar"
    )
    # email_symbol_verbalization joined the phrasing bucket in v2.
    assert (
        bucket_for_factor(CalibrationJudge.NATIVENESS, "email_symbol_verbalization")
        == "phrasing"
    )
    # A factor unknown to the catalog reports SHADOW, never crashes.
    assert (
        factor_disposition(CalibrationJudge.NATIVENESS, "brand_new_factor")
        is FactorDisposition.SHADOW
    )


def test_bucket_label_priority_any_yes_wins():
    assert bucket_label([ColdLabel.NO, ColdLabel.YES]) is ColdLabel.YES
    assert bucket_label([ColdLabel.NO, ColdLabel.UNSURE]) is ColdLabel.UNSURE
    assert bucket_label([ColdLabel.NO_OPPORTUNITY, ColdLabel.NO]) is ColdLabel.NO
    assert bucket_label([ColdLabel.NO_OPPORTUNITY]) is ColdLabel.NO_OPPORTUNITY
    assert bucket_label([None, None]) is None
    assert bucket_label([None, ColdLabel.NO]) is ColdLabel.NO


def test_bucket_outcome_priority_any_fail_wins():
    assert bucket_outcome([JudgeOutcome.PASS, JudgeOutcome.FAIL]) is JudgeOutcome.FAIL
    assert (
        bucket_outcome([JudgeOutcome.NO_OPPORTUNITY, JudgeOutcome.PASS])
        is JudgeOutcome.PASS
    )
    assert (
        bucket_outcome([JudgeOutcome.NO_OPPORTUNITY, JudgeOutcome.NO_OPPORTUNITY])
        is JudgeOutcome.NO_OPPORTUNITY
    )
    # Only errored/deferred constituents: unscorable, reported as ERROR so
    # accumulate_cold_cell books judge_unscored.
    assert (
        bucket_outcome([JudgeOutcome.ERROR, JudgeOutcome.DEFERRED])
        is JudgeOutcome.ERROR
    )
    assert bucket_outcome([]) is None


def test_misattribution_cancels_at_the_bucket_level():
    """The motivating case: the rater files a defect under factor A, the
    judge under sibling factor B. Factor-level that is an FN plus an FP;
    bucket-level the merged cell is a single TP."""
    factor_a = ColdCounts()
    accumulate_cold_cell(factor_a, ColdLabel.YES, JudgeOutcome.PASS)
    factor_b = ColdCounts()
    accumulate_cold_cell(factor_b, ColdLabel.NO, JudgeOutcome.FAIL)
    assert factor_a.fn == 1 and factor_b.fp_opp == 1

    merged = ColdCounts()
    accumulate_cold_cell(
        merged,
        bucket_label([ColdLabel.YES, ColdLabel.NO]),
        bucket_outcome([JudgeOutcome.PASS, JudgeOutcome.FAIL]),
    )
    assert (merged.tp, merged.fn, merged.fp_opp) == (1, 0, 0)
