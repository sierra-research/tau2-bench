"""Calibration buckets: the construct-level rollup of judge factors.

The 2026-08-31 recall_20 calibration round showed that a large share of
per-factor disagreement is boundary misattribution between sibling factors,
not judge blindness: the judge and raters agree a call carries a defect but
file it under different factor ids (a re-ask loop is both a repeated question
and a duplicate lookup; an auth mismatch IS an incorrect tool parameter).
Scored at the factor level those cross-filings count as an FP *and* an FN;
merged at the construct level they cancel. Measured on the recall_20 gold set
the rollup moved redundancy 0.72/0.76 -> 0.84, wrong-arguments 0.89/0.97 ->
0.94, and nativeness wording 0.72-0.85 -> 0.90.

Buckets are therefore the unit of CALIBRATION and REPORTING. Factors remain
the unit of JUDGING. The calibrated ``natural_word_choice`` factor already
combines word choice, translationese, and verb morphology in its own rubric;
those former standalone ids must not be added back by the reporting rollup.
Other constituents keep their independent rubric lines and evidence. Nothing
in this module changes what judges score at run time.

A bucket cell is a clip-level either-flag merge per rater:
- human bucket label: YES if any constituent label is YES, else UNSURE if any
  is UNSURE, else NO if any is NO, else NO_OPPORTUNITY if any is that, else
  unlabeled.
- judge bucket outcome: FAIL if any constituent outcome is FAIL, else PASS if
  any is PASS, else NO_OPPORTUNITY if any is that, else ERROR (unscorable).
The merged (label, outcome) pair then folds through the one shared
``accumulate_cold_cell`` so bucket 2x2 semantics can never drift from factor
semantics.
"""

from enum import Enum
from typing import Annotated, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from tau2.annotation.calibration_draw import CalibrationJudge
from tau2.annotation.models import ColdLabel
from tau2.data_model.simulation import JudgeOutcome

#: Bump when bucket membership or merge semantics change; stamped on reports.
#: v2: wording_phrasing renamed to phrasing + gains the deterministic
#: email_symbol_verbalization; ko honorific_levels and backchannel_frequency
#: retired; quality buckets grouped under the "efficiency" display label;
#: intonation (prosody) retired outright (calibration F1 0.61, no prompting
#: round planned).
#: v3: translationese and verb_morphology are absorbed by the calibrated
#: natural_word_choice factor. regional_consistency remains independent in
#: the non-Spanish packs that still define it.
#: v4: Korean honorific agreement is restored to the grammar bucket after its
#: utterance-level precision and recall evidence was normalized into the
#: release archive.
CALIBRATION_BUCKETS_VERSION = "calibration-buckets-v4"

#: Display label for each judge family's bucket group in reports. The quality
#: judge's buckets measure execution efficiency (correct tool use, no wasted
#: turns or calls), not "quality" at large.
JUDGE_GROUP_LABELS: dict[CalibrationJudge, str] = {
    CalibrationJudge.NATIVENESS: "nativeness",
    CalibrationJudge.QUALITY: "efficiency",
    CalibrationJudge.DELIVERY: "delivery",
}


class FactorDisposition(str, Enum):
    """Why a factor does or does not participate in bucket calibration."""

    SCORED = "scored"  # member of a calibration bucket
    RETIRED = "retired"  # prompt-immune FP flood; not judged toward buckets
    AUDITION = "audition"  # voice-level property; gated at voice audition
    SHADOW = "shadow"  # judged and reported per-factor, never bucketed


class CalibrationBucket(BaseModel):
    """One construct-level rollup of sibling factors under a judge family."""

    model_config = ConfigDict(frozen=True)

    bucket_id: Annotated[str, Field(description="Stable rollup identifier.")]
    judge: Annotated[CalibrationJudge, Field(description="Owning judge family.")]
    factor_ids: Annotated[
        tuple[str, ...],
        Field(min_length=1, description="Constituent factor ids (either-flag)."),
    ]
    rationale: Annotated[str, Field(description="Why these factors are one construct.")]


CALIBRATION_BUCKETS: tuple[CalibrationBucket, ...] = (
    CalibrationBucket(
        bucket_id="phrasing",
        judge=CalibrationJudge.NATIVENESS,
        factor_ids=(
            "natural_word_choice",
            "regional_consistency",
            "modal_particles",
            "name_address_conventions",
            "email_symbol_verbalization",
        ),
        rationale=(
            "Does the agent word things like a native rep. The combined "
            "natural-word-choice factor owns word choice, calques, and verb "
            "morphology. Regional consistency remains a separate signal only "
            "in packs that still define it, so it continues to roll up here. "
            "email_symbol_verbalization is deterministic word choice (arroba "
            "vs 'at') and rides along unvalidated."
        ),
    ),
    CalibrationBucket(
        bucket_id="grammar",
        judge=CalibrationJudge.NATIVENESS,
        factor_ids=(
            "gender_agreement",
            "honorific_agreement",
            "counting_units",
            "register_formality",
        ),
        rationale=(
            "Independent grammatical signals: gender agreement, honorific "
            "agreement, counters/units, and politeness register "
            "(pronoun/verb-level formality). Verb morphology is already owned "
            "by natural_word_choice."
        ),
    ),
    CalibrationBucket(
        bucket_id="tool_correctness",
        judge=CalibrationJudge.QUALITY,
        factor_ids=(
            "incorrect_tool_parameters",
            "auth_arg_mismatch",
            "agent_caused_tool_error",
        ),
        rationale=(
            "Did the agent use tools correctly: wrong or invented argument "
            "values (auth_arg_mismatch is definitionally a specialization of "
            "incorrect_tool_parameters, so cross-filing between them is "
            "structural) and culpable errored calls."
        ),
    ),
    CalibrationBucket(
        bucket_id="redundancy",
        judge=CalibrationJudge.QUALITY,
        factor_ids=("unnecessary_repetition", "unnecessary_tool_call"),
        rationale=(
            "A re-ask loop is one defect that surfaces as both a repeated "
            "question and a duplicate lookup; judge and raters file it under "
            "different ids on the same calls."
        ),
    ),
    CalibrationBucket(
        bucket_id="faithfulness",
        judge=CalibrationJudge.DELIVERY,
        factor_ids=("fidelity", "tone_meaning_flip"),
        rationale=(
            "Did the audio convey the intended words: a tone error that flips "
            "meaning is acoustically a word substitution."
        ),
    ),
)

#: Retired by owner ruling. 2026-08-31: request_politeness produced
#: prompt-immune FP floods (recall_20 F1 0.15-0.18). 2026-09-01: ko
#: honorific_levels (zero fails in
#: the calibration corpus; its only historical firings are fragment-split
#: false positives) and backchannel_frequency (acknowledgment-word rate with
#: an uncalibrated 1-6/min pass band; left out of the reporting groups).
RETIRED_FACTORS: frozenset[str] = frozenset(
    {
        "request_politeness",
        "honorific_levels",
        "backchannel_frequency",
        "intonation",
    }
)

#: Voice-level property: pooled per (language, voice) in the audition gate
#: table, never per-call precision/recall.
AUDITION_FACTORS: frozenset[str] = frozenset({"regional_accent_consistency"})

#: Judged and reported per-factor but excluded from buckets. Empty since the
#: 2026-09-01 rulings (intonation moved to RETIRED); factors unknown to this
#: catalog still report SHADOW via ``factor_disposition``.
SHADOW_FACTORS: frozenset[str] = frozenset()

_MEMBERSHIP: dict[tuple[CalibrationJudge, str], str] = {
    (bucket.judge, factor_id): bucket.bucket_id
    for bucket in CALIBRATION_BUCKETS
    for factor_id in bucket.factor_ids
}

_UNBUCKETED = RETIRED_FACTORS | AUDITION_FACTORS | SHADOW_FACTORS
if _UNBUCKETED & {factor_id for _, factor_id in _MEMBERSHIP}:
    raise ValueError("a retired/audition/shadow factor appears in a bucket")


def bucket_for_factor(judge: CalibrationJudge, factor_id: str) -> Optional[str]:
    """The bucket a factor rolls into, or None for unbucketed dispositions."""
    return _MEMBERSHIP.get((judge, factor_id))


def factor_disposition(judge: CalibrationJudge, factor_id: str) -> FactorDisposition:
    """Classify a factor id for reporting."""
    if factor_id in RETIRED_FACTORS:
        return FactorDisposition.RETIRED
    if factor_id in AUDITION_FACTORS:
        return FactorDisposition.AUDITION
    if bucket_for_factor(judge, factor_id) is not None:
        return FactorDisposition.SCORED
    return FactorDisposition.SHADOW


_LABEL_PRIORITY = (
    ColdLabel.YES,
    ColdLabel.UNSURE,
    ColdLabel.NO,
    ColdLabel.NO_OPPORTUNITY,
)


def bucket_label(labels: Iterable[Optional[ColdLabel]]) -> Optional[ColdLabel]:
    """Merge constituent human labels into one bucket label (any-yes wins)."""
    present = {label for label in labels if label is not None}
    for label in _LABEL_PRIORITY:
        if label in present:
            return label
    return None


_OUTCOME_PRIORITY = (
    JudgeOutcome.FAIL,
    JudgeOutcome.PASS,
    JudgeOutcome.NO_OPPORTUNITY,
)


def bucket_outcome(
    outcomes: Iterable[Optional[JudgeOutcome]],
) -> Optional[JudgeOutcome]:
    """Merge constituent judge outcomes into one bucket outcome (any-FAIL wins).

    A mix of only errored/deferred verdicts is unscorable and reports as
    ERROR so ``accumulate_cold_cell`` books it under judge_unscored.
    """
    present = {outcome for outcome in outcomes if outcome is not None}
    for outcome in _OUTCOME_PRIORITY:
        if outcome in present:
            return outcome
    if present:
        return JudgeOutcome.ERROR
    return None
