# Copyright Sierra
"""Judge-calibration agreement: the cold-sheet and precision-sheet semantics.

Everything here is pure (no I/O, no LLM): the cold-sheet opportunity-gated
precision/recall, precision adjudication, inter-annotator kappa, and the
pipeline/native confusion summary shared by the translation and
communicate-judge calibrations.

This is the round that must happen BEFORE a wave is fitted — a judge feeding
``judged_*`` features is a judge whose per-language precision has been
adjudicated. ``gate`` and ``cohens_kappa`` live in ``evaluation.py`` (one gate,
one place) and are imported from there.
"""

from typing import Annotated, Iterable, Optional

from pydantic import BaseModel, Field

from tau2.annotation.evaluation import cohens_kappa
from tau2.annotation.models import (
    ColdLabel,
    ColdSheet,
    ColdSidecarRow,
    IssueTypeCell,
    JudgePrecisionRow,
    PrecisionVerdict,
)
from tau2.data_model.simulation import JudgeOutcome


def _ratio(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


# ---------------------------------------------------------------------------
# Cold (blind) calibration: opportunity-gated confusion semantics
# ---------------------------------------------------------------------------


class ColdCounts(BaseModel):
    """Accumulated cold-sheet cells for one (language, factor)."""

    tp: Annotated[int, Field(description="Human yes, judge FAIL.")] = 0
    fn: Annotated[int, Field(description="Human yes, judge missed (PASS/no-opp).")] = 0
    fp_opp: Annotated[int, Field(description="Human no, judge FAIL (false alarm).")] = 0
    tn: Annotated[int, Field(description="Human no, judge PASS.")] = 0
    fa_noopp: Annotated[
        int,
        Field(
            description="Judge FAIL where the human said no-opportunity — a "
            "false alarm charged to precision but off the 2x2."
        ),
    ] = 0
    noopp_opp: Annotated[
        int,
        Field(
            description="Human no (opportunity, no violation) but judge "
            "NO_OPPORTUNITY: they disagree about whether an opportunity "
            "existed, so NOT a confident TN — excluded from the 2x2 so it "
            "can't inflate agreement/precision/recall/kappa."
        ),
    ] = 0
    unsure: Annotated[int, Field(description="Human marked unsure.")] = 0
    unlabeled: Annotated[int, Field(description="Cell left blank.")] = 0
    judge_unscored: Annotated[
        int,
        Field(description="Judge verdict error/missing/deferred — unscorable."),
    ] = 0


class ColdFactorMetrics(BaseModel):
    """Precision/recall/F1/kappa for one (language, factor) from cold counts."""

    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    kappa: Annotated[
        Optional[float], Field(description="Kappa over opportunity rows only.")
    ] = None
    n_opportunity: Annotated[
        int, Field(description="Rows inside the 2x2 (tp+fn+fp_opp+tn).")
    ] = 0
    counts: ColdCounts


def cold_metrics(counts: ColdCounts) -> ColdFactorMetrics:
    """Precision/recall/F1/kappa from accumulated cold counts."""
    tp, fn, fp_opp, tn = counts.tp, counts.fn, counts.fp_opp, counts.tn
    fa = counts.fa_noopp  # judge FAIL where human said no-opportunity
    precision = _ratio(tp, tp + fp_opp + fa)  # false alarms charged to precision
    recall = _ratio(tp, tp + fn)
    # Guard on None (undefined), not falsiness — a real 0.0 precision/recall
    # must still yield F1 = 0.0, not None.
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return ColdFactorMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        kappa=cohens_kappa(tp, fn, fp_opp, tn),  # opportunity rows only
        n_opportunity=tp + fn + fp_opp + tn,
        counts=counts,
    )


_UNSCORED = {None, JudgeOutcome.ERROR, JudgeOutcome.DEFERRED}


def accumulate_cold_cell(
    counts: ColdCounts, label: Optional[ColdLabel], judge: Optional[JudgeOutcome]
) -> None:
    """Fold one (human label, judge outcome) cell into the confusion counts.

    THE opportunity-gated semantics — shared by the wide cold sheet and the
    VE grid so the two calibrations can never disagree about what a TP is.
    """
    # Unlabeled precedes judge_unscored: a cell the annotator never
    # touched must report as unlabeled regardless of the judge's state.
    if label is None:
        counts.unlabeled += 1
        return
    if judge in _UNSCORED:
        counts.judge_unscored += 1
        return
    judge_fail = judge is JudgeOutcome.FAIL
    judge_noopp = judge is JudgeOutcome.NO_OPPORTUNITY
    if label is ColdLabel.UNSURE:
        counts.unsure += 1
    elif label is ColdLabel.YES:
        # Real violation: judge FAIL catches it; PASS/NO_OPPORTUNITY miss it.
        if judge_fail:
            counts.tp += 1
        else:
            counts.fn += 1
    elif label is ColdLabel.NO:
        # Opportunity but no violation. A judge FAIL is a false alarm; a
        # judge PASS is a true negative; a judge NO_OPPORTUNITY disagrees
        # with the human about whether an opportunity existed — it is NOT
        # a confident TN, so bucket it separately (excluded from the 2x2).
        if judge_fail:
            counts.fp_opp += 1
        elif judge_noopp:
            counts.noopp_opp += 1
        else:
            counts.tn += 1
    elif label is ColdLabel.NO_OPPORTUNITY:
        # Judge FAIL on a no-opportunity row is a false alarm (charged to
        # precision); a judge agreeing there's nothing to flag needs no
        # counter.
        if judge_fail:
            counts.fa_noopp += 1


def analyze_cold(
    sheet: ColdSheet, sidecar: list[ColdSidecarRow]
) -> dict[tuple[str, str], ColdFactorMetrics]:
    """Per-(language, factor) cold metrics: annotator labels vs the sidecar."""
    judge_by_key = {(r.sim_id, r.factor_id): r.judge_outcome for r in sidecar}
    acc: dict[tuple[str, str], ColdCounts] = {}
    for lang, sim_id, factor_id, label in sheet.melt():
        counts = acc.setdefault((lang, factor_id), ColdCounts())
        accumulate_cold_cell(counts, label, judge_by_key.get((sim_id, factor_id)))
    return {key: cold_metrics(counts) for key, counts in acc.items()}


# The opportunity gate shared by every human-human kappa: only cells BOTH
# raters marked yes/no (a genuine opportunity) enter the 2x2.
_OPPORTUNITY_LABELS = frozenset({ColdLabel.YES, ColdLabel.NO})


def accumulate_pair_cell(
    acc: dict[tuple[str, str], list[int]],
    key: tuple[str, str],
    a: Optional[ColdLabel],
    b: Optional[ColdLabel],
) -> None:
    """Fold one double-labeled cell into the per-key 2x2 (opportunity-gated).

    Rater A as rows, B as cols; yes = positive -> ``[tp, fn, fp, tn]``. A pair
    where either rater did not mark a genuine opportunity is skipped. Shared
    by ``inter_annotator_kappa`` and ``ve_human_kappa`` so the gate cannot
    drift between the cold and VE calibrations.
    """
    if a not in _OPPORTUNITY_LABELS or b not in _OPPORTUNITY_LABELS:
        return
    cells = acc.setdefault(key, [0, 0, 0, 0])
    if a is ColdLabel.YES and b is ColdLabel.YES:
        cells[0] += 1
    elif a is ColdLabel.YES:
        cells[1] += 1
    elif b is ColdLabel.YES:
        cells[2] += 1
    else:
        cells[3] += 1


def inter_annotator_kappa(
    sheet_a: ColdSheet, sheet_b: ColdSheet
) -> dict[tuple[str, str], Optional[float]]:
    """Human-human kappa per (language, factor) over shared opportunity rows.

    The ceiling the judge is chasing: if two natives don't agree, the factor
    is underspecified, not the judge wrong. Only cells both annotators marked
    yes/no (a genuine opportunity) enter the 2x2.
    """
    a_labels: dict[tuple[str, str], Optional[ColdLabel]] = {}
    a_lang: dict[tuple[str, str], str] = {}
    for lang, sim_id, factor_id, label in sheet_a.melt():
        a_labels[(sim_id, factor_id)] = label
        a_lang[(sim_id, factor_id)] = lang
    b_labels = {
        (sim_id, factor_id): label for _lang, sim_id, factor_id, label in sheet_b.melt()
    }
    acc: dict[tuple[str, str], list[int]] = {}
    for key, a in a_labels.items():
        accumulate_pair_cell(acc, (a_lang.get(key, ""), key[1]), a, b_labels.get(key))
    return {key: cohens_kappa(*cells) for key, cells in acc.items()}


# ---------------------------------------------------------------------------
# Precision (adjudicate) calibration
# ---------------------------------------------------------------------------


class PrecisionFactorMetrics(BaseModel):
    """Adjudicated precision for one (language, factor)."""

    precision: Optional[float] = None
    n_adjudicated: Annotated[
        int, Field(description="Rows adjudicated real or false (unsure excluded).")
    ] = 0
    real: int = 0
    false_alarm: int = 0
    unsure: int = 0
    unlabeled: int = 0


def tally_precision_verdict(
    m: PrecisionFactorMetrics, verdict: object, *, real: object, false_alarm: object
) -> None:
    """Fold one adjudicated verdict into the precision tally.

    ``real`` / ``false_alarm`` are the calling surface's enum members (the
    precision sheet's ``PrecisionVerdict``, the VE seeds' ``VeVerdict``);
    None counts as unlabeled and anything else as unsure. Shared so the two
    adjudication surfaces bucket identically.
    """
    if verdict is None:
        m.unlabeled += 1
    elif verdict is real:
        m.real += 1
    elif verdict is false_alarm:
        m.false_alarm += 1
    else:
        m.unsure += 1


def finalize_precision(
    acc: dict[tuple[str, str], PrecisionFactorMetrics],
) -> dict[tuple[str, str], PrecisionFactorMetrics]:
    """Derive n_adjudicated + precision on every tally (returns ``acc``)."""
    for m in acc.values():
        m.n_adjudicated = m.real + m.false_alarm
        m.precision = _ratio(m.real, m.n_adjudicated)
    return acc


def analyze_precision(
    rows: Iterable[JudgePrecisionRow],
) -> dict[tuple[str, str], PrecisionFactorMetrics]:
    """Per-(language, factor) precision from adjudicated judge FAIL rows."""
    acc: dict[tuple[str, str], PrecisionFactorMetrics] = {}
    for row in rows:
        m = acc.setdefault((row.language, row.factor_id), PrecisionFactorMetrics())
        tally_precision_verdict(
            m,
            row.verdict,
            real=PrecisionVerdict.REAL,
            false_alarm=PrecisionVerdict.FALSE_ALARM,
        )
    return finalize_precision(acc)


# ---------------------------------------------------------------------------
# Pipeline/native agreement (translation + communicate-judge calibrations)
# ---------------------------------------------------------------------------


class AgreementRecord(BaseModel):
    """One annotated row compared against the pipeline/judge verdict."""

    task_id: str = ""
    sim_id: str = ""
    field: Annotated[
        str,
        Field(description="Row grouping key: field label, or criterion text."),
    ] = ""
    pipeline_ok: Annotated[
        Optional[bool],
        Field(
            description="The automated verdict; None for verdict-less rows "
            "(excluded from the comparison, still counted in n_records)."
        ),
    ] = None
    native_ok: Annotated[bool, Field(description="The native's verdict.")]
    issue_type: Annotated[
        IssueTypeCell,
        Field(description="Closed-taxonomy issue named by the annotator."),
    ] = None
    comments: str = ""


class ConfusionCells(BaseModel):
    """Pipeline vs native 2x2 (both said ok / only one did / neither)."""

    both_ok: int = 0
    pipeline_only: int = 0
    native_only: int = 0
    neither: int = 0


class AgreementSummary(BaseModel):
    """Agreement summary over ingest records (shared by both judge surfaces)."""

    n_records: int
    n_compared: Annotated[
        int, Field(description="Records with a pipeline verdict to compare.")
    ]
    percent_agreement: Optional[float] = None
    cohens_kappa: Optional[float] = None
    confusion: ConfusionCells
    per_field: dict[str, ConfusionCells]
    per_issue_type: dict[str, ConfusionCells]


def confusion_cells(records: Iterable[AgreementRecord]) -> ConfusionCells:
    cells = ConfusionCells()
    for record in records:
        if record.pipeline_ok is None:
            continue
        if record.pipeline_ok and record.native_ok:
            cells.both_ok += 1
        elif record.pipeline_ok:
            cells.pipeline_only += 1
        elif record.native_ok:
            cells.native_only += 1
        else:
            cells.neither += 1
    return cells


def summarize_agreement(records: list[AgreementRecord]) -> AgreementSummary:
    """Percent agreement + kappa + confusion, overall and per field/issue."""
    pairs = [
        (bool(r.pipeline_ok), r.native_ok) for r in records if r.pipeline_ok is not None
    ]
    tp = sum(1 for a, b in pairs if a and b)
    fn = sum(1 for a, b in pairs if a and not b)
    fp = sum(1 for a, b in pairs if not a and b)
    tn = sum(1 for a, b in pairs if not a and not b)
    per_field = {
        field: confusion_cells([r for r in records if r.field == field])
        for field in sorted({r.field for r in records})
    }
    per_issue_type = {
        issue.value: confusion_cells([r for r in records if r.issue_type is issue])
        for issue in sorted({r.issue_type for r in records if r.issue_type is not None})
    }
    return AgreementSummary(
        n_records=len(records),
        n_compared=len(pairs),
        percent_agreement=(
            sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else None
        ),
        cohens_kappa=cohens_kappa(tp, fn, fp, tn),
        confusion=confusion_cells(records),
        per_field=per_field,
        per_issue_type=per_issue_type,
    )
