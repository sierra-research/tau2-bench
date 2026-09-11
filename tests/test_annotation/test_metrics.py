# Copyright Sierra
"""Agreement math: kappa, cold confusion semantics, precision, summaries.

Ports every math case from the legacy calibration-analyze and factory
agreement suites onto the typed models. One deliberate deviation: the two
legacy kappa implementations disagreed on the degenerate pe==1 case (None vs
1.0/0.0); the unified ``cohens_kappa`` keeps the defined convention (perfect
agreement → 1.0).
"""

import pytest

from tau2.annotation.evaluation import gate
from tau2.annotation.metrics import (
    AgreementRecord,
    ColdCounts,
    analyze_cold,
    analyze_precision,
    cohens_kappa,
    cold_metrics,
    inter_annotator_kappa,
    summarize_agreement,
)
from tau2.annotation.models import (
    ColdLabel,
    ColdSheet,
    ColdSheetRow,
    ColdSidecarRow,
    FactorKeyRow,
    JudgePrecisionRow,
    PrecisionVerdict,
)

F = "register_formality"


def test_kappa_perfect_chance_and_degenerate():
    assert cohens_kappa(5, 0, 0, 5) == pytest.approx(1.0)  # perfect agreement
    assert abs(cohens_kappa(1, 1, 1, 1)) < 1e-9  # chance level
    assert cohens_kappa(0, 0, 0, 0) is None  # empty
    # Both raters constant: pe == 1. Perfect agreement is defined as 1.0
    # (legacy analyze half said None here; the factory half said 1.0 — the
    # unified form keeps the defined convention).
    assert cohens_kappa(10, 0, 0, 0) == pytest.approx(1.0)


def test_kappa_hand_computed_case():
    # a=4, b=1, c=2, d=3: po=0.7, pe=0.5*0.6+0.5*0.4=0.5, kappa=0.4.
    assert cohens_kappa(4, 1, 2, 3) == pytest.approx(0.4)


def _sheet(cells: dict[str, ColdLabel | None], lang="hi") -> ColdSheet:
    """A wide sheet with one factor column F, one row per sim in ``cells``."""
    key = FactorKeyRow(language=lang, factor_id=F, nuance=F)
    rows = [
        ColdSheetRow(
            transcript=f"tx-{sim}", sim_id=sim, language=lang, labels={F: label}
        )
        for sim, label in cells.items()
    ]
    return ColdSheet(rows=rows, factors=[key])


def _side(sim: str, outcome: str | None, lang="hi") -> ColdSidecarRow:
    return ColdSidecarRow(sim_id=sim, factor_id=F, language=lang, judge_outcome=outcome)


def test_analyze_cold_confusion_and_gate():
    sheet = _sheet(
        {
            "s1": ColdLabel.YES,  # human violation
            "s2": ColdLabel.YES,  # human violation
            "s3": ColdLabel.NO,  # human fine
            "s4": ColdLabel.NO_OPPORTUNITY,  # no opportunity
            "s5": ColdLabel.YES,  # human violation, judge errored -> unscored
        }
    )
    sidecar = [
        _side("s1", "fail"),  # TP
        _side("s2", "pass"),  # FN (judge missed)
        _side("s3", "fail"),  # FP (false alarm on a fine call)
        _side("s4", "fail"),  # false alarm on no-opportunity (charged to precision)
        _side("s5", "error"),  # unscored
    ]
    m = analyze_cold(sheet, sidecar)[("hi", F)]
    c = m.counts
    assert (c.tp, c.fn, c.fp_opp, c.tn, c.fa_noopp) == (1, 1, 1, 0, 1)
    assert c.judge_unscored == 1
    # precision = tp/(tp+fp_opp+fa_noopp) = 1/3 ; recall = tp/(tp+fn) = 1/2
    assert m.precision == pytest.approx(1 / 3)
    assert m.recall == pytest.approx(1 / 2)
    assert m.n_opportunity == 3  # s1,s2,s3 (noopp + unscored excluded)
    assert gate(m.precision) == "shadow"
    assert gate(1.0) == "LIVE"


def test_analyze_cold_readable_headers_map_via_key():
    # The factor column is headed by a readable nuance title; the factors key
    # maps it back to factor_id so the sidecar (keyed by factor_id) joins.
    header = "Formal address (usted vs tú)"
    key = FactorKeyRow(language="hi", factor_id=F, nuance=header)
    sheet = ColdSheet(
        rows=[
            ColdSheetRow(
                transcript="tx",
                sim_id="s1",
                language="hi",
                labels={header: ColdLabel.YES},
            ),
            ColdSheetRow(
                transcript="tx",
                sim_id="s2",
                language="hi",
                labels={header: ColdLabel.NO},
            ),
        ],
        factors=[key],
    )
    sidecar = [_side("s1", "fail"), _side("s2", "fail")]
    metrics = analyze_cold(sheet, sidecar)
    # Keyed by factor_id (mapped from the header), not the readable title.
    assert ("hi", F) in metrics and ("hi", header) not in metrics
    m = metrics[("hi", F)]
    assert m.counts.tp == 1 and m.counts.fp_opp == 1
    assert m.precision == pytest.approx(1 / 2)


def test_analyze_cold_noopp_not_true_negative():
    # Human saw an opportunity with no violation ("no"); the judge abstained
    # ("no_opportunity"). They disagree about whether an opportunity existed,
    # so this must NOT count as a confident true negative.
    m = analyze_cold(_sheet({"s1": ColdLabel.NO}), [_side("s1", "no_opportunity")])[
        ("hi", F)
    ]
    assert m.counts.tn == 0
    assert m.counts.noopp_opp == 1
    assert m.n_opportunity == 0  # excluded from the 2x2
    # A judge PASS on the same human "no" still IS a true negative.
    m2 = analyze_cold(_sheet({"s2": ColdLabel.NO}), [_side("s2", "pass")])[("hi", F)]
    assert m2.counts.tn == 1 and m2.counts.noopp_opp == 0


def test_analyze_cold_missing_verdict_is_unscored():
    # A blank sidecar outcome (no stored verdict) never enters the 2x2.
    m = analyze_cold(_sheet({"s1": ColdLabel.YES}), [_side("s1", None)])[("hi", F)]
    assert m.counts.judge_unscored == 1 and m.n_opportunity == 0


def test_analyze_cold_unlabeled_precedes_judge_unscored():
    # A cell the annotator never touched is 'unlabeled' EVEN IF the judge also
    # has no verdict — the annotator gap, not the judge gap, explains the row.
    m = analyze_cold(_sheet({"s1": None}), [_side("s1", "error")])[("hi", F)]
    assert m.counts.unlabeled == 1
    assert m.counts.judge_unscored == 0


def test_cold_metrics_f1_zero_not_none():
    # Zero precision (a false alarm, no true positives) must give F1 = 0.0,
    # not None — 0.0 is falsy and a `precision and recall` guard would drop it.
    m = cold_metrics(ColdCounts(tp=0, fn=1, fp_opp=1, tn=0, fa_noopp=0))
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_inter_annotator_kappa_wide():
    a = _sheet({"s1": ColdLabel.YES, "s2": ColdLabel.NO, "s3": ColdLabel.YES})
    b = _sheet({"s1": ColdLabel.YES, "s2": ColdLabel.NO, "s3": ColdLabel.NO})
    k = inter_annotator_kappa(a, b)[("hi", F)]
    assert k is not None and -1.0 <= k <= 1.0
    # Cells where either annotator saw no opportunity are excluded entirely.
    c = _sheet({"s1": ColdLabel.NO_OPPORTUNITY})
    assert inter_annotator_kappa(c, c) == {}


def test_analyze_precision_only():
    def row(sim, verdict):
        return JudgePrecisionRow(
            sim_id=sim, factor_id=F, language="hi", verdict=verdict
        )

    m = analyze_precision(
        [
            row("s1", PrecisionVerdict.REAL),
            row("s2", PrecisionVerdict.FALSE_ALARM),
            row("s3", PrecisionVerdict.REAL),
            row("s4", PrecisionVerdict.UNSURE),
            row("s5", None),
        ]
    )[("hi", F)]
    assert m.real == 2 and m.false_alarm == 1 and m.unsure == 1 and m.unlabeled == 1
    assert m.precision == pytest.approx(2 / 3)  # unsure excluded from denom
    assert m.n_adjudicated == 3


def test_summarize_agreement_hand_computed():
    # Known confusion matrix: a=4, b=1, c=2, d=3 -> agreement 0.7, kappa 0.4.
    records = (
        [AgreementRecord(field="f", pipeline_ok=True, native_ok=True)] * 4
        + [AgreementRecord(field="f", pipeline_ok=True, native_ok=False)] * 1
        + [AgreementRecord(field="f", pipeline_ok=False, native_ok=True)] * 2
        + [AgreementRecord(field="f", pipeline_ok=False, native_ok=False)] * 3
    )
    summary = summarize_agreement(records)
    assert summary.n_records == 10 and summary.n_compared == 10
    assert summary.percent_agreement == pytest.approx(0.7)
    assert summary.cohens_kappa == pytest.approx(0.4)
    assert summary.confusion.both_ok == 4
    assert summary.confusion.pipeline_only == 1
    assert summary.confusion.native_only == 2
    assert summary.confusion.neither == 3
    assert "f" in summary.per_field


def test_summarize_agreement_verdictless_rows_excluded():
    records = [AgreementRecord(field="f", pipeline_ok=None, native_ok=True)] * 5
    summary = summarize_agreement(records)
    assert summary.n_records == 5
    assert summary.n_compared == 0
    assert summary.percent_agreement is None
    assert summary.cohens_kappa is None
