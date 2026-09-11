# Copyright Sierra
"""Judge-calibration agreement: rater CSVs + stored judge verdicts → κ / P / R.

Consumes the calibration wave's returns (rubric-answer and nativeness-label
CSVs from the blind packet) plus the builder's coverage sidecar, reloads the
STORED judge verdicts for the cohort (never re-judging), and emits per-judge /
per-factor agreement against the pooled human labels — precision, recall, F1,
Cohen's κ — next to the rater–rater ceiling (κ and percent agreement over
shared-opportunity cells).

All confusion semantics are the existing cold-calibration machinery
(``tau2.annotation.metrics``): human labels normalize to ``ColdLabel`` and
judge verdicts stay ``JudgeOutcome``, so a FAIL on a human no-opportunity cell
charges precision without entering the 2x2, exactly as in the nativeness cold
sheets. The judge↔human question mapping is fixed here:

- nativeness factors — the nativeness-label CSV, aggregated to a call verdict
  per factor (any violation → yes).
- quality factors — the interaction section's binary answers.
- semantic dimensions — the semantic section's own questions, plus the two
  interaction answers that double as progression labels
  (``SEMANTIC_DIMENSION_SOURCES``).
- delivery — the audio section: the generic fidelity dimensions aggregate to
  the ``fidelity`` axis, ``intonation`` maps to its axis, and pack delivery
  factors match by id.

**Owner adjudications** (the calibration wave has few examples, so blind
annotation alone is not enough): the adjudicate page's decisions CSV
(``CalibrationDecisionRow``, one Confirmed/Rejected row per judge-positive
candidate) is ingested alongside the rater CSVs via ``collect_owner_decisions``
and reported as ADJUDICATED PRECISION per (judge, factor) — the share of
decided candidates the owner confirmed. Decisions are candidate-level (an
utterance-factor positive contributes one candidate per flagged utterance),
so adjudicated precision charges every individual claim the judge made, not
just one per call. Owner decisions never enter the rater-label 2x2 or the
rater–rater ceiling; they are a parallel precision estimate.

**Judge-visible packets are the TWO-PHASE pass**: an annotate build made with
``--show-judge`` brands its batch name with ``JUDGE_VISIBLE_BATCH_SUFFIX``
(and marks its coverage sidecar + exported nativeness provenance) and runs
each call blind-then-reveal: the rater answers everything with judge findings
hidden, locks that layer, and may then revise after the reveal. Its exports
carry BOTH layers with a ``phase`` column, routed per row: ``pre_reveal``
rows join the BLIND pool (they measure recall like any blind label) and
``post_reveal`` rows — plus legacy phase-less judge-visible returns — are the
PRECISION pool, reported as VERIFIED PRECISION per (judge, factor) —
confirmed / verified over judge-positive cells — and never enter the blind
2x2, κ, or the rater–rater ceiling. Because the precision pool carries
post-reveal labels for EVERY cell (not just judge positives), the same pool
is also scored through the full cold 2x2 as the PHASE-2 metrics — the
post-correction precision AND recall per (judge, factor) — reported
alongside, never replacing, the blind 2x2. The agreement build itself still runs on
the BLIND build's coverage sidecar (a judge-visible sidecar is refused; the
two-phase build shares its draw, so the cohorts are identical): the blind
sidecar is the canonical cohort evidence both pools are scored against.

**Severity join**: semantic-source YES answers carry a human minor/major
(``SemanticSeverity``); the report compares them against the judge's stored
progression rating (1 = major, 2 = minor) on judge-FAIL cells and reports
per-dimension severity agreement.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from tau2.annotation.artifacts import git_sha
from tau2.annotation.buckets import (
    CALIBRATION_BUCKETS,
    CALIBRATION_BUCKETS_VERSION,
    FactorDisposition,
    bucket_for_factor,
    bucket_label,
    bucket_outcome,
    factor_disposition,
)
from tau2.annotation.calibration_draw import (
    CALIBRATION_CELLS_VERSION,
    FOLDED_FIDELITY_CELLS_VERSION,
    JUDGE_VISIBLE_BATCH_SUFFIX,
    CalibrationCoverageSidecar,
    CalibrationJudge,
    JudgeVerdictCell,
    ProgressionSeverity,
    SemanticCallRecord,
    calibration_instrument_cells,
    fold_fidelity_cells,
    judge_verdict_cells,
    load_semantic_artifact,
)
from tau2.annotation.evaluation import Gate, cohens_kappa, gate
from tau2.annotation.loading import LoadedSim, iter_loaded_sims
from tau2.annotation.metrics import (
    ColdCounts,
    ColdFactorMetrics,
    accumulate_cold_cell,
    accumulate_pair_cell,
    cold_metrics,
)
from tau2.annotation.models import (
    OVERALL_SCORE_FACTOR_IDS,
    CalibrationDecisionRow,
    ColdLabel,
    NativenessAdjudicationDecision,
    NativenessAnnotationLabel,
    NativenessHumanLabelRow,
    RubricAnnotationRow,
    RubricAnswer,
    RubricPhase,
    SemanticSeverity,
)
from tau2.annotation.packets.forms import (
    CALIBRATION_DECISIONS_KIND,
    NATIVENESS_LABEL_KIND,
    RUBRIC_PACKET_KIND,
    ingest_browser_csv,
)
from tau2.annotation.packets.rubric import (
    SEMANTIC_DIMENSION_SOURCES,
    fidelity_dimension_ids,
)
from tau2.config import DEFAULT_ANNOTATION_PRECISION_BAR
from tau2.data_model.simulation import JudgeOutcome

# v4: phase-2 (post-reveal) 2x2 as first-class metrics.
# v5: construct-level bucket rollup rows (see tau2.annotation.buckets) —
# factor rows carry their bucket/disposition, bucket_rows carry the merged
# clip-level 2x2 that cancels sibling-factor misattribution.
CALIBRATION_AGREEMENT_VERSION = "judge-calibration-agreement-v5"

#: One human label instance: (judge, factor, clip, rater) → ColdLabel.
HumanLabelKey = tuple[CalibrationJudge, str, str, str]

#: Display labels for the adjudication page.
JUDGE_STATE_LABELS: dict[JudgeOutcome, str] = {
    JudgeOutcome.PASS: "pass",
    JudgeOutcome.FAIL: "FAIL",
    JudgeOutcome.NO_OPPORTUNITY: "n/a",
    JudgeOutcome.DEFERRED: "not judged",
    JudgeOutcome.ERROR: "judge error",
}
HUMAN_STATE_LABELS: dict[Optional[ColdLabel], str] = {
    ColdLabel.YES: "issue",
    ColdLabel.NO: "clean",
    ColdLabel.NO_OPPORTUNITY: "n/a",
    ColdLabel.UNSURE: "unsure",
    None: "unlabeled",
}

#: Tri-state used for disagreement highlighting: defect / clean / n-a.
_JUDGE_TRI: dict[JudgeOutcome, Optional[str]] = {
    JudgeOutcome.FAIL: "defect",
    JudgeOutcome.PASS: "clean",
    JudgeOutcome.NO_OPPORTUNITY: "na",
    JudgeOutcome.DEFERRED: None,
    JudgeOutcome.ERROR: None,
}
_HUMAN_TRI: dict[Optional[ColdLabel], Optional[str]] = {
    ColdLabel.YES: "defect",
    ColdLabel.NO: "clean",
    ColdLabel.NO_OPPORTUNITY: "na",
    ColdLabel.UNSURE: None,
    None: None,
}


def tri_state_disagreement(
    judge_outcome: JudgeOutcome, human_labels: Iterable[Optional[ColdLabel]]
) -> bool:
    """Whether the judge and the raters substantively split on one cell.

    Unscored states (deferred/error judges, unsure/unlabeled raters) never
    create a disagreement on their own; a split needs two distinct scored
    states among the participants.
    """
    states = {_JUDGE_TRI[judge_outcome]}
    states.update(_HUMAN_TRI[label] for label in human_labels)
    states.discard(None)
    return len(states) > 1


# ---------------------------------------------------------------------------
# Human labels from returned CSVs
# ---------------------------------------------------------------------------

_BINARY_TO_COLD: dict[RubricAnswer, ColdLabel] = {
    RubricAnswer.YES: ColdLabel.YES,
    RubricAnswer.NO: ColdLabel.NO,
    RubricAnswer.NO_OPPORTUNITY: ColdLabel.NO_OPPORTUNITY,
}
_AUDIO_TO_COLD: dict[RubricAnswer, ColdLabel] = {
    RubricAnswer.AUDIO_NO_ISSUE: ColdLabel.NO,
    RubricAnswer.AUDIO_MINOR: ColdLabel.YES,
    RubricAnswer.AUDIO_CLEAR: ColdLabel.YES,
    RubricAnswer.AUDIO_SEVERE: ColdLabel.YES,
    RubricAnswer.AUDIO_NO_OPPORTUNITY: ColdLabel.NO_OPPORTUNITY,
}

#: Interaction answers that double as progression-dimension labels (also the
#: judge-visible annotate build's question→semantic-cell join).
INTERACTION_SEMANTIC_TWINS: dict[str, str] = {
    source_factor: dimension
    for dimension, (section, source_factor) in SEMANTIC_DIMENSION_SOURCES.items()
    if section == "interaction"
}


def _aggregate(labels: list[ColdLabel]) -> Optional[ColdLabel]:
    """Any yes → yes; else any no → no; else any n/a → n/a."""
    for wanted in (ColdLabel.YES, ColdLabel.NO, ColdLabel.NO_OPPORTUNITY):
        if wanted in labels:
            return wanted
    return None


#: Human minor/major → the semantic judges' severity vocabulary (major =
#: rating 1, minor = rating 2), so the two sides join on one enum.
_HUMAN_TO_JUDGE_SEVERITY: dict[SemanticSeverity, ProgressionSeverity] = {
    SemanticSeverity.MINOR: ProgressionSeverity.MINOR,
    SemanticSeverity.MAJOR: ProgressionSeverity.MAJOR,
}


def _rubric_row_labels(
    row: RubricAnnotationRow,
) -> list[tuple[CalibrationJudge, str, ColdLabel, Optional[ProgressionSeverity]]]:
    """The (judge, factor, label, severity) instances one completed rubric
    row carries. Severity rides ONLY the SEMANTIC-family entries: the
    semantic judges are the ones that grade major (rating 1) vs minor
    (rating 2), so a YES's minor/major joins their scale; the quality twins
    of the same answers are plain binary verdicts."""
    if not row.completed or row.is_custom or row.answer is None:
        return []
    if row.factor_id in OVERALL_SCORE_FACTOR_IDS:
        return []
    severity = (
        _HUMAN_TO_JUDGE_SEVERITY[row.severity] if row.severity is not None else None
    )
    out: list[
        tuple[CalibrationJudge, str, ColdLabel, Optional[ProgressionSeverity]]
    ] = []
    if row.section is not None and row.section.value == "interaction":
        label = _BINARY_TO_COLD.get(row.answer)
        if label is None:
            return []
        out.append((CalibrationJudge.QUALITY, row.factor_id, label, None))
        semantic_dim = INTERACTION_SEMANTIC_TWINS.get(row.factor_id)
        if semantic_dim is not None:
            out.append((CalibrationJudge.SEMANTIC, semantic_dim, label, severity))
    elif row.section is not None and row.section.value == "semantic":
        label = _BINARY_TO_COLD.get(row.answer)
        if label is not None:
            out.append((CalibrationJudge.SEMANTIC, row.factor_id, label, severity))
    elif row.section is not None and row.section.value == "audio":
        label = _AUDIO_TO_COLD.get(row.answer)
        if label is not None:
            out.append((CalibrationJudge.DELIVERY, row.factor_id, label, None))
    return out


def _is_judge_visible_row(batch: str, provenance_json: str) -> bool:
    """Whether a returned row came from a judge-visible (--show-judge) build.

    Judge-visible packets are the two-phase (blind-then-reveal) pass; which
    POOL a row lands in is decided per row by ``_row_pool`` from its phase.
    """
    if batch.endswith(JUDGE_VISIBLE_BATCH_SUFFIX):
        return True
    if provenance_json:
        try:
            provenance = json.loads(provenance_json)
        except json.JSONDecodeError:
            provenance = {}
        if isinstance(provenance, dict) and bool(provenance.get("judge_visible")):
            return True
    return False


def _row_pool(
    path_name: str, judge_visible: bool, phase: Optional[RubricPhase]
) -> bool:
    """Which pool one returned row belongs to: False = blind (recall),
    True = precision.

    Blind-packet rows carry no phase and land in the blind pool. Two-phase
    judge-visible packets export both layers: the locked pre-reveal answers
    were given with the findings hidden (blind pool); post-reveal revisions
    saw the judge's positives (precision pool). A phase-less judge-visible
    row is a legacy single-layer precision-pass return and stays precision.
    A phase on an unbranded row is contradictory and loudly refused.
    """
    if not judge_visible:
        if phase is not None:
            raise ValueError(
                f"{path_name}: row carries phase '{phase.value}' but no "
                "judge-visible brand — a blind packet never exports phased "
                "rows"
            )
        return False
    return phase is not RubricPhase.PRE_REVEAL


class CollectedHumanLabels(BaseModel):
    """Returned rater rows, split into the blind pool and the precision pool."""

    labels: Annotated[
        dict[HumanLabelKey, ColdLabel],
        Field(description="Blind-wave labels: (judge, factor, clip, rater)."),
    ]
    raters: Annotated[list[str], Field(description="Blind-wave rater roster.")]
    verifications: Annotated[
        dict[HumanLabelKey, ColdLabel],
        Field(
            description="Precision-pool labels (post-reveal layers of "
            "two-phase packets, plus legacy single-layer judge-visible "
            "returns), keyed like ``labels``; scored only against judge "
            "positives."
        ),
    ]
    verifiers: Annotated[list[str], Field(description="Precision-pool rater roster.")]
    severities: Annotated[
        dict[HumanLabelKey, ProgressionSeverity],
        Field(
            default_factory=dict,
            description="Blind-pool human minor/major severities on "
            "semantic-family YES labels, joinable against the judge's "
            "rating (1 = major, 2 = minor).",
        ),
    ]
    verifier_severities: Annotated[
        dict[HumanLabelKey, ProgressionSeverity],
        Field(
            default_factory=dict,
            description="Precision-pool counterpart of ``severities``.",
        ),
    ]


def collect_human_labels(filled: Sequence[Path]) -> CollectedHumanLabels:
    """Normalize returned rater CSVs into per-(judge, factor, clip, rater)
    ColdLabels, split into the blind pool and the precision pool.

    Only the calibration packet's two CSV contracts are accepted; any other
    packet kind is a loud error. Pooling is per ROW (``_row_pool``): blind
    packets fill ``labels``; a two-phase judge-visible packet's pre-reveal
    rows fill ``labels`` too (they were answered blind) while its post-reveal
    rows — and legacy phase-less judge-visible returns — fill
    ``verifications``. Nativeness rows aggregate to one call verdict per
    factor; the generic audio fidelity dimensions aggregate to the delivery
    ``fidelity`` axis. Human minor/major severities on semantic-family YES
    labels ride ``severities`` / ``verifier_severities``.
    """
    fidelity_ids = fidelity_dimension_ids()
    # Index 0 = blind pool, index 1 = precision pool.
    pooled_labels: tuple[dict[HumanLabelKey, ColdLabel], ...] = ({}, {})
    pooled_severities: tuple[dict[HumanLabelKey, ProgressionSeverity], ...] = ({}, {})
    pooled_raters: tuple[set[str], ...] = (set(), set())
    # Raw per-dimension audio answers, folded into axes after ingest.
    audio_by_cell: dict[tuple[bool, str, str], dict[str, ColdLabel]] = {}
    nativeness_by_cell: dict[tuple[bool, str, str, str], list[ColdLabel]] = {}

    for path in filled:
        parsed = ingest_browser_csv(Path(path))
        if parsed.kind == RUBRIC_PACKET_KIND:
            for row in parsed.rows:
                assert isinstance(row, RubricAnnotationRow)
                if not row.completed:
                    continue
                visible = _is_judge_visible_row(row.batch, "")
                pool = _row_pool(Path(path).name, visible, row.phase)
                labels = pooled_labels[pool]
                pooled_raters[pool].add(row.rater)
                for judge, factor_id, label, severity in _rubric_row_labels(row):
                    if judge is CalibrationJudge.DELIVERY and factor_id in (
                        fidelity_ids | {"intonation"}
                    ):
                        audio_by_cell.setdefault((pool, row.clip_id, row.rater), {})[
                            factor_id
                        ] = label
                    else:
                        key = (judge, factor_id, row.clip_id, row.rater)
                        labels[key] = label
                        if severity is not None and label is ColdLabel.YES:
                            pooled_severities[pool][key] = severity
        elif parsed.kind == NATIVENESS_LABEL_KIND:
            for row in parsed.rows:
                assert isinstance(row, NativenessHumanLabelRow)
                if not row.completed or row.annotation_label is None:
                    continue
                visible = _is_judge_visible_row(row.batch, row.provenance_json)
                pool = _row_pool(Path(path).name, visible, row.phase)
                pooled_raters[pool].add(row.rater)
                cold = {
                    NativenessAnnotationLabel.VIOLATION: ColdLabel.YES,
                    NativenessAnnotationLabel.PASS: ColdLabel.NO,
                    NativenessAnnotationLabel.NOT_APPLICABLE: (
                        ColdLabel.NO_OPPORTUNITY
                    ),
                }[row.annotation_label]
                nativeness_by_cell.setdefault(
                    (pool, row.clip_id, row.factor_id, row.rater), []
                ).append(cold)
        else:
            raise ValueError(
                f"{Path(path).name} is a '{parsed.kind}' export — the "
                "calibration wave returns rubric-answer and nativeness-label "
                "CSVs only"
            )

    for (pool, clip_id, rater), by_dimension in audio_by_cell.items():
        labels = pooled_labels[pool]
        fidelity = _aggregate(
            [by_dimension[d] for d in sorted(by_dimension) if d in fidelity_ids]
        )
        if fidelity is not None:
            labels[(CalibrationJudge.DELIVERY, "fidelity", clip_id, rater)] = fidelity
        intonation = by_dimension.get("intonation")
        if intonation is not None:
            labels[(CalibrationJudge.DELIVERY, "intonation", clip_id, rater)] = (
                intonation
            )
    for (pool, clip_id, factor_id, rater), cell_labels in nativeness_by_cell.items():
        aggregated = _aggregate(cell_labels)
        if aggregated is not None:
            pooled_labels[pool][
                (CalibrationJudge.NATIVENESS, factor_id, clip_id, rater)
            ] = aggregated
    return CollectedHumanLabels(
        labels=pooled_labels[False],
        raters=sorted(pooled_raters[False]),
        verifications=pooled_labels[True],
        verifiers=sorted(pooled_raters[True]),
        severities=pooled_severities[False],
        verifier_severities=pooled_severities[True],
    )


# ---------------------------------------------------------------------------
# Owner adjudications (the adjudicate page's decisions CSV)
# ---------------------------------------------------------------------------


class DecisionTally(BaseModel):
    """Confirmed/rejected counts of one (judge, factor)'s decided candidates."""

    n_confirmed: Annotated[int, Field(ge=0)] = 0
    n_rejected: Annotated[int, Field(ge=0)] = 0

    @property
    def n_decided(self) -> int:
        return self.n_confirmed + self.n_rejected

    @property
    def precision(self) -> Optional[float]:
        return self.n_confirmed / self.n_decided if self.n_decided else None


def collect_owner_decisions(
    decisions: Sequence[Path],
) -> tuple[dict[tuple[CalibrationJudge, str], DecisionTally], list[str]]:
    """Tally the adjudicate page's decisions CSVs per (judge, factor).

    Only the ``packet_judge_calibration_decisions`` contract is accepted.
    Every completed row is one decided candidate (utterance-level positives
    contribute one row per flagged utterance); duplicate candidate decisions
    across files are deduplicated on (adjudicator, candidate_id), keeping the
    last. Returns the tallies and the sorted adjudicator roster.
    """
    by_candidate: dict[
        tuple[str, str], tuple[CalibrationJudge, str, NativenessAdjudicationDecision]
    ] = {}
    adjudicators: set[str] = set()
    for path in decisions:
        parsed = ingest_browser_csv(Path(path))
        if parsed.kind != CALIBRATION_DECISIONS_KIND:
            raise ValueError(
                f"{Path(path).name} is a '{parsed.kind}' export — owner "
                "adjudications must be the calibration adjudicate page's "
                "decisions CSV"
            )
        for row in parsed.rows:
            assert isinstance(row, CalibrationDecisionRow)
            if not row.completed or row.adjudication_decision is None:
                continue
            adjudicators.add(row.rater)
            by_candidate[(row.rater, row.candidate_id)] = (
                CalibrationJudge(row.judge),
                row.factor_id,
                row.adjudication_decision,
            )
    tallies: dict[tuple[CalibrationJudge, str], DecisionTally] = {}
    for judge, factor_id, decision in by_candidate.values():
        tally = tallies.setdefault((judge, factor_id), DecisionTally())
        if decision is NativenessAdjudicationDecision.CONFIRMED:
            tally.n_confirmed += 1
        else:
            tally.n_rejected += 1
    return tallies, sorted(adjudicators)


# ---------------------------------------------------------------------------
# Cohort evidence (stored judge verdicts, reloaded — never re-judged)
# ---------------------------------------------------------------------------


class CohortEvidence(BaseModel):
    """Reloaded cohort calls and their normalized stored judge verdicts."""

    sims: Annotated[
        dict[str, LoadedSim], Field(description="clip_id -> reloaded call.")
    ]
    judge_cells: Annotated[
        dict[str, dict[tuple[CalibrationJudge, str], JudgeVerdictCell]],
        Field(description="clip_id -> (judge, factor) -> stored verdict."),
    ]


def load_cohort_evidence(sidecar: CalibrationCoverageSidecar) -> CohortEvidence:
    """Reload every cohort call and normalize its stored judge verdicts.

    The cell view is the one the sidecar's cohort was built with: a
    ``cells_version``-stamped sidecar gets the ``calibration_instrument_cells``
    view — additionally folded through ``fold_fidelity_cells`` when the stamp
    is cells-v2, the one version whose packets asked a single folded fidelity
    question — while a pre-cut sidecar keeps the raw ``judge_verdict_cells``
    view. Either way the sidecar's conversation artifact is loaded when
    present — the instrument view still needs it for the LLM-quality verdicts
    on corpora that store them there."""
    stamped = sidecar.cells_version in (
        CALIBRATION_CELLS_VERSION,
        FOLDED_FIDELITY_CELLS_VERSION,
    )
    fold_v2 = sidecar.cells_version == FOLDED_FIDELITY_CELLS_VERSION
    semantic_by_key: dict[tuple[str, str], SemanticCallRecord] = {}
    if sidecar.conversation_artifact:
        semantic_by_key = load_semantic_artifact(
            Path(sidecar.conversation_artifact)
        ).by_key()
    wanted = {(row.results_path, row.sim_id): row.clip_id for row in sidecar.rows}
    paths = sorted({Path(row.results_path) for row in sidecar.rows})
    sims: dict[str, LoadedSim] = {}
    for item in iter_loaded_sims(paths):
        key = (str((item.results_dir / "results.json").resolve()), item.sim.id)
        clip_id = wanted.get(key)
        if clip_id is not None and clip_id not in sims:
            sims[clip_id] = item
    missing = sorted(set(wanted.values()) - set(sims))
    if missing:
        raise ValueError(
            f"{len(missing)} cohort call(s) missing from their recorded "
            f"results paths: {', '.join(missing[:5])}"
        )

    def _cells(item: LoadedSim):
        cells = (calibration_instrument_cells if stamped else judge_verdict_cells)(
            item.sim,
            sidecar.language,
            semantic_by_key.get(
                (str((item.results_dir / "results.json").resolve()), item.sim.id)
            ),
        )
        return fold_fidelity_cells(cells) if fold_v2 else cells

    judge_cells = {
        clip_id: {(cell.judge, cell.factor_id): cell for cell in _cells(item)}
        for clip_id, item in sims.items()
    }
    return CohortEvidence(sims=sims, judge_cells=judge_cells)


# ---------------------------------------------------------------------------
# The agreement report
# ---------------------------------------------------------------------------


class CalibrationAgreementRow(BaseModel):
    """Agreement of one judge factor against the wave's human labels."""

    judge: Annotated[CalibrationJudge, Field(description="Judge family.")]
    factor_id: Annotated[str, Field(description="Factor / axis / dimension id.")]
    metrics: Annotated[
        ColdFactorMetrics,
        Field(description="Judge-vs-human precision/recall/F1/κ (cold semantics)."),
    ]
    human_kappa: Annotated[
        Optional[float],
        Field(
            description="Rater–rater Cohen's κ over shared-opportunity cells "
            "(the human ceiling); None without doubly-labeled cells."
        ),
    ] = None
    human_agreement: Annotated[
        Optional[float],
        Field(description="Rater–rater raw percent agreement on the same cells."),
    ] = None
    n_confirmed: Annotated[
        int,
        Field(ge=0, description="Judge-positive candidates the owner confirmed."),
    ] = 0
    n_rejected: Annotated[
        int,
        Field(ge=0, description="Judge-positive candidates the owner rejected."),
    ] = 0
    adjudicated_precision: Annotated[
        Optional[float],
        Field(
            description="Owner-adjudicated precision: confirmed / decided "
            "candidates (candidate-level; utterance positives count each "
            "flagged utterance); None without decisions on this factor."
        ),
    ] = None
    n_verifier_confirmed: Annotated[
        int,
        Field(
            ge=0,
            description="Judge positives confirmed by precision-pass "
            "(--show-judge) raters; each rater's verdict counts once.",
        ),
    ] = 0
    n_verifier_rejected: Annotated[
        int,
        Field(ge=0, description="Judge positives rejected by precision-pass raters."),
    ] = 0
    verified_precision: Annotated[
        Optional[float],
        Field(
            description="Rater-verified precision from the judge-visible "
            "precision pass: confirmed / verified; None without "
            "precision-pass returns on this factor."
        ),
    ] = None
    post_reveal_metrics: Annotated[
        Optional[ColdFactorMetrics],
        Field(
            description="Phase-2 (post-reveal) judge-vs-human 2x2: the exact "
            "cold semantics of ``metrics`` accumulated over the "
            "precision-pool labels instead of the blind pool, so it carries "
            "post-correction precision AND recall — a reveal-ward revision "
            "moves fp→tp here while the blind 2x2 keeps the blind answer. "
            "None without precision-pool labels on this factor."
        ),
    ] = None
    n_severity_compared: Annotated[
        int,
        Field(
            ge=0,
            description="Semantic dimensions only: judge-FAIL cells with a "
            "stored major/minor rating where a blind rater said YES with a "
            "minor/major severity.",
        ),
    ] = 0
    n_severity_matched: Annotated[
        int,
        Field(
            ge=0,
            description="Compared cells where the human severity equals the "
            "judge's (major = rating 1, minor = rating 2).",
        ),
    ] = 0
    severity_agreement: Annotated[
        Optional[float],
        Field(
            description="matched / compared; None without comparable severity cells."
        ),
    ] = None
    gate: Annotated[
        Gate, Field(description="LIVE/shadow per the annotation precision bar.")
    ]
    bucket: Annotated[
        Optional[str],
        Field(
            description="Calibration bucket this factor rolls into; None for "
            "retired/audition/shadow dispositions."
        ),
    ] = None
    disposition: Annotated[
        FactorDisposition,
        Field(description="Why the factor is or is not bucket-calibrated."),
    ] = FactorDisposition.SCORED


class CalibrationBucketRow(BaseModel):
    """Construct-level agreement: constituent factors merged per clip.

    A clip counts once per rater: the human label is the any-yes merge of the
    constituent labels, the judge outcome the any-FAIL merge of the
    constituent outcomes (``tau2.annotation.buckets``). This cancels
    sibling-factor misattribution — the judge and rater agreeing a defect
    exists but filing it under different factor ids.
    """

    bucket_id: Annotated[str, Field(description="Bucket identifier.")]
    judge: Annotated[CalibrationJudge, Field(description="Owning judge family.")]
    factor_ids: Annotated[tuple[str, ...], Field(description="Constituent factor ids.")]
    metrics: Annotated[
        ColdFactorMetrics,
        Field(description="Judge-vs-human 2x2 over merged bucket cells."),
    ]
    human_kappa: Annotated[
        Optional[float],
        Field(description="Rater-rater κ over merged bucket cells; the ceiling."),
    ] = None
    human_agreement: Annotated[
        Optional[float],
        Field(description="Rater-rater raw agreement on the same merged cells."),
    ] = None
    gate: Annotated[
        Gate, Field(description="LIVE/shadow per the annotation precision bar.")
    ]


class CalibrationAgreementReport(BaseModel):
    """The persisted per-judge/per-factor calibration agreement artifact."""

    schema_version: Literal[1] = 1
    report_version: str = CALIBRATION_AGREEMENT_VERSION
    created_at: Annotated[str, Field(description="Build wall-clock time (UTC).")]
    git_sha: Annotated[str, Field(description="Repo HEAD at build time.")]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    sidecar_path: Annotated[str, Field(description="Coverage sidecar consumed.")]
    filled: Annotated[list[str], Field(description="Rater CSVs consumed.")]
    raters: Annotated[list[str], Field(description="Blind-wave rater roster found.")]
    verifiers: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Precision-pass (--show-judge) rater roster found.",
        ),
    ]
    decisions: Annotated[
        list[str], Field(default_factory=list, description="Decisions CSVs consumed.")
    ]
    adjudicators: Annotated[
        list[str],
        Field(default_factory=list, description="Owner adjudicator roster found."),
    ]
    n_calls: Annotated[int, Field(ge=0, description="Cohort calls compared.")]
    precision_bar: Annotated[float, Field(description="LIVE/shadow gate bar.")]
    judge_versions: Annotated[
        dict[str, list[str]],
        Field(description="Judge version stamps observed on the cohort."),
    ]
    rows: Annotated[
        list[CalibrationAgreementRow],
        Field(description="One row per (judge, factor)."),
    ]
    buckets_version: Annotated[
        str,
        Field(description="Bucket catalog version the rollup was built with."),
    ] = CALIBRATION_BUCKETS_VERSION
    bucket_rows: Annotated[
        list[CalibrationBucketRow],
        Field(
            default_factory=list,
            description="Construct-level rollup, one row per bucket with "
            "stored judge cells on this cohort.",
        ),
    ]


def build_calibration_agreement(
    sidecar_path: Path,
    filled: Sequence[Path],
    *,
    decisions: Sequence[Path] = (),
    precision_bar: float = DEFAULT_ANNOTATION_PRECISION_BAR,
) -> CalibrationAgreementReport:
    """Compute per-judge/per-factor agreement from returns + stored verdicts.

    ``filled`` takes rater CSVs from BOTH instruments — blind-packet rows and
    two-phase pre-reveal rows score the 2x2/κ tables (the recall pool), while
    post-reveal rows and legacy phase-less judge-visible returns score
    VERIFIED PRECISION over judge positives — split automatically per row by
    the batch brand + phase column. The precision pool labels every cell, so
    it is additionally scored through the same cold 2x2 as the PHASE-2
    metrics (post-correction precision/recall/F1) per (judge, factor),
    reported next to — never in place of — the blind 2x2. ``decisions`` are the owner's
    adjudicate-page decisions CSVs, reported as adjudicated precision per
    factor. Either may be empty, but not both. The sidecar must come from the
    BLIND build (a judge-visible sidecar is a loud error): the two-phase
    build shares its draw (same results + seed → same cohort and clip ids),
    so the blind sidecar is the canonical cohort evidence for both pools.
    Human minor/major severities on semantic YES labels are joined against
    the judge's stored rating (1 = major, 2 = minor) per dimension.
    """
    from tau2.annotation.calibration_draw import load_calibration_sidecar

    if not filled and not decisions:
        raise ValueError(
            "calibration-agreement needs rater CSVs (--filled) and/or owner "
            "decisions CSVs (--decisions)"
        )
    sidecar = load_calibration_sidecar(sidecar_path)
    if sidecar.judge_visible:
        raise ValueError(
            f"{Path(sidecar_path).name}: this coverage sidecar belongs to a "
            "judge-visible (--show-judge) build — pass the BLIND build's "
            "sidecar instead (the two-phase build shares its draw, so the "
            "cohort and clip ids are identical)"
        )
    evidence = load_cohort_evidence(sidecar)
    collected = collect_human_labels(filled)
    labels, raters = collected.labels, collected.raters
    tallies, adjudicators = collect_owner_decisions(decisions)

    # Precision pass: judge-visible returns verify the judge's POSITIVES.
    # A YES confirms the shown claim; NO / no-opportunity rejects it. They
    # never enter the blind counts, κ, or the rater-rater ceiling above.
    verified: dict[tuple[CalibrationJudge, str], DecisionTally] = {}
    for clip_id, cells in evidence.judge_cells.items():
        for (judge, factor_id), cell in cells.items():
            if cell.outcome is not JudgeOutcome.FAIL:
                continue
            for verifier in collected.verifiers:
                label = collected.verifications.get(
                    (judge, factor_id, clip_id, verifier)
                )
                if label is None:
                    continue
                tally = verified.setdefault((judge, factor_id), DecisionTally())
                if label is ColdLabel.YES:
                    tally.n_confirmed += 1
                else:
                    tally.n_rejected += 1

    counts: dict[tuple[CalibrationJudge, str], ColdCounts] = {}
    # Phase-2 2x2: the SAME cold accumulation, run over the precision-pool
    # labels; a factor only reports it when at least one post-reveal label
    # actually touched one of its cells.
    post_counts: dict[tuple[CalibrationJudge, str], ColdCounts] = {}
    post_touched: set[tuple[CalibrationJudge, str]] = set()
    pair_cells: dict[tuple[CalibrationJudge, str], list[int]] = {}
    # Semantic severity join: judge-FAIL cells with a stored major/minor
    # rating vs blind YES labels carrying a human minor/major.
    severity_compared: Counter[tuple[CalibrationJudge, str]] = Counter()
    severity_matched: Counter[tuple[CalibrationJudge, str]] = Counter()
    for clip_id, cells in evidence.judge_cells.items():
        for (judge, factor_id), cell in cells.items():
            key = (judge, factor_id)
            counts.setdefault(key, ColdCounts())
            post_counts.setdefault(key, ColdCounts())
            for verifier in collected.verifiers:
                label = collected.verifications.get(
                    (judge, factor_id, clip_id, verifier)
                )
                accumulate_cold_cell(post_counts[key], label, cell.outcome)
                if label is not None:
                    post_touched.add(key)
            for rater in raters:
                accumulate_cold_cell(
                    counts[key],
                    labels.get((judge, factor_id, clip_id, rater)),
                    cell.outcome,
                )
                if cell.outcome is JudgeOutcome.FAIL and cell.severity is not None:
                    human_severity = collected.severities.get(
                        (judge, factor_id, clip_id, rater)
                    )
                    if human_severity is not None:
                        severity_compared[key] += 1
                        severity_matched[key] += human_severity is cell.severity
            rated = [labels.get((judge, factor_id, clip_id, rater)) for rater in raters]
            present = [label for label in rated if label is not None]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    accumulate_pair_cell(pair_cells, key, present[i], present[j])

    # Bucket rollup: per (bucket, clip, rater) merge the constituent labels
    # (any-yes) and outcomes (any-FAIL), then fold through the SAME
    # accumulate_cold_cell so the bucket 2x2 can never drift from factor
    # semantics. Cancels sibling-factor misattribution.
    bucket_counts: dict[str, ColdCounts] = {}
    bucket_pair_cells: dict[tuple[CalibrationJudge, str], list[int]] = {}
    bucket_judges = {bucket.bucket_id: bucket.judge for bucket in CALIBRATION_BUCKETS}
    for clip_id, cells in evidence.judge_cells.items():
        members: dict[str, list[tuple[CalibrationJudge, str, JudgeOutcome]]] = {}
        for (judge, factor_id), cell in cells.items():
            bucket_id = bucket_for_factor(judge, factor_id)
            if bucket_id is not None:
                members.setdefault(bucket_id, []).append(
                    (judge, factor_id, cell.outcome)
                )
        for bucket_id, constituent in members.items():
            bucket_counts.setdefault(bucket_id, ColdCounts())
            merged_outcome = bucket_outcome(outcome for _, _, outcome in constituent)
            merged_by_rater: list[Optional[ColdLabel]] = []
            for rater in raters:
                merged = bucket_label(
                    labels.get((judge, factor_id, clip_id, rater))
                    for judge, factor_id, _ in constituent
                )
                merged_by_rater.append(merged)
                accumulate_cold_cell(bucket_counts[bucket_id], merged, merged_outcome)
            present_merged = [label for label in merged_by_rater if label is not None]
            pair_key = (bucket_judges[bucket_id], bucket_id)
            for i in range(len(present_merged)):
                for j in range(i + 1, len(present_merged)):
                    accumulate_pair_cell(
                        bucket_pair_cells,
                        pair_key,
                        present_merged[i],
                        present_merged[j],
                    )

    unknown_decisions = sorted(set(tallies) - set(counts))
    if unknown_decisions:
        raise ValueError(
            "decisions CSV adjudicates factors with no stored judge cells on "
            f"this cohort: {unknown_decisions[:5]} — wrong sidecar?"
        )

    rows: list[CalibrationAgreementRow] = []
    for key in sorted(counts, key=lambda k: (k[0].value, k[1])):
        metrics = cold_metrics(counts[key])
        human_kappa = human_agreement = None
        cells = pair_cells.get(key)
        if cells is not None and sum(cells) > 0:
            tp, fn, fp, tn = cells
            human_kappa = cohens_kappa(tp, fn, fp, tn)
            human_agreement = (tp + tn) / (tp + fn + fp + tn)
        tally = tallies.get(key, DecisionTally())
        verifier_tally = verified.get(key, DecisionTally())
        rows.append(
            CalibrationAgreementRow(
                judge=key[0],
                factor_id=key[1],
                metrics=metrics,
                human_kappa=human_kappa,
                human_agreement=human_agreement,
                n_confirmed=tally.n_confirmed,
                n_rejected=tally.n_rejected,
                adjudicated_precision=tally.precision,
                n_verifier_confirmed=verifier_tally.n_confirmed,
                n_verifier_rejected=verifier_tally.n_rejected,
                verified_precision=verifier_tally.precision,
                post_reveal_metrics=(
                    cold_metrics(post_counts[key]) if key in post_touched else None
                ),
                n_severity_compared=severity_compared[key],
                n_severity_matched=severity_matched[key],
                severity_agreement=(
                    severity_matched[key] / severity_compared[key]
                    if severity_compared[key]
                    else None
                ),
                gate=gate(metrics.precision, precision_bar),
                bucket=bucket_for_factor(key[0], key[1]),
                disposition=factor_disposition(key[0], key[1]),
            )
        )

    bucket_rows: list[CalibrationBucketRow] = []
    for bucket in CALIBRATION_BUCKETS:
        counts_for_bucket = bucket_counts.get(bucket.bucket_id)
        if counts_for_bucket is None:
            continue
        bucket_metrics = cold_metrics(counts_for_bucket)
        human_kappa = human_agreement = None
        cells = bucket_pair_cells.get((bucket.judge, bucket.bucket_id))
        if cells is not None and sum(cells) > 0:
            tp, fn, fp, tn = cells
            human_kappa = cohens_kappa(tp, fn, fp, tn)
            human_agreement = (tp + tn) / (tp + fn + fp + tn)
        bucket_rows.append(
            CalibrationBucketRow(
                bucket_id=bucket.bucket_id,
                judge=bucket.judge,
                factor_ids=bucket.factor_ids,
                metrics=bucket_metrics,
                human_kappa=human_kappa,
                human_agreement=human_agreement,
                gate=gate(bucket_metrics.precision, precision_bar),
            )
        )

    return CalibrationAgreementReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
        language=sidecar.language,
        sidecar_path=str(sidecar_path),
        filled=[str(path) for path in filled],
        raters=raters,
        verifiers=collected.verifiers,
        decisions=[str(path) for path in decisions],
        adjudicators=adjudicators,
        n_calls=len(sidecar.rows),
        precision_bar=precision_bar,
        judge_versions=sidecar.judge_versions,
        rows=rows,
        bucket_rows=bucket_rows,
    )


def _fmt(value: Optional[float]) -> str:
    return "  -  " if value is None else f"{value:.3f}"


def render_calibration_agreement(report: CalibrationAgreementReport) -> str:
    """Fixed-width table: per-judge/per-factor agreement, the human ceiling,
    the owner-adjudicated precision, and the phase-2 (post-reveal) 2x2."""
    lines = [
        f"Judge calibration agreement — {report.language} "
        f"({report.n_calls} calls, raters: {', '.join(report.raters) or 'none'}, "
        f"verifiers: {', '.join(report.verifiers) or 'none'}, "
        f"adjudicators: {', '.join(report.adjudicators) or 'none'})",
        "",
        f"{'judge':<11} {'factor':<32} {'prec':>6} {'recall':>6} {'f1':>6} "
        f"{'kappa':>6} {'hkappa':>6} {'hagree':>6} {'aprec':>6} {'adj':>4} "
        f"{'vprec':>6} {'ver':>4} {'p2prec':>6} {'p2rec':>6} {'p2f1':>6} "
        f"{'p2n':>4} {'sev':>6} {'nsev':>4} {'n':>4}  gate",
    ]
    for row in report.rows:
        metrics = row.metrics
        post = row.post_reveal_metrics
        lines.append(
            f"{row.judge.value:<11} {row.factor_id:<32} "
            f"{_fmt(metrics.precision):>6} {_fmt(metrics.recall):>6} "
            f"{_fmt(metrics.f1):>6} {_fmt(metrics.kappa):>6} "
            f"{_fmt(row.human_kappa):>6} {_fmt(row.human_agreement):>6} "
            f"{_fmt(row.adjudicated_precision):>6} "
            f"{row.n_confirmed + row.n_rejected:>4} "
            f"{_fmt(row.verified_precision):>6} "
            f"{row.n_verifier_confirmed + row.n_verifier_rejected:>4} "
            f"{_fmt(post.precision if post else None):>6} "
            f"{_fmt(post.recall if post else None):>6} "
            f"{_fmt(post.f1 if post else None):>6} "
            f"{(post.n_opportunity if post else 0):>4} "
            f"{_fmt(row.severity_agreement):>6} "
            f"{row.n_severity_compared:>4} "
            f"{metrics.n_opportunity:>4}  {row.gate}"
            + (
                f"  [{row.disposition.value}]"
                if row.disposition is not FactorDisposition.SCORED
                else f"  -> {row.bucket}"
            )
        )
        diag = row.metrics.counts
        extras = {
            "judge_unscored": diag.judge_unscored,
            "noopp_opp": diag.noopp_opp,
            "fa_noopp": diag.fa_noopp,
            "unsure": diag.unsure,
            "unlabeled": diag.unlabeled,
        }
        noted = {name: count for name, count in extras.items() if count}
        if noted:
            lines.append(
                "  " + "  ".join(f"{name}={count}" for name, count in noted.items())
            )
    if report.bucket_rows:
        lines.append("")
        lines.append(
            f"Calibration buckets ({report.buckets_version}) — construct-level "
            "rollup, clip merged per rater (any-yes / any-FAIL):"
        )
        lines.append(
            f"{'judge':<11} {'bucket':<32} {'prec':>6} {'recall':>6} {'f1':>6} "
            f"{'kappa':>6} {'hkappa':>6} {'hagree':>6} {'n':>4}  gate"
        )
        for bucket_row in report.bucket_rows:
            bucket_metrics = bucket_row.metrics
            lines.append(
                f"{bucket_row.judge.value:<11} {bucket_row.bucket_id:<32} "
                f"{_fmt(bucket_metrics.precision):>6} "
                f"{_fmt(bucket_metrics.recall):>6} "
                f"{_fmt(bucket_metrics.f1):>6} {_fmt(bucket_metrics.kappa):>6} "
                f"{_fmt(bucket_row.human_kappa):>6} "
                f"{_fmt(bucket_row.human_agreement):>6} "
                f"{bucket_metrics.n_opportunity:>4}  {bucket_row.gate}"
            )
            lines.append(f"  = {' + '.join(bucket_row.factor_ids)}")
    if report.judge_versions:
        lines.append("")
        lines.append("Judge versions observed on the cohort:")
        for family, versions in sorted(report.judge_versions.items()):
            lines.append(f"  {family}: {', '.join(versions)}")
    return "\n".join(lines)


__all__ = [
    "CALIBRATION_AGREEMENT_VERSION",
    "HUMAN_STATE_LABELS",
    "INTERACTION_SEMANTIC_TWINS",
    "JUDGE_STATE_LABELS",
    "CalibrationAgreementReport",
    "CalibrationAgreementRow",
    "CalibrationBucketRow",
    "CohortEvidence",
    "DecisionTally",
    "build_calibration_agreement",
    "collect_human_labels",
    "collect_owner_decisions",
    "load_cohort_evidence",
    "render_calibration_agreement",
    "tri_state_disagreement",
]
