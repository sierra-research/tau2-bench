# Copyright Sierra
"""Defect-enriched draw for the held-out judge-calibration annotation wave.

The paper's calibration protocol (METHODS_AND_EXPERIMENTS_V2.md §4) needs ONE
held-out, double-annotated, judge-blind wave — a per-language cohort
(``DEFAULT_CALIBRATION_CALLS_PER_LANGUAGE``, owner-sized to 30; ``--n-calls``
overrides), defect-enriched so every live judged factor sees human labels,
plus judge-clean controls so recall/false-alarm rates are estimable. This
module owns the SAMPLING side of `tau2 annotate calibration-packets`:

- ``judge_verdict_cells`` normalizes every stored judge verdict on a call —
  nativeness factors, delivery axes + pack delivery factors, quality factors,
  and the semantic (progression/conciseness) suite from a stored
  conversation-judge artifact — into one typed cell list. The draw, the adjudication packet, and
  the agreement report all read the SAME cells, so a call can serve several
  judges' calibration at once.
- ``build_calibration_frame`` enumerates the whole sampling frame (every
  eligible stored call, its stratum memberships, and why ineligible calls
  were dropped) and ``draw_calibration_cohort`` marks the seeded draw on it.
  The frame is persisted whole, with per-stratum inclusion probabilities, so
  the draw is auditable and reproducible (same inputs + seed → same cohort).
- ``build_adjudication_frame`` / ``draw_adjudication_cohort`` are the
  adjudicate-mode siblings: an INDEPENDENT per-factor draw straight off the
  stored results — every defect stratum takes ``min(per_factor_cap,
  frame_size)`` of its flagged calls (capped, never padded), with no clean
  controls and no shared budget. Per-factor drawn/covered counts persist on
  the frame's stratum accounting.

Strata are ``<judge>:<factor>`` cells (a judge FAIL on that factor) plus one
``clean_control`` cell (no FAIL on any judged factor). Semantic PROGRESSION
cells split by severity — ``semantic:<dimension>:major`` (rating 1) and
``semantic:<dimension>:minor`` (rating 2) — because the dimensions flag
minors on most calls (question_quality flags ~93% of a 240-call corpus,
~87% of them rating-2), and a flag-keyed stratum would let minors swamp the
draw's coverage of the rare majors. Draws are uniform
within a stratum with an independent per-(seed, language, stratum) stream —
the same discipline as the preference sampler — so adding a stratum never
reshuffles another stratum's draw. Because one call can belong to several
defect strata, per-call inclusion probabilities are not uniform; the frame
records per-stratum coverage rates and the full membership matrix, which is
sufficient to recompute any inclusion probability after the fact.

Judges ship as-is: this module only READS stored verdicts (never re-judges).
The semantic suite is read from its stored artifact (``tau2 judges legacy
conversation``'s ``ConversationJudgeArtifact``) through a minimal local reader
model rather than importing ``tau2.judges.conversation`` internals — the
stored artifact JSON is the stable contract.
"""

import random
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau2.annotation.artifacts import git_sha
from tau2.annotation.loading import LoadedSim
from tau2.config import (
    DEFAULT_CALIBRATION_ADJUDICATION_FACTOR_CAP,
    DEFAULT_CALIBRATION_CALLS_PER_LANGUAGE,
    DEFAULT_CALIBRATION_CONTROL_FRACTION,
    DEFAULT_CALIBRATION_SEED,
)
from tau2.data_model.simulation import JudgeOutcome, QualityInfo, SimulationRun
from tau2.judges.delivery.factors import delivery_factors_for
from tau2.judges.nativeness.factors import judge_factors_for
from tau2.judges.quality.factors import QUALITY_FACTORS

#: v5: the annotate draw gains the judge-blind ``rule="random"`` (recall arm);
#: the enriched rule's per-stratum streams are untouched — same seed, same
#: enriched cohort as v4.
#: v6: standalone translationese and verb-morphology strata leave the cold
#: coverage list after being absorbed into calibrated natural_word_choice.
CALIBRATION_SAMPLER_VERSION = "judge-calibration-draw-v6"

#: Version stamp of the instrument-level cell view (``calibration_instrument_cells``).
#: v2 (instrument cut, 2026-08-25): deterministic quality checkers (EVA
#: turn-taking) and the semantic progression/conciseness suite are OUT of the
#: calibration instrument, and the two delivery axes fold into ONE
#: ``fidelity`` cell. Stamped on coverage sidecars so the adjudication page
#: and the agreement report reconstruct the SAME cell view the packet was
#: built with; sidecars without the stamp are pre-cut waves and keep the raw
#: ``judge_verdict_cells`` view (with the sidecar's semantic artifact).
#: v3 (intonation unfold, 2026-08-26, owner ruling "intonation should be its
#: own thing"): the delivery axes stay SEPARATE ``fidelity`` and
#: ``intonation`` cells — each axis gets its own question, stratum, and
#: precision/recall numbers. Only v2 sidecars carry the folded view
#: (``FOLDED_FIDELITY_CELLS_VERSION``).
CALIBRATION_CELLS_VERSION = "calibration-cells-v3"

#: The one cells version whose sidecars carry the FOLDED fidelity view —
#: adjudication over a v2 sidecar must fold, v3+ and pre-cut must not.
FOLDED_FIDELITY_CELLS_VERSION = "calibration-cells-v2"

#: Stratum id for judge-clean control calls (no FAIL on any judged factor).
CLEAN_CONTROL_STRATUM = "clean_control"

#: ``drawn_for`` label of the judge-blind uniform draw (``rule="random"``):
#: selection never conditions on judge verdicts, so the only honest stratum
#: is the whole frame. The per-stratum accounting rows a random-rule frame
#: still carries are DESCRIPTIVE (what the cohort happened to cover), never
#: selective.
RANDOM_DRAW_STRATUM = "random"

#: The nativeness factors that went live without cold validation
#: (audit-rubric shadow round follow-up, 2026-08-13). The draw must cover
#: them whenever the corpus contains a FAIL; an empty stratum is recorded
#: loudly on the frame instead of silently thinning the wave.
COLD_UNVALIDATED_FACTORS: dict[str, tuple[str, ...]] = {
    "es": ("request_politeness",),
    "zh": ("request_politeness",),
}

#: Delivery axes judged on every voice call (the delivery judge's first two
#: axes; the third axis is the pack's per-factor rubric).
DELIVERY_AXES = ("fidelity", "intonation")

#: Forced batch-name brand of a judge-visible (--show-judge) annotate build.
#: The builder ALWAYS appends it, so every artifact the build emits — packet
#: directory, manifest, page header, and every browser-exported CSV row's
#: ``batch`` column — carries the mark, and the agreement report can refuse
#: judge-visible returns even when only the CSV survives.
JUDGE_VISIBLE_BATCH_SUFFIX = "_judge_visible"

#: Semantic conciseness is flagged below this normalized score — the EVA-X
#: pass threshold (EVA_X_CONCISENESS_THRESHOLD in the conversation-judge
#: suite; kept as a local constant so this module never imports the package).
SEMANTIC_CONCISENESS_FLAG_BELOW = 0.5


class CalibrationJudge(str, Enum):
    """The judge families the calibration wave collects human labels for."""

    NATIVENESS = "nativeness"
    DELIVERY = "delivery"
    QUALITY = "quality"
    SEMANTIC = "semantic"


class ProgressionSeverity(str, Enum):
    """Severity of one flagged semantic progression dimension (EVA rating
    1 = major, 2 = minor; an unflagged dimension carries no severity)."""

    MAJOR = "major"
    MINOR = "minor"


def stratum_of(
    judge: CalibrationJudge,
    factor_id: str,
    severity: Optional[ProgressionSeverity] = None,
) -> str:
    """The stratum id of one (judge, factor) defect cell; semantic
    progression cells carry a severity suffix (``semantic:<dim>:major``)."""
    base = f"{judge.value}:{factor_id}"
    return f"{base}:{severity.value}" if severity is not None else base


class JudgeVerdictCell(BaseModel):
    """One stored judge verdict on one call, normalized across judge families."""

    judge: Annotated[CalibrationJudge, Field(description="Judge family.")]
    factor_id: Annotated[
        str,
        Field(
            description="Factor / axis / dimension id within the judge family "
            "(e.g. a nativeness factor, 'fidelity', a quality factor, or a "
            "progression dimension)."
        ),
    ]
    outcome: Annotated[JudgeOutcome, Field(description="Normalized verdict.")]
    severity: Annotated[
        Optional[ProgressionSeverity],
        Field(
            description="Severity of a FLAGGED semantic progression "
            "dimension (rating 1 = major, 2 = minor); None on every other "
            "cell. Splits the semantic strata so rare majors are drawn "
            "independently of the plentiful minors."
        ),
    ] = None
    evidence: Annotated[
        str, Field(description="Judge rationale / measurement, when stored.")
    ] = ""
    quote: Annotated[
        str, Field(description="Exact offending span, when the judge pinned one.")
    ] = ""

    @property
    def stratum_id(self) -> str:
        return stratum_of(self.judge, self.factor_id, self.severity)


# ---------------------------------------------------------------------------
# Conversation-judge artifact reader — minimal, local, extra-tolerant.
# Target shape: tau2.judges.conversation.models.ConversationJudgeArtifact
# (PR #747): HEADLINE per-dimension progression + per-turn conciseness on
# each call, composites demoted to shadow_scores.
# ---------------------------------------------------------------------------


class SemanticProgressionDimension(BaseModel):
    """One HEADLINE progression dimension from the conversation artifact."""

    model_config = ConfigDict(extra="ignore")

    name: Annotated[str, Field(description="Closed progression dimension name.")]
    flagged: Annotated[bool, Field(description="Whether the defect occurred.")]
    rating: Annotated[
        int,
        Field(ge=1, le=3, description="EVA severity rating (3 clean, 2/1 flagged)."),
    ]
    evidence: Annotated[str, Field(description="Judge rationale.")] = ""

    @model_validator(mode="after")
    def _flag_and_rating_agree(self) -> "SemanticProgressionDimension":
        # Mirrors the producer contract (EvaProgressionDimension): a stored
        # artifact contradicting it is a loud error, never a silent stratum.
        if self.flagged is (self.rating == 3):
            raise ValueError(
                f"{self.name}: flagged={self.flagged} contradicts rating={self.rating}"
            )
        return self

    @property
    def severity(self) -> Optional[ProgressionSeverity]:
        if not self.flagged:
            return None
        return (
            ProgressionSeverity.MAJOR if self.rating == 1 else ProgressionSeverity.MINOR
        )


class SemanticConcisenessTurn(BaseModel):
    """One HEADLINE per-agent-turn conciseness judgment."""

    model_config = ConfigDict(extra="ignore")

    rating: Annotated[int, Field(description="EVA conciseness rating, 1..3.")]
    evidence: Annotated[str, Field(description="Judge rationale.")] = ""


class SemanticShadowScores(BaseModel):
    """The artifact's demoted composites; only the quality info is read."""

    model_config = ConfigDict(extra="ignore")

    ours: Annotated[
        Optional[QualityInfo],
        Field(
            description="Tau quality-suite output stored in the artifact; used "
            "as the quality verdict source when the run itself carries none."
        ),
    ] = None


class SemanticCallRecord(BaseModel):
    """One call's conversation-judge record, keyed by (source, sim_id)."""

    model_config = ConfigDict(extra="ignore")

    source: Annotated[str, Field(description="Absolute source results path.")]
    sim_id: Annotated[str, Field(description="Source simulation id.")]
    task_id: Annotated[
        Optional[str],
        Field(description="Source task id, when recorded by the artifact."),
    ] = None
    progression_dimensions: Annotated[
        list[SemanticProgressionDimension],
        Field(default_factory=list, description="HEADLINE progression verdicts."),
    ]
    conciseness_turns: Annotated[
        list[SemanticConcisenessTurn],
        Field(default_factory=list, description="HEADLINE per-turn conciseness."),
    ]
    shadow_scores: SemanticShadowScores = Field(default_factory=SemanticShadowScores)

    @property
    def ours(self) -> Optional[QualityInfo]:
        return self.shadow_scores.ours

    @property
    def conciseness_score(self) -> Optional[float]:
        """Mean normalized turn rating recomputed from the HEADLINE turns
        ((rating-1)/2 per turn); None when the call has no spoken agent turns."""
        if not self.conciseness_turns:
            return None
        return sum((turn.rating - 1) / 2 for turn in self.conciseness_turns) / len(
            self.conciseness_turns
        )


def _normalize_source(source: str) -> str:
    """Fold a recorded source (results dir OR results.json) onto the
    results.json path this module keys sims by."""
    trimmed = source.rstrip("/")
    if Path(trimmed).name != "results.json":
        return f"{trimmed}/results.json"
    return trimmed


class SemanticArtifactReader(BaseModel):
    """Minimal typed reader for a stored conversation-judge artifact."""

    model_config = ConfigDict(extra="ignore")

    calls: list[SemanticCallRecord] = Field(default_factory=list)

    def by_key(self) -> dict[tuple[str, str], SemanticCallRecord]:
        return {
            (_normalize_source(record.source), record.sim_id): record
            for record in self.calls
        }


def load_semantic_artifact(path: Path) -> SemanticArtifactReader:
    """Read the semantic slice of an conversation-judge artifact."""
    return SemanticArtifactReader.model_validate_json(Path(path).read_text())


# ---------------------------------------------------------------------------
# Verdict normalization (one cell list per call, shared by draw + report)
# ---------------------------------------------------------------------------


def judge_verdict_cells(
    sim: SimulationRun,
    language: str,
    semantic: Optional[SemanticCallRecord] = None,
) -> list[JudgeVerdictCell]:
    """Every stored judge verdict on one call, normalized to typed cells.

    Purely a read of stored state — nothing here invokes a judge. Only the
    factors the calibration instrument asks humans about are emitted: every
    nativeness factor live in the current pack (LLM-judged AND deterministic
    checkers), every quality factor (LLM, hybrid, and deterministic), the two
    delivery axes plus pack delivery factors, and the semantic
    progression/conciseness suite.
    """
    cells: list[JudgeVerdictCell] = []

    if sim.nativeness_info is not None:
        live_factors = {
            factor.id for factor in judge_factors_for(language) if factor.enabled
        }
        for check in sim.nativeness_info.factor_checks:
            if check.id not in live_factors:
                continue
            cells.append(
                JudgeVerdictCell(
                    judge=CalibrationJudge.NATIVENESS,
                    factor_id=check.id,
                    outcome=check.outcome,
                    evidence=check.evidence or "",
                    quote=check.quote or "",
                )
            )

    quality_info = sim.quality_info or (semantic.ours if semantic else None)
    if quality_info is not None:
        asked_quality = {factor.id for factor in QUALITY_FACTORS}
        for check in quality_info.factor_checks:
            if check.id not in asked_quality:
                continue
            cells.append(
                JudgeVerdictCell(
                    judge=CalibrationJudge.QUALITY,
                    factor_id=check.id,
                    outcome=check.outcome,
                    evidence=check.evidence or "",
                    quote=check.quote or "",
                )
            )

    delivery = sim.delivery_info
    if delivery is not None and delivery.num_judged > 0:
        for axis in DELIVERY_AXES:
            findings = [
                finding
                for result in delivery.utterance_results
                for finding in result.findings
                if finding.axis == axis
            ]
            cells.append(
                JudgeVerdictCell(
                    judge=CalibrationJudge.DELIVERY,
                    factor_id=axis,
                    outcome=JudgeOutcome.FAIL if findings else JudgeOutcome.PASS,
                    evidence="; ".join(
                        f"{finding.category}: {finding.issue or ''}".strip(": ")
                        for finding in findings[:5]
                    ),
                )
            )
        pack_factor_ids = {
            factor.id for factor in delivery_factors_for(language) if factor.enabled
        }
        for check in delivery.factor_checks:
            if check.id not in pack_factor_ids:
                continue
            cells.append(
                JudgeVerdictCell(
                    judge=CalibrationJudge.DELIVERY,
                    factor_id=check.id,
                    outcome=check.outcome,
                    evidence=check.evidence or "",
                    quote=check.quote or "",
                )
            )

    if semantic is not None:
        for dimension in semantic.progression_dimensions:
            cells.append(
                JudgeVerdictCell(
                    judge=CalibrationJudge.SEMANTIC,
                    factor_id=dimension.name,
                    outcome=(
                        JudgeOutcome.FAIL if dimension.flagged else JudgeOutcome.PASS
                    ),
                    severity=dimension.severity,
                    evidence=dimension.evidence,
                )
            )
        # The conciseness cell exists only when the suite produced headline
        # output for the call (dimensions or turns); an errored call has none.
        if semantic.progression_dimensions or semantic.conciseness_turns:
            score = semantic.conciseness_score
            if score is None:
                outcome = JudgeOutcome.NO_OPPORTUNITY
            elif score < SEMANTIC_CONCISENESS_FLAG_BELOW:
                outcome = JudgeOutcome.FAIL
            else:
                outcome = JudgeOutcome.PASS
            worst = [
                turn.evidence
                for turn in semantic.conciseness_turns
                if turn.rating < 3 and turn.evidence
            ]
            cells.append(
                JudgeVerdictCell(
                    judge=CalibrationJudge.SEMANTIC,
                    factor_id="conciseness",
                    outcome=outcome,
                    evidence=(
                        f"mean normalized turn score {score:.3f}"
                        + (f" — {'; '.join(worst[:3])}" if worst else "")
                        if score is not None
                        else "no spoken agent turns"
                    ),
                )
            )
    return cells


def calibration_instrument_cells(
    sim: SimulationRun,
    language: str,
    semantic: Optional[SemanticCallRecord] = None,
) -> list[JudgeVerdictCell]:
    """The judge cells the CURRENT calibration instrument asks humans about
    (``CALIBRATION_CELLS_VERSION``).

    ``judge_verdict_cells`` stays the raw read of every stored verdict; this
    is the instrument's view of it: deterministic quality checkers (the EVA
    turn-taking suite) are out — they are metric threshold events, not judged
    defects; the semantic progression/conciseness cells are out; and the two
    delivery axes stay SEPARATE ``fidelity`` and ``intonation`` cells
    (cells-v3 unfold), matching the instrument's two generic audio
    questions. The ``semantic`` record is still consumed: on corpora whose
    quality verdicts live in the conversation artifact (``semantic.ours``)
    rather than on the sim, it is the ONLY source of the surviving
    LLM-quality cells — only its semantic progression/conciseness cells are
    dropped. Every new draw, annotate build, and per-factor adjudication
    reads THIS view; agreement and adjudication over pre-cut sidecars keep
    the raw view.
    """
    llm_quality = {
        factor.id for factor in QUALITY_FACTORS if factor.evaluator != "deterministic"
    }
    cells: list[JudgeVerdictCell] = []
    for cell in judge_verdict_cells(sim, language, semantic):
        if cell.judge is CalibrationJudge.SEMANTIC:
            continue
        if cell.judge is CalibrationJudge.QUALITY and cell.factor_id not in llm_quality:
            continue
        cells.append(cell)
    return cells


def fold_fidelity_cells(cells: list[JudgeVerdictCell]) -> list[JudgeVerdictCell]:
    """The cells-v2 FOLDED view of an instrument cell list: the two delivery
    axes collapse into ONE ``fidelity`` cell. v2 sidecars were built and
    annotated against this view, so reading them back (adjudication,
    agreement) must reconstruct it; nothing built at cells-v3+ folds."""
    out: list[JudgeVerdictCell] = []
    axes: list[JudgeVerdictCell] = []
    for cell in cells:
        if cell.judge is CalibrationJudge.DELIVERY and cell.factor_id in DELIVERY_AXES:
            axes.append(cell)
            continue
        out.append(cell)
    if axes:
        failed = [cell for cell in axes if cell.outcome is JudgeOutcome.FAIL]
        out.append(
            JudgeVerdictCell(
                judge=CalibrationJudge.DELIVERY,
                factor_id="fidelity",
                outcome=JudgeOutcome.FAIL if failed else JudgeOutcome.PASS,
                evidence="; ".join(cell.evidence for cell in failed if cell.evidence),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Frame + draw
# ---------------------------------------------------------------------------


class FrameCall(BaseModel):
    """One eligible stored call in the calibration sampling frame."""

    results_path: Annotated[
        str, Field(description="Resolved results.json path of the source run.")
    ]
    sim_id: Annotated[str, Field(description="Source simulation id.")]
    task_id: Annotated[str, Field(description="Source task id.")]
    trial: Annotated[int, Field(description="Source trial index.")] = 0
    domain: Annotated[str, Field(description="Source domain.")] = ""
    provider: Annotated[
        Optional[str], Field(description="Audio-native provider, when recorded.")
    ] = None
    experiment: Annotated[
        str, Field(description="Human label of the source run (its dir name).")
    ] = ""
    has_audio: Annotated[
        bool, Field(description="Whether the run stores the call's both.wav.")
    ] = False
    has_agent_speech: Annotated[
        bool,
        Field(
            description="False only when the call stores ticks and NONE "
            "carries agent speech — a fully-mute call cannot render an "
            "agent-only clip, so it is ineligible for every calibration "
            "packet (annotate and adjudicate alike)."
        ),
    ] = True
    judges_present: Annotated[
        list[CalibrationJudge],
        Field(description="Judge families with stored verdicts on this call."),
    ]
    strata: Annotated[
        list[str],
        Field(
            description="Stratum memberships: '<judge>:<factor>' defect cells "
            "(a stored FAIL; semantic progression cells are severity-keyed "
            "'semantic:<dimension>:major|minor') and/or 'clean_control'."
        ),
    ]
    drawn: Annotated[
        bool, Field(description="Whether the seeded draw selected this call.")
    ] = False
    drawn_for: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Strata whose draw selected this call (empty unless drawn).",
        ),
    ]

    @property
    def key(self) -> tuple[str, str]:
        return (self.results_path, self.sim_id)


class CalibrationStratumAccounting(BaseModel):
    """How one stratum fared in the draw."""

    stratum_id: Annotated[str, Field(description="'<judge>:<factor>' or control.")]
    judge: Annotated[
        Optional[CalibrationJudge],
        Field(description="Judge family; None for the clean-control stratum."),
    ] = None
    frame_size: Annotated[
        int, Field(ge=0, description="Frame calls belonging to this stratum.")
    ]
    drawn: Annotated[
        int, Field(ge=0, description="Calls this stratum's own draw selected.")
    ]
    covered: Annotated[
        int,
        Field(
            ge=0,
            description="Stratum members in the final cohort, however drawn — "
            "the count that matters for calibration coverage.",
        ),
    ]
    inclusion_probability: Annotated[
        Optional[float],
        Field(
            description="Achieved coverage rate covered/frame_size; None when "
            "the stratum is empty. Per-call probabilities are not uniform "
            "across overlapping strata — recompute from the persisted "
            "membership matrix when a design-based estimator needs them."
        ),
    ] = None


class CalibrationDrawConfig(BaseModel):
    """Everything one annotate-mode (blind wave) calibration draw depends on."""

    draw: Annotated[
        Literal["annotate"],
        Field(description="Draw rule discriminator: the blind-wave draw."),
    ] = "annotate"
    rule: Annotated[
        Literal["enriched", "random"],
        Field(
            description="Cohort selection rule. 'enriched' is the "
            "defect-enriched draw (coverage → enrichment → controls) for "
            "waves that must exercise every judge factor. 'random' is the "
            "judge-blind uniform draw for the RECALL arm: n_calls seeded "
            "uniformly from the whole eligible frame, never conditioned on "
            "judge verdicts, so human-found defects on undrawn-by-any-judge "
            "calls stay reachable. control_fraction is ignored — clean calls "
            "enter at their natural rate."
        ),
    ] = "enriched"
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    n_calls: Annotated[int, Field(gt=0, description="Target cohort size (calls).")] = (
        DEFAULT_CALIBRATION_CALLS_PER_LANGUAGE
    )
    control_fraction: Annotated[
        float,
        Field(
            ge=0.0,
            lt=1.0,
            description="Share of the cohort reserved for judge-clean controls.",
        ),
    ] = DEFAULT_CALIBRATION_CONTROL_FRACTION
    seed: Annotated[int, Field(description="Draw seed.")] = DEFAULT_CALIBRATION_SEED
    require_audio: Annotated[
        bool,
        Field(
            description="Drop calls without stored audio (the annotate packet "
            "needs full + agent-only WAVs)."
        ),
    ] = True

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def n_controls(self) -> int:
        return round(self.n_calls * self.control_fraction)

    @property
    def n_defects(self) -> int:
        return self.n_calls - self.n_controls


class AdjudicationDrawConfig(BaseModel):
    """Everything one adjudicate-mode PER-FACTOR draw depends on.

    Independent of the blind wave: every defect stratum takes
    ``min(per_factor_cap, frame_size)`` of its flagged calls — capped, never
    padded (owner ruling: "max 20, if 20 don't exist, then whatever exists")
    — with no clean controls and no shared defect budget. Audio is required
    by default: the adjudication page ships the full + agent-only WAV pair
    (audio-judge decisions are made by listening), so a call whose wav was
    pruned or that never produced agent speech cannot build.
    """

    draw: Annotated[
        Literal["adjudicate_per_factor"],
        Field(description="Draw rule discriminator: the per-factor draw."),
    ] = "adjudicate_per_factor"
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    per_factor_cap: Annotated[
        int,
        Field(gt=0, description="Max flagged calls drawn per defect stratum."),
    ] = DEFAULT_CALIBRATION_ADJUDICATION_FACTOR_CAP
    seed: Annotated[int, Field(description="Draw seed.")] = DEFAULT_CALIBRATION_SEED
    require_audio: Annotated[
        bool, Field(description="Drop calls without stored audio.")
    ] = True

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return value.strip().lower()


class CalibrationFrame(BaseModel):
    """The persisted whole sampling frame, with the draw marked on it."""

    schema_version: Literal[1] = 1
    sampler_version: str = CALIBRATION_SAMPLER_VERSION
    created_at: Annotated[str, Field(description="Draw wall-clock time (UTC).")]
    git_sha: Annotated[str, Field(description="Repo HEAD at draw time.")]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    config: Annotated[
        CalibrationDrawConfig | AdjudicationDrawConfig,
        Field(
            discriminator="draw",
            description="Draw settings; the `draw` field names the rule.",
        ),
    ]
    results: Annotated[
        list[str], Field(description="Input results dirs / files, as given.")
    ]
    conversation_artifact: Annotated[
        Optional[str],
        Field(description="Semantic-suite artifact path, when supplied."),
    ] = None
    n_calls_scanned: Annotated[
        int, Field(ge=0, description="Stored calls streamed from the inputs.")
    ]
    n_dropped_no_judges: Annotated[
        int,
        Field(ge=0, description="Calls with no stored judge verdicts at all."),
    ]
    n_dropped_no_audio: Annotated[
        int,
        Field(ge=0, description="Calls dropped for missing audio (require_audio)."),
    ]
    n_dropped_mute: Annotated[
        int,
        Field(
            ge=0,
            description="Fully-mute calls dropped (require_audio): stored "
            "ticks carry NO agent speech, so no agent-only clip can exist.",
        ),
    ] = 0
    n_dropped_language: Annotated[
        int,
        Field(
            ge=0,
            description="Calls whose recorded language contradicts the draw's.",
        ),
    ]
    n_dropped_language_unknown: Annotated[
        int,
        Field(
            ge=0,
            description="Calls carrying NO recorded language — unattributable "
            "calls never enter a language-specific frame.",
        ),
    ] = 0
    must_cover: Annotated[
        list[str],
        Field(description="Strata the draw must cover when the frame has them."),
    ]
    must_cover_empty: Annotated[
        list[str],
        Field(
            description="Must-cover strata with ZERO frame calls — the corpus "
            "holds no such judge FAIL; the wave cannot validate these cells "
            "and the owner must extend the corpus or accept the gap."
        ),
    ]
    strata: Annotated[
        list[CalibrationStratumAccounting],
        Field(description="Per-stratum sizes, draws, coverage, and rates."),
    ]
    calls: Annotated[
        list[FrameCall],
        Field(description="EVERY eligible call, drawn or not (the whole frame)."),
    ]

    def drawn_calls(self) -> list[FrameCall]:
        return [call for call in self.calls if call.drawn]


def load_calibration_frame(path: Path) -> CalibrationFrame:
    return CalibrationFrame.model_validate_json(Path(path).read_text())


def frame_call_from_loaded(
    item: LoadedSim,
    config: "CalibrationDrawConfig | AdjudicationDrawConfig",
    semantic_by_key: dict[tuple[str, str], SemanticCallRecord],
) -> Optional[FrameCall]:
    """One frame row from a streamed sim; None when the call is ineligible
    for reasons the caller counts (no judges / no audio / wrong language are
    signalled via the returned row's fields — see build_calibration_frame)."""
    from tau2.annotation.packets.audio_quality import agent_speech_tick_spans
    from tau2.annotation.packets.builder import find_audio

    sim = item.sim
    results_path = str((item.results_dir / "results.json").resolve())
    semantic = semantic_by_key.get((results_path, sim.id))
    cells = calibration_instrument_cells(sim, config.language, semantic)
    if not cells:
        return None
    defect_strata = sorted(
        {cell.stratum_id for cell in cells if cell.outcome is JudgeOutcome.FAIL}
    )
    strata = defect_strata or [CLEAN_CONTROL_STRATUM]
    return FrameCall(
        results_path=results_path,
        sim_id=sim.id,
        task_id=str(sim.task_id),
        trial=sim.trial or 0,
        domain=item.domain,
        provider=sim.agent_provider,
        experiment=item.experiment_label,
        has_audio=find_audio(item.results_dir, sim) is not None,
        has_agent_speech=(bool(agent_speech_tick_spans(sim)) if sim.ticks else True),
        judges_present=sorted({cell.judge for cell in cells}),
        strata=strata,
    )


def _stratum_rng(seed: int, language: str, stratum: str) -> random.Random:
    """Independent per-(seed, language, stratum) stream, so adding a stratum
    or a language never reshuffles another stratum's draw."""
    return random.Random(f"tau2-calib-draw|{seed}|{language}|{stratum}")


def _draw_one(
    pool: list[FrameCall], rng: random.Random, stratum: str
) -> Optional[FrameCall]:
    undrawn = [call for call in pool if not call.drawn]
    if not undrawn:
        return None
    pick = undrawn[rng.randrange(len(undrawn))]
    pick.drawn = True
    pick.drawn_for.append(stratum)
    return pick


def _descriptive_stratum_accounting(
    members: dict[str, list[FrameCall]],
) -> list[CalibrationStratumAccounting]:
    """Per-stratum coverage rows over an already-marked draw (drawn counts
    reflect ``drawn_for``; a stratum nothing was drawn FOR reads drawn=0)."""
    accounting: list[CalibrationStratumAccounting] = []
    for stratum in sorted(members):
        pool = members[stratum]
        covered = sum(1 for call in pool if call.drawn)
        accounting.append(
            CalibrationStratumAccounting(
                stratum_id=stratum,
                judge=(
                    CalibrationJudge(stratum.split(":", 1)[0])
                    if stratum != CLEAN_CONTROL_STRATUM
                    else None
                ),
                frame_size=len(pool),
                drawn=sum(1 for call in pool if stratum in call.drawn_for),
                covered=covered,
                inclusion_probability=(covered / len(pool)) if pool else None,
            )
        )
    return accounting


def _draw_random_cohort(
    calls: list[FrameCall], config: CalibrationDrawConfig
) -> list[CalibrationStratumAccounting]:
    """Mark the seeded judge-blind uniform draw on the frame calls, in place.

    One pass: ``n_calls`` drawn uniformly (without replacement) from the whole
    eligible frame in stable order — judge verdicts never influence selection,
    so every call has the same inclusion probability and the cohort estimates
    recall without inheriting any judge's blind spots. Short frames are taken
    whole and logged. The accounting leads with the one selective stratum
    (the whole frame) followed by descriptive per-stratum coverage rows.
    """
    calls_sorted = sorted(
        calls, key=lambda c: (c.results_path, c.task_id, c.trial, c.sim_id)
    )
    rng = random.Random(
        f"tau2-calib-draw|{config.seed}|{config.language}|{RANDOM_DRAW_STRATUM}"
    )
    n = min(config.n_calls, len(calls_sorted))
    if n < config.n_calls:
        logger.warning(
            f"random frame exhausted at {n}/{config.n_calls} eligible calls "
            f"for {config.language} — the frame is the frame"
        )
    for pick in rng.sample(calls_sorted, n):
        pick.drawn = True
        pick.drawn_for.append(RANDOM_DRAW_STRATUM)

    members: dict[str, list[FrameCall]] = {}
    for call in calls_sorted:
        for stratum in call.strata:
            members.setdefault(stratum, []).append(call)
    frame_size = len(calls_sorted)
    return [
        CalibrationStratumAccounting(
            stratum_id=RANDOM_DRAW_STRATUM,
            judge=None,
            frame_size=frame_size,
            drawn=n,
            covered=n,
            inclusion_probability=(n / frame_size) if frame_size else None,
        ),
        *_descriptive_stratum_accounting(members),
    ]


def draw_calibration_cohort(
    calls: list[FrameCall], config: CalibrationDrawConfig
) -> list[CalibrationStratumAccounting]:
    """Mark the seeded draw on the frame calls, in place.

    ``rule="random"`` delegates to the judge-blind uniform draw
    (``_draw_random_cohort``). The default enriched rule runs three
    deterministic passes:

    1. **Coverage** — every non-empty defect stratum gets at least one cohort
       member (must-cover strata first), drawn uniformly from its undrawn
       members unless an earlier draw already covers it.
    2. **Enrichment** — the remaining defect budget is dealt round-robin
       across defect strata in the same order, one uniform draw per stratum
       per round, until the budget is spent or the frame is exhausted.
    3. **Controls** — ``n_controls`` uniform draws from the judge-clean
       stratum (short control pools are taken whole and logged, never
       back-filled from defect calls).
    """
    if config.rule == "random":
        return _draw_random_cohort(calls, config)
    calls_sorted = sorted(
        calls, key=lambda c: (c.results_path, c.task_id, c.trial, c.sim_id)
    )
    members: dict[str, list[FrameCall]] = {}
    for call in calls_sorted:
        for stratum in call.strata:
            members.setdefault(stratum, []).append(call)

    must_cover = [
        stratum_of(CalibrationJudge.NATIVENESS, factor_id)
        for factor_id in COLD_UNVALIDATED_FACTORS.get(config.language, ())
    ]
    defect_strata = [s for s in must_cover if s in members] + sorted(
        s for s in members if s != CLEAN_CONTROL_STRATUM and s not in must_cover
    )
    rngs = {
        stratum: _stratum_rng(config.seed, config.language, stratum)
        for stratum in [*defect_strata, CLEAN_CONTROL_STRATUM]
    }

    drawn_defects = 0
    # Pass 1: minimum coverage.
    for stratum in defect_strata:
        if any(call.drawn for call in members[stratum]):
            continue
        if _draw_one(members[stratum], rngs[stratum], stratum) is not None:
            drawn_defects += 1
    if drawn_defects > config.n_defects:
        logger.warning(
            f"stratum coverage needed {drawn_defects} defect calls, over the "
            f"defect budget of {config.n_defects} — coverage wins, the cohort "
            "runs larger than n_calls"
        )
    # Pass 2: round-robin enrichment.
    progressed = True
    while drawn_defects < config.n_defects and progressed:
        progressed = False
        for stratum in defect_strata:
            if drawn_defects >= config.n_defects:
                break
            if _draw_one(members[stratum], rngs[stratum], stratum) is not None:
                drawn_defects += 1
                progressed = True
    if drawn_defects < config.n_defects:
        logger.warning(
            f"defect frame exhausted at {drawn_defects}/{config.n_defects} "
            f"defect calls for {config.language} — the frame is the frame; "
            "no back-filling from controls"
        )
    # Pass 3: controls.
    control_pool = members.get(CLEAN_CONTROL_STRATUM, [])
    drawn_controls = 0
    for _ in range(config.n_controls):
        if (
            _draw_one(control_pool, rngs[CLEAN_CONTROL_STRATUM], CLEAN_CONTROL_STRATUM)
            is None
        ):
            break
        drawn_controls += 1
    if drawn_controls < config.n_controls:
        logger.warning(
            f"clean-control pool exhausted at {drawn_controls}/"
            f"{config.n_controls} for {config.language}"
        )

    accounting: list[CalibrationStratumAccounting] = []
    for stratum in [*defect_strata, CLEAN_CONTROL_STRATUM]:
        pool = members.get(stratum, [])
        covered = sum(1 for call in pool if call.drawn)
        accounting.append(
            CalibrationStratumAccounting(
                stratum_id=stratum,
                judge=(
                    CalibrationJudge(stratum.split(":", 1)[0])
                    if stratum != CLEAN_CONTROL_STRATUM
                    else None
                ),
                frame_size=len(pool),
                drawn=sum(1 for call in pool if stratum in call.drawn_for),
                covered=covered,
                inclusion_probability=(covered / len(pool)) if pool else None,
            )
        )
    return accounting


class FrameScan(BaseModel):
    """The eligible calls streamed off the inputs, plus the drop counters."""

    calls: Annotated[list[FrameCall], Field(description="Eligible frame calls.")]
    n_scanned: Annotated[int, Field(ge=0, description="Stored calls streamed.")]
    n_no_judges: Annotated[
        int, Field(ge=0, description="Calls with no stored judge verdicts.")
    ]
    n_no_audio: Annotated[
        int, Field(ge=0, description="Calls dropped for missing audio.")
    ]
    n_mute: Annotated[
        int,
        Field(ge=0, description="Fully-mute calls dropped (no agent speech)."),
    ] = 0
    n_language: Annotated[
        int, Field(ge=0, description="Calls in a different recorded language.")
    ]
    n_language_unknown: Annotated[
        int, Field(ge=0, description="Calls carrying no recorded language.")
    ]


def _scan_frame(
    loaded_sims,
    config: "CalibrationDrawConfig | AdjudicationDrawConfig",
    semantic_by_key: dict[tuple[str, str], SemanticCallRecord],
) -> FrameScan:
    """Stream the inputs into eligible frame calls (shared by both draws).

    Sims are dropped after their frame row is built, so large runs never sit
    in memory whole.
    """
    calls: list[FrameCall] = []
    seen: set[tuple[str, str]] = set()
    n_scanned = n_no_judges = n_no_audio = n_language = n_language_unknown = 0
    n_mute = 0
    for item in loaded_sims:
        n_scanned += 1
        sim_language = (
            item.sim.nativeness_info.language
            if item.sim.nativeness_info is not None
            else None
        )
        # A call that cannot be attributed to a language must never enter a
        # language-specific frame — quality/delivery/semantic verdicts alone
        # say nothing about the conversation's language.
        if not sim_language:
            n_language_unknown += 1
            continue
        if sim_language.lower() != config.language:
            n_language += 1
            continue
        row = frame_call_from_loaded(item, config, semantic_by_key)
        if row is None:
            n_no_judges += 1
            continue
        if config.require_audio and not row.has_audio:
            n_no_audio += 1
            continue
        if config.require_audio and not row.has_agent_speech:
            n_mute += 1
            continue
        if row.key in seen:
            continue
        seen.add(row.key)
        calls.append(row)
    return FrameScan(
        calls=calls,
        n_scanned=n_scanned,
        n_no_judges=n_no_judges,
        n_no_audio=n_no_audio,
        n_mute=n_mute,
        n_language=n_language,
        n_language_unknown=n_language_unknown,
    )


def _must_cover_strata(language: str, present: set[str]) -> tuple[list[str], list[str]]:
    must_cover = [
        stratum_of(CalibrationJudge.NATIVENESS, factor_id)
        for factor_id in COLD_UNVALIDATED_FACTORS.get(language, ())
    ]
    must_cover_empty = [s for s in must_cover if s not in present]
    for stratum in must_cover_empty:
        logger.warning(
            f"must-cover stratum {stratum} has NO frame calls — the corpus "
            "holds no such judge FAIL; this cell stays cold-unvalidated"
        )
    return must_cover, must_cover_empty


def _frame_from_scan(
    scan: FrameScan,
    config: "CalibrationDrawConfig | AdjudicationDrawConfig",
    strata_accounting: list[CalibrationStratumAccounting],
    *,
    results: list[Path],
    conversation_artifact: Optional[Path],
) -> CalibrationFrame:
    if getattr(config, "rule", None) == "random":
        # A judge-blind uniform draw cannot promise stratum coverage; claiming
        # must-cover cells on its frame would misstate the design.
        must_cover, must_cover_empty = [], []
    else:
        must_cover, must_cover_empty = _must_cover_strata(
            config.language, {row.stratum_id for row in strata_accounting}
        )
    return CalibrationFrame(
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
        language=config.language,
        config=config,
        results=[str(path) for path in results],
        conversation_artifact=(
            str(conversation_artifact) if conversation_artifact is not None else None
        ),
        n_calls_scanned=scan.n_scanned,
        n_dropped_no_judges=scan.n_no_judges,
        n_dropped_no_audio=scan.n_no_audio,
        n_dropped_mute=scan.n_mute,
        n_dropped_language=scan.n_language,
        n_dropped_language_unknown=scan.n_language_unknown,
        must_cover=must_cover,
        must_cover_empty=must_cover_empty,
        strata=strata_accounting,
        calls=scan.calls,
    )


def build_calibration_frame(
    loaded_sims,
    config: CalibrationDrawConfig,
    *,
    results: list[Path],
    conversation_artifact: Optional[Path] = None,
    semantic_by_key: Optional[dict[tuple[str, str], SemanticCallRecord]] = None,
) -> CalibrationFrame:
    """Enumerate the frame from streamed sims and mark the seeded blind-wave
    draw on it (coverage → enrichment → controls; see
    ``draw_calibration_cohort``)."""
    scan = _scan_frame(loaded_sims, config, semantic_by_key or {})
    strata_accounting = draw_calibration_cohort(scan.calls, config)
    return _frame_from_scan(
        scan,
        config,
        strata_accounting,
        results=results,
        conversation_artifact=conversation_artifact,
    )


def _adjudication_rng(seed: int, language: str, stratum: str) -> random.Random:
    """The per-factor draw's own independent per-(seed, language, stratum)
    stream — keyed apart from the blind wave's so the two draws never share
    (or perturb) a sequence."""
    return random.Random(f"tau2-calib-adj-draw|{seed}|{language}|{stratum}")


def draw_adjudication_cohort(
    calls: list[FrameCall], config: AdjudicationDrawConfig
) -> list[CalibrationStratumAccounting]:
    """Mark the seeded PER-FACTOR draw on the frame calls, in place.

    One pass over the defect strata in stable order: each stratum draws
    uniformly from its undrawn members until ``min(per_factor_cap,
    frame_size)`` of its members are in the cohort — members already drawn
    for an overlapping stratum count toward the target, so overlap shrinks
    the cohort instead of double-drawing. Capped, never padded: a stratum
    with fewer flagged calls than the cap is taken whole. Clean controls are
    never drawn (the adjudication reviews judge POSITIVES); the control
    stratum is still accounted so the frame states how many clean calls the
    corpus held.
    """
    calls_sorted = sorted(
        calls, key=lambda c: (c.results_path, c.task_id, c.trial, c.sim_id)
    )
    members: dict[str, list[FrameCall]] = {}
    for call in calls_sorted:
        for stratum in call.strata:
            members.setdefault(stratum, []).append(call)

    defect_strata = sorted(s for s in members if s != CLEAN_CONTROL_STRATUM)
    for stratum in defect_strata:
        pool = members[stratum]
        rng = _adjudication_rng(config.seed, config.language, stratum)
        target = min(config.per_factor_cap, len(pool))
        while sum(1 for call in pool if call.drawn) < target:
            if _draw_one(pool, rng, stratum) is None:
                break

    accounting: list[CalibrationStratumAccounting] = []
    for stratum in [*defect_strata, CLEAN_CONTROL_STRATUM]:
        pool = members.get(stratum, [])
        covered = sum(1 for call in pool if call.drawn)
        accounting.append(
            CalibrationStratumAccounting(
                stratum_id=stratum,
                judge=(
                    CalibrationJudge(stratum.split(":", 1)[0])
                    if stratum != CLEAN_CONTROL_STRATUM
                    else None
                ),
                frame_size=len(pool),
                drawn=sum(1 for call in pool if stratum in call.drawn_for),
                covered=covered,
                inclusion_probability=(covered / len(pool)) if pool else None,
            )
        )
    return accounting


def build_adjudication_frame(
    loaded_sims,
    config: AdjudicationDrawConfig,
    *,
    results: list[Path],
    conversation_artifact: Optional[Path] = None,
    semantic_by_key: Optional[dict[tuple[str, str], SemanticCallRecord]] = None,
) -> CalibrationFrame:
    """Enumerate the frame from streamed sims and mark the seeded per-factor
    draw on it (``draw_adjudication_cohort``). Independent of any annotate
    build: it reads the stored results directly, never a coverage sidecar."""
    scan = _scan_frame(loaded_sims, config, semantic_by_key or {})
    strata_accounting = draw_adjudication_cohort(scan.calls, config)
    return _frame_from_scan(
        scan,
        config,
        strata_accounting,
        results=results,
        conversation_artifact=conversation_artifact,
    )


# ---------------------------------------------------------------------------
# Coverage sidecar (owner-facing; NEVER written inside the rater packet)
# ---------------------------------------------------------------------------


class CalibrationCoverageRow(BaseModel):
    """Why one cohort call was sampled — owner-facing only."""

    clip_id: Annotated[str, Field(description="Blind clip id in the packet.")]
    sim_id: Annotated[str, Field(description="Source simulation id.")]
    task_id: Annotated[str, Field(description="Source task id.")]
    trial: Annotated[int, Field(description="Source trial index.")] = 0
    results_path: Annotated[str, Field(description="Source results.json path.")]
    provider: Annotated[Optional[str], Field(description="Arm provider.")] = None
    experiment: Annotated[
        str, Field(description="Source run label (its dir name; encodes the arm).")
    ] = ""
    judges_present: Annotated[
        list[CalibrationJudge],
        Field(description="Judge families this call calibrates."),
    ]
    strata: Annotated[
        list[str], Field(description="Every stratum this call belongs to.")
    ]
    drawn_for: Annotated[
        list[str], Field(description="Strata whose draw selected this call.")
    ]


class CalibrationCoverageSidecar(BaseModel):
    """The builder's clip→source join and sampling rationale.

    Written OUTSIDE the packet directory; the adjudication packet and the
    agreement report consume it. It never ships to raters.
    """

    schema_version: Literal[1] = 1
    sampler_version: Annotated[
        str, Field(description="Sampler that drew this cohort.")
    ] = CALIBRATION_SAMPLER_VERSION
    created_at: Annotated[str, Field(description="Build wall-clock time (UTC).")]
    git_sha: Annotated[str, Field(description="Repo HEAD at build time.")]
    batch_id: Annotated[str, Field(description="The rater packet's batch id.")]
    batch_name: Annotated[str, Field(description="The rater packet's batch name.")]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    frame_path: Annotated[
        str, Field(description="The persisted CalibrationFrame this draw used.")
    ]
    per_factor_counts: Annotated[
        Optional[dict[str, int]],
        Field(
            description="Adjudicate-mode per-factor draws only: covered cohort "
            "calls per defect stratum (the min(cap, #flagged) accounting; the "
            "frame's stratum table carries the full frame_size/drawn/covered "
            "detail). None on blind-wave (annotate) sidecars."
        ),
    ] = None
    conversation_artifact: Annotated[
        Optional[str],
        Field(
            description="Conversation-judge artifact path, when supplied. "
            "Pre-cut sidecars read its semantic cells; the folded view still "
            "reads it for the LLM-quality verdicts (``ours``) on corpora "
            "that store them there rather than on the sim."
        ),
    ] = None
    cells_version: Annotated[
        Optional[str],
        Field(
            description="Instrument cell view this cohort was built with "
            "(``CALIBRATION_CELLS_VERSION``). None = a pre-cut sidecar: the "
            "adjudication page and agreement report reconstruct the raw "
            "``judge_verdict_cells`` view (with the sidecar's semantic "
            "artifact) instead of the folded instrument view."
        ),
    ] = None
    judge_visible: Annotated[
        bool,
        Field(
            description="Whether the annotate build rendered judge "
            "annotations inline (--show-judge). A judge-visible build is for "
            "enriched internal review only: its rater returns are NOT valid "
            "blind calibration data and calibration-agreement refuses them."
        ),
    ] = False
    judge_versions: Annotated[
        dict[str, list[str]],
        Field(
            default_factory=dict,
            description="Observed judge version stamps per judge family "
            "(model ids, prompt/rubric versions) across the cohort.",
        ),
    ]
    rows: Annotated[
        list[CalibrationCoverageRow], Field(description="One row per cohort call.")
    ]


def load_calibration_sidecar(path: Path) -> CalibrationCoverageSidecar:
    return CalibrationCoverageSidecar.model_validate_json(Path(path).read_text())


def observed_judge_versions(sims: list[SimulationRun]) -> dict[str, list[str]]:
    """Version stamps of the stored verdicts a cohort was judged under."""
    versions: dict[str, set[str]] = {}

    def note(family: str, value: Optional[str]) -> None:
        if value:
            versions.setdefault(family, set()).add(value)

    for sim in sims:
        if sim.nativeness_info is not None:
            note("nativeness_model", sim.nativeness_info.judge_model)
            note("nativeness_prompt", sim.nativeness_info.judge_prompt_version)
            note("nativeness_rubric", sim.nativeness_info.rubric_version)
        if sim.quality_info is not None:
            note("quality_rubric", sim.quality_info.rubric_version)
        if sim.delivery_info is not None:
            note("delivery_model", sim.delivery_info.judge_model)
            note("delivery_prompt", sim.delivery_info.judge_prompt_version)
    return {family: sorted(values) for family, values in versions.items()}


__all__ = [
    "CALIBRATION_SAMPLER_VERSION",
    "CLEAN_CONTROL_STRATUM",
    "COLD_UNVALIDATED_FACTORS",
    "DELIVERY_AXES",
    "JUDGE_VISIBLE_BATCH_SUFFIX",
    "SEMANTIC_CONCISENESS_FLAG_BELOW",
    "AdjudicationDrawConfig",
    "CalibrationCoverageRow",
    "CalibrationCoverageSidecar",
    "CalibrationDrawConfig",
    "CalibrationFrame",
    "CalibrationJudge",
    "CalibrationStratumAccounting",
    "FrameCall",
    "FrameScan",
    "JudgeVerdictCell",
    "ProgressionSeverity",
    "SemanticArtifactReader",
    "SemanticCallRecord",
    "build_adjudication_frame",
    "build_calibration_frame",
    "draw_adjudication_cohort",
    "draw_calibration_cohort",
    "frame_call_from_loaded",
    "judge_verdict_cells",
    "calibration_instrument_cells",
    "CALIBRATION_CELLS_VERSION",
    "load_calibration_frame",
    "load_calibration_sidecar",
    "load_semantic_artifact",
    "observed_judge_versions",
    "stratum_of",
]
