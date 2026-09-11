# Copyright Sierra
"""Facts-first interaction suite (``tau2 metrics interaction-facts``).

The τ-ML paper's HEADLINE interaction report is this deterministic fact
table — per call and aggregated per (language x domain x provider x
reasoning-effort) cell. Composite scores (the EVA-X adaptation's turn-taking
component here; the full EVA-X conjunction and the τ quality composite in
``tau2 judges legacy conversation``) are demoted to a permanent, structurally
separate ``shadow_scores`` section: still computed and version-stamped for
comparability, labeled ``uncalibrated-shadow``, never ranked or headlined.
EVA-Bench-style thresholds and composites are arbitrary and lossy across
languages (binding decision, 2026-08-18); the facts are not.

Everything is derived from stored simulations on the DISCRETE tick clock.
Event extraction is REUSED, never re-derived:

- turn routing, latencies, overlaps, and yields come from the canonical
  typed-event scorer in ``tau2.metrics.turn_taking`` (which itself reads the
  orchestrator's typed sim actions through
  ``tau2.metrics.voice_interaction_metrics`` — one event taxonomy);
- response rate and the three selectivity rates are PROMOTED from the τ-voice
  leaderboard panel (``compute_interaction_metrics_for_ticks``), not
  reimplemented;
- dead air is the ONLY new instrument: maximal spans of the (end-filtered)
  tick timeline where neither party is speaking. A gap that IS a measured
  ordinary response latency is still dead air acoustically and counts once in
  the totals; ``dead_air_excl_response_s`` additionally excludes those
  ordinary response-latency gaps.

Every fact that can be unscoreable is a typed :class:`Fact` carrying an
N/A-with-reason (the #743 ``agent_mute`` pattern generalized), and cell
aggregates report N/A counts per reason. Calls without a tick timeline (text
runs) are carried as fully reasoned-N/A rows, never silently dropped.
"""

import hashlib
from enum import Enum
from pathlib import Path
from statistics import mean, median
from typing import Annotated, Iterable, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from tau2.config import DEFAULT_FEATURE_LONG_SILENCE_SECONDS
from tau2.data_model.message import Tick
from tau2.data_model.simulation import resolve_tick_duration_seconds
from tau2.metrics.run_loading import LoadedCall, RunMeta, iter_loaded_calls
from tau2.metrics.turn_taking import (
    EVA_X_ADAPTATION_VERSION,
    EVA_X_SOURCE_VERSION,
    UNCALIBRATED_SHADOW_LABEL,
    EvaTurnRoute,
    EvaTurnTakingMetric,
    EvaTurnTakingTurn,
    TurnTakingNAReason,
    TurnTakingNotApplicable,
    TurnTakingParams,
    evaluate_turn_taking,
    percentile,
    resolve_turn_taking_params,
)
from tau2.metrics.voice_interaction_metrics import (
    InteractionMetricsConfig,
    VoiceInteractionMetricCounts,
    compute_interaction_metrics_for_ticks,
    filter_end_of_conversation_ticks,
)
from tau2.utils.utils import get_now

#: Version of the fact definitions. Bump on any semantic change.
INTERACTION_FACTS_VERSION = "1.0.0"


class FactNAReason(str, Enum):
    """Typed reason a single fact is N/A rather than a number."""

    #: The call carries no tick timeline (text/half-duplex run): every
    #: perceptual interaction fact is unmeasurable.
    NO_TICK_TIMELINE = "no_tick_timeline"
    #: The timeline exists but produced zero scored real caller turns, so
    #: turn-derived rates and latencies have no denominator.
    NO_SCORED_TURNS = "no_scored_turns"
    #: The fact's own event class never occurred (no ordinary responses, no
    #: incursions, no injected tics, ...): an empty denominator, never a fail.
    NO_EVENTS = "no_events"


class Fact(BaseModel):
    """One deterministic fact: a value, or a typed N/A with reason."""

    value: Optional[float] = Field(
        default=None,
        description="The measured value; None exactly when na_reason is set.",
    )
    na_reason: Optional[FactNAReason] = Field(
        default=None,
        description="Typed reason the fact is unmeasurable on this call; "
        "set exactly when value is None.",
    )

    @model_validator(mode="after")
    def _value_xor_reason(self) -> "Fact":
        if (self.value is None) != (self.na_reason is not None):
            raise ValueError(
                "a fact is either a value or a reasoned N/A: value is None "
                "exactly when na_reason is set"
            )
        return self

    @classmethod
    def of(cls, value: Optional[float], *, na: FactNAReason) -> "Fact":
        """A value fact, or the given reasoned N/A when the value is None."""
        if value is None:
            return cls(na_reason=na)
        return cls(value=float(value))


class InteractionFacts(BaseModel):
    """The per-call deterministic fact table (every column a typed Fact).

    Latencies are milliseconds on the tick clock; rates and fractions are in
    [0, 1] (except ``agent_interruption_rate``-style route fractions, which
    are fractions of scored turns); dead-air columns are seconds.
    """

    # --- response latency (ordinary turns) ---
    response_latency_mean_ms: Annotated[
        Fact, Field(description="Mean ordinary caller-to-agent response latency.")
    ]
    response_latency_p50_ms: Annotated[
        Fact, Field(description="Median ordinary response latency.")
    ]
    response_latency_p90_ms: Annotated[
        Fact, Field(description="90th-percentile ordinary response latency.")
    ]
    tool_turn_latency_mean_ms: Annotated[
        Fact,
        Field(
            description="Mean ordinary response latency over turns where tool "
            "use preceded the settled response."
        ),
    ]
    plain_turn_latency_mean_ms: Annotated[
        Fact,
        Field(description="Mean ordinary response latency over tool-free turns."),
    ]
    # --- responsiveness ---
    missed_response_rate: Annotated[
        Fact,
        Field(
            description="Fraction of scored real caller turns left unanswered "
            "(the caller had to speak again)."
        ),
    ]
    response_rate: Annotated[
        Fact,
        Field(
            description="Canonical τ-voice panel R_R: fraction of "
            "response-eligible caller turns answered (promoted from "
            "VoiceInteractionMetrics)."
        ),
    ]
    # --- interruption behavior ---
    agent_interruption_rate: Annotated[
        Fact,
        Field(description="Fraction of scored turns containing an agent incursion."),
    ]
    caller_interruption_rate: Annotated[
        Fact,
        Field(description="Fraction of scored turns where the caller interrupted."),
    ]
    incursion_overlap_mean_ms: Annotated[
        Fact,
        Field(
            description="Mean total agent-over-caller overlap per turn with an "
            "incursion."
        ),
    ]
    yield_latency_p50_ms: Annotated[
        Fact, Field(description="Median caller-interruption yield latency.")
    ]
    yield_latency_p90_ms: Annotated[
        Fact, Field(description="90th-percentile yield latency.")
    ]
    post_interrupt_latency_p50_ms: Annotated[
        Fact, Field(description="Median settled-response recovery latency.")
    ]
    post_interrupt_latency_p90_ms: Annotated[
        Fact, Field(description="90th-percentile recovery latency.")
    ]
    # --- turn route mix (fractions of scored turns; sum to 1 when scored) ---
    route_ordinary_fraction: Annotated[
        Fact, Field(description="Fraction of scored turns routed ordinary.")
    ]
    route_agent_interruption_fraction: Annotated[
        Fact, Field(description="Fraction routed agent-interruption.")
    ]
    route_user_interruption_fraction: Annotated[
        Fact, Field(description="Fraction routed caller-interruption.")
    ]
    route_dual_interruption_fraction: Annotated[
        Fact, Field(description="Fraction routed dual-interruption.")
    ]
    route_missed_response_fraction: Annotated[
        Fact, Field(description="Fraction routed missed-response.")
    ]
    # --- selectivity (promoted τ-voice panel values, not reimplemented) ---
    selectivity_backchannel: Annotated[
        Fact, Field(description="Panel S_BC: backchannels correctly talked through.")
    ]
    selectivity_vocal_tic: Annotated[
        Fact, Field(description="Panel S_VT: vocal tics correctly ignored.")
    ]
    selectivity_non_directed: Annotated[
        Fact,
        Field(description="Panel S_ND: non-directed speech correctly ignored."),
    ]
    # --- dead air (the one NEW instrument) ---
    dead_air_total_s: Annotated[
        Fact,
        Field(
            description="Total seconds where neither party is speaking on the "
            "end-filtered timeline. Gaps that are measured response latencies "
            "still count once here (they are acoustically dead air)."
        ),
    ]
    dead_air_fraction: Annotated[
        Fact, Field(description="dead_air_total_s over the call duration.")
    ]
    dead_air_max_gap_s: Annotated[
        Fact, Field(description="Longest single neither-speaking gap, seconds.")
    ]
    dead_air_long_gap_count: Annotated[
        Fact,
        Field(
            description="Gaps strictly exceeding the long-silence threshold "
            "(config.long_gap_threshold_s; default 3.0s)."
        ),
    ]
    dead_air_excl_response_s: Annotated[
        Fact,
        Field(
            description="Dead-air seconds excluding spans that ARE measured "
            "ordinary response-latency gaps — silence no pending response "
            "accounts for."
        ),
    ]


class ShadowTurnTakingScore(BaseModel):
    """The demoted EVA turn-taking composite: stamped, never headlined."""

    label: Literal["uncalibrated-shadow"] = Field(
        default=UNCALIBRATED_SHADOW_LABEL,
        description="Fixed shadow marker: computed for comparability only.",
    )
    score: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None,
        description="Mean per-turn EVA score; None exactly when not_applicable is set.",
    )
    gated_score: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None,
        description="Zero when any turn scored a hard zero, else the mean.",
    )
    passed: Optional[bool] = Field(
        default=None, description="Whether the score clears the EVA threshold."
    )
    threshold: float = Field(description="Applied EVA pass threshold.")
    not_applicable: Optional[TurnTakingNotApplicable] = Field(
        default=None,
        description="Typed N/A reason (no scoreable turns / agent mute); set "
        "exactly when score is None.",
    )
    params: TurnTakingParams = Field(
        description="Resolved curve constants (EVA defaults or pack override)."
    )
    adaptation_version: str = Field(
        default=EVA_X_ADAPTATION_VERSION,
        description="Version of the τ EVA adaptation that produced the score.",
    )
    source_version: str = Field(
        default=EVA_X_SOURCE_VERSION, description="Upstream EVA contract version."
    )

    @model_validator(mode="after")
    def _na_state_is_consistent(self) -> "ShadowTurnTakingScore":
        if (self.score is None) != (self.not_applicable is not None):
            raise ValueError("shadow score N/A must carry a typed reason")
        return self

    @classmethod
    def from_metric(cls, metric: EvaTurnTakingMetric) -> "ShadowTurnTakingScore":
        return cls(
            score=metric.score,
            gated_score=metric.gated_score,
            passed=metric.passed,
            threshold=metric.threshold,
            not_applicable=metric.not_applicable,
            params=metric.params,
        )


class InteractionShadowScores(BaseModel):
    """The per-call shadow section: structurally separate from the facts."""

    label: Literal["uncalibrated-shadow"] = Field(
        default=UNCALIBRATED_SHADOW_LABEL,
        description="Fixed shadow marker for the whole section.",
    )
    turn_taking: ShadowTurnTakingScore = Field(
        description="Deterministic EVA turn-taking composite. The full EVA-X "
        "conjunction and the τ quality composite are LLM-dependent and live "
        "in the tau2 judges legacy conversation artifact's shadow section."
    )


class InteractionCallRecord(BaseModel):
    """One call: identity + headline facts + per-turn evidence + shadow."""

    results_path: Annotated[str, Field(description="Path of the run's results.json.")]
    experiment_label: Annotated[str, Field(description="Source-run dir name.")]
    sim_id: Annotated[str, Field(description="Simulation id (join key).")]
    task_id: Annotated[str, Field(description="Task id (join key).")]
    trial: Annotated[int, Field(description="Trial index.")]
    language: Annotated[str, Field(description="ISO 639-1 language of the call.")]
    domain: Annotated[str, Field(description="Domain of the run.")]
    modality: Annotated[str, Field(description="'voice' or 'text' (run-level).")]
    provider: Annotated[
        str, Field(description="Audio-native provider ('' on text runs).")
    ]
    agent_model: Annotated[str, Field(description="Agent model of the run.")]
    reasoning_effort: Annotated[
        Optional[str],
        Field(description="Recorded reasoning effort of the run, if any."),
    ] = None
    reward: Annotated[
        Optional[float], Field(description="Recorded task reward, if any.")
    ] = None
    termination_reason: Annotated[str, Field(description="How the sim terminated.")] = (
        ""
    )
    facts: Annotated[
        InteractionFacts, Field(description="The headline per-call fact table.")
    ]
    scored_turn_count: Annotated[
        int, Field(ge=0, description="Real caller turns behind the turn facts.")
    ] = 0
    event_counts: Annotated[
        Optional[VoiceInteractionMetricCounts],
        Field(
            description="Canonical panel denominators backing the promoted "
            "rates; None when the call has no tick timeline."
        ),
    ] = None
    turns: Annotated[
        list[EvaTurnTakingTurn],
        Field(
            default_factory=list,
            description="Per-turn routes and raw measurements (the fact "
            "evidence); empty when include_turns is off or no timeline exists.",
        ),
    ]
    shadow_scores: Annotated[
        InteractionShadowScores,
        Field(description="Demoted composite scores; never headlined."),
    ]


class FactCellSummary(BaseModel):
    """Mean/median/coverage of one fact over one cell, with N/A accounting."""

    mean: Annotated[Optional[float], Field(description="Mean over valued calls.")]
    median: Annotated[Optional[float], Field(description="Median over valued calls.")]
    n: Annotated[int, Field(ge=0, description="Calls with a value.")]
    n_na: Annotated[int, Field(ge=0, description="Calls with a reasoned N/A.")]
    na_reasons: Annotated[
        dict[FactNAReason, int],
        Field(
            default_factory=dict,
            description="N/A counts per typed reason (absent reasons omitted).",
        ),
    ]


class ShadowCellSummary(BaseModel):
    """Shadow-score means for one cell — separate from the fact columns."""

    label: Literal["uncalibrated-shadow"] = Field(
        default=UNCALIBRATED_SHADOW_LABEL,
        description="Fixed shadow marker for the whole block.",
    )
    turn_taking_score_mean: Annotated[
        Optional[float],
        Field(description="Mean shadow turn-taking score over scored calls."),
    ]
    turn_taking_gated_score_mean: Annotated[
        Optional[float], Field(description="Mean shadow gated score.")
    ]
    turn_taking_pass_rate: Annotated[
        Optional[float],
        Field(description="Fraction of scored calls clearing the EVA threshold."),
    ]
    n_scored: Annotated[int, Field(ge=0, description="Calls with a shadow score.")]
    n_na: Annotated[int, Field(ge=0, description="Calls whose shadow score is N/A.")]
    na_reasons: Annotated[
        dict[TurnTakingNAReason, int],
        Field(
            default_factory=dict,
            description="Shadow N/A counts per typed reason.",
        ),
    ]


class InteractionCellAggregate(BaseModel):
    """One (language x domain x provider x reasoning-effort x modality) cell."""

    language: str
    domain: str
    modality: str
    provider: str
    reasoning_effort: Annotated[
        Optional[str],
        Field(description="Recorded reasoning effort; None when unrecorded."),
    ] = None
    n_calls: Annotated[int, Field(description="Calls aggregated into the cell.")]
    facts: Annotated[
        dict[str, FactCellSummary],
        Field(
            description="Per-fact summaries, keyed by InteractionFacts field "
            "name (the key set is the model's field set)."
        ),
    ]
    shadow_scores: Annotated[
        ShadowCellSummary,
        Field(description="Shadow-score means; structurally separate from facts."),
    ]

    @model_validator(mode="after")
    def _keys_are_fact_fields(self) -> "InteractionCellAggregate":
        unknown = set(self.facts) - set(InteractionFacts.model_fields)
        if unknown:
            raise ValueError(
                f"cell facts carry keys that are not InteractionFacts fields: "
                f"{sorted(unknown)}"
            )
        return self


class InteractionFactsConfig(BaseModel):
    """Extraction scope and knobs, recorded in the artifact's provenance."""

    langs: Annotated[
        Optional[list[str]],
        Field(description="Restrict to these ISO 639-1 codes (None = all)."),
    ] = None
    domain: Annotated[
        Optional[str], Field(description="Restrict to one domain (None = all).")
    ] = None
    max_sims: Annotated[
        Optional[int], Field(description="Cap on extracted sims (dry runs).")
    ] = None
    long_gap_threshold_s: Annotated[
        float,
        Field(
            gt=0,
            description="Dead-air gaps strictly longer than this count as "
            "long gaps (default: the shared long-silence constant, 3.0s).",
        ),
    ] = DEFAULT_FEATURE_LONG_SILENCE_SECONDS
    include_turns: Annotated[
        bool,
        Field(
            description="Keep per-turn evidence on each record (drop for very "
            "large pools; facts and cells are unaffected)."
        ),
    ] = True

    @field_validator("langs")
    @classmethod
    def _fold_langs(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return None if value is None else [item.strip().lower() for item in value]


class InteractionSourceRun(BaseModel):
    """Per-run provenance: run identity plus how many calls it contributed."""

    meta: RunMeta
    n_calls: Annotated[int, Field(description="Calls extracted from this run.")]


class InteractionFactsArtifact(BaseModel):
    """The provenance-bearing facts-first interaction artifact."""

    schema_version: int = 1
    instrument_version: Annotated[
        str, Field(description="INTERACTION_FACTS_VERSION at build time.")
    ] = INTERACTION_FACTS_VERSION
    artifact_id: Annotated[
        str,
        Field(
            description="Content-derived id: sha256(version, config, records) "
            "truncated to 12 hex — identical inputs reproduce it."
        ),
    ]
    created_at: Annotated[str, Field(description="Build wall-clock time.")]
    git_sha: Annotated[str, Field(description="Repo HEAD at build time.")]
    config: InteractionFactsConfig
    source_runs: list[InteractionSourceRun]
    records: Annotated[
        list[InteractionCallRecord], Field(description="One record per call.")
    ]
    cells: Annotated[
        list[InteractionCellAggregate],
        Field(
            description="In-verb cell aggregates (language x domain x "
            "provider x reasoning-effort x modality): facts + N/A counts + "
            "shadow means."
        ),
    ]


# ---------------------------------------------------------------------------
# Dead air
# ---------------------------------------------------------------------------


class DeadAirFacts(BaseModel):
    """The five dead-air facts for one call."""

    total_s: Fact
    fraction: Fact
    max_gap_s: Fact
    long_gap_count: Fact
    excl_response_s: Fact

    @classmethod
    def na(cls, reason: FactNAReason) -> "DeadAirFacts":
        blank = Fact(na_reason=reason)
        return cls(
            total_s=blank,
            fraction=blank,
            max_gap_s=blank,
            long_gap_count=blank,
            excl_response_s=blank,
        )


def _speaking(chunk) -> bool:
    return chunk is not None and bool(getattr(chunk, "contains_speech", False))


def _silence_runs(ticks: list[Tick]) -> list[tuple[int, int]]:
    """Maximal [start, end) tick spans where neither party is speaking."""
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for index, tick in enumerate(ticks):
        silent = not _speaking(tick.user_chunk) and not _speaking(tick.agent_chunk)
        if silent and start is None:
            start = index
        elif not silent and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(ticks)))
    return runs


def _interval_overlap_s(
    run_s: tuple[float, float], intervals_s: list[tuple[float, float]]
) -> float:
    """Total overlap of one span with a set of (possibly touching) intervals."""
    total = 0.0
    for low, high in intervals_s:
        total += max(0.0, min(run_s[1], high) - max(run_s[0], low))
    return total


def compute_dead_air(
    ticks: list[Tick],
    *,
    long_gap_threshold_s: float,
    ordinary_response_gaps_s: Optional[list[tuple[float, float]]] = None,
) -> DeadAirFacts:
    """Dead-air facts from the typed tick timeline.

    ``ordinary_response_gaps_s`` are the [caller-turn end, settled response
    onset) spans of ordinary-routed turns, in seconds on the same clock; a
    dead-air gap counts once in the totals even when it IS such a measured
    response latency, and ``excl_response_s`` reports the remainder after
    removing those spans. A call with no usable timeline is a reasoned N/A.
    """
    if not ticks:
        return DeadAirFacts.na(FactNAReason.NO_TICK_TIMELINE)
    filtered = filter_end_of_conversation_ticks(ticks)
    if not filtered:
        return DeadAirFacts.na(FactNAReason.NO_TICK_TIMELINE)
    dur = resolve_tick_duration_seconds(filtered, None)
    duration_s = len(filtered) * dur
    runs = _silence_runs(filtered)
    gap_lengths_s = [(end - start) * dur for start, end in runs]
    total_s = sum(gap_lengths_s)
    response_gaps = ordinary_response_gaps_s or []
    excl_response_s = sum(
        (end - start) * dur
        - _interval_overlap_s((start * dur, end * dur), response_gaps)
        for start, end in runs
    )
    return DeadAirFacts(
        total_s=Fact(value=total_s),
        fraction=Fact(value=total_s / duration_s),
        max_gap_s=Fact(value=max(gap_lengths_s, default=0.0)),
        long_gap_count=Fact(
            value=float(
                sum(1 for length in gap_lengths_s if length > long_gap_threshold_s)
            )
        ),
        excl_response_s=Fact(value=max(0.0, excl_response_s)),
    )


def ordinary_response_gaps_s(
    turns: list[EvaTurnTakingTurn],
) -> list[tuple[float, float]]:
    """[caller-turn end, settled response onset) spans of ordinary turns."""
    gaps: list[tuple[float, float]] = []
    for turn in turns:
        if (
            turn.route is EvaTurnRoute.ORDINARY
            and turn.latency_ms is not None
            and turn.latency_ms > 0
        ):
            start = turn.user_end_ms / 1000.0
            gaps.append((start, start + turn.latency_ms / 1000.0))
    return gaps


# ---------------------------------------------------------------------------
# Per-call extraction
# ---------------------------------------------------------------------------


def _all_na_facts(reason: FactNAReason) -> InteractionFacts:
    blank = Fact(na_reason=reason)
    return InteractionFacts(**{name: blank for name in InteractionFacts.model_fields})


def _route_fraction(metric: EvaTurnTakingMetric, route: EvaTurnRoute) -> Fact:
    count = metric.stats.turn_count
    if count == 0:
        return Fact(na_reason=FactNAReason.NO_SCORED_TURNS)
    matching = next(
        (row.turn_count for row in metric.stats.routes if row.route is route), 0
    )
    return Fact(value=matching / count)


def extract_interaction_facts(
    call: LoadedCall, config: Optional[InteractionFactsConfig] = None
) -> InteractionCallRecord:
    """All headline facts plus the shadow turn-taking score for one call."""
    cfg = config or InteractionFactsConfig()
    sim = call.sim
    params = resolve_turn_taking_params(call.language)

    if not sim.ticks:
        return _record(
            call,
            cfg,
            facts=_all_na_facts(FactNAReason.NO_TICK_TIMELINE),
            metric=evaluate_turn_taking([], params=params),
            event_counts=None,
            turns=[],
        )

    ticks = sim.ticks
    dur = resolve_tick_duration_seconds(ticks, None)
    metric = evaluate_turn_taking(ticks, params=params)
    panel = compute_interaction_metrics_for_ticks(
        ticks, InteractionMetricsConfig(tick_duration_sec=dur)
    )
    stats = metric.stats
    turn_na = (
        FactNAReason.NO_SCORED_TURNS
        if stats.turn_count == 0
        else FactNAReason.NO_EVENTS
    )

    ordinary = [turn for turn in metric.turns if turn.route is EvaTurnRoute.ORDINARY]
    tool_latencies = [
        turn.latency_ms
        for turn in ordinary
        if turn.uses_tool and turn.latency_ms is not None
    ]
    plain_latencies = [
        turn.latency_ms
        for turn in ordinary
        if not turn.uses_tool and turn.latency_ms is not None
    ]
    overlaps = [
        turn.agent_overlap_ms
        for turn in metric.turns
        if turn.agent_overlap_ms is not None
    ]
    yield_latencies = [
        turn.yield_latency_ms
        for turn in metric.turns
        if turn.yield_latency_ms is not None
    ]
    recovery_latencies = [
        turn.post_interrupt_latency_ms
        for turn in metric.turns
        if turn.post_interrupt_latency_ms is not None
    ]

    counts = panel.counts
    dead_air = compute_dead_air(
        ticks,
        long_gap_threshold_s=cfg.long_gap_threshold_s,
        ordinary_response_gaps_s=ordinary_response_gaps_s(metric.turns),
    )

    facts = InteractionFacts(
        response_latency_mean_ms=Fact.of(stats.response_latency_mean_ms, na=turn_na),
        response_latency_p50_ms=Fact.of(stats.response_latency_p50_ms, na=turn_na),
        response_latency_p90_ms=Fact.of(stats.response_latency_p90_ms, na=turn_na),
        tool_turn_latency_mean_ms=Fact.of(
            mean(tool_latencies) if tool_latencies else None, na=turn_na
        ),
        plain_turn_latency_mean_ms=Fact.of(
            mean(plain_latencies) if plain_latencies else None, na=turn_na
        ),
        missed_response_rate=Fact.of(
            stats.missed_response_rate, na=FactNAReason.NO_SCORED_TURNS
        ),
        response_rate=Fact.of(panel.response_rate, na=FactNAReason.NO_EVENTS),
        agent_interruption_rate=Fact.of(
            stats.agent_interruption_rate, na=FactNAReason.NO_SCORED_TURNS
        ),
        caller_interruption_rate=Fact.of(
            stats.user_interruption_rate, na=FactNAReason.NO_SCORED_TURNS
        ),
        incursion_overlap_mean_ms=Fact.of(
            mean(overlaps) if overlaps else None, na=turn_na
        ),
        yield_latency_p50_ms=Fact.of(percentile(yield_latencies, 0.5), na=turn_na),
        yield_latency_p90_ms=Fact.of(percentile(yield_latencies, 0.9), na=turn_na),
        post_interrupt_latency_p50_ms=Fact.of(
            percentile(recovery_latencies, 0.5), na=turn_na
        ),
        post_interrupt_latency_p90_ms=Fact.of(
            percentile(recovery_latencies, 0.9), na=turn_na
        ),
        route_ordinary_fraction=_route_fraction(metric, EvaTurnRoute.ORDINARY),
        route_agent_interruption_fraction=_route_fraction(
            metric, EvaTurnRoute.AGENT_INTERRUPTION
        ),
        route_user_interruption_fraction=_route_fraction(
            metric, EvaTurnRoute.USER_INTERRUPTION
        ),
        route_dual_interruption_fraction=_route_fraction(
            metric, EvaTurnRoute.DUAL_INTERRUPTION
        ),
        route_missed_response_fraction=_route_fraction(
            metric, EvaTurnRoute.MISSED_RESPONSE
        ),
        selectivity_backchannel=Fact.of(
            panel.selectivity_backchannel, na=FactNAReason.NO_EVENTS
        ),
        selectivity_vocal_tic=Fact.of(
            panel.selectivity_vocal_tic, na=FactNAReason.NO_EVENTS
        ),
        selectivity_non_directed=Fact.of(
            panel.selectivity_non_directed, na=FactNAReason.NO_EVENTS
        ),
        dead_air_total_s=dead_air.total_s,
        dead_air_fraction=dead_air.fraction,
        dead_air_max_gap_s=dead_air.max_gap_s,
        dead_air_long_gap_count=dead_air.long_gap_count,
        dead_air_excl_response_s=dead_air.excl_response_s,
    )
    return _record(
        call, cfg, facts=facts, metric=metric, event_counts=counts, turns=metric.turns
    )


def _record(
    call: LoadedCall,
    cfg: InteractionFactsConfig,
    *,
    facts: InteractionFacts,
    metric: EvaTurnTakingMetric,
    event_counts: Optional[VoiceInteractionMetricCounts],
    turns: list[EvaTurnTakingTurn],
) -> InteractionCallRecord:
    meta, sim = call.meta, call.sim
    return InteractionCallRecord(
        results_path=meta.results_path,
        experiment_label=meta.experiment_label,
        sim_id=str(sim.id),
        task_id=str(sim.task_id),
        trial=sim.trial or 0,
        language=call.language,
        domain=meta.domain,
        modality=meta.modality,
        provider=meta.provider,
        agent_model=meta.agent_model,
        reasoning_effort=meta.reasoning_effort,
        reward=(sim.reward_info.reward if sim.reward_info else None),
        termination_reason=(
            sim.termination_reason.value if sim.termination_reason else ""
        ),
        facts=facts,
        scored_turn_count=metric.stats.turn_count,
        event_counts=event_counts,
        turns=(turns if cfg.include_turns else []),
        shadow_scores=InteractionShadowScores(
            turn_taking=ShadowTurnTakingScore.from_metric(metric)
        ),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_cells(
    records: list[InteractionCallRecord],
) -> list[InteractionCellAggregate]:
    """Per (language x domain x provider x reasoning-effort x modality) cells."""
    by_cell: dict[tuple[str, str, str, str, str], list[InteractionCallRecord]] = {}
    for record in records:
        key = (
            record.language,
            record.domain,
            record.modality,
            record.provider,
            record.reasoning_effort or "",
        )
        by_cell.setdefault(key, []).append(record)
    cells: list[InteractionCellAggregate] = []
    for key, cell_records in sorted(by_cell.items()):
        language, domain, modality, provider, effort = key
        facts: dict[str, FactCellSummary] = {}
        for field_name in InteractionFacts.model_fields:
            values: list[float] = []
            na_reasons: dict[FactNAReason, int] = {}
            for record in cell_records:
                fact: Fact = getattr(record.facts, field_name)
                if fact.value is not None:
                    values.append(fact.value)
                else:
                    na_reasons[fact.na_reason] = na_reasons.get(fact.na_reason, 0) + 1
            facts[field_name] = FactCellSummary(
                mean=(mean(values) if values else None),
                median=(median(values) if values else None),
                n=len(values),
                n_na=sum(na_reasons.values()),
                na_reasons=na_reasons,
            )
        shadow_values = [record.shadow_scores.turn_taking for record in cell_records]
        scored = [shadow for shadow in shadow_values if shadow.score is not None]
        shadow_na: dict[TurnTakingNAReason, int] = {}
        for shadow in shadow_values:
            if shadow.not_applicable is not None:
                reason = shadow.not_applicable.reason
                shadow_na[reason] = shadow_na.get(reason, 0) + 1
        cells.append(
            InteractionCellAggregate(
                language=language,
                domain=domain,
                modality=modality,
                provider=provider,
                reasoning_effort=(effort or None),
                n_calls=len(cell_records),
                facts=facts,
                shadow_scores=ShadowCellSummary(
                    turn_taking_score_mean=(
                        mean(shadow.score for shadow in scored) if scored else None
                    ),
                    turn_taking_gated_score_mean=(
                        mean(shadow.gated_score for shadow in scored)
                        if scored
                        else None
                    ),
                    turn_taking_pass_rate=(
                        mean(float(shadow.passed) for shadow in scored)
                        if scored
                        else None
                    ),
                    n_scored=len(scored),
                    n_na=sum(shadow_na.values()),
                    na_reasons=shadow_na,
                ),
            )
        )
    return cells


def _derive_artifact_id(
    config: InteractionFactsConfig, records: list[InteractionCallRecord]
) -> str:
    digest = hashlib.sha256()
    digest.update(INTERACTION_FACTS_VERSION.encode())
    digest.update(b"\0")
    digest.update(config.model_dump_json().encode())
    for record in records:
        digest.update(b"\0")
        digest.update(record.model_dump_json().encode())
    return digest.hexdigest()[:12]


def build_interaction_facts(
    paths: Iterable[Path | str],
    config: Optional[InteractionFactsConfig] = None,
) -> InteractionFactsArtifact:
    """Extract every call under ``paths`` and assemble the artifact."""
    from tau2.annotation.artifacts import git_sha

    cfg = config or InteractionFactsConfig()
    records: list[InteractionCallRecord] = []
    calls_per_run: dict[str, int] = {}
    metas: dict[str, RunMeta] = {}
    for call in iter_loaded_calls(
        paths, langs=cfg.langs, domain=cfg.domain, max_sims=cfg.max_sims
    ):
        records.append(extract_interaction_facts(call, cfg))
        metas[call.meta.results_path] = call.meta
        calls_per_run[call.meta.results_path] = (
            calls_per_run.get(call.meta.results_path, 0) + 1
        )
    return InteractionFactsArtifact(
        artifact_id=_derive_artifact_id(cfg, records),
        created_at=get_now(),
        git_sha=git_sha(),
        config=cfg,
        source_runs=[
            InteractionSourceRun(meta=metas[path], n_calls=count)
            for path, count in sorted(calls_per_run.items())
        ],
        records=records,
        cells=aggregate_cells(records),
    )
