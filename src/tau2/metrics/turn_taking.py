# Copyright Sierra
"""τ tick-stream adaptation of EVA turn-taking v0.2 (deterministic).

Deterministic computation lives under ``tau2.metrics``: this module owns the
typed turn-taking event routing and the EVA-adapted scoring formulas. The
facts-first instrument (``tau2 metrics interaction-facts``) reads the routed
per-turn measurements as its headline fact table; the score itself is a
permanent SHADOW column — computed and stamped for comparability with
EVA-Bench-style pipelines, labeled uncalibrated, never ranked or headlined
(binding decision, 2026-08-18).

The score curves and interruption formulas follow ServiceNow EVA (Tables
16-18). Event sourcing is the orchestrator's TYPED sim actions and injected
effects, read exclusively through the canonical extractor layer in
``tau2.metrics.voice_interaction_metrics`` — there is ONE event taxonomy:

- caller/agent segments (``extract_user_segments`` / ``extract_agent_segments``)
  carry the typed ``turn_taking_action`` (``backchannel`` etc.) and the
  injected-effect spans, so a sim backchannel or a pure vocal-tic /
  non-directed utterance is never scored as a real caller turn — while a real
  turn that merely CONTAINS one injected tic tick still is;
- ordinary and missed responses come from ``extract_turn_transitions`` — the
  same events behind the leaderboard response rate/latency;
- agent incursions and caller-interruption yields come from
  ``extract_interruption_events`` — the same events behind the I_A rate and
  yield rate/latency, which also route mixed tic/non-directed overlap events
  into the selectivity domain instead of the yield formula.

Every constant lives on :class:`TurnTakingParams` (EVA defaults, provisional
against the Stivers et al. 2009 cross-language turn-gap norms, pending
per-language fits) and is pack-overridable per language via
``LanguagePack.turn_taking``.

N/A is always typed (``EvaTurnTakingMetric.not_applicable``): zero scoreable
turns is one reason; the other is the agent-mute infrastructure failure (the
known ko×OpenAI realtime permanently-mute-mid-call population). A dead audio
channel makes turn-taking BEHAVIOR unobservable, so such a call is a reasoned
N/A — excluded from aggregate means because ``score`` is None — while its raw
per-turn routes and stats stay on the record for inspection. Detection is
evidence-grounded: terminal agent silence (``mute_min_trailing_turns`` real
caller turns starting at-or-after the agent's final audio emission) or
missed-response dominance (``mute_missed_fraction`` of scored turns missed,
backed by at least ``mute_min_missed_responses`` misses).
"""

from enum import Enum
from statistics import mean
from typing import Annotated, Optional

from pydantic import BaseModel, Field, model_validator

from tau2.data_model.message import Tick
from tau2.data_model.simulation import resolve_tick_duration_seconds
from tau2.metrics.voice_interaction_metrics import (
    AgentSpeechSegment,
    InterruptionEvent,
    TurnTransitionEvent,
    UserSpeechSegment,
    extract_agent_segments,
    extract_interruption_events,
    extract_turn_transitions,
    extract_user_segments,
    filter_end_of_conversation_ticks,
)

# v2: turn-taking events re-sourced from the orchestrator's typed sim actions
# via the canonical extractors (segments, transitions, interruption events —
# consistent with the VoiceInteractionMetrics panel), a quantization-aware
# latency band, real N/A states instead of zero-evidence fails,
# pack-overridable curve params, the gated variant, and per-route sub-scores.
# v3: every turn-taking N/A carries a typed reason, and the
# permanently-mute-mid-call infrastructure failure (the known ko×OpenAI
# realtime population) is detected in-verb — terminal agent silence from tick
# audio evidence, or missed-response dominance — and emitted as a reasoned
# N/A (score/gated_score/passed all None) with the raw turns and stats
# retained, instead of contaminating cell means as near-zero scores.
EVA_X_ADAPTATION_VERSION = "eva-x-tau-v3"
EVA_X_SOURCE_VERSION = "eva-turn-taking-v0.2"
EVA_X_PASS_THRESHOLD = 0.8

#: Fixed label stamped on every demoted composite score: computed for
#: comparability, uncalibrated against human preference, never headlined.
UNCALIBRATED_SHADOW_LABEL = "uncalibrated-shadow"

#: Provenance tag for stock EVA v0.2 turn-taking constants.
EVA_DEFAULT_PARAMS_ORIGIN = "eva-v0.2-defaults"

#: End-of-call marker the sim emits (see ``tau2.user.user_simulator_base.STOP``,
#: not imported to keep this module free of simulator dependencies). A caller
#: segment carrying it ended the call; no response was owed.
_STOP_MARKER = "###STOP###"

#: Injected-effect types that are not caller speech directed at the agent.
_NON_DIRECTED_EFFECTS = frozenset({"vocal_tic", "non_directed_speech"})


class TurnTakingParams(BaseModel):
    """Every constant of the EVA turn-taking formulas, in milliseconds.

    Defaults are the published EVA v0.2 values (compatible with the
    cross-language turn-gap norms of Stivers et al. 2009, whose mode sits
    near +200ms with language offsets under ~500ms) and are PROVISIONAL
    pending per-language fits from caller-tolerance annotations. Resolve
    per-language pack overrides with :func:`resolve_turn_taking_params`;
    the resolved instance rides on the emitted metric so a re-fit is visible
    on the results it produced.
    """

    hard_early_ms: Annotated[
        float,
        Field(
            description="Response onset at or before this (negative) latency "
            "scores 0: premature interruption."
        ),
    ] = -500.0
    sweet_low_ms: Annotated[
        float,
        Field(description="Lower bound of the optimal window (score reaches 1)."),
    ] = 500.0
    sweet_high_ms: Annotated[
        float,
        Field(description="Upper bound of the optimal window without tool calls."),
    ] = 2000.0
    hard_late_ms: Annotated[
        float,
        Field(description="Latency at which a non-tool response scores 0."),
    ] = 3500.0
    tool_sweet_high_ms: Annotated[
        float,
        Field(description="Upper bound of the optimal window for tool-call turns."),
    ] = 3000.0
    tool_hard_late_ms: Annotated[
        float,
        Field(description="Latency at which a tool-call response scores 0."),
    ] = 5000.0
    agent_interrupt_max_score: Annotated[
        float,
        Field(
            description="Cap M on every agent-interruption sub-score: an "
            "interruption is never cost-free."
        ),
    ] = 0.5
    agent_overlap_hard_ms: Annotated[
        float,
        Field(
            description="Total agent-over-caller overlap at which the overlap "
            "sub-score reaches 0."
        ),
    ] = 2000.0
    agent_interrupt_count_hard: Annotated[
        int,
        Field(
            description="Distinct agent incursions at which the count "
            "sub-score reaches 0.",
            gt=1,
        ),
    ] = 3
    user_yield_hard_ms: Annotated[
        float,
        Field(
            description="Yield latency at which the caller-interruption score "
            "reaches 0."
        ),
    ] = 2000.0
    # τ mute-detector extension (not an EVA formula constant): thresholds for
    # routing "the agent's audio channel died" to a reasoned N/A instead of a
    # scored fail. Defaults are grounded in the ko×OpenAI realtime
    # permanently-mute-mid-call population from the 2026-08-18 first-tables
    # corpus (400 ko×openai calls inspected at tick level).
    mute_min_trailing_turns: Annotated[
        int,
        Field(
            description="Real caller turns starting at-or-after the agent's "
            "FINAL audio emission that establish terminal agent silence "
            "(infrastructure mute); below this, a trailing missed run is "
            "ordinary bad turn-taking.",
            gt=0,
        ),
    ] = 3
    mute_missed_fraction: Annotated[
        float,
        Field(
            description="Missed-response fraction over scored turns at which "
            "the call is treated as (intermittently) mute even when the agent "
            "emitted audio after the last miss.",
            gt=0,
            le=1,
        ),
    ] = 0.5
    mute_min_missed_responses: Annotated[
        int,
        Field(
            description="Minimum missed responses backing the fraction "
            "trigger: one isolated miss in a short call is bad turn-taking "
            "evidence, never infrastructure mute.",
            gt=0,
        ),
    ] = 2
    origin: Annotated[
        str,
        Field(
            description="Where these constants came from: the EVA defaults or "
            "a language pack override (``pack:<language>``)."
        ),
    ] = EVA_DEFAULT_PARAMS_ORIGIN

    @model_validator(mode="after")
    def _validate_curve_order(self) -> "TurnTakingParams":
        for sweet_high, hard_late, label in (
            (self.sweet_high_ms, self.hard_late_ms, "standard"),
            (self.tool_sweet_high_ms, self.tool_hard_late_ms, "tool-call"),
        ):
            if not self.hard_early_ms < self.sweet_low_ms <= sweet_high < hard_late:
                raise ValueError(
                    f"{label} latency breakpoints must satisfy "
                    "hard_early < sweet_low <= sweet_high < hard_late"
                )
        if self.agent_overlap_hard_ms <= 0 or self.user_yield_hard_ms <= 0:
            raise ValueError("overlap and yield hard bounds must be positive")
        if not 0 < self.agent_interrupt_max_score <= 1:
            raise ValueError("agent_interrupt_max_score must be in (0, 1]")
        return self


class EvaTurnRoute(str, Enum):
    """Event route used to select the EVA turn-taking formula."""

    ORDINARY = "ordinary"
    AGENT_INTERRUPTION = "agent_interruption"
    USER_INTERRUPTION = "user_interruption"
    DUAL_INTERRUPTION = "dual_interruption"
    MISSED_RESPONSE = "missed_response"


class TurnTakingNAReason(str, Enum):
    """Why a call's turn-taking score is N/A rather than a number."""

    #: Zero scoreable real caller turns: no evidence is never a fail.
    NO_SCOREABLE_TURNS = "no_scoreable_turns"
    #: The agent's audio channel died (infrastructure mute): the caller kept
    #: taking real turns that the agent never answered — terminal silence
    #: after the agent's final audio emission, or missed-response dominance.
    #: Turn-taking BEHAVIOR is unobservable on a dead channel, so the call is
    #: N/A, not a zero.
    AGENT_MUTE = "agent_mute"


class TurnTakingNotApplicable(BaseModel):
    """Typed reason and raw evidence behind a turn-taking N/A.

    Present exactly when the metric's ``score`` is None. The evidence fields
    keep the call inspectable: a reasoned-N/A call still reports what the
    detector saw, and the full per-turn routes and stats stay on the metric.
    """

    reason: TurnTakingNAReason = Field(description="Typed N/A cause.")
    trailing_unanswered_turns: Annotated[
        int,
        Field(
            ge=0,
            description="Real caller turns starting at-or-after the agent's "
            "final audio emission (every one unanswered by construction).",
        ),
    ] = 0
    last_agent_audio_ms: Optional[Annotated[float, Field(ge=0)]] = Field(
        default=None,
        description="Call-timeline end of the agent's final speech segment; "
        "None when the agent never emitted audio at all.",
    )
    missed_response_count: Annotated[
        int,
        Field(ge=0, description="Scored turns routed as missed responses."),
    ] = 0
    missed_response_fraction: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None,
        description="Missed responses over scored turns; None when the call "
        "produced no scored turns.",
    )
    evidence: str = Field(
        default="", description="Compact human-readable detector evidence."
    )


class EvaTurnTakingTurn(BaseModel):
    """One real caller turn and the event-specific evidence used to score it."""

    turn_id: Annotated[
        int, Field(ge=0, description="Zero-based real caller-turn identifier.")
    ]
    route: EvaTurnRoute = Field(description="Event-specific EVA scoring route.")
    score: Annotated[
        float, Field(ge=0, le=1, description="Event-specific normalized score.")
    ]
    user_start_ms: Annotated[
        float, Field(ge=0, description="Caller-turn start on the call timeline.")
    ]
    user_end_ms: Annotated[
        float, Field(ge=0, description="Caller-turn end on the call timeline.")
    ]
    uses_tool: bool = Field(
        default=False,
        description="Whether tool use occurred before the settled response.",
    )
    latency_ms: Optional[float] = Field(
        default=None, description="Ordinary caller-to-agent response latency."
    )
    agent_overlap_ms: Optional[Annotated[float, Field(ge=0)]] = Field(
        default=None, description="Total agent overlap during the caller turn."
    )
    agent_interrupt_count: Optional[Annotated[int, Field(ge=0)]] = Field(
        default=None, description="Distinct agent speech incursions into the turn."
    )
    post_interrupt_latency_ms: Optional[float] = Field(
        default=None, description="Latency to a settled post-interruption response."
    )
    yield_latency_ms: Optional[Annotated[float, Field(ge=0)]] = Field(
        default=None, description="Time for the agent to yield after caller onset."
    )
    evidence: str = Field(default="", description="Compact event-routing evidence.")


class EvaTurnRouteBreakdown(BaseModel):
    """Per-event-type sub-score: how one routing class scored on this call."""

    route: EvaTurnRoute = Field(description="Event route this row aggregates.")
    turn_count: Annotated[
        int, Field(gt=0, description="Real caller turns scored on this route.")
    ]
    mean_score: Annotated[
        float, Field(ge=0, le=1, description="Mean per-turn score on this route.")
    ]


class EvaTurnTakingStats(BaseModel):
    """Tail-aware diagnostics accompanying the per-turn EVA score."""

    turn_count: Annotated[
        int, Field(ge=0, description="Number of scored real caller turns.")
    ]
    routes: list[EvaTurnRouteBreakdown] = Field(
        default_factory=list,
        description="Per-event-type turn counts and mean scores (routes with "
        "no turns are omitted).",
    )
    missed_response_rate: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None, description="Fraction of real caller turns left unanswered."
    )
    agent_interruption_rate: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None, description="Fraction containing an agent interruption."
    )
    user_interruption_rate: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None, description="Fraction where the caller interrupted the agent."
    )
    response_latency_mean_ms: Optional[float] = Field(
        default=None, description="Mean ordinary response latency."
    )
    response_latency_p50_ms: Optional[float] = Field(
        default=None, description="Median ordinary response latency."
    )
    response_latency_p90_ms: Optional[float] = Field(
        default=None, description="90th-percentile ordinary response latency."
    )
    post_interrupt_latency_mean_ms: Optional[float] = Field(
        default=None, description="Mean settled-response recovery latency."
    )
    post_interrupt_latency_p50_ms: Optional[float] = Field(
        default=None, description="Median settled-response recovery latency."
    )
    post_interrupt_latency_p90_ms: Optional[float] = Field(
        default=None, description="90th-percentile recovery latency."
    )
    yield_latency_mean_ms: Optional[float] = Field(
        default=None, description="Mean caller-interruption yield latency."
    )
    yield_latency_p50_ms: Optional[float] = Field(
        default=None, description="Median caller-interruption yield latency."
    )
    yield_latency_p90_ms: Optional[float] = Field(
        default=None, description="90th-percentile yield latency."
    )


class EvaTurnTakingMetric(BaseModel):
    """Continuous EVA turn-taking score plus per-turn and tail evidence.

    The ``score``/``gated_score``/``passed`` triple is a SHADOW composite;
    the per-turn measurements (``turns``) and the rate/latency diagnostics
    (``stats``) are the fact layer the interaction-facts instrument reports.
    """

    score: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None,
        description="Mean normalized real-turn score; None exactly when "
        "not_applicable is set (no scoreable caller turns, or an agent-mute "
        "call whose turn-taking behavior is unobservable) — reasoned N/A "
        "drops out of aggregate means by construction.",
    )
    gated_score: Optional[Annotated[float, Field(ge=0, le=1)]] = Field(
        default=None,
        description="Shadow gated aggregation: 0 when ANY real turn "
        "scored a hard zero (missed response, failed yield, saturated "
        "interruption), else the mean; None whenever the call is N/A. "
        "Never used for pass/fail.",
    )
    passed: Optional[bool] = Field(
        default=None,
        description="Whether the score clears the EVA threshold; None "
        "whenever the call is N/A.",
    )
    not_applicable: Optional[TurnTakingNotApplicable] = Field(
        default=None,
        description="Typed N/A reason and detector evidence; set exactly "
        "when score is None. An agent-mute N/A still carries the raw turns "
        "and stats so the call remains inspectable.",
    )
    params: TurnTakingParams = Field(
        default_factory=TurnTakingParams,
        description="Resolved curve constants this call was scored with "
        "(EVA defaults or a language-pack override).",
    )
    threshold: float = Field(
        default=EVA_X_PASS_THRESHOLD, description="Applied turn-taking pass threshold."
    )
    turns: list[EvaTurnTakingTurn] = Field(
        description="Per-real-caller-turn routes, measurements, and scores."
    )
    stats: EvaTurnTakingStats = Field(
        description="Call-level rates and latency distribution diagnostics."
    )

    @model_validator(mode="after")
    def _na_state_is_consistent(self) -> "EvaTurnTakingMetric":
        if (self.score is None) != (self.not_applicable is not None):
            raise ValueError(
                "turn-taking N/A must be reasoned: score is None exactly "
                "when not_applicable is set"
            )
        if self.not_applicable is not None and not (
            self.gated_score is None and self.passed is None
        ):
            raise ValueError(
                "an N/A turn-taking metric cannot carry a gated score or a pass verdict"
            )
        return self


def resolve_turn_taking_params(language: Optional[str]) -> TurnTakingParams:
    """EVA defaults, overridden by the language pack's ``turn_taking`` block.

    Unknown language / no pack / no block all resolve to the defaults —
    per-language constants are opt-in calibration, never an error.
    """
    defaults = TurnTakingParams()
    if not language:
        return defaults
    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack(language.lower())
    if pack is None or pack.turn_taking is None:
        return defaults
    overrides = pack.turn_taking.model_dump(exclude_none=True)
    if not overrides:
        return defaults
    return defaults.model_copy(
        update={**overrides, "origin": f"pack:{language.lower()}"}
    )


def eva_latency_score(
    latency_ms: float,
    *,
    uses_tool: bool,
    params: Optional[TurnTakingParams] = None,
    overlap_excluded: bool = False,
) -> float:
    """Score one response latency on EVA's tool-aware piecewise curve.

    ``overlap_excluded`` is τ's explicit quantization handling: canonical
    turn transitions are non-overlapping BY CONSTRUCTION (an onset during
    caller speech routes to the interruption formulas instead), so the
    sub-``sweet_low`` early ramp — EVA's premature-interruption penalty —
    cannot apply to them. Tick quantization maps a genuinely fast ~0-300ms
    response to a measured 0ms gap; with the flag set, any non-negative
    latency below the sweet spot scores 1.0 instead of riding the ramp.
    """
    p = params or TurnTakingParams()
    sweet_high = p.tool_sweet_high_ms if uses_tool else p.sweet_high_ms
    hard_late = p.tool_hard_late_ms if uses_tool else p.hard_late_ms
    if overlap_excluded and 0.0 <= latency_ms <= sweet_high:
        return 1.0
    if latency_ms <= p.hard_early_ms or latency_ms >= hard_late:
        return 0.0
    if latency_ms < p.sweet_low_ms:
        return (latency_ms - p.hard_early_ms) / (p.sweet_low_ms - p.hard_early_ms)
    if latency_ms <= sweet_high:
        return 1.0
    return (hard_late - latency_ms) / (hard_late - sweet_high)


def _effect_only_segment(segment: UserSpeechSegment) -> bool:
    """Whether injected tic/non-directed spans cover the WHOLE segment.

    Only a segment that IS the injected event is routed out of scoring; a
    real caller turn containing one tic tick keeps its floor-taking status.
    """
    if not (segment.has_vocal_tic or segment.has_non_directed_speech):
        return False
    covered: set[int] = set()
    for effect in segment.audio_effects:
        if effect.effect_type in _NON_DIRECTED_EFFECTS:
            covered.update(range(effect.start_tick, effect.end_tick))
    return set(range(segment.start_tick, segment.end_tick)) <= covered


def _is_real_caller_turn(segment: UserSpeechSegment) -> bool:
    """A floor-taking caller turn, from the typed action and effect spans.

    Routed out: typed backchannels (``turn_taking_action.action``), segments
    fully covered by injected vocal-tic / non-directed-speech effects (those
    are selectivity events), and the typed end-of-call marker segment.
    """
    return not (
        segment.is_backchannel
        or _effect_only_segment(segment)
        or _STOP_MARKER in (segment.transcript or "")
    )


def percentile(values: list[float], fraction: float) -> Optional[float]:
    """Linear-interpolated percentile; None on an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _uses_tool(ticks: list[Tick], start_tick: int, response_start_tick: int) -> bool:
    start = max(0, start_tick)
    stop = min(len(ticks), response_start_tick + 1)
    return any(tick.agent_tool_calls for tick in ticks[start:stop])


def _overlap_ms(
    user: UserSpeechSegment, agent: AgentSpeechSegment, tick_ms: float
) -> float:
    overlap_ticks = min(user.end_tick, agent.end_tick) - max(
        user.start_tick, agent.start_tick
    )
    return max(0.0, overlap_ticks * tick_ms)


def _immediate_response(
    user: UserSpeechSegment, agent_segments: list[AgentSpeechSegment]
) -> Optional[AgentSpeechSegment]:
    """The canonical taxonomy's quantization blind spot, made explicit.

    An agent response starting the very tick the caller stopped is the
    fastest measurable response — no tick contains both speakers, so it is
    not an overlap — yet ``extract_turn_transitions`` skips the caller
    segment entirely (``other_speaking_at_end``). Returning it lets the
    scorer route it as an ordinary ~0ms-latency response instead of silently
    dropping (or worse, penalizing) genuinely fast agents.
    """
    for agent in agent_segments:
        if agent.start_tick == user.end_tick:
            return agent
    return None


def _na_metric(
    params: TurnTakingParams,
    not_applicable: TurnTakingNotApplicable,
    *,
    turns: Optional[list[EvaTurnTakingTurn]] = None,
    stats: Optional[EvaTurnTakingStats] = None,
) -> EvaTurnTakingMetric:
    """A reasoned N/A: no number, but the raw evidence stays inspectable."""
    return EvaTurnTakingMetric(
        score=None,
        gated_score=None,
        passed=None,
        not_applicable=not_applicable,
        params=params,
        turns=turns or [],
        stats=stats or EvaTurnTakingStats(turn_count=0),
    )


def _no_scoreable_turns(params: TurnTakingParams) -> EvaTurnTakingMetric:
    """The zero-evidence N/A: no scoreable caller turns is never a fail."""
    return _na_metric(
        params,
        TurnTakingNotApplicable(
            reason=TurnTakingNAReason.NO_SCOREABLE_TURNS,
            evidence="no scoreable real caller turns",
        ),
    )


def _detect_agent_mute(
    scored: list[EvaTurnTakingTurn],
    real_turns: list[UserSpeechSegment],
    agent_segments: list[AgentSpeechSegment],
    tick_ms: float,
    params: TurnTakingParams,
) -> Optional[TurnTakingNotApplicable]:
    """Infrastructure-mute evidence, or None when turn-taking is observable.

    Two prongs, both grounded in the emitted audio rather than score shape:

    - terminal silence: at least ``mute_min_trailing_turns`` real caller
      turns begin at-or-after the end of the agent's FINAL speech segment —
      the agent never emitted audio again while the caller kept re-taking the
      floor (the permanently-mute-mid-call signature; also covers an agent
      that never spoke at all);
    - missed-response dominance: at least ``mute_missed_fraction`` of scored
      turns are missed responses, backed by ``mute_min_missed_responses``
      misses — the intermittently-dead-channel signature, where sparse agent
      audio after the last miss defeats the terminal-silence prong.
    """
    last_agent_audio_end: Optional[int] = max(
        (segment.end_tick for segment in agent_segments), default=None
    )
    trailing = [
        segment
        for segment in real_turns
        if last_agent_audio_end is None or segment.start_tick >= last_agent_audio_end
    ]
    missed_count = sum(turn.route == EvaTurnRoute.MISSED_RESPONSE for turn in scored)
    missed_fraction = missed_count / len(scored) if scored else None
    terminal_silence = len(trailing) >= params.mute_min_trailing_turns
    missed_dominance = (
        missed_count >= params.mute_min_missed_responses
        and missed_fraction is not None
        and missed_fraction >= params.mute_missed_fraction
    )
    if not (terminal_silence or missed_dominance):
        return None
    prongs = [
        prong
        for prong, fired in (
            ("terminal_silence", terminal_silence),
            ("missed_dominance", missed_dominance),
        )
        if fired
    ]
    return TurnTakingNotApplicable(
        reason=TurnTakingNAReason.AGENT_MUTE,
        trailing_unanswered_turns=len(trailing),
        last_agent_audio_ms=(
            last_agent_audio_end * tick_ms if last_agent_audio_end is not None else None
        ),
        missed_response_count=missed_count,
        missed_response_fraction=missed_fraction,
        evidence=(
            f"{'+'.join(prongs)}: {len(trailing)} real caller turns after "
            f"final agent audio (threshold {params.mute_min_trailing_turns}); "
            f"missed {missed_count}/{len(scored)} scored responses "
            f"(threshold {params.mute_missed_fraction:.2f} with >= "
            f"{params.mute_min_missed_responses} misses)"
        ),
    )


def evaluate_turn_taking(
    ticks: list[Tick],
    *,
    params: Optional[TurnTakingParams] = None,
) -> EvaTurnTakingMetric:
    """Route and score every real caller turn from a τ full-duplex timeline."""
    p = params or TurnTakingParams()
    if not ticks:
        return _no_scoreable_turns(p)
    duration = resolve_tick_duration_seconds(ticks, None)
    tick_ms = duration * 1000.0
    filtered = filter_end_of_conversation_ticks(ticks)
    user_segments = extract_user_segments(filtered, duration)
    agent_segments = extract_agent_segments(filtered, duration)
    transitions = extract_turn_transitions(user_segments, agent_segments)
    interruptions = extract_interruption_events(
        user_segments,
        agent_segments,
        filtered,
        tick_duration_sec=duration,
        no_yield_window_sec=p.user_yield_hard_ms / 1000.0,
    )
    transition_by_idx: dict[int, TurnTransitionEvent] = {
        event.user_segment_idx: event for event in transitions
    }
    # Canonical caller-side overlap events, keyed by user segment: only
    # "user_interrupts_agent" enters the yield formula — the extractor routes
    # backchannel / tic / non-directed overlaps to the selectivity domain.
    yield_event_by_idx: dict[int, InterruptionEvent] = {
        event.interrupter_segment_idx: event
        for event in interruptions
        if event.event_type == "user_interrupts_agent"
        and event.interrupter_segment_idx >= 0
    }
    # Canonical agent incursions (the I_A events), attributed to the caller
    # segment they broke into by onset containment.
    incursion_events = [
        event
        for event in interruptions
        if event.event_type == "agent_interrupts_user"
        and event.interrupter_segment_idx >= 0
    ]

    real_turns = [
        (idx, segment)
        for idx, segment in enumerate(user_segments)
        if _is_real_caller_turn(segment)
    ]

    scored: list[EvaTurnTakingTurn] = []
    for segment_idx, user in real_turns:
        user_start_ms = user.start_tick * tick_ms
        user_end_ms = user.end_tick * tick_ms
        transition = transition_by_idx.get(segment_idx)
        settled_latency_ms: Optional[float] = None
        settled_start_tick: Optional[int] = None
        if transition is not None and transition.outcome == "response":
            settled_latency_ms = transition.gap_sec * 1000.0
            settled_start_tick = transition.agent_start_tick

        incursions = [
            agent_segments[event.interrupter_segment_idx]
            for event in incursion_events
            if user.start_tick <= event.interrupter_start_tick < user.end_tick
        ]
        yield_event = yield_event_by_idx.get(segment_idx)

        uses_tool = False
        post_latency: Optional[float] = None
        overlap: Optional[float] = None
        agent_interrupt_score: Optional[float] = None
        if incursions:
            overlap = sum(_overlap_ms(user, agent, tick_ms) for agent in incursions)
            cap = p.agent_interrupt_max_score
            overlap_score = cap * max(0.0, 1.0 - overlap / p.agent_overlap_hard_ms)
            count_score = cap * max(
                0.0,
                1.0 - (len(incursions) - 1) / float(p.agent_interrupt_count_hard - 1),
            )
            components = [overlap_score, count_score]
            if settled_latency_ms is not None and settled_start_tick is not None:
                # EVA Table 18: the post-interrupt recovery sub-score is
                # omitted when no settled response exists or the agent was
                # still speaking at the caller's segment end (those turns
                # never emit a "response" transition).
                post_latency = settled_latency_ms
                uses_tool = _uses_tool(filtered, user.start_tick, settled_start_tick)
                components.append(
                    eva_latency_score(
                        post_latency,
                        uses_tool=uses_tool,
                        params=p,
                        overlap_excluded=True,
                    )
                )
            agent_interrupt_score = min(components)

        yield_latency: Optional[float] = None
        user_interrupt_score: Optional[float] = None
        if yield_event is not None:
            if yield_event.interrupted_yielded:
                yield_latency = yield_event.yield_time_sec * 1000.0
                user_interrupt_score = max(
                    0.0, 1.0 - yield_latency / p.user_yield_hard_ms
                )
            else:
                # The canonical no-yield event: the agent spoke through the
                # entire window. Hard zero; no finite yield latency exists.
                user_interrupt_score = 0.0

        if agent_interrupt_score is not None and user_interrupt_score is not None:
            route = EvaTurnRoute.DUAL_INTERRUPTION
            score = min(agent_interrupt_score, user_interrupt_score)
        elif agent_interrupt_score is not None:
            route = EvaTurnRoute.AGENT_INTERRUPTION
            score = agent_interrupt_score
        elif user_interrupt_score is not None:
            route = EvaTurnRoute.USER_INTERRUPTION
            score = user_interrupt_score
        elif transition is None:
            immediate = _immediate_response(user, agent_segments)
            if immediate is None:
                # No canonical evidence for this turn: nothing followed it
                # (the call ended there) or the extractor deemed it
                # non-eligible. The canonical response rate does not count
                # it, so neither do we.
                continue
            route = EvaTurnRoute.ORDINARY
            post_latency = 0.0
            uses_tool = _uses_tool(filtered, user.start_tick, immediate.start_tick)
            score = eva_latency_score(
                post_latency,
                uses_tool=uses_tool,
                params=p,
                overlap_excluded=True,
            )
        elif settled_latency_ms is None:
            # Canonical "no_response": the caller had to speak again before
            # any agent response settled.
            route = EvaTurnRoute.MISSED_RESPONSE
            score = 0.0
        else:
            route = EvaTurnRoute.ORDINARY
            post_latency = settled_latency_ms
            uses_tool = _uses_tool(filtered, user.start_tick, settled_start_tick)
            score = eva_latency_score(
                post_latency,
                uses_tool=uses_tool,
                params=p,
                overlap_excluded=True,
            )

        scored.append(
            EvaTurnTakingTurn(
                turn_id=len(scored),
                route=route,
                score=score,
                user_start_ms=user_start_ms,
                user_end_ms=user_end_ms,
                uses_tool=uses_tool,
                latency_ms=(post_latency if route == EvaTurnRoute.ORDINARY else None),
                agent_overlap_ms=overlap,
                agent_interrupt_count=(len(incursions) if incursions else None),
                post_interrupt_latency_ms=(post_latency if incursions else None),
                yield_latency_ms=yield_latency,
                evidence=(
                    f"route={route.value}; user={user_start_ms:.0f}-"
                    f"{user_end_ms:.0f}ms; action={user.action or 'unknown'}"
                ),
            )
        )

    stats = _build_stats(scored)
    # Infrastructure mute preempts both the scored path and the zero-evidence
    # N/A: a dead audio channel is the more informative reason, and the raw
    # per-turn routes and stats stay on the record either way.
    mute = _detect_agent_mute(
        scored, [segment for _, segment in real_turns], agent_segments, tick_ms, p
    )
    if mute is not None:
        return _na_metric(p, mute, turns=scored, stats=stats)
    if not scored:
        return _no_scoreable_turns(p)

    score = mean(turn.score for turn in scored)
    return EvaTurnTakingMetric(
        score=score,
        gated_score=0.0 if any(turn.score == 0.0 for turn in scored) else score,
        passed=score >= EVA_X_PASS_THRESHOLD,
        params=p,
        turns=scored,
        stats=stats,
    )


def _build_stats(scored: list[EvaTurnTakingTurn]) -> EvaTurnTakingStats:
    """Call-level rates and latency diagnostics; tolerates zero scored turns."""
    if not scored:
        return EvaTurnTakingStats(turn_count=0)
    response_latencies = [
        turn.latency_ms for turn in scored if turn.latency_ms is not None
    ]
    post_interrupt_latencies = [
        turn.post_interrupt_latency_ms
        for turn in scored
        if turn.post_interrupt_latency_ms is not None
    ]
    yield_latencies = [
        turn.yield_latency_ms for turn in scored if turn.yield_latency_ms is not None
    ]
    count = len(scored)
    routes = [
        EvaTurnRouteBreakdown(
            route=route,
            turn_count=len(route_scores),
            mean_score=mean(route_scores),
        )
        for route in EvaTurnRoute
        if (route_scores := [turn.score for turn in scored if turn.route == route])
    ]
    return EvaTurnTakingStats(
        turn_count=count,
        routes=routes,
        missed_response_rate=(
            sum(turn.route == EvaTurnRoute.MISSED_RESPONSE for turn in scored) / count
        ),
        agent_interruption_rate=(
            sum(
                turn.route
                in {
                    EvaTurnRoute.AGENT_INTERRUPTION,
                    EvaTurnRoute.DUAL_INTERRUPTION,
                }
                for turn in scored
            )
            / count
        ),
        user_interruption_rate=(
            sum(
                turn.route
                in {
                    EvaTurnRoute.USER_INTERRUPTION,
                    EvaTurnRoute.DUAL_INTERRUPTION,
                }
                for turn in scored
            )
            / count
        ),
        response_latency_mean_ms=(
            mean(response_latencies) if response_latencies else None
        ),
        response_latency_p50_ms=percentile(response_latencies, 0.5),
        response_latency_p90_ms=percentile(response_latencies, 0.9),
        post_interrupt_latency_mean_ms=(
            mean(post_interrupt_latencies) if post_interrupt_latencies else None
        ),
        post_interrupt_latency_p50_ms=percentile(post_interrupt_latencies, 0.5),
        post_interrupt_latency_p90_ms=percentile(post_interrupt_latencies, 0.9),
        yield_latency_mean_ms=(mean(yield_latencies) if yield_latencies else None),
        yield_latency_p50_ms=percentile(yield_latencies, 0.5),
        yield_latency_p90_ms=percentile(yield_latencies, 0.9),
    )
