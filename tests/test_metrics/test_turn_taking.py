# Copyright Sierra
"""Deterministic EVA turn-taking routing/scoring contracts (metrics-owned)."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tau2.data_model.audio_effects import SpeechEffectsResult, UserSpeechInsert
from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    TurnTakingAction,
    UserMessage,
)
from tau2.metrics.turn_taking import (
    EvaTurnTakingMetric,
    EvaTurnTakingStats,
    TurnTakingNAReason,
    TurnTakingNotApplicable,
    TurnTakingParams,
    eva_latency_score,
    evaluate_turn_taking,
    resolve_turn_taking_params,
)
from tau2.metrics.voice_interaction_metrics import (
    InteractionMetricsConfig,
    compute_interaction_metrics_for_ticks,
)
from tau2.multilingual.schema import TurnTakingPackConfig


def _tick(
    tick_id: int,
    *,
    user: bool = False,
    agent: bool = False,
    action: str | None = None,
    effect: str | None = None,
    tool: bool = False,
    content: str = "caller",
) -> Tick:
    speech_effects = (
        SpeechEffectsResult(speech_insert=UserSpeechInsert(text="um", type=effect))
        if effect
        else None
    )
    user_chunk = None
    if user or effect:
        user_chunk = UserMessage(
            role="user",
            content=content,
            contains_speech=user,
            turn_taking_action=TurnTakingAction(action=action) if action else None,
            speech_effects=speech_effects,
        )
    return Tick(
        tick_id=tick_id,
        timestamp=str(tick_id),
        tick_duration_seconds=0.5,
        user_chunk=user_chunk,
        agent_chunk=AssistantMessage(
            role="assistant", content="agent", contains_speech=agent
        ),
        agent_tool_calls=(
            [ToolCall(id=f"tool-{tick_id}", name="lookup", arguments={})]
            if tool
            else []
        ),
    )


def _ticks(spec: list[dict]) -> list[Tick]:
    return [_tick(i, **row) for i, row in enumerate(spec)]


@pytest.mark.parametrize(
    ("latency_ms", "uses_tool", "expected"),
    [
        (-500.0, False, 0.0),
        (0.0, False, 0.5),
        (500.0, False, 1.0),
        (2000.0, False, 1.0),
        (2750.0, False, 0.5),
        (3500.0, False, 0.0),
        (3000.0, True, 1.0),
        (4000.0, True, 0.5),
        (5000.0, True, 0.0),
    ],
)
def test_eva_latency_curve(latency_ms, uses_tool, expected):
    assert eva_latency_score(latency_ms, uses_tool=uses_tool) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("latency_ms", "expected"),
    [
        # Canonical transitions are non-overlapping by construction, so the
        # premature-interruption early ramp cannot apply: a genuinely fast
        # quantized ~0ms response is not penalized.
        (0.0, 1.0),
        (200.0, 1.0),
        (2000.0, 1.0),
        # Late side is unchanged.
        (2750.0, 0.5),
        # A negative latency would mean overlap: the band never fires.
        (-100.0, 0.4),
    ],
)
def test_overlap_excluded_band_disables_early_ramp(latency_ms, expected):
    assert eva_latency_score(
        latency_ms, uses_tool=False, overlap_excluded=True
    ) == pytest.approx(expected)


def test_ordinary_turn_uses_tool_aware_latency_curve():
    # Caller ends at 1s; the agent begins at 5s. Four seconds is outside the
    # ordinary curve but halfway down EVA's tool-aware late ramp.
    base = (
        [{"user": True, "action": "keep_talking"}] * 2
        + [{}] * 8
        + [{"agent": True}]
        + [{}]
    )
    without_tool = evaluate_turn_taking(_ticks(base))
    assert without_tool.turns[0].route == "ordinary"
    assert without_tool.turns[0].latency_ms == pytest.approx(4000.0)
    assert without_tool.turns[0].score == 0.0

    with_tool_spec = list(base)
    with_tool_spec[5] = {"tool": True}
    with_tool = evaluate_turn_taking(_ticks(with_tool_spec))
    assert with_tool.turns[0].uses_tool is True
    assert with_tool.turns[0].score == pytest.approx(0.5)


def test_agent_interruption_scores_overlap_count_and_recovery():
    # Two distinct agent incursions overlap a 2s caller turn for 1s total.
    # Both overlap and count components score .25; recovery is prompt.
    result = evaluate_turn_taking(
        _ticks(
            [
                {"user": True, "action": "keep_talking"},
                {"user": True, "agent": True},
                {"user": True},
                {"user": True, "agent": True},
                {},
                {"agent": True},
                {},
            ]
        )
    )
    turn = result.turns[0]
    assert turn.route == "agent_interruption"
    assert turn.agent_interrupt_count == 2
    assert turn.agent_overlap_ms == pytest.approx(1000.0)
    assert turn.post_interrupt_latency_ms == pytest.approx(500.0)
    assert turn.score == pytest.approx(0.25)


def test_user_interruption_no_yield_is_a_hard_zero_without_fake_latency():
    # The agent speaks through the entire canonical no-yield window: the
    # canonical event is a no-yield, scored 0 with no finite yield latency.
    result = evaluate_turn_taking(
        _ticks(
            [{"agent": True}] * 2
            + [{"agent": True, "user": True, "action": "keep_talking"}] * 2
            + [{"agent": True}] * 2
            + [{}]
        )
    )
    turn = result.turns[0]
    assert turn.route == "user_interruption"
    assert turn.yield_latency_ms is None
    assert turn.score == 0.0
    assert result.gated_score == 0.0


def test_user_interruption_scores_canonical_yield_latency():
    # The agent yields one tick (500ms) after the typed caller onset.
    result = evaluate_turn_taking(
        _ticks(
            [
                {"agent": True},
                {"agent": True},
                {"agent": True, "user": True, "action": "keep_talking"},
                {"user": True},
                {},
            ]
        )
    )
    turn = result.turns[0]
    assert turn.route == "user_interruption"
    assert turn.yield_latency_ms == pytest.approx(500.0)
    assert turn.score == pytest.approx(0.75)


def test_backchannel_is_not_a_turn_and_final_turn_without_evidence_is_na():
    # The typed backchannel never becomes a caller turn, and the final real
    # turn has no canonical evidence (nothing followed it), so the call has
    # zero scoreable turns: N/A, never a zero-fail.
    result = evaluate_turn_taking(
        _ticks(
            [
                {"agent": True},
                {"agent": True, "user": True, "action": "backchannel"},
                {},
                {"user": True, "action": "keep_talking"},
                {"user": True},
                {},
            ]
        )
    )
    assert result.turns == []
    assert result.stats.turn_count == 0
    assert result.score is None
    assert result.gated_score is None
    assert result.passed is None
    assert result.not_applicable is not None
    assert result.not_applicable.reason == TurnTakingNAReason.NO_SCOREABLE_TURNS


def test_mid_call_no_response_is_missed_and_gates_the_call():
    # Turn one is never answered before the caller must speak again
    # (canonical "no_response"); turn two gets a prompt answer.
    result = evaluate_turn_taking(
        _ticks(
            [
                {"user": True, "action": "keep_talking"},
                {},
                {"user": True, "action": "keep_talking"},
                {},
                {"agent": True},
                {},
            ]
        )
    )
    assert [turn.route for turn in result.turns] == [
        "missed_response",
        "ordinary",
    ]
    assert result.turns[0].score == 0.0
    assert result.turns[1].score == pytest.approx(1.0)
    assert result.score == pytest.approx(0.5)
    assert result.gated_score == 0.0
    assert result.passed is False
    routes = {row.route: row for row in result.stats.routes}
    assert routes["missed_response"].turn_count == 1
    assert routes["ordinary"].mean_score == pytest.approx(1.0)


# One answered caller turn: 1s of caller speech, a 500ms gap, a 1s agent
# response, a trailing gap. One unanswered turn: the same caller speech with
# nothing but silence after it.
ANSWERED_TURN = (
    [{"user": True, "action": "keep_talking"}] * 2 + [{}] + [{"agent": True}] * 2 + [{}]
)
UNANSWERED_TURN = [{"user": True, "action": "keep_talking"}] * 2 + [{}] * 3


def test_permanent_mid_call_mute_is_reasoned_na_not_a_zero_score():
    # The real ko×OpenAI failure shape: the agent answers, then its audio
    # channel dies mid-call and the caller re-takes the floor again and again
    # into silence. The call is N/A with a typed agent-mute reason — never a
    # near-zero score contaminating cell means — while the raw per-turn
    # routes and stats stay inspectable.
    result = evaluate_turn_taking(_ticks(ANSWERED_TURN + UNANSWERED_TURN * 4))
    assert result.score is None
    assert result.gated_score is None
    assert result.passed is None
    na = result.not_applicable
    assert na is not None
    assert na.reason == TurnTakingNAReason.AGENT_MUTE
    # All four post-mute caller turns start after the agent's final audio.
    assert na.trailing_unanswered_turns == 4
    assert na.last_agent_audio_ms == pytest.approx(2500.0)
    assert na.missed_response_count == 3
    assert na.missed_response_fraction == pytest.approx(0.75)
    assert "terminal_silence" in na.evidence
    # Raw evidence is retained: the answered turn plus the three missed
    # responses (the final caller turn has no canonical evidence).
    assert [turn.route for turn in result.turns] == [
        "ordinary",
        "missed_response",
        "missed_response",
        "missed_response",
    ]
    assert result.stats.turn_count == 4
    assert result.stats.missed_response_rate == pytest.approx(0.75)


def test_late_mute_is_caught_by_terminal_silence_below_the_fraction_rule():
    # The channel dies late: six answered turns, then three caller turns into
    # silence. The missed fraction (2/8) is far below the dominance
    # threshold — the downstream >=50% report rule missed this population —
    # but the tick-level evidence (three real turns after the agent's final
    # audio) still establishes infrastructure mute.
    ticks = _ticks(ANSWERED_TURN * 6 + UNANSWERED_TURN * 3)
    result = evaluate_turn_taking(ticks)
    na = result.not_applicable
    assert na is not None
    assert na.reason == TurnTakingNAReason.AGENT_MUTE
    assert na.trailing_unanswered_turns == 3
    assert na.missed_response_fraction == pytest.approx(0.25)
    assert "terminal_silence" in na.evidence
    # The thresholds govern: a stricter trailing-turn requirement returns the
    # same call to the scored population.
    strict = evaluate_turn_taking(
        ticks, params=TurnTakingParams(mute_min_trailing_turns=5)
    )
    assert strict.not_applicable is None
    assert strict.score is not None


def test_intermittent_mute_is_caught_by_missed_dominance():
    # Half the responses are missing but the agent still emits audio after
    # the last miss, defeating the terminal-silence prong: the
    # missed-dominance prong (>=50% missed, >=2 misses) still routes the call
    # to a reasoned N/A.
    result = evaluate_turn_taking(
        _ticks(
            ANSWERED_TURN
            + [{"user": True, "action": "keep_talking"}] * 2
            + [{}]
            + [{"user": True, "action": "keep_talking"}] * 2
            + [{}]
            + ANSWERED_TURN
        )
    )
    na = result.not_applicable
    assert na is not None
    assert na.reason == TurnTakingNAReason.AGENT_MUTE
    assert na.trailing_unanswered_turns == 0
    assert na.missed_response_count == 2
    assert na.missed_response_fraction == pytest.approx(0.5)
    assert "missed_dominance" in na.evidence
    assert "terminal_silence" not in na.evidence


def test_bad_turn_taking_below_mute_thresholds_stays_a_scored_fail():
    # One missed response and two trailing caller turns: real turn-taking
    # defects, below both mute prongs. The call stays scored (and gated to
    # zero) — reasoned N/A never absorbs ordinary bad behavior.
    result = evaluate_turn_taking(_ticks(ANSWERED_TURN * 3 + UNANSWERED_TURN * 2))
    assert result.not_applicable is None
    assert result.score is not None
    routes = [turn.route for turn in result.turns]
    assert routes.count("missed_response") == 1
    assert result.gated_score == 0.0
    assert result.passed is False


def test_agent_that_never_speaks_is_mute_not_merely_unscoreable():
    # A channel dead from tick zero: every real caller turn is trailing and
    # unanswered. The typed reason is agent-mute with no agent-audio
    # timestamp at all.
    result = evaluate_turn_taking(_ticks(UNANSWERED_TURN * 4))
    na = result.not_applicable
    assert na is not None
    assert na.reason == TurnTakingNAReason.AGENT_MUTE
    assert na.trailing_unanswered_turns == 4
    assert na.last_agent_audio_ms is None


def test_na_metric_must_carry_a_typed_reason():
    with pytest.raises(ValidationError):
        EvaTurnTakingMetric(
            score=None, turns=[], stats=EvaTurnTakingStats(turn_count=0)
        )
    with pytest.raises(ValidationError):
        EvaTurnTakingMetric(
            score=0.5,
            gated_score=0.5,
            passed=False,
            not_applicable=TurnTakingNotApplicable(
                reason=TurnTakingNAReason.AGENT_MUTE
            ),
            turns=[],
            stats=EvaTurnTakingStats(turn_count=0),
        )


def test_fast_response_at_the_callers_end_tick_scores_one():
    # The agent starts the very tick the caller stopped: the canonical
    # transition extractor skips this segment (other_speaking_at_end), so
    # the scorer routes it explicitly as an ordinary ~0ms response.
    result = evaluate_turn_taking(
        _ticks(
            [
                {"user": True, "action": "keep_talking"},
                {"user": True},
                {"agent": True},
                {},
            ]
        )
    )
    turn = result.turns[0]
    assert turn.route == "ordinary"
    assert turn.latency_ms == pytest.approx(0.0)
    assert turn.score == pytest.approx(1.0)


def test_partial_vocal_tic_does_not_drop_a_real_turn():
    # One injected tic tick inside a three-tick real turn: the turn keeps
    # its floor-taking status and is scored normally.
    result = evaluate_turn_taking(
        _ticks(
            [
                {"user": True, "action": "keep_talking"},
                {"user": True, "effect": "vocal_tic"},
                {"user": True},
                {},
                {"agent": True},
                {},
            ]
        )
    )
    assert len(result.turns) == 1
    assert result.turns[0].route == "ordinary"
    assert result.turns[0].score == pytest.approx(1.0)


def test_effect_only_segment_is_not_a_turn():
    # A segment fully covered by the injected effect is a selectivity event,
    # not a caller turn.
    result = evaluate_turn_taking(
        _ticks(
            [
                {"user": True, "effect": "vocal_tic"},
                {},
                {"agent": True},
                {},
            ]
        )
    )
    assert result.turns == []
    assert result.score is None
    assert result.not_applicable.reason == TurnTakingNAReason.NO_SCOREABLE_TURNS


def test_stop_marker_segment_is_never_a_missed_response():
    # Real speech merged with the ###STOP### end-of-call marker survives
    # filter_end_of_conversation_ticks; the typed marker routes it out.
    result = evaluate_turn_taking(
        _ticks(
            [
                {"agent": True},
                {},
                {"user": True, "action": "keep_talking"},
                {"user": True, "content": "###STOP###"},
                {},
                {"agent": True},
                {},
            ]
        )
    )
    assert result.turns == []
    assert result.score is None
    assert result.not_applicable.reason == TurnTakingNAReason.NO_SCOREABLE_TURNS


def test_empty_ticks_are_na():
    result = evaluate_turn_taking([])
    assert result.score is None
    assert result.passed is None
    assert result.stats.turn_count == 0
    assert result.not_applicable.reason == TurnTakingNAReason.NO_SCOREABLE_TURNS


def test_event_rates_are_consistent_with_the_canonical_panel():
    # One agent incursion, one caller interruption with a canonical yield,
    # and one ordinary answered turn: the scorer's event counts must agree
    # with the VoiceInteractionMetrics panel for the same ticks.
    ticks = _ticks(
        [
            {"user": True, "action": "keep_talking"},
            {"user": True, "agent": True},
            {"user": True},
            {},
            {"agent": True},
            {},
            {"user": True, "action": "keep_talking"},
            {},
            {"agent": True},
            {},
            {"agent": True},
            {"agent": True, "user": True, "action": "keep_talking"},
            {"user": True},
            {},
            {"agent": True},
            {},
        ]
    )
    result = evaluate_turn_taking(ticks)
    panel = compute_interaction_metrics_for_ticks(
        ticks, InteractionMetricsConfig(tick_duration_sec=0.5)
    )
    routed = {turn.route for turn in result.turns}
    assert routed == {"agent_interruption", "ordinary", "user_interruption"}
    assert panel.counts.agent_interrupts_count == sum(
        turn.agent_interrupt_count or 0 for turn in result.turns
    )
    assert panel.counts.yield_total == sum(
        turn.route in {"user_interruption", "dual_interruption"}
        for turn in result.turns
    )
    assert panel.counts.backchannel_total == 0


def test_pack_overrides_resolve_onto_eva_defaults(monkeypatch):
    packs = {
        "xx": SimpleNamespace(turn_taking=TurnTakingPackConfig(sweet_high_ms=2500.0)),
        "yy": SimpleNamespace(turn_taking=None),
    }
    monkeypatch.setattr(
        "tau2.multilingual.registry.get_language_pack",
        lambda language: packs.get(language),
    )
    overridden = resolve_turn_taking_params("xx")
    assert overridden.sweet_high_ms == pytest.approx(2500.0)
    assert overridden.hard_late_ms == pytest.approx(3500.0)
    assert overridden.origin == "pack:xx"
    for language in (None, "yy", "zz"):
        params = resolve_turn_taking_params(language)
        assert params.sweet_high_ms == pytest.approx(2000.0)
        assert params.origin == "eva-v0.2-defaults"
