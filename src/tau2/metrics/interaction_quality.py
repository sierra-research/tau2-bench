"""Deterministic per-call turn-taking measurements behind call-level Interaction.

These are the counts the release's quality checkers read for nonresponse,
interruption, selectivity and monologue. They come from the call's ticks alone:
no judge or model is involved. The τ-voice event extraction
(``extract_voice_quality_events_from_simulation``) supplies response, yield and
selectivity events; agent interruptions are counted from the same filtered
segments, and the longest agent floor hold from the raw tick timeline.
"""

from statistics import fmean
from typing import Annotated

from pydantic import BaseModel, Field

from tau2.config import DEFAULT_TICK_DURATION_SECONDS
from tau2.data_model.simulation import SimulationRun, Tick
from tau2.metrics.voice_interaction_metrics import (
    InteractionMetricsConfig,
    extract_all_segments,
    extract_interruption_events,
    extract_out_of_turn_effects,
    extract_voice_quality_events_from_simulation,
    filter_end_of_conversation_ticks,
)
from tau2.utils.transcript.segments import contiguous_ranges


class InteractionMeasurements(BaseModel):
    """Event counts for one tick-bearing call."""

    response_total: Annotated[int, Field(ge=0, description="Caller turns due a reply.")]
    responses: Annotated[int, Field(ge=0, description="Caller turns answered.")]
    response_latency_mean: Annotated[
        float | None, Field(description="Mean reply latency in seconds.")
    ]
    yield_total: Annotated[
        int, Field(ge=0, description="Caller interruptions of agent speech.")
    ]
    yields: Annotated[int, Field(ge=0, description="Interruptions yielded to.")]
    backchannel_total: Annotated[int, Field(ge=0, description="Caller backchannels.")]
    backchannel_errors: Annotated[
        int, Field(ge=0, description="Backchannels the agent stopped for.")
    ]
    vocal_tic_total: Annotated[int, Field(ge=0, description="Caller vocal tics.")]
    vocal_tic_errors: Annotated[
        int, Field(ge=0, description="Vocal tics the agent stopped for or answered.")
    ]
    non_directed_total: Annotated[
        int, Field(ge=0, description="Caller speech not directed at the agent.")
    ]
    non_directed_errors: Annotated[
        int, Field(ge=0, description="Non-directed speech the agent reacted to.")
    ]
    agent_interruptions: Annotated[
        int, Field(ge=0, description="Agent speech onsets during caller speech.")
    ]
    max_agent_floor_hold_seconds: Annotated[
        float | None,
        Field(description="Longest agent hold between caller turns; null if none."),
    ]

    @property
    def selectivity_total(self) -> int:
        return self.backchannel_total + self.vocal_tic_total + self.non_directed_total

    @property
    def selectivity_errors(self) -> int:
        return (
            self.backchannel_errors + self.vocal_tic_errors + self.non_directed_errors
        )


def tick_duration_seconds(ticks: list[Tick]) -> float:
    """The recorded tick step, else the repository default."""
    for tick in ticks:
        if tick.tick_duration_seconds:
            return tick.tick_duration_seconds
    return DEFAULT_TICK_DURATION_SECONDS


def _agent_interruptions(ticks: list[Tick], config: InteractionMetricsConfig) -> int:
    """``agent_interrupts_user`` events on the event extraction's filtered segments."""
    filtered = filter_end_of_conversation_ticks(ticks)
    user_segments, agent_segments = extract_all_segments(
        filtered, config.tick_duration_sec
    )
    events = extract_interruption_events(
        user_segments,
        agent_segments,
        filtered,
        tick_duration_sec=config.tick_duration_sec,
        no_yield_window_sec=config.no_yield_window_sec,
        backchannel_yield_window_sec=config.backchannel_yield_window_sec,
        vocal_tic_yield_window_sec=config.vocal_tic_yield_window_sec,
        non_directed_yield_window_sec=config.non_directed_yield_window_sec,
        vocal_tic_response_window_sec=config.vocal_tic_response_window_sec,
        non_directed_response_window_sec=config.non_directed_response_window_sec,
        out_of_turn_effects=extract_out_of_turn_effects(
            filtered, config.tick_duration_sec
        ),
    )
    return sum(event.event_type == "agent_interrupts_user" for event in events)


def _max_agent_floor_hold(ticks: list[Tick], duration: float) -> float | None:
    """Longest span from first to last agent speech tick between caller turns.

    Pauses inside the span count: the agent holds the floor until the caller
    speaks again.
    """
    agent = [bool(t.agent_chunk and t.agent_chunk.contains_speech) for t in ticks]
    user = [bool(t.user_chunk and t.user_chunk.contains_speech) for t in ticks]
    holds = []
    for span in contiguous_ranges(not speaking for speaking in user):
        speaking = [i for i in range(span.start, span.end) if agent[i]]
        if speaking:
            holds.append((speaking[-1] - speaking[0] + 1) * duration)
    return max(holds) if holds else None


def measure_interaction(simulation: SimulationRun) -> InteractionMeasurements | None:
    """Measure one call; ``None`` when it has no ticks (nothing is measurable)."""
    ticks = simulation.ticks
    if not ticks:
        return None
    duration = tick_duration_seconds(ticks)
    config = InteractionMetricsConfig(tick_duration_sec=duration)
    events = extract_voice_quality_events_from_simulation(
        ticks,
        tick_duration_sec=config.tick_duration_sec,
        no_yield_window_sec=config.no_yield_window_sec,
        backchannel_yield_window_sec=config.backchannel_yield_window_sec,
        vocal_tic_yield_window_sec=config.vocal_tic_yield_window_sec,
        non_directed_yield_window_sec=config.non_directed_yield_window_sec,
        vocal_tic_response_window_sec=config.vocal_tic_response_window_sec,
        non_directed_response_window_sec=config.non_directed_response_window_sec,
    )

    def count(event_type: str) -> int:
        return sum(event.event_type == event_type for event in events)

    latencies = [
        event.latency_sec
        for event in events
        if event.event_type == "response" and event.latency_sec is not None
    ]
    return InteractionMeasurements(
        response_total=count("response") + count("no_response"),
        responses=count("response"),
        response_latency_mean=fmean(latencies) if latencies else None,
        yield_total=count("yield") + count("no_yield"),
        yields=count("yield"),
        backchannel_total=count("backchannel_correct") + count("backchannel_error"),
        backchannel_errors=count("backchannel_error"),
        vocal_tic_total=count("vocal_tic_correct") + count("vocal_tic_error"),
        vocal_tic_errors=count("vocal_tic_error"),
        non_directed_total=count("non_directed_correct") + count("non_directed_error"),
        non_directed_errors=count("non_directed_error"),
        agent_interruptions=_agent_interruptions(ticks, config),
        max_agent_floor_hold_seconds=_max_agent_floor_hold(ticks, duration),
    )
