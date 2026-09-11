# Copyright Sierra
"""One ordered timeline per call: caller utterance units + agent tool events.

The deterministic loss instruments (caller-cost, entity-trace) both need the
same two event streams positioned on ONE order axis, so re-dictations can be
placed before/after tool outcomes:

- **Caller units** — what the caller said, in order. Voice calls reuse the
  canonical segment extractor (``tau2.metrics.voice_interaction_metrics.
  extract_user_segments``) so the unit boundaries are exactly the τ-voice
  segment model, never a parallel re-derivation from raw ticks; text calls
  use content-bearing user messages.
- **Tool events** — every agent tool call paired with its result. Voice
  calls read the inline tick tool activity; text calls pair assistant tool
  calls with their tool messages by id.

The shared order axis is the tick index for voice calls and the message index
for text calls; the two streams of one call always use the same axis, so
``unit.order_key < event.order_key`` means "said before the tool call".
"""

from typing import Annotated, Optional

from pydantic import BaseModel, Field

from tau2.data_model.message import ToolCall
from tau2.data_model.simulation import SimulationRun


class CallerUnit(BaseModel):
    """One caller utterance unit (voice segment or text message)."""

    order_key: Annotated[
        int,
        Field(
            description="Position on the call's order axis: start tick index "
            "(voice) or message index (text)."
        ),
    ]
    text: Annotated[str, Field(description="What the caller said in this unit.")]
    is_backchannel: Annotated[
        bool,
        Field(
            description="True for a backchannel-classified voice segment "
            "(never counts as a caller turn; always False on text units)."
        ),
    ] = False
    is_barge_in: Annotated[
        bool,
        Field(
            description="True when this unit started while the agent was "
            "speaking and it is not a backchannel (voice only)."
        ),
    ] = False


class ToolEvent(BaseModel):
    """One agent tool call paired with its result (when one was recorded)."""

    order_key: Annotated[
        int,
        Field(
            description="Position of the CALL on the call's order axis "
            "(tick index for voice, message index for text)."
        ),
    ]
    call: Annotated[ToolCall, Field(description="The tool call as recorded.")]
    result_order_key: Annotated[
        Optional[int],
        Field(description="Position of the paired result; None when unpaired."),
    ] = None
    result_content: Annotated[
        Optional[str], Field(description="Content of the paired result.")
    ] = None
    result_error: Annotated[
        Optional[bool],
        Field(description="Error flag of the paired result; None when unpaired."),
    ] = None


def extract_caller_units(sim: SimulationRun) -> list[CallerUnit]:
    """The caller's utterance units in order, on the call's order axis."""
    if sim.ticks:
        from tau2.metrics.voice_interaction_metrics import extract_user_segments

        units = []
        for segment in extract_user_segments(sim.ticks):
            text = (segment.transcript or "").strip()
            if not text:
                continue
            units.append(
                CallerUnit(
                    order_key=segment.start_tick,
                    text=text,
                    is_backchannel=segment.is_backchannel,
                    is_barge_in=(
                        segment.is_interruption and not segment.is_backchannel
                    ),
                )
            )
        return units
    units = []
    for index, message in enumerate(sim.get_messages()):
        if getattr(message, "role", None) != "user":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            units.append(CallerUnit(order_key=index, text=content.strip()))
    return units


def extract_tool_events(sim: SimulationRun) -> list[ToolEvent]:
    """Every agent tool call, in order, paired with its result by call id."""
    events: list[ToolEvent] = []
    by_call_id: dict[str, ToolEvent] = {}
    if sim.ticks:
        for tick_index, tick in enumerate(sim.ticks):
            for call in tick.agent_tool_calls:
                event = ToolEvent(order_key=tick_index, call=call)
                events.append(event)
                if call.id:
                    by_call_id[call.id] = event
            for result in tick.agent_tool_results:
                event = by_call_id.get(result.id)
                if event is not None:
                    event.result_order_key = tick_index
                    event.result_content = result.content
                    event.result_error = result.error
        return events
    for index, message in enumerate(sim.get_messages()):
        for call in getattr(message, "tool_calls", None) or []:
            if call.requestor != "assistant":
                continue
            event = ToolEvent(order_key=index, call=call)
            events.append(event)
            if call.id:
                by_call_id[call.id] = event
        if (
            getattr(message, "role", None) == "tool"
            and getattr(message, "requestor", None) == "assistant"
        ):
            event = by_call_id.get(message.id)
            if event is not None:
                event.result_order_key = index
                event.result_content = message.content
                event.result_error = message.error
    return events
