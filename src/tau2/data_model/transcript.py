"""Typed transcript events and participant-perspective tick contracts."""

from enum import Enum
from typing import Annotated, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

from tau2.data_model.message import Message, ParticipantMessageBase

InputMessageType = TypeVar("InputMessageType", bound=Message)
OutputMessageType = TypeVar("OutputMessageType", bound=Message)


class LiveTranscriptFragment(BaseModel):
    """Original Live transcript evidence, timed on the provider session clock."""

    event_id: str | None = None
    speaker: Literal["agent", "caller"]
    text: str
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    ingress_audio_bytes: int | None = Field(default=None, ge=0)


class LiveTranscriptTick(BaseModel):
    """Serializable evidence; playback activity uses tick-relative audio time."""

    session_id: str
    origin: Literal["provider", "recorded_audio"] = "provider"
    fragments: list[LiveTranscriptFragment] = Field(default_factory=list)
    speech_intervals_ms: list[tuple[float, float]] = Field(default_factory=list)


class ParticipantTick(BaseModel, Generic[InputMessageType, OutputMessageType]):
    """
    Represents events from a single tick from a participant's perspective.

    In full-duplex mode, both parties can emit chunks simultaneously at each tick.
    This captures three concurrent channels per tick:

    - self_chunk: what I emitted (speech, or a message containing tool_calls)
    - other_chunk: speech/audio received from the other participant
    - env_chunk: tool results received from the environment (ToolMessage or
      MultiToolMessage), delivered in response to a tool call from a previous tick

    Before linearization, ticks with env_chunk are expanded into old-format
    ticks (env_chunk merged into other_chunk) by expand_env_chunks, so that
    linearization code sees the same structure as the base branch.

    Attributes:
        tick_id: The sequential identifier for this tick.
        timestamp: When this tick occurred.
        self_chunk: The chunk emitted by this participant (if any, may include tool calls).
        other_chunk: The chunk received from the other participant (speech/audio).
        env_chunk: Tool results from the environment (ToolMessage or MultiToolMessage).
    """

    model_config = {"arbitrary_types_allowed": True}

    tick_id: int
    timestamp: str
    self_chunk: Optional[OutputMessageType] = None
    other_chunk: Optional[InputMessageType] = None
    env_chunk: Optional[Message] = None


class SpeechSegment(BaseModel):
    """Speech grouped across short pauses, indexed in the input tick list."""

    start_index: int = Field(description="First input-list position in the segment.")
    end_index: int = Field(description="Last input-list position in the segment.")
    chunks: dict[int, Message] = Field(
        description="Speech chunks by input-list position."
    )


class ToolExchange(BaseModel):
    """A tool call and its matching result, which may not have arrived yet."""

    call_index: int = Field(description="Input-list position of the tool call.")
    call: Message = Field(description="Message containing the tool call.")
    result: Message | None = Field(
        description="Matching tool result, or None while pending."
    )


class ParticipantSegments(BaseModel):
    """Speech and tool activity from a participant's perspective."""

    self_speech: list[SpeechSegment] = Field(
        description="Speech emitted by this participant."
    )
    other_speech: list[SpeechSegment] = Field(
        description="Speech received from the other participant."
    )
    tool_exchanges: list[ToolExchange] = Field(
        description="Matched and pending tool calls."
    )


class LinearizationStrategy(str, Enum):
    """
    Strategies for converting tick-based history to sequential messages.

    LLMs expect sequential message history, but full-duplex ticks contain
    concurrent events. These strategies define how to order concurrent events
    into a sequence suitable for LLM consumption.
    """

    TIMESTAMP_ORDER = "timestamp"  # Sort all messages by timestamp
    SELF_FIRST_PER_TICK = "self_first"  # Each tick: self chunk first, then other chunk
    OTHER_FIRST_PER_TICK = (
        "other_first"  # Each tick: other chunk first, then self chunk
    )
    CONSOLIDATED = "consolidated"  # Merge consecutive same-speaker messages
    CONTAINMENT_AWARE = "containment_aware"


class UtteranceGroup(BaseModel):
    """One participant's chunks and their original timeline anchors."""

    speaker: Annotated[
        Literal["agent", "caller"],
        Field(description="Participant who emitted these chunks."),
    ]
    start_index: Annotated[
        int, Field(description="Position in the source tick list, used for ordering.")
    ]
    end_index: Annotated[
        int, Field(description="Last source-list position containing this utterance.")
    ]
    start_tick: Annotated[
        int, Field(description="First original tick ID, not a list position.")
    ]
    end_tick: Annotated[
        int, Field(description="Last original tick ID containing this utterance.")
    ]
    chunks: Annotated[
        list[ParticipantMessageBase],
        Field(
            description="Source chunks, or a derived Live segment retaining its fragments."
        ),
    ]


class SpokenTurn(BaseModel):
    """Raw spoken text in call order; judge markers are never stored here."""

    speaker: Annotated[
        Literal["agent", "caller"],
        Field(description="Participant who delivered the turn."),
    ]
    text: Annotated[str, Field(description="Raw concatenated transcript content.")]
    interrupted: Annotated[
        bool,
        Field(description="Union of recorded truncation and timeline overlap signals."),
    ] = False


class AgentTurn(BaseModel):
    """Delivered agent utterance with its preceding completed caller context."""

    index: Annotated[int, Field(ge=0, description="Stable delivered-turn ordinal.")]
    segment_id: str | None = Field(default=None, exclude=True)
    start_tick: int | None = Field(default=None, exclude=True)
    end_tick: int | None = Field(default=None, exclude=True)
    text: Annotated[
        str,
        Field(
            description="Delivered gold text when available, otherwise transcript text."
        ),
    ]
    preceding_user_text: Annotated[
        str | None,
        Field(description="Immediately preceding completed caller utterance."),
    ] = None
    overlapping_user_text: str | None = Field(
        default=None,
        exclude=True,
        description="Recorded caller speech overlapping the agent audio span.",
    )
    interrupted: Annotated[
        bool,
        Field(
            description="Explicit Live cut, or legacy chunk/timeline interruption signals."
        ),
    ] = False


class TickRange(BaseModel):
    """Half-open positions in an input tick list (not provider tick IDs)."""

    start: Annotated[int, Field(ge=0, description="Included source-list position.")]
    end: Annotated[int, Field(ge=0, description="Excluded source-list position.")]


class DisplayTick(BaseModel):
    """Visible content/activity facts used to group a rendered transcript row."""

    agent_content: Annotated[
        str, Field(description="Visible agent transcript text.")
    ] = ""
    user_content: Annotated[
        str, Field(description="Visible caller transcript text.")
    ] = ""
    agent_turn_action: Annotated[
        str, Field(description="Visible agent turn-taking label.")
    ] = ""
    user_turn_action: Annotated[
        str, Field(description="Visible caller turn-taking label.")
    ] = ""
    has_tool_activity: Annotated[
        bool, Field(description="Whether this presentation exposes tool activity.")
    ] = False
