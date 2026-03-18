"""
Test fixtures for containment-aware linearization.

These fixtures define test scenarios for the CONTAINMENT_AWARE linearization strategy
which uses the following rules:
1. Form continuous speech segments for self and other
2. If one segment is contained within another, split the containing segment and insert
3. For partial overlaps, order by start time (tie-breaker: other first)
4. Non-overlapping segments are placed chronologically

Categories:
1. No Overlap - chronological order
2. Partial Overlap - order by start time
3. Containment - split and insert
4. Integration Window - merge/split based on gap tolerance
5. Edge Cases - boundary conditions
6. Tool Calls - proper call/result ordering (including MultiToolMessage)

See: src/experiments/tau_voice/linearization/cases.md
"""

from dataclasses import dataclass
from typing import Optional

from tau2.agent.base.streaming import LinearizationStrategy, ParticipantTick
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


@dataclass
class ExpectedMessage:
    """Expected message in linearization output."""

    role: str  # "user", "assistant", or "tool"
    content: str
    is_tool_call: bool = False


@dataclass
class LinearizationFixture:
    """A test fixture for linearization.

    Attributes:
        name: Descriptive name for the fixture
        description: What scenario this fixture tests
        ticks: List of (self_content, other_content) tuples for each tick.
               None means no speech in that tick.
               self = assistant (from assistant's perspective)
               other = user
        expected_output: List of expected messages after linearization
        strategy: Linearization strategy to use (default: CONTAINMENT_AWARE)
        integration_ticks: Integration ticks parameter (default: 1)
    """

    name: str
    description: str
    ticks: list[tuple[Optional[str], Optional[str]]]
    expected_output: list[ExpectedMessage]
    strategy: LinearizationStrategy = LinearizationStrategy.CONTAINMENT_AWARE
    integration_ticks: int = 1

    def create_participant_ticks(self) -> list[ParticipantTick]:
        """Convert tick tuples to ParticipantTick objects."""
        result = []
        for i, (self_content, other_content) in enumerate(self.ticks):
            self_chunk = None
            other_chunk = None

            if self_content is not None:
                self_chunk = AssistantMessage(
                    role="assistant",
                    content=self_content,
                    contains_speech=True,
                )

            if other_content is not None:
                other_chunk = UserMessage(
                    role="user",
                    content=other_content,
                    contains_speech=True,
                )

            result.append(
                ParticipantTick(
                    tick_id=i,
                    timestamp=f"2024-01-01T00:00:{i:02d}",
                    self_chunk=self_chunk,
                    other_chunk=other_chunk,
                )
            )
        return result


# =============================================================================
# CATEGORY 1: NO OVERLAP (Chronological Order)
# =============================================================================

CASE_1_1_OTHER_THEN_SELF = LinearizationFixture(
    name="1.1_other_then_self",
    description="Other speaks first, then self (no overlap)",
    ticks=[
        (None, "Hello "),
        (None, "there"),
        (None, None),  # silence
        ("Hi ", None),
        ("back", None),
    ],
    expected_output=[
        ExpectedMessage(role="user", content="Hello there"),
        ExpectedMessage(role="assistant", content="Hi back"),
    ],
)

CASE_1_2_SELF_THEN_OTHER = LinearizationFixture(
    name="1.2_self_then_other",
    description="Self speaks first, then other (no overlap)",
    ticks=[
        ("I need ", None),
        ("help", None),
        (None, None),  # silence
        (None, "Sure "),
        (None, "thing"),
    ],
    expected_output=[
        ExpectedMessage(role="assistant", content="I need help"),
        ExpectedMessage(role="user", content="Sure thing"),
    ],
)

# =============================================================================
# CATEGORY 2: PARTIAL OVERLAP (Order by Start Time)
# =============================================================================

CASE_2_1_OTHER_STARTS_FIRST = LinearizationFixture(
    name="2.1_other_starts_first",
    description="Other starts first, self joins and continues",
    ticks=[
        (None, "I can "),
        (None, "help you "),
        ("Actually ", "with"),  # overlap
        ("I need ", None),
        ("something else", None),
    ],
    expected_output=[
        # Other starts at tick 0, self starts at tick 2 → other first
        ExpectedMessage(role="user", content="I can help you with"),
        ExpectedMessage(role="assistant", content="Actually I need something else"),
    ],
)

CASE_2_2_SELF_STARTS_FIRST = LinearizationFixture(
    name="2.2_self_starts_first",
    description="Self starts first, other joins and continues",
    ticks=[
        ("I want ", None),
        ("to return ", None),
        ("this", "Got it "),  # overlap
        (None, "let me "),
        (None, "check"),
    ],
    expected_output=[
        # Self starts at tick 0, other starts at tick 2 → self first
        ExpectedMessage(role="assistant", content="I want to return this"),
        ExpectedMessage(role="user", content="Got it let me check"),
    ],
)

CASE_2_3_SAME_START_TIEBREAKER = LinearizationFixture(
    name="2.3_same_start_tiebreaker",
    description="Same start time - tiebreaker: other first",
    ticks=[
        ("Hello ", "Hi "),  # both start
        ("friend ", "there "),
        ("!", "!"),
    ],
    expected_output=[
        # Same start (tick 0) → tie-breaker: other first
        ExpectedMessage(role="user", content="Hi there !"),
        ExpectedMessage(role="assistant", content="Hello friend !"),
    ],
)

# =============================================================================
# CATEGORY 3: CONTAINMENT (Split and Insert)
# =============================================================================

CASE_3_1_SELF_CONTAINED_IN_OTHER = LinearizationFixture(
    name="3.1_self_contained_in_other",
    description="Self contained in other - split other, insert self",
    ticks=[
        (None, "Let me "),
        (None, "explain the "),
        ("Wait ", "process "),  # self starts
        ("please", "in detail. "),  # self ends
        (None, "First you "),
        (None, "need to "),
        (None, "log in"),
    ],
    expected_output=[
        # Self (2-3) contained in other (0-6) → split other at tick 3
        ExpectedMessage(role="user", content="Let me explain the process in detail."),
        ExpectedMessage(role="assistant", content="Wait please"),
        ExpectedMessage(role="user", content="First you need to log in"),
    ],
)

CASE_3_2_OTHER_CONTAINED_IN_SELF = LinearizationFixture(
    name="3.2_other_contained_in_self",
    description="Other contained in self - split self, insert other",
    ticks=[
        ("I want ", None),
        ("to return ", None),
        ("this item ", "Mm-hmm "),  # other starts
        ("because it's ", "I see"),  # other ends
        ("broken and ", None),
        ("doesn't work", None),
    ],
    expected_output=[
        # Other (2-3) contained in self (0-5) → split self at tick 3
        ExpectedMessage(
            role="assistant", content="I want to return this item because it's "
        ),
        ExpectedMessage(role="user", content="Mm-hmm I see"),
        ExpectedMessage(role="assistant", content="broken and doesn't work"),
    ],
)

CASE_3_3_MULTIPLE_CONTAINED = LinearizationFixture(
    name="3.3_multiple_contained",
    description="Multiple contained segments",
    ticks=[
        (None, "So the "),
        ("Wait", "return "),  # first self segment
        (None, "process "),
        ("Stop", "requires "),  # second self segment
        (None, "you to "),
        (None, "log in"),
    ],
    expected_output=[
        # Two self segments each contained in other
        ExpectedMessage(role="user", content="So the return"),
        ExpectedMessage(role="assistant", content="Wait"),
        ExpectedMessage(role="user", content="process requires"),
        ExpectedMessage(role="assistant", content="Stop"),
        ExpectedMessage(role="user", content="you to log in"),
    ],
)

# =============================================================================
# CATEGORY 4: INTEGRATION WINDOW EFFECTS
# =============================================================================

CASE_4_1_GAP_MERGES = LinearizationFixture(
    name="4.1_gap_merges",
    description="Gap within integration window merges segments",
    integration_ticks=3,
    ticks=[
        ("I want ", None),
        ("to cancel ", None),
        (None, None),  # silence
        (None, None),  # silence (gap of 2 ticks)
        ("my order", None),
    ],
    expected_output=[
        # Gap of 2 ticks ≤ 3 → segments merge
        ExpectedMessage(role="assistant", content="I want to cancel my order"),
    ],
)

CASE_4_2_GAP_SPLITS = LinearizationFixture(
    name="4.2_gap_splits",
    description="Gap exceeds integration window keeps segments separate",
    integration_ticks=1,
    ticks=[
        ("I want ", None),
        ("to cancel", None),
        (None, None),  # silence
        (None, None),  # silence (gap of 2 ticks > 1)
        ("my order", None),
    ],
    expected_output=[
        # Gap of 2 ticks > 1 → segments stay separate
        ExpectedMessage(role="assistant", content="I want to cancel"),
        ExpectedMessage(role="assistant", content="my order"),
    ],
)

# =============================================================================
# CATEGORY 5: EDGE CASES
# =============================================================================

CASE_5_1_EMPTY_INPUT = LinearizationFixture(
    name="5.1_empty_input",
    description="Empty input returns empty output",
    ticks=[],
    expected_output=[],
)

CASE_5_2_SINGLE_TICK_BOTH = LinearizationFixture(
    name="5.2_single_tick_both",
    description="Single tick with both speaking",
    ticks=[
        ("Hi", "Hello"),
    ],
    expected_output=[
        # Same start, same end → tie-breaker: other first
        ExpectedMessage(role="user", content="Hello"),
        ExpectedMessage(role="assistant", content="Hi"),
    ],
)

CASE_5_3_ONLY_SELF = LinearizationFixture(
    name="5.3_only_self",
    description="Only self speaks",
    ticks=[
        ("Hello ", None),
        ("world", None),
    ],
    expected_output=[
        ExpectedMessage(role="assistant", content="Hello world"),
    ],
)

CASE_5_4_ONLY_OTHER = LinearizationFixture(
    name="5.4_only_other",
    description="Only other speaks",
    ticks=[
        (None, "Hello "),
        (None, "there"),
    ],
    expected_output=[
        ExpectedMessage(role="user", content="Hello there"),
    ],
)

# =============================================================================
# CATEGORY 6: TOOL CALLS
# =============================================================================


@dataclass
class ToolCallFixture(LinearizationFixture):
    """Fixture that supports tool calls with custom tick creation."""

    raw_ticks: Optional[list[ParticipantTick]] = None

    def create_participant_ticks(self) -> list[ParticipantTick]:
        """Use raw_ticks if provided, otherwise fall back to parent."""
        if self.raw_ticks is not None:
            return self.raw_ticks
        return super().create_participant_ticks()


def _create_tool_call_msg(call_id: str, function_name: str) -> AssistantMessage:
    """Helper to create a tool call message."""
    return AssistantMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=call_id, name=function_name, arguments={})],
        contains_speech=False,
    )


def _create_multi_tool_call_msg(
    call_ids: list[str], function_names: list[str]
) -> AssistantMessage:
    """Helper to create a message with multiple tool calls."""
    return AssistantMessage(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(id=cid, name=fname, arguments={})
            for cid, fname in zip(call_ids, function_names)
        ],
        contains_speech=False,
    )


def _create_tool_result_msg(call_id: str, result: str) -> ToolMessage:
    """Helper to create a tool result message."""
    return ToolMessage(role="tool", content=result, tool_call_id=call_id, id=call_id)


def _create_multi_tool_result_msg(
    call_ids: list[str], results: list[str]
) -> MultiToolMessage:
    """Helper to create a MultiToolMessage with multiple results."""
    return MultiToolMessage(
        role="tool",
        tool_messages=[
            ToolMessage(role="tool", content=result, tool_call_id=cid, id=cid)
            for cid, result in zip(call_ids, results)
        ],
    )


CASE_6_1_BASIC_TOOL_CALL = ToolCallFixture(
    name="6.1_basic_tool_call",
    description="Basic tool call followed by result",
    ticks=[],  # Will use raw_ticks instead
    raw_ticks=[
        ParticipantTick(
            tick_id=0,
            timestamp="2024-01-01T00:00:00",
            self_chunk=AssistantMessage(
                role="assistant", content="Let me check", contains_speech=True
            ),
            other_chunk=None,
        ),
        ParticipantTick(
            tick_id=1,
            timestamp="2024-01-01T00:00:01",
            self_chunk=_create_tool_call_msg("call_1", "get_order"),
            other_chunk=None,
        ),
        ParticipantTick(
            tick_id=2,
            timestamp="2024-01-01T00:00:02",
            self_chunk=None,
            other_chunk=_create_tool_result_msg("call_1", '{"status": "shipped"}'),
        ),
        ParticipantTick(
            tick_id=3,
            timestamp="2024-01-01T00:00:03",
            self_chunk=None,
            other_chunk=UserMessage(
                role="user", content="Your order has shipped", contains_speech=True
            ),
        ),
    ],
    expected_output=[
        ExpectedMessage(role="assistant", content="Let me check"),
        ExpectedMessage(role="assistant", content="", is_tool_call=True),
        ExpectedMessage(role="tool", content='{"status": "shipped"}'),
        ExpectedMessage(role="user", content="Your order has shipped"),
    ],
)

CASE_6_2_MULTIPLE_TOOL_CALLS = ToolCallFixture(
    name="6.2_multiple_tool_calls",
    description="Multiple simultaneous tool calls with MultiToolMessage result",
    ticks=[],  # Will use raw_ticks instead
    raw_ticks=[
        ParticipantTick(
            tick_id=0,
            timestamp="2024-01-01T00:00:00",
            self_chunk=AssistantMessage(
                role="assistant",
                content="Creating both tasks",
                contains_speech=True,
            ),
            other_chunk=None,
        ),
        ParticipantTick(
            tick_id=1,
            timestamp="2024-01-01T00:00:01",
            self_chunk=_create_multi_tool_call_msg(
                ["call_1", "call_2"], ["create_task", "create_task"]
            ),
            other_chunk=None,
        ),
        ParticipantTick(
            tick_id=2,
            timestamp="2024-01-01T00:00:02",
            self_chunk=None,
            other_chunk=_create_multi_tool_result_msg(
                ["call_1", "call_2"], ['{"id": "task_1"}', '{"id": "task_2"}']
            ),
        ),
        ParticipantTick(
            tick_id=3,
            timestamp="2024-01-01T00:00:03",
            self_chunk=None,
            other_chunk=UserMessage(
                role="user", content="Both tasks created", contains_speech=True
            ),
        ),
    ],
    expected_output=[
        ExpectedMessage(role="assistant", content="Creating both tasks"),
        ExpectedMessage(role="assistant", content="", is_tool_call=True),
        ExpectedMessage(role="tool", content='{"id": "task_1"}'),
        ExpectedMessage(role="tool", content='{"id": "task_2"}'),
        ExpectedMessage(role="user", content="Both tasks created"),
    ],
)

CASE_6_3_TOOL_CALL_WITH_OVERLAP = ToolCallFixture(
    name="6.3_tool_call_with_overlap",
    description="Tool call with overlapping speech maintains order",
    ticks=[],  # Will use raw_ticks instead
    raw_ticks=[
        ParticipantTick(
            tick_id=0,
            timestamp="2024-01-01T00:00:00",
            self_chunk=AssistantMessage(
                role="assistant", content="Let me check", contains_speech=True
            ),
            other_chunk=None,
        ),
        ParticipantTick(
            tick_id=1,
            timestamp="2024-01-01T00:00:01",
            self_chunk=AssistantMessage(
                role="assistant", content=" your order", contains_speech=True
            ),
            other_chunk=UserMessage(
                role="user", content="What's my ", contains_speech=True
            ),
        ),
        ParticipantTick(
            tick_id=2,
            timestamp="2024-01-01T00:00:02",
            self_chunk=None,
            other_chunk=UserMessage(
                role="user", content="status please", contains_speech=True
            ),
        ),
        ParticipantTick(
            tick_id=3,
            timestamp="2024-01-01T00:00:03",
            self_chunk=_create_tool_call_msg("call_1", "get_order"),
            other_chunk=None,
        ),
        ParticipantTick(
            tick_id=4,
            timestamp="2024-01-01T00:00:04",
            self_chunk=None,
            other_chunk=_create_tool_result_msg("call_1", '{"status": "shipped"}'),
        ),
    ],
    expected_output=[
        # Self starts first (tick 0), other joins at tick 1 → self first (partial overlap)
        ExpectedMessage(role="assistant", content="Let me check your order"),
        ExpectedMessage(role="user", content="What's my status please"),
        ExpectedMessage(role="assistant", content="", is_tool_call=True),
        ExpectedMessage(role="tool", content='{"status": "shipped"}'),
    ],
)

# =============================================================================
# All fixtures
# =============================================================================
FIXTURES = [
    # Category 1: No Overlap
    CASE_1_1_OTHER_THEN_SELF,
    CASE_1_2_SELF_THEN_OTHER,
    # Category 2: Partial Overlap
    CASE_2_1_OTHER_STARTS_FIRST,
    CASE_2_2_SELF_STARTS_FIRST,
    CASE_2_3_SAME_START_TIEBREAKER,
    # Category 3: Containment
    CASE_3_1_SELF_CONTAINED_IN_OTHER,
    CASE_3_2_OTHER_CONTAINED_IN_SELF,
    CASE_3_3_MULTIPLE_CONTAINED,
    # Category 4: Integration Window
    CASE_4_1_GAP_MERGES,
    CASE_4_2_GAP_SPLITS,
    # Category 5: Edge Cases
    CASE_5_1_EMPTY_INPUT,
    CASE_5_2_SINGLE_TICK_BOTH,
    CASE_5_3_ONLY_SELF,
    CASE_5_4_ONLY_OTHER,
    # Category 6: Tool Calls
    CASE_6_1_BASIC_TOOL_CALL,
    CASE_6_2_MULTIPLE_TOOL_CALLS,
    CASE_6_3_TOOL_CALL_WITH_OVERLAP,
]


def get_fixture_by_name(name: str) -> LinearizationFixture:
    """Get a fixture by name."""
    for fixture in FIXTURES:
        if fixture.name == name:
            return fixture
    raise ValueError(f"Unknown fixture: {name}")
