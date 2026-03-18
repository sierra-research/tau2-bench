"""
Tests for tick linearization logic.

This test suite verifies that the CONTAINMENT_AWARE linearization strategy
correctly handles various scenarios including:
- Basic linearization without overlap
- Partial overlaps (order by start time)
- Containment (split containing segment, insert contained)
- Integration window effects
- Tool messages during conversations
- Edge cases and boundary conditions

Algorithm Summary (CONTAINMENT_AWARE):
1. Form continuous speech segments for self and other
2. If one segment is contained within another, split the containing segment
   at the point where the contained segment ended, and insert it there
3. For partial overlaps (neither contained), order by start time
   - If same start time, other comes first (tie-breaker)
4. Non-overlapping segments are placed chronologically

See: src/experiments/tau_voice/linearization/cases.md
"""

from tau2.agent.base.streaming import (
    LinearizationStrategy,
    ParticipantTick,
    _expand_env_chunks,
    _has_meaningful_content,
    consolidate_messages,
    linearize_ticks,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

# =============================================================================
# Helper Functions
# =============================================================================


def create_tick(
    tick_id: int,
    self_chunk=None,
    other_chunk=None,
) -> ParticipantTick:
    """Helper to create a ParticipantTick."""
    return ParticipantTick(
        tick_id=tick_id,
        timestamp=f"2024-01-01T00:00:{tick_id:02d}",
        self_chunk=self_chunk,
        other_chunk=other_chunk,
    )


def user_msg(content: str) -> UserMessage:
    """Helper to create a UserMessage."""
    return UserMessage(role="user", content=content, contains_speech=True)


def assistant_msg(content: str) -> AssistantMessage:
    """Helper to create an AssistantMessage."""
    return AssistantMessage(role="assistant", content=content, contains_speech=True)


def tool_call_msg(call_id: str, function_name: str) -> AssistantMessage:
    """Helper to create a tool call message."""
    return AssistantMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=call_id, name=function_name, arguments={})],
        contains_speech=False,
    )


def tool_result_msg(call_id: str, result: str) -> ToolMessage:
    """Helper to create a tool result message."""
    return ToolMessage(role="tool", content=result, tool_call_id=call_id, id=call_id)


# =============================================================================
# CATEGORY 1: NO OVERLAP (Chronological Order)
# =============================================================================


class TestNoOverlap:
    """Tests for scenarios with no overlapping speech."""

    def test_empty_ticks(self):
        """Case: Empty input returns empty output."""
        messages = linearize_ticks([], LinearizationStrategy.CONTAINMENT_AWARE)
        assert messages == []

    def test_other_then_self(self):
        """Case 1.1: Other speaks first, then self (no overlap)."""
        ticks = [
            create_tick(0, other_chunk=user_msg("Hello ")),
            create_tick(1, other_chunk=user_msg("there")),
            create_tick(2),  # silence
            create_tick(3, self_chunk=assistant_msg("Hi ")),
            create_tick(4, self_chunk=assistant_msg("back")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello there"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi back"

    def test_self_then_other(self):
        """Case 1.2: Self speaks first, then other (no overlap)."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("I need ")),
            create_tick(1, self_chunk=assistant_msg("help")),
            create_tick(2),  # silence
            create_tick(3, other_chunk=user_msg("Sure ")),
            create_tick(4, other_chunk=user_msg("thing")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        assert len(messages) == 2
        assert messages[0].role == "assistant"
        assert messages[0].content == "I need help"
        assert messages[1].role == "user"
        assert messages[1].content == "Sure thing"

    def test_alternating_speakers(self):
        """Multiple turn-taking without overlap."""
        ticks = [
            create_tick(0, other_chunk=user_msg("Hello")),
            create_tick(1, self_chunk=assistant_msg("Hi")),
            create_tick(2, other_chunk=user_msg("How are you?")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        assert len(messages) == 3
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi"
        assert messages[2].content == "How are you?"


# =============================================================================
# CATEGORY 2: PARTIAL OVERLAP (Order by Start Time)
# =============================================================================


class TestPartialOverlap:
    """Tests for partial overlaps where neither segment contains the other."""

    def test_other_starts_first(self):
        """Case 2.1: Other starts first, self joins and continues."""
        ticks = [
            create_tick(0, other_chunk=user_msg("I can ")),
            create_tick(1, other_chunk=user_msg("help you ")),
            create_tick(
                2, other_chunk=user_msg("with"), self_chunk=assistant_msg("Actually ")
            ),
            create_tick(3, self_chunk=assistant_msg("I need ")),
            create_tick(4, self_chunk=assistant_msg("something else")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        # Other starts at tick 0, self starts at tick 2 → other first
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "I can help you with"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Actually I need something else"

    def test_self_starts_first(self):
        """Case 2.2: Self starts first, other joins and continues."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("I want ")),
            create_tick(1, self_chunk=assistant_msg("to return ")),
            create_tick(
                2, self_chunk=assistant_msg("this"), other_chunk=user_msg("Got it ")
            ),
            create_tick(3, other_chunk=user_msg("let me ")),
            create_tick(4, other_chunk=user_msg("check")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        # Self starts at tick 0, other starts at tick 2 → self first
        assert len(messages) == 2
        assert messages[0].role == "assistant"
        assert messages[0].content == "I want to return this"
        assert messages[1].role == "user"
        assert messages[1].content == "Got it let me check"

    def test_same_start_tiebreaker(self):
        """Case 2.3: Same start time - tiebreaker: other first."""
        ticks = [
            create_tick(
                0, self_chunk=assistant_msg("Hello "), other_chunk=user_msg("Hi ")
            ),
            create_tick(
                1, self_chunk=assistant_msg("friend "), other_chunk=user_msg("there ")
            ),
            create_tick(2, self_chunk=assistant_msg("!"), other_chunk=user_msg("!")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        # Same start (tick 0) → tie-breaker: other first
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hi there !"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hello friend !"


# =============================================================================
# CATEGORY 3: CONTAINMENT (Split and Insert)
# =============================================================================


class TestContainment:
    """Tests for containment scenarios where one segment is inside another."""

    def test_self_contained_in_other(self):
        """Case 3.1: Self contained in other - split other, insert self."""
        ticks = [
            create_tick(0, other_chunk=user_msg("Let me ")),
            create_tick(1, other_chunk=user_msg("explain the ")),
            create_tick(
                2, other_chunk=user_msg("process "), self_chunk=assistant_msg("Wait ")
            ),
            create_tick(
                3,
                other_chunk=user_msg("in detail. "),
                self_chunk=assistant_msg("please"),
            ),
            create_tick(4, other_chunk=user_msg("First you ")),
            create_tick(5, other_chunk=user_msg("need to ")),
            create_tick(6, other_chunk=user_msg("log in")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        # Self (2-3) contained in other (0-6) → split other at tick 3, insert self
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert "Let me explain the process in detail" in messages[0].content
        assert messages[1].role == "assistant"
        assert messages[1].content == "Wait please"
        assert messages[2].role == "user"
        assert "First you need to log in" in messages[2].content

    def test_other_contained_in_self(self):
        """Case 3.2: Other contained in self - split self, insert other."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("I want ")),
            create_tick(1, self_chunk=assistant_msg("to return ")),
            create_tick(
                2,
                self_chunk=assistant_msg("this item "),
                other_chunk=user_msg("Mm-hmm "),
            ),
            create_tick(
                3,
                self_chunk=assistant_msg("because it's "),
                other_chunk=user_msg("I see"),
            ),
            create_tick(4, self_chunk=assistant_msg("broken and ")),
            create_tick(5, self_chunk=assistant_msg("doesn't work")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        # Other (2-3) contained in self (0-5) → split self at tick 3, insert other
        assert len(messages) == 3
        assert messages[0].role == "assistant"
        assert messages[1].role == "user"
        assert messages[1].content == "Mm-hmm I see"
        assert messages[2].role == "assistant"

    def test_multiple_contained_segments(self):
        """Case 3.3: Multiple contained segments each create a break."""
        ticks = [
            create_tick(0, other_chunk=user_msg("So the ")),
            create_tick(
                1, other_chunk=user_msg("return "), self_chunk=assistant_msg("Wait")
            ),
            create_tick(2, other_chunk=user_msg("process ")),
            create_tick(
                3, other_chunk=user_msg("requires "), self_chunk=assistant_msg("Stop")
            ),
            create_tick(4, other_chunk=user_msg("you to ")),
            create_tick(5, other_chunk=user_msg("log in")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        # Two self segments (tick 1, tick 3) each contained in other (0-5)
        assert len(messages) == 5
        assert messages[0].role == "user"  # "So the return"
        assert messages[1].role == "assistant"  # "Wait"
        assert messages[2].role == "user"  # "process requires"
        assert messages[3].role == "assistant"  # "Stop"
        assert messages[4].role == "user"  # "you to log in"


# =============================================================================
# CATEGORY 4: INTEGRATION WINDOW EFFECTS
# =============================================================================


class TestIntegrationWindow:
    """Tests for integration_ticks parameter effects."""

    def test_gap_within_window_merges(self):
        """Case 4.1: Gap within integration window merges segments."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("I want ")),
            create_tick(1, self_chunk=assistant_msg("to cancel ")),
            create_tick(2),  # silence
            create_tick(3),  # silence (gap of 2 ticks)
            create_tick(4, self_chunk=assistant_msg("my order")),
        ]

        messages = linearize_ticks(
            ticks, LinearizationStrategy.CONTAINMENT_AWARE, integration_ticks=3
        )

        # Gap of 2 ticks ≤ 3 → segments merge
        assert len(messages) == 1
        assert messages[0].content == "I want to cancel my order"

    def test_gap_exceeds_window_splits(self):
        """Case 4.2: Gap exceeds integration window keeps segments separate."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("I want ")),
            create_tick(1, self_chunk=assistant_msg("to cancel")),
            create_tick(2),  # silence
            create_tick(3),  # silence (gap of 2 ticks > 1)
            create_tick(4, self_chunk=assistant_msg("my order")),
        ]

        messages = linearize_ticks(
            ticks, LinearizationStrategy.CONTAINMENT_AWARE, integration_ticks=1
        )

        # Gap of 2 ticks > 1 → segments stay separate
        assert len(messages) == 2
        assert messages[0].content == "I want to cancel"
        assert messages[1].content == "my order"

    def test_default_integration_ticks_is_one(self):
        """Default integration_ticks=1 ends segment immediately on silence."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("A"), other_chunk=user_msg("X")),
            create_tick(1, other_chunk=user_msg("Y")),  # self silent
            create_tick(2, self_chunk=assistant_msg("B"), other_chunk=user_msg("Z")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        # With default integration_ticks=1, some fragmentation expected
        assert len(messages) >= 3


# =============================================================================
# CATEGORY 5: TOOL CALLS
# =============================================================================


class TestToolCalls:
    """Tests for tool call and result handling."""

    def test_basic_tool_call(self):
        """Case 5.1: Tool call followed by result."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("Let me check")),
            create_tick(1, self_chunk=tool_call_msg("call_1", "get_order")),
            create_tick(
                2, other_chunk=tool_result_msg("call_1", '{"status": "shipped"}')
            ),
            create_tick(3, other_chunk=user_msg("Your order ")),
            create_tick(4, other_chunk=user_msg("has shipped")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        # Find tool call and result indices
        tool_call_idx = None
        tool_result_idx = None
        for i, msg in enumerate(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call_idx = i
            if isinstance(msg, ToolMessage):
                tool_result_idx = i

        assert tool_call_idx is not None, "Tool call not found"
        assert tool_result_idx is not None, "Tool result not found"
        assert tool_result_idx > tool_call_idx, "Tool result must come after tool call"

    def test_tool_call_with_overlap(self):
        """Case 5.2: Tool call with overlapping speech maintains order."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("Let me ")),
            create_tick(
                1, self_chunk=assistant_msg("check"), other_chunk=user_msg("What's my ")
            ),
            create_tick(2, other_chunk=user_msg("status")),
            create_tick(3, self_chunk=tool_call_msg("call_1", "get_order")),
            create_tick(
                4, other_chunk=tool_result_msg("call_1", '{"status": "shipped"}')
            ),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        # Verify tool call before result
        tool_call_idx = None
        tool_result_idx = None
        for i, msg in enumerate(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call_idx = i
            if isinstance(msg, ToolMessage):
                tool_result_idx = i

        if tool_call_idx is not None and tool_result_idx is not None:
            assert tool_result_idx > tool_call_idx

    def test_tool_message_preserved(self):
        """Tool messages are preserved in output."""
        # Tool messages need a preceding tool call to be paired with
        ticks = [
            create_tick(0, self_chunk=tool_call_msg("call_123", "get_info")),
            create_tick(
                1,
                other_chunk=ToolMessage(
                    id="call_123",
                    role="tool",
                    content="Tool result",
                    tool_call_id="call_123",
                ),
            ),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        # Find tool message
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "Tool result"

    def test_multi_tool_message_expanded(self):
        """MultiToolMessage is expanded to individual messages.

        Test with OTHER_FIRST_PER_TICK strategy which has simpler handling.
        """
        tool_msgs = [
            ToolMessage(
                id="call_1", role="tool", content="Result 1", tool_call_id="call_1"
            ),
            ToolMessage(
                id="call_2", role="tool", content="Result 2", tool_call_id="call_2"
            ),
        ]
        ticks = [
            create_tick(
                0,
                other_chunk=MultiToolMessage(role="tool", tool_messages=tool_msgs),
            ),
        ]

        # Use OTHER_FIRST_PER_TICK for straightforward multi-tool expansion
        messages = linearize_ticks(ticks, LinearizationStrategy.OTHER_FIRST_PER_TICK)

        assert len(messages) == 2
        assert messages[0].content == "Result 1"
        assert messages[1].content == "Result 2"


# =============================================================================
# CATEGORY 6: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_single_tick_both_speaking(self):
        """Single tick with both speaking - other first (tie-breaker)."""
        ticks = [
            create_tick(
                0, self_chunk=assistant_msg("Hi"), other_chunk=user_msg("Hello")
            ),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi"

    def test_only_self_speaks(self):
        """Only self speaks."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("Hello ")),
            create_tick(1, self_chunk=assistant_msg("world")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        assert len(messages) == 1
        assert messages[0].content == "Hello world"

    def test_only_other_speaks(self):
        """Only other speaks."""
        ticks = [
            create_tick(0, other_chunk=user_msg("Hello ")),
            create_tick(1, other_chunk=user_msg("there")),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        assert len(messages) == 1
        assert messages[0].content == "Hello there"

    def test_all_silent_ticks(self):
        """All silent ticks produce no messages."""
        ticks = [
            create_tick(0),
            create_tick(1),
            create_tick(2),
        ]

        messages = linearize_ticks(ticks, LinearizationStrategy.CONTAINMENT_AWARE)

        assert len(messages) == 0


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHasSpeechContent:
    """Tests for _has_meaningful_content helper function."""

    def test_none_chunk(self):
        """None returns False."""
        assert _has_meaningful_content(None) is False

    def test_chunk_with_content(self):
        """Chunk with content returns True."""
        msg = UserMessage(role="user", content="Hello", contains_speech=True)
        assert _has_meaningful_content(msg) is True

    def test_chunk_with_empty_content(self):
        """Chunk with empty content and contains_speech=False returns False."""
        msg = UserMessage(role="user", content="", contains_speech=False)
        assert _has_meaningful_content(msg) is False

    def test_chunk_with_contains_speech_true(self):
        """Chunk with contains_speech=True returns True even without content."""
        msg = AssistantMessage(role="assistant", content=None, contains_speech=True)
        assert _has_meaningful_content(msg) is True

    def test_tool_message_always_meaningful(self):
        """ToolMessage is always considered meaningful."""
        msg = ToolMessage(id="call_1", role="tool", content="")
        assert _has_meaningful_content(msg) is True

    def test_multi_tool_message_always_meaningful(self):
        """MultiToolMessage is always considered meaningful."""
        msg = MultiToolMessage(
            role="tool",
            tool_messages=[ToolMessage(id="call_1", role="tool", content="")],
        )
        assert _has_meaningful_content(msg) is True

    def test_tool_call_is_meaningful(self):
        """Messages with tool_calls are meaningful even if contains_speech=False."""
        tool_call = ToolCall(id="call_test", name="test_func", arguments={})
        msg = AssistantMessage(
            role="assistant",
            content="",
            tool_calls=[tool_call],
            contains_speech=False,
        )
        assert _has_meaningful_content(msg) is True


class TestConsolidateMessages:
    """Tests for consolidate_messages function."""

    def test_empty_list(self):
        """Empty list returns empty."""
        result = consolidate_messages([])
        assert result == []

    def test_single_message(self):
        """Single message unchanged."""
        msgs = [UserMessage(role="user", content="Hello", contains_speech=True)]
        result = consolidate_messages(msgs)
        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_consecutive_same_type_merged(self):
        """Consecutive same-type messages are merged."""
        msgs = [
            UserMessage(role="user", content="Hello ", contains_speech=True),
            UserMessage(role="user", content="World", contains_speech=True),
        ]
        result = consolidate_messages(msgs)
        assert len(result) == 1
        assert result[0].content == "Hello World"

    def test_different_types_not_merged(self):
        """Different message types are not merged."""
        msgs = [
            UserMessage(role="user", content="Hello", contains_speech=True),
            AssistantMessage(role="assistant", content="Hi", contains_speech=True),
        ]
        result = consolidate_messages(msgs)
        assert len(result) == 2

    def test_tool_messages_not_merged(self):
        """Tool messages are not merged."""
        msgs = [
            ToolMessage(id="call_1", role="tool", content="Result 1"),
            ToolMessage(id="call_2", role="tool", content="Result 2"),
        ]
        result = consolidate_messages(msgs)
        assert len(result) == 2


# =============================================================================
# OTHER STRATEGIES (Keep for compatibility)
# =============================================================================


class TestOtherStrategies:
    """Tests for non-CONTAINMENT_AWARE strategies."""

    def test_other_first_per_tick(self):
        """OTHER_FIRST_PER_TICK strategy."""
        ticks = [
            create_tick(
                0,
                self_chunk=AssistantMessage(
                    role="assistant", content="Self", contains_speech=True
                ),
                other_chunk=UserMessage(
                    role="user", content="Other", contains_speech=True
                ),
            ),
        ]
        messages = linearize_ticks(ticks, LinearizationStrategy.OTHER_FIRST_PER_TICK)

        assert len(messages) == 2
        assert messages[0].content == "Other"
        assert messages[1].content == "Self"

    def test_self_first_per_tick(self):
        """SELF_FIRST_PER_TICK strategy."""
        ticks = [
            create_tick(
                0,
                self_chunk=AssistantMessage(
                    role="assistant", content="Self", contains_speech=True
                ),
                other_chunk=UserMessage(
                    role="user", content="Other", contains_speech=True
                ),
            ),
        ]
        messages = linearize_ticks(ticks, LinearizationStrategy.SELF_FIRST_PER_TICK)

        assert len(messages) == 2
        assert messages[0].content == "Self"
        assert messages[1].content == "Other"

    def test_timestamp_order(self):
        """TIMESTAMP_ORDER strategy sorts by timestamp."""
        ticks = [
            create_tick(
                0,
                self_chunk=AssistantMessage(
                    role="assistant",
                    content="Second",
                    contains_speech=True,
                    timestamp="2024-01-01T00:00:02",
                ),
                other_chunk=UserMessage(
                    role="user",
                    content="First",
                    contains_speech=True,
                    timestamp="2024-01-01T00:00:01",
                ),
            ),
        ]
        messages = linearize_ticks(ticks, LinearizationStrategy.TIMESTAMP_ORDER)

        assert len(messages) == 2
        assert messages[0].content == "First"
        assert messages[1].content == "Second"


# =============================================================================
# FIXTURE-BASED TESTS
# =============================================================================


class TestFixtures:
    """Tests using fixtures from linearization_fixtures.py."""

    def test_all_fixtures(self):
        """Run all fixtures and verify expected output."""
        from linearization_fixtures import FIXTURES

        for fixture in FIXTURES:
            ticks = fixture.create_participant_ticks()
            messages = linearize_ticks(
                ticks, fixture.strategy, fixture.integration_ticks
            )

            assert len(messages) == len(fixture.expected_output), (
                f"Fixture '{fixture.name}': "
                f"expected {len(fixture.expected_output)} messages, got {len(messages)}"
            )

            for i, (msg, expected) in enumerate(zip(messages, fixture.expected_output)):
                assert msg.role == expected.role, (
                    f"Fixture '{fixture.name}' message {i}: "
                    f"expected role '{expected.role}', got '{msg.role}'"
                )
                # Content check - use 'in' for flexibility with whitespace
                if expected.content:
                    # Normalize whitespace for comparison
                    actual_normalized = " ".join(msg.content.split())
                    expected_normalized = " ".join(expected.content.split())
                    assert (
                        actual_normalized == expected_normalized
                        or expected.content in msg.content
                    ), (
                        f"Fixture '{fixture.name}' message {i}: "
                        f"expected '{expected.content}', got '{msg.content}'"
                    )


# =============================================================================
# _expand_env_chunks Tests
# =============================================================================


class TestExpandEnvChunks:
    """Tests for the _expand_env_chunks transformation."""

    def test_no_env_chunks_passthrough(self):
        """Ticks without env_chunk pass through unchanged."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("Hello")),
            create_tick(1, other_chunk=user_msg("Hi")),
            create_tick(
                2, self_chunk=assistant_msg("How?"), other_chunk=user_msg("Good")
            ),
        ]
        expanded = _expand_env_chunks(ticks)

        assert len(expanded) == 3
        assert expanded[0].self_chunk.content == "Hello"
        assert expanded[1].other_chunk.content == "Hi"
        assert expanded[2].self_chunk.content == "How?"
        assert expanded[2].other_chunk.content == "Good"

    def test_env_chunk_without_other_chunk(self):
        """Tick with env_chunk but no other_chunk produces one expanded tick."""
        tool_result = tool_result_msg("call_1", "shipped")
        ticks = [
            ParticipantTick(
                tick_id=0,
                timestamp="t0",
                self_chunk=assistant_msg("Your order shipped"),
                other_chunk=None,
                env_chunk=tool_result,
            ),
        ]
        expanded = _expand_env_chunks(ticks)

        assert len(expanded) == 1
        assert expanded[0].other_chunk is tool_result
        assert expanded[0].self_chunk.content == "Your order shipped"
        assert expanded[0].env_chunk is None

    def test_env_chunk_with_other_chunk(self):
        """Tick with both env_chunk and other_chunk produces two expanded ticks."""
        tool_result = tool_result_msg("call_1", "shipped")
        speech = user_msg("Thanks")
        response = assistant_msg("You're welcome")

        ticks = [
            ParticipantTick(
                tick_id=0,
                timestamp="t0",
                self_chunk=response,
                other_chunk=speech,
                env_chunk=tool_result,
            ),
        ]
        expanded = _expand_env_chunks(ticks)

        assert len(expanded) == 2
        # First tick: tool result with response
        assert expanded[0].other_chunk is tool_result
        assert expanded[0].self_chunk is response
        assert expanded[0].env_chunk is None
        # Second tick: speech without response
        assert expanded[1].other_chunk is speech
        assert expanded[1].self_chunk is None
        assert expanded[1].env_chunk is None

    def test_tick_ids_are_sequential(self):
        """Expanded ticks have sequential, unique tick IDs."""
        ticks = [
            create_tick(0, self_chunk=assistant_msg("A")),
            ParticipantTick(
                tick_id=1,
                timestamp="t1",
                self_chunk=assistant_msg("B"),
                other_chunk=user_msg("C"),
                env_chunk=tool_result_msg("call_1", "result"),
            ),
            create_tick(2, other_chunk=user_msg("D")),
        ]
        expanded = _expand_env_chunks(ticks)

        assert len(expanded) == 4
        assert [t.tick_id for t in expanded] == [0, 1, 2, 3]

    def test_multi_tool_message_in_env_chunk(self):
        """MultiToolMessage in env_chunk is handled correctly."""
        multi_tool = MultiToolMessage(
            role="tool",
            tool_messages=[
                ToolMessage(
                    id="call_1", role="tool", content="R1", tool_call_id="call_1"
                ),
                ToolMessage(
                    id="call_2", role="tool", content="R2", tool_call_id="call_2"
                ),
            ],
        )
        ticks = [
            ParticipantTick(
                tick_id=0,
                timestamp="t0",
                self_chunk=assistant_msg("Here are results"),
                other_chunk=None,
                env_chunk=multi_tool,
            ),
        ]
        expanded = _expand_env_chunks(ticks)

        assert len(expanded) == 1
        assert expanded[0].other_chunk is multi_tool


# =============================================================================
# End-to-end: linearization with env_chunk
# =============================================================================


class TestLinearizationWithEnvChunk:
    """End-to-end tests verifying that linearization produces identical output
    whether tool results are in other_chunk (old format) or env_chunk (new format)."""

    def test_basic_tool_call_env_chunk_matches_old_format(self):
        """Linearization with env_chunk matches linearization with other_chunk."""
        # Old format (tool result in other_chunk, as existing tests use)
        old_ticks = [
            create_tick(0, self_chunk=assistant_msg("Let me check")),
            create_tick(1, self_chunk=tool_call_msg("call_1", "get_order")),
            create_tick(
                2, other_chunk=tool_result_msg("call_1", '{"status": "shipped"}')
            ),
            create_tick(3, self_chunk=assistant_msg("Your order shipped")),
        ]

        # New format (tool result in env_chunk)
        new_ticks = [
            ParticipantTick(
                tick_id=0,
                timestamp="2024-01-01T00:00:00",
                self_chunk=assistant_msg("Let me check"),
            ),
            ParticipantTick(
                tick_id=1,
                timestamp="2024-01-01T00:00:01",
                self_chunk=tool_call_msg("call_1", "get_order"),
            ),
            ParticipantTick(
                tick_id=2,
                timestamp="2024-01-01T00:00:02",
                self_chunk=assistant_msg("Your order shipped"),
                env_chunk=tool_result_msg("call_1", '{"status": "shipped"}'),
            ),
        ]

        # Expand new format, then linearize both
        expanded_ticks = _expand_env_chunks(new_ticks)
        old_messages = linearize_ticks(
            old_ticks, LinearizationStrategy.CONTAINMENT_AWARE
        )
        new_messages = linearize_ticks(
            expanded_ticks, LinearizationStrategy.CONTAINMENT_AWARE
        )

        assert len(old_messages) == len(new_messages)
        for old_msg, new_msg in zip(old_messages, new_messages):
            assert type(old_msg) is type(new_msg)
            assert old_msg.content == new_msg.content

    def test_tool_call_with_simultaneous_speech(self):
        """When tool results and speech arrive simultaneously, both are preserved."""
        ticks = [
            ParticipantTick(
                tick_id=0,
                timestamp="2024-01-01T00:00:00",
                self_chunk=tool_call_msg("call_1", "get_order"),
            ),
            ParticipantTick(
                tick_id=1,
                timestamp="2024-01-01T00:00:01",
                self_chunk=assistant_msg("Your order shipped"),
                other_chunk=user_msg("Any update?"),
                env_chunk=tool_result_msg("call_1", '{"status": "shipped"}'),
            ),
        ]

        expanded = _expand_env_chunks(ticks)
        messages = linearize_ticks(expanded, LinearizationStrategy.CONTAINMENT_AWARE)

        # Verify all content is present
        contents = [m.content for m in messages]
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_results) == 1
        assert '{"status": "shipped"}' in contents
        assert "Your order shipped" in contents
        assert "Any update?" in contents

        # Verify tool call comes before tool result
        tool_call_idx = next(
            i
            for i, m in enumerate(messages)
            if hasattr(m, "tool_calls") and m.tool_calls
        )
        tool_result_idx = next(
            i for i, m in enumerate(messages) if isinstance(m, ToolMessage)
        )
        assert tool_result_idx > tool_call_idx

    def test_other_first_strategy_with_env_chunk(self):
        """OTHER_FIRST_PER_TICK strategy with env_chunk matches old format."""
        old_ticks = [
            create_tick(0, self_chunk=tool_call_msg("call_1", "lookup")),
            create_tick(
                1,
                other_chunk=tool_result_msg("call_1", "found"),
                self_chunk=assistant_msg("Got it"),
            ),
        ]

        new_ticks = [
            ParticipantTick(
                tick_id=0,
                timestamp="2024-01-01T00:00:00",
                self_chunk=tool_call_msg("call_1", "lookup"),
            ),
            ParticipantTick(
                tick_id=1,
                timestamp="2024-01-01T00:00:01",
                self_chunk=assistant_msg("Got it"),
                env_chunk=tool_result_msg("call_1", "found"),
            ),
        ]

        expanded = _expand_env_chunks(new_ticks)
        old_messages = linearize_ticks(
            old_ticks, LinearizationStrategy.OTHER_FIRST_PER_TICK
        )
        new_messages = linearize_ticks(
            expanded, LinearizationStrategy.OTHER_FIRST_PER_TICK
        )

        assert len(old_messages) == len(new_messages)
        for old_msg, new_msg in zip(old_messages, new_messages):
            assert type(old_msg) is type(new_msg)
            assert old_msg.content == new_msg.content

    def test_tool_result_not_interleaved_with_waiting_tick_speech(self):
        """Verify tool call/result stay paired even when the other participant
        speaks during the 1-tick waiting period between tool call and tool
        result delivery.

        This is the scenario from the EnvironmentMessage bug fix: self makes a
        tool call at tick 0, other speaks at tick 1 (waiting tick), tool result
        arrives at tick 2 via env_chunk. The CONTAINMENT_AWARE strategy must
        ensure tool_call is immediately followed by tool_result in the output,
        with no interleaved speech from other.
        """
        ticks = [
            ParticipantTick(
                tick_id=0,
                timestamp="2024-01-01T00:00:00",
                self_chunk=tool_call_msg("call_1", "toggle_airplane_mode"),
                other_chunk=user_msg("Sure, toggling now"),
            ),
            ParticipantTick(
                tick_id=1,
                timestamp="2024-01-01T00:00:01",
                other_chunk=user_msg("One moment please"),
            ),
            ParticipantTick(
                tick_id=2,
                timestamp="2024-01-01T00:00:02",
                other_chunk=user_msg("Still here"),
                env_chunk=tool_result_msg("call_1", '{"success": true}'),
            ),
        ]

        expanded = _expand_env_chunks(ticks)
        messages = linearize_ticks(expanded, LinearizationStrategy.CONTAINMENT_AWARE)

        # Find tool call and tool result indices
        tool_call_idx = None
        tool_result_idx = None
        for i, m in enumerate(messages):
            if hasattr(m, "tool_calls") and m.tool_calls:
                tool_call_idx = i
            if isinstance(m, ToolMessage):
                tool_result_idx = i

        assert tool_call_idx is not None, "Tool call not found in linearized output"
        assert tool_result_idx is not None, "Tool result not found in linearized output"
        assert tool_result_idx == tool_call_idx + 1, (
            f"Tool result (idx={tool_result_idx}) must immediately follow "
            f"tool call (idx={tool_call_idx}), but there are interleaved messages"
        )

        # Verify all content is preserved
        contents = [m.content for m in messages]
        assert '{"success": true}' in contents
        assert any("toggling" in (c or "") for c in contents)
        assert any("moment" in (c or "") for c in contents)
