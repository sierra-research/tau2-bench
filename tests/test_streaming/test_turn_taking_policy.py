"""
Tests for basic_turn_taking_policy.

Focuses on the delivering_tool_result_speech threshold override
and overlap/interruption logic.
"""

from tau2.agent.base.streaming import (
    StreamingState,
    basic_turn_taking_policy,
)
from tau2.data_model.message import AssistantMessage, UserMessage


def _make_state(
    output_queue_size: int = 0,
    input_buffer_speech_ticks: int = 0,
    consecutive_self_speaking: int = 0,
    consecutive_other_speaking: int = 0,
    time_since_last_talk: int = 0,
    time_since_last_other_talk: int = 0,
    delivering_tool_result_speech: bool = False,
) -> StreamingState:
    """Build a StreamingState with the given overlap/timing configuration."""
    output_queue = [
        AssistantMessage(role="assistant", content=f"chunk_{i}", contains_speech=True)
        for i in range(output_queue_size)
    ]

    input_buffer = [
        UserMessage(role="user", content=f"speech_{i}", contains_speech=True)
        for i in range(input_buffer_speech_ticks)
    ]

    return StreamingState(
        output_streaming_queue=output_queue,
        input_turn_taking_buffer=input_buffer,
        consecutive_self_speaking_ticks=consecutive_self_speaking,
        consecutive_other_speaking_ticks=consecutive_other_speaking,
        time_since_last_talk=time_since_last_talk,
        time_since_last_other_talk=time_since_last_other_talk,
        delivering_tool_result_speech=delivering_tool_result_speech,
    )


class TestDeliveringToolResultSpeech:
    """Tests for the delivering_tool_result_speech threshold override."""

    def test_tool_result_speech_uses_interrupting_threshold(self):
        """When delivering tool result speech, the high 'interrupting' threshold
        should be used regardless of who initiated the overlap.

        This is the test that would have caught the overlap_initiator scoping bug.
        """
        state = _make_state(
            output_queue_size=5,
            input_buffer_speech_ticks=3,
            consecutive_self_speaking=10,
            consecutive_other_speaking=3,
            delivering_tool_result_speech=True,
        )

        # With the high threshold (10), 3 ticks of interruption is not enough
        action, info = basic_turn_taking_policy(
            state,
            yield_threshold_when_interrupted=2,
            yield_threshold_when_interrupting=10,
        )
        assert action == "keep_talking"

    def test_tool_result_speech_still_yields_past_threshold(self):
        """Tool result speech is harder to interrupt but not immune.
        Past the high threshold, the participant should yield."""
        state = _make_state(
            output_queue_size=5,
            input_buffer_speech_ticks=15,
            consecutive_self_speaking=20,
            consecutive_other_speaking=15,
            delivering_tool_result_speech=True,
        )

        action, info = basic_turn_taking_policy(
            state,
            yield_threshold_when_interrupted=2,
            yield_threshold_when_interrupting=10,
        )
        assert action == "stop_talking"

    def test_normal_speech_uses_interrupted_threshold_when_other_initiated(self):
        """Without delivering_tool_result_speech, when other initiated the overlap,
        the lower 'interrupted' threshold is used."""
        # Other has been speaking longer -> other initiated the overlap
        # (self was speaking first, other barged in)
        state = _make_state(
            output_queue_size=5,
            input_buffer_speech_ticks=3,
            consecutive_self_speaking=10,
            consecutive_other_speaking=3,
        )

        action, info = basic_turn_taking_policy(
            state,
            yield_threshold_when_interrupted=2,
            yield_threshold_when_interrupting=10,
        )
        # 3 ticks > threshold of 2 -> should stop
        assert action == "stop_talking"

    def test_normal_speech_uses_interrupting_threshold_when_self_initiated(self):
        """Without delivering_tool_result_speech, when self initiated the overlap,
        the higher 'interrupting' threshold is used."""
        # Self has been speaking shorter -> self initiated the overlap
        # (other was speaking first, self barged in)
        state = _make_state(
            output_queue_size=5,
            input_buffer_speech_ticks=3,
            consecutive_self_speaking=3,
            consecutive_other_speaking=10,
        )

        action, info = basic_turn_taking_policy(
            state,
            yield_threshold_when_interrupted=2,
            yield_threshold_when_interrupting=10,
        )
        # 3 ticks < threshold of 10 -> should keep talking
        assert action == "keep_talking"


class TestBasicTurnTakingActions:
    """Smoke tests for basic turn-taking action selection."""

    def test_wait_when_not_talking_and_below_threshold(self):
        """When not talking and other hasn't spoken long enough, wait."""
        state = _make_state(
            time_since_last_other_talk=0,
            time_since_last_talk=0,
        )

        action, _ = basic_turn_taking_policy(
            state,
            wait_to_respond_threshold_other=5,
            wait_to_respond_threshold_self=5,
        )
        assert action == "wait"

    def test_generate_when_thresholds_met(self):
        """When both wait thresholds are met, generate a message."""
        state = _make_state(
            input_buffer_speech_ticks=3,
            time_since_last_other_talk=10,
            time_since_last_talk=10,
        )

        action, _ = basic_turn_taking_policy(
            state,
            wait_to_respond_threshold_other=2,
            wait_to_respond_threshold_self=2,
        )
        assert action == "generate_message"

    def test_keep_talking_when_no_interruption(self):
        """When talking and other is silent, keep talking."""
        state = _make_state(
            output_queue_size=3,
            consecutive_self_speaking=5,
            consecutive_other_speaking=0,
        )

        action, _ = basic_turn_taking_policy(
            state,
            yield_threshold_when_interrupted=2,
        )
        assert action == "keep_talking"
