"""
Unit tests for DiscreteTimeAudioNativeAgent.

This test suite verifies that the discrete-time audio native agent works correctly with:
- Tick-based audio exchange
- State management with DiscreteTimeAgentState
- Tool call handling
- Speech detection
- Audio extraction from user messages
- Response creation

Note: Most tests use mock adapters to avoid real API calls.
See tests/test_voice/test_audio_native/test_openai/test_discrete_time_adapter.py
for adapter-specific tests.
"""

import base64
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from tau2.agent.discrete_time_audio_native_agent import (
    DiscreteTimeAgentState,
    DiscreteTimeAudioNativeAgent,
)
from tau2.data_model.audio import TELEPHONY_SAMPLE_RATE, AudioEncoding, AudioFormat
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.voice.audio_native.openai.tick_runner import TickResult

# =============================================================================
# Mock TickResult Helper
# =============================================================================


def make_mock_tick_result(
    tick_number: int = 1,
    agent_audio: bytes = b"",
    transcript: str = "",
    events: Optional[List] = None,
    was_truncated: bool = False,
    item_ids: Optional[List[str]] = None,
    tool_calls: Optional[List] = None,
) -> TickResult:
    """Create a mock TickResult for testing."""
    result = MagicMock(spec=TickResult)
    result.tick_number = tick_number
    result.agent_audio_data = agent_audio
    result.proportional_transcript = transcript
    result.events = events or []
    result.was_truncated = was_truncated
    result.get_played_agent_audio.return_value = agent_audio or b"\x7f" * 8000
    result.item_ids = item_ids or []
    result.tool_calls = tool_calls or []
    result.model_dump.return_value = {
        "tick_number": tick_number,
        "agent_audio_bytes": len(agent_audio),
        "proportional_transcript": transcript,
        "events": events or [],
        "was_truncated": was_truncated,
        "item_ids": item_ids or [],
        "tool_calls": tool_calls or [],
    }
    return result


# =============================================================================
# Fixtures
# =============================================================================


def _test_tool_func(arg: str) -> str:
    """A test tool for testing.

    Args:
        arg: An argument string.

    Returns:
        The result string.
    """
    return f"Result: {arg}"


@pytest.fixture
def mock_tools() -> List[Tool]:
    """Create mock tools for testing."""
    return [Tool(_test_tool_func)]


@pytest.fixture
def domain_policy() -> str:
    """Create a domain policy for testing."""
    return "You are a helpful assistant."


@pytest.fixture
def mock_adapter():
    """Create a mock adapter that doesn't make real API calls."""
    adapter = MagicMock()
    adapter.is_connected = False

    def mock_connect(*args, **kwargs):
        adapter.is_connected = True

    def mock_disconnect():
        adapter.is_connected = False

    adapter.connect.side_effect = mock_connect
    adapter.disconnect.side_effect = mock_disconnect
    adapter.run_tick.return_value = make_mock_tick_result()

    return adapter


@pytest.fixture
def agent_with_mock_adapter(
    mock_tools: List[Tool], domain_policy: str, mock_adapter
) -> DiscreteTimeAudioNativeAgent:
    """Create an agent with a mock adapter."""
    return DiscreteTimeAudioNativeAgent(
        tools=mock_tools,
        domain_policy=domain_policy,
        tick_duration_ms=1000,
        modality="audio",
        adapter=mock_adapter,
    )


@pytest.fixture
def audio_format() -> AudioFormat:
    """Create a standard audio format for testing."""
    return AudioFormat(
        encoding=AudioEncoding.ULAW,
        sample_rate=TELEPHONY_SAMPLE_RATE,
        channels=1,
    )


@pytest.fixture
def user_audio_message(audio_format) -> UserMessage:
    """Create a sample audio user message."""
    audio_bytes = b"\x50" * 8000  # 1 second of audio at 8kHz
    return UserMessage(
        role="user",
        content="Hello",
        is_audio=True,
        audio_content=base64.b64encode(audio_bytes).decode("utf-8"),
        audio_format=audio_format,
        contains_speech=True,
    )


# =============================================================================
# Basic Interface Tests
# =============================================================================


class TestBasicInterface:
    """Tests for basic agent interface."""

    def test_agent_has_get_next_chunk(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that agent has get_next_chunk method."""
        assert hasattr(agent_with_mock_adapter, "get_next_chunk")
        assert callable(agent_with_mock_adapter.get_next_chunk)

    def test_agent_is_full_duplex(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that agent inherits from FullDuplexAgent."""
        # Full-duplex agents have get_next_chunk, not generate_next_message
        assert hasattr(agent_with_mock_adapter, "get_next_chunk")
        assert not hasattr(agent_with_mock_adapter, "generate_next_message")

    def test_agent_has_stop_method(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that agent has stop method for cleanup."""
        assert hasattr(agent_with_mock_adapter, "stop")
        assert callable(agent_with_mock_adapter.stop)

    def test_agent_has_speech_detection(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that agent has speech_detection method."""
        assert hasattr(agent_with_mock_adapter, "speech_detection")
        assert callable(agent_with_mock_adapter.speech_detection)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for agent initialization."""

    def test_agent_initialization_params(
        self, mock_tools: List[Tool], domain_policy: str, mock_adapter
    ):
        """Test that agent initializes with correct parameters."""
        agent = DiscreteTimeAudioNativeAgent(
            tools=mock_tools,
            domain_policy=domain_policy,
            tick_duration_ms=500,
            modality="audio_in_text_out",
            adapter=mock_adapter,
        )

        assert agent.tools == mock_tools
        assert agent.domain_policy == domain_policy
        assert agent.tick_duration_ms == 500
        assert agent.modality == "audio_in_text_out"
        assert agent.bytes_per_tick == int(TELEPHONY_SAMPLE_RATE * 500 / 1000)

    def test_agent_creates_adapter_lazily(
        self, mock_tools: List[Tool], domain_policy: str
    ):
        """Test that adapter is created lazily if not provided."""
        agent = DiscreteTimeAudioNativeAgent(
            tools=mock_tools,
            domain_policy=domain_policy,
            tick_duration_ms=1000,
        )

        # Adapter should not be created yet
        assert agent._adapter is None
        assert agent._owns_adapter is True

        # Accessing adapter property should create it
        # (We can't actually test this without network, so just verify setup)
        assert agent._adapter is None  # Still None until first use


# =============================================================================
# State Tests
# =============================================================================


class TestStateManagement:
    """Tests for state management."""

    def test_get_init_state_returns_correct_type(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that get_init_state returns DiscreteTimeAgentState."""
        state = agent_with_mock_adapter.get_init_state()

        assert isinstance(state, DiscreteTimeAgentState)

    def test_state_has_required_fields(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that state has all required fields."""
        state = agent_with_mock_adapter.get_init_state()

        # Streaming state fields
        assert hasattr(state, "input_turn_taking_buffer")
        assert hasattr(state, "output_streaming_queue")
        assert hasattr(state, "time_since_last_talk")
        assert hasattr(state, "time_since_last_other_talk")

        # Discrete-time specific fields
        assert hasattr(state, "tick_count")
        assert hasattr(state, "total_user_audio_bytes")
        assert hasattr(state, "total_agent_audio_bytes")
        assert hasattr(state, "pending_tool_calls")
        assert hasattr(state, "messages")

    def test_state_initializes_with_defaults(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that state initializes with correct defaults."""
        state = agent_with_mock_adapter.get_init_state()

        assert state.tick_count == 0
        assert state.total_user_audio_bytes == 0
        assert state.total_agent_audio_bytes == 0
        assert state.pending_tool_calls == []
        assert state.messages == []
        assert state.time_since_last_talk == 0
        assert state.time_since_last_other_talk == 0

    def test_state_connects_adapter_on_init(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent, mock_adapter
    ):
        """Test that get_init_state connects the adapter."""
        assert not mock_adapter.is_connected

        agent_with_mock_adapter.get_init_state()

        assert mock_adapter.is_connected

    def test_state_with_message_history(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that state can be initialized with message history."""
        history = [
            UserMessage(role="user", content="Hello"),
            AssistantMessage(role="assistant", content="Hi there"),
        ]

        state = agent_with_mock_adapter.get_init_state(message_history=history)

        assert len(state.messages) == 2


# =============================================================================
# get_next_chunk Tests
# =============================================================================


class TestGetNextChunk:
    """Tests for get_next_chunk method."""

    def test_get_next_chunk_returns_tuple(
        self,
        agent_with_mock_adapter: DiscreteTimeAudioNativeAgent,
        user_audio_message: UserMessage,
    ):
        """Test that get_next_chunk returns (AssistantMessage, state) tuple."""
        state = agent_with_mock_adapter.get_init_state()

        result = agent_with_mock_adapter.get_next_chunk(state, user_audio_message)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], AssistantMessage)
        assert isinstance(result[1], DiscreteTimeAgentState)

    def test_get_next_chunk_increments_tick_count(
        self,
        agent_with_mock_adapter: DiscreteTimeAudioNativeAgent,
        user_audio_message: UserMessage,
    ):
        """Test that tick_count increments with each call."""
        state = agent_with_mock_adapter.get_init_state()
        assert state.tick_count == 0

        _, state = agent_with_mock_adapter.get_next_chunk(state, user_audio_message)
        assert state.tick_count == 1

        _, state = agent_with_mock_adapter.get_next_chunk(state, user_audio_message)
        assert state.tick_count == 2

    def test_get_next_chunk_tracks_user_audio(
        self,
        agent_with_mock_adapter: DiscreteTimeAudioNativeAgent,
        user_audio_message: UserMessage,
    ):
        """Test that user audio bytes are tracked."""
        state = agent_with_mock_adapter.get_init_state()

        _, state = agent_with_mock_adapter.get_next_chunk(state, user_audio_message)

        # User message had 8000 bytes of audio
        assert state.total_user_audio_bytes == 8000

    def test_get_next_chunk_updates_time_counters(
        self,
        agent_with_mock_adapter: DiscreteTimeAudioNativeAgent,
        user_audio_message: UserMessage,
    ):
        """Test that time counters update correctly."""
        state = agent_with_mock_adapter.get_init_state()

        # First chunk with speech
        _, state = agent_with_mock_adapter.get_next_chunk(state, user_audio_message)
        assert state.time_since_last_other_talk == 0  # Just received speech

        # Silence chunk
        silence = UserMessage(role="user", content=None, contains_speech=False)
        _, state = agent_with_mock_adapter.get_next_chunk(state, silence)
        assert state.time_since_last_other_talk == 1


# =============================================================================
# Speech Detection Tests
# =============================================================================


class TestSpeechDetection:
    """Tests for speech detection."""

    def test_speech_detection_user_message_with_speech(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that speech is detected in user message with contains_speech=True."""
        msg = UserMessage(role="user", content="Hello", contains_speech=True)
        assert agent_with_mock_adapter.speech_detection(msg) is True

    def test_speech_detection_user_message_without_speech(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that no speech is detected in silence."""
        msg = UserMessage(role="user", content=None, contains_speech=False)
        assert agent_with_mock_adapter.speech_detection(msg) is False

    def test_speech_detection_non_user_message(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that speech detection returns False for non-user messages."""
        msg = AssistantMessage(role="assistant", content="Hi", contains_speech=True)
        assert agent_with_mock_adapter.speech_detection(msg) is False

    def test_speech_detection_tool_message(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that speech detection returns False for tool messages."""
        msg = ToolMessage(role="tool", id="call_123", content="Result")
        assert agent_with_mock_adapter.speech_detection(msg) is False


# =============================================================================
# Audio Extraction Tests
# =============================================================================


class TestAudioExtraction:
    """Tests for audio extraction from user messages."""

    def test_extract_audio_from_user_message(
        self,
        agent_with_mock_adapter: DiscreteTimeAudioNativeAgent,
        user_audio_message: UserMessage,
    ):
        """Test extracting audio from user message."""
        audio = agent_with_mock_adapter._extract_user_audio(user_audio_message)

        assert isinstance(audio, bytes)
        assert len(audio) == 8000  # 1 second at 8kHz

    def test_extract_audio_from_text_only_message(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test extracting audio from text-only message returns silence."""
        msg = UserMessage(role="user", content="Hello", is_audio=False)

        audio = agent_with_mock_adapter._extract_user_audio(msg)

        # Should return silence of tick duration
        assert isinstance(audio, bytes)
        assert len(audio) == agent_with_mock_adapter.bytes_per_tick
        assert audio == b"\x7f" * agent_with_mock_adapter.bytes_per_tick

    def test_extract_audio_from_tool_message(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test extracting audio from tool message returns silence."""
        msg = ToolMessage(role="tool", id="call_123", content="Result")

        audio = agent_with_mock_adapter._extract_user_audio(msg)

        # Should return silence
        assert audio == b"\x7f" * agent_with_mock_adapter.bytes_per_tick


# =============================================================================
# Tool Handling Tests
# =============================================================================


class TestToolHandling:
    """Tests for tool call handling."""

    def test_tool_result_sent_to_adapter(
        self,
        agent_with_mock_adapter: DiscreteTimeAudioNativeAgent,
        mock_adapter,
        user_audio_message: UserMessage,
    ):
        """Test that tool results are sent to the adapter."""
        state = agent_with_mock_adapter.get_init_state()

        tool_msg = ToolMessage(role="tool", id="call_123", content="Tool result")
        agent_with_mock_adapter.get_next_chunk(
            state, participant_chunk=user_audio_message, tool_results=tool_msg
        )

        # Verify send_tool_result was called
        mock_adapter.send_tool_result.assert_called_once()

    def test_multi_tool_message_handling(
        self,
        agent_with_mock_adapter: DiscreteTimeAudioNativeAgent,
        mock_adapter,
        user_audio_message: UserMessage,
    ):
        """Test that MultiToolMessage sends all results."""
        state = agent_with_mock_adapter.get_init_state()

        multi_tool = MultiToolMessage(
            role="tool",
            tool_messages=[
                ToolMessage(role="tool", id="call_1", content="Result 1"),
                ToolMessage(role="tool", id="call_2", content="Result 2"),
            ],
        )
        agent_with_mock_adapter.get_next_chunk(
            state, participant_chunk=user_audio_message, tool_results=multi_tool
        )

        # Should have sent two tool results
        assert mock_adapter.send_tool_result.call_count == 2


# =============================================================================
# Response Creation Tests
# =============================================================================


class TestResponseCreation:
    """Tests for response message creation."""

    def test_response_has_required_fields(
        self,
        agent_with_mock_adapter: DiscreteTimeAudioNativeAgent,
        user_audio_message: UserMessage,
    ):
        """Test that response has all required fields."""
        state = agent_with_mock_adapter.get_init_state()
        response, _ = agent_with_mock_adapter.get_next_chunk(state, user_audio_message)

        assert response.role == "assistant"
        assert hasattr(response, "contains_speech")
        assert hasattr(response, "audio_content")
        assert hasattr(response, "audio_format")
        assert hasattr(response, "timestamp")

    def test_response_audio_format_is_telephony(
        self,
        agent_with_mock_adapter: DiscreteTimeAudioNativeAgent,
        user_audio_message: UserMessage,
    ):
        """Test that response uses telephony audio format."""
        state = agent_with_mock_adapter.get_init_state()
        response, _ = agent_with_mock_adapter.get_next_chunk(state, user_audio_message)

        assert response.audio_format.encoding == AudioEncoding.ULAW
        assert response.audio_format.sample_rate == TELEPHONY_SAMPLE_RATE

    def test_empty_response_on_error(
        self, agent_with_mock_adapter: DiscreteTimeAudioNativeAgent
    ):
        """Test that empty response is created correctly."""
        response = agent_with_mock_adapter._create_empty_response()

        assert response.role == "assistant"
        assert response.content is None
        assert response.contains_speech is False
        assert response.audio_content is not None  # Contains silence


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanup:
    """Tests for agent cleanup."""

    def test_stop_disconnects_adapter_when_owned(
        self,
        mock_tools: List[Tool],
        domain_policy: str,
    ):
        """Test that stop() disconnects the adapter when the agent owns it.

        Note: The agent only disconnects adapters it owns (i.e., created internally).
        When an external adapter is provided, the agent does NOT disconnect it.
        """
        # Create agent without adapter - it will create and own one
        mock_adapter_instance = MagicMock()
        with patch(
            "tau2.agent.discrete_time_audio_native_agent.create_adapter",
            return_value=(mock_adapter_instance, "gpt-realtime-2026-01-12"),
        ):
            agent = DiscreteTimeAudioNativeAgent(
                tools=mock_tools,
                domain_policy=domain_policy,
                tick_duration_ms=1000,
            )

            # The agent should own the adapter since none was provided
            assert agent._owns_adapter is True

            # Initialize state (creates adapter)
            agent.get_init_state()

            # Stop should disconnect because agent owns the adapter
            agent.stop()

            mock_adapter_instance.disconnect.assert_called()

    def test_stop_does_not_disconnect_external_adapter(
        self,
        agent_with_mock_adapter: DiscreteTimeAudioNativeAgent,
        mock_adapter,
    ):
        """Test that stop() does NOT disconnect an externally provided adapter."""
        # Initialize to connect
        agent_with_mock_adapter.get_init_state()

        # The agent should NOT own the adapter since it was provided externally
        assert agent_with_mock_adapter._owns_adapter is False

        # Stop should NOT disconnect because agent doesn't own the adapter
        agent_with_mock_adapter.stop()

        mock_adapter.disconnect.assert_not_called()

    def test_cleanup_disconnects_adapter_when_owned(
        self,
        mock_tools: List[Tool],
        domain_policy: str,
    ):
        """Test that cleanup() disconnects the adapter when the agent owns it."""
        mock_adapter_instance = MagicMock()
        with patch(
            "tau2.agent.discrete_time_audio_native_agent.create_adapter",
            return_value=(mock_adapter_instance, "gpt-realtime-2026-01-12"),
        ):
            agent = DiscreteTimeAudioNativeAgent(
                tools=mock_tools,
                domain_policy=domain_policy,
                tick_duration_ms=1000,
            )

            agent.get_init_state()
            agent.cleanup()

            mock_adapter_instance.disconnect.assert_called()


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for agent configuration."""

    def test_default_vad_config(
        self, mock_tools: List[Tool], domain_policy: str, mock_adapter
    ):
        """Test that default VAD config is SERVER_VAD."""
        from tau2.voice.audio_native.openai.provider import OpenAIVADMode

        agent = DiscreteTimeAudioNativeAgent(
            tools=mock_tools,
            domain_policy=domain_policy,
            adapter=mock_adapter,
        )

        assert agent.vad_config.mode == OpenAIVADMode.SERVER_VAD

    def test_custom_vad_config(
        self, mock_tools: List[Tool], domain_policy: str, mock_adapter
    ):
        """Test that custom VAD config is used."""
        from tau2.voice.audio_native.openai.provider import (
            OpenAIVADConfig,
            OpenAIVADMode,
        )

        custom_vad = OpenAIVADConfig(
            mode=OpenAIVADMode.MANUAL,
            threshold=0.8,
        )

        agent = DiscreteTimeAudioNativeAgent(
            tools=mock_tools,
            domain_policy=domain_policy,
            adapter=mock_adapter,
            vad_config=custom_vad,
        )

        assert agent.vad_config.mode == OpenAIVADMode.MANUAL
        assert agent.vad_config.threshold == 0.8

    def test_buffer_until_complete_flag(
        self, mock_tools: List[Tool], domain_policy: str, mock_adapter
    ):
        """Test buffer_until_complete flag is stored."""
        agent = DiscreteTimeAudioNativeAgent(
            tools=mock_tools,
            domain_policy=domain_policy,
            adapter=mock_adapter,
            buffer_until_complete=True,
        )

        assert agent.buffer_until_complete is True

    def test_send_audio_instant_flag(
        self, mock_tools: List[Tool], domain_policy: str, mock_adapter
    ):
        """Test send_audio_instant flag is stored."""
        agent = DiscreteTimeAudioNativeAgent(
            tools=mock_tools,
            domain_policy=domain_policy,
            adapter=mock_adapter,
            send_audio_instant=False,
        )

        assert agent.send_audio_instant is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
