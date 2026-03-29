"""
Unit tests for VoiceStreamingUserSimulator.

This test suite verifies that the voice streaming user simulator works correctly with:
- Audio chunk output (receives text, sends audio)
- State management with VoiceState
- Turn-taking logic
- Interruption and backchanneling parameters
- Speech detection
- Time counters
"""

import pytest

from tau2.data_model.audio import AudioEncoding, AudioFormat
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.voice import SynthesisConfig, VoiceSettings
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator


@pytest.fixture
def user_instructions() -> str:
    return (
        "You are Mia Li. You want to fly from New York to Seattle on May 20 (one way)."
    )


@pytest.fixture
def voice_settings():
    """Create voice settings for testing (synthesis enabled, transcription disabled)."""
    return VoiceSettings(
        transcription_config=None,
        synthesis_config=SynthesisConfig(),
    )


@pytest.fixture
def streaming_user(
    user_instructions: str, voice_settings
) -> VoiceStreamingUserSimulator:
    """Create a voice streaming user with default behavior."""
    return VoiceStreamingUserSimulator(
        llm="gpt-4o-mini",
        instructions=user_instructions,
        tools=None,
        voice_settings=voice_settings,
        chunk_size=8000,  # Audio chunk size in samples
        wait_to_respond_threshold_other=2,
        wait_to_respond_threshold_self=4,
        yield_threshold_when_interrupted=2,
        backchannel_min_threshold=None,
    )


@pytest.fixture
def first_agent_message() -> AssistantMessage:
    """Create a simple agent message."""
    return AssistantMessage(
        role="assistant",
        content="Hello, how can I help you today?",
        contains_speech=True,
    )


@pytest.fixture
def audio_format():
    """Create a standard audio format for testing."""
    return AudioFormat(
        encoding=AudioEncoding.PCM_S16LE,
        sample_rate=16000,
    )


# --- Basic Interface Tests ---


def test_voice_streaming_user_has_get_next_chunk(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that voice streaming user has get_next_chunk method."""
    assert hasattr(streaming_user, "get_next_chunk")
    assert callable(streaming_user.get_next_chunk)


def test_voice_streaming_user_is_full_duplex_only(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that voice streaming user is full-duplex only (no generate_next_message)."""
    assert not hasattr(streaming_user, "generate_next_message")
    assert hasattr(streaming_user, "get_next_chunk")


def test_voice_streaming_user_has_voice_methods(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that voice streaming user has voice-specific methods."""
    # Should have voice methods from VoiceMixin
    assert hasattr(streaming_user, "synthesize_voice")
    assert hasattr(streaming_user, "speech_detection")
    assert hasattr(streaming_user, "voice_settings")


# --- State Initialization Tests ---


def test_voice_streaming_user_state_initialization(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that state initializes correctly with streaming and voice fields."""
    state = streaming_user.get_init_state()

    # Streaming state fields
    assert hasattr(state, "input_turn_taking_buffer")
    assert hasattr(state, "output_streaming_queue")
    assert state.input_turn_taking_buffer == []
    assert state.output_streaming_queue == []

    # Time counters
    assert hasattr(state, "time_since_last_talk")
    assert hasattr(state, "time_since_last_other_talk")
    assert state.time_since_last_talk == 0
    assert state.time_since_last_other_talk == 0

    # Voice state fields (from VoiceState)
    assert hasattr(state, "noise_generator")


def test_voice_streaming_user_state_independence(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that states are properly isolated."""
    state1 = streaming_user.get_init_state()
    state2 = streaming_user.get_init_state()

    # Verify states are independent
    assert id(state1) != id(state2)
    assert id(state1.input_turn_taking_buffer) != id(state2.input_turn_taking_buffer)

    # Modify one state
    state1.input_turn_taking_buffer.append(
        AssistantMessage(role="assistant", content="State 1 chunk")
    )

    # Verify other state is unaffected
    assert len(state1.input_turn_taking_buffer) == 1
    assert len(state2.input_turn_taking_buffer) == 0


def test_voice_streaming_user_state_serialization(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that state can be serialized."""
    state = streaming_user.get_init_state()

    # Test that model_dump works and includes streaming fields
    state_dict = state.model_dump()
    assert "input_turn_taking_buffer" in state_dict
    assert "output_streaming_queue" in state_dict
    assert "time_since_last_talk" in state_dict
    assert "time_since_last_other_talk" in state_dict


# --- get_next_chunk Behavior Tests ---


def test_voice_streaming_user_get_next_chunk_returns_message(
    streaming_user: VoiceStreamingUserSimulator, first_agent_message: AssistantMessage
):
    """Test that get_next_chunk always returns a message (never None)."""
    state = streaming_user.get_init_state()

    chunk, state = streaming_user.get_next_chunk(state, first_agent_message)

    assert chunk is not None
    assert isinstance(chunk, UserMessage)
    assert hasattr(chunk, "contains_speech")
    assert chunk.contains_speech is not None


def test_voice_streaming_user_chunk_accumulation(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that incoming chunks accumulate in state."""
    state = streaming_user.get_init_state()

    # Send first text chunk from agent
    chunk_0 = AssistantMessage(
        role="assistant",
        content="Hello ",
        chunk_id=0,
        is_final_chunk=False,
        contains_speech=True,
    )
    response_0, state = streaming_user.get_next_chunk(state, chunk_0)

    assert response_0 is not None
    assert len(state.input_turn_taking_buffer) >= 1

    # Send second text chunk
    chunk_1 = AssistantMessage(
        role="assistant",
        content="there!",
        chunk_id=1,
        is_final_chunk=False,
        contains_speech=True,
    )
    response_1, state = streaming_user.get_next_chunk(state, chunk_1)

    assert response_1 is not None
    assert len(state.input_turn_taking_buffer) >= 2


def test_voice_streaming_user_contains_speech_on_all_responses(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that all response chunks have contains_speech set."""
    state = streaming_user.get_init_state()

    # Test with various input types
    test_inputs = [
        AssistantMessage(role="assistant", content="Hello", contains_speech=True),
        AssistantMessage(role="assistant", content=None, contains_speech=False),
        AssistantMessage(role="assistant", content="Help me", contains_speech=True),
    ]

    for incoming in test_inputs:
        response, state = streaming_user.get_next_chunk(state, incoming)

        assert response is not None
        assert hasattr(response, "contains_speech")
        assert response.contains_speech is not None
        assert isinstance(response.contains_speech, bool)


# --- Time Counter Tests ---


def test_voice_streaming_user_time_counters(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that time counters update correctly."""
    state = streaming_user.get_init_state()

    # Initial values
    assert state.time_since_last_talk == 0
    assert state.time_since_last_other_talk == 0

    # Send speech chunk
    speech = AssistantMessage(role="assistant", content="Hello", contains_speech=True)
    _, state = streaming_user.get_next_chunk(state, speech)

    # After receiving speech, time_since_last_other_talk should reset
    assert state.time_since_last_other_talk == 0

    # Send silence chunks
    silence = AssistantMessage(role="assistant", content=None, contains_speech=False)
    for i in range(3):
        _, state = streaming_user.get_next_chunk(state, silence)
        assert state.time_since_last_other_talk == i + 1


# --- Speech Detection Tests ---


def test_voice_streaming_user_speech_detection_text(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test speech detection with text chunks (from agent)."""
    # Agent text chunk with speech
    speech_chunk = AssistantMessage(
        role="assistant", content="Hello", contains_speech=True
    )
    assert streaming_user.speech_detection(speech_chunk) is True

    # Agent text chunk without speech
    silence_chunk = AssistantMessage(
        role="assistant", content=None, contains_speech=False
    )
    assert streaming_user.speech_detection(silence_chunk) is False


def test_voice_streaming_user_speech_detection_wrong_role(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that speech detection returns False for wrong role (user)."""
    # User should only detect speech from agent messages
    user_chunk = UserMessage(role="user", content="Hello", contains_speech=True)
    assert streaming_user.speech_detection(user_chunk) is False


# --- Interruption Parameter Tests ---


def test_voice_streaming_user_yield_threshold_when_interrupted(
    user_instructions, voice_settings
):
    """Test that yield_threshold_when_interrupted is properly set."""
    user = VoiceStreamingUserSimulator(
        llm="gpt-4o-mini",
        instructions=user_instructions,
        tools=None,
        voice_settings=voice_settings,
        chunk_size=8000,
        yield_threshold_when_interrupted=5,
    )

    assert user.yield_threshold_when_interrupted == 5


def test_voice_streaming_user_yield_threshold_when_interrupted_none_disables(
    user_instructions, voice_settings
):
    """Test that yield_threshold_when_interrupted=None disables interruption."""
    user = VoiceStreamingUserSimulator(
        llm="gpt-4o-mini",
        instructions=user_instructions,
        tools=None,
        voice_settings=voice_settings,
        chunk_size=8000,
        yield_threshold_when_interrupted=None,
    )

    assert user.yield_threshold_when_interrupted is None


# --- Backchanneling Parameter Tests ---


def test_voice_streaming_user_backchannel_threshold(user_instructions, voice_settings):
    """Test that backchannel_min_threshold is properly set."""
    user = VoiceStreamingUserSimulator(
        llm="gpt-4o-mini",
        instructions=user_instructions,
        tools=None,
        voice_settings=voice_settings,
        chunk_size=8000,
        backchannel_min_threshold=3,
    )

    assert user.backchannel_min_threshold == 3


def test_voice_streaming_user_backchannel_threshold_none_disables(
    user_instructions, voice_settings
):
    """Test that backchannel_min_threshold=None disables backchanneling."""
    user = VoiceStreamingUserSimulator(
        llm="gpt-4o-mini",
        instructions=user_instructions,
        tools=None,
        voice_settings=voice_settings,
        chunk_size=8000,
        backchannel_min_threshold=None,
    )

    assert user.backchannel_min_threshold is None


# --- Voice Settings Tests ---


def test_voice_streaming_user_requires_synthesis(user_instructions):
    """Test that voice streaming user requires synthesis to be enabled."""
    # Settings with synthesis disabled should fail
    # Note: transcription_config must be None for user (not supported yet)
    invalid_settings = VoiceSettings(
        transcription_config=None,
        synthesis_config=None,
    )

    with pytest.raises(ValueError, match="synthesis"):
        VoiceStreamingUserSimulator(
            llm="gpt-4o-mini",
            instructions=user_instructions,
            tools=None,
            voice_settings=invalid_settings,
            chunk_size=8000,
        )


def test_voice_streaming_user_thresholds(user_instructions, voice_settings):
    """Test that wait thresholds are properly set."""
    user = VoiceStreamingUserSimulator(
        llm="gpt-4o-mini",
        instructions=user_instructions,
        tools=None,
        voice_settings=voice_settings,
        chunk_size=8000,
        wait_to_respond_threshold_other=3,
        wait_to_respond_threshold_self=5,
    )

    assert user.wait_to_respond_threshold_other == 3
    assert user.wait_to_respond_threshold_self == 5


# --- Audio Chunking Configuration Tests ---


def test_voice_streaming_user_chunk_size(streaming_user: VoiceStreamingUserSimulator):
    """Test that chunk_size is properly set for audio chunking."""
    assert hasattr(streaming_user, "chunk_size")
    assert streaming_user.chunk_size > 0


def test_voice_streaming_user_outputs_audio_chunks(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that voice streaming user outputs audio chunks.

    VoiceStreamingUserSimulator receives text but outputs audio (synthesized speech).
    """
    state = streaming_user.get_init_state()

    agent_msg = AssistantMessage(
        role="assistant", content="Hello, how can I help?", contains_speech=True
    )

    # Keep sending messages until user responds with speech
    chunk, state = streaming_user.get_next_chunk(state, agent_msg)

    # Output should be a UserMessage
    assert chunk is not None
    assert isinstance(chunk, UserMessage)
    # Audio content may or may not be present depending on whether user is speaking
    # But the message should have the is_audio attribute
    assert hasattr(chunk, "is_audio")


# --- Role Flipping Tests ---


def test_voice_streaming_user_role_flipping(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that user state properly flips roles for LLM interaction."""
    state = streaming_user.get_init_state()

    agent_msg = AssistantMessage(
        role="assistant", content="Hello!", is_final_chunk=True, contains_speech=True
    )
    _, state = streaming_user.get_next_chunk(state, agent_msg)

    # Test flip_roles method
    flipped = state.flip_roles()
    assert flipped is not None


# --- Tool Call Tests ---


def test_voice_streaming_user_tool_calls_not_chunked(
    streaming_user: VoiceStreamingUserSimulator,
):
    """Test that if user makes tool calls, they're sent as single chunks."""
    state = streaming_user.get_init_state()

    agent_msg = AssistantMessage(
        role="assistant", content="What would you like me to do?"
    )
    chunk, state = streaming_user.get_next_chunk(state, agent_msg)

    # If it's a tool call, it should be complete
    if chunk and chunk.tool_calls:
        assert chunk.is_final_chunk is True or not hasattr(chunk, "is_final_chunk")
        assert chunk.tool_calls is not None


# --- Integration Ticks Tests ---


def test_voice_streaming_user_integration_ticks(user_instructions, voice_settings):
    """Test that integration_ticks parameter is properly set."""
    user = VoiceStreamingUserSimulator(
        llm="gpt-4o-mini",
        instructions=user_instructions,
        tools=None,
        voice_settings=voice_settings,
        chunk_size=8000,
        integration_ticks=3,
    )

    assert user.integration_ticks == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
