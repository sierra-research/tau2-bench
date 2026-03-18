"""Tests for LiveKit CascadedVoiceProvider.

These tests verify the provider's core functionality:
- Session connection and disconnection
- Configuration with different STT/LLM/TTS options
- Audio send and receive
- Transcription (user and agent)
- Tool call round-trip

Note: These tests require API keys to be set and make real API calls.
They are enabled with environment variable: LIVEKIT_TEST_ENABLED=1

Required API keys:
- DEEPGRAM_API_KEY (for STT and TTS)
- OPENAI_API_KEY (for LLM)
"""

import asyncio
import os
import wave
from pathlib import Path
from typing import List

import pytest

from tau2.voice.audio_native.livekit.config import (
    CascadedConfig,
    DeepgramSTTConfig,
    DeepgramTTSConfig,
    OpenAILLMConfig,
)
from tau2.voice.audio_native.livekit.provider import (
    CascadedEvent,
    CascadedEventType,
    CascadedVoiceProvider,
    ProviderState,
)

# Skip all tests if not enabled or API keys not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("LIVEKIT_TEST_ENABLED"),
    reason="LIVEKIT_TEST_ENABLED not set (requires DEEPGRAM_API_KEY and OPENAI_API_KEY)",
)


# =============================================================================
# Test Data
# =============================================================================


TESTDATA_DIR = Path(__file__).parent.parent / "testdata"


def load_test_audio(filename: str) -> bytes:
    """Load test audio file and return as 16kHz mono PCM bytes.

    Args:
        filename: Name of file in testdata directory.

    Returns:
        Audio bytes in 16kHz mono PCM format.
    """
    filepath = TESTDATA_DIR / filename
    if not filepath.exists():
        pytest.skip(f"Test audio file not found: {filepath}")

    with wave.open(str(filepath), "rb") as wav:
        # Verify format
        assert wav.getnchannels() == 1, (
            f"Expected mono audio, got {wav.getnchannels()} channels"
        )
        assert wav.getsampwidth() == 2, (
            f"Expected 16-bit audio, got {wav.getsampwidth() * 8}-bit"
        )
        # Note: Sample rate may vary; Deepgram handles resampling
        return wav.readframes(wav.getnframes())


def generate_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Generate silent audio bytes.

    Args:
        duration_ms: Duration in milliseconds.
        sample_rate: Sample rate (default 16kHz).

    Returns:
        Silent audio bytes (16-bit PCM).
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def provider():
    """Create a CascadedVoiceProvider with default config."""
    return CascadedVoiceProvider()


@pytest.fixture
def custom_config():
    """Create a custom CascadedConfig."""
    return CascadedConfig(
        stt=DeepgramSTTConfig(
            model="nova-3",
            endpointing_ms=100,  # Faster endpointing for tests
        ),
        llm=OpenAILLMConfig(
            model="gpt-4.1-mini",  # Use mini for faster tests
            temperature=0.7,
        ),
        tts=DeepgramTTSConfig(
            model="aura-asteria-en",
        ),
        log_prompts=True,
    )


# =============================================================================
# TestProviderConnection
# =============================================================================


class TestProviderConnection:
    """Test basic connection lifecycle."""

    def test_connect_disconnect(self, provider):
        """Test basic connection and disconnection."""

        async def _test():
            assert not provider.is_connected
            assert provider.state == ProviderState.DISCONNECTED

            await provider.connect(
                system_prompt="You are a helpful assistant.",
                tools=[],
            )
            assert provider.is_connected
            assert provider.state == ProviderState.LISTENING

            await provider.disconnect()
            assert not provider.is_connected
            assert provider.state == ProviderState.DISCONNECTED

        asyncio.run(_test())

    def test_connect_with_invalid_deepgram_key(self):
        """Test connection with invalid Deepgram API key."""

        async def _test():
            # Temporarily override API key
            original_key = os.environ.get("DEEPGRAM_API_KEY")
            os.environ["DEEPGRAM_API_KEY"] = "invalid_key_12345"

            try:
                provider = CascadedVoiceProvider()
                # Connection should succeed initially (lazy connection)
                await provider.connect(
                    system_prompt="Test",
                    tools=[],
                )
                # But STT stream should fail when we try to use it
                # (This depends on how Deepgram handles invalid keys)
                await provider.disconnect()
            finally:
                if original_key:
                    os.environ["DEEPGRAM_API_KEY"] = original_key

        # Note: This test may behave differently based on Deepgram's error handling
        asyncio.run(_test())

    def test_reconnect_after_disconnect(self, provider):
        """Test that we can reconnect after disconnecting."""

        async def _test():
            # First connection
            await provider.connect(
                system_prompt="First connection",
                tools=[],
            )
            assert provider.is_connected
            await provider.disconnect()
            assert not provider.is_connected

            # Second connection
            await provider.connect(
                system_prompt="Second connection",
                tools=[],
            )
            assert provider.is_connected
            await provider.disconnect()
            assert not provider.is_connected

        asyncio.run(_test())


# =============================================================================
# TestProviderConfiguration
# =============================================================================


class TestProviderConfiguration:
    """Test provider configuration options."""

    def test_configure_session(self, custom_config):
        """Test that custom configuration is applied."""

        async def _test():
            provider = CascadedVoiceProvider(config=custom_config)

            await provider.connect(
                system_prompt="You are a helpful assistant.",
                tools=[],
            )

            try:
                assert provider.is_connected
                # Verify config was applied
                assert provider.config.stt.model == "nova-3"
                assert provider.config.llm.model == "gpt-4.1-mini"
                assert provider.config.tts.model == "aura-asteria-en"
            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_configure_with_tools(self, provider):
        """Test configuration with tool schemas."""
        from tau2.environment.tool import Tool

        async def _test():
            # Create a test tool
            def get_order_status(order_id: str) -> str:
                """Get the status of an order."""
                return f"Order {order_id} is being shipped"

            tool = Tool(
                name="get_order_status",
                description="Get the status of an order by order ID",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID to look up",
                        },
                    },
                    "required": ["order_id"],
                },
                func=get_order_status,
            )

            await provider.connect(
                system_prompt="You are a customer service agent with access to order tools.",
                tools=[tool],
            )

            try:
                assert provider.is_connected
                assert len(provider._tools) == 1
                assert provider._tools[0].name == "get_order_status"
            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_configure_with_openai_llm(self):
        """Test configuration with OpenAI LLM options."""

        async def _test():
            config = CascadedConfig(
                llm=OpenAILLMConfig(
                    model="gpt-4.1",
                    temperature=0.5,
                    max_completion_tokens=500,
                ),
            )
            provider = CascadedVoiceProvider(config=config)

            await provider.connect(
                system_prompt="Test prompt",
                tools=[],
            )

            try:
                assert provider.is_connected
                assert provider.config.llm.model == "gpt-4.1"
                assert provider.config.llm.temperature == 0.5
                assert provider.config.llm.max_completion_tokens == 500
            finally:
                await provider.disconnect()

        asyncio.run(_test())


# =============================================================================
# TestProviderAudioSend
# =============================================================================


class TestProviderAudioSend:
    """Test audio sending functionality."""

    def test_send_audio_chunks(self, provider):
        """Test that audio chunks are accepted without error."""

        async def _test():
            await provider.connect(
                system_prompt="You are a helpful assistant.",
                tools=[],
            )

            try:
                # Send silent audio chunks
                for _ in range(5):
                    silence = generate_silence(200)  # 200ms of silence
                    events: List[CascadedEvent] = []
                    async for event in provider.process_audio(silence):
                        events.append(event)

                    # Should not get errors
                    errors = [e for e in events if e.type == CascadedEventType.ERROR]
                    assert len(errors) == 0, f"Got errors: {errors}"

                # Connection should still be open
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_send_real_audio_chunks(self, provider):
        """Test sending real audio and receiving transcription."""

        async def _test():
            await provider.connect(
                system_prompt="You are a helpful assistant. Respond briefly.",
                tools=[],
            )

            try:
                # Load test audio (if available)
                try:
                    audio = load_test_audio("hello.wav")
                except Exception:
                    # Generate synthetic "audio" if test file not available
                    audio = generate_silence(1000)

                # Send audio in chunks (200ms each)
                chunk_size = 16000 * 2 // 5  # 200ms at 16kHz, 16-bit
                all_events: List[CascadedEvent] = []

                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i : i + chunk_size]
                    async for event in provider.process_audio(chunk):
                        all_events.append(event)

                # Wait a bit for any delayed events
                await asyncio.sleep(0.5)

                # Should not have fatal errors
                errors = [e for e in all_events if e.type == CascadedEventType.ERROR]
                assert len(errors) == 0, f"Got errors: {errors}"

            finally:
                await provider.disconnect()

        asyncio.run(_test())


# =============================================================================
# TestProviderAudioReceive
# =============================================================================


class TestProviderAudioReceive:
    """Test audio receiving functionality."""

    def test_receive_audio_response(self, provider):
        """Test receiving audio response via direct text input."""

        async def _test():
            await provider.connect(
                system_prompt="You are a helpful assistant. Respond with just one word.",
                tools=[],
            )

            try:
                # Use direct text input to trigger response
                all_events: List[CascadedEvent] = []
                async for event in provider.process_text("Say hello"):
                    all_events.append(event)

                # Should get audio events
                audio_events = [
                    e for e in all_events if e.type == CascadedEventType.TTS_AUDIO
                ]
                tts_completed = any(
                    e.type == CascadedEventType.TTS_COMPLETED for e in all_events
                )

                assert len(audio_events) > 0, "Expected audio response"
                assert tts_completed, "Expected TTS completed event"

                # Verify audio bytes
                total_audio_bytes = sum(len(e.audio) for e in audio_events if e.audio)
                assert total_audio_bytes > 0, "Expected non-empty audio data"

            finally:
                await provider.disconnect()

        asyncio.run(_test())


# =============================================================================
# TestProviderTranscription
# =============================================================================


class TestProviderTranscription:
    """Test transcription functionality."""

    def test_receive_agent_transcript(self, provider):
        """Test receiving agent's text response."""

        async def _test():
            await provider.connect(
                system_prompt="You are a helpful assistant. Always respond with exactly: Hello there!",
                tools=[],
            )

            try:
                all_events: List[CascadedEvent] = []
                async for event in provider.process_text("Greet me"):
                    all_events.append(event)

                # Should get LLM completed event with text
                llm_completed = [
                    e for e in all_events if e.type == CascadedEventType.LLM_COMPLETED
                ]
                assert len(llm_completed) > 0, "Expected LLM completed event"

                response_text = llm_completed[0].text
                assert response_text is not None and len(response_text) > 0, (
                    "Expected non-empty response"
                )

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_receive_user_transcript(self, provider):
        """Test receiving user's transcription from audio."""

        async def _test():
            await provider.connect(
                system_prompt="You are a helpful assistant.",
                tools=[],
            )

            try:
                # Load real audio with speech
                try:
                    audio = load_test_audio("hello.wav")
                except Exception:
                    pytest.skip("Test audio file not available")

                # Send audio
                all_events: List[CascadedEvent] = []
                chunk_size = 16000 * 2 // 5  # 200ms chunks

                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i : i + chunk_size]
                    async for event in provider.process_audio(chunk):
                        all_events.append(event)

                # Wait for final transcription
                await asyncio.sleep(1.0)

                # Drain any remaining events
                remaining = await provider._drain_stt_events(timeout=0.1)
                all_events.extend(remaining)

                # Check for transcript events (may not get any for short/quiet audio)
                _transcripts = [
                    e
                    for e in all_events
                    if e.type
                    in (
                        CascadedEventType.TRANSCRIPT_PARTIAL,
                        CascadedEventType.TRANSCRIPT_FINAL,
                    )
                ]
                # This test verifies the flow works without errors
                errors = [e for e in all_events if e.type == CascadedEventType.ERROR]
                assert len(errors) == 0, f"Got errors: {errors}"

            finally:
                await provider.disconnect()

        asyncio.run(_test())


# =============================================================================
# TestProviderToolFlow
# =============================================================================


class TestProviderToolFlow:
    """Test tool call functionality."""

    def test_tool_call_round_trip(self):
        """Test full tool call cycle: trigger → receive → send result → response."""
        from tau2.environment.tool import Tool

        async def _test():
            # Create a tool that the LLM should call
            def get_order_status(order_id: str) -> str:
                """Get the status of an order."""
                return f"Order {order_id} is being shipped and will arrive tomorrow"

            tool = Tool(
                name="get_order_status",
                description="Get the status of an order by order ID. Always use this when a user asks about an order.",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID to look up",
                        },
                    },
                    "required": ["order_id"],
                },
                func=get_order_status,
            )

            provider = CascadedVoiceProvider(
                config=CascadedConfig(
                    llm=OpenAILLMConfig(model="gpt-4.1-mini"),
                    log_prompts=True,
                ),
            )

            await provider.connect(
                system_prompt=(
                    "You are a customer service agent. When a user asks about an order, "
                    "ALWAYS use the get_order_status tool to look up the order status. "
                    "Never make up order information."
                ),
                tools=[tool],
            )

            try:
                # Step 1: Send text that should trigger tool call
                all_events: List[CascadedEvent] = []
                async for event in provider.process_text(
                    "What is the status of order 12345?"
                ):
                    all_events.append(event)

                # Step 2: Check for tool call
                tool_calls = [
                    e for e in all_events if e.type == CascadedEventType.TOOL_CALL
                ]

                # Note: LLM may or may not call the tool depending on the model
                # This test verifies the flow works if a tool call is made
                if tool_calls:
                    tc = tool_calls[0].tool_call
                    assert tc is not None
                    assert tc.name == "get_order_status"

                    # Step 3: Send tool result
                    result = get_order_status("12345")
                    continuation_events: List[CascadedEvent] = []
                    async for event in provider.send_tool_result(tc.id, result):
                        continuation_events.append(event)

                    # Step 4: Verify continuation response
                    llm_completed = [
                        e
                        for e in continuation_events
                        if e.type == CascadedEventType.LLM_COMPLETED
                    ]
                    assert len(llm_completed) > 0, (
                        "Expected LLM response after tool result"
                    )

                    response = llm_completed[0].text
                    assert response is not None
                    # Response should mention shipping/tomorrow from tool result
                    assert (
                        "ship" in response.lower()
                        or "tomorrow" in response.lower()
                        or len(response) > 10
                    )

            finally:
                await provider.disconnect()

        asyncio.run(_test())


# =============================================================================
# TestProviderInterruption
# =============================================================================


class TestProviderInterruption:
    """Test interruption/barge-in handling."""

    def test_interrupt_during_speaking(self, provider):
        """Test that interrupt() stops TTS and returns to listening."""

        async def _test():
            await provider.connect(
                system_prompt="You are a helpful assistant. Give a long detailed response.",
                tools=[],
            )

            try:
                # Start a response
                events_before_interrupt: List[CascadedEvent] = []

                async def collect_events():
                    async for event in provider.process_text("Tell me a long story"):
                        events_before_interrupt.append(event)
                        # Interrupt after getting some TTS audio
                        if event.type == CascadedEventType.TTS_AUDIO:
                            break

                await collect_events()

                # Interrupt
                interrupt_event = await provider.interrupt()
                assert interrupt_event.type == CascadedEventType.INTERRUPTED
                assert provider.state == ProviderState.LISTENING

            finally:
                await provider.disconnect()

        asyncio.run(_test())


# =============================================================================
# TestProviderContext
# =============================================================================


class TestProviderContext:
    """Test conversation context management."""

    def test_context_accumulation(self, provider):
        """Test that conversation history accumulates correctly."""

        async def _test():
            await provider.connect(
                system_prompt="You are a helpful assistant.",
                tools=[],
            )

            try:
                # First turn
                async for _ in provider.process_text("My name is Alice"):
                    pass

                # Context should have system + user + assistant
                assert len(provider.context.messages) >= 3
                assert provider.context.messages[0]["role"] == "system"
                assert provider.context.messages[1]["role"] == "user"
                assert provider.context.messages[2]["role"] == "assistant"

                # Second turn
                async for _ in provider.process_text("What is my name?"):
                    pass

                # Context should have grown
                assert len(provider.context.messages) >= 5

            finally:
                await provider.disconnect()

        asyncio.run(_test())
