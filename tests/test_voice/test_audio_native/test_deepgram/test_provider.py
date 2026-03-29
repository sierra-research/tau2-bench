"""
Integration tests for DeepgramVoiceAgentProvider.

These tests require a Deepgram API key and make real API calls.
Set DEEPGRAM_TEST_ENABLED=1 along with DEEPGRAM_API_KEY to run these tests.

Run with: DEEPGRAM_TEST_ENABLED=1 pytest tests/test_voice/test_audio_native/test_deepgram/test_provider.py -v -s

NOTE: For full end-to-end testing with audio responses, use the standalone test:
    python src/tau2/voice/audio_native/deepgram/test_provider_standalone.py
"""

import os

import pytest

# Skip all tests unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("DEEPGRAM_TEST_ENABLED"),
    reason="DEEPGRAM_TEST_ENABLED not set (requires DEEPGRAM_API_KEY)",
)


@pytest.fixture
def provider():
    """Create a DeepgramVoiceAgentProvider instance."""
    from tau2.voice.audio_native.deepgram.provider import DeepgramVoiceAgentProvider

    return DeepgramVoiceAgentProvider()


@pytest.fixture
def vad_config():
    """Create a default VAD config."""
    from tau2.voice.audio_native.deepgram.provider import DeepgramVADConfig

    return DeepgramVADConfig()


class TestDeepgramProviderBasic:
    """Basic connectivity and session tests."""

    def test_provider_initialization(self, provider):
        """Test that the provider initializes correctly."""
        assert provider is not None
        assert not provider.is_connected
        assert provider.api_key is not None

    def test_connect_and_disconnect(self, provider, vad_config):
        """Test basic connection and disconnection."""
        import asyncio

        async def _test():
            assert not provider.is_connected

            await provider.connect()
            assert provider.is_connected

            await provider.disconnect()
            assert not provider.is_connected

        asyncio.run(_test())

    def test_configure_session(self, provider, vad_config):
        """Test configuring the session with system prompt."""
        import asyncio

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a helpful assistant. Keep responses brief.",
                    tools=[],
                    vad_config=vad_config,
                )
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_send_audio(self, provider, vad_config):
        """Test sending audio data."""
        import asyncio

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=vad_config,
                )

                # Create test audio (silence in linear16 format)
                # 100ms at 16kHz, 16-bit = 3200 bytes
                silence = bytes([0x00] * 3200)

                # Send audio
                await provider.send_audio(silence)

                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestDeepgramProviderAudioFormats:
    """Tests for audio format handling."""

    def test_supported_audio_formats(self):
        """Test that we handle the expected audio formats."""
        from tau2.voice.audio_native.deepgram.provider import (
            DEEPGRAM_INPUT_SAMPLE_RATE,
            DEEPGRAM_OUTPUT_SAMPLE_RATE,
        )

        # Deepgram Voice Agent typically uses:
        # - Input: linear16 at 16kHz (configurable)
        # - Output: linear16 at 16kHz or 24kHz (configurable)
        assert DEEPGRAM_INPUT_SAMPLE_RATE in [8000, 16000, 24000, 48000]
        assert DEEPGRAM_OUTPUT_SAMPLE_RATE in [8000, 16000, 24000, 48000]


class TestDeepgramProviderToolCalls:
    """Tests for tool/function calling."""

    def test_configure_with_tools(self, provider, vad_config):
        """Test configuring session with tools."""
        import asyncio

        from tau2.environment.tool import Tool

        # Create a mock tool
        class MockTool(Tool):
            def __init__(self):
                pass

            @property
            def openai_schema(self):
                return {
                    "type": "function",
                    "function": {
                        "name": "get_order_status",
                        "description": "Get the status of a customer order",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "order_id": {
                                    "type": "string",
                                    "description": "The order ID to look up",
                                }
                            },
                            "required": ["order_id"],
                        },
                    },
                }

            def __call__(self, **kwargs):
                return {"status": "shipped", "tracking": "1234567890"}

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a customer service agent.",
                    tools=[MockTool()],
                    vad_config=vad_config,
                )
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_send_tool_result(self, provider, vad_config):
        """Test sending a tool result."""
        import asyncio

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=vad_config,
                )

                # This should not raise (even without a preceding tool call)
                # In real usage, this is called after receiving a function call request
                await provider.send_tool_result(
                    call_id="test-function-123",
                    result='{"status": "success", "data": "test result"}',
                )

                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestDeepgramProviderEvents:
    """Tests for event handling."""

    def test_receive_events_timeout(self, provider, vad_config):
        """Test that timeout events are returned when no data arrives."""
        import asyncio

        from tau2.voice.audio_native.deepgram.events import DeepgramTimeoutEvent

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=vad_config,
                )

                # Try to receive events - should get timeout since no audio sent
                events = []
                async for event in provider.receive_events():
                    events.append(event)
                    if len(events) >= 3:
                        break

                # Should receive timeout events
                timeout_events = [
                    e for e in events if isinstance(e, DeepgramTimeoutEvent)
                ]
                assert len(timeout_events) > 0

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestDeepgramProviderLLMConfiguration:
    """Tests for LLM provider configuration."""

    def test_configure_with_openai_llm(self, provider, vad_config):
        """Test configuring session with OpenAI as the LLM provider."""
        import asyncio

        async def _test():
            await provider.connect()

            try:
                # Configure with OpenAI as the LLM provider
                # NOTE: Deepgram uses "open_ai" (with underscore)
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=vad_config,
                    llm_provider="open_ai",
                    llm_model="gpt-4o-mini",
                )
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_configure_with_anthropic_llm(self, provider, vad_config):
        """Test configuring session with Anthropic as the LLM provider."""
        import asyncio

        async def _test():
            await provider.connect()

            try:
                # Configure with Anthropic as the LLM provider
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=vad_config,
                    llm_provider="anthropic",
                    llm_model="claude-3-5-sonnet-20241022",
                )
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestDeepgramProviderTTSConfiguration:
    """Tests for TTS provider configuration."""

    def test_configure_with_deepgram_tts(self, provider, vad_config):
        """Test configuring session with Deepgram's Aura TTS."""
        import asyncio

        async def _test():
            await provider.connect()

            try:
                # For Deepgram TTS, voice is part of the model name
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=vad_config,
                    tts_provider="deepgram",
                    tts_model="aura-2-asteria-en",  # Voice included in model name
                )
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestDeepgramProviderBargeIn:
    """Tests for barge-in (interruption) behavior."""

    def test_barge_in_detection(self, provider, vad_config):
        """Test that the provider supports barge-in detection.

        NOTE: Full barge-in testing requires actual speech audio.
        This test just verifies the configuration is accepted.
        """
        import asyncio

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=vad_config,
                )

                # Verify VAD config was applied
                assert provider._current_vad_config is not None

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestDeepgramProviderContextHistory:
    """Tests for conversation context and history."""

    def test_configure_with_context(self, provider, vad_config):
        """Test configuring session with prior conversation context."""
        import asyncio

        async def _test():
            await provider.connect()

            try:
                # Configure with prior conversation history
                context_messages = [
                    {"role": "user", "content": "Hi, I need help with my order."},
                    {
                        "role": "assistant",
                        "content": "Of course! What's your order number?",
                    },
                ]

                await provider.configure_session(
                    system_prompt="You are a customer service agent.",
                    tools=[],
                    vad_config=vad_config,
                    context_messages=context_messages,
                )
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())


# =============================================================================
# Audio Receive Tests (b)
# =============================================================================


class TestDeepgramProviderAudioReceive:
    """Tests for receiving audio from the agent."""

    def test_receive_audio_via_greeting(self, provider, vad_config):
        """Test receiving agent audio by using the greeting feature.

        This test verifies:
        - (b) audio receive: We receive DeepgramAudioEvent containing audio data
        - Agent can generate audio without requiring user speech input
        """
        import asyncio
        import base64

        from tau2.voice.audio_native.deepgram.events import (
            DeepgramAgentAudioDoneEvent,
            DeepgramAudioEvent,
            DeepgramTimeoutEvent,
        )

        async def _test():
            await provider.connect()

            try:
                # Configure with a greeting - agent speaks first
                await provider.configure_session(
                    system_prompt="You are a helpful assistant. Keep responses brief.",
                    tools=[],
                    vad_config=vad_config,
                    greeting="Hello, how can I help you today?",
                )

                # Collect events for up to 10 seconds
                audio_events = []
                max_events = 200  # Safety limit

                async for event in provider.receive_events():
                    if isinstance(event, DeepgramAudioEvent):
                        audio_events.append(event)
                    elif isinstance(event, DeepgramAgentAudioDoneEvent):
                        break
                    elif not isinstance(event, DeepgramTimeoutEvent):
                        pass  # Other events are fine

                    if len(audio_events) >= max_events:
                        break

                # Verify we received audio
                assert len(audio_events) > 0, "Expected audio events from greeting"

                # Verify audio is valid base64
                total_audio_bytes = 0
                for event in audio_events:
                    audio_bytes = base64.b64decode(event.audio)
                    total_audio_bytes += len(audio_bytes)

                assert total_audio_bytes > 0, "Expected non-empty audio data"
                # At 16kHz 16-bit, "Hello" should be at least a few hundred ms
                # 16000 samples/sec * 2 bytes/sample = 32000 bytes/sec
                # 500ms = 16000 bytes minimum for a short greeting
                assert total_audio_bytes > 1000, (
                    f"Audio too short: {total_audio_bytes} bytes"
                )

            finally:
                await provider.disconnect()

        asyncio.run(_test())


# =============================================================================
# Transcription Tests (c)
# =============================================================================


class TestDeepgramProviderTranscription:
    """Tests for agent audio transcription (ConversationText events)."""

    def test_receive_conversation_text_from_greeting(self, provider, vad_config):
        """Test receiving ConversationText event from agent's greeting.

        This test verifies:
        - (c) agent audio transcription: We receive text content of agent speech
        - The ConversationText contains the actual greeting text
        """
        import asyncio

        from tau2.voice.audio_native.deepgram.events import (
            DeepgramAgentAudioDoneEvent,
            DeepgramConversationTextEvent,
            DeepgramTimeoutEvent,
        )

        async def _test():
            await provider.connect()

            try:
                greeting_text = "Hello, I am ready to help you."
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=vad_config,
                    greeting=greeting_text,
                )

                # Collect conversation text events
                conversation_events = []
                timeout_count = 0
                max_timeouts = 50  # 5 seconds max wait after audio done

                async for event in provider.receive_events():
                    if isinstance(event, DeepgramConversationTextEvent):
                        conversation_events.append(event)
                    elif isinstance(event, DeepgramAgentAudioDoneEvent):
                        # Agent finished speaking, but ConversationText might come after
                        # Wait a bit more for any remaining events
                        pass
                    elif isinstance(event, DeepgramTimeoutEvent):
                        timeout_count += 1
                        if timeout_count > max_timeouts:
                            break

                    if len(conversation_events) >= 10:
                        break

                # Verify we received conversation text
                assert len(conversation_events) > 0, (
                    "Expected ConversationText events but received none. "
                    "Check if Deepgram is sending transcript alongside audio."
                )

                # Check that at least one is from the assistant with content
                assistant_texts = [
                    e
                    for e in conversation_events
                    if e.role == "assistant" and e.content
                ]
                assert len(assistant_texts) > 0, (
                    f"Expected assistant ConversationText with content. "
                    f"Got: {conversation_events}"
                )

                # Verify the greeting text is in one of the responses
                all_content = " ".join(e.content for e in assistant_texts)
                assert "help" in all_content.lower(), (
                    f"Expected greeting content containing 'help'. Got: {all_content}"
                )

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_send_audio_receive_transcription(self, provider, vad_config):
        """Test sending audio and receiving user transcription.

        This test verifies:
        - (b) audio send/receive flow
        - (c) transcription of user speech

        Uses pre-recorded test audio files. Sends audio in chunks to simulate
        real-time streaming (Deepgram expects continuous audio).
        """
        import asyncio
        import os
        import wave

        from tau2.voice.audio_native.deepgram.events import (
            DeepgramAgentAudioDoneEvent,
            DeepgramAudioEvent,
            DeepgramConversationTextEvent,
            DeepgramErrorEvent,
            DeepgramTimeoutEvent,
        )

        # Path to test audio
        testdata_dir = os.path.join(os.path.dirname(__file__), "..", "testdata")
        audio_file = os.path.join(testdata_dir, "hello.wav")

        if not os.path.exists(audio_file):
            pytest.skip("Test audio file not found. Run generate_test_audio.py first.")

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a helpful assistant. Keep responses very brief.",
                    tools=[],
                    vad_config=vad_config,
                )

                # Read test audio file
                with wave.open(audio_file, "rb") as wav:
                    audio_data = wav.readframes(wav.getnframes())

                # Send audio in chunks (200ms each at 16kHz = 6400 bytes)
                # This simulates real-time streaming
                chunk_size = 6400
                num_chunks = len(audio_data) // chunk_size + 1

                # Variables to track events
                user_transcripts = []
                agent_audio_received = False
                agent_audio_done = False

                # Send audio chunks and receive events concurrently
                async def send_audio():
                    for i in range(num_chunks):
                        start = i * chunk_size
                        end = min(start + chunk_size, len(audio_data))
                        chunk = audio_data[start:end]
                        if chunk:
                            await provider.send_audio(chunk)
                            await asyncio.sleep(0.2)  # 200ms per chunk
                    # Add silence after speech to trigger end of turn
                    silence = bytes(chunk_size)
                    for _ in range(5):  # 1 second of silence
                        await provider.send_audio(silence)
                        await asyncio.sleep(0.2)

                async def receive_events():
                    nonlocal agent_audio_received, agent_audio_done
                    timeout_count = 0
                    max_timeouts = 100  # 10 seconds max wait

                    async for event in provider.receive_events():
                        if isinstance(event, DeepgramConversationTextEvent):
                            if event.role == "user":
                                user_transcripts.append(event.content)
                        elif isinstance(event, DeepgramAudioEvent):
                            agent_audio_received = True
                        elif isinstance(event, DeepgramAgentAudioDoneEvent):
                            agent_audio_done = True
                            break
                        elif isinstance(event, DeepgramErrorEvent):
                            # Some errors are expected if we don't trigger response
                            break
                        elif isinstance(event, DeepgramTimeoutEvent):
                            timeout_count += 1
                            if timeout_count > max_timeouts:
                                break
                            if agent_audio_done:
                                break

                # Run both concurrently with a total timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(send_audio(), receive_events()), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    pass  # Expected if agent doesn't respond quickly

                # Verify we got at least some activity
                # Either transcription or agent audio (greeting test covers audio)
                assert len(user_transcripts) > 0 or agent_audio_received, (
                    "Expected transcription or agent audio events"
                )

            finally:
                await provider.disconnect()

        asyncio.run(_test())


# =============================================================================
# Full Tool Flow Tests (e)
# =============================================================================


class TestDeepgramProviderToolFlow:
    """Tests for complete tool/function calling flow."""

    def test_tool_configuration_and_result(self, provider, vad_config):
        """Test configuring tools and sending function results.

        This test verifies:
        - (d) tool configuration: Tools are properly configured
        - (e) tool result flow: We can send function results back

        Note: Triggering an actual function call requires specific user input.
        This test verifies the mechanics work without triggering a real call.
        """
        import asyncio

        from tau2.environment.tool import Tool

        # Create a test tool
        class OrderStatusTool(Tool):
            def __init__(self):
                pass

            @property
            def openai_schema(self):
                return {
                    "type": "function",
                    "function": {
                        "name": "get_order_status",
                        "description": "Get the status of a customer order by order ID",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "order_id": {
                                    "type": "string",
                                    "description": "The order ID to look up",
                                }
                            },
                            "required": ["order_id"],
                        },
                    },
                }

            def __call__(self, order_id: str):
                return {"status": "shipped", "tracking": "1Z999AA10123456784"}

        async def _test():
            await provider.connect()

            try:
                # Configure with tool
                await provider.configure_session(
                    system_prompt=(
                        "You are a customer service agent. "
                        "Use the get_order_status function when asked about orders."
                    ),
                    tools=[OrderStatusTool()],
                    vad_config=vad_config,
                )
                assert provider.is_connected

                # Verify we can send a tool result (doesn't require preceding call)
                await provider.send_tool_result(
                    call_id="test-call-id-12345",
                    result='{"status": "delivered", "date": "2024-01-15"}',
                )

                # Connection should still be valid
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_tool_call_round_trip(self, provider, vad_config):
        """Test full tool call round-trip: audio → function call → result → response.

        This test verifies:
        - Agent receives user audio requesting order status
        - Agent triggers FunctionCallRequest for get_order_status
        - We send FunctionCallResponse with result
        - Agent responds with the order status

        This is the definitive test for (e) tool result flow.
        """
        import asyncio
        import json
        import os
        import wave

        from tau2.environment.tool import Tool
        from tau2.voice.audio_native.deepgram.events import (
            DeepgramAgentAudioDoneEvent,
            DeepgramAudioEvent,
            DeepgramConversationTextEvent,
            DeepgramErrorEvent,
            DeepgramFunctionCallRequestEvent,
            DeepgramTimeoutEvent,
        )

        # Path to test audio
        testdata_dir = os.path.join(os.path.dirname(__file__), "..", "testdata")
        # Use check_order_12345.wav which says "Can you check the status of my order number 12345?"
        # This includes a specific order ID to trigger the function call
        audio_file = os.path.join(testdata_dir, "check_order_12345.wav")

        if not os.path.exists(audio_file):
            pytest.skip("Test audio file not found. Run generate_test_audio.py first.")

        # Create a test tool
        class OrderStatusTool(Tool):
            def __init__(self):
                pass

            @property
            def openai_schema(self):
                return {
                    "type": "function",
                    "function": {
                        "name": "get_order_status",
                        "description": "Get the status of a customer order by order ID. Use this when the user asks about their order status.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "order_id": {
                                    "type": "string",
                                    "description": "The order ID to look up",
                                }
                            },
                            "required": ["order_id"],
                        },
                    },
                }

            def __call__(self, order_id: str):
                return {"status": "shipped", "tracking": "1Z999AA10123456784"}

        async def _test():
            await provider.connect()

            try:
                # Configure with tool and explicit instructions to use it
                await provider.configure_session(
                    system_prompt=(
                        "You are a customer service agent for an online store. "
                        "When a user asks about their order status, you MUST use the "
                        "get_order_status function to look it up. Always use the tool. "
                        "If they don't provide an order ID, ask for it or use a placeholder."
                    ),
                    tools=[OrderStatusTool()],
                    vad_config=vad_config,
                )

                # Read test audio file
                with wave.open(audio_file, "rb") as wav:
                    audio_data = wav.readframes(wav.getnframes())

                # Track events
                function_call_requests = []
                agent_responses = []
                agent_audio_received = False

                # Send audio in chunks
                chunk_size = 6400  # 200ms at 16kHz

                async def send_audio():
                    num_chunks = len(audio_data) // chunk_size + 1
                    for i in range(num_chunks):
                        start = i * chunk_size
                        end = min(start + chunk_size, len(audio_data))
                        chunk = audio_data[start:end]
                        if chunk:
                            await provider.send_audio(chunk)
                            await asyncio.sleep(0.2)
                    # Add silence to trigger end of turn
                    silence = bytes(chunk_size)
                    for _ in range(10):  # 2 seconds of silence
                        await provider.send_audio(silence)
                        await asyncio.sleep(0.2)

                async def receive_and_respond():
                    nonlocal agent_audio_received
                    timeout_count = 0
                    max_timeouts = 200  # 20 seconds max
                    function_call_handled = False

                    async for event in provider.receive_events():
                        if isinstance(event, DeepgramFunctionCallRequestEvent):
                            function_call_requests.append(event)
                            # Respond to the function call
                            result = json.dumps(
                                {
                                    "status": "shipped",
                                    "tracking_number": "1Z999AA10123456784",
                                    "estimated_delivery": "January 25, 2026",
                                }
                            )
                            await provider.send_tool_result(
                                call_id=event.call_id,
                                result=result,
                                function_name=event.function_name,
                            )
                            function_call_handled = True

                        elif isinstance(event, DeepgramConversationTextEvent):
                            if event.role == "assistant" and event.content:
                                agent_responses.append(event.content)

                        elif isinstance(event, DeepgramAudioEvent):
                            agent_audio_received = True

                        elif isinstance(event, DeepgramAgentAudioDoneEvent):
                            if function_call_handled:
                                # Agent finished responding after function call
                                break

                        elif isinstance(event, DeepgramTimeoutEvent):
                            timeout_count += 1
                            if timeout_count > max_timeouts:
                                break

                        elif isinstance(event, DeepgramErrorEvent):
                            # Log but continue
                            pass

                # Run concurrently
                try:
                    await asyncio.wait_for(
                        asyncio.gather(send_audio(), receive_and_respond()),
                        timeout=25.0,
                    )
                except asyncio.TimeoutError:
                    pass

                # Verify function call was triggered
                assert len(function_call_requests) > 0, (
                    "Expected FunctionCallRequest but none received. "
                    f"Agent responses: {agent_responses}"
                )

                # Verify it was the right function
                assert function_call_requests[0].function_name == "get_order_status", (
                    f"Expected get_order_status, got {function_call_requests[0].function_name}"
                )

                # Verify agent responded with audio after function call
                assert agent_audio_received, (
                    "Expected agent audio response after function call"
                )

            finally:
                await provider.disconnect()

        asyncio.run(_test())
