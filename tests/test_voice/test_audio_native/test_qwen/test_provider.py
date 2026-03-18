"""Tests for QwenRealtimeProvider.

These tests verify the provider's core functionality:
- Session connection and disconnection
- Audio/text send and receive
- Event parsing
- Multiple tick-like cycles
- Tool configuration and function calling

Note: These tests require DASHSCOPE_API_KEY to be set and make real API calls.
They are marked with pytest.mark.skipif to skip when API key is not available.

Reference: https://www.alibabacloud.com/help/en/model-studio/realtime
"""

import asyncio
import base64
import os
from typing import List

import pytest

# Skip all tests if API key not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY"),
    reason="DASHSCOPE_API_KEY not set",
)


@pytest.fixture
def provider():
    """Create a QwenRealtimeProvider instance."""
    from tau2.voice.audio_native.qwen.provider import QwenRealtimeProvider

    return QwenRealtimeProvider()


class TestQwenRealtimeProviderBasic:
    """Basic provider connection and communication tests."""

    def test_connect_and_disconnect(self, provider):
        """Test basic connection and disconnection."""

        async def _test():
            from tau2.voice.audio_native.qwen.provider import QwenVADConfig

            assert not provider.is_connected

            await provider.connect()
            assert provider.is_connected

            # Configure session
            await provider.configure_session(
                system_prompt="You are a helpful assistant.",
                tools=[],
                vad_config=QwenVADConfig(),
            )

            await provider.disconnect()
            assert not provider.is_connected

        asyncio.run(_test())

    def test_reconnect_after_disconnect(self, provider):
        """Test that reconnection works after disconnect."""

        async def _test():
            from tau2.voice.audio_native.qwen.provider import QwenVADConfig

            # First connection
            await provider.connect()
            assert provider.is_connected
            await provider.disconnect()
            assert not provider.is_connected

            # Reconnect
            await provider.connect()
            assert provider.is_connected

            await provider.configure_session(
                system_prompt="You are a helpful assistant.",
                tools=[],
                vad_config=QwenVADConfig(),
            )

            await provider.disconnect()

        asyncio.run(_test())

    def test_send_text_receive_audio(self, provider):
        """Test sending text and receiving audio response."""

        async def _test():
            from tau2.voice.audio_native.qwen.events import (
                QwenAudioDeltaEvent,
                QwenResponseDoneEvent,
            )
            from tau2.voice.audio_native.qwen.provider import QwenVADConfig

            await provider.connect()
            await provider.configure_session(
                system_prompt="You are a helpful assistant. Respond briefly.",
                tools=[],
                vad_config=QwenVADConfig(),
            )

            try:
                # Send text to trigger response
                await provider.send_text("Hello, please say hi briefly.")

                # Receive events for up to 10 seconds
                events = await provider.receive_events_for_duration(10.0)

                # Verify we got audio and response done
                audio_chunks_received = sum(
                    1 for e in events if isinstance(e, QwenAudioDeltaEvent)
                )
                response_done_received = any(
                    isinstance(e, QwenResponseDoneEvent) for e in events
                )

                assert audio_chunks_received > 0, "Expected audio response"
                assert response_done_received, "Expected response.done event"

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_send_text_receive_transcript(self, provider):
        """Test that audio responses include transcription."""

        async def _test():
            from tau2.voice.audio_native.qwen.events import (
                QwenAudioTranscriptDeltaEvent,
            )
            from tau2.voice.audio_native.qwen.provider import QwenVADConfig

            await provider.connect()
            await provider.configure_session(
                system_prompt="You are a helpful assistant. Respond briefly.",
                tools=[],
                vad_config=QwenVADConfig(),
            )

            try:
                await provider.send_text("Say hello in one sentence.")

                # Collect transcript
                transcript = ""
                events = await provider.receive_events_for_duration(10.0)

                for e in events:
                    if isinstance(e, QwenAudioTranscriptDeltaEvent):
                        transcript += e.delta

                assert len(transcript) > 0, "Expected transcript in response"

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestQwenRealtimeProviderMultipleTicks:
    """Test provider with multiple tick-like cycles."""

    def test_multiple_tick_cycles(self, provider):
        """Test multiple send/receive cycles simulating ticks."""

        async def _test():
            from tau2.voice.audio_native.qwen.events import QwenErrorEvent
            from tau2.voice.audio_native.qwen.provider import QwenVADConfig

            await provider.connect()
            await provider.configure_session(
                system_prompt="You are a helpful assistant. Respond briefly.",
                tools=[],
                vad_config=QwenVADConfig(),
            )

            try:
                tick_duration_sec = 0.2
                num_ticks = 5
                errors: List[str] = []

                for tick in range(num_ticks):
                    # Receive for tick duration
                    events = await provider.receive_events_for_duration(
                        tick_duration_sec
                    )

                    # Check for errors
                    for e in events:
                        if isinstance(e, QwenErrorEvent):
                            errors.append(f"Tick {tick}: {e.message}")

                assert len(errors) == 0, f"Errors during ticks: {errors}"

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_text_then_silence_ticks(self, provider):
        """Test sending text followed by silence ticks to collect response."""

        async def _test():
            from tau2.voice.audio_native.qwen.events import (
                QwenAudioDeltaEvent,
                QwenResponseDoneEvent,
            )
            from tau2.voice.audio_native.qwen.provider import QwenVADConfig

            await provider.connect()
            await provider.configure_session(
                system_prompt="You are a helpful assistant. Respond with just one word.",
                tools=[],
                vad_config=QwenVADConfig(),
            )

            try:
                # First, send text to trigger response
                await provider.send_text("Say hello")

                # Then run tick-like receive cycles
                total_audio_chunks = 0
                response_done = False

                for tick in range(50):  # Up to 50 ticks (10 seconds)
                    events = await provider.receive_events_for_duration(0.2)

                    for e in events:
                        if isinstance(e, QwenAudioDeltaEvent):
                            total_audio_chunks += 1
                        elif isinstance(e, QwenResponseDoneEvent):
                            response_done = True

                    if response_done:
                        break

                assert total_audio_chunks > 0, "Expected audio response"
                assert response_done, "Expected response.done event"

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestQwenRealtimeProviderToolCalls:
    """Test tool call functionality."""

    def test_tools_configured(self, provider):
        """Test that tools are properly configured."""

        async def _test():
            from tau2.environment.tool import Tool
            from tau2.voice.audio_native.qwen.provider import QwenVADConfig

            # Create a simple test tool
            def get_weather(location: str) -> str:
                """Get the weather for a location."""
                return f"Sunny in {location}"

            tool = Tool(
                name="get_weather",
                description="Get the weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
                func=get_weather,
            )

            await provider.connect()
            await provider.configure_session(
                system_prompt="You are a helpful assistant with weather tools.",
                tools=[tool],
                vad_config=QwenVADConfig(),
            )

            try:
                # Just verify connection succeeds with tools
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_function_call_triggered(self, provider):
        """Test that a function call can be triggered and completed."""

        async def _test():
            from tau2.environment.tool import Tool
            from tau2.voice.audio_native.qwen.events import (
                QwenAudioTranscriptDeltaEvent,
                QwenFunctionCallArgumentsDoneEvent,
                QwenResponseDoneEvent,
            )
            from tau2.voice.audio_native.qwen.provider import QwenVADConfig

            # Create a weather tool
            def get_weather(location: str) -> str:
                """Get the current weather for a location."""
                return f"72°F and sunny in {location}"

            tool = Tool(
                name="get_weather",
                description="Get the current weather for a location. Use this when the user asks about weather.",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name",
                        },
                    },
                    "required": ["location"],
                },
                func=get_weather,
            )

            await provider.connect()
            await provider.configure_session(
                system_prompt="You are a helpful assistant. Always use the get_weather tool when asked about weather.",
                tools=[tool],
                vad_config=QwenVADConfig(),
            )

            try:
                # Send text that should trigger the tool
                await provider.send_text(
                    "What's the weather in San Francisco right now?"
                )

                # Receive events for up to 15 seconds
                events = await provider.receive_events_for_duration(15.0)

                # Check if function call was triggered
                function_calls = [
                    e
                    for e in events
                    if isinstance(e, QwenFunctionCallArgumentsDoneEvent)
                ]

                # Note: The model may or may not trigger the tool depending on its behavior
                # This test verifies the infrastructure is working
                print(f"Received {len(function_calls)} function calls")
                for fc in function_calls:
                    print(f"  - {fc.name}({fc.arguments})")

                assert len(function_calls) > 0, "Expected at least one function call"

                # If we got a function call, send the result and verify response
                if function_calls:
                    fc = function_calls[0]
                    assert fc.name == "get_weather", (
                        f"Expected get_weather, got {fc.name}"
                    )

                    # Send tool result
                    result = '{"temperature": "72°F", "condition": "sunny"}'
                    await provider.send_tool_result(fc.call_id, result)

                    # Wait for final response
                    final_events = await provider.receive_events_for_duration(10.0)

                    # Verify we got a response after tool result
                    response_done = any(
                        isinstance(e, QwenResponseDoneEvent) for e in final_events
                    )
                    transcript = "".join(
                        e.delta
                        for e in final_events
                        if isinstance(e, QwenAudioTranscriptDeltaEvent)
                    )

                    assert response_done, "Expected response.done after tool result"
                    assert len(transcript) > 0, "Expected transcript in final response"

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestQwenRealtimeProviderAudioFormats:
    """Test audio format handling."""

    def test_audio_response_is_base64_decodable(self, provider):
        """Test that audio responses can be decoded from base64."""

        async def _test():
            from tau2.voice.audio_native.qwen.events import (
                QwenAudioDeltaEvent,
            )
            from tau2.voice.audio_native.qwen.provider import QwenVADConfig

            await provider.connect()
            await provider.configure_session(
                system_prompt="Say hello briefly.",
                tools=[],
                vad_config=QwenVADConfig(),
            )

            try:
                await provider.send_text("Hi")
                events = await provider.receive_events_for_duration(10.0)

                audio_events = [e for e in events if isinstance(e, QwenAudioDeltaEvent)]
                assert len(audio_events) > 0, "Expected audio events"

                # Verify all audio chunks are valid base64
                total_bytes = 0
                for event in audio_events:
                    if event.delta:
                        audio_bytes = base64.b64decode(event.delta)
                        total_bytes += len(audio_bytes)

                assert total_bytes > 0, "Expected decoded audio bytes"
                print(f"Received {total_bytes} bytes of audio")

            finally:
                await provider.disconnect()

        asyncio.run(_test())
