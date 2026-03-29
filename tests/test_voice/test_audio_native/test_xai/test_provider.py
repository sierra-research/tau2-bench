"""Tests for XAIRealtimeProvider.

These tests verify the provider's core functionality:
- Session connection and disconnection
- Audio/text send and receive
- Event parsing
- Multiple tick-like cycles
- Tool configuration

Note: These tests require XAI_API_KEY to be set and make real API calls.
They are marked with pytest.mark.skipif to skip when API key is not available.

Reference: https://docs.x.ai/docs/guides/voice/agent
"""

import asyncio
import os
from typing import List

import pytest

# Skip all tests if API key not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY"),
    reason="XAI_API_KEY not set",
)


@pytest.fixture
def provider():
    """Create an XAIRealtimeProvider instance."""
    from tau2.voice.audio_native.xai.provider import XAIRealtimeProvider

    return XAIRealtimeProvider()


class TestXAIRealtimeProviderBasic:
    """Basic provider connection and communication tests."""

    def test_connect_and_disconnect(self, provider):
        """Test basic connection and disconnection."""

        async def _test():
            from tau2.voice.audio_native.xai.provider import XAIVADConfig

            assert not provider.is_connected

            await provider.connect()
            assert provider.is_connected

            # Configure session
            await provider.configure_session(
                system_prompt="You are a helpful assistant.",
                tools=[],
                vad_config=XAIVADConfig(),
            )

            await provider.disconnect()
            assert not provider.is_connected

        asyncio.run(_test())

    def test_send_text_receive_audio(self, provider):
        """Test sending text and receiving audio response."""

        async def _test():
            from tau2.voice.audio_native.xai.events import (
                XAIAudioDeltaEvent,
                XAIResponseDoneEvent,
            )
            from tau2.voice.audio_native.xai.provider import XAIVADConfig

            await provider.connect()
            await provider.configure_session(
                system_prompt="You are a helpful assistant. Respond briefly.",
                tools=[],
                vad_config=XAIVADConfig(),
            )

            try:
                # Send text to trigger response
                await provider.send_text("Hello, please say hi briefly.")

                # Receive events for up to 5 seconds
                events = await provider.receive_events_for_duration(5.0)

                # Verify we got audio and response done
                audio_chunks_received = sum(
                    1 for e in events if isinstance(e, XAIAudioDeltaEvent)
                )
                response_done_received = any(
                    isinstance(e, XAIResponseDoneEvent) for e in events
                )

                assert audio_chunks_received > 0, "Expected audio response"
                assert response_done_received, "Expected response.done event"

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_send_audio_connection_stays_open(self, provider):
        """Test sending audio keeps connection open."""

        async def _test():
            from tau2.voice.audio_native.xai.provider import XAIVADConfig

            await provider.connect()
            await provider.configure_session(
                system_prompt="You are a helpful assistant.",
                tools=[],
                vad_config=XAIVADConfig(),
            )

            try:
                # Generate test audio (200ms at 8kHz = 1600 bytes for μ-law)
                # Just send silence (0x7f is μ-law silence)
                audio_bytes = b"\x7f" * 1600

                # Send audio
                await provider.send_audio(audio_bytes)

                # Verify connection is still open after sending
                assert provider.is_connected, (
                    "Connection should stay open after sending audio"
                )

                # Receive events briefly
                _ = await provider.receive_events_for_duration(0.5)

                # Connection should still be open
                assert provider.is_connected, (
                    "Connection should stay open after receiving"
                )

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestXAIRealtimeProviderMultipleTicks:
    """Test provider with multiple tick-like cycles."""

    def test_multiple_tick_cycles(self, provider):
        """Test multiple send/receive cycles simulating ticks."""

        async def _test():
            from tau2.voice.audio_native.xai.events import XAIErrorEvent
            from tau2.voice.audio_native.xai.provider import XAIVADConfig

            await provider.connect()
            await provider.configure_session(
                system_prompt="You are a helpful assistant. Respond briefly.",
                tools=[],
                vad_config=XAIVADConfig(),
            )

            try:
                tick_duration_sec = 0.2
                num_ticks = 5
                errors: List[str] = []

                for tick in range(num_ticks):
                    # Generate audio for this tick (200ms = 1600 bytes at 8kHz μ-law)
                    audio_bytes = b"\x7f" * 1600  # μ-law silence

                    # Send audio
                    await provider.send_audio(audio_bytes)

                    # Receive for tick duration
                    events = await provider.receive_events_for_duration(
                        tick_duration_sec
                    )

                    # Check for errors
                    for e in events:
                        if isinstance(e, XAIErrorEvent):
                            errors.append(f"Tick {tick}: {e.message}")

                assert len(errors) == 0, f"Errors during ticks: {errors}"

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_text_then_silence_ticks(self, provider):
        """Test sending text followed by silence ticks."""

        async def _test():
            from tau2.voice.audio_native.xai.events import (
                XAIAudioDeltaEvent,
                XAIResponseDoneEvent,
            )
            from tau2.voice.audio_native.xai.provider import XAIVADConfig

            await provider.connect()
            await provider.configure_session(
                system_prompt="You are a helpful assistant. Respond with just one word.",
                tools=[],
                vad_config=XAIVADConfig(),
            )

            try:
                # First, send text to trigger response
                await provider.send_text("Say hello")

                # Then run tick-like receive cycles
                total_audio_chunks = 0
                response_done = False

                for tick in range(25):  # Up to 25 ticks (5 seconds)
                    events = await provider.receive_events_for_duration(0.2)

                    for e in events:
                        if isinstance(e, XAIAudioDeltaEvent):
                            total_audio_chunks += 1
                        elif isinstance(e, XAIResponseDoneEvent):
                            response_done = True

                    if response_done:
                        break

                assert total_audio_chunks > 0, "Expected audio response"
                assert response_done, "Expected response.done event"

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestXAIRealtimeProviderToolCalls:
    """Test tool call functionality."""

    def test_tools_configured(self, provider):
        """Test that tools are properly configured."""

        async def _test():
            from tau2.environment.tool import Tool
            from tau2.voice.audio_native.xai.provider import XAIVADConfig

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
                vad_config=XAIVADConfig(),
            )

            try:
                # Just verify connection succeeds with tools
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_function_call_triggered(self, provider):
        """Test that a function call can be triggered."""

        async def _test():
            from tau2.environment.tool import Tool
            from tau2.voice.audio_native.xai.events import (
                XAIFunctionCallArgumentsDoneEvent,
            )
            from tau2.voice.audio_native.xai.provider import XAIVADConfig

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
                vad_config=XAIVADConfig(),
            )

            try:
                # Send text that should trigger the tool
                await provider.send_text(
                    "What's the weather in San Francisco right now?"
                )

                # Receive events for up to 10 seconds
                events = await provider.receive_events_for_duration(10.0)

                # Check if function call was triggered
                function_calls = [
                    e
                    for e in events
                    if isinstance(e, XAIFunctionCallArgumentsDoneEvent)
                ]

                # Note: The model may or may not trigger the tool depending on its behavior
                # This test verifies the infrastructure is working, not model behavior
                print(f"Received {len(function_calls)} function calls")
                for fc in function_calls:
                    print(f"  - {fc.name}({fc.arguments})")

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestXAIRealtimeProviderAudioFormats:
    """Test different audio format configurations."""

    def test_pcmu_format(self):
        """Test G.711 μ-law format (default, optimal for telephony)."""

        async def _test():
            from tau2.voice.audio_native.xai.provider import (
                XAIAudioFormat,
                XAIRealtimeProvider,
                XAIVADConfig,
            )

            provider = XAIRealtimeProvider(audio_format=XAIAudioFormat.PCMU)

            await provider.connect()
            await provider.configure_session(
                system_prompt="Say hello briefly.",
                tools=[],
                vad_config=XAIVADConfig(),
            )

            try:
                await provider.send_text("Hi")
                events = await provider.receive_events_for_duration(3.0)

                # Should receive audio in μ-law format
                from tau2.voice.audio_native.xai.events import XAIAudioDeltaEvent

                audio_events = [e for e in events if isinstance(e, XAIAudioDeltaEvent)]
                assert len(audio_events) > 0, "Expected audio events"

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_pcm_format_24khz(self):
        """Test PCM format at 24kHz."""

        async def _test():
            from tau2.voice.audio_native.xai.provider import (
                XAIAudioFormat,
                XAIRealtimeProvider,
                XAIVADConfig,
            )

            provider = XAIRealtimeProvider(
                audio_format=XAIAudioFormat.PCM, sample_rate=24000
            )

            await provider.connect()
            await provider.configure_session(
                system_prompt="Say hello briefly.",
                tools=[],
                vad_config=XAIVADConfig(),
            )

            try:
                await provider.send_text("Hi")
                events = await provider.receive_events_for_duration(3.0)

                # Should receive audio in PCM format
                from tau2.voice.audio_native.xai.events import XAIAudioDeltaEvent

                audio_events = [e for e in events if isinstance(e, XAIAudioDeltaEvent)]
                assert len(audio_events) > 0, "Expected audio events"

            finally:
                await provider.disconnect()

        asyncio.run(_test())
