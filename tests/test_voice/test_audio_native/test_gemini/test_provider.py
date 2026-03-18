"""Tests for GeminiLiveProvider.

These tests verify the provider's core functionality:
- Session connection and disconnection
- Audio/text send and receive
- Event parsing
- Multiple tick-like cycles

Note: These tests require GEMINI_API_KEY to be set and make real API calls.
They are marked with @pytest.mark.integration to allow skipping in CI.
"""

import asyncio
import os
from typing import List

import pytest

# Skip all tests if API key not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)


@pytest.fixture
def provider():
    """Create a GeminiLiveProvider instance."""
    from tau2.voice.audio_native.gemini.provider import GeminiLiveProvider

    return GeminiLiveProvider()


class TestGeminiLiveProviderBasic:
    """Basic provider connection and communication tests."""

    def test_connect_and_disconnect(self, provider):
        """Test basic connection and disconnection."""

        async def _test():
            assert not provider.is_connected

            await provider.connect(
                system_prompt="You are a helpful assistant.",
                tools=[],
                modality="audio",
            )
            assert provider.is_connected

            await provider.disconnect()
            assert not provider.is_connected

        asyncio.run(_test())

    def test_send_text_receive_audio(self, provider):
        """Test sending text and receiving audio response."""

        async def _test():
            from tau2.voice.audio_native.gemini.events import (
                GeminiAudioDeltaEvent,
                GeminiTurnCompleteEvent,
            )

            await provider.connect(
                system_prompt="You are a helpful assistant. Respond briefly.",
                tools=[],
                modality="audio",
            )

            try:
                # Send text (more reliable than random audio for testing)
                await provider.send_text("Hello, please say hi.", end_of_turn=True)

                # Receive events for up to 5 seconds
                events = await provider.receive_events_for_duration(5.0)

                # Verify we got audio and turn complete
                audio_bytes_received = sum(
                    len(e.data) for e in events if isinstance(e, GeminiAudioDeltaEvent)
                )
                turn_complete_received = any(
                    isinstance(e, GeminiTurnCompleteEvent) for e in events
                )

                assert audio_bytes_received > 0, "Expected audio response"
                assert turn_complete_received, "Expected turn complete event"

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_send_audio_connection_stays_open(self, provider):
        """Test sending audio keeps connection open."""

        async def _test():
            await provider.connect(
                system_prompt="You are a helpful assistant.",
                tools=[],
                modality="audio",
            )

            try:
                # Generate test audio (200ms at 16kHz = 3200 samples)
                import random

                random.seed(42)
                audio_samples = [random.randint(-100, 100) for _ in range(3200)]
                audio_bytes = b"".join(
                    s.to_bytes(2, "little", signed=True) for s in audio_samples
                )

                # Send audio
                await provider.send_audio(audio_bytes)

                # Verify connection is still open after sending
                assert provider.is_connected, (
                    "Connection should stay open after sending audio"
                )

                # Receive events briefly (random noise may not trigger response)
                _ = await provider.receive_events_for_duration(0.5)

                # Connection should still be open
                assert provider.is_connected, (
                    "Connection should stay open after receiving"
                )

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestGeminiLiveProviderMultipleTicks:
    """Test provider with multiple tick-like cycles."""

    def test_multiple_tick_cycles(self, provider):
        """Test multiple send/receive cycles simulating ticks."""

        async def _test():
            from tau2.voice.audio_native.gemini.events import (
                GeminiErrorEvent,
            )

            await provider.connect(
                system_prompt="You are a helpful assistant. Respond briefly.",
                tools=[],
                modality="audio",
            )

            try:
                import random

                random.seed(42)

                tick_duration_sec = 0.2
                num_ticks = 5
                errors: List[str] = []

                for tick in range(num_ticks):
                    # Generate audio for this tick (200ms = 3200 samples at 16kHz)
                    audio_samples = [random.randint(-100, 100) for _ in range(3200)]
                    audio_bytes = b"".join(
                        s.to_bytes(2, "little", signed=True) for s in audio_samples
                    )

                    # Send audio
                    await provider.send_audio(audio_bytes)

                    # Receive for tick duration
                    events = await provider.receive_events_for_duration(
                        tick_duration_sec
                    )

                    # Check for errors
                    for e in events:
                        if isinstance(e, GeminiErrorEvent):
                            errors.append(f"Tick {tick}: {e.message}")

                assert len(errors) == 0, f"Errors during ticks: {errors}"

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_text_then_silence_ticks(self, provider):
        """Test sending text followed by silence ticks."""

        async def _test():
            from tau2.voice.audio_native.gemini.events import (
                GeminiAudioDeltaEvent,
                GeminiTurnCompleteEvent,
            )

            await provider.connect(
                system_prompt="You are a helpful assistant. Respond with just one word.",
                tools=[],
                modality="audio",
            )

            try:
                # First, send text to trigger response
                await provider.send_text("Say hello", end_of_turn=True)

                # Then run tick-like receive cycles
                total_audio_bytes = 0
                turn_complete = False

                for tick in range(25):  # Up to 25 ticks (5 seconds)
                    events = await provider.receive_events_for_duration(0.2)

                    for e in events:
                        if isinstance(e, GeminiAudioDeltaEvent):
                            total_audio_bytes += len(e.data)
                        elif isinstance(e, GeminiTurnCompleteEvent):
                            turn_complete = True

                    if turn_complete:
                        break

                assert total_audio_bytes > 0, "Expected audio response"
                # Note: turn_complete may not always arrive within timeout
                # The key test is that we received audio

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestGeminiLiveProviderToolCalls:
    """Test tool call functionality."""

    def test_tools_configured(self, provider):
        """Test that tools are properly configured."""

        async def _test():
            from tau2.environment.tool import Tool

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

            await provider.connect(
                system_prompt="You are a helpful assistant with weather tools.",
                tools=[tool],
                modality="audio",
            )

            try:
                # Just verify connection succeeds with tools
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())
