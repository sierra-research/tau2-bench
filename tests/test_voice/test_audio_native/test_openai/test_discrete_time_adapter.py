"""Tests for DiscreteTimeAudioNativeAdapter using a mock provider.

These tests verify the adapter's core logic without making real API calls:
- Tick-based audio exchange
- Audio buffering across ticks
- Proportional transcript distribution
- Interruption handling (SpeechStartedEvent)
- Tool call extraction
- buffer_until_complete mode
"""

import asyncio
import base64
import time
from typing import AsyncGenerator, List, Optional

import pytest

from tau2.data_model.audio import TELEPHONY_SAMPLE_RATE
from tau2.environment.tool import Tool
from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeAudioNativeAdapter,
)
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    AudioDoneEvent,
    AudioTranscriptDeltaEvent,
    AudioTranscriptDoneEvent,
    BaseRealtimeEvent,
    FunctionCallArgumentsDoneEvent,
    ResponseDoneEvent,
    SpeechStartedEvent,
    TimeoutEvent,
)
from tau2.voice.audio_native.openai.provider import OpenAIVADConfig, OpenAIVADMode

# =============================================================================
# Mock Provider
# =============================================================================


class MockOpenAIRealtimeProvider:
    """Mock provider that yields scripted event sequences per tick.

    This allows testing adapter logic without making real API calls.
    Events are yielded based on simulated timing within each tick.
    """

    def __init__(
        self,
        tick_duration_ms: int = 1000,
        event_script: Optional[List[List[BaseRealtimeEvent]]] = None,
    ):
        """Initialize mock provider.

        Args:
            tick_duration_ms: Duration of each tick in ms (for timing simulation).
            event_script: List of event lists, one per tick. Each inner list
                contains events to yield during that tick.
        """
        self.tick_duration_ms = tick_duration_ms
        self.event_script = event_script or []
        self.tick_index = 0
        self._connected = False

        # Track calls for verification
        self.audio_sent: List[bytes] = []
        self.tool_results_sent: List[tuple] = []
        self.configure_calls: List[dict] = []
        self.truncate_calls: List[dict] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def configure_session(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: OpenAIVADConfig,
        modality: str = "audio",
        audio_format: Optional[object] = None,
    ) -> None:
        self.configure_calls.append(
            {
                "system_prompt": system_prompt,
                "tools": tools,
                "vad_config": vad_config,
                "modality": modality,
                "audio_format": audio_format,
            }
        )

    async def send_audio(self, audio_data: bytes) -> None:
        self.audio_sent.append(audio_data)

    async def send_tool_result(
        self, call_id: str, result: str, request_response: bool = True
    ) -> None:
        self.tool_results_sent.append((call_id, result, request_response))

    async def truncate_item(
        self, item_id: str, content_index: int, audio_end_ms: int
    ) -> None:
        """Mock truncate_item - records calls for verification."""
        self.truncate_calls.append(
            {
                "item_id": item_id,
                "content_index": content_index,
                "audio_end_ms": audio_end_ms,
            }
        )

    async def cancel_response(self) -> None:
        """Mock cancel_response - no-op for tests."""
        pass

    async def receive_events(self) -> AsyncGenerator[BaseRealtimeEvent, None]:
        """Yield events for the current tick, then stop.

        Events are yielded with simulated delays to mimic real API timing.
        The generator stops after yielding all events for the current tick
        plus a timeout event to signal tick completion.
        """
        if self.tick_index < len(self.event_script):
            events = self.event_script[self.tick_index]
            self.tick_index += 1

            # Simulate events arriving over time
            if events:
                delay_per_event = (self.tick_duration_ms / 1000) / (len(events) + 1)
                for event in events:
                    await asyncio.sleep(delay_per_event)
                    yield event

        # Yield timeout events until tick time is up (the adapter will exit)
        for _ in range(100):  # Safety limit
            await asyncio.sleep(0.01)
            yield TimeoutEvent(type="timeout")


# =============================================================================
# Helper Functions
# =============================================================================


def make_audio_bytes(duration_ms: int) -> bytes:
    """Create audio bytes for a given duration at telephony sample rate."""
    num_bytes = int(TELEPHONY_SAMPLE_RATE * duration_ms / 1000)
    # Use non-silence values so we can detect them
    return bytes([0x50] * num_bytes)


def make_silence_bytes(duration_ms: int) -> bytes:
    """Create silence bytes for a given duration."""
    num_bytes = int(TELEPHONY_SAMPLE_RATE * duration_ms / 1000)
    return bytes([0x7F] * num_bytes)  # μ-law silence


def make_audio_delta_event(
    audio_bytes: bytes, item_id: str = "item_1"
) -> AudioDeltaEvent:
    """Create an AudioDeltaEvent with base64-encoded audio."""
    return AudioDeltaEvent(
        type="response.audio.delta",
        delta=base64.b64encode(audio_bytes).decode("utf-8"),
        item_id=item_id,
    )


def make_transcript_delta_event(
    text: str, item_id: str = "item_1"
) -> AudioTranscriptDeltaEvent:
    """Create an AudioTranscriptDeltaEvent."""
    return AudioTranscriptDeltaEvent(
        type="response.audio_transcript.delta",
        delta=text,
        item_id=item_id,
    )


def make_audio_done_event(item_id: str = "item_1") -> AudioDoneEvent:
    """Create an AudioDoneEvent."""
    return AudioDoneEvent(
        type="response.audio.done",
        item_id=item_id,
    )


def make_transcript_done_event(item_id: str = "item_1") -> AudioTranscriptDoneEvent:
    """Create an AudioTranscriptDoneEvent."""
    return AudioTranscriptDoneEvent(
        type="response.audio_transcript.done",
        transcript="",
        item_id=item_id,
    )


def make_speech_started_event(audio_start_ms: int) -> SpeechStartedEvent:
    """Create a SpeechStartedEvent (user interrupted)."""
    return SpeechStartedEvent(
        type="input_audio_buffer.speech_started",
        audio_start_ms=audio_start_ms,
    )


def make_function_call_done_event(
    call_id: str, name: str, arguments: str
) -> FunctionCallArgumentsDoneEvent:
    """Create a FunctionCallArgumentsDoneEvent."""
    return FunctionCallArgumentsDoneEvent(
        type="response.function_call_arguments.done",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tick_duration_ms() -> int:
    """Standard tick duration for tests."""
    return 1000  # 1 second


@pytest.fixture
def bytes_per_tick(tick_duration_ms: int) -> int:
    """Bytes per tick at telephony sample rate."""
    return int(TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000)


@pytest.fixture
def user_audio(bytes_per_tick: int) -> bytes:
    """Standard user audio for one tick."""
    return make_silence_bytes(1000)


@pytest.fixture
def vad_config() -> OpenAIVADConfig:
    """VAD config for tests."""
    return OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)


# =============================================================================
# Test: Basic Tick Execution
# =============================================================================


class TestBasicTickExecution:
    """Tests for basic tick execution without complex scenarios."""

    def test_adapter_connects_and_configures(
        self, tick_duration_ms: int, vad_config: OpenAIVADConfig
    ):
        """Test that adapter properly connects and configures the provider."""
        mock_provider = MockOpenAIRealtimeProvider(tick_duration_ms=tick_duration_ms)

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test prompt",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        assert adapter.is_connected
        assert mock_provider.is_connected
        assert len(mock_provider.configure_calls) == 1
        assert mock_provider.configure_calls[0]["system_prompt"] == "Test prompt"

        adapter.disconnect()
        assert not adapter.is_connected

    def test_tick_with_no_agent_response(
        self,
        tick_duration_ms: int,
        bytes_per_tick: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test a tick where agent doesn't respond (silence)."""
        # No events - agent is silent
        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[[]],  # Empty event list for tick 1
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        assert result.tick_number == 1
        assert result.audio_sent_bytes == len(user_audio)
        assert result.agent_audio_bytes == 0  # No agent audio
        assert result.proportional_transcript == ""

        # Played audio should be silence (padded)
        played = result.get_played_agent_audio()
        assert len(played) == bytes_per_tick
        assert played == make_silence_bytes(tick_duration_ms)

        adapter.disconnect()

    def test_tick_with_agent_audio_response(
        self,
        tick_duration_ms: int,
        bytes_per_tick: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test a tick where agent responds with audio."""
        agent_audio = make_audio_bytes(500)  # 500ms of audio

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    make_audio_delta_event(agent_audio, item_id="item_1"),
                    make_audio_done_event(item_id="item_1"),
                ]
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        assert result.tick_number == 1
        assert result.agent_audio_bytes == len(agent_audio)

        # Played audio should be padded to full tick
        played = result.get_played_agent_audio()
        assert len(played) == bytes_per_tick
        # First part is agent audio, rest is silence padding
        assert played[: len(agent_audio)] == agent_audio

        adapter.disconnect()

    def test_user_audio_is_sent_to_provider(
        self,
        tick_duration_ms: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that user audio is properly sent to the provider."""
        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[[]],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        adapter.run_tick(user_audio, tick_number=1)

        # Verify audio was sent
        assert len(mock_provider.audio_sent) == 1
        assert mock_provider.audio_sent[0] == user_audio

        adapter.disconnect()


# =============================================================================
# Test: Audio Buffering Across Ticks
# =============================================================================


class TestAudioBuffering:
    """Tests for audio buffering when agent audio exceeds tick capacity."""

    def test_excess_audio_buffered_for_next_tick(
        self,
        tick_duration_ms: int,
        bytes_per_tick: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that audio exceeding tick capacity is buffered."""
        # Agent responds with 1.5 ticks worth of audio in first tick
        agent_audio = make_audio_bytes(1500)  # 1.5 seconds

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [make_audio_delta_event(agent_audio, item_id="item_1")],
                [],  # Second tick - no new events, just buffered audio
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        # First tick: should cap at bytes_per_tick
        result1 = adapter.run_tick(user_audio, tick_number=1)
        played1 = result1.get_played_agent_audio()
        assert len(played1) == bytes_per_tick
        assert played1 == agent_audio[:bytes_per_tick]

        # Second tick: should get the buffered remainder
        result2 = adapter.run_tick(user_audio, tick_number=2)
        played2 = result2.get_played_agent_audio()
        assert len(played2) == bytes_per_tick

        # First part should be the buffered audio
        remaining_audio = agent_audio[bytes_per_tick:]
        assert played2[: len(remaining_audio)] == remaining_audio

        adapter.disconnect()

    def test_buffered_audio_appears_at_tick_start(
        self,
        tick_duration_ms: int,
        bytes_per_tick: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that buffered audio appears at the start of next tick."""
        # 1.2 ticks of audio - 0.2 ticks will be buffered
        agent_audio = make_audio_bytes(1200)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [make_audio_delta_event(agent_audio, item_id="item_1")],
                [],  # No new audio in tick 2
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        _result1 = adapter.run_tick(user_audio, tick_number=1)
        result2 = adapter.run_tick(user_audio, tick_number=2)

        # Tick 2 should have the buffered audio (0.2 ticks = 200ms)
        expected_buffered = agent_audio[bytes_per_tick:]
        assert result2.agent_audio_bytes == len(expected_buffered)

        adapter.disconnect()


# =============================================================================
# Test: Proportional Transcript
# =============================================================================


class TestProportionalTranscript:
    """Tests for proportional transcript distribution based on audio played."""

    def test_transcript_proportional_to_audio(
        self,
        tick_duration_ms: int,
        bytes_per_tick: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that transcript is distributed proportionally to audio."""
        # 2 seconds of audio with transcript "Hello World!"
        agent_audio = make_audio_bytes(2000)
        transcript = "Hello World!"

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    make_audio_delta_event(agent_audio, item_id="item_1"),
                    make_transcript_delta_event(transcript, item_id="item_1"),
                ],
                [],  # Second tick gets buffered audio
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result1 = adapter.run_tick(user_audio, tick_number=1)
        result2 = adapter.run_tick(user_audio, tick_number=2)

        # Tick 1 plays 1 second (50% of 2 second audio)
        # Should get ~50% of transcript (first 6 chars of 12)
        assert len(result1.proportional_transcript) > 0

        # Tick 2 plays remaining 1 second
        # Should get remaining transcript
        assert len(result2.proportional_transcript) > 0

        # Combined should be full transcript
        full_transcript = (
            result1.proportional_transcript + result2.proportional_transcript
        )
        assert full_transcript == transcript

        adapter.disconnect()

    def test_transcript_without_audio_is_empty(
        self,
        tick_duration_ms: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that transcript events without audio don't produce output."""
        # Only transcript, no audio
        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [make_transcript_delta_event("Hello", item_id="item_1")],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        # No audio played, so no transcript shown
        assert result.proportional_transcript == ""

        adapter.disconnect()


# =============================================================================
# Test: Interruption Handling
# =============================================================================


class TestInterruptionHandling:
    """Tests for handling user interruptions (SpeechStartedEvent)."""

    def test_speech_started_truncates_audio(
        self,
        tick_duration_ms: int,
        bytes_per_tick: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that SpeechStartedEvent causes audio truncation."""
        agent_audio = make_audio_bytes(1000)  # Full tick of audio

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    make_audio_delta_event(agent_audio, item_id="item_1"),
                    # User interrupts at 500ms into the tick
                    make_speech_started_event(audio_start_ms=500),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        # Result should be marked as truncated
        assert result.was_truncated
        assert result.interruption_audio_start_ms == 500

        # get_played_agent_audio() returns the actual audio that would play,
        # which includes audio up to the interruption point + silence padding.
        # The key is that was_truncated is set correctly for downstream handling.
        played_audio = result.get_played_agent_audio()
        assert len(played_audio) == bytes_per_tick  # Always padded to full tick

        # The non-silence portion should be less than full audio (truncated at 500ms)
        # Count non-silence bytes (0x50 is our test audio marker)
        non_silence = sum(1 for b in played_audio if b == 0x50)
        expected_max_bytes = int(TELEPHONY_SAMPLE_RATE * 500 / 1000)  # 500ms worth
        assert non_silence <= expected_max_bytes

        adapter.disconnect()

    def test_buffered_audio_cleared_on_interruption(
        self,
        tick_duration_ms: int,
        bytes_per_tick: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that buffered audio is discarded when user interrupts."""
        # Agent sends 1.5 ticks of audio, gets interrupted in tick 2
        agent_audio = make_audio_bytes(1500)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [make_audio_delta_event(agent_audio, item_id="item_1")],
                [make_speech_started_event(audio_start_ms=1200)],  # Interrupt in tick 2
                [],  # Tick 3 should have no buffered audio
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result1 = adapter.run_tick(user_audio, tick_number=1)
        result2 = adapter.run_tick(user_audio, tick_number=2)
        result3 = adapter.run_tick(user_audio, tick_number=3)

        # Tick 1 should have full audio
        assert result1.agent_audio_bytes == bytes_per_tick

        # Tick 2 was interrupted
        assert result2.was_truncated

        # Tick 3 should have no buffered audio from the interrupted utterance
        assert result3.agent_audio_bytes == 0

        adapter.disconnect()


# =============================================================================
# Test: Tool Call Extraction
# =============================================================================


class TestToolCallExtraction:
    """Tests for extracting tool calls from events."""

    def test_function_call_captured_in_events(
        self,
        tick_duration_ms: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that FunctionCallArgumentsDoneEvent is captured in tick events."""
        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    make_function_call_done_event(
                        call_id="call_123",
                        name="get_weather",
                        arguments='{"city": "Seattle"}',
                    ),
                    ResponseDoneEvent(type="response.done"),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        # Check that the function call event is in the events list
        function_call_events = [
            e for e in result.events if isinstance(e, FunctionCallArgumentsDoneEvent)
        ]
        assert len(function_call_events) == 1
        assert function_call_events[0].call_id == "call_123"
        assert function_call_events[0].name == "get_weather"
        assert function_call_events[0].arguments == '{"city": "Seattle"}'

        adapter.disconnect()

    def test_tool_result_sent_to_provider(
        self,
        tick_duration_ms: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that send_tool_result queues result for next tick."""
        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[[], []],  # Two empty ticks
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        # Queue a tool result
        adapter.send_tool_result(
            call_id="call_123",
            result="Sunny, 72°F",
            request_response=True,
        )

        # Result should be queued, not sent yet
        assert len(mock_provider.tool_results_sent) == 0

        # Run a tick - this should send the queued tool result
        adapter.run_tick(user_audio, tick_number=1)

        # Now it should be sent
        assert len(mock_provider.tool_results_sent) == 1
        assert mock_provider.tool_results_sent[0] == ("call_123", "Sunny, 72°F", True)

        adapter.disconnect()


# =============================================================================
# Test: Buffer Until Complete Mode
# =============================================================================


class TestBufferUntilCompleteMode:
    """Tests for buffer_until_complete=True mode."""

    def test_audio_held_until_done_event(
        self,
        tick_duration_ms: int,
        bytes_per_tick: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that audio is buffered until AudioDoneEvent."""
        agent_audio = make_audio_bytes(500)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                # Tick 1: Audio arrives but no done event
                [make_audio_delta_event(agent_audio, item_id="item_1")],
                # Tick 2: Done events arrive
                [
                    make_audio_done_event(item_id="item_1"),
                    make_transcript_done_event(item_id="item_1"),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=True,  # <-- Key setting
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        # Tick 1: Audio should NOT appear yet (waiting for done)
        result1 = adapter.run_tick(user_audio, tick_number=1)
        assert result1.agent_audio_bytes == 0

        # Tick 2: Audio should appear after done events
        result2 = adapter.run_tick(user_audio, tick_number=2)
        assert result2.agent_audio_bytes == len(agent_audio)

        adapter.disconnect()

    def test_transcript_released_with_audio(
        self,
        tick_duration_ms: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that transcript is released when utterance is complete."""
        agent_audio = make_audio_bytes(500)
        transcript = "Hello there"

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    make_audio_delta_event(agent_audio, item_id="item_1"),
                    make_transcript_delta_event(transcript, item_id="item_1"),
                ],
                [
                    make_audio_done_event(item_id="item_1"),
                    make_transcript_done_event(item_id="item_1"),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=True,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result1 = adapter.run_tick(user_audio, tick_number=1)
        result2 = adapter.run_tick(user_audio, tick_number=2)

        # Tick 1: No transcript (waiting)
        assert result1.proportional_transcript == ""

        # Tick 2: Full transcript (released)
        assert result2.proportional_transcript == transcript

        adapter.disconnect()


# =============================================================================
# Test: Timing Guarantees
# =============================================================================


class TestTimingGuarantees:
    """Tests for tick timing requirements."""

    def test_tick_takes_at_least_tick_duration(
        self,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that each tick takes at least tick_duration_ms wall-clock time."""
        # Use a short tick for faster test
        tick_duration_ms = 100

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[[]],  # No events - quick processing
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        start_time = time.time()
        adapter.run_tick(user_audio[:100], tick_number=1)
        elapsed_ms = (time.time() - start_time) * 1000

        # Should take at least tick_duration_ms (with some tolerance for test overhead)
        assert elapsed_ms >= tick_duration_ms * 0.9  # 90% tolerance

        adapter.disconnect()

    @pytest.mark.skip(reason="tick_wall_duration_ms not implemented in TickResult")
    def test_tick_wall_duration_recorded(
        self,
        tick_duration_ms: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that tick records wall-clock duration."""
        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[[]],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        # Wall duration should be at least tick duration
        assert result.tick_wall_duration_ms >= tick_duration_ms * 0.9

        adapter.disconnect()


# =============================================================================
# Test: Multi-Tick Conversations
# =============================================================================


class TestMultiTickConversations:
    """Tests for multi-tick conversation scenarios."""

    def test_multiple_utterances_across_ticks(
        self,
        tick_duration_ms: int,
        bytes_per_tick: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test handling multiple agent utterances across ticks."""
        audio1 = make_audio_bytes(500)
        audio2 = make_audio_bytes(500)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    make_audio_delta_event(audio1, item_id="item_1"),
                    make_transcript_delta_event("First ", item_id="item_1"),
                    make_audio_done_event(item_id="item_1"),
                ],
                [
                    make_audio_delta_event(audio2, item_id="item_2"),
                    make_transcript_delta_event("Second", item_id="item_2"),
                    make_audio_done_event(item_id="item_2"),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result1 = adapter.run_tick(user_audio, tick_number=1)
        result2 = adapter.run_tick(user_audio, tick_number=2)

        # Both ticks should have audio
        assert result1.agent_audio_bytes == len(audio1)
        assert result2.agent_audio_bytes == len(audio2)

        # Transcripts should be separate
        assert result1.proportional_transcript == "First "
        assert result2.proportional_transcript == "Second"

        adapter.disconnect()

    def test_cumulative_user_audio_tracking(
        self,
        tick_duration_ms: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that cumulative user audio is tracked correctly across ticks."""
        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[[], [], []],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result1 = adapter.run_tick(user_audio, tick_number=1)
        result2 = adapter.run_tick(user_audio, tick_number=2)
        result3 = adapter.run_tick(user_audio, tick_number=3)

        # Each tick should have correct cumulative tracking
        assert result1.cumulative_user_audio_at_tick_start_ms == 0
        assert result2.cumulative_user_audio_at_tick_start_ms == tick_duration_ms
        assert result3.cumulative_user_audio_at_tick_start_ms == 2 * tick_duration_ms

        adapter.disconnect()


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_invalid_tick_duration_raises_error(self):
        """Test that invalid tick duration raises ValueError."""
        mock_provider = MockOpenAIRealtimeProvider()

        with pytest.raises(ValueError, match="tick_duration_ms must be > 0"):
            DiscreteTimeAudioNativeAdapter(
                tick_duration_ms=0,
                send_audio_instant=True,
                buffer_until_complete=False,
                provider=mock_provider,
            )

        with pytest.raises(ValueError, match="tick_duration_ms must be > 0"):
            DiscreteTimeAudioNativeAdapter(
                tick_duration_ms=-100,
                send_audio_instant=True,
                buffer_until_complete=False,
                provider=mock_provider,
            )

    def test_run_tick_without_connect_raises_error(
        self,
        tick_duration_ms: int,
        user_audio: bytes,
    ):
        """Test that run_tick before connect raises RuntimeError."""
        mock_provider = MockOpenAIRealtimeProvider(tick_duration_ms=tick_duration_ms)

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        with pytest.raises(RuntimeError, match="Not connected"):
            adapter.run_tick(user_audio, tick_number=1)

    def test_empty_user_audio(
        self,
        tick_duration_ms: int,
        vad_config: OpenAIVADConfig,
    ):
        """Test handling of empty user audio."""
        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[[]],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        # Empty audio should work (just 0 bytes sent)
        result = adapter.run_tick(b"", tick_number=1)
        assert result.audio_sent_bytes == 0

        adapter.disconnect()

    def test_clear_buffers(
        self,
        tick_duration_ms: int,
        bytes_per_tick: int,
        user_audio: bytes,
        vad_config: OpenAIVADConfig,
    ):
        """Test that clear_buffers removes all pending state."""
        # Agent sends 1.5 ticks of audio
        agent_audio = make_audio_bytes(1500)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [make_audio_delta_event(agent_audio, item_id="item_1")],
                [],  # Tick 2 would get buffered audio normally
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        # Run tick 1 to buffer excess audio
        adapter.run_tick(user_audio, tick_number=1)

        # Clear buffers
        adapter.clear_buffers()

        # Run tick 2 - should have no audio (buffers cleared)
        result2 = adapter.run_tick(user_audio, tick_number=2)
        assert result2.agent_audio_bytes == 0

        adapter.disconnect()
