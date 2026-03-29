"""Precision tests for interruption handling in DiscreteTimeAudioNativeAdapter.

These tests verify the exact timing and byte-level accuracy of interruption handling,
specifically addressing concerns about audio being truncated too early.
"""

import base64

# Import mock provider from existing tests
import sys
from pathlib import Path

import pytest

from tau2.data_model.audio import TELEPHONY_SAMPLE_RATE
from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeAudioNativeAdapter,
)
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    SpeechStartedEvent,
)
from tau2.voice.audio_native.openai.provider import OpenAIVADConfig, OpenAIVADMode

test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir))
from test_discrete_time_adapter import (  # noqa: E402
    MockOpenAIRealtimeProvider,
    make_audio_bytes,
    make_silence_bytes,
)


class TestInterruptionPrecision:
    """Tests for precise interruption timing behavior."""

    @pytest.mark.parametrize(
        "interrupt_at_ms,expected_audio_ms",
        [
            (100, 100),  # Interrupt at 100ms
            (250, 250),  # Interrupt at 250ms (quarter tick)
            (500, 500),  # Interrupt at 500ms (half tick)
            (750, 750),  # Interrupt at 750ms (three-quarter tick)
            (900, 900),  # Interrupt near end
        ],
    )
    def test_interruption_at_exact_position(
        self, interrupt_at_ms: int, expected_audio_ms: int
    ):
        """Test that interruption at specific millisecond positions is precise.

        This test verifies that when a user interrupts at a specific time,
        the audio is truncated at EXACTLY that position, not earlier.
        """
        tick_duration_ms = 1000
        bytes_per_tick = int(TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000)

        # Create 1 full tick of agent audio
        agent_audio = make_audio_bytes(tick_duration_ms)
        user_audio = make_silence_bytes(tick_duration_ms)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=interrupt_at_ms,
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        # Verify truncation metadata
        assert result.was_truncated, "Result should be marked as truncated"
        assert result.interruption_audio_start_ms == interrupt_at_ms, (
            f"Expected interruption at {interrupt_at_ms}ms, got {result.interruption_audio_start_ms}ms"
        )

        # Get the played audio
        played_audio = result.get_played_agent_audio()
        assert len(played_audio) == bytes_per_tick, (
            "Audio should be padded to full tick"
        )

        # Count non-silence bytes (0x50 is the test audio marker)
        non_silence_bytes = sum(1 for b in played_audio if b == 0x50)

        # Calculate expected bytes based on interrupt position
        # This should be EXACT, not "at most"
        expected_bytes = int(TELEPHONY_SAMPLE_RATE * expected_audio_ms / 1000)

        # Allow small tolerance (±1 byte) due to rounding
        tolerance = 1
        assert abs(non_silence_bytes - expected_bytes) <= tolerance, (
            f"Expected {expected_bytes} bytes of audio (±{tolerance}), "
            f"but got {non_silence_bytes} bytes. "
            f"Audio may be truncated too early or too late. "
            f"Interrupt at {interrupt_at_ms}ms, expected {expected_audio_ms}ms of audio."
        )

        adapter.disconnect()

    def test_interruption_at_tick_boundary(self):
        """Test interruption exactly at tick start (0ms into tick)."""
        tick_duration_ms = 1000
        bytes_per_tick = int(TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000)

        # Agent sends audio, user interrupts at start of tick
        agent_audio = make_audio_bytes(tick_duration_ms)
        user_audio = make_silence_bytes(tick_duration_ms)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                    # Interrupt at 0ms (tick start)
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=0,
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        assert result.was_truncated
        assert result.interruption_audio_start_ms == 0

        played_audio = result.get_played_agent_audio()
        assert len(played_audio) == bytes_per_tick

        # Should have NO agent audio (all silence)
        non_silence_bytes = sum(1 for b in played_audio if b == 0x50)
        assert non_silence_bytes == 0, (
            f"Expected no audio at tick start, got {non_silence_bytes} bytes"
        )

        adapter.disconnect()

    def test_interruption_across_multiple_ticks(self):
        """Test interruption timing when agent audio spans multiple ticks."""
        tick_duration_ms = 1000
        bytes_per_tick = int(TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000)

        # Agent sends 2.5 ticks worth of audio, interrupted at 1500ms (halfway through tick 2)
        agent_audio = make_audio_bytes(2500)
        user_audio = make_silence_bytes(tick_duration_ms)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                # Tick 1: Agent starts speaking (1000ms of audio)
                [
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                ],
                # Tick 2: User interrupts at 1500ms (500ms into this tick)
                [
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=1500,  # 500ms into tick 2
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        # Tick 1: Full audio
        result1 = adapter.run_tick(user_audio, tick_number=1)
        played1 = result1.get_played_agent_audio()
        assert len(played1) == bytes_per_tick
        assert not result1.was_truncated

        # Tick 2: Interrupted at 500ms into tick
        result2 = adapter.run_tick(user_audio, tick_number=2)
        played2 = result2.get_played_agent_audio()
        assert len(played2) == bytes_per_tick
        assert result2.was_truncated
        assert result2.interruption_audio_start_ms == 1500

        # Verify tick 2 has exactly 500ms of audio
        non_silence_bytes = sum(1 for b in played2 if b == 0x50)
        expected_bytes = int(TELEPHONY_SAMPLE_RATE * 500 / 1000)
        tolerance = 1

        assert abs(non_silence_bytes - expected_bytes) <= tolerance, (
            f"Tick 2 should have ~{expected_bytes} bytes (500ms), "
            f"but got {non_silence_bytes} bytes"
        )

        adapter.disconnect()

    def test_get_played_agent_audio_without_interruption(self):
        """Test that get_played_agent_audio works correctly without interruption."""
        tick_duration_ms = 1000
        bytes_per_tick = int(TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000)

        # Agent sends 700ms of audio (less than full tick)
        agent_audio = make_audio_bytes(700)
        user_audio = make_silence_bytes(tick_duration_ms)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        # Should NOT be truncated
        assert not result.was_truncated
        assert result.interruption_audio_start_ms is None

        # Get played audio
        played = result.get_played_agent_audio()
        assert len(played) == bytes_per_tick, "Audio should be padded to full tick"

        # Count non-silence bytes
        non_silence_bytes = sum(1 for b in played if b == 0x50)
        expected_bytes = len(agent_audio)

        assert non_silence_bytes == expected_bytes, (
            f"Expected {expected_bytes} bytes of audio, got {non_silence_bytes}"
        )

        # Verify padding is at the end
        # First 700ms should be audio (0x50), remaining should be silence (0x7f)
        assert played[:expected_bytes] == agent_audio
        assert all(b == 0x7F for b in played[expected_bytes:])

        adapter.disconnect()

    def test_interruption_with_no_agent_audio(self):
        """Test interruption when agent hasn't produced any audio yet."""
        tick_duration_ms = 1000
        bytes_per_tick = int(TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000)
        user_audio = make_silence_bytes(tick_duration_ms)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    # User interrupts but agent hasn't produced audio
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=500,
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        # Should NOT be marked as truncated (no audio to truncate)
        assert not result.was_truncated

        # Should return full silence
        played = result.get_played_agent_audio()
        assert len(played) == bytes_per_tick
        assert all(b == 0x7F for b in played), "Should be all silence"

        adapter.disconnect()


class TestInterruptionEdgeCases:
    """Test edge cases and potential bugs in interruption handling."""

    def test_interruption_past_tick_end(self):
        """Test interruption with audio_start_ms beyond current tick."""
        tick_duration_ms = 1000
        bytes_per_tick = int(TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000)

        agent_audio = make_audio_bytes(1000)
        user_audio = make_silence_bytes(tick_duration_ms)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                    # Interrupt at 1500ms (beyond this tick's end at 1000ms)
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=1500,
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        # Should be marked as truncated
        assert result.was_truncated

        # But should play FULL tick of audio (clamped to tick end)
        played = result.get_played_agent_audio()
        non_silence_bytes = sum(1 for b in played if b == 0x50)

        # Should have full tick of audio (clamped to tick duration)
        expected_bytes = bytes_per_tick
        tolerance = 1
        assert abs(non_silence_bytes - expected_bytes) <= tolerance

        adapter.disconnect()

    def test_interruption_with_negative_position(self):
        """Test interruption with audio_start_ms before tick start (edge case)."""
        tick_duration_ms = 1000

        agent_audio = make_audio_bytes(1000)
        user_audio = make_silence_bytes(tick_duration_ms)

        # Tick 2: cumulative audio = 1000ms, so tick start = 1000ms
        # If audio_start_ms = 500ms, that's BEFORE this tick start
        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [],  # Tick 1: nothing
                [
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                    # Interrupt at 500ms (before tick 2 start at 1000ms)
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=500,
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        _result1 = adapter.run_tick(user_audio, tick_number=1)
        result2 = adapter.run_tick(user_audio, tick_number=2)

        # Should be marked as truncated
        assert result2.was_truncated

        # Should have NO audio (clamped to 0)
        played = result2.get_played_agent_audio()
        non_silence_bytes = sum(1 for b in played if b == 0x50)
        assert non_silence_bytes == 0, "Should have no audio (position before tick)"

        adapter.disconnect()


class TestCumulativeAudioTracking:
    """Test that cumulative audio tracking works correctly with interruptions."""

    def test_cumulative_tracking_matches_interruption_timing(self):
        """Test that cumulative audio tracking is correct across multiple ticks."""
        tick_duration_ms = 1000
        bytes_per_tick = int(TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000)
        user_audio = make_silence_bytes(tick_duration_ms)

        # Send 2.5 ticks worth of audio
        agent_audio = make_audio_bytes(2500)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [],  # Tick 1: silence
                [  # Tick 2: agent speaks (sends 2.5 ticks of audio)
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                ],
                [  # Tick 3: user interrupts at 2500ms (500ms into tick 3)
                    # Agent still has buffered audio playing
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=2500,
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result1 = adapter.run_tick(user_audio, tick_number=1)
        result2 = adapter.run_tick(user_audio, tick_number=2)
        result3 = adapter.run_tick(user_audio, tick_number=3)

        # Verify cumulative tracking
        assert result1.cumulative_user_audio_at_tick_start_ms == 0
        assert result2.cumulative_user_audio_at_tick_start_ms == 1000
        assert result3.cumulative_user_audio_at_tick_start_ms == 2000

        # Tick 2 should have full audio (1000ms)
        assert result2.agent_audio_bytes == bytes_per_tick

        # Tick 3 should be truncated at 2500ms
        # 2500ms - 2000ms (tick start) = 500ms into tick
        # Tick 3 has buffered audio from tick 2
        assert result3.was_truncated
        assert result3.interruption_audio_start_ms == 2500

        # Verify that get_played_agent_audio is truncated at 500ms
        played3 = result3.get_played_agent_audio()
        assert len(played3) == bytes_per_tick

        # Should have ~500ms of audio
        non_silence_bytes = sum(1 for b in played3 if b == 0x50)
        expected_bytes = int(TELEPHONY_SAMPLE_RATE * 500 / 1000)
        tolerance = 1
        assert abs(non_silence_bytes - expected_bytes) <= tolerance

        adapter.disconnect()

    def test_interruption_timing_with_variable_tick_sizes(self):
        """Test interruption with different user audio sizes per tick."""
        tick_duration_ms = 1000

        # Send different amounts of user audio per tick
        user_audio_full = make_silence_bytes(tick_duration_ms)
        user_audio_half = make_silence_bytes(500)

        agent_audio = make_audio_bytes(1000)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [  # Tick 1: agent speaks
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                ],
                [  # Tick 2: continues with more audio, user interrupts
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                    # Interrupt at 1250ms (250ms into tick 2)
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=1250,
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        # Tick 1: send full audio
        result1 = adapter.run_tick(user_audio_full, tick_number=1)
        assert result1.cumulative_user_audio_at_tick_start_ms == 0
        assert not result1.was_truncated

        # Tick 2: send half audio (500ms), but interrupt at 1250ms
        result2 = adapter.run_tick(user_audio_half, tick_number=2)
        assert result2.cumulative_user_audio_at_tick_start_ms == 1000
        assert result2.was_truncated
        assert result2.interruption_audio_start_ms == 1250

        # Verify audio is truncated at 250ms into tick
        played2 = result2.get_played_agent_audio()
        non_silence_bytes = sum(1 for b in played2 if b == 0x50)
        expected_bytes = int(TELEPHONY_SAMPLE_RATE * 250 / 1000)
        tolerance = 1

        assert abs(non_silence_bytes - expected_bytes) <= tolerance, (
            f"Expected ~{expected_bytes} bytes (250ms), got {non_silence_bytes}"
        )

        adapter.disconnect()


class TestGetPlayedAgentAudioDetailedBehavior:
    """Detailed tests for get_played_agent_audio edge cases."""

    def test_audio_chunks_split_across_calls(self):
        """Test that multiple audio chunks in one tick are handled correctly."""
        tick_duration_ms = 1000
        bytes_per_tick = int(TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000)
        user_audio = make_silence_bytes(tick_duration_ms)

        # Agent sends audio in 3 chunks
        chunk1 = make_audio_bytes(300)
        chunk2 = make_audio_bytes(300)
        chunk3 = make_audio_bytes(400)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(chunk1).decode("utf-8"),
                        item_id="item_1",
                    ),
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(chunk2).decode("utf-8"),
                        item_id="item_1",
                    ),
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(chunk3).decode("utf-8"),
                        item_id="item_1",
                    ),
                    # Interrupt at 700ms (partway through chunk3)
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=700,
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        assert result.was_truncated
        assert result.interruption_audio_start_ms == 700

        played = result.get_played_agent_audio()
        assert len(played) == bytes_per_tick

        # Should have ~700ms of audio (chunk1 + chunk2 + part of chunk3)
        non_silence_bytes = sum(1 for b in played if b == 0x50)
        expected_bytes = int(TELEPHONY_SAMPLE_RATE * 700 / 1000)
        tolerance = 1

        assert abs(non_silence_bytes - expected_bytes) <= tolerance, (
            f"Expected ~{expected_bytes} bytes (700ms), got {non_silence_bytes}"
        )

        adapter.disconnect()

    def test_bytes_per_second_calculation(self):
        """Test that bytes_per_second is used correctly in truncation."""
        tick_duration_ms = 1000
        user_audio = make_silence_bytes(tick_duration_ms)

        agent_audio = make_audio_bytes(1000)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                [
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=333,  # Odd number to test rounding
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        result = adapter.run_tick(user_audio, tick_number=1)

        assert result.was_truncated
        played = result.get_played_agent_audio()

        # Verify that bytes_per_second = TELEPHONY_SAMPLE_RATE = 8000
        # 333ms * 8000 bytes/sec / 1000 ms/sec = 2664 bytes
        expected_bytes = int(TELEPHONY_SAMPLE_RATE * 333 / 1000)
        non_silence_bytes = sum(1 for b in played if b == 0x50)

        tolerance = 1
        assert abs(non_silence_bytes - expected_bytes) <= tolerance, (
            f"Expected {expected_bytes} bytes (333ms), got {non_silence_bytes}"
        )

        adapter.disconnect()


class TestDelayedInterruptionDetection:
    """Tests for scenarios where SpeechStartedEvent arrives many ticks after audio_start_ms.

    This can happen due to network latency or server-side VAD processing delays.
    The audio_start_ms in SpeechStartedEvent indicates when speech was detected
    in the cumulative user audio buffer, but the event may arrive much later.
    """

    def test_interruption_detected_many_ticks_later(self):
        """Test when SpeechStartedEvent arrives 9 ticks after audio_start_ms.

        Scenario (mimics real-world observation):
        - Tick 1-5: Agent sends audio that spans multiple ticks
        - Ticks 6-9: Buffered audio continues playing, no new events
        - Tick 10: SpeechStartedEvent arrives with audio_start_ms=1000 (tick 5!)

        Expected behavior:
        - Ticks 1-5 already completed (audio was returned, can't be undone)
        - Tick 6-9 already completed with buffered audio
        - Tick 10: was_truncated=True, but audio_start_ms is BEFORE tick 10 start
        """
        tick_duration_ms = 200  # 200ms ticks like in QA
        bytes_per_tick = int(
            TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000
        )  # 1600 bytes

        # Agent sends 3 seconds of audio (15 ticks worth) in one burst
        agent_audio = make_audio_bytes(3000)
        user_audio = make_silence_bytes(tick_duration_ms)

        # Script: Audio arrives at tick 1, SpeechStartedEvent at tick 10
        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                # Tick 1: All audio arrives
                [
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                ],
                # Ticks 2-9: Empty (buffered audio distributes)
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                # Tick 10: SpeechStartedEvent with audio_start_ms=1000ms (tick 5)
                [
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=1000,  # Points to 1000ms = tick 5
                    ),
                ],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        results = []
        for tick_num in range(1, 12):
            result = adapter.run_tick(user_audio, tick_number=tick_num)
            results.append(result)

        # Analyze results
        # Ticks 1-9: should NOT be truncated (SpeechStartedEvent hasn't arrived yet)
        for i in range(9):
            assert not results[i].was_truncated, f"Tick {i + 1} should not be truncated"

        # Tick 10: SHOULD be truncated
        assert results[9].was_truncated, "Tick 10 should be truncated"
        assert results[9].interruption_audio_start_ms == 1000

        # Critical check: What audio does tick 10 return?
        # cumulative_user_audio_at_tick_start for tick 10 = 9 * 200 = 1800ms
        # position_within_tick_ms = 1000 - 1800 = -800ms, clamped to 0
        # So tick 10 should have 0 bytes of audio (all silence)
        played_10 = results[9].get_played_agent_audio()
        non_silence_10 = sum(1 for b in played_10 if b == 0x50)
        assert non_silence_10 == 0, (
            f"Tick 10 should have 0 audio bytes (interruption was before tick start), "
            f"but got {non_silence_10}"
        )

        # Key observation: Ticks 6-9 STILL have audio that was "played" before we knew
        # about the interruption. This is a fundamental limitation - we can't retroactively
        # undo audio that was already returned from previous ticks.

        # Calculate total "played" audio across all ticks
        total_played_bytes = 0
        for i, result in enumerate(results):
            played = result.get_played_agent_audio()
            non_silence = sum(1 for b in played if b == 0x50)
            total_played_bytes += non_silence
            print(
                f"Tick {i + 1}: {non_silence} audio bytes, was_truncated={result.was_truncated}"
            )

        # Expected: First 9 ticks have audio (bytes_per_tick each, up to available audio)
        # Tick 10+ have 0 audio
        # This means ~1800ms of audio is "played" even though interruption was at 1000ms
        # This is the "late interruption detection" problem
        expected_max_audio = 9 * bytes_per_tick  # 9 ticks of 200ms each
        print(f"Total played: {total_played_bytes}, Expected max: {expected_max_audio}")

        adapter.disconnect()

    def test_buffered_audio_cleared_on_delayed_interruption(self):
        """Test that buffered audio is cleared when SpeechStartedEvent finally arrives.

        Even if the interruption detection is delayed, once it arrives:
        - Buffered audio (for future ticks) should be discarded
        - Current and future ticks should not play the interrupted utterance
        """
        tick_duration_ms = 200

        # Agent sends 5 seconds of audio (25 ticks worth)
        agent_audio = make_audio_bytes(5000)
        user_audio = make_silence_bytes(tick_duration_ms)

        mock_provider = MockOpenAIRealtimeProvider(
            tick_duration_ms=tick_duration_ms,
            event_script=[
                # Tick 1: All audio arrives
                [
                    AudioDeltaEvent(
                        type="response.audio.delta",
                        delta=base64.b64encode(agent_audio).decode("utf-8"),
                        item_id="item_1",
                    ),
                ],
                # Ticks 2-5: Empty
                [],
                [],
                [],
                [],
                # Tick 6: SpeechStartedEvent with audio_start_ms=400ms (tick 2)
                [
                    SpeechStartedEvent(
                        type="input_audio_buffer.speech_started",
                        audio_start_ms=400,
                    ),
                ],
                # Ticks 7-10: Should be empty (buffered audio was cleared)
                [],
                [],
                [],
                [],
            ],
        )

        adapter = DiscreteTimeAudioNativeAdapter(
            tick_duration_ms=tick_duration_ms,
            send_audio_instant=True,
            buffer_until_complete=False,
            provider=mock_provider,
        )

        vad_config = OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD)
        adapter.connect(
            system_prompt="Test",
            tools=[],
            vad_config=vad_config,
            modality="audio",
        )

        results = []
        for tick_num in range(1, 11):
            result = adapter.run_tick(user_audio, tick_number=tick_num)
            results.append(result)

        # Ticks 1-5: Not truncated, have audio
        for i in range(5):
            assert not results[i].was_truncated
            played = results[i].get_played_agent_audio()
            non_silence = sum(1 for b in played if b == 0x50)
            assert non_silence > 0, f"Tick {i + 1} should have audio"

        # Tick 6: Truncated
        assert results[5].was_truncated

        # Ticks 7-10: Should have NO audio (buffered audio was cleared)
        for i in range(6, 10):
            result = results[i]
            # These ticks should have no raw audio (buffer was cleared)
            assert result.agent_audio_bytes == 0, (
                f"Tick {i + 1} should have no raw audio (buffer was cleared), "
                f"but got {result.agent_audio_bytes} bytes"
            )

        adapter.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
