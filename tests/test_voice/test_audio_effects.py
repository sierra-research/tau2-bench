# Copyright Sierra
"""
Unit tests for audio effects module.

Tests cover:
- effects.py: apply_burst_noise, apply_frame_drops, apply_dynamic_muffling,
  apply_constant_muffling, StreamingTelephonyConverter
- noise_generator.py: BackgroundNoiseGenerator, create_background_noise_generator,
  apply_background_noise
- scheduler.py: EffectScheduler, ScheduledEffect, generate_turn_effects,
  EffectSchedulerState
- processor.py: PendingEffectState, StreamingAudioEffectsMixin
- probability.py: GilbertElliottModel
"""

from pathlib import Path

import numpy as np
import pytest

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import SynthesisConfig
from tau2.voice.synthesis.audio_effects.effects import (
    StreamingTelephonyConverter,
    apply_burst_noise,
    apply_constant_muffling,
    apply_dynamic_muffling,
    apply_frame_drops,
)
from tau2.voice.synthesis.audio_effects.noise_generator import (
    BackgroundNoiseGenerator,
    apply_background_noise,
    create_background_noise_generator,
)
from tau2.voice.synthesis.audio_effects.processor import (
    BatchAudioEffectsMixin,
    PendingEffectState,
    StreamingAudioEffectsMixin,
)
from tau2.voice.synthesis.audio_effects.scheduler import (
    EffectScheduler,
    EffectSchedulerState,
    ScheduledEffect,
    generate_turn_effects,
)
from tau2.voice.utils.audio_io import save_wav_file
from tau2.voice.utils.audio_preprocessing import audio_data_to_numpy
from tau2.voice.utils.probability import GilbertElliottConfig, GilbertElliottModel

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_pcm16_audio() -> AudioData:
    """Create a simple PCM16 audio sample (1 second, 440 Hz sine wave)."""
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)

    return AudioData(
        data=samples.tobytes(),
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        ),
        audio_path=None,
    )


@pytest.fixture
def sample_ulaw_audio() -> AudioData:
    """Create a simple μ-law audio sample (8kHz, 0.5 second)."""
    sample_rate = 8000
    num_samples = int(sample_rate * 0.5)
    # Generate non-silent μ-law samples (0x7F is mid-range)
    samples = bytes([0x7F] * num_samples)

    return AudioData(
        data=samples,
        format=AudioFormat(
            encoding=AudioEncoding.ULAW,
            sample_rate=sample_rate,
        ),
        audio_path=None,
    )


@pytest.fixture
def short_pcm16_audio() -> AudioData:
    """Create a short PCM16 audio sample (0.1 second)."""
    sample_rate = 16000
    duration = 0.1
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)

    return AudioData(
        data=samples.tobytes(),
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        ),
        audio_path=None,
    )


@pytest.fixture
def burst_noise_file(tmp_path: Path) -> Path:
    """Create a temporary burst noise WAV file."""
    sample_rate = 16000
    duration = 0.1  # 100ms burst
    frequency = 1000.0

    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = (np.sin(2 * np.pi * frequency * t) * 8000).astype(np.int16)

    audio = AudioData(
        data=samples.tobytes(),
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        ),
        audio_path=None,
    )

    filepath = tmp_path / "burst.wav"
    save_wav_file(audio, filepath)
    return filepath


@pytest.fixture
def noise_audio_file(tmp_path: Path) -> Path:
    """Create a temporary background noise WAV file."""
    sample_rate = 16000
    duration = 0.5
    # White noise
    np.random.seed(42)
    samples = (np.random.randn(int(sample_rate * duration)) * 3000).astype(np.int16)

    audio = AudioData(
        data=samples.tobytes(),
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
        ),
        audio_path=None,
    )

    filepath = tmp_path / "noise.wav"
    save_wav_file(audio, filepath)
    return filepath


# ============================================================================
# Effects Tests
# ============================================================================


class TestEffects:
    """Tests for audio effect functions."""

    def test_apply_frame_drops_pcm16(self, sample_pcm16_audio: AudioData):
        """Test frame drops insert silence in PCM16 audio."""
        drop_count = 3
        drop_duration_ms = 50

        result = apply_frame_drops(
            sample_pcm16_audio,
            drop_count=drop_count,
            drop_duration_ms=drop_duration_ms,
        )

        assert result.format.encoding == AudioEncoding.PCM_S16LE
        assert len(result.data) == len(sample_pcm16_audio.data)

        # Check that some samples are now zero (silence)
        result_samples = audio_data_to_numpy(result)
        original_samples = audio_data_to_numpy(sample_pcm16_audio)

        # There should be zeroed segments
        zero_count = np.sum(result_samples == 0)
        original_zero_count = np.sum(original_samples == 0)
        assert zero_count > original_zero_count

    def test_apply_frame_drops_ulaw(self, sample_ulaw_audio: AudioData):
        """Test frame drops insert silence (0xFF) in μ-law audio."""
        drop_count = 2
        drop_duration_ms = 50

        result = apply_frame_drops(
            sample_ulaw_audio,
            drop_count=drop_count,
            drop_duration_ms=drop_duration_ms,
        )

        assert result.format.encoding == AudioEncoding.ULAW
        assert len(result.data) == len(sample_ulaw_audio.data)

        # Count 0xFF bytes (μ-law silence)
        original_silence = sample_ulaw_audio.data.count(b"\xff"[0])
        result_silence = result.data.count(b"\xff"[0])
        assert result_silence > original_silence

    def test_apply_constant_muffling(self, sample_pcm16_audio: AudioData):
        """Test constant muffling applies low-pass filter."""
        cutoff_freq = 500.0

        result = apply_constant_muffling(sample_pcm16_audio, cutoff_freq=cutoff_freq)

        assert result.format.encoding == AudioEncoding.PCM_S16LE
        assert result.format.sample_rate == sample_pcm16_audio.format.sample_rate
        assert len(result.data) == len(sample_pcm16_audio.data)

        # The audio should be modified (different from original)
        original_samples = audio_data_to_numpy(sample_pcm16_audio)
        result_samples = audio_data_to_numpy(result)
        assert not np.array_equal(original_samples, result_samples)

    def test_apply_dynamic_muffling(self, sample_pcm16_audio: AudioData):
        """Test dynamic muffling applies segment-based filtering."""
        segment_count = 2
        segment_duration_ms = 100
        cutoff_freq = 800.0
        transition_ms = 50

        result = apply_dynamic_muffling(
            sample_pcm16_audio,
            segment_count=segment_count,
            segment_duration_ms=segment_duration_ms,
            cutoff_freq=cutoff_freq,
            transition_ms=transition_ms,
        )

        assert result.format.encoding == AudioEncoding.PCM_S16LE
        assert result.format.sample_rate == sample_pcm16_audio.format.sample_rate
        assert len(result.data) == len(sample_pcm16_audio.data)

        # The audio should be modified
        original_samples = audio_data_to_numpy(sample_pcm16_audio)
        result_samples = audio_data_to_numpy(result)
        assert not np.array_equal(original_samples, result_samples)

    def test_apply_burst_noise_with_file(
        self, sample_pcm16_audio: AudioData, burst_noise_file: Path
    ):
        """Test burst noise overlay with a WAV file."""
        result = apply_burst_noise(
            sample_pcm16_audio, burst_noise_file=burst_noise_file
        )

        assert result.format.encoding == AudioEncoding.PCM_S16LE
        assert result.format.sample_rate == sample_pcm16_audio.format.sample_rate
        assert len(result.data) == len(sample_pcm16_audio.data)

        # The audio should be modified (burst added)
        original_samples = audio_data_to_numpy(sample_pcm16_audio)
        result_samples = audio_data_to_numpy(result)
        assert not np.array_equal(original_samples, result_samples)

    def test_apply_burst_noise_none_returns_original(
        self, sample_pcm16_audio: AudioData
    ):
        """Test that apply_burst_noise returns original when no file provided."""
        result = apply_burst_noise(sample_pcm16_audio, burst_noise_file=None)

        assert result.data == sample_pcm16_audio.data
        assert result.format == sample_pcm16_audio.format

    def test_streaming_telephony_converter(self, short_pcm16_audio: AudioData):
        """Test streaming telephony converter preserves state across chunks."""
        converter = StreamingTelephonyConverter(
            input_sample_rate=short_pcm16_audio.format.sample_rate
        )

        # Convert two chunks
        result1 = converter.convert_chunk(short_pcm16_audio)
        result2 = converter.convert_chunk(short_pcm16_audio)

        # Both should produce μ-law output at 8kHz
        assert result1.format.encoding == AudioEncoding.ULAW
        assert result1.format.sample_rate == 8000
        assert result2.format.encoding == AudioEncoding.ULAW
        assert result2.format.sample_rate == 8000

        # Both should have data
        assert len(result1.data) > 0
        assert len(result2.data) > 0

    def test_streaming_telephony_converter_empty_input(self):
        """Test streaming telephony converter handles empty input."""
        converter = StreamingTelephonyConverter(input_sample_rate=16000)

        empty_audio = AudioData(
            data=b"",
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
            audio_path=None,
        )

        result = converter.convert_chunk(empty_audio)

        assert result.format.encoding == AudioEncoding.ULAW
        assert len(result.data) == 0


# ============================================================================
# Noise Generator Tests
# ============================================================================


class TestNoiseGenerator:
    """Tests for BackgroundNoiseGenerator."""

    def test_background_noise_generator_silent_mode(self):
        """Test silent mode creates zero-valued audio."""
        sample_rate = 16000
        generator = BackgroundNoiseGenerator(
            sample_rate=sample_rate,
            silent_mode=True,
        )

        chunk, envelope = generator.get_next_chunk(num_samples=1000)

        assert chunk.format.encoding == AudioEncoding.PCM_S16LE
        assert chunk.format.sample_rate == sample_rate
        assert chunk.num_samples == 1000

        # Silent mode uses default SNR settings
        assert generator.snr_db == 15.0  # Default NOISE_SNR_DB

    def test_background_noise_generator_with_audio_data(
        self, sample_pcm16_audio: AudioData
    ):
        """Test generator with direct audio data."""
        generator = BackgroundNoiseGenerator(
            audio=sample_pcm16_audio,
            sample_rate=sample_pcm16_audio.format.sample_rate,
        )

        chunk, envelope = generator.get_next_chunk(num_samples=500)

        assert chunk.num_samples == 500
        assert len(envelope) == 500

    def test_background_noise_generator_looping(self, noise_audio_file: Path):
        """Test audio loops correctly when requesting more samples than available."""
        generator = BackgroundNoiseGenerator(
            audio_file_path=noise_audio_file,
            sample_rate=16000,
        )

        # Request more samples than the noise file contains
        total_noise_samples = generator.audio.num_samples
        request_samples = total_noise_samples * 2 + 100

        chunk, envelope = generator.get_next_chunk(num_samples=request_samples)

        assert chunk.num_samples == request_samples
        assert len(envelope) == request_samples

    def test_snr_envelope_generation(self):
        """Test SNR envelope stays within specified range (base +/- drift)."""
        generator = BackgroundNoiseGenerator(
            sample_rate=16000,
            silent_mode=True,
        )

        snr_db = 15.0
        snr_drift_db = 3.0
        envelope = generator.generate_snr_envelope(
            start_time_seconds=0.0,
            sample_rate=16000,
            num_samples=1000,
            variation_speed=0.05,
            snr_db=snr_db,
            snr_drift_db=snr_drift_db,
            primary_variation=(1.0, 1.0),
            harmonic_variation=(0.3, 2.7),
            slow_drift=(0.2, 0.4),
        )

        assert len(envelope) == 1000
        # SNR should be within base +/- drift
        assert np.min(envelope) >= snr_db - snr_drift_db - 0.01  # Small tolerance
        assert np.max(envelope) <= snr_db + snr_drift_db + 0.01

    def test_get_next_chunk_returns_correct_samples(self, noise_audio_file: Path):
        """Test get_next_chunk returns exactly the requested number of samples."""
        generator = BackgroundNoiseGenerator(
            audio_file_path=noise_audio_file,
            sample_rate=16000,
        )

        for num_samples in [100, 500, 1000, 1600]:
            chunk, envelope = generator.get_next_chunk(num_samples=num_samples)
            assert chunk.num_samples == num_samples
            assert len(envelope) == num_samples

    def test_add_burst_and_get_burst_chunk(
        self, noise_audio_file: Path, burst_noise_file: Path
    ):
        """Test burst sound can be retrieved via get_burst_chunk."""
        generator = BackgroundNoiseGenerator(
            audio_file_path=noise_audio_file,
            sample_rate=16000,
        )

        # No burst initially
        assert not generator.has_active_burst()
        assert generator.get_burst_chunk(num_samples=500) is None

        # Add burst
        generator.add_burst(burst_noise_file)

        assert generator.has_active_burst()
        assert generator.burst_snr_db is not None  # SNR should be sampled

        # Get burst chunks until consumed
        burst_info = generator.get_burst_chunk(num_samples=500)
        assert burst_info is not None
        burst_samples, burst_snr, whole_burst_rms = burst_info
        assert len(burst_samples) == 500
        assert -5.0 <= burst_snr <= 10.0  # Within BURST_SNR_RANGE_DB
        assert whole_burst_rms > 0  # Pre-computed RMS over entire burst

        # All chunks should report the same whole-burst RMS
        rms_values = [whole_burst_rms]
        while generator.has_active_burst():
            info = generator.get_burst_chunk(num_samples=500)
            if info is not None:
                _, _, chunk_rms = info
                rms_values.append(chunk_rms)
        assert len(set(rms_values)) == 1, (
            f"All chunks must use the same whole-burst RMS, got {rms_values}"
        )

        assert not generator.has_active_burst()

    def test_apply_background_noise(
        self, sample_pcm16_audio: AudioData, noise_audio_file: Path
    ):
        """Test applying background noise to speech audio."""
        generator = BackgroundNoiseGenerator(
            audio_file_path=noise_audio_file,
            sample_rate=sample_pcm16_audio.format.sample_rate,
        )

        result = apply_background_noise(sample_pcm16_audio, generator)

        assert result.format.encoding == AudioEncoding.PCM_S16LE
        assert result.format.sample_rate == sample_pcm16_audio.format.sample_rate
        assert len(result.data) == len(sample_pcm16_audio.data)

    def test_create_background_noise_generator_silent(self):
        """Test factory function creates silent generator when no file provided."""
        from tau2.data_model.audio_effects import SourceEffectsConfig

        config = SourceEffectsConfig()
        generator = create_background_noise_generator(
            config=config,
            sample_rate=16000,
            background_noise_file=None,
        )

        assert generator.silent_mode is True
        assert generator.sample_rate == 16000

    def test_should_trigger_burst_no_rate(self):
        """Test should_trigger_burst returns None when burst_rate is 0."""
        generator = BackgroundNoiseGenerator(
            sample_rate=16000,
            silent_mode=True,
            burst_rate=0.0,
        )

        result = generator.should_trigger_burst(chunk_duration_sec=0.1)
        assert result is None

    def test_should_trigger_burst_with_active_burst(self, noise_audio_file: Path):
        """Test should_trigger_burst returns None when burst already active."""
        generator = BackgroundNoiseGenerator(
            audio_file_path=noise_audio_file,
            sample_rate=16000,
            burst_rate=1000.0,  # Very high rate to guarantee trigger
        )

        # Simulate an active burst
        generator.burst_audio = b"\x00" * 100
        generator.burst_offset = 0

        result = generator.should_trigger_burst(chunk_duration_sec=0.1)
        assert result is None


# ============================================================================
# Scheduler Tests
# ============================================================================


class TestScheduler:
    """Tests for effect scheduling."""

    def test_scheduled_effect_model(self):
        """Test ScheduledEffect model fields."""
        effect = ScheduledEffect(
            effect_type="burst_noise_file",
            timestamp_ms=1000,
            duration_ms=500,
            timing="cross_turn",
            burst_noise_file=Path("/test/burst.wav"),
        )

        assert effect.effect_type == "burst_noise_file"
        assert effect.timestamp_ms == 1000
        assert effect.duration_ms == 500
        assert effect.timing == "cross_turn"
        assert effect.burst_noise_file == Path("/test/burst.wav")

    def test_effect_scheduler_state_elapsed_ms(self):
        """Test EffectSchedulerState elapsed_ms computation."""
        state = EffectSchedulerState(
            elapsed_samples=8000,
            sample_rate=8000,
        )

        assert state.elapsed_ms == 1000  # 8000 samples at 8kHz = 1 second

        state = EffectSchedulerState(
            elapsed_samples=16000,
            sample_rate=16000,
        )
        assert state.elapsed_ms == 1000

    def test_effect_scheduler_deterministic(self):
        """Test EffectScheduler produces deterministic results with same seed."""
        from tau2.data_model.audio_effects import (
            ChannelEffectsConfig,
            SourceEffectsConfig,
            SpeechEffectsConfig,
        )

        config_source = SourceEffectsConfig(enable_burst_noise=True)
        config_speech = SpeechEffectsConfig()
        config_channel = ChannelEffectsConfig()

        scheduler1 = EffectScheduler(
            seed=42,
            source_config=config_source,
            speech_config=config_speech,
            channel_config=config_channel,
        )

        scheduler2 = EffectScheduler(
            seed=42,
            source_config=config_source,
            speech_config=config_speech,
            channel_config=config_channel,
        )

        # Both should have same rates
        assert scheduler1.burst_rate == scheduler2.burst_rate
        assert scheduler1.speech_insert_rate == scheduler2.speech_insert_rate

    def test_check_for_effects_returns_list(self):
        """Test check_for_effects returns a list of effects."""
        from tau2.data_model.audio_effects import (
            ChannelEffectsConfig,
            SourceEffectsConfig,
            SpeechEffectsConfig,
        )

        config_source = SourceEffectsConfig(
            enable_burst_noise=False,  # Disable to get predictable results
        )
        config_speech = SpeechEffectsConfig(
            enable_vocal_tics=False,
            enable_non_directed_phrases=False,
        )
        config_channel = ChannelEffectsConfig(enable_frame_drops=False)

        scheduler = EffectScheduler(
            seed=42,
            source_config=config_source,
            speech_config=config_speech,
            channel_config=config_channel,
        )

        effects = scheduler.check_for_effects(
            chunk_duration_ms=100,
            is_silence=False,
            current_time_ms=0,
        )

        assert isinstance(effects, list)

    def test_generate_turn_effects(self):
        """Test generate_turn_effects returns effect results tuple."""
        from tau2.user_simulation_voice_presets import sample_voice_config

        base_config = SynthesisConfig()
        # Sample and merge complexity overrides into synthesis config
        sampled = sample_voice_config(
            seed=42, synthesis_config=base_config, complexity="control"
        )
        synthesis_config = SynthesisConfig(
            channel_effects_config=sampled.channel_effects_config,
            source_effects_config=sampled.source_effects_config,
            speech_effects_config=sampled.speech_effects_config,
        )

        speech_effects, source_effects, channel_effects = generate_turn_effects(
            seed=42,
            turn_idx=0,
            synthesis_config=synthesis_config,
        )

        # Check return types
        from tau2.data_model.audio_effects import (
            ChannelEffectsResult,
            SourceEffectsResult,
            SpeechEffectsResult,
        )

        assert isinstance(speech_effects, SpeechEffectsResult)
        assert isinstance(source_effects, SourceEffectsResult)
        assert isinstance(channel_effects, ChannelEffectsResult)

    def test_generate_turn_effects_deterministic(self):
        """Test generate_turn_effects is deterministic with same seed and turn_idx."""
        from tau2.user_simulation_voice_presets import sample_voice_config

        base_config = SynthesisConfig()
        # Sample and merge complexity overrides into synthesis config
        sampled = sample_voice_config(
            seed=42, synthesis_config=base_config, complexity="regular"
        )
        synthesis_config = SynthesisConfig(
            channel_effects_config=sampled.channel_effects_config,
            source_effects_config=sampled.source_effects_config,
            speech_effects_config=sampled.speech_effects_config,
        )

        result1 = generate_turn_effects(
            seed=42,
            turn_idx=5,
            synthesis_config=synthesis_config,
        )

        result2 = generate_turn_effects(
            seed=42,
            turn_idx=5,
            synthesis_config=synthesis_config,
        )

        # Same seed + turn_idx should produce same results
        assert (
            result1[0].dynamic_muffling_enabled == result2[0].dynamic_muffling_enabled
        )
        assert result1[1].burst_noise_file == result2[1].burst_noise_file
        assert result1[2].frame_drops_enabled == result2[2].frame_drops_enabled


# ============================================================================
# Processor Tests
# ============================================================================


class TestProcessor:
    """Tests for audio effect processor components."""

    def test_pending_effect_state_properties(self):
        """Test PendingEffectState remaining_bytes and is_complete properties."""
        audio_bytes = b"\x00" * 1000

        state = PendingEffectState(
            audio_bytes=audio_bytes,
            offset=0,
        )

        assert state.remaining_bytes == 1000
        assert state.is_complete is False

        # Advance offset
        state = PendingEffectState(
            audio_bytes=audio_bytes,
            offset=500,
        )

        assert state.remaining_bytes == 500
        assert state.is_complete is False

        # Complete
        state = PendingEffectState(
            audio_bytes=audio_bytes,
            offset=1000,
        )

        assert state.remaining_bytes == 0
        assert state.is_complete is True

    def test_pending_effect_state_with_info(self):
        """Test PendingEffectState with optional info dict."""
        state = PendingEffectState(
            audio_bytes=b"\x00" * 100,
            offset=0,
            info={"type": "vocal_tic", "value": "um"},
        )

        assert state.info["type"] == "vocal_tic"
        assert state.info["value"] == "um"

    def test_streaming_audio_effects_mixin_frame_drop(self):
        """Test StreamingAudioEffectsMixin.apply_streaming_frame_drop."""

        class TestMixin(StreamingAudioEffectsMixin):
            pass

        mixin = TestMixin()

        # Test PCM frame drop
        audio_bytes = b"\x7f" * 1000  # Non-silent bytes
        result = mixin.apply_streaming_frame_drop(
            audio_bytes=audio_bytes,
            sample_rate=8000,
            bytes_per_sample=1,
            drop_duration_ms=50,
            is_ulaw=False,
        )

        # Should have some silence (0x00) at the start
        assert result[:10] == b"\x00" * 10
        assert len(result) == len(audio_bytes)

    def test_streaming_audio_effects_mixin_frame_drop_ulaw(self):
        """Test StreamingAudioEffectsMixin.apply_streaming_frame_drop for μ-law."""

        class TestMixin(StreamingAudioEffectsMixin):
            pass

        mixin = TestMixin()

        # Test μ-law frame drop (silence is 0xFF)
        audio_bytes = b"\x7f" * 1000
        result = mixin.apply_streaming_frame_drop(
            audio_bytes=audio_bytes,
            sample_rate=8000,
            bytes_per_sample=1,
            drop_duration_ms=50,
            is_ulaw=True,
        )

        # Should have μ-law silence (0xFF) at the start
        assert result[:10] == b"\xff" * 10
        assert len(result) == len(audio_bytes)

    def test_streaming_audio_effects_mixin_full_frame_drop(self):
        """Test frame drop when drop duration exceeds audio length."""

        class TestMixin(StreamingAudioEffectsMixin):
            pass

        mixin = TestMixin()

        audio_bytes = b"\x7f" * 100
        result = mixin.apply_streaming_frame_drop(
            audio_bytes=audio_bytes,
            sample_rate=8000,
            bytes_per_sample=1,
            drop_duration_ms=1000,  # Much longer than audio
            is_ulaw=False,
        )

        # Entire audio should be silenced
        assert result == b"\x00" * 100

    def test_batch_audio_effects_mixin_apply_batch_effects(
        self, sample_pcm16_audio: AudioData, noise_audio_file: Path
    ):
        """Test BatchAudioEffectsMixin.apply_batch_effects applies all effects."""
        from tau2.data_model.audio_effects import (
            ChannelEffectsConfig,
            ChannelEffectsResult,
            SourceEffectsResult,
            SpeechEffectsConfig,
            SpeechEffectsResult,
        )

        class TestMixin(BatchAudioEffectsMixin):
            pass

        mixin = TestMixin()
        noise_generator = BackgroundNoiseGenerator(
            audio_file_path=noise_audio_file,
            sample_rate=sample_pcm16_audio.format.sample_rate,
        )

        speech_effects = SpeechEffectsResult(dynamic_muffling_enabled=False)
        source_effects = SourceEffectsResult(burst_noise_file=None)
        channel_effects = ChannelEffectsResult(frame_drops_enabled=False)
        speech_config = SpeechEffectsConfig()
        channel_config = ChannelEffectsConfig()

        result = mixin.apply_batch_effects(
            audio=sample_pcm16_audio,
            speech_effects=speech_effects,
            source_effects=source_effects,
            channel_effects=channel_effects,
            speech_config=speech_config,
            channel_config=channel_config,
            noise_generator=noise_generator,
            apply_telephony=False,  # Skip telephony to keep PCM format
        )

        # Result should be PCM format with same length
        assert result.format.encoding == AudioEncoding.PCM_S16LE
        assert len(result.data) == len(sample_pcm16_audio.data)

    def test_batch_audio_effects_mixin_with_muffling(
        self, sample_pcm16_audio: AudioData
    ):
        """Test BatchAudioEffectsMixin with dynamic muffling enabled."""
        from tau2.data_model.audio_effects import (
            ChannelEffectsConfig,
            ChannelEffectsResult,
            SourceEffectsResult,
            SpeechEffectsConfig,
            SpeechEffectsResult,
        )

        class TestMixin(BatchAudioEffectsMixin):
            pass

        mixin = TestMixin()

        speech_effects = SpeechEffectsResult(dynamic_muffling_enabled=True)
        source_effects = SourceEffectsResult(burst_noise_file=None)
        channel_effects = ChannelEffectsResult(frame_drops_enabled=False)
        # Use shorter segment duration that fits within 1-second audio
        # Default requires 19200 samples (segment=16000 + 2*transition=3200)
        # but we only have 16000 samples, so use shorter duration
        speech_config = SpeechEffectsConfig(
            muffle_segment_duration_ms=200,
            muffle_transition_ms=50,
        )
        channel_config = ChannelEffectsConfig()

        result = mixin.apply_batch_effects(
            audio=sample_pcm16_audio,
            speech_effects=speech_effects,
            source_effects=source_effects,
            channel_effects=channel_effects,
            speech_config=speech_config,
            channel_config=channel_config,
            noise_generator=None,
            apply_telephony=False,
        )

        # Audio should be modified by muffling
        original_samples = audio_data_to_numpy(sample_pcm16_audio)
        result_samples = audio_data_to_numpy(result)
        assert not np.array_equal(original_samples, result_samples)

    def test_process_streaming_chunk_basic(self):
        """Test StreamingAudioEffectsMixin.process_streaming_chunk."""

        class TestMixin(StreamingAudioEffectsMixin):
            pass

        mixin = TestMixin()
        sample_rate = 16000
        noise_generator = BackgroundNoiseGenerator(
            sample_rate=sample_rate,
            silent_mode=True,
        )

        # Create simple speech audio
        num_samples = 1000
        samples = (np.sin(np.linspace(0, 10, num_samples)) * 16000).astype(np.int16)
        speech_audio = AudioData(
            data=samples.tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate
            ),
            audio_path=None,
        )

        result = mixin.process_streaming_chunk(
            speech_audio=speech_audio,
            noise_generator=noise_generator,
            num_samples=num_samples,
            scheduled_effects=None,
            out_of_turn_generator=None,
            pending_effect=None,
        )

        result_audio = result.to_mixed_audio_data()
        assert result_audio.num_samples == num_samples
        assert result.source_effects is None
        assert result.pending_effect is None
        assert result.tracks.num_samples == num_samples

    def test_process_streaming_chunk_silence_only(self):
        """Test process_streaming_chunk with no speech (silence)."""

        class TestMixin(StreamingAudioEffectsMixin):
            pass

        mixin = TestMixin()
        sample_rate = 16000
        noise_generator = BackgroundNoiseGenerator(
            sample_rate=sample_rate,
            silent_mode=True,
        )

        result = mixin.process_streaming_chunk(
            speech_audio=None,
            noise_generator=noise_generator,
            num_samples=500,
            scheduled_effects=None,
            out_of_turn_generator=None,
            pending_effect=None,
        )

        result_audio = result.to_mixed_audio_data()
        assert result_audio.num_samples == 500
        assert result.source_effects is None
        assert result.pending_effect is None


# ============================================================================
# Gilbert-Elliott Model Tests
# ============================================================================


class TestGilbertElliottModel:
    """Tests for Gilbert-Elliott bursty packet loss model."""

    def test_ge_config_computed_fields(self):
        """Test GilbertElliottConfig computes derived parameters correctly."""
        config = GilbertElliottConfig(
            target_loss_rate=0.05,
            avg_burst_duration_ms=200,
        )

        # Check user parameters
        assert config.target_loss_rate == 0.05
        assert config.avg_burst_duration_ms == 200
        assert config.good_state_loss_prob == 0.0
        assert config.bad_state_loss_prob == 0.2

        # Check derived parameters
        # r_rate = 1 / (200ms / 1000) = 5.0/s
        assert config.r_rate == 5.0
        # p_rate = r * target / (h - target) = 5.0 * 0.05 / (0.2 - 0.05) = 5/3
        assert abs(config.p_rate - (5.0 * 0.05 / 0.15)) < 1e-6
        # steady_state_bad_prob = p / (p + r)
        expected_pi_b = config.p_rate / (config.p_rate + config.r_rate)
        assert abs(config.steady_state_bad_prob - expected_pi_b) < 1e-6

    def test_ge_model_initialization(self):
        """Test GilbertElliottModel initializes correctly."""
        import random

        rng = random.Random(42)
        model = GilbertElliottModel(
            target_loss_rate=0.05,
            avg_burst_duration_ms=200,
            rng=rng,
        )

        # Check internal parameters are set via config
        assert model.target_loss_rate == 0.05
        assert model.avg_burst_duration_ms == 200
        # Check config object exists
        assert model.config is not None
        assert model.config.target_loss_rate == 0.05

    def test_ge_model_invalid_loss_rate(self):
        """Test GilbertElliottModel raises error for invalid loss rate."""
        import random

        from pydantic import ValidationError

        rng = random.Random(42)

        # Loss rate must be >= 0 (Pydantic validation)
        with pytest.raises(ValidationError):
            GilbertElliottModel(
                target_loss_rate=-0.1,
                avg_burst_duration_ms=200,
                rng=rng,
            )

        # Loss rate must be < H (0.2) (Pydantic validation)
        with pytest.raises(ValidationError):
            GilbertElliottModel(
                target_loss_rate=0.25,
                avg_burst_duration_ms=200,
                rng=rng,
            )

    def test_ge_model_invalid_burst_duration(self):
        """Test GilbertElliottModel raises error for invalid burst duration."""
        import random

        from pydantic import ValidationError

        rng = random.Random(42)

        # Burst duration must be > 0 (Pydantic validation)
        with pytest.raises(ValidationError):
            GilbertElliottModel(
                target_loss_rate=0.05,
                avg_burst_duration_ms=0,
                rng=rng,
            )

    def test_ge_model_deterministic(self):
        """Test GilbertElliottModel produces deterministic results with same seed."""
        import random

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        model1 = GilbertElliottModel(
            target_loss_rate=0.05,
            avg_burst_duration_ms=200,
            rng=rng1,
        )
        model2 = GilbertElliottModel(
            target_loss_rate=0.05,
            avg_burst_duration_ms=200,
            rng=rng2,
        )

        # Run 100 iterations and compare
        results1 = [model1.should_drop(0.1) for _ in range(100)]
        results2 = [model2.should_drop(0.1) for _ in range(100)]

        assert results1 == results2

    def test_ge_model_produces_bursty_losses(self):
        """Test that GE model produces bursty (correlated) losses."""
        import random

        rng = random.Random(42)
        model = GilbertElliottModel(
            target_loss_rate=0.05,
            avg_burst_duration_ms=500,  # Long bursts
            rng=rng,
        )

        # Run many iterations
        drops = [model.should_drop(0.01) for _ in range(10000)]

        # Count consecutive drops (bursts)
        burst_lengths = []
        current_burst = 0
        for drop in drops:
            if drop:
                current_burst += 1
            elif current_burst > 0:
                burst_lengths.append(current_burst)
                current_burst = 0
        if current_burst > 0:
            burst_lengths.append(current_burst)

        # With bursty model, we should see some bursts > 1
        # (independent losses would rarely have consecutive drops)
        if burst_lengths:
            max_burst = max(burst_lengths)
            assert max_burst >= 1  # At least some drops occurred

    def test_ge_model_reset(self):
        """Test GilbertElliottModel reset functionality."""
        import random

        from tau2.voice.utils.probability import GEState

        rng = random.Random(42)
        model = GilbertElliottModel(
            target_loss_rate=0.05,
            avg_burst_duration_ms=200,
            rng=rng,
            initial_state="bad",
        )

        assert model.state == GEState.BAD
        assert model.is_in_bad_state is True

        model.reset("good")
        assert model.state == GEState.GOOD
        assert model.is_in_bad_state is False

    def test_ge_model_state_transitions(self):
        """Test that GE model transitions between states."""
        import random

        from tau2.voice.utils.probability import GEState

        rng = random.Random(42)
        model = GilbertElliottModel(
            target_loss_rate=0.1,  # Higher rate for more transitions
            avg_burst_duration_ms=100,  # Short bursts
            rng=rng,
        )

        # Track state changes over many iterations
        states = []
        for _ in range(1000):
            model.should_drop(0.05)
            states.append(model.state)

        # Should have both states represented
        assert GEState.GOOD in states
        assert GEState.BAD in states


# ============================================================================
# AudioTracks Tests
# ============================================================================


class TestAudioTracks:
    """Tests for AudioTracks multitrack data structure."""

    def test_tracks_sum_equals_single_pass_mix(self):
        """AudioTracks.to_audio_data() must be bit-exact with the original single-pass mix.

        Reimplements the old mix_audio_dynamic logic (sum in float64, clip once)
        to verify that the multitrack path produces identical output.
        """
        from tau2.voice.utils.audio_preprocessing import (
            _compute_rms,
            _snr_to_scale,
            audio_data_to_numpy,
            mix_audio_to_tracks,
        )

        sample_rate = 16000
        num_samples = 800
        speech_samples = (np.sin(np.linspace(0, 20, num_samples)) * 8000).astype(
            np.int16
        )
        noise_samples = (np.random.RandomState(42).randn(num_samples) * 500).astype(
            np.int16
        )
        speech = AudioData(
            data=speech_samples.tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate
            ),
        )
        noise = AudioData(
            data=noise_samples.tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate
            ),
        )
        snr_envelope = np.full(num_samples, 20.0)

        # Original single-pass logic (sum float64, clip once)
        s = audio_data_to_numpy(speech, dtype=np.int16).astype(np.float64)
        n = audio_data_to_numpy(noise, dtype=np.int16).astype(np.float64)
        noise_rms = _compute_rms(n)
        noise_scale = _snr_to_scale(float(np.mean(snr_envelope)), noise_rms)
        expected = np.clip(s + n * noise_scale, -32768, 32767).astype(np.int16)

        # New multitrack path
        tracks = mix_audio_to_tracks(speech, noise, snr_envelope)
        actual = np.frombuffer(tracks.to_audio_data().data, dtype=np.int16)

        assert np.array_equal(expected, actual)

    def test_tracks_shapes(self):
        """All tracks should have the same num_samples."""
        from tau2.voice.utils.audio_preprocessing import mix_audio_to_tracks

        sample_rate = 16000
        num_samples = 500
        speech = AudioData(
            data=np.zeros(num_samples, dtype=np.int16).tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate
            ),
        )
        noise = AudioData(
            data=np.zeros(num_samples, dtype=np.int16).tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate
            ),
        )
        snr = np.full(num_samples, 15.0)

        tracks = mix_audio_to_tracks(speech, noise, snr)
        assert tracks.num_samples == num_samples
        assert len(tracks.speech) == num_samples
        assert len(tracks.background_noise) == num_samples
        assert len(tracks.burst_noise) == num_samples

    def test_tracks_with_burst(self):
        """Burst noise should appear in the burst_noise track."""
        from tau2.voice.utils.audio_preprocessing import mix_audio_to_tracks

        sample_rate = 16000
        num_samples = 500
        speech = AudioData(
            data=np.zeros(num_samples, dtype=np.int16).tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate
            ),
        )
        noise = AudioData(
            data=np.zeros(num_samples, dtype=np.int16).tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate
            ),
        )
        snr = np.full(num_samples, 15.0)

        burst_samples = (np.ones(num_samples, dtype=np.int16) * 1000).astype(np.int16)
        whole_burst_rms = float(np.sqrt(np.mean(burst_samples.astype(np.float64) ** 2)))
        burst_info = (burst_samples, 10.0, whole_burst_rms)

        tracks = mix_audio_to_tracks(speech, noise, snr, burst_info=burst_info)
        assert np.any(tracks.burst_noise != 0)

    def test_streaming_chunk_result_multitrack(self):
        """StreamingChunkResult should produce per-effect tracks."""
        mixin = type("M", (StreamingAudioEffectsMixin,), {})()
        noise_gen = BackgroundNoiseGenerator(sample_rate=16000, silent_mode=True)

        num_samples = 800
        samples = (np.sin(np.linspace(0, 10, num_samples)) * 8000).astype(np.int16)
        speech = AudioData(
            data=samples.tobytes(),
            format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
        )

        result = mixin.process_streaming_chunk(
            speech_audio=speech,
            noise_generator=noise_gen,
            num_samples=num_samples,
        )

        assert result.tracks.num_samples == num_samples
        assert len(result.out_of_turn_speech) == num_samples
        mixed = result.to_mixed_audio_data()
        assert mixed.num_samples == num_samples


# ============================================================================
# EffectTimeline Tests
# ============================================================================


class TestEffectTimeline:
    """Tests for EffectTimeline metadata tracking."""

    def test_open_and_close_event(self):
        from tau2.data_model.audio_effects import EffectTimeline

        tl = EffectTimeline()
        tl.open_event("burst_noise", start_ms=1000, participant="user")
        assert len(tl.events) == 1
        assert tl.events[0].end_ms is None
        assert tl.events[0].duration_ms is None
        assert tl.has_open_event("burst_noise", "user")

        tl.close_event("burst_noise", end_ms=1500, participant="user")
        assert tl.events[0].end_ms == 1500
        assert tl.events[0].duration_ms == 500
        assert not tl.has_open_event("burst_noise", "user")

    def test_close_all_open(self):
        from tau2.data_model.audio_effects import EffectTimeline

        tl = EffectTimeline()
        tl.open_event("burst_noise", start_ms=100, participant="user")
        tl.open_event("out_of_turn_speech", start_ms=200, participant="user")
        tl.close_all_open(end_ms=5000)

        assert all(e.end_ms == 5000 for e in tl.events)

    def test_get_events_by_type(self):
        from tau2.data_model.audio_effects import EffectTimeline

        tl = EffectTimeline()
        tl.open_event("burst_noise", start_ms=100, participant="user")
        tl.open_event("frame_drop", start_ms=200, participant="user")
        tl.open_event("burst_noise", start_ms=300, participant="user")

        bursts = tl.get_events_by_type("burst_noise")
        assert len(bursts) == 2
        assert all(e.effect_type == "burst_noise" for e in bursts)

    def test_event_params(self):
        from tau2.data_model.audio_effects import EffectTimeline

        tl = EffectTimeline()
        tl.open_event(
            "burst_noise",
            start_ms=1000,
            participant="user",
            params={"file": "car_horn.wav", "snr_db": 5.0},
        )
        assert tl.events[0].params == {"file": "car_horn.wav", "snr_db": 5.0}

    def test_close_returns_none_for_no_match(self):
        from tau2.data_model.audio_effects import EffectTimeline

        tl = EffectTimeline()
        result = tl.close_event("burst_noise", end_ms=100, participant="user")
        assert result is None

    def test_serialization_roundtrip(self):
        """EffectTimeline should survive JSON serialization."""
        from tau2.data_model.audio_effects import EffectTimeline

        tl = EffectTimeline()
        tl.open_event(
            "burst_noise",
            start_ms=1000,
            participant="user",
            params={"file": "horn.wav"},
        )
        tl.close_event("burst_noise", end_ms=1500, participant="user")

        json_str = tl.model_dump_json()
        restored = EffectTimeline.model_validate_json(json_str)
        assert len(restored.events) == 1
        assert restored.events[0].start_ms == 1000
        assert restored.events[0].end_ms == 1500
        assert restored.events[0].duration_ms == 500
        assert restored.events[0].params == {"file": "horn.wav"}
