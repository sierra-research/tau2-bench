# Copyright Sierra
"""
Minimal high-level tests for audio preprocessing utilities.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.voice.synthesis.audio_effects.effects import (
    apply_band_pass_filter,
    convert_to_telephony,
)
from tau2.voice.utils.audio_io import load_wav_file, save_wav_file
from tau2.voice.utils.audio_preprocessing import (
    audio_data_to_numpy,
    convert_to_mono,
    convert_to_pcm16,
    convert_to_ulaw,
    mix_audio_dynamic,
    normalize_audio,
    numpy_to_audio_data,
    resample_audio,
)


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
def sample_stereo_audio() -> AudioData:
    """Create a simple stereo PCM16 audio sample."""
    sample_rate = 16000
    duration = 0.5

    # Create stereo samples (interleaved)
    samples_left = (
        np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        * 16000
    ).astype(np.int16)
    samples_right = (
        np.sin(2 * np.pi * 550 * np.linspace(0, duration, int(sample_rate * duration)))
        * 16000
    ).astype(np.int16)
    stereo = np.empty(len(samples_left) * 2, dtype=np.int16)
    stereo[0::2] = samples_left
    stereo[1::2] = samples_right

    return AudioData(
        data=stereo.tobytes(),
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=sample_rate,
            channels=2,
        ),
        audio_path=None,
    )


class TestFormatConversions:
    """Test audio format conversion functions."""

    def test_convert_to_pcm16(self, sample_pcm16_audio):
        """Test conversion to PCM16 format."""
        result = convert_to_pcm16(sample_pcm16_audio)

        assert result.format.encoding == AudioEncoding.PCM_S16LE
        assert result.format.sample_width == 2
        assert len(result.data) > 0

    def test_convert_to_ulaw(self, sample_pcm16_audio):
        """Test conversion to μ-law format."""
        result = convert_to_ulaw(sample_pcm16_audio)

        assert result.format.encoding == AudioEncoding.ULAW
        assert result.format.sample_width == 1
        assert len(result.data) > 0

    def test_convert_to_mono(self, sample_stereo_audio):
        """Test conversion to mono."""
        result = convert_to_mono(sample_stereo_audio)

        assert result.format.channels == 1
        # Mono should have half the data size (2 channels -> 1 channel)
        assert len(result.data) == len(sample_stereo_audio.data) // 2


class TestNumpyConversions:
    """Test numpy array conversions."""

    def test_audio_data_to_numpy(self, sample_pcm16_audio):
        """Test conversion from AudioData to numpy array."""
        samples = audio_data_to_numpy(sample_pcm16_audio)

        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.int16
        assert len(samples) == len(sample_pcm16_audio.data) // 2

    def test_numpy_to_audio_data(self):
        """Test conversion from numpy array to AudioData."""
        samples = np.array([1000, 2000, 3000], dtype=np.int16)

        result = numpy_to_audio_data(
            samples,
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=8000,
            channels=1,
            dtype=np.int16,
        )

        assert result.format.encoding == AudioEncoding.PCM_S16LE
        assert result.format.sample_rate == 8000
        assert len(result.data) == 6  # 3 samples * 2 bytes


class TestFileOperations:
    """Test file loading and saving operations."""

    def test_save_and_load_wav_file(self, sample_pcm16_audio):
        """Test saving and loading WAV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.wav"

            # Save
            save_wav_file(sample_pcm16_audio, filepath)
            assert filepath.exists()

            # Load
            loaded = load_wav_file(str(filepath))
            assert loaded.format.sample_rate == sample_pcm16_audio.format.sample_rate
            assert loaded.format.channels == sample_pcm16_audio.format.channels
            assert len(loaded.data) == len(sample_pcm16_audio.data)


class TestAudioProcessing:
    """Test audio processing functions."""

    def test_resample_audio(self, sample_pcm16_audio):
        """Test audio resampling."""
        new_rate = 8000
        result = resample_audio(sample_pcm16_audio, new_rate)

        assert result.format.sample_rate == new_rate
        assert result.format.encoding == sample_pcm16_audio.format.encoding
        # Downsampling should reduce data size (roughly by the ratio of sample rates)
        expected_size = (
            len(sample_pcm16_audio.data)
            * new_rate
            // sample_pcm16_audio.format.sample_rate
        )
        # Allow some tolerance due to resampling algorithms
        assert abs(len(result.data) - expected_size) < 100

    def test_normalize_audio(self, sample_pcm16_audio):
        """Test audio normalization."""
        result = normalize_audio(
            sample_pcm16_audio, max_value=32767, normalize_ratio=0.9
        )

        assert result.format.sample_rate == sample_pcm16_audio.format.sample_rate
        assert result.format.sample_width == sample_pcm16_audio.format.sample_width
        assert result.num_samples == sample_pcm16_audio.num_samples

        # Check that normalized audio is not clipping
        samples = audio_data_to_numpy(result)
        assert np.max(np.abs(samples)) <= 32767
        # Check that audio is actually normalized to near the target
        assert np.max(np.abs(samples)) > 20000  # Should be reasonably loud

    def test_apply_band_pass_filter(self, sample_pcm16_audio):
        """Test band-pass filtering."""
        result = apply_band_pass_filter(
            sample_pcm16_audio, low_freq=300, high_freq=3400
        )

        assert result.format.sample_rate == sample_pcm16_audio.format.sample_rate
        assert len(result.data) == len(sample_pcm16_audio.data)


class TestTelephonyConversion:
    """Test telephony format conversion."""

    def test_convert_to_telephony(self, sample_pcm16_audio):
        """Test conversion to telephony format (μ-law 8kHz)."""
        result = convert_to_telephony(sample_pcm16_audio)

        assert result.format.encoding == AudioEncoding.ULAW
        assert result.format.sample_rate == 8000
        assert result.format.sample_width == 1
        assert len(result.data) > 0


class TestAudioMixing:
    """Test audio mixing functions."""

    def test_mix_audio_dynamic(self, sample_pcm16_audio):
        """Test SNR-based dynamic audio mixing."""
        # Create two audio samples of same length
        speech = sample_pcm16_audio
        noise = sample_pcm16_audio  # Using same audio as noise for simplicity

        # Generate SNR envelope in dB (e.g., 10-20 dB range)
        num_samples = len(audio_data_to_numpy(speech))
        snr_envelope_db = np.linspace(10.0, 20.0, num_samples)

        result = mix_audio_dynamic(speech, noise, snr_envelope_db)

        # For linear PCM input, output is PCM_S16LE
        assert result.format.encoding == AudioEncoding.PCM_S16LE
        assert result.format.sample_rate == speech.format.sample_rate
        assert len(result.data) == len(speech.data)

        # Check that mixed audio is not clipping
        samples = audio_data_to_numpy(result)
        assert np.all(samples >= -32768)
        assert np.all(samples <= 32767)
