# Copyright Sierra
"""Background noise generator for voice synthesis."""

import random
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.audio_effects import SourceEffectsConfig
from tau2.voice.utils.audio_io import load_wav_file
from tau2.voice.utils.audio_preprocessing import (
    apply_fade_out,
    convert_to_pcm16,
    mix_audio_dynamic,
    normalize_audio,
    resample_audio,
)
from tau2.voice.utils.probability import poisson_should_trigger
from tau2.voice.utils.utils import BURST_NOISE_FILES
from tau2.voice_config import (
    BURST_SNR_RANGE_DB,
    NOISE_SNR_DB,
    NOISE_SNR_DRIFT_DB,
    NOISE_VARIATION_SPEED,
    NORMALIZE_RATIO,
)

CHANNELS = 1
MAX_VOLUME_FACTOR = 0.6


class BackgroundNoiseGenerator:
    """Looping background noise with SNR-based mixing and burst sound support."""

    def __init__(
        self,
        audio: Optional[AudioData] = None,
        audio_file_path: Optional[Path] = None,
        sample_rate: Optional[int] = None,
        max_volume_factor: float = MAX_VOLUME_FACTOR,
        # SNR-based parameters
        snr_db: float = NOISE_SNR_DB,
        snr_drift_db: float = NOISE_SNR_DRIFT_DB,
        variation_speed: float = NOISE_VARIATION_SPEED,
        primary_variation: tuple[float, float] = (1.0, 1.0),
        harmonic_variation: tuple[float, float] = (0.3, 2.7),
        slow_drift: tuple[float, float] = (0.2, 0.4),
        position: int = 0,
        silent_mode: bool = False,
        seed: int = 42,
        burst_rate: float = 0.0,
        burst_noise_files: Optional[list[Path]] = None,
        burst_snr_range_db: tuple[float, float] = BURST_SNR_RANGE_DB,
    ):
        self.audio = audio
        self.audio_file_path = audio_file_path
        self.sample_rate = sample_rate
        self.max_volume_factor = max_volume_factor
        # SNR-based parameters
        self.snr_db = snr_db
        self.snr_drift_db = snr_drift_db
        self.variation_speed = variation_speed
        self.primary_variation = primary_variation
        self.harmonic_variation = harmonic_variation
        self.slow_drift = slow_drift
        self.position = 0
        self.silent_mode = silent_mode
        self.total_samples_processed: int = 0

        # Burst sound state (one-shot, plays through then clears)
        self.burst_audio: Optional[bytes] = None
        self.burst_offset: int = 0
        self.burst_snr_db: Optional[float] = None  # SNR for current burst
        self.burst_rms: Optional[float] = None  # RMS computed over entire burst
        self.burst_snr_range_db = burst_snr_range_db

        # Scheduling state
        self.burst_rate = burst_rate
        self.trigger_rng = random.Random(seed)
        self.selection_rng = random.Random(seed + 1)
        self.burst_snr_rng = random.Random(seed + 2)

        self.burst_noise_files = (
            burst_noise_files if burst_noise_files is not None else BURST_NOISE_FILES
        )

        # Handle silent mode
        if silent_mode:
            if sample_rate is None:
                raise ValueError("sample_rate must be provided for silent_mode")
            self.audio = self._create_silent_audio(sample_rate)
            self.sample_rate = sample_rate
        elif self.audio is not None and self.audio_file_path is not None:
            raise ValueError("Either audio or audio_file_path must be provided")
        elif self.audio is not None:
            if not audio.format.is_pcm16:
                raise ValueError(
                    f"Audio must be PCM_S16LE, got {audio.format.encoding}"
                )
            self.audio = audio
        elif self.audio_file_path is not None:
            self._load_noise_from_wav(self.audio_file_path)
        else:
            raise ValueError(
                "Either audio, audio_file_path, or silent_mode must be set to True"
            )

    @staticmethod
    def _create_silent_audio(sample_rate: int) -> AudioData:
        """Create a small buffer of silent audio (1 second) to use for looping."""
        duration_ms = 1000
        encoding = AudioEncoding.PCM_S16LE
        num_bytes = int((duration_ms / 1000) * sample_rate * encoding.sample_width)
        return AudioData(
            data=b"\x00" * num_bytes,
            format=AudioFormat(encoding=encoding, sample_rate=sample_rate),
            audio_path=None,
        )

    def _prepare_audio(
        self, file_path: Path, normalize_ratio: float = NORMALIZE_RATIO
    ) -> AudioData:
        """Load, resample, convert to PCM16, and normalize audio from a WAV file."""
        audio = load_wav_file(file_path)
        if audio.format.channels > CHANNELS:
            raise ValueError(
                f"Audio file must have {CHANNELS} channels, got {audio.format.channels}"
            )
        if self.sample_rate is not None:
            audio = resample_audio(audio, new_sample_rate=self.sample_rate)
        audio = convert_to_pcm16(audio)
        audio = normalize_audio(audio, max_value=32767, normalize_ratio=normalize_ratio)
        return audio

    def _load_noise_from_wav(self, file_path: Path) -> None:
        """Load background noise from a WAV file."""
        if self.sample_rate is None:
            audio = load_wav_file(file_path)
            self.sample_rate = audio.format.sample_rate
        self.audio = self._prepare_audio(
            file_path, normalize_ratio=self.max_volume_factor
        )
        self.position = 0

    def _load_burst_from_wav(self, file_path: Path) -> None:
        """Load a burst sound from a WAV file and pre-compute its RMS."""
        audio = self._prepare_audio(file_path, normalize_ratio=NORMALIZE_RATIO)
        self.burst_audio = audio.data
        self.burst_offset = 0
        # Pre-compute RMS over the entire burst so that all chunks use the
        # same scale factor, preserving the burst's natural dynamics.
        all_samples = np.frombuffer(audio.data, dtype=np.int16).astype(np.float64)
        self.burst_rms = float(np.sqrt(np.mean(all_samples**2)))

    def generate_snr_envelope(
        self,
        start_time_seconds: float,
        sample_rate: int,
        num_samples: int,
        variation_speed: float,
        snr_db: float,
        snr_drift_db: float,
        primary_variation: tuple[float, float],
        harmonic_variation: tuple[float, float],
        slow_drift: tuple[float, float],
    ) -> np.ndarray:
        """Generate smooth SNR envelope in dB oscillating around snr_db +/- drift."""
        duration_seconds = num_samples / sample_rate
        t = np.linspace(
            start_time_seconds, start_time_seconds + duration_seconds, num_samples
        )

        primary_amp, primary_freq = primary_variation
        harmonic_amp, harmonic_freq = harmonic_variation
        drift_amp, drift_freq = slow_drift

        envelope = (
            primary_amp * np.sin(2 * np.pi * variation_speed * primary_freq * t)
            + harmonic_amp * np.sin(2 * np.pi * variation_speed * harmonic_freq * t)
            + drift_amp * np.sin(2 * np.pi * variation_speed * drift_freq * t)
        )

        # Normalize to [-1, 1]
        amplitude_sum = primary_amp + harmonic_amp + drift_amp
        envelope_min, envelope_max = -amplitude_sum, amplitude_sum
        envelope = (envelope - envelope_min) / (envelope_max - envelope_min)
        envelope = envelope * 2 - 1  # Map [0, 1] to [-1, 1]

        # Map to SNR range: snr_db +/- snr_drift_db
        return snr_db + envelope * snr_drift_db

    def add_burst(self, burst_file: Path) -> None:
        """Add a one-shot burst sound to be mixed into subsequent chunks."""
        self._load_burst_from_wav(burst_file)
        # Sample SNR for this burst from the range
        min_snr, max_snr = self.burst_snr_range_db
        self.burst_snr_db = self.burst_snr_rng.uniform(min_snr, max_snr)

    def has_active_burst(self) -> bool:
        """Check if there's a burst currently playing."""
        return self.burst_audio is not None

    def should_trigger_burst(self, chunk_duration_sec: float) -> Optional[Path]:
        """Check if a burst should be triggered for this chunk."""
        if self.burst_rate <= 0 or not self.burst_noise_files:
            return None
        if self.has_active_burst():
            return None

        if poisson_should_trigger(
            self.burst_rate, chunk_duration_sec, self.trigger_rng
        ):
            return self.selection_rng.choice(self.burst_noise_files)
        return None

    def get_next_chunk(self, num_samples: int) -> tuple[AudioData, np.ndarray]:
        """Get next chunk of background noise and SNR envelope."""
        if self.audio is None:
            raise ValueError("Noise audio not loaded")

        noise_data_bytes = self.audio.data
        total_num_samples = self.audio.num_samples
        bytes_per_sample = self.audio.format.bytes_per_sample

        chunk_bytes = b""
        samples_remaining = num_samples

        while samples_remaining > 0:
            samples_available = total_num_samples - self.position
            samples_to_take = min(samples_available, samples_remaining)

            start_byte = self.position * bytes_per_sample
            end_byte = (self.position + samples_to_take) * bytes_per_sample
            chunk_bytes += noise_data_bytes[start_byte:end_byte]

            self.position += samples_to_take
            samples_remaining -= samples_to_take

            if self.position >= total_num_samples:
                self.position = 0

        # Generate SNR envelope (in dB) instead of volume envelope
        snr_envelope = self.generate_snr_envelope(
            start_time_seconds=self.total_samples_processed / self.sample_rate,
            sample_rate=self.sample_rate,
            num_samples=num_samples,
            variation_speed=self.variation_speed,
            snr_db=self.snr_db,
            snr_drift_db=self.snr_drift_db,
            primary_variation=self.primary_variation,
            harmonic_variation=self.harmonic_variation,
            slow_drift=self.slow_drift,
        )
        self.total_samples_processed += num_samples
        next_audio_chunk = AudioData(
            data=chunk_bytes,
            format=deepcopy(self.audio.format),
            audio_path=None,
        )

        assert next_audio_chunk.num_samples == num_samples, (
            f"Expected {num_samples} samples, got {next_audio_chunk.num_samples}"
        )

        # Note: burst mixing is now handled in mix_audio_dynamic with SNR
        # We just return the raw noise chunk and the burst info is stored

        return next_audio_chunk, snr_envelope

    def get_burst_chunk(
        self, num_samples: int
    ) -> Optional[tuple[np.ndarray, float, float]]:
        """Get the next chunk of burst audio if active, or None.

        Returns:
            Tuple of (samples, snr_db, whole_burst_rms) or None if no active burst.
            whole_burst_rms is pre-computed over the entire burst at load time so
            that every chunk uses the same scale factor.
        """
        if self.burst_audio is None or self.burst_snr_db is None:
            return None

        bytes_per_sample = 2
        chunk_bytes = num_samples * bytes_per_sample

        remaining = len(self.burst_audio) - self.burst_offset
        if remaining <= 0:
            self._clear_burst()
            return None

        portion_length = min(chunk_bytes, remaining)
        burst_portion = self.burst_audio[
            self.burst_offset : self.burst_offset + portion_length
        ]

        burst_np = np.frombuffer(burst_portion, dtype=np.int16).copy()

        # Pad with zeros if burst is shorter than requested
        if len(burst_np) < num_samples:
            burst_np = apply_fade_out(burst_np)
            padded = np.zeros(num_samples, dtype=np.int16)
            padded[: len(burst_np)] = burst_np
            burst_np = padded

        self.burst_offset += portion_length

        snr = self.burst_snr_db
        rms = self.burst_rms

        if self.burst_offset >= len(self.burst_audio):
            self._clear_burst()

        return burst_np, snr, rms

    def _clear_burst(self) -> None:
        """Reset all burst state."""
        self.burst_audio = None
        self.burst_offset = 0
        self.burst_snr_db = None
        self.burst_rms = None


def create_background_noise_generator(
    config: SourceEffectsConfig,
    sample_rate: int,
    background_noise_file: Optional[Path] = None,
    burst_noise_files: Optional[list[Path]] = None,
) -> BackgroundNoiseGenerator:
    """Create a background noise generator (silent mode if no file provided)."""
    if background_noise_file is None:
        return BackgroundNoiseGenerator(
            sample_rate=sample_rate,
            silent_mode=True,
            burst_noise_files=burst_noise_files,
            burst_snr_range_db=config.burst_snr_range_db,
        )

    return BackgroundNoiseGenerator(
        audio_file_path=background_noise_file,
        sample_rate=sample_rate,
        snr_db=config.noise_snr_db,
        snr_drift_db=config.noise_snr_drift_db,
        variation_speed=config.noise_variation_speed,
        burst_noise_files=burst_noise_files,
        burst_snr_range_db=config.burst_snr_range_db,
    )


def apply_background_noise(
    audio: AudioData,
    noise_generator: BackgroundNoiseGenerator,
) -> AudioData:
    """Apply background noise to audio using SNR-based mixing."""
    if not audio.format.is_pcm16:
        raise ValueError(f"Audio must be PCM_S16LE, got {audio.format.encoding}")
    if not noise_generator.audio.format.is_pcm16:
        raise ValueError(
            f"Noise generator must be PCM_S16LE, got {noise_generator.audio.format.encoding}"
        )
    if noise_generator.audio.format != audio.format:
        raise ValueError(
            f"Noise generator audio format {noise_generator.audio.format} does not match audio format {audio.format}"
        )
    num_bytes = len(audio.data)
    num_samples = num_bytes // noise_generator.audio.format.bytes_per_sample
    noise_data, snr_envelope = noise_generator.get_next_chunk(num_samples)

    # Get burst chunk if active
    burst_info = noise_generator.get_burst_chunk(num_samples)

    return mix_audio_dynamic(audio, noise_data, snr_envelope, burst_info=burst_info)
