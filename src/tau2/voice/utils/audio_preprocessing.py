# Copyright Sierra
"""Audio preprocessing utilities for voice synthesis and transcription."""

import audioop
from copy import deepcopy
from dataclasses import dataclass
from math import gcd
from typing import Literal, Optional

import numpy as np
from loguru import logger

from tau2.data_model.audio import (
    PCM_SAMPLE_RATE,
    AudioData,
    AudioEncoding,
    AudioFormat,
)
from tau2.voice_config import (
    DEFAULT_FADE_OUT_SAMPLES,
    MIN_BURST_RMS,
    MIN_RMS_THRESHOLD,
    NORMALIZE_RATIO,
    SNR_SPEECH_REFERENCE_RMS,
)

# μ-law companding parameter (G.711 standard, not tunable)
MULAW_MU = 255


def apply_fade_out(
    samples: np.ndarray,
    fade_samples: int = DEFAULT_FADE_OUT_SAMPLES,
) -> np.ndarray:
    """Apply a linear fade-out to the end of audio samples to prevent clicking."""
    if len(samples) == 0:
        return samples

    # Limit fade to actual sample length
    fade_samples = min(fade_samples, len(samples))
    if fade_samples <= 0:
        return samples

    # Make a copy to avoid modifying the original
    samples = samples.copy()

    # Apply linear fade-out, preserving original dtype
    fade_window = np.linspace(1.0, 0.0, fade_samples)
    faded = samples[-fade_samples:].astype(np.float64) * fade_window
    samples[-fade_samples:] = faded.astype(samples.dtype)

    return samples


def audio_data_to_numpy(audio: AudioData, dtype: np.dtype = np.int16) -> np.ndarray:
    """Convert AudioData to numpy array."""
    if not audio.format.is_pcm16:
        raise ValueError(
            f"Audio must be in PCM_S16LE format, got {audio.format.encoding}"
        )
    samples = np.frombuffer(audio.data, dtype=dtype)
    return samples.astype(dtype)


def numpy_to_audio_data(
    samples: np.ndarray,
    encoding: AudioEncoding,
    sample_rate: int,
    channels: Literal[1, 2],
    dtype: np.dtype,
) -> AudioData:
    """Convert numpy array to AudioData."""
    return AudioData(
        data=samples.astype(dtype).tobytes(),
        format=AudioFormat(
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,
        ),
        audio_path=None,
    )


def overlay_audio_samples(
    base: np.ndarray,
    overlay: np.ndarray,
    volume: float = 1.0,
) -> np.ndarray:
    """Mix overlay audio into base audio with volume scaling."""
    base = base.copy()
    overlay_scaled = (overlay * volume).astype(np.int16)

    # Mix into base (overlay may be shorter than base)
    mix_length = min(len(base), len(overlay_scaled))
    base[:mix_length] = np.clip(
        base[:mix_length].astype(np.int32)
        + overlay_scaled[:mix_length].astype(np.int32),
        -32768,
        32767,
    ).astype(np.int16)

    return base


def convert_to_pcm16(audio: AudioData) -> AudioData:
    """Convert AudioData of any supported format to PCM_S16LE."""
    raw_data = audio.data
    encoding = audio.format.encoding
    channels = audio.format.channels
    sample_rate = audio.format.sample_rate

    if encoding == AudioEncoding.PCM_U8:
        # 8-bit unsigned (0-255) -> signed 16-bit
        raw_data = audioop.bias(raw_data, 1, -128)  # Convert to signed
        raw_data = audioop.lin2lin(raw_data, 1, 2)
    elif encoding == AudioEncoding.PCM_S16LE:
        # Already 16-bit PCM
        pass
    elif encoding == AudioEncoding.PCM_S24LE:
        # 24-bit PCM -> 16-bit (lossy): take upper 2 bytes of each 3-byte sample
        logger.warning(
            "Converting 24-bit PCM to 16-bit PCM, lossy conversion is performed."
        )
        # Extract bytes 1 and 2 from each 3-byte sample (little-endian: MSB is last)
        raw_data = bytes(b for i, b in enumerate(raw_data) if i % 3 != 0)
    elif encoding == AudioEncoding.PCM_S32LE:
        # 32-bit PCM -> 16-bit (lossy)
        logger.warning(
            "Converting 32-bit PCM to 16-bit PCM, lossy conversion is performed."
        )
        raw_data = audioop.lin2lin(raw_data, 4, 2)
    elif encoding == AudioEncoding.ULAW:
        raw_data = audioop.ulaw2lin(raw_data, 2)
    elif encoding == AudioEncoding.ALAW:
        raw_data = audioop.alaw2lin(raw_data, 2)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    fmt = AudioFormat(
        encoding=AudioEncoding.PCM_S16LE,
        sample_rate=sample_rate,
        channels=channels,
    )
    return AudioData(data=raw_data, format=fmt, audio_path=audio.audio_path)


def convert_to_ulaw(audio: AudioData) -> AudioData:
    """Convert any AudioData to 8-bit μ-law (G.711) encoding."""
    # If already µ-law, just return a copy
    if audio.format.is_ulaw:
        return AudioData(
            data=audio.data,
            format=AudioFormat(
                encoding=AudioEncoding.ULAW,
                sample_rate=audio.format.sample_rate,
                channels=audio.format.channels,
            ),
            audio_path=audio.audio_path,
        )

    # Step 1: Convert to PCM16
    pcm16_audio = convert_to_pcm16(audio)

    # Step 2: Convert PCM16 -> µ-law
    ulaw_bytes = audioop.lin2ulaw(pcm16_audio.data, 2)

    return AudioData(
        data=ulaw_bytes,
        format=AudioFormat(
            encoding=AudioEncoding.ULAW,
            sample_rate=pcm16_audio.format.sample_rate,
            channels=pcm16_audio.format.channels,
        ),
        audio_path=audio.audio_path,
    )


def convert_to_alaw(audio: AudioData) -> AudioData:
    """Convert any AudioData to 8-bit A-law (G.711) encoding."""
    # If already A-law, just return a copy
    if audio.format.is_alaw:
        return AudioData(
            data=audio.data,
            format=AudioFormat(
                encoding=AudioEncoding.ALAW,
                sample_rate=audio.format.sample_rate,
                channels=audio.format.channels,
            ),
            audio_path=audio.audio_path,
        )

    # Step 1: Convert to PCM16
    pcm16_audio = convert_to_pcm16(audio)

    # Step 2: Convert PCM16 -> A-law
    alaw_bytes = audioop.lin2alaw(pcm16_audio.data, 2)

    return AudioData(
        data=alaw_bytes,
        format=AudioFormat(
            encoding=AudioEncoding.ALAW,
            sample_rate=pcm16_audio.format.sample_rate,
            channels=pcm16_audio.format.channels,
        ),
        audio_path=audio.audio_path,
    )


def generate_silence_audio(duration_ms: int) -> AudioData:
    """Generate silence audio of a given duration (16kHz PCM_S16LE mono)."""
    duration_seconds = duration_ms / 1000  # seconds
    sample_rate = PCM_SAMPLE_RATE
    encoding = AudioEncoding.PCM_S16LE
    num_bytes = int(duration_seconds * sample_rate * encoding.sample_width)

    return AudioData(
        data=b"\x00" * num_bytes,
        format=AudioFormat(
            encoding=encoding,
            sample_rate=sample_rate,
        ),
        audio_path=None,
    )


def merge_audio_datas(
    audio_datas: list[AudioData], silence_duration_ms: Optional[int] = 1000
) -> AudioData:
    """Merge a list of AudioData objects into a single AudioData with silence gaps."""
    # Convert all audio datas to 16-bit PCM.
    audio_datas = [convert_to_pcm16(audio_data) for audio_data in audio_datas]
    # Generate silence audio between each audio data.
    if silence_duration_ms is not None:
        silence_audio = generate_silence_audio(silence_duration_ms)
    else:
        silence_audio = None
    merged_audio_data = b""
    for i, audio_data in enumerate(audio_datas):
        merged_audio_data += audio_data.data
        if silence_audio is not None and i < len(audio_datas) - 1:
            merged_audio_data += silence_audio.data
    return AudioData(
        data=merged_audio_data,
        format=deepcopy(audio_datas[0].format),
        audio_path=None,
    )


def convert_to_mono(audio: AudioData) -> AudioData:
    """Convert stereo audio to mono by averaging channels."""
    if audio.format.channels == 1:
        return audio
    elif audio.format.channels == 2:
        mono_data = audioop.tomono(audio.data, audio.format.sample_width, 1, 1)
        return AudioData(
            data=mono_data,
            format=AudioFormat(
                encoding=audio.format.encoding,
                sample_rate=audio.format.sample_rate,
                channels=1,
            ),
            audio_path=None,
        )
    else:
        raise ValueError(f"Unsupported number of channels: {audio.format.channels}")


def convert_to_stereo(audio1: AudioData, audio2: AudioData) -> AudioData:
    """Convert two mono audio tracks to stereo by interleaving samples."""
    import numpy as np

    if audio1.format.channels != 1 or audio2.format.channels != 1:
        raise ValueError(
            f"Audio must be mono, got {audio1.format.channels} and {audio2.format.channels}"
        )
    if audio1.format != audio2.format:
        raise ValueError(
            f"Audio must have the same format, got {audio1.format} and {audio2.format}"
        )
    if len(audio1.data) != len(audio2.data):
        raise ValueError(
            f"Audio must have the same length, got {len(audio1.data)} and {len(audio2.data)}"
        )

    # Determine numpy dtype based on sample width
    sample_width = audio1.format.sample_width
    if sample_width == 1:
        dtype = np.int8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Convert to numpy arrays
    left_samples = np.frombuffer(audio1.data, dtype=dtype)
    right_samples = np.frombuffer(audio2.data, dtype=dtype)

    # Interleave: [L0, R0, L1, R1, L2, R2, ...]
    stereo_samples = np.empty(len(left_samples) * 2, dtype=dtype)
    stereo_samples[0::2] = left_samples
    stereo_samples[1::2] = right_samples

    return AudioData(
        data=stereo_samples.tobytes(),
        format=AudioFormat(
            encoding=audio1.format.encoding,
            sample_rate=audio1.format.sample_rate,
            channels=2,
        ),
        audio_path=None,
    )


def resample_audio(audio: AudioData, new_sample_rate: int) -> AudioData:
    """Resample audio to a new sample rate with proper anti-aliasing."""

    # Handle empty audio
    if len(audio.data) == 0:
        logger.warning("Cannot resample empty audio data")
        return AudioData(
            data=b"",
            format=AudioFormat(
                encoding=audio.format.encoding,
                sample_rate=new_sample_rate,
                channels=audio.format.channels,
            ),
            audio_path=None,
        )

    # No resampling needed
    if audio.format.sample_rate == new_sample_rate:
        return AudioData(
            data=audio.data,
            format=AudioFormat(
                encoding=audio.format.encoding,
                sample_rate=new_sample_rate,
                channels=audio.format.channels,
            ),
            audio_path=None,
        )

    from scipy.signal import resample_poly

    samples = np.frombuffer(audio.data, dtype=np.int16).astype(np.float64)

    # Calculate up/down factors
    g = gcd(audio.format.sample_rate, new_sample_rate)
    up = new_sample_rate // g
    down = audio.format.sample_rate // g

    # Resample (includes anti-aliasing filter)
    resampled = resample_poly(samples, up, down)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

    return AudioData(
        data=resampled.tobytes(),
        format=AudioFormat(
            encoding=audio.format.encoding,
            sample_rate=new_sample_rate,
            channels=audio.format.channels,
        ),
        audio_path=None,
    )


def normalize_audio(
    audio: AudioData, max_value: float = 32767, normalize_ratio: float = 1.0
) -> AudioData:
    """Normalize audio to the given max value and ratio."""
    if not audio.format.is_pcm16:
        raise ValueError(
            f"Audio must be in PCM_S16LE format, got {audio.format.encoding}"
        )
    # Read as int16 first, then convert to float32 for processing
    samples = audio_data_to_numpy(audio, dtype=np.int16).astype(np.float32)

    # Check if array is empty
    if len(samples) == 0:
        logger.warning("Cannot normalize empty audio data")
        return audio

    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples * (max_value / max_val) * normalize_ratio

    return numpy_to_audio_data(
        samples,
        encoding=AudioEncoding.PCM_S16LE,
        sample_rate=audio.format.sample_rate,
        channels=audio.format.channels,
        dtype=np.int16,
    )


def pad_audio_with_zeros(
    audio: AudioData, target_num_samples: int, apply_fade_out_flag: bool = True
) -> AudioData:
    """Pad audio with zeros to reach target number of samples."""
    current_num_samples = audio.num_samples

    # If already at or above target, return as is
    if current_num_samples >= target_num_samples:
        return audio

    # Apply fade-out to prevent clicking at the transition to silence
    if apply_fade_out_flag and current_num_samples > 0 and audio.format.is_pcm16:
        samples = np.frombuffer(audio.data, dtype=np.int16)
        samples = apply_fade_out(samples)
        audio = AudioData(
            data=samples.tobytes(),
            format=deepcopy(audio.format),
            audio_path=audio.audio_path,
        )

    # Calculate how many samples to add
    samples_to_add = target_num_samples - current_num_samples
    bytes_to_add = samples_to_add * audio.format.bytes_per_sample

    # Pad with zeros (silence)
    padded_data = audio.data + (b"\x00" * bytes_to_add)

    return AudioData(
        data=padded_data,
        format=deepcopy(audio.format),
        audio_path=audio.audio_path,
    )


def _compute_rms(samples: np.ndarray) -> float:
    """Compute RMS of audio samples."""
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _snr_to_scale(snr_db: float, noise_rms: float) -> float:
    """Convert target SNR to noise scale factor using fixed reference speech level."""
    if noise_rms < MIN_RMS_THRESHOLD:
        return 0.0  # No noise to scale

    return (SNR_SPEECH_REFERENCE_RMS / noise_rms) * (10 ** (-snr_db / 20))


@dataclass
class AudioTracks:
    """Individual audio tracks before mixing, stored as float64 for precision.

    Each track is independently scaled and can be summed to reproduce the
    mixed output. Zero regions mean the effect is inactive for that portion.
    Tracks are always time-aligned and the same length.

    Tracks are kept in float64 so that summing them produces the same result
    as the original single-pass mix (no intermediate int16 quantization).
    Conversion to int16 happens only at output boundaries (WAV taps, final mix).
    """

    speech: np.ndarray
    background_noise: np.ndarray
    burst_noise: np.ndarray
    sample_rate: int
    num_samples: int
    metadata: dict

    def sum(self) -> np.ndarray:
        """Sum all tracks and clip to int16 range."""
        mixed = self.speech + self.background_noise + self.burst_noise
        return np.clip(mixed, -32768, 32767).astype(np.int16)

    def to_audio_data(self) -> AudioData:
        """Sum tracks and return as AudioData."""
        return numpy_to_audio_data(
            self.sum(),
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=self.sample_rate,
            channels=1,
            dtype=np.int16,
        )

    def track_to_int16(self, track_name: str) -> np.ndarray:
        """Get a track as int16 (for WAV tap recording)."""
        arr = getattr(self, track_name)
        return np.clip(arr, -32768, 32767).astype(np.int16)


def mix_audio_to_tracks(
    speech: Optional[AudioData],
    noise: Optional[AudioData],
    snr_envelope_db: Optional[np.ndarray],
    allow_resample_snr: bool = True,
    burst_info: Optional[tuple[np.ndarray, float, float]] = None,
) -> AudioTracks:
    """Compute individually scaled audio tracks without mixing them.

    Returns an AudioTracks with speech, background_noise, and burst_noise
    as separate float64 arrays. Call .sum() or .to_audio_data() to get the
    mixed result.

    Args:
        speech: Speech audio data (PCM_S16LE)
        noise: Background noise audio data (PCM_S16LE)
        snr_envelope_db: SNR envelope in dB for noise mixing
        allow_resample_snr: Whether to allow resampling the SNR envelope
        burst_info: Tuple of (samples, snr_db, whole_burst_rms) for burst mixing
    """
    if speech is None and noise is None:
        raise ValueError("Either speech or noise must be provided")

    if speech is not None and not speech.format.is_pcm16:
        raise ValueError(f"Speech must be PCM_S16LE, got {speech.format.encoding}")
    if noise is not None and not noise.format.is_pcm16:
        raise ValueError(f"Noise must be PCM_S16LE, got {noise.format.encoding}")

    if speech is not None and noise is not None:
        if len(speech.data) != len(noise.data):
            raise ValueError(
                f"Audio must have the same length, got {len(speech.data)} and {len(noise.data)}"
            )

    format_source = speech if speech is not None else noise
    sample_rate = format_source.format.sample_rate

    # Convert to numpy arrays (float64 for scaling precision)
    if speech is not None:
        speech_f64 = audio_data_to_numpy(speech, dtype=np.int16).astype(np.float64)
    else:
        speech_f64 = None

    if noise is not None:
        noise_f64 = audio_data_to_numpy(noise, dtype=np.int16).astype(np.float64)
    else:
        noise_f64 = None

    if speech_f64 is None:
        speech_f64 = np.zeros_like(noise_f64)
    if noise_f64 is None:
        noise_f64 = np.zeros_like(speech_f64)

    num_samples = len(speech_f64)
    metadata: dict = {}

    # Scale background noise by SNR
    scaled_noise = np.zeros(num_samples, dtype=np.float64)
    noise_rms = _compute_rms(noise_f64)

    if snr_envelope_db is not None and noise_rms >= MIN_RMS_THRESHOLD:
        if len(snr_envelope_db) != num_samples:
            if not allow_resample_snr:
                raise ValueError(
                    f"SNR envelope must have the same length as the audio, "
                    f"got {len(snr_envelope_db)} and {num_samples}"
                )
            snr_envelope_db = np.interp(
                np.linspace(0, len(snr_envelope_db) - 1, num_samples),
                np.arange(len(snr_envelope_db)),
                snr_envelope_db,
            )

        avg_snr_db = float(np.mean(snr_envelope_db))
        noise_scale = _snr_to_scale(avg_snr_db, noise_rms)
        scaled_noise = noise_f64 * noise_scale
        metadata["bg_noise_avg_snr_db"] = avg_snr_db
        metadata["bg_noise_scale"] = noise_scale

    # Scale burst noise by its SNR using the whole-burst RMS
    scaled_burst = np.zeros(num_samples, dtype=np.float64)
    if burst_info is not None:
        burst_samples, burst_snr_db, whole_burst_rms = burst_info
        burst_f64 = burst_samples.astype(np.float64)

        if whole_burst_rms >= MIN_BURST_RMS:
            burst_scale = _snr_to_scale(burst_snr_db, whole_burst_rms)
            if len(burst_f64) < num_samples:
                padded = np.zeros(num_samples, dtype=np.float64)
                padded[: len(burst_f64)] = burst_f64
                burst_f64 = padded
            elif len(burst_f64) > num_samples:
                burst_f64 = burst_f64[:num_samples]

            scaled_burst = burst_f64 * burst_scale
            metadata["burst_snr_db"] = burst_snr_db
            metadata["burst_scale"] = burst_scale

    return AudioTracks(
        speech=speech_f64,
        background_noise=scaled_noise,
        burst_noise=scaled_burst,
        sample_rate=sample_rate,
        num_samples=num_samples,
        metadata=metadata,
    )


def mix_audio_dynamic(
    speech: Optional[AudioData],
    noise: Optional[AudioData],
    snr_envelope_db: Optional[np.ndarray],
    allow_resample_snr: bool = True,
    burst_info: Optional[tuple[np.ndarray, float, float]] = None,
) -> AudioData:
    """Mix speech with background noise using SNR-based scaling.

    Convenience wrapper around mix_audio_to_tracks() that sums the tracks
    and returns a single AudioData.

    Args:
        speech: Speech audio data (PCM_S16LE)
        noise: Background noise audio data (PCM_S16LE)
        snr_envelope_db: SNR envelope in dB for noise mixing
        allow_resample_snr: Whether to allow resampling the SNR envelope
        burst_info: Tuple of (samples, snr_db, whole_burst_rms) for burst mixing
    """
    tracks = mix_audio_to_tracks(
        speech=speech,
        noise=noise,
        snr_envelope_db=snr_envelope_db,
        allow_resample_snr=allow_resample_snr,
        burst_info=burst_info,
    )
    return tracks.to_audio_data()


# Default volume for overlay audio
DEFAULT_OVERLAY_VOLUME = 0.75


def overlay_audio_on_chunk(
    base_audio: AudioData,
    overlay_audio: AudioData,
    volume: float = DEFAULT_OVERLAY_VOLUME,
) -> AudioData:
    """Overlay audio on a chunk, starting at the beginning."""
    if not base_audio.format.is_pcm16:
        raise ValueError(
            f"Base audio must be PCM_S16LE, got {base_audio.format.encoding}"
        )

    try:
        if overlay_audio.format.sample_rate != base_audio.format.sample_rate:
            overlay_audio = resample_audio(overlay_audio, base_audio.format.sample_rate)

        overlay_pcm = convert_to_pcm16(overlay_audio)
        overlay_pcm = normalize_audio(
            overlay_pcm, max_value=32767, normalize_ratio=NORMALIZE_RATIO
        )

        base_np = audio_data_to_numpy(base_audio, dtype=np.int16)
        overlay_np = audio_data_to_numpy(overlay_pcm, dtype=np.int16)
        mixed_np = overlay_audio_samples(base_np, overlay_np, volume)

        return numpy_to_audio_data(
            mixed_np,
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=base_audio.format.sample_rate,
            channels=base_audio.format.channels,
            dtype=np.int16,
        )

    except Exception as e:
        logger.error(f"Error overlaying audio: {e}. Returning original audio.")
        return base_audio
