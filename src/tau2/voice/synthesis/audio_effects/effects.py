# Copyright Sierra
"""Audio effect functions (burst noise, frame drops, muffling, telephony conversion)."""

import audioop
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from scipy import signal

from tau2.config import DEFAULT_PCM_SAMPLE_RATE
from tau2.data_model.audio import (
    TELEPHONY_SAMPLE_RATE,
    AudioData,
    AudioEncoding,
    AudioFormat,
)
from tau2.voice.utils.audio_io import load_wav_file
from tau2.voice.utils.audio_preprocessing import (
    audio_data_to_numpy,
    convert_to_pcm16,
    convert_to_ulaw,
    normalize_audio,
    numpy_to_audio_data,
    overlay_audio_samples,
    resample_audio,
)
from tau2.voice_config import (
    BURST_SNR_RANGE_DB,
    CONSTANT_MUFFLE_CUTOFF_FREQ,
    MIN_RMS_THRESHOLD,
    MUFFLE_TRANSITION_MS,
    NORMALIZE_RATIO,
    SNR_SPEECH_REFERENCE_RMS,
    TELEPHONY_HEADROOM,
    TELEPHONY_HIGH_FREQ,
    TELEPHONY_LOW_FREQ,
)

TELEPHONY_FORMAT = AudioFormat(
    encoding=AudioEncoding.ULAW,
    sample_rate=TELEPHONY_SAMPLE_RATE,
    channels=1,
)


class StreamingTelephonyConverter:
    """Streaming telephony converter that preserves filter state between chunks."""

    def __init__(self, input_sample_rate: int = DEFAULT_PCM_SAMPLE_RATE):
        self.input_sample_rate = input_sample_rate
        self.resample_state = None

        # Design bandpass filter (300-3400Hz) for 8kHz output
        nyquist = TELEPHONY_SAMPLE_RATE / 2
        low = TELEPHONY_LOW_FREQ / nyquist
        high = TELEPHONY_HIGH_FREQ / nyquist
        self.b, self.a = signal.butter(4, [low, high], btype="band")

        # Filter state - preserving this between chunks prevents clicks
        self.filter_zi = np.zeros(max(len(self.a), len(self.b)) - 1)

    def convert_chunk(self, audio: AudioData) -> AudioData:
        """Convert a single PCM chunk to μ-law 8kHz telephony format."""
        if len(audio.data) == 0:
            return AudioData(data=b"", format=TELEPHONY_FORMAT, audio_path=None)

        if not audio.format.is_pcm16:
            raise ValueError(f"Expected PCM_S16LE input, got {audio.format.encoding}")

        # Resample to 8kHz with state preservation
        resampled_bytes, self.resample_state = audioop.ratecv(
            audio.data,
            audio.format.sample_width,
            audio.format.channels,
            self.input_sample_rate,
            TELEPHONY_SAMPLE_RATE,
            self.resample_state,
        )

        samples = np.frombuffer(resampled_bytes, dtype=np.int16).astype(np.float64)

        # Apply bandpass filter with state preservation (prevents chunk boundary clicks)
        if len(samples) > 0:
            filtered, self.filter_zi = signal.lfilter(
                self.b, self.a, samples, zi=self.filter_zi
            )
        else:
            filtered = samples

        # Apply headroom and clip
        filtered = filtered * TELEPHONY_HEADROOM
        filtered = np.clip(filtered, -32768, 32767).astype(np.int16)

        pcm_filtered = AudioData(
            data=filtered.tobytes(),
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE,
                sample_rate=TELEPHONY_SAMPLE_RATE,
                channels=1,
            ),
            audio_path=None,
        )

        return convert_to_ulaw(pcm_filtered)


def apply_burst_noise(
    audio: AudioData,
    burst_noise_file: Optional[Path] = None,
    burst_snr_db: Optional[float] = None,
) -> AudioData:
    """Apply burst noise at center of audio using SNR-based mixing."""
    if burst_noise_file is None:
        return audio

    if not audio.format.is_pcm16:
        raise ValueError(f"Audio must be PCM_S16LE, got {audio.format.encoding}")

    try:
        burst_audio = load_wav_file(burst_noise_file)

        if burst_audio.format.channels > 1:
            raise ValueError(
                f"Burst audio must be mono, got {burst_audio.format.channels} channels"
            )

        if burst_audio.format.sample_rate != audio.format.sample_rate:
            burst_audio = resample_audio(burst_audio, audio.format.sample_rate)

        burst_pcm = convert_to_pcm16(burst_audio)
        burst_pcm = normalize_audio(
            burst_pcm, max_value=32767, normalize_ratio=NORMALIZE_RATIO
        )

        main_np = audio_data_to_numpy(audio, dtype=np.int16).astype(np.float64)
        burst_np = audio_data_to_numpy(burst_pcm, dtype=np.int16).astype(np.float64)

        main_length = len(main_np)
        burst_length = len(burst_np)
        edge_buffer = 4000

        if main_length <= burst_length + 2 * edge_buffer:
            return audio

        # Sample SNR if not provided
        if burst_snr_db is None:
            import random

            min_snr, max_snr = BURST_SNR_RANGE_DB
            burst_snr_db = random.uniform(min_snr, max_snr)

        # Compute RMS for SNR-based scaling (use fixed reference speech level)
        start_pos = (main_length - burst_length) // 2
        end_pos = min(start_pos + burst_length, main_length)

        burst_segment = burst_np[: end_pos - start_pos]
        burst_rms = float(np.sqrt(np.mean(burst_segment**2)))

        # Use fixed reference speech level for consistent burst volume
        if burst_rms < MIN_RMS_THRESHOLD:
            burst_scale = 0.0
        else:
            burst_scale = (SNR_SPEECH_REFERENCE_RMS / burst_rms) * (
                10 ** (-burst_snr_db / 20)
            )

        # Mix burst into main audio
        main_np[start_pos:end_pos] = overlay_audio_samples(
            main_np[start_pos:end_pos], burst_segment, volume=burst_scale
        )

        return numpy_to_audio_data(
            main_np,
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=audio.format.sample_rate,
            channels=audio.format.channels,
            dtype=np.int16,
        )

    except Exception as e:
        logger.error(f"Error applying burst noise: {e}. Returning original audio.")
        return audio


def apply_frame_drops(
    audio: AudioData,
    drop_count: int,
    drop_duration_ms: int,
) -> AudioData:
    """Simulate packet loss by dropping audio segments at evenly spaced intervals."""
    is_ulaw = audio.format.encoding == AudioEncoding.ULAW
    is_pcm = audio.format.is_pcm16

    if not is_ulaw and not is_pcm:
        raise ValueError(
            f"Audio must be PCM_S16LE or ULAW, got {audio.format.encoding}"
        )

    sample_rate = audio.format.sample_rate
    num_samples = audio.num_samples

    drop_samples = int(drop_duration_ms * sample_rate / 1000)
    drop_samples = min(drop_samples, num_samples)

    total_drop_samples = drop_count * drop_samples
    if num_samples < total_drop_samples:
        return audio

    section_size = num_samples // (drop_count + 1)

    if is_ulaw:
        # For ULAW: work directly with bytes, silence is 0xFF
        data = bytearray(audio.data)
        for i in range(drop_count):
            center = section_size * (i + 1)
            drop_start = max(0, center - drop_samples // 2)
            drop_end = min(num_samples, drop_start + drop_samples)
            for j in range(drop_start, drop_end):
                data[j] = 0xFF
        return AudioData(data=bytes(data), format=audio.format)
    else:
        # For PCM: use numpy
        samples = audio_data_to_numpy(audio, dtype=np.int16).astype(np.float32)
        for i in range(drop_count):
            center = section_size * (i + 1)
            drop_start = max(0, center - drop_samples // 2)
            drop_end = min(num_samples, drop_start + drop_samples)
            samples[drop_start:drop_end] = 0

        return numpy_to_audio_data(
            samples,
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=audio.format.sample_rate,
            channels=audio.format.channels,
            dtype=np.int16,
        )


def apply_dynamic_muffling(
    audio: AudioData,
    segment_count: int,
    segment_duration_ms: int,
    cutoff_freq: float,
    transition_ms: int = MUFFLE_TRANSITION_MS,
) -> AudioData:
    """Apply segment-based muffling to simulate occasional speaker movement."""
    if not audio.format.is_pcm16:
        raise ValueError(f"Audio must be PCM_S16LE, got {audio.format.encoding}")

    samples = audio_data_to_numpy(audio, dtype=np.int16).astype(np.float32)
    sample_rate = audio.format.sample_rate
    num_samples = len(samples)
    nyquist = sample_rate / 2

    segment_samples = int(segment_duration_ms * sample_rate / 1000)
    segment_samples = min(segment_samples, num_samples)
    transition_samples = int(transition_ms * sample_rate / 1000)

    min_required = segment_samples + 2 * transition_samples
    if num_samples < min_required:
        return audio

    section_size = num_samples // (segment_count + 1)

    segments = []
    for i in range(segment_count):
        center = section_size * (i + 1)
        seg_start = max(transition_samples, center - segment_samples // 2)
        seg_end = min(num_samples - transition_samples, seg_start + segment_samples)
        if seg_end > seg_start:
            segments.append((seg_start, seg_end))

    if not segments:
        return audio

    output = samples.copy()

    normalized_cutoff = min(cutoff_freq / nyquist, 0.99)
    normalized_cutoff = max(normalized_cutoff, 0.01)

    b, a = signal.butter(4, normalized_cutoff, btype="low")

    for seg_start, seg_end in segments:
        pad_start = max(0, seg_start - transition_samples)
        pad_end = min(num_samples, seg_end + transition_samples)

        segment_audio = samples[pad_start:pad_end]
        if len(segment_audio) < 10:
            continue

        filtered_segment = signal.filtfilt(b, a, segment_audio)

        segment_len = pad_end - pad_start
        envelope = np.ones(segment_len)

        fade_in_len = min(transition_samples, seg_start - pad_start)
        if fade_in_len > 0:
            envelope[:fade_in_len] = np.linspace(0, 1, fade_in_len)

        fade_out_start = segment_len - min(transition_samples, pad_end - seg_end)
        fade_out_len = segment_len - fade_out_start
        if fade_out_len > 0:
            envelope[fade_out_start:] = np.linspace(1, 0, fade_out_len)

        blended = envelope * filtered_segment + (1 - envelope) * segment_audio
        output[pad_start:pad_end] = blended

    max_val = np.max(np.abs(output))
    original_max = np.max(np.abs(samples))
    if max_val > 0 and original_max > 0:
        output = output * (original_max / max_val)

    return numpy_to_audio_data(
        output,
        encoding=AudioEncoding.PCM_S16LE,
        sample_rate=audio.format.sample_rate,
        channels=audio.format.channels,
        dtype=np.int16,
    )


def apply_constant_muffling(
    audio: AudioData,
    cutoff_freq: float = CONSTANT_MUFFLE_CUTOFF_FREQ,
) -> AudioData:
    """Apply constant muffling to entire audio (speaker facing away from mic)."""
    if not audio.format.is_pcm16:
        raise ValueError(f"Audio must be PCM_S16LE, got {audio.format.encoding}")

    samples = audio_data_to_numpy(audio, dtype=np.int16).astype(np.float32)
    sample_rate = audio.format.sample_rate
    nyquist = sample_rate / 2

    normalized_cutoff = min(cutoff_freq / nyquist, 0.99)
    normalized_cutoff = max(normalized_cutoff, 0.01)

    b, a = signal.butter(4, normalized_cutoff, btype="low")

    filtered = signal.filtfilt(b, a, samples)

    max_val = np.max(np.abs(filtered))
    original_max = np.max(np.abs(samples))
    if max_val > 0 and original_max > 0:
        filtered = filtered * (original_max / max_val)

    return numpy_to_audio_data(
        filtered,
        encoding=AudioEncoding.PCM_S16LE,
        sample_rate=audio.format.sample_rate,
        channels=audio.format.channels,
        dtype=np.int16,
    )


def apply_band_pass_filter(
    audio: AudioData,
    low_freq: float,
    high_freq: float,
) -> AudioData:
    """Apply a band-pass filter to the audio."""
    if not audio.format.is_pcm16:
        raise ValueError(
            f"Audio must be in PCM_S16LE format, got {audio.format.encoding}"
        )
    pcm_bytes = audio.data

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)

    nyquist = audio.format.sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype="band")

    min_length = 3 * max(len(a), len(b))

    if len(samples) > min_length:
        filtered = signal.filtfilt(b, a, samples)
    else:
        logger.warning(
            f"Audio too short ({len(samples)} samples) for band-pass filter "
            f"(requires > {min_length} samples). Skipping filter."
        )
        filtered = samples

    return numpy_to_audio_data(
        filtered,
        encoding=AudioEncoding.PCM_S16LE,
        sample_rate=audio.format.sample_rate,
        channels=audio.format.channels,
        dtype=np.int16,
    )


def convert_to_telephony(audio: AudioData) -> AudioData:
    """Convert audio to μ-law 8kHz telephony format."""
    logger.debug("Converting to telephony format")

    if len(audio.data) == 0:
        logger.warning("Cannot convert empty audio to telephony format")
        return AudioData(
            data=b"",
            format=TELEPHONY_FORMAT,
            audio_path=None,
        )

    if audio.format.is_ulaw:
        pcm_audio = convert_to_pcm16(audio)
    elif audio.format.is_pcm16:
        pcm_audio = audio
    else:
        raise ValueError(f"Unsupported format: {audio.format.encoding}")

    pcm_audio_resampled = resample_audio(pcm_audio, TELEPHONY_SAMPLE_RATE)

    pcm_audio_filtered = apply_band_pass_filter(
        audio=pcm_audio_resampled,
        low_freq=TELEPHONY_LOW_FREQ,
        high_freq=TELEPHONY_HIGH_FREQ,
    )

    samples = audio_data_to_numpy(pcm_audio_filtered, dtype=np.int16).astype(np.float32)
    samples = samples * TELEPHONY_HEADROOM
    samples = np.clip(samples, -32768, 32767)
    pcm_audio_filtered = numpy_to_audio_data(
        samples,
        encoding=AudioEncoding.PCM_S16LE,
        sample_rate=pcm_audio_filtered.format.sample_rate,
        channels=pcm_audio_filtered.format.channels,
        dtype=np.int16,
    )

    return convert_to_ulaw(pcm_audio_filtered)
