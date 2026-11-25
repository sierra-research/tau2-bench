"""Shared streaming audio converter between telephony and provider formats.

All voice providers use PCM16 at various sample rates (16kHz, 24kHz) while
the simulation framework uses telephony format (8kHz mu-law). This module
provides a single converter that handles both directions with streaming
resample state to avoid artifacts at chunk boundaries.

Resampling is band-limited (see ``tau2.voice.utils.streaming_resampler``): no
aliasing into the 8 kHz telephone band on the way down and no mirror images
above 4 kHz on the way up. Each direction delays the audio content by a
fixed 3.625 ms (the filter's group delay); cumulative sample counts follow the
rate ratio exactly, and ``reset()`` discards the filter state.

Usage:
    converter = StreamingTelephonyConverter(
        input_sample_rate=16000,   # provider expects 16kHz PCM16 input
        output_sample_rate=24000,  # provider produces 24kHz PCM16 output
    )

    # In each tick:
    provider_audio = converter.convert_input(telephony_audio)
    telephony_audio = converter.convert_output(provider_audio)

    # On interruption or new session:
    converter.reset()
"""

from typing import Optional, Tuple

from tau2.config import DEFAULT_TELEPHONY_RATE
from tau2.voice.utils.streaming_resampler import (
    StreamingResampler,
    pcm16_to_ulaw,
    ulaw_to_pcm16,
)

TELEPHONY_SAMPLE_RATE = DEFAULT_TELEPHONY_RATE


def telephony_to_pcm16(
    audio_bytes: bytes,
    target_sample_rate: int,
    resample_state: Optional[StreamingResampler] = None,
) -> Tuple[bytes, Optional[StreamingResampler]]:
    """Convert telephony audio (8kHz mu-law) to PCM16 at target rate.

    Returns:
        Tuple of (converted audio bytes, new resample state for streaming).
    """
    if len(audio_bytes) == 0:
        return b"", resample_state
    if resample_state is None:
        resample_state = StreamingResampler(TELEPHONY_SAMPLE_RATE, target_sample_rate)
    return resample_state.process(ulaw_to_pcm16(audio_bytes)), resample_state


def pcm16_to_telephony(
    audio_bytes: bytes,
    source_sample_rate: int,
    resample_state: Optional[StreamingResampler] = None,
) -> Tuple[bytes, Optional[StreamingResampler]]:
    """Convert PCM16 at source rate to telephony audio (8kHz mu-law).

    Returns:
        Tuple of (converted audio bytes, new resample state for streaming).
    """
    if len(audio_bytes) == 0:
        return b"", resample_state
    if resample_state is None:
        resample_state = StreamingResampler(source_sample_rate, TELEPHONY_SAMPLE_RATE)
    return pcm16_to_ulaw(resample_state.process(audio_bytes)), resample_state


class StreamingTelephonyConverter:
    """Streaming audio converter between telephony (8kHz mu-law) and PCM16.

    Maintains resample state across calls to avoid audio artifacts at chunk
    boundaries. Used by all provider adapters that need format conversion
    (everything except xAI which uses native mu-law).

    Args:
        input_sample_rate: Provider's expected input rate (e.g., 16000).
        output_sample_rate: Provider's output rate (e.g., 24000).
    """

    def __init__(
        self,
        input_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self._input_resample_state: Optional[StreamingResampler] = None
        self._output_resample_state: Optional[StreamingResampler] = None

    def convert_input(self, telephony_audio: bytes) -> bytes:
        """Convert telephony (8kHz mu-law) to provider input (PCM16)."""
        result, self._input_resample_state = telephony_to_pcm16(
            telephony_audio, self.input_sample_rate, self._input_resample_state
        )
        return result

    def convert_output(self, provider_audio: bytes) -> bytes:
        """Convert provider output (PCM16) to telephony (8kHz mu-law)."""
        result, self._output_resample_state = pcm16_to_telephony(
            provider_audio, self.output_sample_rate, self._output_resample_state
        )
        return result

    def reset(self) -> None:
        """Reset resample state. Call on interruption or new session."""
        self._input_resample_state = None
        self._output_resample_state = None
