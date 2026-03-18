"""Audio format conversion utilities for LiveKit cascaded voice pipeline.

The LiveKit cascaded pipeline uses:
- STT Input:  16kHz PCM16 mono (Deepgram expects this)
- TTS Output: Variable rate PCM16 mono (default 24kHz for Deepgram TTS)

The simulation framework uses telephony format:
- 8kHz μ-law mono

This module provides conversion functions to bridge between formats.
"""

import audioop
from typing import Optional, Tuple

from tau2.config import DEFAULT_PCM_SAMPLE_RATE, DEFAULT_TELEPHONY_RATE

# Telephony format: 8kHz μ-law, 1 byte per sample
TELEPHONY_SAMPLE_RATE = DEFAULT_TELEPHONY_RATE
TELEPHONY_BYTES_PER_SECOND = DEFAULT_TELEPHONY_RATE  # 1 byte/sample for μ-law

# STT input format (Deepgram): 16kHz PCM16
STT_SAMPLE_RATE = DEFAULT_PCM_SAMPLE_RATE
STT_BYTES_PER_SECOND = STT_SAMPLE_RATE * 2  # 2 bytes/sample for PCM16

# TTS output format: Variable (default 24kHz for Deepgram Aura)
DEFAULT_TTS_SAMPLE_RATE = 24000


def telephony_to_stt_input(
    audio_bytes: bytes,
    resample_state: Optional[Tuple] = None,
) -> Tuple[bytes, Optional[Tuple]]:
    """Convert telephony audio (8kHz μ-law) to STT input format (16kHz PCM16).

    Args:
        audio_bytes: Raw audio bytes in 8kHz μ-law format.
        resample_state: Optional state from previous call for streaming.

    Returns:
        Tuple of (converted audio bytes, new resample state).
    """
    if len(audio_bytes) == 0:
        return b"", resample_state

    # Step 1: Decode μ-law to PCM16 (still at 8kHz)
    pcm16_8khz = audioop.ulaw2lin(audio_bytes, 2)

    # Step 2: Resample from 8kHz to 16kHz
    pcm16_16khz, new_state = audioop.ratecv(
        pcm16_8khz,
        2,  # sample width (16-bit = 2 bytes)
        1,  # channels (mono)
        TELEPHONY_SAMPLE_RATE,  # input rate
        STT_SAMPLE_RATE,  # output rate
        resample_state,
    )

    return pcm16_16khz, new_state


def tts_output_to_telephony(
    audio_bytes: bytes,
    tts_sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
    resample_state: Optional[Tuple] = None,
) -> Tuple[bytes, Optional[Tuple]]:
    """Convert TTS output audio (variable rate PCM16) to telephony format (8kHz μ-law).

    Args:
        audio_bytes: Raw audio bytes in PCM16 format at tts_sample_rate.
        tts_sample_rate: Sample rate of the TTS output (default 24kHz).
        resample_state: Optional state from previous call for streaming.

    Returns:
        Tuple of (converted audio bytes, new resample state).
    """
    if len(audio_bytes) == 0:
        return b"", resample_state

    # Step 1: Resample from TTS rate to 8kHz
    pcm16_8khz, new_state = audioop.ratecv(
        audio_bytes,
        2,  # sample width (16-bit = 2 bytes)
        1,  # channels (mono)
        tts_sample_rate,  # input rate (e.g., 24kHz)
        TELEPHONY_SAMPLE_RATE,  # output rate (8kHz)
        resample_state,
    )

    # Step 2: Encode to μ-law
    ulaw_8khz = audioop.lin2ulaw(pcm16_8khz, 2)

    return ulaw_8khz, new_state


class StreamingLiveKitConverter:
    """Streaming audio converter for LiveKit that preserves state between chunks.

    Use this when processing audio in a tick-by-tick manner to avoid
    audio artifacts at chunk boundaries.

    Args:
        tts_sample_rate: Sample rate of TTS output (default 24kHz for Deepgram).
    """

    def __init__(self, tts_sample_rate: int = DEFAULT_TTS_SAMPLE_RATE):
        """Initialize the streaming converter.

        Args:
            tts_sample_rate: The sample rate of TTS audio output.
        """
        self._tts_sample_rate = tts_sample_rate
        self._input_resample_state: Optional[Tuple] = None
        self._output_resample_state: Optional[Tuple] = None

    def convert_input(self, telephony_audio: bytes) -> bytes:
        """Convert telephony audio to STT input format.

        Args:
            telephony_audio: Raw audio bytes in 8kHz μ-law format.

        Returns:
            Converted audio bytes in 16kHz PCM16 format.
        """
        result, self._input_resample_state = telephony_to_stt_input(
            telephony_audio, self._input_resample_state
        )
        return result

    def convert_output(self, tts_audio: bytes) -> bytes:
        """Convert TTS output to telephony format.

        Args:
            tts_audio: Raw audio bytes in PCM16 format at TTS sample rate.

        Returns:
            Converted audio bytes in 8kHz μ-law format.
        """
        result, self._output_resample_state = tts_output_to_telephony(
            tts_audio, self._tts_sample_rate, self._output_resample_state
        )
        return result

    def reset(self) -> None:
        """Reset the converter state.

        Call this when starting a new conversation or after an interruption.
        """
        self._input_resample_state = None
        self._output_resample_state = None

    @property
    def tts_sample_rate(self) -> int:
        """Get the TTS sample rate."""
        return self._tts_sample_rate

    @tts_sample_rate.setter
    def tts_sample_rate(self, value: int) -> None:
        """Set the TTS sample rate and reset output state."""
        if value != self._tts_sample_rate:
            self._tts_sample_rate = value
            self._output_resample_state = None  # Reset on rate change
