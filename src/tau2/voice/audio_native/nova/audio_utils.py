"""Audio format conversion utilities for Amazon Nova Sonic.

Nova Sonic uses different audio formats than the telephony standard:
- Input:  16kHz LPCM mono (vs 8kHz μ-law for telephony)
- Output: 24kHz LPCM mono (vs 8kHz μ-law for telephony)

This module provides conversion functions to bridge between formats.
Note: Output is 24kHz (different from input 16kHz).
"""

import audioop
from typing import Optional, Tuple

from tau2.config import (
    DEFAULT_NOVA_INPUT_SAMPLE_RATE,
    DEFAULT_NOVA_OUTPUT_SAMPLE_RATE,
    DEFAULT_TELEPHONY_RATE,
)

# Telephony format: 8kHz μ-law, 1 byte per sample
TELEPHONY_SAMPLE_RATE = DEFAULT_TELEPHONY_RATE
TELEPHONY_BYTES_PER_SECOND = DEFAULT_TELEPHONY_RATE  # 1 byte/sample for μ-law

# Nova Sonic audio formats (from config, PCM16 mono, 2 bytes per sample)
NOVA_INPUT_SAMPLE_RATE = DEFAULT_NOVA_INPUT_SAMPLE_RATE
NOVA_OUTPUT_SAMPLE_RATE = DEFAULT_NOVA_OUTPUT_SAMPLE_RATE


def telephony_to_nova_input(
    audio_bytes: bytes,
    resample_state: Optional[Tuple] = None,
) -> Tuple[bytes, Optional[Tuple]]:
    """Convert telephony audio (8kHz μ-law) to Nova input format (16kHz PCM16).

    Args:
        audio_bytes: Raw audio bytes in 8kHz μ-law format.
        resample_state: Optional state from previous call for streaming.

    Returns:
        Tuple of (converted audio bytes, new resample state).
        The state should be passed to the next call for streaming.
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
        NOVA_INPUT_SAMPLE_RATE,  # output rate
        resample_state,
    )

    return pcm16_16khz, new_state


def nova_output_to_telephony(
    audio_bytes: bytes,
    resample_state: Optional[Tuple] = None,
) -> Tuple[bytes, Optional[Tuple]]:
    """Convert Nova output audio (24kHz PCM16) to telephony format (8kHz μ-law).

    Args:
        audio_bytes: Raw audio bytes in 24kHz PCM16 format.
        resample_state: Optional state from previous call for streaming.

    Returns:
        Tuple of (converted audio bytes, new resample state).
        The state should be passed to the next call for streaming.
    """
    if len(audio_bytes) == 0:
        return b"", resample_state

    # Step 1: Resample from 24kHz to 8kHz
    pcm16_8khz, new_state = audioop.ratecv(
        audio_bytes,
        2,  # sample width (16-bit = 2 bytes)
        1,  # channels (mono)
        NOVA_OUTPUT_SAMPLE_RATE,  # input rate (24kHz)
        TELEPHONY_SAMPLE_RATE,  # output rate (8kHz)
        resample_state,
    )

    # Step 2: Encode to μ-law
    ulaw_8khz = audioop.lin2ulaw(pcm16_8khz, 2)

    return ulaw_8khz, new_state


class StreamingNovaConverter:
    """Streaming audio converter for Nova Sonic that preserves state between chunks.

    Use this when processing audio in a tick-by-tick manner to avoid
    audio artifacts at chunk boundaries.
    """

    def __init__(self):
        """Initialize the streaming converter."""
        self._input_resample_state: Optional[Tuple] = None
        self._output_resample_state: Optional[Tuple] = None

    def convert_input(self, telephony_audio: bytes) -> bytes:
        """Convert telephony audio to Nova input format.

        Args:
            telephony_audio: Raw audio bytes in 8kHz μ-law format.

        Returns:
            Converted audio bytes in 16kHz PCM16 format.
        """
        result, self._input_resample_state = telephony_to_nova_input(
            telephony_audio, self._input_resample_state
        )
        return result

    def convert_output(self, nova_audio: bytes) -> bytes:
        """Convert Nova output to telephony format.

        Args:
            nova_audio: Raw audio bytes in 24kHz PCM16 format.

        Returns:
            Converted audio bytes in 8kHz μ-law format.
        """
        result, self._output_resample_state = nova_output_to_telephony(
            nova_audio, self._output_resample_state
        )
        return result

    def reset(self) -> None:
        """Reset the converter state.

        Call this when starting a new conversation or after an interruption.
        """
        self._input_resample_state = None
        self._output_resample_state = None


def calculate_telephony_bytes_per_tick(tick_duration_ms: int) -> int:
    """Calculate the expected audio bytes per tick for telephony format.

    Args:
        tick_duration_ms: Duration of each tick in milliseconds.

    Returns:
        Expected number of bytes per tick (8kHz μ-law = 8000 bytes/sec).
    """
    return int(TELEPHONY_BYTES_PER_SECOND * tick_duration_ms / 1000)
