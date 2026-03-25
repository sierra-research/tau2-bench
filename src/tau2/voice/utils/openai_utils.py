"""OpenAI Realtime API audio format utilities.

This module provides conversion functions between tau2's AudioFormat
and OpenAI Realtime API format strings.

OpenAI Realtime API supports:
- g711_ulaw: 8kHz, 8-bit μ-law (telephony)
- g711_alaw: 8kHz, 8-bit A-law (telephony)
- pcm16: 24kHz, 16-bit signed PCM
"""

from tau2.config import DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE
from tau2.data_model.audio import (
    TELEPHONY_SAMPLE_RATE,
    AudioEncoding,
    AudioFormat,
)

# OpenAI Realtime API uses 24kHz for pcm16 format
OPENAI_PCM16_SAMPLE_RATE = DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE


def audio_format_to_openai_string(audio_format: AudioFormat) -> str:
    """Convert AudioFormat to OpenAI Realtime API format string.

    Args:
        audio_format: The AudioFormat to convert.

    Returns:
        OpenAI format string: "g711_ulaw", "g711_alaw", or "pcm16".

    Raises:
        ValueError: If the format is not supported by OpenAI Realtime API.

    Example:
        >>> fmt = AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000)
        >>> audio_format_to_openai_string(fmt)
        'g711_ulaw'
    """
    if audio_format.encoding == AudioEncoding.ULAW:
        if audio_format.sample_rate != TELEPHONY_SAMPLE_RATE:
            raise ValueError(
                f"OpenAI g711_ulaw requires {TELEPHONY_SAMPLE_RATE}Hz, "
                f"got {audio_format.sample_rate}Hz"
            )
        return "g711_ulaw"
    elif audio_format.encoding == AudioEncoding.ALAW:
        if audio_format.sample_rate != TELEPHONY_SAMPLE_RATE:
            raise ValueError(
                f"OpenAI g711_alaw requires {TELEPHONY_SAMPLE_RATE}Hz, "
                f"got {audio_format.sample_rate}Hz"
            )
        return "g711_alaw"
    elif audio_format.encoding == AudioEncoding.PCM_S16LE:
        if audio_format.sample_rate != OPENAI_PCM16_SAMPLE_RATE:
            raise ValueError(
                f"OpenAI pcm16 requires {OPENAI_PCM16_SAMPLE_RATE}Hz, "
                f"got {audio_format.sample_rate}Hz"
            )
        return "pcm16"
    else:
        raise ValueError(
            f"Unsupported encoding for OpenAI: {audio_format.encoding}. "
            "Supported: ULAW (g711_ulaw), ALAW (g711_alaw), PCM_S16LE (pcm16)"
        )


def openai_string_to_audio_format(format_string: str) -> AudioFormat:
    """Create AudioFormat from OpenAI Realtime API format string.

    Args:
        format_string: "g711_ulaw", "g711_alaw", or "pcm16".

    Returns:
        AudioFormat configured for the specified format.

    Raises:
        ValueError: If the format string is not recognized.

    Example:
        >>> fmt = openai_string_to_audio_format("g711_ulaw")
        >>> fmt.encoding
        <AudioEncoding.ULAW: 'ulaw'>
        >>> fmt.sample_rate
        8000
    """
    if format_string == "g711_ulaw":
        return AudioFormat(
            encoding=AudioEncoding.ULAW, sample_rate=TELEPHONY_SAMPLE_RATE
        )
    elif format_string == "g711_alaw":
        return AudioFormat(
            encoding=AudioEncoding.ALAW, sample_rate=TELEPHONY_SAMPLE_RATE
        )
    elif format_string == "pcm16":
        return AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=OPENAI_PCM16_SAMPLE_RATE
        )
    else:
        raise ValueError(
            f"Unknown OpenAI format: {format_string}. "
            "Supported: g711_ulaw, g711_alaw, pcm16"
        )
