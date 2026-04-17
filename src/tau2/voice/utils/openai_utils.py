"""OpenAI Realtime API audio format utilities.

This module provides conversion functions between tau2's AudioFormat
and OpenAI Realtime API format objects.

OpenAI GA API audio formats:
- audio/pcmu: G.711 μ-law (telephony)
- audio/pcma: G.711 A-law (telephony)
- audio/pcm: 24kHz, 16-bit signed PCM (requires rate: 24000)
"""

from tau2.config import DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE
from tau2.data_model.audio import (
    TELEPHONY_SAMPLE_RATE,
    AudioEncoding,
    AudioFormat,
)

OPENAI_PCM16_SAMPLE_RATE = DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE


def audio_format_to_openai(audio_format: AudioFormat) -> dict:
    """Convert AudioFormat to OpenAI GA API format object.

    Args:
        audio_format: The AudioFormat to convert.

    Returns:
        Dict with 'type' and optionally 'rate' for the GA API.

    Raises:
        ValueError: If the format is not supported by OpenAI Realtime API.
    """
    if audio_format.encoding == AudioEncoding.ULAW:
        if audio_format.sample_rate != TELEPHONY_SAMPLE_RATE:
            raise ValueError(
                f"OpenAI audio/pcmu requires {TELEPHONY_SAMPLE_RATE}Hz, "
                f"got {audio_format.sample_rate}Hz"
            )
        return {"type": "audio/pcmu"}
    elif audio_format.encoding == AudioEncoding.ALAW:
        if audio_format.sample_rate != TELEPHONY_SAMPLE_RATE:
            raise ValueError(
                f"OpenAI audio/pcma requires {TELEPHONY_SAMPLE_RATE}Hz, "
                f"got {audio_format.sample_rate}Hz"
            )
        return {"type": "audio/pcma"}
    elif audio_format.encoding == AudioEncoding.PCM_S16LE:
        if audio_format.sample_rate != OPENAI_PCM16_SAMPLE_RATE:
            raise ValueError(
                f"OpenAI audio/pcm requires {OPENAI_PCM16_SAMPLE_RATE}Hz, "
                f"got {audio_format.sample_rate}Hz"
            )
        return {"type": "audio/pcm", "rate": 24000}
    else:
        raise ValueError(
            f"Unsupported encoding for OpenAI: {audio_format.encoding}. "
            "Supported: ULAW (audio/pcmu), ALAW (audio/pcma), PCM_S16LE (audio/pcm)"
        )


def openai_format_to_audio_format(fmt: dict) -> AudioFormat:
    """Create AudioFormat from OpenAI GA API format object.

    Args:
        fmt: Dict with 'type' key (e.g. {"type": "audio/pcmu"}).

    Returns:
        AudioFormat configured for the specified format.

    Raises:
        ValueError: If the format type is not recognized.
    """
    fmt_type = fmt.get("type", "")
    if fmt_type == "audio/pcmu":
        return AudioFormat(
            encoding=AudioEncoding.ULAW, sample_rate=TELEPHONY_SAMPLE_RATE
        )
    elif fmt_type == "audio/pcma":
        return AudioFormat(
            encoding=AudioEncoding.ALAW, sample_rate=TELEPHONY_SAMPLE_RATE
        )
    elif fmt_type == "audio/pcm":
        return AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=OPENAI_PCM16_SAMPLE_RATE
        )
    else:
        raise ValueError(
            f"Unknown OpenAI format: {fmt_type}. "
            "Supported: audio/pcmu, audio/pcma, audio/pcm"
        )
