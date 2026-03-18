import os
import re
from copy import deepcopy
from typing import Tuple

from elevenlabs import ElevenLabs
from loguru import logger

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import ElevenLabsTTSConfig


def make_elevenlabs_output_format(codec: str, sample_rate: int, bitrate: int) -> str:
    """
    Make an ElevenLabs output format string from codec, sample rate, and bitrate

    Args:
        codec: The codec to use
        sample_rate: The sample rate to use
        bitrate: The bitrate to use

    Returns:
        The ElevenLabs output format string
    """
    return f"{codec}_{sample_rate}_{bitrate}"


def parse_elevenlabs_output_format(output_format: str) -> Tuple[str, int, int]:
    """
    Parse an ElevenLabs output format string into codec, sample rate, and bitrate

    Args:
        output_format: The ElevenLabs output format string

    Returns:
        Tuple of (codec, sample rate, bitrate)
    """
    pat = re.compile(r"^(\w+)_(\d+)_(\d+)$")
    match = pat.match(output_format)
    if not match:
        raise ValueError(f"Invalid ElevenLabs output format: {output_format}")
    return match.group(1), int(match.group(2)), int(match.group(3))


AUDIO_TAG_PATTERN = re.compile(r"\[(cough|sneeze|sniffle)\]")
PAUSE_TAG_PATTERN = re.compile(r"\[pause\]", re.IGNORECASE)


def tts_elevenlabs(
    text: str,
    config: ElevenLabsTTSConfig,
) -> AudioData:
    """Text to speech using ElevenLabs API

    Args:
        text: The text to synthesize
        config: The configuration to use
    Returns:
        AudioData with the specified output format
    """
    api_key = config.api_key or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not found in config or environment")

    is_v3 = "v3" in config.model_id.lower()

    # Replace [pause] with ellipsis for non-v3 models (v3 supports [pause] natively)
    if not is_v3 and PAUSE_TAG_PATTERN.search(text):
        text = PAUSE_TAG_PATTERN.sub("...", text)

    # Warn if audio tags are present but model isn't v3 (tags only work with v3)
    if AUDIO_TAG_PATTERN.search(text) and not is_v3:
        logger.warning(
            f"Audio tags detected in text but model is {config.model_id}. "
            "Audio tags like [cough], [sneeze], [sniffle] only work with v3 models."
        )

    client = ElevenLabs(api_key=api_key)

    voice_id = config.voice_id

    # Log before making API call to help diagnose timeouts
    text_preview = text[:50] + "..." if len(text) > 50 else text
    logger.debug(
        f"ElevenLabs TTS: calling API for text '{text_preview}' "
        f"(voice_id={voice_id}, model={config.model_id})"
    )

    try:
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            voice_settings=config.voice_settings,
            model_id=config.model_id,
            output_format=config.output_format_name,
            seed=config.seed,
        )

        audio_bytes = b"".join(audio)
    except Exception as e:
        # Log at debug level - the retry wrapper (tts_retry) will log with attempt info
        logger.debug(
            f"ElevenLabs TTS API call failed: {type(e).__name__}: {e} "
            f"(text='{text_preview}', voice_id={voice_id}, model={config.model_id})"
        )
        raise

    logger.debug(f"ElevenLabs TTS: received {len(audio_bytes)} bytes of audio")

    # Validate that we received audio data
    if len(audio_bytes) == 0:
        logger.error(f"ElevenLabs TTS returned empty audio for text: '{text}'")
        raise ValueError(f"ElevenLabs TTS returned empty audio for text: '{text}'")

    audio_data = AudioData(
        data=audio_bytes,
        format=deepcopy(config.output_audio_format),
    )

    return audio_data
