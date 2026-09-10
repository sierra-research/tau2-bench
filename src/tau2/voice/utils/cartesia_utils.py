"""Cartesia customer TTS, using the same PCM format as the effects pipeline."""

import os
import re

import httpx

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import CartesiaTTSConfig


def tts_cartesia(text: str, config: CartesiaTTSConfig) -> AudioData:
    """Synthesize 16 kHz mono PCM without sending ElevenLabs-only vocal tags."""
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        raise ValueError("CARTESIA_API_KEY is not set")
    if not config.voice_id:
        raise ValueError("Cartesia voice_id is required")
    if re.search(r"\[(?:cough|sneeze|sniffle)\]", text, re.IGNORECASE):
        raise ValueError(
            "Cartesia synthesis does not support ElevenLabs vocal-tic tags"
        )
    transcript = re.sub(
        r"\[pause\]", '<break time="500ms"/>', text, flags=re.IGNORECASE
    )
    response = httpx.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Cartesia-Version": config.api_version,
        },
        json={
            "model_id": config.model_id,
            "voice": config.voice_id,
            "transcript": transcript,
            "locale": config.locale,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": config.output_audio_format.sample_rate,
            },
        },
        timeout=60.0,
    )
    response.raise_for_status()
    audio = response.content
    if not audio or len(audio) % 2:
        raise ValueError("Cartesia returned empty or incomplete PCM16 audio")
    return AudioData(data=audio, format=config.output_audio_format)
