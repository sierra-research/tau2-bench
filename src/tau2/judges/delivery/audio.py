# Copyright Sierra
"""Convert a message's stored audio into WAV base64 for the multimodal judge.

The agent's synthesized speech is stored on the message as ``audio_content``
(base64) with ``audio_format`` metadata — typically 8kHz μ-law telephony. Gemini
takes an inline ``input_audio`` part of WAV bytes, so we decode, convert any
companded/PCM format to 16-bit PCM (reusing ``convert_to_pcm16``), and wrap it in
an in-memory WAV container (mirroring ``save_wav_file`` but without touching disk).
"""

import io
import wave
from typing import TYPE_CHECKING, Optional

from tau2.data_model.audio import AudioData, audio_bytes_to_string
from tau2.voice.utils.audio_preprocessing import convert_to_pcm16

if TYPE_CHECKING:
    from tau2.data_model.message import Message


def audio_data_to_wav_b64(audio: AudioData) -> str:
    """Encode AudioData as a base64 WAV string (always 16-bit linear PCM)."""
    if audio.format.encoding.is_companded:
        audio = convert_to_pcm16(audio)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(audio.format.channels)
        wav_file.setsampwidth(audio.format.sample_width)
        wav_file.setframerate(audio.format.sample_rate)
        wav_file.setcomptype("NONE", "not compressed")
        wav_file.writeframes(audio.data)
    return audio_bytes_to_string(buffer.getvalue())


def message_audio_to_wav_b64(message: "Message") -> Optional[str]:
    """Return the message's audio as a base64 WAV string, or None if it has none."""
    audio_bytes = message.get_audio_bytes()
    if not audio_bytes or message.audio_format is None:
        return None
    audio = AudioData(data=audio_bytes, format=message.audio_format)
    return audio_data_to_wav_b64(audio)
