# Copyright Sierra
"""Voice transcription module."""

from .transcribe import (
    TranscriptionConfig,
    TranscriptionResult,
    transcribe_audio,
)

__all__ = [
    "TranscriptionConfig",
    "TranscriptionResult",
    "transcribe_audio",
]
