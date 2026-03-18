# Copyright Sierra
"""Voice synthesis module."""

from tau2.data_model.voice_personas import ALL_PERSONA_NAMES

from .audio_effects import BackgroundNoiseGenerator

__all__ = [
    "ALL_PERSONA_NAMES",
    "BackgroundNoiseGenerator",
]
