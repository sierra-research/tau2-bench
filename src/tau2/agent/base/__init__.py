"""
Base agent components.

This module exports the foundational building blocks for creating
conversation participants with support for half-duplex, full-duplex,
streaming, and voice communication.
"""

import warnings

# LLM configuration mixin
from tau2.agent.base.llm_config import LLMConfigMixin

# Protocol base classes
from tau2.agent.base.participant import (
    FullDuplexParticipant,
    HalfDuplexParticipant,
    VoiceParticipantMixin,
)

# Streaming components
from tau2.agent.base.streaming import (
    AudioChunkingMixin,
    BasicActionType,
    StreamingMixin,
    StreamingState,
    basic_turn_taking_policy,
    merge_homogeneous_chunks,
)

# Streaming utilities for audio script gold processing
from tau2.agent.base.streaming_utils import (
    extract_active_chunk_ids,
    extract_all_chunk_ids,
    extract_chunks_with_text,
    extract_gold_text,
    extract_message_uuid,
    format_transcript_comparison,
    merge_audio_script_gold,
)

# Voice components
from tau2.agent.base.voice import VoiceMixin, VoiceState

# =============================================================================
# DEPRECATION ALIASES
# =============================================================================


def _deprecated_alias(old_name: str, new_name: str, new_class):
    """Create a deprecated alias that warns on use."""

    def __getattr__(name):
        if name == old_name:
            warnings.warn(
                f"{old_name} is deprecated, use {new_name} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return new_class
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return __getattr__


# Deprecated aliases for backward compatibility
BaseConversationParticipant = HalfDuplexParticipant
BaseStreamingParticipant = FullDuplexParticipant
BaseVoiceParticipant = VoiceParticipantMixin


__all__ = [
    # Protocol base classes
    "HalfDuplexParticipant",
    "FullDuplexParticipant",
    "VoiceParticipantMixin",
    # LLM configuration
    "LLMConfigMixin",
    # Streaming components
    "StreamingMixin",
    "StreamingState",
    "AudioChunkingMixin",
    "BasicActionType",
    "basic_turn_taking_policy",
    "merge_homogeneous_chunks",
    # Streaming utilities
    "extract_message_uuid",
    "extract_active_chunk_ids",
    "extract_all_chunk_ids",
    "merge_audio_script_gold",
    "extract_gold_text",
    "extract_chunks_with_text",
    "format_transcript_comparison",
    # Voice components
    "VoiceMixin",
    "VoiceState",
    # Deprecated aliases (kept for backward compatibility)
    "BaseConversationParticipant",
    "BaseStreamingParticipant",
    "BaseVoiceParticipant",
]
