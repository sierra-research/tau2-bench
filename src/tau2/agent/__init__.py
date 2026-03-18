import warnings

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base.participant import FullDuplexParticipant, HalfDuplexParticipant
from tau2.agent.base.streaming import StreamingMixin, StreamingState
from tau2.agent.base.streaming_utils import (
    extract_active_chunk_ids,
    extract_all_chunk_ids,
    extract_chunks_with_text,
    extract_gold_text,
    extract_message_uuid,
    format_transcript_comparison,
    merge_audio_script_gold,
)
from tau2.agent.base_agent import (
    FullDuplexAgent,
    FullDuplexVoiceAgent,
    HalfDuplexAgent,
    HalfDuplexVoiceAgent,
    ValidAgentInputMessage,
)
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMGTAgent, LLMSoloAgent
from tau2.voice.audio_native.openai import OpenAIRealtimeProvider
from tau2.voice.audio_native.openai.provider import OpenAIVADMode

# =============================================================================
# DEPRECATION ALIASES
# =============================================================================
# These aliases maintain backward compatibility with code using old names.
# They will emit DeprecationWarning when used.


def __getattr__(name: str):
    """Module-level __getattr__ for deprecation warnings."""
    deprecated_aliases = {
        "BaseConversationParticipant": ("HalfDuplexParticipant", HalfDuplexParticipant),
        "BaseStreamingParticipant": ("FullDuplexParticipant", FullDuplexParticipant),
        "BaseAgent": ("HalfDuplexAgent", HalfDuplexAgent),
        "LocalAgent": ("HalfDuplexAgent", HalfDuplexAgent),
        "BaseStreamingAgent": ("FullDuplexAgent", FullDuplexAgent),
        "BaseVoiceAgent": ("HalfDuplexVoiceAgent", HalfDuplexVoiceAgent),
    }

    if name in deprecated_aliases:
        new_name, new_class = deprecated_aliases[name]
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_class

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Direct aliases for static analysis tools (these don't trigger warnings on import)
BaseConversationParticipant = HalfDuplexParticipant
BaseStreamingParticipant = FullDuplexParticipant
BaseAgent = HalfDuplexAgent
LocalAgent = HalfDuplexAgent
BaseStreamingAgent = FullDuplexAgent
BaseVoiceAgent = HalfDuplexVoiceAgent


__all__ = [
    # Generic base classes
    "HalfDuplexParticipant",
    "FullDuplexParticipant",
    # LLM configuration mixin
    "LLMConfigMixin",
    # Generic streaming mixins
    "StreamingMixin",
    "StreamingState",
    # Streaming utilities
    "extract_message_uuid",
    "extract_active_chunk_ids",
    "extract_all_chunk_ids",
    "merge_audio_script_gold",
    "extract_gold_text",
    "extract_chunks_with_text",
    "format_transcript_comparison",
    # Agent-specific base classes
    "HalfDuplexAgent",
    "FullDuplexAgent",
    "HalfDuplexVoiceAgent",
    "FullDuplexVoiceAgent",
    "ValidAgentInputMessage",
    # LLM Agents
    "LLMAgent",
    "LLMAgentState",
    "LLMGTAgent",
    "LLMSoloAgent",
    # Audio Native
    "OpenAIRealtimeProvider",
    "OpenAIVADMode",
    # Deprecated aliases (kept for backward compatibility)
    "BaseConversationParticipant",
    "BaseStreamingParticipant",
    "BaseAgent",
    "LocalAgent",
    "BaseStreamingAgent",
    "BaseVoiceAgent",
]
