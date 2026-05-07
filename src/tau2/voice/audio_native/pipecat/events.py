"""Event types emitted by the Pipecat-based cascaded provider.

This is a thin wrapper that flattens Pipecat's frame-based event stream into
a small set of named events that the discrete-time adapter can consume in
the same way it consumes events from other providers.

The provider does NOT expose Pipecat ``Frame`` objects directly to the
adapter; everything funnels through ``PipecatEvent`` so the rest of the
adapter code stays Pipecat-agnostic.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from tau2.data_model.message import ToolCall


class PipecatEventType(str, Enum):
    """Event types emitted by ``PipecatVoiceProvider``."""

    # VAD / speech events
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"

    # Transcription events
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"

    # LLM events
    LLM_STARTED = "llm_started"
    LLM_TOKEN = "llm_token"
    LLM_COMPLETED = "llm_completed"
    TOOL_CALL = "tool_call"

    # TTS events
    TTS_STARTED = "tts_started"
    TTS_AUDIO = "tts_audio"
    TTS_COMPLETED = "tts_completed"

    # Control events
    INTERRUPTED = "interrupted"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class PipecatEvent:
    """Event emitted by the Pipecat-based cascaded provider.

    Mirrors the shape of ``CascadedEvent`` from the LiveKit provider so the
    discrete-time adapter can stay close to that pattern.
    """

    type: PipecatEventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def transcript(self) -> Optional[str]:
        return self.data.get("transcript")

    @property
    def audio(self) -> Optional[bytes]:
        return self.data.get("audio")

    @property
    def sample_rate(self) -> Optional[int]:
        return self.data.get("sample_rate")

    @property
    def tool_call(self) -> Optional[ToolCall]:
        return self.data.get("tool_call")

    @property
    def text(self) -> Optional[str]:
        return self.data.get("text")
