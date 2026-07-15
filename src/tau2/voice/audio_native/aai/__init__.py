"""aai audio-native integration.

aai (the @alexkroman1/aai voice-agent framework) provides speech-to-speech
capabilities through a local WebSocket-based voice-agent host — distinct from
the separate `assemblyai` provider. The host communicates via bidirectional WebSocket with
speech transcription, agent responses, and tool/function calling support.

Key features:
- Input: PCM16 16kHz audio
- Output: PCM16 24kHz audio
- Server-side VAD and turn-taking
- Tool/function calling support
- Local execution for privacy and latency

Reference: AssemblyAI documentation
https://www.assemblyai.com/docs/
"""

from tau2.voice.audio_native.aai.discrete_time_adapter import DiscreteTimeAAIAdapter
from tau2.voice.audio_native.aai.events import (
    AAIAudioChunkEvent,
    AAIErrorEvent,
    AAIToolCallEvent,
    AAIUserTranscriptEvent,
    parse_aai_event,
)
from tau2.voice.audio_native.aai.provider import AAIVADConfig, AAIVoiceAgentProvider

__all__ = [
    "AAIAudioChunkEvent",
    "AAIErrorEvent",
    "AAIToolCallEvent",
    "AAIUserTranscriptEvent",
    "AAIVADConfig",
    "AAIVoiceAgentProvider",
    "DiscreteTimeAAIAdapter",
    "parse_aai_event",
]
