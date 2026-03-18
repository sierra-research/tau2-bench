"""Deepgram Voice Agent API integration for audio native adapters.

NOTE: This is a CASCADED provider (STT → LLM → TTS), not a native audio provider.
Unlike OpenAI Realtime, Gemini Live, or Nova Sonic which process audio natively,
Deepgram Voice Agent API chains separate transcription, reasoning, and synthesis steps.

This means:
- Higher latency than native audio providers (~500-800ms E2E vs ~200-400ms)
- Explicit text transcript available at each step (good for debugging/logging)
- Flexibility to mix best-in-class STT, LLM, and TTS providers
- BYO (Bring Your Own) LLM/TTS support

Key features:
- Input: Configurable (linear16, mulaw, etc.) at various sample rates
- Output: Configurable audio format and voice
- STT: Nova-3 model (Deepgram's latest)
- LLM: Configurable (OpenAI, Anthropic, Deepgram, or custom)
- TTS: Aura-2 or external (ElevenLabs, OpenAI, etc.)
- Built-in VAD, barge-in detection, turn-taking

Reference: Deepgram Voice Agent API documentation
https://developers.deepgram.com/docs/voice-agent
"""

from tau2.voice.audio_native.deepgram.events import (
    BaseDeepgramEvent,
    DeepgramAgentAudioDoneEvent,
    DeepgramAgentStartedSpeakingEvent,
    DeepgramAgentThinkingEvent,
    DeepgramConversationTextEvent,
    DeepgramErrorEvent,
    DeepgramEvent,
    DeepgramFunctionCallBeginEvent,
    DeepgramFunctionCallingEvent,
    DeepgramFunctionCallRequestEvent,
    DeepgramSettingsAppliedEvent,
    DeepgramTimeoutEvent,
    DeepgramUnknownEvent,
    DeepgramUserStartedSpeakingEvent,
    DeepgramWelcomeEvent,
    parse_deepgram_event,
)
from tau2.voice.audio_native.deepgram.provider import (
    DeepgramVADConfig,
    DeepgramVoiceAgentProvider,
)

__all__ = [
    # Events
    "BaseDeepgramEvent",
    "DeepgramEvent",
    "DeepgramAgentAudioDoneEvent",
    "DeepgramAgentStartedSpeakingEvent",
    "DeepgramAgentThinkingEvent",
    "DeepgramConversationTextEvent",
    "DeepgramErrorEvent",
    "DeepgramFunctionCallBeginEvent",
    "DeepgramFunctionCallingEvent",
    "DeepgramFunctionCallRequestEvent",
    "DeepgramSettingsAppliedEvent",
    "DeepgramTimeoutEvent",
    "DeepgramUnknownEvent",
    "DeepgramUserStartedSpeakingEvent",
    "DeepgramWelcomeEvent",
    "parse_deepgram_event",
    # Provider
    "DeepgramVoiceAgentProvider",
    "DeepgramVADConfig",
]
