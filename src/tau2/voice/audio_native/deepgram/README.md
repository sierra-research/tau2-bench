# Deepgram Voice Agent API - Discrete-Time Adapter

This module provides a **tick-based interface** for Deepgram's Voice Agent API, designed for discrete-time full-duplex voice simulation.

## ⚠️ Important: Cascaded Architecture

```
================================================================================
  CASCADED PROVIDER - Not Native Audio
================================================================================

Unlike OpenAI Realtime, Gemini Live, Nova Sonic, or xAI which process audio
NATIVELY through end-to-end models, Deepgram Voice Agent is a CASCADED system:

                STT → LLM → TTS

This means:
✓ Higher latency (~500-800ms E2E vs ~200-400ms for native audio)
✓ Explicit text transcript at each step (great for debugging)
✓ Flexibility to mix best-in-class STT, LLM, and TTS providers
✓ BYO (Bring Your Own) LLM and TTS support

================================================================================
```

## Overview

Deepgram Voice Agent API orchestrates three separate steps:
1. **Listen (STT)**: Nova-3 speech-to-text (Deepgram)
2. **Think (LLM)**: Configurable - OpenAI, Anthropic, or Deepgram
3. **Speak (TTS)**: Aura-2 or external (ElevenLabs, OpenAI)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FullDuplexOrchestrator                       │
│            (coordinates agent ↔ user, handles tools)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               DiscreteTimeAudioNativeAgent                      │
│         (provider-agnostic, selects adapter by config)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               DiscreteTimeDeepgramAdapter                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • Audio format conversion (telephony ↔ PCM16)          │   │
│   │  • Audio capping & buffering                            │   │
│   │  • Proportional transcript distribution                 │   │
│   │  • Interruption handling via VAD events                 │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                DeepgramVoiceAgentProvider                       │
│                   (async WebSocket client)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Deepgram Voice Agent API                       │
│                  (WebSocket endpoint)                           │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│   │  Listen  │ -> │  Think   │ -> │  Speak   │                 │
│   │ (Nova-3) │    │  (LLM)   │    │ (Aura-2) │                 │
│   └──────────┘    └──────────┘    └──────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## Comparison with Native Audio Providers

| Feature | Native (OpenAI, Gemini, Nova, xAI) | Deepgram (Cascaded) |
|---------|-------------------------------------|---------------------|
| **Architecture** | End-to-end audio model | STT → LLM → TTS |
| **Latency** | ~200-400ms | ~500-800ms |
| **Transcript** | Inferred from audio | Explicit at each step |
| **LLM Flexibility** | Provider's model only | BYO (OpenAI, Anthropic, etc.) |
| **TTS Flexibility** | Provider's voices only | BYO (ElevenLabs, OpenAI, etc.) |
| **Tool Calling** | Varies by provider | ✅ Works |

## Components

| File | Description |
|------|-------------|
| `events.py` | Pydantic models for Deepgram Voice Agent events |
| `provider.py` | `DeepgramVoiceAgentProvider` - async WebSocket client |
| `discrete_time_adapter.py` | `DiscreteTimeDeepgramAdapter` - implements `DiscreteTimeAdapter` |
| `audio_utils.py` | Audio format conversion utilities |

## Usage

### CLI

```bash
# Run with Deepgram Voice Agent provider
tau2 run --domain retail --audio-native --audio-native-provider deepgram \
    --num-tasks 1 --seed 42 --tick-duration 0.2 --max-steps-seconds 120 \
    --speech-complexity control --verbose-logs --save-to my_simulation
```

### Programmatic

```python
from tau2.voice.audio_native.deepgram import (
    DeepgramVoiceAgentProvider,
    DeepgramVADConfig,
)
from tau2.voice.audio_native.deepgram.discrete_time_adapter import (
    DiscreteTimeDeepgramAdapter,
)

adapter = DiscreteTimeDeepgramAdapter(
    tick_duration_ms=200,
    send_audio_instant=True,
    llm_provider="openai",        # BYO LLM
    llm_model="gpt-4o-mini",
    tts_model="aura-2-thalia-en", # Deepgram TTS with voice
)

adapter.connect(
    system_prompt="You are a helpful assistant.",
    tools=my_tools,  # Tool calling works!
    vad_config=DeepgramVADConfig(endpointing_ms=500),
)

for tick in range(max_ticks):
    result = adapter.run_tick(user_audio_chunk, tick_number=tick)
    # result.get_played_agent_audio() - agent audio (telephony format)
    # result.proportional_transcript - text for this tick
    # result.tool_calls - function calls

adapter.disconnect()
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DEEPGRAM_API_KEY` | Deepgram API key |

### STT Models (Listen)

| Model | Description |
|-------|-------------|
| `nova-3` | Deepgram's latest STT model (default) |

### LLM Providers (Think)

| Provider | Models |
|----------|--------|
| `openai` | `gpt-4o`, `gpt-4o-mini`, etc. |
| `anthropic` | `claude-3-5-sonnet`, etc. |
| `deepgram` | Deepgram's native LLM |

### TTS Models (Speak)

Deepgram TTS (Aura-2) combines model and voice in the model name:

| Model | Description |
|-------|-------------|
| `aura-2-thalia-en` | Aura-2, Thalia voice, English |
| `aura-2-helios-en` | Aura-2, Helios voice, English |
| `aura-2-luna-en` | Aura-2, Luna voice, English |

External TTS providers (ElevenLabs, OpenAI) can also be configured.

### Audio Formats

| Direction | Format | Sample Rate |
|-----------|--------|-------------|
| Input | linear16 (PCM16) | 16kHz |
| Output | linear16 (PCM16) | 16kHz |

The adapter handles conversion to/from telephony format (8kHz μ-law).

### VAD Configuration

```python
class DeepgramVADConfig(BaseModel):
    mode: DeepgramVADMode = DeepgramVADMode.SERVER_VAD  # Server handles VAD
    endpointing_ms: int = 500  # Silence duration to trigger end of turn
```

## Event Types

| Event | Description |
|-------|-------------|
| `Welcome` | Initial connection acknowledgment |
| `SettingsApplied` | Configuration confirmed |
| `UserStartedSpeaking` | VAD detected user speech |
| `ConversationText` | Transcription (user) or response text (agent) |
| `AgentThinking` | Agent is processing |
| `AgentStartedSpeaking` | Agent audio output starting |
| `AgentAudioDone` | Agent finished speaking |
| `FunctionCallRequest` | Agent wants to call a function |
| `Audio` | Binary audio chunk |

## Function Calling

Deepgram Voice Agent supports function/tool calling:

1. Agent sends `FunctionCallRequest` with function name, ID, and arguments
2. Execute the function and get result
3. Send `FunctionCallResponse` with call ID and result content

```python
# Deepgram FunctionCallResponse format
{
    "type": "FunctionCallResponse",
    "id": "call_abc123",
    "name": "get_order_status",
    "content": "{\"status\": \"shipped\"}"
}
```

## Known Issues & Caveats

1. **Higher Latency**: As a cascaded system, expect ~500-800ms E2E latency vs ~200-400ms for native audio providers.

2. **Audio Format Conversion**: Requires conversion between telephony (8kHz μ-law) and Deepgram's format (16kHz PCM16).

3. **Transcript Instructions**: The adapter automatically appends instructions to the system prompt asking users to spell out names/IDs to reduce transcription errors.

## Implementation Status

- [x] Events (`events.py`)
- [x] Provider (`provider.py`)
- [x] Discrete-time adapter (`discrete_time_adapter.py`)
- [x] Audio format conversion (`audio_utils.py`)
- [x] Server-side VAD
- [x] Interruption handling (barge-in)
- [x] Tool/function calling
- [x] BYO LLM configuration
- [x] BYO TTS configuration
- [x] Proportional transcript distribution

## Reference

- [Deepgram Voice Agent API Documentation](https://developers.deepgram.com/docs/voice-agent)
