# xAI Grok Voice Agent API - Discrete-Time Adapter

This module provides a **tick-based interface** for xAI's Grok Voice Agent API, designed for discrete-time full-duplex voice simulation.

## Overview

xAI's Grok Voice Agent API is a WebSocket-based real-time voice API that is highly compatible with OpenAI's Realtime API protocol. The key advantage is **native G.711 μ-law support**, eliminating audio format conversion overhead for telephony applications.

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
│                  DiscreteTimeXAIAdapter                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • NO audio conversion needed (native G.711 μ-law!)     │   │
│   │  • Audio capping & buffering                            │   │
│   │  • Proportional transcript distribution                 │   │
│   │  • Interruption handling via VAD events                 │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    XAIRealtimeProvider                          │
│                   (async WebSocket client)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  xAI Grok Voice Agent API                       │
│                    (WebSocket server)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Key Advantages

| Feature | xAI | OpenAI | Gemini |
|---------|-----|--------|--------|
| Native G.711 μ-law | ✅ Yes | ✅ Yes | ❌ No (requires conversion) |
| Audio conversion overhead | None | None | PCM16 ↔ μ-law conversion |
| Protocol compatibility | OpenAI-like | - | Different |
| Voice options | 5 voices | 1 voice | Multiple |

### No Audio Conversion Required

xAI natively supports G.711 μ-law at 8kHz, which is the standard telephony format. This means:
- Audio passes through directly without conversion
- Lower latency
- No quality loss from resampling

## Components

| File | Description |
|------|-------------|
| `events.py` | Pydantic models for xAI Realtime events |
| `provider.py` | `XAIRealtimeProvider` - async WebSocket client |
| `discrete_time_adapter.py` | `DiscreteTimeXAIAdapter` - implements `DiscreteTimeAdapter` |

## Usage

### CLI

```bash
# Run with xAI Grok Voice Agent provider
tau2 run --domain retail --audio-native --audio-native-provider xai \
    --num-tasks 1 --seed 42 --tick-duration 0.2 --max-steps-seconds 120 \
    --speech-complexity control --verbose-logs --save-to my_simulation
```

### Programmatic

```python
from tau2.voice.audio_native.xai import (
    DiscreteTimeXAIAdapter,
    XAIVADConfig,
)

adapter = DiscreteTimeXAIAdapter(
    tick_duration_ms=200,
    send_audio_instant=True,
    voice="Ara",  # Options: Ara, Rex, Sal, Eve, Leo
)

adapter.connect(
    system_prompt="You are a helpful assistant.",
    tools=tools,
    vad_config=XAIVADConfig(),
)

for tick in range(max_ticks):
    result = adapter.run_tick(user_audio_chunk, tick_number=tick)
    # result.get_played_agent_audio() - agent audio (8kHz μ-law, no conversion!)
    # result.proportional_transcript - text for this tick
    # result.tool_calls - function calls

adapter.disconnect()
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `XAI_API_KEY` | xAI API key for authentication |

### Voice Options

| Voice | Description |
|-------|-------------|
| `Ara` | Default voice |
| `Rex` | Alternative voice |
| `Sal` | Alternative voice |
| `Eve` | Alternative voice |
| `Leo` | Alternative voice |

### Audio Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `audio/pcmu` | G.711 μ-law at 8kHz | Telephony (default, optimal) |
| `audio/pcma` | G.711 A-law at 8kHz | International telephony |
| `audio/pcm` | PCM Linear16 | Custom sample rates |

### VAD Configuration

```python
class XAIVADConfig(BaseModel):
    mode: XAIVADMode = XAIVADMode.SERVER_VAD  # or MANUAL
```

| Mode | Description |
|------|-------------|
| `SERVER_VAD` | Server handles voice activity detection automatically |
| `MANUAL` | Client controls turns explicitly |

## Key Differences from OpenAI

While xAI's API is highly compatible with OpenAI's Realtime API, there are some differences:

### Event Names

| Purpose | OpenAI | xAI |
|---------|--------|-----|
| Session start | `session.created` | `conversation.created` |
| Audio delta | `response.audio.delta` | `response.output_audio.delta` |
| Audio done | `response.audio.done` | `response.output_audio.done` |
| Transcript delta | `response.audio_transcript.delta` | `response.output_audio_transcript.delta` |

### Audio Configuration

```python
# xAI audio config structure
{
    "audio": {
        "input": {"format": {"type": "audio/pcmu"}},
        "output": {"format": {"type": "audio/pcmu"}}
    }
}

# OpenAI audio config
{
    "input_audio_format": "g711_ulaw",
    "output_audio_format": "g711_ulaw"
}
```

## Implementation Status

- [x] Events (`events.py`)
- [x] Provider (`provider.py`)
- [x] Discrete-time adapter (`discrete_time_adapter.py`)
- [x] Native G.711 μ-law support (no conversion needed!)
- [x] Tool call parsing
- [x] Input transcription
- [x] Output transcription
- [x] Interruption handling (via VAD events)
- [x] Multiple voice options

## Reference

- [xAI Voice Agent Documentation](https://docs.x.ai/docs/guides/voice/agent)
