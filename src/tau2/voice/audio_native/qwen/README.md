# Qwen Omni Flash Realtime API - Discrete-Time Adapter

This module provides a **tick-based interface** for Alibaba Cloud's Qwen Omni Flash Realtime API via DashScope, designed for discrete-time full-duplex voice simulation.

## ⚠️ Critical Limitation: Tool Calling Does NOT Work

```
================================================================================
⚠️  TOOL/FUNCTION CALLING DOES NOT WORK WITH QWEN REALTIME API  ⚠️
================================================================================

Despite accepting tools configuration, the Qwen Realtime WebSocket API
(qwen3-omni-flash-realtime) does NOT actually invoke functions:

1. The model accepts tool definitions in session.update
2. The model may SAY "let me check that for you"
3. BUT it never emits function_call events - it generates audio instead

This is a limitation of the REALTIME API specifically. The HTTP API
(qwen3-omni-flash) DOES support tool calling correctly.

Tested: January 2026
Status: Audio streaming works, tool calling broken

If tool calling is required, use OpenAI, Gemini, xAI, or Nova instead.
================================================================================
```

## Overview

Qwen Omni Flash Realtime is Alibaba Cloud's real-time voice API that uses an OpenAI-compatible WebSocket protocol. It supports bidirectional audio streaming with server-side VAD for barge-in detection.

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
│                 DiscreteTimeQwenAdapter                         │
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
│                   QwenRealtimeProvider                          │
│                   (async WebSocket client)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DashScope Realtime API                       │
│                    (WebSocket server)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Key Differences from OpenAI

| Feature | OpenAI Realtime | Qwen Realtime |
|---------|-----------------|---------------|
| **Audio Input** | 8kHz G.711 μ-law | 16kHz PCM16 |
| **Audio Output** | 8kHz G.711 μ-law | 24kHz PCM16 |
| **Tool Calling** | ✅ Works | ❌ Broken (accepts but never invokes) |
| **Protocol** | OpenAI WebSocket | OpenAI-compatible WebSocket |
| **Authentication** | `OPENAI_API_KEY` | `DASHSCOPE_API_KEY` |

## Audio Format Conversion

The adapter handles conversion between telephony and Qwen formats:

```
                     AUDIO INPUT
    ════════════════════════════════════════════════════
    
    Telephony (8kHz μ-law)  →  convert_input()  →  Qwen (16kHz PCM16)
    
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  8000 Hz        │     │  Resample       │     │  16000 Hz       │
    │  μ-law          │ ──▶ │  Decode μ-law   │ ──▶ │  PCM16          │
    │  1 byte/sample  │     │                 │     │  2 bytes/sample │
    └─────────────────┘     └─────────────────┘     └─────────────────┘


                     AUDIO OUTPUT
    ════════════════════════════════════════════════════
    
    Qwen (24kHz PCM16)  →  convert_output()  →  Telephony (8kHz μ-law)
    
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  24000 Hz       │     │  Resample       │     │  8000 Hz        │
    │  PCM16          │ ──▶ │  Encode μ-law   │ ──▶ │  μ-law          │
    │  2 bytes/sample │     │                 │     │  1 byte/sample  │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Components

| File | Description |
|------|-------------|
| `events.py` | Pydantic models for Qwen Realtime events (OpenAI-compatible) |
| `provider.py` | `QwenRealtimeProvider` - async WebSocket client |
| `discrete_time_adapter.py` | `DiscreteTimeQwenAdapter` - implements `DiscreteTimeAdapter` |
| `audio_utils.py` | Audio format conversion utilities |

## Usage

### CLI

```bash
# Run with Qwen Realtime provider (NOTE: tools will not work!)
tau2 run --domain retail --audio-native --audio-native-provider qwen \
    --num-tasks 1 --seed 42 --tick-duration 0.2 --max-steps-seconds 120 \
    --speech-complexity control --verbose-logs --save-to my_simulation
```

### Programmatic

```python
from tau2.voice.audio_native.qwen import (
    DiscreteTimeQwenAdapter,
    QwenVADConfig,
)

adapter = DiscreteTimeQwenAdapter(
    tick_duration_ms=200,
    send_audio_instant=True,
    voice="Cherry",
)

# NOTE: Pass empty tools list - tool calling does NOT work!
adapter.connect(
    system_prompt="You are a helpful assistant.",
    tools=[],  # Tool calling is broken in Qwen Realtime API
    vad_config=QwenVADConfig(),
)

for tick in range(max_ticks):
    result = adapter.run_tick(user_audio_chunk, tick_number=tick)
    # result.get_played_agent_audio() - agent audio (telephony format)
    # result.proportional_transcript - text for this tick

adapter.disconnect()
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DASHSCOPE_API_KEY` | Alibaba Cloud DashScope API key |

### Models

| Model | Description |
|-------|-------------|
| `qwen3-omni-flash-realtime` | Default Qwen Omni Flash Realtime model |

### Voice Options

| Voice | Description |
|-------|-------------|
| `Cherry` | Default voice |

### VAD Configuration

```python
class QwenVADConfig(BaseModel):
    mode: QwenVADMode = QwenVADMode.SERVER_VAD  # or MANUAL
    threshold: float = 0.5           # Speech detection sensitivity (0.0-1.0)
    prefix_padding_ms: int = 300     # Audio before speech start
    silence_duration_ms: int = 800   # Silence before turn end
```

## OpenAI-Compatible Protocol

Qwen uses the same WebSocket event types as OpenAI:

| Event Type | Description |
|------------|-------------|
| `session.created` | Session started |
| `session.updated` | Session configured |
| `response.audio.delta` | Audio chunk |
| `response.audio_transcript.delta` | Transcript chunk |
| `input_audio_buffer.speech_started` | VAD: speech started |
| `input_audio_buffer.speech_stopped` | VAD: speech stopped |
| `response.done` | Response complete |

## Known Issues & Caveats

1. **Tool Calling Broken**: The most significant limitation. The API accepts tool configurations but never emits `function_call_arguments.done` events. Use OpenAI, Gemini, xAI, or Nova if you need tool calling.

2. **Audio Format Conversion**: Requires conversion between telephony (8kHz μ-law) and Qwen formats (16kHz/24kHz PCM16). Some quality loss is inherent in resampling.

3. **Limited Voice Options**: Currently only "Cherry" voice is documented.

## Implementation Status

- [x] Events (`events.py`)
- [x] Provider (`provider.py`)
- [x] Discrete-time adapter (`discrete_time_adapter.py`)
- [x] Audio format conversion (`audio_utils.py`)
- [x] Server-side VAD
- [x] Interruption handling
- [x] Output transcription
- [ ] Tool/function calling (BROKEN - API limitation)

## Reference

- [Alibaba Cloud Model Studio Realtime API](https://www.alibabacloud.com/help/en/model-studio/realtime)
