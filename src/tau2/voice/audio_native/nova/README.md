# Amazon Nova Sonic API - Discrete-Time Adapter

This module provides a **tick-based interface** for Amazon Nova Sonic API via AWS Bedrock, designed for discrete-time full-duplex voice simulation.

## Overview

Amazon Nova Sonic is an audio-native LLM that processes speech-to-speech in real-time using AWS Bedrock's bidirectional streaming API.

```
┌─────────────────────────────────────────────────────────────────┐
│               DiscreteTimeAudioNativeAgent                      │
│                  (get_next_chunk per tick)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               DiscreteTimeNovaAdapter                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • Audio format conversion (telephony ↔ Nova)           │   │
│   │  • Audio capping (max bytes_per_tick)                   │   │
│   │  • Audio buffering (excess → next tick)                 │   │
│   │  • SPECULATIVE/FINAL content filtering                  │   │
│   │  • Proportional transcript distribution                 │   │
│   │  • Interruption handling (barge-in)                     │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     NovaSonicProvider                           │
│              (AWS Bedrock bidirectional stream)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS Bedrock Runtime API                      │
│                    (HTTP/2 bidirectional)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Differences from OpenAI

| Feature | OpenAI Realtime | Nova Sonic |
|---------|-----------------|------------|
| **Protocol** | WebSocket | AWS Bedrock HTTP/2 bidirectional stream |
| **Authentication** | API Key | AWS SigV4 (boto3 credentials) |
| **Audio Input** | 8kHz μ-law (G.711) | 16kHz PCM16 LPCM |
| **Audio Output** | 8kHz μ-law (G.711) | 24kHz PCM16 LPCM |
| **VAD Modes** | SERVER_VAD, SEMANTIC_VAD, MANUAL | SERVER_VAD only |
| **Generation** | Streaming (committed) | SPECULATIVE + FINAL |

## SPECULATIVE vs FINAL Generation

Nova Sonic uses **speculative generation** - it starts generating content before fully committing to it:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPECULATIVE GENERATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. User speaks: "What's the weather?"                          │
│                                                                  │
│  2. Nova starts SPECULATIVE responses (may be revised):        │
│     ├─ contentStart (generationStage: SPECULATIVE) → ignored   │
│     ├─ textOutput "The weather is..." → ignored                 │
│     ├─ audioOutput → ignored                                    │
│     └─ ... (multiple speculative versions may be generated)    │
│                                                                  │
│  3. Nova commits FINAL response:                                │
│     ├─ contentStart (generationStage: FINAL) → tracked         │
│     ├─ textOutput "The weather today is sunny..." → processed  │
│     ├─ audioOutput → converted and played                      │
│     └─ contentEnd                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

IMPORTANT: We only process content from FINAL generation stage.
           SPECULATIVE content is logged and ignored.
```

The adapter tracks which `content_id` values are FINAL and filters audio/text accordingly:

```python
# In _process_event():
if isinstance(event, NovaContentStartEvent):
    if event.generation_stage == "FINAL":
        self._final_content_ids.add(event.content_id)
    # SPECULATIVE content_ids are not added

if isinstance(event, NovaAudioOutputEvent):
    if content_id not in self._final_content_ids:
        return  # Skip speculative audio

if isinstance(event, NovaTextOutputEvent):
    if content_id not in self._final_content_ids:
        return  # Skip speculative text
```

## Audio Format Conversion

Nova Sonic uses different audio formats than telephony standard:

```
                     AUDIO INPUT
    ════════════════════════════════════════════════════
    
    Telephony (8kHz μ-law)  →  convert_input()  →  Nova (16kHz PCM16)
    
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  8000 Hz        │     │  Resample       │     │  16000 Hz       │
    │  μ-law          │ ──▶ │  Decode μ-law   │ ──▶ │  PCM16          │
    │  1 byte/sample  │     │                 │     │  2 bytes/sample │
    └─────────────────┘     └─────────────────┘     └─────────────────┘


                     AUDIO OUTPUT
    ════════════════════════════════════════════════════
    
    Nova (24kHz PCM16)  →  convert_output()  →  Telephony (8kHz μ-law)
    
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  24000 Hz       │     │  Resample       │     │  8000 Hz        │
    │  PCM16          │ ──▶ │  Encode μ-law   │ ──▶ │  μ-law          │
    │  2 bytes/sample │     │                 │     │  1 byte/sample  │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
```

The `StreamingNovaConverter` class handles this conversion with resampling state preservation for seamless streaming.

## Interruption Handling (Barge-In)

When user speech is detected, Nova sends `speechStarted` or `bargeIn` events:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTERRUPTION FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Agent is speaking (FINAL audio being played)                │
│                                                                  │
│  2. User starts speaking                                        │
│     └─ Nova detects via server-side VAD                         │
│                                                                  │
│  3. Nova sends: speechStarted or bargeIn event                  │
│                                                                  │
│  4. Adapter handles interruption:                               │
│     ├─ Clear buffered audio                                     │
│     ├─ Mark result.was_truncated = True                         │
│     ├─ Set skip_item_id (ignore remaining audio from utterance) │
│     ├─ Clear _final_content_ids (new response will have new IDs)│
│     └─ Reset audio converter                                    │
│                                                                  │
│  5. Agent stops, new response begins with fresh content IDs     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Event Types

Nova Sonic uses a different event protocol than OpenAI:

| Event | Description |
|-------|-------------|
| `sessionStart` / `sessionEnd` | Session lifecycle |
| `contentStart` | Content block starting (includes `generationStage`) |
| `contentEnd` | Content block ended |
| `textOutput` | Text content (ASR for USER, response for ASSISTANT) |
| `audioOutput` | Audio chunk (base64 LPCM 24kHz) |
| `toolUse` | Tool/function call request |
| `speechStarted` | User started speaking (VAD) |
| `speechEnded` | User stopped speaking (VAD) |
| `bargeIn` | User interrupted (barge-in) |
| `completionStart` / `completionEnd` | Completion lifecycle |

### Key Event Fields

The `NovaContentStartEvent` includes:
- `role`: "USER" or "ASSISTANT"
- `content_id`: Unique identifier for this content block
- `type`: "AUDIO", "TEXT", "TOOL_USE", "TOOL_RESULT"
- `generation_stage`: "SPECULATIVE" or "FINAL" (parsed from `additionalModelFields`)

## Usage

### CLI

```bash
# Run with Nova Sonic provider
tau2 run --domain retail --audio-native --audio-native-provider nova \
    --num-tasks 1 --seed 42 --tick-duration 0.2 --max-steps-seconds 120 \
    --speech-complexity control --verbose-logs --save-to my_simulation
```

### Programmatic

```python
from tau2.voice.audio_native.nova import (
    DiscreteTimeNovaAdapter,
    NovaVADConfig,
)

adapter = DiscreteTimeNovaAdapter(
    tick_duration_ms=200,     # Duration of each tick
    send_audio_instant=True,  # Send all audio at once per tick
    voice="tiffany",          # Voice: matthew, tiffany, amy
)

adapter.connect(
    system_prompt="You are a helpful assistant.",
    tools=tools,
    vad_config=NovaVADConfig(),  # Only SERVER_VAD supported
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
| `AWS_ACCESS_KEY_ID` | AWS access key ID |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key |
| `AWS_DEFAULT_REGION` | AWS region (default: `us-east-1`) |
| `AWS_PROFILE` | AWS profile name (alternative to access keys) |

### Models

| Model | Description |
|-------|-------------|
| `amazon.nova-2-sonic-v1:0` | Default Nova Sonic model |

### Voice Options

| Voice | Description |
|-------|-------------|
| `tiffany` | Default voice |
| `matthew` | Alternative voice |
| `amy` | Alternative voice |

## AWS Authentication

Nova Sonic uses standard AWS credentials via boto3:

```python
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# Option 2: AWS profile
export AWS_PROFILE=my-profile

# Option 3: EC2 instance profile (automatic)
```

The `NovaSonicProvider` uses `boto3.Session()` to resolve credentials.

## Files in This Module

| File | Description |
|------|-------------|
| `provider.py` | Low-level AWS Bedrock bidirectional stream client |
| `discrete_time_adapter.py` | Sync wrapper with SPECULATIVE/FINAL filtering |
| `events.py` | Pydantic models for Nova events (includes `generation_stage`) |
| `audio_utils.py` | Audio format conversion (telephony ↔ Nova) |

## Common Issues

### Agent appears to talk continuously

**Cause**: SPECULATIVE content was being processed instead of only FINAL.

**Solution**: The adapter now filters by `generation_stage`. Only content with `"generationStage": "FINAL"` in `additionalModelFields` is processed.

### Audio quality issues

**Cause**: Audio format mismatch or resampling artifacts.

**Solution**: The `StreamingNovaConverter` preserves resampling state for seamless streaming. Reset with `reset()` on interruption.

### VAD not working

**Cause**: Nova Sonic only supports server-side VAD.

**Solution**: Use `NovaVADConfig()` with default settings. MANUAL mode is not supported.

## Implementation Status

- [x] Events (`events.py`)
- [x] Provider (`provider.py`)
- [x] Discrete-time adapter (`discrete_time_adapter.py`)
- [x] Audio format conversion (`audio_utils.py`)
- [x] SPECULATIVE/FINAL content filtering
- [x] Tool call parsing
- [x] Input transcription (ASR)
- [x] Output transcription
- [x] Interruption handling (barge-in)

## Reference

- [AWS Nova Sonic Documentation](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html)
