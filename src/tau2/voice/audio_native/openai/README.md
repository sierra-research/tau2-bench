# OpenAI Realtime API - Discrete-Time Adapter

This module provides a **tick-based interface** for OpenAI's Realtime API, designed for discrete-time full-duplex voice simulation.

## Overview

In discrete-time simulation, time advances in fixed increments called **ticks**. Each tick represents a fixed duration of simulated audio time (e.g., 200ms). This differs from real-time streaming where events happen continuously.

```
Real-Time (Continuous)          Discrete-Time (Tick-Based)
─────────────────────           ──────────────────────────

Audio flows continuously        Audio exchanged in fixed chunks

     ┌──────────────┐                ┌──────────────┐
     │   User       │                │   User       │
     │   Audio      │────stream────▶ │   Audio      │──┬─ Tick 1 ─▶
     └──────────────┘                └──────────────┘  ├─ Tick 2 ─▶
                                                       └─ Tick 3 ─▶
```

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
│                  (get_next_chunk per tick)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│             DiscreteTimeAudioNativeAdapter                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    TickRunner                           │   │
│   │  • Audio capping (max bytes_per_tick)                   │   │
│   │  • Audio buffering (excess → next tick)                 │   │
│   │  • Proportional transcript distribution                 │   │
│   │  • Interruption handling                                │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OpenAIRealtimeProvider                        │
│                   (async WebSocket client)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OpenAI Realtime API                          │
│                    (WebSocket server)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Tick Flow

Each call to `run_tick()` represents one tick of the simulation:

```
                              TICK N
    ════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────┐
    │ 1. RECEIVE USER AUDIO                                   │
    │    • User simulator provides audio for this tick        │
    │    • Fixed size: bytes_per_tick bytes                   │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 2. PREPEND BUFFERED AUDIO                               │
    │    • Add any agent audio buffered from Tick N-1         │
    │    • This audio was received but exceeded the cap       │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 3. SEND & RECEIVE                                       │
    │    • Send user audio to OpenAI API                      │
    │    • Receive events for tick_duration_ms                │
    │    • Collect agent audio (AudioDeltaEvent)              │
    │    • Collect transcript (AudioTranscriptDeltaEvent)     │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 4. CAP & BUFFER                                         │
    │    • Cap agent audio at bytes_per_tick                  │
    │    • Buffer excess for Tick N+1                         │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 5. CALCULATE PROPORTIONAL TRANSCRIPT                    │
    │    • Distribute transcript text proportionally          │
    │    • Based on audio bytes played this tick              │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ 6. RETURN TICK RESULT                                   │
    │    • Agent audio (exactly bytes_per_tick, padded)       │
    │    • Proportional transcript                            │
    │    • Events, timing info, tool calls                    │
    └─────────────────────────────────────────────────────────┘
```

## Audio Buffering

Agent audio may arrive faster than tick duration allows. Excess audio is buffered for the next tick:

```
                        Agent generates 300ms of audio
                        but tick is only 200ms
    
    Tick N                              Tick N+1
    ────────────────────────────────    ────────────────────────────────
    
    Agent audio from API:               Buffered audio prepended:
    ┌──────────────────────────────┐    ┌────────────┬─────────────────┐
    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│    │░░░░░░░░░░░░│                 │
    │        300ms of audio        │    │  from N    │  new audio      │
    └──────────────────────────────┘    └────────────┴─────────────────┘
              │                                   │
              ▼                                   ▼
    ┌────────────────────┬─────────┐    ┌────────────────────┬─────────┐
    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░│    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│░░░░░░░░░│
    │   Play 200ms       │ Buffer  │    │   Play 200ms       │ Buffer  │
    │                    │  100ms  │    │                    │  ...    │
    └────────────────────┴─────────┘    └────────────────────┴─────────┘
    
    ▓▓▓ = Played this tick    ░░░ = Buffered for next tick
```

## Proportional Transcript

Text arrives from the API independently of audio. To synchronize display, transcript is distributed proportionally based on audio played:

```
    Full utterance: "Hello, how can I help you today?"
    Total audio:    1600ms (4 ticks @ 400ms each)
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    PROPORTIONAL DISTRIBUTION                     │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  Tick 1:  ▓▓▓▓░░░░░░░░░░░░  "Hello, ho"     (25% of text)       │
    │  Tick 2:  ▓▓▓▓▓▓▓▓░░░░░░░░  "w can I h"     (next 25%)          │
    │  Tick 3:  ▓▓▓▓▓▓▓▓▓▓▓▓░░░░  "elp you t"     (next 25%)          │
    │  Tick 4:  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  "oday?"         (final 25%)         │
    │                                                                  │
    │  ▓▓▓ = Audio played    ░░░ = Audio remaining                    │
    └─────────────────────────────────────────────────────────────────┘
    
    Formula:
    ┌────────────────────────────────────────────────────────────────┐
    │                    audio_bytes_played                          │
    │  chars_to_show = ───────────────────── × total_transcript_len  │
    │                   audio_bytes_received                         │
    └────────────────────────────────────────────────────────────────┘
```

## Interruption Handling (Barge-In)

When the user speaks while the agent is responding, the system handles the interruption:

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    INTERRUPTION FLOW                             │
    └─────────────────────────────────────────────────────────────────┘
    
    
         Agent speaking                    User interrupts
              │                                  │
              ▼                                  ▼
    ══════════════════════════════════════════════════════════════════
         Tick 1         Tick 2         Tick 3         Tick 4
    ──────────────────────────────────────────────────────────────────
    
    Agent: ▓▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓░░░░░░  ░░░░░░░░░░░░
           "Hello, I"    "can help"    "you t"--      (discarded)
                                           │
    User:  ░░░░░░░░░░░░  ░░░░░░░░░░░░  ░░░░▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓▓
           (silence)     (silence)         "Wait!"    "I need..."
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │  SpeechStartedEvent     │
                              │  audio_start_ms: 850    │
                              └─────────────────────────┘
                                           │
                                           ▼
                         ┌─────────────────────────────────────┐
                         │         CLIENT ACTIONS               │
                         ├─────────────────────────────────────┤
                         │ 1. Stop playing agent audio         │
                         │ 2. Clear buffered audio             │
                         │ 3. Calculate audio_end_ms           │
                         │ 4. Send truncate to server ────────────────┐
                         └─────────────────────────────────────┘      │
                                                                      │
                         ┌─────────────────────────────────────┐      │
                         │         SERVER ACTIONS               │◀─────┘
                         ├─────────────────────────────────────┤
                         │ 1. Cancel in-progress response      │
                         │ 2. Truncate conversation history    │
                         │ 3. Model knows what user heard      │
                         └─────────────────────────────────────┘
    
    
    ▓▓▓ = Actual audio    ░░░ = Silence/Discarded
```

### Truncation Message

When interrupted, the client sends `conversation.item.truncate` to the server:

```
    ┌────────────────────────────────────────────────────────────────┐
    │  {                                                             │
    │    "type": "conversation.item.truncate",                       │
    │    "item_id": "item_abc123",    // interrupted response        │
    │    "content_index": 0,                                         │
    │    "audio_end_ms": 850          // what user actually heard    │
    │  }                                                             │
    └────────────────────────────────────────────────────────────────┘
    
    This ensures the model's memory matches what the user heard,
    enabling natural follow-ups like "What was that last thing?"
```

### VAD Mode vs Push-to-Talk

```
    VAD Mode (default)                   Push-to-Talk (VAD disabled)
    ──────────────────────────────       ──────────────────────────────
    
    Server detects speech                Client controls manually
    
    1. SpeechStartedEvent arrives        1. User releases "talk" button
    2. Server auto-cancels response      2. Client sends response.cancel
    3. Client sends truncate             3. Client sends truncate
                                         4. Client sends input_audio_buffer.commit
                                         5. Client sends response.create
    
    Note: In VAD mode, the server handles step 2 automatically.
    The provider.cancel_response() method is available for push-to-talk.
```

## Tool Call Flow

When the agent requests a tool call, it spans multiple ticks:

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                      TOOL CALL FLOW                              │
    └─────────────────────────────────────────────────────────────────┘
    
    
         Tick N              Tick N+1             Tick N+2
    ═══════════════════  ═══════════════════  ═══════════════════
    
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │ Agent requests  │  │ Tool result     │  │ Agent continues │
    │ tool call       │  │ sent to API     │  │ with response   │
    │                 │  │                 │  │                 │
    │ FunctionCall-   │  │ Orchestrator    │  │ Agent generates │
    │ ArgumentsDone   │  │ executes tool   │  │ audio based on  │
    │ Event received  │  │ and provides    │  │ tool result     │
    │                 │  │ result          │  │                 │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │ Return tool_    │  │ send_tool_      │  │ Return audio    │
    │ calls in        │  │ result() queues │  │ response to     │
    │ AssistantMsg    │  │ for next tick   │  │ orchestrator    │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
    
    
    Note: Tool results are queued and sent at the START of the next tick.
    This ensures proper timing in discrete-time simulation.
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tick_duration_ms` | Duration of each tick in milliseconds | - |
| `send_audio_instant` | **Deprecated.** If True, sends all audio at once; if False, streams in 20ms chunks. This option will be removed in a future release. | Required |
| `buffer_until_complete` | **Deprecated.** If True, waits for complete utterances before releasing audio. This option will be removed in a future release. | `False` |
| `audio_format` | Audio format for API communication | Telephony (8kHz μ-law) |
| `fast_forward_mode` | **Deprecated.** If True, exits tick early when enough audio is buffered (speeds up simulation). This option will be removed in a future release. | `False` |

### Modes

```
    send_audio_instant=True (deprecated) send_audio_instant=False (deprecated)
    ────────────────────────────         ────────────────────────────
    
    User audio sent instantly            User audio sent in chunks
    (discrete-time simulation)           (VoIP-style streaming)
    
    ┌────────────────────────┐           ┌────┬────┬────┬────┬────┐
    │ All 200ms at once      │           │20ms│20ms│20ms│20ms│... │
    └────────────────────────┘           └────┴────┴────┴────┴────┘
          │                                │    │    │    │
          └──────────────────▶             └─┬──┴─┬──┴─┬──┴─┬──▶
                                             │    │    │    │
                                           sleep sleep sleep sleep
    
    
    buffer_until_complete=False          buffer_until_complete=True (deprecated)
    ────────────────────────────         ────────────────────────────
    
    Stream audio as received             Wait for utterance to complete
    (lower latency)                      (accurate timing)
    
    Audio arrives:  ▓▓▓░░░░░░░░          Audio arrives:  ▓▓▓░░░░░░░░
    Tick returns:   ▓▓▓                  Tick returns:   (nothing)
                      │                                    │
    Next chunk:     ░░░▓▓▓░░░░           AudioDoneEvent: ▓▓▓▓▓▓▓▓▓▓▓
    Tick returns:   ░░░▓▓▓               Tick returns:   ▓▓▓▓▓▓▓▓▓▓▓


    fast_forward_mode=False              fast_forward_mode=True (deprecated)
    ────────────────────────────         ────────────────────────────
    
    Wait full tick_duration_ms           Exit early when enough audio
    (real-time pacing)                   (faster simulation)
    
    API responds in 50ms                 API responds in 50ms
    ┌────────────────────────┐           ┌────────────────────────┐
    │ Wait remaining 150ms   │           │ Return immediately     │
    │ before returning       │           │ if bytes >= bytes_per  │
    └────────────────────────┘           │ _tick buffered         │
              │                          └────────────────────────┘
              ▼                                    │
    Tick takes 200ms wall-clock                    ▼
                                         Tick takes ~50ms wall-clock
```

## Key Data Structures

### TickResult

Returned by each `run_tick()` call:

```
    TickResult
    ├── tick_number: int                    # 1-indexed tick number
    ├── user_audio_data: bytes              # User audio sent this tick
    ├── agent_audio_chunks: List            # Agent audio received (raw)
    ├── events: List[BaseRealtimeEvent]     # All API events
    ├── proportional_transcript: str        # Text for this tick's audio
    ├── was_truncated: bool                 # True if interrupted
    └── Methods:
        └── get_played_agent_audio()        # Padded to exactly bytes_per_tick
```

### UtteranceTranscript

Tracks audio/transcript for proportional distribution:

```
    UtteranceTranscript
    ├── item_id: str                        # OpenAI item ID
    ├── audio_bytes_received: int           # Total audio from API
    ├── audio_bytes_played: int             # Audio "played" so far
    ├── transcript_received: str            # Full transcript text
    └── transcript_chars_shown: int         # Characters shown so far
```

## Usage

### CLI

```bash
# Run with OpenAI Realtime provider
tau2 run --domain retail --audio-native --audio-native-provider openai \
    --num-tasks 1 --seed 42 --tick-duration 0.2 --max-steps-seconds 120 \
    --speech-complexity control --verbose-logs --save-to my_simulation
```

### Programmatic

```python
from tau2.voice.audio_native.openai import (
    DiscreteTimeAudioNativeAdapter,
    OpenAIVADConfig,
    OpenAIVADMode,
)

adapter = DiscreteTimeAudioNativeAdapter(
    tick_duration_ms=200,
    send_audio_instant=True,     # Deprecated: will be removed in a future release
    buffer_until_complete=False, # Deprecated: will be removed in a future release
)

adapter.connect(
    system_prompt="You are a helpful assistant.",
    tools=tools,
    vad_config=OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD),
    modality="audio",
)

for tick in range(max_ticks):
    result = adapter.run_tick(user_audio_chunk, tick_number=tick)
    # result.get_played_agent_audio() - agent audio (8kHz μ-law)
    # result.proportional_transcript - text for this tick
    # result.tool_calls - function calls

adapter.disconnect()
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for authentication |

### Models

| Model | Description |
|-------|-------------|
| `gpt-realtime-2025-08-28` | Default model for OpenAI Realtime API |

### Voice

The voice is currently hardcoded to `alloy`. See [OpenAI's documentation](https://platform.openai.com/docs/guides/realtime) for available voices.

## Files in This Module

| File | Description |
|------|-------------|
| `provider.py` | Low-level async WebSocket client for OpenAI Realtime API |
| `tick_runner.py` | Core tick logic: buffering, capping, proportional transcript |
| `discrete_time_adapter.py` | Sync wrapper with background thread for tick-based interface |
| `events.py` | Pydantic models for API events (e.g., `SpeechStartedEvent`) |

