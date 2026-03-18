# Gemini Live API - Discrete-Time Adapter

This module provides a **tick-based interface** for Google's Gemini Live API, designed for discrete-time full-duplex voice simulation.

## Overview

Gemini Live is Google's real-time multimodal API that supports bidirectional audio streaming with native audio understanding and generation. This implementation follows the same discrete-time simulation pattern as the OpenAI adapter.

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
│               DiscreteTimeGeminiAdapter                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • Audio format conversion (telephony ↔ PCM16)          │   │
│   │  • Audio capping & buffering                            │   │
│   │  • Proportional transcript distribution                 │   │
│   │  • Event processing (must process ALL events)           │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GeminiLiveProvider                           │
│              (async google-genai SDK client)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gemini Live API                              │
│                    (WebSocket server)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Key Differences from OpenAI Realtime

### Audio Formats

| Direction | OpenAI Realtime | Gemini Live |
|-----------|-----------------|-------------|
| Input | 8kHz G.711 μ-law | 16kHz PCM16 mono |
| Output | 8kHz G.711 μ-law | 24kHz PCM16 mono |
| Bytes/second (in) | 8,000 | 32,000 |
| Bytes/second (out) | 8,000 | 48,000 |

The adapter handles conversion between telephony format (used by the orchestrator) and Gemini's PCM16 formats.

### Authentication

| Method | OpenAI | Gemini |
|--------|--------|--------|
| API Key | `OPENAI_API_KEY` | `GEMINI_API_KEY` (AI Studio) |
| Service Account | N/A | `GOOGLE_APPLICATION_CREDENTIALS` + `GOOGLE_CLOUD_PROJECT` |
| Inline Service Account | N/A | `GOOGLE_SERVICE_ACCOUNT_KEY` (JSON content) |

The provider auto-detects authentication method based on available environment variables.

### Client Library & Session Management

```
OpenAI                              Gemini
──────────────────────────          ──────────────────────────

Raw WebSocket connection            google-genai SDK
+ custom event parsing              + async context manager

# OpenAI                            # Gemini
ws = await connect(url)             ctx = client.aio.live.connect(...)
await ws.send(data)                 session = await ctx.__aenter__()
event = await ws.recv()             async for resp in session.receive():
await ws.close()                    await ctx.__aexit__(None, None, None)

Key difference: Gemini requires careful management of the
async context manager for proper session lifecycle.
```

### Tool Calls

| Aspect | OpenAI | Gemini |
|--------|--------|--------|
| Event type | `function_call_arguments.done` | `response.tool_call` container |
| Structure | Single tool call per event | `function_calls[]` array |
| Arguments | `arguments` (JSON string) | `args` (dict) |

```python
# OpenAI tool call event
{
    "type": "response.function_call_arguments.done",
    "call_id": "call_abc123",
    "name": "get_weather",
    "arguments": "{\"location\": \"NYC\"}"
}

# Gemini tool call (in response.tool_call)
{
    "_type": "LiveServerToolCall",
    "function_calls": [
        {
            "id": "call_abc123",
            "name": "get_weather",
            "args": {"location": "NYC"}
        }
    ]
}
```

### Tool Schema Handling

Tool schemas are passed using `parametersJsonSchema`, which lets the SDK handle complex JSON Schema features like `$ref` and `$defs` natively.

```python
# Default behavior - SDK handles $ref/$defs
types.FunctionDeclaration(
    name="my_tool",
    description="...",
    parametersJsonSchema=raw_openai_schema,  # Includes $ref, $defs
)
```

A fallback manual flattening mode (`use_raw_json_schema=False`) is available if needed, which inlines `$ref` references and removes `additionalProperties`.

### Transcription Events

| Type | OpenAI | Gemini |
|------|--------|--------|
| Input (user speech) | `input_audio_transcription.completed` | `server_content.input_transcription` |
| Output (agent speech) | `response.audio_transcript.delta` | `server_content.output_transcription` |

### Audio Data Location

```
OpenAI                              Gemini
──────────────────────────          ──────────────────────────

Base64 encoded in event             Raw bytes in response

{                                   response.data = b'...'
  "type": "response.audio.delta",   OR
  "delta": "SGVsbG8gV29ybGQ="       response.server_content
}                                     .model_turn.parts[0]
                                      .inline_data.data = b'...'
```

### Interruption Handling

| Aspect | OpenAI | Gemini |
|--------|--------|--------|
| Signal | `SpeechStartedEvent` with `audio_start_ms` | `server_content.interrupted = True` |
| Client action | Send `conversation.item.truncate` | Automatic (server handles) |

### Event Processing

**Critical difference**: In Gemini, text and audio events are interleaved. You **must process ALL events** in each receive call - do not break early when audio buffer is full, or you will lose transcript text.

```python
# WRONG - loses transcript events
for event in events:
    process(event)
    if audio_buffer_full:
        break  # ← Text events after this are LOST!

# CORRECT - process everything, buffer excess audio
for event in events:
    process(event)  # Process ALL events
# Then limit audio output in buffering step
```

## Components

| File | Description |
|------|-------------|
| `events.py` | Pydantic models for Gemini Live events |
| `provider.py` | `GeminiLiveProvider` - async session client |
| `discrete_time_adapter.py` | `DiscreteTimeGeminiAdapter` - implements `DiscreteTimeAdapter` |
| `audio_utils.py` | Audio format conversion utilities |

## Usage

### CLI

```bash
# Run with Gemini Live provider
tau2 run --domain retail --audio-native --audio-native-provider gemini \
    --num-tasks 1 --seed 42 --tick-duration 0.2 --max-steps-seconds 120 \
    --speech-complexity regular --verbose-logs --save-to my_simulation
```

### Programmatic

```python
from tau2.voice.audio_native.gemini import (
    DiscreteTimeGeminiAdapter,
    GeminiVADConfig,
)

adapter = DiscreteTimeGeminiAdapter(
    tick_duration_ms=200,
    send_audio_instant=True,
)

adapter.connect(
    system_prompt="You are a helpful assistant.",
    tools=tools,
    vad_config=GeminiVADConfig(),
    modality="audio",
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
| `GEMINI_API_KEY` | API key for Google AI Studio |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON file |
| `GOOGLE_SERVICE_ACCOUNT_KEY` | Service account JSON content (inline) |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID (required for Vertex AI) |

### VAD Configuration

```python
class GeminiVADConfig(BaseModel):
    mode: GeminiVADMode = GeminiVADMode.AUTOMATIC
    enable_input_transcription: bool = True
```

### Voice Configuration

The voice is configured when connecting via the provider. Default voice: `Zephyr`

```python
# Voice is set internally when provider connects
# To use a different voice, pass it to the provider:
provider = GeminiLiveProvider()
await provider.connect(
    system_prompt="...",
    tools=[],
    voice="Puck",  # Override default voice
)
```

Available voices depend on the model. Check [Google's documentation](https://ai.google.dev/gemini-api/docs/speech-generation) for the current list of supported voices.

### Models

| Model | Description |
|-------|-------------|
| `models/gemini-live-2.5-flash-native-audio` | Default for AI Studio |
| `gemini-live-2.5-flash-native-audio` | Default for Vertex AI |

Note: Model names may change. Check [Google's Gemini Live documentation](https://ai.google.dev/api/live) for the latest available models.

## Known Issues & Caveats

1. **Session Lifecycle**: The `google-genai` SDK uses async context managers. Improper management causes `ConnectionClosedOK` errors.

2. **Tool Schema Handling**: By default, tool schemas are passed directly to the SDK via `parametersJsonSchema`. If issues arise, set `use_raw_json_schema=False` to use manual schema flattening.

3. **Audio Format Conversion**: The adapter converts between telephony (8kHz μ-law) and Gemini formats. Some audio quality loss is inherent in format conversion.

4. **Context Window Compression**: Gemini supports automatic context window compression for long sessions:
   ```python
   config["context_window_compression"] = ContextWindowCompressionConfig(
       trigger_tokens=25600,
       sliding_window=SlidingWindow(target_tokens=12800),
   )
   ```

5. **Output Transcription**: Enabled by default to get text of what Gemini says:
   ```python
   config["output_audio_transcription"] = AudioTranscriptionConfig()
   ```

6. **Null Tool Call IDs**: Gemini sometimes sends empty or null `call_id` values for function calls. The adapter generates synthetic IDs internally for tracking and maps them back to original IDs when sending tool responses.

7. **Session Timeout**: Sessions timeout after ~10 minutes. Enable session resumption (`max_resumptions > 0`) for long-running conversations. See the [Session Resumption](#session-resumption) section for details.

## Session Resumption

Gemini Live sessions have a ~10 minute timeout. The adapter supports automatic session resumption to maintain long-running conversations.

### Configuration

```python
adapter = DiscreteTimeGeminiAdapter(
    tick_duration_ms=200,
    max_resumptions=3,           # Max reconnection attempts (default: 3, 0 to disable)
    resume_only_on_timeout=True, # Only resume on planned timeout (default: True)
)
```

### How It Works

1. Server periodically sends `SessionResumption` events with resumption handles
2. Server sends `GoAway` event ~30 seconds before planned disconnect
3. When connection closes after GoAway, adapter automatically reconnects with saved handle
4. Conversation context is preserved across reconnection

### Events

| Event | Description |
|-------|-------------|
| `GeminiGoAwayEvent` | Server warning before disconnect, includes `time_left_seconds` |
| `GeminiSessionResumptionEvent` | Contains `new_handle` and `resumable` flag for session recovery |

### Behavior

- If `resume_only_on_timeout=True` (default): Only attempts resumption when GoAway was received before disconnect (planned timeout)
- If `resume_only_on_timeout=False`: Attempts resumption on any connection close
- After `max_resumptions` attempts, raises `RuntimeError`

## Implementation Status

- [x] Events (`events.py`)
- [x] Provider (`provider.py`)
- [x] Discrete-time adapter (`discrete_time_adapter.py`)
- [x] Audio format conversion (`audio_utils.py`)
- [x] Tool call parsing
- [x] Tool schema handling (via `parametersJsonSchema`, with manual fallback)
- [x] Input transcription
- [x] Output transcription
- [x] Interruption handling
- [x] Context window compression
- [x] Session resumption
