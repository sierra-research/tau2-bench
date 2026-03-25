# Audio Native Architecture

This module provides end-to-end audio processing via provider-specific realtime APIs. Unlike the traditional voice pipeline (STT → LLM → TTS), audio native APIs process audio directly without intermediate text transcription steps.

**Supported Providers:**
- **OpenAI Realtime API** - WebSocket-based, G.711 μ-law audio
- **Google Gemini Live** - google-genai SDK, PCM16 audio with session resumption
- **xAI Grok Voice Agent** - WebSocket-based, native G.711 μ-law (no conversion needed)

**Experimental/Internal Providers** (not exposed via `--audio-native-provider` CLI):
- **Amazon Nova Sonic** - AWS Bedrock bidirectional stream, LPCM audio with SPECULATIVE/FINAL generation
- **Qwen Omni Flash** - DashScope WebSocket, PCM16 audio (⚠️ tool calling broken)
- **Deepgram Voice Agent** - Cascaded STT→LLM→TTS with BYO LLM/TTS support

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Audio Native Stack                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Agent Layer                                  │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  DiscreteTimeAudioNativeAgent                                 │  │   │
│  │  │  (Tick-based, discrete-time simulation)                       │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Adapter Layer                                 │   │
│  │                                                                      │   │
│  │  Abstract Interface (adapter.py):                                    │   │
│  │  ┌─────────────────────────────────┐                                │   │
│  │  │  DiscreteTimeAdapter            │                                │   │
│  │  │  (Tick-based simulation)        │                                │   │
│  │  └─────────────────────────────────┘                                │   │
│  │              │                                                       │   │
│  │              ▼                                                       │   │
│  │  OpenAI Implementation (openai/):                                    │   │
│  │  ┌─────────────────────────────────┐                                │   │
│  │  │ DiscreteTimeAudioNativeAdapter  │                                │   │
│  │  └─────────────────────────────────┘                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Provider Layer (openai/)                       │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  OpenAIRealtimeProvider                                      │   │   │
│  │  │  - WebSocket connection to wss://api.openai.com/v1/realtime  │   │   │
│  │  │  - Async send/receive                                        │   │   │
│  │  │  - Session configuration                                     │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
audio_native/
├── README.md                    # This file
├── __init__.py                  # Re-exports from openai/
├── adapter.py                   # Abstract base class + factory:
│                                #   - DiscreteTimeAdapter (tick-based)
│                                #   - create_adapter() factory function
├── async_loop.py                # BackgroundAsyncLoop helper class
├── tick_result.py               # TickResult, UtteranceTranscript,
│                                #   buffer_excess_audio(), get_proportional_transcript()
├── openai/
│   ├── __init__.py              # Public exports for OpenAI implementation
│   ├── provider.py              # OpenAIRealtimeProvider (async WebSocket)
│   ├── discrete_time_adapter.py # DiscreteTimeAudioNativeAdapter (tick-based)
│   ├── tick_runner.py           # TickRunner (OpenAI-specific tick logic)
│   ├── events.py                # Pydantic event models
│   └── README.md                # OpenAI-specific documentation
├── nova/
│   ├── __init__.py              # Public exports for Nova Sonic implementation
│   ├── provider.py              # NovaSonicProvider (AWS bidirectional stream)
│   ├── discrete_time_adapter.py # DiscreteTimeNovaAdapter (tick-based)
│   ├── events.py                # Pydantic event models (with generationStage)
│   ├── audio_utils.py           # Audio format conversion utilities
│   └── README.md                # Nova-specific documentation
├── gemini/
│   ├── __init__.py              # Public exports for Gemini implementation
│   ├── provider.py              # GeminiLiveProvider (google-genai SDK client)
│   ├── discrete_time_adapter.py # DiscreteTimeGeminiAdapter (tick-based)
│   ├── events.py                # Pydantic event models
│   ├── audio_utils.py           # Audio format conversion utilities
│   └── README.md                # Gemini-specific documentation
├── xai/
│   ├── __init__.py              # Public exports for xAI implementation
│   ├── provider.py              # XAIRealtimeProvider (WebSocket client)
│   ├── discrete_time_adapter.py # DiscreteTimeXAIAdapter (tick-based)
│   ├── events.py                # Pydantic event models
│   └── README.md                # xAI-specific documentation
├── qwen/
│   ├── __init__.py              # Public exports for Qwen implementation
│   ├── provider.py              # QwenRealtimeProvider (DashScope WebSocket)
│   ├── discrete_time_adapter.py # DiscreteTimeQwenAdapter (tick-based)
│   ├── events.py                # Pydantic event models (OpenAI-compatible)
│   ├── audio_utils.py           # Audio format conversion utilities
│   └── README.md                # Qwen-specific documentation
└── deepgram/
    ├── __init__.py              # Public exports for Deepgram implementation
    ├── provider.py              # DeepgramVoiceAgentProvider (WebSocket)
    ├── discrete_time_adapter.py # DiscreteTimeDeepgramAdapter (tick-based)
    ├── events.py                # Pydantic event models
    ├── audio_utils.py           # Audio format conversion utilities
    └── README.md                # Deepgram-specific documentation
```

---

## Adapter Pattern

All audio native providers implement the tick-based `DiscreteTimeAdapter` pattern, where audio time is the primary clock:

```python
for tick in range(max_ticks):
    result = adapter.run_tick(user_audio_chunk, tick_number=tick)
    # result.agent_audio_data - agent audio for this tick
    # result.proportional_transcript - text for this tick
```

---

## Components

### 1. Abstract Base Classes (`adapter.py`)

#### `DiscreteTimeAdapter` (ABC)

Provider-agnostic interface for tick-based discrete-time simulation:

```python
class DiscreteTimeAdapter(ABC):
    def connect(self, system_prompt, tools, vad_config, modality="audio") -> None
    def disconnect(self) -> None
    def is_connected -> bool
    
    # Primary method - one tick of simulation
    def run_tick(self, user_audio: bytes, tick_number: int) -> TickResult
    
    # Tool handling (queued for next tick)
    def send_tool_result(self, call_id, result, request_response=True) -> None
```

The key difference is that `DiscreteTimeAdapter.run_tick()` handles one complete tick:
- Sends user audio
- Collects events for tick duration
- Returns agent audio (capped to tick duration)
- Returns proportional transcript

#### `create_adapter()` (Factory Function)

Creates the correct adapter subclass for a given provider, handling parameter validation and model default resolution in one place. This is the single entry point for adapter construction — `DiscreteTimeAudioNativeAgent` delegates to it, and new providers must register here.

```python
from tau2.voice.audio_native import create_adapter

adapter, resolved_model = create_adapter(
    provider="gemini",
    tick_duration_ms=1000,
    model=None,  # uses provider default
)
```

The factory:
1. **Validates parameters** — raises `ValueError` if provider-specific parameters are used with incompatible providers.
2. **Resolves model defaults** — uses `DEFAULT_AUDIO_NATIVE_MODELS[provider]` when no model is given, or the `CascadedConfig` default for livekit.
3. **Warns on unsupported model selection** — logs a warning if `model` is provided for providers that ignore it (xai, nova, qwen).
4. **Constructs the adapter** — returns `(adapter, resolved_model)` so callers also get the effective model name.

---

### 2. Provider Implementations

See provider-specific READMEs for detailed documentation:
- [OpenAI README](openai/README.md)
- [Nova README](nova/README.md)

#### OpenAI Implementation (`openai/`)

#### `OpenAIRealtimeProvider` (`provider.py`)

Low-level async WebSocket client for OpenAI Realtime API.

**Connection:**
```python
url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
headers = {
    "Authorization": "Bearer {api_key}",
    "OpenAI-Beta": "realtime=v1"
}
```

**Session Configuration:**
```python
await provider.configure_session(
    system_prompt="You are a helpful assistant...",
    tools=[tool1, tool2],
    vad_config=OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD),
    modality="audio"  # or "text", "audio_in_text_out"
)
```

**VAD (Voice Activity Detection) Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `SERVER_VAD` | Server detects speech end via silence threshold | Full-duplex with auto turn detection |
| `SEMANTIC_VAD` | Server uses semantic understanding | Natural conversation flow |
| `MANUAL` | No automatic detection | Client controls turns explicitly |

**VAD Configuration:**
```python
@dataclass
class OpenAIVADConfig:
    mode: OpenAIVADMode = SERVER_VAD
    threshold: float = 0.5           # Speech detection sensitivity
    prefix_padding_ms: int = 300     # Audio to include before speech
    silence_duration_ms: int = 500   # Silence before turn end
    eagerness: str = "medium"        # For semantic_vad mode
```

**Modalities:**

| Modality | Input | Output | API Config |
|----------|-------|--------|------------|
| `"text"` | Text | Text | `modalities: ["text"]` |
| `"audio"` | Audio (G.711 μ-law) | Audio (G.711 μ-law) | `modalities: ["text", "audio"]` |
| `"audio_in_text_out"` | Audio | Text | `modalities: ["text"]` + audio input config |

---

#### `DiscreteTimeAudioNativeAdapter` (`discrete_time_adapter.py`)

Tick-based adapter for discrete-time simulation. Implements `DiscreteTimeAdapter`.

**Key Features:**
- **Tick-based interface**: `run_tick()` instead of request-response
- **Audio capping**: Agent audio capped to `bytes_per_tick` per tick
- **Audio buffering**: Excess agent audio carries to next tick
- **Proportional transcript**: Text distributed based on audio played
- **Interruption handling**: Client-side truncation on `SpeechStartedEvent`

**Usage:**
```python
from tau2.voice.audio_native.openai import (
    DiscreteTimeAudioNativeAdapter,
    OpenAIVADConfig,
    OpenAIVADMode,
)

adapter = DiscreteTimeAudioNativeAdapter(
    tick_duration_ms=1000,
)

adapter.connect(
    system_prompt="You are a helpful assistant.",
    tools=tools,
    vad_config=OpenAIVADConfig(mode=OpenAIVADMode.SERVER_VAD),
    modality="audio",
)

for tick in range(max_ticks):
    result = adapter.run_tick(user_audio_chunk, tick_number=tick)
    
    # Raw audio (for speech detection)
    if result.agent_audio_bytes > 0:
        print(f"Agent spoke: {result.agent_audio_bytes} bytes")
    
    # Padded audio (for time-aligned playback)
    played_audio = result.get_played_agent_audio()  # Exactly bytes_per_tick
    
    # Proportional transcript
    print(f"Text: {result.proportional_transcript}")

adapter.disconnect()
```

---

#### `TickResult` (`tick_result.py`) and `TickRunner` (`openai/tick_runner.py`)

Core components for tick-based simulation:

**`TickResult`** - Result of a single tick:

| Field | Type | Description |
|-------|------|-------------|
| `tick_number` | `int` | 1-indexed tick number |
| `user_audio_data` | `bytes` | User audio sent this tick |
| `agent_audio_chunks` | `list` | Raw agent audio chunks |
| `proportional_transcript` | `str` | Text for audio played this tick |
| `events` | `list[BaseRealtimeEvent]` | All API events received |
| `was_truncated` | `bool` | True if agent was interrupted |
| `bytes_per_tick` | `int` | Expected audio bytes per tick |

**Key Methods:**
- `agent_audio_data` (property): Raw unpadded agent audio
- `agent_audio_bytes` (property): Raw audio byte count
- `get_played_agent_audio()`: Returns exactly `bytes_per_tick` bytes, padded with silence

**`UtteranceTranscript`** - Proportional text distribution:

Distributes transcript text proportionally across audio duration, so text appears at roughly the same rate as speech.

**`TickRunner`** - Manages tick-by-tick simulation and timing-aligned buffering.

---

#### Event Types (`events.py`)

Pydantic models for all OpenAI Realtime API events:

**Response Streaming Events:**
| Event | Description |
|-------|-------------|
| `TextDeltaEvent` | Incremental text content |
| `AudioDeltaEvent` | Incremental audio data (base64) |
| `AudioTranscriptDeltaEvent` | Incremental transcript of audio output |
| `AudioDoneEvent` | Audio for item complete |
| `AudioTranscriptDoneEvent` | Transcript for item complete |
| `ResponseDoneEvent` | Response complete, includes usage stats |

**Function Call Events:**
| Event | Description |
|-------|-------------|
| `OutputItemAddedEvent` | New output item (message or function_call) |
| `FunctionCallArgumentsDeltaEvent` | Incremental function arguments |
| `FunctionCallArgumentsDoneEvent` | Function call complete |

**Speech Detection Events:**
| Event | Description |
|-------|-------------|
| `SpeechStartedEvent` | User started speaking (includes `audio_start_ms`) |
| `SpeechStoppedEvent` | User stopped speaking |
| `InputAudioTranscriptionCompletedEvent` | Transcription of user input ready |

**Event Parsing:**
```python
from tau2.voice.audio_native.openai import parse_realtime_event

raw_data = {"type": "response.text.delta", "delta": "Hello"}
event = parse_realtime_event(raw_data)  # Returns TextDeltaEvent
```

**Timeout Event Convention:**

All providers must include a `TimeoutEvent` class with `type: Literal["timeout"]`. This is a synthetic event emitted when no API events arrive during a tick's collection window. The `TickResult.has_provider_activity` property uses this to detect provider stalls - it returns `True` if any event with `type != "timeout"` was received.

```python
class TimeoutEvent(BaseRealtimeEvent):
    type: Literal["timeout"]
```

---

#### Nova Sonic Implementation (`nova/`)

Amazon Nova Sonic API provider via AWS Bedrock.

**Key Differences from OpenAI:**
- Uses AWS SigV4 authentication (boto3 credentials)
- Audio format: 16kHz PCM16 input, 24kHz PCM16 output (converted to/from telephony)
- **SPECULATIVE/FINAL generation**: Nova generates speculative content that may be revised before committing as FINAL
- Only SERVER_VAD mode supported

**SPECULATIVE vs FINAL:**

Nova Sonic uses speculative generation - it starts generating content before fully committing:

```python
# contentStart events include generationStage in additionalModelFields:
# {"generationStage": "SPECULATIVE"}  - may be revised, ignored
# {"generationStage": "FINAL"}        - committed, processed

# The adapter filters:
if event.generation_stage == "FINAL":
    self._final_content_ids.add(event.content_id)

# Audio/text from SPECULATIVE content_ids is ignored
if content_id not in self._final_content_ids:
    return  # Skip speculative content
```

**Usage:**
```python
from tau2.voice.audio_native.nova import (
    DiscreteTimeNovaAdapter,
    NovaVADConfig,
)

adapter = DiscreteTimeNovaAdapter(
    tick_duration_ms=50,
    voice="tiffany",  # matthew, tiffany, amy
)

adapter.connect(
    system_prompt="You are a helpful assistant.",
    tools=tools,
    vad_config=NovaVADConfig(),
)

for tick in range(max_ticks):
    result = adapter.run_tick(user_audio_chunk, tick_number=tick)
    # Audio is automatically converted to telephony format

adapter.disconnect()
```

See [Nova README](nova/README.md) for detailed documentation.

---

## Agent Integration

### `DiscreteTimeAudioNativeAgent`

Tick-based simulation for discrete-time full-duplex interaction.

```python
from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent

agent = DiscreteTimeAudioNativeAgent(
    tools=tools,
    domain_policy="...",
    tick_duration_ms=1000,
    modality="audio",
)

state = agent.get_init_state()

for tick in range(max_ticks):
    chunk, state = agent.get_next_chunk(state, user_audio_chunk)
    # chunk contains agent audio (capped) + text (proportional)
    # state.pending_tool_calls if tool execution needed
```

**Key Features:**
- Each `get_next_chunk()` call = one tick of simulation
- Agent audio capped to tick duration, excess buffered
- Proportional transcript distribution
- Tool calls detected and returned to orchestrator

---

## Audio Format

OpenAI Realtime API uses **G.711 μ-law** (8kHz telephony standard):

```python
from tau2.data_model.audio import TELEPHONY_SAMPLE_RATE

# TELEPHONY_AUDIO_FORMAT:
#   sample_rate: 8000 Hz
#   channels: 1
#   bytes_per_sample: 1
#   encoding: μ-law
```

**Timing Calculations:**
```python
tick_duration_ms = 1000
bytes_per_tick = TELEPHONY_SAMPLE_RATE * tick_duration_ms / 1000  # = 8000 bytes
```

---

## Error Handling

Errors are surfaced through:

1. **`ErrorEvent`**: API-level errors (rate limits, invalid requests). Each provider defines an `ErrorEvent` in its `events.py`. These appear in `TickResult.events` after `run_tick()`.
2. **`RuntimeError`**: Connection/threading issues (e.g., WebSocket failures, background loop errors).

```python
result = adapter.run_tick(user_audio_chunk, tick_number=tick)
for event in result.events:
    if hasattr(event, "type") and event.type == "error":
        logger.error(f"API error: {event.message}")
```

---

## Shared Adapter Utilities

Most discrete-time adapters (xAI, Qwen, Gemini, Nova, Deepgram) share the same
tick lifecycle logic.  Common building blocks are factored into two modules so
new providers can reuse them instead of copying boilerplate.

### `BackgroundAsyncLoop` (`async_loop.py`)

Manages a daemon thread running an `asyncio` event loop.  Adapters that expose
a synchronous interface (`connect`, `run_tick`, `disconnect`) but talk to an
async provider use this to bridge the two worlds:

```python
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop

self._bg_loop = BackgroundAsyncLoop()
self._bg_loop.start()

# Schedule an async coroutine and block for the result:
result = self._bg_loop.run_coroutine(some_async_fn(), timeout=30.0)

self._bg_loop.stop()
```

`run_coroutine()` wraps the common `asyncio.run_coroutine_threadsafe` +
`future.result(timeout=...)` pattern into a single call.

### `buffer_excess_audio()` (`tick_result.py`)

Caps `result.agent_audio_chunks` to `bytes_per_tick` and returns the excess
for the caller to buffer until the next tick.  When the tick was truncated
(interruption), excess is discarded instead.

```python
from tau2.voice.audio_native.tick_result import buffer_excess_audio

self._buffered_agent_audio = buffer_excess_audio(result, self.bytes_per_tick)
```

### `get_proportional_transcript()` (`tick_result.py`)

Computes the proportional transcript for the audio chunks kept in this tick.
An optional `item_id_map` handles providers (Nova) where audio and text arrive
under different content IDs.

```python
from tau2.voice.audio_native.tick_result import get_proportional_transcript

# Most providers:
result.proportional_transcript = get_proportional_transcript(
    result.agent_audio_chunks, self._utterance_transcripts
)

# Nova (audio and text use different content IDs):
result.proportional_transcript = get_proportional_transcript(
    result.agent_audio_chunks, self._utterance_transcripts,
    item_id_map=self._audio_to_text_map,
)
```

---

## Adding New Providers

For detailed requirements and implementation steps, see [`.cursor/rules/audio-native-provider.md`](../../../.cursor/rules/audio-native-provider.md) and [AGENTS.md](AGENTS.md).

Quick overview:
1. Create `audio_native/<provider>/` directory
2. Implement `events.py` with provider-specific event models
3. Implement `provider.py` with async API client
4. Implement `discrete_time_adapter.py` extending `DiscreteTimeAdapter`
5. Add audio conversion utilities if needed (`audio_utils.py`)
6. Register the new adapter in `create_adapter()` in `adapter.py`
7. Add provider config defaults to `src/tau2/config.py`
8. Add CLI option in `cli.py`

The `create_adapter()` factory in `adapter.py` is the central place where all adapter construction and parameter validation lives. The `DiscreteTimeAudioNativeAgent` delegates to it, so new providers only need to be added to the factory — not to the agent itself.
