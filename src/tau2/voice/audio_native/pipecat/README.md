# Pipecat Cascaded Voice Provider

This module wraps [Pipecat](https://github.com/pipecat-ai/pipecat) — a real-time voice AI framework — into the τ-bench `DiscreteTimeAdapter` interface so the simulator can drive a Pipecat `Pipeline` tick-by-tick.

Like LiveKit, this is a **cascaded** STT → LLM → TTS pipeline (as opposed to a single audio-native model). Where LiveKit composes plugins manually, Pipecat composes plugins as a `Pipeline` of `FrameProcessors`, and we feed audio in/out via a custom in-memory transport.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  PipecatVoiceProvider (provider.py)                      │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                     pipecat.Pipeline                             │   │
│   │                                                                  │   │
│   │  QueueInputTransport ─▶ STT ─▶ ContextAggregator(user) ─▶ LLM    │   │
│   │                                                  │               │   │
│   │       QueueOutputTransport ◀ TTS ◀ ContextAggregator(asst) ◀─┘   │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                    PipecatEvent stream                           │   │
│   │   (SPEECH_STARTED, TRANSCRIPT_*, LLM_*, TTS_*, TOOL_CALL, …)     │   │
│   └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│            DiscreteTimePipecatAdapter (discrete_time_adapter.py)         │
│                                                                          │
│  • Tick-based interface for the simulator                                │
│  • μ-law 8 kHz ↔ PCM16 16 kHz / 24 kHz conversion                        │
│  • Audio buffering for tick alignment                                    │
│  • PipecatEvent → TickResult mapping                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Typed configs for STT (Deepgram, OpenAI), LLM (OpenAI, Anthropic), TTS (Cartesia, Deepgram, ElevenLabs, OpenAI), plus `PIPECAT_CONFIGS` presets. |
| `events.py` | `PipecatEvent` / `PipecatEventType` — neutral event types emitted by the provider, mirroring LiveKit's `CascadedEvent`. |
| `queue_transport.py` | In-memory `QueueInputTransport` / `QueueOutputTransport` — the bridge that lets the simulator push audio frames into Pipecat and read them back. |
| `provider.py` | `PipecatVoiceProvider` — builds the Pipeline, handles connect/disconnect, feeds audio in, drains frames out, registers tools, sends tool results. |
| `discrete_time_adapter.py` | `DiscreteTimePipecatAdapter` — runs the provider's pipeline on a background asyncio loop and exposes a synchronous `run_tick()` to the simulator. |

## Why a custom transport?

The simulator drives the pipeline in fixed `tick_duration_ms` chunks (default 200 ms μ-law 8 kHz audio). Pipecat's standard transports (Daily, WebRTC, FastAPI WebSocket) all assume a real-time external client. We need direct in-process audio I/O with deterministic timing, so `QueueTransport` exposes the same `BaseInputTransport` / `BaseOutputTransport` API but reads from / writes to `asyncio.Queue`s. Audio resampling between telephony (8 kHz μ-law) and PCM16 (16 / 24 kHz) is handled by `StreamingTelephonyConverter` from the shared `audio_converter.py`.

## Configuration

```python
from tau2.voice.audio_native.pipecat.config import (
    PipecatConfig, DeepgramSTTConfig, OpenAILLMConfig, CartesiaTTSConfig,
)

config = PipecatConfig(
    stt=DeepgramSTTConfig(model="nova-3"),
    llm=OpenAILLMConfig(model="gpt-4.1"),
    tts=CartesiaTTSConfig(voice_id="..."),
    enable_vad=True,
    allow_interruptions=True,
)
```

Or use a preset by name (in `PIPECAT_CONFIGS`):

| Preset | STT | LLM | TTS | Required keys |
|--------|-----|-----|-----|---------------|
| `default` | Deepgram nova-3 | OpenAI gpt-4.1 | Cartesia sonic-2 | `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY` |
| `openai-thinking` | Deepgram nova-3 | OpenAI gpt-5.2 (high reasoning) | Cartesia sonic-2 | `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY` |
| `openai-only` | OpenAI gpt-4o-transcribe | OpenAI gpt-4.1 | OpenAI gpt-4o-mini-tts | `OPENAI_API_KEY` |

## CLI usage

```bash
# Default preset
tau2 run --domain retail --audio-native --audio-native-provider pipecat \
  --num-tasks 1 --verbose-logs

# A specific preset
tau2 run --domain airline --audio-native --audio-native-provider pipecat \
  --pipecat-config openai-only --num-tasks 1
```

## Design Decisions

### 1. Reuse Pipecat's pipeline, don't reimplement

Pipecat already handles transcription, context aggregation, sentence-aware TTS chunking, interruption frame propagation, and graceful turn-taking. We let it own all of that and only translate the *output* frames into our `PipecatEvent` stream. This keeps the integration shallow and easy to upgrade.

### 2. Custom in-memory transport

A `QueueTransport` mimics Pipecat's standard transport interface so we can reuse `PipelineRunner`, `PipelineTask`, and the standard `OpenAILLMContext` aggregator. This gives us the same lifecycle semantics as a "real" transport but with full deterministic control over audio I/O.

### 3. Tools are registered with placeholder handlers

The simulator owns tool execution (so tools can be evaluated against the environment). When the LLM emits a tool call, we capture it as a `TOOL_CALL` event, the simulator runs it, and the result is delivered back into the LLM context via `send_tool_result()`. The tool handlers registered with Pipecat's `LLMService` are placeholders that simply yield the call to our event stream.

### 4. Lazy / guarded imports

All `pipecat.*` imports happen inside functions or under `TYPE_CHECKING`. The module loads cleanly without `pipecat-ai` installed; the import error is raised only when you actually try to construct a `PipecatVoiceProvider` (or run with `--audio-native-provider pipecat`). Install with:

```bash
uv sync --extra voice
```
