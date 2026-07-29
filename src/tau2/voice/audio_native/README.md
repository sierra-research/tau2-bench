# Audio Native Providers

Full-duplex voice evaluation via provider-specific realtime APIs. Each provider connects to a different voice AI service and exposes the same `DiscreteTimeAdapter` interface to the simulation framework.

## Supported Providers

| Provider | Type | API | Model |
|----------|------|-----|-------|
| **openai** | Native audio | OpenAI Realtime API | gpt-realtime-1.5 |
| **gptlive** | Native audio (full-duplex) | GPT-Live API (confidential alpha) | gpt-live-1-boulder-alpha |
| **gemini** | Native audio | Google Gemini Live | gemini-3.1-flash-live-preview |
| **xai** | Native audio | xAI Grok Voice Agent | xai-realtime |
| **nova** | Native audio | Amazon Nova Sonic | amazon.nova-2-sonic-v1:0 |
| **qwen** | Native audio | Alibaba Qwen Omni | qwen3-omni-flash-realtime |
| **livekit** | Cascaded (STT→LLM→TTS) | LiveKit + Deepgram + OpenAI | Configurable |

## Architecture

```
DiscreteTimeAdapter (adapter.py)          ← shared base class
├── _async_run_tick()                     ← template method: buffering, transcript, timing
├── _send_audio_chunked()                 ← shared VoIP send helper
├── StreamingTelephonyConverter           ← shared audio format conversion (audio_converter.py)
│
├── gemini/discrete_time_adapter.py       ← implements _execute_tick + _flush_pending_tool_results
├── openai/discrete_time_adapter.py
├── gptlive/discrete_time_adapter.py
├── xai/discrete_time_adapter.py
├── qwen/discrete_time_adapter.py
├── nova/discrete_time_adapter.py
└── livekit/discrete_time_adapter.py      ← own run_tick (cascaded pipeline, different interaction model)
```

Each provider has:
- `provider.py` — WebSocket/API client (connect, send audio, receive events)
- `events.py` — Pydantic models for provider-specific event types
- `discrete_time_adapter.py` — Implements `_execute_tick()`: convert audio, send, receive, process events

The base class handles: TickResult creation, audio buffering/capping, proportional transcript distribution, barge-in buffer clearing, tool result queuing, tick timing enforcement, and cumulative state tracking.

## Adding a New Provider

1. Create `provider.py` with `connect()`, `send_audio()`, `receive_events_for_duration()`, `send_tool_result()`
2. Create `events.py` with typed event models and a `parse_*_event()` function
3. Create `discrete_time_adapter.py` extending `DiscreteTimeAdapter`:
   - `__init__`: set up converter, chunk size, provider
   - `_execute_tick`: convert audio, send + receive concurrently, process events
   - `_flush_pending_tool_results`: send queued tool results to API
   - `connect` / `disconnect` / `run_tick` / `is_connected`: lifecycle (use `BackgroundAsyncLoop`)
4. Register in `adapter.py` `create_adapter()` factory
5. Add to `config.py` (model, sample rates, endpoints)
6. Run `test_provider_suite.py` to validate

## Testing

```bash
# Run suite for all available providers
uv run tests/test_voice/test_audio_native/run_provider_suite.py

# Run for a specific provider
uv run pytest tests/test_voice/test_audio_native/test_provider_suite.py -v -k gemini
```

See `test_provider_suite.py` docstring for architecture details.
