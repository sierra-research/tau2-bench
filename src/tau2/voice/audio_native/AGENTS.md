# AGENTS.md — src/tau2/voice/audio_native/

> See `README.md` for architecture overview and provider list.

## Rules for Working in This Directory

1. **Use the template method.** New providers should implement `_execute_tick()` and `_flush_pending_tool_results()` on the `DiscreteTimeAdapter` base class, not their own `_async_run_tick()`. The base class handles buffering, transcript, timing, and state.

2. **Use `StreamingTelephonyConverter`.** Audio format conversion lives in `audio_converter.py`. Don't create per-provider converter classes.

3. **Import constants from `config.py` directly.** Don't re-declare them under new names in provider code.

4. **Run the provider suite after changes.** `uv run tests/test_voice/test_audio_native/run_provider_suite.py` tests all available providers through the `DiscreteTimeAdapter` interface.

5. **Provider-specific event handling goes in `_process_event()`.** The provider's `_execute_tick()` should call `_process_event()` for each event. Audio chunks, tool calls, VAD events, and utterance transcripts are populated there.

6. **LiveKit and Pipecat are different.** Both use cascaded pipelines (STT→LLM→TTS) with their own `run_tick()` and don't use the template method. This is intentional — the interaction model is fundamentally different from WebSocket-based providers. The Pipecat adapter wraps a Pipecat `Pipeline` and bridges it to the tick loop via an in-memory `QueueTransport`.

## File Structure

```
adapter.py              — DiscreteTimeAdapter base class + create_adapter() factory
audio_converter.py      — StreamingTelephonyConverter (shared, parameterized by sample rates)
tick_result.py          — TickResult, UtteranceTranscript, buffer_excess_audio, get_proportional_transcript
async_loop.py           — BackgroundAsyncLoop (sync/async bridge for adapters)

<provider>/
  provider.py           — API client (WebSocket, connect, send, receive)
  events.py             — Pydantic event models + parse function
  discrete_time_adapter.py — _execute_tick, _flush_pending_tool_results, connect/disconnect
```
