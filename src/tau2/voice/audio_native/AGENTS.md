# AGENTS.md — src/tau2/voice/audio_native/

> See `README.md` for full architecture, adapter patterns, and event types.

## General Rules for Working in This Directory

### Audio Format

The framework-wide default is telephony: **8kHz mono μ-law (G.711)**. All providers must convert to/from this format. Never change the framework-wide telephony constants — add conversion in the provider's `audio_utils.py` instead.

### TimeoutEvent Requirement

Every provider's `events.py` MUST define a `TimeoutEvent` with `type: Literal["timeout"]`. This synthetic event is emitted when no API events arrive during a tick's collection window. `TickResult.has_provider_activity` depends on it.

### Event Models Are Provider-Specific

When modifying or creating `events.py`, base models exclusively on that provider's API documentation. Each provider has unique event structures, field names, and nesting — do not assume they match another provider's format.

### Config Defaults

Add user-facing config defaults (model names, tick durations, etc.) to `src/tau2/config.py` — the single source of truth. Import constants from there; do not define local duplicates in provider modules.

### Shared Adapter Utilities

New adapters should reuse the shared building blocks instead of reimplementing them:

- **`BackgroundAsyncLoop`** (`async_loop.py`) — manages a background thread + asyncio event loop. Use `self._bg_loop = BackgroundAsyncLoop()` and call `start()`, `run_coroutine()`, `stop()` instead of manually creating threads and calling `asyncio.run_coroutine_threadsafe`.
- **`buffer_excess_audio()`** (`tick_result.py`) — caps `result.agent_audio_chunks` to `bytes_per_tick`, returns excess to buffer. Handles both normal and truncated (interruption) cases.
- **`get_proportional_transcript()`** (`tick_result.py`) — computes proportional transcript from audio chunks and `UtteranceTranscript` trackers. Accepts an optional `item_id_map` for providers where audio and text arrive under different IDs (e.g., Nova).

See the "Shared Adapter Utilities" section in `README.md` for usage examples.

### Modifying Existing Providers

When fixing bugs or updating an existing provider:
- Check the provider's latest API documentation for any breaking changes.
- Run the provider's tests: `{PROVIDER}_TEST_ENABLED=1 pytest tests/test_voice/test_audio_native/test_<provider>/ -v`
- If you change the adapter's tick behavior, also verify with an E2E run: `tau2 run --domain retail --audio-native --audio-native-provider <name> --task-ids 0 --verbose-logs`

---

## Adding a New Audio Native Provider

When integrating a new real-time voice API, follow this structured approach.

### Provider Architecture Types

Before starting, identify which architecture type the provider uses:

| Type | Description | Examples |
|------|-------------|----------|
| **Native Multimodal** | Single model processes audio input → generates audio output directly | OpenAI Realtime, Gemini Live, xAI Grok Voice |
| **Cascaded** | Orchestrated pipeline: STT → LLM → TTS (may be exposed as single API) | Deepgram Voice Agent, Amazon Nova Sonic, LiveKit |

Both types are valid. Document the type in the provider's `__init__.py` docstring.

### (A) Requirements Check

Before implementation, verify the API supports **all** of these capabilities:

1. **Bidirectional audio streaming** - Real-time send/receive audio over WebSocket or similar persistent connection
2. **Audio input processing** - Either native audio understanding OR high-quality STT
3. **Audio output generation** - Either native audio generation OR high-quality TTS
4. **Tool/function calling** - Ability to invoke tools with structured arguments during conversation
5. **Voice Activity Detection (VAD)** - Server-side detection of when user starts/stops speaking
6. **Input transcription** - Provides text transcription of user's speech
7. **Output transcription** - Provides text transcription of model's/agent's speech (critical for verification)
8. **Session persistence** - Maintains session state across multiple audio exchanges without reconnecting
9. **Reasonable latency** - Response time suitable for real-time conversation (sub-second first audio)

If the API lacks any critical capability, document the limitation and assess whether a workaround exists.

### (B) Implementation Steps

#### Directory Structure

```
src/tau2/voice/audio_native/{provider_name}/
├── __init__.py              # Exports, architecture type docstring
├── events.py                # Pydantic models for API events
├── provider.py              # WebSocket client, session management
├── audio_utils.py           # Format conversion (if needed)
└── discrete_time_adapter.py # DiscreteTimeAdapter implementation

tests/test_voice/test_audio_native/test_{provider_name}/
├── __init__.py
└── test_provider.py         # Provider integration tests
```

#### Step 1: Research API Documentation (CRITICAL)

**DO NOT skip or rush this step.** Read the official provider documentation thoroughly for:

- [ ] **Exact WebSocket endpoint URL** (including version path like `/v1/`)
- [ ] **Authentication method** (API key in header, query param, or connection message)
- [ ] **Audio format requirements** (sample rate, encoding, mono/stereo)
- [ ] **Session configuration message format** (exact field names, nesting structure)
- [ ] **All event types** the API can send (with exact JSON structure)
- [ ] **Tool/function calling format** - both request and response structures
- [ ] **Error event format** and error codes

**Common pitfalls from past implementations:**
- WebSocket URL missing version prefix or path segments
- Settings/config fields have unexpected names or nesting (always verify exact field names in docs)
- Tool schemas may need different format than OpenAI standard (wrapping, unwrapping, or flattening)
- Function call events may be arrays, single objects, or have different field names
- String constants (provider names, model names) may have unexpected formatting

#### Step 2: Define Event Models (`events.py`)

Create Pydantic models for **all** API response types. Base these **exclusively** on provider documentation, not by copying other providers.

Required event categories (exact names vary by provider - check their docs):
- **Connection**: Session established, configuration acknowledged
- **Transcription**: Text versions of user speech and agent speech
- **Audio**: Audio data/chunks from agent
- **Turn-taking**: Agent started/stopped speaking, user started/stopped speaking
- **Tool calls**: Function/tool invocation requests from agent
- **Errors**: Error messages with codes/descriptions
- **TimeoutEvent** - MUST include `type: Literal["timeout"]` for `TickResult.has_provider_activity`

Use `Field(alias="camelCase")` for camelCase→snake_case mapping.

#### Step 3: Implement Provider Class (`provider.py`)

Async WebSocket client with these methods:
- `connect()` / `disconnect()` - Connection lifecycle
- `configure_session(...)` - Send settings/configuration message
- `send_audio(bytes)` - Stream audio chunks to provider
- `receive_events()` / `receive_events_for_duration(seconds)` - Get parsed events
- `send_tool_result(call_id, ...)` - Return function call results

**Environment variable naming**: `{PROVIDER}_API_KEY` (e.g., `DEEPGRAM_API_KEY`)

#### Step 4: Write Provider Tests (`test_provider.py`)

Location: `tests/test_voice/test_audio_native/test_{provider_name}/test_provider.py`

**Test data**: Use shared audio files in `tests/test_voice/test_audio_native/testdata/`
- Run `generate_test_audio.py` to create new test audio if needed
- Audio should be 16kHz mono PCM WAV (convert in test if provider needs different format)

**Required test coverage** (all must pass before proceeding):

```python
# Enable with environment variable: {PROVIDER}_TEST_ENABLED=1

class TestProviderConnection:
    async def test_connect_disconnect()           # Basic lifecycle
    async def test_connect_with_invalid_key()     # Auth error handling

class TestProviderConfiguration:
    async def test_configure_session()            # Basic config works
    async def test_configure_with_tools()         # Tool schema accepted
    # Cascaded providers only: test LLM/TTS provider configuration
    async def test_configure_with_openai_llm()    # (if provider supports multiple LLMs)

class TestProviderAudioSend:
    async def test_send_audio_chunks()            # Audio accepted without error

class TestProviderAudioReceive:
    async def test_receive_audio_response()       # Get audio back from agent
                                                  # (trigger via greeting, prompt, or user audio)

class TestProviderTranscription:
    async def test_receive_agent_transcript()     # Text version of agent speech
    async def test_receive_user_transcript()      # Text version of user speech

class TestProviderToolFlow:
    async def test_tool_call_round_trip()         # Full cycle:
                                                  # 1. Send audio triggering tool
                                                  # 2. Receive tool call request
                                                  # 3. Send tool result
                                                  # 4. Receive agent response
```

**Run tests with:**
```bash
{PROVIDER}_TEST_ENABLED=1 pytest tests/test_voice/test_audio_native/test_{provider_name}/ -v
```

#### Step 5: Implement Discrete-Time Adapter (`discrete_time_adapter.py`)

Bridge provider to `DiscreteTimeAdapter` interface:
- Audio format conversion (telephony 8kHz μ-law ↔ provider format)
- Audio buffering for tick-based processing
- Proportional transcript distribution across ticks
- Tool call coordination

Use the shared utilities instead of reimplementing common logic:
- `BackgroundAsyncLoop` from `async_loop.py` for the background event loop
- `buffer_excess_audio()` from `tick_result.py` for audio capping/buffering
- `get_proportional_transcript()` from `tick_result.py` for transcript distribution

Reference: `src/tau2/voice/audio_native/adapter.py` for interface definition. Study an existing adapter (e.g., `xai/discrete_time_adapter.py`) for the standard tick lifecycle pattern.

#### Step 6: Add Audio Conversion Utilities (`audio_utils.py`)

Only needed if provider doesn't support 8kHz μ-law (G.711). Convert between:
- Telephony format: 8kHz mono μ-law
- Provider format: Usually 16kHz/24kHz mono linear16 PCM

#### Step 7: Register in Adapter Factory

Add the new adapter to the `create_adapter()` factory function in `src/tau2/voice/audio_native/adapter.py`. This is the **single entry point** for adapter construction — `DiscreteTimeAudioNativeAgent` delegates to it.

1. Add a new `elif provider == "{provider_name}":` branch in `create_adapter()` with a lazy import of your adapter class (inside the branch body, to avoid circular imports).
2. Pass the relevant parameters to your adapter's constructor. If the provider doesn't support model selection, add it to `_PROVIDERS_WITHOUT_MODEL_SELECTION` at the top of the file so users get a warning.
3. Add CLI option: `--audio-native-provider {provider_name}` in `cli.py`.
4. **Add user-facing config defaults to `src/tau2/config.py`** (single source of truth):
   - Add the default model to `DEFAULT_AUDIO_NATIVE_MODELS` dict.
   - Add the provider name to `AUDIO_NATIVE_PROVIDER_TYPES`.
   - Import these constants in your `provider.py` instead of defining local duplicates.

**Do NOT** add adapter construction or parameter validation logic in `DiscreteTimeAudioNativeAgent` — all of that belongs in `create_adapter()`.

#### Step 8: End-to-End Testing

Run with retail domain:
```bash
tau2 run --domain retail --audio-native  \
  --audio-native-provider {provider_name} \
  --task-ids 0 --verbose-logs --max-steps-seconds 60 \
  --save-to {provider_name}_retail_test
```

**Verify in output files** (`data/simulations/{save_to}/tasks/task_0/*/`):
- [ ] Multi-turn conversation (user ↔ agent back-and-forth)
- [ ] Audio present in simulation result
- [ ] Transcripts present for both user and agent
- [ ] At least 1 tool call made and result received
- [ ] Conversation continues correctly after tool call

#### Step 9: Test with Airline Domain (Schema Refs)

Airline domain uses `$ref/$defs` in tool schemas. Verify:
- Provider handles JSON Schema references correctly
- Or implement manual schema flattening in `_format_tools_for_api()`

### Debugging Tips

1. **WebSocket 404/401**: Verify exact URL path and authentication method (header vs query param vs message)
2. **"Error parsing message"**: Configuration/settings message format is wrong - compare field-by-field with provider docs
3. **Tool calls not working**: Verify tool schema format matches provider's expectations (OpenAI format, unwrapped, or custom)
4. **No transcripts received**: Check event parsing for the provider's transcript event type and field names
5. **Tool response rejected**: Verify exact field names for tool result message (varies: `id`/`call_id`, `content`/`output`/`result`)
6. **Audio not received**: Confirm audio format matches provider requirements (sample rate, encoding, channels)

### Reference Implementations

Study existing providers as patterns (but always verify against target provider's docs):
- **Native multimodal**: `openai/`, `gemini/`, `xai/`
- **Cascaded**: `deepgram/`, `livekit/`
