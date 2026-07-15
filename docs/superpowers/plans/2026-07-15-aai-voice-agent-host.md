# aai voice-agent host + tau2 `aai` provider — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the aai framework a "voice-agent host" (external harness supplies prompt + tool schemas per session; tool calls relay to the client) and add a tau2 `aai` audio-native provider that benchmarks it, with tool execution in tau2's environment.

**Architecture:** Phase A (repo `~/Code/aai`) adds host mode to the WS protocol + runtime. Phase B (repo `~/Code/tau2-bench`) adds a `DiscreteTimeAdapter`-based `aai` provider that injects the domain policy + tools, streams PCM16 audio (μ-law↔PCM16 conversion), and returns tool results.

**Tech Stack:** aai — TypeScript, zod, `ws`, vitest, the `@alexkroman1/aai/host` runtime. tau2 — Python 3.12+, `websockets`, Pydantic v2, pytest, `uv`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-15-aai-voice-agent-host-design.md`.
- aai WS endpoint: `ws://localhost:3000/websocket`. Client sends `config` first, then **raw binary PCM16** audio frames. Server streams **binary PCM16** audio + JSON events.
- aai audio: **PCM16, input 16000 Hz, output 24000 Hz** (`DEFAULT_STT_SAMPLE_RATE`/`DEFAULT_TTS_SAMPLE_RATE`).
- aai `ToolSchema` = `{ type:"function", name, description, parameters: <JSON Schema object> }`. Parameters are JSON Schema **directly** (no zod conversion).
- aai `ExecuteTool` signature: `(name, args, sessionId?, messages?, opts?: { signal?, toolCallId? }) => Promise<unknown>`.
- aai runtime hooks: `createRuntime({ agent, env, executeTool?, toolSchemas?, systemPrompt?/via agent, stt?, llm?, tts? })`; `toolSchemas` is **required when** `executeTool` is set.
- Host mode is gated behind env `AAI_ALLOW_HOST` (default **on** in the dev server, off otherwise).
- tau2 tool schema source: `tool.openai_schema` → `{"type":"function","function":{"name","description","parameters"}}`.
- tau2 telephony format is μ-law 8 kHz mono; conversion via `tau2.voice.utils.audio_preprocessing` (`convert_to_pcm16`, `convert_to_ulaw`, `resample_audio`).
- New protocol messages: client→server `tool_result { type:"tool_result", toolCallId, result, error? }`; client `config` gains optional `host { systemPrompt, greeting?, tools: ToolSchema[] }`.

---

# Phase A — aai host mode (repo `~/Code/aai/agent`)

### Task A1: Map the session config → runtime wiring (discovery + contract)

Unfamiliar-code discovery. Produces the exact integration contract the next tasks implement against. No production code changes.

**Files:** read `packages/aai/host/server.ts`, `packages/aai/host/ws-handler.ts`, `packages/aai/host/session-core.ts`, `packages/aai/host/runtime.ts` (`startSession`, `SessionStartOptions`, `RuntimeOptions`), `packages/aai-cli/_dev-server.ts`, `packages/aai-server/transport-websocket.ts`.

- [ ] **Step 1: Answer these questions in a report**, each with a `file:line` citation:
  1. Where is the inbound `config` message parsed for a WS session, and where is `runtime.startSession(ws, startOpts)` called?
  2. Does `SessionStartOptions` (or `startSession`) already allow per-session overrides of system prompt, `toolSchemas`, and `executeTool`? If yes, name the fields. If no, what is the smallest change to thread a per-session `{ systemPrompt, toolSchemas, executeTool }` into the session's pipeline/S2S transport (`buildPipelineTransport` in `runtime.ts:382`)?
  3. How does an inbound WS JSON message get dispatched to a handler (so a new `tool_result` type can be routed), and how does a session send a JSON event to the client (to emit `tool_call`)?
  4. What does `_dev-server.ts` pass as the agent, and how would a session with `config.host` bypass the deployed agent?
  5. The zod schema location for inbound client messages (confirm it is `packages/aai-server/schemas.ts` vs a schema in `packages/aai`).

- [ ] **Step 2: Write the contract** to `~/Code/aai/agent/HOST_MODE_CONTRACT.md` (gitignored scratch): the chosen wiring point, the per-session override mechanism (field names or the new plumbing), and the dispatch/send functions for `tool_result`/`tool_call`. Subsequent tasks cite this.

No test. Deliverable = the contract doc.

---

### Task A2: Protocol schema — `config.host` + `tool_result`

**Files:**
- Modify: the inbound-client-message zod schema (confirmed in A1; likely `packages/aai-server/schemas.ts` and/or `packages/aai/sdk/_internal-types.ts`)
- Test: the sibling schema test (e.g. `packages/aai-server/schemas.test.ts`)

**Interfaces:**
- Produces: `HostConfigSchema` = `{ systemPrompt: string (min 1), greeting?: string, tools: ToolSchemaSchema[] }`; `config` message gains optional `host: HostConfigSchema`; new inbound message `{ type:"tool_result", toolCallId: string (min 1), result: string, error?: string }` added to the client→server union.

- [ ] **Step 1: Write the failing test** (adapt to the repo's vitest style; use the real schema import path from A1):

```ts
import { describe, it, expect } from "vitest";
import { ClientMessageSchema } from "./<path-from-A1>.ts"; // the inbound union

describe("host mode schema", () => {
  it("accepts config.host with tool schemas", () => {
    const msg = {
      type: "config", audioFormat: "pcm16", sampleRate: 16000, ttsSampleRate: 24000,
      host: {
        systemPrompt: "You are a retail agent.",
        tools: [{ type: "function", name: "get_order_details",
          description: "Look up an order.",
          parameters: { type: "object", properties: { order_id: { type: "string" } }, required: ["order_id"] } }],
      },
    };
    expect(ClientMessageSchema.parse(msg)).toMatchObject({ host: { systemPrompt: "You are a retail agent." } });
  });

  it("accepts tool_result", () => {
    expect(ClientMessageSchema.parse({ type: "tool_result", toolCallId: "c1", result: "{}" }))
      .toMatchObject({ type: "tool_result", toolCallId: "c1" });
  });

  it("rejects host with empty systemPrompt", () => {
    expect(() => ClientMessageSchema.parse({
      type: "config", audioFormat: "pcm16", sampleRate: 16000, ttsSampleRate: 24000,
      host: { systemPrompt: "", tools: [] },
    })).toThrow();
  });
});
```

- [ ] **Step 2: Run it, expect FAIL** (`config.host`/`tool_result` unknown). Command: the repo's test runner scoped to the schema test (e.g. `npx vitest run <schema>.test.ts` or `deno task test` — use what the repo's `package.json`/`CLAUDE.md` prescribes).

- [ ] **Step 3: Implement** — reuse the existing `ToolSchemaSchema` (`packages/aai/sdk/_internal-types.ts`) for `host.tools`; add `HostConfigSchema`; add `host: HostConfigSchema.optional()` to the config message; add the `tool_result` variant to the inbound union.

- [ ] **Step 4: Run it, expect PASS.**

- [ ] **Step 5: Commit** — `git commit -m "feat(host): config.host + tool_result protocol schema"` (on a new branch `feat/voice-agent-host` in the aai repo).

---

### Task A3: Host runtime — inject prompt/tools + relay executeTool

**Files:**
- Modify: the WS session wiring point (from A1) + a new helper `packages/aai/host/host-mode.ts` for the relay executor
- Test: `packages/aai/host/host-mode.test.ts`

**Interfaces:**
- Consumes: A2 schemas; `createRuntime`/`startSession`/`buildPipelineTransport` (per A1 contract); `ExecuteTool`, `ToolSchema`.
- Produces: `createRelayExecuteTool({ send, register, timeoutMs })` returning `{ executeTool: ExecuteTool, onToolResult(msg), dispose() }` where:
  - `executeTool(name, args, sessionId, messages, opts)` generates/uses `opts.toolCallId`, sends `{type:"tool_call", toolCallId, toolName:name, args}` via `send`, and returns a Promise stored by `toolCallId`.
  - `onToolResult({toolCallId, result, error})` resolves/rejects the stored Promise (JSON-parse `result`).
  - a `timeoutMs` (default 120000) rejects the Promise with a tool error and cleans up.
- A session with `config.host` present builds its pipeline with `systemPrompt = host.systemPrompt`, `toolSchemas = host.tools`, `executeTool = relay.executeTool`, gated by `AAI_ALLOW_HOST` (default on in dev). Inbound `tool_result` routes to `relay.onToolResult`.

- [ ] **Step 1: Write the failing test** for the relay executor (pure, no WS):

```ts
import { describe, it, expect, vi } from "vitest";
import { createRelayExecuteTool } from "./host-mode.ts";

describe("relay executeTool", () => {
  it("emits tool_call and resolves on matching tool_result", async () => {
    const sent: any[] = [];
    const relay = createRelayExecuteTool({ send: (m) => sent.push(m), timeoutMs: 1000 });
    const p = relay.executeTool("get_order_details", { order_id: "#W1" }, "s1", [], { toolCallId: "c1" });
    expect(sent[0]).toMatchObject({ type: "tool_call", toolCallId: "c1", toolName: "get_order_details", args: { order_id: "#W1" } });
    relay.onToolResult({ toolCallId: "c1", result: JSON.stringify({ status: "pending" }) });
    await expect(p).resolves.toMatchObject({ status: "pending" });
  });

  it("times out when no tool_result arrives", async () => {
    vi.useFakeTimers();
    const relay = createRelayExecuteTool({ send: () => {}, timeoutMs: 50 });
    const p = relay.executeTool("x", {}, "s1", [], { toolCallId: "c2" });
    vi.advanceTimersByTime(60);
    await expect(p).rejects.toThrow();
    vi.useRealTimers();
  });
});
```

- [ ] **Step 2: Run it, expect FAIL** (`host-mode.ts` missing).

- [ ] **Step 3: Implement `host-mode.ts`** — a `Map<string, {resolve, reject, timer}>`; `executeTool` creates the entry, sends the `tool_call`, returns the Promise; `onToolResult` looks up by `toolCallId`, clears the timer, JSON-parses `result` (or rejects on `error`); timeout rejects + deletes. Then wire it at the session point (A1): when `config.host` and `AAI_ALLOW_HOST`, build the session pipeline with the host prompt/tools/executeTool and route inbound `tool_result` to `relay.onToolResult`.

- [ ] **Step 4: Run it, expect PASS.**

- [ ] **Step 5: Integration test** — extend the repo's existing WS integration test (`packages/aai-server/ws-integration.test.ts` pattern): open a session with `config.host`, drive a fake LLM (the repo's test LLM provider) to call a tool, assert the client receives `tool_call` and that sending `tool_result` unblocks the turn (`reply_done`). If the repo's test harness can't easily force a tool call, assert the wiring path (host runtime built, `tool_result` routed) and note the live check is deferred to Task A4.

- [ ] **Step 6: Commit** — `git commit -m "feat(host): relay executeTool + per-session host prompt/tools"`.

---

### Task A4: Local host-mode smoke check

**Files:** none (verification). Optionally add `packages/aai-cli/scripts/host-smoke.mjs`.

- [ ] **Step 1:** Start the dev server (`npm run dev` / the repo's dev command; confirm `AAI_ALLOW_HOST` defaults on).
- [ ] **Step 2:** Connect a throwaway node/deno script to `ws://localhost:3000/websocket`, send `config` with a `host` block (systemPrompt "You are a test agent. When asked, call ping." + one tool `ping`), send a short PCM16 silence buffer, and log inbound events. Confirm: server accepts `config.host`, session becomes ready, and if the model calls `ping` you receive a `tool_call` and can reply with `tool_result`. Report the observed event sequence.
- [ ] **Step 3:** If host mode isn't reachable (gating/wiring), fix and re-run before declaring Phase A done.

---

# Phase B — tau2 `aai` provider (repo `~/Code/tau2-bench`)

### Task B1: Event models (`aai/events.py`)

**Files:**
- Create: `src/tau2/voice/audio_native/aai/__init__.py` (empty marker)
- Create: `src/tau2/voice/audio_native/aai/events.py`
- Test: `tests/test_voice/test_audio_native/test_aai_events.py`

**Interfaces:**
- Produces: `parse_aai_event(data: dict) -> BaseAAIEvent`; classes `AAIConfigEvent`, `AAISpeechStartedEvent`, `AAISpeechStoppedEvent`, `AAIUserTranscriptEvent{text, turn_order?}`, `AAIAgentTranscriptEvent{text}`, `AAIToolCallEvent{tool_call_id, tool_name, args}`, `AAIReplyDoneEvent`, `AAIAudioDoneEvent`, `AAICancelledEvent`, `AAIResetEvent`, `AAIIdleTimeoutEvent`, `AAIErrorEvent{code, message}`, `AAICustomEvent{event, data}`, `AAITimeoutEvent`, `AAIUnknownEvent`. (Binary audio frames are NOT parsed here — handled in the provider.)

- [ ] **Step 1: Write the failing test**

```python
from tau2.voice.audio_native.aai.events import (
    AAIAgentTranscriptEvent, AAIToolCallEvent, AAIErrorEvent, AAIUnknownEvent,
    parse_aai_event,
)

def test_parse_agent_transcript():
    ev = parse_aai_event({"type": "agent_transcript", "text": "hello"})
    assert isinstance(ev, AAIAgentTranscriptEvent) and ev.text == "hello"

def test_parse_tool_call_maps_fields():
    ev = parse_aai_event({"type": "tool_call", "toolCallId": "c1", "toolName": "get_order_details", "args": {"order_id": "#W1"}})
    assert isinstance(ev, AAIToolCallEvent)
    assert (ev.tool_call_id, ev.tool_name, ev.args) == ("c1", "get_order_details", {"order_id": "#W1"})

def test_parse_error():
    ev = parse_aai_event({"type": "error", "code": "llm", "message": "boom"})
    assert isinstance(ev, AAIErrorEvent) and ev.code == "llm"

def test_unknown():
    assert isinstance(parse_aai_event({"type": "nope"}), AAIUnknownEvent)
```

- [ ] **Step 2: Run, expect FAIL** — `uv run --extra dev --extra voice python -m pytest tests/test_voice/test_audio_native/test_aai_events.py -v`

- [ ] **Step 3: Implement `events.py`** — Pydantic v2 models with `ConfigDict(extra="ignore")`, camelCase JSON fields mapped via `Field(alias=...)` + `populate_by_name=True` (e.g. `tool_call_id: str = Field("", alias="toolCallId")`, `tool_name` ← `toolName`). `_EVENT_TYPE_MAP` + `parse_aai_event` mirroring `assemblyai/events.py`. Include `AAITimeoutEvent`/`AAIUnknownEvent`.

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `git commit -m "feat(voice): aai event models"`.

---

### Task B2: Provider client (`aai/provider.py`)

**Files:**
- Create: `src/tau2/voice/audio_native/aai/provider.py`
- Modify: `src/tau2/config.py` (aai constants — see Global Constraints)
- Test: `tests/test_voice/test_audio_native/test_aai_provider.py`

**Interfaces:**
- Consumes: B1 events; `Tool` (`tool.openai_schema`).
- Produces:
  - `AAIVADConfig(BaseModel)` — empty (aai does turn detection server-side); present for interface parity.
  - `AAIVoiceAgentProvider(api_key=None, ws_url=DEFAULT_AAI_WS_URL, input_sample_rate=16000, tts_sample_rate=24000, system_prompt="", tools=())` with: `connect()` (open WS, send `config` incl. `host`, await server `config`), `send_audio(pcm16: bytes)` (binary frame), `send_tool_result(tool_call_id, result)` (`{"type":"tool_result",...}`), `receive_events()`/`receive_events_for_duration()` (JSON→events; **binary frames yielded as `AAIAudioChunkEvent(pcm16=<bytes>)`**), `is_connected`.
  - Pure builders: `_build_config_message(system_prompt, tools)` and `_format_tools_for_api(tools) -> list[ToolSchema-dict]`.

- [ ] **Step 1: Write the failing test** (pure builders, no network):

```python
from tau2.environment.tool import Tool
from tau2.voice.audio_native.aai.provider import AAIVoiceAgentProvider

def _p():
    return AAIVoiceAgentProvider(system_prompt="You are a retail agent.")

def test_config_message_has_host_and_pcm16():
    def get_order(order_id: str) -> str:
        """Look up an order.

        Args:
            order_id: the id.
        """
        return order_id
    p = AAIVoiceAgentProvider(system_prompt="POLICY", tools=[Tool(func=get_order)])
    msg = p._build_config_message("POLICY", p.tools)
    assert msg["type"] == "config" and msg["audioFormat"] == "pcm16"
    assert msg["sampleRate"] == 16000 and msg["ttsSampleRate"] == 24000
    assert msg["host"]["systemPrompt"] == "POLICY"
    t0 = msg["host"]["tools"][0]
    assert t0["type"] == "function" and t0["name"] == "get_order"
    assert "parameters" in t0 and "function" not in t0

def test_tool_result_shape(monkeypatch):
    # _build not needed; assert send_tool_result payload via a fake ws captured in B3-style test
    pass
```

- [ ] **Step 2: Run, expect FAIL** (module/const missing).

- [ ] **Step 3: Add config constants** to `src/tau2/config.py` near the provider block:

```python
# =============================================================================
# AAI PROVIDER (local voice-agent host; overridable URL/rates)
# =============================================================================
DEFAULT_AAI_WS_URL = "ws://localhost:3000/websocket"  # overridable via AAI_WS_URL
DEFAULT_AAI_MODEL = "host"  # fixed, determined by endpoint
DEFAULT_AAI_INPUT_SAMPLE_RATE = 16000  # PCM16 sent to aai (STT)
DEFAULT_AAI_OUTPUT_SAMPLE_RATE = 24000  # PCM16 received from aai (TTS)
```

- [ ] **Step 4: Implement `provider.py`** — mirror `assemblyai/provider.py` structure. `_format_tools_for_api` flattens `tool.openai_schema["function"]` into `{type:"function", name, description, parameters}`. `_build_config_message` returns `{"type":"config","audioFormat":"pcm16","sampleRate":self.input_sample_rate,"ttsSampleRate":self.tts_sample_rate,"host":{"systemPrompt":system_prompt,"tools":[...]}}`. `connect()` reads `AAI_WS_URL` env (fallback `DEFAULT_AAI_WS_URL`), opens WS, sends the config JSON, waits for server `config` (bounded per-frame `asyncio.wait_for`, buffering non-handshake frames — reuse the assemblyai handshake pattern). `receive_events()`: on `recv()`, if `bytes` → yield `AAIAudioChunkEvent(pcm16=raw)`; if `str` → `parse_aai_event(json.loads(...))`. `send_audio` sends bytes; `send_tool_result` sends the JSON. Add `AAIAudioChunkEvent` to `events.py` (or define in provider) as needed by B3.

- [ ] **Step 5: Run, expect PASS** (the config-builder test).

- [ ] **Step 6: Commit** — `git commit -m "feat(voice): aai provider client (config.host + pcm16)"`.

---

### Task B3: Discrete-time adapter (`aai/discrete_time_adapter.py`)

**Files:**
- Create: `src/tau2/voice/audio_native/aai/discrete_time_adapter.py`
- Test: `tests/test_voice/test_audio_native/test_aai_adapter.py`

**Interfaces:**
- Consumes: `DiscreteTimeAdapter`, `TickResult`, `UtteranceTranscript`, `BackgroundAsyncLoop`, B1 events, B2 provider; audio utils `convert_to_pcm16`, `convert_to_ulaw`, `resample_audio` (`tau2.voice.utils.audio_preprocessing`), `AudioData`/`AudioFormat`/`AudioEncoding` (`tau2.data_model.audio`).
- Produces: `DiscreteTimeAAIAdapter(tick_duration_ms, send_audio_instant=True, reasoning_effort=None, voice=None, provider=None, system_prompt="", tools=())`. External format = telephony μ-law 8k (default). Public `_process_event(result, event)`.

- [ ] **Step 1: Write the failing test** (drive `_process_event` with synthetic events; audio helpers stubbed via small μ-law/PCM buffers):

```python
from tau2.voice.audio_native.aai.discrete_time_adapter import DiscreteTimeAAIAdapter
from tau2.voice.audio_native.aai.events import (
    AAIAgentTranscriptEvent, AAIToolCallEvent, AAISpeechStartedEvent, AAIReplyDoneEvent, AAIAudioChunkEvent,
)
from tau2.voice.audio_native.tick_result import TickResult

def _a():
    return DiscreteTimeAAIAdapter(tick_duration_ms=200, send_audio_instant=True, system_prompt="P")

def _r():
    return TickResult(tick_number=1, audio_sent_bytes=0, audio_sent_duration_ms=0.0, bytes_per_tick=1600, bytes_per_second=8000)

def test_reasoning_effort_rejected():
    import pytest
    with pytest.raises(ValueError):
        DiscreteTimeAAIAdapter(tick_duration_ms=200, reasoning_effort="high")

def test_audio_chunk_converted_and_appended():
    a, r = _a(), _r()
    # 24kHz PCM16: 240 samples = 480 bytes -> ~10ms; after resample to 8k ulaw -> 80 bytes
    a._process_event(r, AAIAudioChunkEvent(pcm16=b"\x00\x01" * 240))
    assert r.agent_audio_bytes > 0  # some μ-law bytes appended

def test_agent_transcript_overwrites():
    a, r = _a(), _r()
    a._current_item_id = "t1"
    a._process_event(r, AAIAgentTranscriptEvent(text="hi"))
    a._process_event(r, AAIAgentTranscriptEvent(text="hi there"))
    assert a._utterance_transcripts["t1"].transcript_received == "hi there"

def test_tool_call_recorded():
    a, r = _a(), _r()
    a._process_event(r, AAIToolCallEvent(tool_call_id="c1", tool_name="get_order", args={"x": 1}))
    assert (r.tool_calls[0].id, r.tool_calls[0].name, r.tool_calls[0].arguments) == ("c1", "get_order", {"x": 1})

def test_barge_in_truncates():
    a, r = _a(), _r()
    a._process_event(r, AAISpeechStartedEvent())
    assert r.was_truncated and "speech_started" in r.vad_events
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — mirror `assemblyai/discrete_time_adapter.py` for lifecycle/`_execute_tick`/`_flush_pending_tool_results`. Differences:
  - `connect()` passes `system_prompt`/`tools` to the provider (so `config.host` carries them).
  - **Audio out (send):** in `_execute_tick`, before sending, convert the tick's μ-law-8k `user_audio` → PCM16-16k: `resample_audio(convert_to_pcm16(AudioData(user_audio, TELEPHONY_FORMAT)), 16000)` → `.data`; send those bytes via `provider.send_audio`. Compute `_chunk_size` from the 16k PCM16 rate.
  - **Audio in (`_process_event` for `AAIAudioChunkEvent`):** wrap `event.pcm16` as PCM16-24k `AudioData`, `resample_audio(..., 8000)` → `convert_to_ulaw(...)` → append `.data` to `result.agent_audio_chunks` under the current turn id; `UtteranceTranscript.add_audio`. Respect `result.skip_item_id` (barge-in) exactly like assemblyai.
  - `AAIAgentTranscriptEvent` → set `ut.transcript_received = event.text` (overwrite). `AAIToolCallEvent` → `ToolCall(id=tool_call_id, name=tool_name, arguments=args)`. `AAISpeechStartedEvent` → barge-in truncate + clear buffer. `AAIReplyDoneEvent`/`AAIAudioDoneEvent` → turn end. `AAIErrorEvent` → `logger.error`. Loud-failure guard on reply/turn end with zero audio+transcript (as in assemblyai).
  - `_flush_pending_tool_results`: for each `(call_id, result_str, _rr, _err)` → `await self.provider.send_tool_result(call_id, result_str)`.
  - Use a running turn id: set `_current_item_id` on first audio/transcript of a turn; reset on `reply_done`.

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `git commit -m "feat(voice): aai discrete-time adapter with pcm16 conversion"`.

---

### Task B4: Registration (config, factory, Literal, CLI, agent VAD, exports)

**Files:**
- Modify: `src/tau2/config.py` (registry dicts), `src/tau2/voice/audio_native/adapter.py` (factory + endpoint tuple), `src/tau2/data_model/simulation.py` (Literal), `src/tau2/cli.py` (choices), `src/tau2/agent/discrete_time_audio_native_agent.py` (**VAD branch + Literal + Union**), `src/tau2/voice/audio_native/aai/__init__.py` (exports)
- Test: `tests/test_voice/test_audio_native/test_aai_registration.py`

**Interfaces:**
- Produces: `create_adapter(provider="aai", ...)` → `DiscreteTimeAAIAdapter`; `AudioNativeConfig(provider="aai")` validates; agent builds `AAIVADConfig` for `provider="aai"`.

- [ ] **Step 1: Write the failing test**

```python
from tau2.config import AUDIO_NATIVE_PROVIDER_TYPES, DEFAULT_AUDIO_NATIVE_MODELS, DEFAULT_AUDIO_NATIVE_REASONING_EFFORT
from tau2.data_model.simulation import AudioNativeConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.aai.discrete_time_adapter import DiscreteTimeAAIAdapter

def test_registry():
    assert DEFAULT_AUDIO_NATIVE_MODELS["aai"] == "host"
    assert DEFAULT_AUDIO_NATIVE_REASONING_EFFORT["aai"] is None
    assert AUDIO_NATIVE_PROVIDER_TYPES["aai"] == "audio_native"

def test_config_literal():
    assert AudioNativeConfig(provider="aai").provider == "aai"

def test_factory_builds_without_connecting():
    adapter, model = create_adapter(provider="aai", tick_duration_ms=200, model=None)
    assert isinstance(adapter, DiscreteTimeAAIAdapter) and model == "host"
    assert adapter.is_connected is False
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement registration** — add `"aai"` to `DEFAULT_AUDIO_NATIVE_MODELS` (→ `DEFAULT_AAI_MODEL`), `DEFAULT_AUDIO_NATIVE_REASONING_EFFORT` (`None`), `AUDIO_NATIVE_PROVIDER_TYPES` (`"audio_native"`); add `"aai"` to `_PROVIDERS_WITH_ENDPOINT_DETERMINED_MODEL` and a `create_adapter()` branch constructing `DiscreteTimeAAIAdapter(tick_duration_ms=..., send_audio_instant=..., reasoning_effort=...)`; add `"aai"` to the `AudioNativeConfig.provider` Literal + description; add `"aai"` to the CLI `--audio-native-provider` choices; in `discrete_time_audio_native_agent.py` add an `elif provider == "aai":` branch importing and building `AAIVADConfig()`, and add `"aai"` to that file's `AudioNativeProvider` Literal + `VADConfig` union + TYPE_CHECKING import; fill `aai/__init__.py` exports.

- [ ] **Step 4: Run, expect PASS**, then the full new group + ruff:

```bash
uv run --extra dev --extra voice python -m pytest tests/test_voice/test_audio_native/ -k aai -q
uv run ruff check src/tau2/voice/audio_native/aai/ && uv run ruff format --check src/tau2/voice/audio_native/aai/
```

- [ ] **Step 5: Commit** — `git commit -m "feat(voice): register aai audio-native provider"`.

---

### Task B5: Standalone smoke test + e2e

**Files:**
- Create: `src/tau2/voice/audio_native/aai/test_provider_standalone.py`
- Modify: `.env.example` (comment: `AAI_WS_URL=ws://localhost:3000/websocket`), `src/tau2/voice/README.md` + `src/tau2/voice/audio_native/README.md` (provider rows)

- [ ] **Step 1: Write the standalone smoke script** (mirrors the assemblyai one) — connect to `AAI_WS_URL`, send `config.host` with a tiny policy + one tool, push μ-law→PCM16 silence, collect events, assert `config`+events flow, exit 0; skip if the URL isn't reachable (short connect timeout → clear SKIP message).

- [ ] **Step 2: Run it against the live dev server** — `uv run python src/tau2/voice/audio_native/aai/test_provider_standalone.py` (aai dev server must be running on :3000). Report the event sequence. **Verify open items:** binary audio arrives and decodes; `tool_call` round-trips via `tool_result`; agent produces a first turn.

- [ ] **Step 3: Docs/env edits** (provider tables + `AAI_WS_URL`).

- [ ] **Step 4: Commit** — `git commit -m "feat(voice): aai provider smoke test + docs"`.

- [ ] **Step 5: End-to-end** (manual; aai dev server running, ElevenLabs personas + Deepgram + user-llm keys set):

```bash
uv run tau2 run --domain retail --audio-native --audio-native-provider aai \
  --user-llm gpt-4.1 --speech-complexity control --num-tasks 1 --verbose-logs
```
Confirm a trajectory is produced with agent audio, tool calls executed by tau2's env, and a reward computed. Report the actual outcome.

---

## Self-Review

**Spec coverage:** host mode protocol (`config.host` + `tool_result`) → A2; relay executeTool + per-session prompt/tools → A3; aai discovery of wiring → A1; local host verification → A4; tau2 events → B1; provider client + config.host builder + constants → B2; adapter + **PCM16↔μ-law conversion** + tool relay → B3; registration incl. **agent VAD branch** → B4; smoke + docs + e2e → B5. ✓

**Placeholder scan:** aai Tasks A2/A3 reference paths "from A1" because the exact WS-session file is the one genuinely-unknown integration point in an unfamiliar repo; A1 exists precisely to resolve it and write the contract before code. All tau2 tasks carry concrete code. No "add error handling"-style hand-waves.

**Type consistency:** provider `AAIVoiceAgentProvider`, adapter `DiscreteTimeAAIAdapter`, VAD `AAIVADConfig`, events `AAI*` + `parse_aai_event`, audio event `AAIAudioChunkEvent`, `ToolCall(id,name,arguments)`, `send_tool_result(call_id, result)` (2 args, matches flush) — consistent across B1–B5. aai `ToolSchema` `{type,name,description,parameters}` and `ExecuteTool(name,args,sessionId?,messages?,opts?)` used as defined in A2/A3.

**Scope:** Phase A gates Phase B's live tests (B5) but B1–B4 (code + unit tests) are independent of Phase A and can proceed in parallel; note this at execution time.
