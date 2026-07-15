# aai voice-agent host + tau2 `aai` provider

**Date:** 2026-07-15
**Status:** Approved — ready for implementation planning
**Repos:** `~/Code/aai` (host-mode feature) and `~/Code/tau2-bench` (new `aai` provider)

## Goal

Benchmark the **aai** voice-agent framework on tau2. To do that, aai must become a
**voice-agent host**: a mode where an *external* harness (tau2) supplies the system
prompt + tool schemas at connect time, aai runs its own STT→LLM→TTS pipeline, and
aai **relays each tool call back to the harness** to execute (so tau2's environment
runs the domain tools against its DB — which is what the reward scores). A new tau2
`aai` audio-native provider drives this over aai's existing WebSocket.

### Why a host mode is required

aai agents are normally defined in TypeScript (`agent({ systemPrompt, tools:{ name: tool({parameters, execute}) } })`) and their tools execute **server-side in a sandbox** against `ctx.kv`. tau2 requires the opposite: the agent runs the *domain's* policy + tool *schemas*, and **tau2's Python environment executes the tools**. Host mode bridges this: prompt + tool *schemas* are injected per session, and tool *execution* is relayed to the tau2 client.

## Confirmed facts (from code inspection)

- WS endpoint: `ws://localhost:3000/websocket` (`?sessionId=`/`?resume=1` optional). Server name `pipeline-simple`.
- Client→server: `config` first (`{type:"config", audioFormat:"pcm16", sampleRate, ttsSampleRate, sessionId?}`), then **raw binary PCM16** audio frames; control msgs `audio_ready` / `cancel` / `reset` / `history{messages:[{role,content}]}`.
- Server→client: **binary PCM16** audio (TTS) + JSON events `config`, `speech_started`, `speech_stopped`, `user_transcript{text,turnOrder?}`, `agent_transcript{text}`, `tool_call{toolCallId,toolName,args}`, `tool_call_done{toolCallId,result}`, `reply_done`, `audio_done`, `cancelled`, `reset`, `idle_timeout`, `error{code,message}`, `custom_event{event,data}`.
- Default sample rates: **input 16000 Hz, output 24000 Hz**, PCM16 (`DEFAULT_STT_SAMPLE_RATE` / `DEFAULT_TTS_SAMPLE_RATE`; compat fixture `sampleRate:16000, ttsSampleRate:24000`).
- The aai package already has a `host/` runtime (`runtime.ts`, `s2s.ts`, `session-core.ts`, `runtime-config.ts`, providers, `pipeline-transport.ts`) — the home for host mode.
- Server message schemas are validated with zod in `packages/aai-server/schemas.ts` (+ the pipeline transport/session schemas in `packages/aai`). Tool execution + relay must integrate with the aai `host` runtime and its ws transport.

## Part A — aai-side: host mode

**Protocol additions (backward-compatible):**

1. Extend the client `config` message with an optional `host` block:
   ```
   { type: "config", audioFormat: "pcm16", sampleRate, ttsSampleRate,
     host: {
       systemPrompt: string,
       greeting?: string,
       tools: Array<{ name: string, description: string, parameters: <JSON Schema object> }>
     } }
   ```
   When `host` is present, the server builds the agent **dynamically** from it (instead of the deployed agent).
2. Add a client→server message `tool_result`:
   ```
   { type: "tool_result", toolCallId: string, result: <JSON string>, error?: string }
   ```

**Behavior:**
- Each injected tool's `execute(args, ctx)` emits the existing `tool_call{toolCallId, toolName, args}` and returns a Promise that resolves when the matching `tool_result` arrives (or rejects on `error`). A per-call **timeout** (configurable, default ~120s) rejects with a tool error so the pipeline continues.
- `systemPrompt` and `greeting` come from the injected block. Everything else (pcm16 audio in/out, transcripts, `reply_done`, `audio_done`, barge-in, `speech_started/stopped`) is unchanged.
- The injected `parameters` JSON Schema is converted to the internal tool-parameter representation (the runtime currently takes zod; add a JSON-Schema → runtime-schema path, or accept JSON Schema directly for host tools).

**Safety:** host mode lets a client run arbitrary prompts/tools, so gate it behind an env flag (e.g. `AAI_ALLOW_HOST`, default **on for local dev**, off in production deploys). The relayed tools do NOT run sandbox code — they only round-trip to the client — so the sandbox threat surface is unchanged.

**Files (aai):** primarily `packages/aai-server/schemas.ts` (config `host` + `tool_result`), the ws transport / orchestrator that dispatches inbound messages (`transport-websocket.ts`, `orchestrator.ts`), and the `packages/aai/host` runtime that builds the agent + defines relayed tools. Exact files finalized in the plan after reading the host runtime.

## Part B — tau2-side: `aai` provider

New provider `src/tau2/voice/audio_native/aai/`, modeled on the assemblyai provider but with **PCM16 audio conversion** (aai is not μ-law).

- `provider.py` — `AAIVoiceAgentProvider` (WS client): `connect()` (open `AAI_WS_URL` default `ws://localhost:3000/websocket`, send `config` with `host{systemPrompt, tools}` + `audioFormat:"pcm16"` + sample rates, await server `config`), `send_audio(pcm16_bytes)` (binary frame), `send_tool_result(tool_call_id, result)` (`tool_result` JSON), `receive_events*()`. Plus `AAIVADConfig` (server-side VAD; likely a no-op/empty config since aai handles turn detection).
- `events.py` — pydantic models for the aai server messages + `parse_aai_event()`; binary frames handled out-of-band (not JSON).
- `discrete_time_adapter.py` — `DiscreteTimeAAIAdapter(DiscreteTimeAdapter)`:
  - **Audio conversion at the WS boundary:** external format stays tau2 telephony **μ-law 8 kHz**; convert μ-law-8k → PCM16-16k when sending, PCM16-24k → μ-law-8k when receiving, using `tau2.voice.utils.audio_preprocessing` (`convert_to_pcm16`, `resample_audio`, `convert_to_ulaw`) / `StreamingTelephonyConverter`. `bytes_per_tick` etc. remain in telephony terms.
  - `_execute_tick`: send converted audio + receive events/binary-audio for the tick window.
  - `_process_event`: binary audio → convert → `agent_audio_chunks` (item = current `reply`/turn id); `agent_transcript{text}` → `UtteranceTranscript` (full-text overwrite, like assemblyai); `tool_call` → `ToolCall(id=toolCallId, name=toolName, arguments=args)`; `speech_started` → barge-in (truncate + clear buffer); `reply_done`/`audio_done` → turn end; `error` → log.
  - `_flush_pending_tool_results`: for each queued `(call_id, result, ...)` send `tool_result{toolCallId=call_id, result}`. (No extra "continue" message — aai auto-continues on tool_result.)
  - Lifecycle via `BackgroundAsyncLoop`; `reasoning_effort` unsupported → `None`.
- `__init__.py` exports; `test_provider_standalone.py` smoke test against live `localhost:3000`.

**Registration (tau2):**
- `config.py`: `DEFAULT_AAI_WS_URL="ws://localhost:3000/websocket"`, `DEFAULT_AAI_MODEL="host"` (endpoint-determined), `DEFAULT_AAI_INPUT_SAMPLE_RATE=16000`, `DEFAULT_AAI_OUTPUT_SAMPLE_RATE=24000`; add `"aai"` to `DEFAULT_AUDIO_NATIVE_MODELS`, `DEFAULT_AUDIO_NATIVE_REASONING_EFFORT` (None), `AUDIO_NATIVE_PROVIDER_TYPES` ("audio_native").
- `adapter.py`: add `"aai"` to `_PROVIDERS_WITH_ENDPOINT_DETERMINED_MODEL` and a `create_adapter()` branch.
- `data_model/simulation.py`: add `"aai"` to the `AudioNativeConfig.provider` Literal + description.
- `cli.py`: add `"aai"` to `--audio-native-provider` choices.
- `agent/discrete_time_audio_native_agent.py`: add the **`aai` VAD-config branch** (build `AAIVADConfig`) + `"aai"` in that file's `AudioNativeProvider` Literal and `VADConfig` union. *(This is the integration point the assemblyai work initially missed — do not skip it.)*

## Key differences from the assemblyai provider
- **Audio is PCM16 (16k in / 24k out), not μ-law** → resampling both directions (assemblyai was passthrough).
- **Prompt + tool schemas injected in `config.host`** (assemblyai used `session.update`).
- **Tools relayed via `tool_call` → `tool_result`** with client execution (same need as tau2; aai must add `tool_result`).
- **Two-repo change** (aai host mode + tau2 provider).

## Testing
- **aai**: vitest for (a) `config.host` builds a dynamic agent with the injected prompt/tools, (b) a `tool_call` round-trips through `tool_result` (with timeout path), following aai's existing test setup (`transport-websocket.test.ts`, `ws-integration.test.ts`, host runtime tests).
- **tau2**: unit tests — `parse_aai_event`, the `config.host` builder + tool-schema formatting, `_process_event` mapping (audio/transcript/tool_call/barge-in), `_flush_pending_tool_results` sends `tool_result` and no extra continue; μ-law↔PCM16 conversion sanity. Standalone smoke test against live `localhost:3000`.
- **End-to-end**: `uv run tau2 run --domain retail --audio-native --audio-native-provider aai --user-llm gpt-4.1 --speech-complexity control --num-tasks 1 --verbose-logs`.

## Open items to resolve during planning
- Exact aai host-runtime integration points (read `packages/aai/host/runtime.ts`, `s2s.ts`, `pipeline-transport.ts`, and `aai-server/orchestrator.ts` / `transport-websocket.ts`).
- JSON-Schema → aai tool-parameter conversion (aai tools use zod today).
- Whether aai VAD/turn-detection needs any client config, or is fully server-side (then `AAIVADConfig` is empty).
- Confirm the aai input sample rate the pipeline actually expects (16000) and that arbitrary μ-law-derived PCM16 is accepted.

## Non-goals
- Deploying a persistent tau2 domain agent to aai (host mode is per-session, ephemeral).
- Changing aai's sandbox/tool-execution model for normal (non-host) agents.
- Multi-trial/leaderboard submission plumbing for aai (separate follow-up).
