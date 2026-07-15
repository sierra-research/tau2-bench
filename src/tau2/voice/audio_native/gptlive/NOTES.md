# GPT-Live Provider Implementation Notes

Working notes for adding `gptlive` as an audio-native provider.
Source material: `../tmp_gpt_live_notes.md` and `../gpt-live-confidential-instructions.pdf`
(CONFIDENTIAL alpha docs — model `gpt-live-1-boulder-alpha`).

## TODOs

- [x] Step 1: Review OpenAI Realtime integration; map GPT Live vs GPT Realtime API differences
- [x] Step 2: Decide 1:1 carry-forward vs. divergent parts
- [x] Step 3: Implement gptlive provider; horizontal test suite passes (gptlive only)
- [x] Step 4: E2E run on 2 retail tasks (control/clean audio); verify tool calls + responses in tau2 view
- [x] Step 5: Review/clean up all added code
- [x] Step 6: 5-7 bullet implementation summary

---

## Step 1 — API differences (GPT Live vs GPT Realtime)

- **Endpoint/auth**: `wss://api.openai.com/v1/live?model=...` + required header
  `OpenAI-Alpha: quicksilver=v2` (vs `/v1/realtime`, no alpha header). Same
  `OPENAI_API_KEY` bearer auth.
- **Handshake inverted**: Realtime: server sends `session.created` first, then client
  configures and waits for `session.updated`. Live: *client* sends `session.update`
  as the first event, server replies `session.started`. Model must NOT be repeated
  in the session object (URL query string only).
- **Audio format**: Live WebSocket only supports raw headerless 24kHz mono PCM16LE
  base64 (both directions). Realtime supports g711_ulaw natively (no conversion).
  Live needs telephony (8kHz μ-law) ↔ 24kHz PCM16 conversion both ways.
- **Audio events**: input via `input_audio.append` (vs `input_audio_buffer.append`);
  output via `output_audio.delta` with server-timeline `start_ms`/`end_ms` and NO
  `item_id`, NO `output_audio.done`. Gaps between output ranges = omitted silence.
- **No VAD**: Full-duplex architecture — no turn_detection config, no
  `speech_started`/`speech_stopped` events, no `conversation.item.truncate`.
  The model listens while speaking and yields on its own.
- **Transcripts**: complete timed fragments (`input_transcript.added` /
  `output_transcript.added`) instead of deltas keyed by item_id. `turn.*` events are
  a heuristic projection (not context items).
- **Tool calling via delegation**: tools live under
  `session.delegation.responses.tools` (Responses delegation mode). Function calls
  arrive as pass-through `response.function_call_arguments.done` events (same field
  shape as Realtime: call_id/name/arguments). Results returned via
  `delegation.function_call_output.create` — and clients must NOT send
  `response.create` (Realtime requires it after tool results).
- **Graceful close**: `session.close` → drain until `session.closed` (max 10s server
  side) vs plain WS close.
- **Usage**: `session.usage.updated` ~1/min + final usage in `session.closed`.

## Step 2 — Carry forward vs. diverge

**Carried 1:1 (same pattern as openai/qwen adapters):**
- `DiscreteTimeAdapter` template method: `_execute_tick` + `_flush_pending_tool_results`,
  `BackgroundAsyncLoop`, `_send_audio_chunked`, buffer/cap/proportional-transcript logic.
- Provider class shape: `connect / configure_session / send_audio /
  receive_events_for_duration / send_tool_result / disconnect`.
- Event parsing style: pydantic models + `parse_gptlive_event()` + required `TimeoutEvent`.
- ToolCall extraction from `response.function_call_arguments.done` (identical fields).
- Audio conversion via shared `StreamingTelephonyConverter(24000, 24000)` (like qwen,
  but 24kHz input instead of 16kHz).

**Divergent decisions:**
1. **Delegation mode = `responses`** (not `client`): tau2 needs structured function
   calls with call_ids; client delegation is free-text only. Delegated backend model
   configurable, default in config.py (`DEFAULT_GPTLIVE_DELEGATION_MODEL`). The domain
   policy is passed as BOTH live-session `instructions` and
   `delegation.responses.instructions` (the delegated model chooses/executes tools, so
   it needs the policy too).
2. **Synthetic utterance IDs**: `output_audio.delta` has no item_id, but the framework's
   proportional-transcript machinery is keyed on item_id. We maintain a synthetic
   utterance id (`live_utt_N`) that both audio deltas and output transcript fragments
   feed into; rotated on interruption so stale transcript doesn't bleed into the next
   response.
3. **Synthesized barge-in signal**: no server VAD events exist. We emit
   `speech_started` + local truncation when an `input_transcript.added` fragment
   arrives while agent audio is playing/buffered (evidence the user spoke over the
   agent). `event.start_ms` (server timeline ≈ cumulative user-audio clock) is used as
   the interruption timestamp. No server-side truncate call exists — we only clear
   local buffers; the model yields on its own.
4. **No skip_item_id**: Realtime skips in-flight audio from a truncated item by id;
   Live deltas carry no ids, so skipping would be meaningless (or would drop all
   future audio if we used a constant id). We rely on the model yielding + local
   buffer clear instead.
5. **VAD config ignored** (warn if provided); `session.close`/`session.closed`
   drain in disconnect with a bounded timeout.
6. **Test gating**: suite entry gated on `GPTLIVE_TEST_ENABLED=1` (like nova/livekit),
   not bare `OPENAI_API_KEY` — this is a confidential limited-access alpha, so people
   with a regular OpenAI key but no alpha access shouldn't get failures.

### Step 1+2 summary
- Realtime and Live share event-stream style and function-call payload shape, so the
  adapter skeleton carries over; the transport lifecycle, audio format, tool-result
  channel, and interruption model are all different.
- Biggest impedance mismatches with the framework: no item_ids on audio and no VAD
  events — solved with synthetic utterance ids and transcript-based barge-in synthesis.

---

## Step 3 — Implementation

Files: `gptlive/{__init__,events,provider,discrete_time_adapter}.py`; registered in
`adapter.py` `create_adapter()`, `config.py`, `cli.py`, `AudioNativeConfig`, and the
provider suite (gated on `GPTLIVE_TEST_ENABLED=1`).

Surprises found by testing against the real API (doc didn't match reality):
1. **`response.function_call_arguments.done` does NOT carry `call_id`/`name`** (despite
   the alpha guide's example). They're announced earlier on `response.output_item.added`
   → adapter keeps an `item_id → (call_id, name)` map and joins.
2. **Output audio streams continuously in real time, INCLUDING silence** (RMS ~0
   chunks with contiguous ranges), despite the guide's "gap = omitted silence" claim.
   Without handling, the framework thinks the agent never stops speaking. Added an
   RMS hysteresis gate (open at >=100, close at <15; TTS noise floor within an
   utterance is ~18-130, so it doesn't false-close in inter-word pauses). Gate close
   also rotates the synthetic utterance id.
3. **Barge-in needs timeline overlap, not "audio present in this tick"**: input
   transcript fragments lag real speech by ~600-1000ms, and the full-duplex model
   stops so fast that tick-level co-occurrence checks both false-positived (user's own
   turn tail overlapping response onset) and false-negatived (agent yielded before the
   fragment arrived). Now compares fragment.start_ms (server clock == cumulative
   user-audio clock) to when the agent utterance started (>=300ms before) and when
   agent audio last played (<=200ms margin).
4. **Graceful close needed tighter budgets**: session.close → session.closed drain
   capped at 2s + websocket close_timeout=2s to fit the framework's 5s disconnect.
5. **Suite prompt tweak**: `BARGE_IN_SYSTEM_PROMPT` needed an extra sentence
   ("even for greetings... at least thirty seconds") because gpt-live answers
   greetings briefly regardless of "give long answers" (alpha IF is weak — matches
   the tmp notes' "continued IF improvements to come"). Verified openai still passes
   with the updated prompt.

### Step 3 summary
- Suite result: **11/11 passed** for gptlive, serial and `-n 4` parallel (repeated runs).
- Tool round-trip works via Responses delegation with the item_id join.
- Real-API behavior diverged from the alpha PDF in two load-bearing places
  (function-call fields, continuous silence streaming) — both handled in the adapter.



## Step 4 — E2E retail run

Command: `tau2 run --domain retail --audio-native --audio-native-provider gptlive
--task-ids 0 1 --speech-complexity control --max-steps-seconds 300 --save-to
gptlive_retail_test2` (first attempt used a 120s cap — too short for retail tasks;
both hit max_steps mid-verification).

### Step 4 summary
- **Task 1: reward 1.0** — 6 tool calls (`find_user_id_by_name_zip`,
  `get_order_details`, 3x `get_product_details`, `exchange_delivered_order_items`
  write), all action checks + DB check pass, user_stop termination. Many agent
  speech turns; verified the full tick trajectory (agent text, user text, tool
  calls/results) renders through the same `Results`/`ConsoleDisplay` path
  `tau2 view` uses.
- Task 0: agent does the flow (9 tool calls on a retry run) but is slow —
  re-verifies identity repeatedly and pauses ("One moment... checking") while
  delegation runs, so it didn't complete within 300s (and in one run the user
  hung up during a ~15s silent delegation wait). Model-behavior/latency trait of
  the alpha, not an adapter bug.



## Step 5 — Cleanup review

- Rewrote the adapter module docstring to describe the silence gate and
  timeline-based barge-in as implemented (the original docstring described the
  pre-debugging design).
- Deleted the three throwaway debug scripts (`scripts/debug_gptlive_*.py`) used to
  observe raw API behavior.
- Added gptlive to `audio_native/README.md` provider table/diagram; reverted an
  unrelated ruff reformat and the regenerated `provider_suite_results.txt`.
- `make check-all` clean; gptlive suite 11/11 (final confirmation run); `make test`
  (core, 193 tests) passes.

## Step 6 — Implementation summary

See final chat summary. Key decisions: Responses delegation for tools; RMS
hysteresis silence gate; synthetic utterance ids; timeline-overlap barge-in
synthesis; item_id join for function-call metadata; GPTLIVE_TEST_ENABLED gating.
