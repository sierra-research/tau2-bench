# Gemini Live Provider

Adapter for Google's Gemini Live API (native audio-to-audio).

## Authentication

Auto-detected from environment variables (checked in order):
1. `GEMINI_API_KEY` → AI Studio
2. `GOOGLE_SERVICE_ACCOUNT_KEY` (JSON content) → Vertex AI
3. `GOOGLE_APPLICATION_CREDENTIALS` (file path) → Vertex AI

Vertex AI also requires `GOOGLE_CLOUD_PROJECT` (or extracted from service account JSON).

## Gemini 3.1 vs 2.5

The older `gemini-3.1-flash-live-preview` model remains available via
`--audio-native-model`. The provider auto-detects 3.1 models and applies feature gates:

| Feature | 2.5 | 3.1 |
|---------|-----|-----|
| Context window compression | Yes | No (not supported) |
| Input audio transcription | Yes | No (not supported) |
| Proactive audio | Yes | No (not supported) |
| Thinking config | No | Yes (`DEFAULT_GEMINI_THINKING_LEVEL`) |
| Text input path | `session.send()` | `send_realtime_input()` (non-Vertex) |

## Gemini 3.8 Live Extended Thinking

The default model is `gemini-3.8-live-extended-thinking`.
Use `--reasoning-effort low`, `medium`, or `high` (the current provider default
is `high`). Extended Thinking requires nonblocking tool declarations. Its
`turn_complete` marks a spoken utterance, while `interaction_status=IDLE` marks
the end of the full interaction. The Google Gen AI SDK must be at least 2.30.1.
The provider rotates audio items at each spoken turn boundary even while
`session.receive()` stays open until the interaction is idle.
Confirm the configured Gemini credential can open a session before scheduling
a full evaluation.

## Session Resumption

Gemini sessions have a ~10 minute timeout. The provider handles this via:
1. Server sends `GoAway` event with time remaining
2. Provider sets `needs_reconnection` flag
3. Adapter checks at tick boundary and calls `perform_pending_reconnection()`
4. New session resumes from the resumption handle

Configurable via `max_resumptions` (default 3) and `resume_only_on_timeout` (default True).

## Tool Calls

Gemini sends null/empty tool call IDs, so the adapter generates synthetic IDs for internal tracking and maps them back to original IDs when sending results. Tool results are batched into a single `send_tool_response` call to prevent the model from re-calling tools before seeing all results.

## Audio Formats

- Input: 16kHz PCM16 mono
- Output: 24kHz PCM16 mono
- Conversion handled by `StreamingTelephonyConverter` in the adapter
