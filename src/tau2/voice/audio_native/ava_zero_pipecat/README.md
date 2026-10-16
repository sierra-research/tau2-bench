# Ava Zero provider (Pipecat RTVI)

This adapter connects τ-bench's discrete-time audio agent to Avartha's Ava Zero
voice agent. It talks to an already-running Ava Zero server over Pipecat's
protobuf RTVI WebSocket transport; τ-bench does not start or configure the
server. Model, speech, and turn-taking settings belong to the server.

## Install

The provider uses Pipecat's frame serializer, which ships with τ-bench's voice
dependencies:

```bash
uv sync --extra voice
```

## Run

```bash
tau2 run --domain retail --audio-native \
  --audio-native-provider ava_zero_pipecat \
  --ava-zero-base-url wss://<your-ava-zero-host> \
  --num-tasks 1
```

| Option | Default | Description |
|---|---|---|
| `--audio-native-model` | `conductor-v2` | Server-side agent ID to start for each call |
| `--ava-zero-base-url` | `ws://localhost:7860` | Base URL of the server; `http(s)` and `ws(s)` are both accepted |
| `--ava-zero-websocket-token-env` | *(none)* | Name of an environment variable whose value is sent as the WebSocket `token` query parameter |

`TAU2_AVA_ZERO_BASE_URL`, when set, takes precedence over `--ava-zero-base-url`.
`--reasoning-effort` is rejected: reasoning is configured on the server.

## What happens on each call

1. The adapter creates a session with `POST /api/sessions`, carrying the
   unmodified domain policy as the agent prompt.
2. It opens `/ws-client?agent=<id>&session_id=<id>`, announces RTVI
   `client-ready`, and waits for the server's `session_boot` readiness verdict.
   A session that does not boot is retried, for up to three attempts.
3. It registers the domain tools as self-contained MCP tool definitions
   (`inputSchema`) under the versioned `avartha-harness-tools/1` protocol and
   waits for the server to acknowledge them.
4. Each tick converts τ-bench's 8 kHz μ-law audio to 16 kHz PCM16, sends it,
   and converts the returned PCM16 frames from their declared sample rate back
   to telephony audio. `tool_call` messages become τ-bench tool calls and their
   results are returned as MCP `CallToolResult` payloads.
5. On disconnect it requests the server's `cost_report`.

The agent's transcript is revealed only as far as its audio has actually
played. The server sends each sentence of text before the audio that speaks
it; the adapter holds the text back until the server's word-timing events or
its end-of-utterance playback caption show that the words aired. On barge-in,
text the caller has not heard is discarded.

## Protocol compatibility

The server's `tools_registered` or `tools_registration_error` reply must carry
`"protocol": "avartha-harness-tools/1"`. A missing or different protocol fails
the run immediately with `Incompatible Ava Zero tool protocol`: this is a
configuration error, so local and worker runs stop scheduling tasks and skip
simulation retries instead of recording the mismatch as a benchmark score.

## Result metadata

The server's session ID is saved as `provider_session_id` and its native cost
and usage report as `provider_cost_report` on `SimulationRun`. When the report
carries a valid `total_cost_usd`, that total is used as the simulation's agent
cost. Leaderboard trajectory trimming strips `provider_cost_report`.

## Live provider suite

With a compatible server running:

```bash
AVA_ZERO_PIPECAT_TEST_ENABLED=1 \
TAU2_AVA_ZERO_BASE_URL=ws://localhost:7860 \
uv run pytest tests/test_voice/test_audio_native/test_provider_suite.py \
  -v -s -k ava_zero_pipecat
```

Set `TAU2_AVA_ZERO_AGENT` to test an agent other than `conductor-v2`.
