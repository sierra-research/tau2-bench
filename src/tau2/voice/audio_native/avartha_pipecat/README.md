# Avartha provider (Pipecat RTVI)

This adapter connects τ-bench's discrete-time audio agent to a hosted Avartha
voice agent. It calls the agent through Pipecat's agent-server (SmallWebRTC)
protocol. τ-bench does not start or configure the agent. Model, speech, and
turn-taking settings belong to the agent.

## Install

The WebRTC client (aiortc) ships in the live extra:

```bash
uv sync --extra voice --extra live
```

## Run

```bash
export AVARTHA_API_KEY=<your-api-key>
tau2 run --audio-native --benchmark none --domain retail \
  --audio-native-provider avartha_pipecat \
  --avartha-base-url https://<host>/inference/serverless/pipecat/v1/public/<agent_id> \
  --num-tasks 1
```

| Option or variable | Description |
|---|---|
| `--avartha-base-url` | The hosted agent's URL (required) |
| `AVARTHA_API_KEY` | The agent's API key, sent as a bearer token when starting each call (required) |
| `--audio-native-model` | Label recorded with the results, `avartha` by default; the URL selects the agent |

`--reasoning-effort` is rejected: reasoning is configured on the agent.

The agent speaks and listens on the wall clock, so ticks are paced in real
time and the user simulator's LLM and TTS run without blocking them, as for
`openai_live`. `--no-realtime-generation` turns the latter off.

## What happens on each call

1. `POST <agent-url>/start` with `caller_tools: true` and the unmodified domain
   policy as the agent prompt override. The reply carries the session ID and
   ICE servers.
2. `POST <agent-url>/sessions/<session-id>/api/offer` exchanges the SDP offer
   and answer. The call carries one Opus audio track each way and an ordered
   `chat` data channel for RTVI messages.
3. The adapter announces RTVI `client-ready` and waits for the agent's
   `session_boot` readiness verdict. A call that does not boot is retried, for
   up to three attempts.
4. It registers the domain tools as self-contained MCP tool definitions
   (`inputSchema`) under the versioned `avartha-harness-tools/1` protocol and
   waits for the agent to acknowledge them.
5. Each tick converts τ-bench's 8 kHz μ-law audio to 16 kHz PCM16 with a
   windowed-sinc filter (`caller_audio.py`; the shared linear-interpolation
   converter leaves a mirror image of the voice above 4 kHz) and paces it onto
   the caller track in 20 ms frames. The agent's track is resampled to 24 kHz
   PCM16, frames of digital silence between utterances are dropped, and the
   rest is converted back to telephony audio. `tool_call` messages become
   τ-bench tool calls and their results are returned as MCP `CallToolResult`
   payloads.
6. On disconnect it sends RTVI `disconnect-bot` and closes the peer connection.

The agent's transcript is revealed only as far as its audio has actually
played. The agent sends each sentence of text before the audio that speaks it;
the adapter holds the text back until the agent's word-timing events or its
end-of-utterance playback caption show that the words aired. A word the
alignment splits in two ("T-shir" + "t") counts once both pieces have aired.
On barge-in, text the caller has not heard is discarded.

## Protocol compatibility

The agent's `tools_registered` or `tools_registration_error` reply must carry
`"protocol": "avartha-harness-tools/1"`. A missing or different protocol fails
the run immediately with `Incompatible Avartha tool protocol`: this is a
configuration error, so local and worker runs stop scheduling tasks and skip
simulation retries instead of recording the mismatch as a benchmark score.

## Result metadata

The call's session ID is saved as `provider_session_id` on `SimulationRun`.
The hosted agent reports no cost.

## Live provider suite

```bash
AVARTHA_PIPECAT_TEST_ENABLED=1 \
TAU2_AVARTHA_BASE_URL=https://<host>/inference/serverless/pipecat/v1/public/<agent_id> \
AVARTHA_API_KEY=<your-api-key> \
uv run pytest tests/test_voice/test_audio_native/test_provider_suite.py \
  -v -s -k avartha_pipecat
```
