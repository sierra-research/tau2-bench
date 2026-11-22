# Smallest AI Hydra

The `smallest` provider connects τ-voice to Smallest AI's Hydra speech-to-speech
server over its `/v1/hydra` WebSocket. Hydra runs server-side and handles the
whole conversation, including turn-taking. τ-voice sends the benchmark's system prompt and tool schemas when
the call starts, and benchmark tools execute in τ-voice: Hydra executes no
tools.

## Setup

Install the `voice` extra and point the adapter at a Hydra server:

```bash
uv sync --extra voice
export SMALLEST_WS_URL=wss://<host>/v1/hydra   # required
# Set SMALLEST_API_KEY through your secret-management mechanism if the server
# requires a key.
```

| Variable | Description |
|---|---|
| `SMALLEST_WS_URL` | Hydra's WebSocket URL (required). `ws://` is fine for a local server; credentials need `wss://` unless the host is local |
| `SMALLEST_API_KEY` | Sent as `Authorization: Bearer <key>` (optional; no credential is sent without it) |
| `SMALLEST_VOICE` | Voice for the session (optional; Hydra's configured default otherwise) |
| `SMALLEST_TRANSCRIPT_REVEAL_CPS` | Transcript reveal cap in characters per second of agent audio played, default `15`; `0` turns the cap off (see below). The effective value is recorded in every saved run's `info.agent_info.llm_args` |

`--audio-native-model` is a label recorded with the results
(`hydra` by default). `--reasoning-effort` and `--voice` are not supported
(use `SMALLEST_VOICE` for the voice). Only telephony audio (8 kHz
μ-law) is supported.

## Evaluation

```bash
tau2 run --audio-native --audio-native-provider smallest --benchmark voice-v1
```

## Audio transport

Each tick converts τ-voice's 8 kHz μ-law audio to 16 kHz PCM16 and sends it as
`input_audio_buffer.append` frames of 20 ms (one frame per tick when the run
config sets `send_audio_instant`). The agent's PCM16 audio arrives at the rate Hydra
announces in `session.configured` and is converted back to telephony audio.
The connection fails if Hydra rejects the session or announces no output rate,
so a call never runs unconfigured or plays audio at the wrong speed. Hydra's
session ID is saved as `provider_session_id` on each simulation.

Tool calls (`response.function_call_arguments.done`) become τ-voice tool calls.
Each result goes back on the next tick as a `function_call_output` item, then
`response.create` (once per batch of results), and Hydra continues the reply.

Hydra decides barge-ins itself. `input_audio_buffer.speech_started` marks the
first sound of a caller turn and cuts nothing: Hydra may keep talking. When it
commits a barge-in it sends `response.barge_in`; the adapter then cuts the
queued agent audio, as the other adapters do on speech onset, and skips the
rest of the reply that is still streaming.

The adapter records no usage or cost for the agent.

## Transcript pacing

The caller never sees agent words before the audio that speaks them has
played. Hydra sends the caption of each clause just before the clause's
audio, so revealing text in proportion to audio played over audio received
(the default for other providers) would show a whole clause on its first tick.
The adapter therefore caps the reveal: after `s` seconds of an utterance's
audio have played, at most `floor(15 × s)` of its characters are shown. When
Hydra reports the reply `completed` and all of its audio has played, the rest
is shown; a reply cut short never shows its unspoken rest.

Example with 200 ms ticks: the caption "I can change that flight for you."
(33 characters) arrives with the first 200 ms of its 2.2 s of audio.
Proportional reveal alone shows all 33 characters on the first tick. With the
cap the first tick shows 3 characters ("I c"), each later tick 3 more, and a
caller who barges in after 1 s has seen 15 ("I can change th") and never sees
the rest.

The cap lives in the adapter (`pacing.py`); `SMALLEST_TRANSCRIPT_REVEAL_CPS=0`
restores the default proportional reveal exactly.

## Live provider suite

```bash
SMALLEST_TEST_ENABLED=1 SMALLEST_WS_URL=ws://localhost:<port>/v1/hydra \
uv run pytest tests/test_voice/test_audio_native/test_provider_suite.py \
  -v -s -k smallest
```

The offline tests (`tests/test_voice/test_audio_native/test_smallest/`) run the
adapter against a fake Hydra server on the loopback interface. In the live
suite, `TestUsageReporting` fails by design: the adapter reports no usage
records, and the pricing table has no rate for Hydra.
