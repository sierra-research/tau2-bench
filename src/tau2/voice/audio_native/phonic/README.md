# Phonic

The `phonic` provider connects τ-voice to Phonic's speech-to-speech WebSocket API.
The benchmark's system prompt and tool schemas are sent with every session, and
benchmark tools execute in τ-voice.

## Setup

Install the `voice` extra and set your Phonic API key:

```bash
uv sync --extra voice
# Set PHONIC_API_KEY through your secret-management mechanism.
export PHONIC_PROJECT=main  # optional (default: main)
```

`--audio-native-model` sets the Phonic model (default `phonic-1`) and
`--reasoning-effort` its intelligence level (`high` and `xhigh` map to high, the
default; lower efforts map to standard).

## Evaluation

```bash
tau2 run --audio-native --audio-native-provider phonic --benchmark voice-v1
```

## Audio transport

Phonic processes audio as a live call, so it expects a continuous real-time
stream. The orchestrator can block between ticks while the user simulator
generates speech, so the adapter runs a background pump that streams user audio,
or silence when none is queued, at a steady real-time cadence and reads agent
events continuously. Agent speech is classified from the audio played in each
tick, as in the OpenAI Live adapter.
