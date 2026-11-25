# Resia Fullplex

The `resia_fullplex` provider connects τ-voice to Resia Fullplex 1 (`resia-fullplex-1`)
through the Resia Live API. The API serves the OpenAI GPT-Live session protocol over
one WebSocket, so this provider is the OpenAI Live adapter with a WebSocket transport.
The benchmark's system prompt and tool schemas are sent with every session, and
benchmark tools execute in τ-voice.

## Setup

Install the `live` extra and set your Resia API key and the backend model:

```bash
uv sync --extra live
# Set RESIA_API_KEY and OPENAI_API_KEY through your secret-management mechanism.
export RESIA_FULLPLEX_BACKEND_MODEL=...  # the delegated backend model; required
```

The provider connects to `wss://api.resia.ai/v1/live/sessions`.
`--audio-native-model` defaults to `resia-fullplex-1`, and `--reasoning-effort` sets
the backend model's reasoning effort.

## Evaluation

```bash
tau2 run --audio-native --audio-native-provider resia_fullplex --benchmark voice-v1
```

## Audio transport

Audio is G.711 mu-law at 8 kHz in both directions. The provider sends caller audio in
20 ms frames at real-time pace and plays agent audio as it arrives. Agent speech is
classified from the audio played in each tick, as in the OpenAI Live adapter.
