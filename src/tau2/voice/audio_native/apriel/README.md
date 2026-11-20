# Apriel Realtime provider

This integration connects tau to an Apriel realtime gateway through an
OpenAI-compatible WebSocket protocol.

The adapter uses Tau's shared discrete-time template for audio pacing, buffering,
transcript alignment and timing. Apriel-specific event handling preserves
response ownership, clears interrupted audio, rejects late cancelled output,
and dispatches each completed tool call once. Tau still executes the tools.

## Configuration

Required environment variables:

```bash
APRIEL_REALTIME_BASE_URL=wss://your-host/v1/realtime
APRIEL_REALTIME_API_KEY=...
APRIEL_MODEL_NAME=Apriel-Voice-27B-Preview
```

Set `APRIEL_MODEL_NAME` (or `--audio-native-model`) to the attached backend model
returned by `/v1/models`. This branch's existing model default is unchanged.
The URL accepts an HTTP(S) origin, `/v1`, or a WS(S) `/v1/realtime` URL; it is
normalized to the WebSocket endpoint. Credentials must never be placed in URLs.

For a protected EAI proxy also set `APRIEL_CLOUD_AUTH_TOKEN`. The cloud token is
sent as the Authorization bearer; the realtime key is sent in the
`openai-insecure-api-key.<key>` subprotocol alongside `realtime`. Model discovery
uses cloud Authorization plus `X-Apriel-Authorization: Bearer <realtime-key>`.
The transport helper `connection.resolve_model` performs verified HTTPS discovery.
Without the cloud token, direct realtime bearer authentication is retained.
`APRIEL_API_KEY` and its file form remain legacy fallbacks. The optional
`APRIEL_REALTIME_TLS_SERVER_NAME` controls verified TLS SNI for endpoint aliases.

Each session requests and verifies `apriel.playback_clock="media"` and
`apriel.turn_clock="media"` before sending audio. The backend must support these
session options. A missing or conflicting clock acknowledgment fails the
connection rather than silently running with wall timing.

Prompts, tasks, tools, simulator behavior and scoring are not modified. The
branch's tracing default and model-selection configuration are preserved.

## Running

```bash
uv run tau2 run --domain airline --audio-native --audio-native-provider apriel --num-tasks 1
```
