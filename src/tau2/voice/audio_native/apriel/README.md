# Apriel Realtime provider

This integration connects tau to an Apriel realtime gateway through an
OpenAI-compatible WebSocket protocol.

The adapter is intentionally thin: it mirrors tau's OpenAI discrete-time runner
behavior (audio pacing, transcript alignment, interruption handling, tool-call
gating), but uses an Apriel-configured realtime provider and a local Apriel
`tick_runner` implementation.

## Configuration

Required environment variables:

```bash
APRIEL_REALTIME_BASE_URL=wss://your-host/v1/realtime
APRIEL_API_KEY=...
APRIEL_MODEL_NAME=Apriel-Voice-27B-Preview
```

## Running

```bash
uv run tau2 run --domain airline --audio-native --audio-native-provider apriel --num-tasks 1
```
