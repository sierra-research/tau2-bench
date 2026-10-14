# BaiRong Voice Realtime provider

`bairong` reuses tau2's OpenAI Realtime event and tick implementation and connects
it to BaiRong's OpenAI-Realtime-compatible voice service.

The provider connects to the hosted BaiRong service by default:

```text
wss://model-api.resultscloud.com/realtime-service/v1/realtime
```

Run a tau-voice evaluation with:

```bash
uv run tau2 run --domain retail --audio-native \
  --audio-native-provider bairong \
  --audio-native-model BaiRong-Voice-Realtime \
  --num-tasks 1 --verbose-logs
```

The endpoint must implement the subset used by tau2's OpenAI provider:
`session.created`, `session.updated`, `input_audio_buffer.append`, response
audio deltas, response transcripts, speech-started events, response completion,
and function-call argument events.
