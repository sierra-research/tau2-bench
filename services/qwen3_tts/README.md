# Qwen3-TTS compatibility service

This standalone GPU service implements the one ElevenLabs endpoint used by
tau2's voice user simulator. It returns headerless, mono, signed 16-bit PCM at
16 kHz, matching `pcm_16000`.

## Start the service

Install it in the environment that already runs Qwen3-TTS:

```bash
pip install -r services/qwen3_tts/requirements.txt
```

Configure the model and the fallback cloned voice:

```bash
export QWEN_TTS_MODEL_PATH=/kpfs-data/huan.shen/code/ttsllm/Qwen3-TTS/pretrained_models/Qwen3-TTS-12Hz-1.7B-Base/
export QWEN_TTS_DEVICE=cuda:0
export QWEN_TTS_DTYPE=bfloat16
export QWEN_TTS_ATTN_IMPLEMENTATION=flash_attention_2
export QWEN_TTS_REF_AUDIO=/kpfs-data/huan.shen/dataset/Benchmark/Ref_audios/KeSpeech_Trimmed_Ref/audios/1000015_377f41d7.wav
export QWEN_TTS_REF_TEXT='行情回顾原油多头受库欣库存减少的支撑'
export QWEN_TTS_LANGUAGE=English
export QWEN_TTS_API_KEY=local-qwen

uvicorn services.qwen3_tts.app:app --host 0.0.0.0 --port 8008 --workers 1
```

Use exactly one worker per GPU because each worker loads its own model. Requests
inside a worker are serialized to keep model inference and seeding isolated.

## Connect tau2

On the machine or container running tau2:

```bash
export ELEVENLABS_BASE_URL=http://127.0.0.1:8008
export ELEVENLABS_API_KEY=local-qwen
```

If the service is remote, replace `127.0.0.1` with its reachable address. The
API key may be any non-empty value when `QWEN_TTS_API_KEY` is unset on the
service.

Smoke-test the compatible endpoint:

```bash
curl -sS -X POST \
  'http://127.0.0.1:8008/v1/text-to-speech/test?output_format=pcm_16000' \
  -H 'content-type: application/json' \
  -H 'xi-api-key: local-qwen' \
  -d '{"text":"Hello, I would like to check my order."}' \
  --output output.pcm

ffmpeg -f s16le -ar 16000 -ac 1 -i output.pcm output.wav
```

Then run a small clean-condition evaluation before enabling realistic effects:

```bash
uv run tau2 run --domain retail --audio-native --speech-complexity control \
  --num-tasks 1 --max-concurrency 1 --verbose-logs
```

## Multiple tau2 personas

By default every ElevenLabs `voice_id` uses `QWEN_TTS_REF_AUDIO` and
`QWEN_TTS_REF_TEXT`. To preserve distinct tau2 personas, copy
`voices.example.json`, add references keyed by the tau2 voice IDs, and set:

```bash
export QWEN_TTS_VOICES_FILE=/absolute/path/to/voices.json
```

Unknown voice IDs continue to use the fallback reference.

## Compatibility scope

The facade intentionally implements only the endpoint and `pcm_16000` format
used by tau2. ElevenLabs voice settings and model IDs are accepted but ignored.
Qwen3-TTS voice cloning does not reproduce ElevenLabs v3 cough, sneeze, or
sniffle tags; the service strips those tags and turns a tag-only request into
200 ms of silence.
