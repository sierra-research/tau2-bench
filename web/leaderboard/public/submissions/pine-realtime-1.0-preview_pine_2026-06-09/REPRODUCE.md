# Reproducing the Pine Realtime 1.0 Preview tau2-voice results

Pine is exposed as an OpenAI-Realtime-API-compatible endpoint, so the benchmark
connects to it through the stock audio-native adapter that ships in this PR
(`tau2.voice.audio_native.pine`). No harness modifications are required; you
only need (1) a Pine API key and (2) the standard tau2-bench user-simulator and
voice keys.

The public endpoint is `wss://api-preview.pinevoice.ai/v1/realtime`.

---

## 1. Get a Pine API key (email + verification code)

Authenticate through the Pine gateway. It proxies email verification and caches
the token → user mapping as it issues your token, so the token then works
against the realtime API with a plain `Authorization: Bearer <token>` header.

**a. Request a code** — a 4-digit code is emailed to you:

```bash
curl -X POST https://api-preview.pinevoice.ai/api/v2/auth/setup \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'
# -> {"status":"success","data":{"request_token":"abc..."}}
```

Save the `request_token` and check your inbox for the 4-digit code.

**b. Verify the code** — exchange it for your durable access token:

```bash
curl -X POST https://api-preview.pinevoice.ai/api/v2/auth/verify \
  -H "Content-Type: application/json" \
  -d '{
        "email": "you@example.com",
        "code": "1234",
        "request_token": "abc..."
      }'
# -> response includes "access_token": "<your durable Pine API key>"
```

The `access_token` is your `PINE_API_KEY`. Because you verified *through the
gateway*, it is already registered with the realtime API — no extra headers
needed. (See https://tau-bench.pinevoice.ai/ for the same flow with a live demo.)

---

## 2. Install tau2-bench (this branch)

```bash
git clone <this-fork> tau2-bench && cd tau2-bench
git checkout submission/pine-realtime-1.0-preview-2026-06-09
uv sync                      # installs the `voice` extras (websockets, pyaudio, scipy, ...)
```

`pyaudio` builds against PortAudio; on Debian/Ubuntu install it first with
`sudo apt-get install -y portaudio19-dev`.

---

## 3. Set environment variables

```bash
# Pine agent (the system under test)
export PINE_API_KEY="<your Pine access_token from step 1>"
# PINE_BASE_URL defaults to wss://api-preview.pinevoice.ai/v1/realtime

# Standard tau2-bench user-simulator + grader (OpenAI; needs gpt-4.1 access)
export OPENAI_API_KEY="<your OpenAI key>"

# User-side TTS voice (ElevenLabs) — official tau2-bench voice setup
export ELEVENLABS_API_KEY="<your ElevenLabs key>"
```

This submission used `openrouter/openai/gpt-5.2` as the simulated user via
`--user-llm` (set `OPENROUTER_API_KEY`); any tau2-supported user LLM works.

> Note: the per-persona `TAU2_VOICE_ID_*` overrides used to produce the
> published audio are non-official voice IDs. Omit them to use the official
> tau2-bench default voices (recommended for leaderboard-comparable numbers).

---

## 4. Run

Single task (smoke test):

```bash
uv run tau2 run \
  --domain airline --task-ids 0 \
  --audio-native --audio-native-provider pine \
  --user-llm openrouter/openai/gpt-5.2 \
  --speech-complexity regular \
  --num-trials 1 --max-concurrency 1 \
  --save-to my_pine_run
```

Full domain (all tasks):

```bash
for d in airline retail telecom; do
  uv run tau2 run \
    --domain "$d" --audio-native --audio-native-provider pine \
    --user-llm openrouter/openai/gpt-5.2 \
    --speech-complexity regular --num-trials 1 \
    --save-to "pine_${d}"
done
```

The agent loads as `discrete_time_audio_native_agent → pine/pine-realtime-1.0-preview`.
Results (per-task reward, transcripts, audio) are written under
`data/simulations/<save-to>/`.

---

## What "provider pine" does

`--audio-native-provider pine` selects `tau2.voice.audio_native.pine`, a thin
sibling of the OpenAI Realtime provider: it speaks the same wire protocol
(`session.update`, `input_audio_buffer.*`, `response.*`, function calls,
barge-in via `conversation.item.truncate`) and only changes the base URL and
bearer credential. The model label `pine-realtime-1.0-preview` is client-side
only; the server uses its configured model.
