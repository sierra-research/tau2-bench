# Reproducing the Pine Realtime 1.0 Preview tau2-voice results

Pine is exposed as an OpenAI-Realtime-API-compatible endpoint, so the benchmark
connects to it through the stock audio-native adapter that ships in this PR
(`tau2.voice.audio_native.pine`). No harness modifications are required; you
only need (1) a Pine API key and (2) the standard tau2-bench user-simulator and
voice keys.

The public endpoint is `wss://api-preview.pinevoice.ai/v1/realtime`.

---

## 1. Get a Pine API key (email + verification code)

Authenticate through the Pine gateway. A Pine user-id cannot be derived from an
access token, so the realtime API resolves your user from a token → user mapping
the gateway maintains. Verifying *through the gateway* (the `api-preview.pinevoice.ai`
host below — **not** `19pine.ai` directly) registers that mapping as it issues your
token, so the token then works on the socket with a plain `Authorization: Bearer
<token>` header. Capture **both** the `access_token` and your user `id` from the
verify response — you need the `id` if you have to register the token manually
(see the note after step **b** and Troubleshooting).

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
# -> {"status":"success","data":{"access_token":"<durable Pine API key>","id":"<your user id>"}}
```

The `data.access_token` is your `PINE_API_KEY`; keep `data.id` (your user id) too.
Because you verified *through the gateway*, the token is normally registered with
the realtime API already — no extra headers needed.

**If your first run 401s on the WebSocket upgrade** (`invalid or expired
credential` / `token not registered …`), the token → user mapping wasn't seeded.
Register it once with a single call that carries your user id, then re-run:

```bash
curl -X POST https://api-preview.pinevoice.ai/v1/realtime/client_secrets \
  -H "Authorization: Bearer ${PINE_API_KEY}" \
  -H "X-Pine-User-Id: <data.id from above>" \
  -H "Content-Type: application/json" -d '{}'
# 200 here registers the token; you can discard the returned ephemeral secret.
```

(See https://tau-bench.pinevoice.ai/ for the same flow with a live demo.)

---

## 2. Install tau2-bench (this branch)

```bash
git clone <this-fork> tau2-bench && cd tau2-bench
git checkout submission/pine-realtime-1.0-preview
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

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401` on the WS upgrade — `token not registered with the realtime gateway: …` or `invalid or expired credential` | The `pine` provider presents `PINE_API_KEY` as a bearer token directly on the socket, and the gateway has no token → user mapping for it (verified off-gateway, or the verify response didn't seed it). | Run the one-time registration `curl` in step **1** (it sends `X-Pine-User-Id`), then re-run. Make sure you verified against `api-preview.pinevoice.ai`, not `19pine.ai`. |
| `Pine API key not provided. Set the PINE_API_KEY env var.` | `PINE_API_KEY` unset. | Export the `data.access_token` from step **1**. |
| Voice deps missing / `libportaudio.so.2: cannot open shared object file` | PortAudio not installed (or not on the linker path on minimal hosts). | `sudo apt-get install -y portaudio19-dev`, then `uv sync`. |
| `ELEVENLABS_API_KEY not found` | User-simulator TTS key unset. | Export `ELEVENLABS_API_KEY` (step **3**). |

---

## What "provider pine" does

`--audio-native-provider pine` selects `tau2.voice.audio_native.pine`, a thin
sibling of the OpenAI Realtime provider: it speaks the same wire protocol
(`session.update`, `input_audio_buffer.*`, `response.*`, function calls,
barge-in via `conversation.item.truncate`) and only changes the base URL and
bearer credential. The model label `pine-realtime-1.0-preview` is client-side
only; the server uses its configured model.
