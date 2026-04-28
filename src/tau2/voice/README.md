# Voice (Full-Duplex)

τ-bench supports end-to-end voice evaluation using real-time audio APIs. In this mode, a user simulator streams synthesized speech to the agent, and the agent responds with audio — both sides operating simultaneously (full-duplex).

```bash
tau2 run --domain retail --audio-native --num-tasks 1 --verbose-logs
```

## Providers

| Provider | Flag | Requirements |
|----------|------|-------------|
| OpenAI Realtime | `--audio-native-provider openai` | `OPENAI_API_KEY` |
| Google Gemini Live | `--audio-native-provider gemini` | `GEMINI_API_KEY` or Vertex AI credentials |
| xAI Grok Voice | `--audio-native-provider xai` | `XAI_API_KEY` |
| Amazon Nova Sonic | `--audio-native-provider nova` | AWS credentials |
| Alibaba Qwen Omni | `--audio-native-provider qwen` | `DASHSCOPE_API_KEY` |
| Boson realtime voice chat | `--audio-native-provider boson` | `BOSON_API_KEY` |
| LiveKit cascaded voice | `--audio-native-provider livekit` | LiveKit and cascaded provider credentials |

The default provider is `openai`. Use `--audio-native-model` to override the default model for a provider.

## Speech Complexity

The `--speech-complexity` flag controls the realism of the user simulator's speech environment:

| Preset | Description |
|--------|-------------|
| `control` | Clean baseline — no audio effects, American accents, patient user |
| `regular` | Full realistic conditions — background noise, accents, interruptions |

Ablation presets isolate individual factors: `control_audio`, `control_accents`, `control_behavior`, and pairwise combinations (`control_audio_accents`, `control_audio_behavior`, `control_accents_behavior`).

```bash
# Clean baseline
tau2 run --domain retail --audio-native --speech-complexity control

# Full realistic conditions (default)
tau2 run --domain retail --audio-native --speech-complexity regular
```

## Key CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--audio-native` | — | Enable voice full-duplex mode |
| `--audio-native-provider` | `openai` | Provider to use (see table above) |
| `--audio-native-model` | per-provider | Override model |
| `--speech-complexity` | `regular` | Speech complexity level |
| `--tick-duration` | `0.2` | Simulation timestep in seconds |
| `--max-steps-seconds` | `1200` | Maximum conversation duration |
| `--verbose-logs` | — | Save audio files, LLM logs, and tick data |

See `tau2 run --help` or the [CLI Reference](../../docs/cli-reference.md) for the full list including turn-taking thresholds and debugging options.

## End-to-End Examples

### Boson Realtime End-to-End Smoke Test

This runs a small full-duplex τ-bench voice simulation with the Boson realtime
voice chat provider. The user side is still τ-bench's voice user simulator:
it uses an LLM to decide what the customer says, ElevenLabs to synthesize the
customer audio, and Boson to handle the agent's realtime speech-to-speech API.

1. Install the voice and test dependencies.

```bash
uv sync --extra voice --extra dev
```

2. Create a local `.env` and set the required keys.

```bash
cp .env.example .env
```

Add these values to `.env`:

```bash
# Agent realtime voice API
BOSON_API_KEY=<your_boson_key>

# Default tau2 user simulator / evaluator LLMs
OPENAI_API_KEY=<your_openai_key>

# User simulator speech synthesis
ELEVENLABS_API_KEY=<your_elevenlabs_key>
```

`BOSON_REALTIME_URL` is optional. If it is not set, tau2 uses the default
Boson staging endpoint configured in `tau2.config`.

3. Make sure the control voice personas are configured.

For an end-to-end voice simulation, the user simulator needs valid ElevenLabs
voice IDs. If your `.env` already has `TAU2_VOICE_ID_MATT_DELANEY` and
`TAU2_VOICE_ID_LISA_BRENNER`, you can skip this setup command. Otherwise,
create the minimal control personas:

```bash
uv run python -m tau2.voice.scripts.setup_voices --complexity control
```

The script prints `TAU2_VOICE_ID_*` lines. Add the printed values to `.env`,
at minimum:

```bash
TAU2_VOICE_ID_MATT_DELANEY=<voice_id_from_setup_script>
TAU2_VOICE_ID_LISA_BRENNER=<voice_id_from_setup_script>
```

Do not use public/library voice IDs here unless your ElevenLabs plan supports
library voices through the API. A `402 paid_plan_required` error from
ElevenLabs means the voice ID is not API-usable on your current plan; replace it
with a voice saved in your own account.

4. Run a small end-to-end Boson voice simulation.

```bash
uv run tau2 run \
  --domain retail \
  --audio-native \
  --audio-native-provider boson \
  --user-llm gpt-4.1-2025-04-14 \
  --speech-complexity control \
  --num-trials 1 \
  --num-tasks 5 \
  --verbose-logs \
  --save-to boson_realtime_smoke \
  --max-concurrency 1
```

The default Boson model is `Qwen2.5-72B-Instruct`. To test a different Boson
model, add `--audio-native-model <model-name>`.

5. Inspect the output.

The run writes results under `data/simulations/boson_realtime_smoke/`. With
`--verbose-logs`, the most useful files are:

```text
data/simulations/boson_realtime_smoke/
├── results.json
├── simulations/
└── artifacts/
    └── task_<id>/
        └── sim_<uuid>/
            ├── task.log
            ├── audio/
            │   ├── both.wav
            │   ├── assistant_labels.txt
            │   └── user_labels.txt
            └── llm_debug/
```

6. Inspect the run with `tau2 view`.

For the smoke-test run above, open the terminal result viewer directly on the
saved `results.json` file:

```bash
uv run tau2 view --file data/simulations/boson_realtime_smoke/results.json
```

For full-duplex voice debugging, add `--expanded-ticks` to inspect the
tick-by-tick stream state:

```bash
uv run tau2 view \
  --file data/simulations/boson_realtime_smoke/results.json \
  --expanded-ticks
```

To browse all saved simulation runs and select the smoke test interactively,
point `tau2 view` at the parent simulations directory:

```bash
uv run tau2 view --dir data/simulations
```

7. View the results in a browser.

From the repository root, start the local result viewer server:

```bash
uv run python result_view_website/server.py \
  --results data/simulations/boson_realtime_smoke \
  --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

By default, the server binds to `127.0.0.1`, so it is only reachable from the
same machine. If you are running it on a remote host and want to access it by
hostname, bind to all interfaces:

```bash
uv run python result_view_website/server.py \
  --results data/simulations/boson_realtime_smoke \
  --host 0.0.0.0 \
  --port 8765
```

Then open `http://<host>:8765/`. If the page still does not load, the port may
be blocked by firewall or network policy. In that case, use SSH port forwarding:

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<host>
```

Then open `http://127.0.0.1:8765/` on your local machine.

The viewer is read-only. Use the left sidebar to choose a simulation, then use
the tabs to inspect the grouped conversation timeline, expanded ticks, task
definition, reward/review details, run config, and raw simulation JSON. If the
run was created with `--verbose-logs` and `both.wav` exists, the audio player at
the top of the detail pane plays the full conversation audio.

To browse multiple saved runs from the same server, point `--results` at the
parent simulations directory instead:

```bash
uv run python result_view_website/server.py \
  --results data/simulations \
  --port 8765
```

For a lower-level provider smoke test, run the shared adapter suite with
`BOSON_API_KEY` set:

```bash
uv run pytest tests/test_voice/test_audio_native/test_provider_suite.py -v -s \
  -k "boson and (test_connect_disconnect or test_single_turn_reply)"
```

## Programmatic Usage

```python
from tau2 import VoiceRunConfig
from tau2.data_model.simulation import AudioNativeConfig
from tau2 import run_domain

config = VoiceRunConfig(
    domain="airline",
    audio_native_config=AudioNativeConfig(
        provider="openai",
        model="gpt-4o-realtime-preview",
    ),
    llm_user="openai/gpt-4.1",
    speech_complexity="regular",
)

results = run_domain(config)
```

See [Running Simulations](../../docs/running_simulations.md) for more examples and instance-level control.

## Output Structure

With `--verbose-logs`, voice runs produce:

```
data/simulations/<run_name>/
├── results.json                        # Metadata and task definitions
├── simulations/                        # Individual simulation data files
│   ├── sim_0.json
│   └── ...
└── artifacts/
    └── task_<id>/
        └── sim_<uuid>/
            ├── sim_status.json         # Simulation status
            ├── task.log                # Per-task log
            ├── audio/
            │   ├── both.wav            # Full conversation audio (stereo)
            │   ├── assistant_labels.txt # Audacity labels for agent speech
            │   ├── user_labels.txt     # Audacity labels for user speech
            │   └── assistant_tool_calls_labels.txt
            └── llm_debug/
                └── *.json              # LLM call logs
```

Voice runs use a directory-based storage format: `results.json` holds metadata and task definitions, while each simulation is stored as a separate file under `simulations/`. Runtime artifacts (audio, logs) live under `artifacts/`.

## Architecture

The voice module has two main components:

- **`audio_native/`** — Real-time provider adapters (OpenAI, Gemini, xAI, Nova, Qwen, Boson, LiveKit). Each provider implements a `DiscreteTimeAdapter` that bridges the provider's streaming API to the tick-based simulation. See [audio_native/README.md](audio_native/README.md) for architecture details.

- **`synthesis/`** — User simulator speech generation. Converts user text to audio via ElevenLabs TTS, applies audio effects (background noise, burst sounds, frame drops), and converts to telephony format (G.711 μ-law 8kHz).

- **`transcription/`** — Speech-to-text for evaluation. Supports Deepgram (nova-2, nova-3) and OpenAI (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe).

- **`utils/`** — Audio format conversion, WAV I/O, and shared helpers.

## Voice Persona Setup

The user simulator uses ElevenLabs voices defined in `src/tau2/data_model/voice_personas.py`. The default voice IDs are Sierra-internal and **will not work** for external users.

To run voice evaluations, create your own voices in ElevenLabs and configure them via environment variables:

```bash
# In your .env file:
TAU2_VOICE_ID_MATT_DELANEY=your_voice_id_here
TAU2_VOICE_ID_LISA_BRENNER=your_voice_id_here
# ... (one per persona)
```

For a minimal setup, create just the two control personas and use `--speech-complexity control`.

See the [Voice Persona Setup Guide](../../docs/voice-personas.md) for step-by-step instructions on creating matching voices with ElevenLabs Voice Design.

## Environment Variables

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI Realtime provider |
| `GEMINI_API_KEY` | Gemini Live provider via AI Studio |
| `GOOGLE_SERVICE_ACCOUNT_KEY` / `GOOGLE_APPLICATION_CREDENTIALS` | Gemini Live provider via Vertex AI |
| `XAI_API_KEY` | xAI Grok Voice provider |
| `DASHSCOPE_API_KEY` | Alibaba Qwen Omni provider |
| `BOSON_API_KEY` | Boson realtime voice chat provider |
| `BOSON_REALTIME_URL` | Optional Boson WebSocket endpoint override |
| `ELEVENLABS_API_KEY` | User simulator TTS (synthesis) |
| `DEEPGRAM_API_KEY` | Transcription (Deepgram nova-2, nova-3) |
| `TAU2_VOICE_ID_*` | Custom voice ID overrides (see [Voice Persona Setup](../../docs/voice-personas.md)) |
