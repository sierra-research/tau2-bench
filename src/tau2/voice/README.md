# Voice (Full-Duplex)

τ-bench supports end-to-end voice evaluation using real-time audio APIs. In this mode, a user simulator streams synthesized speech to the agent, and the agent responds with audio — both sides operating simultaneously (full-duplex).

```bash
tau2 run --domain retail --audio-native --num-tasks 1 --verbose-logs
```

## Providers

| Provider | Flag | Requirements |
|----------|------|-------------|
| OpenAI Realtime | `--audio-native-provider openai` | `OPENAI_API_KEY` |
| Google Gemini Live | `--audio-native-provider gemini` | `GOOGLE_API_KEY` |
| xAI Grok Voice | `--audio-native-provider xai` | `XAI_API_KEY` |

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
| `--max-steps-seconds` | `600` | Maximum conversation duration |
| `--verbose-logs` | — | Save audio files, LLM logs, and tick data |

See `tau2 run --help` or the [CLI Reference](../../docs/cli-reference.md) for the full list including turn-taking thresholds and debugging options.

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

- **`audio_native/`** — Real-time provider adapters (OpenAI, Gemini, xAI). Each provider implements a `DiscreteTimeAdapter` that bridges the provider's streaming API to the tick-based simulation. See [audio_native/README.md](audio_native/README.md) for architecture details.

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
| `GOOGLE_API_KEY` | Gemini Live provider |
| `XAI_API_KEY` | xAI Grok Voice provider |
| `ELEVENLABS_API_KEY` | User simulator TTS (synthesis) |
| `DEEPGRAM_API_KEY` | Transcription (Deepgram nova-2, nova-3) |
| `TAU2_VOICE_ID_*` | Custom voice ID overrides (see [Voice Persona Setup](../../docs/voice-personas.md)) |
