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

## Cartesia customer speech

The customer simulator can use Cartesia TTS independently of the agent provider.
Install the voice dependencies and set `CARTESIA_API_KEY`, then run:

```bash
tau2 run --domain retail --audio-native --num-tasks 1 --num-trials 1 \
  --speech-complexity control --verbose-logs \
  --user-tts-config examples/voice/cartesia-customer.json
```

This example uses the single Hao stock voice for **all** customer personas and
pins `sonic-3.6-2026-08-27`. It is a smoke-test configuration, not a reproduction
of the standard customer accents. The command retains the default OpenAI agent
and therefore also needs OpenAI access. It does not configure the agent's ASR or
TTS; customer synthesis and the evaluated agent are independent.

For persona-specific voices, remove the example's `voice_id` and supply
`provider_config.persona_voice_ids`, mapping persona names such as `matt_delaney`
and `lisa_brenner` to Cartesia voice IDs. Alternatively set
`TAU2_CARTESIA_VOICE_ID_MATT_DELANEY`, `TAU2_CARTESIA_VOICE_ID_LISA_BRENNER`, etc.
Resolution order is JSON persona mapping, persona environment variable, then an
explicit shared `voice_id`. Missing voices fail instead of falling back to an
ElevenLabs ID. Choose and audition voices matching the intended personas before
comparing configurations. The standard `TAU2_VOICE_ID_*` variables remain
ElevenLabs-only.

The customer text LLM, policies, task definitions, and scoring are unchanged.
Speech uses 16 kHz mono PCM before the existing noise, muffling, telephony, and
frame-loss processing. Normal utterances, backchannels, and non-directed phrases
all use the selected Cartesia voice. `[pause]` becomes a 500 ms Cartesia break.
ElevenLabs-only cough/sneeze/sniffle inserts are disabled **after** task presets
are loaded, including for `regular`; this is recorded in the task's effective
speech settings. Those tags are rejected if they reach the Cartesia API helper.
Other speech conditions and their timing settings remain in place.

These are **modified customer-simulator conditions**, not directly comparable
with the standard ElevenLabs results. Saved settings include the provider,
model snapshot, API version, resolved task voice IDs, and effective effects.
Credentials are read from the environment and are not part of the Cartesia
configuration. No ElevenLabs API key is needed for Cartesia customer synthesis.

Programmatically, pass `VoiceSettings(transcription_config=None,
synthesis_config=SynthesisConfig(provider="cartesia",
provider_config=CartesiaTTSConfig(...)))` as `VoiceRunConfig.user_voice_settings`.

To check only the customer side (one short Cartesia request; no agent/LLM API):

```bash
uv run examples/voice/cartesia_customer_smoke.py
```

It saves a WAV and effective configuration under
`data/simulations/cartesia-customer-smoke/` after passing through the real
customer simulator's synthesis and telephony conversion.

### Ten conversations with Cartesia on both sides

With `ANTHROPIC_API_KEY` and `CARTESIA_API_KEY` in the environment or `.env`:

```bash
uv run examples/voice/run_cartesia_customer_examples.py
```

This runs retail tasks 0, 2, 5, 10, 12, 15, 16, 17, 18, and 21 once,
with two conversations at a time and a 300-second conversation limit.
The evaluated agent uses LiveKit's Cartesia Ink 2 ASR, Claude Haiku 4.5
(`claude-haiku-4-5-20251001`), and Cartesia Sonic 3.6 with the Hao voice.
The customer uses Haiku 4.5 for dialogue and the same Cartesia TTS snapshot.
The `control` speech setting uses clean audio. This is an example run of a
cascaded agent, not a test of Cartesia Managed Agents orchestration.
Anthropic requests receive a "Please continue." user cue when initializing
the customer or prompting it again after silence; this avoids assistant prefill.

Results and audio are saved under `data/simulations/cartesia-haiku-both-10/`.
Cartesia usage quantities are recorded, but total dollar cost is unavailable
because the pricing table does not encode Cartesia's plan-dependent credits.
To run live adapter validation, set `CARTESIA_TEST_ENABLED=1` and run
`uv run tests/test_voice/test_audio_native/run_provider_suite.py`; its dollar-cost
coverage assertion currently fails for Cartesia for that reason.

If a batch is interrupted or blocked by API billing, rerun the example script
with `--auto-resume` after resolving the issue. Completed calls are retained;
infrastructure failures are retried. Use
`uv run examples/voice/summarize_cartesia_examples.py` to export readable
transcripts, tool calls, and `example-index.json` alongside the recordings.
