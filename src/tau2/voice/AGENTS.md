# AGENTS.md — src/tau2/voice/

> See `README.md` for full architecture, CLI usage, and audio effects data flow diagrams.

## Module Boundaries

| Submodule | Responsibility |
|-----------|---------------|
| `synthesis/` | Text-to-speech (ElevenLabs), audio effects pipeline |
| `synthesis/audio_effects/` | Three-tier effects system (speech/source/channel) |
| `transcription/` | Speech-to-text (Deepgram, OpenAI Whisper/Realtime) |
| `audio_native/` | Real-time voice providers (see `audio_native/AGENTS.md`) |
| `utils/` | Audio format conversion, WAV I/O, mixing, noise file paths |

## API Keys

| Provider | Env Variable | Used By |
|----------|-------------|---------|
| ElevenLabs | `ELEVENLABS_API_KEY` | TTS synthesis |
| Deepgram | `DEEPGRAM_API_KEY` | `nova-2`, `nova-3` transcription |
| OpenAI | `OPENAI_API_KEY` | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` |

## Audio Format Conventions

Different parts of the pipeline use different formats. Do not mix them up:

| Context | Format | Notes |
|---------|--------|-------|
| **Telephony (framework default)** | μ-law 8kHz mono | Used by orchestrator, audio native adapters |
| **TTS output** | PCM_S16LE 16kHz | Required by synthesis pipeline |
| **Transcription input** | PCM_S16LE mono 24kHz | Internal conversion before API calls |
| **Background/burst noise files** | PCM 16-bit mono 16kHz | Recommended; others auto-converted at runtime |

Canonical format constants are in `tau2.data_model.audio` (`TELEPHONY_AUDIO_FORMAT`, `AudioEncoding`, etc.).

## Config and Data Model Separation

- **Effect config classes** (`ChannelEffectsConfig`, `SourceEffectsConfig`, `SpeechEffectsConfig`) live in `tau2.data_model.voice` and `tau2.data_model.audio_effects` — NOT in the voice module.
- **Voice settings** (`VoiceSettings`) live in `tau2.data_model.voice`.
- **Voice-level constants** (noise directories, default rates) live in `src/tau2/voice_config.py` (at the `tau2` package level, not under `voice/`).
- Import config constants from `tau2.voice_config` or `tau2.config` — do not define local duplicates.

## Audio Effects Pipeline

### Three-Tier Taxonomy

| Tier | Config Class | Effects | When Applied |
|------|-------------|---------|-------------|
| **Speech** | `SpeechEffectsConfig` | Dynamic muffling, vocal tics | During active speech |
| **Source** | `SourceEffectsConfig` | Background noise, burst noise, out-of-turn speech | Any time (cross-turn or out-of-turn) |
| **Channel** | `ChannelEffectsConfig` | Telephony conversion, frame drops | Post-mixing |

### Batch Pipeline Order (Fixed)

```
Speech Effects → Background Noise → Burst Noise → Telephony (μ-law 8kHz) → Frame Drops
```

Do NOT change this order — it affects noise levels and SNR behavior.

### Streaming vs Batch

- `StreamingAudioEffectsMixin` — stateful, chunk-by-chunk processing with `PendingEffectState`
- `BatchAudioEffectsMixin` — stateless, full-audio processing
- `StreamingTelephonyConverter` keeps filter state between chunks to avoid boundary clicks. Reset on interruption.

### Key Components

- **`BackgroundNoiseGenerator`**: Looping background noise with SNR-based mixing. Also owns burst scheduling (Poisson).
- **`OutOfTurnSpeechGenerator`**: Pre-generates vocal tics and non-directed phrases via TTS. Non-directed phrases get muffling; vocal tics are clear.
- **`EffectScheduler`**: Poisson for burst noise/speech inserts, Gilbert-Elliott for frame drops.

### Audio Tags

`[cough]`, `[sneeze]`, `[sniffle]` only work with ElevenLabs v3 models. Using them with other models produces a warning.

## Adding Background Noise Audio Files

Audio files live in `data/voice/background_noise_audio_pcm_mono_verified/`.

### Format Requirements

**Hard requirements (validated in code):**

| Property | Requirement | What Happens If Violated |
|----------|-------------|--------------------------|
| File format | WAV (`.wav`) | Load fails |
| Channels | **Mono (1 channel)** | `ValueError` raised (`noise_generator.py`) |

**Recommended format (auto-converted at runtime):**

| Property | Recommended | Notes |
|----------|-------------|-------|
| Encoding | PCM 16-bit signed (`PCM_S16LE`) | Other encodings auto-converted |
| Sample rate | **16000 Hz (16kHz)** | Other rates auto-resampled |

Why 16kHz? Upsampling from lower rates degrades quality. Pre-convert to avoid runtime resampling artifacts.

### Directory Layout

| Type | Directory | Duration | Description |
|------|-----------|----------|-------------|
| **Continuous** | `continuous/` | At least 30s | Looping ambient (busy street, TV, people talking) |
| **Burst** | `bursts/` | Max 5s | Short sounds (car horn, dog bark, siren) |

### Conversion Tool

```bash
python -m tau2.voice.utils.convert_audio_files \
    /path/to/input/dir \
    /path/to/output/dir \
    --sample-rate 16000
```

Accepts PCM WAV (8/16/24/32-bit), μ-law, A-law, and stereo input.

### Content Guidelines

- Files are automatically normalized to prevent clipping.
- SNR-based mixing means raw volume matters less than audio quality.
- Seamless loops are ideal for continuous files but not required.

## Gotchas

1. **Two `conversation_builder` modules**: `synthesis/conversation_builder.py` (Audacity labels, stereo merge) vs `utils/conversation_builder.py` (merge audio from messages). Different responsibilities — don't confuse them.
2. **Noise files loaded at import**: `utils/utils.py` loads `CONTINUOUS_NOISE_FILES` and `BURST_NOISE_FILES` at import time. Missing directories or empty directories produce warnings.
3. **Graceful failure**: If synthesis fails, the message is sent without audio. If transcription fails, original text is preserved. Errors are logged but never interrupt the simulation.
