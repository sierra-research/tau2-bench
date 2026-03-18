# Audio Effects

Audio effects processing for voice synthesis, simulating realistic telephony call environments.

## Overview

This module provides a comprehensive suite of audio effects that simulate real-world telephone call conditions. Effects are categorized into three layers:

| Layer | Description | Examples |
|-------|-------------|----------|
| **Speech Effects** | Applied to the speaker's voice | Dynamic muffling, vocal tics |
| **Source Effects** | Acoustic environment simulation | Background noise, burst noise |
| **Channel Effects** | Transmission/network artifacts | Frame drops, telephony conversion |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Audio Effects Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │   Speech     │   │   Source     │   │      Channel         │ │
│  │   Effects    │──▶│   Effects    │──▶│      Effects         │ │
│  └──────────────┘   └──────────────┘   └──────────────────────┘ │
│        │                   │                     │              │
│        ▼                   ▼                     ▼              │
│  • Dynamic muffling  • Background noise   • Telephony (μ-law)  │
│  • Vocal tics        • Burst noise        • Frame drops        │
│  • Non-directed      • Out-of-turn                             │
│    phrases             speech                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Modules

### `effects.py`

Core audio effect functions:

- **`apply_burst_noise(audio, burst_noise_file)`** - Overlays a burst sound (e.g., cough, door slam) at the center of the audio
- **`apply_frame_drops(audio, drop_count, drop_duration_ms)`** - Simulates packet loss by inserting silence gaps
- **`apply_dynamic_muffling(audio, segment_count, ...)`** - Applies low-pass filter to random segments (simulates speaker moving away from mic)
- **`apply_constant_muffling(audio, cutoff_freq)`** - Applies constant low-pass filter (speaker facing away)
- **`convert_to_telephony(audio)`** - Converts to μ-law 8kHz telephony format with bandpass filtering (300-3400Hz)
- **`StreamingTelephonyConverter`** - Stateful converter for streaming chunks without boundary artifacts

### `noise_generator.py`

Background noise generation with natural volume variations:

- **`BackgroundNoiseGenerator`** - Loops background noise with dynamic volume envelope using multi-frequency sinusoidal modulation
  - Supports one-shot burst sounds mixed into the stream
  - Poisson-based automatic burst triggering
- **`create_background_noise_generator(config, sample_rate, ...)`** - Factory function
- **`apply_background_noise(audio, noise_generator)`** - Mix noise into audio

### `speech_generator.py`

Out-of-turn speech synthesis:

- **`OutOfTurnSpeechGenerator`** - Pre-generates and caches speech for:
  - **Vocal tics**: "um", "uh", "hmm" (during speech)
  - **Non-directed phrases**: "one sec", "hold on" (during silence, muffled)
- **`create_streaming_audio_generators(config, voice_id, ...)`** - Creates both noise and speech generators

### `scheduler.py`

Effect scheduling using Poisson processes and Gilbert-Elliott model:

- **`EffectScheduler`** - Hybrid effect triggering
  - `burst_rate` - Burst noise events per second (Poisson)
  - `speech_insert_rate` - Out-of-turn speech events per second (Poisson)
  - Frame drops use Gilbert-Elliott model for bursty packet loss
- **`ScheduledEffect`** - Scheduled effect event with timing and parameters
- **`generate_turn_effects(seed, turn_idx, ...)`** - Generate per-turn effects for batch mode

Effect timings:
- `cross_turn` - Can occur anytime (burst noise, frame drops)
- `out_of_turn` - Only during silence (out-of-turn speech)
- `in_turn` - During active speech (vocal tics)

### `processor.py`

Processing mixins for batch and streaming modes:

- **`BatchAudioEffectsMixin`** - Apply all effects to complete audio segments
  - Processing order: Speech → Noise → Burst → Telephony → Frame drops
- **`StreamingAudioEffectsMixin`** - Process audio chunk-by-chunk
  - Handles effect state across chunk boundaries
  - Manages pending effects that span multiple chunks
- **`PendingEffectState`** - Tracks multi-chunk effect state

## Configuration

Effect configuration is defined in `tau2.data_model.voice`:

```python
from tau2.data_model.voice import (
    SpeechEffectsConfig,   # Muffling, vocal tics, non-directed phrases
    SourceEffectsConfig,   # Background noise, burst noise
    ChannelEffectsConfig,  # Frame drops, telephony
    SynthesisConfig,       # Combined configuration
)
```

### Complexity Presets

The scheduler supports complexity presets (`control`, `regular`) that adjust effect rates:

| Preset   | Burst Rate | Speech Insert Rate | Frame Drop Rate | Muffling |
|----------|------------|-------------------|-----------------|----------|
| control  | 0/min      | 0/min             | 0/min           | Off      |
| regular  | 2.0/min    | 1.0/min           | 1.0/min         | On       |

## Usage

### Batch Mode

```python
from tau2.voice.synthesis.audio_effects import (
    apply_burst_noise,
    apply_dynamic_muffling,
    apply_frame_drops,
    create_background_noise_generator,
    apply_background_noise,
)
from tau2.voice.synthesis.audio_effects.effects import convert_to_telephony

# Apply effects in order
audio = apply_dynamic_muffling(audio, segment_count=2, segment_duration_ms=500, cutoff_freq=1000)
audio = apply_background_noise(audio, noise_generator)
audio = apply_burst_noise(audio, burst_noise_file=Path("cough.wav"))
audio = convert_to_telephony(audio)  # μ-law 8kHz
audio = apply_frame_drops(audio, drop_count=2, drop_duration_ms=50)
```

### Streaming Mode

```python
from tau2.voice.synthesis.audio_effects import (
    BackgroundNoiseGenerator,
    EffectScheduler,
    StreamingAudioEffectsMixin,
    create_streaming_audio_generators,
)

# Create generators
noise_gen, speech_gen = create_streaming_audio_generators(
    synthesis_config=config,
    voice_id="voice_123",
    sample_rate=8000,
    background_noise_file=Path("office_noise.wav"),
)

# Create scheduler (uses GE model for frame drops, Poisson for others)
scheduler = EffectScheduler(
    seed=42,
    source_config=config.source_effects_config,
    speech_config=config.speech_effects_config,
    channel_config=config.channel_effects_config,
)

# Process chunks
class MyProcessor(StreamingAudioEffectsMixin):
    def process(self, speech_chunk, elapsed_ms):
        # Check for scheduled effects
        effects = scheduler.check_for_effects(
            chunk_duration_ms=100,
            is_silence=(speech_chunk is None),
            current_time_ms=elapsed_ms,
        )
        
        # Process chunk with effects
        result, source_effects, pending = self.process_streaming_chunk(
            speech_audio=speech_chunk,
            noise_generator=noise_gen,
            num_samples=800,
            scheduled_effects=effects,
            out_of_turn_generator=speech_gen,
        )
        return result
```

## Audio Format Requirements

Most effects require **PCM_S16LE** (16-bit signed little-endian PCM) input:
- Frame drops also support **μ-law** directly
- Telephony conversion outputs **μ-law 8kHz mono**

## Dependencies

- `numpy` - Array operations
- `scipy.signal` - Butterworth filters for muffling/bandpass
- `audioop` - Resampling with state preservation (streaming telephony)

