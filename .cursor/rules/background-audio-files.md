---
description: Guidelines for adding background noise and burst audio files
globs:
  - data/voice/background_noise_audio_pcm_mono_verified/**
  - src/tau2/voice/utils/convert_audio_files.py
---

# Adding Background Noise Audio Files

The authoritative guide for background noise audio files lives in:

**`src/tau2/voice/AGENTS.md`** — see the "Adding Background Noise Audio Files" section.

Refer to that file for:
- Format requirements (WAV, mono, PCM 16-bit 16kHz recommended)
- Directory layout (`continuous/` vs `bursts/`)
- Recommended durations
- Conversion workflow (`convert_audio_files.py`)
- Content guidelines

## Locale beds are generated, not hand-sourced

Per-locale continuous beds (e.g. `continuous/india/`) are **generated** by
`tau2.multilingual.factory.acoustic_generation` via
`tau2 factory generate-assets --lang <lang>` (ElevenLabs Sound-Effects API, plus
a designed TV-broadcaster voice muffled over kitchen ambience for indoor beds).
Per-locale prompts/scripts/lengths + the rationale live in that module's
`RECIPES` and are echoed into `<locale>/acoustic_rationale.yaml`. **Bursts stay
constant across locales** — presets reference the shared English burst files,
not locale copies. Don't hand-edit generated locale WAVs; change the recipe and
re-run.
