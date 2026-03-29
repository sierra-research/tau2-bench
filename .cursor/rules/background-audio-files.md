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
