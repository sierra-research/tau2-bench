# Korean acoustic presets

The Korean continuous background beds are **real generated audio** (not
placeholder copies of the English files), produced by
`tau2.multilingual.factory.acoustic_generation` via
`tau2 factory generate-assets --lang ko` using the ElevenLabs Sound-Effects
API plus, for the household bed, a designed Korean TV-news broadcaster voice
muffled over kitchen ambience. Recipes are **authored on the fly** from the
pack's locale info (no hand-written `RECIPES['ko']` entry).

All files are **16 kHz PCM mono WAV**, matching the verified-format requirements
of the surrounding directories.

## Current beds (generated, in `continuous/ko_KR/`)

| File | Replaces (English) | How it's made |
| --- | --- | --- |
| `busy_street_iphone_mic.wav` | `busy_street_iphone_mic.wav` | Locale outdoor-traffic SFX prompt, crossfade-tiled to ~181 s |
| `medium_size_room_tv_news_iphone_mic.wav` | `medium_size_room_tv_news_iphone_mic.wav` | Kitchen SFX bed + designed Korean broadcaster TTS, ~150 s |

See [`acoustic_rationale.yaml`](./acoustic_rationale.yaml) for the exact prompts,
broadcaster script, voice id, models, and lengths.

## Bursts

Burst sounds are **constant across locales** and are NOT generated per-locale;
the presets reference the shared English burst files (`car_horn.wav`,
`engine_idling.wav`, `dog_bark.wav`).

Regenerate with `tau2 factory generate-assets --lang ko --acoustics-only --force`.
