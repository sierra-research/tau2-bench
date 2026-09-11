# India acoustic presets

The India continuous background beds are **real generated audio** (no longer
placeholder copies of the English files). They are produced by
`tau2.multilingual.factory.acoustic_generation` — run as part of language addition
via `tau2 factory generate-assets --lang hi` — using the ElevenLabs Sound-Effects
API plus, for the household bed, a designed Hindi TV-news broadcaster voice
muffled over kitchen ambience.

All files are **16 kHz PCM mono WAV**, matching the verified-format requirements
of the surrounding directories.

## Current beds (generated, in `continuous/hi_IN/`)

Filenames match the English bed each one replaces — the `hi_IN/` folder already
scopes them to the locale, so no extra prefix is needed.

| File | Replaces (English) | Content | How it's made |
| --- | --- | --- | --- |
| `busy_street_iphone_mic.wav` | `busy_street_iphone_mic.wav` | Close-mic chaotic Mumbai street traffic (horns, 2-wheelers, vendors, high noise floor) | SFX prompt, crossfade-tiled to ≈181 s |
| `medium_size_room_tv_news_iphone_mic.wav` | `medium_size_room_tv_news_iphone_mic.wav` | Clear Hindi TV news over live kitchen ambience | Kitchen SFX bed + designed broadcaster TTS, mixed at 0.85, ≈150 s |

`hi_IN_office` is **not yet localized** — its preset references the shared
English `people_talking.wav`.

## Bursts

Burst sounds are intentionally **constant across locales** and are NOT
generated per-locale. The India presets reference the shared English burst
files directly (`car_horn.wav`, `engine_idling.wav`, `dog_bark.wav`).

## Provenance & history

- **Why these acoustic choices + full generation provenance** (prompts,
  broadcaster script, voice id, models, lengths): see
  [`acoustic_rationale.yaml`](./acoustic_rationale.yaml).
- These beds replaced earlier placeholder copies of the English files (since
  removed); regenerate with `tau2 factory generate-assets --lang hi`.
