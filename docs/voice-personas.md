# Voice Persona Setup

The voice evaluation pipeline uses [ElevenLabs](https://elevenlabs.io/) to synthesize user simulator speech. Each caller persona in a language pack has an ElevenLabs **voice ID** that determines how the simulated caller sounds. Every voice-v2 language (`en`, `es`, `pt`, `hi`, `zh`, `ko`) has two personas, one male and one female; English voice runs speak with `mamadou_diallo` and `priya_patil`.

The default voice IDs in the codebase are Sierra's internal voices and **will not work** for external users. This guide walks you through creating your own voices and configuring the framework to use them.

> **Note:** Sierra runs final evaluations with its own voices to ensure parity across published results. Your custom voices will sound different but are functionally equivalent for development and experimentation.

## Prerequisites

- An [ElevenLabs](https://elevenlabs.io/) account (free tier works for testing)
- `ELEVENLABS_API_KEY` set in your `.env` file

## Automated Setup (Recommended)

The script calls the ElevenLabs Voice Design API to create a voice for each persona of the selected language packs and prints the environment variables to paste into your `.env`. The prompts, design model and native audition line for each persona live in `data/tau2/voice_design/<lang>.yaml`. It uses a fixed seed for reproducibility, so everyone running the script gets the same voices.

```bash
# Create the caller voices for every voice-v2 language (12 voices)
python -m tau2.voice.scripts.setup_voices --language all

# Create the English caller voices only (the default)
python -m tau2.voice.scripts.setup_voices

# Pick languages (repeatable)
python -m tau2.voice.scripts.setup_voices --language es --language hi

# Dry run — see what would be created (also shows existing tau2 voices)
python -m tau2.voice.scripts.setup_voices --language all --dry-run

# Preview each voice audio before saving
python -m tau2.voice.scripts.setup_voices --preview
```

The script uses a fixed random seed (`--seed 42` by default) so that the same prompt always produces the same voice. You can override it with `--seed <value>`.

Each persona is designed with the model recorded in its prompt file (`eleven_ttv_v3`). To use the older ElevenLabs TTV v2 model for every persona instead:

```bash
python -m tau2.voice.scripts.setup_voices --model eleven_multilingual_ttv_v2
```

Voices are saved as `tau2_<lang>_<persona>` (for example `tau2_es_camila_es_v1`), so languages never collide.

**Duplicate detection:** If voices named `tau2_*` already exist in your ElevenLabs account, the script will list them and ask what to do: **replace** (delete old, create new), **duplicate** (keep both), **skip** (only create missing), or **abort**. Use `--force` to skip the prompt and replace automatically.

The script outputs `TAU2_VOICE_ID_<LANG>_<PERSONA>=...` lines (for example `TAU2_VOICE_ID_ES_CAMILA_ES_V1=...`) — copy them into your `.env` file and you're done. Skip to [Step 3: Verify](#step-3-verify).

## Manual Setup via ElevenLabs UI

> **Note:** The [automated setup](#automated-setup-recommended) above is strongly recommended. It ensures correct parameters (seed, loudness, guidance scale, model) and avoids manual errors.

### Step 1: Create Voices with Voice Design

1. Go to [ElevenLabs Voice Library](https://elevenlabs.io/app/voice-lab) (or navigate to **Voices** in the ElevenLabs dashboard)
2. Click **Add a voice** → **Voice Design** (the "voice from prompt" option)
3. For each persona, paste its `prompt` from `data/tau2/voice_design/<lang>.yaml` into the voice description field, and use its `audition_text` as the preview text
4. Set the following parameters:
   - **Model**: the persona's `model` (Eleven v3)
   - **Loudness**: 75%
   - **Guidance scale**: 38%
5. Generate and save the voice
6. Copy the **Voice ID** from the voice details page (click the voice → look for the ID string)

### Persona Prompts

Every persona's prompt is in `data/tau2/voice_design/<lang>.yaml`, the file the [automated setup script](#automated-setup-recommended) reads. The English prompts are shown here as an example.

**Mamadou Diallo** (`mamadou_diallo`) — Senegalese man in his mid-30s, hurried with French accent

> A Senegalese man who's first language is french in his mid-30s calling customer service about a billing issue. He speaks English with a strong French accent. His tone is hurried, slightly annoyed, and matter-of-fact, as if he's been transferred between agents and just wants the problem fixed.

**Priya Patil** (`priya_patil`) — Maharashtrian woman in her early 30s, hurried and focused

> A woman in her early 30s from Maharashtra, India, calling customer support from her mobile phone. She speaks Indian English with a strong Maharashtrian accent — noticeable regional intonation and rhythm. Her tone is slightly annoyed and hurried, matter-of-fact, and focused on getting the issue resolved quickly. Her voice has medium pitch, firm delivery, short sentences, and faint background room tone typical of a phone call.

### Step 2: Configure Voice IDs

Set one environment variable per persona in your `.env` file. The framework picks these up automatically — no code changes needed.

```bash
TAU2_VOICE_ID_EN_MAMADOU_DIALLO=your_mamadou_voice_id
TAU2_VOICE_ID_EN_PRIYA_PATIL=your_priya_voice_id
TAU2_VOICE_ID_ES_ALEJANDRO_ES_V1=your_alejandro_voice_id
TAU2_VOICE_ID_ES_CAMILA_ES_V1=your_camila_voice_id
# ... one line per persona for pt, hi, zh and ko
```

The pattern is `TAU2_VOICE_ID_<LANG>_<PERSONA>`, upper-cased, and it works the same way for every language pack (see [Language Packs](language-packs.md#voice-id-overrides)). If a variable is not set, the pack's built-in voice ID is used (which only works for Sierra-internal use). Runs that use an override log a warning and record the substituted voice; their results are not comparable to released runs.

## Step 3: Verify

> Both the automated script and manual setup paths converge here.

Run a quick voice simulation:

```bash
tau2 run --domain retail --audio-native --benchmark none --num-tasks 1 --verbose-logs
```

Check the generated audio in the output directory to confirm the voices sound correct.

## How It Works

The voice ID resolution follows this priority:

1. **Environment variable** `TAU2_VOICE_ID_<LANG>_<PERSONA>` (if set)
2. **Built-in default** in the pack's `data/tau2/multilingual/<lang>/pack.yaml`

The persona's behavior (register, interruptions, speech inserts) comes from the pack regardless of which voice ID is used. Only the ElevenLabs voice that renders the speech changes.

The older per-persona catalog in `src/tau2/data_model/voice_personas.py` (`matt_delaney`, `lisa_brenner`, … with `TAU2_VOICE_ID_<PERSONA>` overrides) no longer selects the caller voice in `tau2 run` or supplies the setup prompts; it is used only by the standalone synthesis tools.

## Regenerating Pre-sampled Voice Configs

If you use pre-sampled voice configurations (`tasks_voice.json` files in domain data directories), note that these files store persona *names*, not voice IDs. The voice ID is resolved at synthesis time from the persona name. So your custom voice IDs will be picked up automatically without regenerating these files.
