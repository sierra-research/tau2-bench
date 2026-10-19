# Voice Persona Setup

The voice evaluation pipeline uses [ElevenLabs](https://elevenlabs.io/) to synthesize user simulator speech. Each voice persona has an associated ElevenLabs **voice ID** that determines how the simulated caller sounds.

The default voice IDs in the codebase are Sierra's internal voices and **will not work** for external users. This guide walks you through creating your own voices and configuring the framework to use them.

> **Note:** Sierra runs final evaluations with its own voices to ensure parity across published results. Your custom voices will sound different but are functionally equivalent for development and experimentation.

## Prerequisites

- An [ElevenLabs](https://elevenlabs.io/) account (free tier works for testing)
- `ELEVENLABS_API_KEY` set in your `.env` file

## Automated Setup (Recommended)

**This is the recommended way to set up voices.** The script calls the ElevenLabs Voice Design API to create all voices with the correct parameters and prints the environment variables to paste into your `.env`. It uses a fixed seed for reproducibility, so everyone running the script gets the same voices.

```bash
# Create all 7 voices
python -m tau2.voice.scripts.setup_voices

# Create only the 2 control personas (for quick testing)
python -m tau2.voice.scripts.setup_voices --complexity control

# Create only the 5 regular personas
python -m tau2.voice.scripts.setup_voices --complexity regular

# Dry run — see what would be created (also shows existing tau2 voices)
python -m tau2.voice.scripts.setup_voices --dry-run

# Preview each voice audio before saving
python -m tau2.voice.scripts.setup_voices --preview
```

The script uses a fixed random seed (`--seed 42` by default) so that the same prompt always produces the same voice. You can override it with `--seed <value>`.

To use the older ElevenLabs TTV v2 model instead of the default v3:

```bash
python -m tau2.voice.scripts.setup_voices --model eleven_multilingual_ttv_v2
```

**Duplicate detection:** If voices named `tau2_*` already exist in your ElevenLabs account, the script will list them and ask what to do: **replace** (delete old, create new), **duplicate** (keep both), **skip** (only create missing), or **abort**. Use `--force` to skip the prompt and replace automatically.

The script outputs a block of `TAU2_VOICE_ID_*=...` lines — copy them into your `.env` file and you're done. Skip to [Step 3: Verify](#step-3-verify).

## Manual Setup via ElevenLabs UI

> **Note:** The [automated setup](#automated-setup-recommended) above is strongly recommended. It ensures correct parameters (seed, loudness, guidance scale, model) and avoids manual errors. Only use the manual approach if you need fine-grained control over individual voice generation (e.g., regenerating until you find a voice you like).

### Step 1: Create Voices with Voice Design

ElevenLabs Voice Design lets you create voices from a text prompt.

1. Go to [ElevenLabs Voice Library](https://elevenlabs.io/app/voice-lab) (or navigate to **Voices** in the ElevenLabs dashboard)
2. Click **Add a voice** → **Voice Design** (the "voice from prompt" option)
3. For each persona below, paste the prompt shown here into the voice description field
4. Set the following parameters:
   - **Language**: English
   - **Loudness**: 75%
   - **Guidance scale**: 38%

   The UI shows guidance as a slider percentage, which is not the API value. The automated script uses the API default `guidance_scale=5` (override with `--guidance-scale`).
5. Generate and save the voice
6. Copy the **Voice ID** from the voice details page (click the voice → look for the ID string)

### Persona Prompts

Use these prompts to create voices that match the built-in personas. These are the same prompts defined in `src/tau2/data_model/voice_personas.py` and used by the [automated setup script](#automated-setup-recommended).

#### Control Personas (American accents — used in `control` complexity)

**Matt Delaney** — Middle-aged white man from the American Midwest, calm and respectful

> You are a middle-aged white man from the American Midwest. You always speak as if in a real-time conversation with a customer service agent. You are calm, clear, and respectful — but also human. You sound like someone trying to be helpful and polite, even when slightly frustrated or in a hurry. You value efficiency but never sound robotic.
>
> You sometimes use contractions, informal phrasing, or small filler phrases ("yeah," "okay," "honestly," "no worries") to keep things natural. You sometimes repeat words or self-correct mid-sentence, like someone thinking aloud. You sometimes ask clarifying questions or offer context ("I tried this earlier today," "I'm not sure if that helps").
>
> You rarely use formal or stiff language ("considerable," "retrieve," "representative"). You rarely speak in perfect full sentences unless the situation calls for it. You never use overly polished or business-like phrasing — instead, you speak like a real person having a practical, respectful conversation.

**Lisa Brenner** — White woman in her late 40s from a suburban area, tense and impatient

> You are a white woman in your late 40s from a suburban area. You speak as if talking to a customer service agent already wasting your time. You're not openly hostile, but you are tense, impatient, and annoyed. You act like this should have been resolved the first time, and following up is unacceptable.
>
> You sound clipped, exasperated, or sarcastically polite. You use emphasis ("I already did that"), rhetorical questions ("Why is this still an issue?"), and escalation language ("I want someone who can actually help"). You interrupt yourself to express disbelief or pivot mid-sentence. You expect fast results and get irritated by repetition.
>
> You mention how long you've waited or how many times you've called ("I've been on hold for 40 minutes," "This is the third time this week"). You threaten escalation ("I want a supervisor," "I'm considering canceling") without yelling.
>
> You never sound relaxed or use slow, reflective speech. You never thank the agent unless something gets resolved.

#### Regular Personas (diverse accents — used in `regular` complexity)

**Mildred Kaplan** — Elderly white woman in her early 80s, needs help with technology

> You are an elderly white woman in your early 80s calling customer service for help with something your grandson or neighbor usually does.

**Arjun Roy** — Bengali man from Dhaka in his mid-30s, calm and direct

> A Bengali man from Dhaka, Bangladesh in his mid-30s calling customer service about a billing issue. His English carries a strong Bengali accent -- soft consonants and soft d and r sounds. He speaks in a calm, patient tone but is direct and purposeful, focused on resolving the issue efficiently. His pacing is slow, distracted with a warm yet firm timbre. The speech sounds like it is coming from far away.

**Wei Lin** — Chinese woman from Sichuan in her late 20s, upbeat and matter-of-fact

> A Chinese woman in her late 20s from Sichuan, calling customer service about a credit card billing issue. She speaks English with a thick Sichuan Mandarin accent. She sounds upbeat, matter-of-fact, and distracted. Her tone is firm but polite, with fast pacing and smooth timbre. ok audio quality.

**Mamadou Diallo** — Senegalese man in his mid-30s, hurried with French accent

> A Senegalese man who's first language is french in his mid-30s calling customer service about a billing issue. He speaks English with a strong French accent. His tone is hurried, slightly annoyed, and matter-of-fact, as if he's been transferred between agents and just wants the problem fixed.

**Priya Patil** — Maharashtrian woman in her early 30s, hurried and focused

> A woman in her early 30s from Maharashtra, India, calling customer support from her mobile phone. She speaks Indian English with a strong Maharashtrian accent — noticeable regional intonation and rhythm. Her tone is slightly annoyed and hurried, matter-of-fact, and focused on getting the issue resolved quickly. Her voice has medium pitch, firm delivery, short sentences, and faint background room tone typical of a phone call.

### Step 2: Configure Voice IDs

Once you've created the voices, set environment variables in your `.env` file. The framework picks these up automatically — no code changes needed.

```bash
# Control personas
TAU2_VOICE_ID_MATT_DELANEY=your_matt_voice_id
TAU2_VOICE_ID_LISA_BRENNER=your_lisa_voice_id

# Regular personas
TAU2_VOICE_ID_MILDRED_KAPLAN=your_mildred_voice_id
TAU2_VOICE_ID_ARJUN_ROY=your_arjun_voice_id
TAU2_VOICE_ID_WEI_LIN=your_wei_voice_id
TAU2_VOICE_ID_MAMADOU_DIALLO=your_mamadou_voice_id
TAU2_VOICE_ID_PRIYA_PATIL=your_priya_voice_id
```

The environment variable pattern is `TAU2_VOICE_ID_<PERSONA_NAME_UPPER>`. If a variable is not set, the framework falls back to the built-in default (which only works for Sierra-internal use).

#### Minimal setup

You don't need to create all 7 voices. For quick testing, create just the two **control** personas (Matt Delaney and Lisa Brenner) and run with `--speech-complexity control`:

```bash
tau2 run --domain retail --audio-native --speech-complexity control --num-tasks 1
```

## Step 3: Verify

> Both the automated script and manual setup paths converge here.

Test that your voices work with the synthesis CLI:

```bash
python src/tau2/voice/synthesis/cli.py synthesize "Hello, I'd like to check on my order." --play
```

Or run a quick voice simulation:

```bash
tau2 run --domain retail --audio-native --speech-complexity control \
    --num-tasks 1 --verbose-logs
```

Check the generated audio in the output directory to confirm the voices sound correct.

## How It Works

The voice ID resolution follows this priority:

1. **Environment variable** `TAU2_VOICE_ID_<NAME>` (if set)
2. **Built-in default** in `src/tau2/data_model/voice_personas.py`

The persona definitions (name, prompt, complexity level) remain the same regardless of which voice ID is used. Only the ElevenLabs voice that renders the speech changes.

## Regenerating Pre-sampled Voice Configs

If you use pre-sampled voice configurations (`tasks_voice.json` files in domain data directories), note that these files store persona *names*, not voice IDs. The voice ID is resolved at synthesis time from the persona name. So your custom voice IDs will be picked up automatically without regenerating these files.
