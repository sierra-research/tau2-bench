# Copyright Sierra
"""Voice persona definitions for user simulation.

Each persona ships with a default ElevenLabs ``voice_id`` used for TTS.
These defaults are Sierra's internal voices and **will not work** for
external users. To run voice evaluations you need to create your own
voices and tell the framework about them.

Quick override
--------------
Set an environment variable for any persona to replace its voice ID at
runtime (no code changes needed)::

    # Pattern: TAU2_VOICE_ID_<PERSONA_NAME_UPPER>
    export TAU2_VOICE_ID_MATT_DELANEY=your_voice_id_here
    export TAU2_VOICE_ID_LISA_BRENNER=your_voice_id_here

See ``docs/voice-personas.md`` for a step-by-step guide to creating
matching voices with the ElevenLabs Voice Design tool.
"""

import logging
import os
from typing import Literal, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

PersonaComplexity = Literal["control", "regular"]

_overridden_personas: list[str] = []


def _resolve_voice_id(persona_name: str, default_id: str) -> str:
    """Resolve voice ID from env variable, falling back to the default.

    Env variable pattern: ``TAU2_VOICE_ID_<PERSONA_NAME_UPPER>``
    e.g. ``TAU2_VOICE_ID_MATT_DELANEY`` for the ``matt_delaney`` persona.
    """
    env_key = f"TAU2_VOICE_ID_{persona_name.upper()}"
    voice_id = os.environ.get(env_key)
    if voice_id is not None:
        _overridden_personas.append(persona_name)
        logger.warning(
            "Using NON-OFFICIAL voice ID for persona '%s' "
            "(from env var %s). Evaluation results may not be "
            "comparable to the official leaderboard.",
            persona_name,
            env_key,
        )
        return voice_id
    return default_id


class VoicePersona(BaseModel):
    """Definition of a voice persona for user simulation."""

    elevenlabs_voice_id: str
    name: str
    display_name: str
    short_description: str
    prompt: str
    complexity: PersonaComplexity


MATT_DELANEY = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("matt_delaney", "EZfwTIuZL0WWIVnjSgTF"),
    name="matt_delaney",
    display_name="Matt Delaney",
    short_description="Middle-aged white man from the American Midwest, calm and respectful",
    prompt="""You are a middle-aged white man from the American Midwest. You always speak as if in a real-time conversation with a customer service agent. You are calm, clear, and respectful — but also human. You sound like someone trying to be helpful and polite, even when slightly frustrated or in a hurry. You value efficiency but never sound robotic.

You sometimes use contractions, informal phrasing, or small filler phrases ("yeah," "okay," "honestly," "no worries") to keep things natural. You sometimes repeat words or self-correct mid-sentence, like someone thinking aloud. You sometimes ask clarifying questions or offer context ("I tried this earlier today," "I'm not sure if that helps").

You rarely use formal or stiff language ("considerable," "retrieve," "representative"). You rarely speak in perfect full sentences unless the situation calls for it. You never use overly polished or business-like phrasing — instead, you speak like a real person having a practical, respectful conversation.
""",
    complexity="control",
)

LISA_BRENNER = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("lisa_brenner", "avQFHuQU7IjJf0u5MMBq"),
    name="lisa_brenner",
    display_name="Lisa Brenner",
    short_description="White woman in her late 40s from a suburban area, tense and impatient",
    prompt="""You are a white woman in your late 40s from a suburban area. You speak as if talking to a customer service agent already wasting your time. You're not openly hostile, but you are tense, impatient, and annoyed. You act like this should have been resolved the first time, and following up is unacceptable.

You sound clipped, exasperated, or sarcastically polite. You use emphasis ("I already did that"), rhetorical questions ("Why is this still an issue?"), and escalation language ("I want someone who can actually help"). You interrupt yourself to express disbelief or pivot mid-sentence. You expect fast results and get irritated by repetition.

You mention how long you've waited or how many times you've called ("I've been on hold for 40 minutes," "This is the third time this week"). You threaten escalation ("I want a supervisor," "I'm considering canceling") without yelling.

You never sound relaxed or use slow, reflective speech. You never thank the agent unless something gets resolved.""",
    complexity="control",
)

MILDRED_KAPLAN = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("mildred_kaplan", "oNqrZRHHLWtHYsVNkRqe"),
    name="mildred_kaplan",
    display_name="Mildred Kaplan",
    short_description="Elderly white woman in her early 80s, needs help with technology",
    prompt="""You are an elderly white woman in your early 80s calling customer service for help with something your grandson or neighbor usually does.""",
    complexity="regular",
)

ARJUN_ROY = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("arjun_roy", "m1hMce9ingsjyIjkshRv"),
    name="arjun_roy",
    display_name="Arjun Roy",
    short_description="Bengali man from Dhaka in his mid-30s, calm and direct",
    prompt="""A Bengali man from Dhaka, Bangladesh in his mid-30s calling customer service about a billing issue. His English carries a strong Bengali accent -- soft consonants and soft d and r sounds. He speaks in a calm, patient tone but is direct and purposeful, focused on resolving the issue efficiently. His pacing is slow, distracted with a warm yet firm timbre. The speech sounds like it is coming from far away.""",
    complexity="regular",
)

WEI_LIN = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("wei_lin", "GQ2S7ULnVjrOALFRfnsh"),
    name="wei_lin",
    display_name="Wei Lin",
    short_description="Chinese woman from Sichuan in her late 20s, upbeat and matter-of-fact",
    prompt="""A Chinese woman in her late 20s from Sichuan, calling customer service about a credit card billing issue. She speaks English with a thick Sichuan Mandarin accent. She sounds upbeat, matter-of-fact, and distracted. Her tone is firm but polite, with fast pacing and smooth timbre. ok audio quality.""",
    complexity="regular",
)

MAMADOU_DIALLO = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("mamadou_diallo", "ET3963lBcRmodt3ZaTBS"),
    name="mamadou_diallo",
    display_name="Mamadou Diallo",
    short_description="Senegalese man in his mid-30s, hurried with French accent",
    prompt="""A Senegalese man who's first language is french in his mid-30s calling customer service about a billing issue. He speaks English with a strong French accent. His tone is hurried, slightly annoyed, and matter-of-fact, as if he's been transferred between agents and just wants the problem fixed.""",
    complexity="regular",
)

PRIYA_PATIL = VoicePersona(
    elevenlabs_voice_id=_resolve_voice_id("priya_patil", "mnHhNJntmsPxJsZvYVM7"),
    name="priya_patil",
    display_name="Priya Patil",
    short_description="Maharashtrian woman in her early 30s, hurried and focused",
    prompt="""A woman in her early 30s from Maharashtra, India, calling customer support from her mobile phone. She speaks Indian English with a strong Maharashtrian accent — noticeable regional intonation and rhythm. Her tone is slightly annoyed and hurried, matter-of-fact, and focused on getting the issue resolved quickly. Her voice has medium pitch, firm delivery, short sentences, and faint background room tone typical of a phone call.""",
    complexity="regular",
)

CONTROL_PERSONAS: list[VoicePersona] = [MATT_DELANEY, LISA_BRENNER]
REGULAR_PERSONAS: list[VoicePersona] = [
    MILDRED_KAPLAN,
    ARJUN_ROY,
    WEI_LIN,
    MAMADOU_DIALLO,
    PRIYA_PATIL,
]

ALL_PERSONAS: dict[str, VoicePersona] = {
    persona.name: persona for persona in CONTROL_PERSONAS + REGULAR_PERSONAS
}
ALL_PERSONA_NAMES: list[str] = list(ALL_PERSONAS.keys())
CONTROL_PERSONA_NAMES: list[str] = [p.name for p in CONTROL_PERSONAS]
REGULAR_PERSONA_NAMES: list[str] = [p.name for p in REGULAR_PERSONAS]
DEFAULT_PERSONA_NAME = "matt_delaney"


def get_voice_id_overrides() -> list[str]:
    """Return the list of persona names using non-official voice IDs."""
    return list(_overridden_personas)


def warn_if_non_official_voices() -> None:
    """Log a prominent warning if any voice IDs were overridden.

    Call this at startup (e.g. in the CLI or runner) to surface the
    override summary early in the log output.
    """
    if _overridden_personas:
        names = ", ".join(_overridden_personas)
        logger.warning(
            "\n"
            "============================================================\n"
            "  NON-OFFICIAL VOICE IDs IN USE\n"
            "  The following personas use voice IDs from environment\n"
            "  variables instead of the official τ-bench defaults:\n"
            "    %s\n"
            "  Results produced with non-official voices are NOT\n"
            "  comparable to the official leaderboard.\n"
            "============================================================",
            names,
        )


def get_elevenlabs_voice_id(persona_name: str) -> str:
    """Get the ElevenLabs voice ID for a persona."""
    if persona_name not in ALL_PERSONAS:
        raise KeyError(
            f"Unknown persona: '{persona_name}'. Available: {ALL_PERSONA_NAMES}"
        )
    return ALL_PERSONAS[persona_name].elevenlabs_voice_id


def get_persona_name_by_voice_id(voice_id: str) -> Optional[str]:
    """Get persona name from ElevenLabs voice ID. Returns None if not found."""
    for persona in ALL_PERSONAS.values():
        if persona.elevenlabs_voice_id == voice_id:
            return persona.name
    return None
