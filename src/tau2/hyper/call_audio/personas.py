# Copyright Sierra
"""Voice casting registry for rendering hyper-tau phone-call transcripts.

Unlike ``tau2.data_model.voice_personas`` (behavioral prompts that steer a
live user simulator), these personas cast *fixed, authored* transcripts: the
words are already written, so each entry only carries the ElevenLabs voice,
TTS delivery settings, and casting attributes (gender presentation, age band,
accent, energy) used to match a voice to the customer a case describes.

The ElevenLabs voice IDs are borrowed from tau-voice, where they are proven
to work with ``eleven_v3``. Add new voices by appending to the pools below.
"""

from typing import Literal

from pydantic import BaseModel, Field

CallRole = Literal["agent", "customer"]


class CallVoice(BaseModel):
    """A voice available for casting one side of a rendered call."""

    name: str
    elevenlabs_voice_id: str
    display_name: str
    # Casting notes a reviewer uses to match voice to transcript content.
    casting_notes: str
    suitable_roles: list[CallRole]
    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    style: float = Field(default=0.0, ge=0.0, le=1.0)


# Voice IDs from tau2.data_model.voice_personas (user-simulator personas).
MATT_DELANEY = CallVoice(
    name="matt_delaney",
    elevenlabs_voice_id="EZfwTIuZL0WWIVnjSgTF",
    display_name="Matt Delaney",
    casting_notes="Male, middle-aged, US Midwest, calm and measured",
    suitable_roles=["agent", "customer"],
)

LISA_BRENNER = CallVoice(
    name="lisa_brenner",
    elevenlabs_voice_id="avQFHuQU7IjJf0u5MMBq",
    display_name="Lisa Brenner",
    casting_notes="Female, late 40s, US suburban, tense and clipped",
    suitable_roles=["customer"],
)

MILDRED_KAPLAN = CallVoice(
    name="mildred_kaplan",
    elevenlabs_voice_id="oNqrZRHHLWtHYsVNkRqe",
    display_name="Mildred Kaplan",
    casting_notes="Female, early 80s, US, gentle and unhurried",
    suitable_roles=["customer"],
)

ARJUN_ROY = CallVoice(
    name="arjun_roy",
    elevenlabs_voice_id="m1hMce9ingsjyIjkshRv",
    display_name="Arjun Roy",
    casting_notes="Male, mid-30s, Bengali accent, calm and direct",
    suitable_roles=["customer"],
)

WEI_LIN = CallVoice(
    name="wei_lin",
    elevenlabs_voice_id="GQ2S7ULnVjrOALFRfnsh",
    display_name="Wei Lin",
    casting_notes="Female, late 20s, Sichuan Mandarin accent, upbeat and quick",
    suitable_roles=["customer"],
)

MAMADOU_DIALLO = CallVoice(
    name="mamadou_diallo",
    elevenlabs_voice_id="ET3963lBcRmodt3ZaTBS",
    display_name="Mamadou Diallo",
    casting_notes="Male, mid-30s, Senegalese French accent, hurried",
    suitable_roles=["customer"],
)

PRIYA_PATIL = CallVoice(
    name="priya_patil",
    elevenlabs_voice_id="mnHhNJntmsPxJsZvYVM7",
    display_name="Priya Patil",
    casting_notes="Female, early 30s, Maharashtrian accent, firm and focused",
    suitable_roles=["agent", "customer"],
)

# Voice IDs from src/tau2/voice/README.md (documented available voices).
AVA_BAILEY = CallVoice(
    name="ava_bailey",
    elevenlabs_voice_id="ycvyTVVIzO2xfIGZC7tZ",
    display_name="Ava Bailey",
    casting_notes="Female, US, even and professional",
    suitable_roles=["agent"],
)

CHRIS = CallVoice(
    name="chris",
    elevenlabs_voice_id="iP95p4xoKVk53GoZ742B",
    display_name="Chris",
    casting_notes="Male, US, even and professional",
    suitable_roles=["agent"],
)

ALL_CALL_VOICES: dict[str, CallVoice] = {
    voice.name: voice
    for voice in [
        MATT_DELANEY,
        LISA_BRENNER,
        MILDRED_KAPLAN,
        ARJUN_ROY,
        WEI_LIN,
        MAMADOU_DIALLO,
        PRIYA_PATIL,
        AVA_BAILEY,
        CHRIS,
    ]
}

AGENT_VOICES: list[CallVoice] = [
    voice for voice in ALL_CALL_VOICES.values() if "agent" in voice.suitable_roles
]
CUSTOMER_VOICES: list[CallVoice] = [
    voice for voice in ALL_CALL_VOICES.values() if "customer" in voice.suitable_roles
]


def get_call_voice(name: str) -> CallVoice:
    """Look up a call voice by name."""
    if name not in ALL_CALL_VOICES:
        raise KeyError(
            f"Unknown call voice: {name!r}. Available: {sorted(ALL_CALL_VOICES)}"
        )
    return ALL_CALL_VOICES[name]
