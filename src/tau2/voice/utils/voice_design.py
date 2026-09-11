# Copyright Sierra
"""Generic ElevenLabs Voice Design wrappers.

Provider-agnostic of any pack/persona semantics: design a voice preview from a
description, save a preview to the account, or do both in one shot. The
audition ``text`` passed to the API must be at least 100 characters.
"""

from __future__ import annotations

from typing import Optional

# Voice Design parameters.
VOICE_DESIGN_MODEL = "eleven_ttv_v3"
VOICE_DESIGN_LOUDNESS = 0.5
VOICE_DESIGN_GUIDANCE_SCALE = 38
VOICE_DESIGN_SEED = 42

# Voice Design rejects audition `text` shorter than this.
MIN_AUDITION_CHARS = 100


def design_voice_preview(
    client,
    *,
    voice_description: str,
    audition_text: str,
    model_id: str = VOICE_DESIGN_MODEL,
    loudness: float = VOICE_DESIGN_LOUDNESS,
    guidance_scale: float = VOICE_DESIGN_GUIDANCE_SCALE,
    seed: int = VOICE_DESIGN_SEED,
):
    """Call Voice Design and return the first preview (or ``None`` if empty)."""
    if len(audition_text) < MIN_AUDITION_CHARS:
        raise ValueError(
            f"audition_text must be >= {MIN_AUDITION_CHARS} chars "
            f"(got {len(audition_text)}); Voice Design rejects shorter text"
        )
    result = client.text_to_voice.design(
        voice_description=voice_description,
        text=audition_text,
        model_id=model_id,
        loudness=loudness,
        guidance_scale=guidance_scale,
        seed=seed,
        auto_generate_text=False,
    )
    previews = getattr(result, "previews", None)
    return previews[0] if previews else None


def create_voice_from_preview(
    client, *, voice_name: str, short_description: str, generated_voice_id: str
) -> str:
    """Save a designed preview to the account and return its ``voice_id``."""
    voice = client.text_to_voice.create(
        voice_name=voice_name,
        voice_description=short_description,
        generated_voice_id=generated_voice_id,
    )
    return voice.voice_id


def design_and_create_voice(
    client,
    *,
    voice_description: str,
    voice_name: str,
    short_description: str,
    audition_text: str,
    model_id: str = VOICE_DESIGN_MODEL,
    loudness: float = VOICE_DESIGN_LOUDNESS,
    guidance_scale: float = VOICE_DESIGN_GUIDANCE_SCALE,
    seed: int = VOICE_DESIGN_SEED,
) -> Optional[str]:
    """Design a voice from a prompt and save it to the account.

    Returns the saved ``voice_id``, or ``None`` if the API returned no previews.
    Raises on API/transport errors so callers can surface them.
    """
    preview = design_voice_preview(
        client,
        voice_description=voice_description,
        audition_text=audition_text,
        model_id=model_id,
        loudness=loudness,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    if preview is None:
        return None
    return create_voice_from_preview(
        client,
        voice_name=voice_name,
        short_description=short_description,
        generated_voice_id=preview.generated_voice_id,
    )
