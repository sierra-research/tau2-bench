# Copyright Sierra
"""Registry for Language Packs.

Mirrors the pattern of ``tau2.registry`` (agents/domains/users): packs are
registered into a module-level dict, and shared code looks them up by
language code or persona id. Registration also makes each pack persona
available as a ``VoicePersona`` so the existing TTS voice-id plumbing
(``get_elevenlabs_voice_id``, speech generators) works unchanged.

All getters lazily trigger pack discovery (``tau2.multilingual.loader``), so
callers never need to import language packs explicitly. With no packs
installed, every lookup is a cheap miss and English behavior is untouched.
"""

from typing import Optional

from loguru import logger

from tau2.data_model.voice_personas import (
    VoicePersona,
    _resolve_voice_id,
    register_voice_persona,
)
from tau2.multilingual.schema import LanguagePack, MultilingualPersonaConfig

_LANGUAGE_PACKS: dict[str, LanguagePack] = {}


def register_language_pack(pack: LanguagePack) -> None:
    """Register a language pack and its personas.

    Each persona is also registered as a ``VoicePersona`` (name=persona_id) so
    voice-id resolution and audio generation work through the existing
    registries. Persona voice ids honor the ``TAU2_VOICE_ID_<PERSONA_ID>``
    env-var override, like all other voice personas.

    Raises:
        ValueError: If the language or any persona_id is already registered.
    """
    if pack.language in _LANGUAGE_PACKS:
        raise ValueError(f"Language pack '{pack.language}' already registered")

    for persona in pack.personas.values():
        voice_persona = VoicePersona(
            elevenlabs_voice_id=_resolve_voice_id(
                persona.persona_id, persona.voice_id or ""
            ),
            name=persona.persona_id,
            display_name=persona.display_name,
            short_description=persona.short_description,
            prompt=persona.tts_voice_prompt,
            complexity="regular",
            language=pack.language,
        )
        register_voice_persona(voice_persona)

    _LANGUAGE_PACKS[pack.language] = pack
    logger.info(
        f"Registered language pack '{pack.language}' "
        f"({len(pack.personas)} personas: {sorted(pack.personas)})"
    )


def get_language_pack(language: str) -> Optional[LanguagePack]:
    """Get the pack for a language code, or None (English has no pack)."""
    _ensure_loaded()
    return _LANGUAGE_PACKS.get(language)


def list_language_packs() -> list[str]:
    """List registered language codes."""
    _ensure_loaded()
    return sorted(_LANGUAGE_PACKS)


def get_multilingual_persona(
    persona_id: str,
) -> Optional[tuple[LanguagePack, MultilingualPersonaConfig]]:
    """Look up a persona across all packs by its persona_id.

    Returns the (pack, persona) pair, or None if the id does not belong to
    any language pack (e.g. it is a plain English voice persona).
    """
    _ensure_loaded()
    for pack in _LANGUAGE_PACKS.values():
        persona = pack.get_persona(persona_id)
        if persona is not None:
            return pack, persona
    return None


def _ensure_loaded() -> None:
    from tau2.multilingual.loader import load_language_packs

    load_language_packs()
