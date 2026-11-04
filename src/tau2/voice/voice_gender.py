# Copyright Sierra
"""Perceived gender of provider TTS voices.

Resolves a ``(provider, voice)`` pair to ``"male"``/``"female"``, falling back to
the provider's pinned default voice. Only voices with a clear, widely-agreed
perceived gender are listed; anything absent (or an officially neutral voice)
resolves to ``None``.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tau2.data_model.simulation import RunConfig

from tau2.config import (
    DEFAULT_GEMINI_VOICE,
    DEFAULT_NOVA_VOICE,
    DEFAULT_OPENAI_VOICE,
    DEFAULT_PHONIC_VOICE,
    DEFAULT_QWEN_VOICE,
    DEFAULT_XAI_VOICE,
)

# Per-provider default voice (what the adapters actually use today).
PROVIDER_DEFAULT_VOICE: dict[str, str] = {
    "openai": DEFAULT_OPENAI_VOICE,
    "gemini": DEFAULT_GEMINI_VOICE,
    "xai": DEFAULT_XAI_VOICE,
    "nova": DEFAULT_NOVA_VOICE,
    "qwen": DEFAULT_QWEN_VOICE,
    "phonic": DEFAULT_PHONIC_VOICE,
}

# (provider -> {voice_lower -> "male" | "female"}). Only confidently-gendered
# voices are listed; absent/neutral voices resolve to None. When adding a voice,
# prefer omitting it to guessing.
VOICE_GENDER: dict[str, dict[str, str]] = {
    "elevenlabs": {"9BWtsMINqrJLrRacOk9x".lower(): "female"},  # Aria
    "deepgram": {"aura-asteria-en": "female", "aura-luna-en": "female"},
    # OpenAI does not officially assign gender; these are the realtime voices with a
    # clear perceived gender. Neutral voices (alloy, ash, ballad, sage, verse) omitted.
    "openai": {
        "coral": "female",
        "shimmer": "female",
        "marin": "female",
        "echo": "male",
        "cedar": "male",
    },
    # Gemini genders are from Google's published Gemini-TTS voice list.
    "gemini": {
        "zephyr": "female",
        "puck": "male",
        "charon": "male",
        "kore": "female",
        "fenrir": "male",
        "aoede": "female",
        "leda": "female",
        "orus": "male",
    },
    # xAI named voices (Ara, Rex, Sal, Eve, Leo); "sal" is ambiguous -> omitted.
    "xai": {
        "ara": "female",
        "rex": "male",
        "eve": "female",
        "leo": "male",
    },
    # Amazon Nova Sonic voices (Polly-style named voices, well-known genders).
    "nova": {
        "matthew": "male",
        "tiffany": "female",
        "amy": "female",
    },
    # Phonic voices (genders as listed by Phonic's voices API).
    "phonic": {
        "grant": "male",
        "landon": "male",
        "sabrina": "female",
        "shelby": "female",
    },
    "qwen": {
        "cherry": "female",
        "ethan": "male",
        "tina": "female",  # Qwen realtime default voice (config.DEFAULT_QWEN_VOICE)
    },
}


def resolve_agent_gender(
    provider: Optional[str], voice: Optional[str] = None
) -> Optional[str]:
    """Best-effort agent gender ("male"/"female") from the provider voice.

    Falls back to the provider's default voice when ``voice`` is None. Returns
    None for unknown providers/voices and for officially neutral voices.
    """
    if not provider:
        return None
    provider = provider.strip().lower()
    voice = (voice or PROVIDER_DEFAULT_VOICE.get(provider) or "").strip().lower()
    if not voice:
        return None
    return VOICE_GENDER.get(provider, {}).get(voice)


VOICE_GENDER_CATALOG_REVISION = "voice-gender-v2"


def selected_agent_voice(config: "RunConfig") -> tuple[str | None, str | None]:
    """Resolve recorded actual voice through the existing benchmark catalog."""

    audio = getattr(config, "audio_native_config", None)
    if audio is None:
        return None, None
    if audio.provider == "openai_live":
        voice = audio.live_config.voice
        return voice, resolve_agent_gender("openai", voice)
    if audio.provider == "livekit":
        from tau2.data_model.cascaded import ElevenLabsTTSConfig

        tts = audio.cascaded_config.tts
        voice = tts.voice_id if isinstance(tts, ElevenLabsTTSConfig) else tts.model
        return voice, resolve_agent_gender(tts.provider, voice)
    voice = audio.voice or PROVIDER_DEFAULT_VOICE.get(audio.provider)
    return voice, resolve_agent_gender(audio.provider, voice) if voice else None
