# Copyright Sierra
"""Provider language capabilities, independent of installed packs and benchmark cells."""

from tau2.data_model.language import cross_lingual

# https://developers.deepgram.com/docs/models-languages-overview
# https://elevenlabs.io/docs/overview/models
# Reviewed Nova-3 / ElevenLabs v2 intersection. Regional aliases are explicit:
# Cantonese must not be treated as Mandarin by stripping its region suffix.
CASCADED_LANGUAGE_CODES: dict[str, frozenset[str]] = {
    code: frozenset({code})
    for code in (
        "bg",
        "de",
        "el",
        "fi",
        "hi",
        "hr",
        "id",
        "it",
        "ja",
        "ms",
        "nl",
        "pl",
        "ro",
        "ru",
        "sk",
        "ta",
        "tl",
        "uk",
        "hu",
        "no",
        "vi",
    )
}
CASCADED_LANGUAGE_CODES.update(
    {
        "ar": frozenset({"ar", "ar-SA", "ar-AE"}),
        "cs": frozenset({"cs", "cs-CZ"}),
        "da": frozenset({"da", "da-DK"}),
        "en": frozenset({"en", "en-US", "en-AU", "en-GB", "en-IN", "en-NZ"}),
        "es": frozenset({"es", "es-419"}),
        "fr": frozenset({"fr", "fr-CA"}),
        "ko": frozenset({"ko", "ko-KR"}),
        "pt": frozenset({"pt", "pt-BR", "pt-PT"}),
        "sv": frozenset({"sv", "sv-SE"}),
        "tr": frozenset({"tr", "tr-TR"}),
        "zh": frozenset({"zh", "zh-CN", "zh-Hans", "zh-TW", "zh-Hant"}),
    }
)
CASCADED_DEFAULT_CODES = {"pt": "pt-BR", "zh": "zh-CN", "fr-FR": "fr"}
ELEVENLABS_V2_5_ONLY_LANGUAGES = frozenset({"hu", "no", "vi"})


def cascaded_language(language: str) -> tuple[str, str]:
    """Return a verified base language and Nova-3 code, or reject unsupported input."""
    code = CASCADED_DEFAULT_CODES.get(language, language)
    for base, codes in CASCADED_LANGUAGE_CODES.items():
        if code in codes:
            return base, code
    raise ValueError(
        f"Unsupported cascaded language {language!r}; no reviewed STT/TTS mapping"
    )


# https://docs.x.ai/developers/model-capabilities/audio/speech-to-speech
XAI_LANGUAGE_CODES = frozenset(
    {
        "en",
        "ar-EG",
        "ar-SA",
        "ar-AE",
        "bn",
        "zh",
        "fr",
        "de",
        "hi",
        "id",
        "it",
        "ja",
        "ko",
        "pt-BR",
        "pt-PT",
        "ru",
        "es-MX",
        "es-ES",
        "tr",
        "vi",
    }
)
XAI_LANGUAGE_ALIASES = {
    "es": "es-ES",
    "pt": "pt-BR",
    "fr-FR": "fr",
    "zh-CN": "zh",
    "hi-IN": "hi",
    "ko-KR": "ko",
    "en-US": "en",
}


def xai_language_hint(language: str | None) -> str | None:
    """Resolve a documented hint; unsupported codes must not silently auto-detect."""
    if language is None or not cross_lingual(language):
        return None
    code = XAI_LANGUAGE_ALIASES.get(language, language)
    if code not in XAI_LANGUAGE_CODES:
        raise ValueError(f"Unsupported xAI transcription language {language!r}")
    return code


# https://cloud.google.com/vertex-ai/generative-ai/docs/live-api/configure-language-voice
# Includes the existing half-cascade mappings preserved from tau-multi.
GEMINI_LANGUAGE_CODES = {
    "en": "en-US",
    "es": "es-ES",
    "hi": "hi-IN",
    "ko": "ko-KR",
    "pt": "pt-BR",
    "zh": "cmn-CN",
    "fr": "fr-FR",
    "de": "de-DE",
    "ar": "ar-EG",
    "bn": "bn-BD",
    "nl": "nl-NL",
    "id": "id-ID",
    "it": "it-IT",
    "ja": "ja-JP",
    "mr": "mr-IN",
    "pl": "pl-PL",
    "ro": "ro-RO",
    "ru": "ru-RU",
    "ta": "ta-IN",
    "te": "te-IN",
    "th": "th-TH",
    "tr": "tr-TR",
    "uk": "uk-UA",
    "vi": "vi-VN",
}


def gemini_language_code(language: str) -> str:
    """Resolve a reviewed Live language without consulting benchmark membership."""
    if language in GEMINI_LANGUAGE_CODES:
        return GEMINI_LANGUAGE_CODES[language]
    if language in GEMINI_LANGUAGE_CODES.values():
        return language
    if language in {"zh-CN", "zh-Hans"}:
        return "cmn-CN"
    raise ValueError(f"Unsupported Gemini language hint {language!r}")
