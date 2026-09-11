# Copyright Sierra
"""Fixed location/variety catalogs for language-pinned prompt material.

Owner ruling (2026-08-20): wherever a prompt names the caller's language, it
names their location and speech variety too. A raw persona locale code
('ES-MD') is useless to a model; the pinned variety ('Peninsular Spanish')
was previously stated nowhere outside the persona clauses. Two closed,
in-code catalogs carry that information:

- :data:`LOCALE_DISPLAY_NAMES` — persona locale code (the ISO 3166-2
  subdivision codes used by pack.yaml ``locale:`` fields) → human-readable
  place phrase ('ES-MD' → 'Madrid, Spain'). Entries name the pack persona's
  home CITY where the persona pins one (locale codes are subdivision-level).
- :data:`LANGUAGE_VARIETY_NAMES` — ISO 639-1 language code → the language's
  pinned speech-variety name ('es' → 'Peninsular Spanish (Spain)'). Entries
  must agree with the packs' persona clauses (clause 1 states each persona's
  variety).

Both catalogs are the single source of truth shared by the agent-side
language clause (:func:`tau2.multilingual.registry.get_agent_language_clause`)
and the user-simulator target-language directive
(``tau2.multilingual.english_prompts``, template v3+).

Fail-loudly contract: lookups RAISE on unmapped codes at prompt-RENDER time,
not at pack load. Rendering a prompt with a bare locale code or a missing
variety would silently ship an under-specified arm; but a draft pack carrying
a brand-new locale must not brick pack discovery (the loader registers every
pack under data/tau2/multilingual/) for every other language's runs. Extending
a pack to a new locale or language therefore requires extending these catalogs
before any prompt for it can render.
"""

# Persona locale code → human-readable place phrase. Covers every ``locale:``
# value in the shipped packs (data/tau2/multilingual/*/pack.yaml).
LOCALE_DISPLAY_NAMES: dict[str, str] = {
    "BR-RJ": "Rio de Janeiro, Brazil",
    "BR-SP": "São Paulo, Brazil",
    "CN-BJ": "Beijing, China",
    "CN-SH": "Shanghai, China",
    "ES-MD": "Madrid, Spain",
    "IN-DL": "Delhi, India",
    "IN-MH": "Mumbai, India",
    "KR-11": "Seoul, South Korea",
    "KR-26": "Busan, South Korea",
    "US-NJ": "New Jersey, US",
    "US-OH": "Ohio, US",
}

# ISO 639-1 language code → pinned speech-variety name, verified against each
# pack's persona clause 1 / short_description (never guessed against the pack).
LANGUAGE_VARIETY_NAMES: dict[str, str] = {
    "en": "American English",
    "es": "Peninsular Spanish (Spain)",
    "hi": "standard urban Hindustani",
    "ko": "standard Seoul Korean",
    "pt": "Brazilian Portuguese",
    "zh": "Mainland Standard Mandarin (Putonghua)",
}


def locale_display_name(locale: str) -> str:
    """The human-readable place phrase for a persona locale code.

    Raises:
        ValueError: If the locale code is not in the catalog. This fires at
            prompt-render time — a new pack locale must be added to
            :data:`LOCALE_DISPLAY_NAMES` before its prompts can render.
    """
    try:
        return LOCALE_DISPLAY_NAMES[locale]
    except KeyError:
        raise ValueError(
            f"Persona locale '{locale}' has no display name. Add it to "
            "tau2.multilingual.varieties.LOCALE_DISPLAY_NAMES — prompts never "
            "render raw locale codes."
        ) from None


def language_variety_name(language: str) -> str:
    """The pinned speech-variety name for a language code.

    Raises:
        ValueError: If the language is not in the catalog. This fires at
            prompt-render time — a new language pack must pin its variety in
            :data:`LANGUAGE_VARIETY_NAMES` before its prompts can render.
    """
    try:
        return LANGUAGE_VARIETY_NAMES[language]
    except KeyError:
        raise ValueError(
            f"Language '{language}' has no pinned variety name. Add it to "
            "tau2.multilingual.varieties.LANGUAGE_VARIETY_NAMES — language-"
            "pinned prompts always name the expected variety."
        ) from None
