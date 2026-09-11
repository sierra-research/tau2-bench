# Copyright Sierra
"""The closed location/variety catalogs behind language-pinned prompts.

The catalogs raise at prompt-RENDER time (not pack load), so the load-time
safety lives here instead: a coverage guard asserting every shipped pack's
language and every shipped persona's locale has an entry. A pack extended
with a new locale/language must extend ``tau2.multilingual.varieties`` in the
same change, or these tests fail before any run does.
"""

import pytest

from tau2.multilingual.registry import get_language_pack, list_language_packs
from tau2.multilingual.varieties import (
    LANGUAGE_VARIETY_NAMES,
    LOCALE_DISPLAY_NAMES,
    language_variety_name,
    locale_display_name,
)

LANGUAGES = list_language_packs()


class TestShippedCatalogCoverage:
    def test_there_are_registered_languages(self):
        assert LANGUAGES, "no packs registered — discovery is broken"

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_every_pack_language_has_a_variety_name(self, language):
        assert language_variety_name(language)

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_every_persona_locale_has_a_display_name(self, language):
        pack = get_language_pack(language)
        for persona_id, persona in sorted(pack.personas.items()):
            assert persona.locale, f"{persona_id} carries no locale"
            assert locale_display_name(persona.locale), persona_id


class TestFailLoudLookups:
    def test_unmapped_locale_raises(self):
        with pytest.raises(ValueError, match="no display name"):
            locale_display_name("ZZ-99")

    def test_unmapped_language_raises(self):
        with pytest.raises(ValueError, match="no pinned variety name"):
            language_variety_name("zz")


class TestCatalogShape:
    def test_display_names_are_phrases_not_codes(self):
        for code, phrase in LOCALE_DISPLAY_NAMES.items():
            assert phrase != code
            assert ", " in phrase, f"{code}: expected 'Place, Country' shape"

    def test_variety_names_are_nonempty(self):
        for language, variety in LANGUAGE_VARIETY_NAMES.items():
            assert variety.strip(), language
