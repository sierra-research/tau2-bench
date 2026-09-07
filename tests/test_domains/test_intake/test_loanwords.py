"""Tests for the loanword adaptation tables (design doc §8b): golden
anglicization vectors covering every pinned language, the per-language
stress rules, nasal-vowel handling, segment normalization (ich-laut ç,
precomposed nasals, affricate tie bars), and the closed-catalog guards that
keep the language enum, lexicon pins, and stress rules in lockstep."""

import pytest

from tau2.domains.intake.tasks.loanwords import (
    LANGUAGE_STRESS,
    LEXICON_SOURCES,
    LoanLanguage,
    adapt_foreign_ipa,
)
from tau2.domains.intake.tasks.pronunciation import (
    NATURAL_VOWEL_GRAPHEMES,
    PronunciationError,
    render_natural,
    render_respelling,
)

# One (or two) real pinned-lexicon rows per language: the fluent-American
# anglicized reading the adaptation tables must keep producing. Changing a
# table changes spoken audio — a diff here is a deliberate re-review, not a
# refactor detail.
GOLDEN_VECTORS: list[tuple[LoanLanguage, str, str]] = [
    (LoanLanguage.CZECH, "p ɛ n z ɪ j o n", "PEHN-zih-yohn"),  # penzion
    (LoanLanguage.WELSH, "m ə n i", "MUH-nee"),  # mynydd
    (LoanLanguage.GERMAN, "ɡ a s t h oː f", "GAHST-hohf"),  # Gasthof
    (LoanLanguage.GERMAN, "m ɪ l ç", "MIHLKH"),  # Milch: ich-laut -> kh (§8b)
    (LoanLanguage.FRENCH, "o b ɛ ʁ ʒ", "oh-BEHRZH"),  # auberge
    (LoanLanguage.FRENCH, "b ɔ̃ ʒ u ʁ", "bawng-ZHOOR"),  # bonjour: nasal -> ng
    (LoanLanguage.SCOTTISH_GAELIC, "m a x ɪ ɾʲ", "MAH-kihr"),  # machair
    (LoanLanguage.IRISH, "oː sˠ t̪ˠ ɑː nˠ", "OH-stahn"),  # óstán
    (LoanLanguage.HUNGARIAN, "h aː r o m", "HAH-rohm"),  # három
    (LoanLanguage.ITALIAN, "d͡ʒ u l j a", "JOOL-yah"),  # Giulia: tie-bar affricate
    (LoanLanguage.DUTCH, "n eː ɣ ə", "NAY-guh"),  # negen
    (LoanLanguage.NORWEGIAN, "f j ɛ l", "FYEHL"),  # fjell
    (LoanLanguage.POLISH, "b j a w ɘ m", "BYAH-wihm"),  # białym
    (LoanLanguage.PORTUGUESE, "k ĩ t ɐ", "KEENG-tuh"),  # quinta: nasal -> ng (§8b)
    (LoanLanguage.PORTUGUESE, "p o w z a d ɐ", "poh-ZAH-duh"),  # pousada
    (LoanLanguage.ROMANIAN, "s i ŋ ɡ u r a t i k", "seeng-goo-RAH-teek"),  # singuratic
    (LoanLanguage.SPANISH, "θ a ɡ w a n", "ZAH-gwahn"),  # zaguán: Castilian θ -> z
    (LoanLanguage.SWEDISH, "k v ɑː r t", "KVAHRT"),  # kvart
    (LoanLanguage.TURKISH, "k ɯ ɾ ɯ k", "kih-RIHK"),  # kırık: ɯ -> ih, final stress
]


@pytest.mark.parametrize(
    "language, phonemes, expected",
    GOLDEN_VECTORS,
    ids=[f"{lang.value}-{expected}" for lang, _, expected in GOLDEN_VECTORS],
)
def test_golden_anglicizations(language, phonemes, expected):
    assert render_respelling(adapt_foreign_ipa(phonemes, language)) == expected


def test_every_language_is_golden_covered():
    covered = {language for language, _, _ in GOLDEN_VECTORS}
    assert covered == set(LoanLanguage)


def test_adapted_nuclei_render_naturally():
    """Every golden adaptation renders through render_natural — i.e. the
    adaptation tables only emit nuclei the natural-orthography renderer
    knows (the closed NATURAL_VOWEL_GRAPHEMES catalog)."""
    for language, phonemes, _ in GOLDEN_VECTORS:
        syllables = adapt_foreign_ipa(phonemes, language)
        for syllable in syllables:
            assert syllable.nucleus in NATURAL_VOWEL_GRAPHEMES
        assert render_natural(syllables).isalpha()


def test_nasal_vowels_emit_ng():
    """§8b: Portuguese/French nasal vowels read as vowel + ng — both the
    combining-tilde form (ɔ̃) and the precomposed letter (ĩ)."""
    assert (
        render_respelling(adapt_foreign_ipa("b ɔ̃ ʒ u ʁ", LoanLanguage.FRENCH))
        == "bawng-ZHOOR"
    )
    assert (
        render_respelling(adapt_foreign_ipa("k ĩ t ɐ", LoanLanguage.PORTUGUESE))
        == "KEENG-tuh"
    )


def test_ich_laut_survives_normalization():
    """ç decomposes to c + cedilla; the segment normalizer must NOT strip
    the cedilla (that would silently read Milch as 'milk')."""
    syllables = adapt_foreign_ipa("m ɪ l ç", LoanLanguage.GERMAN)
    assert render_natural(syllables) == "milkh"


def test_unknown_segment_fails_loud():
    with pytest.raises(PronunciationError, match="Unknown swe IPA segment"):
        adapt_foreign_ipa("ɧ uː k", LoanLanguage.SWEDISH)  # sj-sound: unmapped


def test_stress_rule_and_lexicon_catalogs_are_closed():
    assert set(LANGUAGE_STRESS) == set(LoanLanguage)
    assert set(LEXICON_SOURCES) == set(LoanLanguage)
    for source in LEXICON_SOURCES.values():
        assert source.filename.endswith(".tsv")
        assert source.url.startswith(
            "https://raw.githubusercontent.com/CUNY-CL/wikipron/"
        )
        assert source.filename in source.url


def test_stress_positions():
    """One vector per stress rule family (initial, penultimate, final, and
    both Romance branches)."""
    # INITIAL (deu): first syllable caps.
    assert (
        render_respelling(adapt_foreign_ipa("ɡ a s t h oː f", LoanLanguage.GERMAN))
        .split("-")[0]
        .isupper()
    )
    # FINAL (fra): last syllable caps.
    assert (
        render_respelling(adapt_foreign_ipa("o b ɛ ʁ ʒ", LoanLanguage.FRENCH))
        .split("-")[-1]
        .isupper()
    )
    # ROMANCE ends-in-vowel -> penult (por pousada).
    assert (
        render_respelling(adapt_foreign_ipa("p o w z a d ɐ", LoanLanguage.PORTUGUESE))
        == "poh-ZAH-duh"
    )
    # ROMANCE ends-in-other-consonant -> final (spa azul).
    assert (
        render_respelling(adapt_foreign_ipa("a θ u l", LoanLanguage.SPANISH))
        == "ah-ZOOL"
    )
    # ROMANCE ends-in-n -> penult (spa cansado ends -o, use hostal ends -l is
    # final; the -n branch: spa "zaguan" reads penult under the rule, which
    # is exactly why the accent-marked native form is authored instead).
    assert (
        render_respelling(adapt_foreign_ipa("θ a ɡ w a n", LoanLanguage.SPANISH))
        == "ZAH-gwahn"
    )
