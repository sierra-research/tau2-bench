# Copyright Sierra
"""Loanword adaptation: foreign-language IPA -> the repo respelling alphabet.

Phase 3 of the intake pronunciation program (design doc
docs/designs/intake-mispronunciation.md §8b): foreign tokens in the
properties and vehicles banks source their phonemes from per-language
WikiPron lexicons (:data:`LEXICON_SOURCES`, pinned at the same commit as the
English lexicon in :mod:`tau2.domains.intake.tasks.name_banks`). The target
pronunciation is the FLUENT-AMERICAN anglicized reading, not the native one:
:func:`adapt_foreign_ipa` maps source-language IPA into the same respelling
units the English converters emit, via fixed per-language adaptation tables
layered over a shared foreign base (German /ç/ -> "kh", Castilian /θ/ from
the letter z -> "z", nasalized vowels -> vowel + "ng", ...), then applies the
language's fixed stress convention (:class:`StressRule`) — WikiPron bulk rows
carry no stress marks.

Everything here is pure and total over the tables below; an unknown segment
raises :class:`PronunciationError` — extend the table deliberately, never
coerce silently. Golden-vector unit tests per language live in
``tests/test_domains/test_intake/test_loanwords.py``.
"""

import unicodedata
from enum import Enum
from typing import Annotated, Optional

from pydantic import Field

from tau2.domains.intake.tasks.pronunciation import (
    IPA_CONSONANTS,
    IPA_DIPHTHONGS,
    IPA_VOWELS,
    PronunciationError,
    Syllable,
    _Phone,
    syllabify,
)
from tau2.utils.pydantic_utils import BaseModelNoExtra

# The pinned WikiPron commit — the same one PR-A pinned for the English
# lexicon (name_banks.WIKIPRON_SOURCE_URL); phase 3 adds the per-language
# bulk TSVs at the same commit.
WIKIPRON_COMMIT = "d282e848a211ea31cfd730f0ced8bc8cdab9e83d"
_WIKIPRON_BASE = (
    "https://raw.githubusercontent.com/CUNY-CL/wikipron/"
    f"{WIKIPRON_COMMIT}/data/scrape/tsv/"
)


class LoanLanguage(str, Enum):
    """Closed set of pin-able source languages (design doc §8b).

    The 12 first-pass languages plus the five the miss decomposition named
    (Dutch, Welsh, Scottish Gaelic, Czech, Hungarian). Greek (ell) was
    measured in the first pass but is NOT carried: its lexicon keys are
    Greek-script and can never fold-match an ASCII bank token — the two
    romanized-Greek tokens (Asimeniou, Feggariou) are authored instead
    (decision recorded in the design doc §8b).
    """

    CZECH = "ces"
    WELSH = "cym"
    GERMAN = "deu"
    FRENCH = "fra"
    SCOTTISH_GAELIC = "gla"
    IRISH = "gle"
    HUNGARIAN = "hun"
    ITALIAN = "ita"
    DUTCH = "nld"
    NORWEGIAN = "nob"
    POLISH = "pol"
    PORTUGUESE = "por"
    ROMANIAN = "ron"
    SPANISH = "spa"
    SWEDISH = "swe"
    TURKISH = "tur"


class LexiconSource(BaseModelNoExtra):
    """One pinned per-language WikiPron bulk TSV."""

    filename: Annotated[str, Field(description="TSV filename at the pinned commit.")]
    url: Annotated[str, Field(description="Raw URL at the pinned commit.")]
    note: Annotated[
        str, Field(description="Which dialect/transcription tier was chosen and why.")
    ]


def _lexicon(filename: str, note: str) -> LexiconSource:
    return LexiconSource(filename=filename, url=_WIKIPRON_BASE + filename, note=note)


# Dialect / transcription-tier choices are pinned here, one per language.
# Broad transcriptions wherever WikiPron publishes them; Czech and Hungarian
# only exist as narrow. Spanish is Castilian (the repo's es variety pin) and
# Portuguese is European — matching the property names' flavor; Welsh is the
# South Wales tier (the larger speaker population).
LEXICON_SOURCES: dict[LoanLanguage, LexiconSource] = {
    LoanLanguage.CZECH: _lexicon("ces_latn_narrow.tsv", "only narrow published"),
    LoanLanguage.WELSH: _lexicon("cym_latn_sw_broad.tsv", "South Wales broad"),
    LoanLanguage.GERMAN: _lexicon("deu_latn_broad.tsv", "broad"),
    LoanLanguage.FRENCH: _lexicon("fra_latn_broad.tsv", "broad"),
    LoanLanguage.SCOTTISH_GAELIC: _lexicon("gla_latn_broad.tsv", "broad"),
    LoanLanguage.IRISH: _lexicon("gle_latn_broad.tsv", "broad"),
    LoanLanguage.HUNGARIAN: _lexicon("hun_latn_narrow.tsv", "only narrow published"),
    LoanLanguage.ITALIAN: _lexicon("ita_latn_broad.tsv", "broad"),
    LoanLanguage.DUTCH: _lexicon("nld_latn_broad.tsv", "broad"),
    LoanLanguage.NORWEGIAN: _lexicon("nob_latn_broad.tsv", "Bokmål broad"),
    LoanLanguage.POLISH: _lexicon("pol_latn_broad.tsv", "broad"),
    LoanLanguage.PORTUGUESE: _lexicon(
        "por_latn_po_broad.tsv", "European broad (the property names' flavor)"
    ),
    LoanLanguage.ROMANIAN: _lexicon("ron_latn_broad.tsv", "broad"),
    LoanLanguage.SPANISH: _lexicon(
        "spa_latn_ca_broad.tsv", "Castilian broad (the repo's es variety pin)"
    ),
    LoanLanguage.SWEDISH: _lexicon("swe_latn_broad.tsv", "broad"),
    LoanLanguage.TURKISH: _lexicon("tur_latn_broad.tsv", "broad"),
}


class StressRule(str, Enum):
    """Fixed per-language stress convention for unmarked bulk rows.

    The anglicized reading keeps the source language's characteristic stress
    (a fluent American says "may-GAHN", not "MAY-gan"). ``romance`` is the
    Spanish/Portuguese written rule: penultimate when the word ends in a
    vowel, n, or s; final otherwise. Words whose native spelling carries an
    explicit stress mark overriding that rule (zaguán) are authored instead
    — the ASCII bank token has lost the mark.
    """

    INITIAL = "initial"
    PENULTIMATE = "penultimate"
    FINAL = "final"
    ROMANCE = "romance"


LANGUAGE_STRESS: dict[LoanLanguage, StressRule] = {
    LoanLanguage.CZECH: StressRule.INITIAL,
    LoanLanguage.WELSH: StressRule.PENULTIMATE,
    LoanLanguage.GERMAN: StressRule.INITIAL,
    LoanLanguage.FRENCH: StressRule.FINAL,
    LoanLanguage.SCOTTISH_GAELIC: StressRule.INITIAL,
    LoanLanguage.IRISH: StressRule.INITIAL,
    LoanLanguage.HUNGARIAN: StressRule.INITIAL,
    LoanLanguage.ITALIAN: StressRule.PENULTIMATE,
    LoanLanguage.DUTCH: StressRule.INITIAL,
    LoanLanguage.NORWEGIAN: StressRule.INITIAL,
    LoanLanguage.POLISH: StressRule.PENULTIMATE,
    LoanLanguage.PORTUGUESE: StressRule.ROMANCE,
    LoanLanguage.ROMANIAN: StressRule.PENULTIMATE,
    LoanLanguage.SPANISH: StressRule.ROMANCE,
    LoanLanguage.SWEDISH: StressRule.INITIAL,
    LoanLanguage.TURKISH: StressRule.FINAL,
}


# ---------------------------------------------------------------------------
# Adaptation tables: foreign IPA segment -> respelling units
# ---------------------------------------------------------------------------

# Shared foreign base, layered over the English IPA tables. The flap rule is
# reversed on purpose: in the US-English lexicon ɾ is a flapped t/d, but in
# every source language here it is a true tapped r.
FOREIGN_BASE_CONSONANTS: dict[str, tuple[str, ...]] = {
    "r": ("r",),  # trills read as plain r
    "ɾ": ("r",),
    "ʀ": ("r",),
    "ʁ": ("r",),
    "ɲ": ("n", "y"),
    "ʎ": ("l", "y"),
    "ɣ": ("g",),
    "β": ("b",),
    "ʝ": ("y",),
    "ɕ": ("sh",),
    "ʑ": ("zh",),
    "tɕ": ("ch",),
    "dʑ": ("j",),
    "tʃ": ("ch",),  # tied affricates (ita t͡ʃ) — the tie bar strips as combining
    "dʒ": ("j",),
    "ʂ": ("sh",),
    "ʐ": ("zh",),
    "ɬ": ("l",),  # Welsh ll: the anglicized reading drops the fricative
    "ç": ("kh",),  # German ich-laut (design doc §8b example)
}

FOREIGN_BASE_VOWELS: dict[str, str] = {
    "ɯ": "ih",  # Turkish ı
    "ɤ": "uh",
    "œ": "eh",  # front rounded: the anglicized reading unrounds
    "ø": "eh",
    "y": "oo",  # French u / Germanic y: anglicized as oo ("ay-KLOOZ")
    "ʏ": "uu",
}

# Per-language overrides (consulted before the foreign base, which is
# consulted before the English base tables).
LANGUAGE_CONSONANTS: dict[LoanLanguage, dict[str, tuple[str, ...]]] = {
    LoanLanguage.GERMAN: {"x": ("kh",)},  # ach-laut ("BAKH")
    LoanLanguage.IRISH: {"x": ("k",)},  # Chnoc reads like "Knock"
    LoanLanguage.SCOTTISH_GAELIC: {"x": ("k",)},  # Machair reads "MAK-er"
    LoanLanguage.HUNGARIAN: {"c": ("t", "s"), "ɟ": ("d", "y")},
    LoanLanguage.SPANISH: {
        # Castilian θ in the bank tokens comes from the letter z (Azul,
        # Descalzo); a fluent American reads the letter, not the lisp.
        "θ": ("z",),
    },
}

LANGUAGE_VOWELS: dict[LoanLanguage, dict[str, str]] = {
    LoanLanguage.FRENCH: {"e": "ay", "ø": "oo"},  # é -> ay; vieux -> "VYOO"
    LoanLanguage.PORTUGUESE: {"e": "ay"},  # closed e: rei -> "RAY"
    LoanLanguage.DUTCH: {"e": "ay"},  # ee: negen -> "NAY-guh"
    LoanLanguage.POLISH: {"ɘ": "ih"},  # y: białym -> "BYAH-wim"
}

# Per-language two-segment merges, layered over the English diphthong table.
LANGUAGE_DIPHTHONGS: dict[LoanLanguage, dict[tuple[str, str], str]] = {
    LoanLanguage.PORTUGUESE: {("o", "w"): "oh"},  # pousada -> "poh-ZAH-duh"
    LoanLanguage.GERMAN: {("ɔ", "ʏ"): "oy"},  # eu/äu
}

# Modifier letters stripped from foreign segments in addition to the English
# strip set (velarization, pharyngealization, unreleased stops).
_FOREIGN_STRIP = frozenset("ːˑ˞ʲʰˠˤ˺ʷ")

_NASAL_MARK = "̃"  # combining tilde


def _strip_foreign_segment(segment: str) -> tuple[str, bool, bool]:
    """(base symbols, is_syllabic, is_nasal) for one lexicon segment.

    The base is built by recomposing (NFC) the decomposition with the nasal
    tilde removed, then dropping the remaining combining marks — NOT by
    stripping every combining mark off the raw NFD: that would also delete
    the cedilla out of the ich-laut ``ç`` (leaving a bare ``c`` that misses
    the adaptation table). Recomposition keeps table symbols like ``ç`` one
    non-combining character and folds precomposed nasal vowels (``ĩ``) onto
    their plain letter, while genuine decorations — dental bridge,
    non-syllabic breve, the affricate tie bar — remain combining marks and
    are stripped along with the ``_FOREIGN_STRIP`` modifier letters.
    """
    decomposed = unicodedata.normalize("NFD", segment)
    syllabic = "̩" in decomposed
    nasal = _NASAL_MARK in decomposed
    recomposed = unicodedata.normalize("NFC", decomposed.replace(_NASAL_MARK, ""))
    base = "".join(
        ch
        for ch in recomposed
        if not unicodedata.combining(ch) and ch not in _FOREIGN_STRIP
    )
    return base, syllabic, nasal


def _lookup_vowel(language: LoanLanguage, base: str) -> Optional[str]:
    for table in (
        LANGUAGE_VOWELS.get(language, {}),
        FOREIGN_BASE_VOWELS,
        IPA_VOWELS,
    ):
        if base in table:
            return table[base]
    return None


def _lookup_consonant(language: LoanLanguage, base: str) -> Optional[tuple[str, ...]]:
    override = LANGUAGE_CONSONANTS.get(language, {})
    if base in override:
        return override[base]
    if base in FOREIGN_BASE_CONSONANTS:
        return FOREIGN_BASE_CONSONANTS[base]
    if base in IPA_CONSONANTS:
        return (IPA_CONSONANTS[base],)
    return None


def foreign_phones(phonemes: str, language: LoanLanguage) -> list[_Phone]:
    """Parse one space-segmented foreign lexicon row into respelling phones.

    Nasalized vowels emit the vowel followed by an ``ng`` consonant — the
    fluent-American reading of the Portuguese/French nasals (design doc §8b).
    """
    stripped: list[tuple[str, bool, bool]] = []
    for segment in phonemes.split():
        base, syllabic, nasal = _strip_foreign_segment(segment)
        if not base or base == "ʔ":
            continue
        stripped.append((base, syllabic, nasal))

    diphthongs = {**IPA_DIPHTHONGS, **LANGUAGE_DIPHTHONGS.get(language, {})}
    phones: list[_Phone] = []
    index = 0
    while index < len(stripped):
        base, _syllabic, nasal = stripped[index]
        if index + 1 < len(stripped):
            pair = (base, stripped[index + 1][0])
            if pair in diphthongs:
                phones.append(_Phone(unit=diphthongs[pair], is_vowel=True, stress=None))
                if nasal or stripped[index + 1][2]:
                    phones.append(_Phone(unit="ng", is_vowel=False))
                index += 2
                continue
        vowel = _lookup_vowel(language, base)
        if vowel is not None:
            phones.append(_Phone(unit=vowel, is_vowel=True, stress=None))
            if nasal:
                phones.append(_Phone(unit="ng", is_vowel=False))
            index += 1
            continue
        consonant = _lookup_consonant(language, base)
        if consonant is not None:
            for unit in consonant:
                phones.append(_Phone(unit=unit, is_vowel=False))
            index += 1
            continue
        raise PronunciationError(
            f"Unknown {language.value} IPA segment {base!r} in {phonemes!r} — "
            "extend the loanword adaptation tables deliberately"
        )
    if not phones:
        raise PronunciationError(f"Empty {language.value} phoneme string: {phonemes!r}")
    return phones


def _apply_stress(syllables: list[Syllable], rule: StressRule) -> list[Syllable]:
    for syllable in syllables:
        syllable.stressed = False
    if rule is StressRule.INITIAL:
        index = 0
    elif rule is StressRule.FINAL:
        index = len(syllables) - 1
    elif rule is StressRule.PENULTIMATE:
        index = max(len(syllables) - 2, 0)
    else:  # ROMANCE: penult when vowel/n/s-final, final otherwise
        coda = syllables[-1].coda
        if not coda or (len(coda) == 1 and coda[0] in ("n", "s")):
            index = max(len(syllables) - 2, 0)
        else:
            index = len(syllables) - 1
    syllables[index].stressed = True
    return syllables


def adapt_foreign_ipa(phonemes: str, language: LoanLanguage) -> list[Syllable]:
    """The full loanword converter: foreign lexicon row -> stressed syllables.

    Parse under the language's adaptation tables, syllabify with the shared
    onset-maximization machinery, then place stress by the language's fixed
    convention (bulk rows carry no stress marks).
    """
    syllables = syllabify(foreign_phones(phonemes, language))
    return _apply_stress(syllables, LANGUAGE_STRESS[language])


__all__ = [
    "FOREIGN_BASE_CONSONANTS",
    "FOREIGN_BASE_VOWELS",
    "LANGUAGE_CONSONANTS",
    "LANGUAGE_DIPHTHONGS",
    "LANGUAGE_STRESS",
    "LANGUAGE_VOWELS",
    "LEXICON_SOURCES",
    "LexiconSource",
    "LoanLanguage",
    "StressRule",
    "WIKIPRON_COMMIT",
    "adapt_foreign_ipa",
    "foreign_phones",
]
