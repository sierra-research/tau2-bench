# Copyright Sierra
"""Fixed phoneme -> respelling converters and distortion operators.

The intake pronunciation columns (design doc §2.4 / §4) are derived by the
code in this module — deterministic, table-driven, no LLM:

1. **Converters**: ARPABET (CMUdict) and IPA (WikiPron ``eng_latn_us_broad``)
   phoneme strings parse into a shared phone sequence, syllabify by onset
   maximization over a closed legal-onset table, and render as a hyphenated
   ASCII respelling with the primary-stressed syllable in caps
   (``Z OW1 G B IY0`` -> ``"ZOHG-bee"``). WikiPron US broad transcriptions
   carry **no stress marks**, so unmarked words take the fixed fallback:
   primary stress on the **penultimate** syllable (the first syllable for
   monosyllables).
2. **Distortion operators** (design doc §4): a closed catalog of
   deterministic functions producing the ``mispronounced`` variant. Most
   operate on the syllabified respelling; ``spelling_pronunciation`` operates
   on the token SPELLING via the fixed naive letter-to-sound table below.
   The stored/spoken variant is rendered by :func:`render_natural` in
   NATURAL ORTHOGRAPHY (a plausible normal word, TTS-ready verbatim).
   Applicability is per bank (:data:`OPERATOR_APPLICABILITY`, owner directive
   2026-08-26), then per-value feasibility is structural (a 2-syllable value
   cannot ``syllable_drop``); ``vowel_swap`` is always feasible and applies
   to every bank, so the feasible set is never empty. Operators are drawn by
   :class:`MispronunciationDrawer` — balance-greedy per bank, seeded
   tie-breaks — and the drawn operator is recorded next to the variant in
   the bank (:mod:`tau2.domains.intake.tasks.name_banks`).

Everything here is pure and total over the tables below; an unknown phoneme
segment raises :class:`PronunciationError` — extend the table deliberately,
never coerce silently.
"""

import random
import unicodedata
from enum import Enum
from typing import Annotated, Literal, Optional, Sequence

from pydantic import Field

from tau2.utils.pydantic_utils import BaseModelNoExtra


class PronunciationError(Exception):
    """A phoneme string cannot be converted; the bank build must not ship."""


class DistortionOperator(str, Enum):
    """Closed catalog of mispronunciation operators (design doc §4)."""

    METATHESIS = "metathesis"  # transpose adjacent phonemes in a syllable
    VOWEL_SWAP = "vowel_swap"  # substitute the stressed vowel
    SYLLABLE_DROP = "syllable_drop"  # drop an unstressed medial syllable
    SPELLING_PRONUNCIATION = "spelling_pronunciation"  # read the spelling naively


class Syllable(BaseModelNoExtra):
    """One syllable of a respelling: onset units, nucleus vowel unit, coda
    units, and whether it carries primary stress (rendered in caps)."""

    onset: Annotated[
        list[str], Field(description="Consonant respelling units before the vowel.")
    ]
    nucleus: Annotated[str, Field(description="The vowel respelling unit.")]
    coda: Annotated[
        list[str], Field(description="Consonant respelling units after the vowel.")
    ]
    stressed: Annotated[
        bool, Field(description="Primary stress (the caps syllable).")
    ] = False


class _Phone(BaseModelNoExtra):
    """One parsed phone: a respelling unit plus vowel/stress metadata."""

    unit: str
    is_vowel: bool
    stress: Optional[int] = None  # ARPABET 0/1/2; None when unmarked (IPA)


# ---------------------------------------------------------------------------
# ARPABET tables (CMUdict)
# ---------------------------------------------------------------------------

ARPABET_CONSONANTS: dict[str, str] = {
    "B": "b",
    "CH": "ch",
    "D": "d",
    "DH": "th",
    "F": "f",
    "G": "g",
    "HH": "h",
    "JH": "j",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ng",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "sh",
    "T": "t",
    "TH": "th",
    "V": "v",
    "W": "w",
    "Y": "y",
    "Z": "z",
    "ZH": "zh",
}

ARPABET_VOWELS: dict[str, str] = {
    "AA": "ah",
    "AE": "a",
    "AH": "uh",
    "AO": "aw",
    "AW": "ow",
    "AY": "eye",  # rendered "y" after an onset (MY), "eye" bare (EYE-so...)
    "EH": "eh",
    "ER": "ur",
    "EY": "ay",
    "IH": "ih",
    "IY": "ee",
    "OW": "oh",
    "OY": "oy",
    "UH": "uu",
    "UW": "oo",
}

# ---------------------------------------------------------------------------
# IPA tables (WikiPron eng_latn_us_broad)
# ---------------------------------------------------------------------------

# Diphthongs written as ONE segment (compared after diacritic stripping).
IPA_ATTACHED_DIPHTHONGS: dict[str, str] = {
    "aɪ": "eye",
    "äɪ": "eye",
    "ɑɪ": "eye",
    "aʊ": "ow",
    "äʊ": "ow",
    "ɑʊ": "ow",
    "eɪ": "ay",
    "oʊ": "oh",
    "ɔɪ": "oy",
}

# Two-segment vowel sequences that read as one diphthong. WikiPron US broad
# usually splits diphthongs into segments ("o ʊ" for /oʊ/); the parser merges
# these pairs (compared after diacritic stripping) into the unit named here.
IPA_DIPHTHONGS: dict[tuple[str, str], str] = {
    ("a", "ɪ"): "eye",
    ("ä", "ɪ"): "eye",
    ("ɑ", "ɪ"): "eye",
    ("a", "ʊ"): "ow",
    ("ä", "ʊ"): "ow",
    ("ɑ", "ʊ"): "ow",
    ("e", "ɪ"): "ay",
    ("o", "ʊ"): "oh",
    ("ɔ", "ɪ"): "oy",
}

IPA_VOWELS: dict[str, str] = {
    "a": "ah",
    "ä": "ah",
    "ɐ": "uh",
    "ɑ": "ah",
    "ɒ": "ah",
    "æ": "a",
    "e": "eh",
    "ø": "ur",
    "ɘ": "uh",
    "ə": "uh",
    "ɚ": "ur",
    "ɛ": "eh",
    "ɜ": "ur",
    "ɝ": "ur",
    "i": "ee",
    "ɨ": "ih",
    "ɪ": "ih",
    "o": "oh",
    "ɔ": "aw",
    "u": "oo",
    "ʉ": "oo",
    "ʊ": "uu",
    "ʌ": "uh",
    "ɵ": "uh",
    "ᵻ": "ih",
    "y": "ee",
}

IPA_CONSONANTS: dict[str, str] = {
    "b": "b",
    "d": "d",
    "f": "f",
    "h": "h",
    "j": "y",
    "k": "k",
    "l": "l",
    "ɫ": "l",
    "m": "m",
    "n": "n",
    "ŋ": "ng",
    "p": "p",
    "s": "s",
    "t": "t",
    "v": "v",
    "w": "w",
    "z": "z",
    "ð": "th",
    "ɡ": "g",
    "g": "g",
    "ɹ": "r",
    "ɾ": "d",  # alveolar flap: reads as a quick d
    "θ": "th",
    "ʃ": "sh",
    "ʒ": "zh",
    "ʋ": "v",
    "ʍ": "w",
    "x": "h",
    "χ": "h",
    "˞": "r",  # bare rhotic hook segment: r-coloring
}

# Multi-character / tied segments checked before single-symbol mapping
# (keys are the tie-stripped forms; syllabic consonants read as vowel units).
IPA_SPECIAL_CONSONANTS: dict[str, tuple[str, ...]] = {
    "tʃ": ("ch",),
    "dʒ": ("j",),
    "ts": ("t", "s"),
    "dɹ": ("d", "r"),
}

IPA_SYLLABIC_VOWELS: dict[str, str] = {
    "l": "ul",
    "m": "um",
    "n": "un",
}

# Glottal stop carries no respelling.
_IPA_DROPPED = frozenset({"ʔ"})

# Combining marks / modifiers stripped from IPA segments before mapping
# (length, nasalization, tone, non-syllabicity, palatalization, ...).
_IPA_STRIP = frozenset("ːˑ˞ʲʰ")


def _strip_ipa_segment(segment: str) -> tuple[str, bool]:
    """(base symbols, is_syllabic) for one WikiPron segment."""
    decomposed = unicodedata.normalize("NFD", segment)
    syllabic = "̩" in decomposed
    base = "".join(
        ch
        for ch in decomposed
        if not unicodedata.combining(ch) and ch not in _IPA_STRIP and ch != "͡"
    )
    return unicodedata.normalize("NFC", base), syllabic


# ---------------------------------------------------------------------------
# Phone parsing
# ---------------------------------------------------------------------------


def phones_from_arpabet(phonemes: str) -> list[_Phone]:
    """Parse a CMUdict ARPABET string (``Z OW1 G B IY0``) into phones."""
    phones: list[_Phone] = []
    for token in phonemes.split():
        stress: Optional[int] = None
        symbol = token
        if symbol and symbol[-1].isdigit():
            stress = int(symbol[-1])
            symbol = symbol[:-1]
        if symbol in ARPABET_VOWELS:
            phones.append(
                _Phone(unit=ARPABET_VOWELS[symbol], is_vowel=True, stress=stress)
            )
        elif symbol in ARPABET_CONSONANTS:
            phones.append(_Phone(unit=ARPABET_CONSONANTS[symbol], is_vowel=False))
        else:
            raise PronunciationError(
                f"Unknown ARPABET symbol {token!r} in {phonemes!r}"
            )
    if not phones:
        raise PronunciationError(f"Empty ARPABET phoneme string: {phonemes!r}")
    return phones


def phones_from_ipa(phonemes: str) -> list[_Phone]:
    """Parse a WikiPron space-segmented IPA string into phones.

    US-broad WikiPron rows carry no stress marks; every vowel parses with
    ``stress=None`` and :func:`syllabify` applies the penultimate fallback.
    """
    stripped: list[tuple[str, bool]] = []
    for segment in phonemes.split():
        base, syllabic = _strip_ipa_segment(segment)
        if not base or base in _IPA_DROPPED:
            continue
        stripped.append((base, syllabic))

    phones: list[_Phone] = []
    index = 0
    while index < len(stripped):
        base, syllabic = stripped[index]
        if syllabic and base in IPA_SYLLABIC_VOWELS:
            phones.append(
                _Phone(unit=IPA_SYLLABIC_VOWELS[base], is_vowel=True, stress=None)
            )
            index += 1
            continue
        if index + 1 < len(stripped):
            pair = (base, stripped[index + 1][0])
            if pair in IPA_DIPHTHONGS:
                phones.append(
                    _Phone(unit=IPA_DIPHTHONGS[pair], is_vowel=True, stress=None)
                )
                index += 2
                continue
        if base in IPA_SPECIAL_CONSONANTS:
            for unit in IPA_SPECIAL_CONSONANTS[base]:
                phones.append(_Phone(unit=unit, is_vowel=False))
            index += 1
            continue
        if base in IPA_ATTACHED_DIPHTHONGS:
            phones.append(
                _Phone(unit=IPA_ATTACHED_DIPHTHONGS[base], is_vowel=True, stress=None)
            )
            index += 1
            continue
        if base in IPA_VOWELS:
            phones.append(_Phone(unit=IPA_VOWELS[base], is_vowel=True, stress=None))
            index += 1
            continue
        if base in IPA_CONSONANTS:
            phones.append(_Phone(unit=IPA_CONSONANTS[base], is_vowel=False))
            index += 1
            continue
        raise PronunciationError(f"Unknown IPA segment {base!r} in {phonemes!r}")
    if not phones:
        raise PronunciationError(f"Empty IPA phoneme string: {phonemes!r}")
    return phones


# ---------------------------------------------------------------------------
# Syllabification (onset maximization over a closed legal-onset table)
# ---------------------------------------------------------------------------

# Legal English syllable onsets, as tuples of consonant respelling units.
# Word-initial clusters are taken whole regardless (they were legal enough
# for the word to exist); this table splits INTERVOCALIC clusters.
_SINGLE_ONSETS = frozenset(unit for unit in ARPABET_CONSONANTS.values() if unit != "ng")
_CLUSTER_ONSETS: frozenset[tuple[str, ...]] = frozenset(
    tuple(cluster.split())
    for cluster in (
        "b l|b r|b y|d r|d w|d y|f l|f r|f y|g l|g r|g w|g y|h y|k l|k r|k w|"
        "k y|m y|n y|p l|p r|p y|s f|s k|s l|s m|s n|s p|s t|s w|sh r|t r|t w|"
        "th r|th w|v y|"
        "s k r|s k w|s k y|s p l|s p r|s p y|s t r|s t y"
    ).split("|")
)


def _is_legal_onset(units: tuple[str, ...]) -> bool:
    if len(units) == 0:
        return True
    if len(units) == 1:
        return units[0] in _SINGLE_ONSETS
    return units in _CLUSTER_ONSETS


def syllabify(phones: list[_Phone]) -> list[Syllable]:
    """Group phones into syllables; exactly one syllable carries stress.

    Primary stress: the first ARPABET stress-1 vowel; else the first
    stress-2 vowel; else the fixed unmarked-word fallback — penultimate
    syllable (first syllable of a monosyllable).
    """
    vowel_positions = [i for i, phone in enumerate(phones) if phone.is_vowel]
    if not vowel_positions:
        raise PronunciationError(
            f"No vowel in phone sequence: {[p.unit for p in phones]}"
        )

    syllables: list[Syllable] = []
    onsets: list[list[str]] = [[] for _ in vowel_positions]
    codas: list[list[str]] = [[] for _ in vowel_positions]

    # Leading consonants: all onset of the first syllable.
    onsets[0] = [p.unit for p in phones[: vowel_positions[0]]]
    # Trailing consonants: all coda of the last syllable.
    codas[-1] = [p.unit for p in phones[vowel_positions[-1] + 1 :]]
    # Intervocalic clusters: longest legal onset goes right, rest goes left.
    for k in range(len(vowel_positions) - 1):
        cluster = [
            p.unit for p in phones[vowel_positions[k] + 1 : vowel_positions[k + 1]]
        ]
        split = 0  # default: whole cluster is the left coda
        for start in range(len(cluster)):
            if _is_legal_onset(tuple(cluster[start:])):
                split = start
                break
        codas[k] = cluster[:split]
        onsets[k + 1] = cluster[split:]

    stressed_index = _primary_stress_index(phones, vowel_positions)
    for k, position in enumerate(vowel_positions):
        syllables.append(
            Syllable(
                onset=onsets[k],
                nucleus=phones[position].unit,
                coda=codas[k],
                stressed=(k == stressed_index),
            )
        )
    return syllables


def _primary_stress_index(phones: list[_Phone], vowel_positions: list[int]) -> int:
    for wanted in (1, 2):
        for k, position in enumerate(vowel_positions):
            if phones[position].stress == wanted:
                return k
    return max(len(vowel_positions) - 2, 0)  # penultimate fallback


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_syllable(syllable: Syllable) -> str:
    nucleus = syllable.nucleus
    if nucleus == "eye" and syllable.onset:
        nucleus = "y"  # "MY", not "MEYE"; bare syllables keep "eye"
    text = "".join(syllable.onset) + nucleus + "".join(syllable.coda)
    return text.upper() if syllable.stressed else text


def render_respelling(syllables: list[Syllable]) -> str:
    """Hyphen-joined syllables, primary-stressed syllable in caps."""
    if not syllables:
        raise PronunciationError("Cannot render an empty syllable list")
    if sum(1 for s in syllables if s.stressed) != 1:
        raise PronunciationError(
            "Respelling requires exactly one stressed syllable: "
            f"{[s.model_dump() for s in syllables]}"
        )
    rendered = "-".join(_render_syllable(s) for s in syllables)
    if not rendered.isascii():
        raise PronunciationError(f"Respelling is not ASCII: {rendered!r}")
    return rendered


def syllabify_phonemes(
    phonemes: str, scheme: Literal["arpabet", "ipa"]
) -> list[Syllable]:
    """Parse + syllabify one phoneme string of the given scheme."""
    if scheme == "arpabet":
        return syllabify(phones_from_arpabet(phonemes))
    return syllabify(phones_from_ipa(phonemes))


def respell_phonemes(phonemes: str, scheme: Literal["arpabet", "ipa"]) -> str:
    """The full converter: phoneme string -> hyphenated caps-stress respelling."""
    return render_respelling(syllabify_phonemes(phonemes, scheme))


# ---------------------------------------------------------------------------
# Naive spelling pronunciation (spelling_pronunciation operator)
# ---------------------------------------------------------------------------

# The fixed letter-to-sound rules a naive American reader applies to an
# unfamiliar spelling ("Chaim" read as CHAYM — chain with an m — instead of
# the correct HY-ihm). Deliberately SMALL and closed: exactly the rules
# below, nothing cleverer. Rules, in application order:
#
# 1. Apostrophes are dropped; the token is lowercased (must be ASCII alpha).
# 2. Adjacent identical consonant letters collapse ("tt" -> "t"), so doubled
#    consonants and the digraphs they hide read once ("Matthew" -> "Mathew").
# 3. A leading "Mc" reads "muh-k" (the one proper-noun convention every
#    naive reader knows; mirrors the census casing rule in name_banks).
# 4. Two-letter graphemes (longest match first, the table below): vowel
#    teams (ai/ay -> "ay"; ei/ey -> "ay" as in "eight"/"they" — the one
#    documented choice for the ambiguous ei), r-controlled vowels
#    (ar -> "ah"+r, or -> "aw"+r, er/ir/ur -> "ur"), and consonant digraphs
#    (ch, sh, th, ph -> f, wh -> w, ck -> k, gh -> hard g, ng, qu -> kw).
# 5. c and g are soft before e/i/y ("s"/"j"), hard otherwise ("k"/"g").
# 6. x -> "ks"; y is the consonant "y" word-initially before a vowel letter,
#    the vowel "ee" word-finally, and "ih" elsewhere.
# 7. Single vowels read short: a -> "a", e -> "eh", i -> "ih", o -> "ah",
#    u -> "uh"; a word-final "a" reads "uh" ("Anna" -> AN-uh).
# 8. Every other consonant letter reads as itself.
# 9. A word-final standalone "e" after a consonant is silent when an earlier
#    vowel exists; when exactly one consonant separates it from that vowel,
#    the vowel lengthens (magic e: a->ay, eh->ee, ih->eye, ah->oh, uh->oo).
# 10. Syllabification and rendering are the shared machinery above; the
#     naive reading carries no stress marks, so the penultimate fallback
#     applies (the same convention as unmarked WikiPron rows).

# Grapheme pair -> (unit, is_vowel) phone sequence.
SPELLING_TWO_LETTER: dict[str, tuple[tuple[str, bool], ...]] = {
    "ai": (("ay", True),),
    "ay": (("ay", True),),
    "au": (("aw", True),),
    "aw": (("aw", True),),
    "ea": (("ee", True),),
    "ee": (("ee", True),),
    "ei": (("ay", True),),
    "ey": (("ay", True),),
    "ie": (("ee", True),),
    "oa": (("oh", True),),
    "oi": (("oy", True),),
    "oy": (("oy", True),),
    "oo": (("oo", True),),
    "ou": (("ow", True),),
    "ow": (("ow", True),),
    "ue": (("oo", True),),
    "ui": (("oo", True),),
    "eu": (("oo", True),),
    "ew": (("oo", True),),
    "ar": (("ah", True), ("r", False)),
    "or": (("aw", True), ("r", False)),
    "er": (("ur", True),),
    "ir": (("ur", True),),
    "ur": (("ur", True),),
    "ch": (("ch", False),),
    "ck": (("k", False),),
    "gh": (("g", False),),
    "ng": (("ng", False),),
    "ph": (("f", False),),
    "qu": (("k", False), ("w", False)),
    "sh": (("sh", False),),
    "th": (("th", False),),
    "wh": (("w", False),),
}

SPELLING_SHORT_VOWELS: dict[str, str] = {
    "a": "a",
    "e": "eh",
    "i": "ih",
    "o": "ah",
    "u": "uh",
}

# Magic-e lengthening over the short vowel UNITS (rule 9).
SPELLING_MAGIC_E: dict[str, str] = {
    "a": "ay",
    "eh": "ee",
    "ih": "eye",
    "ah": "oh",
    "uh": "oo",
}

SPELLING_PLAIN_CONSONANTS: dict[str, str] = {
    letter: letter for letter in "bdfhjklmnprstvwz"
}

_VOWEL_LETTERS = frozenset("aeiou")
_SOFTENING_LETTERS = frozenset("eiy")


def spelling_syllables(token: str) -> list[Syllable]:
    """The naive American reading of ``token``'s SPELLING, syllabified.

    Total over ASCII alphabetic tokens (apostrophes dropped) with at least
    one vowel phone; anything else raises :class:`PronunciationError`.
    """
    letters = token.replace("'", "").lower()
    if not letters or not letters.isascii() or not letters.isalpha():
        raise PronunciationError(f"Cannot naively read non-alphabetic token {token!r}")
    # Rule 2: collapse adjacent identical consonant letters.
    collapsed: list[str] = []
    for letter in letters:
        if collapsed and collapsed[-1] == letter and letter not in _VOWEL_LETTERS:
            continue
        collapsed.append(letter)
    letters = "".join(collapsed)

    phones: list[_Phone] = []
    starts: list[int] = []  # letter index each phone started at (rule 9)

    def emit(unit: str, is_vowel: bool, start: int) -> None:
        phones.append(_Phone(unit=unit, is_vowel=is_vowel, stress=None))
        starts.append(start)

    index = 0
    if letters.startswith("mc") and len(letters) > 2:  # rule 3
        emit("m", False, 0)
        emit("uh", True, 0)
        emit("k", False, 1)
        index = 2
    while index < len(letters):
        letter = letters[index]
        nxt = letters[index + 1] if index + 1 < len(letters) else ""
        pair = letters[index : index + 2]
        if pair in SPELLING_TWO_LETTER:  # rule 4
            for unit, is_vowel in SPELLING_TWO_LETTER[pair]:
                emit(unit, is_vowel, index)
            index += 2
            continue
        if letter == "c":  # rule 5
            emit("s" if nxt in _SOFTENING_LETTERS else "k", False, index)
        elif letter == "g":  # rule 5
            emit("j" if nxt in _SOFTENING_LETTERS else "g", False, index)
        elif letter == "x":  # rule 6
            emit("k", False, index)
            emit("s", False, index)
        elif letter == "y":  # rule 6
            if index == 0 and nxt in _VOWEL_LETTERS:
                emit("y", False, index)
            elif index == len(letters) - 1:
                emit("ee", True, index)
            else:
                emit("ih", True, index)
        elif letter in SPELLING_SHORT_VOWELS:  # rule 7
            if letter == "a" and index == len(letters) - 1:
                emit("uh", True, index)
            else:
                emit(SPELLING_SHORT_VOWELS[letter], True, index)
        elif letter in SPELLING_PLAIN_CONSONANTS:  # rule 8
            emit(SPELLING_PLAIN_CONSONANTS[letter], False, index)
        else:  # pragma: no cover — q outside qu is the only reachable case
            raise PronunciationError(
                f"No naive reading rule for letter {letter!r} in {token!r}"
            )
        index += 1

    # Rule 9: silent final e (+ magic-e lengthening over ONE consonant).
    if (
        letters.endswith("e")
        and phones
        and phones[-1].unit == "eh"
        and starts[-1] == len(letters) - 1
        and any(phone.is_vowel for phone in phones[:-1])
    ):
        del phones[-1], starts[-1]
        vowel_indexes = [i for i, phone in enumerate(phones) if phone.is_vowel]
        trailing = len(phones) - 1 - vowel_indexes[-1]
        lengthened = SPELLING_MAGIC_E.get(phones[vowel_indexes[-1]].unit)
        if trailing == 1 and lengthened is not None:
            phones[vowel_indexes[-1]].unit = lengthened

    return syllabify(phones)  # rule 10 (raises when no vowel phone remains)


def spelling_respelling(token: str) -> str:
    """The rendered naive reading of ``token`` (rules above)."""
    return render_respelling(spelling_syllables(token))


# ---------------------------------------------------------------------------
# Distortion operators (design doc §4)
# ---------------------------------------------------------------------------

# Total stressed-vowel substitution table (vowel_swap). Every nucleus unit
# maps to a different-sounding unit, so the variant always differs audibly.
VOWEL_SWAPS: dict[str, str] = {
    "a": "ah",
    "ah": "a",
    "aw": "ow",
    "ow": "aw",
    "ay": "eh",
    "eh": "ay",
    "ee": "ih",
    "ih": "ee",
    "eye": "ay",
    "oh": "oo",
    "oo": "oh",
    "oy": "oh",
    "uh": "ah",
    "ur": "ar",
    "uu": "oo",
    "ul": "ol",
    "um": "om",
    "un": "on",
}


def _stressed_index(syllables: list[Syllable]) -> int:
    for index, syllable in enumerate(syllables):
        if syllable.stressed:
            return index
    raise PronunciationError("No stressed syllable")


def _metathesis_target(syllables: list[Syllable]) -> Optional[int]:
    """The syllable metathesis applies to: the stressed one when its onset is
    a cluster, else the first syllable with a cluster onset."""
    stressed = _stressed_index(syllables)
    if len(syllables[stressed].onset) >= 2:
        return stressed
    for index, syllable in enumerate(syllables):
        if len(syllable.onset) >= 2:
            return index
    return None


def _droppable_index(syllables: list[Syllable]) -> Optional[int]:
    """The first unstressed medial syllable (never first, last, or stressed)."""
    for index in range(1, len(syllables) - 1):
        if not syllables[index].stressed:
            return index
    return None


# Which operators may appear in which bank (owner directive 2026-08-26):
# syllable_drop applies to medications only — a person name missing a
# syllable reads as a DIFFERENT name, not a mispronounced one. stress_shift
# was RETIRED outright (owner decision, 2026-08-26 audition listen): its
# variant differed from the correct respelling ONLY by caps position, and
# eleven_v3 does not render that caps position reliably (short caps
# syllables read as initialisms), so the operator produced no audible
# mispronunciation. Consulted BEFORE the per-value structural gates;
# validated on bank load (an operator recorded on a bank it does not apply
# to fails loud).
#
# Phase 3 (design doc §8b): properties and vehicles are treated LIKE
# person_names — metathesis / vowel_swap / spelling_pronunciation, NO
# syllable_drop: a place or model name missing a syllable reads as a
# DIFFERENT name, not a mispronounced one. This extension is a phase-3
# proposal implemented pending owner review via the foreign-token packet
# (`tau2 intake-names foreign-packet`).
_NAME_LIKE_OPERATORS = frozenset(
    {
        DistortionOperator.METATHESIS,
        DistortionOperator.VOWEL_SWAP,
        DistortionOperator.SPELLING_PRONUNCIATION,
    }
)

OPERATOR_APPLICABILITY: dict[str, frozenset[DistortionOperator]] = {
    "person_names": _NAME_LIKE_OPERATORS,
    "medications": frozenset(
        {
            DistortionOperator.METATHESIS,
            DistortionOperator.VOWEL_SWAP,
            DistortionOperator.SYLLABLE_DROP,
            DistortionOperator.SPELLING_PRONUNCIATION,
        }
    ),
    "properties": _NAME_LIKE_OPERATORS,
    "vehicles": _NAME_LIKE_OPERATORS,
}


# Hand-tuned hard pins (owner directive 2026-08-26): the balance-greedy draw
# is blind to how HARD each operator's variant is for a given token, so it
# wastes the hard-hitting cases — Chaim drew a mild vowel_swap ("HAY-ihm")
# when its naive spelling reading ("CHAYM") is barely recognizable. Where one
# operator's variant diverges far more audibly than the balance pick, the
# owner pins that operator here; with the pins, roughly half of each bank's
# drawn variants are hard (audibly far from the correct reading) and the rest
# stay mild — both regimes are measured. Closed, owner-curated table
# (curated 2026-08-26 from a divergence ranking of every feasible operator
# variant, then hand-reviewed; e.g. Janey was rejected because its naive
# "JEH-nee" is exactly the name Jenny — a pinned variant must sound WRONG,
# never like a different valid entity). A pinned operator must be feasible
# for its token and every pin must be consumed by the bank build, both
# fail-loud. (14 spelling_pronunciation pins were dropped with the natural-
# orthography rendering, 2026-08-26: their naive readings spelled naturally
# reconstruct the token itself — Olayan -> "olayan" — so the swap would be
# vacuous; those tokens fall back to the balance draw.)
HARD_PIN_OPERATORS: dict[str, dict[str, DistortionOperator]] = {
    "person_names": {
        "Cesario": DistortionOperator.SPELLING_PRONUNCIATION,
        "Chaim": DistortionOperator.SPELLING_PRONUNCIATION,
        "Cherie": DistortionOperator.SPELLING_PRONUNCIATION,
        "Christiane": DistortionOperator.SPELLING_PRONUNCIATION,
        "Cutugno": DistortionOperator.SPELLING_PRONUNCIATION,
        "Denyse": DistortionOperator.VOWEL_SWAP,
        "Durenberger": DistortionOperator.VOWEL_SWAP,
        "Geoff": DistortionOperator.VOWEL_SWAP,
        "Georgakis": DistortionOperator.SPELLING_PRONUNCIATION,
        "Hellerman": DistortionOperator.VOWEL_SWAP,
        "Hester": DistortionOperator.VOWEL_SWAP,
        "Lifland": DistortionOperator.VOWEL_SWAP,
        "Medora": DistortionOperator.SPELLING_PRONUNCIATION,
        "Natalia": DistortionOperator.SPELLING_PRONUNCIATION,
        "Oneil": DistortionOperator.SPELLING_PRONUNCIATION,
        "Renna": DistortionOperator.VOWEL_SWAP,
        "Rimson": DistortionOperator.VOWEL_SWAP,
        "Terrel": DistortionOperator.SPELLING_PRONUNCIATION,
        "Wilford": DistortionOperator.VOWEL_SWAP,
    },
    "medications": {
        "Azithromycin": DistortionOperator.SPELLING_PRONUNCIATION,
        "Chlorpromazine": DistortionOperator.SPELLING_PRONUNCIATION,
        "Colesevelam": DistortionOperator.SPELLING_PRONUNCIATION,
        "Dimenhydrinate": DistortionOperator.SPELLING_PRONUNCIATION,
        "Furosemide": DistortionOperator.SPELLING_PRONUNCIATION,
        "Guaifenesin": DistortionOperator.SPELLING_PRONUNCIATION,
        "Hydrochlorothiazide": DistortionOperator.SPELLING_PRONUNCIATION,
        "Hydroxychloroquine": DistortionOperator.SPELLING_PRONUNCIATION,
        "Hydroxyzine": DistortionOperator.SPELLING_PRONUNCIATION,
        "Lamotrigine": DistortionOperator.SPELLING_PRONUNCIATION,
        "Loratadine": DistortionOperator.SPELLING_PRONUNCIATION,
        "Meloxicam": DistortionOperator.SPELLING_PRONUNCIATION,
        "Naproxen": DistortionOperator.SPELLING_PRONUNCIATION,
        "Olanzapine": DistortionOperator.SPELLING_PRONUNCIATION,
        "Omeprazole": DistortionOperator.SPELLING_PRONUNCIATION,
        "Pioglitazone": DistortionOperator.SPELLING_PRONUNCIATION,
    },
}


# Tokens for which NO feasible operator yields a non-vacuous natural-
# orthography variant (the correct reading spells back to the token itself
# and every swap lands on it too — "-sartan" transparency). Closed allowlist,
# enforced BOTH ways at bank load: a listed token carrying a variant and an
# unlisted mispronounced-required token missing one both fail loud, so the
# list tracks the banks exactly. These tokens are simply never drawn by
# mispronounced_term (the sampler gates per gold value on variant presence).
NO_VARIANT_TOKENS: dict[str, frozenset[str]] = {
    "person_names": frozenset(),
    "medications": frozenset({"Valsartan", "Losartan"}),
    # Phase 3 foreign-token banks (design doc §8b): no hard pins initially.
    # Gasthof (GAHST-hohf) and Harom (HAH-rohm) are vowel-swap-only tokens
    # whose swapped natural rendering IS the token's own spelling
    # ("gasthof", "harom") — deliberately columned without a variant.
    "properties": frozenset({"Gasthof", "Harom"}),
    "vehicles": frozenset(),
}


# Natural-orthography renderer (owner decision, 2026-08-26 audition listens,
# second iteration): the hyphenated caps-stress respelling alphabet misrenders
# in eleven_v3 — caps syllables read as initialisms, hyphens over-articulate,
# and even dehyphenated-lowercase respellings sometimes get spelled out letter
# by letter ("eyebuhr..."). So the SPOKEN form of a mispronounced variant is
# rendered as a plausible normal English word ("iburprofen", "chaym"): each
# respelling unit maps to an ordinary grapheme via the closed table below,
# joined with no hyphens, no caps, no stress mark. The hyphenated caps-stress
# respelling remains the banks' reference notation (correct readings, packet
# review); it never reaches TTS (which is also why stress_shift was retired —
# caps position was its only signal). Every VOWEL_SWAPS pair maps to distinct
# graphemes, so a swap is never erased by the rendering.

NATURAL_VOWEL_GRAPHEMES: dict[str, str] = {
    "a": "a",
    "ah": "o",  # as in "hot"
    "aw": "aw",
    "ow": "ow",
    "ay": "ay",
    "eh": "e",
    "ee": "ee",
    "eye": "i",  # "y" after an onset ("my"), "i" bare ("iburprofen")
    "ih": "i",
    "oh": "o",
    "oo": "oo",
    "oy": "oy",
    "uh": "u",
    "ur": "ur",
    "uu": "u",  # as in "put"
    "ar": "ar",
    "ul": "ul",
    "um": "um",
    "un": "un",
    "ol": "ol",
    "om": "om",
    "on": "on",
}


def render_natural(syllables: list[Syllable]) -> str:
    """Render syllables as a plausible normal English word — the TTS-ready
    spoken form of a mispronounced variant (lowercase plain letters, no
    hyphens, no stress mark). Consonant units are already ordinary graphemes
    and pass through; nuclei map via :data:`NATURAL_VOWEL_GRAPHEMES`."""
    if not syllables:
        raise PronunciationError("Cannot render an empty syllable list")
    parts: list[str] = []
    for syllable in syllables:
        nucleus = NATURAL_VOWEL_GRAPHEMES.get(syllable.nucleus)
        if nucleus is None:
            raise PronunciationError(
                f"No natural grapheme for nucleus {syllable.nucleus!r}"
            )
        if syllable.nucleus == "eye" and syllable.onset:
            nucleus = "y"
        parts.append("".join(syllable.onset) + nucleus + "".join(syllable.coda))
    rendered = "".join(parts)
    if not (rendered.isascii() and rendered.isalpha() and rendered.islower()):
        raise PronunciationError(
            f"Natural rendering is not plain letters: {rendered!r}"
        )
    return rendered


def _spelling_feasible(
    token: str, syllables: list[Syllable], decoy_tokens: Sequence[str]
) -> bool:
    """``spelling_pronunciation`` feasibility, judged in NATURAL-orthography
    space (the spoken form, :func:`render_natural`): the naive reading must
    exist, its spoken form must differ both from the correct reading's
    spoken form and from the token's own spelling (transparent spellings
    like "Metformin" ARE their naive reading, so nothing is mispronounced),
    and it must not collide with any decoy token of the entry (a decoy
    planted in the DB must never become the spoken form)."""
    try:
        naive = render_natural(spelling_syllables(token))
    except PronunciationError:
        return False  # e.g. no vowel letter: no naive reading exists
    if naive == render_natural(syllables) or naive == token.casefold():
        return False
    return naive not in {decoy.casefold() for decoy in decoy_tokens}


def feasible_operators(
    bank: str,
    token: str,
    syllables: list[Syllable],
    decoy_tokens: Sequence[str],
) -> list[DistortionOperator]:
    """The operators this bank/token/respelling supports, in catalog order.

    Bank applicability (:data:`OPERATOR_APPLICABILITY`) gates first, then the
    per-value structural gates. Never empty: ``vowel_swap`` is total over the
    nucleus inventory and applies to every bank.
    """
    applicable = OPERATOR_APPLICABILITY.get(bank)
    if applicable is None:
        raise PronunciationError(f"No operator applicability for bank {bank!r}")
    feasible: list[DistortionOperator] = []
    if (
        DistortionOperator.METATHESIS in applicable
        and _metathesis_target(syllables) is not None
    ):
        feasible.append(DistortionOperator.METATHESIS)
    feasible.append(DistortionOperator.VOWEL_SWAP)
    if (
        DistortionOperator.SYLLABLE_DROP in applicable
        and _droppable_index(syllables) is not None
    ):
        feasible.append(DistortionOperator.SYLLABLE_DROP)
    if DistortionOperator.SPELLING_PRONUNCIATION in applicable and _spelling_feasible(
        token, syllables, decoy_tokens
    ):
        feasible.append(DistortionOperator.SPELLING_PRONUNCIATION)
    return feasible


def apply_operator(
    operator: DistortionOperator, syllables: list[Syllable]
) -> list[Syllable]:
    """Apply one syllable-level distortion operator; deterministic,
    structure-preserving. ``spelling_pronunciation`` is NOT a syllable
    operator — it reads the token spelling, use :func:`spelling_respelling`.
    """
    if operator is DistortionOperator.SPELLING_PRONUNCIATION:
        raise PronunciationError(
            "spelling_pronunciation operates on the token spelling, not the "
            "syllabified respelling — use spelling_respelling(token)"
        )
    syllables = [s.model_copy(deep=True) for s in syllables]
    if operator is DistortionOperator.METATHESIS:
        target = _metathesis_target(syllables)
        if target is None:
            raise PronunciationError("metathesis needs a cluster onset")
        syllable = syllables[target]
        # Swap the last onset consonant with the nucleus: "thro" -> "thor".
        moved = syllable.onset.pop()
        syllable.coda.insert(0, moved)
        return syllables
    if operator is DistortionOperator.VOWEL_SWAP:
        stressed = syllables[_stressed_index(syllables)]
        swapped = VOWEL_SWAPS.get(stressed.nucleus)
        if swapped is None:
            raise PronunciationError(f"No vowel swap for nucleus {stressed.nucleus!r}")
        stressed.nucleus = swapped
        return syllables
    if operator is DistortionOperator.SYLLABLE_DROP:
        index = _droppable_index(syllables)
        if index is None:
            raise PronunciationError("syllable_drop needs an unstressed medial")
        del syllables[index]
        return syllables
    raise PronunciationError(f"Unhandled operator: {operator}")  # pragma: no cover


class MispronunciationDrawer:
    """Balance-greedy operator draw for one bank (design doc §4).

    The retired uniform-among-feasible draw produced badly skewed cells
    (stress_shift 70 / vowel_swap 70 / syllable_drop 30 / metathesis 14):
    an operator that is rarely feasible almost never wins a uniform draw, and
    per-operator ANALYSIS of mispronounced_term outcomes needs non-trivial
    cell sizes. So the drawer visits tokens in the existing seeded build
    order and each token takes the FEASIBLE operator with the lowest running
    count — ties broken by the token's own seeded rng stream, keeping the
    whole draw a pure function of the build seed. Counts run PER BANK, not
    globally: the per-bank applicability sets differ (owner directive
    2026-08-26), so a global count would let one bank's exclusive operator
    starve the other bank's shared ones.
    """

    def __init__(self, bank: str) -> None:
        applicable = OPERATOR_APPLICABILITY.get(bank)
        if applicable is None:
            raise PronunciationError(f"No operator applicability for bank {bank!r}")
        self.bank = bank
        # Catalog order, for deterministic iteration.
        self.counts: dict[DistortionOperator, int] = {
            operator: 0 for operator in DistortionOperator if operator in applicable
        }
        self._pins = HARD_PIN_OPERATORS.get(bank, {})
        self._unconsumed_pins = set(self._pins)

    def draw(
        self,
        token: str,
        syllables: list[Syllable],
        decoy_tokens: Sequence[str],
        rng: random.Random,
    ) -> Optional[tuple[DistortionOperator, str]]:
        """Draw (operator, variant) for one token and advance the counts.

        The variant is the NATURAL-ORTHOGRAPHY spoken form
        (:func:`render_natural` — lowercase plain letters, TTS-ready). It
        must differ both from the natural rendering of the correct reading
        and from the token's own spelling — either equality would make the
        synthesis swap vacuous. Returns None when NO feasible operator
        yields a non-vacuous variant (transparent spellings like "Valsartan"
        whose correct reading is its own spelling and whose swaps land back
        on it): the token then carries no mispronounced variant and is
        simply never drawn by ``mispronounced_term``.
        """
        feasible = feasible_operators(self.bank, token, syllables, decoy_tokens)
        pinned = self._pins.get(token)
        if pinned is not None:
            # Hand-tuned hard pin (HARD_PIN_OPERATORS) beats the balance
            # draw; a pin whose operator stopped being feasible or renders
            # vacuously is stale.
            if pinned not in feasible:
                raise PronunciationError(
                    f"Hard pin {pinned.value} for {self.bank}/{token!r} is not "
                    f"feasible (stale pin after a data or gate change?)"
                )
            variant = self._natural_variant(pinned, token, syllables)
            if variant is None:
                raise PronunciationError(
                    f"Hard pin {pinned.value} for {self.bank}/{token!r} "
                    "renders vacuously in natural orthography (stale pin)"
                )
            self._unconsumed_pins.discard(token)
            operator = pinned
        else:
            # Balance-greedy over the feasible operators, skipping any whose
            # variant renders vacuously in natural orthography (equals the
            # correct reading's spoken form or the token's own spelling).
            candidates = list(feasible)
            variant = None
            while candidates:
                lowest = min(self.counts[op] for op in candidates)
                tied = [op for op in candidates if self.counts[op] == lowest]
                operator = tied[0] if len(tied) == 1 else rng.choice(tied)
                variant = self._natural_variant(operator, token, syllables)
                if variant is not None:
                    break
                candidates.remove(operator)
            if variant is None:
                return None  # no non-vacuous variant: token gets no column
        self.counts[operator] += 1
        return operator, variant

    def _natural_variant(
        self,
        operator: DistortionOperator,
        token: str,
        syllables: list[Syllable],
    ) -> Optional[str]:
        """The operator's variant in natural orthography, or None when it is
        vacuous (spoken form equals the correct reading's spoken form or the
        token's own spelling)."""
        if operator is DistortionOperator.SPELLING_PRONUNCIATION:
            variant_syllables = spelling_syllables(token)
        else:
            variant_syllables = apply_operator(operator, syllables)
        variant = render_natural(variant_syllables)
        if variant == render_natural(syllables) or variant == token.casefold():
            return None
        return variant

    def histogram(self) -> dict[str, int]:
        """Running per-operator counts, catalog-ordered, for manifests."""
        return {operator.value: count for operator, count in self.counts.items()}

    def assert_pins_consumed(self) -> None:
        """Every hand pin for this bank must have matched a drawn token —
        an unconsumed pin means the token left the bank (stale pin)."""
        if self._unconsumed_pins:
            raise PronunciationError(
                f"Hard pins for {self.bank} name tokens no draw produced "
                f"(stale after a bank change?): {sorted(self._unconsumed_pins)}"
            )


__all__ = [
    "ARPABET_CONSONANTS",
    "ARPABET_VOWELS",
    "DistortionOperator",
    "IPA_CONSONANTS",
    "IPA_DIPHTHONGS",
    "IPA_VOWELS",
    "HARD_PIN_OPERATORS",
    "MispronunciationDrawer",
    "NATURAL_VOWEL_GRAPHEMES",
    "NO_VARIANT_TOKENS",
    "OPERATOR_APPLICABILITY",
    "PronunciationError",
    "SPELLING_TWO_LETTER",
    "Syllable",
    "VOWEL_SWAPS",
    "apply_operator",
    "feasible_operators",
    "phones_from_arpabet",
    "phones_from_ipa",
    "render_natural",
    "render_respelling",
    "respell_phonemes",
    "spelling_respelling",
    "spelling_syllables",
    "syllabify",
    "syllabify_phonemes",
]
