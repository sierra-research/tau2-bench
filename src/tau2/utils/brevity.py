# Copyright Sierra
"""How long is a short phrase, measured comparably across languages.

A persona's voice section carries two SHORT-PHRASE INVENTORIES that the runtime
draws from uniformly at random with no context: out-of-turn speech
(``non_directed_phrases``) and backchannel continuers (``backchannel_phrases``).
Both are bounded by a maximum number of entries and a maximum length per entry.

Lengths are stated in WORDS, and a word is only directly countable in languages
whose orthography puts spaces between words. Space-less scripts declare a
character-based conversion in ``WORD_MEASURES``. The factors are deliberately
approximate: the point is a bound that discriminates "a beat" from "a sentence".
Punctuation and whitespace are excluded from character counts.
"""

import math
import re
from typing import Literal, NamedTuple

from tau2.backchannel import MAX_BACKCHANNEL_PHRASE_WORDS, MAX_BACKCHANNEL_PHRASES

# Out-of-turn speech is a glance away from the phone, not a second conversation.
MAX_NON_DIRECTED_PHRASES = 4
MAX_NON_DIRECTED_PHRASE_WORDS = 4


class WordMeasure(NamedTuple):
    """How to convert a language's countable units into spoken words."""

    unit: Literal["token", "char"]
    units_per_word: float


DEFAULT_WORD_MEASURE = WordMeasure(unit="token", units_per_word=1.0)

# Languages whose countable units are NOT one-per-word.
WORD_MEASURES: dict[str, WordMeasure] = {
    "zh": WordMeasure(unit="char", units_per_word=2),
    "th": WordMeasure(unit="char", units_per_word=4),
}

_NON_SPOKEN_RE = re.compile(r"[\s!-/:-@\[-`{-~¡¿‐-›、-〿！-･،؛؟।॥]")


def word_measure(language: str) -> WordMeasure:
    """The unit conversion for a language (default: one token per word)."""
    return WORD_MEASURES.get(language.split("-")[0], DEFAULT_WORD_MEASURE)


def phrase_word_count(text: str, language: str) -> int:
    """Approximate spoken-word length of a short phrase, rounded up."""
    measure = word_measure(language)
    if measure.unit == "char":
        units = len(_NON_SPOKEN_RE.sub("", text))
    else:
        units = len(text.split())
    return math.ceil(units / measure.units_per_word)


class PhraseInventory(NamedTuple):
    """One bounded short-phrase inventory on a persona voice section."""

    field: str
    max_phrases: int
    max_words: int


NON_DIRECTED_INVENTORY = PhraseInventory(
    field="non_directed_phrases",
    max_phrases=MAX_NON_DIRECTED_PHRASES,
    max_words=MAX_NON_DIRECTED_PHRASE_WORDS,
)
BACKCHANNEL_INVENTORY = PhraseInventory(
    field="backchannel_phrases",
    max_phrases=MAX_BACKCHANNEL_PHRASES,
    max_words=MAX_BACKCHANNEL_PHRASE_WORDS,
)


def phrase_inventory_problems(
    owner: str, phrases: list[str], *, inventory: PhraseInventory, language: str
) -> list[str]:
    """Human-readable bound violations; empty means the inventory passes."""
    problems = []
    if len(phrases) > inventory.max_phrases:
        problems.append(
            f"{owner} carries {len(phrases)} {inventory.field}; at most "
            f"{inventory.max_phrases} are allowed because entries are drawn at "
            "random with no context"
        )
    for phrase in phrases:
        count = phrase_word_count(phrase, language)
        if count > inventory.max_words:
            problems.append(
                f"{owner} {inventory.field} entry {phrase!r} runs ~{count} words; "
                f"the budget is {inventory.max_words} words"
            )
    return problems
