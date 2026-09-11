# Copyright Sierra
"""How long is a short phrase, measured comparably across languages.

A persona carries two SHORT-PHRASE INVENTORIES that the runtime draws from
uniformly at random with no context: out-of-turn speech
(``non_directed_phrases``) and backchannel continuers
(``backchannel_phrases``). Both are bounded the same way — a maximum number
of entries and a maximum length per entry — and both bounds are declared
once here as a :class:`PhraseInventory`, so the factory guardrails, the pack
invariant suite and the redraft verbs all check the same thing.

Lengths are stated in WORDS, and a word is only directly countable in
languages whose orthography puts spaces between words. Languages that differ
declare a conversion in ``WORD_MEASURES`` — space-less scripts (Han, Thai)
have no token boundary at all, so the budget is converted to CHARACTERS; a
syllable-per-token orthography would instead discount its tokens.

The factors are approximations and are meant to be: the point is a bound that
discriminates "a beat" from "a sentence" in every pack, not a tokenizer.
Punctuation and whitespace are excluded from character counts so the measure
tracks spoken material rather than how many commas the drafter used.
"""

import math
import re
from typing import Literal, NamedTuple, Optional

from tau2.backchannel import MAX_BACKCHANNEL_PHRASE_WORDS, MAX_BACKCHANNEL_PHRASES
from tau2.voice_config import (
    MAX_NON_DIRECTED_PHRASE_WORDS,
    MAX_NON_DIRECTED_PHRASES,
)


class WordMeasure(NamedTuple):
    """How to convert a language's countable units into spoken words."""

    unit: Literal["token", "char"]
    units_per_word: float


# Whitespace-delimited tokens, one per word — true for most of the benchmark.
DEFAULT_WORD_MEASURE = WordMeasure(unit="token", units_per_word=1.0)

# Languages whose countable units are NOT one-per-word. Han words average
# ~1.5-2 characters; Thai (a worked example for future space-less onboarding)
# runs longer still.
WORD_MEASURES: dict[str, WordMeasure] = {
    "zh": WordMeasure(unit="char", units_per_word=2),
    "th": WordMeasure(unit="char", units_per_word=4),
}

# ISO 15924 script codes written without inter-word spaces. A language in one
# of these MUST declare a character-based measure above — counting whitespace
# tokens there would score a whole sentence as one word.
SPACELESS_SCRIPTS = frozenset({"hans", "hant", "jpan", "thai"})

# Everything that is not spoken material: whitespace, and punctuation in ASCII
# plus the CJK / Arabic / Devanagari forms the packs actually use.
_NON_SPOKEN_RE = re.compile(r"[\s!-/:-@\[-`{-~¡¿‐-›、-〿！-･،؛؟।॥]")


def word_measure(language: str) -> WordMeasure:
    """The unit conversion for a language (default: one token per word)."""
    return WORD_MEASURES.get(language, DEFAULT_WORD_MEASURE)


def phrase_word_count(text: str, language: str) -> int:
    """Approximate spoken-word length of a short phrase.

    Rounded UP, so a phrase never measures shorter than it sounds.
    """
    measure = word_measure(language)
    if measure.unit == "char":
        units = len(_NON_SPOKEN_RE.sub("", text))
    else:
        units = len(text.split())
    return math.ceil(units / measure.units_per_word)


class PhraseInventory(NamedTuple):
    """One bounded short-phrase inventory on a persona.

    Both inventories are drawn from uniformly at random with no context, so
    both need a count bound (a long list dilutes the good entries and lets
    anything land anywhere) and a length bound (a long entry stops sounding
    like the thing it is meant to be). ``why_bounded`` explains the count to
    a reader hitting the guardrail; ``what_it_is`` names what a too-long
    entry stops being.
    """

    field: str
    max_phrases: int
    max_words: int
    why_bounded: str
    what_it_is: str


NON_DIRECTED_INVENTORY = PhraseInventory(
    field="non_directed_phrases",
    max_phrases=MAX_NON_DIRECTED_PHRASES,
    max_words=MAX_NON_DIRECTED_PHRASE_WORDS,
    why_bounded=(
        "the phrase is drawn uniformly at random with no context, so a longer "
        "inventory buys variety nobody hears and dilutes the clipped entries"
    ),
    what_it_is=(
        "Out-of-turn speech is a glance away from the phone, not a second conversation"
    ),
)

BACKCHANNEL_INVENTORY = PhraseInventory(
    field="backchannel_phrases",
    max_phrases=MAX_BACKCHANNEL_PHRASES,
    max_words=MAX_BACKCHANNEL_PHRASE_WORDS,
    why_bounded=(
        "each must be a PURE continuer (the 'mm-hmm'/'uh-huh' equivalent), "
        "because the phrase is drawn uniformly at random with no context. "
        "Acknowledgments belong in the localization block's "
        "conversational_confirmations palette, where the simulator picks them "
        "IN CONTEXT at turn level. A language with no pure continuer in its "
        "authored material gets one from `tau2 factory draft-continuers`, "
        "which authors it through a fixed prompt for its native reviewer — "
        "never invent it here"
    ),
    what_it_is=(
        "A continuer is a brief hum, not an utterance: at that length it is a "
        "turn-level response landing at a random moment"
    ),
)


def phrase_budget_label(language: str, inventory: PhraseInventory) -> str:
    """How the length budget reads for this language, for error messages."""
    measure = word_measure(language)
    if measure == DEFAULT_WORD_MEASURE:
        return f"{inventory.max_words} words"
    budget = inventory.max_words * measure.units_per_word
    noun = "spoken characters" if measure.unit == "char" else "syllables"
    return f"{inventory.max_words} words (~{budget:g} {noun} in '{language}')"


def phrase_counting_rule(language: str, inventory: PhraseInventory) -> str:
    """How the checker counts, spelled out for the drafting prompts.

    A model that knows the exact rule can edit against the bound instead of
    guessing at "short" — and can see that punctuation buys it nothing.
    """
    measure = word_measure(language)
    budget = inventory.max_words * measure.units_per_word
    if measure.unit == "char":
        return (
            "a phrase is measured in CHARACTERS with all punctuation and "
            f"whitespace removed, and the ceiling is {budget:g} characters "
            f"(~{inventory.max_words} spoken words in '{language}')"
        )
    if measure.units_per_word != 1:
        return (
            "a phrase is measured in whitespace-separated tokens — in "
            f"'{language}' each SYLLABLE is its own token — and the ceiling "
            f"is {budget:g} tokens (~{inventory.max_words} spoken words)"
        )
    return (
        "a phrase is measured in whitespace-separated tokens, and the ceiling "
        f"is {budget:g}. Punctuation attached to a token does not split it or "
        "discount it, so commas buy nothing"
    )


def phrase_inventory_problems(
    persona_id: str,
    phrases: object,
    *,
    inventory: PhraseInventory,
    language: str,
    script: Optional[str] = None,
) -> list[str]:
    """Bound violations in one persona's short-phrase inventory.

    Returns human-readable problem strings (empty == pass) — the shape the
    factory guardrails, the pack invariant suite and the redraft verbs all
    consume. Purity/naturalness is not checkable here and is not attempted:
    this is the deterministic half, and the redraft verbs pair it with a
    fixed prompt and a named human reviewer.
    """
    problems: list[str] = []

    # A space-less script measured in whitespace tokens would score any
    # sentence as one "word", silently disabling the bound. Fail loudly so a
    # newly onboarded language gets a deliberate factor instead.
    if script in SPACELESS_SCRIPTS and word_measure(language).unit != "char":
        problems.append(
            f"language '{language}' is written in the space-less script "
            f"'{script}' but declares no character-based measure — an "
            "engineer must add one to "
            "tau2.multilingual.brevity.WORD_MEASURES before its "
            f"{inventory.field} can be bounded"
        )
        return problems

    if phrases is None:
        return problems  # falls back to the English defaults, which are in budget
    if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
        return [f"persona '{persona_id}' {inventory.field} must be a list of strings"]

    if len(phrases) > inventory.max_phrases:
        problems.append(
            f"persona '{persona_id}' carries {len(phrases)} {inventory.field} "
            f"({phrases}); at most {inventory.max_phrases} are allowed — "
            f"{inventory.why_bounded}."
        )
    for phrase in phrases:
        count = phrase_word_count(phrase, language)
        if count > inventory.max_words:
            problems.append(
                f"persona '{persona_id}' {inventory.field} entry {phrase!r} "
                f"runs ~{count} words; the budget is "
                f"{phrase_budget_label(language, inventory)}. "
                f"{inventory.what_it_is}."
            )
    return problems


def non_directed_phrase_problems(
    persona_id: str,
    phrases: object,
    *,
    language: str,
    script: Optional[str] = None,
) -> list[str]:
    """Bound violations in one persona's ``non_directed_phrases``."""
    return phrase_inventory_problems(
        persona_id,
        phrases,
        inventory=NON_DIRECTED_INVENTORY,
        language=language,
        script=script,
    )


def backchannel_phrase_problems(
    persona_id: str,
    phrases: object,
    *,
    language: str,
    script: Optional[str] = None,
) -> list[str]:
    """Bound violations in one persona's ``backchannel_phrases``."""
    return phrase_inventory_problems(
        persona_id,
        phrases,
        inventory=BACKCHANNEL_INVENTORY,
        language=language,
        script=script,
    )
