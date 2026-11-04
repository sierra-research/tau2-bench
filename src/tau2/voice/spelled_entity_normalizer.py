# Copyright Sierra
"""Pronunciation repair for explicit spelling in TTS input.

PT-BR synthesis consistently mispronounces some letter names. A published
persona opts in through RuntimePersona.spelled_entity_tables; the runner copies
those tables into SynthesisConfig. This deterministic transformation affects
only the TTS input and audio_script_gold. Original message content is preserved.

Two forms are recognized:
- Separated uppercase letters/digits, such as "R, E, N, A", "E-H-G-L-P" or
  "4, 8, 1, 2, 3". Period-separated runs require at least three characters;
  two-digit hyphen ranges such as "2-3" remain unchanged.
- Comma-separated native letter/digit names, such as "érre, é, êne, á".
  Only reviewed letter_name_tts substitutions are applied to these runs.

Unsplit identifiers, ordinary numbers, amounts and dates remain unchanged.
This function does not decide when callers should spell: the simulator's
requested-spelling instructions own that behavior. No LLM is called here.
"""

import re
from functools import lru_cache
from typing import Iterable, Optional

from tau2.data_model.voice import SpelledEntityTables

# Each token must already be a separately written character. Boundaries exclude
# fragments of ordinary words and hyphenated identifiers such as "REF-A-B".
_SPELLING_PATTERN = re.compile(
    r"(?<![\w-])[A-Z0-9](?:(?:,\s+|\.\s+|\s+|-)[A-Z0-9])+(?![\w-])"
)

_RUN_TOKEN = re.compile(r"[A-Z0-9]")


def normalize_spelled_entities(text: str, tables: SpelledEntityTables) -> str:
    """Normalize explicit spelling runs in TTS-bound text into native words.

    Pure and idempotent: applies only the narrow match families documented in
    the module docstring; everything else (prose numbers, dates, amounts,
    solitary capitals) passes through byte-identical. A token containing a
    character with no table entry is left untouched rather than half-expanded.

    Args:
        text: The TTS-bound text (post speech-insert markup).
        tables: The opted-in pack's native word tables.

    Returns:
        The text with matched entities expanded; unchanged when nothing
        matches.
    """

    def replace(match: re.Match) -> str:
        raw = match.group(0)
        if _skip_spelling_run(raw):
            return raw
        expanded = _expand_characters(_RUN_TOKEN.findall(raw), tables)
        return expanded if expanded is not None else raw

    text = _SPELLING_PATTERN.sub(replace, text)
    return _hint_native_letter_runs(text, tables)


@lru_cache(maxsize=8)
def _native_run_pattern(vocabulary: tuple[str, ...]) -> re.Pattern:
    """A regex matching two or more comma-separated spelled-out words.

    ``vocabulary`` is the pack's letter and digit words, longest first so an
    alternation never stops at a prefix. Cached per pack (a handful of
    languages opt in), since the alternation is rebuilt on every TTS call
    otherwise.
    """
    alternation = "|".join(re.escape(word) for word in vocabulary)
    token = rf"(?<!\w)(?:{alternation})(?!\w)"
    return re.compile(rf"{token}(?:\s*,\s*{token})+", re.IGNORECASE)


def _hint_native_letter_runs(text: str, tables: SpelledEntityTables) -> str:
    """Respell hinted letter names inside already-native spelling runs.

    The simulator writes the spelling out itself ("érre, é, êne, á"), so the
    separated-character pattern never sees it. Rewriting is confined to runs of
    two or more pack-listed letter/digit words, and only letters the pack
    gives a hint for change — an accidental match on prose is a no-op, and
    hint forms are never themselves letter names, so the pass is idempotent.
    """
    if not tables.letter_name_tts:
        return text
    hints = {
        name.casefold(): tables.letter_name_tts[letter]
        for letter, name in tables.letter_names.items()
        if letter in tables.letter_name_tts
    }
    vocabulary = tuple(
        sorted(
            {*tables.letter_names.values(), *tables.digit_names.values()},
            key=len,
            reverse=True,
        )
    )
    if not vocabulary:
        return text

    def rewrite(match: re.Match) -> str:
        parts = re.split(r"(\s*,\s*)", match.group(0))
        return "".join(
            part if index % 2 else _hinted(part, hints)
            for index, part in enumerate(parts)
        )

    return _native_run_pattern(vocabulary).sub(rewrite, text)


def _hinted(word: str, hints: dict[str, str]) -> str:
    """The hint for ``word``, keeping a sentence-initial capital if any."""
    hint = hints.get(word.casefold())
    if hint is None:
        return word
    return hint[0].upper() + hint[1:] if word[:1].isupper() else hint


def _expand_characters(
    characters: Iterable[str], tables: SpelledEntityTables
) -> Optional[str]:
    """Comma-join the native words for a character sequence, or None.

    None (leave the token untouched) when any character has no table entry —
    a half-expanded token would be worse than the raw one.
    """
    parts: list[str] = []
    for ch in characters:
        name = tables.digit_names.get(ch) if ch.isdigit() else tables.spoken_letter(ch)
        if name is None:
            return None
        parts.append(name)
    return ", ".join(parts)


def _skip_spelling_run(run: str) -> bool:
    """Guards keeping prose out of the spelling-run family."""
    tokens = _RUN_TOKEN.findall(run)
    # Sentence-boundary guard: "portão E. O agente" is prose, not spelling.
    if "." in run and len(tokens) < 3:
        return True
    # Prose-range guard: "2-3" (days, hours) is a range, not a code.
    if len(tokens) == 2 and "-" in run and all(t.isdigit() for t in tokens):
        return True
    return False
