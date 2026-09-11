# Copyright Sierra
"""Pre-TTS spelled-entity normalization.

Per-language TTS rendering defects are concentrated in pt and pl: pt letter
names can be mispronounced when spelling IDs
(S voiced as "éfi", N as "Uni"), pt digits (2 as "dô", 5 as "senco"), and pl
user-IDs / digit strings read with Russian/Ukrainian pronunciation. Most
languages render these forms correctly, so this normalization is OPT-IN per pack
(``LocalizationPackConfig.spelled_entity_normalization``) and deliberately
narrow: it expands only entity-shaped tokens and explicit spelling runs into
the pack's native letter/digit/symbol words, immediately before the
ElevenLabs call. The stored transcript (``message.content``) keeps the
original text; only the TTS-bound text (and hence ``audio_script_gold``, the
delivered-text record, mirroring the vocal-tic markup prior art) carries the
expansion.

Match families (both deliberately narrow — ordinary prose numbers, dates,
and amounts are OUT of scope and must pass through byte-identical):

- **Spelling runs**: two or more consecutive single-character tokens
  (UPPERCASE letters or digits) separated by spaces, comma+space,
  period+space, or hyphens ("E, H, G, L, P, 3", "E-H-G-L-P", "M I A").
  A solitary capital never matches (German noun capitalization is safe), and
  lowercase single-letter words (pt "a"/"e"/"o", pl "a"/"i"/"w"/"z") are
  never tokens. Period-separated runs require >= 3 tokens so a sentence
  boundary followed by a capitalized single-letter word ("... portão E. O
  agente ...") is safe; a 2-token all-digit hyphen run ("2-3") is treated as
  a prose range and skipped.
- **Entity-shaped tokens** (the airline DB grammars, matched precisely):
  user ids ``[A-Za-z]+_[A-Za-z]+_\\d+`` (e.g. 'mia_li_3668'), flight numbers
  ``HAT\\d+``, 6-character uppercase alphanumeric reservation codes
  (e.g. 'VAAOXJ', '4WQ150'), and bare ID-like digit runs of >= 3 digits.
  Digit runs skip decimal/thousands-grouped numbers ('1.250', '50,90'),
  currency-adjacent amounts ('R$ 1500', '250 zł'), and bare 4-digit years
  (1900-2099) — those are prose, not IDs.

- **Native letter-name runs**: two or more
  comma-separated tokens the pack already voices as letter names, written by
  the simulator itself ("érre, é, êne, á" for R-E-N-A). This family exists
  because the LLM spells names in native words rather than capitals — in the
  simulator may produce native-name runs without raw capitals, so the two
  families above do not fire. Only pack-listed letter names count as tokens,
  and only inside a run, so the pt
  demonstrative "esse" and the pronoun "ele" (unaccented, and never
  comma-listed beside other letter names) are untouched.

Expansion: letters -> the pack's ``letter_names``, digits -> ``digit_names``
(digit by digit), symbols ('_', '-', '@', '.') -> the pack's existing
``symbol_readouts``; parts joined with comma+space (the spelled
character-by-character comma convention). Output is native lowercase words,
so a second pass finds no raw match — the function is idempotent.

A pack may additionally ship ``letter_name_tts``: a pronunciation-hint
respelling used for the TTS-bound text only, when the orthographic letter
name is voiced with the wrong vowel. In pt, Brazilian Portuguese raises the
European final -e forms ("érrê", "élê", "ênê") to -i ("érri", "éli", "êni"),
so the pack maps
R -> 'érri', S -> 'éssi', and so on. Hints apply to every family: expanded
capitals come out already hinted, and a native-name run is rewritten in
place. The transcript keeps the orthographic form either way.
"""

import re
from functools import lru_cache
from typing import Iterable, Optional

from pydantic import BaseModel, Field

# Currency markers that mark an adjacent digit run as a prose amount, not an
# ID. Symbols are checked immediately before the run (ignoring spaces); words
# are checked as the token immediately after it. A fixed in-code closed set —
# extend it here, never at run time.
_CURRENCY_SYMBOLS = ("R$", "US$", "$", "€", "£", "zł")
_CURRENCY_WORDS = frozenset(
    {
        "reais",
        "real",
        "zł",
        "zl",
        "złote",
        "złotych",
        "euro",
        "euros",
        "eur",
        "usd",
        "brl",
        "pln",
        "dólares",
        "dolarów",
        "lira",
        "lirası",
        "tl",
    }
)

# The airline DB entity grammars (see data/tau2/domains/airline/db.json):
# user ids like 'mia_li_3668', flight numbers like 'HAT123', reservation
# codes like 'VAAOXJ' / '4WQ150'. Alternation order matters: at the same
# start position the more specific shape wins (HAT123 is a flight number,
# not a reservation code).
_USER_ID = r"[A-Za-z]+_[A-Za-z]+_\d+"
_FLIGHT_NUMBER = r"HAT\d+"
# At least one letter: a pure 6-digit token is classified as a digit run so
# the prose-number guards (currency, grouping) apply to it.
_RESERVATION_CODE = r"(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}"
_SPELLING_RUN = r"[A-Z0-9](?:(?:,\s+|\.\s+|\s+|-)[A-Z0-9])+"
_DIGIT_RUN = r"\d{3,}"

_ENTITY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    rf"(?:(?P<user_id>{_USER_ID})"
    rf"|(?P<flight>{_FLIGHT_NUMBER})"
    rf"|(?P<reservation>{_RESERVATION_CODE})"
    rf"|(?P<spelling_run>{_SPELLING_RUN})"
    rf"|(?P<digit_run>{_DIGIT_RUN}))"
    r"(?![A-Za-z0-9_])"
)

_RUN_TOKEN = re.compile(r"[A-Z0-9]")


class SpelledEntityTables(BaseModel):
    """The native word tables a pack provides for spelled-entity expansion.

    Built from a pack's ``localization`` block (see
    ``LocalizationPackConfig.spelled_entity_tables``); consumers get an
    instance ONLY when the pack opted in, so holding one implies
    normalization applies.
    """

    letter_names: dict[str, str] = Field(
        description="Native spoken letter names, keyed by UPPERCASE letter "
        "A-Z (complete table; e.g. pt 'S' -> 'ésse')."
    )
    digit_names: dict[str, str] = Field(
        description="Native spoken digit words, keyed by digit '0'-'9' "
        "(complete table; e.g. pl '3' -> 'trzy')."
    )
    symbol_names: dict[str, str] = Field(
        description="Native readout per technical symbol ('_', '-', '@', "
        "'.'; the most natural form from the pack's symbol_readouts)."
    )
    letter_name_tts: dict[str, str] = Field(
        default_factory=dict,
        description="Pronunciation-hint respelling per UPPERCASE letter, used "
        "in the TTS-bound text in place of the orthographic letter name "
        "(e.g. pt 'S' -> 'éssi' so the final vowel is voiced Brazilian, not "
        "European). Partial by design: only the letters the TTS gets wrong.",
    )

    def spoken_letter(self, letter: str) -> Optional[str]:
        """The word the TTS should receive for ``letter``, hint-first."""
        upper = letter.upper()
        return self.letter_name_tts.get(upper) or self.letter_names.get(upper)


def normalize_spelled_entities(text: str, tables: SpelledEntityTables) -> str:
    """Expand spelled entities in TTS-bound text into native words.

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
        if match.group("spelling_run") is not None:
            if _skip_spelling_run(raw):
                return raw
            expanded = _expand_characters(_RUN_TOKEN.findall(raw), tables)
        elif match.group("digit_run") is not None:
            if _skip_digit_run(match):
                return raw
            expanded = _expand_characters(raw, tables)
        else:
            expanded = _expand_characters(raw, tables)
        return expanded if expanded is not None else raw

    text = _ENTITY_PATTERN.sub(replace, text)
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
    capital-based families never see it. Rewriting is confined to runs of
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
        if ch.isdigit():
            name = tables.digit_names.get(ch)
        elif ch.isalpha():
            name = tables.spoken_letter(ch)
        else:
            name = tables.symbol_names.get(ch)
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


def _skip_digit_run(match: re.Match) -> bool:
    """Guards keeping prose numbers out of the ID-like digit-run family."""
    text = match.string
    start, end = match.start(), match.end()
    run = match.group(0)
    # Decimal / thousands-grouping adjacency: '1.250', '50,90'.
    if start >= 2 and text[start - 1] in ".," and text[start - 2].isdigit():
        return True
    if end + 1 < len(text) and text[end] in ".," and text[end + 1].isdigit():
        return True
    # Space-grouped thousands (pl '1 250', fr '1 250 000'): a 3-digit group
    # preceded by digit+space, or any run followed by space + exactly 3
    # digits, is a prose number, not an ID.
    spaces = "  "
    if (
        len(run) == 3
        and start >= 2
        and text[start - 1] in spaces
        and text[start - 2].isdigit()
    ):
        return True
    if (
        end + 3 < len(text)
        and text[end] in spaces
        and text[end + 1 : end + 4].isdigit()
        and (end + 4 == len(text) or not text[end + 4].isdigit())
    ):
        return True
    # Currency symbol immediately before (ignoring spaces): 'R$ 1500'.
    prefix = text[:start].rstrip()
    if prefix.endswith(_CURRENCY_SYMBOLS):
        return True
    # Currency word immediately after: '250 zł'.
    following = text[end:].lstrip()
    next_word = re.match(r"[^\s.,;:!?]+", following)
    if next_word and next_word.group(0).lower() in _CURRENCY_WORDS:
        return True
    # Bare 4-digit year: 'em 2026'.
    if len(run) == 4 and 1900 <= int(run) <= 2099:
        return True
    return False
