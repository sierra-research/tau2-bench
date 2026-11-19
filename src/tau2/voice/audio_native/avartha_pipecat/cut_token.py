"""The spoken part of a word a caller cut caught mid-sound.

When a cut catches the agent mid-word before the word's end was reported,
the adapter shows that started word on the cut tick (see
``_HeardTranscript.interrupt``). An ordinary word lasts a fraction of a
second, so the caller heard most of it. A code, an identifier or a long
digit string is one written word the TTS reads unit by unit ("6941833" as
seven digits, "ivan_rossi_8555" as "ivan", "underscore", "rossi", ...) for
several seconds, and a cut can stop it after any unit. Showing such a token
whole would show the caller units it never heard, so only the units whose
sound surely played are shown: the time played over a unit's duration, with
a safety margin, rounded down.

- ``split_token``: the token's kind and its units;
- ``units_heard``: how many units surely played.
"""

from __future__ import annotations

import enum
import math
import re
import unicodedata
from dataclasses import dataclass

# A word with a digit and at least this many letters and digits is read unit
# by unit (a code, a digit string), as is a word of at least this many ASCII
# capitals only ("IFOYYZ", which the TTS spells: 1.2-2.3 s median on the final
# runs' completed words; a rare all-caps name such as "GARCIA" then shows
# fewer letters, erring on fewer). A plain number of 3 or 4 digits ("930",
# "1177") is read as number words.
CODE_MIN_UNITS = 5
# An ordinary word is shown whole if it has at most this many letters and
# digits (an ideograph or Hangul syllable counts one); a longer one lasts too
# long for the caller to have heard most of it, so none of it is shown.
ORDINARY_MAX_CHARS = 10
# A unit's duration, from the previous unit's end to its end, when the call
# has no completed code to measure it on. Language medians on recorded
# calls' completed codes that measure a unit (measures_unit): en 383,
# es 427, pt 383, hi 410, zh 308, ko 288 ms.
DEFAULT_UNIT_MS = 400.0
# The shortest unit a call's measurement may set, so one mis-timed completed
# code cannot shrink it: well under the lowest language median (ko 288 ms).
UNIT_MS_FLOOR = 150.0
# A unit counts as heard only once the code's sound covers it this many times
# over. A completed code's duration over (units x the call's median unit) was
# 0.70 / 1.00 / 1.45 at p10 / p50 / p90 (ko 0.38 / 1.00 / 1.83) and 1.8-2.7
# at p98 (ko 5.2), on the completed codes measures_unit accepts. Plain
# rounding down showed the caller a unit it had not heard in 8.2-18.3 code
# cuts per language; this margin leaves an estimated 0.1-1.5 per language,
# worst in ko (a Monte Carlo estimate on recorded calls' cut codes).
UNIT_SAFETY = 2.3

# Inside an identifier these are read as words ("underscore", "at", "dot").
_IDENTIFIER_SEPARATORS = "_@.-"
# A dot between letters or digits ("hao.h18836", "li.wei1234").
_INNER_DOT = re.compile(r"[^\W_]\.[^\W_]")
# A dotted abbreviation, read as words ("p.m.", "e.g.", "U.S."): two runs of
# at most 2 letters, and a dot after them (the word's suffix). "A.B.C.D.E" and
# "j.li" without a closing dot are identifiers ("jo.li." at a sentence end is
# not told apart from "p.m.", so it is shown whole).
_DOTTED_ABBREVIATION = re.compile(r"[^\W\d_]{1,2}\.[^\W\d_]{1,2}")
# Tokens that measure a unit (SplitToken.measures_unit): capitals and digits,
# a lower-case name and 4+ digits; not one ending in a round number (3+
# trailing zeros: "10000", "AB1000") or a number and a 1-3 letter unit
# ("128GB"), which are read as words.
_CAPS_AND_DIGITS = re.compile(r"[A-Z0-9]{5,}")
_NAME_AND_DIGITS = re.compile(r"[a-z]+\d{4,}")
_ROUND_NUMBER = re.compile(r"000$")
_NUMBER_AND_UNIT = re.compile(r"\d+[A-Za-z]{1,3}")
_CURRENCY = re.compile(r"[$€£¥₹₩원]")
# A digit grouping or decimal mark between digits, or a currency sign
# anywhere in a word with digits: an amount or a number read as number words
# ("1,387", "2.420", "5,27,520", "R$137,30").
_GROUPED_NUMBER = re.compile(r"\d[.,]\d")


class TokenKind(enum.Enum):
    # A short word the caller heard most of when its sound started: shown whole.
    ORDINARY = "ordinary"
    # A word too long for that: none of it is shown.
    LONG_WORD = "long_word"
    # Read unit by unit; only the units whose sound played are shown.
    CODE = "code"  # a spelled code (letters and digits) or a digit string
    IDENTIFIER = "identifier"  # name parts, separators and digits (_ or @)
    # An amount or a number read as number words: its digits are no units of
    # its sound, so none of it is shown.
    NUMBER_WORDS = "number_words"
    # Read unit by unit, but in characters the unit model does not read
    # ("HAT138's"): no units can be counted, so none of it is shown.
    UNPARSED = "unparsed"


@dataclass(frozen=True)
class SplitToken:
    """A started word: ``prefix + "".join(units) + suffix`` is the word.

    ``prefix`` is the space and punctuation before it, ``suffix`` the
    punctuation after it."""

    kind: TokenKind
    prefix: str
    units: tuple[str, ...]
    suffix: str

    @property
    def read_by_units(self) -> bool:
        return self.kind in (TokenKind.CODE, TokenKind.IDENTIFIER)

    @property
    def measures_unit(self) -> bool:
        """Whether a completed token measures the call's unit duration.

        Stricter than display. Display errs toward reading a token by units,
        which shows fewer; but a word read as words ("24-hour", "750ml",
        "10000", "NEVER") measured as units makes the unit short, and a later
        code then shows more than played. So only these measure, with at
        least CODE_MIN_UNITS units:
        - ASCII capitals and digits with a digit ("6941833", "HAT110",
          "3RK2T9"), except one ending in a round number ("10000", "HAT000",
          "AB1000") and a number with a short unit ("128GB", "1500W");
        - an identifier with _ or @ ("ivan_rossi_8555");
        - a lower-case name and at least 4 digits ("garcia5901"; a title-case
          "Garcia5901" does not measure, failing closed).
        Accepted residual: a capitals-and-digits name read partly as a word
        ("COVID19", "RTX4090"), or one with a round number inside ("H1000X"),
        still measures; it is rare, and the call's median and UNIT_MS_FLOOR
        bound its effect."""

        if not self.read_by_units or len(self.units) < CODE_MIN_UNITS:
            return False
        body = "".join(self.units)
        if "_" in body or "@" in body or _NAME_AND_DIGITS.fullmatch(body):
            return True
        return bool(
            _CAPS_AND_DIGITS.fullmatch(body)
            and any(ch.isdigit() for ch in body)
            and not _ROUND_NUMBER.search(body)
            and not _NUMBER_AND_UNIT.fullmatch(body)
        )

    def shown(self, heard_units: int) -> str:
        """The text to show when ``heard_units`` units played: a short word
        whole; of a code the units heard; nothing otherwise."""

        if self.kind is TokenKind.ORDINARY:
            return self.prefix + "".join(self.units) + self.suffix
        if not self.read_by_units or heard_units <= 0:
            return ""
        if heard_units >= len(self.units):
            return self.prefix + "".join(self.units) + self.suffix
        # A hyphen after the last unit heard leads to one not heard.
        return self.prefix + "".join(self.units[:heard_units]).rstrip("-")


def _code_units(body: str) -> list[str]:
    """Units of a code: each digit and capital (ASCII or full-width), any other
    letter run as one ("i3668": "i", "3", ...); a hyphen goes with the unit
    before it."""

    units: list[str] = []
    for match in re.finditer(r"[A-ZＡ-Ｚ]|[^\W\dA-ZＡ-Ｚ_]+|\d|-", body):
        if match.group() == "-" and units:
            units[-1] += "-"
        elif match.group() != "-":
            units.append(match.group())
    return units


def _identifier_units(body: str) -> list[str]:
    """Units of an identifier: each letter run, separator and digit."""

    separators = re.escape(_IDENTIFIER_SEPARATORS)
    return re.findall(r"[^\W\d_]+|\d|[" + separators + "]", body)


def _in_word(ch: str) -> bool:
    return ch.isalnum() or unicodedata.category(ch).startswith("M")


def _word(prefix: str, body: str, suffix: str) -> SplitToken:
    """A word not read unit by unit: short (shown whole) or long."""

    short = sum(ch.isalnum() for ch in body) <= ORDINARY_MAX_CHARS
    kind = TokenKind.ORDINARY if short else TokenKind.LONG_WORD
    return SplitToken(kind, prefix, (body,), suffix)


def split_token(word: str) -> SplitToken:
    """Split the started word ``word`` (as ``_next_word`` returns it).

    Fail-safe: a word with ``_`` or ``@``, a dot between letters or digits,
    or a digit among at least ``CODE_MIN_UNITS`` letters and digits is never
    shown whole; if the unit model cannot parse it, none of it is shown.
    (_next_word ends a word at a script change, so a particle after a code,
    ko "HAT138편은" or hi "HAT138का", is not part of it; an attached
    possessive, "HAT138's", is, and is not parsed.)"""

    first = next((i for i, ch in enumerate(word) if ch.isalnum()), len(word))
    # A combining mark (a Devanagari vowel sign) belongs to the word.
    last = next((i for i in range(len(word) - 1, -1, -1) if _in_word(word[i])), -1)
    prefix, body, suffix = word[:first], word[first : last + 1], word[last + 1 :]
    digits = sum(ch.isdigit() for ch in body)
    if digits and (_CURRENCY.search(prefix + body) or _GROUPED_NUMBER.search(body)):
        # An amount ("R$137,30", "$ 1,387") or a grouped number ("2.420").
        return SplitToken(TokenKind.NUMBER_WORDS, prefix, (body,), suffix)
    # Capitals only, ASCII or full-width ("ＩＦＯＹＹＺ"), are spelled (display
    # only: they never measure the unit).
    if re.fullmatch(r"[A-ZＡ-Ｚ]+", body) and len(body) >= CODE_MIN_UNITS:
        return SplitToken(TokenKind.CODE, prefix, tuple(body), suffix)
    alnum = sum(ch.isalnum() for ch in body)
    if not digits and _DOTTED_ABBREVIATION.fullmatch(body) and suffix.startswith("."):
        return _word(prefix, body, suffix)  # "p.m.", "e.g.", "U.S.": words
    identifier = "_" in body or "@" in body or bool(_INNER_DOT.search(body))
    if not identifier and not (digits and alnum >= CODE_MIN_UNITS):
        if digits >= 3 and re.fullmatch(r"\d[\d-]*", body):
            return SplitToken(TokenKind.NUMBER_WORDS, prefix, (body,), suffix)
        return _word(prefix, body, suffix)
    if identifier:
        kind, units = TokenKind.IDENTIFIER, _identifier_units(body)
    else:
        kind, units = TokenKind.CODE, _code_units(body)
    if "".join(units) != body:
        return SplitToken(TokenKind.UNPARSED, prefix, (body,), suffix)
    return SplitToken(kind, prefix, tuple(units), suffix)


def units_heard(played_ms: float, unit_ms: float) -> int:
    """Units of a token whose sound surely played in ``played_ms`` from its
    start: the time over ``unit_ms`` times ``UNIT_SAFETY``, rounded down."""

    return math.floor(played_ms / (unit_ms * UNIT_SAFETY))
