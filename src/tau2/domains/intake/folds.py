# Copyright Sierra
"""Published per-entity-type fold rules for the intake domain (design doc §7).

Reward in this domain is final-DB hash equality against a golden replay, so
two surface forms of the *same captured value* must enter the DB as the same
byte string. The seam is **canonicalization at write time**: ``submit_intake``
folds every field into its canonical stored form before it lands in the DB. An orthography coin-flip ("maria lopez",
lowercase VINs, "555 123 2002" vs "555-123-2002") must not fail the reward; a
real capture error (digit swap, wrong record, accepted decoy) must.

The canonical form is **purely syntactic**: computable from the writer's input
alone, deterministic, and collision-free over the entity banks (truth never
folds together with any of its decoys — asserted exhaustively in
``tests/test_domains/test_intake/test_folds_intake.py``). The tool cannot know
the pinned truth at runtime, so no rule may depend on it; in particular there
is no "restore bank casing" step — where casing cannot be reconstructed
syntactically (names), the stored form IS the folded form.

Fold rules per entity type:

- ``PHONE`` — the bare digit sequence, with the two purely syntactic dialing
  prefixes normalized away: a ``+`` marker and the international ``00``
  prefix are dropped (``+44…`` == ``0044…`` == ``44…``), and an 11-digit
  sequence starting with ``1`` drops the NANP country code
  (``+1 (206) 555-0111`` == ``(206) 555-0111``). Grouping, separators, and
  extension markers ("ext", "x") carry no identity; every other digit —
  including any remaining leading zero — does.
  ``555 123 2002`` == ``555-123-2002`` -> ``5551232002``;
  ``+44 20 7946 0958`` -> ``442079460958``.
- ``CODE`` — uppercase alphanumeric sequence: separators and spaces dropped,
  case folded up. Covers VINs, plates, member/loyalty IDs, and resolved
  record ids (``prv-203`` == ``PRV 203`` -> ``PRV203``).
- ``NAME`` — person / place / coined names and addresses: diacritics folded,
  case folded down, apostrophes dropped, ``&`` folded to the word "and" and
  ``%`` to "percent" (both are what the symbol says out loud),
  hyphens / commas / slashes / parentheses folded to spaces, periods
  dropped, a space inserted at every digit<->letter boundary (``K9-Plus`` ==
  ``K9 Plus`` == ``K9Plus`` -> ``k 9 plus``), whitespace collapsed.
  Title-casing is deliberately NOT restored ("van der / de la" makes it
  unsafe), so the stored form is the folded lowercase form:
  ``Ana de la Vega-Marchetti`` -> ``ana de la vega marchetti``.
- ``EMAIL`` — case folded down, trimmed; **separators preserved**. Dots vs
  underscores in a local part are identity-bearing (the §4 hard-email decoys
  differ only by them), so nothing beyond case may fold.
- ``DATE`` — ISO ``YYYY-MM-DD``. Year-first numeric forms, unambiguous
  English month-name forms ("July 16, 2025", "16 Jul 2025"), and year-last
  numerics whose day component exceeds 12 (``07/16/2025``, ``16/07/2025`` —
  only one reading is a calendar month) normalize; genuinely ambiguous
  year-last numerics ("07/06/2025" could be June or July) deliberately do
  not fold and are stored as given (they fail the match, which is correct —
  the format was already pinned by the field spec).
- ``NAME_LIST`` — list-valued fields: split on ``;`` / ``,`` / newlines, each
  item NAME-folded, joined with the canonical ``"; "``. Item **order is
  preserved** — for ``current_medications`` the order is caller-pinned
  (document order), not caller-free.
- ``TIME`` — canonical ``h:mm AM`` / ``h:mm PM`` (no leading hour zero,
  two-digit minutes, uppercase meridiem). Meridiem spellings ("9:00 pm",
  "9 p.m.", a leading hour zero) and the unambiguous words ("noon",
  "midnight") normalize; unambiguous 24-hour forms (``14:30``, ``00:15``)
  normalize too. A bare 12-hour time without a meridiem ("2:30") is
  genuinely ambiguous and deliberately stays as given — the meridiem is
  identity-bearing (design doc §11) and was already pinned by policy.
- ``AMOUNT`` — monetary amount as its minimal decimal digit string: currency
  markers ("$", a "USD"/"dollars" word), thousands separators, and trailing
  decimal zeros carry no identity (``$1,250.00`` == ``1250``); every
  remaining digit does (``125.5`` != ``125.05``).
- Explicit none — the pinned sentinel is JSON null (``None``). On fields that
  accept an explicit none, the bare tokens ``none`` / ``n/a`` / ``null``
  (case-insensitive) fold to the sentinel.

Values that cannot fold (a phone without digits, an empty code) are returned
trimmed-as-given: canonicalization never invents a value, and a nonsense
value failing the DB match is the correct outcome.
"""

import re
import unicodedata
from datetime import date
from enum import Enum
from typing import Optional

NONE_TOKENS = frozenset({"none", "n/a", "null"})

LIST_JOINER = "; "


class FoldKind(str, Enum):
    """Closed set of per-entity-type fold rules (design doc §7)."""

    NAME = "name"  # person / place / coined names, addresses
    CODE = "code"  # VINs, plates, member ids, resolved record ids
    PHONE = "phone"  # phone numbers
    EMAIL = "email"  # email addresses
    DATE = "date"  # calendar dates
    NAME_LIST = "name_list"  # list-valued name fields ("; "-joined)
    TIME = "time"  # times of day ("h:mm AM/PM", meridiem identity-bearing)
    AMOUNT = "amount"  # monetary amounts (minimal decimal digit string)


def fold_phone(value: str) -> str:
    """Bare digit sequence, dialing prefixes normalized. See module docstring."""
    trimmed = value.strip()
    digits = "".join(ch for ch in trimmed if ch.isdigit())
    if digits.startswith("00"):
        # International dialing prefix: '0044…' says the same thing as
        # '+44…'. No subscriber number begins with two zeros.
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("1"):
        # NANP country code: '+1 (206) 555-0111' == '(206) 555-0111'.
        digits = digits[1:]
    if not digits:
        return trimmed
    return digits


def fold_code(value: str) -> str:
    """Uppercase alphanumeric sequence. See module docstring."""
    trimmed = value.strip()
    key = "".join(ch for ch in trimmed if ch.isalnum()).upper()
    return key or trimmed


# Latin letters that carry a diacritic-like distinction but do NOT decompose
# under NFD, so mark-stripping alone cannot fold them. Casefolding "I" gives
# "i" while "ı" stays "ı" — without this map the uppercase form of a Turkish
# name would fold differently from the name itself.
_NON_DECOMPOSING = str.maketrans(
    {
        "ı": "i",
        "ø": "o",
        "ł": "l",
        "đ": "d",
        "ð": "d",
        "þ": "th",
        "æ": "ae",
        "œ": "oe",
    }
)


def fold_name(value: str) -> str:
    """Case-, diacritic-, and separator-folded name form. See module docstring."""
    decomposed = unicodedata.normalize("NFD", value.strip())
    folded = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    ).casefold()
    folded = folded.translate(_NON_DECOMPOSING)
    folded = folded.replace("'", "").replace("’", "")
    folded = folded.replace("&", " and ").replace("%", " percent ")
    folded = re.sub(r"[-,/()]", " ", folded)
    folded = folded.replace(".", "")
    # 'K9Plus', 'K9 Plus', and 'K9-Plus' are the same spoken name; a digit
    # next to a letter is a boundary the writer may or may not mark.
    folded = re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def fold_email(value: str) -> str:
    """Case-folded, trimmed; separators preserved. See module docstring."""
    return value.strip().casefold()


_YEAR_FIRST_DATE = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_COMPACT_DATE = re.compile(r"(\d{4})(\d{2})(\d{2})")
_YEAR_LAST_DATE = re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})")

_MONTHS = {
    name: number
    for number, full in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
    for name in (full, full[:3])
}
_MONTH_ALTERNATION = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH_FIRST_DATE = re.compile(
    rf"({_MONTH_ALTERNATION})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})",
    re.IGNORECASE,
)
_DAY_FIRST_DATE = re.compile(
    rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_ALTERNATION})\.?,?\s+(\d{{4}})",
    re.IGNORECASE,
)


def fold_date(value: str) -> str:
    """ISO ``YYYY-MM-DD`` for unambiguous forms. See module docstring."""
    trimmed = value.strip()
    year = month = day = None
    if match := (
        _YEAR_FIRST_DATE.fullmatch(trimmed) or _COMPACT_DATE.fullmatch(trimmed)
    ):
        year, month, day = (int(part) for part in match.groups())
    elif match := _MONTH_FIRST_DATE.fullmatch(trimmed):
        year = int(match.group(3))
        month = _MONTHS[match.group(1).casefold()]
        day = int(match.group(2))
    elif match := _DAY_FIRST_DATE.fullmatch(trimmed):
        year = int(match.group(3))
        month = _MONTHS[match.group(2).casefold()]
        day = int(match.group(1))
    elif match := _YEAR_LAST_DATE.fullmatch(trimmed):
        # Year-last numerics are unambiguous exactly when only one of the two
        # leading components can be a month ('07/16/2025' / '16/07/2025' both
        # mean July 16). When both could be ('07/06/2025'), the form is
        # genuinely ambiguous and deliberately stays as given.
        first, second, year = (int(part) for part in match.groups())
        if first > 12 and second <= 12:
            day, month = first, second
        elif second > 12 and first <= 12:
            month, day = first, second
        else:
            return trimmed
    if year is None:
        return trimmed
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return trimmed


def fold_name_list(value: str) -> str:
    """Order-preserving canonical ``"; "`` join of NAME-folded items."""
    items = [fold_name(item) for item in re.split(r"[;,\n]", value)]
    return LIST_JOINER.join(item for item in items if item)


_TIME_PATTERN = re.compile(
    r"(?P<hour>\d{1,2})(?:[:.](?P<minute>\d{2}))?"
    r"(?:\s*(?P<meridiem>a\.?m\.?|p\.?m\.?|noon|midnight))?",
    re.IGNORECASE,
)
_TIME_WORDS = {"noon": (12, 0, "PM"), "midnight": (12, 0, "AM")}


def fold_time(value: str) -> str:
    """Canonical ``h:mm AM/PM`` for unambiguous forms. See module docstring."""
    trimmed = value.strip()
    lowered = trimmed.casefold()
    if lowered in _TIME_WORDS:
        hour, minute, meridiem = _TIME_WORDS[lowered]
        return f"{hour}:{minute:02d} {meridiem}"
    match = _TIME_PATTERN.fullmatch(trimmed)
    if not match:
        return trimmed
    hour = int(match.group("hour"))
    minute = int(match.group("minute")) if match.group("minute") else None
    token = (match.group("meridiem") or "").replace(".", "").casefold()
    if minute is not None and minute > 59:
        return trimmed
    if token in ("noon", "midnight"):
        # "12 noon" / "12:00 midnight": the word IS the meridiem — but only
        # the 12 o'clock hour can carry it.
        word_hour, word_minute, meridiem = _TIME_WORDS[token]
        if hour != word_hour or (minute or 0) != word_minute:
            return trimmed
        return f"{word_hour}:{word_minute:02d} {meridiem}"
    if token in ("am", "pm"):
        # A 12-hour reading with an explicit meridiem.
        if not 1 <= hour <= 12:
            return trimmed
        return f"{hour}:{minute or 0:02d} {token.upper()}"
    # No meridiem: only a 24-hour reading is unambiguous. A bare 1..12 hour
    # ("2:30") deliberately stays as given — the meridiem was already pinned
    # by policy, and a missing one failing the match is the correct outcome.
    if minute is None:
        return trimmed
    if hour == 0:
        return f"12:{minute:02d} AM"
    if 13 <= hour <= 23:
        return f"{hour - 12}:{minute:02d} PM"
    return trimmed


_AMOUNT_PATTERN = re.compile(
    r"\$?\s*(?P<digits>\d{1,3}(?:,\d{3})*|\d+)(?:\.(?P<cents>\d+))?"
    r"(?:\s*(?:usd|dollars?))?",
    re.IGNORECASE,
)


def fold_amount(value: str) -> str:
    """Minimal decimal digit string of a money amount. See module docstring."""
    trimmed = value.strip()
    match = _AMOUNT_PATTERN.fullmatch(trimmed)
    if not match:
        return trimmed
    whole = match.group("digits").replace(",", "").lstrip("0") or "0"
    cents = (match.group("cents") or "").rstrip("0")
    if cents:
        return f"{whole}.{cents}"
    return whole


_FOLDERS = {
    FoldKind.NAME: fold_name,
    FoldKind.CODE: fold_code,
    FoldKind.PHONE: fold_phone,
    FoldKind.EMAIL: fold_email,
    FoldKind.DATE: fold_date,
    FoldKind.NAME_LIST: fold_name_list,
    FoldKind.TIME: fold_time,
    FoldKind.AMOUNT: fold_amount,
}


def fold_value(kind: FoldKind, value: str) -> str:
    """Fold ``value`` with the rule for ``kind``."""
    return _FOLDERS[kind](value)


def fold_field(
    kind: FoldKind, value: Optional[str], allows_none: bool = False
) -> Optional[str]:
    """Canonical stored form of one submitted field.

    ``None`` stays the explicit-none sentinel. On fields that accept an
    explicit none, the bare NONE_TOKENS fold to the sentinel too; everywhere
    else they stay literal strings (and correctly fail the match).
    """
    if value is None:
        return None
    if allows_none and value.strip().casefold() in NONE_TOKENS:
        return None
    return fold_value(kind, value)
