# Copyright Sierra
"""Published per-entity-type fold rules for the intake domain.

Reward in this domain is final-DB hash equality against a golden replay, so
two surface forms of the *same captured value* must enter the DB as the same
byte string. The seam is **canonicalization at write time**: ``submit_fields``
folds every field into its canonical stored form before it lands in the DB.
An orthography coin-flip ("maria lopez",
lowercase VINs, "555 123 2002" vs "555-123-2002") must not fail the reward; a
real capture error (digit swap, wrong record, accepted decoy) must.

The canonical form is **purely syntactic**: computable from the writer's input
alone and deterministic. Truth never folds together with any bank decoy;
extension-only phone entries deliberately share a scoring key (asserted in
``tests/test_domains/test_intake/test_folds_intake.py``). The tool cannot know
the pinned truth at runtime, so no rule may depend on it; in particular there
is no "restore bank casing" step — where casing cannot be reconstructed
syntactically (names), the stored form IS the folded form.

Fold rules per entity type:

The scoring projection adds decimal-digit normalization to the seven canonical
person-name, code, phone, date, time, address, and medication families. Unicode
decimal digits with the same value share an ASCII canonical form; coined names,
property names, and email addresses keep their original digit script. Native
addresses also ignore commas as separators, while retaining other punctuation.
Medications ignore separator commas and treat written milligram units in
English, Hindi, Korean, Mandarin, Spanish, and Portuguese as ``mg`` when they
follow a numeric dose; written number words in those published-language
milligram strengths also become digits. A comma between dose digits remains a
decimal mark. Korean person/property names ignore Hangul-only spacing. Korean
addresses ignore parentheses around a five-digit postcode, and Portuguese
addresses equate reviewed state abbreviations with full state names in the
state slot. Korean mobile and Chinese landline domestic trunk zeros are folded
only for their published number shapes.

- ``PHONE`` — the domestic digit sequence. Published-language country codes,
  ``+``/``00`` dialing markers, grouping, separators, and extensions do not
  carry identity; every domestic digit does.
  ``555 123 2002`` == ``555-123-2002`` -> ``5551232002``;
  ``+44 20 7946 0958`` -> ``2079460958``.
- ``CODE`` — uppercase alphanumeric sequence: separators and spaces dropped,
  case folded up. Covers VINs, plates, member/loyalty IDs, and resolved
  record ids (``prv-203`` == ``PRV 203`` -> ``PRV203``).
- ``NAME`` — person / place / coined names and addresses: diacritics folded,
  case folded down, apostrophes dropped, ``&`` folded to the word "and" and
  ``%`` to "percent" (both are what the symbol says out loud),
  hyphens / commas / slashes / parentheses folded to spaces, periods
  dropped only when neither neighbor is a digit, a space inserted at every
  digit<->letter boundary (``K9-Plus`` == ``K9 Plus`` == ``K9Plus`` ->
  ``k 9 plus``), whitespace collapsed.
  Generic NAME keeps unit words distinct: "milligram", "milligrams", and
  ``mg`` remain distinct. Numeric periods are preserved, so ``0.5 mg``
  cannot match ``05 mg`` or ``5 mg``.
  Title-casing is deliberately NOT restored ("van der / de la" makes it
  unsafe), so the stored form is the folded lowercase form:
  ``Ana de la Vega-Marchetti`` -> ``ana de la vega marchetti``.
- ``EMAIL`` — case folded down, trimmed; **separators preserved**. Dots vs
  underscores in a local part are identity-bearing (the hard-email decoys
  differ only by them), so nothing beyond case may fold.
- ``DATE`` — no format conversion. Only surrounding whitespace is trimmed;
  the agent must submit the field's required ``YYYY-MM-DD`` representation.
  Alternative formats remain as submitted and fail the golden-record match.
- ``NAME_LIST`` — list-valued fields: split on ``;`` / ``,`` / newlines, each
  item NAME-folded, joined with the canonical ``"; "``. Item **order is
  preserved** — for ``current_medications`` the order is caller-pinned
  (document order), not caller-free.
- ``TIME`` — no format conversion. Only surrounding whitespace is trimmed;
  the agent must submit ``H:MM AM`` / ``H:MM PM`` (no leading hour zero,
  two-digit minutes, uppercase meridiem). Alternate spellings, 24-hour times,
  and words such as "noon" remain as submitted and fail the golden-record match.
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
from enum import Enum
from functools import partial
from typing import Optional

# 1.6.0: locale-bound Korean/Portuguese orthography, Korean/Chinese dialing,
#        and medication number-word strengths.
# 1.5.0: fold written milligram units in every published intake language.
# 1.4.0: fold medication units/commas and phone country codes/extensions.
# 1.3.0: fold decimal digit scripts in seven families and address commas.
# 1.2.0: require date/time formats and preserve numeric periods in names.
# 1.1.0: preserve written units; milligram/milligrams no longer fold to mg.
SCORING_VERSION = "1.6.0"

NONE_TOKENS = frozenset({"none", "n/a", "null"})

LIST_JOINER = "; "


class FoldKind(str, Enum):
    """Closed set of per-entity-type fold rules."""

    NATIVE_NAME = "native_name"  # NFC/case only: marks and punctuation carry identity
    NATIVE_NAME_DIGITS = "native_name_digits"
    NATIVE_MEDICATION = "native_medication"
    HINDI_MEDICATION = "hindi_medication"
    KOREAN_MEDICATION = "korean_medication"
    CHINESE_MEDICATION = "chinese_medication"
    PORTUGUESE_MEDICATION = "portuguese_medication"
    SPANISH_MEDICATION = "spanish_medication"
    NATIVE_ADDRESS = "native_address"
    KOREAN_NAME = "korean_name"
    KOREAN_NAME_DIGITS = "korean_name_digits"
    KOREAN_ADDRESS = "korean_address"
    PORTUGUESE_ADDRESS = "portuguese_address"
    KOREAN_PHONE = "korean_phone"
    CHINESE_PHONE = "chinese_phone"
    NATIVE_NAME_LIST = "native_name_list"
    NAME = "name"  # person / place / coined names, addresses
    NAME_DIGITS = "name_digits"
    MEDICATION = "medication"
    CODE = "code"  # VINs, plates, member ids, resolved record ids
    PHONE = "phone"  # phone numbers
    EMAIL = "email"  # email addresses
    DATE = "date"  # calendar dates
    NAME_LIST = "name_list"  # list-valued name fields ("; "-joined)
    TIME = "time"  # times of day ("h:mm AM/PM", meridiem identity-bearing)
    AMOUNT = "amount"  # monetary amounts (minimal decimal digit string)


def fold_decimal_digits(value: str) -> str:
    """Map Unicode decimal digits to ASCII; preserve other numeric symbols."""
    return "".join(
        str(unicodedata.decimal(ch)) if ch.isdecimal() else ch for ch in value
    )


def fold_phone(value: str) -> str:
    """Domestic digit sequence, with country prefix and extension ignored."""
    trimmed = value.strip()
    number = re.sub(
        r"(?i)\s*(?:ext(?:ension|ensión)?\.?|x|ramal)\s*[:.]?\s*\d+\s*$",
        "",
        fold_decimal_digits(trimmed),
    )
    # Published Mandarin landline extensions use a final hyphenated digit.
    if re.fullmatch(r"0\d{2}-\d{4}-\d{4}-\d{1,2}", number):
        number = number.rsplit("-", 1)[0]
    digits = "".join(ch for ch in number if ch.isdecimal())
    marked_country = number.lstrip().startswith("+") or digits.startswith("00")
    if digits.startswith("00"):
        digits = digits[2:]
    # These are the dialing prefixes in the published intake languages, plus
    # +44 used by the fold contract. Lengths prevent stripping a domestic
    # number that merely starts with the same digits.
    domestic_lengths = {
        "1": {10},
        "34": {9},
        "44": {9, 10},
        "55": {10, 11},
        "82": {9, 10},
        "86": {11},
        "91": {10},
    }
    for prefix in sorted(domestic_lengths, key=len, reverse=True):
        if (
            prefix == "1"
            and not marked_country
            and not re.match(r"^1[\s.-]", number.lstrip())
        ):
            # A bare 11-digit Chinese mobile can start with 1; only treat a
            # bare NANP 1 as a prefix when written separately.
            continue
        if digits.startswith(prefix) and (
            len(digits) - len(prefix) in domestic_lengths[prefix]
        ):
            digits = digits[len(prefix) :]
            break
    if not digits:
        return trimmed
    return digits


def fold_korean_phone(value: str) -> str:
    """Equate Korean 010 mobiles with +82 10, retaining subscriber digits."""
    digits = fold_phone(value)
    if re.fullmatch(r"010\d{8}", digits):
        return digits[1:]
    return digits


def fold_chinese_phone(value: str) -> str:
    """Equate published 010/021 landlines with +86 10/21, not mobiles."""
    digits = fold_phone(value)
    if re.fullmatch(r"86(?:10|21)\d{8}", digits):
        return digits[2:]
    if re.fullmatch(r"0(?:10|21)\d{8}", digits):
        return digits[1:]
    return digits


def fold_code(value: str) -> str:
    """Uppercase alphanumeric sequence. See module docstring."""
    trimmed = value.strip()
    key = "".join(ch for ch in fold_decimal_digits(trimmed) if ch.isalnum()).upper()
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
    # Keep numeric periods: deleting one can change a medication's strength.
    # Ordinary initial/abbreviation periods, such as "Dr.", still fold away.
    folded = re.sub(r"(?<!\d)\.(?!\d)", "", folded)
    # 'K9Plus', 'K9 Plus', and 'K9-Plus' are the same spoken name; a digit
    # next to a letter is a boundary the writer may or may not mark.
    folded = re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def fold_native_name(value: str) -> str:
    """Canonical Unicode composition and casing, retaining script-bearing marks.

    No transliteration, accent deletion, English symbol expansion or fuzzy
    matching: authored variants that differ in written identity remain distinct.
    """
    return unicodedata.normalize("NFC", " ".join(value.strip().casefold().split()))


def fold_korean_name(value: str) -> str:
    """Ignore spacing between Hangul syllables in Korean personal/property names."""
    return re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", fold_native_name(value))


def fold_korean_name_digits(value: str) -> str:
    """Korean personal-name spacing with equivalent decimal digit scripts."""
    return fold_korean_name(fold_decimal_digits(value))


def fold_native_address(value: str) -> str:
    """Fold address digit scripts and commas, retaining all other punctuation."""
    commas = str.maketrans({",": " ", "，": " ", "、": " "})
    return fold_native_name(fold_decimal_digits(value).translate(commas))


def fold_korean_address(value: str) -> str:
    """Ignore parentheses around a five-digit Korean postcode only."""
    normalized = fold_decimal_digits(value)
    normalized = re.sub(r"\(\s*(\d{5})\s*\)(?=\s*$)", r" \1 ", normalized)
    return fold_native_address(normalized)


_BRAZILIAN_STATE_NAMES = {
    "sp": "são paulo",
    "pr": "paraná",
    "mg": "minas gerais",
    "rj": "rio de janeiro",
    "rs": "rio grande do sul",
    "pe": "pernambuco",
    "pa": "pará",
    "sc": "santa catarina",
}
_BRAZILIAN_STATE = re.compile(
    r"(?P<separator>[,-])\s*(?P<state>"
    + "|".join(
        re.escape(form)
        for code, name in _BRAZILIAN_STATE_NAMES.items()
        for form in (code, name)
    )
    + r")\s*(?=,|$)",
    re.IGNORECASE,
)


def fold_portuguese_address(value: str) -> str:
    """Fold reviewed Brazilian state names only in an address state slot."""
    normalized = fold_decimal_digits(value)

    def replace_state(match: re.Match[str]) -> str:
        written = match.group("state").casefold()
        for code, name in _BRAZILIAN_STATE_NAMES.items():
            if written in (code, name):
                return f" {code}"
        raise AssertionError("Matched an unrecognized Brazilian state")

    return fold_native_address(_BRAZILIAN_STATE.sub(replace_state, normalized))


_MILLIGRAM_UNIT = re.compile(
    r"(?P<dose>[0-9]+(?:\.[0-9]+)?)\s*"
    r"(?:milligrams?|miligram(?:o|a)s?|मिलीग्राम|मिलिग्राम|एमजी|"
    r"밀리그램|毫克|mg)(?![A-Za-z])",
    re.IGNORECASE,
)

# Only strengths that occur in the published medication frame are accepted.
# This is a written-form equivalence, not a general word-to-number parser: the
# word must directly modify a milligram dose, never an address, code, or name.
_MEDICATION_STRENGTH_WORDS = {
    "one": "1",
    "two": "2",
    "four": "4",
    "five": "5",
    "ten": "10",
    "thirty": "30",
    "fifty": "50",
    "eighty": "80",
    "one hundred": "100",
    "one hundred fifty": "150",
    "two hundred fifty": "250",
    "six hundred": "600",
    "एक": "1",
    "दो": "2",
    "चार": "4",
    "पाँच": "5",
    "पांच": "5",
    "दस": "10",
    "तीस": "30",
    "पचास": "50",
    "अस्सी": "80",
    "सौ": "100",
    "एक सौ": "100",
    "एक सौ पचास": "150",
    "दो सौ पचास": "250",
    "छह सौ": "600",
    "일": "1",
    "이": "2",
    "사": "4",
    "오": "5",
    "십": "10",
    "삼십": "30",
    "오십": "50",
    "팔십": "80",
    "백": "100",
    "백오십": "150",
    "이백오십": "250",
    "육백": "600",
    "一": "1",
    "二": "2",
    "四": "4",
    "五": "5",
    "十": "10",
    "三十": "30",
    "五十": "50",
    "八十": "80",
    "百": "100",
    "一百": "100",
    "一百五十": "150",
    "二百五十": "250",
    "两百五十": "250",
    "六百": "600",
    "um": "1",
    "uma": "1",
    "dois": "2",
    "duas": "2",
    "quatro": "4",
    "cinco": "5",
    "dez": "10",
    "trinta": "30",
    "cinquenta": "50",
    "oitenta": "80",
    "cem": "100",
    "cento e cinquenta": "150",
    "duzentos e cinquenta": "250",
    "seiscentos": "600",
    "uno": "1",
    "una": "1",
    "dos": "2",
    "cuatro": "4",
    "diez": "10",
    "treinta": "30",
    "cincuenta": "50",
    "ochenta": "80",
    "cien": "100",
    "ciento cincuenta": "150",
    "doscientos cincuenta": "250",
    "seiscientos": "600",
}
_STRENGTH_UNIT = (
    r"(?:milligrams?|miligram(?:o|a)s?|मिलीग्राम|मिलिग्राम|एमजी|"
    r"밀리그램|毫克|mg)"
)
_STRENGTH_WORDS_BY_LANGUAGE = {
    "en": "one|two|four|five|ten|thirty|fifty|eighty|one hundred|"
    "one hundred fifty|two hundred fifty|six hundred",
    "hi": "एक|दो|चार|पाँच|पांच|दस|तीस|पचास|अस्सी|सौ|एक सौ|एक सौ पचास|दो सौ पचास|छह सौ",
    "ko": "일|이|사|오|십|삼십|오십|팔십|백|백오십|이백오십|육백",
    "zh": "一|二|四|五|十|三十|五十|八十|百|一百|一百五十|二百五十|两百五十|六百",
    "pt": "um|uma|dois|duas|quatro|cinco|dez|trinta|cinquenta|oitenta|"
    "cem|cento e cinquenta|duzentos e cinquenta|seiscentos",
    "es": "uno|una|dos|cuatro|cinco|diez|treinta|cincuenta|ochenta|"
    "cien|ciento cincuenta|doscientos cincuenta|seiscientos",
}
_STRENGTH_WORD_PATTERNS = {
    language: re.compile(
        r"(?<![A-Za-z])(?:"
        + "|".join(
            re.escape(word) for word in sorted(words.split("|"), key=len, reverse=True)
        )
        + r")(?=\s*"
        + _STRENGTH_UNIT
        + r"(?![A-Za-z]))",
        re.IGNORECASE,
    )
    for language, words in _STRENGTH_WORDS_BY_LANGUAGE.items()
}


def _fold_medication_strength_words(value: str, language: str | None) -> str:
    """Turn published-language written milligram strengths into their digits."""
    if language is None:
        return value
    return _STRENGTH_WORD_PATTERNS[language].sub(
        lambda match: _MEDICATION_STRENGTH_WORDS[match.group().casefold()], value
    )


def _normalize_medication_units(value: str, language: str | None = None) -> str:
    """Canonicalize numeric milligram doses without changing other unit words."""
    normalized = fold_decimal_digits(_fold_medication_strength_words(value, language))
    # A comma between digits is a decimal mark, not a punctuation separator.
    normalized = re.sub(r"(?<=\d)[,，](?=\d)", ".", normalized)
    normalized = normalized.replace(",", " ").replace("，", " ")

    def replace_unit(match: re.Match[str]) -> str:
        # CJK medicine names/forms often touch the strength with no space.
        before = normalized[match.start() - 1] if match.start() else ""
        separator = " " if before and before.isalpha() else ""
        return f"{separator}{match.group('dose')} mg "

    return _MILLIGRAM_UNIT.sub(replace_unit, normalized)


def fold_medication(value: str) -> str:
    """English medication fold with equivalent numeric milligram units."""
    return fold_name(_normalize_medication_units(value, "en"))


def fold_native_medication(value: str, language: str | None = None) -> str:
    """Native medication fold with equivalent numeric milligram units."""
    return fold_native_name(_normalize_medication_units(value, language))


def fold_name_digits(value: str) -> str:
    """English name fold with decimal digit scripts normalized."""
    return fold_name(fold_decimal_digits(value))


def fold_native_name_digits(value: str) -> str:
    """Native name fold with decimal digit scripts normalized."""
    return fold_native_name(fold_decimal_digits(value))


def fold_protocol_digits(value: str) -> str:
    """Retain required date/time format while folding decimal digit scripts."""
    return fold_decimal_digits(value.strip())


def fold_native_name_list(value: str) -> str:
    """Preserve list order and all identity-bearing marks in each native name."""
    return LIST_JOINER.join(
        fold_native_name(item) for item in re.split(r"[;,\n]", value) if item.strip()
    )


def fold_email(value: str) -> str:
    """Case-folded, trimmed; separators preserved. See module docstring."""
    return value.strip().casefold()


def fold_name_list(value: str) -> str:
    """Order-preserving canonical ``"; "`` join of NAME-folded items."""
    items = [fold_name(item) for item in re.split(r"[;,\n]", value)]
    return LIST_JOINER.join(item for item in items if item)


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
    FoldKind.NAME_DIGITS: fold_name_digits,
    FoldKind.MEDICATION: fold_medication,
    FoldKind.NATIVE_NAME: fold_native_name,
    FoldKind.NATIVE_NAME_DIGITS: fold_native_name_digits,
    FoldKind.KOREAN_NAME: fold_korean_name,
    FoldKind.KOREAN_NAME_DIGITS: fold_korean_name_digits,
    FoldKind.NATIVE_MEDICATION: fold_native_medication,
    FoldKind.HINDI_MEDICATION: partial(fold_native_medication, language="hi"),
    FoldKind.KOREAN_MEDICATION: partial(fold_native_medication, language="ko"),
    FoldKind.CHINESE_MEDICATION: partial(fold_native_medication, language="zh"),
    FoldKind.PORTUGUESE_MEDICATION: partial(fold_native_medication, language="pt"),
    FoldKind.SPANISH_MEDICATION: partial(fold_native_medication, language="es"),
    FoldKind.NATIVE_ADDRESS: fold_native_address,
    FoldKind.KOREAN_ADDRESS: fold_korean_address,
    FoldKind.PORTUGUESE_ADDRESS: fold_portuguese_address,
    FoldKind.NATIVE_NAME_LIST: fold_native_name_list,
    FoldKind.CODE: fold_code,
    FoldKind.PHONE: fold_phone,
    FoldKind.KOREAN_PHONE: fold_korean_phone,
    FoldKind.CHINESE_PHONE: fold_chinese_phone,
    FoldKind.EMAIL: fold_email,
    FoldKind.DATE: fold_protocol_digits,
    FoldKind.NAME_LIST: fold_name_list,
    FoldKind.TIME: fold_protocol_digits,
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
