"""Normalization of times, dates, phone numbers, names, and addresses for the food domain."""

import re
from datetime import datetime
from typing import Dict, List, Optional

_ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "avenue": "ave",
    "road": "rd",
    "drive": "dr",
    "boulevard": "blvd",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "terrace": "ter",
    "parkway": "pkwy",
    "circle": "cir",
    "highway": "hwy",
    "apartment": "apt",
    "unit": "apt",
    "suite": "ste",
    "building": "bldg",
}


def _collapse(text: str) -> str:
    """Lowercases and collapses whitespace (for hashing names and addresses)."""
    return " ".join((text or "").lower().split())


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.strip().split(":")
    return int(hours) * 60 + int(minutes)


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _parse_clock(value: str) -> str:
    """Normalizes a clock time to 'HH:MM' (accepts 'H:MM', 'HH:MM', and am/pm forms)."""
    text = (value or "").strip().lower().replace(".", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not m:
        raise ValueError(f"Use a time like '18:30' (got '{value}').")
    hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if hour > 24 or minute > 59 or (hour == 24 and minute):
        raise ValueError(f"Use a time like '18:30' (got '{value}').")
    return f"{hour:02d}:{minute:02d}"


def _phone_digits(phone: str) -> Optional[str]:
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")


def _normalize_phone(phone: str, allow_default: bool = True) -> str:
    """Normalize a phone number to XXX-XXX-XXXX when 10 US digits are present."""
    digits = _phone_digits(phone)
    if digits is not None:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    cleaned = " ".join((phone or "").strip().split())
    if not cleaned:
        if allow_default:
            return "000-000-0000"
        raise ValueError("Customer phone number cannot be empty.")
    return cleaned


def _normalize_name(name: str, allow_default: bool = True) -> str:
    """Normalize a customer name by collapsing whitespace and title-casing."""
    cleaned = " ".join((name or "").strip().split())
    if not cleaned:
        if allow_default:
            return "Guest"
        raise ValueError("Customer name cannot be empty.")
    return cleaned.title()


def _normalize_order_type(order_type: str) -> str:
    """Normalize order_type into 'pickup', 'delivery', or 'dine_in'."""
    raw = re.sub(r"[\s\-_]+", "_", (order_type or "").strip().lower())
    mapping = {
        "pickup": "pickup",
        "pick_up": "pickup",
        "carryout": "pickup",
        "carry_out": "pickup",
        "takeout": "pickup",
        "take_out": "pickup",
        "to_go": "pickup",
        "togo": "pickup",
        "drive_thru": "pickup",
        "drivethru": "pickup",
        "counter": "pickup",
        "delivery": "delivery",
        "deliver": "delivery",
        "dine_in": "dine_in",
        "dinein": "dine_in",
        "for_here": "dine_in",
        "in_store": "dine_in",
    }
    if raw not in mapping:
        raise ValueError(
            f"Invalid order_type '{order_type}'. Must be one of: 'pickup', 'delivery', 'dine_in'."
        )
    return mapping[raw]


_UNIT_OR_ROUTE_PREFIXES = frozenset(
    {"apt", "ste", "bldg", "fl", "floor", "rm", "room", "lot", "hwy", "rte", "route"}
)
_ORDINAL_STREET_WORDS = frozenset(
    {
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
    }
)
_STREET_SUFFIX_TOKENS = frozenset(_ADDRESS_ABBREVIATIONS) | frozenset(
    _ADDRESS_ABBREVIATIONS.values()
)


def _normalize_address(address: Optional[str]) -> str:
    """Normalize a street address so minor punctuation/suffix differences match."""
    raw = (address or "").strip().lower()
    cleaned = re.sub(r"#", " # ", re.sub(r"[,.;]+", " ", raw))
    tokens = _house_number_to_digits([tok for tok in cleaned.split() if tok])
    mapped = [_ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in tokens]
    out: List[str] = []
    for idx, tok in enumerate(mapped):
        if tok == "#":
            if out and idx + 1 < len(mapped) and out[-1] not in _UNIT_OR_ROUTE_PREFIXES:
                out.append("apt")
            continue
        out.append(tok)
    if not out:
        raise ValueError("Delivery address is required for delivery orders.")
    return " ".join(tok.title() for tok in out)


_NUMBER_WORDS: Dict[str, int] = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_SCALES = {"hundred": 100, "thousand": 1000}


def _house_number_to_digits(tokens: List[str]) -> List[str]:
    """Writes a spoken leading house number as digits ('twelve oak st' -> '12 oak st').

    Handles 'four twenty' (420), 'seven forty-two' (742), 'one oh five' (105), and
    'twelve hundred' (1200). Only the leading run of number words is converted.
    """
    words: List[str] = []
    used = 0
    for idx, tok in enumerate(tokens):
        if (
            tok == "and"
            and words
            and words[-1] in _NUMBER_SCALES
            and idx + 2 < len(tokens)
            and all(
                p in _NUMBER_WORDS or p in _NUMBER_SCALES
                for p in tokens[idx + 1].split("-")
            )
            and tokens[idx + 2] not in _ORDINAL_STREET_WORDS
            and not (
                idx + 2 == len(tokens) - 1 and tokens[idx + 2] in _STREET_SUFFIX_TOKENS
            )
        ):
            used += 1
            continue
        parts = tok.split("-")
        if not all(p in _NUMBER_WORDS or p in _NUMBER_SCALES for p in parts):
            break
        words.extend(parts)
        used += 1
    if not words or words[0] in ("oh", "zero") or words[0] in _NUMBER_SCALES:
        return tokens
    if any(w in _NUMBER_SCALES for w in words):
        total, current = 0, 0
        for w in words:
            if w in _NUMBER_SCALES:
                current = max(current, 1) * _NUMBER_SCALES[w]
                if _NUMBER_SCALES[w] >= 1000:
                    total, current = total + current, 0
            else:
                current += _NUMBER_WORDS[w]
        number = str(total + current)
    else:
        # Spoken in groups: a tens word followed by a ones word is one group ('forty two').
        chunks: List[str] = []
        i = 0
        while i < len(words):
            value = _NUMBER_WORDS[words[i]]
            nxt = _NUMBER_WORDS.get(words[i + 1], 0) if i + 1 < len(words) else 0
            if value >= 20 and value % 10 == 0 and 0 < nxt < 10:
                chunks.append(str(value + nxt))
                i += 2
            else:
                chunks.append(str(value))
                i += 1
        number = "".join(chunks)
    return [number] + tokens[used:]


def _normalize_date(date: str) -> str:
    """Returns a date as 'YYYY-MM-DD' (accepts '2026-6-26'); raises ValueError if malformed."""
    text = (date or "").strip()
    try:
        return datetime.strptime(text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Use a date like '2026-06-26' (got '{date}').") from None


def _date_key(date: str) -> str:
    """The normalized date, or the raw text when it is not a valid date (reported elsewhere)."""
    try:
        return _normalize_date(date)
    except ValueError:
        return date
