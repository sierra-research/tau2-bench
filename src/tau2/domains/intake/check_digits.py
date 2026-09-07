# Copyright Sierra
"""Check-digit code families for the intake domain.

Codes with a check digit are deliberately part of the entity banks (design doc
§4): they let scoring tell "captured wrong and submitted anyway" from
"captured wrong and caught it" with zero extra machinery. Two families ship in
v1:

- **VIN** (ISO 3779): 17 characters, no I/O/Q, position 9 is a mod-11 check
  digit over the transliterated remaining characters.
- **Member ID** (mod-97, IBAN-style): ``MBR-<core>-<cc>`` where the core is
  digits (easy) or alternating letters+digits (hard), letters digitized
  A=10..Z=35, and the two check digits satisfy
  ``(digitized(core) * 100 + cc) % 97 == 1``. Mod-97 catches every
  single-character error and almost every transposition, which is exactly the
  property the benchmark trades on.

The bank loader validates every banked code against its family on load, and
``submit_intake``'s validation dial enforces the same validators at submit
time.
"""

import re

# --- VIN (ISO 3779) ---------------------------------------------------------

_VIN_ALLOWED = re.compile(r"[A-HJ-NPR-Z0-9]{17}")

_VIN_TRANSLITERATION = {
    **{str(d): d for d in range(10)},
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "J": 1,
    "K": 2,
    "L": 3,
    "M": 4,
    "N": 5,
    "P": 7,
    "R": 9,
    "S": 2,
    "T": 3,
    "U": 4,
    "V": 5,
    "W": 6,
    "X": 7,
    "Y": 8,
    "Z": 9,
}

_VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def vin_check_digit(vin: str) -> str:
    """The ISO 3779 check digit (position 9) implied by the other 16 chars.

    Raises:
        ValueError: If the VIN is not 17 valid VIN characters (I, O, Q are
            never valid).
    """
    vin = vin.upper()
    if not _VIN_ALLOWED.fullmatch(vin):
        raise ValueError(
            f"A VIN must be exactly 17 characters from A-Z (excluding I, O, Q) "
            f"and 0-9, got: {vin!r}"
        )
    total = sum(
        _VIN_TRANSLITERATION[ch] * weight
        for ch, weight in zip(vin, _VIN_WEIGHTS)
        if weight  # position 9 (the check digit itself) has weight 0
    )
    remainder = total % 11
    return "X" if remainder == 10 else str(remainder)


def is_valid_vin(vin: str) -> bool:
    """Whether the string is a well-formed VIN with a correct check digit."""
    try:
        return vin.upper()[8] == vin_check_digit(vin)
    except (ValueError, IndexError):
        return False


# --- Member ID (mod-97) -----------------------------------------------------

_MEMBER_ID_PATTERN = re.compile(r"MBR-([A-Z0-9]{4,12})-(\d{2})")


def _digitize(core: str) -> int:
    """IBAN-style digitization: digits pass through, letters map A=10..Z=35."""
    digits = "".join(
        ch if ch.isdigit() else str(ord(ch) - ord("A") + 10) for ch in core
    )
    return int(digits)


def member_id_check_digits(core: str) -> str:
    """The two mod-97 check digits for a member-ID core.

    Raises:
        ValueError: If the core is not 4-12 uppercase alphanumerics.
    """
    core = core.upper()
    if not re.fullmatch(r"[A-Z0-9]{4,12}", core):
        raise ValueError(
            f"A member-ID core must be 4-12 characters A-Z / 0-9, got: {core!r}"
        )
    check = (1 - _digitize(core) * 100) % 97
    return f"{check:02d}"


def format_member_id(core: str) -> str:
    """Format a core as a full ``MBR-<core>-<cc>`` member ID."""
    return f"MBR-{core.upper()}-{member_id_check_digits(core)}"


def is_valid_member_id(member_id: str) -> bool:
    """Whether the string is ``MBR-<core>-<cc>`` with correct check digits.

    Matching is case-insensitive and ignores the exact separators, mirroring
    the published fold rule for codes (digits and letters carry the identity).
    """
    key = "".join(ch for ch in member_id if ch.isalnum()).upper()
    if not key.startswith("MBR") or len(key) < 9:
        return False
    core, check = key[3:-2], key[-2:]
    try:
        return member_id_check_digits(core) == check
    except ValueError:
        return False
