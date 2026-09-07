# Copyright Sierra
"""Load-time grounding validation for the intake phones bank.

Phone numbers are the one entity family where a fictional value can collide
with a real subscriber: a made-up-looking number is still routable. Every
banked phone value AND every decoy must therefore sit inside an *officially
reserved fictional range* — ranges the regulators set aside precisely so
drama/testing content can never reach a real line. The closed range table
lives here (fixed in-code data, per the machine-not-scripts mandate) and the
bank loader rejects any value outside every range, loudly, at load time.

The four reserved ranges (regulator citations in the table below):

- **NANP** — 555-0100 through 555-0199 in any area code is reserved for
  fictional use by the North American Numbering Plan Administrator
  (NANPA; ATIS-0300115, "555 NXX assignment guidelines"). Bank values may
  carry an ``ext. N`` suffix — extensions are private-PBX digits with no
  public routing, so the reserved range still covers the number.
- **UK** — Ofcom reserves "drama numbers" per area; the London block is
  020 7946 0000..0999 (Ofcom, "Telephone numbers for drama purposes").
- **AU** — ACMA reserves fictitious number blocks including
  (02) 5550 0000..9999 (ACMA, "Fictitious numbers for advertising and
  entertainment", Telecommunications Numbering Plan).
- **FR** — ARCEP reserves 06 39 98 00 00..06 39 98 99 99 as the fictional
  mobile range (ARCEP, "Numeros fictifs pour les oeuvres audiovisuelles").

No network access happens here: this is a pure, deterministic format check.
Reality screening (DNS, registries) is the build-time
``tau2 intake-grounding screen`` verb in :mod:`grounding_screen`.
"""

import re
from typing import Iterable

# Regulator-reserved fictional phone ranges, as anchored patterns over the
# bank's pinned formatting. The formats are deliberately strict: banks pin
# exact strings, so accepting loose formatting would only mask authoring
# drift, never help a legitimate value.
FICTIONAL_PHONE_RANGES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        # NANPA / ATIS-0300115: (AAA) 555-0100..0199, any 3-digit area code,
        # optional " ext. N" PBX extension suffix.
        "NANP fictional 555-0100..0199 (NANPA, ATIS-0300115)",
        re.compile(r"\(\d{3}\) 555-01\d{2}(?: ext\. [1-9]\d{0,3})?"),
    ),
    (
        # Ofcom drama numbers, London block: +44 20 7946 0000..0999.
        "UK Ofcom drama range, London (+44 20 7946 0000..0999)",
        re.compile(r"\+44 20 7946 0\d{3}"),
    ),
    (
        # ACMA fictitious numbers: +61 2 5550 0000..9999.
        "AU ACMA fictitious range (+61 2 5550 0000..9999)",
        re.compile(r"\+61 2 5550 \d{4}"),
    ),
    (
        # ARCEP fictional mobile range: +33 6 39 98 00 00..99 99.
        "FR ARCEP fictional mobile range (+33 6 39 98 00 00..99 99)",
        re.compile(r"\+33 6 39 98 \d{2} \d{2}"),
    ),
)


def is_fictional_phone(value: str) -> bool:
    """True when ``value`` sits inside one of the reserved fictional ranges."""
    return any(pattern.fullmatch(value) for _, pattern in FICTIONAL_PHONE_RANGES)


def validate_fictional_phones(values: Iterable[str]) -> None:
    """Fail loud on any phone value outside every reserved fictional range.

    The bank loader calls this over every phones-bank value and decoy: a
    value outside every range could be a real, routable subscriber number,
    which the benchmark must never speak.
    """
    offenders = [value for value in values if not is_fictional_phone(value)]
    if offenders:
        ranges = "; ".join(name for name, _ in FICTIONAL_PHONE_RANGES)
        raise ValueError(
            "Bank phones: value(s) outside every officially reserved "
            f"fictional range: {', '.join(repr(v) for v in offenders)}. "
            f"Accepted ranges: {ranges}"
        )
