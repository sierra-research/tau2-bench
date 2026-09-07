# Copyright Sierra
"""Oddball-token pronunciation table (design doc sec. 8).

A mechanical scan of the invented-content banks (coined, insurance_plans,
rate_plans, shops, addresses, vehicles, properties; values AND decoys) for
TTS-risk token classes drives a shared token-level respelling table,
``data/tau2/domains/intake/banks/token_pronunciations.yaml``. The classes
are fixed in-code regexes (owner-pinned, design doc sec. 8):

- ``mixed_case_inner`` — ``[A-Za-z]*[a-z][A-Z][A-Za-z]*`` (GmbH, XyloNine,
  gCaorach): genuinely unpredictable readings;
- ``all_caps`` — ``[A-Z]{2,}`` (PPO, HDHP, MICE, SMERF): several are real
  words TTS may speak instead of spelling, or vice versa;
- ``ampersand`` — tokens containing ``&`` (B&B): need a literal-token match.

Letter-digit-mix tokens (3B, 6TT, Q7) read predictably and are **ignored by
owner decision** — except the owner-pinned graduates in
``GRADUATED_LETTER_DIGIT`` (leet-spelled coinages like Dr4gline and
convention-bound trim codes like 4MATIC, flagged in the 2026-08-26 packet
review), which carry the ``letter_digit`` class and get authored readings
like any other oddball. The extractor skips the rest and the review packet
lists them as ignored, not auditioned.

The respellings are AUTHORED data, owner-reviewed via the packet
(``tau2 intake-names token-packet``): these are invented or convention-bound
tokens, so the respelling DEFINES the canonical pronunciation rather than
recording one. Loading is validated against the extractor: every extracted
token has exactly one entry, no orphan entries, ASCII only, class recorded
and matching the extractor's classification. The table is REFERENCE
DATA only (owner decision 2026-08-26 after the audition listen): runs use
the provider's default reading, and the table never joins the pre-synthesis
substitution map — see :mod:`tau2.domains.intake.pronunciation_map`.
"""

import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Optional

from pydantic import ConfigDict, Field, model_validator

from tau2.utils import load_file
from tau2.utils.pydantic_utils import BaseModelNoExtra

TOKEN_PRONUNCIATIONS_FILENAME = "token_pronunciations.yaml"

# The banks the extractor scans (design doc sec. 8): the invented-content
# banks whose values are coinages or industry codes, values AND decoys.
ODDBALL_SOURCE_BANKS: tuple[str, ...] = (
    "coined",
    "insurance_plans",
    "rate_plans",
    "shops",
    "addresses",
    "vehicles",
    "properties",
)

# Fixed tokenization + class regexes (owner-pinned; changing any of these is
# a table regeneration event, never a silent drift).
_TOKEN = re.compile(r"[A-Za-z0-9&']+")
_MIXED_CASE_INNER = re.compile(r"[A-Za-z]*[a-z][A-Z][A-Za-z]*")
_ALL_CAPS = re.compile(r"[A-Z]{2,}")
_HAS_LETTER = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"[0-9]")


class OddballTokenClass(str, Enum):
    """Closed set of TTS-risk token classes the extractor recognizes."""

    MIXED_CASE_INNER = "mixed_case_inner"
    ALL_CAPS = "all_caps"
    AMPERSAND = "ampersand"
    LETTER_DIGIT = "letter_digit"


# Letter-digit-mix tokens the owner graduated out of the ignored class
# (packet review 2026-08-26): leet-spelled coinages whose intended word is
# not the naive digit-by-digit reading, plus convention-bound vehicle/rate
# codes. Closed, owner-pinned list — a new letter-digit token stays ignored
# until it is added here AND given a table entry (the loader enforces both).
GRADUATED_LETTER_DIGIT: frozenset[str] = frozenset(
    {
        "15Pax",
        "228i",
        "4MATIC",
        "Dr4gline",
        "Dz6",
        "Dz9",
        "Fj0rdline",
        "Kv4rn",
        "Kw2rtz",
        "Ny7ro",
        "P300e",
        "Qw1kfit",
        "Str8ta",
        "Thr0ttleworx",
        "Thr33fold",
        "Vy8ka",
        "Wr3nchworks",
        "Xen0dyne",
        "Zw0lle",
        "Zw4an",
        "sDrive28i",
    }
)


class OddballToken(BaseModelNoExtra):
    """One authored oddball-token respelling (the table's row shape)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    token: Annotated[str, Field(description="The exact bank token, as extracted.")]
    respelling: Annotated[
        str,
        Field(
            description=(
                "The pinned canonical reading TTS gets: authored ASCII "
                "(letter names written out for initialisms, words for "
                "word-convention codes, phonetic respelling for coinages)."
            )
        ),
    ]
    token_class: Annotated[
        OddballTokenClass,
        Field(
            alias="class",
            description="The extractor class the token matched (recorded so "
            "the table stays auditable against the fixed regexes).",
        ),
    ]
    note: Annotated[
        str,
        Field(
            description="One-line authoring rationale shown in the review "
            "packet (convention cited, compound split, coined reading)."
        ),
    ]

    @model_validator(mode="after")
    def _check_ascii(self) -> "OddballToken":
        for label, text in (("token", self.token), ("respelling", self.respelling)):
            if not text.isascii():
                raise ValueError(
                    f"Oddball {label} for {self.token!r} is not ASCII: {text!r}"
                )
        if not self.respelling:
            raise ValueError(f"Oddball token {self.token!r} has an empty respelling")
        return self


class OddballExtraction(BaseModelNoExtra):
    """The mechanical scan result: classified tokens plus the ignored class."""

    tokens: Annotated[
        dict[str, OddballTokenClass],
        Field(description="Every non-ignored oddball token -> its class."),
    ]
    ignored_letter_digit: Annotated[
        List[str],
        Field(
            description="Letter-digit-mix tokens skipped by owner decision, "
            "sorted (listed in the packet as ignored)."
        ),
    ]


def classify_token(token: str) -> Optional[OddballTokenClass]:
    """The fixed classification of one bank token, or None.

    Order matters and is pinned: a token with both letters and digits is the
    ignored letter-digit class (returns None — callers that need the ignored
    list use :func:`extract_oddball_tokens`) unless the owner graduated it
    via ``GRADUATED_LETTER_DIGIT``; ampersand beats the case classes (an
    ``&`` token can never fullmatch the pure-alpha regexes).
    """
    if not _HAS_LETTER.search(token):
        return None
    if _HAS_DIGIT.search(token):
        if token in GRADUATED_LETTER_DIGIT:
            return OddballTokenClass.LETTER_DIGIT
        return None
    if "&" in token:
        return OddballTokenClass.AMPERSAND
    if _ALL_CAPS.fullmatch(token):
        return OddballTokenClass.ALL_CAPS
    if _MIXED_CASE_INNER.fullmatch(token):
        return OddballTokenClass.MIXED_CASE_INNER
    return None


def extract_oddball_tokens(banks=None) -> OddballExtraction:
    """Mechanically scan the source banks' values and decoys.

    Deterministic pure function of the checked-in banks; the table loader
    validates the authored yaml against exactly this result.
    """
    from tau2.domains.intake.tasks.banks import load_banks

    banks = banks if banks is not None else load_banks()
    tokens: dict[str, OddballTokenClass] = {}
    ignored: set[str] = set()
    for bank_name in ODDBALL_SOURCE_BANKS:
        for entry in getattr(banks, bank_name):
            for value in (entry.value, *entry.decoys):
                for token in _TOKEN.findall(value):
                    token_class = classify_token(token)
                    if token_class is not None:
                        tokens[token] = token_class
                    elif _HAS_LETTER.search(token) and _HAS_DIGIT.search(token):
                        ignored.add(token)
    return OddballExtraction(tokens=tokens, ignored_letter_digit=sorted(ignored))


class _TokenPronunciationsFile(BaseModelNoExtra):
    """The yaml file shape: the authored entries."""

    entries: Annotated[List[OddballToken], Field(description="Authored rows.")]


@lru_cache(maxsize=4)
def load_token_pronunciations(
    banks_dir: Optional[Path] = None,
) -> tuple[OddballToken, ...]:
    """Load the authored table and validate it against the extractor.

    Fails loud on: a missing file, duplicate tokens, an extracted token with
    no entry, an orphan entry no extraction produces, or a recorded class
    that disagrees with the fixed classification.
    """
    from tau2.domains.intake.tasks.banks import INTAKE_BANKS_DIR, load_banks

    banks_dir = banks_dir if banks_dir is not None else INTAKE_BANKS_DIR
    path = banks_dir / TOKEN_PRONUNCIATIONS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing oddball-token table: {path}")
    table = _TokenPronunciationsFile.model_validate(load_file(path))

    seen: set[str] = set()
    for entry in table.entries:
        if entry.token in seen:
            raise ValueError(f"Duplicate oddball-token entry: {entry.token!r}")
        seen.add(entry.token)

    extraction = extract_oddball_tokens(
        load_banks(banks_dir) if banks_dir != INTAKE_BANKS_DIR else None
    )
    stale_graduates = sorted(GRADUATED_LETTER_DIGIT - set(extraction.tokens))
    if stale_graduates:
        raise ValueError(
            "GRADUATED_LETTER_DIGIT lists tokens no bank produces "
            f"(stale after a bank edit?): {stale_graduates}"
        )
    missing = sorted(set(extraction.tokens) - seen)
    orphans = sorted(seen - set(extraction.tokens))
    if missing or orphans:
        raise ValueError(
            "token_pronunciations.yaml out of sync with the extractor: "
            f"missing entries for {missing}, orphan entries {orphans}"
        )
    for entry in table.entries:
        expected = extraction.tokens[entry.token]
        if entry.token_class is not expected:
            raise ValueError(
                f"Oddball token {entry.token!r} recorded as "
                f"{entry.token_class.value} but classifies as {expected.value}"
            )
    return tuple(table.entries)


# ---------------------------------------------------------------------------
# Review packet (tau2 intake-names token-packet)
# ---------------------------------------------------------------------------


class TokenPacket(BaseModelNoExtra):
    """The rendered oddball-token review packet plus summary counts."""

    total: Annotated[int, Field(description="Non-ignored tokens in the table.")]
    per_class: Annotated[
        dict[str, int], Field(description="Token count per extractor class.")
    ]
    ignored: Annotated[
        int, Field(description="Letter-digit-mix tokens listed as ignored.")
    ]
    markdown: Annotated[str, Field(description="The full packet body (deterministic).")]


def build_token_packet() -> TokenPacket:
    """Render the oddball-token table + the ignored list for owner review.

    Deterministic pure function of the checked-in banks and table.
    """
    entries = load_token_pronunciations()
    extraction = extract_oddball_tokens()
    per_class = {token_class.value: 0 for token_class in OddballTokenClass}
    for entry in entries:
        per_class[entry.token_class.value] += 1

    lines: list[str] = []
    lines.append("# Intake oddball-token pronunciation review packet")
    lines.append("")
    lines.append(
        "Provenance: authored table "
        f"banks/{TOKEN_PRONUNCIATIONS_FILENAME}, validated against the fixed "
        "in-code extractor over the "
        f"{', '.join(ODDBALL_SOURCE_BANKS)} banks (values and decoys). "
        "Regenerable byte-identically via `tau2 intake-names token-packet`; "
        "do not hand-edit. The respelling DEFINES each token's canonical "
        "reading as REFERENCE DATA (owner decision 2026-08-26 after the "
        "audition listen: runs use the provider's default reading — the "
        "table is never swapped into synthesis)."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- tokens in the table: {len(entries)}")
    for token_class, count in per_class.items():
        lines.append(f"- {token_class}: {count}")
    lines.append(
        f"- ignored letter-digit-mix tokens (owner decision): "
        f"{len(extraction.ignored_letter_digit)}"
    )
    lines.append("")
    lines.append("## Pinned readings")
    lines.append("")
    lines.append("| token | class | respelling | note |")
    lines.append("|---|---|---|---|")
    for entry in sorted(entries, key=lambda e: (e.token_class.value, e.token)):
        lines.append(
            f"| {entry.token} | {entry.token_class.value} | "
            f"{entry.respelling} | {entry.note} |"
        )
    lines.append("")
    lines.append("## Ignored letter-digit-mix tokens")
    lines.append("")
    lines.append(
        "Excluded from the table by owner decision (read predictably); "
        "listed for completeness, not auditioned:"
    )
    lines.append("")
    lines.append(", ".join(extraction.ignored_letter_digit))
    lines.append("")
    return TokenPacket(
        total=len(entries),
        per_class=per_class,
        ignored=len(extraction.ignored_letter_digit),
        markdown="\n".join(lines),
    )
