# Copyright Sierra
"""Entity banks for the intake task generator (design doc §4).

Banks are checked-in, pinned YAML under ``data/tau2/domains/intake/banks/``.
The bank is the unit (owner call, 2026-08-24): each of the 15 banks holds one
homogeneous value grammar, every entry is tagged **easy** or **hard**, and
every entry carries a ``decoys`` list of plausible same-type neighbors (the
fold suite asserts truth and decoys stay fold-distinct). Loading is
pydantic-validated and fails loud on unknown fields, duplicate values, decoys
that collide with real entries, check-digit violations (VINs and mod-97
member IDs are validated in code, see :mod:`tau2.domains.intake.check_digits`),
and phone values outside the officially reserved fictional ranges (see
:mod:`tau2.domains.intake.grounding`).

**The flat bank contract** (design doc §4): every bank ships exactly
:data:`BANK_CONTRACT_PER_TIER` easy and as many hard values — validated at
build time via :func:`assert_bank_contract`, which the generator calls before
drawing. The canonical freeze draws 10 per bank × tier cell without
replacement, so the pool covers two full additive doublings (10 → 20 → 40)
with no value reuse.

Two banks keep extra columns the generator consumes:

- ``person_names`` — every entry carries a reviewed ``gender`` (any entry can
  be drawn as the callee, and the voice sidecar pins the spoken persona),
  optionally an ``alt_name`` former-name variant, and per-token
  ``pronunciations`` columns (design doc §2.4; hard tokens also carry the
  ``mispronounced`` variant + distortion operator).
- ``medications`` — the drug-name tokens carry ``pronunciations`` columns
  (never dose/form), every one with a ``mispronounced`` variant.
- ``codes`` / ``dates`` / ``coined`` — carry the closed ``family`` / ``kind``
  / ``role`` tag that decides which record field the value fills.
- ``emails`` — owner-linked local-part *patterns*, instantiated with the
  drawn callee's name at generation time (:func:`instantiate_email`).
"""

import hashlib
import re
import unicodedata
from enum import Enum
from pathlib import Path
from typing import Annotated, Generic, List, Optional, TypeVar

from pydantic import Field, model_validator

from tau2.domains.intake.check_digits import is_valid_member_id, is_valid_vin
from tau2.domains.intake.grounding import validate_fictional_phones
from tau2.domains.intake.tasks.loanwords import LoanLanguage
from tau2.domains.intake.tasks.pronunciation import (
    NO_VARIANT_TOKENS,
    OPERATOR_APPLICABILITY,
    DistortionOperator,
)
from tau2.domains.intake.utils import INTAKE_DATA_DIR
from tau2.utils import load_file
from tau2.utils.pydantic_utils import BaseModelNoExtra

INTAKE_BANKS_DIR = INTAKE_DATA_DIR / "banks"

# The flat bank contract (design doc §4): exactly this many values per tier,
# in every bank, no per-bank exceptions.
BANK_CONTRACT_PER_TIER = 40


class Difficulty(str, Enum):
    """The two per-slot difficulty levels (design doc §4)."""

    EASY = "easy"
    HARD = "hard"


class Gender(str, Enum):
    """Callee gender for voice-persona pinning (name_genders.py convention:
    every caller-facing given name carries an explicit, reviewed gender)."""

    MALE = "male"
    FEMALE = "female"


class CodeFamily(str, Enum):
    """Closed set of alphanumeric-code families the codes bank carries."""

    VIN = "vin"  # 17 chars, ISO 3779 check digit — always a hard capture
    MEMBER_ID = "member_id"  # MBR-<core>-<cc>, mod-97 check digits
    PLATE = "plate"  # license plate, no check digit
    LOYALTY = "loyalty"  # loyalty program number, no check digit


class DateKind(str, Enum):
    """What a banked date is usable as."""

    BIRTH_DATE = "birth_date"  # past date, absolute (easy)
    FUTURE_DATE = "future_date"  # scheduling date, absolute (easy)
    RELATIVE_DATE = "relative_date"  # spoken relative form (hard); the value
    # is the resolution against the pinned clock


class CoinedRole(str, Enum):
    """Which capture field a coined proper noun is usable for."""

    EMPLOYER = "employer"  # clinic: employer_name
    COMPANY = "company"  # hotel: company_name
    BRAND = "brand"  # auto: aftermarket_brand


class PronunciationSource(str, Enum):
    """Where a token's phonemes came from (design doc §2.4 / §8b)."""

    CMUDICT = "cmudict"  # ARPABET, pinned cmusphinx/cmudict commit
    WIKIPRON = "wikipron"  # IPA, pinned CUNY-CL/wikipron TSV (English rows
    # carry no language; foreign rows carry the pinned LoanLanguage)
    MANUAL = "manual"  # hand-filled from USAN/MedlinePlus, cited
    WIKTIONARY = "wiktionary"  # exact-form IPA printed on en.wiktionary,
    # per-token source URL in `citation` (design doc §8b sourcing ladder)
    AUTHORED = "authored"  # authored respelling for coinages/archaisms,
    # rationale/convention cited, owner-reviewed via the packet


# Sources whose rows must carry a citation (reference URL or rationale);
# lexicon-bulk sources must not — their provenance lives in the extract.
CITED_SOURCES = frozenset(
    {
        PronunciationSource.MANUAL,
        PronunciationSource.WIKTIONARY,
        PronunciationSource.AUTHORED,
    }
)


def person_name_tokens(value: str) -> list[str]:
    """The pronunciation-bearing tokens of a person value: the first name and
    each surname token, hyphenated constructions split into components
    (``"Ana O'Brien-Smith"`` -> ``["Ana", "O'Brien", "Smith"]``)."""
    tokens: list[str] = []
    for word in value.split():
        tokens.extend(part for part in word.split("-") if part)
    return tokens


def medication_drug_tokens(value: str) -> list[str]:
    """The pronunciation-bearing tokens of a medication value: the drug-name
    words before the dose (first digit-leading token), hyphenated combination
    drugs split into components — never dose or form words
    (``"Levodopa-Carbidopa 25-100 mg tablet"`` -> ``["Levodopa",
    "Carbidopa"]``)."""
    tokens: list[str] = []
    for word in value.split():
        if word[0].isdigit():
            break
        tokens.extend(part for part in word.split("-") if part)
    if not tokens:
        raise ValueError(f"Medication value {value!r} has no drug-name token")
    return tokens


class TokenPronunciation(BaseModelNoExtra):
    """The pronunciation columns of one value token (design doc §2.4)."""

    token: Annotated[
        str, Field(description="The exact value token these columns describe.")
    ]
    source: Annotated[
        PronunciationSource, Field(description="Lexicon the phonemes came from.")
    ]
    phonemes: Annotated[
        str,
        Field(
            description=(
                "ARPABET (cmudict) or space-segmented IPA (wikipron); for "
                "manual entries, the cited reference pronunciation as printed."
            )
        ),
    ]
    respelling: Annotated[
        str,
        Field(
            description=(
                "Derived hyphenated ASCII respelling, stressed syllable in "
                "caps — what TTS gets (design doc §3)."
            )
        ),
    ]
    mispronounced: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "The curated bad variant for the mispronounced_term "
                "complication (design doc §4), stored TTS-ready in natural "
                "orthography (lowercase plain letters, e.g. 'iburprofen'); "
                "hard-tier names and all medications carry one."
            ),
        ),
    ]
    operator: Annotated[
        Optional[DistortionOperator],
        Field(
            default=None,
            description="The distortion operator that produced `mispronounced`.",
        ),
    ]
    citation: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "Reference for cited sources: USAN/MedlinePlus for manual, "
                "the per-token source URL for wiktionary, the authoring "
                "rationale/convention for authored."
            ),
        ),
    ]
    language: Annotated[
        Optional[LoanLanguage],
        Field(
            default=None,
            description=(
                "The pinned source language of a foreign token (design doc "
                "§8b) — decides which lexicon and adaptation table applied. "
                "Always None on English-lexicon rows (person_names, "
                "medications)."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _check_columns(self) -> "TokenPronunciation":
        for label, text in (
            ("token", self.token),
            ("respelling", self.respelling),
            ("mispronounced", self.mispronounced or ""),
        ):
            if not text.isascii():
                raise ValueError(
                    f"Pronunciation {label} for {self.token!r} is not ASCII: {text!r}"
                )
        if not self.respelling:
            raise ValueError(f"Token {self.token!r} has an empty respelling")
        if (self.mispronounced is None) != (self.operator is None):
            raise ValueError(
                f"Token {self.token!r}: mispronounced and operator must be set together"
            )
        if self.mispronounced is not None:
            # Natural orthography: what TTS receives, verbatim — lowercase
            # plain letters, and never the token's own spelling (that swap
            # would be vacuous).
            if not (self.mispronounced.isalpha() and self.mispronounced.islower()):
                raise ValueError(
                    f"Token {self.token!r}: mispronounced "
                    f"{self.mispronounced!r} is not lowercase plain letters "
                    "(natural orthography)"
                )
            if self.mispronounced == self.token.casefold():
                raise ValueError(
                    f"Token {self.token!r}: mispronounced equals the token's "
                    "own spelling (vacuous swap)"
                )
        if (self.source in CITED_SOURCES) != (self.citation is not None):
            raise ValueError(
                f"Token {self.token!r}: citation is required for "
                "manual/wiktionary/authored pronunciations and forbidden "
                "otherwise"
            )
        if self.source is PronunciationSource.WIKTIONARY and not (
            self.citation or ""
        ).startswith("https://en.wiktionary.org/"):
            raise ValueError(
                f"Token {self.token!r}: wiktionary rows cite the per-token "
                f"entry URL, got {self.citation!r}"
            )
        return self


def _check_pronunciations(
    bank: str,
    value: str,
    expected_tokens: list[str],
    pronunciations: List[TokenPronunciation],
    decoys: List[str],
    mispronounced_required: bool,
) -> None:
    """Shared entry-level pronunciation gates: token alignment, the
    mispronounced-presence rule, per-bank operator applicability (owner
    directive 2026-08-26), and decoy-token collision freedom."""
    got_tokens = [p.token for p in pronunciations]
    if got_tokens != expected_tokens:
        raise ValueError(
            f"Entry {value!r}: pronunciation tokens {got_tokens} != value "
            f"tokens {expected_tokens}"
        )
    decoy_tokens = {
        token.casefold() for decoy in decoys for token in person_name_tokens(decoy)
    }
    no_variant = NO_VARIANT_TOKENS[bank]
    for pronunciation in pronunciations:
        if pronunciation.language is not None:
            raise ValueError(
                f"Entry {value!r}: token {pronunciation.token!r} carries a "
                f"language pin — English-lexicon banks ({bank}) never do"
            )
        if pronunciation.source in (
            PronunciationSource.WIKTIONARY,
            PronunciationSource.AUTHORED,
        ):
            raise ValueError(
                f"Entry {value!r}: source {pronunciation.source.value!r} is a "
                f"foreign-token tier — English-lexicon banks ({bank}) use "
                "cmudict/wikipron/manual"
            )
        if mispronounced_required and pronunciation.mispronounced is None:
            if pronunciation.token not in no_variant:
                raise ValueError(
                    f"Entry {value!r}: token {pronunciation.token!r} must "
                    "carry a mispronounced variant"
                )
        if pronunciation.mispronounced is not None and (
            not mispronounced_required or pronunciation.token in no_variant
        ):
            raise ValueError(
                f"Entry {value!r}: token {pronunciation.token!r} must not "
                "carry a mispronounced variant "
                f"({'NO_VARIANT_TOKENS' if mispronounced_required else 'easy tier'})"
            )
        if (
            pronunciation.operator is not None
            and pronunciation.operator not in OPERATOR_APPLICABILITY[bank]
        ):
            raise ValueError(
                f"Entry {value!r}: operator {pronunciation.operator.value!r} "
                f"does not apply to the {bank} bank"
            )
        if (
            pronunciation.mispronounced is not None
            and pronunciation.mispronounced.casefold() in decoy_tokens
        ):
            raise ValueError(
                f"Entry {value!r}: mispronounced "
                f"{pronunciation.mispronounced!r} collides with a decoy token"
            )


class BankEntry(BaseModelNoExtra):
    """One flat bank entry: a value with difficulty and decoy neighbors."""

    value: Annotated[str, Field(description="The exact pinned value.")]
    difficulty: Annotated[
        Difficulty, Field(description="easy or hard per the design §4 table.")
    ]
    decoys: Annotated[
        List[str],
        Field(
            default_factory=list,
            description=(
                "Plausible same-type neighbors; the fold suite asserts they "
                "stay fold-distinct from every real entry of the bank."
            ),
        ),
    ]


class PersonNameEntry(BankEntry):
    """A census/SSA-derived person name (see ``tau2 intake-names``)."""

    gender: Annotated[
        Gender,
        Field(
            description=(
                "The name's gender (male/female), required: any entry can be "
                "drawn as the CALLEE, and the voice sidecar pins the spoken "
                "persona to it (the name_genders.py convention). Unknown or "
                "missing genders are rejected on load."
            )
        ),
    ]
    alt_name: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "A former/maiden-name variant of the same person (kept for "
                "the localization seam's identity derivation)."
            ),
        ),
    ]
    pronunciations: Annotated[
        List[TokenPronunciation],
        Field(
            description=(
                "Pronunciation columns per value token — first name and each "
                "surname token (design doc §2.4). Hard-tier tokens carry a "
                "mispronounced variant; easy-tier tokens never do."
            )
        ),
    ]

    @model_validator(mode="after")
    def _check_pronunciation_columns(self) -> "PersonNameEntry":
        _check_pronunciations(
            "person_names",
            self.value,
            person_name_tokens(self.value),
            self.pronunciations,
            self.decoys,
            mispronounced_required=self.difficulty is Difficulty.HARD,
        )
        return self


class MedicationEntry(BankEntry):
    """A medication; the drug-name tokens carry pronunciation columns."""

    pronunciations: Annotated[
        List[TokenPronunciation],
        Field(
            description=(
                "Pronunciation columns for the drug-name tokens only — never "
                "dose or form (design doc §2.4). Every token carries a "
                "mispronounced variant."
            )
        ),
    ]

    @model_validator(mode="after")
    def _check_pronunciation_columns(self) -> "MedicationEntry":
        _check_pronunciations(
            "medications",
            self.value,
            medication_drug_tokens(self.value),
            self.pronunciations,
            self.decoys,
            mispronounced_required=True,
        )
        return self


# The fixed value tokenizer of the foreign-token banks (design doc §8b):
# identical to the section-8 oddball scan's tokenization, so the two
# ownership boundaries (oddball table vs pronunciation columns) partition
# the same token stream. Hyphens split ("d'El-Rei" -> "d'El", "Rei");
# apostrophes stay inside their token (clitics like "l'Ecluse").
_FOREIGN_VALUE_TOKEN = re.compile(r"[A-Za-z0-9&']+")


def foreign_value_tokens(value: str) -> list[str]:
    """The tokens of a properties/vehicles value, in appearance order."""
    return _FOREIGN_VALUE_TOKEN.findall(value)


def _check_foreign_pronunciations(bank: str, entry: "ForeignTokenEntry") -> None:
    """Entry-level gates for the foreign-token banks (design doc §8b).

    Load-time validation is necessarily weaker than the build's (the
    English-coverage scan needs the pinned lexicons): every pronunciation
    row must name a distinct token of the value; every row carries a
    mispronounced variant unless allow-listed (NO_VARIANT_TOKENS, enforced
    both directions); wikipron rows carry the language pin that selected
    their lexicon; operators respect the per-bank applicability table; and
    no variant collides with a decoy token. Full candidate-set alignment is
    enforced by the build and its guard test.
    """
    value_tokens = foreign_value_tokens(entry.value)
    seen: set[str] = set()
    decoy_tokens = {
        token.casefold()
        for decoy in entry.decoys
        for token in foreign_value_tokens(decoy)
    }
    no_variant = NO_VARIANT_TOKENS[bank]
    for pronunciation in entry.pronunciations:
        token = pronunciation.token
        if token not in value_tokens:
            raise ValueError(
                f"Entry {entry.value!r}: pronunciation token {token!r} is "
                "not a token of the value"
            )
        if token in seen:
            raise ValueError(
                f"Entry {entry.value!r}: duplicate pronunciation token {token!r}"
            )
        seen.add(token)
        if pronunciation.source in (
            PronunciationSource.CMUDICT,
            PronunciationSource.MANUAL,
        ):
            raise ValueError(
                f"Entry {entry.value!r}: source "
                f"{pronunciation.source.value!r} is an English-lexicon tier "
                f"— foreign-token banks ({bank}) use "
                "wikipron/wiktionary/authored"
            )
        if pronunciation.source is PronunciationSource.WIKIPRON:
            if pronunciation.language is None:
                raise ValueError(
                    f"Entry {entry.value!r}: token {token!r} has a bulk-"
                    "lexicon pronunciation but no language pin"
                )
            if pronunciation.language is not entry.language:
                raise ValueError(
                    f"Entry {entry.value!r}: token {token!r} pinned "
                    f"{pronunciation.language.value} but the entry is pinned "
                    f"{entry.language.value if entry.language else None}"
                )
        if pronunciation.mispronounced is None:
            if token not in no_variant:
                raise ValueError(
                    f"Entry {entry.value!r}: token {token!r} must carry a "
                    "mispronounced variant (design doc §8b)"
                )
        elif token in no_variant:
            raise ValueError(
                f"Entry {entry.value!r}: token {token!r} is in "
                "NO_VARIANT_TOKENS and must not carry a variant"
            )
        if (
            pronunciation.operator is not None
            and pronunciation.operator not in OPERATOR_APPLICABILITY[bank]
        ):
            raise ValueError(
                f"Entry {entry.value!r}: operator "
                f"{pronunciation.operator.value!r} does not apply to the "
                f"{bank} bank"
            )
        if (
            pronunciation.mispronounced is not None
            and pronunciation.mispronounced.casefold() in decoy_tokens
        ):
            raise ValueError(
                f"Entry {entry.value!r}: mispronounced "
                f"{pronunciation.mispronounced!r} collides with a decoy token"
            )


class ForeignTokenEntry(BankEntry):
    """A properties/vehicles entry: foreign tokens carry pronunciation
    columns and the entry carries its one-time language pin (design doc §8b).

    ``language`` is the curated per-entry pin — it decided which bulk
    lexicon and adaptation table applied to the entry's tokens and is
    reviewed in the packet like everything else. Entries whose foreign
    tokens are all authored (romanized Greek, archaisms, manufacturer
    coinages) may carry no pin; entries with no foreign tokens carry
    neither pin nor columns.
    """

    language: Annotated[
        Optional[LoanLanguage],
        Field(
            default=None,
            description=(
                "One-time per-entry language pin, curated from the entry's "
                "linguistic flavor (several tokens are attested in multiple "
                "languages, so first-hit lookup is not acceptable)."
            ),
        ),
    ]
    pronunciations: Annotated[
        List[TokenPronunciation],
        Field(
            default_factory=list,
            description=(
                "Pronunciation columns for the entry's foreign tokens only "
                "(design doc §8b); every column-bearing token carries a "
                "mispronounced variant unless allow-listed."
            ),
        ),
    ]


class PropertyEntry(ForeignTokenEntry):
    """A hotel property name (foreign-token pronunciation columns, §8b)."""

    @model_validator(mode="after")
    def _check_pronunciation_columns(self) -> "PropertyEntry":
        _check_foreign_pronunciations("properties", self)
        return self


class VehicleEntry(ForeignTokenEntry):
    """A vehicle make/model (foreign-token pronunciation columns, §8b)."""

    @model_validator(mode="after")
    def _check_pronunciation_columns(self) -> "VehicleEntry":
        _check_foreign_pronunciations("vehicles", self)
        return self


class CodeEntry(BankEntry):
    """An alphanumeric code; check-digit families are validated on load."""

    family: Annotated[
        CodeFamily, Field(description="Which code family the value belongs to.")
    ]

    @model_validator(mode="after")
    def _check_digits(self) -> "CodeEntry":
        if self.family is CodeFamily.VIN:
            for value in (self.value, *self.decoys):
                if not is_valid_vin(value):
                    raise ValueError(f"Invalid VIN check digit: {value!r}")
        if self.family is CodeFamily.MEMBER_ID:
            for value in (self.value, *self.decoys):
                if not is_valid_member_id(value):
                    raise ValueError(f"Invalid member-ID check digits: {value!r}")
        return self


class DateEntry(BankEntry):
    """A date; relative entries carry the spoken form and its resolution."""

    kind: Annotated[DateKind, Field(description="What the date is usable as.")]
    spoken: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "The spoken relative form ('a week from today'); required for "
                "relative_date entries whose value is the resolution against "
                "the pinned clock."
            ),
        ),
    ]

    @model_validator(mode="after")
    def _check_kind(self) -> "DateEntry":
        if self.kind is DateKind.RELATIVE_DATE:
            if not self.spoken:
                raise ValueError(
                    f"relative_date entry {self.value!r} must carry its spoken form"
                )
            if self.difficulty is not Difficulty.HARD:
                raise ValueError(
                    f"relative_date entry {self.value!r} must be hard (the "
                    "callee speaks the relative form; resolving it is the "
                    "hazard)"
                )
        elif self.spoken is not None:
            raise ValueError(
                f"absolute date entry {self.value!r} must not carry a spoken form"
            )
        return self


class CoinedEntry(BankEntry):
    """A coined proper noun (invented, never a dictionary word or real brand)."""

    role: Annotated[
        CoinedRole, Field(description="Which capture field the name is usable for.")
    ]


_EMAIL_PATTERN_TOKENS = re.compile(r"\{(first|last|f|l)\}|[a-z0-9._-]")


class EmailEntry(BaseModelNoExtra):
    """An email *pattern*, instantiated with its owner's name at draw time.

    Emails are owner-linked (a pinned local-part pattern over the drawn
    person's name plus a pinned fictional domain), so a callee's email always
    reads as their own. Placeholders: ``{first}`` / ``{last}`` (first / last
    name token, ASCII-folded lowercase) and ``{f}`` / ``{l}`` (initials).
    The decoy instantiates ``decoy_pattern``/``decoy_domain`` (defaulting to
    the entry's own) with the same owner and must differ from the value.
    """

    pattern: Annotated[
        str,
        Field(
            description=(
                "Local-part pattern: placeholders {first}/{last}/{f}/{l} plus "
                "literal [a-z0-9._-] characters."
            )
        ),
    ]
    domain: Annotated[str, Field(description="Fictional domain of the address.")]
    decoy_pattern: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Decoy local-part pattern (default: the entry's own).",
        ),
    ]
    decoy_domain: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Decoy domain (default: the entry's own).",
        ),
    ]
    difficulty: Annotated[Difficulty, Field(description="easy or hard per §4.")]

    @model_validator(mode="after")
    def _check_patterns(self) -> "EmailEntry":
        for pattern in (self.pattern, self.decoy_pattern):
            if pattern is None:
                continue
            if _EMAIL_PATTERN_TOKENS.sub("", pattern):
                raise ValueError(
                    f"Email pattern {pattern!r} has characters outside "
                    "{{first}}/{{last}}/{{f}}/{{l}} and [a-z0-9._-]"
                )
        if (self.decoy_pattern or self.pattern, self.decoy_domain or self.domain) == (
            self.pattern,
            self.domain,
        ):
            raise ValueError(
                f"Email entry {self.pattern}@{self.domain}: decoy must differ "
                "in pattern or domain"
            )
        return self


def _email_name_tokens(owner_name: str) -> list[str]:
    """ASCII-folded lowercase name tokens (hyphens kept inside tokens)."""
    decomposed = unicodedata.normalize("NFD", owner_name.strip())
    folded = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    ).casefold()
    folded = folded.replace("'", "").replace("’", "").replace(".", "")
    return folded.split()


def instantiate_email(entry: EmailEntry, owner_name: str) -> BankEntry:
    """Instantiate an email pattern with its owner's name.

    Returns a flat :class:`BankEntry` (value + one decoy) so draw sites handle
    emails exactly like every other banked value. Fails loud when the decoy
    collapses onto the value or the local part comes out malformed.
    """
    tokens = _email_name_tokens(owner_name)
    if not tokens:
        raise ValueError(f"Cannot derive email tokens from name {owner_name!r}")
    subs = {
        "first": tokens[0],
        "last": tokens[-1],
        "f": tokens[0][0],
        "l": tokens[-1][0],
    }

    def build(pattern: str, domain: str) -> str:
        local = pattern.format(**subs)
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", local):
            raise ValueError(
                f"Email local part {local!r} (pattern {pattern!r}, owner "
                f"{owner_name!r}) is malformed"
            )
        return f"{local}@{domain}"

    value = build(entry.pattern, entry.domain)
    decoy = build(
        entry.decoy_pattern or entry.pattern, entry.decoy_domain or entry.domain
    )
    if decoy == value:
        raise ValueError(f"Email decoy collapsed onto the value: {value}")
    return BankEntry(value=value, difficulty=entry.difficulty, decoys=[decoy])


EntryT = TypeVar("EntryT", bound=BaseModelNoExtra)


class BankFile(BaseModelNoExtra, Generic[EntryT]):
    """One bank file: the entity name and its entries."""

    entity: Annotated[str, Field(description="Entity bank name (matches filename).")]
    entries: Annotated[List[EntryT], Field(description="The pinned entries.")]


class Banks(BaseModelNoExtra):
    """All 15 loaded entity banks plus per-file provenance shas."""

    person_names: List[PersonNameEntry]
    codes: List[CodeEntry]
    phones: List[BankEntry]
    dates: List[DateEntry]
    times: List[BankEntry]
    addresses: List[BankEntry]
    properties: List[PropertyEntry]
    amounts: List[BankEntry]
    emails: List[EmailEntry]
    medications: List[MedicationEntry]
    insurance_plans: List[BankEntry]
    vehicles: List[VehicleEntry]
    rate_plans: List[BankEntry]
    coined: List[CoinedEntry]
    shops: List[BankEntry]
    shas: Annotated[
        dict[str, str],
        Field(description="sha256 per bank file, provenance for the manifest."),
    ]


# The 15 banks, in the design §4 listing order (providers and parts were
# retired 2026-08-24: providers duplicated person_names post-census-refactor,
# parts are dictionary compounds the pinning policy exempts from spelling).
_BANK_FILES: dict[str, type] = {
    "person_names": PersonNameEntry,
    "codes": CodeEntry,
    "phones": BankEntry,
    "dates": DateEntry,
    "times": BankEntry,
    "addresses": BankEntry,
    "properties": PropertyEntry,
    "amounts": BankEntry,
    "emails": EmailEntry,
    "medications": MedicationEntry,
    "insurance_plans": BankEntry,
    "vehicles": VehicleEntry,
    "rate_plans": BankEntry,
    "coined": CoinedEntry,
    "shops": BankEntry,
}

BANK_NAMES: tuple[str, ...] = tuple(_BANK_FILES)


def _entry_key(entry) -> str:
    """The identity string of an entry, for uniqueness checks."""
    if isinstance(entry, EmailEntry):
        return f"{entry.pattern}@{entry.domain}"
    if isinstance(entry, BankEntry):
        return entry.value
    raise TypeError(f"Bank entry {type(entry).__name__} has no identity attribute")


def _decoy_keys(entry) -> list[str]:
    if isinstance(entry, EmailEntry):
        return [
            f"{entry.decoy_pattern or entry.pattern}@{entry.decoy_domain or entry.domain}"
        ]
    return list(entry.decoys)


def load_banks(banks_dir: Optional[Path] = None) -> Banks:
    """Load and validate every entity bank. Fails loud on a missing file,
    unknown fields, duplicate entry values, or a decoy colliding with a real
    entry (a decoy must stay a decoy — it never doubles as a drawable value).
    """
    banks_dir = banks_dir or INTAKE_BANKS_DIR
    loaded: dict[str, list] = {}
    shas: dict[str, str] = {}
    for bank_name, entry_model in _BANK_FILES.items():
        path = banks_dir / f"{bank_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Missing entity bank: {path}")
        shas[bank_name] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = load_file(path)
        bank_file = BankFile[entry_model].model_validate(raw)
        if bank_file.entity != bank_name:
            raise ValueError(
                f"Bank file {path.name} declares entity {bank_file.entity!r}, "
                f"expected {bank_name!r}"
            )
        keys = [_entry_key(entry) for entry in bank_file.entries]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate entries in bank {bank_name}: {', '.join(duplicates)}"
            )
        key_set = set(keys)
        for entry in bank_file.entries:
            collisions = sorted(set(_decoy_keys(entry)) & key_set)
            if collisions:
                raise ValueError(
                    f"Bank {bank_name} entry {_entry_key(entry)!r} has decoy(s) "
                    f"colliding with real entries: {', '.join(collisions)}"
                )
        loaded[bank_name] = bank_file.entries
    validate_fictional_phones(
        value for entry in loaded["phones"] for value in (entry.value, *entry.decoys)
    )
    _check_given_name_genders(loaded["person_names"])
    return Banks(**loaded, shas=shas)


def assert_bank_contract(banks: Banks) -> None:
    """Enforce the flat bank contract: exactly 40 easy + 40 hard per bank.

    The generator refuses to draw against a violating bank — a short pool
    would silently shrink a cell's headroom for additive doublings, and an
    over-full one would make "10 of the 40" a different draw than reviewed.
    """
    problems: list[str] = []
    for bank_name in _BANK_FILES:
        entries = getattr(banks, bank_name)
        for tier in Difficulty:
            count = sum(1 for entry in entries if entry.difficulty is tier)
            if count == BANK_CONTRACT_PER_TIER:
                continue
            delta = count - BANK_CONTRACT_PER_TIER
            direction = "over by" if delta > 0 else "short by"
            problems.append(
                f"bank {bank_name}: {count} {tier.value} values "
                f"({direction} {abs(delta)}, contract is "
                f"{BANK_CONTRACT_PER_TIER})"
            )
    if problems:
        raise ValueError(
            "Bank contract violation (exactly "
            f"{BANK_CONTRACT_PER_TIER} easy + {BANK_CONTRACT_PER_TIER} hard "
            "per bank): " + "; ".join(problems)
        )


def _check_given_name_genders(entries: List[PersonNameEntry]) -> None:
    """One given name, one gender (the name_genders.py 'unambiguous catalog'
    spirit): two entries sharing a first token but disagreeing on gender is an
    authoring slip that would make the callee's pinned voice a coin flip."""
    by_given: dict[str, tuple[str, Gender]] = {}
    for entry in entries:
        given = entry.value.split()[0]
        seen = by_given.get(given)
        if seen is not None and seen[1] is not entry.gender:
            raise ValueError(
                f"Bank person_names: given name {given!r} is tagged "
                f"{seen[1].value} on {seen[0]!r} but {entry.gender.value} on "
                f"{entry.value!r} — one given name must carry one gender"
            )
        by_given.setdefault(given, (entry.value, entry.gender))
