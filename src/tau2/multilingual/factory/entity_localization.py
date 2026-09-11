# Copyright Sierra
"""Corpus-driven, deterministic locale identity swap for benchmark task sets.

The reusable core behind ``tau2 factory localize-entities`` (and the
``add-a-language`` pipeline). Given a per-locale ``locale_corpus.yaml`` (the
reviewed raw material — gendered name pools, cities, address/zip/email/phone
conventions) it builds a **deterministic identity map** keyed by the domain
profile's caller key — the original caller ``user_id`` for USER_ID_HANDLE
domains (airline), the English caller full name for STRUCTURED_NAME_PHONE
domains (telecom, whose diversified seed callers share one canonical
phone/customer record) — and drives :func:`derive_identity_tasks` to emit the
identity-variant ``<domain>_tasks_<lang>_identity.json`` set plus a caller-gender
sidecar for voice routing.

Gender is not sampled: each identity inherits the gender of the ENGLISH source
caller's given name (``tau2.multilingual.factory.name_genders``), so a task
keeps one caller gender across its plain-localized and ``_identity`` variants
and across languages, and the sidecar covers EVERY localized task id — the
voice sampler pins the persona to the caller's gender on both task sets, never
just the ``_identity`` one.

Determinism: every per-caller choice is seeded by the profile's caller key
(original ``user_id`` or English full name) via a process-stable SHA-256 seed,
so re-running yields byte-identical artifacts. Nothing here calls an LLM.

Only the callers actually referenced by the task set get an identity — the
per-task ``initialization_data`` patch is all any task exercises (see
``docs/multilingual/LOCALE_ENTITY_SWAP_PLAN.md``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Annotated, Literal, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tau2.config import DEFAULT_MULTILINGUAL_DOMAIN
from tau2.multilingual.domain_profiles import (
    CallerIdentityKind,
    DomainLocalizationProfile,
    get_domain_profile,
)
from tau2.multilingual.factory.name_genders import source_caller_gender
from tau2.multilingual.invariants import (
    AddressProseFormat,
    caller_set_user_info,
    resolve_caller_user_id,
)
from tau2.multilingual.localize_lib import (
    customer_by_phone,
    data_dir,
    derive_identity_tasks,
    load_domain_db,
    load_domain_tasks,
    multilingual_data_dir,
    resolve_script_code,
)
from tau2.multilingual.names import (
    NameOrder,
    compose_full_name,
    name_order_for_language,
)
from tau2.multilingual.native_script import (
    NATIVE_IDENTITY_VARIANT,
    native_name_spellout,
)

CORPUS_FILENAME = "locale_corpus.yaml"
IDENTITY_LOCALIZATION_VERSION = "v3_localized_telecom_phone_numbers"
# The native-script DB variant (romanization x localization ablation): same
# deterministic identity draw, but the NAME fields keep the locale's native
# script (Devanagari / Hanzi) instead of the ASCII fold. Versioned separately:
# the two variants are different regimes, not two versions of one artifact.
NATIVE_IDENTITY_LOCALIZATION_VERSION = "v1_native_script_db"


class CorpusCity(BaseModel):
    """One city/region pair for the locale-wide address generator."""

    model_config = ConfigDict(extra="forbid")

    city: Annotated[str, Field(description="City name (used as address city)")]
    region: Annotated[
        str,
        Field(description="Region/state code for the address 'state' field"),
    ] = ""


class ReviewedAddressCity(BaseModel):
    """One reviewed city with internally consistent address material."""

    model_config = ConfigDict(extra="forbid")

    city: str
    region: str
    postal_codes: Annotated[list[str], Field(min_length=1)]
    street_patterns: Annotated[list[str], Field(min_length=1)]


class AddressReviewProvenance(BaseModel):
    """Annotator feedback state for the corpus's city-address material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(description="Human-facing review artifact or round")
    status: Literal["annotator_feedback_applied", "pending_review"] = Field(
        description="Whether packet feedback was applied or review is still pending"
    )


class LocaleCorpus(BaseModel):
    """The reviewed per-locale raw material for the identity swap.

    Authored by hand as ``data/tau2/multilingual/<lang>/locale_corpus.yaml``
    (see the ``es`` corpus for a worked example) and validated on load —
    unknown keys are rejected so a typo'd field fails loudly instead of being
    silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    female_first_names: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Female given-name pool. LATIN/ROMANIZED spelling "
            "(diacritics fine): user_ids and email local parts are derived by "
            "romanizing these names, and non-Latin scripts romanize to nothing.",
        ),
    ]
    male_first_names: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Male given-name pool. LATIN/ROMANIZED spelling "
            "(diacritics fine) — see female_first_names.",
        ),
    ]
    last_names: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Family-name pool. LATIN/ROMANIZED spelling "
            "(diacritics fine) — see female_first_names.",
        ),
    ]
    cities: Annotated[
        list[CorpusCity],
        Field(min_length=1, description="City pool for generated addresses"),
    ]
    reviewed_address_cities: Annotated[
        list[ReviewedAddressCity],
        Field(
            description="Human-reviewed city/street/postcode bundles used by "
            "PROSE_NAME_ZIP domains such as retail."
        ),
    ] = Field(default_factory=list)
    email_domain: Annotated[
        str,
        Field(description="Domain for generated caller emails, e.g. 'example.com'"),
    ]
    street_patterns: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Street-address templates; '{n}' is a house number",
        ),
    ] = ["{n} Main St"]
    secondary_line_patterns: Annotated[
        list[str],
        Field(
            description="Templates for the unit within the street address "
            "('Piso {n}', '{n}-ho'); '{n}' is its number. Drawn only for "
            "domains whose records carry a secondary line (retail). Each "
            "pattern must hold EXACTLY ONE number: tasks that turn on the unit "
            "next door move it by one, and a line with two numbers or none has "
            "nothing to move.",
        ),
    ] = Field(default_factory=list)
    secondary_line_number_max: Annotated[
        int,
        Field(
            ge=1,
            default=60,
            description="Largest number inserted into a secondary-line template.",
        ),
    ] = 60
    country: Annotated[
        str,
        Field(description="Country code for the address 'country' field"),
    ] = ""
    zip_format: Annotated[
        Optional[str],
        Field(
            description="Postal-code template; each '#' becomes a digit "
            "(default 5-digit)"
        ),
    ] = None
    phone_number_format: Annotated[
        str,
        Field(
            description="Domestic phone-number template; each '#' becomes a "
            "digit. Used by STRUCTURED_NAME_PHONE domains by default."
        ),
    ]
    address_format: Annotated[
        Optional[AddressProseFormat],
        Field(description="Locale-natural rendering of structured place fields."),
    ] = None
    address_review: Annotated[
        Optional[AddressReviewProvenance],
        Field(description="Review provenance for city-owned address material."),
    ] = None
    native_female_first_names: Annotated[
        dict[str, str],
        Field(
            description="NATIVE-SCRIPT spelling of every female given name, "
            "keyed by its romanized pool entry (e.g. 'Priya' -> 'प्रिया'). "
            "Required (with full pool coverage) only to build the "
            "native-script DB identity variant; empty otherwise."
        ),
    ] = Field(default_factory=dict)
    native_male_first_names: Annotated[
        dict[str, str],
        Field(
            description="NATIVE-SCRIPT spelling of every male given name, "
            "keyed by its romanized pool entry — see native_female_first_names."
        ),
    ] = Field(default_factory=dict)
    native_last_names: Annotated[
        dict[str, str],
        Field(
            description="NATIVE-SCRIPT spelling of every family name, keyed "
            "by its romanized pool entry. Kept as a separate table from the "
            "given names because one romanization can name different native "
            "forms per role (zh 'Yang': surname 杨, given name 阳)."
        ),
    ] = Field(default_factory=dict)
    character_clarifications: Annotated[
        dict[str, str],
        Field(
            description="Curated per-character clarification glosses for a "
            "Han-script locale (closed catalog): character -> the phone-"
            "convention gloss a speaker uses to disambiguate it (张 -> 弓长张, "
            "伟 -> 伟大的伟). Must cover every character of every native name "
            "form; consumed by the native-script identity spell-out payload. "
            "Empty for non-Han locales."
        ),
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _native_name_tables_consistent(self) -> "LocaleCorpus":
        """Native tables, when present, are closed catalogs over the pools.

        Coverage is all-or-nothing per table: a partially-filled table would
        make the native build fail on whichever caller happens to draw the
        missing name, which is the worst possible time to find out.
        """
        from tau2.multilingual.native_script import is_han

        tables = {
            "native_female_first_names": (
                self.native_female_first_names,
                self.female_first_names,
            ),
            "native_male_first_names": (
                self.native_male_first_names,
                self.male_first_names,
            ),
            "native_last_names": (self.native_last_names, self.last_names),
        }
        any_native = any(table for table, _pool in tables.values())
        for field_name, (table, pool) in tables.items():
            if not any_native:
                continue
            missing = sorted(set(pool) - set(table))
            extra = sorted(set(table) - set(pool))
            if missing or extra:
                raise ValueError(
                    f"{field_name} must cover the romanized pool exactly: "
                    f"missing {missing or 'none'}, unknown keys {extra or 'none'}"
                )
            empty = sorted(k for k, v in table.items() if not v.strip())
            if empty:
                raise ValueError(f"{field_name}: empty native form for {empty}")
        # A Han-script native table requires a gloss for every character it
        # can ever emit — the spell-out payload is curated, never improvised.
        han_chars = {
            ch
            for table, _pool in tables.values()
            for native in table.values()
            if is_han(native)
            for ch in native
        }
        missing_glosses = sorted(han_chars - set(self.character_clarifications))
        if missing_glosses:
            raise ValueError(
                "character_clarifications must gloss every character of every "
                f"native name form; missing: {missing_glosses}"
            )
        return self

    @property
    def has_native_name_tables(self) -> bool:
        """Whether the corpus can build the native-script identity variant."""
        return bool(
            self.native_female_first_names
            and self.native_male_first_names
            and self.native_last_names
        )

    def native_name(self, romanized: str, *, pool: str) -> str:
        """The native-script form of one romanized pool name (fail-loud)."""
        table = getattr(self, f"native_{pool}")
        if romanized not in table:
            raise KeyError(
                f"no native-script form for {pool} entry {romanized!r} — the "
                "corpus native tables must cover the pool exactly."
            )
        return table[romanized]

    @model_validator(mode="after")
    def _secondary_lines_hold_one_number(self) -> "LocaleCorpus":
        """Reject a secondary-line pattern with no single number to move."""
        bad = [
            pattern
            for pattern in self.secondary_line_patterns
            if len(re.findall(r"\d+", pattern.format(n=7))) != 1
        ]
        if bad:
            raise ValueError(
                f"secondary_line_patterns {bad!r} must each hold exactly one "
                "number — the unit-next-door tasks shift it by one"
            )
        return self

    @model_validator(mode="after")
    def _phone_number_format_is_valid(self) -> "LocaleCorpus":
        """Require a plausible domestic template for the locale."""
        pattern = self.phone_number_format
        if not re.fullmatch(r"[0-9# ()-]+", pattern):
            raise ValueError(
                "phone_number_format must contain only digits, '#', spaces, "
                "parentheses, or hyphens"
            )
        digit_count = sum(ch.isdigit() or ch == "#" for ch in pattern)
        if pattern.count("#") < 6 or not 7 <= digit_count <= 15:
            raise ValueError(
                "phone_number_format must contain at least six '#' placeholders "
                "and produce 7-15 digits"
            )
        return self

    @model_validator(mode="after")
    def _reviewed_city_addresses_are_complete(self) -> "LocaleCorpus":
        """A reviewed address corpus is city-owned all the way through."""
        if not self.reviewed_address_cities:
            return self
        if self.address_format is None or self.address_review is None:
            raise ValueError(
                "city-owned address material requires address_format and "
                "address_review provenance"
            )
        bad_streets = [
            f"{city.city}: {pattern}"
            for city in self.reviewed_address_cities
            for pattern in city.street_patterns
            if len(re.findall(r"\d+", pattern.format(n=7))) != 1
        ]
        if bad_streets:
            raise ValueError(
                f"city street_patterns must each hold exactly one number: {bad_streets}"
            )
        return self

    @property
    def has_reviewed_city_addresses(self) -> bool:
        """Whether address generation is fully owned by city entries."""
        return bool(self.reviewed_address_cities)

    @model_validator(mode="after")
    def _names_must_romanize(self) -> "LocaleCorpus":
        """Every name must survive romanization — fail LOUDLY at load time.

        ``build_identity`` derives ``user_id`` and the email local part via
        :func:`_romanize`; a name authored in a non-Latin script (Cyrillic,
        Arabic, ...) romanizes to the empty string and would silently produce
        identities like ``__123`` / ``.123@example.com``. Catch it here, at
        corpus load, with the offending entries named.
        """
        bad = [
            name
            for pool in (
                self.female_first_names,
                self.male_first_names,
                self.last_names,
            )
            for name in pool
            if not _romanize(name)
        ]
        if bad:
            raise ValueError(
                f"corpus name(s) {bad!r} romanize to nothing — user_ids and "
                "emails are derived by romanization, so name pools must be "
                "authored in Latin/romanized spelling (diacritics are fine: "
                "'José' works; 'Иван' must be written 'Ivan')."
            )
        return self


def corpus_path(lang: str) -> Path:
    return multilingual_data_dir() / lang / CORPUS_FILENAME


def identity_map_path(lang: str, domain: str) -> Path:
    return multilingual_data_dir() / lang / f"{domain}_identity_map_{lang}.json"


def identity_manifest_path(lang: str, domain: str) -> Path:
    return multilingual_data_dir() / lang / f"{domain}_identity_manifest_{lang}.json"


def gender_sidecar_path(lang: str, domain: str) -> Path:
    return multilingual_data_dir() / lang / f"{domain}_caller_gender_{lang}.json"


def identity_task_set_path(lang: str, domain: str) -> Path:
    return multilingual_data_dir() / lang / f"{domain}_tasks_{lang}_identity.json"


def native_identity_map_path(lang: str, domain: str) -> Path:
    return multilingual_data_dir() / lang / f"{domain}_identity_native_map_{lang}.json"


def native_identity_manifest_path(lang: str, domain: str) -> Path:
    return (
        multilingual_data_dir()
        / lang
        / f"{domain}_identity_native_manifest_{lang}.json"
    )


def native_identity_task_set_path(lang: str, domain: str) -> Path:
    """The ``<domain>_<lang>_identity_native`` task set file (filename
    convention registers it automatically — see tau2.multilingual.task_sets)."""
    return (
        multilingual_data_dir()
        / lang
        / f"{domain}_tasks_{lang}_{NATIVE_IDENTITY_VARIANT}.json"
    )


def localized_task_set_path(lang: str, domain: str) -> Path:
    return multilingual_data_dir() / lang / f"{domain}_tasks_{lang}.json"


def load_corpus(lang: str) -> LocaleCorpus:
    """The reviewed per-locale corpus, or a clear error if missing/invalid."""
    path = corpus_path(lang)
    if not path.exists():
        raise FileNotFoundError(
            f"No locale corpus for '{lang}' at {path}. Author it (gendered name "
            "pools, cities, address/zip/email conventions) — see the es corpus "
            "for the schema."
        )
    try:
        return LocaleCorpus.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid locale corpus:\n{exc}") from exc


def identity_seed(user_id: str) -> int:
    """Process-stable seed for a caller (Python's hash() is salted; SHA-256 is not)."""
    return int(hashlib.sha256(user_id.encode("utf-8")).hexdigest(), 16)


# Latin letters NFKD does NOT decompose (strokes/ligatures), so map them
# explicitly or they'd be dropped (e.g. Łukasz -> ukasz, Đức -> uc). Both
# cases are mapped: fold_to_ascii preserves case; _romanize lowercases first
# and only exercises the lowercase entries.
_TRANSLIT = str.maketrans(
    {
        "ł": "l",
        "Ł": "L",
        "đ": "d",
        "Đ": "D",
        "ø": "o",
        "Ø": "O",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "Th",
        "ß": "ss",
        "ẞ": "Ss",
        "æ": "ae",
        "Æ": "Ae",
        "œ": "oe",
        "Œ": "Oe",
        "ı": "i",
    }
)


def fold_to_ascii(text: str) -> str:
    """Case-preserving ASCII fold for every DISPLAY value entering an identity.

    The DB record, the golden-action arguments derived from it, and the
    agent's tool-call arguments are compared byte-exactly at reward time
    (``Action.compare_with_tool_call``), while in voice runs whether the
    agent's transcription of a spoken name carries the diacritics is an
    orthography coin-flip. The read tools fold at lookup
    (``tau2.utils.text_match.fold_for_match``), but WRITE arguments and
    golden-action matching have no such seam — so the data itself is kept
    plain ASCII: identities are folded here at build time (``Iván Sánchez``
    -> ``Ivan Sanchez``, ``Łukasz`` -> ``Lukasz``, ``Hauptstraße`` ->
    ``Hauptstrasse``), and each diacritic language's
    ``agent_language_clause`` tells the agent the DB is unaccented.

    Corpora stay authored in natural orthography; only built identities fold.
    A value that is still non-ASCII after folding (a non-Latin script the
    corpus validator's romanization check let through, stray punctuation)
    raises rather than shipping a byte the reward comparison can trip on.
    """
    translated = text.translate(_TRANSLIT)
    folded = "".join(
        c
        for c in unicodedata.normalize("NFKD", translated)
        if not unicodedata.combining(c)
    )
    if not folded.isascii():
        residue = sorted({c for c in folded if not c.isascii()})
        raise ValueError(
            f"corpus value {text!r} does not fold to ASCII (residue: {residue!r}) "
            "— identity values must be authored in Latin/romanized spelling."
        )
    return folded


def _romanize(text: str) -> str:
    """ASCII, lowercase, word-chars only — for a Latin user_id / email local part.

    Diacritics are stripped via NFKD (é->e, ş->s); Latin letters NFKD leaves
    intact (ł, đ, ø, ß, ...) are transliterated first so they aren't dropped.
    """
    lowered = text.lower().translate(_TRANSLIT)
    stripped = "".join(
        c
        for c in unicodedata.normalize("NFKD", lowered)
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]+", "", stripped)


def _numeric_suffix(user_id: str) -> str:
    """The trailing numeric run of a user id (kept to preserve uniqueness)."""
    match = re.search(r"(\d+)$", user_id)
    return match.group(1) if match else "0"


def gen_zip(zip_format: str | None, rng: random.Random) -> str:
    """Fill '#' in a postal-code template with digits (default 5-digit)."""
    template = zip_format or "#####"
    return "".join(str(rng.randint(0, 9)) if ch == "#" else ch for ch in template)


def _sha256(path: Path) -> str:
    """SHA-256 of one factory input or output artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    """Best-effort repository HEAD for advisory provenance."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=data_dir(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=data_dir(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class EntityLocalizationManifest(BaseModel):
    """Typed provenance for one emitted locale/domain identity task set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["v1"] = "v1"
    generator_version: str
    language: str
    domain: str
    # Which identity regime the artifacts belong to: the ASCII-folded
    # "identity" sets or the native-script-DB "identity_native" ablation sets.
    variant: Literal["identity", "identity_native"] = "identity"
    git_sha: str
    corpus_sha256: str
    source_tasks_sha256: str
    localized_tasks_sha256: str
    identity_count: int
    task_count: int
    address_review: Optional[AddressReviewProvenance] = None
    output_sha256: dict[str, str]


def build_identity(
    old_user_id: str,
    old_user: dict,
    corpus: LocaleCorpus,
    address2_default: str | None = None,
    *,
    use_reviewed_city_addresses: bool = False,
) -> dict:
    """A deterministic locale identity for one caller (seeded by its user_id).

    The identity's gender is NOT sampled: it is the gender of the caller's
    ENGLISH given name (the first token of the user_id, e.g. ``emma`` in
    ``emma_kim_9957``), resolved through the closed catalog in
    :mod:`tau2.multilingual.factory.name_genders` — fail-loud for uncovered
    names. The locale name is then drawn from the corpus pool of that gender,
    so the caller's gender is stable across the plain and ``_identity`` task
    variants and across languages.

    ``address2_default`` is what the generated address puts on its secondary
    line, and doubles as the switch for whether the domain HAS one. It is a
    property of the DOMAIN's schema, not of the source record: airline's
    ``Address.address2`` is ``Optional[str]`` and its committed maps carry None
    (so no secondary line is drawn and those maps stay byte-identical), but
    retail's is a required ``str`` — a None there fails ``RetailDB`` validation
    the moment the patched record loads, and retail tasks manipulate the unit
    itself, so a real line is drawn from the corpus when it offers one.
    """
    rng = random.Random(identity_seed(old_user_id))
    gender = source_caller_gender(old_user_id.split("_")[0])
    pool = corpus.female_first_names if gender == "female" else corpus.male_first_names
    first = fold_to_ascii(rng.choice(pool))
    last = fold_to_ascii(rng.choice(corpus.last_names))
    suffix = _numeric_suffix(old_user_id)
    new_id = f"{_romanize(first)}_{_romanize(last)}_{suffix}"
    email = f"{_romanize(first)}.{_romanize(last)}{suffix}@{corpus.email_domain}"
    identity = {
        "user_id": new_id,
        "first_name": first,
        "last_name": last,
        "email": email,
        "gender": gender,
    }
    if old_user.get("address"):
        reviewed_address = (
            use_reviewed_city_addresses and corpus.has_reviewed_city_addresses
        )
        city = rng.choice(
            corpus.reviewed_address_cities if reviewed_address else corpus.cities
        )
        street_patterns = (
            city.street_patterns if reviewed_address else corpus.street_patterns
        )
        street_pattern = rng.choice(street_patterns)
        street = fold_to_ascii(street_pattern.format(n=rng.randint(1, 999)))
        postcode = (
            rng.choice(city.postal_codes)
            if reviewed_address
            else gen_zip(corpus.zip_format, rng)
        )
        address = {
            "address1": street,
            "address2": address2_default,
            "city": fold_to_ascii(city.city),
            "state": fold_to_ascii(city.region),
            "country": fold_to_ascii(corpus.country),
            "zip": postcode,
        }
        # Drawn LAST so adding secondary lines to a corpus leaves every other
        # field of every already-committed identity untouched.
        if address2_default is not None and corpus.secondary_line_patterns:
            address["address2"] = fold_to_ascii(
                rng.choice(corpus.secondary_line_patterns).format(
                    n=rng.randint(1, corpus.secondary_line_number_max)
                )
            )
        identity["address"] = address
    return identity


def build_structured_identity(
    english_full_name: str,
    corpus: LocaleCorpus,
    name_order: NameOrder = NameOrder.GIVEN_FIRST,
) -> dict:
    """A deterministic locale identity for one STRUCTURED_NAME_PHONE caller.

    Keyed AND seeded by the English caller's full name (the diversified
    seed-set caller, not the DB canonical record — every seed task patches
    its own caller over C1001), so each of the pool's callers gets one stable
    locale identity per language. The gender is inherited from the English
    given name, exactly like :func:`build_identity`. The name, email, and phone
    number are locale material; customer and line ids and every other record
    field stay canonical, so no address is generated.

    ``full_name`` is the DISPLAY name and follows the locale's ``name_order``
    (family-first for Mandarin/Korean/Vietnamese) — it is what the caller says
    and what the agent's customer record stores, so an exact-match lookup
    succeeds on the name a native speaker actually gives. ``first_name`` /
    ``last_name`` stay ROLES and never reorder, and the email local part stays
    given.family: it is a Latin handle, spelled out letter by letter.
    """
    rng = random.Random(identity_seed(english_full_name))
    gender = source_caller_gender(english_full_name.split()[0])
    pool = corpus.female_first_names if gender == "female" else corpus.male_first_names
    first = fold_to_ascii(rng.choice(pool))
    last = fold_to_ascii(rng.choice(corpus.last_names))
    return {
        "first_name": first,
        "last_name": last,
        "full_name": compose_full_name(first, last, name_order),
        "email": f"{_romanize(first)}.{_romanize(last)}@{corpus.email_domain}",
        "phone_number": gen_phone_number(corpus.phone_number_format, english_full_name),
        "gender": gender,
    }


def gen_phone_number(pattern: str, seed_key: str, attempt: int = 0) -> str:
    """Fill a reviewed locale phone template deterministically."""
    rng = random.Random(identity_seed(f"phone:{seed_key}:{attempt}"))
    return "".join(str(rng.randrange(10)) if ch == "#" else ch for ch in pattern)


def build_identity_map(
    domain: str,
    corpus: LocaleCorpus,
    name_order: NameOrder = NameOrder.GIVEN_FIRST,
) -> dict[str, dict]:
    """``{caller_key: identity}`` for every caller in the domain's task set.

    The caller key follows the domain profile: the original caller user id
    for USER_ID_HANDLE domains, the English caller full name for
    STRUCTURED_NAME_PHONE domains. Identities are made unique across the map:
    on the rare collision
    (same drawn name + same numeric suffix) a disambiguating counter is
    appended to the colliding fields in lockstep, so id and email stay
    mutually consistent.

    ``name_order`` is the locale's display name order; it affects only the
    composed ``full_name`` of STRUCTURED_NAME_PHONE identities (USER_ID_HANDLE
    identities carry structured name roles and no display string).
    """
    identity_kind = _caller_identity_kind(domain)
    db = load_domain_db(domain)
    tasks = load_domain_tasks(domain)
    if identity_kind is CallerIdentityKind.STRUCTURED_NAME_PHONE:
        return _build_structured_identity_map(db, tasks, corpus, name_order)
    users_db = db.get("users") or {}
    callers = sorted(
        {cid for t in tasks if (cid := resolve_caller_user_id(t, users_db))}
    )
    # Retail's Address.address2 is a required str, airline's is Optional[str];
    # see build_identity. Airline keeps None so its committed maps are stable.
    address2_default = (
        "" if identity_kind is CallerIdentityKind.PROSE_NAME_ZIP else None
    )

    identity_map: dict[str, dict] = {}
    used_ids: set[str] = set()
    used_emails: set[str] = set()
    for caller in callers:
        if caller not in db["users"]:
            continue
        identity = build_identity(
            caller,
            db["users"][caller],
            corpus,
            address2_default,
            use_reviewed_city_addresses=(
                identity_kind is CallerIdentityKind.PROSE_NAME_ZIP
            ),
        )
        base_id = identity["user_id"]
        local, _, mail_domain = identity["email"].partition("@")
        bump = 1
        while identity["user_id"] in used_ids or identity["email"] in used_emails:
            bump += 1
            identity["user_id"] = f"{base_id}{bump}"
            identity["email"] = f"{local}{bump}@{mail_domain}"
        used_ids.add(identity["user_id"])
        used_emails.add(identity["email"])
        identity_map[caller] = identity
    return identity_map


def to_native_identity(identity: dict, corpus: LocaleCorpus) -> dict:
    """The native-script rendition of one built (folded) identity record.

    Composes ON TOP of :func:`build_identity`'s deterministic draw, so the
    native identity is the SAME person as the folded one — same user_id, same
    email, same address, same gender, same numeric suffix — with exactly the
    NAME fields flipped to native script and the per-identity spell-out
    payload attached:

    - ``first_name`` / ``last_name``: the curated native form of the drawn
      romanized name (fail-loud on a missing table entry);
    - ``romanized_first_name`` / ``romanized_last_name``: the folded forms,
      kept for provenance (they are what user_id/email were derived from);
    - ``character_clarifications``: the caller's name spell-out payload —
      Han gloss lines from the corpus's curated catalog, or the algorithmic
      Devanagari akshara spell-out (``tau2.multilingual.native_script``).

    Deliberately NOT flipped: user_id and email (Latin handles by real-world
    convention, and spelled letter-by-letter as before) and the address (the
    ablation isolates the NAME-script axis; address material stays the folded
    reviewed-city text the ``_identity`` sets use).
    """
    pool = (
        "female_first_names" if identity["gender"] == "female" else ("male_first_names")
    )
    native_first = corpus.native_name(identity["first_name"], pool=pool)
    native_last = corpus.native_name(identity["last_name"], pool="last_names")
    native = copy.deepcopy(identity)
    native["romanized_first_name"] = identity["first_name"]
    native["romanized_last_name"] = identity["last_name"]
    native["first_name"] = native_first
    native["last_name"] = native_last
    native["character_clarifications"] = native_name_spellout(
        native_first,
        native_last,
        character_clarifications=corpus.character_clarifications,
    )
    return native


def _caller_identity_kind(domain: str) -> CallerIdentityKind:
    """The domain's caller-identity kind (unprofiled test domains = airline's)."""
    from tau2.multilingual.domain_profiles import DOMAIN_PROFILES

    profile = DOMAIN_PROFILES.get(domain)
    return profile.caller_identity if profile else CallerIdentityKind.USER_ID_HANDLE


def check_identity_map_addresses(
    identity_map: dict[str, dict], corpus: LocaleCorpus
) -> list[str]:
    """Verify every generated address belongs to one reviewed city entry."""
    if not corpus.has_reviewed_city_addresses:
        return []
    cities = {fold_to_ascii(city.city): city for city in corpus.reviewed_address_cities}
    problems: list[str] = []
    for caller, identity in identity_map.items():
        address = identity.get("address")
        if not address:
            continue
        city = cities.get(address.get("city"))
        if city is None:
            problems.append(f"identity {caller}: unknown city {address.get('city')!r}")
            continue
        if address.get("state") != fold_to_ascii(city.region):
            problems.append(
                f"identity {caller}: region {address.get('state')!r} does not "
                f"belong to {city.city!r}"
            )
        valid_postcodes = {fold_to_ascii(code) for code in city.postal_codes}
        if address.get("zip") not in valid_postcodes:
            problems.append(
                f"identity {caller}: postcode {address.get('zip')!r} does not "
                f"belong to {city.city!r}"
            )
        street = address.get("address1") or ""
        valid_street = any(
            re.fullmatch(
                re.escape(fold_to_ascii(pattern)).replace(r"\{n\}", r"\d+"),
                street,
            )
            for pattern in city.street_patterns
        )
        if not valid_street:
            problems.append(
                f"identity {caller}: street {street!r} does not belong to {city.city!r}"
            )
    return problems


def english_caller_gender(
    task: dict,
    profile: DomainLocalizationProfile,
    db: Optional[dict] = None,
) -> Optional[str]:
    """The gender of a task's ENGLISH caller, by the profile's identity kind.

    The single reading of "who is calling, and what gender is their name" that
    every gender consumer shares: the locale identities of
    :func:`build_identity` / :func:`build_structured_identity` inherit it, the
    seed split's English arm sidecar is built from it, and so is the English
    sidecar of a domain seed-tasks cannot seed. Resolved through the closed
    catalog in :mod:`tau2.multilingual.factory.name_genders`, which is
    fail-loud for uncovered names.

    None when the task names no caller (the voice sampler then keeps its
    balanced persona rotation) — never a silent gender guess.

    ``db`` is the domain DB, needed only by ``PROSE_NAME_ZIP`` domains: a
    retail caller anchored by name+zip or email carries no handle in the
    prose, so the given name comes from the resolved DB record. Omit it for
    the handle/structured kinds, which read the caller off the task alone.
    """
    if profile.caller_identity is CallerIdentityKind.STRUCTURED_NAME_PHONE:
        info = caller_set_user_info(task)
        if info is None:
            return None
        return source_caller_gender(info["name"].split()[0])
    users_db = (db or {}).get("users") or {}
    user_id = resolve_caller_user_id(task, users_db)
    if user_id is None:
        return None
    # The DB record is authoritative when we have it: a caller resolved by
    # name+zip has no handle to split, and even for handle callers the record
    # is the same name the handle encodes.
    record_name = (users_db.get(user_id) or {}).get("name") or {}
    given = record_name.get("first_name") or user_id.split("_")[0]
    return source_caller_gender(given)


def build_english_gender_sidecar(domain: str) -> dict[str, str]:
    """``{task_id: gender}`` for a domain's COMMITTED English arm file.

    The English counterpart of the localized sidecar written by
    :func:`localize_entities`. It exists because the voice sampler pins the
    caller's persona gender only for task ids a sidecar covers: without one
    an English arm falls back to the domain's pre-sampled ``tasks_voice.json``
    personas, whose gender is uncorrelated with the caller's name (measured
    25/50 contradictory on airline).

    The sidecar is WRITTEN by whichever verb emits the English arm — the
    shared emitter in ``tau2.multilingual.factory.task_arms``, reached through
    ``tau2 factory seed-tasks`` (generated-pool domains) or ``tau2 factory
    arm-tasks`` (curated-source domains). This function recomputes it from the
    committed arm file instead, which is what lets a test assert the two
    derivations agree — a stale sidecar cannot outlive its arm file.

    Raises:
        FileNotFoundError: The domain has no English arm task file.
    """
    profile = get_domain_profile(domain)
    arm_path = multilingual_data_dir() / "en" / f"{domain}_tasks_en.json"
    if not arm_path.exists():
        raise FileNotFoundError(
            f"No English arm task set at {arm_path} — the sidecar keys its "
            "task ids, so the arm file must exist first."
        )
    tasks = json.loads(arm_path.read_text())
    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]
    db = load_domain_db(domain)
    gender: dict[str, str] = {}
    for task in tasks:
        caller_gender = english_caller_gender(task, profile, db)
        if caller_gender is None:
            continue
        gender[task["id"]] = caller_gender
    return gender


def _caller_key(
    task: dict, db: dict, profile: DomainLocalizationProfile
) -> Optional[str]:
    """The identity-map key of a task's caller (profile-kind dispatch)."""
    if profile.caller_identity is CallerIdentityKind.STRUCTURED_NAME_PHONE:
        info = caller_set_user_info(task)
        if info is None:
            return None
        # The phone must still resolve to a canonical DB customer (validity
        # check), but the KEY is the English caller name the task announces —
        # diversified seed tasks all share C1001's phone.
        if customer_by_phone(db, info["phone_number"]) is None:
            return None
        return info["name"]
    return resolve_caller_user_id(task, db.get("users") or {})


def _build_structured_identity_map(
    db: dict,
    tasks: list[dict],
    corpus: LocaleCorpus,
    name_order: NameOrder = NameOrder.GIVEN_FIRST,
) -> dict[str, dict]:
    """``{english_full_name: identity}`` for every caller the task set references.

    Callers are discovered through each task's ``set_user_info`` action —
    the NAME is the key (the diversified seed set spreads a caller pool over
    one canonical phone/customer record, so the customer id no longer
    identifies a caller); the phone must still resolve to a DB customer, as a
    validity check. Emails and phone numbers are made unique across the map.
    """
    callers: set[str] = set()
    for task in tasks:
        info = caller_set_user_info(task)
        if info is None:
            continue
        if customer_by_phone(db, info["phone_number"]) is not None:
            callers.add(info["name"])

    identity_map: dict[str, dict] = {}
    used_emails: set[str] = set()
    used_phones: set[str] = set()
    for english_full_name in sorted(callers):
        identity = build_structured_identity(english_full_name, corpus, name_order)
        local, _, mail_domain = identity["email"].partition("@")
        bump = 1
        while identity["email"] in used_emails:
            bump += 1
            identity["email"] = f"{local}{bump}@{mail_domain}"
        used_emails.add(identity["email"])
        attempt = 0
        while identity["phone_number"] in used_phones:
            attempt += 1
            identity["phone_number"] = gen_phone_number(
                corpus.phone_number_format, english_full_name, attempt
            )
        used_phones.add(identity["phone_number"])
        identity_map[english_full_name] = identity
    return identity_map


def localize_entities(
    lang: str,
    domain: str = DEFAULT_MULTILINGUAL_DOMAIN,
    *,
    script_code: str | None = None,
    write: bool = True,
    native_script: bool = False,
) -> tuple[list[dict], list[str], dict[str, dict]]:
    """Build + (optionally) write the identity-variant artifacts for a language.

    Reads the reviewed corpus and the domain's ``<domain>_tasks_<lang>`` set
    (the English-prose arm emitted by ``arm-tasks``/``seed-tasks``; translated
    prose only for the retired ``tasks_translated`` regime), builds the
    deterministic identity map, derives the ``_identity`` task set,
    and writes: the identity map, the ``_identity`` tasks, and the caller-gender
    sidecar ``{task_id: gender}`` covering EVERY localized task id — both the
    plain ``<id>_<lang>`` and the ``<id>_<lang>_identity`` variant (for voice
    routing; the gender is the same for both, inherited from the English source
    caller's name).

    Returns ``(identity_tasks, problems, identity_map)``; ``problems`` is empty iff
    the derived set is invariant-clean. Nothing is written when ``problems`` is
    non-empty (or ``write=False``).

    The identity map is built from the domain DB and the localized arm is
    rewritten through ``derive_identity_tasks``.

    ``native_script=True`` builds the NATIVE-SCRIPT DB ablation variant
    instead (romanization x localization 2x2, config A): the same
    deterministic identity draw, with the name fields flipped to the corpus's
    curated native forms plus a per-identity spell-out payload (see
    :func:`to_native_identity`), emitted as ``<domain>_<lang>_identity_native``
    with its own map and manifest. Requires the corpus's native name tables;
    supported for the string-patch USER_ID_HANDLE / PROSE_NAME_ZIP shapes only
    (the structured telecom shape stays folded until that experiment exists).
    """
    profile = get_domain_profile(domain)
    if not profile.identity_swap_supported:
        raise ValueError(
            f"Domain '{domain}' does not support the locale identity swap "
            "(DomainLocalizationProfile.identity_swap_supported is False)."
        )
    corpus = load_corpus(lang)
    if native_script and not corpus.has_native_name_tables:
        raise ValueError(
            f"Native-script identity localization for '{lang}' requires the "
            "curated native name tables (native_female_first_names / "
            "native_male_first_names / native_last_names) in its locale corpus."
        )
    name_order = name_order_for_language(lang)
    localized_path = localized_task_set_path(lang, domain)
    if not localized_path.exists():
        if profile.tasks_translated:
            producer = (
                "the retired task-translation loop (deleted; the localized set "
                "must already be on disk)"
            )
        elif profile.curated_source:
            producer = f"`tau2 factory arm-tasks --domain {domain} --lang {lang}`"
        else:
            producer = f"`tau2 factory seed-tasks --domain {domain} --lang {lang}`"
        raise FileNotFoundError(
            f"No localized task set at {localized_path}; run {producer} first."
        )

    if profile.identity_swap_supported:
        if (
            profile.caller_identity is CallerIdentityKind.PROSE_NAME_ZIP
            and not corpus.has_reviewed_city_addresses
        ):
            raise ValueError(
                f"Retail identity localization for '{lang}' requires reviewed "
                "city-owned postal_codes and street_patterns in its locale corpus."
            )
        identity_map = build_identity_map(domain, corpus, name_order)
        address_problems = (
            check_identity_map_addresses(identity_map, corpus)
            if profile.caller_identity is CallerIdentityKind.PROSE_NAME_ZIP
            else []
        )
        if (
            native_script
            and profile.caller_identity is CallerIdentityKind.STRUCTURED_NAME_PHONE
        ):
            raise ValueError(
                f"Domain '{domain}' anchors its caller by STRUCTURED_NAME_PHONE: "
                "the native-script DB variant is defined for the string-patch "
                "user-record shapes only (retail/airline). Extend the "
                "structured deriver before building it for this domain."
            )
        if native_script:
            identity_map = {
                caller: to_native_identity(identity, corpus)
                for caller, identity in identity_map.items()
            }
        localized_tasks = json.loads(localized_path.read_text())
        source_tasks = load_domain_tasks(domain)
        db = load_domain_db(domain)
        variant = NATIVE_IDENTITY_VARIANT if native_script else "identity"
        identity_tasks, problems = derive_identity_tasks(
            localized_tasks,
            source_tasks,
            db,
            identity_map,
            lang,
            resolve_script_code(lang, script_code),
            domain=domain,
            name_order=name_order,
            address_format=corpus.address_format,
            variant=variant,
        )
        problems = address_problems + problems
        identity_map_payload = identity_map
        identity_count = len(identity_map)
        generator_version = (
            NATIVE_IDENTITY_LOCALIZATION_VERSION
            if native_script
            else IDENTITY_LOCALIZATION_VERSION
        )
        # Caller-gender sidecar keyed by EVERY localized task id the voice
        # sampler can see — the plain set (English caller names) and the
        # _identity set (locale names) share one gender per task, inherited
        # from the English source caller's name. The native variant writes NO
        # sidecar of its own: its gender is the same per task by construction,
        # and caller_gender_for_task resolves a ``*_identity_native`` id by
        # stripping the ``_native`` tail — so the plain sidecar must already
        # cover the ``_identity`` ids (fail here, not mid-run).
        gender = {}
        if native_script:
            existing_sidecar = load_gender_sidecar(lang, domain)
            for task in identity_tasks:
                fallback_id = task["id"].removesuffix("_native")
                if fallback_id not in existing_sidecar:
                    problems.append(
                        f"task {task['id']}: no caller-gender sidecar entry for "
                        f"'{fallback_id}' — run the plain identity build "
                        f"(localize-entities without --native-script) first."
                    )
        else:
            for localized_task in localized_tasks:
                caller = _caller_key(localized_task, db, profile)
                if caller in identity_map:
                    source_id = localized_task["id"].removesuffix(f"_{lang}")
                    caller_gender = identity_map[caller]["gender"]
                    gender[f"{source_id}_{lang}"] = caller_gender
                    gender[f"{source_id}_{lang}_identity"] = caller_gender

    if write and not problems:
        _write_identity_artifacts(
            lang,
            domain,
            profile,
            corpus,
            localized_path,
            identity_tasks,
            identity_map_payload,
            gender,
            identity_count,
            generator_version,
            native_script=native_script,
        )
    return identity_tasks, problems, identity_map_payload


def _write_json_stable(path: Path, payload) -> None:
    """Write a JSON artifact only when its bytes change (idempotent reruns)."""
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_identity_artifacts(
    lang: str,
    domain: str,
    profile: DomainLocalizationProfile,
    corpus: LocaleCorpus,
    localized_path: Path,
    identity_tasks: list[dict],
    identity_map_payload: dict,
    gender: dict[str, str],
    identity_count: int,
    generator_version: str,
    *,
    native_script: bool = False,
) -> None:
    """The single writer of the identity artifacts (map, tasks, gender
    sidecar, manifest). Every write is idempotent: unchanged content leaves
    the file untouched, and an unchanged manifest keeps its original git_sha
    (the seed-tasks precedent — provenance records what PRODUCED the bytes).

    The native-script variant writes its own map/tasks/manifest under the
    ``identity_native`` names and NO gender sidecar (the plain sidecar covers
    its ids via the ``_native``-strip fallback in ``caller_gender_for_task``)."""
    if native_script:
        map_path = native_identity_map_path(lang, domain)
        task_path = native_identity_task_set_path(lang, domain)
    else:
        map_path = identity_map_path(lang, domain)
        task_path = identity_task_set_path(lang, domain)
    sidecar_path = gender_sidecar_path(lang, domain)
    _write_json_stable(map_path, identity_map_payload)
    _write_json_stable(task_path, identity_tasks)
    output_sha256 = {
        map_path.name: _sha256(map_path),
        task_path.name: _sha256(task_path),
    }
    if not native_script:
        _write_json_stable(sidecar_path, gender)
        output_sha256[sidecar_path.name] = _sha256(sidecar_path)
    source_path = (
        data_dir() / "tau2" / "domains" / domain / profile.source_tasks_filename
    )
    manifest = EntityLocalizationManifest(
        generator_version=generator_version,
        language=lang,
        domain=domain,
        variant=NATIVE_IDENTITY_VARIANT if native_script else "identity",
        git_sha=_git_sha(),
        corpus_sha256=_sha256(corpus_path(lang)),
        source_tasks_sha256=_sha256(source_path),
        localized_tasks_sha256=_sha256(localized_path),
        identity_count=identity_count,
        task_count=len(identity_tasks),
        address_review=corpus.address_review,
        output_sha256=output_sha256,
    )
    manifest_path = (
        native_identity_manifest_path(lang, domain)
        if native_script
        else identity_manifest_path(lang, domain)
    )
    manifest_dump = manifest.model_dump(mode="json")
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            existing = None
        if existing is not None and {
            k: v for k, v in existing.items() if k != "git_sha"
        } == {k: v for k, v in manifest_dump.items() if k != "git_sha"}:
            existing_unchanged = True
        else:
            existing_unchanged = False
    else:
        existing_unchanged = False
    if not existing_unchanged:
        _write_json_stable(manifest_path, manifest_dump)
    if native_script:
        logger.info(
            f"native-script identity artifacts for {lang}/{domain}: "
            f"{len(identity_tasks)} tasks, {identity_count} identities "
            "(gender rides the plain sidecar via the _native-strip fallback)"
        )
        return
    split = {
        g: sum(1 for t in identity_tasks if gender.get(t["id"]) == g)
        for g in ("male", "female")
    }
    logger.info(
        f"caller-gender sidecar for {lang}/{domain}: {len(gender)} task ids, "
        f"task-level split male={split['male']} female={split['female']}"
    )


# Sidecar cache keyed by (path, mtime_ns): the voice sampler asks for the
# caller gender once per task, and re-reading the same JSON file per task is
# pure waste. Keying by mtime keeps the cache correct when the factory
# rewrites the sidecar in-process (localize-entities then a run) and when
# tests repoint DATA_DIR — no explicit invalidation hook needed.
_GENDER_SIDECAR_CACHE: dict[Path, tuple[int, dict[str, str]]] = {}


def load_gender_sidecar(
    lang: str, domain: str = DEFAULT_MULTILINGUAL_DOMAIN
) -> dict[str, str]:
    """``{localized_task_id: gender}`` for voice routing (empty if none written).

    Covers both the plain ``<id>_<lang>`` and the ``<id>_<lang>_identity`` task
    ids of every localized task. Cached per file mtime — callers must treat the
    returned mapping as read-only.
    """
    path = gender_sidecar_path(lang, domain)
    if not path.exists():
        return {}
    mtime_ns = path.stat().st_mtime_ns
    cached = _GENDER_SIDECAR_CACHE.get(path)
    if cached is not None and cached[0] == mtime_ns:
        return cached[1]
    data = json.loads(path.read_text())
    _GENDER_SIDECAR_CACHE[path] = (mtime_ns, data)
    return data


def caller_gender_for_task(
    task_id: str, lang: str, domain: str = DEFAULT_MULTILINGUAL_DOMAIN
) -> str | None:
    """The caller's gender for a localized task id, for voice routing.

    Applies to EVERY localized task — plain, ``_identity``, and
    ``_identity_native`` variants alike. The native-script variant carries no
    sidecar of its own (the gender is the same person's, inherited from the
    English source caller), so a ``*_identity_native`` id resolves through its
    ``*_identity`` sibling. None when no sidecar covers the id (the voice
    sampler then falls back to its balanced persona rotation).
    """
    sidecar = load_gender_sidecar(lang, domain)
    gender = sidecar.get(task_id)
    if gender is None and task_id.endswith(f"_{NATIVE_IDENTITY_VARIANT}"):
        gender = sidecar.get(task_id.removesuffix("_native"))
    return gender
