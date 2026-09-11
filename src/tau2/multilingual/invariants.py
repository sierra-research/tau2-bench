# Copyright Sierra
"""Localization invariants — the single implementation.

Shared by the parametrized test suite
(``tests/test_multilingual/test_task_localization_invariants.py``) and the
translator round-trip tool (`the retired translator CSV round-trip`) so the rules and
their enforcement stay in sync. Every checker returns a list of human-readable
problem strings (empty == pass) instead of raising.

The invariants (see docs/multilingual/ADDING_A_LANGUAGE.md):
- every concrete value (emails, user ids, reservation/flight codes, every
  numeric run) in the English instructions appears verbatim in the localized
  instructions;
- ``evaluation_criteria`` is byte-identical to the English source — except
  for identity-variant sets (tasks that rename the caller via
  ``initialization_data``), where it must equal the source modulo exactly
  that rename;
- prose is actually localized (contains the pack's declared script);
- values stay Latin (no localized digit forms).
"""

import copy
import re
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.multilingual.domain_profiles import CallerIdentityKind
from tau2.multilingual.names import (
    NameOrder,
    compose_full_name,
    name_order_for_language,
    split_full_name,
)

# ISO 15924 script code -> regex matching one character of that script.
# Extend deliberately when onboarding a language with a new script.
#
# These are PRESENCE checks (>= 1 char of the expected script), not purity
# checks: other characters are allowed, so Latin-script languages with
# diacritics (Polish ł, Turkish ı/ş/ğ, Vietnamese tone marks) pass under "latn".
SCRIPT_RANGES: dict[str, str] = {
    "deva": r"[ऀ-ॿ]",  # Devanagari
    "hans": r"[一-鿿]",  # Han (Simplified)
    "hant": r"[一-鿿]",  # Han (Traditional)
    "arab": r"[؀-ۿ]",  # Arabic
    "cyrl": r"[Ѐ-ӿ]",  # Cyrillic
    "hang": r"[가-힯]",  # Hangul
    "jpan": r"[぀-ヿ一-鿿]",  # Japanese (kana + kanji)
    "thai": r"[฀-๿]",  # Thai
    "latn": r"[A-Za-z]",  # Latin
}

# Digit forms that must never replace Latin digits in values.
LOCALIZED_DIGIT_RES: dict[str, re.Pattern] = {
    "deva": re.compile(r"[०-९]"),
    "hans": re.compile(r"[０-９]"),  # fullwidth digits
    "hant": re.compile(r"[０-９]"),
    "arab": re.compile(r"[٠-٩]"),
}

# Concrete-value extractors for the English source instructions.
VALUE_PATTERNS = [
    # Emails.
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    # User ids, e.g. anya_garcia_5901 (also matches quoted daiki_muller_1116).
    re.compile(r"\b[a-z]+_[a-z]+_\d+\b"),
    # Uppercase alphanumeric codes containing a digit: reservation ids
    # (JMO1MG, 1N99U6), flight numbers (HAT169), etc.
    re.compile(r"\b(?=[A-Z0-9]*\d)[A-Z][A-Z0-9]{2,}\b|\b\d[A-Z0-9]{2,}\b"),
    # Numbers: amounts, dates, last-4 digits, DOBs (each numeric run must
    # survive verbatim, e.g. 2001-04-12 -> 2001, 04, 12).
    re.compile(r"\d+"),
]

INSTRUCTION_FIELDS = [
    "task_instructions",
    "reason_for_call",
    "known_info",
    "unknown_info",
]

# Fields a localization may change relative to its source task. Everything
# else (initial_state excepted for identity variants) must be untouched.
TRANSLATABLE_FIELDS = INSTRUCTION_FIELDS + ["purpose", "notes", "relevant_policies"]


def script_regex(script_code: str) -> re.Pattern:
    if script_code not in SCRIPT_RANGES:
        raise ValueError(
            f"No Unicode range registered for ISO 15924 script '{script_code}'. "
            "Extend SCRIPT_RANGES in tau2.multilingual.invariants."
        )
    return re.compile(SCRIPT_RANGES[script_code])


def instruction_text(task: dict) -> str:
    instructions = task["user_scenario"]["instructions"]
    return "\n".join(instructions.get(field) or "" for field in INSTRUCTION_FIELDS)


def extract_values(text: str) -> set[str]:
    """Concrete values that must survive a translation verbatim."""
    values: set[str] = set()
    for pattern in VALUE_PATTERNS:
        values.update(pattern.findall(text))
    # Drop pure-English words accidentally caught by the code pattern
    # (e.g. "DOB", "NOT"): keep only tokens containing a digit or '@'.
    return {v for v in values if any(ch.isdigit() for ch in v) or "@" in v}


# The caller's user id as it appears in known_info, e.g. emma_kim_9957.
# SINGLE implementation: the derive-names engine (localize_lib), the identity
# swap (factory.entity_localization), and the invariant checkers all extract
# the caller through here.
# ASCII lookarounds, not \b: in agglutinative scripts the id is legitimately
# followed by a letter with no space (ko "..._5188이다"), and Hangul/CJK are
# \w so \b never fires there.
CALLER_USER_ID_RE = re.compile(r"(?<![A-Za-z0-9_])[a-z]+_[a-z]+_\d+(?![A-Za-z0-9_])")


def find_caller_user_id(task: Optional[dict]) -> Optional[str]:
    """The caller's user id, read from ``known_info`` (None if absent).

    This is the derive-names convention for ``USER_ID_HANDLE`` domains
    (airline): every task of a localized set names its caller's user id in
    ``known_info`` (a hard invariant of the source domains), so identity/name
    maps can be keyed by caller user id.
    """
    if not task:
        return None
    known_info = ((task.get("user_scenario") or {}).get("instructions") or {}).get(
        "known_info"
    ) or ""
    match = CALLER_USER_ID_RE.search(known_info)
    return match.group(0) if match else None


# The other two lookup keys a PROSE_NAME_ZIP caller may be anchored by, since
# the retail policy authenticates "via email, or via name + zip code" and each
# task's author used whichever they preferred.
CALLER_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CALLER_ZIP_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")
# Capitalized bigrams as name CANDIDATES, not as a parse of the sentence. The
# retail prose introduces the caller a dozen different ways ("You are X",
# "Your name is X", "You're X", "an interesting guy called X"), so matching
# idioms means an open-ended regex that the next task phrasing breaks. Reading
# every bigram and letting the DB adjudicate is closed and phrasing-proof:
# a wrong candidate simply misses the (name, zip) index.
CALLER_NAME_BIGRAM_RE = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")


def _known_info(task: Optional[dict]) -> str:
    if not task:
        return ""
    return ((task.get("user_scenario") or {}).get("instructions") or {}).get(
        "known_info"
    ) or ""


def resolve_caller_user_id(
    task: Optional[dict], users_db: Optional[dict]
) -> Optional[str]:
    """The caller's user id, resolved against the domain's ``users`` mapping.

    The ``PROSE_NAME_ZIP`` (retail) counterpart of :func:`find_caller_user_id`:
    the caller is anchored ONLY in free ``known_info`` prose, by whichever
    lookup key the task author wrote. Keys are tried in precedence order and
    each must resolve to EXACTLY ONE db user, else the next is tried:

    1. a ``first_last_1234`` handle that is a real user id (23 retail tasks);
    2. an email address (18) — every email in the prose is tried, because a
       caller may recite two ("you have two emails: ... and ...");
    3. a capitalized name bigram plus a 5-digit zip (73), the dominant shape.

    Ambiguity yields None rather than a guess: a name+zip pair matching two
    users is exactly the case the retail agent itself cannot authenticate.

    With ``users_db`` None or empty this degrades to :func:`find_caller_user_id`,
    so ``USER_ID_HANDLE`` domains and db-less test fixtures are unaffected —
    airline's handle is a hard invariant of its source set, so the first key
    always hits there and the fallbacks never run.
    """
    handle = find_caller_user_id(task)
    if not users_db:
        return handle
    if handle is not None and handle in users_db:
        return handle
    known_info = _known_info(task)

    emails = {e.lower() for e in CALLER_EMAIL_RE.findall(known_info)}
    if emails:
        matched = {
            user_id
            for user_id, user in users_db.items()
            if (user.get("email") or "").lower() in emails
        }
        if len(matched) == 1:
            return next(iter(matched))

    zips = set(CALLER_ZIP_RE.findall(known_info))
    names = {
        (first.lower(), last.lower())
        for first, last in CALLER_NAME_BIGRAM_RE.findall(known_info)
    }
    if zips and names:
        matched = {
            user_id
            for user_id, user in users_db.items()
            if (
                (
                    (user.get("name") or {}).get("first_name", "").lower(),
                    (user.get("name") or {}).get("last_name", "").lower(),
                )
                in names
                and ((user.get("address") or {}).get("zip") or "") in zips
            )
        }
        if len(matched) == 1:
            return next(iter(matched))
    return handle


def caller_set_user_info(task: Optional[dict]) -> Optional[dict]:
    """``{name, phone_number}`` from the ``set_user_info`` init action, or None.

    The ``STRUCTURED_NAME_PHONE`` counterpart of :func:`find_caller_user_id`
    (telecom): the caller's identity is carried as structured arguments of the
    ``set_user_info`` initialization action, and the prose repeats it.
    """
    if not task:
        return None
    actions = (task.get("initial_state") or {}).get("initialization_actions") or []
    for action in actions:
        if (action or {}).get("func_name") != "set_user_info":
            continue
        arguments = action.get("arguments") or {}
        if arguments.get("name") and arguments.get("phone_number"):
            return {
                "name": arguments["name"],
                "phone_number": arguments["phone_number"],
            }
    return None


def phone_digits(phone_number: str) -> str:
    """The digit sequence of a phone number (matching ignores formatting)."""
    return re.sub(r"\D", "", phone_number or "")


def _split_full_name(full_name: str) -> tuple[str, str]:
    """(given, family) ROLES from an English-ordered full-name string.

    Feeds structured ``{first_name, last_name}`` swaps in evaluation criteria,
    never display order — display names are carried verbatim on
    :class:`IdentityRename`, so no caller has to know how a locale writes them.
    """
    return split_full_name(full_name)


def _identity_patch_records(localized_task: dict) -> dict[str, dict]:
    """``{user_id: patch_record}`` for caller patches that carry a ``name``.

    Reads ``initial_state.initialization_data.agent_data.users`` and returns the
    whole patched user record (so the caller identity — name, email — can be
    read from it); missing layers yield an empty mapping rather than raising.
    """
    initial_state = localized_task.get("initial_state") or {}
    init_data = initial_state.get("initialization_data") or {}
    agent_data = init_data.get("agent_data") or {}
    return {
        user_id: (patch or {})
        for user_id, patch in (agent_data.get("users") or {}).items()
        if (patch or {}).get("name")
    }


def effective_agent_records(
    task: Optional[dict], domain_db: dict, collection: str
) -> list[dict]:
    """A top-level agent-data list a task actually establishes at run time.

    Addict-style list patches replace the domain DB's list wholesale. Consumers
    therefore read a task's own list when present and fall back to the DB only
    otherwise.
    """
    patched = (
        ((task or {}).get("initial_state") or {}).get("initialization_data", {}) or {}
    ).get("agent_data", {}) or {}
    records = patched.get(collection)
    if records:
        return records
    return domain_db.get(collection) or []


def effective_customer_records(task: Optional[dict], domain_db: dict) -> list[dict]:
    """The customers list a task actually establishes at run time."""
    return effective_agent_records(task, domain_db, "customers")


def _customer_identity_patches(localized_task: dict) -> dict[str, dict]:
    """``{customer_id: patched_customer}`` from a customers-list patch.

    The STRUCTURED_NAME_PHONE (telecom) identity-patch shape: the domain DB
    keys customers by a top-level ``customers`` LIST, so the ``agent_data``
    patch replaces the whole list (list patches are wholesale) and the caller
    rename is read off the entry whose ``full_name`` changed. Missing layers
    yield an empty mapping rather than raising.
    """
    initial_state = localized_task.get("initial_state") or {}
    init_data = initial_state.get("initialization_data") or {}
    agent_data = init_data.get("agent_data") or {}
    return {
        customer["customer_id"]: customer
        for customer in (agent_data.get("customers") or [])
        if (customer or {}).get("customer_id") and customer.get("full_name")
    }


_ADDRESS_FIELDS = ("address1", "address2", "city", "state", "country", "zip")

# The fields that say WHICH address this is; the street lines say where in it.
_ADDRESS_PLACE_FIELDS = ("city", "state", "country", "zip")

_DIGIT_RUN_RE = re.compile(r"\d+")


class AddressProseFormat(BaseModel):
    """Locale-owned rendering for structured city/region/country fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    city_region: str = Field(
        default="{city}, {region}",
        description="Format used when source prose names city and region.",
    )
    city_country: str = Field(
        default="{city}, {country}",
        description="Format used when source prose names city and country.",
    )
    full_address: Optional[str] = Field(
        default=None,
        description="Optional locale order for a complete caller address.",
    )

    @model_validator(mode="after")
    def _templates_use_address_fields(self) -> "AddressProseFormat":
        allowed = {"city", "region", "country"}
        for name, template, required in (
            ("city_region", self.city_region, {"city", "region"}),
            ("city_country", self.city_country, {"city", "country"}),
        ):
            fields = set(re.findall(r"{([^{}]+)}", template))
            if not required <= fields or not fields <= allowed:
                raise ValueError(
                    f"{name} must use {sorted(required)} and only "
                    f"{sorted(allowed)}; got {template!r}"
                )
            try:
                template.format(city="City", region="Region", country="Country")
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid {name} template {template!r}") from exc
        if self.full_address is not None:
            allowed = {
                "address1",
                "address2",
                "city",
                "region",
                "country",
                "postcode",
            }
            fields = set(re.findall(r"{([^{}]+)}", self.full_address))
            required = {"address1", "city", "postcode"}
            if not required <= fields or not fields <= allowed:
                raise ValueError(
                    "full_address must use address1, city, postcode and only "
                    f"{sorted(allowed)}; got {self.full_address!r}"
                )
            try:
                self.full_address.format(
                    address1="Street 1",
                    address2="Unit 2",
                    city="City",
                    region="Region",
                    country="Country",
                    postcode="12345",
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"invalid full_address template {self.full_address!r}"
                ) from exc
        return self


# US state code -> spelled-out name. The source domains are all US-based, and
# their prose sometimes names the caller's state in full where the structured
# record only carries the code ("you live in Texas in zipcode 76171"). A
# localized caller does not live in Texas, so the swap has to recognize the
# long form; the code alone is two letters that collide with ordinary words
# and is never substituted on its own.
US_STATE_NAMES: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

US_CITY_ALIASES: dict[str, tuple[str, ...]] = {
    "New York": ("Big Apple", "NYC"),
}


def digit_run_delta(old: str, new: str) -> Optional[int]:
    """How far ``new``'s number moved from ``old``'s, or None if incomparable.

    Both strings must carry exactly one digit run and be otherwise identical:
    '443 Maple Drive' -> '445 Maple Drive' is +2. Anything else (a different
    street, two numbers, no number) returns None — this recognizes a
    one-number edit, not an arbitrary address difference.
    """
    old_runs, new_runs = _DIGIT_RUN_RE.findall(old), _DIGIT_RUN_RE.findall(new)
    if len(old_runs) != 1 or len(new_runs) != 1:
        return None
    if _DIGIT_RUN_RE.sub("#", old) != _DIGIT_RUN_RE.sub("#", new):
        return None
    return int(new_runs[0]) - int(old_runs[0])


def shift_digit_run(text: str, delta: int) -> Optional[str]:
    """``text`` with its one digit run moved by ``delta``, or None.

    Zero padding is preserved ('007' + 2 -> '009'), and a shift that would go
    negative is refused rather than wrapped.
    """
    runs = _DIGIT_RUN_RE.findall(text)
    if len(runs) != 1:
        return None
    shifted = int(runs[0]) + delta
    if shifted < 0:
        return None
    return _DIGIT_RUN_RE.sub(str(shifted).zfill(len(runs[0])), text, count=1)


def caller_address_variant(
    source_task: Optional[dict], address: Optional[dict]
) -> Optional[dict]:
    """The near-miss of the caller's own address a task deliberately uses.

    Some retail tasks turn on an address that is the caller's EXCEPT for its
    street number: task 41/42's caller mistyped 443 Maple Drive as 445 and
    wants it corrected, and the one-digit gap IS the task. Such an address is
    still identity material — it has to move to the locale with the caller,
    carrying the same gap — but it is not the caller's address on file, so the
    whole-dict swap never matches it.

    Recognized only where the place fields (city/state/country/zip) all match
    the caller's and address1 differs by a single number. A unit-only change
    is a destination instruction (task 17's Suite 641), not an identity
    correction, and stays canonical. An
    address the caller merely dictates as a new destination ("1234 Elm St,
    Springfield, IL") differs in its place fields and is left canonical, as
    order ids and prices are.
    """
    if not source_task or not address:
        return None
    for node in _walk_dicts(source_task.get("evaluation_criteria") or {}):
        if not all(field in node for field in ("address1", "city", "zip")):
            continue
        if any(
            (node.get(field) or "") != (address.get(field) or "")
            for field in _ADDRESS_PLACE_FIELDS
        ):
            continue
        differing = [
            field
            for field in ("address1", "address2")
            if (node.get(field) or "") != (address.get(field) or "")
        ]
        if differing != ["address1"]:
            continue
        if digit_run_delta(address.get("address1") or "", node["address1"]) is None:
            continue
        return {key: node.get(key) for key in _ADDRESS_FIELDS}
    return None


def _walk_dicts(node) -> Iterator[dict]:
    """Every dict nested anywhere inside ``node``, outermost first."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_dicts(item)


# An email as retail prose writes one. Deliberately narrower than the greedy
# VALUE_PATTERNS extractor: this one must yield a clean address (no trailing
# punctuation) because its matches become replacement literals.
_PROSE_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z]{2,})+")


def variant_email(old_email: str, old_variant: str, new_email: str) -> Optional[str]:
    """The near-miss of ``new_email`` carrying ``old_variant``'s gap to ``old_email``.

    Some retail tasks recite a deliberately-wrong email next to (or instead
    of) the caller's real one — a decoy the task turns on, exactly as the
    mistyped street number is for addresses (:func:`caller_address_variant`).
    The decoy is identity material: leaving it behind ships prose built from
    the SOURCE caller's name ("sofia.thomas3019" beside a caller renamed to
    Irene Díaz). What must survive the swap is the RELATIONSHIP, not the
    string. Recognized gaps (each present in retail's source prose):

    - dots dropped from the local part ('daikisanchez1479' for
      'daiki.sanchez1479' — task 36's caller only ever states the dotless
      form);
    - the given-name segment dropped ('silva7872' for 'amelia.silva7872');
    - the local part's one digit run moved ('sofia.thomas3019' for
      'sofia.thomas3069').

    Any other relationship returns None — better no variant than a wrong one;
    the caller decides whether that is a problem.
    """
    old_local, _, old_domain = old_email.partition("@")
    var_local, _, var_domain = old_variant.partition("@")
    new_local, _, new_domain = new_email.partition("@")
    if var_domain != old_domain:
        return None
    if var_local == old_local.replace(".", "") and var_local != old_local:
        return f"{new_local.replace('.', '')}@{new_domain}"
    if "." in old_local and var_local == old_local.partition(".")[2]:
        rest = new_local.partition(".")[2]
        return f"{rest}@{new_domain}" if rest else None
    delta = digit_run_delta(old_local, var_local)
    if delta is not None and delta != 0:
        shifted = shift_digit_run(new_local, delta)
        if shifted and shifted != new_local:
            return f"{shifted}@{new_domain}"
    return None


def caller_email_variant_pairs(
    source_task: Optional[dict],
    old_email: Optional[str],
    new_email: Optional[str],
    known_emails: Optional[set[str]] = None,
) -> tuple[tuple[str, str], ...]:
    """(old, new) pairs for the near-miss emails a task's source prose recites.

    Scans the SOURCE instructions for emails that differ from the caller's DB
    email by a gap :func:`variant_email` recognizes, and maps each onto the
    swapped-in email carrying the same gap. ``known_emails`` (every email the
    domain DB actually owns) guards against a coincidental near-miss that is
    some OTHER user's real address — that one is a bystander, never rewritten.

    Both prose producers (``derive_identity_tasks`` and english-prompt mode)
    build their :class:`IdentityRename` through this, so the pairs cannot
    diverge between the committed ``_identity`` files and the run-time swap.
    """
    if not (source_task and old_email and new_email) or old_email == new_email:
        return ()
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate in _PROSE_EMAIL_RE.findall(instruction_text(source_task)):
        if candidate == old_email or candidate in seen:
            continue
        seen.add(candidate)
        if known_emails and candidate in known_emails:
            continue
        new_variant = variant_email(old_email, candidate, new_email)
        if new_variant is not None and new_variant != new_email:
            pairs.append((candidate, new_variant))
    return tuple(pairs)


_SOURCE_POSTCODE_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")
_MAX_POSTCODE_VARIANT_DELTA = 100


def _shift_postcode(postcode: str, delta: int) -> Optional[str]:
    """Shift a postcode's digit sequence while preserving its punctuation."""
    digits = "".join(ch for ch in postcode if ch.isdigit())
    if not digits:
        return None
    shifted = int(digits) + delta
    if shifted < 0 or shifted >= 10 ** len(digits):
        return None
    replacements = iter(str(shifted).zfill(len(digits)))
    return "".join(next(replacements) if ch.isdigit() else ch for ch in postcode)


def caller_zip_variant_pairs(
    source_task: Optional[dict],
    old_zip: Optional[str],
    new_zip: Optional[str],
) -> tuple[tuple[str, str], ...]:
    """Localized forms of deliberate near-miss caller postcodes in prose.

    Retail tasks 67/68 make the caller try one or two nearby postcodes before
    giving the real one. Those mistakes are caller identity material just like
    the near-miss street number: preserve each numeric gap against the locale
    postcode, including locale punctuation such as Brazil's ``#####-###``.

    Only source-style five-digit candidates within 100 of the caller's actual
    postcode qualify. That deliberately excludes unrelated dictated
    destinations such as Springfield 62701 in a task whose caller lives at
    95154.
    """
    if not (source_task and old_zip and new_zip) or not old_zip.isdigit():
        return ()
    old_number = int(old_zip)
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for candidate in _SOURCE_POSTCODE_RE.findall(instruction_text(source_task)):
        if candidate == old_zip or candidate in seen:
            continue
        seen.add(candidate)
        delta = int(candidate) - old_number
        if not 0 < abs(delta) <= _MAX_POSTCODE_VARIANT_DELTA:
            continue
        shifted = _shift_postcode(new_zip, delta)
        if shifted is not None and shifted != new_zip:
            pairs.append((candidate, shifted))
    return tuple(pairs)


def localized_address_variant(
    old_address: Optional[dict],
    new_address: Optional[dict],
    old_variant: Optional[dict],
) -> Optional[dict]:
    """``old_variant``'s relationship to the caller's address, in the locale.

    The gap between the caller's address and its near-miss is a number, and a
    number survives localization: 'Calle de Alcalá 79' mistyped the same way
    is 'Calle de Alcalá 81'. Returns None when the locale street line has no
    single number to shift — better no variant than a wrong one.
    """
    if not (old_address and new_address and old_variant):
        return None
    variant = dict(new_address)
    for field in ("address1", "address2"):
        old_line, variant_line = old_address.get(field) or "", old_variant.get(field)
        if (variant_line or "") == old_line:
            continue
        delta = digit_run_delta(old_line, variant_line or "")
        shifted = (
            shift_digit_run(new_address.get(field) or "", delta)
            if delta is not None
            else None
        )
        if shifted is None:
            return None
        variant[field] = shifted
    return variant


def _street_name(line: str) -> str:
    """The street line minus its one house number ('760 Elm Avenue' -> 'Elm
    Avenue', 'Calle de Alcalá 79' -> 'Calle de Alcalá'), or '' when the line
    has no single number to strip. A mid-line number takes its tail with it
    ('Jianshe Lu 280 Hao' -> 'Jianshe Lu'): the counter word after the digits
    names the house number, not the street."""
    runs = list(_DIGIT_RUN_RE.finditer(line))
    if len(runs) != 1:
        return ""
    before = line[: runs[0].start()].strip(" ,")
    after = line[runs[0].end() :].strip(" ,")
    return before or after


def _swap_street_lines(
    text: str, old_address: Optional[dict], new_address: Optional[dict]
) -> str:
    """``text`` with one address's street lines swapped for another's.

    The compound form goes first so "445 Maple Drive, Suite 394" is never left
    half-swapped by the single-line rules that follow. The bare street NAME
    goes last: "You live on Elm Avenue in Houston" names the caller's street
    with no house number, so the literal-line rules never touch it and the
    city swap alone would strand it ("Elm Avenue in Alicante"). The lookbehind
    keeps a number-bearing mention out of it — "915 Elm Avenue" is a dictated
    destination, not the caller's street, and stays canonical.
    """
    if not (old_address and new_address):
        return text

    def field(source: dict, key: str) -> str:
        return (source.get(key) or "").strip()

    old_a1, new_a1 = field(old_address, "address1"), field(new_address, "address1")
    old_a2, new_a2 = field(old_address, "address2"), field(new_address, "address2")
    if old_a1 and new_a1:
        if old_a2:
            text = text.replace(
                f"{old_a1}, {old_a2}", f"{new_a1}, {new_a2}" if new_a2 else new_a1
            )
        text = text.replace(old_a1, new_a1)
        old_street, new_street = _street_name(old_a1), _street_name(new_a1)
        if old_street and new_street and old_street != new_street:
            text = re.sub(r"(?<![0-9] )" + re.escape(old_street), new_street, text)
    if old_a2 and new_a2:
        text = text.replace(old_a2, new_a2)
    return text


def _swap_full_address(
    text: str,
    old_address: Optional[dict],
    new_address: Optional[dict],
    address_format: Optional[AddressProseFormat],
) -> str:
    """Render a complete caller address in locale order when one is present."""
    if not (
        old_address and new_address and address_format and address_format.full_address
    ):
        return text

    def field(source: dict, key: str) -> str:
        return (source.get(key) or "").strip()

    old_a1 = field(old_address, "address1")
    old_a2 = field(old_address, "address2")
    old_city = field(old_address, "city")
    old_region = field(old_address, "state")
    old_country = field(old_address, "country")
    old_postcode = field(old_address, "zip")
    if not (old_a1 and old_city and old_postcode):
        return text

    rendered = address_format.full_address.format(
        address1=field(new_address, "address1"),
        address2=field(new_address, "address2"),
        city=field(new_address, "city"),
        region=field(new_address, "state"),
        country=field(new_address, "country"),
        postcode=field(new_address, "zip"),
    )
    street = ", ".join(part for part in (old_a1, old_a2) if part)
    region_forms = [old_region]
    state_name = US_STATE_NAMES.get(old_region)
    if state_name:
        region_forms.append(state_name)
    old_forms = {
        f"{street}, {old_city}, {region}, {old_postcode}"
        for region in region_forms
        if region
    }
    if old_country:
        old_forms.add(f"{street}, {old_city}, {old_country}, {old_postcode}")
        old_forms.update(
            f"{street}, {old_city}, {region}, {old_country}, {old_postcode}"
            for region in region_forms
            if region
        )
    for old_form in sorted(old_forms, key=len, reverse=True):
        text = text.replace(old_form, rendered)
    return text


class IdentityRename(BaseModel):
    """A caller's full-identity rename, derived from a ``_identity`` task.

    ``old_*`` come from the domain DB (keyed by the source task's caller id);
    ``new_*`` come from the task's ``initialization_data`` patch. Only fields
    that actually change contribute to :attr:`string_pairs` /
    :attr:`renamed_source_values`.
    """

    model_config = ConfigDict(frozen=True)

    old_user_id: str = Field(description="The source caller's user id")
    new_user_id: str = Field(description="The patched (locale) user id")
    old_name: tuple[str, str] = Field(
        description="(given, family) ROLES from the domain DB (English source); "
        "drives structured {first_name, last_name} swaps, never display order"
    )
    new_name: tuple[str, str] = Field(
        description="(given, family) ROLES from the patch — roles, not positions"
    )
    old_full_name: str = Field(
        description="The source caller's DISPLAY name (English: given-first)"
    )
    new_full_name: str = Field(
        description="The locale caller's DISPLAY name, already in the locale's "
        "name order. Carried as a string rather than recomposed from the role "
        "tuple: the STRUCTURED_NAME_PHONE patch states it verbatim, and a "
        "family-first locale would be reordered wrongly by a naive rejoin."
    )
    old_email: Optional[str] = Field(
        default=None, description="The source caller's email, if the DB has one"
    )
    new_email: Optional[str] = Field(
        default=None, description="The patched email, if the patch carries one"
    )
    old_phone: Optional[str] = Field(
        default=None, description="The source caller's phone number, if present"
    )
    new_phone: Optional[str] = Field(
        default=None, description="The patched locale phone number, if present"
    )
    old_address: Optional[dict] = Field(
        default=None,
        description="The source caller's DB address, when the domain makes the "
        "address identity material (PROSE_NAME_ZIP / retail, where the agent "
        "authenticates on name + zip). None leaves every address untouched.",
    )
    new_address: Optional[dict] = Field(
        default=None, description="The patched locale address, if any"
    )
    old_address_variant: Optional[dict] = Field(
        default=None,
        description="The near-miss of the source caller's address the task "
        "itself turns on — a mistyped street number, the suite next door (see "
        ":func:`caller_address_variant`). None for the tasks that have none.",
    )
    new_address_variant: Optional[dict] = Field(
        default=None,
        description="The locale address carrying the SAME near-miss, so the "
        "gap the task is about survives localization",
    )
    email_variant_pairs: tuple[tuple[str, str], ...] = Field(
        default=(),
        description="(old, new) near-miss emails the source prose recites — "
        "the email analogue of the address variant; see "
        ":func:`caller_email_variant_pairs`. Empty for tasks without decoys.",
    )
    zip_variant_pairs: tuple[tuple[str, str], ...] = Field(
        default=(),
        description="(old, new) deliberate near-miss caller postcodes in the "
        "source prose; see :func:`caller_zip_variant_pairs`.",
    )

    @property
    def renames_address(self) -> bool:
        """Whether an address swap is declared (and actually changes it)."""
        return bool(
            self.old_address
            and self.new_address
            and self.old_address != self.new_address
        )

    @property
    def renames_address_variant(self) -> bool:
        """Whether a near-miss address is declared (and actually changes)."""
        return bool(
            self.old_address_variant
            and self.new_address_variant
            and self.old_address_variant != self.new_address_variant
        )

    @property
    def renames_caller(self) -> bool:
        """Whether the caller is actually renamed (display name differs)."""
        return self.old_full_name != self.new_full_name

    @property
    def string_pairs(self) -> list[tuple[str, str]]:
        """(old, new) string substitutions for prose / string assertions.

        The real email precedes its near-miss variants: a given-name-dropped
        decoy ('silva7872@…') is a SUBSTRING of the real address
        ('amelia.silva7872@…'), so replacing it first would corrupt the real
        one and leave nothing for the email rule to match.
        """
        pairs: list[tuple[str, str]] = []
        if self.renames_caller:
            pairs.append((self.old_full_name, self.new_full_name))
        if self.old_user_id != self.new_user_id:
            pairs.append((self.old_user_id, self.new_user_id))
        if self.old_email and self.new_email and self.old_email != self.new_email:
            pairs.append((self.old_email, self.new_email))
        if self.old_phone and self.new_phone and self.old_phone != self.new_phone:
            pairs.append((self.old_phone, self.new_phone))
        pairs.extend((old, new) for old, new in self.email_variant_pairs if old != new)
        return pairs

    def apply_to_prose(
        self,
        text: str,
        caller_identity: CallerIdentityKind,
        address_format: Optional[AddressProseFormat] = None,
    ) -> str:
        """``text`` with the whole caller rename applied — THE single reading.

        Both producers of renamed prose go through here: the identity-set
        emitter (``localize_lib.derive_identity_tasks``, which writes the
        committed ``_identity`` files) and english-prompt mode
        (``english_prompts``, which re-derives the same swap onto the English
        source at run time). They must agree exactly — a consistency check
        compares them per task and fails the simulation on any mismatch — so
        the swap cannot live in only one of them.

        Every shape gets the literal :attr:`string_pairs` (full name, user id,
        email). ``PROSE_NAME_ZIP`` additionally gets the given name and the
        address, because there the caller is anchored in free prose by exactly
        the fields the agent authenticates on. The wider swap is deliberately
        NOT applied to the other shapes: their prose names people and places
        the caller does not own — an airline caller whose record says Houston
        has flights to Houston, and rewriting that city would rewrite the
        itinerary.

        Substitutions run longest-form first, so a compound never has its city
        half consumed by the bare-city rule and its region left stranded:

        1. the street lines of the near-miss address the task turns on, if it
           has one (445 Maple Drive when the record says 443 — see
           :func:`caller_address_variant`), then of the address on file. The
           near-miss goes first: it is a different literal, and doing it after
           the place fields moved would leave nothing to anchor on.
        2. the full street line ("445 Maple Drive, Suite 394") before each line
           on its own, then the bare street NAME ("You live on Elm Avenue" —
           see :func:`_swap_street_lines`).
        3. "City, <region>" as a unit — matched by REGEX rather than a literal
           so it catches the spelled-out state ("Fort Worth, Texas") as well as
           the DB's two-letter code ("San Jose, CA"). Swapping the city ALONE
           is worse than doing nothing: it yields hybrids like "Murcia, Texas".
           The comma is optional against the caller's OWN two-letter code
           ("Houston TX, 77004", "Seattle WA 98187") — anchored on the exact
           code, never guessed from shape. "City, <country>" ("living in
           Denver, USA") moves as a unit too, to the locale city + country.
        4. the bare city, then the caller's state SPELLED OUT ("you live in
           Texas in zipcode 76171" — 4 tasks name the state with no city
           beside it). The long form resolves to the locale city rather than
           its region code, which is both true of the caller and readable;
           'you live in MC' is neither. KNOWN LIMIT: this replace is blanket,
           so a task naming the caller's state about something they do NOT own
           ("an order sent to Texas by accident", tasks 25/26) would corrupt
           if an identity draw ever paired a Texan caller with it. No current
           pairing does, and both producers share the behavior symmetrically,
           so no consistency check can catch it — a fix needs ground truth the
           rename does not carry.
        5. the zip.

        A bare region CODE is never swapped: "CA" or "IN" would hit ordinary
        words and mangle unrelated prose. Only the caller's OWN address is
        rewritten — one they dictate as a new destination ("your new address is
        1234 Elm St, Springfield, IL, 62701") is not identity material and
        stays canonical, exactly as order ids and prices do.
        """
        for old, new in self.string_pairs:
            text = text.replace(old, new)
        if caller_identity is not CallerIdentityKind.PROSE_NAME_ZIP:
            return text
        old_first, new_first = self.old_name[0], self.new_name[0]
        # A prose-anchored caller may be introduced by given name alone ("You
        # are Amelia, and you have two emails: ..." — 4 of retail's 114 tasks).
        # Safe under this shape, whose tasks put exactly one person in the
        # prose — unlike airline, where co-passengers are named alongside the
        # caller and a bare given name is not necessarily theirs.
        if old_first and new_first and old_first != new_first:
            text = text.replace(old_first, new_first)
        if self.renames_caller:
            text = text.replace(
                f"{new_first} ({self.new_user_id})",
                f"{self.new_full_name} ({self.new_user_id})",
            )
        for old_zip_variant, new_zip_variant in self.zip_variant_pairs:
            text = text.replace(old_zip_variant, new_zip_variant)

        old_address, new_address = self.old_address or {}, self.new_address or {}
        for old_lines, new_lines in (
            (self.old_address_variant, self.new_address_variant),
            (old_address, new_address),
        ):
            text = _swap_full_address(text, old_lines, new_lines, address_format)
            text = _swap_street_lines(text, old_lines, new_lines)

        def field(source: dict, key: str) -> str:
            return (source.get(key) or "").strip()

        old_city, new_city = field(old_address, "city"), field(new_address, "city")
        old_region, new_region = (
            field(old_address, "state"),
            field(new_address, "state"),
        )
        old_country = field(old_address, "country")
        new_country = field(new_address, "country")
        if old_city and new_city:
            if new_region:
                comma_city_region = (
                    address_format.city_region.format(
                        city=new_city,
                        region=new_region,
                        country=new_country,
                    )
                    if address_format
                    else f"{new_city}, {new_region}"
                )
                text = re.sub(
                    re.escape(old_city)
                    + r",\s+(?:[A-Z]{2}|[A-Z][a-z]+(?: [A-Z][a-z]+)?)(?=[,.\s])",
                    comma_city_region,
                    text,
                )
                if old_region:
                    # The no-comma form ("Houston TX, 77004"): safe only
                    # against the caller's exact code — a shape guess here
                    # would eat a capitalized word after any city mention.
                    plain_city_region = (
                        comma_city_region
                        if address_format
                        else f"{new_city} {new_region}"
                    )
                    text = re.sub(
                        re.escape(old_city)
                        + r"\s+"
                        + re.escape(old_region)
                        + r"(?=[,.\s])",
                        plain_city_region,
                        text,
                    )
            if old_country and new_country and old_country != new_country:
                city_country = (
                    address_format.city_country.format(
                        city=new_city,
                        region=new_region,
                        country=new_country,
                    )
                    if address_format
                    else f"{new_city}, {new_country}"
                )
                text = text.replace(f"{old_city}, {old_country}", city_country)
            text = text.replace(old_city, new_city)
            for alias in US_CITY_ALIASES.get(old_city, ()):
                text = text.replace(f"the {alias}", new_city)
                text = text.replace(alias, new_city)
            state_name = US_STATE_NAMES.get(old_region)
            if state_name:
                text = text.replace(state_name, new_city)

        old_zip, new_zip = field(old_address, "zip"), field(new_address, "zip")
        if old_zip and new_zip:
            text = text.replace(old_zip, new_zip)
        return text

    @property
    def renamed_source_values(self) -> set[str]:
        """Source concrete-values intentionally replaced (verbatim exemptions)."""
        values: set[str] = set()
        if self.old_user_id != self.new_user_id:
            values.add(self.old_user_id)
        if self.old_email and self.new_email and self.old_email != self.new_email:
            values.add(self.old_email)
            # An email's digits are a concrete value in their own right, and
            # the new local part need not carry them: the generated email
            # reuses the USER-ID's numeric suffix, which in airline happens to
            # equal the email's ('mia.li3668' / 'mia_li_3668') but in retail
            # does not ('yusuf.hernandez8836' / 'yusuf_hernandez_6785').
            # Exempt only digits the new email genuinely drops.
            values |= extract_values(self.old_email) - extract_values(self.new_email)
        if self.old_phone and self.new_phone and self.old_phone != self.new_phone:
            values.add(self.old_phone)
            values |= extract_values(self.old_phone) - extract_values(self.new_phone)
        for old_variant, new_variant in self.email_variant_pairs:
            # Near-miss decoy emails move with the caller exactly as the real
            # one does — same exemption, same digits-genuinely-dropped rule.
            values.add(old_variant)
            values |= extract_values(old_variant) - extract_values(new_variant)
        for old_variant, new_variant in self.zip_variant_pairs:
            values.add(old_variant)
            values |= extract_values(old_variant) - extract_values(new_variant)
        # Same reasoning for the address the caller authenticates on: the old
        # zip (and street number) are deliberately replaced — as is the near
        # miss of it that some tasks turn on.
        pairs: list[tuple[dict, dict]] = []
        if self.renames_address:
            pairs.append((self.old_address, self.new_address))
        if self.renames_address_variant:
            pairs.append((self.old_address_variant, self.new_address_variant))
        for old_address, new_address in pairs:
            old_text = " ".join(
                str(old_address.get(field) or "") for field in _ADDRESS_FIELDS
            )
            new_text = " ".join(
                str(new_address.get(field) or "") for field in _ADDRESS_FIELDS
            )
            values |= extract_values(old_text) - extract_values(new_text)
        return values


def identity_rename(
    source_task: Optional[dict],
    localized_task: dict,
    domain_db: Optional[dict],
    name_order: NameOrder = NameOrder.GIVEN_FIRST,
) -> Optional[IdentityRename]:
    """The caller identity rename declared by a ``_identity`` task, or None.

    ``name_order`` is the LOCALE's display order (from its language pack); it
    governs how the new caller's full name is composed and, for the
    STRUCTURED_NAME_PHONE shape, how the patched ``full_name`` is read back
    into (given, family) roles. The default is the English source order, so
    callers that predate locale name order behave unchanged.

    None for plain localizations (no ``initialization_data`` caller patch) and
    for a patch whose old caller id is absent from the domain DB (that case is
    reported by :func:`check_identity_rename_coverage`).

    Two patch shapes, dispatched on what the task actually carries:
    ``agent_data.users`` (USER_ID_HANDLE domains — airline; ``_identity``
    tasks patch exactly one caller, so the new id is the patch key) and
    ``agent_data.customers`` (STRUCTURED_NAME_PHONE domains — telecom; the
    caller keeps their customer id and the rename is the entry whose
    ``full_name`` differs from the DB record).

    """
    if not domain_db:
        return None
    records = _identity_patch_records(localized_task)
    if records:
        users_db = domain_db.get("users") or {}
        old_id = resolve_caller_user_id(source_task, users_db)
        if old_id is None or old_id not in users_db:
            return None
        old_user = users_db[old_id]
        old_name = old_user.get("name") or {}
        if not old_name:
            return None
        new_id = next(iter(records))
        new_record = records[new_id]
        new_name = new_record.get("name") or old_name
        old_variant = caller_address_variant(source_task, old_user.get("address"))
        known_emails = {
            user.get("email") for user in users_db.values() if user.get("email")
        }
        return IdentityRename(
            old_user_id=old_id,
            new_user_id=new_id,
            old_name=(old_name["first_name"], old_name["last_name"]),
            new_name=(new_name["first_name"], new_name["last_name"]),
            # The USER_ID_HANDLE record stores name ROLES and no display
            # string, so the display name is composed here — the source in
            # English order, the locale caller in the locale's.
            old_full_name=compose_full_name(
                old_name["first_name"], old_name["last_name"]
            ),
            new_full_name=compose_full_name(
                new_name["first_name"], new_name["last_name"], name_order
            ),
            old_email=old_user.get("email"),
            new_email=new_record.get("email"),
            old_address=old_user.get("address"),
            new_address=new_record.get("address"),
            old_address_variant=old_variant,
            new_address_variant=localized_address_variant(
                old_user.get("address"), new_record.get("address"), old_variant
            ),
            email_variant_pairs=caller_email_variant_pairs(
                source_task,
                old_user.get("email"),
                new_record.get("email"),
                known_emails,
            ),
            zip_variant_pairs=caller_zip_variant_pairs(
                source_task,
                (old_user.get("address") or {}).get("zip"),
                (new_record.get("address") or {}).get("zip"),
            ),
        )
    customer = _customer_identity_rename(
        source_task, localized_task, domain_db, name_order
    )
    return customer


def _customer_identity_rename(
    source_task: Optional[dict],
    localized_task: dict,
    domain_db: dict,
    name_order: NameOrder = NameOrder.GIVEN_FIRST,
) -> Optional[IdentityRename]:
    """The STRUCTURED_NAME_PHONE rename: a changed ``full_name`` in a
    customers-list patch, matched by the source caller's ``set_user_info``
    phone number against the SOURCE task's effective customers list (its own
    patch when it carries one — the diversified telecom seed does — else the
    domain DB). The customer id never changes, so the rename contributes only
    full-name (and email) string pairs. When source and localized task carry
    the same identity (the en arm), the rename degenerates to no-op pairs."""
    patches = _customer_identity_patches(localized_task)
    if not patches:
        return None
    caller = caller_set_user_info(source_task)
    if caller is None:
        return None
    wanted = phone_digits(caller["phone_number"])
    old_customer = next(
        (
            customer
            for customer in effective_customer_records(source_task, domain_db)
            if phone_digits(customer.get("phone_number") or "") == wanted
        ),
        None,
    )
    if old_customer is None or not old_customer.get("full_name"):
        return None
    patch = patches.get(old_customer["customer_id"])
    if patch is None:
        return None
    # Both records STATE their display name, so neither is recomposed and the
    # DISPLAY pairs need no order: the patch's full_name is already written the
    # way the locale writes it. Order matters only for the ROLE split behind
    # structured {first_name, last_name} swaps — the source is English
    # (given-first) while the patch is in the locale's order. When the patch
    # repeats the source name (a plain, non-identity localized set), the roles
    # ARE the source's: applying the locale order there would swap them and
    # invent a rename out of a set that renames nobody.
    old_full_name = old_customer["full_name"]
    new_full_name = patch["full_name"]
    old_name = _split_full_name(old_full_name)
    return IdentityRename(
        old_user_id=old_customer["customer_id"],
        new_user_id=old_customer["customer_id"],
        old_name=old_name,
        new_name=(
            split_full_name(new_full_name, name_order)
            if new_full_name != old_full_name
            else old_name
        ),
        old_full_name=old_full_name,
        new_full_name=new_full_name,
        old_email=old_customer.get("email"),
        new_email=patch.get("email"),
        old_phone=old_customer.get("phone_number"),
        new_phone=patch.get("phone_number"),
        email_variant_pairs=caller_email_variant_pairs(
            source_task,
            old_customer.get("email"),
            patch.get("email"),
            {
                customer.get("email")
                for customer in effective_customer_records(source_task, domain_db)
                if customer.get("email")
            },
        ),
    )


def check_identity_rename_coverage(
    localized_task: dict,
    domain_db: Optional[dict],
    source_task: Optional[dict] = None,
) -> list[str]:
    """Problems for renamed callers whose user_id is absent from the domain DB.

    Empty == pass (and always empty for plain, non-identity-variant tasks). The
    caller that must exist in the DB is the SOURCE caller id (read from the
    source task's known_info); the patch key is the new, DB-absent id.
    """
    if not domain_db or source_task is None:
        return []
    task_id = localized_task.get("id", "<no id>")
    if _identity_patch_records(localized_task):
        users_db = domain_db.get("users") or {}
        old_id = resolve_caller_user_id(source_task, users_db)
        if not (users_db.get(old_id or "") or {}).get("name"):
            return [
                f"{task_id}: identity-variant task renames caller '{old_id}' "
                "but no such user with a name exists in the domain DB"
            ]
        return []
    if _customer_identity_patches(localized_task):
        if _customer_identity_rename(source_task, localized_task, domain_db) is None:
            caller = caller_set_user_info(source_task) or {}
            return [
                f"{task_id}: identity-variant task patches customers but the "
                f"source caller ({caller.get('name')!r}, "
                f"{caller.get('phone_number')!r}) does not resolve to a "
                "renamed customer record in the domain DB"
            ]
        return []
    return []


def check_customer_patch_integrity(
    localized_task: dict,
    domain_db: Optional[dict],
    source_task: Optional[dict],
) -> list[str]:
    """Integrity problems for a STRUCTURED_NAME_PHONE identity task's
    wholesale customers patch, phone alias, and its own ``set_user_info``
    (empty == pass).

    The patch ships the ENTIRE customers list (list patches replace
    wholesale), so a corrupted committed file could rename or drop a
    bystander, or mutate the caller's canonical fields (such as
    date_of_birth) — and the rename-coverage check alone would still pass.
    The baseline is the SOURCE task's effective customers list (its own
    patch when it carries one — the diversified seed does — else the domain
    DB), which pins the caller's date of birth to the English task's value
    for free: dates never localize. Enforced here:

    - the patched list carries exactly the baseline's customer ids, in order;
    - every bystander record is byte-identical to its baseline record;
    - the caller's record differs only in ``full_name``/``email``/
      ``phone_number``;
    - the localized phone aliases to the source line number;
    - the task's own ``set_user_info`` speaks the patched name and phone.
    """
    patches = _customer_identity_patches(localized_task)
    if not patches or not domain_db or source_task is None:
        return []
    task_id = localized_task.get("id", "<no id>")

    caller = caller_set_user_info(source_task)
    if caller is None:
        return []  # rename-coverage reports this shape already
    wanted = phone_digits(caller["phone_number"])
    baseline = effective_customer_records(source_task, domain_db)
    caller_baseline = next(
        (
            customer
            for customer in baseline
            if phone_digits(customer.get("phone_number") or "") == wanted
        ),
        None,
    )
    if caller_baseline is None:
        return []  # rename-coverage reports this shape already
    caller_id = caller_baseline["customer_id"]

    initial_state = localized_task.get("initial_state") or {}
    init_data = initial_state.get("initialization_data") or {}
    agent_data = init_data.get("agent_data") or {}
    patched_list = agent_data.get("customers") or []

    patched_ids = [(customer or {}).get("customer_id") for customer in patched_list]
    baseline_ids = [customer["customer_id"] for customer in baseline]
    if patched_ids != baseline_ids:
        return [
            f"{task_id}: customers patch must carry exactly the source "
            f"task's effective customer ids in order (the patch replaces "
            f"the list wholesale); got {patched_ids!r}, expected "
            f"{baseline_ids!r}"
        ]

    problems: list[str] = []
    for baseline_record, patched in zip(baseline, patched_list):
        customer_id = baseline_record["customer_id"]
        if customer_id == caller_id:
            changed = {
                key
                for key in set(baseline_record) | set(patched)
                if baseline_record.get(key) != patched.get(key)
            }
            illegal = changed - {"full_name", "email", "phone_number"}
            if illegal:
                problems.append(
                    f"{task_id}: caller customer '{customer_id}' patch "
                    f"changes {sorted(illegal)} vs the source task's "
                    "effective record — only full_name/email/phone_number "
                    "may differ (ids and date_of_birth stay pinned)"
                )
        elif patched != baseline_record:
            problems.append(
                f"{task_id}: bystander customer '{customer_id}' differs from "
                "the source task's effective record — identity patches must "
                "leave bystanders byte-identical"
            )

    localized_caller = caller_set_user_info(localized_task)
    patch_record = patches.get(caller_id) or {}
    if localized_caller is None:
        problems.append(
            f"{task_id}: identity task patches customers but carries no "
            "set_user_info init action"
        )
    else:
        patched_name = patch_record.get("full_name")
        if patched_name and localized_caller["name"] != patched_name:
            problems.append(
                f"{task_id}: set_user_info name {localized_caller['name']!r} "
                f"does not match the customers-patch rename {patched_name!r}"
            )
        patched_phone = patch_record.get("phone_number")
        if patched_phone and localized_caller["phone_number"] != patched_phone:
            problems.append(
                f"{task_id}: set_user_info phone "
                f"{localized_caller['phone_number']!r} differs from the "
                f"customers-patch phone {patched_phone!r}"
            )

    old_phone = caller_baseline.get("phone_number") or ""
    new_phone = patch_record.get("phone_number") or ""
    if new_phone != old_phone:
        source_agent_data = (
            (source_task.get("initial_state") or {}).get("initialization_data") or {}
        ).get("agent_data") or {}
        expected_aliases = {
            **(domain_db.get("phone_number_aliases") or {}),
            **(source_agent_data.get("phone_number_aliases") or {}),
            new_phone: old_phone,
        }
        if agent_data.get("phone_number_aliases") != expected_aliases:
            problems.append(
                f"{task_id}: phone_number_aliases must map localized phone "
                f"{new_phone!r} to source line phone {old_phone!r}"
            )

    return problems


def _is_same_address(node: dict, address: Optional[dict]) -> bool:
    """Whether ``node`` carries exactly ``address``'s values on every field.

    Tolerates the two ways an absent secondary line is written (``None`` and
    ``""``) so a record and an action argument that mean the same thing match.
    """
    if not address:
        return False
    if not any(field in node for field in _ADDRESS_FIELDS):
        return False
    return all(
        (node.get(field) or "") == (address.get(field) or "")
        for field in _ADDRESS_FIELDS
    )


def expected_eval_criteria(
    source_task: dict,
    localized_task: dict,
    domain_db: Optional[dict] = None,
    name_order: NameOrder = NameOrder.GIVEN_FIRST,
) -> dict:
    """What the localized task's evaluation_criteria must equal.

    Plain localizations: the source criteria byte-for-byte. Identity-variant
    tasks (caller renamed via initialization_data): the source criteria with
    exactly that rename applied — dict-level {first_name, last_name} and
    user_id swaps in action arguments (so unrelated passengers sharing a first
    name are never touched) and full-name / id / email swaps in
    nl_assertions/communicate_info strings.
    """
    criteria = copy.deepcopy(source_task["evaluation_criteria"])
    rename = identity_rename(source_task, localized_task, domain_db, name_order)
    if rename is None:
        return criteria

    old_first, old_last = rename.old_name
    new_first, new_last = rename.new_name
    string_pairs = rename.string_pairs
    # The caller's email is an authentication key exactly as name+zip is:
    # a golden find_user_id_by_email must ask for the SWAPPED address, or the
    # expected action is one no agent following the call can emit. Near-miss
    # decoys move too, carrying their gap (see caller_email_variant_pairs).
    email_swaps = dict(rename.email_variant_pairs)
    if rename.old_email and rename.new_email and rename.old_email != rename.new_email:
        email_swaps[rename.old_email] = rename.new_email

    def swap_structured(node):
        if isinstance(node, dict):
            if (
                node.get("first_name") == old_first
                and node.get("last_name") == old_last
            ):
                node["first_name"] = new_first
                node["last_name"] = new_last
            if node.get("user_id") == rename.old_user_id:
                node["user_id"] = rename.new_user_id
            if node.get("email") in email_swaps:
                node["email"] = email_swaps[node["email"]]
            if rename.renames_address:
                # The caller's own address, moved in lockstep with the DB
                # patch. EXACT-match only: retail's address-changing tasks
                # split into ones that restate the caller's address on file
                # (which must follow the identity) and ones where the caller
                # dictates a brand-new destination (which must not). Matching
                # the whole dict is what tells those apart — a dictated
                # address differs in at least one field.
                if _is_same_address(node, rename.old_address):
                    node.update(
                        {
                            key: value
                            for key, value in rename.new_address.items()
                            if key in node
                        }
                    )
                elif rename.renames_address_variant and _is_same_address(
                    node, rename.old_address_variant
                ):
                    # The near miss the task is ABOUT (the mistyped street
                    # number the caller wants corrected). It moves with the
                    # caller, carrying the same gap — see
                    # :func:`localized_address_variant`.
                    node.update(
                        {
                            key: value
                            for key, value in rename.new_address_variant.items()
                            if key in node
                        }
                    )
                elif node.get("zip") == (rename.old_address or {}).get("zip") and (
                    node.get("first_name") == new_first
                    or node.get("last_name") == new_last
                ):
                    # find_user_id_by_name_zip: name + zip and nothing else, so
                    # there is no full address dict to match. The name half was
                    # just swapped above; the zip must follow it.
                    new_zip = (rename.new_address or {}).get("zip")
                    if new_zip:
                        node["zip"] = new_zip
            for value in node.values():
                swap_structured(value)
        elif isinstance(node, list):
            for item in node:
                swap_structured(item)

    def swap_strings(node):
        if isinstance(node, list):
            return [swap_strings(item) for item in node]
        if isinstance(node, str):
            for old, new in string_pairs:
                node = node.replace(old, new)
            return node
        return node

    swap_structured(criteria.get("actions") or [])
    for field in ("communicate_info", "nl_assertions"):
        if criteria.get(field):
            criteria[field] = swap_strings(criteria[field])
    return criteria


def check_task_localization(
    localized_task: dict,
    source_task: dict,
    script_re: Optional[re.Pattern],
    domain_db: Optional[dict] = None,
    localized_digits_re: Optional[re.Pattern] = None,
    name_order: NameOrder = NameOrder.GIVEN_FIRST,
    domain: Optional[str] = None,
) -> list[str]:
    """All localization-invariant problems for one task (empty == pass).

    ``script_re=None`` skips the prose-is-localized presence checks — for
    domains whose sets deliberately keep the English source prose while task
    translation is deferred (``DomainLocalizationProfile.tasks_translated`` is
    False); the entity and evaluation-criteria invariants still run.

    ``name_order`` is the locale's display name order (see
    :func:`identity_rename`); the default is the English source order.

    ``domain`` is accepted for callers that validate multiple domain profiles.
    """
    problems: list[str] = []
    task_id = localized_task.get("id", "<no id>")

    problems.extend(
        check_identity_rename_coverage(localized_task, domain_db, source_task)
    )
    problems.extend(
        check_customer_patch_integrity(localized_task, domain_db, source_task)
    )
    rename = identity_rename(source_task, localized_task, domain_db, name_order)
    expected = expected_eval_criteria(
        source_task, localized_task, domain_db, name_order
    )
    if localized_task["evaluation_criteria"] != expected:
        suffix = " (modulo the declared caller rename)" if domain_db else ""
        problems.append(
            f"{task_id}: evaluation_criteria differs from the English source{suffix}"
        )

    source_text = instruction_text(source_task)
    localized_text = instruction_text(localized_task)
    # Identity-variant tasks intentionally replace the caller's id/email with a
    # locale identity, so those source values are not required to survive.
    exempt = rename.renamed_source_values if rename else set()
    if rename is not None and rename.renames_caller:
        # The rename must land EVERYWHERE the caller is named — a leftover
        # source-caller mention ships a mixed-identity task.
        old_full = rename.old_full_name
        ticket = localized_task.get("ticket") or ""
        if old_full and (old_full in localized_text or old_full in ticket):
            problems.append(
                f"{task_id}: identity-variant task still names the source "
                f"caller '{old_full}' in its prose/ticket — the rename must "
                "land everywhere"
            )
    for value in sorted(extract_values(source_text)):
        # A renamed id/email may be extracted with trailing punctuation (the
        # email pattern is greedy), so compare with the edges stripped — but
        # never by substring containment: a bare digit-run exemption ('1479')
        # must not shield a whole email that happens to contain it, or the
        # verbatim-survival invariant silently stops checking those values.
        if value in localized_text or value.strip(".,;:!?") in exempt:
            continue
        problems.append(
            f"{task_id}: concrete value '{value}' from the English source is "
            "missing — names, ids, codes, and numbers must survive verbatim"
        )

    # Presence check: >= 1 char of the target script; mixed content is fine.
    if script_re is not None:
        purpose = (localized_task.get("description") or {}).get("purpose") or ""
        if not script_re.search(purpose):
            problems.append(f"{task_id}: description.purpose is not localized")
        instructions = localized_task["user_scenario"]["instructions"]
        for field in ["task_instructions", "reason_for_call", "known_info"]:
            if not script_re.search(instructions.get(field) or ""):
                problems.append(f"{task_id}: {field} is not localized")

    if localized_digits_re and localized_digits_re.search(localized_text):
        problems.append(
            f"{task_id}: localized digit forms found — IDs/amounts must stay "
            "Latin digits"
        )
    return problems


def check_task_set_localization(
    localized_tasks: list[dict],
    source_tasks: list[dict],
    suffix: str,
    script_code: Optional[str],
    domain_db: Optional[dict] = None,
    name_order: Optional[NameOrder] = None,
    domain: Optional[str] = None,
) -> list[str]:
    """All problems for one localized task set (empty == pass).

    ``script_code=None`` skips the prose-is-localized (and localized-digit)
    checks — see :func:`check_task_localization`.

    ``name_order`` defaults to the declared order of the set's own language —
    the first token of ``suffix`` by the task-set naming convention
    (``<lang>`` / ``<lang>_identity``), falling back to the English given-first
    order for an unregistered language. Resolving it here rather than making
    every caller remember means a family-first language cannot be validated
    against the wrong name order. Pass it explicitly to override.

    ``domain`` is forwarded for callers that validate multiple domain profiles.
    """
    if name_order is None:
        name_order = name_order_for_language(suffix.split("_")[0])
    problems: list[str] = []
    source_by_id = {task["id"]: task for task in source_tasks}
    seen_ids: set[str] = set()
    script_re = script_regex(script_code) if script_code else None
    digits_re = LOCALIZED_DIGIT_RES.get(script_code) if script_code else None

    for task in localized_tasks:
        task_id = task.get("id", "<no id>")
        if task_id in seen_ids:
            problems.append(f"duplicate task id '{task_id}'")
            continue
        seen_ids.add(task_id)
        if not task_id.endswith(f"_{suffix}"):
            problems.append(
                f"{task_id}: id must end with '_{suffix}' (the task-set suffix)"
            )
            continue
        source_id = task_id.removesuffix(f"_{suffix}")
        if source_id not in source_by_id:
            problems.append(
                f"{task_id}: no source task '{source_id}' in the domain task set"
            )
            continue
        problems.extend(
            check_task_localization(
                task,
                source_by_id[source_id],
                script_re,
                domain_db=domain_db,
                localized_digits_re=digits_re,
                name_order=name_order,
                domain=domain,
            )
        )
    return problems
