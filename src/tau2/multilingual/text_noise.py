# Copyright Sierra
"""Noisy-text probes: deterministic noise injection into ENTITY conveyance.

The text analog of acoustic degradation (docs/designs/text-mode-multilingual.md,
Item 2). The voice stack degrades the CHANNEL, not the caller's knowledge —
the caller can always repeat the true value — so the text analog injects noise
at the conveyance level:

- a per-task **entity noise plan** is precomputed deterministically from
  ``(catalog_version, seed, task_id, entity)``: one corrupted form per pinned
  identity entity, with the clean ground truth recorded alongside. The plan
  depends on nothing the agent under test does, so every system faces the
  identical corrupted stimulus (paired design);
- at simulation time the text user simulator's FIRST conveyance of each
  pinned entity is substituted with its corrupted form (message content only,
  never tool calls). Later conveyances stay clean, so an agent that notices
  the failed lookup and re-asks can repair — the capture/repair loop the
  clean/corrupted pair exists to measure. Corrupting every conveyance would
  make exact-lookup tasks unsolvable and measure only the corruption.

Operators are a CLOSED, versioned catalog with per-language applicability
(which operators apply to which scripts); an unmapped language fails loudly.
``tau2 factory text-noise-bank`` renders the full corrupted stimulus bank for
a language's task set through this exact code path, so what reviewers approve
is what runs.
"""

import re
import unicodedata
from datetime import datetime, timezone
from random import Random
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.simulation import (
    TextNoiseEntity,
    TextNoiseInfo,
    TextNoiseSettings,
)
from tau2.data_model.tasks import Task
from tau2.multilingual.domain_profiles import CallerIdentityKind, get_domain_profile
from tau2.multilingual.invariants import CALLER_USER_ID_RE, INSTRUCTION_FIELDS

TEXT_NOISE_CATALOG_VERSION = "v1"

TEXT_NOISE_PROMPT_VERSION = "v1"

# Fixed, versioned conveyance-format clause rendered into the TEXT user
# simulator's system prompt on noise-armed runs: it pins values to inline
# verbatim strings so first-conveyance substitution matches them. Calibrated
# prompt material — never edit in place without bumping
# TEXT_NOISE_PROMPT_VERSION. It deliberately says nothing about noise: the
# sim believes what it typed, and repairs from its (clean) scenario when the
# agent re-asks.
TYPED_VALUES_CLAUSE = """
## TYPED VALUES
When you provide a concrete value from your scenario (a name, ID, email, phone number, or code), type it inline as one exact string, exactly as written in your scenario. Do not spell it out character by character, reformat it, or add spaces or punctuation inside it.
""".strip()


# ---------------------------------------------------------------------------
# Operator catalog (closed, versioned)
# ---------------------------------------------------------------------------


class TextNoiseOperatorDef(BaseModel):
    """Language-independent metadata for one noise operator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Stable catalog id, recorded on every corruption.")
    description: str = Field(description="What the operator does, for reviewers.")


_QWERTY_NEIGHBORS: dict[str, str] = {
    "q": "wa",
    "w": "qes",
    "e": "wrd",
    "r": "etf",
    "t": "ryg",
    "y": "tuh",
    "u": "yij",
    "i": "uok",
    "o": "ipl",
    "p": "ol",
    "a": "qsz",
    "s": "adwx",
    "d": "sfec",
    "f": "dgrv",
    "g": "fhtb",
    "h": "gjyn",
    "j": "hkum",
    "k": "jlim",
    "l": "kop",
    "z": "xas",
    "x": "zcsd",
    "c": "xvdf",
    "v": "cbfg",
    "b": "vngh",
    "n": "bmhj",
    "m": "njk",
}

_ASCII_LETTER_RE = re.compile(r"[a-zA-Z]")
_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z]{5,}")


def _ascii_letter_positions(value: str) -> list[int]:
    return [i for i, ch in enumerate(value) if ch.isascii() and ch.isalpha()]


def _match_case(source: str, replacement: str) -> str:
    return replacement.upper() if source.isupper() else replacement


def _can_adjacent_key_typo(value: str, language: str) -> bool:
    return any(
        value[i].lower() in _QWERTY_NEIGHBORS for i in _ascii_letter_positions(value)
    )


def _apply_adjacent_key_typo(value: str, rng: Random, language: str) -> str:
    positions = [
        i
        for i in _ascii_letter_positions(value)
        if value[i].lower() in _QWERTY_NEIGHBORS
    ]
    index = rng.choice(positions)
    neighbor = rng.choice(_QWERTY_NEIGHBORS[value[index].lower()])
    return value[:index] + _match_case(value[index], neighbor) + value[index + 1 :]


def _transposable_pairs(value: str, predicate: Callable[[str], bool]) -> list[int]:
    return [
        i
        for i in range(len(value) - 1)
        if predicate(value[i]) and predicate(value[i + 1]) and value[i] != value[i + 1]
    ]


def _is_ascii_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _can_char_transpose(value: str, language: str) -> bool:
    return bool(_transposable_pairs(value, _is_ascii_letter))


def _apply_char_transpose(value: str, rng: Random, language: str) -> str:
    i = rng.choice(_transposable_pairs(value, _is_ascii_letter))
    return value[:i] + value[i + 1] + value[i] + value[i + 2 :]


def _can_char_drop(value: str, language: str) -> bool:
    return bool(_LATIN_TOKEN_RE.search(value))


def _apply_char_drop(value: str, rng: Random, language: str) -> str:
    match = rng.choice(list(_LATIN_TOKEN_RE.finditer(value)))
    # Drop an INTERIOR letter — edge typos read as different names, interior
    # drops read as typing slips.
    index = match.start() + rng.randrange(1, len(match.group()) - 1)
    return value[:index] + value[index + 1 :]


def _can_char_double(value: str, language: str) -> bool:
    return bool(_ascii_letter_positions(value))


def _apply_char_double(value: str, rng: Random, language: str) -> str:
    index = rng.choice(_ascii_letter_positions(value))
    return value[: index + 1] + value[index] + value[index + 1 :]


def _can_digit_transpose(value: str, language: str) -> bool:
    return sum(ch.isdigit() for ch in value) >= 4 and bool(
        _transposable_pairs(value, str.isdigit)
    )


def _apply_digit_transpose(value: str, rng: Random, language: str) -> str:
    i = rng.choice(_transposable_pairs(value, str.isdigit))
    return value[:i] + value[i + 1] + value[i] + value[i + 2 :]


def _strip_diacritics(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def _can_diacritic_strip(value: str, language: str) -> bool:
    return _strip_diacritics(value) != value


def _apply_diacritic_strip(value: str, rng: Random, language: str) -> str:
    return _strip_diacritics(value)


def _ko_droppable_spaces(value: str) -> list[int]:
    """Indices of spaces flanked by Hangul on both sides (deletable)."""
    return [
        i
        for i in range(1, len(value) - 1)
        if value[i] == " "
        and _HANGUL_RE.match(value[i - 1])
        and _HANGUL_RE.match(value[i + 1])
    ]


def _ko_syllable_boundaries(value: str) -> list[int]:
    """Indices between two adjacent Hangul syllables (spurious-space sites)."""
    return [
        i
        for i in range(1, len(value))
        if _HANGUL_RE.match(value[i - 1]) and _HANGUL_RE.match(value[i])
    ]


def _can_ko_spacing(value: str, language: str) -> bool:
    # Exactly the eligibility _apply_ko_spacing needs: a deletable
    # inter-Hangul space or an adjacent-syllable boundary to split. Hangul
    # count alone is not enough (e.g. syllables separated by Latin chars).
    return bool(_ko_droppable_spaces(value) or _ko_syllable_boundaries(value))


def _apply_ko_spacing(value: str, rng: Random, language: str) -> str:
    spaces = _ko_droppable_spaces(value)
    if spaces:
        # A required space is dropped (the dominant spacing error).
        index = rng.choice(spaces)
        return value[:index] + value[index + 1 :]
    # No inter-Hangul space: insert a spurious one at a syllable boundary.
    index = rng.choice(_ko_syllable_boundaries(value))
    return value[:index] + " " + value[index:]


# Fixed per-language plausible-autocorrect tables (clean lowercase token →
# what a phone keyboard plausibly turns it into). Reviewed data: es/pt
# autocorrect RE-ADDS accents (names arrive accented while the DB is
# ASCII-folded); en/hi autocorrect turns unknown name tokens into dictionary
# words.
AUTOCORRECT_TABLES: dict[str, dict[str, str]] = {
    "en": {"daiki": "daily", "mei": "me", "chen": "when", "raj": "ran"},
    "hi": {"daiki": "daily", "mei": "me", "chen": "when", "raj": "ran"},
    "es": {
        "jose": "josé",
        "garcia": "garcía",
        "martinez": "martínez",
        "gonzalez": "gonzález",
        "hernandez": "hernández",
    },
    "pt": {
        "joao": "joão",
        "sao": "são",
        "goncalves": "gonçalves",
        "conceicao": "conceição",
        "antonio": "antônio",
    },
}

_WORD_RE = re.compile(r"[A-Za-z]+")


def _autocorrect_hits(value: str, language: str) -> list[re.Match]:
    table = AUTOCORRECT_TABLES.get(language, {})
    return [m for m in _WORD_RE.finditer(value) if m.group().lower() in table]


def _can_autocorrect(value: str, language: str) -> bool:
    return bool(_autocorrect_hits(value, language))


def _apply_autocorrect(value: str, rng: Random, language: str) -> str:
    match = rng.choice(_autocorrect_hits(value, language))
    token = match.group()
    replacement = AUTOCORRECT_TABLES[language][token.lower()]
    if token[0].isupper():
        replacement = replacement[0].upper() + replacement[1:]
    return value[: match.start()] + replacement + value[match.end() :]


TEXT_NOISE_OPERATORS: dict[str, TextNoiseOperatorDef] = {
    d.id: d
    for d in [
        TextNoiseOperatorDef(
            id="adjacent_key_typo",
            description="Replaces one Latin letter with a QWERTY neighbor.",
        ),
        TextNoiseOperatorDef(
            id="char_transpose",
            description="Swaps two adjacent Latin letters.",
        ),
        TextNoiseOperatorDef(
            id="char_drop",
            description="Drops one interior letter of a Latin token of 5+ letters.",
        ),
        TextNoiseOperatorDef(
            id="char_double",
            description="Doubles one Latin letter.",
        ),
        TextNoiseOperatorDef(
            id="digit_transpose",
            description="Swaps two adjacent digits in a 4+-digit value.",
        ),
        TextNoiseOperatorDef(
            id="diacritic_strip",
            description="Removes every accent/diacritic (NFD strip); fires "
            "only on strings that carry one.",
        ),
        TextNoiseOperatorDef(
            id="ko_spacing",
            description="Deletes a required space (or inserts a spurious one) "
            "at a Hangul syllable boundary.",
        ),
        TextNoiseOperatorDef(
            id="autocorrect_substitution",
            description="Replaces a token via the language's fixed "
            "plausible-autocorrect table; fires only on table hits.",
        ),
    ]
}

_CAN_FIRE: dict[str, Callable[[str, str], bool]] = {
    "adjacent_key_typo": _can_adjacent_key_typo,
    "char_transpose": _can_char_transpose,
    "char_drop": _can_char_drop,
    "char_double": _can_char_double,
    "digit_transpose": _can_digit_transpose,
    "diacritic_strip": _can_diacritic_strip,
    "ko_spacing": _can_ko_spacing,
    "autocorrect_substitution": _can_autocorrect,
}

_APPLY: dict[str, Callable[[str, Random, str], str]] = {
    "adjacent_key_typo": _apply_adjacent_key_typo,
    "char_transpose": _apply_char_transpose,
    "char_drop": _apply_char_drop,
    "char_double": _apply_char_double,
    "digit_transpose": _apply_digit_transpose,
    "diacritic_strip": _apply_diacritic_strip,
    "ko_spacing": _apply_ko_spacing,
    "autocorrect_substitution": _apply_autocorrect,
}

_LATIN_OPERATORS = (
    "adjacent_key_typo",
    "char_transpose",
    "char_drop",
    "char_double",
    "digit_transpose",
)

# Which operators apply to which language (closed map; an unmapped language
# fails loudly at the plan seam). Rationale per language in
# docs/designs/text-mode-multilingual.md.
NOISE_OPERATORS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "en": (*_LATIN_OPERATORS, "autocorrect_substitution"),
    "hi": (*_LATIN_OPERATORS, "autocorrect_substitution"),
    "es": (*_LATIN_OPERATORS, "diacritic_strip", "autocorrect_substitution"),
    "pt": (*_LATIN_OPERATORS, "diacritic_strip", "autocorrect_substitution"),
    "ko": (
        "adjacent_key_typo",
        "char_transpose",
        "char_double",
        "digit_transpose",
        "ko_spacing",
    ),
    "zh": ("adjacent_key_typo", "char_transpose", "char_double", "digit_transpose"),
}


def get_noise_operator(operator_id: str) -> TextNoiseOperatorDef:
    """Look up a catalog operator by id, or raise if not in the closed set."""
    try:
        return TEXT_NOISE_OPERATORS[operator_id]
    except KeyError:
        raise KeyError(
            f"Unknown text-noise operator '{operator_id}'. It must be one of "
            f"the catalog ids in tau2.multilingual.text_noise: "
            f"{sorted(TEXT_NOISE_OPERATORS)}"
        ) from None


def noise_operators_for(language: str) -> tuple[str, ...]:
    """The closed operator set for one language; unmapped languages raise."""
    try:
        return NOISE_OPERATORS_BY_LANGUAGE[language]
    except KeyError:
        raise KeyError(
            f"Language '{language}' has no text-noise operator mapping. Add it "
            "to NOISE_OPERATORS_BY_LANGUAGE in tau2.multilingual.text_noise "
            "(a reviewed decision, not a fallback)."
        ) from None


# ---------------------------------------------------------------------------
# Entity pinning
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\dA-Za-z])\+?\d[\d\-\.]{5,}\d(?![\dA-Za-z])")
# ISO-formatted dates (DOB is an auth entity in name-authenticated domains).
# Matched before phones so a date is classified as a date, not a digit run.
_DATE_RE = re.compile(r"(?<![\dA-Za-z])\d{4}-\d{2}-\d{2}(?![\dA-Za-z])")
# "You are Marcus Delgado with phone number ..." — the caller-name anchor of
# name-authenticated domains' known_info prose (identity strings are
# ASCII/romanized in the DB per the identity ASCII fold).
_CALLER_NAME_RE = re.compile(r"[Yy]ou are ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+)")


class PinnedEntity(BaseModel):
    """One identity entity pinned for noise, before corruption."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(
        description="Entity class: email | user_id | date | phone | name."
    )
    value: str = Field(
        description="The clean entity string as the scenario carries it."
    )


def _scenario_text(task: Task) -> str:
    scenario = task.user_scenario
    parts = [scenario.persona or ""]
    instructions = scenario.instructions
    if isinstance(instructions, str):
        parts.append(instructions)
    else:
        parts.extend(getattr(instructions, field) or "" for field in INSTRUCTION_FIELDS)
    return "\n".join(part for part in parts if part)


def _caller_identity_kind(domain: str) -> CallerIdentityKind:
    try:
        return get_domain_profile(domain).caller_identity
    except KeyError:
        return CallerIdentityKind.USER_ID_HANDLE


def pinned_entities(task: Task, domain: str) -> list[PinnedEntity]:
    """The identity/conveyance entities noise may target, in stable order.

    Driven by the domain profile's caller-identity shape: emails, user-id
    handles, and phone-shaped values are pinned for every domain; the caller's
    display name only for name-anchored domains. Reservation/order codes and
    amounts are deliberately NOT pinned — corrupting a value the caller reads
    off a record (rather than knows about themself) makes the task unsolvable
    rather than noisy, and amounts change task semantics.
    """
    text = _scenario_text(task)
    entities: list[PinnedEntity] = []
    seen: set[str] = set()

    def _add(kind: str, value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            entities.append(PinnedEntity(kind=kind, value=value))

    for match in _EMAIL_RE.finditer(text):
        _add("email", match.group())
    for match in CALLER_USER_ID_RE.finditer(text):
        _add("user_id", match.group())
    for match in _DATE_RE.finditer(text):
        _add("date", match.group())
    for match in _PHONE_RE.finditer(text):
        if sum(ch.isdigit() for ch in match.group()) >= 7 and not _DATE_RE.fullmatch(
            match.group()
        ):
            _add("phone", match.group())
    if _caller_identity_kind(domain) in (
        CallerIdentityKind.STRUCTURED_NAME_PHONE,
        CallerIdentityKind.PROSE_NAME_ZIP,
    ):
        for match in _CALLER_NAME_RE.finditer(text):
            _add("name", match.group(1))
    return entities


# ---------------------------------------------------------------------------
# Deterministic corruption plans
# ---------------------------------------------------------------------------


class SkippedEntity(BaseModel):
    """A pinned entity no applicable operator could corrupt (bank review)."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="Entity class of the untouched entity.")
    clean: str = Field(description="The entity string, conveyed clean.")
    reason: str = Field(description="Why no corruption applies.")


class TaskNoisePlan(BaseModel):
    """One task's noise plan plus what was pinned but left clean."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="The task the plan corrupts.")
    info: TextNoiseInfo = Field(
        description="The runtime record: catalog version, seed, corruptions."
    )
    skipped: list[SkippedEntity] = Field(
        default_factory=list,
        description="Pinned entities with no applicable operator (reviewable).",
    )


def _entity_rng(seed: int, task_id: str, entity: str) -> Random:
    # String-seeded so the draw is stable across processes and machines
    # (never salted hash()), and independent of everything but
    # (catalog version, seed, task, entity) — the paired-design contract.
    return Random(
        f"tau2-text-noise:{TEXT_NOISE_CATALOG_VERSION}:{seed}:{task_id}:{entity}"
    )


def plan_task_noise(
    task: Task,
    language: str,
    settings: TextNoiseSettings,
    domain: str,
) -> TaskNoisePlan:
    """Build one task's deterministic entity noise plan.

    For each pinned entity: a seeded RNG draws one operator among the
    language's applicable operators that CAN fire on the string, then applies
    it with the same RNG. Same seed → byte-identical plan, every run, every
    machine. ``settings.operators`` (when set) restricts the language set —
    never extends it.
    """
    language_operators = noise_operators_for(language)
    if settings.operators is not None:
        allowed = set(settings.operators)
        operators = tuple(op for op in language_operators if op in allowed)
    else:
        operators = language_operators

    corrupted_entities: list[TextNoiseEntity] = []
    skipped: list[SkippedEntity] = []
    for entity in pinned_entities(task, domain):
        rng = _entity_rng(settings.seed, task.id, entity.value)
        firing = [op for op in operators if _CAN_FIRE[op](entity.value, language)]
        if not firing:
            skipped.append(
                SkippedEntity(
                    kind=entity.kind,
                    clean=entity.value,
                    reason="no applicable operator for this string",
                )
            )
            continue
        operator = rng.choice(firing)
        corrupted = _APPLY[operator](entity.value, rng, language)
        if corrupted == entity.value:
            raise ValueError(
                f"text-noise operator '{operator}' returned the clean string "
                f"for {entity.value!r} — operators must always change a string "
                "they claim to fire on"
            )
        corrupted_entities.append(
            TextNoiseEntity(
                kind=entity.kind,
                clean=entity.value,
                corrupted=corrupted,
                operator=operator,
            )
        )
    return TaskNoisePlan(
        task_id=task.id,
        info=TextNoiseInfo(
            catalog_version=TEXT_NOISE_CATALOG_VERSION,
            seed=settings.seed,
            entities=corrupted_entities,
        ),
        skipped=skipped,
    )


def _conveyance_re(value: str) -> re.Pattern[str]:
    """Whole-token pattern for one conveyed value.

    Same ASCII-alphanumeric lookarounds the pinning regexes use: a value
    counts as conveyed only when no ASCII letter or digit touches it, so a
    short corrupted form (``me``) can never match inside an unrelated word
    (``some``). Non-ASCII neighbors do not block a match — a Hangul particle
    attached to a name (``김지훈입니다``) is still a conveyance.
    """
    return re.compile(rf"(?<![0-9A-Za-z]){re.escape(value)}(?![0-9A-Za-z])")


def inject_first_conveyance_noise(
    content: str,
    info: TextNoiseInfo,
    prior_user_texts: list[str],
) -> str:
    """Substitute each entity's FIRST conveyance with its corrupted form.

    An entity is corrupted in the first user message that conveys its clean
    string as a whole token (every occurrence within that message — one
    consistent typo); once its corrupted form has appeared as a whole token
    in a prior user message, later conveyances pass through clean, so the
    agent's re-ask can repair. Applies to message CONTENT only — callers
    must never pass tool calls.
    """
    # Longest-first so a value that contains another pinned value cannot be
    # partially rewritten by the shorter one's substitution.
    for entity in sorted(info.entities, key=lambda e: len(e.clean), reverse=True):
        clean_re = _conveyance_re(entity.clean)
        if not clean_re.search(content):
            continue
        corrupted_re = _conveyance_re(entity.corrupted)
        if any(corrupted_re.search(text) for text in prior_user_texts):
            continue
        # re.sub with a function so the corrupted string is inserted verbatim
        # (never interpreted as a replacement template).
        content = clean_re.sub(lambda _match, _c=entity.corrupted: _c, content)
    return content


# ---------------------------------------------------------------------------
# The reviewable stimulus bank (tau2 factory text-noise-bank)
# ---------------------------------------------------------------------------


class TextNoiseBankManifest(BaseModel):
    """Provenance for one rendered stimulus bank (human-facing artifact)."""

    model_config = ConfigDict(extra="forbid")

    catalog_version: str = Field(description="TEXT_NOISE_CATALOG_VERSION at render.")
    language: str = Field(description="ISO 639-1 language the bank was built for.")
    domain: str = Field(description="Benchmark domain of the task set.")
    task_set_name: str = Field(description="Registered task set the bank covers.")
    seed: int = Field(description="The noise seed; same seed → identical bank.")
    operators: list[str] = Field(
        description="The resolved operator set the plans drew from."
    )
    created_at: str = Field(description="UTC ISO timestamp of the render.")
    num_tasks: int = Field(description="Tasks scanned.")
    num_corrupted: int = Field(description="Entities corrupted across the bank.")
    num_skipped: int = Field(description="Pinned entities left clean (no operator).")


class TextNoiseBank(BaseModel):
    """The full corrupted stimulus bank for one language x task set."""

    model_config = ConfigDict(extra="forbid")

    manifest: TextNoiseBankManifest
    plans: list[TaskNoisePlan]


def build_text_noise_bank(
    language: str,
    domain: str,
    task_set_name: str,
    settings: Optional[TextNoiseSettings] = None,
) -> TextNoiseBank:
    """Render the corrupted stimulus bank for one language's task set.

    Uses the exact runtime plan builder (:func:`plan_task_noise`), so the bank
    a human reviews is byte-identical to what a run with the same settings
    will inject.
    """
    from tau2.registry import registry

    settings = settings or TextNoiseSettings()
    language_operators = noise_operators_for(language)
    if settings.operators is not None:
        allowed = set(settings.operators)
        operators = [op for op in language_operators if op in allowed]
    else:
        operators = list(language_operators)

    tasks = registry.get_tasks_loader(task_set_name)()
    plans = [plan_task_noise(task, language, settings, domain) for task in tasks]
    return TextNoiseBank(
        manifest=TextNoiseBankManifest(
            catalog_version=TEXT_NOISE_CATALOG_VERSION,
            language=language,
            domain=domain,
            task_set_name=task_set_name,
            seed=settings.seed,
            operators=operators,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            num_tasks=len(plans),
            num_corrupted=sum(len(plan.info.entities) for plan in plans),
            num_skipped=sum(len(plan.skipped) for plan in plans),
        ),
        plans=plans,
    )
