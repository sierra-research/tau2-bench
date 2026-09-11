# Copyright Sierra
"""Deterministic entity capture / tool-argument tracing (``tau2 metrics entity-trace``).

A SHADOW instrument for the τ-ML paper: computed and reported, never scored or
gated. No LLM calls anywhere — everything is derived from the stored run
(ticks / messages / tool traces) and the task definition, which is the ground
truth for what the caller was told to convey ("the sim knows what it conveyed").

For every pinned identity entity of a call (name, user id, customer id, email,
phone, date of birth, zip code) the tracer answers, per call:

(a) **first capture** — did the entity's FIRST realization as a tool-call
    argument match the ground-truth value (fold-aware)?
(b) **wrong/fabricated args** — did a wrong value for the entity ever reach a
    tool-call argument, split by read-only vs state-changing tools (the
    fabrication gate feeding the Authentication Robustness metric)? A wrong
    value the caller never uttered before the call is a FABRICATION: the
    value is AGENT-side — on a voice call that bundles mis-transcription
    with outright invention (indistinguishable without audio), while a wrong
    value the caller DID utter is caller-side conveyance, not an agent
    fabrication.
(c) **transient fabrications** — a wrong arg sent and corrected later is still
    a fabrication event; ``corrected_later`` marks it transient rather than
    excusing it.

Matching is FOLD-AWARE per entity kind so orthography variants are never
booked as fabrications: names and free text compare case- and diacritic-folded
on word tokens (``Álvaro Domínguez`` == ``alvaro dominguez``), ids compare on
the alphanumeric skeleton (``laura_ortega_8020`` == ``Laura Ortega 8020``),
phones and zips compare digits-only, and dates compare on a canonical
``YYYY-MM-DD`` parsed from ISO, numeric, or English-prose forms (``March 21,
1992`` == ``1992-03-21``). This mirrors ``fold_for_match`` (the runtime lookup
fold), so joins across the two stay meaningful.

Known deterministic limits, stated rather than papered over:
- conveyance detection reads the caller's stored text (chunk content /
  messages); an entity conveyed only through language-native month words or
  fully verbalized digits can be missed, which UNDERcounts conveyances and
  can only over-count fabrications toward wrong-but-heard, never invent one;
- a wrong arg is attributed to the single pinned entity of its kind; when a
  task pins several entities of one kind the attribution picks the first in
  pinned order and flags ``assignment_ambiguous``.

Join contract: records carry ``sim_id`` + ``task_id`` + ``entity_kind`` +
``entity_value``, so the fce30 first-critical-error labels (keyed by call id)
join on ``sim_id`` and, where they name an entity, on ``(entity_kind,
entity_value_normalized)``.
"""

import hashlib
import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Iterable, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from tau2.data_model.tasks import Task
from tau2.metrics.call_timeline import (
    ToolEvent,
    extract_caller_units,
    extract_tool_events,
)
from tau2.metrics.run_loading import LoadedCall, RunMeta, iter_loaded_calls
from tau2.utils.text_match import fold_for_match
from tau2.utils.utils import get_now

#: Version of the tracing rules. Bump on any change to entity pinning,
#: normalization, arg-kind assignment, or event semantics.
ENTITY_TRACE_VERSION = "1.0.0"


class EntityKind(str, Enum):
    """The identity entity classes the tracer pins and follows."""

    NAME = "name"
    USER_ID = "user_id"
    CUSTOMER_ID = "customer_id"
    EMAIL = "email"
    PHONE = "phone"
    DATE = "date"
    ZIP = "zip"


# ---------------------------------------------------------------------------
# Kind-aware normalization (fold-aware matching)
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)")
_PROSE_DATE_RE = re.compile(
    r"\b([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\dA-Za-z])\+?\d[\d\-\.\s\(\)]{5,}\d(?![\dA-Za-z])")
_CUSTOMER_ID_RE = re.compile(r"(?<![A-Za-z0-9])C\d{3,}(?![A-Za-z0-9])")
_ZIP_CONTEXT_RE = re.compile(r"zip\s*code\s+(?:is\s+)?(\d{4,6})", re.IGNORECASE)
# "You are Marcus Delgado ..." / "Your name is Javier Dominguez ..." — the
# caller-name anchors of name-authenticated domains' known_info prose (the
# first is the anchor the text-noise entity pinning uses; the second appears
# in retail identity variants, including the typo'd "You name is ...").
_CALLER_NAME_RE = re.compile(
    r"[Yy]ou(?:r)? (?:are|name is) ([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+)"
)


# Fixed in-code digit-word tables (machine principle: reviewed data, never
# improvised). Voice callers routinely SPELL digit entities ("cero, siete,
# cinco, ocho, cuatro" for 07584), so digit containment folds these words to
# digits first. Only word-delimited scripts are mapped: in ko/zh the digit
# morphemes are everyday syllables (이, 一) and folding them would inject
# noise into ordinary prose, while their simulators write digit entities as
# numerals in the stored text.
_DIGIT_WORDS: dict[str, dict[str, str]] = {
    "en": {
        "zero": "0",
        "oh": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    },
    "es": {
        "cero": "0",
        "uno": "1",
        "dos": "2",
        "tres": "3",
        "cuatro": "4",
        "cinco": "5",
        "seis": "6",
        "siete": "7",
        "ocho": "8",
        "nueve": "9",
    },
    "pt": {
        "zero": "0",
        "um": "1",
        "dois": "2",
        "tres": "3",  # compared post-fold, so 'três' arrives accent-stripped
        "quatro": "4",
        "cinco": "5",
        "seis": "6",
        "meia": "6",  # Brazilian phone-number convention
        "sete": "7",
        "oito": "8",
        "nove": "9",
    },
    "hi": {
        "शून्य": "0",
        "एक": "1",
        "दो": "2",
        "तीन": "3",
        "चार": "4",
        "पांच": "5",
        "पाँच": "5",
        "छह": "6",
        "सात": "7",
        "आठ": "8",
        "नौ": "9",
    },
}


def _digit_squash(text: str, language: str) -> str:
    """The text's digit string, folding spelled digit words to digits first."""
    words = _DIGIT_WORDS.get(language)
    if words:
        folded = fold_for_match(text)
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(fold_for_match(w)) for w in words) + r")\b"
        )
        table = {fold_for_match(word): digit for word, digit in words.items()}
        text = pattern.sub(lambda match: table[match.group(1)], folded)
    return _digits(text)


def _fold_words(value: str) -> str:
    """Case/diacritic-folded word skeleton: 'Álvaro  Domínguez' -> 'alvaro dominguez'."""
    return " ".join(re.findall(r"\w+", fold_for_match(value), flags=re.UNICODE))


def _fold_compact(value: str) -> str:
    """Alphanumeric-only fold: 'laura_ortega_8020' -> 'lauraortega8020'."""
    return "".join(ch for ch in fold_for_match(value) if ch.isalnum())


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def normalize_date(value: str) -> str:
    """Canonical ``YYYY-MM-DD`` for ISO / numeric / English-prose dates.

    Falls back to the digit string when no known shape parses, so two
    unparseable-but-identical strings still compare equal.
    """
    text = value.strip()
    match = _ISO_DATE_RE.search(text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    match = _NUMERIC_DATE_RE.search(text)
    if match:
        month, day, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    match = _PROSE_DATE_RE.search(text)
    if match:
        month = _MONTHS.get(match.group(1).lower())
        if month is not None:
            return f"{match.group(3)}-{month:02d}-{int(match.group(2)):02d}"
    return _digits(text)


def normalize_entity_value(kind: EntityKind, value: str) -> str:
    """The canonical comparison form for one entity kind."""
    if kind in (EntityKind.PHONE, EntityKind.ZIP):
        return _digits(value)
    if kind in (EntityKind.USER_ID, EntityKind.CUSTOMER_ID):
        return _fold_compact(value)
    if kind is EntityKind.EMAIL:
        return fold_for_match(value)
    if kind is EntityKind.DATE:
        return normalize_date(value)
    return _fold_words(value)


def value_conveyed_in(
    kind: EntityKind, normalized: str, text: str, language: str = "en"
) -> bool:
    """Was this (already normalized) value conveyed in one caller unit's text?

    Kind-aware containment: digit values match on the unit's digit squash
    (with the language's spelled digit words folded to digits first),
    ids/emails on its alphanumeric fold, dates on any date mention the unit
    carries, and names when every name token appears in the folded text.
    """
    if not normalized:
        return False
    if kind in (EntityKind.PHONE, EntityKind.ZIP):
        return normalized in _digit_squash(text, language)
    if kind in (EntityKind.USER_ID, EntityKind.CUSTOMER_ID):
        return normalized in _fold_compact(text)
    if kind is EntityKind.EMAIL:
        return normalized in fold_for_match(text)
    if kind is EntityKind.DATE:
        mentions = (
            [m.group(0) for m in _ISO_DATE_RE.finditer(text)]
            + [m.group(0) for m in _NUMERIC_DATE_RE.finditer(text)]
            + [m.group(0) for m in _PROSE_DATE_RE.finditer(text)]
        )
        return any(normalize_date(mention) == normalized for mention in mentions)
    folded = _fold_words(text)
    tokens = [token for token in normalized.split() if len(token) > 1]
    if not tokens:
        return False
    # ASCII tokens match as whole words; CJK/other-script tokens match as
    # substrings — \b never fires between adjacent CJK word characters.
    return all(
        re.search(rf"\b{re.escape(token)}\b", folded)
        if token.isascii()
        else token in folded
        for token in tokens
    )


# ---------------------------------------------------------------------------
# Ground-truth entity pinning (the task is what the caller knows)
# ---------------------------------------------------------------------------


class TracedEntity(BaseModel):
    """One ground-truth identity entity pinned for tracing."""

    model_config = {"frozen": True}

    kind: Annotated[EntityKind, Field(description="Entity class.")]
    value: Annotated[
        str, Field(description="The entity string as the task carries it.")
    ]
    normalized: Annotated[
        str, Field(description="Kind-canonical comparison form of the value.")
    ]
    source: Annotated[
        str,
        Field(
            description="Where the value was pinned from: 'scenario' "
            "(persona/instructions prose), 'init_action' (structured "
            "initialization arguments), or 'derived_from_user_id' (display "
            "name reconstructed from a first_last_1234 handle)."
        ),
    ]


def _scenario_text(task: Task) -> str:
    from tau2.multilingual.invariants import INSTRUCTION_FIELDS

    scenario = task.user_scenario
    parts = [scenario.persona or ""]
    instructions = scenario.instructions
    if isinstance(instructions, str):
        parts.append(instructions)
    else:
        parts.extend(
            getattr(instructions, field, None) or "" for field in INSTRUCTION_FIELDS
        )
    return "\n".join(part for part in parts if part)


def expected_entities(task: Task, domain: str) -> list[TracedEntity]:
    """The caller's ground-truth identity entities, in stable pinned order.

    Sources, in precedence order: structured initialization actions
    (telecom's ``set_user_info`` name/phone), then the scenario prose
    (emails, ``first_last_1234`` handles, ``C####`` customer ids, dates,
    phone-shaped digit runs, zip codes, and the "You are <Name>" caller-name
    anchor), then a display name derived from the user-id handle when no
    prose name exists (retail identity variants anchor the caller by handle
    while the auth tool takes first/last name). Values a caller reads off a
    record rather than knows about themself (reservation codes, order ids,
    amounts) are deliberately NOT pinned — same scoping as the text-noise
    entity pinning.
    """
    from tau2.multilingual.invariants import CALLER_USER_ID_RE

    entities: list[TracedEntity] = []
    seen: set[tuple[EntityKind, str]] = set()

    def _add(kind: EntityKind, value: str, source: str) -> None:
        normalized = normalize_entity_value(kind, value)
        if not value or not normalized:
            return
        key = (kind, normalized)
        if key in seen:
            return
        seen.add(key)
        entities.append(
            TracedEntity(kind=kind, value=value, normalized=normalized, source=source)
        )

    initial_state = task.initial_state
    for action in (
        initial_state.initialization_actions if initial_state is not None else None
    ) or []:
        if getattr(action, "func_name", None) != "set_user_info":
            continue
        arguments = getattr(action, "arguments", None) or {}
        if arguments.get("name"):
            _add(EntityKind.NAME, str(arguments["name"]), "init_action")
        if arguments.get("phone_number"):
            _add(EntityKind.PHONE, str(arguments["phone_number"]), "init_action")

    text = _scenario_text(task)
    for match in _EMAIL_RE.finditer(text):
        _add(EntityKind.EMAIL, match.group(), "scenario")
    user_ids = [match.group() for match in CALLER_USER_ID_RE.finditer(text)]
    for user_id in user_ids:
        _add(EntityKind.USER_ID, user_id, "scenario")
    for match in _CUSTOMER_ID_RE.finditer(text):
        _add(EntityKind.CUSTOMER_ID, match.group(), "scenario")
    for match in _ISO_DATE_RE.finditer(text):
        _add(EntityKind.DATE, match.group(), "scenario")
    for match in _PROSE_DATE_RE.finditer(text):
        if match.group(1).lower() in _MONTHS:
            _add(EntityKind.DATE, match.group(), "scenario")
    for match in _ZIP_CONTEXT_RE.finditer(text):
        _add(EntityKind.ZIP, match.group(1), "scenario")
    for match in _PHONE_RE.finditer(text):
        digits = _digits(match.group())
        if len(digits) >= 7 and not _ISO_DATE_RE.fullmatch(match.group().strip()):
            _add(EntityKind.PHONE, match.group().strip(), "scenario")
    for match in _CALLER_NAME_RE.finditer(text):
        _add(EntityKind.NAME, match.group(1), "scenario")

    has_name = any(entity.kind is EntityKind.NAME for entity in entities)
    if not has_name and user_ids:
        # first_last_1234 -> "first last": the display name a caller gives a
        # name-keyed finder even when the scenario anchors them by handle.
        parts = user_ids[0].split("_")
        if len(parts) >= 3:
            _add(
                EntityKind.NAME,
                " ".join(parts[:-1]),
                "derived_from_user_id",
            )
    _ = domain  # pinning is domain-independent today; the seam stays explicit
    return entities


# ---------------------------------------------------------------------------
# Tool-argument kind assignment
# ---------------------------------------------------------------------------

_ARG_NAME_KINDS: tuple[tuple[re.Pattern, EntityKind], ...] = (
    (re.compile(r"phone"), EntityKind.PHONE),
    (re.compile(r"email"), EntityKind.EMAIL),
    (re.compile(r"dob|birth"), EntityKind.DATE),
    (re.compile(r"zip"), EntityKind.ZIP),
    (re.compile(r"user_?id"), EntityKind.USER_ID),
    (re.compile(r"customer_?id"), EntityKind.CUSTOMER_ID),
    (re.compile(r"^(first_?name|last_?name|full_?name|name)$"), EntityKind.NAME),
)


def _arg_leaf(arg_name: str) -> str:
    """The naming segment of a dotted arg path: 'item_ids.0' -> 'item_ids'."""
    segments = [s for s in arg_name.split(".") if not s.isdigit()]
    return (segments[-1] if segments else arg_name).casefold()


def _kind_for_arg(arg_name: str, value: str) -> Optional[EntityKind]:
    """Which entity kind a tool argument carries, or None (not traced).

    The argument NAME decides when it matches a known pattern; otherwise the
    VALUE's shape decides (an email-shaped string is an email whatever the
    parameter is called). Record-handle arguments (``order_id``,
    ``item_ids``, ``product_id``, ...) are values the caller reads off a
    record, not identity conveyance — they never enter the shape fallback,
    so a digit-run item id cannot masquerade as a phone number. Arguments
    that match nothing are skipped.
    """
    leaf = _arg_leaf(arg_name)
    for pattern, kind in _ARG_NAME_KINDS:
        if pattern.search(leaf):
            return kind
    if re.search(r"(^|_)ids?$", leaf):
        return None  # record handle, not identity conveyance
    stripped = value.strip()
    if _EMAIL_RE.fullmatch(stripped):
        return EntityKind.EMAIL
    if _ISO_DATE_RE.fullmatch(stripped):
        return EntityKind.DATE
    from tau2.multilingual.invariants import CALLER_USER_ID_RE

    if CALLER_USER_ID_RE.fullmatch(stripped):
        return EntityKind.USER_ID
    if _CUSTOMER_ID_RE.fullmatch(stripped):
        return EntityKind.CUSTOMER_ID
    if (
        len(_digits(stripped)) >= 7
        and _PHONE_RE.fullmatch(stripped)
        and (stripped.startswith("+") or re.search(r"[\-\.\s\(\)]", stripped))
    ):
        # A bare digit run is ambiguous (ids, codes); only separator-shaped
        # or +-prefixed strings read as phones without a naming hint.
        return EntityKind.PHONE
    return None


def _iter_scalar_args(arguments: dict, prefix: str = "") -> Iterable[tuple[str, str]]:
    """Flatten tool arguments to (dotted_name, string_value) scalar leaves."""
    for name, value in (arguments or {}).items():
        path = f"{prefix}{name}"
        if isinstance(value, dict):
            yield from _iter_scalar_args(value, prefix=f"{path}.")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    yield from _iter_scalar_args(item, prefix=f"{path}.{index}.")
                elif isinstance(item, (str, int, float)) and not isinstance(item, bool):
                    yield f"{path}.{index}", str(item)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            yield path, str(value)


def _name_realizations(event: ToolEvent) -> list[tuple[str, str]]:
    """NAME-kind realizations of one call, combining first+last name args.

    A finder that takes ``first_name`` + ``last_name`` conveys ONE name; the
    two parameters are combined into a single realization so a wrong last
    name is one wrong-name event, not two half-events.
    """
    scalars = dict(_iter_scalar_args(event.call.arguments or {}))
    first = next(
        (v for k, v in scalars.items() if re.search(r"first_?name", k, re.I)), None
    )
    last = next(
        (v for k, v in scalars.items() if re.search(r"last_?name", k, re.I)), None
    )
    realizations: list[tuple[str, str]] = []
    if first is not None or last is not None:
        combined = " ".join(part for part in (first, last) if part)
        realizations.append(("first_name+last_name", combined))
    for key, value in scalars.items():
        if re.fullmatch(r"(full_?name|name)", _arg_leaf(key)):
            realizations.append((key, value))
    return realizations


# ---------------------------------------------------------------------------
# Read-only vs state-changing tool classification
# ---------------------------------------------------------------------------


def tool_write_map(domain: str) -> Optional[dict[str, bool]]:
    """tool name -> mutates_state for one domain, from the domain toolkit.

    Introspects the registered environment (deterministic, no LLM). None when
    the environment cannot be constructed without extra configuration (e.g. a
    retrieval-config domain) — events then carry ``tool_write=None`` and the
    read/write split stays un-attributed rather than guessed.
    """
    from tau2.registry import registry

    try:
        environment = registry.get_env_constructor(domain)()
        toolkit = environment.tools
        return {name: bool(toolkit.tool_mutates_state(name)) for name in toolkit.tools}
    except Exception as error:  # loud, but one domain must not kill a batch
        logger.warning(
            f"could not introspect domain '{domain}' tools ({error}); "
            "read/write split will be unattributed for its calls"
        )
        return None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class EntityArgEvent(BaseModel):
    """One realization of a pinned entity as a tool-call argument."""

    order_key: Annotated[
        int, Field(description="Position of the tool call on the call's order axis.")
    ]
    tool_name: Annotated[str, Field(description="Tool the argument was sent to.")]
    tool_call_id: Annotated[
        str, Field(description="Recorded tool-call id ('' when absent).")
    ] = ""
    tool_write: Annotated[
        Optional[bool],
        Field(
            description="True when the tool mutates environment state, False "
            "for read-only tools, None when the domain toolkit could not be "
            "introspected."
        ),
    ] = None
    arg_name: Annotated[
        str,
        Field(
            description="Dotted argument path ('first_name+last_name' for a "
            "combined name realization)."
        ),
    ]
    value_raw: Annotated[str, Field(description="Argument value as sent.")]
    value_normalized: Annotated[
        str, Field(description="Kind-canonical form of the argument value.")
    ]
    correct: Annotated[
        bool,
        Field(description="Argument matches the ground-truth value (fold-aware)."),
    ]
    value_uttered_by_caller_before: Annotated[
        Optional[bool],
        Field(
            description="For a WRONG argument: had the caller uttered this "
            "exact (normalized) value in any earlier unit? None on correct "
            "arguments."
        ),
    ] = None
    value_seen_in_prior_tool_result: Annotated[
        Optional[bool],
        Field(
            description="For a WRONG argument: did the value appear in an "
            "earlier SUCCESSFUL tool result? True means the agent read it "
            "off a record (a different semantic slot, e.g. an address zip "
            "vs the caller's identity zip) — excluded from the wrong/"
            "fabrication counts. None on correct arguments."
        ),
    ] = None
    fabricated: Annotated[
        bool,
        Field(
            description="Wrong AND never uttered by the caller before the "
            "call was made — an agent-side value (invention, or on voice "
            "calls a mis-transcription; the two are indistinguishable "
            "without audio)."
        ),
    ] = False
    corrected_later: Annotated[
        bool,
        Field(
            description="A later realization of the same entity was correct "
            "(the wrong value was transient). Still counts as a wrong/"
            "fabrication event."
        ),
    ] = False
    assignment_ambiguous: Annotated[
        bool,
        Field(
            description="The task pins several entities of this kind and the "
            "wrong value matched none, so attribution picked the first in "
            "pinned order."
        ),
    ] = False


class EntityTraceRecord(BaseModel):
    """The full trace of one pinned entity through one call."""

    results_path: Annotated[str, Field(description="Path of the run's results.json.")]
    experiment_label: Annotated[str, Field(description="Source-run dir name.")]
    sim_id: Annotated[str, Field(description="Simulation id (join key).")]
    task_id: Annotated[str, Field(description="Task id (join key).")]
    trial: Annotated[int, Field(description="Trial index.")]
    language: Annotated[str, Field(description="ISO 639-1 language of the call.")]
    domain: Annotated[str, Field(description="Domain of the run.")]
    modality: Annotated[str, Field(description="'voice' or 'text' (run-level).")]
    provider: Annotated[
        str, Field(description="Audio-native provider ('' on text runs).")
    ]
    agent_model: Annotated[str, Field(description="Agent model of the run.")]
    entity_kind: Annotated[EntityKind, Field(description="Entity class (join key).")]
    entity_value: Annotated[
        str, Field(description="Ground-truth value as the task carries it.")
    ]
    entity_value_normalized: Annotated[
        str, Field(description="Kind-canonical comparison form (join key).")
    ]
    entity_source: Annotated[
        str, Field(description="Where the ground truth was pinned from.")
    ]
    first_conveyance_order: Annotated[
        Optional[int],
        Field(
            description="Order key of the first caller unit conveying the "
            "entity; None when conveyance was never detected in the caller's "
            "stored text."
        ),
    ] = None
    conveyance_count: Annotated[
        int,
        Field(description="Caller units in which the entity was conveyed."),
    ] = 0
    first_capture_correct: Annotated[
        Optional[bool],
        Field(
            description="Whether the entity's FIRST realization as a tool "
            "argument matched the ground truth; None when the entity never "
            "reached a tool argument."
        ),
    ] = None
    reached_tool_args: Annotated[
        bool, Field(description="The entity was realized as a tool argument.")
    ] = False
    wrong_arg_event_count: Annotated[
        int, Field(description="Realizations that did not match the ground truth.")
    ] = 0
    fabricated_event_count: Annotated[
        int,
        Field(
            description="Wrong realizations whose value the caller never "
            "uttered beforehand (the agent invented them)."
        ),
    ] = 0
    transient_fabrication_count: Annotated[
        int,
        Field(
            description="Fabricated events later followed by a correct "
            "realization of the same entity — corrected, but still counted."
        ),
    ] = 0
    read_only_wrong_arg_count: Annotated[
        int, Field(description="Wrong realizations sent to read-only tools.")
    ] = 0
    state_changing_wrong_arg_count: Annotated[
        int, Field(description="Wrong realizations sent to state-changing tools.")
    ] = 0
    read_only_fabricated_count: Annotated[
        int, Field(description="Fabricated realizations sent to read-only tools.")
    ] = 0
    state_changing_fabricated_count: Annotated[
        int,
        Field(
            description="Fabricated realizations sent to state-changing tools "
            "— the authentication-robustness gate feed."
        ),
    ] = 0
    events: Annotated[
        list[EntityArgEvent],
        Field(default_factory=list, description="Every realization, in order."),
    ]


class EntityTraceCallRollup(BaseModel):
    """Per-call rollup over that call's entity records."""

    results_path: str
    experiment_label: str
    sim_id: str
    task_id: str
    trial: int
    language: str
    domain: str
    modality: str
    provider: str
    agent_model: str
    reward: Annotated[
        Optional[float], Field(description="Recorded task reward, if any.")
    ] = None
    termination_reason: Annotated[str, Field(description="How the sim terminated.")] = (
        ""
    )
    entity_count: Annotated[int, Field(description="Pinned entities in the task.")]
    entities_reaching_tool_args: Annotated[
        int, Field(description="Entities realized as tool arguments at least once.")
    ]
    entities_first_capture_correct: Annotated[
        int, Field(description="Entities whose first realization was correct.")
    ]
    wrong_arg_event_count: Annotated[int, Field(description="Sum over entities.")]
    fabricated_event_count: Annotated[int, Field(description="Sum over entities.")]
    transient_fabrication_count: Annotated[int, Field(description="Sum over entities.")]
    state_changing_fabrication_count: Annotated[
        int,
        Field(description="Fabricated values that reached state-changing tools."),
    ]
    any_fabrication_reached_state_change: Annotated[
        bool,
        Field(description="The per-call authentication-robustness gate bit."),
    ]


class EntityTraceConfig(BaseModel):
    """Extraction scope, recorded in the artifact's provenance."""

    langs: Annotated[
        Optional[list[str]],
        Field(description="Restrict to these ISO 639-1 codes (None = all)."),
    ] = None
    domain: Annotated[
        Optional[str], Field(description="Restrict to one domain (None = all).")
    ] = None
    max_sims: Annotated[
        Optional[int], Field(description="Cap on traced sims (dry runs / smokes).")
    ] = None
    include_events: Annotated[
        bool,
        Field(
            description="Carry per-realization events on each record (False "
            "keeps only the per-entity counters, for large pools)."
        ),
    ] = True

    @field_validator("langs")
    @classmethod
    def _fold_langs(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return None if value is None else [item.strip().lower() for item in value]


class EntityTraceSourceRun(BaseModel):
    """Per-run provenance: run identity plus how many calls it contributed."""

    meta: RunMeta
    n_calls: Annotated[int, Field(description="Calls traced from this run.")]


class EntityTraceArtifact(BaseModel):
    """The provenance-bearing entity-trace artifact (SHADOW metric)."""

    schema_version: int = 1
    instrument_version: Annotated[
        str, Field(description="ENTITY_TRACE_VERSION at build time.")
    ] = ENTITY_TRACE_VERSION
    artifact_id: Annotated[
        str,
        Field(
            description="Content-derived id: sha256(version, config, records, "
            "rollups) truncated to 12 hex — identical inputs reproduce it."
        ),
    ]
    created_at: Annotated[str, Field(description="Build wall-clock time.")]
    git_sha: Annotated[str, Field(description="Repo HEAD at build time.")]
    config: EntityTraceConfig
    source_runs: list[EntityTraceSourceRun]
    records: Annotated[
        list[EntityTraceRecord], Field(description="One record per (call, entity).")
    ]
    rollups: Annotated[
        list[EntityTraceCallRollup], Field(description="One rollup per call.")
    ]


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def _entity_for_arg(
    kind: EntityKind,
    normalized_value: str,
    entities: list[TracedEntity],
) -> tuple[Optional[TracedEntity], bool]:
    """Attribute one argument realization to a pinned entity.

    A value matching an entity of its kind belongs to that entity. A wrong
    value belongs to the single pinned entity of its kind; with several, the
    first in pinned order is picked and the event flagged ambiguous.
    """
    of_kind = [entity for entity in entities if entity.kind is kind]
    if not of_kind:
        return None, False
    for entity in of_kind:
        if entity.normalized == normalized_value:
            return entity, False
    return of_kind[0], len(of_kind) > 1


def trace_call_entities(
    call: LoadedCall,
    *,
    write_map: Optional[dict[str, bool]] = None,
    include_events: bool = True,
) -> tuple[list[EntityTraceRecord], EntityTraceCallRollup]:
    """Trace every pinned entity of one call through its tool trace."""
    meta, sim = call.meta, call.sim
    entities = expected_entities(call.task, meta.domain) if call.task else []
    units = extract_caller_units(sim)
    tool_events = extract_tool_events(sim)

    events_by_entity: dict[tuple[EntityKind, str], list[EntityArgEvent]] = {
        (entity.kind, entity.normalized): [] for entity in entities
    }
    # SUCCESSFUL result payloads: values the agent can legitimately read off
    # a record. Error results are excluded — they echo the failed call's own
    # arguments ("Error: user X not found") and would launder a repeated
    # mishearing into a record-sourced value.
    result_sources: list[tuple[int, str, str]] = [
        (event.result_order_key, event.call.id or "", event.result_content)
        for event in tool_events
        if event.result_order_key is not None
        and event.result_error is False
        and event.result_content
    ]
    for tool_event in tool_events:
        realizations: list[tuple[str, str, EntityKind]] = []
        seen_name_args: set[str] = set()
        for arg_name, combined in _name_realizations(tool_event):
            realizations.append((arg_name, combined, EntityKind.NAME))
            seen_name_args.add(arg_name)
        for arg_name, value in _iter_scalar_args(tool_event.call.arguments or {}):
            leaf = _arg_leaf(arg_name)
            if re.fullmatch(r"(first_?name|last_?name|full_?name|name)", leaf):
                continue  # realized through _name_realizations
            kind = _kind_for_arg(arg_name, value)
            if kind is None:
                continue
            realizations.append((arg_name, value, kind))
        for arg_name, value, kind in realizations:
            normalized = normalize_entity_value(kind, value)
            if not normalized:
                continue
            entity, ambiguous = _entity_for_arg(kind, normalized, entities)
            if entity is None:
                continue
            correct = normalized == entity.normalized
            uttered: Optional[bool] = None
            record_sourced: Optional[bool] = None
            if not correct:
                # A wrong value that is a substring/superstring of the truth
                # is a corruption of the true value (dropped or inserted
                # characters in transcription), never an independently
                # uttered value — containment against the caller's text
                # would false-match it against the TRUE value's utterance
                # ('551232002' is inside '5551232002').
                artifact_of_truth = (
                    normalized in entity.normalized or entity.normalized in normalized
                )
                uttered = not artifact_of_truth and any(
                    value_conveyed_in(kind, normalized, unit.text, call.language)
                    for unit in units
                    if unit.order_key <= tool_event.order_key
                )
                record_sourced = any(
                    value_conveyed_in(kind, normalized, content, call.language)
                    for order, call_id, content in result_sources
                    if order <= tool_event.order_key
                    and call_id != (tool_event.call.id or "")
                )
            events_by_entity[(entity.kind, entity.normalized)].append(
                EntityArgEvent(
                    order_key=tool_event.order_key,
                    tool_name=tool_event.call.name,
                    tool_call_id=tool_event.call.id or "",
                    tool_write=(
                        write_map.get(tool_event.call.name)
                        if write_map is not None
                        else None
                    ),
                    arg_name=arg_name,
                    value_raw=value,
                    value_normalized=normalized,
                    correct=correct,
                    value_uttered_by_caller_before=uttered,
                    value_seen_in_prior_tool_result=record_sourced,
                    fabricated=(
                        not correct and uttered is False and record_sourced is False
                    ),
                    assignment_ambiguous=ambiguous,
                )
            )

    records: list[EntityTraceRecord] = []
    for entity in entities:
        events = events_by_entity[(entity.kind, entity.normalized)]
        events.sort(key=lambda event: event.order_key)
        # corrected_later: a wrong event followed by a correct realization.
        for index, event in enumerate(events):
            if not event.correct:
                event.corrected_later = any(
                    later.correct for later in events[index + 1 :]
                )
        conveyed_orders = [
            unit.order_key
            for unit in units
            if value_conveyed_in(
                entity.kind, entity.normalized, unit.text, call.language
            )
        ]
        # Record-sourced wrong values are a different semantic slot (the agent
        # read them off a record), not attempts to realize THIS entity: they
        # stay visible as flagged events but never enter the counts or decide
        # first capture.
        eligible = [
            event
            for event in events
            if event.correct or not event.value_seen_in_prior_tool_result
        ]
        wrong = [event for event in eligible if not event.correct]
        fabricated = [event for event in wrong if event.fabricated]
        records.append(
            EntityTraceRecord(
                results_path=meta.results_path,
                experiment_label=meta.experiment_label,
                sim_id=str(sim.id),
                task_id=str(sim.task_id),
                trial=sim.trial or 0,
                language=call.language,
                domain=meta.domain,
                modality=meta.modality,
                provider=meta.provider,
                agent_model=meta.agent_model,
                entity_kind=entity.kind,
                entity_value=entity.value,
                entity_value_normalized=entity.normalized,
                entity_source=entity.source,
                first_conveyance_order=(
                    min(conveyed_orders) if conveyed_orders else None
                ),
                conveyance_count=len(conveyed_orders),
                first_capture_correct=(eligible[0].correct if eligible else None),
                reached_tool_args=bool(eligible),
                wrong_arg_event_count=len(wrong),
                fabricated_event_count=len(fabricated),
                transient_fabrication_count=sum(
                    1 for event in fabricated if event.corrected_later
                ),
                read_only_wrong_arg_count=sum(
                    1 for event in wrong if event.tool_write is False
                ),
                state_changing_wrong_arg_count=sum(
                    1 for event in wrong if event.tool_write is True
                ),
                read_only_fabricated_count=sum(
                    1 for event in fabricated if event.tool_write is False
                ),
                state_changing_fabricated_count=sum(
                    1 for event in fabricated if event.tool_write is True
                ),
                events=events if include_events else [],
            )
        )

    rollup = EntityTraceCallRollup(
        results_path=meta.results_path,
        experiment_label=meta.experiment_label,
        sim_id=str(sim.id),
        task_id=str(sim.task_id),
        trial=sim.trial or 0,
        language=call.language,
        domain=meta.domain,
        modality=meta.modality,
        provider=meta.provider,
        agent_model=meta.agent_model,
        reward=(sim.reward_info.reward if sim.reward_info else None),
        termination_reason=(
            sim.termination_reason.value if sim.termination_reason else ""
        ),
        entity_count=len(records),
        entities_reaching_tool_args=sum(
            1 for record in records if record.reached_tool_args
        ),
        entities_first_capture_correct=sum(
            1 for record in records if record.first_capture_correct is True
        ),
        wrong_arg_event_count=sum(record.wrong_arg_event_count for record in records),
        fabricated_event_count=sum(record.fabricated_event_count for record in records),
        transient_fabrication_count=sum(
            record.transient_fabrication_count for record in records
        ),
        state_changing_fabrication_count=sum(
            record.state_changing_fabricated_count for record in records
        ),
        any_fabrication_reached_state_change=any(
            record.state_changing_fabricated_count > 0 for record in records
        ),
    )
    return records, rollup


def _derive_artifact_id(
    config: EntityTraceConfig,
    records: list[EntityTraceRecord],
    rollups: list[EntityTraceCallRollup],
) -> str:
    digest = hashlib.sha256()
    digest.update(ENTITY_TRACE_VERSION.encode())
    digest.update(b"\0")
    digest.update(config.model_dump_json().encode())
    for record in records:
        digest.update(b"\0")
        digest.update(record.model_dump_json().encode())
    for rollup in rollups:
        digest.update(b"\0")
        digest.update(rollup.model_dump_json().encode())
    return digest.hexdigest()[:12]


def build_entity_trace(
    paths: Iterable[Path | str],
    config: Optional[EntityTraceConfig] = None,
) -> EntityTraceArtifact:
    """Trace every call under ``paths`` and assemble the artifact."""
    from tau2.annotation.artifacts import git_sha

    cfg = config or EntityTraceConfig()
    write_maps: dict[str, Optional[dict[str, bool]]] = {}
    records: list[EntityTraceRecord] = []
    rollups: list[EntityTraceCallRollup] = []
    calls_per_run: dict[str, int] = {}
    metas: dict[str, RunMeta] = {}
    for call in iter_loaded_calls(
        paths, langs=cfg.langs, domain=cfg.domain, max_sims=cfg.max_sims
    ):
        domain = call.meta.domain
        if domain not in write_maps:
            write_maps[domain] = tool_write_map(domain)
        call_records, rollup = trace_call_entities(
            call,
            write_map=write_maps[domain],
            include_events=cfg.include_events,
        )
        records.extend(call_records)
        rollups.append(rollup)
        metas[call.meta.results_path] = call.meta
        calls_per_run[call.meta.results_path] = (
            calls_per_run.get(call.meta.results_path, 0) + 1
        )
    return EntityTraceArtifact(
        artifact_id=_derive_artifact_id(cfg, records, rollups),
        created_at=get_now(),
        git_sha=git_sha(),
        config=cfg,
        source_runs=[
            EntityTraceSourceRun(meta=metas[path], n_calls=count)
            for path, count in sorted(calls_per_run.items())
        ],
        records=records,
        rollups=rollups,
    )
