# Copyright Sierra
"""The v4 intake task generator: enumerate, verify, freeze (design doc §6).

Everything a task carries is a deterministic function of the checked-in
banks, the fixed in-code catalogs in this module, the versioned call-frame
templates (:mod:`tau2.domains.intake.call_frame`), and one seed — no LLM, no
improvised prose. ``tau2 intake-tasks freeze`` regenerates the canonical set
byte-identically; a code change that would alter it is visible as a diff on
``tasks.json`` (and blocked by the freeze regression test until
:data:`GENERATOR_VERSION` moves).

**The canonical set** (first freeze, owner-approved 2026-08-24; narrowed to
the final benchmark 10 banks 2026-08-26, :data:`CANONICAL_BANKS`): 10 banks ×
2 tiers × 10 calls = 200 atomic (n=1) tasks. The five retired banks
(amounts, insurance_plans, rate_plans, shops, vehicles) keep their data
files and machinery — they are simply not drawn into the canonical set
(the dropped-languages precedent). Each cell draws
:data:`DRAW_PER_CELL` of its bank's 40 contract values WITHOUT replacement
(named rng stream ``intake-freeze|cell|{bank}|{tier}|{seed}``), so the pool
covers two additive doublings (10 → 20 → 40) with no value reuse. Filler
context (callee, org, submission date, the one filled context field) comes
from separate ``intake-freeze|filler|{task_id}|{seed}`` streams so a filler
change can never perturb a measured draw.

**The bank is the unit**: :data:`_UNIT_RESOLVERS` maps every bank entry to
its one record field (vertical, field name, type description, fold rule).
Multi-valued banks split by their closed tag — code family, date kind,
coined role — exactly as the design §11 owner decisions assign them.

**Verification gates** (generation time, per task, design §6): the
derivation replays through the REAL toolkits (seed → blank → order →
entities), the callback order must name exactly the missing fields, the
golden ``submit_fields`` call must reach the expected final DB (the seeded
complete record under the published folds), values must stay fold-distinct
within the task, and the serialized task must be pure ASCII. A gate failure
aborts the freeze — nothing partial is written.

The ``n_entities`` knob (1|2|3|5) is kept for later additive draws
(``tau2 intake-tasks draw``): n>1 bundles n distinct same-vertical bank
units per call, drawn per (vertical × tier) cell. The canonical set is
atomic-only; multi-entity bands are additive draws, never part of the
freeze (design §1).
"""

import json
import random
from pathlib import Path
from typing import Annotated, Literal, Optional

from loguru import logger
from pydantic import Field

from tau2.data_model.tasks import (
    Action,
    Description,
    EnvFunctionCall,
    EvaluationCriteria,
    InitialState,
    Task,
)
from tau2.domains.intake.call_frame import (
    CALL_FRAME_VERSION,
    render_opener,
    render_user_scenario,
)
from tau2.domains.intake.data_model import IntakeDB, Vertical
from tau2.domains.intake.folds import FoldKind, fold_value
from tau2.domains.intake.tasks.banks import (
    BANK_CONTRACT_PER_TIER,
    BANK_NAMES,
    Banks,
    CodeFamily,
    CoinedRole,
    DateKind,
    Difficulty,
    assert_bank_contract,
    instantiate_email,
    load_banks,
)
from tau2.domains.intake.tools import IntakeTools
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.domains.intake.user_tools import IntakeUserTools
from tau2.domains.intake.utils import INTAKE_DATA_DIR, render_dual_form
from tau2.utils.pydantic_utils import BaseModelNoExtra

GENERATOR_VERSION = "7.0.1"

# The final benchmark bank set (owner decision 2026-08-26, from the
# intake_full300_2026-08-26 entity analysis): every retained bank is either
# a hard complication constraint (medications = syllable_drop + the 0.50
# mispronounced rate; times/dates/person_names = the three lazy_omission
# components) or the sole carrier of a failure channel (codes = structured
# strings + the strongest tier dial, emails = symbol spelling + the floor
# bank, addresses = long free text, phones = ubiquitous digit strings,
# properties = invented names + the phase-3 foreign tokens, coined = the
# hard invented-name representative). Retired: amounts, insurance_plans,
# rate_plans, shops (redundant invented-name/near-ceiling channels) and
# vehicles (mispronounced_term keeps 3 carriers; only 14/92 foreign tokens).
# Bank data and machinery for retired banks stay checked in and usable.
# Order follows BANK_NAMES (the bank-file order) so cell iteration and the
# sequential record ids keep the established convention.
CANONICAL_BANKS: tuple[str, ...] = tuple(
    bank
    for bank in BANK_NAMES
    if bank
    in {
        "person_names",
        "codes",
        "phones",
        "dates",
        "times",
        "addresses",
        "properties",
        "emails",
        "medications",
        "coined",
    }
)

FREEZE_MANIFEST_VERSION = "1.0.0"

DEFAULT_FREEZE_SEED = 20260824
DRAW_PER_CELL = 10
# Bundles draw from CANONICAL_BANKS; the auto vertical offers only 4 of the
# canonical 10, so 5-entity bundles left the frame with the 2026-08-26 cut.
VALID_N_ENTITIES = (1, 2, 3)

TASKS_FILENAME = "tasks.json"
SPLITS_FILENAME = "split_tasks.json"
MANIFEST_FILENAME = "tasks.manifest.json"

RNG_NAMESPACE = "intake-freeze"


class GeneratorError(Exception):
    """A generation gate failed; nothing may be written."""


# ---------------------------------------------------------------------------
# Fixed catalogs (in-code, closed — machine, not scripts)
# ---------------------------------------------------------------------------

# What each record field holds (serves the agent through the callback order).
FIELD_DESCRIPTIONS: dict[str, str] = {
    "contact_phone": "The phone number the customer can be reached back on.",
    "drop_off_date": (
        "The date the vehicle is dropped off at the shop, format YYYY-MM-DD."
    ),
    "check_in_date": "The date the stay begins, format YYYY-MM-DD.",
    "date_of_birth": "The patient's date of birth, format YYYY-MM-DD.",
    "home_address": "The patient's full home address, exactly as written.",
    "appointment_time": (
        "The requested appointment time of day, format H:MM AM or H:MM PM "
        "(the AM/PM part is required)."
    ),
    "approved_budget": (
        "The approved repair budget in US dollars, digits with a decimal "
        "point where cents apply."
    ),
    "vin": (
        "The 17-character vehicle identification number (VIN) of the vehicle "
        "this service request is for."
    ),
    "license_plate": "The vehicle's license plate, exactly as printed.",
    "member_id": (
        "The clinic member ID, format MBR-<core>-<checksum>, exactly as "
        "printed on the membership card."
    ),
    "loyalty_number": (
        "The hotel loyalty program number, exactly as printed on the card."
    ),
    "second_guest_full_name": "The full name of the second guest on the booking.",
    "contact_email": "The email address booking confirmations go to.",
    "destination_property": "The name of the property the stay is booked at.",
    "rate_plan_name": "The name of the rate plan the booking was made under.",
    "company_name": "The company name the booking is billed to.",
    "insurance_plan_name": "The display name of the patient's insurance plan.",
    "current_medication": (
        "The medication the patient currently takes, name and strength as prescribed."
    ),
    "employer_name": "The patient's employer name.",
    "aftermarket_brand": "The brand of the aftermarket part to be installed.",
    "previous_service_shop": "The name of the shop that last serviced the vehicle.",
    "vehicle_model": "The vehicle's make and model as registered.",
}

# Client organizations per vertical (fixed filler catalog; coined, distinct
# from every bank value — asserted at load). The org feeds the deterministic
# opener and is never a measured value.
ORG_CATALOG: dict[Vertical, tuple[str, ...]] = {
    Vertical.CLINIC: (
        "Maple Grove Family Clinic",
        "Northgate Medical Group",
        "Riverbend Health Associates",
        "Cedar Park Primary Care",
        "Stonebridge Community Clinic",
        "Lakeshore Family Practice",
    ),
    Vertical.AUTO: (
        "Beacon Ridge Auto Service",
        "Copperline Garage",
        "Fairhaven Motor Works",
        "Torrey Hills Automotive",
        "Bright Harbor Auto Repair",
        "Silver Creek Service Center",
    ),
    Vertical.HOTEL: (
        "Crestwater Hotels Guest Services",
        "Wayfarer Reservations Desk",
        "Aurora Gate Hospitality",
        "Pinnacle Stay Reservations",
        "Meridian Line Hotels",
        "Opal Coast Booking Office",
    ),
}

# The record's one filled context field per vertical (ordered candidates —
# the first whose field name differs from the measured field is used, so a
# record always shows one completed cell next to the missing one).
_CONTEXT_CANDIDATES: dict[Vertical, tuple[str, ...]] = {
    Vertical.CLINIC: ("contact_phone", "home_address"),
    Vertical.AUTO: ("contact_phone", "drop_off_date"),
    Vertical.HOTEL: ("contact_phone", "check_in_date"),
}

# The submission-date window: the form went in shortly before the pinned
# clock (2025-06-12, tau2.domains.intake.utils.get_now).
_SUBMITTED_ON_DAYS = tuple(range(2, 12))  # 2025-06-02 .. 2025-06-11


class BankUnit(BaseModelNoExtra):
    """One bank entry resolved to the record field it fills (design §4)."""

    bank: Annotated[str, Field(description="Bank the value was drawn from.")]
    vertical: Annotated[Vertical, Field(description="Cover-story vertical.")]
    field_name: Annotated[str, Field(description="Record field the value fills.")]
    fold: Annotated[FoldKind, Field(description="Published fold rule of the field.")]
    value: Annotated[str, Field(description="The pinned true (written) value.")]
    spoken: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "What the callee's records say when it differs from the "
                "written value (relative dates: the spoken form; the agent "
                "resolves it against get_today)."
            ),
        ),
    ]
    email_pattern: Annotated[
        Optional[str],
        Field(
            default=None,
            description="For emails: the owner-linked pattern key that was drawn.",
        ),
    ]

    @property
    def description(self) -> str:
        return FIELD_DESCRIPTIONS[self.field_name]

    @property
    def entity_value(self) -> str:
        """What ``get_entity`` returns. Spoken-form entries (relative dates)
        carry BOTH forms: the sim leads with the spoken form but can give the
        real written value when the caller asks for the exact date — without
        this it has nothing but the phrase and spells THAT out letter by
        letter (2026-08-26 smoke, intake_dates_hard_01)."""
        if self.spoken is None:
            return self.value
        return render_dual_form(self.spoken, self.value)


# Per-bank resolvers: closed tag -> (vertical, field, fold). Emails resolve
# at draw time (owner-linked), so their resolver needs the callee name.

_CODE_FIELDS: dict[CodeFamily, tuple[Vertical, str]] = {
    CodeFamily.VIN: (Vertical.AUTO, "vin"),
    CodeFamily.PLATE: (Vertical.AUTO, "license_plate"),
    CodeFamily.MEMBER_ID: (Vertical.CLINIC, "member_id"),
    CodeFamily.LOYALTY: (Vertical.HOTEL, "loyalty_number"),
}

_DATE_FIELDS: dict[DateKind, tuple[Vertical, str]] = {
    DateKind.BIRTH_DATE: (Vertical.CLINIC, "date_of_birth"),
    DateKind.FUTURE_DATE: (Vertical.AUTO, "drop_off_date"),
    DateKind.RELATIVE_DATE: (Vertical.AUTO, "drop_off_date"),
}

_COINED_FIELDS: dict[CoinedRole, tuple[Vertical, str]] = {
    CoinedRole.EMPLOYER: (Vertical.CLINIC, "employer_name"),
    CoinedRole.COMPANY: (Vertical.HOTEL, "company_name"),
    CoinedRole.BRAND: (Vertical.AUTO, "aftermarket_brand"),
}

_FLAT_UNIT_FIELDS: dict[str, tuple[Vertical, str, FoldKind]] = {
    "person_names": (Vertical.HOTEL, "second_guest_full_name", FoldKind.NAME),
    "phones": (Vertical.AUTO, "contact_phone", FoldKind.PHONE),
    "times": (Vertical.CLINIC, "appointment_time", FoldKind.TIME),
    "addresses": (Vertical.CLINIC, "home_address", FoldKind.NAME),
    "properties": (Vertical.HOTEL, "destination_property", FoldKind.NAME),
    "amounts": (Vertical.AUTO, "approved_budget", FoldKind.AMOUNT),
    "medications": (Vertical.CLINIC, "current_medication", FoldKind.NAME),
    "insurance_plans": (Vertical.CLINIC, "insurance_plan_name", FoldKind.NAME),
    "vehicles": (Vertical.AUTO, "vehicle_model", FoldKind.NAME),
    "rate_plans": (Vertical.HOTEL, "rate_plan_name", FoldKind.NAME),
    "coined": (Vertical.CLINIC, "employer_name", FoldKind.NAME),  # via role map
    "shops": (Vertical.AUTO, "previous_service_shop", FoldKind.NAME),
    "codes": (Vertical.AUTO, "vin", FoldKind.CODE),  # via family map
    "dates": (Vertical.AUTO, "drop_off_date", FoldKind.DATE),  # via kind map
    "emails": (Vertical.HOTEL, "contact_email", FoldKind.EMAIL),
}


def resolve_unit(bank: str, entry, callee_full_name: str) -> BankUnit:
    """Resolve one bank entry to the record field it fills."""
    if bank == "codes":
        vertical, field_name = _CODE_FIELDS[entry.family]
        return BankUnit(
            bank=bank,
            vertical=vertical,
            field_name=field_name,
            fold=FoldKind.CODE,
            value=entry.value,
        )
    if bank == "dates":
        vertical, field_name = _DATE_FIELDS[entry.kind]
        return BankUnit(
            bank=bank,
            vertical=vertical,
            field_name=field_name,
            fold=FoldKind.DATE,
            value=entry.value,
            spoken=entry.spoken,
        )
    if bank == "coined":
        vertical, field_name = _COINED_FIELDS[entry.role]
        return BankUnit(
            bank=bank,
            vertical=vertical,
            field_name=field_name,
            fold=FoldKind.NAME,
            value=entry.value,
        )
    if bank == "emails":
        drawn = instantiate_email(entry, callee_full_name)
        return BankUnit(
            bank=bank,
            vertical=Vertical.HOTEL,
            field_name="contact_email",
            fold=FoldKind.EMAIL,
            value=drawn.value,
            email_pattern=f"{entry.pattern}@{entry.domain}",
        )
    vertical, field_name, fold = _FLAT_UNIT_FIELDS[bank]
    return BankUnit(
        bank=bank,
        vertical=vertical,
        field_name=field_name,
        fold=fold,
        value=entry.value,
    )


def entry_draw_key(bank: str, entry) -> str:
    """The manifest identity of a drawn bank entry (pattern key for emails)."""
    if bank == "emails":
        return f"{entry.pattern}@{entry.domain}"
    return entry.value


# ---------------------------------------------------------------------------
# Manifest models (provenance-bearing artifact, CODE_DESIGN rules)
# ---------------------------------------------------------------------------


class BankContract(BaseModelNoExtra):
    """The flat bank contract the draw ran against."""

    easy_per_bank: Annotated[int, Field(description="Easy values per bank.")]
    hard_per_bank: Annotated[int, Field(description="Hard values per bank.")]
    draw_per_cell: Annotated[
        int, Field(description="Values drawn per bank x tier cell, no replacement.")
    ]


class CellDraw(BaseModelNoExtra):
    """One bank x tier cell's draw."""

    bank: Annotated[str, Field(description="The bank (the hazard unit).")]
    tier: Annotated[Difficulty, Field(description="easy or hard.")]
    drawn: Annotated[
        list[str],
        Field(
            description=(
                "The drawn entry keys in draw order (email cells record the "
                "pattern key; every other bank the value itself)."
            )
        ),
    ]


class TaskDraw(BaseModelNoExtra):
    """Which bank/tier/values one task drew, plus its filler context."""

    task_id: Annotated[str, Field(description="The generated task id.")]
    bank: Annotated[str, Field(description="Bank of the measured unit.")]
    tier: Annotated[Difficulty, Field(description="Tier of the measured unit.")]
    vertical: Annotated[Vertical, Field(description="Cover-story vertical.")]
    field_name: Annotated[str, Field(description="Missing record field collected.")]
    value: Annotated[str, Field(description="Pinned true (written) value.")]
    spoken: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Spoken form the callee's records hold, when it differs.",
        ),
    ]
    email_pattern: Annotated[
        Optional[str],
        Field(default=None, description="Email pattern key (emails bank only)."),
    ]
    record_id: Annotated[str, Field(description="Record the callback completes.")]
    callee_full_name: Annotated[str, Field(description="Filler callee identity.")]
    org_name: Annotated[str, Field(description="Filler client organization.")]
    submitted_on: Annotated[str, Field(description="Filler form-submission date.")]
    context_fields: Annotated[
        dict[str, str],
        Field(description="The record's filled (non-missing) filler cells."),
    ]


class FreezeManifest(BaseModelNoExtra):
    """Provenance for one generated task set (written next to the task file)."""

    manifest_version: Literal["1.0.0"]
    generator_version: Annotated[
        str, Field(description="GENERATOR_VERSION that produced the set.")
    ]
    call_frame_version: Annotated[
        str, Field(description="CALL_FRAME_VERSION the tasks were rendered with.")
    ]
    seed: Annotated[int, Field(description="The one seed behind every draw.")]
    n_entities: Annotated[
        int, Field(description="Missing fields per task (the canonical set is 1).")
    ]
    bank_contract: Annotated[
        BankContract, Field(description="The bank contract the draw ran against.")
    ]
    frame_size: Annotated[
        int,
        Field(
            description=(
                "Enumerated frame size: drawable atomic tasks (banks x tiers "
                "x contract values), recomputed by the generator (design §6)."
            )
        ),
    ]
    task_count: Annotated[int, Field(description="Tasks in the generated set.")]
    bank_shas: Annotated[
        dict[str, str],
        Field(description="sha256 per bank file the draw ran against."),
    ]
    cells: Annotated[
        list[CellDraw], Field(description="Per-cell draws, in generation order.")
    ]
    tasks: Annotated[
        list[TaskDraw], Field(description="Per-task draws, in generation order.")
    ]


class FreezeOutcome(BaseModelNoExtra):
    """What one freeze/draw run produced."""

    seed: int
    n_entities: int
    out_dir: Path
    task_count: int
    frame_size: int
    written: list[Path]


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


class FrameReport(BaseModelNoExtra):
    """The enumerated task frame (a reportable fact, design §6)."""

    banks: int
    tiers: int
    values_per_cell: int
    draw_per_cell: int
    frame_size: Annotated[
        int, Field(description="banks x tiers x values_per_cell (atomic tasks).")
    ]
    canonical_size: Annotated[
        int, Field(description="banks x tiers x draw_per_cell (the frozen set).")
    ]


def enumerate_frame(banks: Optional[Banks] = None) -> FrameReport:
    """Recompute the frame from the banks (contract-checked)."""
    banks = banks or load_banks()
    assert_bank_contract(banks)
    n_banks = len(CANONICAL_BANKS)
    return FrameReport(
        banks=n_banks,
        tiers=len(Difficulty),
        values_per_cell=BANK_CONTRACT_PER_TIER,
        draw_per_cell=DRAW_PER_CELL,
        frame_size=n_banks * len(Difficulty) * BANK_CONTRACT_PER_TIER,
        canonical_size=n_banks * len(Difficulty) * DRAW_PER_CELL,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _stream(*parts) -> random.Random:
    """A named rng stream (design §6: ``intake-freeze|...|{seed}``)."""
    return random.Random("|".join(str(part) for part in parts))


def _tier_entries(banks: Banks, bank: str, tier: Difficulty) -> list:
    entries = [e for e in getattr(banks, bank) if e.difficulty is tier]
    if len(entries) != BANK_CONTRACT_PER_TIER:
        raise GeneratorError(
            f"Bank {bank} has {len(entries)} {tier.value} values, expected "
            f"{BANK_CONTRACT_PER_TIER} (assert_bank_contract should have caught this)"
        )
    return entries


def _draw_filler(rng: random.Random, pool: list[str], reject) -> str:
    """The first acceptable filler value in a seeded shuffle of the pool."""
    order = list(pool)
    rng.shuffle(order)
    for candidate in order:
        if not reject(candidate):
            return candidate
    raise GeneratorError("Filler pool exhausted under the rejection rule")


def _context_fields(
    banks: Banks,
    vertical: Vertical,
    measured_fields: set[str],
    rng: random.Random,
    required: bool = True,
) -> dict[str, tuple[str, FoldKind]]:
    """The record's one filled filler cell: name -> (value, fold).

    Canonical atomic tasks always get one (``required=True``); a multi-entity
    bundle may measure every candidate, in which case the record carries only
    measured cells."""
    for field_name in _CONTEXT_CANDIDATES[vertical]:
        if field_name in measured_fields:
            continue
        if field_name == "contact_phone":
            pool = [e.value for e in banks.phones if e.difficulty is Difficulty.EASY]
            fold = FoldKind.PHONE
        elif field_name == "home_address":
            pool = [e.value for e in banks.addresses if e.difficulty is Difficulty.EASY]
            fold = FoldKind.NAME
        else:  # drop_off_date / check_in_date
            pool = [
                e.value
                for e in banks.dates
                if e.kind is DateKind.FUTURE_DATE and e.difficulty is Difficulty.EASY
            ]
            fold = FoldKind.DATE
        value = _draw_filler(rng, pool, reject=lambda _: False)
        return {field_name: (value, fold)}
    if required:
        raise GeneratorError(
            f"No context field available for vertical {vertical.value} with "
            f"measured fields {sorted(measured_fields)}"
        )
    return {}


# ---------------------------------------------------------------------------
# Task assembly + gates
# ---------------------------------------------------------------------------


def _fixed_task_notes(seed: int) -> str:
    return (
        "Canonical v4 freeze: generated by tau2 intake-tasks freeze "
        f"(GENERATOR_VERSION {GENERATOR_VERSION}, call frame "
        f"{CALL_FRAME_VERSION}, seed {seed}). Regenerable byte-identically; "
        "do not hand-edit."
    )


def _build_task(
    task_id: str,
    units: list[BankUnit],
    record_id: str,
    callee_full_name: str,
    org_name: str,
    submitted_on: str,
    context: dict[str, tuple[str, FoldKind]],
    seed: int,
    purpose: Optional[str] = None,
    notes: Optional[str] = None,
) -> tuple[Task, TaskDraw]:
    """Assemble one task. ``purpose``/``notes`` default to the canonical
    freeze strings; the compose bands (tasks/compose.py) pass their own —
    same shell, different provenance stamp."""
    vertical = units[0].vertical
    fields = [
        {
            "name": name,
            "description": FIELD_DESCRIPTIONS[name],
            "fold": fold.value,
            "value": value,
        }
        for name, (value, fold) in context.items()
    ] + [
        {
            "name": unit.field_name,
            "description": unit.description,
            "fold": unit.fold.value,
            "value": unit.value,
        }
        for unit in units
    ]
    record = {
        "record_id": record_id,
        "vertical": vertical.value,
        "callee_full_name": callee_full_name,
        "submitted_on": submitted_on,
        "fields": fields,
    }
    missing_names = [unit.field_name for unit in units]
    submission = {unit.field_name: unit.value for unit in units}
    entities = {unit.field_name: unit.entity_value for unit in units}
    tier = Difficulty.EASY if task_id.split("_")[-2] == "easy" else Difficulty.HARD
    task = Task(
        id=task_id,
        description=Description(
            purpose=purpose
            if purpose is not None
            else (
                f"Canonical v4 capture cell: {units[0].bank} x {tier.value}, "
                f"n={len(units)}: collect the missing "
                f"{' and '.join(missing_names)} on the {vertical.value} record."
            ),
            notes=notes if notes is not None else _fixed_task_notes(seed),
        ),
        user_scenario=render_user_scenario(
            callee_full_name,
            org_name,
            fetched_fields=[unit.field_name for unit in units],
        ),
        initial_state=InitialState(
            initialization_actions=[
                EnvFunctionCall(
                    env_type="assistant",
                    func_name="seed_record",
                    arguments={"record": record},
                ),
                EnvFunctionCall(
                    env_type="assistant",
                    func_name="blank_fields",
                    arguments={"record_id": record_id, "field_names": missing_names},
                ),
                EnvFunctionCall(
                    env_type="assistant",
                    func_name="create_callback_order",
                    arguments={"record_id": record_id, "org_name": org_name},
                ),
                EnvFunctionCall(
                    env_type="user",
                    func_name="set_entities",
                    arguments={"entities": entities},
                ),
            ]
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id="submit_fields_1",
                    requestor="assistant",
                    name="submit_fields",
                    arguments={
                        "record_id": record_id,
                        "fields": submission,
                        "confirmed_with_user": True,
                    },
                )
            ],
            reward_basis=["DB"],
        ),
        agent_opener=render_opener(org_name),
    )
    unit = units[0]
    draw = TaskDraw(
        task_id=task_id,
        bank=unit.bank,
        tier=tier,
        vertical=vertical,
        field_name=unit.field_name,
        value=unit.value,
        spoken=unit.spoken,
        email_pattern=unit.email_pattern,
        record_id=record_id,
        callee_full_name=callee_full_name,
        org_name=org_name,
        submitted_on=submitted_on,
        context_fields={name: value for name, (value, _) in context.items()},
    )
    return task, draw


def _gate_task(task: Task) -> None:
    """The design §6 verification gates, replayed through the REAL toolkits."""
    tools = IntakeTools(IntakeDB())
    user_tools = IntakeUserTools(IntakeUserDB())
    for call in task.initial_state.initialization_actions:
        toolkit = tools if call.env_type == "assistant" else user_tools
        try:
            getattr(toolkit, call.func_name)(**call.arguments)
        except Exception as error:
            raise GeneratorError(
                f"Task {task.id}: initialization replay failed on "
                f"{call.func_name}: {error}"
            ) from error

    seed_call = next(
        c
        for c in task.initial_state.initialization_actions
        if c.func_name == "seed_record"
    )
    record_spec = seed_call.arguments["record"]
    submit = next(
        a for a in task.evaluation_criteria.actions if a.name == "submit_fields"
    )

    # Gate: the order names exactly the record's missing fields, in order.
    order = tools.get_callback_order()
    missing = [f.name for f in order.missing_fields]
    if missing != list(submit.arguments["fields"]):
        raise GeneratorError(
            f"Task {task.id}: callback order names {missing}, golden submit "
            f"writes {list(submit.arguments['fields'])}"
        )
    # Gate: every entity the callee holds is a field the order asks for.
    entities = next(
        c
        for c in task.initial_state.initialization_actions
        if c.func_name == "set_entities"
    ).arguments["entities"]
    if set(entities) != set(missing):
        raise GeneratorError(
            f"Task {task.id}: callee entities {sorted(entities)} != missing "
            f"fields {sorted(missing)}"
        )

    # Gate: the expected final DB is reachable by exactly one submit call —
    # and equals the seeded complete record under the published folds.
    try:
        tools.submit_fields(
            record_id=submit.arguments["record_id"],
            fields=dict(submit.arguments["fields"]),
            confirmed_with_user=submit.arguments["confirmed_with_user"],
        )
    except Exception as error:
        raise GeneratorError(
            f"Task {task.id}: golden submit_fields failed: {error}"
        ) from error
    expected_tools = IntakeTools(IntakeDB())
    expected_tools.seed_record(record_spec)
    final = tools.db.intake_records[0]
    expected = expected_tools.db.intake_records[0]
    if final.model_dump() != expected.model_dump():
        raise GeneratorError(
            f"Task {task.id}: golden submit does not reach the seeded record "
            f"(final {final.model_dump()} != expected {expected.model_dump()})"
        )

    # Gate: fold-distinctness within the task (measured vs filler cells, and
    # the callee's own name vs a measured name value).
    folded = {}
    for field in record_spec["fields"]:
        key = (field["fold"], fold_value(FoldKind(field["fold"]), field["value"]))
        if key in folded:
            raise GeneratorError(
                f"Task {task.id}: values {folded[key]!r} and "
                f"{field['value']!r} fold together"
            )
        folded[key] = field["value"]
    callee_key = (
        FoldKind.NAME.value,
        fold_value(FoldKind.NAME, record_spec["callee_full_name"]),
    )
    if callee_key in folded:
        raise GeneratorError(
            f"Task {task.id}: callee {record_spec['callee_full_name']!r} "
            f"folds onto record value {folded[callee_key]!r}"
        )

    # Gate: ASCII-only everywhere (owner directive; the banks enforce it for
    # values, this catches template or catalog regressions too).
    serialized = task.model_dump_json()
    if not serialized.isascii():
        raise GeneratorError(f"Task {task.id}: serialized task is not pure ASCII")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_tasks(
    seed: int = DEFAULT_FREEZE_SEED,
    n_entities: int = 1,
    banks: Optional[Banks] = None,
) -> tuple[list[Task], FreezeManifest]:
    """Generate a verified task set: the canonical draw at ``n_entities=1``,
    a same-vertical multi-entity band otherwise."""
    if n_entities not in VALID_N_ENTITIES:
        raise GeneratorError(
            f"n_entities must be one of {VALID_N_ENTITIES}, got {n_entities}"
        )
    banks = banks or load_banks()
    assert_bank_contract(banks)
    _assert_org_catalog(banks)
    frame = enumerate_frame(banks)

    callee_pool = [
        e.value for e in banks.person_names if e.difficulty is Difficulty.EASY
    ]

    tasks: list[Task] = []
    cells: list[CellDraw] = []
    draws: list[TaskDraw] = []
    record_counter = 1000

    if n_entities == 1:
        cell_specs = [(bank, tier) for bank in CANONICAL_BANKS for tier in Difficulty]
        for bank, tier in cell_specs:
            entries = _tier_entries(banks, bank, tier)
            rng = _stream(RNG_NAMESPACE, "cell", bank, tier.value, seed)
            indices = rng.sample(range(BANK_CONTRACT_PER_TIER), DRAW_PER_CELL)
            cells.append(
                CellDraw(
                    bank=bank,
                    tier=tier,
                    drawn=[entry_draw_key(bank, entries[i]) for i in indices],
                )
            )
            for call_index, entry_index in enumerate(indices, start=1):
                entry = entries[entry_index]
                task_id = f"intake_{bank}_{tier.value}_{call_index:02d}"
                record_counter += 1
                task, draw = _assemble_one(
                    banks,
                    bank,
                    entry,
                    task_id,
                    f"REC-{record_counter}",
                    callee_pool,
                    seed,
                )
                tasks.append(task)
                draws.append(draw)
        _gate_cell_value_uniqueness(draws)
    else:
        tasks, cells, draws = _generate_multi_entity(
            banks, seed, n_entities, callee_pool
        )

    for task in tasks:
        _gate_task(task)
    task_ids = [t.id for t in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise GeneratorError("Duplicate task ids generated")

    manifest = FreezeManifest(
        manifest_version="1.0.0",
        generator_version=GENERATOR_VERSION,
        call_frame_version=CALL_FRAME_VERSION,
        seed=seed,
        n_entities=n_entities,
        bank_contract=BankContract(
            easy_per_bank=BANK_CONTRACT_PER_TIER,
            hard_per_bank=BANK_CONTRACT_PER_TIER,
            draw_per_cell=DRAW_PER_CELL,
        ),
        frame_size=frame.frame_size,
        task_count=len(tasks),
        bank_shas=banks.shas,
        cells=cells,
        tasks=draws,
    )
    return tasks, manifest


def _assemble_one(
    banks: Banks,
    bank: str,
    entry,
    task_id: str,
    record_id: str,
    callee_pool: list[str],
    seed: int,
) -> tuple[Task, TaskDraw]:
    filler_rng = _stream(RNG_NAMESPACE, "filler", task_id, seed)
    # Callee first (emails instantiate against it); reject a callee that
    # folds onto the measured value (a person_names cell drawing the callee's
    # own name would make the capture vacuous).
    measured_probe = entry_draw_key(bank, entry)
    callee = _draw_filler(
        filler_rng,
        callee_pool,
        reject=lambda name: bank == "person_names"
        and fold_value(FoldKind.NAME, name)
        == fold_value(FoldKind.NAME, measured_probe),
    )
    unit = resolve_unit(bank, entry, callee)
    org_rng_pool = list(ORG_CATALOG[unit.vertical])
    org = _draw_filler(filler_rng, org_rng_pool, reject=lambda _: False)
    submitted_on = f"2025-06-{_draw_filler(filler_rng, [f'{d:02d}' for d in _SUBMITTED_ON_DAYS], reject=lambda _: False)}"
    context = _context_fields(banks, unit.vertical, {unit.field_name}, filler_rng)
    return _build_task(
        task_id, [unit], record_id, callee, org, submitted_on, context, seed
    )


def _gate_cell_value_uniqueness(draws: list[TaskDraw]) -> None:
    """No value reuse within a bank x tier cell (byte and fold level)."""
    by_cell: dict[tuple[str, str], list[TaskDraw]] = {}
    for draw in draws:
        by_cell.setdefault((draw.bank, draw.tier.value), []).append(draw)
    for (bank, tier), cell_draws in by_cell.items():
        keys = [d.email_pattern or d.value for d in cell_draws]
        if len(keys) != len(set(keys)):
            raise GeneratorError(f"Cell {bank}/{tier}: drawn entry reused")
        values = [d.value for d in cell_draws]
        if len(values) != len(set(values)):
            raise GeneratorError(f"Cell {bank}/{tier}: captured value reused")


def _generate_multi_entity(
    banks: Banks, seed: int, n_entities: int, callee_pool: list[str]
) -> tuple[list[Task], list[CellDraw], list[TaskDraw]]:
    """A later additive band: n same-vertical units per call, 10 calls per
    (vertical x tier) cell. Kept minimal — the canonical set is atomic."""
    tasks: list[Task] = []
    cells: list[CellDraw] = []
    draws: list[TaskDraw] = []
    record_counter = 9000
    for vertical in Vertical:
        for tier in Difficulty:
            eligible = _eligible_units_by_vertical(banks, vertical, tier)
            if len(eligible) < n_entities:
                raise GeneratorError(
                    f"Vertical {vertical.value}/{tier.value} offers only "
                    f"{len(eligible)} banks; cannot bundle {n_entities}"
                )
            rng = _stream(
                RNG_NAMESPACE, "multi", vertical.value, tier.value, n_entities, seed
            )
            consumed: dict[str, list] = {
                bank: rng.sample(entries, len(entries))
                for bank, entries in eligible.items()
            }
            bank_order = list(eligible)
            drawn_keys: list[str] = []
            for call_index in range(1, DRAW_PER_CELL + 1):
                # Deterministic capacity-aware pick: the n banks with the
                # most values left (bank order breaks ties), so small pools
                # (e.g. the 5 hard loyalty codes) drain without starving a
                # call.
                by_capacity = sorted(
                    bank_order,
                    key=lambda bank: (-len(consumed[bank]), bank_order.index(bank)),
                )
                chosen_banks = by_capacity[:n_entities]
                if any(not consumed[bank] for bank in chosen_banks):
                    raise GeneratorError(
                        f"Vertical {vertical.value}/{tier.value}: bank pools "
                        f"exhausted bundling {n_entities} entities"
                    )
                task_id = (
                    f"intake_n{n_entities}_{vertical.value}_{tier.value}"
                    f"_{call_index:02d}"
                )
                record_counter += 1
                filler_rng = _stream(RNG_NAMESPACE, "filler", task_id, seed)
                callee = _draw_filler(filler_rng, callee_pool, reject=lambda _: False)
                units = []
                for bank in chosen_banks:
                    entry = consumed[bank].pop()
                    drawn_keys.append(entry_draw_key(bank, entry))
                    units.append(resolve_unit(bank, entry, callee))
                org = _draw_filler(
                    filler_rng, list(ORG_CATALOG[vertical]), reject=lambda _: False
                )
                submitted_on = f"2025-06-{_draw_filler(filler_rng, [f'{d:02d}' for d in _SUBMITTED_ON_DAYS], reject=lambda _: False)}"
                context = _context_fields(
                    banks,
                    vertical,
                    {u.field_name for u in units},
                    filler_rng,
                    required=False,
                )
                task, draw = _build_task(
                    task_id,
                    units,
                    f"REC-{record_counter}",
                    callee,
                    org,
                    submitted_on,
                    context,
                    seed,
                )
                tasks.append(task)
                draws.append(draw)
            cells.append(
                CellDraw(
                    bank=f"n{n_entities}:{vertical.value}", tier=tier, drawn=drawn_keys
                )
            )
    return tasks, cells, draws


def _eligible_units_by_vertical(
    banks: Banks, vertical: Vertical, tier: Difficulty
) -> dict[str, list]:
    """Bank -> tier entries resolving to ``vertical``, distinct field names."""
    eligible: dict[str, list] = {}
    seen_fields: set[str] = set()
    for bank in CANONICAL_BANKS:
        entries = []
        for entry in _tier_entries(banks, bank, tier):
            unit_vertical = _entry_vertical(bank, entry)
            if unit_vertical is vertical:
                entries.append(entry)
        if not entries:
            continue
        field = _entry_field(bank, entries[0])
        if field in seen_fields:
            continue  # one bank per record field in a bundle
        seen_fields.add(field)
        eligible[bank] = entries
    return eligible


def _entry_vertical(bank: str, entry) -> Vertical:
    if bank == "codes":
        return _CODE_FIELDS[entry.family][0]
    if bank == "dates":
        return _DATE_FIELDS[entry.kind][0]
    if bank == "coined":
        return _COINED_FIELDS[entry.role][0]
    return _FLAT_UNIT_FIELDS[bank][0]


def _entry_field(bank: str, entry) -> str:
    if bank == "codes":
        return _CODE_FIELDS[entry.family][1]
    if bank == "dates":
        return _DATE_FIELDS[entry.kind][1]
    if bank == "coined":
        return _COINED_FIELDS[entry.role][1]
    return _FLAT_UNIT_FIELDS[bank][1]


def _assert_org_catalog(banks: Banks) -> None:
    """No org display name may fold onto any bank value (orgs are filler)."""
    bank_folds = set()
    for bank in BANK_NAMES:
        if bank == "emails":
            continue
        for entry in getattr(banks, bank):
            bank_folds.add(fold_value(FoldKind.NAME, entry.value))
    for orgs in ORG_CATALOG.values():
        for org in orgs:
            if fold_value(FoldKind.NAME, org) in bank_folds:
                raise GeneratorError(
                    f"Org catalog name {org!r} collides with a bank value"
                )


# ---------------------------------------------------------------------------
# Writing (idempotent, byte-identical per seed + code version)
# ---------------------------------------------------------------------------


def _task_payload(tasks: list[Task]) -> str:
    payload = {
        "tasks": [task.model_dump(mode="json", exclude_none=True) for task in tasks]
    }
    return json.dumps(payload, indent=2) + "\n"


def _splits_payload(tasks: list[Task]) -> str:
    return json.dumps({"base": [task.id for task in tasks]}, indent=2) + "\n"


def _manifest_payload(manifest: FreezeManifest) -> str:
    return json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n"


def _write_if_changed(path: Path, text: str, written: list[Path]) -> None:
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    written.append(path)


def freeze_tasks(
    seed: int = DEFAULT_FREEZE_SEED,
    n_entities: int = 1,
    out_dir: Optional[Path] = None,
) -> FreezeOutcome:
    """Generate, verify, and write a task set + manifest.

    The canonical freeze is the default invocation (seed
    :data:`DEFAULT_FREEZE_SEED`, ``n_entities=1``, the domain data dir): it
    replaces ``tasks.json`` / ``split_tasks.json`` and writes
    ``tasks.manifest.json`` next to them. Multi-entity draws must go to an
    explicit ``out_dir`` — they never overwrite the canonical set.
    """
    out_dir = Path(out_dir) if out_dir is not None else INTAKE_DATA_DIR
    if n_entities != 1 and out_dir == INTAKE_DATA_DIR:
        raise GeneratorError(
            "Multi-entity draws are additive bands: pass an explicit out_dir "
            "(the canonical tasks.json is atomic-only, design §1/§6)"
        )
    tasks, manifest = generate_tasks(seed=seed, n_entities=n_entities)
    written: list[Path] = []
    _write_if_changed(out_dir / TASKS_FILENAME, _task_payload(tasks), written)
    _write_if_changed(out_dir / SPLITS_FILENAME, _splits_payload(tasks), written)
    _write_if_changed(out_dir / MANIFEST_FILENAME, _manifest_payload(manifest), written)
    logger.info(
        f"intake-tasks freeze: seed={seed} n_entities={n_entities} "
        f"tasks={len(tasks)} written={len(written)} -> {out_dir}"
    )
    return FreezeOutcome(
        seed=seed,
        n_entities=n_entities,
        out_dir=out_dir,
        task_count=len(tasks),
        frame_size=manifest.frame_size,
        written=written,
    )
