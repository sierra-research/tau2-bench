# Copyright Sierra
"""Paired composition bands (``tau2 intake-tasks compose``).

The entity-composition experiments (docs/designs/intake-entity-composition.md)
need bundles built from the EXACT values of the canonical atomic freeze, so
every bundled entity has a known atomic outcome under the same condition.
This module writes four additive bands in one deterministic pass:

- ``compose_n2`` / ``compose_n3`` — flat bundles (E-COMP): n same-vertical
  canonical values per call, one callback order naming every field, one
  ``submit_fields`` writing them all (the canonical shell, just wider).
- ``chain_n2`` / ``chain_n3`` — the staged twins (E-HIER): the IDENTICAL
  bundles re-emitted as staged chains for the ``intake_staged`` domain
  (:mod:`tau2.domains.intake.staged`): the callback order reveals one field
  at a time, each submission must verify against the record before the next
  field unlocks, and the golden path is one verified submit per slot.

Twin alignment is by construction: both bands come from one draw, so a chain
task and its flat twin share the callee, org, record, values, and slot order —
only the protocol differs. Everything is a deterministic function of the
checked-in banks, the canonical freeze manifest, the fixed in-code catalogs,
and one seed (named rng streams ``intake-compose|...``); no LLM, no
improvised prose. Bands are additive draws — the canonical ``tasks.json`` is
never touched.

**Slot pairing**: per (vertical x tier) cell, parent values are drawn from
the canonical manifest with clean-reference preference (parents whose
regular-condition seed-42 complication draw is "none" are drawn first, so
the paired per-slot analysis keeps clean-vs-clean comparisons), banks are
chosen capacity-aware and distinct within a call, and slot order is balanced
so each bank appears in each position evenly. When a cell's parent pool is
too small (auto x easy holds 18 canonical values for 20 n=2 slots) a value
is reused across calls — at most :data:`MAX_PARENT_USES_PER_BAND` uses per
band, never twice in one call, and every reuse is flagged in the manifest.

**Verification gates**: every flat task replays through the canonical
:func:`~tau2.domains.intake.tasks.generator._gate_task`; every chain task
walks the golden staged path through the REAL staged toolkit — one field
revealed at a time, a wrong-value probe (another slot's value) must be
refused without advancing, and the final DB must equal the seeded record.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Optional

from loguru import logger
from pydantic import Field

from tau2.data_model.simulation import ComplicationProfile
from tau2.data_model.tasks import Action, EnvFunctionCall, Task
from tau2.domains.intake.data_model import MISSING, Vertical
from tau2.domains.intake.staged import StagedIntakeDB, StagedIntakeTools
from tau2.domains.intake.tasks.banks import Banks, Difficulty, load_banks
from tau2.domains.intake.tasks.generator import (
    _SUBMITTED_ON_DAYS,
    DEFAULT_FREEZE_SEED,
    MANIFEST_FILENAME,
    ORG_CATALOG,
    SPLITS_FILENAME,
    TASKS_FILENAME,
    BankUnit,
    FreezeManifest,
    GeneratorError,
    _build_task,
    _context_fields,
    _draw_filler,
    _gate_task,
    _tier_entries,
    entry_draw_key,
    resolve_unit,
)
from tau2.domains.intake.utils import INTAKE_DATA_DIR
from tau2.utils.pydantic_utils import BaseModelNoExtra

# 1.0.1: re-composed under complication catalog 2.3.0 (#966 voice-gated
# spelling_style). The clean-parent preference keys on the seed-42 regular
# reference draw, and that draw re-keys on the catalog version, so the
# trunk sync changed which parents count as clean — the 1.0.0 bands were
# drawn under 2.2.0 semantics and no run ever consumed them.
# 1.0.2: re-composed under complication catalog 2.4.0 (spell_correction
# added, owner directive 2026-09-01) — every reference draw re-keys on the
# catalog bump, changing which parents count as clean.
COMPOSE_VERSION = "1.0.2"
COMPOSE_MANIFEST_VERSION = "1.0.0"
COMPOSE_RNG_NAMESPACE = "intake-compose"
COMPOSE_MANIFEST_FILENAME = "compose.manifest.json"

# The default location runs load bands from (see the band splits in
# tau2.domains.intake.environment / tau2.domains.intake.staged).
BANDS_DIR = INTAKE_DATA_DIR / "bands"

FLAT_BAND_NAMES = ("compose_n2", "compose_n3")
CHAIN_BAND_NAMES = ("chain_n2", "chain_n3")

# Calls per (vertical x tier) cell, per band n. 3 verticals x 2 tiers each:
# n=2 -> 60 calls / 120 slots, n=3 -> 30 calls / 90 slots (design doc §2.1).
CALLS_PER_CELL: dict[int, int] = {2: 10, 3: 5}

# A canonical parent value may back at most this many slots per band (only
# cells whose parent pool is smaller than their slot count reuse at all).
MAX_PARENT_USES_PER_BAND = 2

# The complication draw the clean-reference preference is computed against:
# the canonical regular-condition voice run (seed 42, default profile).
REFERENCE_RUN_SEED = 42
REFERENCE_CHANNEL = "voice"


# ---------------------------------------------------------------------------
# Manifest models (provenance-bearing artifact, CODE_DESIGN rules)
# ---------------------------------------------------------------------------


class ParentSlot(BaseModelNoExtra):
    """One bundled slot and the canonical atomic task it pairs to."""

    slot_index: Annotated[
        int, Field(ge=1, description="1-based position in the call's ask order.")
    ]
    parent_task_id: Annotated[
        str, Field(description="Canonical atomic task this slot's value comes from.")
    ]
    bank: Annotated[str, Field(description="Bank of the slot's value.")]
    tier: Annotated[Difficulty, Field(description="easy or hard.")]
    field_name: Annotated[str, Field(description="Record field the slot fills.")]
    entry_key: Annotated[
        str,
        Field(
            description=(
                "Manifest identity of the bank entry (email slots carry the "
                "pattern key; every other bank the value itself)."
            )
        ),
    ]
    value: Annotated[
        str,
        Field(
            description=(
                "The pinned written value IN THIS TASK. Differs from the "
                "parent's only for emails, which re-instantiate their "
                "pattern against this call's callee."
            )
        ),
    ]
    parent_value: Annotated[
        str, Field(description="The pinned written value in the parent task.")
    ]
    reused: Annotated[
        bool,
        Field(
            description=(
                "True when this parent value already backs an earlier call "
                "in the same band (pool smaller than the cell's slot count)."
            )
        ),
    ]
    parent_triggered_reference: Annotated[
        bool,
        Field(
            description=(
                "Whether the parent task drew a complication in the "
                "reference condition (voice, default profile, seed "
                f"{REFERENCE_RUN_SEED}) — deterministic preview, no run "
                "artifact. Triggered parents are drawn last and excluded "
                "from the paired per-slot analysis."
            )
        ),
    ]


class ComposeTaskDraw(BaseModelNoExtra):
    """One flat bundle draw (the chain twin re-emits it unchanged)."""

    task_id: Annotated[str, Field(description="Flat (compose band) task id.")]
    chain_task_id: Annotated[
        str, Field(description="The staged twin's task id (chain band).")
    ]
    vertical: Annotated[Vertical, Field(description="Cover-story vertical.")]
    tier: Annotated[Difficulty, Field(description="easy or hard (tier-pure).")]
    n_entities: Annotated[int, Field(description="Slots in the bundle.")]
    record_id: Annotated[str, Field(description="Record the callback completes.")]
    callee_full_name: Annotated[str, Field(description="Filler callee identity.")]
    org_name: Annotated[str, Field(description="Filler client organization.")]
    submitted_on: Annotated[str, Field(description="Filler form-submission date.")]
    context_fields: Annotated[
        dict[str, str],
        Field(description="The record's filled (non-missing) filler cells."),
    ]
    slots: Annotated[
        list[ParentSlot], Field(min_length=2, description="Slots in ask order.")
    ]


class ComposeBand(BaseModelNoExtra):
    """One written band."""

    name: Annotated[str, Field(description="Band (and split) name.")]
    protocol: Annotated[
        Literal["flat", "chain"], Field(description="Flat bundle or staged chain.")
    ]
    n_entities: Annotated[int, Field(description="Slots per call.")]
    task_count: Annotated[int, Field(description="Tasks in the band.")]
    task_ids: Annotated[list[str], Field(description="Task ids in file order.")]


class ComposeManifest(BaseModelNoExtra):
    """Provenance for one compose pass (all four bands)."""

    manifest_version: Literal["1.0.0"]
    compose_version: Annotated[
        str, Field(description="COMPOSE_VERSION that produced the bands.")
    ]
    parent_generator_version: Annotated[
        str, Field(description="GENERATOR_VERSION of the canonical freeze.")
    ]
    parent_seed: Annotated[int, Field(description="Seed of the canonical freeze.")]
    parent_manifest_sha256: Annotated[
        str, Field(description="sha256 of the canonical freeze manifest file.")
    ]
    seed: Annotated[int, Field(description="The one seed behind every draw.")]
    reference_run_seed: Annotated[
        int,
        Field(
            description=(
                "Run seed the parent_triggered_reference flags were previewed against."
            )
        ),
    ]
    bands: Annotated[list[ComposeBand], Field(description="The written bands.")]
    tasks: Annotated[
        list[ComposeTaskDraw],
        Field(description="Per-bundle draws, in generation order."),
    ]


class ComposeOutcome(BaseModelNoExtra):
    """What one compose run produced."""

    seed: int
    out_dir: Path
    band_task_counts: dict[str, int]
    reused_slots: int
    triggered_parent_slots: int
    written: list[Path]


# ---------------------------------------------------------------------------
# Parent pools
# ---------------------------------------------------------------------------


class _Parent(BaseModelNoExtra):
    """One canonical atomic task, resolved back to its bank entry."""

    task_id: str
    bank: str
    tier: Difficulty
    vertical: Vertical
    field_name: str
    entry_key: str
    parent_value: str
    triggered_reference: bool


def _load_parent_manifest(parent_dir: Path) -> tuple[FreezeManifest, str]:
    path = parent_dir / MANIFEST_FILENAME
    if not path.exists():
        raise GeneratorError(f"Canonical freeze manifest not found: {path}")
    raw = path.read_bytes()
    manifest = FreezeManifest.model_validate(json.loads(raw))
    if manifest.n_entities != 1:
        raise GeneratorError(
            f"{path} is not an atomic freeze (n_entities="
            f"{manifest.n_entities}); compose pairs against the canonical "
            "atomic set only"
        )
    return manifest, hashlib.sha256(raw).hexdigest()


def _load_parent_tasks(parent_dir: Path) -> dict[str, Task]:
    path = parent_dir / TASKS_FILENAME
    if not path.exists():
        raise GeneratorError(f"Canonical task file not found: {path}")
    raw = json.loads(path.read_text())
    tasks = raw["tasks"] if isinstance(raw, dict) else raw
    return {t["id"]: Task.model_validate(t) for t in tasks}


def _reference_triggered(task: Task) -> bool:
    """Whether the parent drew a complication in the reference condition.

    A deterministic preview of the run-time draw (same function, same
    material) — never a run-artifact read."""
    from tau2.domains.intake.complications import sample_complication

    return (
        sample_complication(
            REFERENCE_RUN_SEED,
            task,
            profile=ComplicationProfile.DEFAULT,
            rate_override=None,
            channel=REFERENCE_CHANNEL,
        )
        is not None
    )


def _entry_by_key(banks: Banks, bank: str, tier: Difficulty, key: str):
    for entry in _tier_entries(banks, bank, tier):
        if entry_draw_key(bank, entry) == key:
            return entry
    raise GeneratorError(
        f"Canonical manifest names entry {key!r} in bank {bank}/{tier.value}, "
        "but the checked-in bank has no such entry (bank drift — regenerate "
        "against the freeze the banks actually produced)"
    )


def _build_parents(
    banks: Banks, manifest: FreezeManifest, parent_tasks: dict[str, Task]
) -> dict[tuple[Vertical, Difficulty], dict[str, list[_Parent]]]:
    """Parent pools per (vertical x tier) cell: bank -> parents in manifest
    order. Emails key by pattern; the slot value re-instantiates later."""
    pools: dict[tuple[Vertical, Difficulty], dict[str, list[_Parent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for draw in manifest.tasks:
        task = parent_tasks.get(draw.task_id)
        if task is None:
            raise GeneratorError(
                f"Manifest task {draw.task_id} missing from the canonical task file"
            )
        parent = _Parent(
            task_id=draw.task_id,
            bank=draw.bank,
            tier=draw.tier,
            vertical=draw.vertical,
            field_name=draw.field_name,
            entry_key=draw.email_pattern or draw.value,
            parent_value=draw.value,
            triggered_reference=_reference_triggered(task),
        )
        pools[(draw.vertical, draw.tier)][parent.bank].append(parent)
    return pools


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _stream(*parts) -> random.Random:
    return random.Random("|".join(str(part) for part in parts))


class _BankStack:
    """Deterministic per-bank parent stack with bounded reuse.

    The base order is a seeded shuffle with triggered-reference parents
    FIRST (stacks pop from the end, so clean parents are drawn first). A
    drained stack refills once in the same order; a parent serves at most
    :data:`MAX_PARENT_USES_PER_BAND` slots per band."""

    def __init__(self, parents: list[_Parent], rng: random.Random) -> None:
        order = list(parents)
        rng.shuffle(order)
        order.sort(key=lambda p: not p.triggered_reference)
        self._base = order
        self._stack = list(order)
        self._use_count: dict[str, int] = defaultdict(int)
        self._refilled = False

    def capacity(self) -> int:
        return sum(
            MAX_PARENT_USES_PER_BAND - self._use_count[p.task_id] for p in self._base
        )

    def pop(self, exclude_task_ids: set[str]) -> _Parent:
        while True:
            while self._stack:
                parent = self._stack.pop()
                if self._use_count[parent.task_id] >= MAX_PARENT_USES_PER_BAND:
                    continue
                if parent.task_id in exclude_task_ids:
                    # Needed again this very call (cannot happen with
                    # distinct banks, but the invariant is cheap to hold).
                    continue
                self._use_count[parent.task_id] += 1
                return parent
            if self._refilled:
                raise GeneratorError("Bank pool exhausted beyond the reuse bound")
            self._refilled = True
            self._stack = list(self._base)

    def used(self, parent: _Parent) -> bool:
        return self._use_count[parent.task_id] > 1


def _flat_task_id(n: int, vertical: Vertical, tier: Difficulty, index: int) -> str:
    return f"intake_c{n}_{vertical.value}_{tier.value}_{index:02d}"


def _chain_task_id(flat_task_id: str) -> str:
    prefix = "intake_c"
    if not flat_task_id.startswith(prefix):
        raise GeneratorError(f"Not a compose task id: {flat_task_id}")
    return "intake_h" + flat_task_id[len(prefix) :]


def _compose_notes(seed: int, parent: FreezeManifest) -> str:
    return (
        "Composition band: generated by tau2 intake-tasks compose "
        f"(COMPOSE_VERSION {COMPOSE_VERSION}, seed {seed}) from the "
        f"canonical freeze (GENERATOR_VERSION {parent.generator_version}, "
        f"seed {parent.seed}). Regenerable byte-identically; do not hand-edit."
    )


def _flat_purpose(units: list[BankUnit], tier: Difficulty) -> str:
    banks = " + ".join(unit.bank for unit in units)
    return (
        f"Compose band (flat): {banks} x {tier.value}, n={len(units)}: "
        f"collect the missing fields on the {units[0].vertical.value} record "
        "in one flat call."
    )


def _chain_purpose(units: list[BankUnit], tier: Difficulty) -> str:
    banks = " -> ".join(unit.bank for unit in units)
    return (
        f"Chain band (staged): {banks} x {tier.value}, n={len(units)}: each "
        f"field on the {units[0].vertical.value} record unlocks only after "
        "the previous one verifies."
    )


def _order_slots(
    chosen: list[str],
    bank_order: list[str],
    position_counts: list[dict[str, int]],
) -> list[str]:
    """Greedy position balance, per bank RELATIVE to its own appearances.

    Banks appear in a cell's calls at very different rates (capacity-aware
    pairing puts the biggest pool in nearly every call), so balancing raw
    per-slot counts across banks lets a rare partner steal a position from a
    frequent one. Instead each slot goes to the not-yet-placed bank whose
    share of that slot is lowest relative to its own total appearances
    (``count*n - total``; canonical bank order breaks ties), which keeps
    every bank's own slot distribution within one of uniform."""
    n = len(chosen)
    totals = {bank: sum(counts[bank] for counts in position_counts) for bank in chosen}
    remaining = list(chosen)
    ordered: list[str] = []
    for slot in range(n):
        remaining.sort(
            key=lambda bank: (
                position_counts[slot][bank] * n - totals[bank],
                bank_order.index(bank),
            )
        )
        placed = remaining.pop(0)
        position_counts[slot][placed] += 1
        totals[placed] += 1
        ordered.append(placed)
    return ordered


def compose_tasks(
    seed: int = DEFAULT_FREEZE_SEED,
    banks: Optional[Banks] = None,
    parent_dir: Optional[Path] = None,
) -> tuple[dict[str, list[Task]], ComposeManifest]:
    """Generate and verify all four bands from the canonical freeze.

    Returns ``{band_name: tasks}`` for the four bands plus the one manifest.
    """
    banks = banks or load_banks()
    parent_dir = Path(parent_dir) if parent_dir is not None else INTAKE_DATA_DIR
    parent_manifest, parent_sha = _load_parent_manifest(parent_dir)
    parent_tasks = _load_parent_tasks(parent_dir)
    pools = _build_parents(banks, parent_manifest, parent_tasks)

    callee_pool = [
        e.value for e in banks.person_names if e.difficulty is Difficulty.EASY
    ]
    notes = _compose_notes(seed, parent_manifest)

    bands: dict[str, list[Task]] = {name: [] for name in FLAT_BAND_NAMES}
    bands.update({name: [] for name in CHAIN_BAND_NAMES})
    draws: list[ComposeTaskDraw] = []
    record_counter = 20000

    for n in sorted(CALLS_PER_CELL):
        flat_band = f"compose_n{n}"
        chain_band = f"chain_n{n}"
        for vertical in Vertical:
            for tier in Difficulty:
                cell = pools.get((vertical, tier), {})
                bank_order = list(cell)
                if len(bank_order) < n:
                    raise GeneratorError(
                        f"Cell {vertical.value}/{tier.value} spans only "
                        f"{len(bank_order)} canonical banks; cannot bundle {n}"
                    )
                rng = _stream(
                    COMPOSE_RNG_NAMESPACE, "cell", n, vertical.value, tier.value, seed
                )
                stacks = {bank: _BankStack(cell[bank], rng) for bank in bank_order}
                position_counts = [defaultdict(int) for _ in range(n)]
                for call_index in range(1, CALLS_PER_CELL[n] + 1):
                    by_capacity = sorted(
                        (b for b in bank_order if stacks[b].capacity() > 0),
                        key=lambda bank: (
                            -stacks[bank].capacity(),
                            bank_order.index(bank),
                        ),
                    )
                    if len(by_capacity) < n:
                        raise GeneratorError(
                            f"Cell {vertical.value}/{tier.value}: only "
                            f"{len(by_capacity)} banks have capacity left; "
                            f"cannot bundle {n} (reuse bound "
                            f"{MAX_PARENT_USES_PER_BAND})"
                        )
                    chosen = _order_slots(by_capacity[:n], bank_order, position_counts)
                    task_id = _flat_task_id(n, vertical, tier, call_index)
                    parents: list[_Parent] = []
                    used_parent_ids: set[str] = set()
                    for bank in chosen:
                        parent = stacks[bank].pop(used_parent_ids)
                        used_parent_ids.add(parent.task_id)
                        parents.append(parent)

                    record_counter += 1
                    record_id = f"REC-{record_counter}"
                    filler_rng = _stream(COMPOSE_RNG_NAMESPACE, "filler", task_id, seed)
                    measured_name_folds = {
                        p.parent_value for p in parents if p.bank == "person_names"
                    }
                    callee = _draw_filler(
                        filler_rng,
                        callee_pool,
                        reject=lambda name: _folds_onto_name(name, measured_name_folds),
                    )
                    units = [
                        resolve_unit(
                            p.bank,
                            _entry_by_key(banks, p.bank, p.tier, p.entry_key),
                            callee,
                        )
                        for p in parents
                    ]
                    org = _draw_filler(
                        filler_rng,
                        list(ORG_CATALOG[vertical]),
                        reject=lambda _: False,
                    )
                    submitted_on = "2025-06-" + _draw_filler(
                        filler_rng,
                        [f"{d:02d}" for d in _SUBMITTED_ON_DAYS],
                        reject=lambda _: False,
                    )
                    context = _context_fields(
                        banks,
                        vertical,
                        {u.field_name for u in units},
                        filler_rng,
                        required=False,
                    )
                    flat_task, _ = _build_task(
                        task_id,
                        units,
                        record_id,
                        callee,
                        org,
                        submitted_on,
                        context,
                        seed,
                        purpose=_flat_purpose(units, tier),
                        notes=notes,
                    )
                    chain_task = _to_chain_task(
                        flat_task, units, _chain_purpose(units, tier)
                    )
                    bands[flat_band].append(flat_task)
                    bands[chain_band].append(chain_task)
                    draws.append(
                        ComposeTaskDraw(
                            task_id=task_id,
                            chain_task_id=chain_task.id,
                            vertical=vertical,
                            tier=tier,
                            n_entities=n,
                            record_id=record_id,
                            callee_full_name=callee,
                            org_name=org,
                            submitted_on=submitted_on,
                            context_fields={
                                name: value for name, (value, _) in context.items()
                            },
                            slots=[
                                ParentSlot(
                                    slot_index=index,
                                    parent_task_id=parent.task_id,
                                    bank=parent.bank,
                                    tier=parent.tier,
                                    field_name=unit.field_name,
                                    entry_key=parent.entry_key,
                                    value=unit.value,
                                    parent_value=parent.parent_value,
                                    reused=stacks[parent.bank].used(parent),
                                    parent_triggered_reference=(
                                        parent.triggered_reference
                                    ),
                                )
                                for index, (parent, unit) in enumerate(
                                    zip(parents, units), start=1
                                )
                            ],
                        )
                    )

    for name in FLAT_BAND_NAMES:
        for task in bands[name]:
            _gate_task(task)
    for name in CHAIN_BAND_NAMES:
        for task in bands[name]:
            _gate_chain_task(task)
    _gate_position_balance(draws)

    all_ids = [t.id for tasks in bands.values() for t in tasks]
    if len(all_ids) != len(set(all_ids)):
        raise GeneratorError("Duplicate task ids across compose bands")

    manifest = ComposeManifest(
        manifest_version="1.0.0",
        compose_version=COMPOSE_VERSION,
        parent_generator_version=parent_manifest.generator_version,
        parent_seed=parent_manifest.seed,
        parent_manifest_sha256=parent_sha,
        seed=seed,
        reference_run_seed=REFERENCE_RUN_SEED,
        bands=[
            ComposeBand(
                name=name,
                protocol="flat" if name in FLAT_BAND_NAMES else "chain",
                n_entities=2 if name.endswith("n2") else 3,
                task_count=len(tasks),
                task_ids=[t.id for t in tasks],
            )
            for name, tasks in bands.items()
        ],
        tasks=draws,
    )
    return bands, manifest


def _folds_onto_name(candidate: str, measured_values: set[str]) -> bool:
    from tau2.domains.intake.folds import FoldKind, fold_value

    folded = fold_value(FoldKind.NAME, candidate)
    return any(folded == fold_value(FoldKind.NAME, value) for value in measured_values)


# ---------------------------------------------------------------------------
# The staged twin
# ---------------------------------------------------------------------------


def _to_chain_task(flat_task: Task, units: list[BankUnit], purpose: str) -> Task:
    """Re-emit a flat bundle as its staged chain twin.

    Same callee, org, record, values, and slot order; the initialization
    gains the ``stage_fields`` derivation (between seed and blank, while the
    true values are still on the record) and the golden path becomes one
    verified ``submit_fields`` per slot, in slot order."""
    task = flat_task.model_copy(deep=True)
    task.id = _chain_task_id(flat_task.id)
    task.description = task.description.model_copy(update={"purpose": purpose})

    init = list(task.initial_state.initialization_actions)
    seed_index = next(
        i for i, call in enumerate(init) if call.func_name == "seed_record"
    )
    record_id = init[seed_index].arguments["record"]["record_id"]
    field_names = [unit.field_name for unit in units]
    init.insert(
        seed_index + 1,
        EnvFunctionCall(
            env_type="assistant",
            func_name="stage_fields",
            arguments={"record_id": record_id, "field_names": field_names},
        ),
    )
    task.initial_state.initialization_actions = init

    task.evaluation_criteria.actions = [
        Action(
            action_id=f"submit_fields_{index}",
            requestor="assistant",
            name="submit_fields",
            arguments={
                "record_id": record_id,
                "fields": {unit.field_name: unit.value},
                "confirmed_with_user": True,
            },
        )
        for index, unit in enumerate(units, start=1)
    ]
    return task


def _gate_chain_task(task: Task) -> None:
    """Walk the golden staged path through the REAL staged toolkit.

    Per slot: exactly this field is revealed; a wrong-value probe (another
    slot's value — fold-distinct by the flat gate) is refused without
    advancing; the golden value verifies and unlocks the next slot. At the
    end the final DB must equal the seeded complete record."""
    from tau2.domains.intake.user_data_model import IntakeUserDB
    from tau2.domains.intake.user_tools import IntakeUserTools

    tools = StagedIntakeTools(StagedIntakeDB())
    user_tools = IntakeUserTools(IntakeUserDB())
    for call in task.initial_state.initialization_actions:
        toolkit = tools if call.env_type == "assistant" else user_tools
        try:
            getattr(toolkit, call.func_name)(**call.arguments)
        except Exception as error:
            raise GeneratorError(
                f"Chain task {task.id}: initialization replay failed on "
                f"{call.func_name}: {error}"
            ) from error

    submits = [a for a in task.evaluation_criteria.actions if a.name == "submit_fields"]
    if len(submits) < 2:
        raise GeneratorError(f"Chain task {task.id}: expected >=2 golden submits")
    golden_values = [
        (next(iter(a.arguments["fields"])), next(iter(a.arguments["fields"].values())))
        for a in submits
    ]
    for index, action in enumerate(submits):
        record_id = action.arguments["record_id"]
        field_name, value = golden_values[index]
        order = tools.get_callback_order()
        revealed = [spec.name for spec in order.missing_fields]
        if revealed != [field_name]:
            raise GeneratorError(
                f"Chain task {task.id}: stage {index + 1} reveals {revealed}, "
                f"golden submit writes {field_name!r}"
            )
        decoy_value = golden_values[(index + 1) % len(golden_values)][1]
        try:
            tools.submit_fields(
                record_id=record_id,
                fields={field_name: decoy_value},
                confirmed_with_user=True,
            )
        except ValueError:
            pass
        else:
            raise GeneratorError(
                f"Chain task {task.id}: stage {index + 1} accepted another "
                "slot's value — verification is not binding"
            )
        after_probe = [spec.name for spec in tools.get_callback_order().missing_fields]
        if after_probe != [field_name]:
            raise GeneratorError(
                f"Chain task {task.id}: a refused probe advanced the chain"
            )
        try:
            tools.submit_fields(
                record_id=record_id,
                fields={field_name: value},
                confirmed_with_user=True,
            )
        except Exception as error:
            raise GeneratorError(
                f"Chain task {task.id}: golden stage {index + 1} submit failed: {error}"
            ) from error

    seed_call = next(
        c
        for c in task.initial_state.initialization_actions
        if c.func_name == "seed_record"
    )
    expected_tools = StagedIntakeTools(StagedIntakeDB())
    expected_tools.seed_record(seed_call.arguments["record"])
    final = tools.db.intake_records[0]
    expected = expected_tools.db.intake_records[0]
    if final.model_dump() != expected.model_dump():
        raise GeneratorError(
            f"Chain task {task.id}: golden staged path does not reach the seeded record"
        )
    if any(field.value == MISSING for field in final.fields):
        raise GeneratorError(f"Chain task {task.id}: missing cells survived")


def _gate_position_balance(draws: list[ComposeTaskDraw]) -> None:
    """Within each cell, no bank may dominate a slot position: any bank's
    appearances in one position exceed its appearances in another by at most
    one for n=2 cells (the greedy filler guarantees this; the gate keeps it)."""
    cells: dict[tuple, list[ComposeTaskDraw]] = defaultdict(list)
    for draw in draws:
        cells[(draw.n_entities, draw.vertical, draw.tier)].append(draw)
    for (n, vertical, tier), cell_draws in cells.items():
        if n != 2:
            continue
        counts: dict[str, list[int]] = defaultdict(lambda: [0] * n)
        for draw in cell_draws:
            for slot in draw.slots:
                counts[slot.bank][slot.slot_index - 1] += 1
        for bank, per_slot in counts.items():
            if max(per_slot) - min(per_slot) > 1:
                raise GeneratorError(
                    f"Cell n{n}/{vertical.value}/{tier.value}: bank {bank} "
                    f"is position-skewed {per_slot}"
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


def _write_if_changed(path: Path, text: str, written: list[Path]) -> None:
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    written.append(path)


def freeze_compose(
    seed: int = DEFAULT_FREEZE_SEED,
    out_dir: Optional[Path] = None,
    parent_dir: Optional[Path] = None,
) -> ComposeOutcome:
    """Generate, verify, and write the four composition bands + manifest."""
    out_dir = Path(out_dir) if out_dir is not None else BANDS_DIR
    bands, manifest = compose_tasks(seed=seed, parent_dir=parent_dir)
    written: list[Path] = []
    for name, tasks in bands.items():
        _write_if_changed(
            out_dir / name / TASKS_FILENAME, _task_payload(tasks), written
        )
        _write_if_changed(
            out_dir / name / SPLITS_FILENAME, _splits_payload(tasks), written
        )
    _write_if_changed(
        out_dir / COMPOSE_MANIFEST_FILENAME,
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        written,
    )
    reused = sum(1 for draw in manifest.tasks for slot in draw.slots if slot.reused)
    triggered = sum(
        1
        for draw in manifest.tasks
        for slot in draw.slots
        if slot.parent_triggered_reference
    )
    logger.info(
        f"intake-tasks compose: seed={seed} bands="
        f"{ {name: len(tasks) for name, tasks in bands.items()} } "
        f"reused_slots={reused} triggered_parent_slots={triggered} -> {out_dir}"
    )
    return ComposeOutcome(
        seed=seed,
        out_dir=out_dir,
        band_task_counts={name: len(tasks) for name, tasks in bands.items()},
        reused_slots=reused,
        triggered_parent_slots=triggered,
        written=written,
    )
