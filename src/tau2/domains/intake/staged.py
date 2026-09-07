# Copyright Sierra
"""The ``intake_staged`` domain: hierarchical (gated) entity capture.

E-HIER of the entity-composition experiments
(docs/designs/intake-entity-composition.md §3): the same bundles as the flat
compose bands, but the slots form a chain — the callback order reveals one
field at a time, every submission must VERIFY against the record before the
field is written and the next one unlocks, and a failed verification leaves
the field missing so the agent can re-elicit and retry until the duration
cap. The verify oracle is deliberate and scoped: it is the definition of the
gated workflow (each field validates before the process moves on, the way
authentication does) and it leaks only a pass/fail bit per attempt.

Registered as its OWN domain so the canonical ``intake`` surface stays
byte-stable: the frozen benchmark's policy, models, and tool contracts are
untouched — this module subclasses them. The user side is unchanged (the
callee holds all values in ``set_entities`` and answers what is asked);
gating is entirely environment-side.

The chain tasks themselves are written by ``tau2 intake-tasks compose``
(:mod:`tau2.domains.intake.tasks.compose`) into ``bands/chain_n2`` /
``bands/chain_n3`` and loaded here by band name.
"""

from pathlib import Path
from typing import Annotated, Dict, List, Optional

from pydantic import Field, ValidationError

from tau2.data_model.tasks import Task
from tau2.domains.intake.data_model import (
    CallbackOrder,
    IntakeDB,
    MissingFieldSpec,
)
from tau2.domains.intake.folds import fold_value
from tau2.domains.intake.tools import IntakeTools
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.domains.intake.user_tools import IntakeUserTools
from tau2.domains.intake.utils import (
    INTAKE_DATA_DIR,
    INTAKE_DB_PATH,
    INTAKE_MAIN_POLICY_PATH,
    INTAKE_USER_DB_PATH,
)
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType, is_tool
from tau2.utils import load_file

STAGED_POLICY_VERSION = "1.0.0"

# Appended to the canonical intake policy for this domain only (versioned,
# fixed in-code — machine, not scripts). The canonical policy file itself is
# never edited: the frozen benchmark's prompt stays byte-stable.
STAGED_POLICY_SECTION = """
## Staged work orders

Every work order on this desk is staged: the record system accepts fields one
at a time, in order, and `get_callback_order` shows only the field it will
accept next — not the full list. This section replaces the one-submission rule
under "Closing the call". Work the order one field at a time:

- Collect, pin, and confirm the shown field exactly as the sections above
  describe, then submit that field alone with `submit_fields` — a one-field
  submission, never two fields in one call.
- The record system validates each submission against the record. If it
  reports that the value did not verify, what you captured differs from the
  value on file: re-elicit the value with the person — re-pin its written form
  from scratch, per "Pinning the written form" — and submit it again. Never
  guess variants of a value the person did not give you, and never move on:
  the next field appears only after the current one verifies.
- A successful submission names the next field to collect; `get_callback_order`
  shows it too. Repeat until the system confirms all fields are complete.
- When every field has verified, thank the person and hang up with `end_call`.
  If the person ends the call early, whatever has verified stays on the
  record; submit nothing else.
"""

# The band names this domain's task splits map to (written by
# ``tau2 intake-tasks compose``).
CHAIN_BAND_SPLITS = ("chain_n2", "chain_n3")


class StagedIntakeDB(IntakeDB):
    """The intake DB plus the staged orders' verification targets.

    ``stage_expected`` is derived at initialization (the ``stage_fields``
    call runs between ``seed_record`` and ``blank_fields``, while the true
    values are still on the record) and never changes afterwards, so the
    gold and predicted environments always agree on it and the DB-hash
    reward comparison stays meaningful."""

    stage_expected: Annotated[
        Dict[str, Dict[str, str]],
        Field(
            default_factory=dict,
            description=(
                "record_id -> field name -> fold-canonical expected value; "
                "the verification targets of the record's staged order."
            ),
        ),
    ]


class StagedIntakeTools(IntakeTools):
    """The intake toolkit under the staged protocol.

    Same tool names and signatures as the canonical desk — only
    ``get_callback_order`` (reveal one field) and ``submit_fields`` (verify
    one field) change behavior, so transcripts stay comparable across the
    flat and chained experiments."""

    db: StagedIntakeDB

    def __init__(self, db: StagedIntakeDB) -> None:
        super().__init__(db)

    # ------------------------------------------------------------------
    # Environment derivation functions (initialization_actions, not tools)
    # ------------------------------------------------------------------

    def stage_fields(self, record_id: str, field_names: List[str]) -> None:
        """Recorded derivation: pin the verification targets for the record's
        staged order. Must run after ``seed_record`` and before
        ``blank_fields`` — it reads the fold-canonical true values off the
        still-complete record."""
        if not field_names:
            raise ValueError("stage_fields requires at least one field name")
        record = self._record_exact(record_id)
        if record.record_id in self.db.stage_expected:
            raise ValueError(f"Record {record_id} is already staged")
        by_name = {field.name: field for field in record.fields}
        unknown = sorted(set(field_names) - set(by_name))
        if unknown:
            raise ValueError(
                f"Record {record_id} has no field(s): {', '.join(unknown)}"
            )
        blanked = [name for name in field_names if by_name[name].is_missing]
        if blanked:
            raise ValueError(
                f"stage_fields must run before blank_fields; already missing: "
                f"{', '.join(blanked)}"
            )
        self.db.stage_expected[record.record_id] = {
            name: by_name[name].value for name in field_names
        }

    def create_callback_order(self, record_id: str, org_name: str) -> None:
        """Recorded derivation: enqueue the task's one STAGED callback order.

        Same derivation as the canonical desk, plus the gate that the staged
        verification targets cover exactly the order's missing fields."""
        super().create_callback_order(record_id, org_name)
        order = self.db.callback_orders[-1]
        expected = self.db.stage_expected.get(order.record_id)
        if expected is None:
            raise ValueError(
                f"Record {order.record_id} has no staged verification targets "
                "(stage_fields must run before create_callback_order on the "
                "staged desk)"
            )
        order_fields = {spec.name for spec in order.missing_fields}
        if set(expected) != order_fields:
            raise ValueError(
                f"Staged targets {sorted(expected)} != order missing fields "
                f"{sorted(order_fields)} for record {order.record_id}"
            )

    # ------------------------------------------------------------------
    # Tools (staged overrides)
    # ------------------------------------------------------------------

    def _current_stage(self, order: CallbackOrder) -> Optional[MissingFieldSpec]:
        """The first field of the order the record still holds as missing."""
        record = self._record_exact(order.record_id)
        by_name = {field.name: field for field in record.fields}
        for spec in order.missing_fields:
            if by_name[spec.name].is_missing:
                return spec
        return None

    @is_tool(ToolType.READ)
    def get_callback_order(self) -> CallbackOrder:
        """
        Get your work order for this call: the record id, the full name of the
        person the record belongs to, the client organization the call is on
        behalf of, and the missing field to collect NEXT, with a description
        of what it holds. This desk's orders are staged: the record system
        accepts fields one at a time, so the order shows only the field it
        will accept next — each further field appears after the current one is
        submitted and verifies.

        Returns:
            The callback order, showing the next field to collect.

        Raises:
            ValueError: If there is no callback order in the queue.
        """
        order = super().get_callback_order()
        current = self._current_stage(order)
        if current is None:
            return order
        return order.model_copy(update={"missing_fields": [current]})

    @is_tool(ToolType.WRITE)
    def submit_fields(
        self, record_id: str, fields: Dict[str, str], confirmed_with_user: bool
    ) -> str:
        """
        Submit ONE collected field value to the record. This desk's orders are
        staged: submit exactly the one field the callback order currently
        shows, after its read-back is confirmed. The record system validates
        the submission against the record — a value that does not verify is
        NOT written and the same field stays current: re-check the value with
        the person and submit it again. When a value verifies, it is written
        and the response names the next field to collect (or confirms the
        record is complete).

        Values are stored in canonical form (phones as their digit sequence,
        codes uppercase without separators, names case/diacritic-folded, dates
        as YYYY-MM-DD), so formatting variants of a correctly captured value
        are equivalent — capture the value itself accurately.

        Args:
            record_id: The record the callback order names.
            fields: Exactly one entry: the current missing field's name from
                the callback order, mapped to its collected value.
            confirmed_with_user: Whether the value in this submission was read
                back to the person and they explicitly confirmed it on this
                call. Answer honestly: true only if the read-back got an
                explicit yes; false otherwise.

        Returns:
            Confirmation naming the field written and the next field to
            collect, or that the record is complete.

        Raises:
            ValueError: If the record does not exist, the submission does not
                contain exactly the current field, the value is empty, the
                value does not verify against the record, or the record's
                missing fields were already submitted.
        """
        record = self._record_folded(record_id)
        expected = self.db.stage_expected.get(record.record_id)
        if expected is None:
            raise ValueError(f"Record {record.record_id} has no staged order")
        order = super().get_callback_order()
        current = self._current_stage(order)
        if current is None:
            raise ValueError(
                f"Record {record.record_id} has no missing fields — its "
                "values were already submitted"
            )
        if len(fields) != 1:
            raise ValueError(
                "Staged orders take one field per submission. Submit exactly: "
                f"{current.name}"
            )
        ((name, value),) = fields.items()
        if name != current.name:
            raise ValueError(
                f"{name!r} is not the field the order currently asks for. "
                f"Submit exactly: {current.name}"
            )
        if not value or not value.strip():
            raise ValueError(f"Field {name!r} must not be an empty value")
        field = next(f for f in record.fields if f.name == name)
        folded = fold_value(field.fold, value)
        if folded != expected[name]:
            raise ValueError(
                f"The value submitted for {name} did not verify against the "
                "record. Re-check the value with the person and submit it "
                "again."
            )
        field.value = folded
        following = self._current_stage(order)
        if following is None:
            return (
                f"Verified and recorded {name}. All fields for record "
                f"{record.record_id} are complete."
            )
        return (
            f"Verified and recorded {name}. Next missing field: "
            f"{following.name} — {following.description}"
        )


# ---------------------------------------------------------------------------
# Environment, task loading, and splits
# ---------------------------------------------------------------------------


def get_environment(
    db: Optional[StagedIntakeDB] = None,
    user_db: Optional[IntakeUserDB] = None,
    solo_mode: bool = False,
) -> Environment:
    """Build the staged intake environment. Solo mode is not supported."""
    if solo_mode:
        raise ValueError("Solo mode not supported for intake_staged domain")
    if db is None:
        db = StagedIntakeDB.load(INTAKE_DB_PATH)
    if user_db is None:
        user_db = IntakeUserDB.load(INTAKE_USER_DB_PATH)
    policy = load_file(INTAKE_MAIN_POLICY_PATH) + STAGED_POLICY_SECTION
    return Environment(
        domain_name="intake_staged",
        policy=policy,
        tools=StagedIntakeTools(db),
        user_tools=IntakeUserTools(user_db),
    )


def _band_path(band: str) -> Path:
    return INTAKE_DATA_DIR / "bands" / band / "tasks.json"


def _load_band(band: str) -> list[Task]:
    path = _band_path(band)
    if not path.exists():
        raise FileNotFoundError(
            f"Chain band {band!r} not found at {path} — run "
            "`tau2 intake-tasks compose` first"
        )
    raw = load_file(path)
    if isinstance(raw, dict) and "tasks" in raw:
        raw = raw["tasks"]
    tasks: list[Task] = []
    for index, item in enumerate(raw):
        try:
            tasks.append(Task.model_validate(item))
        except ValidationError as e:
            raise ValueError(f"Malformed task at index {index} in {path}: {e}") from e
    return tasks


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Load the staged chain tasks by band split.

    ``base`` (and ``None``) concatenates every chain band; a band name loads
    just that band."""
    if task_split_name in (None, "base"):
        tasks = [task for band in CHAIN_BAND_SPLITS for task in _load_band(band)]
    elif task_split_name in CHAIN_BAND_SPLITS:
        tasks = _load_band(task_split_name)
    else:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: "
            f"{['base', *CHAIN_BAND_SPLITS]}"
        )
    task_ids = [task.id for task in tasks]
    duplicates = sorted({tid for tid in task_ids if task_ids.count(tid) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate task id(s) across chain bands: {', '.join(duplicates)}"
        )
    return tasks


def get_tasks_split() -> dict[str, list[str]]:
    """The staged domain's splits: every chain band, plus their union."""
    splits = {band: [t.id for t in _load_band(band)] for band in CHAIN_BAND_SPLITS}
    splits["base"] = [tid for band in CHAIN_BAND_SPLITS for tid in splits[band]]
    return splits
