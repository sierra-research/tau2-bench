# Copyright Sierra
"""Agent toolkit for the intake domain.

Four in-call tools; the callback order is supplied in the pre-call prompt:

- ``log_capture(field_name, value)`` (GENERIC) — silent worksheet note of a
  just-heard value, logged BEFORE the read-back and re-logged after any
  correction. Stateless: validated against the callback order but written
  nowhere — the tool-call trace is the record. Makes first-capture accuracy,
  recovery (first log wrong, submit right), and log/submit consistency
  deterministically measurable, and a submit with no logs is a premature
  submit by construction.
- ``submit_fields(record_id, fields, confirmed_with_user)`` (WRITE) — one
  typed call writing all collected values at once; ends the task's write
  phase. ``confirmed_with_user`` is a required self-report attestation
  (recorded in the trace, not written to the DB and not reward-bearing):
  whether every submitted value's read-back got an explicit yes on the call.
- ``get_today()`` (READ) — the pinned clock.
- ``end_call()`` (GENERIC) — hangs up: the agents treat it as a
  call-terminating tool via the domain's ``additional_stop_tool_names`` hook,
  the outbound counterpart of the inbound transfer-ends-the-call convention.

Reward is final-DB hash equality, so the write canonicalizes at write time
with the published per-entity-type fold rules
(:mod:`tau2.domains.intake.folds`). Names, codes, and phones use their respective
folds; dates and times must already match the required field format. Numeric
decimal points and unit wording remain significant. The non-tool
``seed_record`` / ``blank_fields`` /
``create_callback_order`` methods are the environment derivation functions
tasks replay through ``initialization_actions`` — complete record first, then the
recorded calls that blank the missing fields, so the diff from the complete
record is readable in the task JSON.
"""

from typing import Dict, List

from tau2.domains.intake.data_model import (
    MISSING,
    CallbackOrder,
    IntakeDB,
    IntakeRecord,
    MissingFieldSpec,
)
from tau2.domains.intake.folds import FoldKind, fold_value
from tau2.domains.intake.utils import get_now
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class ScaffoldedIntakeTools(ToolKitBase):
    """Scaffolded callback tools, including the read-back attestation."""

    db: IntakeDB

    def __init__(self, db: IntakeDB) -> None:
        super().__init__(db)

    # ------------------------------------------------------------------
    # Environment derivation functions (initialization_actions, not tools)
    # ------------------------------------------------------------------

    def seed_record(self, record: dict) -> None:
        """Seed one complete submitted record before blanking missing fields.

        Every cell must carry a real value (the recorded ``blank_fields``
        call that follows is what makes fields missing — the diff from the
        complete record stays readable in the task JSON). Values are folded
        into their canonical stored form here, through the same seam
        ``submit_fields`` writes through, so the expected final DB is exactly
        this seeded record.
        """
        seeded = IntakeRecord.model_validate(record)
        if any(r.record_id == seeded.record_id for r in self.db.intake_records):
            raise ValueError(f"Record {seeded.record_id} already exists")
        blank = [field.name for field in seeded.fields if field.is_missing]
        if blank:
            raise ValueError(
                f"seed_record requires a complete record; {seeded.record_id} "
                f"is missing: {', '.join(blank)} (blank fields with the "
                "recorded blank_fields call instead)"
            )
        for field in seeded.fields:
            field.value = fold_value(field.fold, field.value)
        self.db.intake_records.append(seeded)

    def blank_fields(self, record_id: str, field_names: List[str]) -> None:
        """Recorded derivation: blank the task's missing fields on a seeded
        record (set them to the explicit MISSING sentinel)."""
        if not field_names:
            raise ValueError("blank_fields requires at least one field name")
        record = self._record_exact(record_id)
        field_by_name = {field.name: field for field in record.fields}
        for name in field_names:
            field = field_by_name.get(name)
            if field is None:
                raise ValueError(
                    f"Record {record_id} has no field named {name!r}. "
                    f"Fields: {', '.join(field_by_name)}"
                )
            if field.is_missing:
                raise ValueError(
                    f"Field {name!r} of record {record_id} is already missing"
                )
            field.value = MISSING

    def create_callback_order(self, record_id: str, org_name: str) -> None:
        """Recorded derivation: enqueue the task's one callback order.

        The order is DERIVED from the record — callee name from the record,
        missing fields (names + type descriptions) from its MISSING cells in
        form order — so the generation gate 'the record's missing-field set
        matches the order' holds by construction.
        """
        if self.db.callback_orders:
            raise ValueError("A callback order already exists (exactly one per task)")
        if not org_name.strip():
            raise ValueError("org_name must not be blank")
        record = self._record_exact(record_id)
        missing = record.missing_fields
        if not missing:
            raise ValueError(
                f"Record {record_id} has no missing fields to call back about"
            )
        self.db.callback_orders.append(
            CallbackOrder(
                record_id=record.record_id,
                callee_full_name=record.callee_full_name,
                org_name=org_name,
                missing_fields=[
                    MissingFieldSpec(name=field.name, description=field.description)
                    for field in missing
                ],
            )
        )

    def get_callback_order(self) -> CallbackOrder:
        """
        Read the work order during pre-call preparation: the record id, the
        full name of the person the record belongs to, the client organization
        the call is on behalf of, and the ordered list of missing fields to
        collect, each with a description of what it holds. This internal
        accessor is not an agent tool.

        Returns:
            The callback order for this call.

        Raises:
            ValueError: If the queue does not contain exactly one callback order.
        """
        if len(self.db.callback_orders) != 1:
            raise ValueError(
                f"Expected exactly one callback order, found "
                f"{len(self.db.callback_orders)}"
            )
        return self.db.callback_orders[0]

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @is_tool(ToolType.GENERIC)
    def log_capture(self, field_name: str, value: str) -> str:
        """
        Silently log a value you just heard for one of the callback order's
        missing fields. Log each provided value before submission. If you
        read a value back, log it first. Call this again with the corrected
        value whenever the person corrects it. Logging is a private worksheet
        note: it is not a submission, does not fill the field, and the person
        never hears about it.

        Args:
            field_name: The missing field the value is for, exactly as named
                in the callback order.
            value: The value as you captured it, in the field's expected
                written form.

        Returns:
            Confirmation that the value was logged.

        Raises:
            ValueError: If the field is not one of the callback order's
                missing fields, or the value is empty.
        """
        if len(self.db.callback_orders) != 1:
            raise ValueError(
                f"Expected exactly one callback order, found "
                f"{len(self.db.callback_orders)}"
            )
        order = self.db.callback_orders[0]
        valid = [field.name for field in order.missing_fields]
        if field_name not in valid:
            raise ValueError(
                f"{field_name!r} is not a missing field of this callback "
                f"order. Missing fields: {', '.join(valid)}"
            )
        if not value or not value.strip():
            raise ValueError("value must not be empty")
        return f"Logged {field_name}."

    @is_tool(ToolType.WRITE)
    def submit_fields(
        self, record_id: str, fields: Dict[str, str], confirmed_with_user: bool
    ) -> str:
        """
        Write all collected field values to the record at once. Submit exactly
        once, after every missing field has been collected and its read-back
        confirmed: the submission must contain exactly the record's missing
        fields — no others — and a value for each.

        Submit dates as YYYY-MM-DD and times as H:MM AM/PM, with no leading
        hour zero. Convert only information the person provided into these
        formats; ask for any missing or ambiguous components. Date and time
        strings are stored as submitted, apart from surrounding whitespace.
        Preserve decimal points within numbers and the person's unit wording.

        Phones are stored as digit sequences, codes uppercase without
        separators, and names case/diacritic-folded. Capture the full value
        accurately before applying the field's required format.

        Args:
            record_id: The record the callback order names.
            fields: All collected values, keyed by the missing field names
                from the callback order.
            confirmed_with_user: Whether every value in this submission was
                read back to the person and they explicitly confirmed it on
                this call. Answer honestly: true only if each field's
                read-back got an explicit yes; false otherwise.

        Returns:
            Confirmation message listing the fields written.

        Raises:
            ValueError: If the record does not exist, a submitted field is not
                one of the record's missing fields, a missing field has no
                value in the submission, a value is empty, or the record's
                missing fields were already submitted.
        """
        return self._submit_fields(record_id, fields)

    def _submit_fields(self, record_id: str, fields: Dict[str, str]) -> str:
        """Validate and atomically fill the missing cells for either policy."""
        record = self._record_folded(record_id)
        missing_by_name = {field.name: field for field in record.missing_fields}
        if not missing_by_name:
            raise ValueError(
                f"Record {record.record_id} has no missing fields — its "
                "values were already submitted"
            )
        unknown = sorted(set(fields) - set(missing_by_name))
        if unknown:
            raise ValueError(
                f"Not missing on record {record.record_id}: {', '.join(unknown)}. "
                f"Submit exactly: {', '.join(missing_by_name)}"
            )
        absent = sorted(set(missing_by_name) - set(fields))
        if absent:
            raise ValueError(
                f"Missing value(s) for record {record.record_id}: "
                f"{', '.join(absent)} — one submit_fields call writes all "
                "missing fields at once"
            )
        for name, value in fields.items():
            if not value or not value.strip():
                raise ValueError(f"Field {name!r} must not be an empty value")
        for name, value in fields.items():
            field = missing_by_name[name]
            field.value = fold_value(field.fold, value)
        return (
            f"Fields submitted to record {record.record_id}: "
            f"{', '.join(field.name for field in record.fields if field.name in fields)}"
        )

    @is_tool(ToolType.READ)
    def get_today(self) -> str:
        """
        Get today's date.

        Returns:
            Today's date, format YYYY-MM-DD.
        """
        return get_now().date().isoformat()

    @is_tool(ToolType.GENERIC)
    def end_call(self) -> str:
        """
        Hang up and end the call. Call this exactly once, as your last action:
        after submitting the collected fields and thanking the person, or when
        the call cannot proceed. Ending the call is final — nothing can be
        submitted after it.

        Returns:
            Confirmation that the call has ended.
        """
        return "Call ended."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_exact(self, record_id: str) -> IntakeRecord:
        """Record lookup for derivation calls: exact id, fail loud."""
        for record in self.db.intake_records:
            if record.record_id == record_id:
                return record
        raise ValueError(f"Record {record_id} not found")

    def _record_folded(self, record_id: str) -> IntakeRecord:
        """Record lookup for the write tool: the id echo folds like every code
        — 'rec-1001' and 'REC 1001' resolve to the record on
        file; a voice run cannot control the surface form the agent model
        echoes the id back in."""
        record_key = fold_value(FoldKind.CODE, record_id)
        for record in self.db.intake_records:
            if fold_value(FoldKind.CODE, record.record_id) == record_key:
                return record
        raise ValueError(f"Record {record_id} not found")
