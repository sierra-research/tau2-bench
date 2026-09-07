# Copyright Sierra
"""Agent-side data model for the intake domain (design v4, outbound).

The service bureau's callback queue (docs/designs/intake-domain.md §5.1):
``intake_records`` holds pre-existing submitted records whose missing fields
carry an explicit MISSING sentinel, and ``callback_orders`` holds the one
work order per task — record id, callee, client org display name (feeds the
deterministic opener), and the ordered missing fields with their type
descriptions.

The expected final DB is the base DB with exactly the missing cells filled
with the pinned true values under the published fold rules
(:mod:`tau2.domains.intake.folds`); the initial state stays a derivation
(seed data plus the recorded env calls that blank the missing fields), so
the diff from the complete record is readable in the task JSON.
"""

from enum import Enum
from typing import Annotated, Any, Dict, List

from pydantic import Field, model_validator

from tau2.domains.intake.folds import FoldKind
from tau2.domains.intake.utils import INTAKE_DB_PATH
from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra

# The explicit sentinel a record cell holds while its value is still
# uncaptured. Purely syntactic and impossible as a folded real value (folds
# never emit angle brackets), so a record is complete exactly when no cell
# equals it.
MISSING = "<MISSING>"


class Vertical(str, Enum):
    """The v1 verticals. Which banks a task's values were drawn from, nothing more."""

    CLINIC = "clinic"
    AUTO = "auto"
    HOTEL = "hotel"


class RecordField(BaseModelNoExtra):
    """One cell of a submitted record."""

    name: Annotated[str, Field(description="Field key as it appears on the record.")]
    description: Annotated[
        str,
        Field(
            description=(
                "What the field holds and the format expected (the type "
                "description the callback order serves to the agent)."
            )
        ),
    ]
    fold: Annotated[
        FoldKind,
        Field(
            description=(
                "The published fold rule this field's value is canonicalized "
                "with at write time (design doc §5.1/§7)."
            )
        ),
    ]
    value: Annotated[
        str,
        Field(
            description=(
                "The stored value in its fold-canonical form, or the explicit "
                "MISSING sentinel while the value is still uncaptured."
            )
        ),
    ]

    @property
    def is_missing(self) -> bool:
        return self.value == MISSING


class IntakeRecord(BaseModelNoExtra):
    """A previously submitted record, possibly with missing cells."""

    record_id: Annotated[str, Field(description="Unique identifier for the record.")]
    vertical: Annotated[
        Vertical, Field(description="Which vertical the submitting form belongs to.")
    ]
    callee_full_name: Annotated[
        str,
        Field(
            description=(
                "Full name of the person the record belongs to, as submitted "
                "(display form, not folded — it is spoken, never DB-compared)."
            )
        ),
    ]
    submitted_on: Annotated[
        str,
        Field(
            description=(
                "Date the form was originally submitted, format YYYY-MM-DD "
                "(the purpose statement's 'the form submitted {when}')."
            )
        ),
    ]
    fields: Annotated[
        List[RecordField],
        Field(min_length=1, description="The record's cells, in form order."),
    ]

    @model_validator(mode="after")
    def _check_unique_field_names(self) -> "IntakeRecord":
        names = [field.name for field in self.fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Record {self.record_id} has duplicate field(s): "
                f"{', '.join(duplicates)}"
            )
        return self

    @property
    def missing_fields(self) -> List[RecordField]:
        """The record's uncaptured cells, in form order."""
        return [field for field in self.fields if field.is_missing]


class MissingFieldSpec(BaseModelNoExtra):
    """One missing field as the callback order names it to the agent."""

    name: Annotated[str, Field(description="Field key to collect and submit.")]
    description: Annotated[
        str, Field(description="What the field holds and the format expected.")
    ]


class CallbackOrder(BaseModelNoExtra):
    """The work order behind one callback call (one per task, design doc §5.1)."""

    record_id: Annotated[
        str, Field(description="The record whose missing fields this call collects.")
    ]
    callee_full_name: Annotated[
        str,
        Field(
            description=(
                "Full name of the person to call — the purpose statement "
                "names them; there is no separate identity-confirmation step."
            )
        ),
    ]
    org_name: Annotated[
        str,
        Field(
            description=(
                "Display name of the client organization the call is on "
                "behalf of (feeds the deterministic opener)."
            )
        ),
    ]
    missing_fields: Annotated[
        List[MissingFieldSpec],
        Field(
            min_length=1,
            description="The missing fields to collect, in the order to ask them.",
        ),
    ]


class IntakeDB(DB):
    """Database for the intake domain."""

    intake_records: Annotated[
        List[IntakeRecord],
        Field(
            default_factory=list,
            description="Previously submitted records (the reward surface).",
        ),
    ]
    callback_orders: Annotated[
        List[CallbackOrder],
        Field(
            default_factory=list,
            description="The callback queue (exactly one order per task).",
        ),
    ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get the statistics of the database."""
        return {
            "num_intake_records": len(self.intake_records),
            "num_callback_orders": len(self.callback_orders),
            "num_missing_cells": sum(
                len(record.missing_fields) for record in self.intake_records
            ),
        }


def get_db() -> IntakeDB:
    """Get an instance of the intake database."""
    return IntakeDB.load(INTAKE_DB_PATH)


if __name__ == "__main__":
    print(get_db().get_statistics())
