"""Explicit caller roles and identities used by regional scenario publication."""

from typing import Literal

from pydantic import Field

from tau2.data_model.language import Gender, PackModel
from tau2.data_model.tasks import Task


class CallerIdentitySource(PackModel):
    """An explicitly reviewed canonical identity, never inferred from a name."""

    customer_id: str = Field(min_length=1, description="Canonical customer record ID.")
    full_name: str = Field(min_length=1, description="Exact canonical source name.")


class TelecomCallerRoles(PackModel):
    """Source identities whose approved localized names realize caller rotation."""

    domain: Literal["telecom"] = "telecom"
    target: CallerIdentitySource = Field(description="Account used by the tasks.")
    male: CallerIdentitySource = Field(description="Declared male name source.")
    female: CallerIdentitySource = Field(description="Declared female name source.")


class TaskCallerIdentity(PackModel):
    """The exact identity a task uses, with its canonical name provenance."""

    task_id: str = Field(description="Published task ID.")
    canonical_task_id: str = Field(description="Unchanged canonical task ID.")
    gender: Gender = Field(description="Existing reviewed task caller gender.")
    canonical_target_id: str = Field(description="Canonical account being renamed.")
    canonical_name_source_id: str = Field(description="Canonical donor of the name.")
    native_name: str = Field(description="Approved native-script caller name.")
    romanized_name: str = Field(description="Approved romanized caller name.")
    customer_id: str = Field(description="Name-derived ID with target numeric suffix.")
    email: str = Field(description="Name-derived email retaining target source format.")


class TelecomCallerPublication(PackModel):
    """Tasks and explicit identity provenance produced by caller binding."""

    tasks: list[Task]
    identities: list[TaskCallerIdentity]
