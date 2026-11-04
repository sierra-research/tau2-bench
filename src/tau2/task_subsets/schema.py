"""Strict contracts for published subsets and deterministic authoring requests."""

import hashlib
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau2.data_model.tasks import Task

Domain = Literal["airline", "retail", "telecom", "banking_knowledge", "intake"]
SubsetName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Count = Annotated[int, Field(ge=0, strict=True)]


class SubsetModel(BaseModel):
    """Reject undeclared fields rather than ingesting retired research schemas."""

    model_config = ConfigDict(extra="forbid")


def frame_digest(task_ids: list[str]) -> str:
    """Hash canonical IDs in file order, without a trailing newline."""
    return hashlib.sha256("\n".join(task_ids).encode()).hexdigest()


class SubsetFrame(SubsetModel):
    """Exact registered canonical population used for a draw."""

    task_set: Domain = Field(description="Canonical task-set/domain identifier.")
    task_split: str | None = Field(description="Explicit source split; null means all.")
    size: Annotated[int, Field(gt=0, strict=True, description="Population size.")]
    digest: Digest = Field(description="SHA256 of canonical IDs in source order.")


class StratumRecord(SubsetModel):
    """Population and selected counts for one source-derived stratum."""

    key: str = Field(min_length=1, description="Source-derived stratum label.")
    frame_size: Annotated[int, Field(gt=0, strict=True, description="Stratum size.")]
    drawn: Count = Field(description="Selected count, including zero.")

    @model_validator(mode="after")
    def bounded(self) -> "StratumRecord":
        if self.drawn > self.frame_size:
            raise ValueError("stratum draw exceeds its population")
        return self


class TaskSubset(SubsetModel):
    """Immutable named canonical IDs, compatible with the three shipped artifacts."""

    name: SubsetName = Field(description="Artifact filename stem.")
    domain: Domain = Field(description="Domain owning the population.")
    strategy: Literal["census", "stratified"] = Field(description="Selection design.")
    stratifier: Literal["none", "write_profile", "issue_persona"] = Field(
        description="Closed source-derived stratification rule."
    )
    seed: Annotated[int, Field(strict=True, description="Recorded sampling seed.")]
    frame: SubsetFrame = Field(description="Population identity and order.")
    strata: list[StratumRecord] = Field(min_length=1, description="Stratum accounting.")
    created: str = Field(description="UTC ISO date of original artifact creation.")
    git_commit: Annotated[str | None, Field(pattern=r"^[0-9a-f]{40}$")] = Field(
        description="Source commit at authoring, or null when unavailable."
    )
    task_ids: list[str] = Field(
        min_length=1, description="Canonical IDs in frame order."
    )

    @field_validator("created")
    @classmethod
    def iso_date(cls, value: str) -> str:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("created must be an ISO date")
        return value

    @model_validator(mode="after")
    def consistent(self) -> "TaskSubset":
        if self.domain != self.frame.task_set:
            raise ValueError("subset domain and canonical task set disagree")
        if len(self.task_ids) != len(set(self.task_ids)) or any(
            not t for t in self.task_ids
        ):
            raise ValueError("task IDs must be nonempty and unique")
        if len({s.key for s in self.strata}) != len(self.strata):
            raise ValueError("duplicate strata")
        if sum(s.frame_size for s in self.strata) != self.frame.size:
            raise ValueError("strata do not account for the frame")
        if sum(s.drawn for s in self.strata) != self.size:
            raise ValueError("strata do not account for the selected IDs")
        if self.strategy == "census":
            if self.stratifier != "none" or self.size != self.frame.size:
                raise ValueError("a census must include the whole frame")
        elif self.stratifier == "none" or self.size >= self.frame.size:
            raise ValueError("stratified draws must select less than the whole frame")
        return self

    @property
    def size(self) -> int:
        """Selected population size, derived from the ID list."""
        return len(self.task_ids)


class CanonicalFrame(SubsetModel):
    """Ordinary main task objects in their registered loader's source order."""

    domain: Domain = Field(description="Registered canonical domain.")
    task_split: str | None = Field(description="Explicit registered split.")
    tasks: list[Task] = Field(min_length=1, description="Canonical source tasks.")

    @model_validator(mode="after")
    def unique_ids(self) -> "CanonicalFrame":
        if len(self.tasks) != len(set(self.task_ids)):
            raise ValueError("canonical frame repeats task IDs")
        return self

    @property
    def task_ids(self) -> list[str]:
        """Canonical IDs in the registered loader's order."""
        return [task.id for task in self.tasks]


class DrawRequest(SubsetModel):
    """Explicit reproducible selection inputs; defaults are owned by CLI config."""

    name: SubsetName = Field(description="New immutable artifact name.")
    domain: Domain = Field(description="Canonical domain to draw from.")
    size: Annotated[int, Field(gt=0, strict=True, description="Requested sample size.")]
    seed: Annotated[int, Field(strict=True, description="Within-stratum seed.")]


class Verification(SubsetModel):
    """Successful offline frame, draw and accounting verification."""

    name: str = Field(description="Verified artifact.")
    frame_size: int = Field(description="Verified source population size.")
    selected_size: int = Field(description="Verified selected task count.")
    frame_digest: str = Field(description="Verified source ID digest.")
    languages: list[str] = Field(
        default_factory=list, description="Verified pack bindings."
    )
